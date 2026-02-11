//! HuggingFace model directory loader.
//!
//! Loads a transformer model from a HuggingFace-format directory containing:
//! - `config.json`: Model architecture config
//! - `*.safetensors`: Weight shard files
//!
//! Builds a [`ComputationGraph`] from config.json and loads weights from SafeTensors.

use std::collections::HashMap;
use std::path::Path;

use crate::compiler::graph::{ComputationGraph, GraphBuilder, GraphOp, GraphWeights};
use crate::compiler::onnx::{OnnxModel, OnnxError, ModelMetadata, TransformerConfig};
use crate::compiler::quantize_weights::quantize_weight_matrix;
use crate::compiler::safetensors::{discover_shards, list_tensors_sharded, tensor_to_f32};
use crate::components::activation::ActivationType;
use crate::gadgets::quantize::QuantStrategy;

/// Parsed HuggingFace config.json.
#[derive(Debug, Clone)]
pub struct HfConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
}

impl HfConfig {
    /// Parse a HuggingFace config.json file.
    pub fn from_file(path: &Path) -> Result<Self, OnnxError> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| OnnxError::IoError(format!("Cannot read config.json: {e}")))?;

        let json: serde_json::Value = serde_json::from_str(&contents)
            .map_err(|e| OnnxError::ParseError(format!("Invalid config.json: {e}")))?;

        Ok(Self {
            model_type: json["model_type"]
                .as_str()
                .unwrap_or("unknown")
                .to_string(),
            hidden_size: json["hidden_size"]
                .as_u64()
                .ok_or_else(|| OnnxError::ParseError("missing hidden_size".into()))?
                as usize,
            num_attention_heads: json["num_attention_heads"]
                .as_u64()
                .ok_or_else(|| OnnxError::ParseError("missing num_attention_heads".into()))?
                as usize,
            num_key_value_heads: json["num_key_value_heads"]
                .as_u64()
                .unwrap_or(json["num_attention_heads"].as_u64().unwrap_or(1))
                as usize,
            intermediate_size: json["intermediate_size"]
                .as_u64()
                .ok_or_else(|| OnnxError::ParseError("missing intermediate_size".into()))?
                as usize,
            num_hidden_layers: json["num_hidden_layers"]
                .as_u64()
                .ok_or_else(|| OnnxError::ParseError("missing num_hidden_layers".into()))?
                as usize,
            vocab_size: json["vocab_size"]
                .as_u64()
                .unwrap_or(32000)
                as usize,
            hidden_act: json["hidden_act"]
                .as_str()
                .or_else(|| json["hidden_activation"].as_str())
                .unwrap_or("silu")
                .to_string(),
            max_position_embeddings: json["max_position_embeddings"]
                .as_u64()
                .unwrap_or(2048)
                as usize,
        })
    }

    /// Convert to internal TransformerConfig.
    pub fn to_transformer_config(&self) -> TransformerConfig {
        let activation = match self.hidden_act.as_str() {
            "gelu" | "gelu_new" | "gelu_fast" => ActivationType::GELU,
            "relu" => ActivationType::ReLU,
            "silu" | "swiglu" => ActivationType::GELU, // Map SiLU to GELU for now
            _ => ActivationType::GELU,
        };

        TransformerConfig {
            d_model: self.hidden_size,
            num_heads: self.num_attention_heads,
            d_ff: self.intermediate_size,
            activation,
        }
    }
}

/// Load a model from a HuggingFace directory.
///
/// The directory should contain `config.json` and `*.safetensors` files.
///
/// # Arguments
/// * `model_dir` - Path to the model directory
/// * `num_layers` - Number of transformer layers to load (use 0 or config value for all)
///
/// Returns an `OnnxModel` with the graph and loaded weights.
pub fn load_hf_model(
    model_dir: &Path,
    num_layers: Option<usize>,
) -> Result<OnnxModel, OnnxError> {
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return Err(OnnxError::IoError(format!(
            "config.json not found in {}",
            model_dir.display()
        )));
    }

    // Parse config
    let hf_config = HfConfig::from_file(&config_path)?;
    let transformer_config = hf_config.to_transformer_config();

    let layers = num_layers.unwrap_or(hf_config.num_hidden_layers);
    let layers = if layers == 0 { hf_config.num_hidden_layers } else { layers };

    eprintln!("Model: {} ({})", hf_config.model_type, model_dir.display());
    eprintln!(
        "  hidden_size={}, heads={}, ff={}, layers={}/{}",
        hf_config.hidden_size,
        hf_config.num_attention_heads,
        hf_config.intermediate_size,
        layers,
        hf_config.num_hidden_layers,
    );

    // Build computation graph
    let graph = build_hf_transformer_graph(&transformer_config, layers);

    // Discover and load SafeTensors weights
    let shard_paths = discover_shards(model_dir, "model")
        .map_err(|e| OnnxError::WeightError(format!("Cannot discover shards: {e}")))?;

    if shard_paths.is_empty() {
        eprintln!(
            "  WARNING: No SafeTensors shard files found in {}. Using auto-generated weights.",
            model_dir.display()
        );
        // Fall back to auto-generated weights for testing
        let weights = crate::compiler::onnx::generate_weights_for_graph(&graph, 42);
        let num_parameters = crate::compiler::onnx::count_matmul_params(&graph);

        let metadata = ModelMetadata {
            name: format!("{}_{}L", hf_config.model_type, layers),
            num_parameters,
            input_shape: graph.input_shape,
            output_shape: graph.output_shape,
            num_layers: graph.num_layers(),
        };

        return Ok(OnnxModel {
            input_shape: graph.input_shape,
            graph,
            weights,
            metadata,
        });
    }

    eprintln!("  Loading weights from {} shards...", shard_paths.len());

    // List tensors for diagnostics
    let all_tensor_names: Vec<(String, usize)> = list_tensors_sharded(&shard_paths)
        .map_err(|e| OnnxError::WeightError(format!("Cannot list tensors: {e}")))?;
    eprintln!("  Total tensors across shards: {}", all_tensor_names.len());

    // Build name mapping: graph node index → HuggingFace tensor name
    let name_map = build_weight_name_map(&graph, layers, &all_tensor_names);
    eprintln!("  Weight name mapping: {} entries", name_map.len());
    for (idx, name) in &name_map {
        eprintln!("    node {} → {}", idx, name);
    }

    let weights = load_weights_from_shards(
        &shard_paths, &graph, &name_map, QuantStrategy::Symmetric8,
    ).map_err(|e| OnnxError::WeightError(format!("Cannot load weights: {e}")))?;

    let loaded_count = graph.nodes.iter().enumerate()
        .filter(|(idx, _)| weights.get_weight(*idx).is_some())
        .count();
    let matmul_count = graph.nodes.iter()
        .filter(|n| matches!(n.op, GraphOp::MatMul { .. }))
        .count();

    eprintln!("  Loaded weights for {}/{} MatMul layers", loaded_count, matmul_count);

    let num_parameters = crate::compiler::onnx::count_matmul_params(&graph);

    let metadata = ModelMetadata {
        name: format!("{}_{}L", hf_config.model_type, layers),
        num_parameters,
        input_shape: graph.input_shape,
        output_shape: graph.output_shape,
        num_layers: graph.num_layers(),
    };

    Ok(OnnxModel {
        input_shape: graph.input_shape,
        graph,
        weights,
        metadata,
    })
}

/// Build a transformer computation graph matching a HuggingFace architecture.
///
/// Each transformer block: LayerNorm → Q proj → O proj → LayerNorm → FFN up → act → FFN down
fn build_hf_transformer_graph(
    config: &TransformerConfig,
    num_layers: usize,
) -> ComputationGraph {
    let d = config.d_model;
    let d_ff = config.d_ff;

    let mut builder = GraphBuilder::new((1, d));

    for _ in 0..num_layers {
        // Pre-attention LayerNorm
        builder.layer_norm();
        // Q projection (d → d)
        builder.linear(d);
        // O projection (d → d)
        builder.linear(d);
        // Post-attention LayerNorm
        builder.layer_norm();
        // FFN up projection (d → d_ff)
        builder.linear(d_ff);
        // FFN activation
        builder.activation(config.activation);
        // FFN down projection (d_ff → d)
        builder.linear(d);
    }

    // Final LayerNorm
    builder.layer_norm();

    builder.build()
}

/// Build a mapping from graph node indices to HuggingFace tensor names.
///
/// For each transformer block, the graph has 7 nodes:
///   0: LayerNorm (pre-attention)
///   1: MatMul    (Q projection)    → model.layers.{L}.self_attn.q_proj.weight
///   2: MatMul    (O projection)    → model.layers.{L}.self_attn.o_proj.weight
///   3: LayerNorm (post-attention)
///   4: MatMul    (FFN up)          → model.layers.{L}.mlp.up_proj.weight (or gate_proj)
///   5: Activation
///   6: MatMul    (FFN down)        → model.layers.{L}.mlp.down_proj.weight
///
/// Plus a final LayerNorm at the end.
fn build_weight_name_map(
    graph: &ComputationGraph,
    num_layers: usize,
    available_tensors: &[(String, usize)],
) -> HashMap<usize, String> {
    let mut map = HashMap::new();
    let tensor_set: std::collections::HashSet<&str> = available_tensors
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();

    // Each block has 7 nodes. MatMul nodes are at offsets 1, 2, 4, 6 within a block.
    let nodes_per_block = 7;

    for layer_idx in 0..num_layers {
        let block_start = layer_idx * nodes_per_block;

        // Node offset 1: Q projection
        let q_node = block_start + 1;
        // Node offset 2: O projection
        let o_node = block_start + 2;
        // Node offset 4: FFN up
        let up_node = block_start + 4;
        // Node offset 6: FFN down
        let down_node = block_start + 6;

        // Try common HuggingFace naming patterns for the Q projection
        let q_candidates = [
            format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
            format!("model.layers.{layer_idx}.attention.wq.weight"),
            format!("transformer.h.{layer_idx}.attn.q_proj.weight"),
        ];
        for name in &q_candidates {
            if tensor_set.contains(name.as_str()) {
                map.insert(q_node, name.clone());
                break;
            }
        }

        // O projection
        let o_candidates = [
            format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
            format!("model.layers.{layer_idx}.attention.wo.weight"),
            format!("transformer.h.{layer_idx}.attn.o_proj.weight"),
        ];
        for name in &o_candidates {
            if tensor_set.contains(name.as_str()) {
                map.insert(o_node, name.clone());
                break;
            }
        }

        // FFN up projection (some models use gate_proj, some use up_proj)
        let up_candidates = [
            format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
            format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
            format!("model.layers.{layer_idx}.feed_forward.w1.weight"),
            format!("transformer.h.{layer_idx}.mlp.up_proj.weight"),
        ];
        for name in &up_candidates {
            if tensor_set.contains(name.as_str()) {
                map.insert(up_node, name.clone());
                break;
            }
        }

        // FFN down projection
        let down_candidates = [
            format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
            format!("model.layers.{layer_idx}.feed_forward.w2.weight"),
            format!("transformer.h.{layer_idx}.mlp.down_proj.weight"),
        ];
        for name in &down_candidates {
            if tensor_set.contains(name.as_str()) {
                map.insert(down_node, name.clone());
                break;
            }
        }
    }

    map
}

/// Load weights from multiple SafeTensors shards using an explicit name mapping.
fn load_weights_from_shards(
    shard_paths: &[std::path::PathBuf],
    graph: &ComputationGraph,
    name_map: &HashMap<usize, String>,
    strategy: QuantStrategy,
) -> Result<GraphWeights, crate::compiler::quantize_weights::WeightError> {
    use crate::compiler::quantize_weights::WeightError;

    // Memory-map all shards
    let mut shard_data: Vec<(std::fs::File, memmap2::Mmap)> = Vec::new();
    for path in shard_paths {
        let file = std::fs::File::open(path)
            .map_err(|e| WeightError::IoError(e.to_string()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| WeightError::IoError(e.to_string()))?;
        shard_data.push((file, mmap));
    }

    let mut weights = GraphWeights::new();

    for (idx, node) in graph.nodes.iter().enumerate() {
        if let GraphOp::MatMul { dims: (_m, k, n) } = &node.op {
            if let Some(tensor_name) = name_map.get(&idx) {
                // Search through shards for this tensor
                let mut found = false;
                for (_file, mmap) in &shard_data {
                    let tensors = safetensors::SafeTensors::deserialize(mmap)
                        .map_err(|e| WeightError::IoError(e.to_string()))?;

                    if let Ok(tensor) = tensors.tensor(tensor_name) {
                        let data = tensor_to_f32(tensor.data(), tensor.dtype());
                        // HuggingFace stores weights as (out_features, in_features)
                        // Our MatMul expects (k, n) = (in_features, out_features)
                        // So we may need to transpose
                        let shape = tensor.shape();
                        let (weight_data, wk, wn) = if shape.len() == 2 {
                            let rows = shape[0]; // out_features
                            let cols = shape[1]; // in_features
                            if rows == *n && cols == *k {
                                // Already (out, in) — transpose to (in, out) = (k, n)
                                let mut transposed = vec![0.0f32; data.len()];
                                for r in 0..rows {
                                    for c in 0..cols {
                                        transposed[c * rows + r] = data[r * cols + c];
                                    }
                                }
                                (transposed, *k, *n)
                            } else if rows == *k && cols == *n {
                                // Already (k, n)
                                (data, *k, *n)
                            } else {
                                eprintln!(
                                    "    WARNING: tensor {} shape ({}, {}) doesn't match expected ({}, {}), using as-is",
                                    tensor_name, rows, cols, k, n
                                );
                                (data, *k, *n)
                            }
                        } else {
                            (data, *k, *n)
                        };

                        let (matrix, _params) = quantize_weight_matrix(
                            &weight_data, wk, wn, strategy,
                        );
                        weights.add_weight(idx, matrix);
                        found = true;
                        break;
                    }
                }
                if !found {
                    eprintln!(
                        "    WARNING: tensor '{}' not found in any shard for node {}",
                        tensor_name, idx
                    );
                }
            }
        }
    }

    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_config_parse() {
        let json = r#"{
            "model_type": "qwen3",
            "hidden_size": 5120,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "intermediate_size": 13824,
            "num_hidden_layers": 40,
            "vocab_size": 152064,
            "hidden_act": "silu",
            "max_position_embeddings": 40960
        }"#;

        let tmp = std::env::temp_dir().join("test_config.json");
        std::fs::write(&tmp, json).unwrap();

        let config = HfConfig::from_file(&tmp).unwrap();
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.num_attention_heads, 40);
        assert_eq!(config.intermediate_size, 13824);
        assert_eq!(config.num_hidden_layers, 40);
        assert_eq!(config.model_type, "qwen3");

        let tc = config.to_transformer_config();
        assert_eq!(tc.d_model, 5120);
        assert_eq!(tc.num_heads, 40);
        assert_eq!(tc.d_ff, 13824);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_build_hf_transformer_graph() {
        let config = TransformerConfig {
            d_model: 8,
            num_heads: 2,
            d_ff: 16,
            activation: ActivationType::GELU,
        };

        let graph = build_hf_transformer_graph(&config, 2);

        // 2 blocks × 7 ops + 1 final LN = 15
        assert_eq!(graph.num_layers(), 15);
        assert_eq!(graph.input_shape, (1, 8));
        assert_eq!(graph.output_shape, (1, 8));
    }
}
