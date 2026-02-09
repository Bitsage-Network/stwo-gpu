/// Layer commitment chain verification.
///
/// Verifies that Poseidon commitments form a continuous chain across
/// transformer layers: layer[i].output_commitment == layer[i+1].input_commitment.
///
/// This binds per-layer proofs together — a valid matmul proof at layer N
/// only matters if its output feeds into layer N+1's input.
use core::poseidon::poseidon_hash_span;

/// Header for a single layer's commitment chain.
#[derive(Drop, Copy, Serde)]
pub struct LayerProofHeader {
    /// Layer index in the model.
    pub layer_index: u32,
    /// Poseidon commitment of the layer's input.
    pub input_commitment: felt252,
    /// Poseidon commitment of the layer's output.
    pub output_commitment: felt252,
}

/// Result of layer chain verification.
#[derive(Drop, Copy)]
pub struct ChainVerificationResult {
    /// Whether the chain is valid.
    pub is_valid: bool,
    /// Number of layers verified.
    pub num_layers: u32,
    /// Index of the first broken link (0 if valid).
    pub broken_at: u32,
}

/// Verify the commitment chain across all layers.
///
/// Checks:
/// 1. Layer[i].output_commitment == Layer[i+1].input_commitment (continuity)
/// 2. Layer[0].input_commitment == model_input_commitment (optional)
/// 3. Layer[N-1].output_commitment == model_output_commitment (optional)
pub fn verify_layer_chain(
    layer_headers: Span<LayerProofHeader>,
    model_input_commitment: felt252,
    model_output_commitment: felt252,
) -> ChainVerificationResult {
    let num_layers: u32 = layer_headers.len();

    if num_layers == 0 {
        return ChainVerificationResult { is_valid: true, num_layers: 0, broken_at: 0 };
    }

    // Check first layer's input matches model input
    let first = *layer_headers.at(0);
    if model_input_commitment != 0 && first.input_commitment != model_input_commitment {
        return ChainVerificationResult { is_valid: false, num_layers, broken_at: 0 };
    }

    // Check continuity: output[i] == input[i+1]
    let mut i: u32 = 0;
    loop {
        if i + 1 >= num_layers {
            break;
        }
        let current = *layer_headers.at(i);
        let next = *layer_headers.at(i + 1);
        if current.output_commitment != next.input_commitment {
            return ChainVerificationResult { is_valid: false, num_layers, broken_at: i + 1 };
        }
        i += 1;
    };

    // Check last layer's output matches model output
    let last = *layer_headers.at(num_layers - 1);
    if model_output_commitment != 0 && last.output_commitment != model_output_commitment {
        return ChainVerificationResult {
            is_valid: false, num_layers, broken_at: num_layers - 1,
        };
    }

    ChainVerificationResult { is_valid: true, num_layers, broken_at: 0 }
}

/// Compute a summary commitment over the entire layer chain.
/// Used for on-chain storage efficiency: one felt252 represents the entire chain.
pub fn compute_chain_commitment(
    layer_headers: Span<LayerProofHeader>,
) -> felt252 {
    let mut hash_inputs: Array<felt252> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= layer_headers.len() {
            break;
        }
        let header = *layer_headers.at(i);
        hash_inputs.append(header.layer_index.into());
        hash_inputs.append(header.input_commitment);
        hash_inputs.append(header.output_commitment);
        i += 1;
    };
    if hash_inputs.len() == 0 {
        return 0;
    }
    poseidon_hash_span(hash_inputs.span())
}
