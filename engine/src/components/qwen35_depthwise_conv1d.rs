//! Qwen3.5 GatedDeltaNet depthwise causal Conv1D witness helpers.
//!
//! This is the row-level witness contract that the dedicated AIR component
//! should consume. It intentionally verifies row order as part of the witness
//! contract so independently swapped rows cannot remain valid.

use crate::components::matmul::M31Matrix;
use starknet_ff::FieldElement;
use stwo::core::air::Component;
use stwo::core::channel::{Channel, MerkleChannel};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::vcs_lifted::MerkleHasherLifted;
use stwo::core::verifier::verify as stwo_verify;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::prove;
use stwo::prover::CommitmentSchemeProver;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, TraceLocationAllocator,
};

use crate::backend::convert_evaluations;

pub const QWEN35_DEPTHWISE_CONV1D_MIN_LOG_SIZE: u32 = 4;
const DOMAIN_QWEN35_DEPTHWISE_CONV1D_MATRIX: u64 = 0x513344_4d4154; // "Q3D_MAT"
const DOMAIN_QWEN35_DEPTHWISE_CONV1D_WEIGHT: u64 = 0x513344_574754; // "Q3D_WGT"
const DOMAIN_QWEN35_DEPTHWISE_CONV1D_STATEMENT: u64 = 0x513344_53544d; // "Q3D_STM"
const DOMAIN_QWEN35_DEPTHWISE_CONV1D_CHANNEL: u64 = 0x513344_43484e; // "Q3D_CHN"

#[derive(Debug, Clone)]
pub struct Qwen35DepthwiseConv1dEval {
    pub log_n_rows: u32,
    pub kernel: usize,
    pub instance_id: usize,
}

pub type Qwen35DepthwiseConv1dComponent = FrameworkComponent<Qwen35DepthwiseConv1dEval>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DepthwiseConv1dTrace {
    pub log_size: u32,
    pub n_real_rows: usize,
    pub preprocessed: Vec<Vec<M31>>,
    pub execution: Vec<Vec<M31>>,
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen35DepthwiseConv1dProofError {
    #[error("Witness error: {0}")]
    Witness(String),
    #[error("Proving error: {0}")]
    Proving(String),
    #[error("Verification error: {0}")]
    Verification(String),
}

#[derive(Debug)]
pub struct Qwen35DepthwiseConv1dProof<H: MerkleHasherLifted> {
    pub stark_proof: StarkProof<H>,
    pub log_size: u32,
    pub n_real_rows: usize,
    pub kernel: usize,
    pub statement: Qwen35DepthwiseConv1dStatement,
}

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DepthwiseConv1dStatement {
    pub layer_idx: usize,
    pub seq_len: usize,
    pub channels: usize,
    pub kernel: usize,
    pub input_commitment: FieldElement,
    pub weight_commitment: FieldElement,
    pub output_commitment: FieldElement,
    pub statement_hash: FieldElement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DepthwiseConv1dWitnessRow {
    pub token_idx: usize,
    pub channel_idx: usize,
    pub valid_taps: Vec<M31>,
    pub input_taps: Vec<M31>,
    pub weight_taps: Vec<M31>,
    pub masked_input_taps: Vec<M31>,
    pub product_taps: Vec<M31>,
    pub output: M31,
}

impl FrameworkEval for Qwen35DepthwiseConv1dEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let expected_token = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_depthwise_conv1d_preprocessed_id(self.instance_id, "token", None).into(),
        });
        let expected_channel = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_depthwise_conv1d_preprocessed_id(self.instance_id, "channel", None).into(),
        });
        let expected_valid = (0..self.kernel)
            .map(|tap| {
                eval.get_preprocessed_column(PreProcessedColumnId {
                    id: qwen35_depthwise_conv1d_preprocessed_id(
                        self.instance_id,
                        "valid",
                        Some(tap),
                    )
                    .into(),
                })
            })
            .collect::<Vec<_>>();

        let token = eval.next_trace_mask();
        let channel = eval.next_trace_mask();
        let valid_taps = (0..self.kernel)
            .map(|_| eval.next_trace_mask())
            .collect::<Vec<_>>();
        let input_taps = (0..self.kernel)
            .map(|_| eval.next_trace_mask())
            .collect::<Vec<_>>();
        let weight_taps = (0..self.kernel)
            .map(|_| eval.next_trace_mask())
            .collect::<Vec<_>>();
        let masked_input_taps = (0..self.kernel)
            .map(|_| eval.next_trace_mask())
            .collect::<Vec<_>>();
        let product_taps = (0..self.kernel)
            .map(|_| eval.next_trace_mask())
            .collect::<Vec<_>>();
        let output = eval.next_trace_mask();

        eval.add_constraint(token - expected_token);
        eval.add_constraint(channel - expected_channel);

        let mut sum = E::F::from(M31::from(0u32));
        for tap in 0..self.kernel {
            eval.add_constraint(valid_taps[tap].clone() - expected_valid[tap].clone());
            eval.add_constraint(
                masked_input_taps[tap].clone() - valid_taps[tap].clone() * input_taps[tap].clone(),
            );
            eval.add_constraint(
                product_taps[tap].clone()
                    - masked_input_taps[tap].clone() * weight_taps[tap].clone(),
            );
            sum += product_taps[tap].clone();
        }
        eval.add_constraint(output - sum);

        eval
    }
}

pub fn qwen35_depthwise_conv1d_preprocessed_id(
    instance_id: usize,
    name: &str,
    tap: Option<usize>,
) -> String {
    match tap {
        Some(tap) => format!("qwen35_depthwise_conv1d_{name}_{tap}_{instance_id}"),
        None => format!("qwen35_depthwise_conv1d_{name}_{instance_id}"),
    }
}

fn zero() -> M31 {
    M31::from(0u32)
}

fn one() -> M31 {
    M31::from(1u32)
}

fn validate_shapes(input: &M31Matrix, weights: &[M31], kernel: usize) -> Result<(), String> {
    if input.rows == 0 || input.cols == 0 {
        return Err("DepthwiseConv1D input must be non-empty".to_string());
    }
    if kernel == 0 {
        return Err("DepthwiseConv1D kernel must be non-zero".to_string());
    }
    let expected_weights = input.cols * kernel;
    if weights.len() != expected_weights {
        return Err(format!(
            "DepthwiseConv1D weights length {} != channels*kernel {}",
            weights.len(),
            expected_weights
        ));
    }
    Ok(())
}

fn m31_matrix_commitment(domain: u64, matrix: &M31Matrix) -> FieldElement {
    let mut felts = Vec::with_capacity(4 + matrix.data.len());
    felts.push(FieldElement::from(domain));
    felts.push(FieldElement::from(matrix.rows as u64));
    felts.push(FieldElement::from(matrix.cols as u64));
    felts.push(FieldElement::from(matrix.data.len() as u64));
    for value in &matrix.data {
        felts.push(FieldElement::from(value.0 as u64));
    }
    starknet_crypto::poseidon_hash_many(&felts)
}

pub fn qwen35_depthwise_conv1d_input_commitment(input: &M31Matrix) -> FieldElement {
    m31_matrix_commitment(DOMAIN_QWEN35_DEPTHWISE_CONV1D_MATRIX, input)
}

pub fn qwen35_depthwise_conv1d_output_commitment(output: &M31Matrix) -> FieldElement {
    m31_matrix_commitment(DOMAIN_QWEN35_DEPTHWISE_CONV1D_MATRIX + 1, output)
}

pub fn qwen35_depthwise_conv1d_weight_commitment(
    weights: &[M31],
    channels: usize,
    kernel: usize,
) -> Result<FieldElement, String> {
    if channels == 0 || kernel == 0 {
        return Err("DepthwiseConv1D weight commitment dimensions must be non-zero".to_string());
    }
    if weights.len() != channels * kernel {
        return Err(format!(
            "DepthwiseConv1D weights length {} != channels*kernel {}",
            weights.len(),
            channels * kernel
        ));
    }

    let mut felts = Vec::with_capacity(4 + weights.len());
    felts.push(FieldElement::from(DOMAIN_QWEN35_DEPTHWISE_CONV1D_WEIGHT));
    felts.push(FieldElement::from(channels as u64));
    felts.push(FieldElement::from(kernel as u64));
    felts.push(FieldElement::from(weights.len() as u64));
    for value in weights {
        felts.push(FieldElement::from(value.0 as u64));
    }
    Ok(starknet_crypto::poseidon_hash_many(&felts))
}

pub fn qwen35_depthwise_conv1d_statement(
    layer_idx: usize,
    input: &M31Matrix,
    weights: &[M31],
    output: &M31Matrix,
    kernel: usize,
) -> Result<Qwen35DepthwiseConv1dStatement, String> {
    validate_shapes(input, weights, kernel)?;
    if output.rows != input.rows || output.cols != input.cols {
        return Err(format!(
            "DepthwiseConv1D output shape {}x{} != input shape {}x{}",
            output.rows, output.cols, input.rows, input.cols
        ));
    }

    let input_commitment = qwen35_depthwise_conv1d_input_commitment(input);
    let weight_commitment = qwen35_depthwise_conv1d_weight_commitment(weights, input.cols, kernel)?;
    let output_commitment = qwen35_depthwise_conv1d_output_commitment(output);
    let statement_hash = qwen35_depthwise_conv1d_statement_hash(
        layer_idx,
        input.rows,
        input.cols,
        kernel,
        input_commitment,
        weight_commitment,
        output_commitment,
    );

    Ok(Qwen35DepthwiseConv1dStatement {
        layer_idx,
        seq_len: input.rows,
        channels: input.cols,
        kernel,
        input_commitment,
        weight_commitment,
        output_commitment,
        statement_hash,
    })
}

pub fn qwen35_depthwise_conv1d_statement_hash(
    layer_idx: usize,
    seq_len: usize,
    channels: usize,
    kernel: usize,
    input_commitment: FieldElement,
    weight_commitment: FieldElement,
    output_commitment: FieldElement,
) -> FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(DOMAIN_QWEN35_DEPTHWISE_CONV1D_STATEMENT),
        FieldElement::from(layer_idx as u64),
        FieldElement::from(seq_len as u64),
        FieldElement::from(channels as u64),
        FieldElement::from(kernel as u64),
        input_commitment,
        weight_commitment,
        output_commitment,
    ])
}

fn mix_statement_hash<C: Channel>(channel: &mut C, statement_hash: FieldElement) {
    channel.mix_u64(DOMAIN_QWEN35_DEPTHWISE_CONV1D_CHANNEL);
    let bytes = statement_hash.to_bytes_be();
    channel.mix_u64(u64::from_be_bytes(bytes[0..8].try_into().unwrap()));
    channel.mix_u64(u64::from_be_bytes(bytes[8..16].try_into().unwrap()));
    channel.mix_u64(u64::from_be_bytes(bytes[16..24].try_into().unwrap()));
    channel.mix_u64(u64::from_be_bytes(bytes[24..32].try_into().unwrap()));
}

pub fn qwen35_depthwise_conv1d_output(
    input: &M31Matrix,
    weights: &[M31],
    kernel: usize,
) -> Result<M31Matrix, String> {
    validate_shapes(input, weights, kernel)?;
    let mut output = M31Matrix::new(input.rows, input.cols);
    for row in qwen35_depthwise_conv1d_witness(input, weights, kernel)? {
        output.set(row.token_idx, row.channel_idx, row.output);
    }
    Ok(output)
}

pub fn qwen35_depthwise_conv1d_witness(
    input: &M31Matrix,
    weights: &[M31],
    kernel: usize,
) -> Result<Vec<Qwen35DepthwiseConv1dWitnessRow>, String> {
    validate_shapes(input, weights, kernel)?;
    let mut rows = Vec::with_capacity(input.rows * input.cols);

    for token_idx in 0..input.rows {
        for channel_idx in 0..input.cols {
            let mut valid_taps = Vec::with_capacity(kernel);
            let mut input_taps = Vec::with_capacity(kernel);
            let mut weight_taps = Vec::with_capacity(kernel);
            let mut masked_input_taps = Vec::with_capacity(kernel);
            let mut product_taps = Vec::with_capacity(kernel);
            let mut output = zero();

            for tap in 0..kernel {
                let offset = tap as isize + 1 - kernel as isize;
                let source_token = token_idx as isize + offset;
                let valid = if source_token >= 0 { one() } else { zero() };
                let input_tap = if source_token >= 0 {
                    input.get(source_token as usize, channel_idx)
                } else {
                    zero()
                };
                let weight_tap = weights[channel_idx * kernel + tap];
                let masked = valid * input_tap;
                let product = masked * weight_tap;

                valid_taps.push(valid);
                input_taps.push(input_tap);
                weight_taps.push(weight_tap);
                masked_input_taps.push(masked);
                product_taps.push(product);
                output += product;
            }

            rows.push(Qwen35DepthwiseConv1dWitnessRow {
                token_idx,
                channel_idx,
                valid_taps,
                input_taps,
                weight_taps,
                masked_input_taps,
                product_taps,
                output,
            });
        }
    }

    Ok(rows)
}

pub fn qwen35_depthwise_conv1d_trace(
    input: &M31Matrix,
    weights: &[M31],
    kernel: usize,
) -> Result<Qwen35DepthwiseConv1dTrace, String> {
    let witness = qwen35_depthwise_conv1d_witness(input, weights, kernel)?;
    qwen35_depthwise_conv1d_trace_from_witness(input.rows, input.cols, kernel, &witness)
}

pub fn qwen35_depthwise_conv1d_trace_from_witness(
    seq_len: usize,
    channels: usize,
    kernel: usize,
    rows: &[Qwen35DepthwiseConv1dWitnessRow],
) -> Result<Qwen35DepthwiseConv1dTrace, String> {
    if seq_len == 0 || channels == 0 || kernel == 0 {
        return Err("DepthwiseConv1D trace dimensions must be non-zero".to_string());
    }
    let n_real_rows = seq_len * channels;
    if rows.len() != n_real_rows {
        return Err(format!(
            "DepthwiseConv1D witness rows {} != seq_len*channels {}",
            rows.len(),
            n_real_rows
        ));
    }

    let log_size = (n_real_rows.next_power_of_two().trailing_zeros())
        .max(QWEN35_DEPTHWISE_CONV1D_MIN_LOG_SIZE);
    let size = 1usize << log_size;

    let mut preprocessed = vec![vec![zero(); size]; 2 + kernel];
    let mut execution = vec![vec![zero(); size]; 3 + 5 * kernel];

    for (row_idx, row) in rows.iter().enumerate() {
        preprocessed[0][row_idx] = M31::from(row.token_idx as u32);
        preprocessed[1][row_idx] = M31::from(row.channel_idx as u32);
        execution[0][row_idx] = M31::from(row.token_idx as u32);
        execution[1][row_idx] = M31::from(row.channel_idx as u32);

        for tap in 0..kernel {
            preprocessed[2 + tap][row_idx] = row.valid_taps[tap];
            execution[2 + tap][row_idx] = row.valid_taps[tap];
            execution[2 + kernel + tap][row_idx] = row.input_taps[tap];
            execution[2 + 2 * kernel + tap][row_idx] = row.weight_taps[tap];
            execution[2 + 3 * kernel + tap][row_idx] = row.masked_input_taps[tap];
            execution[2 + 4 * kernel + tap][row_idx] = row.product_taps[tap];
        }
        execution[2 + 5 * kernel][row_idx] = row.output;
    }

    Ok(Qwen35DepthwiseConv1dTrace {
        log_size,
        n_real_rows,
        preprocessed,
        execution,
    })
}

fn simd_evals_from_columns(
    columns: &[Vec<M31>],
    log_size: u32,
) -> Vec<CircleEvaluation<SimdBackend, stwo::core::fields::m31::BaseField, BitReversedOrder>> {
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();
    columns
        .iter()
        .map(|column| {
            let mut col = Col::<SimdBackend, stwo::core::fields::m31::BaseField>::zeros(size);
            for (idx, value) in column.iter().copied().enumerate().take(size) {
                col.set(idx, value);
            }
            CircleEvaluation::new(domain, col)
        })
        .collect()
}

pub fn prove_qwen35_depthwise_conv1d_air(
    input: &M31Matrix,
    weights: &[M31],
    kernel: usize,
) -> Result<Qwen35DepthwiseConv1dProof<Blake2sHash>, Qwen35DepthwiseConv1dProofError> {
    prove_qwen35_depthwise_conv1d_air_for_layer(0, input, weights, kernel)
}

pub fn prove_qwen35_depthwise_conv1d_air_for_layer(
    layer_idx: usize,
    input: &M31Matrix,
    weights: &[M31],
    kernel: usize,
) -> Result<Qwen35DepthwiseConv1dProof<Blake2sHash>, Qwen35DepthwiseConv1dProofError> {
    let output = qwen35_depthwise_conv1d_output(input, weights, kernel)
        .map_err(Qwen35DepthwiseConv1dProofError::Witness)?;
    let statement = qwen35_depthwise_conv1d_statement(layer_idx, input, weights, &output, kernel)
        .map_err(Qwen35DepthwiseConv1dProofError::Witness)?;
    let trace = qwen35_depthwise_conv1d_trace(input, weights, kernel)
        .map_err(Qwen35DepthwiseConv1dProofError::Witness)?;
    let pcs_config = PcsConfig::default();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(trace.log_size + 1 + pcs_config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    mix_statement_hash(channel, statement.statement_hash);
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(pcs_config, &twiddles);

    let preprocessed = simd_evals_from_columns(&trace.preprocessed, trace.log_size);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, _>(
        preprocessed,
    ));
    tree_builder.commit(channel);

    let execution = simd_evals_from_columns(&trace.execution, trace.log_size);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, _>(
        execution,
    ));
    tree_builder.commit(channel);

    let component = FrameworkComponent::new(
        &mut TraceLocationAllocator::default(),
        Qwen35DepthwiseConv1dEval {
            log_n_rows: trace.log_size,
            kernel,
            instance_id: 0,
        },
        SecureField::from(M31::from(0u32)),
    );
    let stark_proof =
        prove::<SimdBackend, Blake2sMerkleChannel>(&[&component], channel, commitment_scheme)
            .map_err(|e| Qwen35DepthwiseConv1dProofError::Proving(format!("{e:?}")))?;

    Ok(Qwen35DepthwiseConv1dProof {
        stark_proof,
        log_size: trace.log_size,
        n_real_rows: trace.n_real_rows,
        kernel,
        statement,
    })
}

pub fn verify_qwen35_depthwise_conv1d_air(
    proof: &Qwen35DepthwiseConv1dProof<Blake2sHash>,
) -> Result<(), Qwen35DepthwiseConv1dProofError> {
    verify_qwen35_depthwise_conv1d_air_with_statement_hash(proof, proof.statement.statement_hash)
}

pub fn verify_qwen35_depthwise_conv1d_air_with_statement_hash(
    proof: &Qwen35DepthwiseConv1dProof<Blake2sHash>,
    expected_statement_hash: FieldElement,
) -> Result<(), Qwen35DepthwiseConv1dProofError> {
    if proof.statement.statement_hash != expected_statement_hash {
        return Err(Qwen35DepthwiseConv1dProofError::Verification(
            "DepthwiseConv1D statement hash mismatch".to_string(),
        ));
    }
    let pcs_config = PcsConfig::default();
    let mut allocator = TraceLocationAllocator::default();
    let component = FrameworkComponent::new(
        &mut allocator,
        Qwen35DepthwiseConv1dEval {
            log_n_rows: proof.log_size,
            kernel: proof.kernel,
            instance_id: 0,
        },
        SecureField::from(M31::from(0u32)),
    );
    let bounds = Component::trace_log_degree_bounds(&component);
    if bounds.len() != 2 {
        return Err(Qwen35DepthwiseConv1dProofError::Verification(format!(
            "expected 2 commitment trees, got {}",
            bounds.len()
        )));
    }

    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    mix_statement_hash(channel, expected_statement_hash);
    let mut commitment_scheme = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(pcs_config);
    commitment_scheme.commit(proof.stark_proof.commitments[0], &bounds[0], channel);
    commitment_scheme.commit(proof.stark_proof.commitments[1], &bounds[1], channel);

    stwo_verify::<Blake2sMerkleChannel>(
        &[&component as &dyn Component],
        channel,
        &mut commitment_scheme,
        proof.stark_proof.clone(),
    )
    .map_err(|e| Qwen35DepthwiseConv1dProofError::Verification(format!("{e:?}")))
}

pub fn verify_qwen35_depthwise_conv1d_witness(
    input: &M31Matrix,
    weights: &[M31],
    kernel: usize,
    rows: &[Qwen35DepthwiseConv1dWitnessRow],
) -> Result<(), String> {
    validate_shapes(input, weights, kernel)?;
    let expected_rows = input.rows * input.cols;
    if rows.len() != expected_rows {
        return Err(format!(
            "DepthwiseConv1D witness rows {} != expected {}",
            rows.len(),
            expected_rows
        ));
    }

    for (row_idx, row) in rows.iter().enumerate() {
        let expected_token = row_idx / input.cols;
        let expected_channel = row_idx % input.cols;
        if row.token_idx != expected_token || row.channel_idx != expected_channel {
            return Err(format!(
                "DepthwiseConv1D row {row_idx} has token={}, channel={}, expected token={}, channel={}",
                row.token_idx, row.channel_idx, expected_token, expected_channel
            ));
        }
        if row.valid_taps.len() != kernel
            || row.input_taps.len() != kernel
            || row.weight_taps.len() != kernel
            || row.masked_input_taps.len() != kernel
            || row.product_taps.len() != kernel
        {
            return Err(format!("DepthwiseConv1D row {row_idx} has wrong tap width"));
        }

        let mut expected_output = zero();
        for tap in 0..kernel {
            let offset = tap as isize + 1 - kernel as isize;
            let source_token = expected_token as isize + offset;
            let expected_valid = if source_token >= 0 { one() } else { zero() };
            let expected_input = if source_token >= 0 {
                input.get(source_token as usize, expected_channel)
            } else {
                zero()
            };
            let expected_weight = weights[expected_channel * kernel + tap];
            let expected_masked = expected_valid * expected_input;
            let expected_product = expected_masked * expected_weight;

            if row.valid_taps[tap] != expected_valid
                || row.input_taps[tap] != expected_input
                || row.weight_taps[tap] != expected_weight
                || row.masked_input_taps[tap] != expected_masked
                || row.product_taps[tap] != expected_product
            {
                return Err(format!(
                    "DepthwiseConv1D row {row_idx} tap {tap} violates causal convolution"
                ));
            }
            expected_output += expected_product;
        }

        if row.output != expected_output {
            return Err(format!(
                "DepthwiseConv1D row {row_idx} output violates product sum"
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m31_vec(values: &[u32]) -> Vec<M31> {
        values.iter().map(|value| M31::from(*value)).collect()
    }

    #[test]
    fn qwen35_depthwise_conv1d_witness_matches_left_padded_causal_conv() {
        let input = M31Matrix {
            rows: 3,
            cols: 2,
            data: m31_vec(&[1, 10, 2, 20, 3, 30]),
        };
        let weights = m31_vec(&[
            1, 2, 3, 4, // channel 0
            5, 6, 7, 8, // channel 1
        ]);

        let output = qwen35_depthwise_conv1d_output(&input, &weights, 4).unwrap();
        assert_eq!(output.rows, 3);
        assert_eq!(output.cols, 2);
        assert_eq!(output.get(0, 0), M31::from(4));
        assert_eq!(output.get(1, 0), M31::from(11));
        assert_eq!(output.get(2, 0), M31::from(20));
        assert_eq!(output.get(0, 1), M31::from(80));
        assert_eq!(output.get(1, 1), M31::from(230));
        assert_eq!(output.get(2, 1), M31::from(440));

        let witness = qwen35_depthwise_conv1d_witness(&input, &weights, 4).unwrap();
        verify_qwen35_depthwise_conv1d_witness(&input, &weights, 4, &witness).unwrap();
        assert_eq!(witness[0].valid_taps, m31_vec(&[0, 0, 0, 1]));
        assert_eq!(witness[2].valid_taps, m31_vec(&[0, 0, 1, 1]));
        assert_eq!(witness[4].valid_taps, m31_vec(&[0, 1, 1, 1]));
    }

    #[test]
    fn qwen35_depthwise_conv1d_witness_rejects_tampered_output() {
        let input = M31Matrix {
            rows: 2,
            cols: 1,
            data: m31_vec(&[7, 11]),
        };
        let weights = m31_vec(&[1, 2, 3, 4]);
        let mut witness = qwen35_depthwise_conv1d_witness(&input, &weights, 4).unwrap();
        witness[1].output += M31::from(1);

        let err =
            verify_qwen35_depthwise_conv1d_witness(&input, &weights, 4, &witness).unwrap_err();
        assert!(err.contains("output violates product sum"));
    }

    #[test]
    fn qwen35_depthwise_conv1d_witness_rejects_swapped_rows() {
        let input = M31Matrix {
            rows: 2,
            cols: 2,
            data: m31_vec(&[1, 2, 3, 4]),
        };
        let weights = m31_vec(&[1, 1, 1, 1, 2, 2, 2, 2]);
        let mut witness = qwen35_depthwise_conv1d_witness(&input, &weights, 4).unwrap();
        witness.swap(1, 2);

        let err =
            verify_qwen35_depthwise_conv1d_witness(&input, &weights, 4, &witness).unwrap_err();
        assert!(err.contains("expected token"));
    }

    #[test]
    fn qwen35_depthwise_conv1d_air_constraints_hold_on_trace() {
        use num_traits::Zero;
        use stwo::core::pcs::TreeVec;
        use stwo_constraint_framework::assert_constraints_on_trace;

        let input = M31Matrix {
            rows: 3,
            cols: 2,
            data: m31_vec(&[1, 10, 2, 20, 3, 30]),
        };
        let weights = m31_vec(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let trace = qwen35_depthwise_conv1d_trace(&input, &weights, 4).unwrap();
        let preprocessed = trace.preprocessed.iter().collect::<Vec<_>>();
        let execution = trace.execution.iter().collect::<Vec<_>>();
        let trees = TreeVec::new(vec![preprocessed, execution]);
        let eval = Qwen35DepthwiseConv1dEval {
            log_n_rows: trace.log_size,
            kernel: 4,
            instance_id: 0,
        };

        assert_constraints_on_trace(
            &trees,
            trace.log_size,
            |row_eval| {
                let _ = eval.evaluate(row_eval);
            },
            stwo::core::fields::qm31::SecureField::zero(),
        );
    }

    #[test]
    fn qwen35_depthwise_conv1d_air_rejects_swapped_execution_rows() {
        use num_traits::Zero;
        use stwo::core::pcs::TreeVec;
        use stwo_constraint_framework::assert_constraints_on_trace;

        let input = M31Matrix {
            rows: 2,
            cols: 2,
            data: m31_vec(&[1, 2, 3, 4]),
        };
        let weights = m31_vec(&[1, 1, 1, 1, 2, 2, 2, 2]);
        let mut trace = qwen35_depthwise_conv1d_trace(&input, &weights, 4).unwrap();
        for col in &mut trace.execution {
            col.swap(1, 2);
        }
        let preprocessed = trace.preprocessed.iter().collect::<Vec<_>>();
        let execution = trace.execution.iter().collect::<Vec<_>>();
        let trees = TreeVec::new(vec![preprocessed, execution]);
        let eval = Qwen35DepthwiseConv1dEval {
            log_n_rows: trace.log_size,
            kernel: 4,
            instance_id: 0,
        };

        let result = std::panic::catch_unwind(|| {
            assert_constraints_on_trace(
                &trees,
                trace.log_size,
                |row_eval| {
                    let _ = eval.evaluate(row_eval);
                },
                stwo::core::fields::qm31::SecureField::zero(),
            );
        });
        assert!(result.is_err());
    }

    #[test]
    fn qwen35_depthwise_conv1d_air_proves_and_verifies_standalone() {
        let input = M31Matrix {
            rows: 3,
            cols: 2,
            data: m31_vec(&[1, 10, 2, 20, 3, 30]),
        };
        let weights = m31_vec(&[1, 2, 3, 4, 5, 6, 7, 8]);

        let proof = prove_qwen35_depthwise_conv1d_air(&input, &weights, 4).unwrap();
        assert_eq!(proof.kernel, 4);
        assert_eq!(proof.n_real_rows, 6);
        verify_qwen35_depthwise_conv1d_air(&proof).unwrap();
    }

    #[test]
    fn qwen35_depthwise_conv1d_statement_binds_io_weight_and_layer() {
        let input = M31Matrix {
            rows: 2,
            cols: 2,
            data: m31_vec(&[1, 2, 3, 4]),
        };
        let weights = m31_vec(&[1, 1, 1, 1, 2, 2, 2, 2]);
        let output = qwen35_depthwise_conv1d_output(&input, &weights, 4).unwrap();
        let statement = qwen35_depthwise_conv1d_statement(7, &input, &weights, &output, 4).unwrap();

        let mut tampered_input = input.clone();
        tampered_input.data[0] += M31::from(1);
        let tampered_input_statement =
            qwen35_depthwise_conv1d_statement(7, &tampered_input, &weights, &output, 4).unwrap();
        assert_ne!(
            statement.statement_hash,
            tampered_input_statement.statement_hash
        );

        let mut tampered_weights = weights.clone();
        tampered_weights[0] += M31::from(1);
        let tampered_weight_statement =
            qwen35_depthwise_conv1d_statement(7, &input, &tampered_weights, &output, 4).unwrap();
        assert_ne!(
            statement.statement_hash,
            tampered_weight_statement.statement_hash
        );

        let mut tampered_output = output.clone();
        tampered_output.data[0] += M31::from(1);
        let tampered_output_statement =
            qwen35_depthwise_conv1d_statement(7, &input, &weights, &tampered_output, 4).unwrap();
        assert_ne!(
            statement.statement_hash,
            tampered_output_statement.statement_hash
        );

        let other_layer_statement =
            qwen35_depthwise_conv1d_statement(8, &input, &weights, &output, 4).unwrap();
        assert_ne!(
            statement.statement_hash,
            other_layer_statement.statement_hash
        );
    }

    #[test]
    fn qwen35_depthwise_conv1d_air_rejects_wrong_statement_hash() {
        let input = M31Matrix {
            rows: 3,
            cols: 2,
            data: m31_vec(&[1, 10, 2, 20, 3, 30]),
        };
        let weights = m31_vec(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let proof = prove_qwen35_depthwise_conv1d_air_for_layer(2, &input, &weights, 4).unwrap();
        verify_qwen35_depthwise_conv1d_air_with_statement_hash(
            &proof,
            proof.statement.statement_hash,
        )
        .unwrap();

        let output = qwen35_depthwise_conv1d_output(&input, &weights, 4).unwrap();
        let wrong_statement = qwen35_depthwise_conv1d_statement(3, &input, &weights, &output, 4)
            .unwrap()
            .statement_hash;
        let err = verify_qwen35_depthwise_conv1d_air_with_statement_hash(&proof, wrong_statement)
            .unwrap_err();
        assert!(err.to_string().contains("statement hash mismatch"));
    }
}
