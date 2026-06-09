//! Qwen3.5 GatedDeltaNet NormAndZGate witness and AIR.
//!
//! This component proves the final linear-attention gating step:
//! `gated_value = rmsnorm(attended_value, norm_weight) * z_gate`.

use crate::backend::convert_evaluations;
use crate::components::matmul::M31Matrix;
use crate::components::rmsnorm::{build_rsqrt_table, RMSNormRelation};
use crate::gadgets::lookup_table::PrecomputedTable;
use starknet_ff::FieldElement;
use stwo::core::air::Component;
use stwo::core::channel::{Channel, MerkleChannel};
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::vcs_lifted::MerkleHasherLifted;
use stwo::core::verifier::verify as stwo_verify;
use stwo::prover::backend::simd::m31::PackedBaseField;
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::prove;
use stwo::prover::CommitmentSchemeProver;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, LogupTraceGenerator, RelationEntry,
    TraceLocationAllocator,
};

pub const QWEN35_NORM_AND_Z_GATE_MIN_LOG_SIZE: u32 = 4;
const DOMAIN_QWEN35_NORM_AND_Z_GATE_MATRIX: u64 = 0x51334e_4d4154; // "Q3N_MAT"
const DOMAIN_QWEN35_NORM_AND_Z_GATE_VECTOR: u64 = 0x51334e_564543; // "Q3N_VEC"
const DOMAIN_QWEN35_NORM_AND_Z_GATE_STATEMENT: u64 = 0x51334e_53544d; // "Q3N_STM"
const DOMAIN_QWEN35_NORM_AND_Z_GATE_CHANNEL: u64 = 0x51334e_43484e; // "Q3N_CHN"
const QWEN35_NORM_AND_Z_GATE_TRACE_CHECKSUM_ALPHA: u32 = 65_537;

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

#[derive(Debug, Clone)]
pub struct Qwen35NormAndZGateEval {
    pub log_n_rows: u32,
    pub instance_id: usize,
    pub lookup_elements: RMSNormRelation,
    pub claimed_sum: SecureField,
    pub trace_checksum: M31,
}

pub type Qwen35NormAndZGateComponent = FrameworkComponent<Qwen35NormAndZGateEval>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35NormAndZGateStatement {
    pub layer_idx: usize,
    pub seq_len: usize,
    pub value_heads: usize,
    pub head_dim: usize,
    pub table_log_size: u32,
    pub trace_checksum: M31,
    pub table_commitment: FieldElement,
    pub attended_value_commitment: FieldElement,
    pub norm_weight_commitment: FieldElement,
    pub z_gate_commitment: FieldElement,
    pub output_commitment: FieldElement,
    pub statement_hash: FieldElement,
}

#[derive(Debug)]
pub struct Qwen35NormAndZGateProof<H: MerkleHasherLifted> {
    pub stark_proof: StarkProof<H>,
    pub claimed_sum: SecureField,
    pub log_size: u32,
    pub n_real_rows: usize,
    pub statement: Qwen35NormAndZGateStatement,
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen35NormAndZGateProofError {
    #[error("Witness error: {0}")]
    Witness(String),
    #[error("Proving error: {0}")]
    Proving(String),
    #[error("Verification error: {0}")]
    Verification(String),
}

#[derive(Debug)]
struct Qwen35NormAndZGateColumns {
    preprocessed: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    execution: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    table_rms_sq_col: Col<SimdBackend, BaseField>,
    table_rsqrt_col: Col<SimdBackend, BaseField>,
    trace_rms_sq_col: Col<SimdBackend, BaseField>,
    trace_rsqrt_col: Col<SimdBackend, BaseField>,
    multiplicities: Vec<M31>,
    trace_checksum: M31,
    log_size: u32,
    n_real_rows: usize,
}

impl FrameworkEval for Qwen35NormAndZGateEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let table_rms_sq = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_norm_and_z_gate_preprocessed_id(self.instance_id, "table_rms_sq").into(),
        });
        let table_rsqrt = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_norm_and_z_gate_preprocessed_id(self.instance_id, "table_rsqrt").into(),
        });
        let expected_token = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_norm_and_z_gate_preprocessed_id(self.instance_id, "token").into(),
        });
        let expected_head = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_norm_and_z_gate_preprocessed_id(self.instance_id, "head").into(),
        });
        let expected_dim = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_norm_and_z_gate_preprocessed_id(self.instance_id, "dim").into(),
        });
        let is_first_dim = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_norm_and_z_gate_preprocessed_id(self.instance_id, "is_first_dim").into(),
        });
        let is_last_dim = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_norm_and_z_gate_preprocessed_id(self.instance_id, "is_last_dim").into(),
        });
        let is_dim_chain = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_norm_and_z_gate_preprocessed_id(self.instance_id, "is_dim_chain").into(),
        });
        let is_first = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_norm_and_z_gate_preprocessed_id(self.instance_id, "is_first").into(),
        });
        let is_last = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_norm_and_z_gate_preprocessed_id(self.instance_id, "is_last").into(),
        });
        let has_next = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_norm_and_z_gate_preprocessed_id(self.instance_id, "has_next").into(),
        });

        let token = eval.next_trace_mask();
        let head = eval.next_trace_mask();
        let dim = eval.next_trace_mask();
        let attended_value = eval.next_trace_mask();
        let norm_weight = eval.next_trace_mask();
        let z_gate = eval.next_trace_mask();
        let output = eval.next_trace_mask();
        let sq_term = eval.next_trace_mask();
        let sq_prefix_before = eval.next_trace_mask();
        let sq_prefix_after = eval.next_trace_mask();
        let shifted_next_sq_prefix_before = eval.next_trace_mask();
        let rms_sq = eval.next_trace_mask();
        let rsqrt = eval.next_trace_mask();
        let shifted_next_rms_sq = eval.next_trace_mask();
        let shifted_next_rsqrt = eval.next_trace_mask();
        let normed_value = eval.next_trace_mask();
        let gated_value = eval.next_trace_mask();
        let trace_acc_before = eval.next_trace_mask();
        let trace_acc_after = eval.next_trace_mask();
        let shifted_next_trace_acc_before = eval.next_trace_mask();
        let multiplicity = eval.next_trace_mask();

        eval.add_constraint(token.clone() - expected_token);
        eval.add_constraint(head.clone() - expected_head);
        eval.add_constraint(dim.clone() - expected_dim);
        eval.add_constraint(sq_term.clone() - attended_value.clone() * attended_value.clone());
        eval.add_constraint(sq_prefix_after.clone() - sq_prefix_before.clone() - sq_term.clone());
        eval.add_constraint(is_first_dim.clone() * sq_prefix_before.clone());
        eval.add_constraint(
            is_dim_chain.clone() * (shifted_next_sq_prefix_before - sq_prefix_after.clone()),
        );
        eval.add_constraint(is_last_dim.clone() * (sq_prefix_after.clone() - rms_sq.clone()));
        eval.add_constraint(is_dim_chain.clone() * (shifted_next_rms_sq - rms_sq.clone()));
        eval.add_constraint(is_dim_chain.clone() * (shifted_next_rsqrt - rsqrt.clone()));
        eval.add_constraint(
            normed_value.clone() - attended_value.clone() * rsqrt.clone() * norm_weight.clone(),
        );
        eval.add_constraint(gated_value.clone() - normed_value * z_gate.clone());
        eval.add_constraint(output.clone() - gated_value.clone());

        let alpha = E::F::from(BaseField::from(M31::from(
            QWEN35_NORM_AND_Z_GATE_TRACE_CHECKSUM_ALPHA,
        )));
        let row_term = E::F::from(BaseField::from(M31::from(193u32)))
            + token * E::F::from(BaseField::from(M31::from(3u32)))
            + head * E::F::from(BaseField::from(M31::from(5u32)))
            + dim * E::F::from(BaseField::from(M31::from(7u32)))
            + attended_value * E::F::from(BaseField::from(M31::from(11u32)))
            + norm_weight * E::F::from(BaseField::from(M31::from(13u32)))
            + z_gate * E::F::from(BaseField::from(M31::from(17u32)))
            + output * E::F::from(BaseField::from(M31::from(19u32)))
            + sq_term * E::F::from(BaseField::from(M31::from(23u32)))
            + sq_prefix_before * E::F::from(BaseField::from(M31::from(29u32)))
            + sq_prefix_after * E::F::from(BaseField::from(M31::from(31u32)))
            + rms_sq.clone() * E::F::from(BaseField::from(M31::from(37u32)))
            + rsqrt.clone() * E::F::from(BaseField::from(M31::from(41u32)))
            + gated_value * E::F::from(BaseField::from(M31::from(43u32)));
        eval.add_constraint(is_first * trace_acc_before.clone());
        eval.add_constraint(trace_acc_after.clone() - trace_acc_before.clone() * alpha - row_term);
        eval.add_constraint(has_next * (shifted_next_trace_acc_before - trace_acc_after.clone()));
        eval.add_constraint(
            is_last
                * (trace_acc_after - E::F::from(BaseField::from(M31::from(self.trace_checksum.0)))),
        );

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::from(multiplicity),
            &[table_rms_sq, table_rsqrt],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::from(E::F::from(BaseField::from(M31::from(1u32)))),
            &[rms_sq, rsqrt],
        ));
        eval.finalize_logup_in_pairs();

        eval
    }
}

pub fn qwen35_norm_and_z_gate_preprocessed_id(instance_id: usize, name: &str) -> String {
    format!("qwen35_norm_and_z_gate_{name}_{instance_id}")
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

fn m31_vector_commitment(domain: u64, values: &[M31]) -> FieldElement {
    let mut felts = Vec::with_capacity(3 + values.len());
    felts.push(FieldElement::from(domain));
    felts.push(FieldElement::from(values.len() as u64));
    for value in values {
        felts.push(FieldElement::from(value.0 as u64));
    }
    starknet_crypto::poseidon_hash_many(&felts)
}

pub fn qwen35_norm_and_z_gate_attended_value_commitment(input: &M31Matrix) -> FieldElement {
    m31_matrix_commitment(DOMAIN_QWEN35_NORM_AND_Z_GATE_MATRIX, input)
}

pub fn qwen35_norm_and_z_gate_z_gate_commitment(z_gate: &M31Matrix) -> FieldElement {
    m31_matrix_commitment(DOMAIN_QWEN35_NORM_AND_Z_GATE_MATRIX + 1, z_gate)
}

pub fn qwen35_norm_and_z_gate_output_commitment(output: &M31Matrix) -> FieldElement {
    m31_matrix_commitment(DOMAIN_QWEN35_NORM_AND_Z_GATE_MATRIX + 2, output)
}

pub fn qwen35_norm_and_z_gate_norm_weight_commitment(norm_weight: &[M31]) -> FieldElement {
    m31_vector_commitment(DOMAIN_QWEN35_NORM_AND_Z_GATE_VECTOR, norm_weight)
}

pub fn qwen35_norm_and_z_gate_table_commitment(table_log_size: u32) -> FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(DOMAIN_QWEN35_NORM_AND_Z_GATE_STATEMENT),
        FieldElement::from(table_log_size as u64),
    ])
}

pub fn qwen35_norm_and_z_gate_statement_hash(
    layer_idx: usize,
    seq_len: usize,
    value_heads: usize,
    head_dim: usize,
    table_log_size: u32,
    trace_checksum: M31,
    table_commitment: FieldElement,
    attended_value_commitment: FieldElement,
    norm_weight_commitment: FieldElement,
    z_gate_commitment: FieldElement,
    output_commitment: FieldElement,
) -> FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(DOMAIN_QWEN35_NORM_AND_Z_GATE_STATEMENT),
        FieldElement::from(layer_idx as u64),
        FieldElement::from(seq_len as u64),
        FieldElement::from(value_heads as u64),
        FieldElement::from(head_dim as u64),
        FieldElement::from(table_log_size as u64),
        FieldElement::from(trace_checksum.0 as u64),
        table_commitment,
        attended_value_commitment,
        norm_weight_commitment,
        z_gate_commitment,
        output_commitment,
    ])
}

fn validate_shapes(
    attended_value: &M31Matrix,
    norm_weight: &[M31],
    z_gate: &M31Matrix,
    output: Option<&M31Matrix>,
    head_dim: usize,
) -> Result<(), String> {
    if attended_value.rows == 0 || attended_value.cols == 0 {
        return Err("NormAndZGate attended_value must be non-empty".to_string());
    }
    if head_dim == 0 || attended_value.cols % head_dim != 0 {
        return Err(format!(
            "NormAndZGate width {} must divide by head_dim {}",
            attended_value.cols, head_dim
        ));
    }
    if norm_weight.len() != head_dim {
        return Err(format!(
            "NormAndZGate norm_weight length {} != head_dim {}",
            norm_weight.len(),
            head_dim
        ));
    }
    if z_gate.rows != attended_value.rows || z_gate.cols != attended_value.cols {
        return Err(format!(
            "NormAndZGate z_gate shape [{}x{}] != attended_value [{}x{}]",
            z_gate.rows, z_gate.cols, attended_value.rows, attended_value.cols
        ));
    }
    if let Some(output) = output {
        if output.rows != attended_value.rows || output.cols != attended_value.cols {
            return Err(format!(
                "NormAndZGate output shape [{}x{}] != attended_value [{}x{}]",
                output.rows, output.cols, attended_value.rows, attended_value.cols
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn trace_row_term(
    token: M31,
    head: M31,
    dim: M31,
    attended_value: M31,
    norm_weight: M31,
    z_gate: M31,
    output: M31,
    sq_term: M31,
    prefix_before: M31,
    prefix_after: M31,
    rms_sq: M31,
    rsqrt: M31,
    gated_value: M31,
) -> M31 {
    M31::from(193u32)
        + token * M31::from(3u32)
        + head * M31::from(5u32)
        + dim * M31::from(7u32)
        + attended_value * M31::from(11u32)
        + norm_weight * M31::from(13u32)
        + z_gate * M31::from(17u32)
        + output * M31::from(19u32)
        + sq_term * M31::from(23u32)
        + prefix_before * M31::from(29u32)
        + prefix_after * M31::from(31u32)
        + rms_sq * M31::from(37u32)
        + rsqrt * M31::from(41u32)
        + gated_value * M31::from(43u32)
}

pub fn qwen35_norm_and_z_gate_output(
    attended_value: &M31Matrix,
    norm_weight: &[M31],
    z_gate: &M31Matrix,
    head_dim: usize,
    table_log_size: u32,
) -> Result<M31Matrix, String> {
    validate_shapes(attended_value, norm_weight, z_gate, None, head_dim)?;
    let table = build_rsqrt_table(table_log_size);
    let value_heads = attended_value.cols / head_dim;
    let mut output = M31Matrix::new(attended_value.rows, attended_value.cols);
    for token_idx in 0..attended_value.rows {
        for head_idx in 0..value_heads {
            let mut sum_sq = M31::from(0u32);
            for dim_idx in 0..head_dim {
                let col_idx = head_idx * head_dim + dim_idx;
                let value = attended_value.get(token_idx, col_idx);
                sum_sq += value * value;
            }
            let rsqrt = table.lookup(sum_sq).ok_or_else(|| {
                format!(
                    "NormAndZGate rms_sq {} is outside rsqrt table domain",
                    sum_sq.0
                )
            })?;
            for dim_idx in 0..head_dim {
                let col_idx = head_idx * head_dim + dim_idx;
                let normed = attended_value.get(token_idx, col_idx) * rsqrt * norm_weight[dim_idx];
                output.set(token_idx, col_idx, normed * z_gate.get(token_idx, col_idx));
            }
        }
    }
    Ok(output)
}

pub fn qwen35_norm_and_z_gate_trace_checksum(
    attended_value: &M31Matrix,
    norm_weight: &[M31],
    z_gate: &M31Matrix,
    output: &M31Matrix,
    head_dim: usize,
    table_log_size: u32,
) -> Result<M31, String> {
    validate_shapes(attended_value, norm_weight, z_gate, Some(output), head_dim)?;
    let value_heads = attended_value.cols / head_dim;
    let n_real_rows = attended_value.rows * value_heads * head_dim;
    let log_size = table_log_size
        .max(n_real_rows.max(1).next_power_of_two().trailing_zeros())
        .max(QWEN35_NORM_AND_Z_GATE_MIN_LOG_SIZE);
    let size = 1usize << log_size;
    let table = build_rsqrt_table(log_size);
    let mut checksum = M31::from(0u32);

    for token_idx in 0..attended_value.rows {
        for head_idx in 0..value_heads {
            let mut sum_sq = M31::from(0u32);
            for dim_idx in 0..head_dim {
                let col_idx = head_idx * head_dim + dim_idx;
                let value = attended_value.get(token_idx, col_idx);
                sum_sq += value * value;
            }
            let rsqrt = table.lookup(sum_sq).ok_or_else(|| {
                format!(
                    "NormAndZGate rms_sq {} is outside rsqrt table domain",
                    sum_sq.0
                )
            })?;
            let mut prefix = M31::from(0u32);
            for dim_idx in 0..head_dim {
                let col_idx = head_idx * head_dim + dim_idx;
                let value = attended_value.get(token_idx, col_idx);
                let sq_term = value * value;
                let next_prefix = prefix + sq_term;
                let normed = value * rsqrt * norm_weight[dim_idx];
                let expected_output = normed * z_gate.get(token_idx, col_idx);
                let actual_output = output.get(token_idx, col_idx);
                if actual_output != expected_output {
                    return Err(format!(
                        "NormAndZGate output mismatch at token={token_idx} head={head_idx} dim={dim_idx}: got {}, expected {}",
                        actual_output.0, expected_output.0
                    ));
                }
                checksum = checksum * M31::from(QWEN35_NORM_AND_Z_GATE_TRACE_CHECKSUM_ALPHA)
                    + trace_row_term(
                        M31::from(token_idx as u32),
                        M31::from(head_idx as u32),
                        M31::from(dim_idx as u32),
                        value,
                        norm_weight[dim_idx],
                        z_gate.get(token_idx, col_idx),
                        actual_output,
                        sq_term,
                        prefix,
                        next_prefix,
                        sum_sq,
                        rsqrt,
                        expected_output,
                    );
                prefix = next_prefix;
            }
        }
    }

    let pad_rms_sq = table.inputs[0];
    let pad_rsqrt = table.outputs[0];
    for _ in n_real_rows..size {
        checksum = checksum * M31::from(QWEN35_NORM_AND_Z_GATE_TRACE_CHECKSUM_ALPHA)
            + trace_row_term(
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                pad_rms_sq,
                pad_rsqrt,
                M31::from(0u32),
            );
    }
    Ok(checksum)
}

pub fn qwen35_norm_and_z_gate_statement(
    layer_idx: usize,
    attended_value: &M31Matrix,
    norm_weight: &[M31],
    z_gate: &M31Matrix,
    output: &M31Matrix,
    head_dim: usize,
    table_log_size: u32,
) -> Result<Qwen35NormAndZGateStatement, String> {
    validate_shapes(attended_value, norm_weight, z_gate, Some(output), head_dim)?;
    let value_heads = attended_value.cols / head_dim;
    let trace_checksum = qwen35_norm_and_z_gate_trace_checksum(
        attended_value,
        norm_weight,
        z_gate,
        output,
        head_dim,
        table_log_size,
    )?;
    let table_commitment = qwen35_norm_and_z_gate_table_commitment(table_log_size);
    let attended_value_commitment =
        qwen35_norm_and_z_gate_attended_value_commitment(attended_value);
    let norm_weight_commitment = qwen35_norm_and_z_gate_norm_weight_commitment(norm_weight);
    let z_gate_commitment = qwen35_norm_and_z_gate_z_gate_commitment(z_gate);
    let output_commitment = qwen35_norm_and_z_gate_output_commitment(output);
    let statement_hash = qwen35_norm_and_z_gate_statement_hash(
        layer_idx,
        attended_value.rows,
        value_heads,
        head_dim,
        table_log_size,
        trace_checksum,
        table_commitment,
        attended_value_commitment,
        norm_weight_commitment,
        z_gate_commitment,
        output_commitment,
    );
    Ok(Qwen35NormAndZGateStatement {
        layer_idx,
        seq_len: attended_value.rows,
        value_heads,
        head_dim,
        table_log_size,
        trace_checksum,
        table_commitment,
        attended_value_commitment,
        norm_weight_commitment,
        z_gate_commitment,
        output_commitment,
        statement_hash,
    })
}

fn multiplicities(trace_rms_sq: &[M31], table: &PrecomputedTable) -> Vec<M31> {
    let mut multiplicities = vec![M31::from(0u32); table.inputs.len()];
    for &value in trace_rms_sq {
        if let Some(idx) = table.lookup_index(value) {
            multiplicities[idx] += M31::from(1u32);
        }
    }
    multiplicities
}

fn preprocessed_columns(
    log_size: u32,
    seq_len: usize,
    value_heads: usize,
    head_dim: usize,
    table: &PrecomputedTable,
) -> (
    Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    Col<SimdBackend, BaseField>,
    Col<SimdBackend, BaseField>,
) {
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();
    let mut table_rms_sq_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut table_rsqrt_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut token_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut head_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut dim_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut is_first_dim_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut is_last_dim_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut is_dim_chain_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut is_first_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut is_last_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut has_next_col = Col::<SimdBackend, BaseField>::zeros(size);

    for (idx, (&input, &output)) in table.inputs.iter().zip(table.outputs.iter()).enumerate() {
        table_rms_sq_col.set(idx, input);
        table_rsqrt_col.set(idx, output);
    }
    let n_real_rows = seq_len * value_heads * head_dim;
    for row_idx in 0..n_real_rows.min(size) {
        let dim_idx = row_idx % head_dim;
        let head_idx = (row_idx / head_dim) % value_heads;
        let token_idx = row_idx / (value_heads * head_dim);
        token_col.set(row_idx, M31::from(token_idx as u32));
        head_col.set(row_idx, M31::from(head_idx as u32));
        dim_col.set(row_idx, M31::from(dim_idx as u32));
        is_first_dim_col.set(row_idx, M31::from((dim_idx == 0) as u32));
        is_last_dim_col.set(row_idx, M31::from((dim_idx + 1 == head_dim) as u32));
        is_dim_chain_col.set(row_idx, M31::from((dim_idx + 1 < head_dim) as u32));
    }
    is_first_col.set(0, M31::from(1u32));
    is_last_col.set(size - 1, M31::from(1u32));
    for row_idx in 0..size.saturating_sub(1) {
        has_next_col.set(row_idx, M31::from(1u32));
    }

    (
        vec![
            CircleEvaluation::new(domain, table_rms_sq_col.clone()),
            CircleEvaluation::new(domain, table_rsqrt_col.clone()),
            CircleEvaluation::new(domain, token_col),
            CircleEvaluation::new(domain, head_col),
            CircleEvaluation::new(domain, dim_col),
            CircleEvaluation::new(domain, is_first_dim_col),
            CircleEvaluation::new(domain, is_last_dim_col),
            CircleEvaluation::new(domain, is_dim_chain_col),
            CircleEvaluation::new(domain, is_first_col),
            CircleEvaluation::new(domain, is_last_col),
            CircleEvaluation::new(domain, has_next_col),
        ],
        table_rms_sq_col,
        table_rsqrt_col,
    )
}

fn columns(
    attended_value: &M31Matrix,
    norm_weight: &[M31],
    z_gate: &M31Matrix,
    output: &M31Matrix,
    head_dim: usize,
    table_log_size: u32,
) -> Result<Qwen35NormAndZGateColumns, String> {
    validate_shapes(attended_value, norm_weight, z_gate, Some(output), head_dim)?;
    let value_heads = attended_value.cols / head_dim;
    let n_real_rows = attended_value.rows * value_heads * head_dim;
    let log_size = table_log_size
        .max(n_real_rows.max(1).next_power_of_two().trailing_zeros())
        .max(QWEN35_NORM_AND_Z_GATE_MIN_LOG_SIZE);
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();
    let table = build_rsqrt_table(log_size);
    let (preprocessed, table_rms_sq_col, table_rsqrt_col) =
        preprocessed_columns(log_size, attended_value.rows, value_heads, head_dim, &table);

    let mut execution_cols = (0..21)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(size))
        .collect::<Vec<_>>();
    let mut trace_acc_before_values = vec![M31::from(0u32); size];
    let mut trace_acc_after_values = vec![M31::from(0u32); size];
    let mut trace_rms_values = vec![table.inputs[0]; size];
    let mut trace_rsqrt_values = vec![table.outputs[0]; size];
    let mut trace_checksum = M31::from(0u32);

    for token_idx in 0..attended_value.rows {
        for head_idx in 0..value_heads {
            let mut sum_sq = M31::from(0u32);
            for dim_idx in 0..head_dim {
                let col_idx = head_idx * head_dim + dim_idx;
                let value = attended_value.get(token_idx, col_idx);
                sum_sq += value * value;
            }
            let rsqrt = table.lookup(sum_sq).ok_or_else(|| {
                format!(
                    "NormAndZGate rms_sq {} is outside rsqrt table domain",
                    sum_sq.0
                )
            })?;
            let mut prefix = M31::from(0u32);
            for dim_idx in 0..head_dim {
                let row_idx = ((token_idx * value_heads + head_idx) * head_dim) + dim_idx;
                let col_idx = head_idx * head_dim + dim_idx;
                let value = attended_value.get(token_idx, col_idx);
                let sq_term = value * value;
                let next_prefix = prefix + sq_term;
                let z_value = z_gate.get(token_idx, col_idx);
                let normed = value * rsqrt * norm_weight[dim_idx];
                let gated = normed * z_value;
                let actual_output = output.get(token_idx, col_idx);
                if actual_output != gated {
                    return Err(format!(
                        "NormAndZGate output mismatch at token={token_idx} head={head_idx} dim={dim_idx}: got {}, expected {}",
                        actual_output.0, gated.0
                    ));
                }

                trace_acc_before_values[row_idx] = trace_checksum;
                let values = [
                    M31::from(token_idx as u32),
                    M31::from(head_idx as u32),
                    M31::from(dim_idx as u32),
                    value,
                    norm_weight[dim_idx],
                    z_value,
                    actual_output,
                    sq_term,
                    prefix,
                    next_prefix,
                    next_prefix,
                    sum_sq,
                    rsqrt,
                    sum_sq,
                    rsqrt,
                    normed,
                    gated,
                ];
                for (idx, value) in values.iter().copied().enumerate() {
                    execution_cols[idx].set(row_idx, value);
                }
                trace_rms_values[row_idx] = sum_sq;
                trace_rsqrt_values[row_idx] = rsqrt;
                trace_checksum = trace_checksum
                    * M31::from(QWEN35_NORM_AND_Z_GATE_TRACE_CHECKSUM_ALPHA)
                    + trace_row_term(
                        M31::from(token_idx as u32),
                        M31::from(head_idx as u32),
                        M31::from(dim_idx as u32),
                        value,
                        norm_weight[dim_idx],
                        z_value,
                        actual_output,
                        sq_term,
                        prefix,
                        next_prefix,
                        sum_sq,
                        rsqrt,
                        gated,
                    );
                trace_acc_after_values[row_idx] = trace_checksum;
                prefix = next_prefix;
            }
        }
    }

    for row_idx in n_real_rows..size {
        trace_acc_before_values[row_idx] = trace_checksum;
        execution_cols[11].set(row_idx, table.inputs[0]);
        execution_cols[12].set(row_idx, table.outputs[0]);
        execution_cols[13].set(row_idx, table.inputs[0]);
        execution_cols[14].set(row_idx, table.outputs[0]);
        trace_checksum = trace_checksum * M31::from(QWEN35_NORM_AND_Z_GATE_TRACE_CHECKSUM_ALPHA)
            + trace_row_term(
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                table.inputs[0],
                table.outputs[0],
                M31::from(0u32),
            );
        trace_acc_after_values[row_idx] = trace_checksum;
    }
    for idx in 0..size {
        execution_cols[17].set(idx, trace_acc_before_values[idx]);
        execution_cols[18].set(idx, trace_acc_after_values[idx]);
        if idx + 1 < size {
            execution_cols[19].set(idx, trace_acc_before_values[idx + 1]);
        }
    }

    let mults = multiplicities(&trace_rms_values, &table);
    for (idx, &multiplicity) in mults.iter().enumerate().take(size) {
        execution_cols[20].set(idx, multiplicity);
    }
    let execution = execution_cols
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect();

    Ok(Qwen35NormAndZGateColumns {
        preprocessed,
        execution,
        table_rms_sq_col,
        table_rsqrt_col,
        trace_rms_sq_col: {
            let mut col = Col::<SimdBackend, BaseField>::zeros(size);
            for (idx, value) in trace_rms_values.iter().copied().enumerate() {
                col.set(idx, value);
            }
            col
        },
        trace_rsqrt_col: {
            let mut col = Col::<SimdBackend, BaseField>::zeros(size);
            for (idx, value) in trace_rsqrt_values.iter().copied().enumerate() {
                col.set(idx, value);
            }
            col
        },
        multiplicities: mults,
        trace_checksum,
        log_size,
        n_real_rows,
    })
}

fn logup_trace(
    columns: &Qwen35NormAndZGateColumns,
    lookup_elements: &RMSNormRelation,
) -> (
    Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    use stwo::prover::backend::simd::m31::LOG_N_LANES;

    let size = 1usize << columns.log_size;
    let vec_size = size >> LOG_N_LANES;
    let mut logup_gen = LogupTraceGenerator::new(columns.log_size);
    let mut col_gen = logup_gen.new_col();
    for vec_row in 0..vec_size {
        let q_table: PackedSecureField = lookup_elements.lookup_elements().combine(&[
            columns.table_rms_sq_col.data[vec_row],
            columns.table_rsqrt_col.data[vec_row],
        ]);
        let q_trace: PackedSecureField = lookup_elements.lookup_elements().combine(&[
            columns.trace_rms_sq_col.data[vec_row],
            columns.trace_rsqrt_col.data[vec_row],
        ]);
        let mult_packed = mult_packed(&columns.multiplicities, vec_row);
        let numerator = q_table - mult_packed * q_trace;
        let denominator = q_table * q_trace;
        col_gen.write_frac(vec_row, numerator, denominator);
    }
    col_gen.finalize_col();
    logup_gen.finalize_last()
}

fn mult_packed(multiplicities: &[M31], vec_row: usize) -> PackedSecureField {
    let base = vec_row * 16;
    let mut values = [M31::from(0u32); 16];
    for (idx, value) in values.iter_mut().enumerate() {
        let multiplicity_idx = base + idx;
        if multiplicity_idx < multiplicities.len() {
            *value = multiplicities[multiplicity_idx];
        }
    }
    PackedBaseField::from_array(std::array::from_fn(|idx| values[idx])).into()
}

fn mix_statement_hash<C: Channel>(channel: &mut C, statement_hash: FieldElement) {
    channel.mix_u64(DOMAIN_QWEN35_NORM_AND_Z_GATE_CHANNEL);
    let bytes = statement_hash.to_bytes_be();
    channel.mix_u64(u64::from_be_bytes(bytes[0..8].try_into().unwrap()));
    channel.mix_u64(u64::from_be_bytes(bytes[8..16].try_into().unwrap()));
    channel.mix_u64(u64::from_be_bytes(bytes[16..24].try_into().unwrap()));
    channel.mix_u64(u64::from_be_bytes(bytes[24..32].try_into().unwrap()));
}

fn expected_preprocessed_root(
    statement: &Qwen35NormAndZGateStatement,
) -> <Blake2sHash as MerkleHasherLifted>::Hash {
    let table = build_rsqrt_table(statement.table_log_size);
    let (preprocessed, _, _) = preprocessed_columns(
        statement.table_log_size,
        statement.seq_len,
        statement.value_heads,
        statement.head_dim,
        &table,
    );
    let pcs_config = PcsConfig::default();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(statement.table_log_size + 1 + pcs_config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(pcs_config, &twiddles);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, BaseField>(
        preprocessed,
    ));
    tree_builder.commit(channel);
    commitment_scheme.roots()[0]
}

pub fn prove_qwen35_norm_and_z_gate_air(
    layer_idx: usize,
    attended_value: &M31Matrix,
    norm_weight: &[M31],
    z_gate: &M31Matrix,
    head_dim: usize,
    table_log_size: u32,
) -> Result<Qwen35NormAndZGateProof<Blake2sHash>, Qwen35NormAndZGateProofError> {
    let output = qwen35_norm_and_z_gate_output(
        attended_value,
        norm_weight,
        z_gate,
        head_dim,
        table_log_size,
    )
    .map_err(Qwen35NormAndZGateProofError::Witness)?;
    let statement = qwen35_norm_and_z_gate_statement(
        layer_idx,
        attended_value,
        norm_weight,
        z_gate,
        &output,
        head_dim,
        table_log_size,
    )
    .map_err(Qwen35NormAndZGateProofError::Witness)?;
    let columns = columns(
        attended_value,
        norm_weight,
        z_gate,
        &output,
        head_dim,
        table_log_size,
    )
    .map_err(Qwen35NormAndZGateProofError::Witness)?;
    if columns.log_size != statement.table_log_size {
        return Err(Qwen35NormAndZGateProofError::Witness(format!(
            "NormAndZGate table_log_size {} is smaller than required log_size {}",
            statement.table_log_size, columns.log_size
        )));
    }
    if columns.trace_checksum != statement.trace_checksum {
        return Err(Qwen35NormAndZGateProofError::Witness(
            "NormAndZGate trace checksum mismatch".to_string(),
        ));
    }

    let pcs_config = PcsConfig::default();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(columns.log_size + 1 + pcs_config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    mix_statement_hash(channel, statement.statement_hash);
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(pcs_config, &twiddles);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, BaseField>(
        columns.preprocessed.clone(),
    ));
    tree_builder.commit(channel);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, BaseField>(
        columns.execution.clone(),
    ));
    tree_builder.commit(channel);

    let lookup_elements: RMSNormRelation = RMSNormRelation::draw(channel);
    let (interaction_trace, claimed_sum) = logup_trace(&columns, &lookup_elements);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, BaseField>(
        interaction_trace,
    ));
    tree_builder.commit(channel);

    let component = FrameworkComponent::new(
        &mut TraceLocationAllocator::default(),
        Qwen35NormAndZGateEval {
            log_n_rows: columns.log_size,
            instance_id: 0,
            lookup_elements,
            claimed_sum,
            trace_checksum: statement.trace_checksum,
        },
        claimed_sum,
    );
    let stark_proof =
        prove::<SimdBackend, Blake2sMerkleChannel>(&[&component], channel, commitment_scheme)
            .map_err(|err| Qwen35NormAndZGateProofError::Proving(format!("{err:?}")))?;

    Ok(Qwen35NormAndZGateProof {
        stark_proof,
        claimed_sum,
        log_size: columns.log_size,
        n_real_rows: columns.n_real_rows,
        statement,
    })
}

pub fn verify_qwen35_norm_and_z_gate_air(
    proof: &Qwen35NormAndZGateProof<Blake2sHash>,
) -> Result<(), Qwen35NormAndZGateProofError> {
    verify_qwen35_norm_and_z_gate_air_with_statement_hash(proof, proof.statement.statement_hash)
}

pub fn verify_qwen35_norm_and_z_gate_air_with_statement_hash(
    proof: &Qwen35NormAndZGateProof<Blake2sHash>,
    expected_statement_hash: FieldElement,
) -> Result<(), Qwen35NormAndZGateProofError> {
    if proof.statement.statement_hash != expected_statement_hash {
        return Err(Qwen35NormAndZGateProofError::Verification(
            "NormAndZGate statement hash mismatch".to_string(),
        ));
    }
    if proof.log_size != proof.statement.table_log_size {
        return Err(Qwen35NormAndZGateProofError::Verification(
            "NormAndZGate proof log_size must equal statement table_log_size".to_string(),
        ));
    }
    let expected_hash = qwen35_norm_and_z_gate_statement_hash(
        proof.statement.layer_idx,
        proof.statement.seq_len,
        proof.statement.value_heads,
        proof.statement.head_dim,
        proof.statement.table_log_size,
        proof.statement.trace_checksum,
        proof.statement.table_commitment,
        proof.statement.attended_value_commitment,
        proof.statement.norm_weight_commitment,
        proof.statement.z_gate_commitment,
        proof.statement.output_commitment,
    );
    if proof.statement.statement_hash != expected_hash {
        return Err(Qwen35NormAndZGateProofError::Verification(
            "NormAndZGate statement is malformed".to_string(),
        ));
    }
    if proof.statement.table_commitment
        != qwen35_norm_and_z_gate_table_commitment(proof.statement.table_log_size)
    {
        return Err(Qwen35NormAndZGateProofError::Verification(
            "NormAndZGate table commitment mismatch".to_string(),
        ));
    }
    if proof.stark_proof.commitments.len() < 3 {
        return Err(Qwen35NormAndZGateProofError::Verification(format!(
            "expected at least 3 commitment trees for NormAndZGate LogUp, got {}",
            proof.stark_proof.commitments.len()
        )));
    }
    if proof.stark_proof.commitments[0] != expected_preprocessed_root(&proof.statement) {
        return Err(Qwen35NormAndZGateProofError::Verification(
            "NormAndZGate preprocessed root mismatch".to_string(),
        ));
    }

    let pcs_config = PcsConfig::default();
    let mut allocator = TraceLocationAllocator::default();
    let dummy_component = FrameworkComponent::new(
        &mut allocator,
        Qwen35NormAndZGateEval {
            log_n_rows: proof.log_size,
            instance_id: 0,
            lookup_elements: RMSNormRelation::dummy(),
            claimed_sum: proof.claimed_sum,
            trace_checksum: proof.statement.trace_checksum,
        },
        proof.claimed_sum,
    );
    let bounds = Component::trace_log_degree_bounds(&dummy_component);
    if proof.stark_proof.commitments.len() < bounds.len() {
        return Err(Qwen35NormAndZGateProofError::Verification(format!(
            "proof commitment count {} is smaller than trace bound count {}",
            proof.stark_proof.commitments.len(),
            bounds.len()
        )));
    }

    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    mix_statement_hash(channel, expected_statement_hash);
    let mut commitment_scheme = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(pcs_config);
    commitment_scheme.commit(proof.stark_proof.commitments[0], &bounds[0], channel);
    commitment_scheme.commit(proof.stark_proof.commitments[1], &bounds[1], channel);
    let lookup_elements: RMSNormRelation = RMSNormRelation::draw(channel);
    for idx in 2..bounds.len() {
        commitment_scheme.commit(proof.stark_proof.commitments[idx], &bounds[idx], channel);
    }

    let mut allocator = TraceLocationAllocator::default();
    let component = FrameworkComponent::new(
        &mut allocator,
        Qwen35NormAndZGateEval {
            log_n_rows: proof.log_size,
            instance_id: 0,
            lookup_elements,
            claimed_sum: proof.claimed_sum,
            trace_checksum: proof.statement.trace_checksum,
        },
        proof.claimed_sum,
    );
    stwo_verify::<Blake2sMerkleChannel>(
        &[&component as &dyn Component],
        channel,
        &mut commitment_scheme,
        proof.stark_proof.clone(),
    )
    .map_err(|err| Qwen35NormAndZGateProofError::Verification(format!("{err:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, cols: usize, start: u32) -> M31Matrix {
        let data = (0..rows * cols)
            .map(|idx| M31::from(start + idx as u32))
            .collect();
        M31Matrix { rows, cols, data }
    }

    fn filled_matrix(rows: usize, cols: usize, value: u32) -> M31Matrix {
        M31Matrix {
            rows,
            cols,
            data: vec![M31::from(value); rows * cols],
        }
    }

    #[test]
    fn qwen35_norm_and_z_gate_statement_binds_inputs_weight_z_and_output() {
        let attended = filled_matrix(2, 4, 1);
        let norm_weight = vec![M31::from(2u32), M31::from(3u32)];
        let z_gate = matrix(2, 4, 5);
        let output = qwen35_norm_and_z_gate_output(&attended, &norm_weight, &z_gate, 2, 4).unwrap();
        let statement =
            qwen35_norm_and_z_gate_statement(7, &attended, &norm_weight, &z_gate, &output, 2, 4)
                .unwrap();
        let mut tampered_z = z_gate.clone();
        tampered_z.set(0, 0, tampered_z.get(0, 0) + M31::from(1u32));
        let tampered_output =
            qwen35_norm_and_z_gate_output(&attended, &norm_weight, &tampered_z, 2, 4).unwrap();
        let tampered_statement = qwen35_norm_and_z_gate_statement(
            7,
            &attended,
            &norm_weight,
            &tampered_z,
            &tampered_output,
            2,
            4,
        )
        .unwrap();
        assert_ne!(
            statement.z_gate_commitment,
            tampered_statement.z_gate_commitment
        );
        assert_ne!(statement.statement_hash, tampered_statement.statement_hash);

        let mut bad_output = output.clone();
        bad_output.set(0, 0, bad_output.get(0, 0) + M31::from(1u32));
        assert!(qwen35_norm_and_z_gate_statement(
            7,
            &attended,
            &norm_weight,
            &z_gate,
            &bad_output,
            2,
            4,
        )
        .unwrap_err()
        .contains("output mismatch"));
    }

    #[test]
    fn qwen35_norm_and_z_gate_air_proves_and_verifies_standalone() {
        let attended = filled_matrix(2, 4, 1);
        let norm_weight = vec![M31::from(2u32), M31::from(3u32)];
        let z_gate = matrix(2, 4, 5);
        let proof =
            prove_qwen35_norm_and_z_gate_air(3, &attended, &norm_weight, &z_gate, 2, 4).unwrap();
        verify_qwen35_norm_and_z_gate_air(&proof).unwrap();
        let output = qwen35_norm_and_z_gate_output(&attended, &norm_weight, &z_gate, 2, 4).unwrap();
        let wrong_statement =
            qwen35_norm_and_z_gate_statement(4, &attended, &norm_weight, &z_gate, &output, 2, 4)
                .unwrap();
        let err = verify_qwen35_norm_and_z_gate_air_with_statement_hash(
            &proof,
            wrong_statement.statement_hash,
        )
        .unwrap_err();
        assert!(err.to_string().contains("statement hash mismatch"));
    }
}
