//! Qwen3.5 GatedDeltaNet DeltaRecurrence statement binding.
//!
//! This module defines the typed statement that the future dedicated
//! DeltaRecurrence AIR must emit. It also pins the upstream Qwen3.5 recurrence
//! contract so the eventual AIR can be checked against the real model
//! semantics instead of a simplified recurrence.
//!
//! The trace-binding witness below is intentionally narrower than the final
//! arithmetic AIR: it proves row ordering and tensor/state binding for active
//! prover ingestion. The gated-delta math still has to be constrained by the
//! production AIR before this component is production-ready.

use crate::backend::convert_evaluations;
use crate::components::activation::{compute_multiplicities, ActivationRelation, ActivationType};
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

pub const QWEN35_DELTA_RECURRENCE_TRACE_BINDING_MIN_LOG_SIZE: u32 = 4;
pub const QWEN35_DELTA_RECURRENCE_ARITHMETIC_MIN_LOG_SIZE: u32 = 4;
pub const QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE: u32 = 4;
const DOMAIN_QWEN35_DELTA_RECURRENCE_MATRIX: u64 = 0x513352_4d4154; // "Q3R_MAT"
const DOMAIN_QWEN35_DELTA_RECURRENCE_VECTOR: u64 = 0x513352_564543; // "Q3R_VEC"
const DOMAIN_QWEN35_DELTA_RECURRENCE_STATEMENT: u64 = 0x513352_53544d; // "Q3R_STM"
const DOMAIN_QWEN35_DELTA_RECURRENCE_AIR_SPEC: u64 = 0x513352_414952; // "Q3R_AIR"
const DOMAIN_QWEN35_DELTA_RECURRENCE_TRACE_BINDING: u64 = 0x513352_545243; // "Q3R_TRC"
const DOMAIN_QWEN35_DELTA_RECURRENCE_CHANNEL: u64 = 0x513352_43484e; // "Q3R_CHN"
const DOMAIN_QWEN35_DELTA_RECURRENCE_ARITHMETIC_STATEMENT: u64 = 0x513352_415354; // "Q3R_AST"
const DOMAIN_QWEN35_DELTA_RECURRENCE_TRANSFORM_STATEMENT: u64 = 0x513352_545354; // "Q3R_TST"
const DOMAIN_QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING: u64 = 0x513352_544246; // "Q3R_TBF"
const DOMAIN_QWEN35_DELTA_RECURRENCE_BETA_SIGMOID_STATEMENT: u64 = 0x513352_425349; // "Q3R_BSI"
const DOMAIN_QWEN35_DELTA_RECURRENCE_NORM_STATEMENT: u64 = 0x513352_4e524d; // "Q3R_NRM"
const DOMAIN_QWEN35_DELTA_RECURRENCE_DECAY_STATEMENT: u64 = 0x513352_444543; // "Q3R_DEC"
const QWEN35_DELTA_RECURRENCE_TRACE_CHECKSUM_ALPHA: u32 = 65_537;

pub const QWEN35_DELTA_RECURRENCE_UPSTREAM: &str =
    "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35DeltaRecurrenceMode {
    ChunkPrefill,
    RecurrentDecode,
}

impl Qwen35DeltaRecurrenceMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ChunkPrefill => "chunk-prefill",
            Self::RecurrentDecode => "recurrent-decode",
        }
    }

    pub fn as_u64(self) -> u64 {
        match self {
            Self::ChunkPrefill => 1,
            Self::RecurrentDecode => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DeltaRecurrenceAirSpec {
    pub upstream: &'static str,
    pub mode: Qwen35DeltaRecurrenceMode,
    pub seq_len: usize,
    pub query_width: usize,
    pub key_width: usize,
    pub value_width: usize,
    pub state_rows: usize,
    pub value_head_dim: usize,
    pub qk_head_dim: usize,
    pub qk_repeat_factor: usize,
    pub binds_initial_recurrent_state: bool,
    pub emits_final_recurrent_state: bool,
    pub uses_qk_l2norm: bool,
    pub query_scale_is_inverse_sqrt_head_dim: bool,
    pub beta_is_sigmoid_b: bool,
    pub decay_is_neg_exp_a_log_times_softplus_a_plus_dt_bias: bool,
    pub state_update_is_gated_delta_rule: bool,
    pub output_is_pre_norm_attended_value: bool,
    pub spec_hash: FieldElement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DeltaRecurrenceStatement {
    pub layer_idx: usize,
    pub seq_len: usize,
    pub query_width: usize,
    pub key_width: usize,
    pub value_width: usize,
    pub state_rows: usize,
    pub value_head_dim: usize,
    pub mode: Qwen35DeltaRecurrenceMode,
    pub air_spec_hash: FieldElement,
    pub query_commitment: FieldElement,
    pub key_commitment: FieldElement,
    pub projected_value_commitment: FieldElement,
    pub a_gate_commitment: FieldElement,
    pub b_gate_commitment: FieldElement,
    pub a_log_weight_commitment: FieldElement,
    pub dt_bias_commitment: FieldElement,
    pub initial_recurrent_state_commitment: FieldElement,
    pub final_recurrent_state_commitment: FieldElement,
    pub output_commitment: FieldElement,
    pub statement_hash: FieldElement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DeltaRecurrenceArithmeticStatement {
    pub layer_idx: usize,
    pub seq_len: usize,
    pub query_width: usize,
    pub key_width: usize,
    pub value_width: usize,
    pub state_rows: usize,
    pub value_head_dim: usize,
    pub mode: Qwen35DeltaRecurrenceMode,
    pub air_spec_hash: FieldElement,
    pub scaled_query_commitment: FieldElement,
    pub normalized_key_commitment: FieldElement,
    pub projected_value_commitment: FieldElement,
    pub decay_commitment: FieldElement,
    pub beta_commitment: FieldElement,
    pub initial_recurrent_state_commitment: FieldElement,
    pub final_recurrent_state_commitment: FieldElement,
    pub output_commitment: FieldElement,
    pub statement_hash: FieldElement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DeltaRecurrenceTransformStatement {
    pub layer_idx: usize,
    pub seq_len: usize,
    pub query_width: usize,
    pub key_width: usize,
    pub state_rows: usize,
    pub value_head_dim: usize,
    pub mode: Qwen35DeltaRecurrenceMode,
    pub air_spec_hash: FieldElement,
    pub query_commitment: FieldElement,
    pub key_commitment: FieldElement,
    pub a_gate_commitment: FieldElement,
    pub b_gate_commitment: FieldElement,
    pub a_log_weight_commitment: FieldElement,
    pub dt_bias_commitment: FieldElement,
    pub scaled_query_commitment: FieldElement,
    pub normalized_key_commitment: FieldElement,
    pub decay_commitment: FieldElement,
    pub beta_commitment: FieldElement,
    pub statement_hash: FieldElement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DeltaRecurrenceBetaSigmoidStatement {
    pub layer_idx: usize,
    pub seq_len: usize,
    pub state_rows: usize,
    pub table_log_size: u32,
    pub trace_checksum: M31,
    pub table_commitment: FieldElement,
    pub b_gate_commitment: FieldElement,
    pub beta_commitment: FieldElement,
    pub statement_hash: FieldElement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35DeltaRecurrenceNormKind {
    Query,
    Key,
}

impl Qwen35DeltaRecurrenceNormKind {
    pub fn as_u64(self) -> u64 {
        match self {
            Self::Query => 1,
            Self::Key => 2,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Key => "key",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DeltaRecurrenceNormStatement {
    pub kind: Qwen35DeltaRecurrenceNormKind,
    pub layer_idx: usize,
    pub seq_len: usize,
    pub state_rows: usize,
    pub qk_head_dim: usize,
    pub table_log_size: u32,
    pub trace_checksum: M31,
    pub table_commitment: FieldElement,
    pub input_commitment: FieldElement,
    pub output_commitment: FieldElement,
    pub post_scale: M31,
    pub statement_hash: FieldElement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DeltaRecurrenceDecayStatement {
    pub layer_idx: usize,
    pub seq_len: usize,
    pub state_rows: usize,
    pub table_log_size: u32,
    pub trace_checksum: M31,
    pub softplus_table_commitment: FieldElement,
    pub exp_table_commitment: FieldElement,
    pub decay_table_commitment: FieldElement,
    pub a_gate_commitment: FieldElement,
    pub a_log_weight_commitment: FieldElement,
    pub dt_bias_commitment: FieldElement,
    pub decay_commitment: FieldElement,
    pub statement_hash: FieldElement,
}

#[derive(Debug, Clone)]
pub struct Qwen35DeltaRecurrenceInputs<'a> {
    pub query: &'a M31Matrix,
    pub key: &'a M31Matrix,
    pub projected_value: &'a M31Matrix,
    pub a_gate: &'a M31Matrix,
    pub b_gate: &'a M31Matrix,
    pub a_log_weight: &'a [M31],
    pub dt_bias: &'a [M31],
    pub initial_recurrent_state: &'a M31Matrix,
    pub final_recurrent_state: &'a M31Matrix,
    pub output: &'a M31Matrix,
    pub state_rows: usize,
    pub value_head_dim: usize,
    pub mode: Qwen35DeltaRecurrenceMode,
}

#[derive(Debug, Clone)]
pub struct Qwen35DeltaRecurrenceTraceBindingEval {
    pub log_n_rows: u32,
    pub instance_id: usize,
}

pub type Qwen35DeltaRecurrenceTraceBindingComponent =
    FrameworkComponent<Qwen35DeltaRecurrenceTraceBindingEval>;

#[derive(Debug, Clone)]
pub struct Qwen35DeltaRecurrenceTransformBindingEval {
    pub log_n_rows: u32,
    pub instance_id: usize,
}

pub type Qwen35DeltaRecurrenceTransformBindingComponent =
    FrameworkComponent<Qwen35DeltaRecurrenceTransformBindingEval>;

#[derive(Debug, Clone)]
pub struct Qwen35DeltaRecurrenceBetaSigmoidEval {
    pub log_n_rows: u32,
    pub instance_id: usize,
    pub lookup_elements: ActivationRelation,
    pub claimed_sum: SecureField,
    pub trace_checksum: M31,
}

pub type Qwen35DeltaRecurrenceBetaSigmoidComponent =
    FrameworkComponent<Qwen35DeltaRecurrenceBetaSigmoidEval>;

#[derive(Debug, Clone)]
pub struct Qwen35DeltaRecurrenceNormEval {
    pub log_n_rows: u32,
    pub instance_id: usize,
    pub lookup_elements: RMSNormRelation,
    pub claimed_sum: SecureField,
    pub inv_qk_head_dim: M31,
    pub post_scale: M31,
    pub trace_checksum: M31,
}

pub type Qwen35DeltaRecurrenceNormComponent = FrameworkComponent<Qwen35DeltaRecurrenceNormEval>;

#[derive(Debug, Clone)]
pub struct Qwen35DeltaRecurrenceDecayEval {
    pub log_n_rows: u32,
    pub instance_id: usize,
    pub lookup_elements: ActivationRelation,
    pub claimed_sum: SecureField,
    pub trace_checksum: M31,
}

pub type Qwen35DeltaRecurrenceDecayComponent = FrameworkComponent<Qwen35DeltaRecurrenceDecayEval>;

#[derive(Debug, Clone)]
pub struct Qwen35DeltaRecurrenceArithmeticEval {
    pub log_n_rows: u32,
    pub instance_id: usize,
}

pub type Qwen35DeltaRecurrenceArithmeticComponent =
    FrameworkComponent<Qwen35DeltaRecurrenceArithmeticEval>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DeltaRecurrenceTraceBindingTrace {
    pub log_size: u32,
    pub n_real_rows: usize,
    pub preprocessed: Vec<Vec<M31>>,
    pub execution: Vec<Vec<M31>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DeltaRecurrenceTransformBindingTrace {
    pub log_size: u32,
    pub n_real_rows: usize,
    pub preprocessed: Vec<Vec<M31>>,
    pub execution: Vec<Vec<M31>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DeltaRecurrenceArithmeticTrace {
    pub log_size: u32,
    pub n_real_rows: usize,
    pub preprocessed: Vec<Vec<M31>>,
    pub execution: Vec<Vec<M31>>,
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen35DeltaRecurrenceTraceBindingProofError {
    #[error("Witness error: {0}")]
    Witness(String),
    #[error("Proving error: {0}")]
    Proving(String),
    #[error("Verification error: {0}")]
    Verification(String),
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen35DeltaRecurrenceTransformBindingProofError {
    #[error("Witness error: {0}")]
    Witness(String),
    #[error("Proving error: {0}")]
    Proving(String),
    #[error("Verification error: {0}")]
    Verification(String),
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen35DeltaRecurrenceBetaSigmoidProofError {
    #[error("Witness error: {0}")]
    Witness(String),
    #[error("Proving error: {0}")]
    Proving(String),
    #[error("Verification error: {0}")]
    Verification(String),
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen35DeltaRecurrenceNormProofError {
    #[error("Witness error: {0}")]
    Witness(String),
    #[error("Proving error: {0}")]
    Proving(String),
    #[error("Verification error: {0}")]
    Verification(String),
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen35DeltaRecurrenceDecayProofError {
    #[error("Witness error: {0}")]
    Witness(String),
    #[error("Proving error: {0}")]
    Proving(String),
    #[error("Verification error: {0}")]
    Verification(String),
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen35DeltaRecurrenceArithmeticProofError {
    #[error("Witness error: {0}")]
    Witness(String),
    #[error("Proving error: {0}")]
    Proving(String),
    #[error("Verification error: {0}")]
    Verification(String),
}

#[derive(Debug)]
pub struct Qwen35DeltaRecurrenceTraceBindingProof<H: MerkleHasherLifted> {
    pub stark_proof: StarkProof<H>,
    pub log_size: u32,
    pub n_real_rows: usize,
    pub statement: Qwen35DeltaRecurrenceStatement,
}

#[derive(Debug)]
pub struct Qwen35DeltaRecurrenceTransformBindingProof<H: MerkleHasherLifted> {
    pub stark_proof: StarkProof<H>,
    pub log_size: u32,
    pub n_real_rows: usize,
    pub statement: Qwen35DeltaRecurrenceTransformStatement,
}

#[derive(Debug)]
pub struct Qwen35DeltaRecurrenceBetaSigmoidProof<H: MerkleHasherLifted> {
    pub stark_proof: StarkProof<H>,
    pub claimed_sum: SecureField,
    pub log_size: u32,
    pub n_real_rows: usize,
    pub statement: Qwen35DeltaRecurrenceBetaSigmoidStatement,
}

#[derive(Debug)]
pub struct Qwen35DeltaRecurrenceNormProof<H: MerkleHasherLifted> {
    pub stark_proof: StarkProof<H>,
    pub claimed_sum: SecureField,
    pub log_size: u32,
    pub n_real_rows: usize,
    pub statement: Qwen35DeltaRecurrenceNormStatement,
}

#[derive(Debug)]
pub struct Qwen35DeltaRecurrenceDecayProof<H: MerkleHasherLifted> {
    pub stark_proof: StarkProof<H>,
    pub claimed_sum: SecureField,
    pub log_size: u32,
    pub n_real_rows: usize,
    pub statement: Qwen35DeltaRecurrenceDecayStatement,
}

#[derive(Debug)]
pub struct Qwen35DeltaRecurrenceArithmeticProof<H: MerkleHasherLifted> {
    pub stark_proof: StarkProof<H>,
    pub log_size: u32,
    pub n_real_rows: usize,
    pub statement: Qwen35DeltaRecurrenceArithmeticStatement,
}

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DeltaRecurrenceTraceBindingRow {
    pub token_idx: usize,
    pub state_row_idx: usize,
    pub qk_col_idx: usize,
    pub value_col_idx: usize,
    pub query: M31,
    pub key: M31,
    pub projected_value: M31,
    pub a_gate: M31,
    pub b_gate: M31,
    pub a_log_weight: M31,
    pub dt_bias: M31,
    pub initial_recurrent_state: M31,
    pub final_recurrent_state: M31,
    pub output: M31,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DeltaRecurrenceTraceBindingWitness {
    pub rows: Vec<Qwen35DeltaRecurrenceTraceBindingRow>,
    pub witness_hash: FieldElement,
}

#[derive(Debug, Clone)]
pub struct Qwen35DeltaRecurrenceArithmeticInputs<'a> {
    /// Q after Q/K L2 normalization and inverse-sqrt head scaling.
    pub scaled_query: &'a M31Matrix,
    /// K after Q/K L2 normalization.
    pub normalized_key: &'a M31Matrix,
    pub projected_value: &'a M31Matrix,
    /// exp(g), where g = -exp(A_log) * softplus(a + dt_bias).
    pub decay: &'a M31Matrix,
    /// sigmoid(b).
    pub beta: &'a M31Matrix,
    pub initial_recurrent_state: &'a M31Matrix,
    pub final_recurrent_state: &'a M31Matrix,
    pub output: &'a M31Matrix,
    pub state_rows: usize,
    pub value_head_dim: usize,
}

#[derive(Debug, Clone)]
pub struct Qwen35DeltaRecurrenceTransformInputs<'a> {
    pub query: &'a M31Matrix,
    pub key: &'a M31Matrix,
    pub a_gate: &'a M31Matrix,
    pub b_gate: &'a M31Matrix,
    pub a_log_weight: &'a [M31],
    pub dt_bias: &'a [M31],
    /// Q after Q/K L2 normalization and inverse-sqrt head scaling.
    pub scaled_query: &'a M31Matrix,
    /// K after Q/K L2 normalization.
    pub normalized_key: &'a M31Matrix,
    /// exp(g), where g = -exp(A_log) * softplus(a + dt_bias).
    pub decay: &'a M31Matrix,
    /// sigmoid(b).
    pub beta: &'a M31Matrix,
    pub state_rows: usize,
    pub value_head_dim: usize,
    pub mode: Qwen35DeltaRecurrenceMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DeltaRecurrenceArithmeticRow {
    pub token_idx: usize,
    pub state_row_idx: usize,
    pub qk_col_idx: usize,
    pub value_col_idx: usize,
    pub scaled_query: M31,
    pub normalized_key: M31,
    pub projected_value: M31,
    pub decay: M31,
    pub beta: M31,
    pub state_before: M31,
    pub decayed_state: M31,
    pub kv_term: M31,
    pub kv_residual_before: M31,
    pub kv_residual_after: M31,
    pub kv_mem: M31,
    pub delta: M31,
    pub state_after: M31,
    pub output_term: M31,
    pub output_prefix_before: M31,
    pub output_prefix_after: M31,
    pub output: M31,
}

#[derive(Debug, Clone)]
pub struct Qwen35DeltaRecurrenceArithmeticWitness {
    pub rows: Vec<Qwen35DeltaRecurrenceArithmeticRow>,
    pub final_recurrent_state: M31Matrix,
    pub output: M31Matrix,
    pub witness_hash: FieldElement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DeltaRecurrenceTransformBindingRow {
    pub token_idx: usize,
    pub state_row_idx: usize,
    pub qk_col_idx: usize,
    pub query: M31,
    pub key: M31,
    pub scaled_query: M31,
    pub normalized_key: M31,
    pub a_gate: M31,
    pub b_gate: M31,
    pub a_log_weight: M31,
    pub dt_bias: M31,
    pub decay: M31,
    pub beta: M31,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DeltaRecurrenceTransformBindingWitness {
    pub rows: Vec<Qwen35DeltaRecurrenceTransformBindingRow>,
    pub witness_hash: FieldElement,
}

impl FrameworkEval for Qwen35DeltaRecurrenceTraceBindingEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let expected_token = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_trace_binding_preprocessed_id(self.instance_id, "token")
                .into(),
        });
        let expected_state_row = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_trace_binding_preprocessed_id(
                self.instance_id,
                "state_row",
            )
            .into(),
        });
        let expected_qk_col = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_trace_binding_preprocessed_id(self.instance_id, "qk_col")
                .into(),
        });
        let expected_value_col = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_trace_binding_preprocessed_id(
                self.instance_id,
                "value_col",
            )
            .into(),
        });

        let token = eval.next_trace_mask();
        let state_row = eval.next_trace_mask();
        let qk_col = eval.next_trace_mask();
        let value_col = eval.next_trace_mask();
        for _ in 0..10 {
            let _ = eval.next_trace_mask();
        }

        eval.add_constraint(token.clone() - expected_token);
        eval.add_constraint(state_row.clone() - expected_state_row);
        eval.add_constraint(qk_col - expected_qk_col);
        eval.add_constraint(value_col - expected_value_col);

        eval
    }
}

impl FrameworkEval for Qwen35DeltaRecurrenceTransformBindingEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let expected_token = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_transform_binding_preprocessed_id(
                self.instance_id,
                "token",
            )
            .into(),
        });
        let expected_state_row = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_transform_binding_preprocessed_id(
                self.instance_id,
                "state_row",
            )
            .into(),
        });
        let expected_qk_col = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_transform_binding_preprocessed_id(
                self.instance_id,
                "qk_col",
            )
            .into(),
        });

        let token = eval.next_trace_mask();
        let state_row = eval.next_trace_mask();
        let qk_col = eval.next_trace_mask();
        for _ in 0..10 {
            let _ = eval.next_trace_mask();
        }

        eval.add_constraint(token.clone() - expected_token);
        eval.add_constraint(state_row.clone() - expected_state_row);
        eval.add_constraint(qk_col - expected_qk_col);

        eval
    }
}

impl FrameworkEval for Qwen35DeltaRecurrenceBetaSigmoidEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let table_input = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_beta_sigmoid_preprocessed_id(
                self.instance_id,
                "table_input",
            )
            .into(),
        });
        let table_output = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_beta_sigmoid_preprocessed_id(
                self.instance_id,
                "table_output",
            )
            .into(),
        });
        let expected_token = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_beta_sigmoid_preprocessed_id(self.instance_id, "token")
                .into(),
        });
        let expected_state_row = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_beta_sigmoid_preprocessed_id(self.instance_id, "state_row")
                .into(),
        });
        let is_first = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_beta_sigmoid_preprocessed_id(self.instance_id, "is_first")
                .into(),
        });
        let is_last = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_beta_sigmoid_preprocessed_id(self.instance_id, "is_last")
                .into(),
        });
        let has_next = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_beta_sigmoid_preprocessed_id(self.instance_id, "has_next")
                .into(),
        });

        let token = eval.next_trace_mask();
        let state_row = eval.next_trace_mask();
        let trace_input = eval.next_trace_mask();
        let trace_output = eval.next_trace_mask();
        let trace_acc_before = eval.next_trace_mask();
        let trace_acc_after = eval.next_trace_mask();
        let shifted_next_trace_acc_before = eval.next_trace_mask();
        let multiplicity = eval.next_trace_mask();

        eval.add_constraint(token.clone() - expected_token);
        eval.add_constraint(state_row.clone() - expected_state_row);
        let trace_row_term = E::F::from(BaseField::from(M31::from(151u32)))
            + token * E::F::from(BaseField::from(M31::from(3u32)))
            + state_row * E::F::from(BaseField::from(M31::from(5u32)))
            + trace_input.clone() * E::F::from(BaseField::from(M31::from(7u32)))
            + trace_output.clone() * E::F::from(BaseField::from(M31::from(11u32)));
        let checksum_alpha = E::F::from(BaseField::from(M31::from(
            QWEN35_DELTA_RECURRENCE_TRACE_CHECKSUM_ALPHA,
        )));
        eval.add_constraint(
            trace_acc_after.clone() - trace_acc_before.clone() * checksum_alpha - trace_row_term,
        );
        eval.add_constraint(is_first * trace_acc_before);
        eval.add_constraint(has_next * (shifted_next_trace_acc_before - trace_acc_after.clone()));
        eval.add_constraint(
            is_last * (trace_acc_after - E::F::from(BaseField::from(self.trace_checksum))),
        );

        let tag = E::F::from(BaseField::from(ActivationType::Sigmoid.type_tag()));
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::from(multiplicity),
            &[tag.clone(), table_input, table_output],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::from(E::F::from(BaseField::from(1))),
            &[tag, trace_input, trace_output],
        ));
        eval.finalize_logup_in_pairs();

        eval
    }
}

impl FrameworkEval for Qwen35DeltaRecurrenceNormEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let table_rms_sq = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_norm_preprocessed_id(self.instance_id, "table_rms_sq")
                .into(),
        });
        let table_rsqrt = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_norm_preprocessed_id(self.instance_id, "table_rsqrt")
                .into(),
        });
        let expected_token = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_norm_preprocessed_id(self.instance_id, "token").into(),
        });
        let expected_state_row = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_norm_preprocessed_id(self.instance_id, "state_row").into(),
        });
        let expected_qk_col = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_norm_preprocessed_id(self.instance_id, "qk_col").into(),
        });
        let is_first_qk = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_norm_preprocessed_id(self.instance_id, "is_first_qk")
                .into(),
        });
        let is_last_qk = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_norm_preprocessed_id(self.instance_id, "is_last_qk").into(),
        });
        let is_qk_chain = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_norm_preprocessed_id(self.instance_id, "is_qk_chain")
                .into(),
        });
        let is_first = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_norm_preprocessed_id(self.instance_id, "is_first").into(),
        });
        let is_last = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_norm_preprocessed_id(self.instance_id, "is_last").into(),
        });
        let has_next = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_norm_preprocessed_id(self.instance_id, "has_next").into(),
        });

        let token = eval.next_trace_mask();
        let state_row = eval.next_trace_mask();
        let qk_col = eval.next_trace_mask();
        let input = eval.next_trace_mask();
        let output = eval.next_trace_mask();
        let sq_term = eval.next_trace_mask();
        let sq_prefix_before = eval.next_trace_mask();
        let sq_prefix_after = eval.next_trace_mask();
        let shifted_next_sq_prefix_before = eval.next_trace_mask();
        let rms_sq = eval.next_trace_mask();
        let rsqrt = eval.next_trace_mask();
        let shifted_next_rms_sq = eval.next_trace_mask();
        let shifted_next_rsqrt = eval.next_trace_mask();
        let trace_acc_before = eval.next_trace_mask();
        let trace_acc_after = eval.next_trace_mask();
        let shifted_next_trace_acc_before = eval.next_trace_mask();
        let multiplicity = eval.next_trace_mask();

        let norm_accumulator_scale = E::F::from(BaseField::from(self.inv_qk_head_dim));
        let post_scale = E::F::from(BaseField::from(self.post_scale));
        let one = E::EF::from(E::F::from(BaseField::from(1)));

        eval.add_constraint(token.clone() - expected_token);
        eval.add_constraint(state_row.clone() - expected_state_row);
        eval.add_constraint(qk_col.clone() - expected_qk_col);

        eval.add_constraint(sq_term.clone() - input.clone() * input.clone());
        eval.add_constraint(sq_prefix_after.clone() - sq_prefix_before.clone() - sq_term.clone());
        eval.add_constraint(is_first_qk.clone() * sq_prefix_before.clone());
        eval.add_constraint(
            is_qk_chain.clone() * (shifted_next_sq_prefix_before.clone() - sq_prefix_after.clone()),
        );
        eval.add_constraint(
            is_last_qk.clone()
                * (sq_prefix_after.clone() * norm_accumulator_scale - rms_sq.clone()),
        );
        eval.add_constraint(is_qk_chain.clone() * (shifted_next_rms_sq - rms_sq.clone()));
        eval.add_constraint(is_qk_chain * (shifted_next_rsqrt - rsqrt.clone()));
        eval.add_constraint(output.clone() - input.clone() * rsqrt.clone() * post_scale);
        let trace_row_term = E::F::from(BaseField::from(M31::from(181u32)))
            + token * E::F::from(BaseField::from(M31::from(3u32)))
            + state_row * E::F::from(BaseField::from(M31::from(5u32)))
            + qk_col * E::F::from(BaseField::from(M31::from(7u32)))
            + input * E::F::from(BaseField::from(M31::from(11u32)))
            + output * E::F::from(BaseField::from(M31::from(13u32)))
            + sq_term * E::F::from(BaseField::from(M31::from(17u32)))
            + sq_prefix_before * E::F::from(BaseField::from(M31::from(19u32)))
            + sq_prefix_after * E::F::from(BaseField::from(M31::from(23u32)))
            + rms_sq.clone() * E::F::from(BaseField::from(M31::from(29u32)))
            + rsqrt.clone() * E::F::from(BaseField::from(M31::from(31u32)));
        let checksum_alpha = E::F::from(BaseField::from(M31::from(
            QWEN35_DELTA_RECURRENCE_TRACE_CHECKSUM_ALPHA,
        )));
        eval.add_constraint(
            trace_acc_after.clone() - trace_acc_before.clone() * checksum_alpha - trace_row_term,
        );
        eval.add_constraint(is_first * trace_acc_before);
        eval.add_constraint(has_next * (shifted_next_trace_acc_before - trace_acc_after.clone()));
        eval.add_constraint(
            is_last * (trace_acc_after - E::F::from(BaseField::from(self.trace_checksum))),
        );

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::from(multiplicity),
            &[table_rms_sq, table_rsqrt],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            one,
            &[rms_sq, rsqrt],
        ));
        eval.finalize_logup_in_pairs();

        eval
    }
}

impl FrameworkEval for Qwen35DeltaRecurrenceDecayEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let softplus_table_input = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_decay_preprocessed_id(
                self.instance_id,
                "softplus_table_input",
            )
            .into(),
        });
        let softplus_table_output = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_decay_preprocessed_id(
                self.instance_id,
                "softplus_table_output",
            )
            .into(),
        });
        let exp_a_log_table_input = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_decay_preprocessed_id(
                self.instance_id,
                "exp_a_log_table_input",
            )
            .into(),
        });
        let exp_a_log_table_output = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_decay_preprocessed_id(
                self.instance_id,
                "exp_a_log_table_output",
            )
            .into(),
        });
        let decay_exp_table_input = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_decay_preprocessed_id(
                self.instance_id,
                "decay_exp_table_input",
            )
            .into(),
        });
        let decay_exp_table_output = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_decay_preprocessed_id(
                self.instance_id,
                "decay_exp_table_output",
            )
            .into(),
        });
        let expected_token = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_decay_preprocessed_id(self.instance_id, "token").into(),
        });
        let expected_state_row = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_decay_preprocessed_id(self.instance_id, "state_row").into(),
        });
        let is_first = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_decay_preprocessed_id(self.instance_id, "is_first").into(),
        });
        let is_last = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_decay_preprocessed_id(self.instance_id, "is_last").into(),
        });
        let has_next = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_decay_preprocessed_id(self.instance_id, "has_next").into(),
        });

        let token = eval.next_trace_mask();
        let state_row = eval.next_trace_mask();
        let a_gate = eval.next_trace_mask();
        let dt_bias = eval.next_trace_mask();
        let a_sum = eval.next_trace_mask();
        let softplus = eval.next_trace_mask();
        let a_log_weight = eval.next_trace_mask();
        let exp_a_log = eval.next_trace_mask();
        let neg_product = eval.next_trace_mask();
        let decay = eval.next_trace_mask();
        let trace_acc_before = eval.next_trace_mask();
        let trace_acc_after = eval.next_trace_mask();
        let shifted_next_trace_acc_before = eval.next_trace_mask();
        let softplus_multiplicity = eval.next_trace_mask();
        let exp_a_log_multiplicity = eval.next_trace_mask();
        let decay_multiplicity = eval.next_trace_mask();

        let softplus_tag = E::F::from(BaseField::from(ActivationType::Softplus.type_tag()));
        let exp_tag = E::F::from(BaseField::from(ActivationType::Softmax.type_tag()));
        let one = E::EF::from(E::F::from(BaseField::from(1)));

        eval.add_constraint(token.clone() - expected_token);
        eval.add_constraint(state_row.clone() - expected_state_row);
        eval.add_constraint(a_sum.clone() - a_gate.clone() - dt_bias.clone());
        let fixed_scale = E::F::from(BaseField::from(M31::from(
            crate::gadgets::lookup_table::activations::GELU_FIXED_POINT_SCALE,
        )));
        eval.add_constraint(
            neg_product.clone() * fixed_scale - exp_a_log.clone() * softplus.clone(),
        );
        let trace_row_term = E::F::from(BaseField::from(M31::from(101u32)))
            + token * E::F::from(BaseField::from(M31::from(3u32)))
            + state_row * E::F::from(BaseField::from(M31::from(5u32)))
            + a_gate * E::F::from(BaseField::from(M31::from(7u32)))
            + dt_bias * E::F::from(BaseField::from(M31::from(11u32)))
            + a_sum.clone() * E::F::from(BaseField::from(M31::from(13u32)))
            + softplus.clone() * E::F::from(BaseField::from(M31::from(17u32)))
            + a_log_weight.clone() * E::F::from(BaseField::from(M31::from(19u32)))
            + exp_a_log.clone() * E::F::from(BaseField::from(M31::from(23u32)))
            + neg_product.clone() * E::F::from(BaseField::from(M31::from(29u32)))
            + decay.clone() * E::F::from(BaseField::from(M31::from(31u32)));
        let checksum_alpha = E::F::from(BaseField::from(M31::from(
            QWEN35_DELTA_RECURRENCE_TRACE_CHECKSUM_ALPHA,
        )));
        eval.add_constraint(
            trace_acc_after.clone() - trace_acc_before.clone() * checksum_alpha - trace_row_term,
        );
        eval.add_constraint(is_first * trace_acc_before);
        eval.add_constraint(has_next * (shifted_next_trace_acc_before - trace_acc_after.clone()));
        eval.add_constraint(
            is_last * (trace_acc_after - E::F::from(BaseField::from(self.trace_checksum))),
        );

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::from(softplus_multiplicity),
            &[
                softplus_tag.clone(),
                softplus_table_input,
                softplus_table_output,
            ],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            one.clone(),
            &[softplus_tag, a_sum, softplus],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::from(exp_a_log_multiplicity),
            &[
                exp_tag.clone(),
                exp_a_log_table_input,
                exp_a_log_table_output,
            ],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            one.clone(),
            &[exp_tag.clone(), a_log_weight, exp_a_log],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::from(decay_multiplicity),
            &[
                exp_tag.clone(),
                decay_exp_table_input,
                decay_exp_table_output,
            ],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            one,
            &[exp_tag, neg_product, decay],
        ));
        eval.finalize_logup_in_pairs();

        eval
    }
}

impl FrameworkEval for Qwen35DeltaRecurrenceArithmeticEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let expected_token = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_arithmetic_preprocessed_id(self.instance_id, "token")
                .into(),
        });
        let expected_state_row = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_arithmetic_preprocessed_id(self.instance_id, "state_row")
                .into(),
        });
        let expected_qk_col = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_arithmetic_preprocessed_id(self.instance_id, "qk_col")
                .into(),
        });
        let expected_value_col = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_arithmetic_preprocessed_id(self.instance_id, "value_col")
                .into(),
        });
        let is_first_qk = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_arithmetic_preprocessed_id(self.instance_id, "is_first_qk")
                .into(),
        });
        let is_last_qk = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_arithmetic_preprocessed_id(self.instance_id, "is_last_qk")
                .into(),
        });
        let is_qk_chain = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_arithmetic_preprocessed_id(self.instance_id, "is_qk_chain")
                .into(),
        });
        let has_next_token = eval.get_preprocessed_column(PreProcessedColumnId {
            id: qwen35_delta_recurrence_arithmetic_preprocessed_id(
                self.instance_id,
                "has_next_token",
            )
            .into(),
        });

        let token = eval.next_trace_mask();
        let state_row = eval.next_trace_mask();
        let qk_col = eval.next_trace_mask();
        let value_col = eval.next_trace_mask();
        let scaled_query = eval.next_trace_mask();
        let normalized_key = eval.next_trace_mask();
        let projected_value = eval.next_trace_mask();
        let decay = eval.next_trace_mask();
        let beta = eval.next_trace_mask();
        let state_before = eval.next_trace_mask();
        let decayed_state = eval.next_trace_mask();
        let kv_term = eval.next_trace_mask();
        let kv_residual_before = eval.next_trace_mask();
        let kv_residual_after = eval.next_trace_mask();
        let shifted_next_kv_residual_before = eval.next_trace_mask();
        let kv_mem = eval.next_trace_mask();
        let delta = eval.next_trace_mask();
        let state_after = eval.next_trace_mask();
        let output_term = eval.next_trace_mask();
        let output_prefix_before = eval.next_trace_mask();
        let output_prefix_after = eval.next_trace_mask();
        let shifted_next_output_prefix_before = eval.next_trace_mask();
        let output = eval.next_trace_mask();
        let next_token_state_before = eval.next_trace_mask();

        eval.add_constraint(token - expected_token);
        eval.add_constraint(state_row - expected_state_row);
        eval.add_constraint(qk_col - expected_qk_col);
        eval.add_constraint(value_col - expected_value_col);

        eval.add_constraint(decayed_state.clone() - state_before.clone() * decay.clone());
        eval.add_constraint(kv_term.clone() - decayed_state.clone() * normalized_key.clone());
        eval.add_constraint(
            kv_residual_after.clone() - kv_residual_before.clone() + kv_term.clone(),
        );
        eval.add_constraint(is_first_qk.clone() * (kv_residual_before.clone() - kv_mem.clone()));
        eval.add_constraint(is_last_qk.clone() * kv_residual_after.clone());
        eval.add_constraint(
            is_qk_chain.clone()
                * (shifted_next_kv_residual_before.clone() - kv_residual_after.clone()),
        );

        eval.add_constraint(delta.clone() - (projected_value - kv_mem.clone()) * beta);
        eval.add_constraint(state_after.clone() - decayed_state - normalized_key * delta);
        eval.add_constraint(output_term.clone() - state_after.clone() * scaled_query);
        eval.add_constraint(
            output_prefix_after.clone() - output_prefix_before.clone() - output_term,
        );
        eval.add_constraint(is_first_qk.clone() * output_prefix_before);
        eval.add_constraint(is_last_qk * (output_prefix_after.clone() - output));
        eval.add_constraint(
            is_qk_chain * (shifted_next_output_prefix_before - output_prefix_after),
        );
        eval.add_constraint(has_next_token * (next_token_state_before - state_after));

        eval
    }
}

pub fn qwen35_delta_recurrence_trace_binding_preprocessed_id(
    instance_id: usize,
    name: &str,
) -> String {
    format!("qwen35_delta_recurrence_trace_binding_{name}_{instance_id}")
}

pub fn qwen35_delta_recurrence_transform_binding_preprocessed_id(
    instance_id: usize,
    name: &str,
) -> String {
    format!("qwen35_delta_recurrence_transform_binding_{name}_{instance_id}")
}

pub fn qwen35_delta_recurrence_beta_sigmoid_preprocessed_id(
    instance_id: usize,
    name: &str,
) -> String {
    format!("qwen35_delta_recurrence_beta_sigmoid_{name}_{instance_id}")
}

pub fn qwen35_delta_recurrence_norm_preprocessed_id(instance_id: usize, name: &str) -> String {
    format!("qwen35_delta_recurrence_norm_{name}_{instance_id}")
}

pub fn qwen35_delta_recurrence_decay_preprocessed_id(instance_id: usize, name: &str) -> String {
    format!("qwen35_delta_recurrence_decay_{name}_{instance_id}")
}

pub fn qwen35_delta_recurrence_arithmetic_preprocessed_id(
    instance_id: usize,
    name: &str,
) -> String {
    format!("qwen35_delta_recurrence_arithmetic_{name}_{instance_id}")
}

pub fn qwen35_delta_recurrence_air_spec_from_statement(
    statement: &Qwen35DeltaRecurrenceStatement,
    _mode: Qwen35DeltaRecurrenceMode,
    binds_initial_recurrent_state: bool,
    emits_final_recurrent_state: bool,
) -> Result<Qwen35DeltaRecurrenceAirSpec, String> {
    qwen35_delta_recurrence_air_spec(
        statement.seq_len,
        statement.query_width,
        statement.key_width,
        statement.value_width,
        statement.state_rows,
        statement.value_head_dim,
        statement.mode,
        binds_initial_recurrent_state,
        emits_final_recurrent_state,
    )
}

pub fn qwen35_delta_recurrence_air_spec(
    seq_len: usize,
    query_width: usize,
    key_width: usize,
    value_width: usize,
    state_rows: usize,
    value_head_dim: usize,
    mode: Qwen35DeltaRecurrenceMode,
    binds_initial_recurrent_state: bool,
    emits_final_recurrent_state: bool,
) -> Result<Qwen35DeltaRecurrenceAirSpec, String> {
    if seq_len == 0 || query_width == 0 || key_width == 0 || value_width == 0 {
        return Err("DeltaRecurrence AIR dimensions must be non-zero".to_string());
    }
    if state_rows == 0 || value_head_dim == 0 {
        return Err("DeltaRecurrence AIR state dimensions must be non-zero".to_string());
    }
    if query_width != key_width {
        return Err(format!(
            "DeltaRecurrence AIR consumes repeated q/k with matching widths, got query={} key={}",
            query_width, key_width
        ));
    }
    if value_width != state_rows * value_head_dim {
        return Err(format!(
            "DeltaRecurrence value width {} != state_rows*value_head_dim {}",
            value_width,
            state_rows * value_head_dim
        ));
    }
    if query_width % state_rows != 0 {
        return Err(format!(
            "DeltaRecurrence query width {} must divide by state_rows {}",
            query_width, state_rows
        ));
    }
    let qk_head_dim = query_width / state_rows;
    if qk_head_dim == 0 {
        return Err("DeltaRecurrence q/k head dimension must be non-zero".to_string());
    }
    let qk_repeat_factor = if value_head_dim == qk_head_dim {
        1
    } else {
        value_head_dim.div_ceil(qk_head_dim)
    };
    let spec_hash = qwen35_delta_recurrence_air_spec_hash(
        seq_len,
        query_width,
        key_width,
        value_width,
        state_rows,
        value_head_dim,
        qk_head_dim,
        qk_repeat_factor,
        mode,
        binds_initial_recurrent_state,
        emits_final_recurrent_state,
    );
    Ok(Qwen35DeltaRecurrenceAirSpec {
        upstream: QWEN35_DELTA_RECURRENCE_UPSTREAM,
        mode,
        seq_len,
        query_width,
        key_width,
        value_width,
        state_rows,
        value_head_dim,
        qk_head_dim,
        qk_repeat_factor,
        binds_initial_recurrent_state,
        emits_final_recurrent_state,
        uses_qk_l2norm: true,
        query_scale_is_inverse_sqrt_head_dim: true,
        beta_is_sigmoid_b: true,
        decay_is_neg_exp_a_log_times_softplus_a_plus_dt_bias: true,
        state_update_is_gated_delta_rule: true,
        output_is_pre_norm_attended_value: true,
        spec_hash,
    })
}

fn qwen35_delta_recurrence_air_spec_hash(
    seq_len: usize,
    query_width: usize,
    key_width: usize,
    value_width: usize,
    state_rows: usize,
    value_head_dim: usize,
    qk_head_dim: usize,
    qk_repeat_factor: usize,
    mode: Qwen35DeltaRecurrenceMode,
    binds_initial_recurrent_state: bool,
    emits_final_recurrent_state: bool,
) -> FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(DOMAIN_QWEN35_DELTA_RECURRENCE_AIR_SPEC),
        FieldElement::from(mode.as_u64()),
        FieldElement::from(seq_len as u64),
        FieldElement::from(query_width as u64),
        FieldElement::from(key_width as u64),
        FieldElement::from(value_width as u64),
        FieldElement::from(state_rows as u64),
        FieldElement::from(value_head_dim as u64),
        FieldElement::from(qk_head_dim as u64),
        FieldElement::from(qk_repeat_factor as u64),
        FieldElement::from(binds_initial_recurrent_state as u64),
        FieldElement::from(emits_final_recurrent_state as u64),
        FieldElement::from(1u64),
        FieldElement::from(1u64),
        FieldElement::from(1u64),
        FieldElement::from(1u64),
        FieldElement::from(1u64),
        FieldElement::from(1u64),
    ])
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
    let mut felts = Vec::with_capacity(2 + values.len());
    felts.push(FieldElement::from(domain));
    felts.push(FieldElement::from(values.len() as u64));
    for value in values {
        felts.push(FieldElement::from(value.0 as u64));
    }
    starknet_crypto::poseidon_hash_many(&felts)
}

fn zero() -> M31 {
    M31::from(0u32)
}

pub fn qwen35_delta_recurrence_query_commitment(query: &M31Matrix) -> FieldElement {
    m31_matrix_commitment(DOMAIN_QWEN35_DELTA_RECURRENCE_MATRIX, query)
}

pub fn qwen35_delta_recurrence_key_commitment(key: &M31Matrix) -> FieldElement {
    m31_matrix_commitment(DOMAIN_QWEN35_DELTA_RECURRENCE_MATRIX + 1, key)
}

pub fn qwen35_delta_recurrence_projected_value_commitment(
    projected_value: &M31Matrix,
) -> FieldElement {
    m31_matrix_commitment(DOMAIN_QWEN35_DELTA_RECURRENCE_MATRIX + 2, projected_value)
}

pub fn qwen35_delta_recurrence_a_gate_commitment(a_gate: &M31Matrix) -> FieldElement {
    m31_matrix_commitment(DOMAIN_QWEN35_DELTA_RECURRENCE_MATRIX + 3, a_gate)
}

pub fn qwen35_delta_recurrence_b_gate_commitment(b_gate: &M31Matrix) -> FieldElement {
    m31_matrix_commitment(DOMAIN_QWEN35_DELTA_RECURRENCE_MATRIX + 4, b_gate)
}

pub fn qwen35_delta_recurrence_a_log_weight_commitment(a_log_weight: &[M31]) -> FieldElement {
    m31_vector_commitment(DOMAIN_QWEN35_DELTA_RECURRENCE_VECTOR, a_log_weight)
}

pub fn qwen35_delta_recurrence_dt_bias_commitment(dt_bias: &[M31]) -> FieldElement {
    m31_vector_commitment(DOMAIN_QWEN35_DELTA_RECURRENCE_VECTOR + 1, dt_bias)
}

pub fn qwen35_delta_recurrence_initial_state_commitment(state: &M31Matrix) -> FieldElement {
    m31_matrix_commitment(DOMAIN_QWEN35_DELTA_RECURRENCE_MATRIX + 5, state)
}

pub fn qwen35_delta_recurrence_final_state_commitment(state: &M31Matrix) -> FieldElement {
    m31_matrix_commitment(DOMAIN_QWEN35_DELTA_RECURRENCE_MATRIX + 6, state)
}

pub fn qwen35_delta_recurrence_output_commitment(output: &M31Matrix) -> FieldElement {
    m31_matrix_commitment(DOMAIN_QWEN35_DELTA_RECURRENCE_MATRIX + 7, output)
}

pub fn qwen35_delta_recurrence_scaled_query_commitment(scaled_query: &M31Matrix) -> FieldElement {
    m31_matrix_commitment(DOMAIN_QWEN35_DELTA_RECURRENCE_MATRIX + 8, scaled_query)
}

pub fn qwen35_delta_recurrence_normalized_key_commitment(
    normalized_key: &M31Matrix,
) -> FieldElement {
    m31_matrix_commitment(DOMAIN_QWEN35_DELTA_RECURRENCE_MATRIX + 9, normalized_key)
}

pub fn qwen35_delta_recurrence_decay_commitment(decay: &M31Matrix) -> FieldElement {
    m31_matrix_commitment(DOMAIN_QWEN35_DELTA_RECURRENCE_MATRIX + 10, decay)
}

pub fn qwen35_delta_recurrence_beta_commitment(beta: &M31Matrix) -> FieldElement {
    m31_matrix_commitment(DOMAIN_QWEN35_DELTA_RECURRENCE_MATRIX + 11, beta)
}

pub fn qwen35_delta_recurrence_transform_binding_witness(
    inputs: &Qwen35DeltaRecurrenceTransformInputs<'_>,
) -> Result<Qwen35DeltaRecurrenceTransformBindingWitness, String> {
    validate_transform_shapes(inputs)?;
    let qk_head_dim = inputs.query.cols / inputs.state_rows;
    let row_count = inputs.query.rows * inputs.state_rows * qk_head_dim;
    let mut rows = Vec::with_capacity(row_count);

    for token_idx in 0..inputs.query.rows {
        for state_row_idx in 0..inputs.state_rows {
            for qk_col_idx in 0..qk_head_dim {
                let qk_abs_col = state_row_idx * qk_head_dim + qk_col_idx;
                rows.push(Qwen35DeltaRecurrenceTransformBindingRow {
                    token_idx,
                    state_row_idx,
                    qk_col_idx,
                    query: inputs.query.get(token_idx, qk_abs_col),
                    key: inputs.key.get(token_idx, qk_abs_col),
                    scaled_query: inputs.scaled_query.get(token_idx, qk_abs_col),
                    normalized_key: inputs.normalized_key.get(token_idx, qk_abs_col),
                    a_gate: inputs.a_gate.get(token_idx, state_row_idx),
                    b_gate: inputs.b_gate.get(token_idx, state_row_idx),
                    a_log_weight: inputs.a_log_weight[state_row_idx],
                    dt_bias: inputs.dt_bias[state_row_idx],
                    decay: inputs.decay.get(token_idx, state_row_idx),
                    beta: inputs.beta.get(token_idx, state_row_idx),
                });
            }
        }
    }

    let witness_hash = qwen35_delta_recurrence_transform_binding_witness_hash(inputs, &rows)?;
    Ok(Qwen35DeltaRecurrenceTransformBindingWitness { rows, witness_hash })
}

pub fn verify_qwen35_delta_recurrence_transform_binding_witness(
    inputs: &Qwen35DeltaRecurrenceTransformInputs<'_>,
    witness: &Qwen35DeltaRecurrenceTransformBindingWitness,
) -> Result<(), String> {
    validate_transform_shapes(inputs)?;
    verify_qwen35_delta_recurrence_transform_binding_rows(inputs, &witness.rows)?;
    let expected_hash =
        qwen35_delta_recurrence_transform_binding_witness_hash(inputs, &witness.rows)?;
    if witness.witness_hash != expected_hash {
        return Err("DeltaRecurrence transform-binding witness hash mismatch".to_string());
    }
    Ok(())
}

pub fn verify_qwen35_delta_recurrence_transform_binding_rows(
    inputs: &Qwen35DeltaRecurrenceTransformInputs<'_>,
    rows: &[Qwen35DeltaRecurrenceTransformBindingRow],
) -> Result<(), String> {
    validate_transform_shapes(inputs)?;
    let qk_head_dim = inputs.query.cols / inputs.state_rows;
    let expected_rows = inputs.query.rows * inputs.state_rows * qk_head_dim;
    if rows.len() != expected_rows {
        return Err(format!(
            "DeltaRecurrence transform-binding rows {} != expected {}",
            rows.len(),
            expected_rows
        ));
    }

    for (row_idx, row) in rows.iter().enumerate() {
        let qk_col_idx = row_idx % qk_head_dim;
        let state_major = row_idx / qk_head_dim;
        let state_row_idx = state_major % inputs.state_rows;
        let token_idx = state_major / inputs.state_rows;

        if row.token_idx != token_idx
            || row.state_row_idx != state_row_idx
            || row.qk_col_idx != qk_col_idx
        {
            return Err(format!(
                "DeltaRecurrence transform-binding row {row_idx} has token={}, state_row={}, qk_col={}, expected token={}, state_row={}, qk_col={}",
                row.token_idx,
                row.state_row_idx,
                row.qk_col_idx,
                token_idx,
                state_row_idx,
                qk_col_idx
            ));
        }

        let qk_abs_col = state_row_idx * qk_head_dim + qk_col_idx;
        let expected = Qwen35DeltaRecurrenceTransformBindingRow {
            token_idx,
            state_row_idx,
            qk_col_idx,
            query: inputs.query.get(token_idx, qk_abs_col),
            key: inputs.key.get(token_idx, qk_abs_col),
            scaled_query: inputs.scaled_query.get(token_idx, qk_abs_col),
            normalized_key: inputs.normalized_key.get(token_idx, qk_abs_col),
            a_gate: inputs.a_gate.get(token_idx, state_row_idx),
            b_gate: inputs.b_gate.get(token_idx, state_row_idx),
            a_log_weight: inputs.a_log_weight[state_row_idx],
            dt_bias: inputs.dt_bias[state_row_idx],
            decay: inputs.decay.get(token_idx, state_row_idx),
            beta: inputs.beta.get(token_idx, state_row_idx),
        };
        if row != &expected {
            return Err(format!(
                "DeltaRecurrence transform-binding row {row_idx} does not match active transform tensors"
            ));
        }
    }

    Ok(())
}

pub fn qwen35_delta_recurrence_transform_binding_witness_hash(
    inputs: &Qwen35DeltaRecurrenceTransformInputs<'_>,
    rows: &[Qwen35DeltaRecurrenceTransformBindingRow],
) -> Result<FieldElement, String> {
    validate_transform_shapes(inputs)?;
    let mut felts = Vec::with_capacity(20 + rows.len() * 13);
    felts.extend([
        FieldElement::from(DOMAIN_QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING),
        FieldElement::from(inputs.mode.as_u64()),
        FieldElement::from(inputs.query.rows as u64),
        FieldElement::from(inputs.query.cols as u64),
        FieldElement::from(inputs.key.cols as u64),
        FieldElement::from(inputs.state_rows as u64),
        FieldElement::from(inputs.value_head_dim as u64),
        qwen35_delta_recurrence_query_commitment(inputs.query),
        qwen35_delta_recurrence_key_commitment(inputs.key),
        qwen35_delta_recurrence_a_gate_commitment(inputs.a_gate),
        qwen35_delta_recurrence_b_gate_commitment(inputs.b_gate),
        qwen35_delta_recurrence_a_log_weight_commitment(inputs.a_log_weight),
        qwen35_delta_recurrence_dt_bias_commitment(inputs.dt_bias),
        qwen35_delta_recurrence_scaled_query_commitment(inputs.scaled_query),
        qwen35_delta_recurrence_normalized_key_commitment(inputs.normalized_key),
        qwen35_delta_recurrence_decay_commitment(inputs.decay),
        qwen35_delta_recurrence_beta_commitment(inputs.beta),
    ]);
    for row in rows {
        felts.extend([
            FieldElement::from(row.token_idx as u64),
            FieldElement::from(row.state_row_idx as u64),
            FieldElement::from(row.qk_col_idx as u64),
            FieldElement::from(row.query.0 as u64),
            FieldElement::from(row.key.0 as u64),
            FieldElement::from(row.scaled_query.0 as u64),
            FieldElement::from(row.normalized_key.0 as u64),
            FieldElement::from(row.a_gate.0 as u64),
            FieldElement::from(row.b_gate.0 as u64),
            FieldElement::from(row.a_log_weight.0 as u64),
            FieldElement::from(row.dt_bias.0 as u64),
            FieldElement::from(row.decay.0 as u64),
            FieldElement::from(row.beta.0 as u64),
        ]);
    }
    Ok(starknet_crypto::poseidon_hash_many(&felts))
}

pub fn qwen35_delta_recurrence_transform_binding_trace(
    inputs: &Qwen35DeltaRecurrenceTransformInputs<'_>,
) -> Result<Qwen35DeltaRecurrenceTransformBindingTrace, String> {
    let witness = qwen35_delta_recurrence_transform_binding_witness(inputs)?;
    qwen35_delta_recurrence_transform_binding_trace_from_rows(
        inputs.query.rows,
        inputs.state_rows,
        inputs.query.cols / inputs.state_rows,
        &witness.rows,
    )
}

pub fn qwen35_delta_recurrence_transform_binding_trace_from_rows(
    seq_len: usize,
    state_rows: usize,
    qk_head_dim: usize,
    rows: &[Qwen35DeltaRecurrenceTransformBindingRow],
) -> Result<Qwen35DeltaRecurrenceTransformBindingTrace, String> {
    if seq_len == 0 || state_rows == 0 || qk_head_dim == 0 {
        return Err(
            "DeltaRecurrence transform-binding trace dimensions must be non-zero".to_string(),
        );
    }
    let n_real_rows = seq_len * state_rows * qk_head_dim;
    if rows.len() != n_real_rows {
        return Err(format!(
            "DeltaRecurrence transform-binding rows {} != expected {}",
            rows.len(),
            n_real_rows
        ));
    }

    let log_size = (n_real_rows.next_power_of_two().trailing_zeros())
        .max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE);
    let size = 1usize << log_size;

    let mut preprocessed = vec![vec![zero(); size]; 3];
    let mut execution = vec![vec![zero(); size]; 13];

    for (row_idx, row) in rows.iter().enumerate() {
        preprocessed[0][row_idx] = M31::from(row.token_idx as u32);
        preprocessed[1][row_idx] = M31::from(row.state_row_idx as u32);
        preprocessed[2][row_idx] = M31::from(row.qk_col_idx as u32);

        execution[0][row_idx] = M31::from(row.token_idx as u32);
        execution[1][row_idx] = M31::from(row.state_row_idx as u32);
        execution[2][row_idx] = M31::from(row.qk_col_idx as u32);
        execution[3][row_idx] = row.query;
        execution[4][row_idx] = row.key;
        execution[5][row_idx] = row.scaled_query;
        execution[6][row_idx] = row.normalized_key;
        execution[7][row_idx] = row.a_gate;
        execution[8][row_idx] = row.b_gate;
        execution[9][row_idx] = row.a_log_weight;
        execution[10][row_idx] = row.dt_bias;
        execution[11][row_idx] = row.decay;
        execution[12][row_idx] = row.beta;
    }

    Ok(Qwen35DeltaRecurrenceTransformBindingTrace {
        log_size,
        n_real_rows,
        preprocessed,
        execution,
    })
}

#[derive(Debug)]
struct Qwen35BetaSigmoidColumns {
    preprocessed: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    execution: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    table_input_col: Col<SimdBackend, BaseField>,
    table_output_col: Col<SimdBackend, BaseField>,
    trace_input_col: Col<SimdBackend, BaseField>,
    trace_output_col: Col<SimdBackend, BaseField>,
    trace_checksum: M31,
    multiplicities: Vec<M31>,
    log_size: u32,
}

pub fn qwen35_delta_recurrence_beta_sigmoid_table_commitment(table_log_size: u32) -> FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(DOMAIN_QWEN35_DELTA_RECURRENCE_BETA_SIGMOID_STATEMENT),
        FieldElement::from(ActivationType::Sigmoid.type_tag() as u64),
        FieldElement::from(table_log_size as u64),
    ])
}

pub fn qwen35_delta_recurrence_beta_sigmoid_statement_hash(
    layer_idx: usize,
    seq_len: usize,
    state_rows: usize,
    table_log_size: u32,
    trace_checksum: M31,
    table_commitment: FieldElement,
    b_gate_commitment: FieldElement,
    beta_commitment: FieldElement,
) -> FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(DOMAIN_QWEN35_DELTA_RECURRENCE_BETA_SIGMOID_STATEMENT),
        FieldElement::from(layer_idx as u64),
        FieldElement::from(seq_len as u64),
        FieldElement::from(state_rows as u64),
        FieldElement::from(table_log_size as u64),
        FieldElement::from(trace_checksum.0 as u64),
        table_commitment,
        b_gate_commitment,
        beta_commitment,
    ])
}

pub fn qwen35_delta_recurrence_beta_sigmoid_statement(
    layer_idx: usize,
    b_gate: &M31Matrix,
    beta: &M31Matrix,
    table_log_size: u32,
) -> Result<Qwen35DeltaRecurrenceBetaSigmoidStatement, String> {
    validate_beta_sigmoid_shapes(b_gate, beta)?;
    let table_commitment = qwen35_delta_recurrence_beta_sigmoid_table_commitment(table_log_size);
    let b_gate_commitment = qwen35_delta_recurrence_b_gate_commitment(b_gate);
    let beta_commitment = qwen35_delta_recurrence_beta_commitment(beta);
    let trace_checksum =
        qwen35_delta_recurrence_beta_sigmoid_trace_checksum(b_gate, beta, table_log_size)?;
    let statement_hash = qwen35_delta_recurrence_beta_sigmoid_statement_hash(
        layer_idx,
        b_gate.rows,
        b_gate.cols,
        table_log_size,
        trace_checksum,
        table_commitment,
        b_gate_commitment,
        beta_commitment,
    );

    Ok(Qwen35DeltaRecurrenceBetaSigmoidStatement {
        layer_idx,
        seq_len: b_gate.rows,
        state_rows: b_gate.cols,
        table_log_size,
        trace_checksum,
        table_commitment,
        b_gate_commitment,
        beta_commitment,
        statement_hash,
    })
}

fn qwen35_beta_sigmoid_table(table_log_size: u32) -> PrecomputedTable {
    PrecomputedTable::build(
        crate::gadgets::lookup_table::activations::sigmoid_approx,
        table_log_size.max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE),
    )
}

fn validate_beta_sigmoid_shapes(b_gate: &M31Matrix, beta: &M31Matrix) -> Result<(), String> {
    if b_gate.rows == 0 || b_gate.cols == 0 {
        return Err("DeltaRecurrence beta sigmoid b_gate must be non-empty".to_string());
    }
    if beta.rows != b_gate.rows || beta.cols != b_gate.cols {
        return Err(format!(
            "DeltaRecurrence beta sigmoid beta shape [{}x{}] != b_gate [{}x{}]",
            beta.rows, beta.cols, b_gate.rows, b_gate.cols
        ));
    }
    Ok(())
}

fn qwen35_beta_sigmoid_trace_row_term(token: M31, state_row: M31, input: M31, output: M31) -> M31 {
    M31::from(151u32)
        + token * M31::from(3u32)
        + state_row * M31::from(5u32)
        + input * M31::from(7u32)
        + output * M31::from(11u32)
}

pub fn qwen35_delta_recurrence_beta_sigmoid_trace_checksum(
    b_gate: &M31Matrix,
    beta: &M31Matrix,
    table_log_size: u32,
) -> Result<M31, String> {
    validate_beta_sigmoid_shapes(b_gate, beta)?;
    let n_real_rows = b_gate.rows * b_gate.cols;
    let required_log_size = (n_real_rows.max(1).next_power_of_two().trailing_zeros())
        .max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE);
    let log_size = table_log_size
        .max(required_log_size)
        .max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE);
    let size = 1usize << log_size;
    let table = qwen35_beta_sigmoid_table(log_size);
    let pad_input = table.inputs[0];
    let pad_output = table.outputs[0];

    let mut checksum = M31::from(0u32);
    for row_idx in 0..size {
        let (token, state_row, input, output) = if row_idx < n_real_rows {
            let token_idx = row_idx / b_gate.cols;
            let state_row_idx = row_idx % b_gate.cols;
            let input = b_gate.get(token_idx, state_row_idx);
            let expected = table.lookup(input).ok_or_else(|| {
                format!(
                    "DeltaRecurrence beta sigmoid input at row {row_idx} is outside table domain: {}",
                    input.0
                )
            })?;
            let output = beta.get(token_idx, state_row_idx);
            if output != expected {
                return Err(format!(
                    "DeltaRecurrence beta sigmoid output mismatch at row {row_idx}: got {}, expected {}",
                    output.0, expected.0
                ));
            }
            (
                M31::from(token_idx as u32),
                M31::from(state_row_idx as u32),
                input,
                output,
            )
        } else {
            (M31::from(0u32), M31::from(0u32), pad_input, pad_output)
        };
        checksum = checksum * M31::from(QWEN35_DELTA_RECURRENCE_TRACE_CHECKSUM_ALPHA)
            + qwen35_beta_sigmoid_trace_row_term(token, state_row, input, output);
    }
    Ok(checksum)
}

fn qwen35_beta_sigmoid_preprocessed_columns(
    log_size: u32,
    seq_len: usize,
    state_rows: usize,
    table: &PrecomputedTable,
) -> (
    Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    Col<SimdBackend, BaseField>,
    Col<SimdBackend, BaseField>,
) {
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();
    let mut table_input_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut table_output_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut token_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut state_row_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut is_first_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut is_last_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut has_next_col = Col::<SimdBackend, BaseField>::zeros(size);

    for idx in 0..size {
        table_input_col.set(idx, table.inputs[idx]);
        table_output_col.set(idx, table.outputs[idx]);
    }
    let n_real_rows = seq_len * state_rows;
    for row_idx in 0..n_real_rows.min(size) {
        let token_idx = row_idx / state_rows;
        let state_row_idx = row_idx % state_rows;
        token_col.set(row_idx, M31::from(token_idx as u32));
        state_row_col.set(row_idx, M31::from(state_row_idx as u32));
    }
    is_first_col.set(0, M31::from(1u32));
    is_last_col.set(size - 1, M31::from(1u32));
    for row_idx in 0..size.saturating_sub(1) {
        has_next_col.set(row_idx, M31::from(1u32));
    }

    (
        vec![
            CircleEvaluation::new(domain, table_input_col.clone()),
            CircleEvaluation::new(domain, table_output_col.clone()),
            CircleEvaluation::new(domain, token_col),
            CircleEvaluation::new(domain, state_row_col),
            CircleEvaluation::new(domain, is_first_col),
            CircleEvaluation::new(domain, is_last_col),
            CircleEvaluation::new(domain, has_next_col),
        ],
        table_input_col,
        table_output_col,
    )
}

fn qwen35_beta_sigmoid_columns(
    b_gate: &M31Matrix,
    beta: &M31Matrix,
    table_log_size: u32,
) -> Result<Qwen35BetaSigmoidColumns, String> {
    validate_beta_sigmoid_shapes(b_gate, beta)?;
    let n_real_rows = b_gate.rows * b_gate.cols;
    let required_log_size = (n_real_rows.max(1).next_power_of_two().trailing_zeros())
        .max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE);
    let log_size = table_log_size
        .max(required_log_size)
        .max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE);
    let table = qwen35_beta_sigmoid_table(log_size);
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();

    let inputs = &b_gate.data;
    let outputs = &beta.data;
    for (idx, (&input, &output)) in inputs.iter().zip(outputs.iter()).enumerate() {
        let expected = table.lookup(input).ok_or_else(|| {
            format!(
                "DeltaRecurrence beta sigmoid input at row {idx} is outside table domain: {}",
                input.0
            )
        })?;
        if output != expected {
            return Err(format!(
                "DeltaRecurrence beta sigmoid output mismatch at row {idx}: got {}, expected {}",
                output.0, expected.0
            ));
        }
    }

    let pad_input = table.inputs[0];
    let pad_output = table.outputs[0];
    let padding_count = size.saturating_sub(inputs.len());
    let mut multiplicities = compute_multiplicities(inputs, &table);
    if padding_count > 0 {
        multiplicities[0] += M31::from(padding_count as u32);
    }

    let (preprocessed, table_input_col, table_output_col) =
        qwen35_beta_sigmoid_preprocessed_columns(log_size, b_gate.rows, b_gate.cols, &table);

    let mut token_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut state_row_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut trace_input_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut trace_output_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut trace_acc_before_values = vec![M31::from(0u32); size];
    let mut trace_acc_after_values = vec![M31::from(0u32); size];
    let mut mult_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut trace_checksum = M31::from(0u32);
    for token_idx in 0..b_gate.rows {
        for state_row_idx in 0..b_gate.cols {
            let row_idx = token_idx * b_gate.cols + state_row_idx;
            let input = b_gate.get(token_idx, state_row_idx);
            let output = beta.get(token_idx, state_row_idx);
            trace_acc_before_values[row_idx] = trace_checksum;
            token_col.set(row_idx, M31::from(token_idx as u32));
            state_row_col.set(row_idx, M31::from(state_row_idx as u32));
            trace_input_col.set(row_idx, input);
            trace_output_col.set(row_idx, output);
            trace_checksum = trace_checksum
                * M31::from(QWEN35_DELTA_RECURRENCE_TRACE_CHECKSUM_ALPHA)
                + qwen35_beta_sigmoid_trace_row_term(
                    M31::from(token_idx as u32),
                    M31::from(state_row_idx as u32),
                    input,
                    output,
                );
            trace_acc_after_values[row_idx] = trace_checksum;
        }
    }
    for idx in inputs.len()..size {
        trace_acc_before_values[idx] = trace_checksum;
        trace_input_col.set(idx, pad_input);
        trace_output_col.set(idx, pad_output);
        trace_checksum = trace_checksum * M31::from(QWEN35_DELTA_RECURRENCE_TRACE_CHECKSUM_ALPHA)
            + qwen35_beta_sigmoid_trace_row_term(
                M31::from(0u32),
                M31::from(0u32),
                pad_input,
                pad_output,
            );
        trace_acc_after_values[idx] = trace_checksum;
    }
    let mut trace_acc_before_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut trace_acc_after_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut shifted_next_trace_acc_before_col = Col::<SimdBackend, BaseField>::zeros(size);
    for idx in 0..size {
        trace_acc_before_col.set(idx, trace_acc_before_values[idx]);
        trace_acc_after_col.set(idx, trace_acc_after_values[idx]);
        if idx + 1 < size {
            shifted_next_trace_acc_before_col.set(idx, trace_acc_before_values[idx + 1]);
        }
    }
    for (idx, &multiplicity) in multiplicities.iter().enumerate().take(size) {
        mult_col.set(idx, multiplicity);
    }

    let execution = vec![
        CircleEvaluation::new(domain, token_col),
        CircleEvaluation::new(domain, state_row_col),
        CircleEvaluation::new(domain, trace_input_col.clone()),
        CircleEvaluation::new(domain, trace_output_col.clone()),
        CircleEvaluation::new(domain, trace_acc_before_col),
        CircleEvaluation::new(domain, trace_acc_after_col),
        CircleEvaluation::new(domain, shifted_next_trace_acc_before_col),
        CircleEvaluation::new(domain, mult_col),
    ];

    Ok(Qwen35BetaSigmoidColumns {
        preprocessed,
        execution,
        table_input_col,
        table_output_col,
        trace_input_col,
        trace_output_col,
        trace_checksum,
        multiplicities,
        log_size,
    })
}

fn qwen35_beta_sigmoid_logup_trace(
    columns: &Qwen35BetaSigmoidColumns,
    lookup_elements: &ActivationRelation,
) -> (
    Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    use stwo::prover::backend::simd::m31::LOG_N_LANES;

    let size = 1usize << columns.log_size;
    let vec_size = size >> LOG_N_LANES;
    let tag_packed = PackedBaseField::broadcast(M31::from(ActivationType::Sigmoid.type_tag()));

    let mut logup_gen = LogupTraceGenerator::new(columns.log_size);
    let mut col_gen = logup_gen.new_col();
    for vec_row in 0..vec_size {
        let q_table: PackedSecureField = lookup_elements.lookup_elements().combine(&[
            tag_packed,
            columns.table_input_col.data[vec_row],
            columns.table_output_col.data[vec_row],
        ]);
        let q_trace: PackedSecureField = lookup_elements.lookup_elements().combine(&[
            tag_packed,
            columns.trace_input_col.data[vec_row],
            columns.trace_output_col.data[vec_row],
        ]);
        let mult_packed = qwen35_beta_sigmoid_mult_packed(&columns.multiplicities, vec_row);
        let numerator = q_table - mult_packed * q_trace;
        let denominator = q_table * q_trace;
        col_gen.write_frac(vec_row, numerator, denominator);
    }
    col_gen.finalize_col();
    logup_gen.finalize_last()
}

fn qwen35_beta_sigmoid_mult_packed(multiplicities: &[M31], vec_row: usize) -> PackedSecureField {
    let base = vec_row * 16;
    let mut values = [M31::from(0); 16];
    for (idx, value) in values.iter_mut().enumerate() {
        let multiplicity_idx = base + idx;
        if multiplicity_idx < multiplicities.len() {
            *value = multiplicities[multiplicity_idx];
        }
    }
    PackedBaseField::from_array(std::array::from_fn(|idx| values[idx])).into()
}

fn qwen35_beta_sigmoid_expected_preprocessed_root(
    table_log_size: u32,
    seq_len: usize,
    state_rows: usize,
) -> <Blake2sHash as MerkleHasherLifted>::Hash {
    let n_real_rows = seq_len * state_rows;
    let required_log_size = (n_real_rows.max(1).next_power_of_two().trailing_zeros())
        .max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE);
    let log_size = table_log_size
        .max(required_log_size)
        .max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE);
    let table = qwen35_beta_sigmoid_table(log_size);
    let log_size = table.log_size;
    let (preprocessed, _, _) =
        qwen35_beta_sigmoid_preprocessed_columns(log_size, seq_len, state_rows, &table);
    let pcs_config = PcsConfig::default();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_size + 1 + pcs_config.fri_config.log_blowup_factor)
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

#[derive(Debug)]
struct Qwen35NormColumns {
    preprocessed: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    execution: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    table_rms_sq_col: Col<SimdBackend, BaseField>,
    table_rsqrt_col: Col<SimdBackend, BaseField>,
    trace_rms_sq_col: Col<SimdBackend, BaseField>,
    trace_rsqrt_col: Col<SimdBackend, BaseField>,
    trace_checksum: M31,
    multiplicities: Vec<M31>,
    log_size: u32,
}

pub fn qwen35_delta_recurrence_norm_table_commitment(table_log_size: u32) -> FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(DOMAIN_QWEN35_DELTA_RECURRENCE_NORM_STATEMENT),
        FieldElement::from(table_log_size as u64),
    ])
}

pub fn qwen35_delta_recurrence_norm_statement_hash(
    kind: Qwen35DeltaRecurrenceNormKind,
    layer_idx: usize,
    seq_len: usize,
    state_rows: usize,
    qk_head_dim: usize,
    table_log_size: u32,
    trace_checksum: M31,
    table_commitment: FieldElement,
    input_commitment: FieldElement,
    output_commitment: FieldElement,
    post_scale: M31,
) -> FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(DOMAIN_QWEN35_DELTA_RECURRENCE_NORM_STATEMENT),
        FieldElement::from(kind.as_u64()),
        FieldElement::from(layer_idx as u64),
        FieldElement::from(seq_len as u64),
        FieldElement::from(state_rows as u64),
        FieldElement::from(qk_head_dim as u64),
        FieldElement::from(table_log_size as u64),
        FieldElement::from(trace_checksum.0 as u64),
        table_commitment,
        input_commitment,
        output_commitment,
        FieldElement::from(post_scale.0 as u64),
    ])
}

pub fn qwen35_delta_recurrence_norm_statement(
    kind: Qwen35DeltaRecurrenceNormKind,
    layer_idx: usize,
    input: &M31Matrix,
    output: &M31Matrix,
    state_rows: usize,
    table_log_size: u32,
    post_scale: M31,
) -> Result<Qwen35DeltaRecurrenceNormStatement, String> {
    validate_qwen35_norm_shapes(input, output, state_rows)?;
    let qk_head_dim = input.cols / state_rows;
    let table_commitment = qwen35_delta_recurrence_norm_table_commitment(table_log_size);
    let input_commitment = match kind {
        Qwen35DeltaRecurrenceNormKind::Query => qwen35_delta_recurrence_query_commitment(input),
        Qwen35DeltaRecurrenceNormKind::Key => qwen35_delta_recurrence_key_commitment(input),
    };
    let output_commitment = match kind {
        Qwen35DeltaRecurrenceNormKind::Query => {
            qwen35_delta_recurrence_scaled_query_commitment(output)
        }
        Qwen35DeltaRecurrenceNormKind::Key => {
            qwen35_delta_recurrence_normalized_key_commitment(output)
        }
    };
    let trace_checksum = qwen35_delta_recurrence_norm_trace_checksum(
        input,
        output,
        state_rows,
        table_log_size,
        post_scale,
    )?;
    let statement_hash = qwen35_delta_recurrence_norm_statement_hash(
        kind,
        layer_idx,
        input.rows,
        state_rows,
        qk_head_dim,
        table_log_size,
        trace_checksum,
        table_commitment,
        input_commitment,
        output_commitment,
        post_scale,
    );

    Ok(Qwen35DeltaRecurrenceNormStatement {
        kind,
        layer_idx,
        seq_len: input.rows,
        state_rows,
        qk_head_dim,
        table_log_size,
        trace_checksum,
        table_commitment,
        input_commitment,
        output_commitment,
        post_scale,
        statement_hash,
    })
}

fn validate_qwen35_norm_shapes(
    input: &M31Matrix,
    output: &M31Matrix,
    state_rows: usize,
) -> Result<(), String> {
    if input.rows == 0 || input.cols == 0 {
        return Err("DeltaRecurrence Q/K norm input must be non-empty".to_string());
    }
    if state_rows == 0 || input.cols % state_rows != 0 {
        return Err(format!(
            "DeltaRecurrence Q/K norm input width {} must divide by state_rows {}",
            input.cols, state_rows
        ));
    }
    if output.rows != input.rows || output.cols != input.cols {
        return Err(format!(
            "DeltaRecurrence Q/K norm output shape [{}x{}] != input [{}x{}]",
            output.rows, output.cols, input.rows, input.cols
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn qwen35_norm_trace_row_term(
    token: M31,
    state_row: M31,
    qk_col: M31,
    input: M31,
    output: M31,
    sq_term: M31,
    sq_prefix_before: M31,
    sq_prefix_after: M31,
    rms_sq: M31,
    rsqrt: M31,
) -> M31 {
    M31::from(181u32)
        + token * M31::from(3u32)
        + state_row * M31::from(5u32)
        + qk_col * M31::from(7u32)
        + input * M31::from(11u32)
        + output * M31::from(13u32)
        + sq_term * M31::from(17u32)
        + sq_prefix_before * M31::from(19u32)
        + sq_prefix_after * M31::from(23u32)
        + rms_sq * M31::from(29u32)
        + rsqrt * M31::from(31u32)
}

pub fn qwen35_delta_recurrence_norm_trace_checksum(
    input: &M31Matrix,
    output: &M31Matrix,
    state_rows: usize,
    table_log_size: u32,
    post_scale: M31,
) -> Result<M31, String> {
    validate_qwen35_norm_shapes(input, output, state_rows)?;
    let qk_head_dim = input.cols / state_rows;
    let n_real_rows = input.rows * state_rows * qk_head_dim;
    let required_log_size = (n_real_rows.max(1).next_power_of_two().trailing_zeros())
        .max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE);
    let log_size = table_log_size
        .max(required_log_size)
        .max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE);
    let size = 1usize << log_size;
    let table = build_rsqrt_table(log_size);
    let pad_rms_sq = table.inputs[0];
    let pad_rsqrt = table.outputs[0];
    let norm_accumulator_scale = M31::from(1u32);

    let mut checksum = M31::from(0u32);
    for token_idx in 0..input.rows {
        for state_row_idx in 0..state_rows {
            let mut sum_sq = M31::from(0u32);
            for qk_idx in 0..qk_head_dim {
                let col_idx = state_row_idx * qk_head_dim + qk_idx;
                let value = input.get(token_idx, col_idx);
                sum_sq += value * value;
            }
            let rms_sq = sum_sq * norm_accumulator_scale;
            let rsqrt = table.lookup(rms_sq).ok_or_else(|| {
                format!(
                    "DeltaRecurrence Q/K norm rms_sq {} is outside rsqrt table domain",
                    rms_sq.0
                )
            })?;

            let mut prefix = M31::from(0u32);
            for qk_idx in 0..qk_head_dim {
                let col_idx = state_row_idx * qk_head_dim + qk_idx;
                let value = input.get(token_idx, col_idx);
                let sq_term = value * value;
                let next_prefix = prefix + sq_term;
                let expected_output = value * rsqrt * post_scale;
                let actual_output = output.get(token_idx, col_idx);
                if actual_output != expected_output {
                    return Err(format!(
                        "DeltaRecurrence Q/K norm output mismatch at token={token_idx} state_row={state_row_idx} qk_col={qk_idx}: got {}, expected {}",
                        actual_output.0, expected_output.0
                    ));
                }
                checksum = checksum * M31::from(QWEN35_DELTA_RECURRENCE_TRACE_CHECKSUM_ALPHA)
                    + qwen35_norm_trace_row_term(
                        M31::from(token_idx as u32),
                        M31::from(state_row_idx as u32),
                        M31::from(qk_idx as u32),
                        value,
                        actual_output,
                        sq_term,
                        prefix,
                        next_prefix,
                        rms_sq,
                        rsqrt,
                    );
                prefix = next_prefix;
            }
        }
    }
    for _ in n_real_rows..size {
        checksum = checksum * M31::from(QWEN35_DELTA_RECURRENCE_TRACE_CHECKSUM_ALPHA)
            + qwen35_norm_trace_row_term(
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
            );
    }
    Ok(checksum)
}

fn qwen35_norm_multiplicities(trace_rms_sq: &[M31], table: &PrecomputedTable) -> Vec<M31> {
    let mut multiplicities = vec![M31::from(0u32); table.inputs.len()];
    for &value in trace_rms_sq {
        if let Some(idx) = table.lookup_index(value) {
            multiplicities[idx] += M31::from(1u32);
        }
    }
    multiplicities
}

fn qwen35_norm_preprocessed_columns(
    log_size: u32,
    seq_len: usize,
    state_rows: usize,
    qk_head_dim: usize,
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
    let mut state_row_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut qk_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut is_first_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut is_last_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut is_chain_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut is_first_global_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut is_last_global_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut has_next_col = Col::<SimdBackend, BaseField>::zeros(size);

    for (idx, (&input, &output)) in table.inputs.iter().zip(table.outputs.iter()).enumerate() {
        table_rms_sq_col.set(idx, input);
        table_rsqrt_col.set(idx, output);
    }

    let n_real_rows = seq_len * state_rows * qk_head_dim;
    for row_idx in 0..n_real_rows.min(size) {
        let qk_idx = row_idx % qk_head_dim;
        let state_row_idx = (row_idx / qk_head_dim) % state_rows;
        let token_idx = row_idx / (state_rows * qk_head_dim);
        token_col.set(row_idx, M31::from(token_idx as u32));
        state_row_col.set(row_idx, M31::from(state_row_idx as u32));
        qk_col.set(row_idx, M31::from(qk_idx as u32));
        is_first_col.set(row_idx, M31::from((qk_idx == 0) as u32));
        is_last_col.set(row_idx, M31::from((qk_idx + 1 == qk_head_dim) as u32));
        is_chain_col.set(row_idx, M31::from((qk_idx + 1 < qk_head_dim) as u32));
    }
    is_first_global_col.set(0, M31::from(1u32));
    is_last_global_col.set(size - 1, M31::from(1u32));
    for row_idx in 0..size.saturating_sub(1) {
        has_next_col.set(row_idx, M31::from(1u32));
    }

    (
        vec![
            CircleEvaluation::new(domain, table_rms_sq_col.clone()),
            CircleEvaluation::new(domain, table_rsqrt_col.clone()),
            CircleEvaluation::new(domain, token_col),
            CircleEvaluation::new(domain, state_row_col),
            CircleEvaluation::new(domain, qk_col),
            CircleEvaluation::new(domain, is_first_col),
            CircleEvaluation::new(domain, is_last_col),
            CircleEvaluation::new(domain, is_chain_col),
            CircleEvaluation::new(domain, is_first_global_col),
            CircleEvaluation::new(domain, is_last_global_col),
            CircleEvaluation::new(domain, has_next_col),
        ],
        table_rms_sq_col,
        table_rsqrt_col,
    )
}

fn qwen35_norm_columns(
    input: &M31Matrix,
    output: &M31Matrix,
    state_rows: usize,
    table_log_size: u32,
    post_scale: M31,
) -> Result<Qwen35NormColumns, String> {
    validate_qwen35_norm_shapes(input, output, state_rows)?;
    let qk_head_dim = input.cols / state_rows;
    let n_real_rows = input.rows * state_rows * qk_head_dim;
    let required_log_size = (n_real_rows.max(1).next_power_of_two().trailing_zeros())
        .max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE);
    let log_size = table_log_size
        .max(required_log_size)
        .max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE);
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();
    let table = build_rsqrt_table(log_size);
    let (preprocessed, table_rms_sq_col, table_rsqrt_col) =
        qwen35_norm_preprocessed_columns(log_size, input.rows, state_rows, qk_head_dim, &table);
    let norm_accumulator_scale = M31::from(1u32);

    let mut token_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut state_row_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut qk_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut input_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut output_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut sq_term_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut sq_prefix_before_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut sq_prefix_after_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut shifted_next_sq_prefix_before_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut rms_sq_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut rsqrt_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut shifted_next_rms_sq_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut shifted_next_rsqrt_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut trace_acc_before_values = vec![M31::from(0u32); size];
    let mut trace_acc_after_values = vec![M31::from(0u32); size];
    let mut trace_rms_values = vec![table.inputs[0]; size];
    let mut trace_rsqrt_values = vec![table.outputs[0]; size];
    let mut trace_checksum = M31::from(0u32);

    for token_idx in 0..input.rows {
        for state_row_idx in 0..state_rows {
            let mut sum_sq = M31::from(0u32);
            for qk_idx in 0..qk_head_dim {
                let col_idx = state_row_idx * qk_head_dim + qk_idx;
                let value = input.get(token_idx, col_idx);
                sum_sq += value * value;
            }
            let rms_sq = sum_sq * norm_accumulator_scale;
            let rsqrt = table.lookup(rms_sq).ok_or_else(|| {
                format!(
                    "DeltaRecurrence Q/K norm rms_sq {} is outside rsqrt table domain",
                    rms_sq.0
                )
            })?;

            let mut prefix = M31::from(0u32);
            for qk_idx in 0..qk_head_dim {
                let row_idx = ((token_idx * state_rows + state_row_idx) * qk_head_dim) + qk_idx;
                let col_idx = state_row_idx * qk_head_dim + qk_idx;
                let value = input.get(token_idx, col_idx);
                let sq_term = value * value;
                let next_prefix = prefix + sq_term;
                let expected_output = value * rsqrt * post_scale;
                let actual_output = output.get(token_idx, col_idx);
                if actual_output != expected_output {
                    return Err(format!(
                        "DeltaRecurrence Q/K norm output mismatch at token={token_idx} state_row={state_row_idx} qk_col={qk_idx}: got {}, expected {}",
                        actual_output.0, expected_output.0
                    ));
                }

                trace_acc_before_values[row_idx] = trace_checksum;
                token_col.set(row_idx, M31::from(token_idx as u32));
                state_row_col.set(row_idx, M31::from(state_row_idx as u32));
                qk_col.set(row_idx, M31::from(qk_idx as u32));
                input_col.set(row_idx, value);
                output_col.set(row_idx, actual_output);
                sq_term_col.set(row_idx, sq_term);
                sq_prefix_before_col.set(row_idx, prefix);
                sq_prefix_after_col.set(row_idx, next_prefix);
                shifted_next_sq_prefix_before_col.set(row_idx, next_prefix);
                rms_sq_col.set(row_idx, rms_sq);
                rsqrt_col.set(row_idx, rsqrt);
                shifted_next_rms_sq_col.set(row_idx, rms_sq);
                shifted_next_rsqrt_col.set(row_idx, rsqrt);
                trace_rms_values[row_idx] = rms_sq;
                trace_rsqrt_values[row_idx] = rsqrt;
                trace_checksum = trace_checksum
                    * M31::from(QWEN35_DELTA_RECURRENCE_TRACE_CHECKSUM_ALPHA)
                    + qwen35_norm_trace_row_term(
                        M31::from(token_idx as u32),
                        M31::from(state_row_idx as u32),
                        M31::from(qk_idx as u32),
                        value,
                        actual_output,
                        sq_term,
                        prefix,
                        next_prefix,
                        rms_sq,
                        rsqrt,
                    );
                trace_acc_after_values[row_idx] = trace_checksum;
                prefix = next_prefix;
            }
        }
    }
    for row_idx in n_real_rows..size {
        trace_acc_before_values[row_idx] = trace_checksum;
        rms_sq_col.set(row_idx, table.inputs[0]);
        rsqrt_col.set(row_idx, table.outputs[0]);
        shifted_next_rms_sq_col.set(row_idx, table.inputs[0]);
        shifted_next_rsqrt_col.set(row_idx, table.outputs[0]);
        trace_checksum = trace_checksum * M31::from(QWEN35_DELTA_RECURRENCE_TRACE_CHECKSUM_ALPHA)
            + qwen35_norm_trace_row_term(
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
            );
        trace_acc_after_values[row_idx] = trace_checksum;
    }
    let mut trace_acc_before_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut trace_acc_after_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut shifted_next_trace_acc_before_col = Col::<SimdBackend, BaseField>::zeros(size);
    for idx in 0..size {
        trace_acc_before_col.set(idx, trace_acc_before_values[idx]);
        trace_acc_after_col.set(idx, trace_acc_after_values[idx]);
        if idx + 1 < size {
            shifted_next_trace_acc_before_col.set(idx, trace_acc_before_values[idx + 1]);
        }
    }

    let mut multiplicities = qwen35_norm_multiplicities(&trace_rms_values, &table);
    let mut multiplicity_col = Col::<SimdBackend, BaseField>::zeros(size);
    for (idx, &multiplicity) in multiplicities.iter().enumerate().take(size) {
        multiplicity_col.set(idx, multiplicity);
    }

    let execution = vec![
        CircleEvaluation::new(domain, token_col),
        CircleEvaluation::new(domain, state_row_col),
        CircleEvaluation::new(domain, qk_col),
        CircleEvaluation::new(domain, input_col),
        CircleEvaluation::new(domain, output_col),
        CircleEvaluation::new(domain, sq_term_col),
        CircleEvaluation::new(domain, sq_prefix_before_col),
        CircleEvaluation::new(domain, sq_prefix_after_col),
        CircleEvaluation::new(domain, shifted_next_sq_prefix_before_col),
        CircleEvaluation::new(domain, rms_sq_col.clone()),
        CircleEvaluation::new(domain, rsqrt_col.clone()),
        CircleEvaluation::new(domain, shifted_next_rms_sq_col),
        CircleEvaluation::new(domain, shifted_next_rsqrt_col),
        CircleEvaluation::new(domain, trace_acc_before_col),
        CircleEvaluation::new(domain, trace_acc_after_col),
        CircleEvaluation::new(domain, shifted_next_trace_acc_before_col),
        CircleEvaluation::new(domain, multiplicity_col),
    ];

    Ok(Qwen35NormColumns {
        preprocessed,
        execution,
        table_rms_sq_col,
        table_rsqrt_col,
        trace_rms_sq_col: rms_sq_col,
        trace_rsqrt_col: rsqrt_col,
        trace_checksum,
        multiplicities: std::mem::take(&mut multiplicities),
        log_size,
    })
}

fn qwen35_norm_logup_trace(
    columns: &Qwen35NormColumns,
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
        let mult_packed = qwen35_norm_mult_packed(&columns.multiplicities, vec_row);
        let numerator = q_table - mult_packed * q_trace;
        let denominator = q_table * q_trace;
        col_gen.write_frac(vec_row, numerator, denominator);
    }
    col_gen.finalize_col();
    logup_gen.finalize_last()
}

fn qwen35_norm_mult_packed(multiplicities: &[M31], vec_row: usize) -> PackedSecureField {
    let base = vec_row * 16;
    let mut values = [M31::from(0); 16];
    for (idx, value) in values.iter_mut().enumerate() {
        let multiplicity_idx = base + idx;
        if multiplicity_idx < multiplicities.len() {
            *value = multiplicities[multiplicity_idx];
        }
    }
    PackedBaseField::from_array(std::array::from_fn(|idx| values[idx])).into()
}

fn qwen35_norm_expected_preprocessed_root(
    statement: &Qwen35DeltaRecurrenceNormStatement,
) -> <Blake2sHash as MerkleHasherLifted>::Hash {
    let table = build_rsqrt_table(statement.table_log_size);
    let (preprocessed, _, _) = qwen35_norm_preprocessed_columns(
        statement.table_log_size,
        statement.seq_len,
        statement.state_rows,
        statement.qk_head_dim,
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

pub fn prove_qwen35_delta_recurrence_norm_air(
    kind: Qwen35DeltaRecurrenceNormKind,
    layer_idx: usize,
    input: &M31Matrix,
    output: &M31Matrix,
    state_rows: usize,
    table_log_size: u32,
    post_scale: M31,
) -> Result<Qwen35DeltaRecurrenceNormProof<Blake2sHash>, Qwen35DeltaRecurrenceNormProofError> {
    let statement = qwen35_delta_recurrence_norm_statement(
        kind,
        layer_idx,
        input,
        output,
        state_rows,
        table_log_size,
        post_scale,
    )
    .map_err(Qwen35DeltaRecurrenceNormProofError::Witness)?;
    let columns = qwen35_norm_columns(input, output, state_rows, table_log_size, post_scale)
        .map_err(Qwen35DeltaRecurrenceNormProofError::Witness)?;
    if columns.log_size != statement.table_log_size {
        return Err(Qwen35DeltaRecurrenceNormProofError::Witness(format!(
            "DeltaRecurrence Q/K norm table_log_size {} is smaller than required log_size {}",
            statement.table_log_size, columns.log_size
        )));
    }
    if columns.trace_checksum != statement.trace_checksum {
        return Err(Qwen35DeltaRecurrenceNormProofError::Witness(
            "DeltaRecurrence Q/K norm trace checksum mismatch".to_string(),
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
    let (interaction_trace, claimed_sum) = qwen35_norm_logup_trace(&columns, &lookup_elements);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, BaseField>(
        interaction_trace,
    ));
    tree_builder.commit(channel);

    let component = FrameworkComponent::new(
        &mut TraceLocationAllocator::default(),
        Qwen35DeltaRecurrenceNormEval {
            log_n_rows: columns.log_size,
            instance_id: 0,
            lookup_elements,
            claimed_sum,
            inv_qk_head_dim: M31::from(1u32),
            post_scale: statement.post_scale,
            trace_checksum: statement.trace_checksum,
        },
        claimed_sum,
    );
    let stark_proof =
        prove::<SimdBackend, Blake2sMerkleChannel>(&[&component], channel, commitment_scheme)
            .map_err(|err| Qwen35DeltaRecurrenceNormProofError::Proving(format!("{err:?}")))?;

    Ok(Qwen35DeltaRecurrenceNormProof {
        stark_proof,
        claimed_sum,
        log_size: columns.log_size,
        n_real_rows: input.data.len(),
        statement,
    })
}

pub fn verify_qwen35_delta_recurrence_norm_air(
    proof: &Qwen35DeltaRecurrenceNormProof<Blake2sHash>,
) -> Result<(), Qwen35DeltaRecurrenceNormProofError> {
    verify_qwen35_delta_recurrence_norm_air_with_statement_hash(
        proof,
        proof.statement.statement_hash,
    )
}

pub fn verify_qwen35_delta_recurrence_norm_air_with_statement_hash(
    proof: &Qwen35DeltaRecurrenceNormProof<Blake2sHash>,
    expected_statement_hash: FieldElement,
) -> Result<(), Qwen35DeltaRecurrenceNormProofError> {
    if proof.statement.statement_hash != expected_statement_hash {
        return Err(Qwen35DeltaRecurrenceNormProofError::Verification(
            "DeltaRecurrence Q/K norm statement hash mismatch".to_string(),
        ));
    }
    if proof.log_size != proof.statement.table_log_size {
        return Err(Qwen35DeltaRecurrenceNormProofError::Verification(
            "DeltaRecurrence Q/K norm proof log_size must equal statement table_log_size"
                .to_string(),
        ));
    }
    let expected_table_commitment =
        qwen35_delta_recurrence_norm_table_commitment(proof.statement.table_log_size);
    if proof.statement.table_commitment != expected_table_commitment {
        return Err(Qwen35DeltaRecurrenceNormProofError::Verification(
            "DeltaRecurrence Q/K norm table commitment mismatch".to_string(),
        ));
    }
    let expected_hash = qwen35_delta_recurrence_norm_statement_hash(
        proof.statement.kind,
        proof.statement.layer_idx,
        proof.statement.seq_len,
        proof.statement.state_rows,
        proof.statement.qk_head_dim,
        proof.statement.table_log_size,
        proof.statement.trace_checksum,
        proof.statement.table_commitment,
        proof.statement.input_commitment,
        proof.statement.output_commitment,
        proof.statement.post_scale,
    );
    if proof.statement.statement_hash != expected_hash {
        return Err(Qwen35DeltaRecurrenceNormProofError::Verification(
            "DeltaRecurrence Q/K norm statement is malformed".to_string(),
        ));
    }
    if proof.stark_proof.commitments.len() < 3 {
        return Err(Qwen35DeltaRecurrenceNormProofError::Verification(format!(
            "expected at least 3 commitment trees for Q/K norm LogUp, got {}",
            proof.stark_proof.commitments.len()
        )));
    }
    let expected_root = qwen35_norm_expected_preprocessed_root(&proof.statement);
    if proof.stark_proof.commitments[0] != expected_root {
        return Err(Qwen35DeltaRecurrenceNormProofError::Verification(
            "DeltaRecurrence Q/K norm preprocessed root mismatch".to_string(),
        ));
    }

    let pcs_config = PcsConfig::default();
    let mut allocator = TraceLocationAllocator::default();
    let dummy_component = FrameworkComponent::new(
        &mut allocator,
        Qwen35DeltaRecurrenceNormEval {
            log_n_rows: proof.log_size,
            instance_id: 0,
            lookup_elements: RMSNormRelation::dummy(),
            claimed_sum: proof.claimed_sum,
            inv_qk_head_dim: M31::from(1u32),
            post_scale: proof.statement.post_scale,
            trace_checksum: proof.statement.trace_checksum,
        },
        proof.claimed_sum,
    );
    let bounds = Component::trace_log_degree_bounds(&dummy_component);
    if proof.stark_proof.commitments.len() < bounds.len() {
        return Err(Qwen35DeltaRecurrenceNormProofError::Verification(format!(
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
        Qwen35DeltaRecurrenceNormEval {
            log_n_rows: proof.log_size,
            instance_id: 0,
            lookup_elements,
            claimed_sum: proof.claimed_sum,
            inv_qk_head_dim: M31::from(1u32),
            post_scale: proof.statement.post_scale,
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
    .map_err(|err| Qwen35DeltaRecurrenceNormProofError::Verification(format!("{err:?}")))
}

#[derive(Debug)]
struct Qwen35DecayColumns {
    preprocessed: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    execution: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    softplus_table_input_col: Col<SimdBackend, BaseField>,
    softplus_table_output_col: Col<SimdBackend, BaseField>,
    exp_a_log_table_input_col: Col<SimdBackend, BaseField>,
    exp_a_log_table_output_col: Col<SimdBackend, BaseField>,
    decay_exp_table_input_col: Col<SimdBackend, BaseField>,
    decay_exp_table_output_col: Col<SimdBackend, BaseField>,
    trace_a_sum_col: Col<SimdBackend, BaseField>,
    trace_softplus_col: Col<SimdBackend, BaseField>,
    trace_a_log_weight_col: Col<SimdBackend, BaseField>,
    trace_exp_a_log_col: Col<SimdBackend, BaseField>,
    trace_neg_product_col: Col<SimdBackend, BaseField>,
    trace_decay_col: Col<SimdBackend, BaseField>,
    trace_checksum: M31,
    softplus_multiplicities: Vec<M31>,
    exp_a_log_multiplicities: Vec<M31>,
    decay_multiplicities: Vec<M31>,
    log_size: u32,
}

pub fn qwen35_delta_recurrence_decay_table_commitments(
    table_log_size: u32,
) -> (FieldElement, FieldElement, FieldElement) {
    let softplus = starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(DOMAIN_QWEN35_DELTA_RECURRENCE_DECAY_STATEMENT),
        FieldElement::from(ActivationType::Softplus.type_tag() as u64),
        FieldElement::from(table_log_size as u64),
    ]);
    let exp = starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(DOMAIN_QWEN35_DELTA_RECURRENCE_DECAY_STATEMENT),
        FieldElement::from(ActivationType::Softmax.type_tag() as u64),
        FieldElement::from(table_log_size as u64),
    ]);
    let decay = starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(DOMAIN_QWEN35_DELTA_RECURRENCE_DECAY_STATEMENT),
        FieldElement::from(0x4445434159u64),
        FieldElement::from(table_log_size as u64),
    ]);
    (softplus, exp, decay)
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_delta_recurrence_decay_statement_hash(
    layer_idx: usize,
    seq_len: usize,
    state_rows: usize,
    table_log_size: u32,
    trace_checksum: M31,
    softplus_table_commitment: FieldElement,
    exp_table_commitment: FieldElement,
    decay_table_commitment: FieldElement,
    a_gate_commitment: FieldElement,
    a_log_weight_commitment: FieldElement,
    dt_bias_commitment: FieldElement,
    decay_commitment: FieldElement,
) -> FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(DOMAIN_QWEN35_DELTA_RECURRENCE_DECAY_STATEMENT),
        FieldElement::from(layer_idx as u64),
        FieldElement::from(seq_len as u64),
        FieldElement::from(state_rows as u64),
        FieldElement::from(table_log_size as u64),
        FieldElement::from(trace_checksum.0 as u64),
        softplus_table_commitment,
        exp_table_commitment,
        decay_table_commitment,
        a_gate_commitment,
        a_log_weight_commitment,
        dt_bias_commitment,
        decay_commitment,
    ])
}

pub fn qwen35_delta_recurrence_decay_statement(
    layer_idx: usize,
    a_gate: &M31Matrix,
    a_log_weight: &[M31],
    dt_bias: &[M31],
    decay: &M31Matrix,
    table_log_size: u32,
) -> Result<Qwen35DeltaRecurrenceDecayStatement, String> {
    validate_decay_shapes(a_gate, a_log_weight, dt_bias, decay)?;
    let (softplus_table_commitment, exp_table_commitment, decay_table_commitment) =
        qwen35_delta_recurrence_decay_table_commitments(table_log_size);
    let a_gate_commitment = qwen35_delta_recurrence_a_gate_commitment(a_gate);
    let a_log_weight_commitment = qwen35_delta_recurrence_a_log_weight_commitment(a_log_weight);
    let dt_bias_commitment = qwen35_delta_recurrence_dt_bias_commitment(dt_bias);
    let decay_commitment = qwen35_delta_recurrence_decay_commitment(decay);
    let trace_checksum = qwen35_delta_recurrence_decay_trace_checksum(
        a_gate,
        a_log_weight,
        dt_bias,
        decay,
        table_log_size,
    )?;
    let statement_hash = qwen35_delta_recurrence_decay_statement_hash(
        layer_idx,
        a_gate.rows,
        a_gate.cols,
        table_log_size,
        trace_checksum,
        softplus_table_commitment,
        exp_table_commitment,
        decay_table_commitment,
        a_gate_commitment,
        a_log_weight_commitment,
        dt_bias_commitment,
        decay_commitment,
    );

    Ok(Qwen35DeltaRecurrenceDecayStatement {
        layer_idx,
        seq_len: a_gate.rows,
        state_rows: a_gate.cols,
        table_log_size,
        trace_checksum,
        softplus_table_commitment,
        exp_table_commitment,
        decay_table_commitment,
        a_gate_commitment,
        a_log_weight_commitment,
        dt_bias_commitment,
        decay_commitment,
        statement_hash,
    })
}

fn qwen35_softplus_table(table_log_size: u32) -> PrecomputedTable {
    PrecomputedTable::build(
        crate::gadgets::lookup_table::activations::softplus_approx,
        table_log_size.max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE),
    )
}

fn qwen35_exp_table(table_log_size: u32) -> PrecomputedTable {
    PrecomputedTable::build(
        crate::gadgets::lookup_table::activations::softmax_exp,
        table_log_size.max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE),
    )
}

fn qwen35_decay_exp_table(table_log_size: u32) -> PrecomputedTable {
    PrecomputedTable::build(
        |x| crate::gadgets::lookup_table::activations::softmax_exp(M31::from(0u32) - x),
        table_log_size.max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE),
    )
}

fn validate_decay_shapes(
    a_gate: &M31Matrix,
    a_log_weight: &[M31],
    dt_bias: &[M31],
    decay: &M31Matrix,
) -> Result<(), String> {
    if a_gate.rows == 0 || a_gate.cols == 0 {
        return Err("DeltaRecurrence decay a_gate must be non-empty".to_string());
    }
    if decay.rows != a_gate.rows || decay.cols != a_gate.cols {
        return Err(format!(
            "DeltaRecurrence decay shape [{}x{}] != a_gate [{}x{}]",
            decay.rows, decay.cols, a_gate.rows, a_gate.cols
        ));
    }
    if a_log_weight.len() != a_gate.cols || dt_bias.len() != a_gate.cols {
        return Err(format!(
            "DeltaRecurrence decay vector widths a_log_weight={} dt_bias={} must equal state_rows {}",
            a_log_weight.len(),
            dt_bias.len(),
            a_gate.cols
        ));
    }
    Ok(())
}

fn qwen35_decay_trace_row_term(
    token: M31,
    state_row: M31,
    a_gate: M31,
    dt_bias: M31,
    a_sum: M31,
    softplus: M31,
    a_log_weight: M31,
    exp_a_log: M31,
    product: M31,
    decay: M31,
) -> M31 {
    M31::from(101u32)
        + token * M31::from(3u32)
        + state_row * M31::from(5u32)
        + a_gate * M31::from(7u32)
        + dt_bias * M31::from(11u32)
        + a_sum * M31::from(13u32)
        + softplus * M31::from(17u32)
        + a_log_weight * M31::from(19u32)
        + exp_a_log * M31::from(23u32)
        + product * M31::from(29u32)
        + decay * M31::from(31u32)
}

pub fn qwen35_delta_recurrence_decay_trace_checksum(
    a_gate: &M31Matrix,
    a_log_weight: &[M31],
    dt_bias: &[M31],
    decay: &M31Matrix,
    table_log_size: u32,
) -> Result<M31, String> {
    validate_decay_shapes(a_gate, a_log_weight, dt_bias, decay)?;
    let n_real_rows = a_gate.rows * a_gate.cols;
    let required_log_size = (n_real_rows.max(1).next_power_of_two().trailing_zeros())
        .max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE);
    let log_size = table_log_size
        .max(required_log_size)
        .max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE);
    let size = 1usize << log_size;
    let softplus_table = qwen35_softplus_table(log_size);
    let exp_table = qwen35_exp_table(log_size);
    let decay_table = qwen35_decay_exp_table(log_size);
    let pad_softplus = softplus_table.outputs[0];
    let pad_exp_input = exp_table.inputs[0];
    let pad_exp = exp_table.outputs[0];
    let pad_decay_input = M31::from(((pad_exp.0 as u64 * pad_softplus.0 as u64) >> 16) as u32);
    let pad_decay = decay_table.lookup(pad_decay_input).ok_or_else(|| {
        format!(
            "DeltaRecurrence decay padding input is outside decay table: {}",
            pad_decay_input.0
        )
    })?;

    let mut checksum = M31::from(0u32);
    for row_idx in 0..size {
        let (
            token,
            state_row,
            a_gate_value,
            dt_bias_value,
            a_sum,
            softplus,
            a_log,
            exp_a_log,
            product,
            decay_value,
        ) = if row_idx < n_real_rows {
            let token_idx = row_idx / a_gate.cols;
            let state_row_idx = row_idx % a_gate.cols;
            let a_gate_value = a_gate.get(token_idx, state_row_idx);
            let dt_bias_value = dt_bias[state_row_idx];
            let a_sum = a_gate_value + dt_bias_value;
            let softplus = softplus_table.lookup(a_sum).ok_or_else(|| {
                format!(
                    "DeltaRecurrence decay a+dt_bias at row {row_idx} is outside softplus table: {}",
                    a_sum.0
                )
            })?;
            let a_log = a_log_weight[state_row_idx];
            let exp_a_log = exp_table.lookup(a_log).ok_or_else(|| {
                format!(
                    "DeltaRecurrence decay a_log_weight at state row {state_row_idx} is outside exp table: {}",
                    a_log.0
                )
            })?;
            let product = M31::from(((exp_a_log.0 as u64 * softplus.0 as u64) >> 16) as u32);
            let expected_decay = decay_table.lookup(product).ok_or_else(|| {
                format!(
                    "DeltaRecurrence decay exp input at row {row_idx} is outside exp table: {}",
                    product.0
                )
            })?;
            let actual_decay = decay.get(token_idx, state_row_idx);
            if actual_decay != expected_decay {
                return Err(format!(
                    "DeltaRecurrence decay output mismatch at token={token_idx} state_row={state_row_idx}: got {}, expected {}",
                    actual_decay.0, expected_decay.0
                ));
            }
            (
                M31::from(token_idx as u32),
                M31::from(state_row_idx as u32),
                a_gate_value,
                dt_bias_value,
                a_sum,
                softplus,
                a_log,
                exp_a_log,
                product,
                actual_decay,
            )
        } else {
            (
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                pad_softplus,
                pad_exp_input,
                pad_exp,
                pad_decay_input,
                pad_decay,
            )
        };
        checksum = checksum * M31::from(QWEN35_DELTA_RECURRENCE_TRACE_CHECKSUM_ALPHA)
            + qwen35_decay_trace_row_term(
                token,
                state_row,
                a_gate_value,
                dt_bias_value,
                a_sum,
                softplus,
                a_log,
                exp_a_log,
                product,
                decay_value,
            );
    }
    Ok(checksum)
}

fn qwen35_decay_preprocessed_columns(
    log_size: u32,
    seq_len: usize,
    state_rows: usize,
    softplus_table: &PrecomputedTable,
    exp_table: &PrecomputedTable,
    decay_table: &PrecomputedTable,
) -> (
    Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    Col<SimdBackend, BaseField>,
    Col<SimdBackend, BaseField>,
    Col<SimdBackend, BaseField>,
    Col<SimdBackend, BaseField>,
    Col<SimdBackend, BaseField>,
    Col<SimdBackend, BaseField>,
) {
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();
    let mut softplus_input_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut softplus_output_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut exp_a_log_input_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut exp_a_log_output_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut decay_exp_input_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut decay_exp_output_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut token_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut state_row_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut is_first_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut is_last_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut has_next_col = Col::<SimdBackend, BaseField>::zeros(size);

    for idx in 0..size {
        softplus_input_col.set(idx, softplus_table.inputs[idx]);
        softplus_output_col.set(idx, softplus_table.outputs[idx]);
        exp_a_log_input_col.set(idx, exp_table.inputs[idx]);
        exp_a_log_output_col.set(idx, exp_table.outputs[idx]);
        decay_exp_input_col.set(idx, decay_table.inputs[idx]);
        decay_exp_output_col.set(idx, decay_table.outputs[idx]);
    }
    let n_real_rows = seq_len * state_rows;
    for row_idx in 0..n_real_rows.min(size) {
        let token_idx = row_idx / state_rows;
        let state_row_idx = row_idx % state_rows;
        token_col.set(row_idx, M31::from(token_idx as u32));
        state_row_col.set(row_idx, M31::from(state_row_idx as u32));
    }
    is_first_col.set(0, M31::from(1u32));
    is_last_col.set(size - 1, M31::from(1u32));
    for row_idx in 0..size.saturating_sub(1) {
        has_next_col.set(row_idx, M31::from(1u32));
    }

    (
        vec![
            CircleEvaluation::new(domain, softplus_input_col.clone()),
            CircleEvaluation::new(domain, softplus_output_col.clone()),
            CircleEvaluation::new(domain, exp_a_log_input_col.clone()),
            CircleEvaluation::new(domain, exp_a_log_output_col.clone()),
            CircleEvaluation::new(domain, decay_exp_input_col.clone()),
            CircleEvaluation::new(domain, decay_exp_output_col.clone()),
            CircleEvaluation::new(domain, token_col),
            CircleEvaluation::new(domain, state_row_col),
            CircleEvaluation::new(domain, is_first_col),
            CircleEvaluation::new(domain, is_last_col),
            CircleEvaluation::new(domain, has_next_col),
        ],
        softplus_input_col,
        softplus_output_col,
        exp_a_log_input_col,
        exp_a_log_output_col,
        decay_exp_input_col,
        decay_exp_output_col,
    )
}

fn qwen35_decay_multiplicities(trace_inputs: &[M31], table: &PrecomputedTable) -> Vec<M31> {
    compute_multiplicities(trace_inputs, table)
}

fn qwen35_decay_columns(
    a_gate: &M31Matrix,
    a_log_weight: &[M31],
    dt_bias: &[M31],
    decay: &M31Matrix,
    table_log_size: u32,
) -> Result<Qwen35DecayColumns, String> {
    validate_decay_shapes(a_gate, a_log_weight, dt_bias, decay)?;
    let n_real_rows = a_gate.rows * a_gate.cols;
    let required_log_size = (n_real_rows.max(1).next_power_of_two().trailing_zeros())
        .max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE);
    let log_size = table_log_size
        .max(required_log_size)
        .max(QWEN35_DELTA_RECURRENCE_TRANSFORM_BINDING_MIN_LOG_SIZE);
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();
    let softplus_table = qwen35_softplus_table(log_size);
    let exp_table = qwen35_exp_table(log_size);
    let decay_table = qwen35_decay_exp_table(log_size);
    let (
        preprocessed,
        softplus_table_input_col,
        softplus_table_output_col,
        exp_a_log_table_input_col,
        exp_a_log_table_output_col,
        decay_exp_table_input_col,
        decay_exp_table_output_col,
    ) = qwen35_decay_preprocessed_columns(
        log_size,
        a_gate.rows,
        a_gate.cols,
        &softplus_table,
        &exp_table,
        &decay_table,
    );

    let mut token_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut state_row_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut a_gate_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut dt_bias_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut a_sum_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut softplus_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut a_log_weight_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut exp_a_log_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut neg_product_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut decay_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut trace_acc_before_values = vec![M31::from(0u32); size];
    let mut trace_acc_after_values = vec![M31::from(0u32); size];
    let pad_softplus_input = softplus_table.inputs[0];
    let pad_softplus = softplus_table.outputs[0];
    let pad_exp_input = exp_table.inputs[0];
    let pad_exp = exp_table.outputs[0];
    let pad_decay_input = M31::from(((pad_exp.0 as u64 * pad_softplus.0 as u64) >> 16) as u32);
    let pad_decay = decay_table.lookup(pad_decay_input).ok_or_else(|| {
        format!(
            "DeltaRecurrence decay padding input is outside decay table: {}",
            pad_decay_input.0
        )
    })?;
    let mut softplus_inputs = vec![pad_softplus_input; size];
    let mut exp_a_log_inputs = vec![pad_exp_input; size];
    let mut decay_inputs = vec![pad_decay_input; size];
    let mut trace_checksum = M31::from(0u32);

    for token_idx in 0..a_gate.rows {
        for state_row_idx in 0..a_gate.cols {
            let row_idx = token_idx * a_gate.cols + state_row_idx;
            trace_acc_before_values[row_idx] = trace_checksum;
            let a_gate_value = a_gate.get(token_idx, state_row_idx);
            let dt_bias_value = dt_bias[state_row_idx];
            let a_sum = a_gate_value + dt_bias_value;
            let softplus = softplus_table.lookup(a_sum).ok_or_else(|| {
                format!(
                    "DeltaRecurrence decay a+dt_bias at row {row_idx} is outside softplus table: {}",
                    a_sum.0
                )
            })?;
            let a_log = a_log_weight[state_row_idx];
            let exp_a_log = exp_table.lookup(a_log).ok_or_else(|| {
                format!(
                    "DeltaRecurrence decay a_log_weight at state row {state_row_idx} is outside exp table: {}",
                    a_log.0
                )
            })?;
            let neg_product = M31::from(((exp_a_log.0 as u64 * softplus.0 as u64) >> 16) as u32);
            let expected_decay = decay_table.lookup(neg_product).ok_or_else(|| {
                format!(
                    "DeltaRecurrence decay exp input at row {row_idx} is outside exp table: {}",
                    neg_product.0
                )
            })?;
            let actual_decay = decay.get(token_idx, state_row_idx);
            if actual_decay != expected_decay {
                return Err(format!(
                    "DeltaRecurrence decay output mismatch at token={token_idx} state_row={state_row_idx}: got {}, expected {}",
                    actual_decay.0, expected_decay.0
                ));
            }

            token_col.set(row_idx, M31::from(token_idx as u32));
            state_row_col.set(row_idx, M31::from(state_row_idx as u32));
            a_gate_col.set(row_idx, a_gate_value);
            dt_bias_col.set(row_idx, dt_bias_value);
            a_sum_col.set(row_idx, a_sum);
            softplus_col.set(row_idx, softplus);
            a_log_weight_col.set(row_idx, a_log);
            exp_a_log_col.set(row_idx, exp_a_log);
            neg_product_col.set(row_idx, neg_product);
            decay_col.set(row_idx, actual_decay);
            softplus_inputs[row_idx] = a_sum;
            exp_a_log_inputs[row_idx] = a_log;
            decay_inputs[row_idx] = neg_product;
            trace_checksum = trace_checksum
                * M31::from(QWEN35_DELTA_RECURRENCE_TRACE_CHECKSUM_ALPHA)
                + qwen35_decay_trace_row_term(
                    M31::from(token_idx as u32),
                    M31::from(state_row_idx as u32),
                    a_gate_value,
                    dt_bias_value,
                    a_sum,
                    softplus,
                    a_log,
                    exp_a_log,
                    neg_product,
                    actual_decay,
                );
            trace_acc_after_values[row_idx] = trace_checksum;
        }
    }
    for row_idx in n_real_rows..size {
        trace_acc_before_values[row_idx] = trace_checksum;
        softplus_col.set(row_idx, pad_softplus);
        exp_a_log_col.set(row_idx, pad_exp);
        neg_product_col.set(row_idx, pad_decay_input);
        decay_col.set(row_idx, pad_decay);
        trace_checksum = trace_checksum * M31::from(QWEN35_DELTA_RECURRENCE_TRACE_CHECKSUM_ALPHA)
            + qwen35_decay_trace_row_term(
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                M31::from(0u32),
                pad_softplus,
                pad_exp_input,
                pad_exp,
                pad_decay_input,
                pad_decay,
            );
        trace_acc_after_values[row_idx] = trace_checksum;
    }
    let mut trace_acc_before_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut trace_acc_after_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut shifted_next_trace_acc_before_col = Col::<SimdBackend, BaseField>::zeros(size);
    for idx in 0..size {
        trace_acc_before_col.set(idx, trace_acc_before_values[idx]);
        trace_acc_after_col.set(idx, trace_acc_after_values[idx]);
        if idx + 1 < size {
            shifted_next_trace_acc_before_col.set(idx, trace_acc_before_values[idx + 1]);
        }
    }

    let softplus_multiplicities = qwen35_decay_multiplicities(&softplus_inputs, &softplus_table);
    let exp_a_log_multiplicities = qwen35_decay_multiplicities(&exp_a_log_inputs, &exp_table);
    let decay_multiplicities = qwen35_decay_multiplicities(&decay_inputs, &decay_table);
    let mut softplus_multiplicity_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut exp_a_log_multiplicity_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut decay_multiplicity_col = Col::<SimdBackend, BaseField>::zeros(size);
    for idx in 0..size {
        softplus_multiplicity_col.set(idx, softplus_multiplicities[idx]);
        exp_a_log_multiplicity_col.set(idx, exp_a_log_multiplicities[idx]);
        decay_multiplicity_col.set(idx, decay_multiplicities[idx]);
    }

    let execution = vec![
        CircleEvaluation::new(domain, token_col),
        CircleEvaluation::new(domain, state_row_col),
        CircleEvaluation::new(domain, a_gate_col),
        CircleEvaluation::new(domain, dt_bias_col),
        CircleEvaluation::new(domain, a_sum_col.clone()),
        CircleEvaluation::new(domain, softplus_col.clone()),
        CircleEvaluation::new(domain, a_log_weight_col.clone()),
        CircleEvaluation::new(domain, exp_a_log_col.clone()),
        CircleEvaluation::new(domain, neg_product_col.clone()),
        CircleEvaluation::new(domain, decay_col.clone()),
        CircleEvaluation::new(domain, trace_acc_before_col),
        CircleEvaluation::new(domain, trace_acc_after_col),
        CircleEvaluation::new(domain, shifted_next_trace_acc_before_col),
        CircleEvaluation::new(domain, softplus_multiplicity_col),
        CircleEvaluation::new(domain, exp_a_log_multiplicity_col),
        CircleEvaluation::new(domain, decay_multiplicity_col),
    ];

    Ok(Qwen35DecayColumns {
        preprocessed,
        execution,
        softplus_table_input_col,
        softplus_table_output_col,
        exp_a_log_table_input_col,
        exp_a_log_table_output_col,
        decay_exp_table_input_col,
        decay_exp_table_output_col,
        trace_a_sum_col: a_sum_col,
        trace_softplus_col: softplus_col,
        trace_a_log_weight_col: a_log_weight_col,
        trace_exp_a_log_col: exp_a_log_col,
        trace_neg_product_col: neg_product_col,
        trace_decay_col: decay_col,
        trace_checksum,
        softplus_multiplicities,
        exp_a_log_multiplicities,
        decay_multiplicities,
        log_size,
    })
}

fn qwen35_decay_logup_trace(
    columns: &Qwen35DecayColumns,
    lookup_elements: &ActivationRelation,
) -> (
    Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    use stwo::prover::backend::simd::m31::LOG_N_LANES;

    let size = 1usize << columns.log_size;
    let vec_size = size >> LOG_N_LANES;
    let softplus_tag = PackedBaseField::broadcast(M31::from(ActivationType::Softplus.type_tag()));
    let exp_tag = PackedBaseField::broadcast(M31::from(ActivationType::Softmax.type_tag()));

    let mut logup_gen = LogupTraceGenerator::new(columns.log_size);
    let mut softplus_col = logup_gen.new_col();
    for vec_row in 0..vec_size {
        let q_table: PackedSecureField = lookup_elements.lookup_elements().combine(&[
            softplus_tag,
            columns.softplus_table_input_col.data[vec_row],
            columns.softplus_table_output_col.data[vec_row],
        ]);
        let q_trace: PackedSecureField = lookup_elements.lookup_elements().combine(&[
            softplus_tag,
            columns.trace_a_sum_col.data[vec_row],
            columns.trace_softplus_col.data[vec_row],
        ]);
        let mult_packed =
            qwen35_beta_sigmoid_mult_packed(&columns.softplus_multiplicities, vec_row);
        softplus_col.write_frac(vec_row, q_table - mult_packed * q_trace, q_table * q_trace);
    }
    softplus_col.finalize_col();

    let mut exp_a_log_col = logup_gen.new_col();
    for vec_row in 0..vec_size {
        let q_table: PackedSecureField = lookup_elements.lookup_elements().combine(&[
            exp_tag,
            columns.exp_a_log_table_input_col.data[vec_row],
            columns.exp_a_log_table_output_col.data[vec_row],
        ]);
        let q_trace: PackedSecureField = lookup_elements.lookup_elements().combine(&[
            exp_tag,
            columns.trace_a_log_weight_col.data[vec_row],
            columns.trace_exp_a_log_col.data[vec_row],
        ]);
        let mult_packed =
            qwen35_beta_sigmoid_mult_packed(&columns.exp_a_log_multiplicities, vec_row);
        exp_a_log_col.write_frac(vec_row, q_table - mult_packed * q_trace, q_table * q_trace);
    }
    exp_a_log_col.finalize_col();

    let mut decay_col = logup_gen.new_col();
    for vec_row in 0..vec_size {
        let q_table: PackedSecureField = lookup_elements.lookup_elements().combine(&[
            exp_tag,
            columns.decay_exp_table_input_col.data[vec_row],
            columns.decay_exp_table_output_col.data[vec_row],
        ]);
        let q_trace: PackedSecureField = lookup_elements.lookup_elements().combine(&[
            exp_tag,
            columns.trace_neg_product_col.data[vec_row],
            columns.trace_decay_col.data[vec_row],
        ]);
        let mult_packed = qwen35_beta_sigmoid_mult_packed(&columns.decay_multiplicities, vec_row);
        decay_col.write_frac(vec_row, q_table - mult_packed * q_trace, q_table * q_trace);
    }
    decay_col.finalize_col();

    logup_gen.finalize_last()
}

fn qwen35_decay_expected_preprocessed_root(
    statement: &Qwen35DeltaRecurrenceDecayStatement,
) -> <Blake2sHash as MerkleHasherLifted>::Hash {
    let softplus_table = qwen35_softplus_table(statement.table_log_size);
    let exp_table = qwen35_exp_table(statement.table_log_size);
    let decay_table = qwen35_decay_exp_table(statement.table_log_size);
    let (preprocessed, _, _, _, _, _, _) = qwen35_decay_preprocessed_columns(
        statement.table_log_size,
        statement.seq_len,
        statement.state_rows,
        &softplus_table,
        &exp_table,
        &decay_table,
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

pub fn prove_qwen35_delta_recurrence_decay_air(
    layer_idx: usize,
    a_gate: &M31Matrix,
    a_log_weight: &[M31],
    dt_bias: &[M31],
    decay: &M31Matrix,
    table_log_size: u32,
) -> Result<Qwen35DeltaRecurrenceDecayProof<Blake2sHash>, Qwen35DeltaRecurrenceDecayProofError> {
    let statement = qwen35_delta_recurrence_decay_statement(
        layer_idx,
        a_gate,
        a_log_weight,
        dt_bias,
        decay,
        table_log_size,
    )
    .map_err(Qwen35DeltaRecurrenceDecayProofError::Witness)?;
    let columns = qwen35_decay_columns(a_gate, a_log_weight, dt_bias, decay, table_log_size)
        .map_err(Qwen35DeltaRecurrenceDecayProofError::Witness)?;
    if columns.log_size != statement.table_log_size {
        return Err(Qwen35DeltaRecurrenceDecayProofError::Witness(format!(
            "DeltaRecurrence decay table_log_size {} is smaller than required log_size {}",
            statement.table_log_size, columns.log_size
        )));
    }
    if columns.trace_checksum != statement.trace_checksum {
        return Err(Qwen35DeltaRecurrenceDecayProofError::Witness(
            "DeltaRecurrence decay trace checksum mismatch".to_string(),
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

    let lookup_elements: ActivationRelation = ActivationRelation::draw(channel);
    let (interaction_trace, claimed_sum) = qwen35_decay_logup_trace(&columns, &lookup_elements);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, BaseField>(
        interaction_trace,
    ));
    tree_builder.commit(channel);

    let component = FrameworkComponent::new(
        &mut TraceLocationAllocator::default(),
        Qwen35DeltaRecurrenceDecayEval {
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
            .map_err(|err| Qwen35DeltaRecurrenceDecayProofError::Proving(format!("{err:?}")))?;

    Ok(Qwen35DeltaRecurrenceDecayProof {
        stark_proof,
        claimed_sum,
        log_size: columns.log_size,
        n_real_rows: a_gate.data.len(),
        statement,
    })
}

pub fn verify_qwen35_delta_recurrence_decay_air(
    proof: &Qwen35DeltaRecurrenceDecayProof<Blake2sHash>,
) -> Result<(), Qwen35DeltaRecurrenceDecayProofError> {
    verify_qwen35_delta_recurrence_decay_air_with_statement_hash(
        proof,
        proof.statement.statement_hash,
    )
}

pub fn verify_qwen35_delta_recurrence_decay_air_with_statement_hash(
    proof: &Qwen35DeltaRecurrenceDecayProof<Blake2sHash>,
    expected_statement_hash: FieldElement,
) -> Result<(), Qwen35DeltaRecurrenceDecayProofError> {
    if proof.statement.statement_hash != expected_statement_hash {
        return Err(Qwen35DeltaRecurrenceDecayProofError::Verification(
            "DeltaRecurrence decay statement hash mismatch".to_string(),
        ));
    }
    if proof.log_size != proof.statement.table_log_size {
        return Err(Qwen35DeltaRecurrenceDecayProofError::Verification(
            "DeltaRecurrence decay proof log_size must equal statement table_log_size".to_string(),
        ));
    }
    let (expected_softplus_commitment, expected_exp_commitment, expected_decay_commitment) =
        qwen35_delta_recurrence_decay_table_commitments(proof.statement.table_log_size);
    if proof.statement.softplus_table_commitment != expected_softplus_commitment
        || proof.statement.exp_table_commitment != expected_exp_commitment
        || proof.statement.decay_table_commitment != expected_decay_commitment
    {
        return Err(Qwen35DeltaRecurrenceDecayProofError::Verification(
            "DeltaRecurrence decay table commitment mismatch".to_string(),
        ));
    }
    let expected_hash = qwen35_delta_recurrence_decay_statement_hash(
        proof.statement.layer_idx,
        proof.statement.seq_len,
        proof.statement.state_rows,
        proof.statement.table_log_size,
        proof.statement.trace_checksum,
        proof.statement.softplus_table_commitment,
        proof.statement.exp_table_commitment,
        proof.statement.decay_table_commitment,
        proof.statement.a_gate_commitment,
        proof.statement.a_log_weight_commitment,
        proof.statement.dt_bias_commitment,
        proof.statement.decay_commitment,
    );
    if proof.statement.statement_hash != expected_hash {
        return Err(Qwen35DeltaRecurrenceDecayProofError::Verification(
            "DeltaRecurrence decay statement is malformed".to_string(),
        ));
    }
    if proof.stark_proof.commitments.len() < 3 {
        return Err(Qwen35DeltaRecurrenceDecayProofError::Verification(format!(
            "expected at least 3 commitment trees for decay LogUp, got {}",
            proof.stark_proof.commitments.len()
        )));
    }
    let expected_root = qwen35_decay_expected_preprocessed_root(&proof.statement);
    if proof.stark_proof.commitments[0] != expected_root {
        return Err(Qwen35DeltaRecurrenceDecayProofError::Verification(
            "DeltaRecurrence decay preprocessed root mismatch".to_string(),
        ));
    }

    let pcs_config = PcsConfig::default();
    let mut allocator = TraceLocationAllocator::default();
    let dummy_component = FrameworkComponent::new(
        &mut allocator,
        Qwen35DeltaRecurrenceDecayEval {
            log_n_rows: proof.log_size,
            instance_id: 0,
            lookup_elements: ActivationRelation::dummy(),
            claimed_sum: proof.claimed_sum,
            trace_checksum: proof.statement.trace_checksum,
        },
        proof.claimed_sum,
    );
    let bounds = Component::trace_log_degree_bounds(&dummy_component);
    if proof.stark_proof.commitments.len() < bounds.len() {
        return Err(Qwen35DeltaRecurrenceDecayProofError::Verification(format!(
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
    let lookup_elements: ActivationRelation = ActivationRelation::draw(channel);
    for idx in 2..bounds.len() {
        commitment_scheme.commit(proof.stark_proof.commitments[idx], &bounds[idx], channel);
    }

    let mut allocator = TraceLocationAllocator::default();
    let component = FrameworkComponent::new(
        &mut allocator,
        Qwen35DeltaRecurrenceDecayEval {
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
    .map_err(|err| Qwen35DeltaRecurrenceDecayProofError::Verification(format!("{err:?}")))
}

pub fn prove_qwen35_delta_recurrence_beta_sigmoid_air(
    layer_idx: usize,
    b_gate: &M31Matrix,
    beta: &M31Matrix,
    table_log_size: u32,
) -> Result<
    Qwen35DeltaRecurrenceBetaSigmoidProof<Blake2sHash>,
    Qwen35DeltaRecurrenceBetaSigmoidProofError,
> {
    let statement =
        qwen35_delta_recurrence_beta_sigmoid_statement(layer_idx, b_gate, beta, table_log_size)
            .map_err(Qwen35DeltaRecurrenceBetaSigmoidProofError::Witness)?;
    let columns = qwen35_beta_sigmoid_columns(b_gate, beta, table_log_size)
        .map_err(Qwen35DeltaRecurrenceBetaSigmoidProofError::Witness)?;
    if columns.trace_checksum != statement.trace_checksum {
        return Err(Qwen35DeltaRecurrenceBetaSigmoidProofError::Witness(
            "DeltaRecurrence beta sigmoid trace checksum mismatch".to_string(),
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

    let lookup_elements: ActivationRelation = ActivationRelation::draw(channel);
    let (interaction_trace, claimed_sum) =
        qwen35_beta_sigmoid_logup_trace(&columns, &lookup_elements);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, BaseField>(
        interaction_trace,
    ));
    tree_builder.commit(channel);

    let component = FrameworkComponent::new(
        &mut TraceLocationAllocator::default(),
        Qwen35DeltaRecurrenceBetaSigmoidEval {
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
            .map_err(|err| {
                Qwen35DeltaRecurrenceBetaSigmoidProofError::Proving(format!("{err:?}"))
            })?;

    Ok(Qwen35DeltaRecurrenceBetaSigmoidProof {
        stark_proof,
        claimed_sum,
        log_size: columns.log_size,
        n_real_rows: b_gate.data.len(),
        statement,
    })
}

pub fn verify_qwen35_delta_recurrence_beta_sigmoid_air(
    proof: &Qwen35DeltaRecurrenceBetaSigmoidProof<Blake2sHash>,
) -> Result<(), Qwen35DeltaRecurrenceBetaSigmoidProofError> {
    verify_qwen35_delta_recurrence_beta_sigmoid_air_with_statement_hash(
        proof,
        proof.statement.statement_hash,
    )
}

pub fn verify_qwen35_delta_recurrence_beta_sigmoid_air_with_statement_hash(
    proof: &Qwen35DeltaRecurrenceBetaSigmoidProof<Blake2sHash>,
    expected_statement_hash: FieldElement,
) -> Result<(), Qwen35DeltaRecurrenceBetaSigmoidProofError> {
    if proof.statement.statement_hash != expected_statement_hash {
        return Err(Qwen35DeltaRecurrenceBetaSigmoidProofError::Verification(
            "DeltaRecurrence beta sigmoid statement hash mismatch".to_string(),
        ));
    }
    let expected_table_commitment =
        qwen35_delta_recurrence_beta_sigmoid_table_commitment(proof.statement.table_log_size);
    if proof.statement.table_commitment != expected_table_commitment {
        return Err(Qwen35DeltaRecurrenceBetaSigmoidProofError::Verification(
            "DeltaRecurrence beta sigmoid table commitment mismatch".to_string(),
        ));
    }
    let expected_hash = qwen35_delta_recurrence_beta_sigmoid_statement_hash(
        proof.statement.layer_idx,
        proof.statement.seq_len,
        proof.statement.state_rows,
        proof.statement.table_log_size,
        proof.statement.trace_checksum,
        proof.statement.table_commitment,
        proof.statement.b_gate_commitment,
        proof.statement.beta_commitment,
    );
    if proof.statement.statement_hash != expected_hash {
        return Err(Qwen35DeltaRecurrenceBetaSigmoidProofError::Verification(
            "DeltaRecurrence beta sigmoid statement is malformed".to_string(),
        ));
    }
    if proof.stark_proof.commitments.len() < 3 {
        return Err(Qwen35DeltaRecurrenceBetaSigmoidProofError::Verification(
            format!(
                "expected at least 3 commitment trees for beta sigmoid LogUp, got {}",
                proof.stark_proof.commitments.len()
            ),
        ));
    }
    let expected_root = qwen35_beta_sigmoid_expected_preprocessed_root(
        proof.statement.table_log_size,
        proof.statement.seq_len,
        proof.statement.state_rows,
    );
    if proof.stark_proof.commitments[0] != expected_root {
        return Err(Qwen35DeltaRecurrenceBetaSigmoidProofError::Verification(
            "DeltaRecurrence beta sigmoid preprocessed table root mismatch".to_string(),
        ));
    }

    let pcs_config = PcsConfig::default();
    let mut allocator = TraceLocationAllocator::default();
    let dummy_component = FrameworkComponent::new(
        &mut allocator,
        Qwen35DeltaRecurrenceBetaSigmoidEval {
            log_n_rows: proof.log_size,
            instance_id: 0,
            lookup_elements: ActivationRelation::dummy(),
            claimed_sum: proof.claimed_sum,
            trace_checksum: proof.statement.trace_checksum,
        },
        proof.claimed_sum,
    );
    let bounds = Component::trace_log_degree_bounds(&dummy_component);
    if proof.stark_proof.commitments.len() < bounds.len() {
        return Err(Qwen35DeltaRecurrenceBetaSigmoidProofError::Verification(
            format!(
                "proof commitment count {} is smaller than trace bound count {}",
                proof.stark_proof.commitments.len(),
                bounds.len(),
            ),
        ));
    }

    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    mix_statement_hash(channel, expected_statement_hash);
    let mut commitment_scheme = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(pcs_config);
    commitment_scheme.commit(proof.stark_proof.commitments[0], &bounds[0], channel);
    commitment_scheme.commit(proof.stark_proof.commitments[1], &bounds[1], channel);
    let lookup_elements: ActivationRelation = ActivationRelation::draw(channel);
    for idx in 2..bounds.len() {
        commitment_scheme.commit(proof.stark_proof.commitments[idx], &bounds[idx], channel);
    }

    let mut allocator = TraceLocationAllocator::default();
    let component = FrameworkComponent::new(
        &mut allocator,
        Qwen35DeltaRecurrenceBetaSigmoidEval {
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
    .map_err(|err| Qwen35DeltaRecurrenceBetaSigmoidProofError::Verification(format!("{err:?}")))
}

pub fn qwen35_delta_recurrence_trace_binding_witness(
    inputs: &Qwen35DeltaRecurrenceInputs<'_>,
) -> Result<Qwen35DeltaRecurrenceTraceBindingWitness, String> {
    validate_shapes(inputs)?;
    let qk_head_dim = inputs.query.cols / inputs.state_rows;
    let row_count = inputs.query.rows * inputs.state_rows * qk_head_dim * inputs.value_head_dim;
    let mut rows = Vec::with_capacity(row_count);

    for token_idx in 0..inputs.query.rows {
        for state_row_idx in 0..inputs.state_rows {
            for qk_col_idx in 0..qk_head_dim {
                for value_col_idx in 0..inputs.value_head_dim {
                    let qk_abs_col = state_row_idx * qk_head_dim + qk_col_idx;
                    let value_abs_col = state_row_idx * inputs.value_head_dim + value_col_idx;
                    let state_abs_row = state_row_idx * qk_head_dim + qk_col_idx;

                    rows.push(Qwen35DeltaRecurrenceTraceBindingRow {
                        token_idx,
                        state_row_idx,
                        qk_col_idx,
                        value_col_idx,
                        query: inputs.query.get(token_idx, qk_abs_col),
                        key: inputs.key.get(token_idx, qk_abs_col),
                        projected_value: inputs.projected_value.get(token_idx, value_abs_col),
                        a_gate: inputs.a_gate.get(token_idx, state_row_idx),
                        b_gate: inputs.b_gate.get(token_idx, state_row_idx),
                        a_log_weight: inputs.a_log_weight[state_row_idx],
                        dt_bias: inputs.dt_bias[state_row_idx],
                        initial_recurrent_state: inputs
                            .initial_recurrent_state
                            .get(state_abs_row, value_col_idx),
                        final_recurrent_state: inputs
                            .final_recurrent_state
                            .get(state_abs_row, value_col_idx),
                        output: inputs.output.get(token_idx, value_abs_col),
                    });
                }
            }
        }
    }

    let witness_hash = qwen35_delta_recurrence_trace_binding_witness_hash(inputs, &rows)?;
    Ok(Qwen35DeltaRecurrenceTraceBindingWitness { rows, witness_hash })
}

pub fn verify_qwen35_delta_recurrence_trace_binding_witness(
    inputs: &Qwen35DeltaRecurrenceInputs<'_>,
    witness: &Qwen35DeltaRecurrenceTraceBindingWitness,
) -> Result<(), String> {
    validate_shapes(inputs)?;
    verify_qwen35_delta_recurrence_trace_binding_rows(inputs, &witness.rows)?;
    let expected_hash = qwen35_delta_recurrence_trace_binding_witness_hash(inputs, &witness.rows)?;
    if witness.witness_hash != expected_hash {
        return Err("DeltaRecurrence trace-binding witness hash mismatch".to_string());
    }
    Ok(())
}

pub fn verify_qwen35_delta_recurrence_trace_binding_rows(
    inputs: &Qwen35DeltaRecurrenceInputs<'_>,
    rows: &[Qwen35DeltaRecurrenceTraceBindingRow],
) -> Result<(), String> {
    validate_shapes(inputs)?;
    let qk_head_dim = inputs.query.cols / inputs.state_rows;
    let expected_rows = inputs.query.rows * inputs.state_rows * qk_head_dim * inputs.value_head_dim;
    if rows.len() != expected_rows {
        return Err(format!(
            "DeltaRecurrence trace-binding rows {} != expected {}",
            rows.len(),
            expected_rows
        ));
    }

    for (row_idx, row) in rows.iter().enumerate() {
        let value_col_idx = row_idx % inputs.value_head_dim;
        let qk_major = row_idx / inputs.value_head_dim;
        let qk_col_idx = qk_major % qk_head_dim;
        let state_major = qk_major / qk_head_dim;
        let state_row_idx = state_major % inputs.state_rows;
        let token_idx = state_major / inputs.state_rows;

        if row.token_idx != token_idx
            || row.state_row_idx != state_row_idx
            || row.qk_col_idx != qk_col_idx
            || row.value_col_idx != value_col_idx
        {
            return Err(format!(
                "DeltaRecurrence trace-binding row {row_idx} has token={}, state_row={}, qk_col={}, value_col={}, expected token={}, state_row={}, qk_col={}, value_col={}",
                row.token_idx,
                row.state_row_idx,
                row.qk_col_idx,
                row.value_col_idx,
                token_idx,
                state_row_idx,
                qk_col_idx,
                value_col_idx
            ));
        }

        let qk_abs_col = state_row_idx * qk_head_dim + qk_col_idx;
        let value_abs_col = state_row_idx * inputs.value_head_dim + value_col_idx;
        let state_abs_row = state_row_idx * qk_head_dim + qk_col_idx;
        let expected = Qwen35DeltaRecurrenceTraceBindingRow {
            token_idx,
            state_row_idx,
            qk_col_idx,
            value_col_idx,
            query: inputs.query.get(token_idx, qk_abs_col),
            key: inputs.key.get(token_idx, qk_abs_col),
            projected_value: inputs.projected_value.get(token_idx, value_abs_col),
            a_gate: inputs.a_gate.get(token_idx, state_row_idx),
            b_gate: inputs.b_gate.get(token_idx, state_row_idx),
            a_log_weight: inputs.a_log_weight[state_row_idx],
            dt_bias: inputs.dt_bias[state_row_idx],
            initial_recurrent_state: inputs
                .initial_recurrent_state
                .get(state_abs_row, value_col_idx),
            final_recurrent_state: inputs
                .final_recurrent_state
                .get(state_abs_row, value_col_idx),
            output: inputs.output.get(token_idx, value_abs_col),
        };
        if row != &expected {
            return Err(format!(
                "DeltaRecurrence trace-binding row {row_idx} does not match active tensors/state"
            ));
        }
    }

    Ok(())
}

pub fn qwen35_delta_recurrence_trace_binding_witness_hash(
    inputs: &Qwen35DeltaRecurrenceInputs<'_>,
    rows: &[Qwen35DeltaRecurrenceTraceBindingRow],
) -> Result<FieldElement, String> {
    validate_shapes(inputs)?;
    let mut felts = Vec::with_capacity(14 + rows.len() * 14);
    felts.extend([
        FieldElement::from(DOMAIN_QWEN35_DELTA_RECURRENCE_TRACE_BINDING),
        FieldElement::from(inputs.mode.as_u64()),
        FieldElement::from(inputs.query.rows as u64),
        FieldElement::from(inputs.query.cols as u64),
        FieldElement::from(inputs.key.cols as u64),
        FieldElement::from(inputs.projected_value.cols as u64),
        FieldElement::from(inputs.state_rows as u64),
        FieldElement::from(inputs.value_head_dim as u64),
        qwen35_delta_recurrence_query_commitment(inputs.query),
        qwen35_delta_recurrence_key_commitment(inputs.key),
        qwen35_delta_recurrence_projected_value_commitment(inputs.projected_value),
        qwen35_delta_recurrence_initial_state_commitment(inputs.initial_recurrent_state),
        qwen35_delta_recurrence_final_state_commitment(inputs.final_recurrent_state),
        qwen35_delta_recurrence_output_commitment(inputs.output),
    ]);
    for row in rows {
        felts.extend([
            FieldElement::from(row.token_idx as u64),
            FieldElement::from(row.state_row_idx as u64),
            FieldElement::from(row.qk_col_idx as u64),
            FieldElement::from(row.value_col_idx as u64),
            FieldElement::from(row.query.0 as u64),
            FieldElement::from(row.key.0 as u64),
            FieldElement::from(row.projected_value.0 as u64),
            FieldElement::from(row.a_gate.0 as u64),
            FieldElement::from(row.b_gate.0 as u64),
            FieldElement::from(row.a_log_weight.0 as u64),
            FieldElement::from(row.dt_bias.0 as u64),
            FieldElement::from(row.initial_recurrent_state.0 as u64),
            FieldElement::from(row.final_recurrent_state.0 as u64),
            FieldElement::from(row.output.0 as u64),
        ]);
    }
    Ok(starknet_crypto::poseidon_hash_many(&felts))
}

pub fn qwen35_delta_recurrence_arithmetic_witness(
    inputs: &Qwen35DeltaRecurrenceArithmeticInputs<'_>,
) -> Result<Qwen35DeltaRecurrenceArithmeticWitness, String> {
    validate_arithmetic_shapes(inputs)?;
    let qk_head_dim = inputs.scaled_query.cols / inputs.state_rows;
    let mut state = inputs.initial_recurrent_state.clone();
    let mut output = M31Matrix::new(inputs.scaled_query.rows, inputs.projected_value.cols);
    let row_count =
        inputs.scaled_query.rows * inputs.state_rows * qk_head_dim * inputs.value_head_dim;
    let mut rows = Vec::with_capacity(row_count);

    for token_idx in 0..inputs.scaled_query.rows {
        for state_row_idx in 0..inputs.state_rows {
            let decay = inputs.decay.get(token_idx, state_row_idx);
            let beta = inputs.beta.get(token_idx, state_row_idx);

            for value_col_idx in 0..inputs.value_head_dim {
                let value_abs_col = state_row_idx * inputs.value_head_dim + value_col_idx;
                let projected_value = inputs.projected_value.get(token_idx, value_abs_col);

                let mut kv_mem = zero();
                let mut kv_terms = Vec::with_capacity(qk_head_dim);
                let mut decayed_states = Vec::with_capacity(qk_head_dim);
                for qk_col_idx in 0..qk_head_dim {
                    let qk_abs_col = state_row_idx * qk_head_dim + qk_col_idx;
                    let state_abs_row = state_row_idx * qk_head_dim + qk_col_idx;
                    let decayed_state = state.get(state_abs_row, value_col_idx) * decay;
                    let kv_term = decayed_state * inputs.normalized_key.get(token_idx, qk_abs_col);
                    decayed_states.push(decayed_state);
                    kv_terms.push(kv_term);
                    kv_mem += kv_term;
                }

                let delta = (projected_value - kv_mem) * beta;
                let mut attended_value = zero();
                let mut kv_residual_before = kv_mem;
                let mut output_prefix_before = zero();
                let mut local_rows = Vec::with_capacity(qk_head_dim);
                for qk_col_idx in 0..qk_head_dim {
                    let qk_abs_col = state_row_idx * qk_head_dim + qk_col_idx;
                    let state_abs_row = state_row_idx * qk_head_dim + qk_col_idx;
                    let state_before = state.get(state_abs_row, value_col_idx);
                    let decayed_state = decayed_states[qk_col_idx];
                    let kv_term = kv_terms[qk_col_idx];
                    let kv_residual_after = kv_residual_before - kv_term;
                    let key = inputs.normalized_key.get(token_idx, qk_abs_col);
                    let state_after = decayed_state + key * delta;
                    let output_term = state_after * inputs.scaled_query.get(token_idx, qk_abs_col);
                    let output_prefix_after = output_prefix_before + output_term;
                    state.set(state_abs_row, value_col_idx, state_after);
                    attended_value = output_prefix_after;

                    local_rows.push(Qwen35DeltaRecurrenceArithmeticRow {
                        token_idx,
                        state_row_idx,
                        qk_col_idx,
                        value_col_idx,
                        scaled_query: inputs.scaled_query.get(token_idx, qk_abs_col),
                        normalized_key: key,
                        projected_value,
                        decay,
                        beta,
                        state_before,
                        decayed_state,
                        kv_term,
                        kv_residual_before,
                        kv_residual_after,
                        kv_mem,
                        delta,
                        state_after,
                        output_term,
                        output_prefix_before,
                        output_prefix_after,
                        output: zero(),
                    });
                    kv_residual_before = kv_residual_after;
                    output_prefix_before = output_prefix_after;
                }

                output.set(token_idx, value_abs_col, attended_value);
                for row in &mut local_rows {
                    row.output = attended_value;
                }
                rows.extend(local_rows);
            }
        }
    }

    let witness_hash =
        qwen35_delta_recurrence_arithmetic_witness_hash(inputs, &rows, &state, &output)?;
    Ok(Qwen35DeltaRecurrenceArithmeticWitness {
        rows,
        final_recurrent_state: state,
        output,
        witness_hash,
    })
}

pub fn verify_qwen35_delta_recurrence_arithmetic_witness(
    inputs: &Qwen35DeltaRecurrenceArithmeticInputs<'_>,
    witness: &Qwen35DeltaRecurrenceArithmeticWitness,
) -> Result<(), String> {
    validate_arithmetic_shapes(inputs)?;
    let expected = qwen35_delta_recurrence_arithmetic_witness(inputs)?;
    if witness.rows != expected.rows {
        return Err("DeltaRecurrence arithmetic witness rows do not match recurrence".to_string());
    }
    if witness.final_recurrent_state.rows != expected.final_recurrent_state.rows
        || witness.final_recurrent_state.cols != expected.final_recurrent_state.cols
        || witness.final_recurrent_state.data != expected.final_recurrent_state.data
    {
        return Err("DeltaRecurrence arithmetic final recurrent state mismatch".to_string());
    }
    if witness.output.rows != expected.output.rows
        || witness.output.cols != expected.output.cols
        || witness.output.data != expected.output.data
    {
        return Err("DeltaRecurrence arithmetic output mismatch".to_string());
    }
    if witness.final_recurrent_state.rows != inputs.final_recurrent_state.rows
        || witness.final_recurrent_state.cols != inputs.final_recurrent_state.cols
        || witness.final_recurrent_state.data != inputs.final_recurrent_state.data
    {
        return Err(
            "DeltaRecurrence arithmetic witness does not match claimed final state".to_string(),
        );
    }
    if witness.output.rows != inputs.output.rows
        || witness.output.cols != inputs.output.cols
        || witness.output.data != inputs.output.data
    {
        return Err("DeltaRecurrence arithmetic witness does not match claimed output".to_string());
    }
    if witness.witness_hash != expected.witness_hash {
        return Err("DeltaRecurrence arithmetic witness hash mismatch".to_string());
    }
    Ok(())
}

pub fn qwen35_delta_recurrence_arithmetic_witness_hash(
    inputs: &Qwen35DeltaRecurrenceArithmeticInputs<'_>,
    rows: &[Qwen35DeltaRecurrenceArithmeticRow],
    final_recurrent_state: &M31Matrix,
    output: &M31Matrix,
) -> Result<FieldElement, String> {
    validate_arithmetic_shapes(inputs)?;
    let mut felts = Vec::with_capacity(18 + rows.len() * 21);
    felts.extend([
        FieldElement::from(DOMAIN_QWEN35_DELTA_RECURRENCE_TRACE_BINDING + 1),
        FieldElement::from(inputs.scaled_query.rows as u64),
        FieldElement::from(inputs.scaled_query.cols as u64),
        FieldElement::from(inputs.normalized_key.cols as u64),
        FieldElement::from(inputs.projected_value.cols as u64),
        FieldElement::from(inputs.state_rows as u64),
        FieldElement::from(inputs.value_head_dim as u64),
        qwen35_delta_recurrence_scaled_query_commitment(inputs.scaled_query),
        qwen35_delta_recurrence_normalized_key_commitment(inputs.normalized_key),
        qwen35_delta_recurrence_projected_value_commitment(inputs.projected_value),
        qwen35_delta_recurrence_decay_commitment(inputs.decay),
        qwen35_delta_recurrence_beta_commitment(inputs.beta),
        qwen35_delta_recurrence_initial_state_commitment(inputs.initial_recurrent_state),
        qwen35_delta_recurrence_final_state_commitment(inputs.final_recurrent_state),
        qwen35_delta_recurrence_output_commitment(inputs.output),
        qwen35_delta_recurrence_final_state_commitment(final_recurrent_state),
        qwen35_delta_recurrence_output_commitment(output),
    ]);
    for row in rows {
        felts.extend([
            FieldElement::from(row.token_idx as u64),
            FieldElement::from(row.state_row_idx as u64),
            FieldElement::from(row.qk_col_idx as u64),
            FieldElement::from(row.value_col_idx as u64),
            FieldElement::from(row.scaled_query.0 as u64),
            FieldElement::from(row.normalized_key.0 as u64),
            FieldElement::from(row.projected_value.0 as u64),
            FieldElement::from(row.decay.0 as u64),
            FieldElement::from(row.beta.0 as u64),
            FieldElement::from(row.state_before.0 as u64),
            FieldElement::from(row.decayed_state.0 as u64),
            FieldElement::from(row.kv_term.0 as u64),
            FieldElement::from(row.kv_residual_before.0 as u64),
            FieldElement::from(row.kv_residual_after.0 as u64),
            FieldElement::from(row.kv_mem.0 as u64),
            FieldElement::from(row.delta.0 as u64),
            FieldElement::from(row.state_after.0 as u64),
            FieldElement::from(row.output_term.0 as u64),
            FieldElement::from(row.output_prefix_before.0 as u64),
            FieldElement::from(row.output_prefix_after.0 as u64),
            FieldElement::from(row.output.0 as u64),
        ]);
    }
    Ok(starknet_crypto::poseidon_hash_many(&felts))
}

pub fn qwen35_delta_recurrence_arithmetic_trace(
    inputs: &Qwen35DeltaRecurrenceArithmeticInputs<'_>,
) -> Result<Qwen35DeltaRecurrenceArithmeticTrace, String> {
    let witness = qwen35_delta_recurrence_arithmetic_witness(inputs)?;
    qwen35_delta_recurrence_arithmetic_trace_from_rows(
        inputs.scaled_query.rows,
        inputs.state_rows,
        inputs.scaled_query.cols / inputs.state_rows,
        inputs.value_head_dim,
        &witness.rows,
    )
}

pub fn qwen35_delta_recurrence_arithmetic_trace_from_rows(
    seq_len: usize,
    state_rows: usize,
    qk_head_dim: usize,
    value_head_dim: usize,
    rows: &[Qwen35DeltaRecurrenceArithmeticRow],
) -> Result<Qwen35DeltaRecurrenceArithmeticTrace, String> {
    if seq_len == 0 || state_rows == 0 || qk_head_dim == 0 || value_head_dim == 0 {
        return Err("DeltaRecurrence arithmetic trace dimensions must be non-zero".to_string());
    }
    let n_real_rows = seq_len * state_rows * value_head_dim * qk_head_dim;
    if rows.len() != n_real_rows {
        return Err(format!(
            "DeltaRecurrence arithmetic rows {} != expected {}",
            rows.len(),
            n_real_rows
        ));
    }

    let log_size = (n_real_rows.next_power_of_two().trailing_zeros())
        .max(QWEN35_DELTA_RECURRENCE_ARITHMETIC_MIN_LOG_SIZE);
    let size = 1usize << log_size;

    let token_stride = state_rows * value_head_dim * qk_head_dim;
    let mut preprocessed = vec![vec![zero(); size]; 8];
    let mut execution = vec![vec![zero(); size]; 24];

    for (row_idx, row) in rows.iter().enumerate() {
        let is_first_qk = row.qk_col_idx == 0;
        let is_last_qk = row.qk_col_idx + 1 == qk_head_dim;
        let is_qk_chain = !is_last_qk;
        let has_next_token = row.token_idx + 1 < seq_len;

        preprocessed[0][row_idx] = M31::from(row.token_idx as u32);
        preprocessed[1][row_idx] = M31::from(row.state_row_idx as u32);
        preprocessed[2][row_idx] = M31::from(row.qk_col_idx as u32);
        preprocessed[3][row_idx] = M31::from(row.value_col_idx as u32);
        preprocessed[4][row_idx] = M31::from(is_first_qk as u32);
        preprocessed[5][row_idx] = M31::from(is_last_qk as u32);
        preprocessed[6][row_idx] = M31::from(is_qk_chain as u32);
        preprocessed[7][row_idx] = M31::from(has_next_token as u32);

        execution[0][row_idx] = M31::from(row.token_idx as u32);
        execution[1][row_idx] = M31::from(row.state_row_idx as u32);
        execution[2][row_idx] = M31::from(row.qk_col_idx as u32);
        execution[3][row_idx] = M31::from(row.value_col_idx as u32);
        execution[4][row_idx] = row.scaled_query;
        execution[5][row_idx] = row.normalized_key;
        execution[6][row_idx] = row.projected_value;
        execution[7][row_idx] = row.decay;
        execution[8][row_idx] = row.beta;
        execution[9][row_idx] = row.state_before;
        execution[10][row_idx] = row.decayed_state;
        execution[11][row_idx] = row.kv_term;
        execution[12][row_idx] = row.kv_residual_before;
        execution[13][row_idx] = row.kv_residual_after;
        execution[14][row_idx] = if is_qk_chain {
            rows[row_idx + 1].kv_residual_before
        } else {
            zero()
        };
        execution[15][row_idx] = row.kv_mem;
        execution[16][row_idx] = row.delta;
        execution[17][row_idx] = row.state_after;
        execution[18][row_idx] = row.output_term;
        execution[19][row_idx] = row.output_prefix_before;
        execution[20][row_idx] = row.output_prefix_after;
        execution[21][row_idx] = if is_qk_chain {
            rows[row_idx + 1].output_prefix_before
        } else {
            zero()
        };
        execution[22][row_idx] = row.output;
        execution[23][row_idx] = if has_next_token {
            rows[row_idx + token_stride].state_before
        } else {
            zero()
        };
    }

    Ok(Qwen35DeltaRecurrenceArithmeticTrace {
        log_size,
        n_real_rows,
        preprocessed,
        execution,
    })
}

pub fn qwen35_delta_recurrence_trace_binding_trace(
    inputs: &Qwen35DeltaRecurrenceInputs<'_>,
) -> Result<Qwen35DeltaRecurrenceTraceBindingTrace, String> {
    let witness = qwen35_delta_recurrence_trace_binding_witness(inputs)?;
    qwen35_delta_recurrence_trace_binding_trace_from_rows(
        inputs.query.rows,
        inputs.state_rows,
        inputs.query.cols / inputs.state_rows,
        inputs.value_head_dim,
        &witness.rows,
    )
}

pub fn qwen35_delta_recurrence_trace_binding_trace_from_rows(
    seq_len: usize,
    state_rows: usize,
    qk_head_dim: usize,
    value_head_dim: usize,
    rows: &[Qwen35DeltaRecurrenceTraceBindingRow],
) -> Result<Qwen35DeltaRecurrenceTraceBindingTrace, String> {
    if seq_len == 0 || state_rows == 0 || qk_head_dim == 0 || value_head_dim == 0 {
        return Err("DeltaRecurrence trace-binding dimensions must be non-zero".to_string());
    }
    let n_real_rows = seq_len * state_rows * qk_head_dim * value_head_dim;
    if rows.len() != n_real_rows {
        return Err(format!(
            "DeltaRecurrence trace-binding rows {} != expected {}",
            rows.len(),
            n_real_rows
        ));
    }

    let log_size = (n_real_rows.next_power_of_two().trailing_zeros())
        .max(QWEN35_DELTA_RECURRENCE_TRACE_BINDING_MIN_LOG_SIZE);
    let size = 1usize << log_size;

    let mut preprocessed = vec![vec![zero(); size]; 4];
    let mut execution = vec![vec![zero(); size]; 14];

    for (row_idx, row) in rows.iter().enumerate() {
        preprocessed[0][row_idx] = M31::from(row.token_idx as u32);
        preprocessed[1][row_idx] = M31::from(row.state_row_idx as u32);
        preprocessed[2][row_idx] = M31::from(row.qk_col_idx as u32);
        preprocessed[3][row_idx] = M31::from(row.value_col_idx as u32);

        execution[0][row_idx] = M31::from(row.token_idx as u32);
        execution[1][row_idx] = M31::from(row.state_row_idx as u32);
        execution[2][row_idx] = M31::from(row.qk_col_idx as u32);
        execution[3][row_idx] = M31::from(row.value_col_idx as u32);
        execution[4][row_idx] = row.query;
        execution[5][row_idx] = row.key;
        execution[6][row_idx] = row.projected_value;
        execution[7][row_idx] = row.a_gate;
        execution[8][row_idx] = row.b_gate;
        execution[9][row_idx] = row.a_log_weight;
        execution[10][row_idx] = row.dt_bias;
        execution[11][row_idx] = row.initial_recurrent_state;
        execution[12][row_idx] = row.final_recurrent_state;
        execution[13][row_idx] = row.output;
    }

    Ok(Qwen35DeltaRecurrenceTraceBindingTrace {
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

fn mix_statement_hash<C: Channel>(channel: &mut C, statement_hash: FieldElement) {
    channel.mix_u64(DOMAIN_QWEN35_DELTA_RECURRENCE_CHANNEL);
    let bytes = statement_hash.to_bytes_be();
    channel.mix_u64(u64::from_be_bytes(bytes[0..8].try_into().unwrap()));
    channel.mix_u64(u64::from_be_bytes(bytes[8..16].try_into().unwrap()));
    channel.mix_u64(u64::from_be_bytes(bytes[16..24].try_into().unwrap()));
    channel.mix_u64(u64::from_be_bytes(bytes[24..32].try_into().unwrap()));
}

pub fn prove_qwen35_delta_recurrence_trace_binding_air(
    layer_idx: usize,
    inputs: &Qwen35DeltaRecurrenceInputs<'_>,
) -> Result<
    Qwen35DeltaRecurrenceTraceBindingProof<Blake2sHash>,
    Qwen35DeltaRecurrenceTraceBindingProofError,
> {
    let statement = qwen35_delta_recurrence_statement(layer_idx, inputs)
        .map_err(Qwen35DeltaRecurrenceTraceBindingProofError::Witness)?;
    let trace = qwen35_delta_recurrence_trace_binding_trace(inputs)
        .map_err(Qwen35DeltaRecurrenceTraceBindingProofError::Witness)?;
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
        Qwen35DeltaRecurrenceTraceBindingEval {
            log_n_rows: trace.log_size,
            instance_id: 0,
        },
        SecureField::from(M31::from(0u32)),
    );
    let stark_proof =
        prove::<SimdBackend, Blake2sMerkleChannel>(&[&component], channel, commitment_scheme)
            .map_err(|e| Qwen35DeltaRecurrenceTraceBindingProofError::Proving(format!("{e:?}")))?;

    Ok(Qwen35DeltaRecurrenceTraceBindingProof {
        stark_proof,
        log_size: trace.log_size,
        n_real_rows: trace.n_real_rows,
        statement,
    })
}

pub fn verify_qwen35_delta_recurrence_trace_binding_air(
    proof: &Qwen35DeltaRecurrenceTraceBindingProof<Blake2sHash>,
) -> Result<(), Qwen35DeltaRecurrenceTraceBindingProofError> {
    verify_qwen35_delta_recurrence_trace_binding_air_with_statement_hash(
        proof,
        proof.statement.statement_hash,
    )
}

pub fn verify_qwen35_delta_recurrence_trace_binding_air_with_statement_hash(
    proof: &Qwen35DeltaRecurrenceTraceBindingProof<Blake2sHash>,
    expected_statement_hash: FieldElement,
) -> Result<(), Qwen35DeltaRecurrenceTraceBindingProofError> {
    if proof.statement.statement_hash != expected_statement_hash {
        return Err(Qwen35DeltaRecurrenceTraceBindingProofError::Verification(
            "DeltaRecurrence trace-binding statement hash mismatch".to_string(),
        ));
    }
    let pcs_config = PcsConfig::default();
    let mut allocator = TraceLocationAllocator::default();
    let component = FrameworkComponent::new(
        &mut allocator,
        Qwen35DeltaRecurrenceTraceBindingEval {
            log_n_rows: proof.log_size,
            instance_id: 0,
        },
        SecureField::from(M31::from(0u32)),
    );
    let bounds = Component::trace_log_degree_bounds(&component);
    if bounds.len() != 2 {
        return Err(Qwen35DeltaRecurrenceTraceBindingProofError::Verification(
            format!("expected 2 commitment trees, got {}", bounds.len()),
        ));
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
    .map_err(|e| Qwen35DeltaRecurrenceTraceBindingProofError::Verification(format!("{e:?}")))
}

pub fn prove_qwen35_delta_recurrence_transform_binding_air(
    layer_idx: usize,
    inputs: &Qwen35DeltaRecurrenceTransformInputs<'_>,
) -> Result<
    Qwen35DeltaRecurrenceTransformBindingProof<Blake2sHash>,
    Qwen35DeltaRecurrenceTransformBindingProofError,
> {
    let statement = qwen35_delta_recurrence_transform_statement(layer_idx, inputs)
        .map_err(Qwen35DeltaRecurrenceTransformBindingProofError::Witness)?;
    let trace = qwen35_delta_recurrence_transform_binding_trace(inputs)
        .map_err(Qwen35DeltaRecurrenceTransformBindingProofError::Witness)?;
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
        Qwen35DeltaRecurrenceTransformBindingEval {
            log_n_rows: trace.log_size,
            instance_id: 0,
        },
        SecureField::from(M31::from(0u32)),
    );
    let stark_proof =
        prove::<SimdBackend, Blake2sMerkleChannel>(&[&component], channel, commitment_scheme)
            .map_err(|e| {
                Qwen35DeltaRecurrenceTransformBindingProofError::Proving(format!("{e:?}"))
            })?;

    Ok(Qwen35DeltaRecurrenceTransformBindingProof {
        stark_proof,
        log_size: trace.log_size,
        n_real_rows: trace.n_real_rows,
        statement,
    })
}

pub fn verify_qwen35_delta_recurrence_transform_binding_air(
    proof: &Qwen35DeltaRecurrenceTransformBindingProof<Blake2sHash>,
) -> Result<(), Qwen35DeltaRecurrenceTransformBindingProofError> {
    verify_qwen35_delta_recurrence_transform_binding_air_with_statement_hash(
        proof,
        proof.statement.statement_hash,
    )
}

pub fn verify_qwen35_delta_recurrence_transform_binding_air_with_statement_hash(
    proof: &Qwen35DeltaRecurrenceTransformBindingProof<Blake2sHash>,
    expected_statement_hash: FieldElement,
) -> Result<(), Qwen35DeltaRecurrenceTransformBindingProofError> {
    if proof.statement.statement_hash != expected_statement_hash {
        return Err(
            Qwen35DeltaRecurrenceTransformBindingProofError::Verification(
                "DeltaRecurrence transform-binding statement hash mismatch".to_string(),
            ),
        );
    }
    let pcs_config = PcsConfig::default();
    let mut allocator = TraceLocationAllocator::default();
    let component = FrameworkComponent::new(
        &mut allocator,
        Qwen35DeltaRecurrenceTransformBindingEval {
            log_n_rows: proof.log_size,
            instance_id: 0,
        },
        SecureField::from(M31::from(0u32)),
    );
    let bounds = Component::trace_log_degree_bounds(&component);
    if bounds.len() != 2 {
        return Err(
            Qwen35DeltaRecurrenceTransformBindingProofError::Verification(format!(
                "expected 2 commitment trees, got {}",
                bounds.len()
            )),
        );
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
    .map_err(|e| Qwen35DeltaRecurrenceTransformBindingProofError::Verification(format!("{e:?}")))
}

pub fn prove_qwen35_delta_recurrence_arithmetic_air(
    layer_idx: usize,
    mode: Qwen35DeltaRecurrenceMode,
    inputs: &Qwen35DeltaRecurrenceArithmeticInputs<'_>,
) -> Result<
    Qwen35DeltaRecurrenceArithmeticProof<Blake2sHash>,
    Qwen35DeltaRecurrenceArithmeticProofError,
> {
    let statement = qwen35_delta_recurrence_arithmetic_statement(layer_idx, mode, inputs)
        .map_err(Qwen35DeltaRecurrenceArithmeticProofError::Witness)?;
    let witness = qwen35_delta_recurrence_arithmetic_witness(inputs)
        .map_err(Qwen35DeltaRecurrenceArithmeticProofError::Witness)?;
    verify_qwen35_delta_recurrence_arithmetic_witness(inputs, &witness)
        .map_err(Qwen35DeltaRecurrenceArithmeticProofError::Witness)?;
    let trace = qwen35_delta_recurrence_arithmetic_trace_from_rows(
        inputs.scaled_query.rows,
        inputs.state_rows,
        inputs.scaled_query.cols / inputs.state_rows,
        inputs.value_head_dim,
        &witness.rows,
    )
    .map_err(Qwen35DeltaRecurrenceArithmeticProofError::Witness)?;
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
        Qwen35DeltaRecurrenceArithmeticEval {
            log_n_rows: trace.log_size,
            instance_id: 0,
        },
        SecureField::from(M31::from(0u32)),
    );
    let stark_proof =
        prove::<SimdBackend, Blake2sMerkleChannel>(&[&component], channel, commitment_scheme)
            .map_err(|e| Qwen35DeltaRecurrenceArithmeticProofError::Proving(format!("{e:?}")))?;

    Ok(Qwen35DeltaRecurrenceArithmeticProof {
        stark_proof,
        log_size: trace.log_size,
        n_real_rows: trace.n_real_rows,
        statement,
    })
}

pub fn verify_qwen35_delta_recurrence_arithmetic_air(
    proof: &Qwen35DeltaRecurrenceArithmeticProof<Blake2sHash>,
) -> Result<(), Qwen35DeltaRecurrenceArithmeticProofError> {
    verify_qwen35_delta_recurrence_arithmetic_air_with_statement_hash(
        proof,
        proof.statement.statement_hash,
    )
}

pub fn verify_qwen35_delta_recurrence_arithmetic_air_with_statement_hash(
    proof: &Qwen35DeltaRecurrenceArithmeticProof<Blake2sHash>,
    expected_statement_hash: FieldElement,
) -> Result<(), Qwen35DeltaRecurrenceArithmeticProofError> {
    if proof.statement.statement_hash != expected_statement_hash {
        return Err(Qwen35DeltaRecurrenceArithmeticProofError::Verification(
            "DeltaRecurrence arithmetic statement hash mismatch".to_string(),
        ));
    }
    let pcs_config = PcsConfig::default();
    let mut allocator = TraceLocationAllocator::default();
    let component = FrameworkComponent::new(
        &mut allocator,
        Qwen35DeltaRecurrenceArithmeticEval {
            log_n_rows: proof.log_size,
            instance_id: 0,
        },
        SecureField::from(M31::from(0u32)),
    );
    let bounds = Component::trace_log_degree_bounds(&component);
    if bounds.len() != 2 {
        return Err(Qwen35DeltaRecurrenceArithmeticProofError::Verification(
            format!("expected 2 commitment trees, got {}", bounds.len()),
        ));
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
    .map_err(|e| Qwen35DeltaRecurrenceArithmeticProofError::Verification(format!("{e:?}")))
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_delta_recurrence_statement_hash(
    layer_idx: usize,
    seq_len: usize,
    query_width: usize,
    key_width: usize,
    value_width: usize,
    state_rows: usize,
    value_head_dim: usize,
    mode_tag: u64,
    air_spec_hash: FieldElement,
    query_commitment: FieldElement,
    key_commitment: FieldElement,
    projected_value_commitment: FieldElement,
    a_gate_commitment: FieldElement,
    b_gate_commitment: FieldElement,
    a_log_weight_commitment: FieldElement,
    dt_bias_commitment: FieldElement,
    initial_recurrent_state_commitment: FieldElement,
    final_recurrent_state_commitment: FieldElement,
    output_commitment: FieldElement,
) -> FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(DOMAIN_QWEN35_DELTA_RECURRENCE_STATEMENT),
        FieldElement::from(layer_idx as u64),
        FieldElement::from(seq_len as u64),
        FieldElement::from(query_width as u64),
        FieldElement::from(key_width as u64),
        FieldElement::from(value_width as u64),
        FieldElement::from(state_rows as u64),
        FieldElement::from(value_head_dim as u64),
        FieldElement::from(mode_tag),
        air_spec_hash,
        query_commitment,
        key_commitment,
        projected_value_commitment,
        a_gate_commitment,
        b_gate_commitment,
        a_log_weight_commitment,
        dt_bias_commitment,
        initial_recurrent_state_commitment,
        final_recurrent_state_commitment,
        output_commitment,
    ])
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_delta_recurrence_arithmetic_statement_hash(
    layer_idx: usize,
    seq_len: usize,
    query_width: usize,
    key_width: usize,
    value_width: usize,
    state_rows: usize,
    value_head_dim: usize,
    mode_tag: u64,
    air_spec_hash: FieldElement,
    scaled_query_commitment: FieldElement,
    normalized_key_commitment: FieldElement,
    projected_value_commitment: FieldElement,
    decay_commitment: FieldElement,
    beta_commitment: FieldElement,
    initial_recurrent_state_commitment: FieldElement,
    final_recurrent_state_commitment: FieldElement,
    output_commitment: FieldElement,
) -> FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(DOMAIN_QWEN35_DELTA_RECURRENCE_ARITHMETIC_STATEMENT),
        FieldElement::from(layer_idx as u64),
        FieldElement::from(seq_len as u64),
        FieldElement::from(query_width as u64),
        FieldElement::from(key_width as u64),
        FieldElement::from(value_width as u64),
        FieldElement::from(state_rows as u64),
        FieldElement::from(value_head_dim as u64),
        FieldElement::from(mode_tag),
        air_spec_hash,
        scaled_query_commitment,
        normalized_key_commitment,
        projected_value_commitment,
        decay_commitment,
        beta_commitment,
        initial_recurrent_state_commitment,
        final_recurrent_state_commitment,
        output_commitment,
    ])
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_delta_recurrence_transform_statement_hash(
    layer_idx: usize,
    seq_len: usize,
    query_width: usize,
    key_width: usize,
    state_rows: usize,
    value_head_dim: usize,
    mode_tag: u64,
    air_spec_hash: FieldElement,
    query_commitment: FieldElement,
    key_commitment: FieldElement,
    a_gate_commitment: FieldElement,
    b_gate_commitment: FieldElement,
    a_log_weight_commitment: FieldElement,
    dt_bias_commitment: FieldElement,
    scaled_query_commitment: FieldElement,
    normalized_key_commitment: FieldElement,
    decay_commitment: FieldElement,
    beta_commitment: FieldElement,
) -> FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(DOMAIN_QWEN35_DELTA_RECURRENCE_TRANSFORM_STATEMENT),
        FieldElement::from(layer_idx as u64),
        FieldElement::from(seq_len as u64),
        FieldElement::from(query_width as u64),
        FieldElement::from(key_width as u64),
        FieldElement::from(state_rows as u64),
        FieldElement::from(value_head_dim as u64),
        FieldElement::from(mode_tag),
        air_spec_hash,
        query_commitment,
        key_commitment,
        a_gate_commitment,
        b_gate_commitment,
        a_log_weight_commitment,
        dt_bias_commitment,
        scaled_query_commitment,
        normalized_key_commitment,
        decay_commitment,
        beta_commitment,
    ])
}

pub fn qwen35_delta_recurrence_transform_statement(
    layer_idx: usize,
    inputs: &Qwen35DeltaRecurrenceTransformInputs<'_>,
) -> Result<Qwen35DeltaRecurrenceTransformStatement, String> {
    validate_transform_shapes(inputs)?;

    let query_commitment = qwen35_delta_recurrence_query_commitment(inputs.query);
    let key_commitment = qwen35_delta_recurrence_key_commitment(inputs.key);
    let a_gate_commitment = qwen35_delta_recurrence_a_gate_commitment(inputs.a_gate);
    let b_gate_commitment = qwen35_delta_recurrence_b_gate_commitment(inputs.b_gate);
    let a_log_weight_commitment =
        qwen35_delta_recurrence_a_log_weight_commitment(inputs.a_log_weight);
    let dt_bias_commitment = qwen35_delta_recurrence_dt_bias_commitment(inputs.dt_bias);
    let scaled_query_commitment =
        qwen35_delta_recurrence_scaled_query_commitment(inputs.scaled_query);
    let normalized_key_commitment =
        qwen35_delta_recurrence_normalized_key_commitment(inputs.normalized_key);
    let decay_commitment = qwen35_delta_recurrence_decay_commitment(inputs.decay);
    let beta_commitment = qwen35_delta_recurrence_beta_commitment(inputs.beta);
    let value_width = inputs.state_rows * inputs.value_head_dim;
    let air_spec = qwen35_delta_recurrence_air_spec(
        inputs.query.rows,
        inputs.query.cols,
        inputs.key.cols,
        value_width,
        inputs.state_rows,
        inputs.value_head_dim,
        inputs.mode,
        true,
        true,
    )?;
    let statement_hash = qwen35_delta_recurrence_transform_statement_hash(
        layer_idx,
        inputs.query.rows,
        inputs.query.cols,
        inputs.key.cols,
        inputs.state_rows,
        inputs.value_head_dim,
        inputs.mode.as_u64(),
        air_spec.spec_hash,
        query_commitment,
        key_commitment,
        a_gate_commitment,
        b_gate_commitment,
        a_log_weight_commitment,
        dt_bias_commitment,
        scaled_query_commitment,
        normalized_key_commitment,
        decay_commitment,
        beta_commitment,
    );

    Ok(Qwen35DeltaRecurrenceTransformStatement {
        layer_idx,
        seq_len: inputs.query.rows,
        query_width: inputs.query.cols,
        key_width: inputs.key.cols,
        state_rows: inputs.state_rows,
        value_head_dim: inputs.value_head_dim,
        mode: inputs.mode,
        air_spec_hash: air_spec.spec_hash,
        query_commitment,
        key_commitment,
        a_gate_commitment,
        b_gate_commitment,
        a_log_weight_commitment,
        dt_bias_commitment,
        scaled_query_commitment,
        normalized_key_commitment,
        decay_commitment,
        beta_commitment,
        statement_hash,
    })
}

pub fn qwen35_delta_recurrence_arithmetic_statement(
    layer_idx: usize,
    mode: Qwen35DeltaRecurrenceMode,
    inputs: &Qwen35DeltaRecurrenceArithmeticInputs<'_>,
) -> Result<Qwen35DeltaRecurrenceArithmeticStatement, String> {
    validate_arithmetic_shapes(inputs)?;

    let scaled_query_commitment =
        qwen35_delta_recurrence_scaled_query_commitment(inputs.scaled_query);
    let normalized_key_commitment =
        qwen35_delta_recurrence_normalized_key_commitment(inputs.normalized_key);
    let projected_value_commitment =
        qwen35_delta_recurrence_projected_value_commitment(inputs.projected_value);
    let decay_commitment = qwen35_delta_recurrence_decay_commitment(inputs.decay);
    let beta_commitment = qwen35_delta_recurrence_beta_commitment(inputs.beta);
    let initial_recurrent_state_commitment =
        qwen35_delta_recurrence_initial_state_commitment(inputs.initial_recurrent_state);
    let final_recurrent_state_commitment =
        qwen35_delta_recurrence_final_state_commitment(inputs.final_recurrent_state);
    let output_commitment = qwen35_delta_recurrence_output_commitment(inputs.output);
    let air_spec = qwen35_delta_recurrence_air_spec(
        inputs.scaled_query.rows,
        inputs.scaled_query.cols,
        inputs.normalized_key.cols,
        inputs.projected_value.cols,
        inputs.state_rows,
        inputs.value_head_dim,
        mode,
        true,
        true,
    )?;
    let statement_hash = qwen35_delta_recurrence_arithmetic_statement_hash(
        layer_idx,
        inputs.scaled_query.rows,
        inputs.scaled_query.cols,
        inputs.normalized_key.cols,
        inputs.projected_value.cols,
        inputs.state_rows,
        inputs.value_head_dim,
        mode.as_u64(),
        air_spec.spec_hash,
        scaled_query_commitment,
        normalized_key_commitment,
        projected_value_commitment,
        decay_commitment,
        beta_commitment,
        initial_recurrent_state_commitment,
        final_recurrent_state_commitment,
        output_commitment,
    );

    Ok(Qwen35DeltaRecurrenceArithmeticStatement {
        layer_idx,
        seq_len: inputs.scaled_query.rows,
        query_width: inputs.scaled_query.cols,
        key_width: inputs.normalized_key.cols,
        value_width: inputs.projected_value.cols,
        state_rows: inputs.state_rows,
        value_head_dim: inputs.value_head_dim,
        mode,
        air_spec_hash: air_spec.spec_hash,
        scaled_query_commitment,
        normalized_key_commitment,
        projected_value_commitment,
        decay_commitment,
        beta_commitment,
        initial_recurrent_state_commitment,
        final_recurrent_state_commitment,
        output_commitment,
        statement_hash,
    })
}

pub fn qwen35_delta_recurrence_statement(
    layer_idx: usize,
    inputs: &Qwen35DeltaRecurrenceInputs<'_>,
) -> Result<Qwen35DeltaRecurrenceStatement, String> {
    validate_shapes(inputs)?;

    let query_commitment = qwen35_delta_recurrence_query_commitment(inputs.query);
    let key_commitment = qwen35_delta_recurrence_key_commitment(inputs.key);
    let projected_value_commitment =
        qwen35_delta_recurrence_projected_value_commitment(inputs.projected_value);
    let a_gate_commitment = qwen35_delta_recurrence_a_gate_commitment(inputs.a_gate);
    let b_gate_commitment = qwen35_delta_recurrence_b_gate_commitment(inputs.b_gate);
    let a_log_weight_commitment =
        qwen35_delta_recurrence_a_log_weight_commitment(inputs.a_log_weight);
    let dt_bias_commitment = qwen35_delta_recurrence_dt_bias_commitment(inputs.dt_bias);
    let initial_recurrent_state_commitment =
        qwen35_delta_recurrence_initial_state_commitment(inputs.initial_recurrent_state);
    let final_recurrent_state_commitment =
        qwen35_delta_recurrence_final_state_commitment(inputs.final_recurrent_state);
    let output_commitment = qwen35_delta_recurrence_output_commitment(inputs.output);
    let air_spec = qwen35_delta_recurrence_air_spec(
        inputs.query.rows,
        inputs.query.cols,
        inputs.key.cols,
        inputs.projected_value.cols,
        inputs.state_rows,
        inputs.value_head_dim,
        inputs.mode,
        true,
        true,
    )?;
    let statement_hash = qwen35_delta_recurrence_statement_hash(
        layer_idx,
        inputs.query.rows,
        inputs.query.cols,
        inputs.key.cols,
        inputs.projected_value.cols,
        inputs.state_rows,
        inputs.value_head_dim,
        inputs.mode.as_u64(),
        air_spec.spec_hash,
        query_commitment,
        key_commitment,
        projected_value_commitment,
        a_gate_commitment,
        b_gate_commitment,
        a_log_weight_commitment,
        dt_bias_commitment,
        initial_recurrent_state_commitment,
        final_recurrent_state_commitment,
        output_commitment,
    );

    Ok(Qwen35DeltaRecurrenceStatement {
        layer_idx,
        seq_len: inputs.query.rows,
        query_width: inputs.query.cols,
        key_width: inputs.key.cols,
        value_width: inputs.projected_value.cols,
        state_rows: inputs.state_rows,
        value_head_dim: inputs.value_head_dim,
        mode: inputs.mode,
        air_spec_hash: air_spec.spec_hash,
        query_commitment,
        key_commitment,
        projected_value_commitment,
        a_gate_commitment,
        b_gate_commitment,
        a_log_weight_commitment,
        dt_bias_commitment,
        initial_recurrent_state_commitment,
        final_recurrent_state_commitment,
        output_commitment,
        statement_hash,
    })
}

fn validate_shapes(inputs: &Qwen35DeltaRecurrenceInputs<'_>) -> Result<(), String> {
    if inputs.query.rows == 0 || inputs.query.cols == 0 {
        return Err("DeltaRecurrence query must be non-empty".to_string());
    }
    if inputs.state_rows == 0 || inputs.value_head_dim == 0 {
        return Err("DeltaRecurrence state dimensions must be non-zero".to_string());
    }
    let seq_len = inputs.query.rows;
    for (name, matrix) in [
        ("key", inputs.key),
        ("projected_value", inputs.projected_value),
        ("a_gate", inputs.a_gate),
        ("b_gate", inputs.b_gate),
        ("output", inputs.output),
    ] {
        if matrix.rows != seq_len {
            return Err(format!(
                "DeltaRecurrence {name} rows {} != query rows {}",
                matrix.rows, seq_len
            ));
        }
        if matrix.cols == 0 {
            return Err(format!("DeltaRecurrence {name} width must be non-zero"));
        }
    }
    if inputs.query.cols != inputs.key.cols {
        return Err(format!(
            "DeltaRecurrence repeated query/key widths must match, got query={} key={}",
            inputs.query.cols, inputs.key.cols
        ));
    }
    if inputs.query.cols % inputs.state_rows != 0 {
        return Err(format!(
            "DeltaRecurrence query width {} must divide by state_rows {}",
            inputs.query.cols, inputs.state_rows
        ));
    }
    let qk_head_dim = inputs.query.cols / inputs.state_rows;
    let expected_state_shape = (inputs.state_rows * qk_head_dim, inputs.value_head_dim);
    for (name, matrix) in [
        ("initial_recurrent_state", inputs.initial_recurrent_state),
        ("final_recurrent_state", inputs.final_recurrent_state),
    ] {
        if (matrix.rows, matrix.cols) != expected_state_shape {
            return Err(format!(
                "DeltaRecurrence {name} shape [{}x{}] != expected [{}x{}]",
                matrix.rows, matrix.cols, expected_state_shape.0, expected_state_shape.1
            ));
        }
    }
    if inputs.a_gate.cols != inputs.state_rows || inputs.b_gate.cols != inputs.state_rows {
        return Err(format!(
            "DeltaRecurrence gate widths a={} b={} must equal state_rows {}",
            inputs.a_gate.cols, inputs.b_gate.cols, inputs.state_rows
        ));
    }
    if inputs.a_log_weight.len() != inputs.state_rows || inputs.dt_bias.len() != inputs.state_rows {
        return Err(format!(
            "DeltaRecurrence vector widths a_log={} dt_bias={} must equal state_rows {}",
            inputs.a_log_weight.len(),
            inputs.dt_bias.len(),
            inputs.state_rows
        ));
    }
    if inputs.projected_value.cols != inputs.state_rows * inputs.value_head_dim {
        return Err(format!(
            "DeltaRecurrence projected_value width {} != state_rows*value_head_dim {}",
            inputs.projected_value.cols,
            inputs.state_rows * inputs.value_head_dim
        ));
    }
    if inputs.output.cols != inputs.state_rows * inputs.value_head_dim {
        return Err(format!(
            "DeltaRecurrence output width {} != state_rows*value_head_dim {}",
            inputs.output.cols,
            inputs.state_rows * inputs.value_head_dim
        ));
    }
    Ok(())
}

fn validate_transform_shapes(
    inputs: &Qwen35DeltaRecurrenceTransformInputs<'_>,
) -> Result<(), String> {
    if inputs.query.rows == 0 || inputs.query.cols == 0 {
        return Err("DeltaRecurrence transform query must be non-empty".to_string());
    }
    if inputs.state_rows == 0 || inputs.value_head_dim == 0 {
        return Err("DeltaRecurrence transform state dimensions must be non-zero".to_string());
    }
    let seq_len = inputs.query.rows;
    for (name, matrix) in [
        ("key", inputs.key),
        ("a_gate", inputs.a_gate),
        ("b_gate", inputs.b_gate),
        ("scaled_query", inputs.scaled_query),
        ("normalized_key", inputs.normalized_key),
        ("decay", inputs.decay),
        ("beta", inputs.beta),
    ] {
        if matrix.rows != seq_len {
            return Err(format!(
                "DeltaRecurrence transform {name} rows {} != query rows {}",
                matrix.rows, seq_len
            ));
        }
        if matrix.cols == 0 {
            return Err(format!(
                "DeltaRecurrence transform {name} width must be non-zero"
            ));
        }
    }
    if inputs.query.cols != inputs.key.cols
        || inputs.query.cols != inputs.scaled_query.cols
        || inputs.query.cols != inputs.normalized_key.cols
    {
        return Err(format!(
            "DeltaRecurrence transform q/k widths must match original and transformed tensors, got query={} key={} scaled_query={} normalized_key={}",
            inputs.query.cols,
            inputs.key.cols,
            inputs.scaled_query.cols,
            inputs.normalized_key.cols
        ));
    }
    if inputs.query.cols % inputs.state_rows != 0 {
        return Err(format!(
            "DeltaRecurrence transform query width {} must divide by state_rows {}",
            inputs.query.cols, inputs.state_rows
        ));
    }
    if inputs.a_gate.cols != inputs.state_rows
        || inputs.b_gate.cols != inputs.state_rows
        || inputs.decay.cols != inputs.state_rows
        || inputs.beta.cols != inputs.state_rows
    {
        return Err(format!(
            "DeltaRecurrence transform gate/decay/beta widths a={} b={} decay={} beta={} must equal state_rows {}",
            inputs.a_gate.cols,
            inputs.b_gate.cols,
            inputs.decay.cols,
            inputs.beta.cols,
            inputs.state_rows
        ));
    }
    if inputs.a_log_weight.len() != inputs.state_rows || inputs.dt_bias.len() != inputs.state_rows {
        return Err(format!(
            "DeltaRecurrence transform vector widths a_log={} dt_bias={} must equal state_rows {}",
            inputs.a_log_weight.len(),
            inputs.dt_bias.len(),
            inputs.state_rows
        ));
    }
    Ok(())
}

fn validate_arithmetic_shapes(
    inputs: &Qwen35DeltaRecurrenceArithmeticInputs<'_>,
) -> Result<(), String> {
    if inputs.scaled_query.rows == 0 || inputs.scaled_query.cols == 0 {
        return Err("DeltaRecurrence arithmetic query must be non-empty".to_string());
    }
    if inputs.state_rows == 0 || inputs.value_head_dim == 0 {
        return Err("DeltaRecurrence arithmetic state dimensions must be non-zero".to_string());
    }
    let seq_len = inputs.scaled_query.rows;
    for (name, matrix) in [
        ("normalized_key", inputs.normalized_key),
        ("projected_value", inputs.projected_value),
        ("decay", inputs.decay),
        ("beta", inputs.beta),
        ("output", inputs.output),
    ] {
        if matrix.rows != seq_len {
            return Err(format!(
                "DeltaRecurrence arithmetic {name} rows {} != query rows {}",
                matrix.rows, seq_len
            ));
        }
        if matrix.cols == 0 {
            return Err(format!(
                "DeltaRecurrence arithmetic {name} width must be non-zero"
            ));
        }
    }
    if inputs.scaled_query.cols != inputs.normalized_key.cols {
        return Err(format!(
            "DeltaRecurrence arithmetic repeated query/key widths must match, got query={} key={}",
            inputs.scaled_query.cols, inputs.normalized_key.cols
        ));
    }
    if inputs.scaled_query.cols % inputs.state_rows != 0 {
        return Err(format!(
            "DeltaRecurrence arithmetic query width {} must divide by state_rows {}",
            inputs.scaled_query.cols, inputs.state_rows
        ));
    }
    let qk_head_dim = inputs.scaled_query.cols / inputs.state_rows;
    let expected_state_shape = (inputs.state_rows * qk_head_dim, inputs.value_head_dim);
    for (name, matrix) in [
        ("initial_recurrent_state", inputs.initial_recurrent_state),
        ("final_recurrent_state", inputs.final_recurrent_state),
    ] {
        if (matrix.rows, matrix.cols) != expected_state_shape {
            return Err(format!(
                "DeltaRecurrence arithmetic {name} shape [{}x{}] != expected [{}x{}]",
                matrix.rows, matrix.cols, expected_state_shape.0, expected_state_shape.1
            ));
        }
    }
    if inputs.decay.cols != inputs.state_rows || inputs.beta.cols != inputs.state_rows {
        return Err(format!(
            "DeltaRecurrence arithmetic decay/beta widths decay={} beta={} must equal state_rows {}",
            inputs.decay.cols, inputs.beta.cols, inputs.state_rows
        ));
    }
    if inputs.projected_value.cols != inputs.state_rows * inputs.value_head_dim {
        return Err(format!(
            "DeltaRecurrence arithmetic projected_value width {} != state_rows*value_head_dim {}",
            inputs.projected_value.cols,
            inputs.state_rows * inputs.value_head_dim
        ));
    }
    if inputs.output.cols != inputs.state_rows * inputs.value_head_dim {
        return Err(format!(
            "DeltaRecurrence arithmetic output width {} != state_rows*value_head_dim {}",
            inputs.output.cols,
            inputs.state_rows * inputs.value_head_dim
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, cols: usize, seed: u32) -> M31Matrix {
        let mut matrix = M31Matrix::new(rows, cols);
        for row in 0..rows {
            for col in 0..cols {
                matrix.set(row, col, M31::from(seed + (row * cols + col) as u32));
            }
        }
        matrix
    }

    fn qwen35_norm_output(
        input: &M31Matrix,
        state_rows: usize,
        table_log_size: u32,
        post_scale: M31,
    ) -> M31Matrix {
        let qk_head_dim = input.cols / state_rows;
        let table = build_rsqrt_table(table_log_size);
        let mut output = M31Matrix::new(input.rows, input.cols);
        for token_idx in 0..input.rows {
            for state_row_idx in 0..state_rows {
                let mut sum_sq = M31::from(0u32);
                for qk_idx in 0..qk_head_dim {
                    let col_idx = state_row_idx * qk_head_dim + qk_idx;
                    let value = input.get(token_idx, col_idx);
                    sum_sq += value * value;
                }
                let rms_sq = sum_sq;
                let rsqrt = table.lookup(rms_sq).unwrap();
                for qk_idx in 0..qk_head_dim {
                    let col_idx = state_row_idx * qk_head_dim + qk_idx;
                    output.set(
                        token_idx,
                        col_idx,
                        input.get(token_idx, col_idx) * rsqrt * post_scale,
                    );
                }
            }
        }
        output
    }

    fn qwen35_decay_output(
        a_gate: &M31Matrix,
        a_log_weight: &[M31],
        dt_bias: &[M31],
        table_log_size: u32,
    ) -> M31Matrix {
        let softplus_table = qwen35_softplus_table(table_log_size);
        let exp_table = qwen35_exp_table(table_log_size);
        let decay_table = qwen35_decay_exp_table(table_log_size);
        let mut decay = M31Matrix::new(a_gate.rows, a_gate.cols);
        for token_idx in 0..a_gate.rows {
            for state_row_idx in 0..a_gate.cols {
                let a_sum = a_gate.get(token_idx, state_row_idx) + dt_bias[state_row_idx];
                let softplus = softplus_table.lookup(a_sum).unwrap();
                let exp_a_log = exp_table.lookup(a_log_weight[state_row_idx]).unwrap();
                let product = M31::from(((exp_a_log.0 as u64 * softplus.0 as u64) >> 16) as u32);
                decay.set(
                    token_idx,
                    state_row_idx,
                    decay_table.lookup(product).unwrap(),
                );
            }
        }
        decay
    }

    #[test]
    fn qwen35_delta_recurrence_statement_binds_all_inputs_output_and_layer() {
        let query = matrix(2, 4, 10);
        let key = matrix(2, 4, 20);
        let projected_value = matrix(2, 6, 30);
        let a_gate = matrix(2, 2, 40);
        let b_gate = matrix(2, 2, 50);
        let initial_recurrent_state = matrix(4, 3, 55);
        let final_recurrent_state = matrix(4, 3, 56);
        let output = matrix(2, 6, 60);
        let a_log_weight = vec![M31::from(70u32), M31::from(71u32)];
        let dt_bias = vec![M31::from(80u32), M31::from(81u32)];
        let inputs = Qwen35DeltaRecurrenceInputs {
            query: &query,
            key: &key,
            projected_value: &projected_value,
            a_gate: &a_gate,
            b_gate: &b_gate,
            a_log_weight: &a_log_weight,
            dt_bias: &dt_bias,
            initial_recurrent_state: &initial_recurrent_state,
            final_recurrent_state: &final_recurrent_state,
            output: &output,
            state_rows: 2,
            value_head_dim: 3,
            mode: Qwen35DeltaRecurrenceMode::ChunkPrefill,
        };

        let statement = qwen35_delta_recurrence_statement(7, &inputs).unwrap();
        assert_eq!(statement.layer_idx, 7);
        assert_eq!(statement.seq_len, 2);
        assert_eq!(statement.query_width, 4);
        assert_eq!(statement.key_width, 4);
        assert_eq!(statement.value_width, 6);
        assert_eq!(statement.state_rows, 2);
        assert_eq!(statement.value_head_dim, 3);
        assert_eq!(statement.mode, Qwen35DeltaRecurrenceMode::ChunkPrefill);
        assert_ne!(statement.air_spec_hash, FieldElement::ZERO);

        let mut other_output = output.clone();
        other_output.set(1, 5, M31::from(999u32));
        let other_inputs = Qwen35DeltaRecurrenceInputs {
            output: &other_output,
            ..inputs
        };
        let other_statement = qwen35_delta_recurrence_statement(7, &other_inputs).unwrap();
        assert_ne!(
            statement.output_commitment,
            other_statement.output_commitment
        );
        assert_ne!(statement.statement_hash, other_statement.statement_hash);

        let mut other_final_state = final_recurrent_state.clone();
        other_final_state.set(3, 2, M31::from(1001u32));
        let other_state_inputs = Qwen35DeltaRecurrenceInputs {
            final_recurrent_state: &other_final_state,
            ..inputs
        };
        let other_state_statement =
            qwen35_delta_recurrence_statement(7, &other_state_inputs).unwrap();
        assert_ne!(
            statement.final_recurrent_state_commitment,
            other_state_statement.final_recurrent_state_commitment
        );
        assert_ne!(
            statement.statement_hash,
            other_state_statement.statement_hash
        );

        let other_layer = qwen35_delta_recurrence_statement(8, &inputs).unwrap();
        assert_ne!(statement.statement_hash, other_layer.statement_hash);

        let decode_inputs = Qwen35DeltaRecurrenceInputs {
            mode: Qwen35DeltaRecurrenceMode::RecurrentDecode,
            ..inputs
        };
        let decode_statement = qwen35_delta_recurrence_statement(7, &decode_inputs).unwrap();
        assert_ne!(statement.air_spec_hash, decode_statement.air_spec_hash);
        assert_ne!(statement.statement_hash, decode_statement.statement_hash);
    }

    #[test]
    fn qwen35_delta_recurrence_statement_rejects_shape_drift() {
        let query = matrix(2, 4, 10);
        let key = matrix(3, 4, 20);
        let projected_value = matrix(2, 6, 30);
        let a_gate = matrix(2, 2, 40);
        let b_gate = matrix(2, 2, 50);
        let initial_recurrent_state = matrix(4, 3, 55);
        let final_recurrent_state = matrix(4, 3, 56);
        let output = matrix(2, 6, 60);
        let a_log_weight = vec![M31::from(70u32), M31::from(71u32)];
        let dt_bias = vec![M31::from(80u32), M31::from(81u32)];
        let inputs = Qwen35DeltaRecurrenceInputs {
            query: &query,
            key: &key,
            projected_value: &projected_value,
            a_gate: &a_gate,
            b_gate: &b_gate,
            a_log_weight: &a_log_weight,
            dt_bias: &dt_bias,
            initial_recurrent_state: &initial_recurrent_state,
            final_recurrent_state: &final_recurrent_state,
            output: &output,
            state_rows: 2,
            value_head_dim: 3,
            mode: Qwen35DeltaRecurrenceMode::ChunkPrefill,
        };
        assert!(qwen35_delta_recurrence_statement(0, &inputs)
            .unwrap_err()
            .contains("key rows"));
    }

    #[test]
    fn qwen35_delta_recurrence_transform_binding_binds_source_and_target_tensors() {
        let query = matrix(2, 4, 10);
        let key = matrix(2, 4, 20);
        let scaled_query = matrix(2, 4, 30);
        let normalized_key = matrix(2, 4, 40);
        let a_gate = matrix(2, 2, 50);
        let b_gate = matrix(2, 2, 60);
        let decay = matrix(2, 2, 70);
        let beta = matrix(2, 2, 80);
        let a_log_weight = vec![M31::from(90u32), M31::from(91u32)];
        let dt_bias = vec![M31::from(100u32), M31::from(101u32)];
        let inputs = Qwen35DeltaRecurrenceTransformInputs {
            query: &query,
            key: &key,
            a_gate: &a_gate,
            b_gate: &b_gate,
            a_log_weight: &a_log_weight,
            dt_bias: &dt_bias,
            scaled_query: &scaled_query,
            normalized_key: &normalized_key,
            decay: &decay,
            beta: &beta,
            state_rows: 2,
            value_head_dim: 3,
            mode: Qwen35DeltaRecurrenceMode::RecurrentDecode,
        };

        let statement = qwen35_delta_recurrence_transform_statement(3, &inputs).unwrap();
        assert_eq!(statement.layer_idx, 3);
        assert_eq!(statement.seq_len, 2);
        assert_eq!(statement.query_width, 4);
        assert_eq!(statement.state_rows, 2);
        assert_ne!(statement.statement_hash, FieldElement::ZERO);

        let witness = qwen35_delta_recurrence_transform_binding_witness(&inputs).unwrap();
        verify_qwen35_delta_recurrence_transform_binding_witness(&inputs, &witness).unwrap();
        assert_eq!(witness.rows.len(), 8);
        assert_eq!(witness.rows[0].query, query.get(0, 0));
        assert_eq!(witness.rows[0].scaled_query, scaled_query.get(0, 0));

        let trace = qwen35_delta_recurrence_transform_binding_trace(&inputs).unwrap();
        {
            use num_traits::Zero;
            use stwo::core::pcs::TreeVec;
            use stwo_constraint_framework::assert_constraints_on_trace;

            let preprocessed = trace.preprocessed.iter().collect::<Vec<_>>();
            let execution = trace.execution.iter().collect::<Vec<_>>();
            let trees = TreeVec::new(vec![preprocessed, execution]);
            let eval = Qwen35DeltaRecurrenceTransformBindingEval {
                log_n_rows: trace.log_size,
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

            let mut tampered_trace = trace.clone();
            tampered_trace.execution[2][0] += M31::from(1u32);
            let preprocessed = tampered_trace.preprocessed.iter().collect::<Vec<_>>();
            let execution = tampered_trace.execution.iter().collect::<Vec<_>>();
            let trees = TreeVec::new(vec![preprocessed, execution]);
            let result = std::panic::catch_unwind(|| {
                assert_constraints_on_trace(
                    &trees,
                    tampered_trace.log_size,
                    |row_eval| {
                        let _ = eval.evaluate(row_eval);
                    },
                    stwo::core::fields::qm31::SecureField::zero(),
                );
            });
            assert!(result.is_err());
        }

        let proof = prove_qwen35_delta_recurrence_transform_binding_air(3, &inputs).unwrap();
        assert_eq!(proof.n_real_rows, 8);
        assert_eq!(proof.statement.statement_hash, statement.statement_hash);
        verify_qwen35_delta_recurrence_transform_binding_air(&proof).unwrap();
        let wrong_statement = qwen35_delta_recurrence_transform_statement(4, &inputs)
            .unwrap()
            .statement_hash;
        let err = verify_qwen35_delta_recurrence_transform_binding_air_with_statement_hash(
            &proof,
            wrong_statement,
        )
        .unwrap_err();
        assert!(err.to_string().contains("statement hash mismatch"));

        let mut swapped = witness.clone();
        swapped.rows.swap(1, 2);
        swapped.witness_hash =
            qwen35_delta_recurrence_transform_binding_witness_hash(&inputs, &swapped.rows).unwrap();
        let err = verify_qwen35_delta_recurrence_transform_binding_witness(&inputs, &swapped)
            .unwrap_err();
        assert!(err.contains("expected token"));

        let mut tampered = witness.clone();
        tampered.rows[0].scaled_query += M31::from(1u32);
        tampered.witness_hash =
            qwen35_delta_recurrence_transform_binding_witness_hash(&inputs, &tampered.rows)
                .unwrap();
        let err = verify_qwen35_delta_recurrence_transform_binding_witness(&inputs, &tampered)
            .unwrap_err();
        assert!(err.contains("active transform tensors"));

        let mut other_decay = decay.clone();
        other_decay.set(1, 1, other_decay.get(1, 1) + M31::from(1u32));
        let other_inputs = Qwen35DeltaRecurrenceTransformInputs {
            decay: &other_decay,
            ..inputs
        };
        let other_statement =
            qwen35_delta_recurrence_transform_statement(3, &other_inputs).unwrap();
        assert_ne!(statement.decay_commitment, other_statement.decay_commitment);
        assert_ne!(statement.statement_hash, other_statement.statement_hash);
    }

    #[test]
    fn qwen35_delta_recurrence_beta_sigmoid_logup_proves_table_bound_beta() {
        let b_gate = matrix(2, 2, 0);
        let mut beta = M31Matrix::new(2, 2);
        for row in 0..2 {
            for col in 0..2 {
                let input = b_gate.get(row, col);
                let output = crate::gadgets::lookup_table::activations::sigmoid_approx(input);
                beta.set(row, col, output);
            }
        }

        let statement = qwen35_delta_recurrence_beta_sigmoid_statement(9, &b_gate, &beta, 4)
            .expect("statement");
        assert_eq!(statement.layer_idx, 9);
        assert_eq!(statement.seq_len, 2);
        assert_eq!(statement.state_rows, 2);
        assert_eq!(
            statement.table_commitment,
            qwen35_delta_recurrence_beta_sigmoid_table_commitment(4)
        );
        assert_eq!(
            statement.trace_checksum,
            qwen35_delta_recurrence_beta_sigmoid_trace_checksum(&b_gate, &beta, 4).unwrap()
        );

        let proof = prove_qwen35_delta_recurrence_beta_sigmoid_air(9, &b_gate, &beta, 4)
            .expect("beta sigmoid proof");
        assert_eq!(proof.n_real_rows, 4);
        assert_eq!(proof.statement.statement_hash, statement.statement_hash);
        verify_qwen35_delta_recurrence_beta_sigmoid_air(&proof).unwrap();

        let wrong_statement = qwen35_delta_recurrence_beta_sigmoid_statement(10, &b_gate, &beta, 4)
            .unwrap()
            .statement_hash;
        let err = verify_qwen35_delta_recurrence_beta_sigmoid_air_with_statement_hash(
            &proof,
            wrong_statement,
        )
        .unwrap_err();
        assert!(err.to_string().contains("statement hash mismatch"));

        let mut bad_table_proof =
            prove_qwen35_delta_recurrence_beta_sigmoid_air(9, &b_gate, &beta, 4).unwrap();
        bad_table_proof.statement.table_commitment += FieldElement::ONE;
        let err = verify_qwen35_delta_recurrence_beta_sigmoid_air(&bad_table_proof).unwrap_err();
        assert!(err.to_string().contains("table commitment mismatch"));

        let mut bad_beta = beta.clone();
        bad_beta.set(0, 0, bad_beta.get(0, 0) + M31::from(1u32));
        let err =
            prove_qwen35_delta_recurrence_beta_sigmoid_air(9, &b_gate, &bad_beta, 4).unwrap_err();
        assert!(err.to_string().contains("output mismatch"));

        let mut swapped_b_gate = b_gate.clone();
        let mut swapped_beta = beta.clone();
        let b_00 = swapped_b_gate.get(0, 0);
        let b_01 = swapped_b_gate.get(0, 1);
        let beta_00 = swapped_beta.get(0, 0);
        let beta_01 = swapped_beta.get(0, 1);
        swapped_b_gate.set(0, 0, b_01);
        swapped_b_gate.set(0, 1, b_00);
        swapped_beta.set(0, 0, beta_01);
        swapped_beta.set(0, 1, beta_00);
        let swapped_statement =
            qwen35_delta_recurrence_beta_sigmoid_statement(9, &swapped_b_gate, &swapped_beta, 4)
                .unwrap();
        assert_ne!(statement.trace_checksum, swapped_statement.trace_checksum);
    }

    #[test]
    fn qwen35_delta_recurrence_norm_air_proves_bound_qk_rms_transform() {
        let input = matrix(2, 4, 1);
        let state_rows = 2;
        let table_log_size = 8;
        let post_scale = M31::from(1u32);
        let output = qwen35_norm_output(&input, state_rows, table_log_size, post_scale);

        let statement = qwen35_delta_recurrence_norm_statement(
            Qwen35DeltaRecurrenceNormKind::Query,
            4,
            &input,
            &output,
            state_rows,
            table_log_size,
            post_scale,
        )
        .expect("statement");
        assert_eq!(statement.layer_idx, 4);
        assert_eq!(statement.seq_len, 2);
        assert_eq!(statement.state_rows, 2);
        assert_eq!(statement.qk_head_dim, 2);
        assert_eq!(
            statement.input_commitment,
            qwen35_delta_recurrence_query_commitment(&input)
        );
        assert_eq!(
            statement.output_commitment,
            qwen35_delta_recurrence_scaled_query_commitment(&output)
        );
        assert_eq!(
            statement.trace_checksum,
            qwen35_delta_recurrence_norm_trace_checksum(
                &input,
                &output,
                state_rows,
                table_log_size,
                post_scale
            )
            .unwrap()
        );

        let proof = prove_qwen35_delta_recurrence_norm_air(
            Qwen35DeltaRecurrenceNormKind::Query,
            4,
            &input,
            &output,
            state_rows,
            table_log_size,
            post_scale,
        )
        .expect("norm proof");
        assert_eq!(proof.n_real_rows, 8);
        assert_eq!(proof.statement.statement_hash, statement.statement_hash);
        verify_qwen35_delta_recurrence_norm_air(&proof).unwrap();

        let wrong_statement = qwen35_delta_recurrence_norm_statement(
            Qwen35DeltaRecurrenceNormKind::Key,
            4,
            &input,
            &output,
            state_rows,
            table_log_size,
            post_scale,
        )
        .unwrap()
        .statement_hash;
        let err =
            verify_qwen35_delta_recurrence_norm_air_with_statement_hash(&proof, wrong_statement)
                .unwrap_err();
        assert!(err.to_string().contains("statement hash mismatch"));

        let mut bad_table_proof = prove_qwen35_delta_recurrence_norm_air(
            Qwen35DeltaRecurrenceNormKind::Query,
            4,
            &input,
            &output,
            state_rows,
            table_log_size,
            post_scale,
        )
        .unwrap();
        bad_table_proof.statement.table_commitment += FieldElement::ONE;
        let err = verify_qwen35_delta_recurrence_norm_air(&bad_table_proof).unwrap_err();
        assert!(err.to_string().contains("table commitment mismatch"));

        let mut bad_output = output.clone();
        bad_output.set(0, 0, bad_output.get(0, 0) + M31::from(1u32));
        let err = prove_qwen35_delta_recurrence_norm_air(
            Qwen35DeltaRecurrenceNormKind::Query,
            4,
            &input,
            &bad_output,
            state_rows,
            table_log_size,
            post_scale,
        )
        .unwrap_err();
        assert!(err.to_string().contains("output mismatch"));

        let mut swapped_input = input.clone();
        let mut swapped_output = output.clone();
        let in_00 = swapped_input.get(0, 0);
        let in_01 = swapped_input.get(0, 1);
        let out_00 = swapped_output.get(0, 0);
        let out_01 = swapped_output.get(0, 1);
        swapped_input.set(0, 0, in_01);
        swapped_input.set(0, 1, in_00);
        swapped_output.set(0, 0, out_01);
        swapped_output.set(0, 1, out_00);
        let swapped_statement = qwen35_delta_recurrence_norm_statement(
            Qwen35DeltaRecurrenceNormKind::Query,
            4,
            &swapped_input,
            &swapped_output,
            state_rows,
            table_log_size,
            post_scale,
        )
        .unwrap();
        assert_ne!(statement.trace_checksum, swapped_statement.trace_checksum);
    }

    #[test]
    fn qwen35_delta_recurrence_decay_air_proves_bound_nonlinear_decay() {
        let a_gate = matrix(2, 2, 1);
        let a_log_weight = vec![M31::from(2u32), M31::from(3u32)];
        let dt_bias = vec![M31::from(4u32), M31::from(5u32)];
        let table_log_size = 16;
        let decay = qwen35_decay_output(&a_gate, &a_log_weight, &dt_bias, table_log_size);

        let statement = qwen35_delta_recurrence_decay_statement(
            5,
            &a_gate,
            &a_log_weight,
            &dt_bias,
            &decay,
            table_log_size,
        )
        .expect("statement");
        assert_eq!(statement.layer_idx, 5);
        assert_eq!(statement.seq_len, 2);
        assert_eq!(statement.state_rows, 2);
        assert_eq!(
            statement.a_gate_commitment,
            qwen35_delta_recurrence_a_gate_commitment(&a_gate)
        );
        assert_eq!(
            statement.decay_commitment,
            qwen35_delta_recurrence_decay_commitment(&decay)
        );
        assert_eq!(
            statement.trace_checksum,
            qwen35_delta_recurrence_decay_trace_checksum(
                &a_gate,
                &a_log_weight,
                &dt_bias,
                &decay,
                table_log_size
            )
            .unwrap()
        );

        let mut swapped_a_gate = a_gate.clone();
        let first = swapped_a_gate.get(0, 0);
        swapped_a_gate.set(0, 0, swapped_a_gate.get(1, 1));
        swapped_a_gate.set(1, 1, first);
        let swapped_decay =
            qwen35_decay_output(&swapped_a_gate, &a_log_weight, &dt_bias, table_log_size);
        let swapped_statement = qwen35_delta_recurrence_decay_statement(
            5,
            &swapped_a_gate,
            &a_log_weight,
            &dt_bias,
            &swapped_decay,
            table_log_size,
        )
        .unwrap();
        assert_ne!(statement.trace_checksum, swapped_statement.trace_checksum);

        let proof = prove_qwen35_delta_recurrence_decay_air(
            5,
            &a_gate,
            &a_log_weight,
            &dt_bias,
            &decay,
            table_log_size,
        )
        .expect("decay proof");
        assert_eq!(proof.n_real_rows, 4);
        assert_eq!(proof.statement.statement_hash, statement.statement_hash);
        verify_qwen35_delta_recurrence_decay_air(&proof).unwrap();

        let wrong_statement = qwen35_delta_recurrence_decay_statement(
            6,
            &a_gate,
            &a_log_weight,
            &dt_bias,
            &decay,
            table_log_size,
        )
        .unwrap()
        .statement_hash;
        let err =
            verify_qwen35_delta_recurrence_decay_air_with_statement_hash(&proof, wrong_statement)
                .unwrap_err();
        assert!(err.to_string().contains("statement hash mismatch"));

        let mut bad_table_proof = prove_qwen35_delta_recurrence_decay_air(
            5,
            &a_gate,
            &a_log_weight,
            &dt_bias,
            &decay,
            table_log_size,
        )
        .unwrap();
        bad_table_proof.statement.exp_table_commitment += FieldElement::ONE;
        let err = verify_qwen35_delta_recurrence_decay_air(&bad_table_proof).unwrap_err();
        assert!(err.to_string().contains("table commitment mismatch"));

        let mut bad_decay = decay.clone();
        bad_decay.set(0, 0, bad_decay.get(0, 0) + M31::from(1u32));
        let err = prove_qwen35_delta_recurrence_decay_air(
            5,
            &a_gate,
            &a_log_weight,
            &dt_bias,
            &bad_decay,
            table_log_size,
        )
        .unwrap_err();
        assert!(err.to_string().contains("output mismatch"));
    }

    #[test]
    fn qwen35_delta_recurrence_arithmetic_witness_enforces_gated_delta_rule() {
        let scaled_query = matrix(2, 4, 2);
        let normalized_key = matrix(2, 4, 7);
        let projected_value = matrix(2, 6, 11);
        let decay = matrix(2, 2, 3);
        let beta = matrix(2, 2, 5);
        let initial_recurrent_state = matrix(4, 3, 13);
        let placeholder_final_state = matrix(4, 3, 0);
        let placeholder_output = matrix(2, 6, 0);
        let draft_inputs = Qwen35DeltaRecurrenceArithmeticInputs {
            scaled_query: &scaled_query,
            normalized_key: &normalized_key,
            projected_value: &projected_value,
            decay: &decay,
            beta: &beta,
            initial_recurrent_state: &initial_recurrent_state,
            final_recurrent_state: &placeholder_final_state,
            output: &placeholder_output,
            state_rows: 2,
            value_head_dim: 3,
        };
        let expected = qwen35_delta_recurrence_arithmetic_witness(&draft_inputs).unwrap();
        let final_recurrent_state = expected.final_recurrent_state.clone();
        let output = expected.output.clone();
        let inputs = Qwen35DeltaRecurrenceArithmeticInputs {
            final_recurrent_state: &final_recurrent_state,
            output: &output,
            ..draft_inputs
        };

        let witness = qwen35_delta_recurrence_arithmetic_witness(&inputs).unwrap();
        verify_qwen35_delta_recurrence_arithmetic_witness(&inputs, &witness).unwrap();
        assert_eq!(witness.rows.len(), 24);

        let first = &witness.rows[0];
        assert_eq!(first.token_idx, 0);
        assert_eq!(first.state_row_idx, 0);
        assert_eq!(first.qk_col_idx, 0);
        assert_eq!(first.value_col_idx, 0);
        assert_eq!(first.decayed_state, first.state_before * first.decay);
        assert_eq!(
            first.state_after,
            first.decayed_state + first.normalized_key * first.delta
        );

        let mut tampered = witness.clone();
        tampered.rows[0].state_after += M31::from(1u32);
        let err =
            verify_qwen35_delta_recurrence_arithmetic_witness(&inputs, &tampered).unwrap_err();
        assert!(err.contains("rows do not match recurrence"));

        let mut bad_output = output.clone();
        bad_output.set(1, 5, bad_output.get(1, 5) + M31::from(1u32));
        let bad_output_inputs = Qwen35DeltaRecurrenceArithmeticInputs {
            output: &bad_output,
            ..inputs.clone()
        };
        let err = verify_qwen35_delta_recurrence_arithmetic_witness(&bad_output_inputs, &witness)
            .unwrap_err();
        assert!(err.contains("claimed output"));

        let mut bad_state = final_recurrent_state.clone();
        bad_state.set(3, 2, bad_state.get(3, 2) + M31::from(1u32));
        let bad_state_inputs = Qwen35DeltaRecurrenceArithmeticInputs {
            final_recurrent_state: &bad_state,
            ..inputs.clone()
        };
        let err = verify_qwen35_delta_recurrence_arithmetic_witness(&bad_state_inputs, &witness)
            .unwrap_err();
        assert!(err.contains("claimed final state"));
    }

    #[test]
    fn qwen35_delta_recurrence_arithmetic_air_constraints_hold_and_reject_tamper() {
        use num_traits::Zero;
        use stwo::core::pcs::TreeVec;
        use stwo_constraint_framework::assert_constraints_on_trace;

        let scaled_query = matrix(2, 4, 2);
        let normalized_key = matrix(2, 4, 7);
        let projected_value = matrix(2, 6, 11);
        let decay = matrix(2, 2, 3);
        let beta = matrix(2, 2, 5);
        let initial_recurrent_state = matrix(4, 3, 13);
        let placeholder_final_state = matrix(4, 3, 0);
        let placeholder_output = matrix(2, 6, 0);
        let draft_inputs = Qwen35DeltaRecurrenceArithmeticInputs {
            scaled_query: &scaled_query,
            normalized_key: &normalized_key,
            projected_value: &projected_value,
            decay: &decay,
            beta: &beta,
            initial_recurrent_state: &initial_recurrent_state,
            final_recurrent_state: &placeholder_final_state,
            output: &placeholder_output,
            state_rows: 2,
            value_head_dim: 3,
        };
        let expected = qwen35_delta_recurrence_arithmetic_witness(&draft_inputs).unwrap();
        let final_recurrent_state = expected.final_recurrent_state.clone();
        let output = expected.output.clone();
        let inputs = Qwen35DeltaRecurrenceArithmeticInputs {
            final_recurrent_state: &final_recurrent_state,
            output: &output,
            ..draft_inputs
        };
        let trace = qwen35_delta_recurrence_arithmetic_trace(&inputs).unwrap();
        let preprocessed = trace.preprocessed.iter().collect::<Vec<_>>();
        let execution = trace.execution.iter().collect::<Vec<_>>();
        let trees = TreeVec::new(vec![preprocessed, execution]);
        let eval = Qwen35DeltaRecurrenceArithmeticEval {
            log_n_rows: trace.log_size,
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

        let mut tampered = trace.clone();
        tampered.execution[10][0] += M31::from(1u32);
        let preprocessed = tampered.preprocessed.iter().collect::<Vec<_>>();
        let execution = tampered.execution.iter().collect::<Vec<_>>();
        let trees = TreeVec::new(vec![preprocessed, execution]);
        let result = std::panic::catch_unwind(|| {
            assert_constraints_on_trace(
                &trees,
                tampered.log_size,
                |row_eval| {
                    let _ = eval.evaluate(row_eval);
                },
                stwo::core::fields::qm31::SecureField::zero(),
            );
        });
        assert!(result.is_err());

        let mut broken_fold = trace;
        broken_fold.execution[14][0] += M31::from(1u32);
        let preprocessed = broken_fold.preprocessed.iter().collect::<Vec<_>>();
        let execution = broken_fold.execution.iter().collect::<Vec<_>>();
        let trees = TreeVec::new(vec![preprocessed, execution]);
        let result = std::panic::catch_unwind(|| {
            assert_constraints_on_trace(
                &trees,
                broken_fold.log_size,
                |row_eval| {
                    let _ = eval.evaluate(row_eval);
                },
                stwo::core::fields::qm31::SecureField::zero(),
            );
        });
        assert!(result.is_err());

        let mut broken_continuity = qwen35_delta_recurrence_arithmetic_trace(&inputs).unwrap();
        broken_continuity.execution[23][0] += M31::from(1u32);
        let preprocessed = broken_continuity.preprocessed.iter().collect::<Vec<_>>();
        let execution = broken_continuity.execution.iter().collect::<Vec<_>>();
        let trees = TreeVec::new(vec![preprocessed, execution]);
        let result = std::panic::catch_unwind(|| {
            assert_constraints_on_trace(
                &trees,
                broken_continuity.log_size,
                |row_eval| {
                    let _ = eval.evaluate(row_eval);
                },
                stwo::core::fields::qm31::SecureField::zero(),
            );
        });
        assert!(result.is_err());
    }

    #[test]
    fn qwen35_delta_recurrence_arithmetic_air_proves_and_verifies_standalone() {
        let scaled_query = matrix(1, 4, 2);
        let normalized_key = matrix(1, 4, 7);
        let projected_value = matrix(1, 6, 11);
        let decay = matrix(1, 2, 3);
        let beta = matrix(1, 2, 5);
        let initial_recurrent_state = matrix(4, 3, 13);
        let placeholder_final_state = matrix(4, 3, 0);
        let placeholder_output = matrix(1, 6, 0);
        let draft_inputs = Qwen35DeltaRecurrenceArithmeticInputs {
            scaled_query: &scaled_query,
            normalized_key: &normalized_key,
            projected_value: &projected_value,
            decay: &decay,
            beta: &beta,
            initial_recurrent_state: &initial_recurrent_state,
            final_recurrent_state: &placeholder_final_state,
            output: &placeholder_output,
            state_rows: 2,
            value_head_dim: 3,
        };
        let expected = qwen35_delta_recurrence_arithmetic_witness(&draft_inputs).unwrap();
        let final_recurrent_state = expected.final_recurrent_state.clone();
        let output = expected.output.clone();
        let inputs = Qwen35DeltaRecurrenceArithmeticInputs {
            final_recurrent_state: &final_recurrent_state,
            output: &output,
            ..draft_inputs
        };

        let proof = prove_qwen35_delta_recurrence_arithmetic_air(
            7,
            Qwen35DeltaRecurrenceMode::RecurrentDecode,
            &inputs,
        )
        .unwrap();
        assert_eq!(proof.n_real_rows, 12);
        assert_eq!(
            proof.statement.output_commitment,
            qwen35_delta_recurrence_output_commitment(&output)
        );
        verify_qwen35_delta_recurrence_arithmetic_air(&proof).unwrap();

        let wrong_statement = qwen35_delta_recurrence_arithmetic_statement(
            8,
            Qwen35DeltaRecurrenceMode::RecurrentDecode,
            &inputs,
        )
        .unwrap()
        .statement_hash;
        let err = verify_qwen35_delta_recurrence_arithmetic_air_with_statement_hash(
            &proof,
            wrong_statement,
        )
        .unwrap_err();
        assert!(err.to_string().contains("statement hash mismatch"));

        let mut bad_output = output.clone();
        bad_output.set(0, 5, bad_output.get(0, 5) + M31::from(1u32));
        let bad_inputs = Qwen35DeltaRecurrenceArithmeticInputs {
            output: &bad_output,
            ..inputs
        };
        let err = prove_qwen35_delta_recurrence_arithmetic_air(
            7,
            Qwen35DeltaRecurrenceMode::RecurrentDecode,
            &bad_inputs,
        )
        .unwrap_err();
        assert!(err.to_string().contains("claimed output"));
    }

    #[test]
    fn qwen35_delta_recurrence_trace_binding_witness_binds_order_and_state() {
        let query = matrix(2, 4, 10);
        let key = matrix(2, 4, 20);
        let projected_value = matrix(2, 6, 30);
        let a_gate = matrix(2, 2, 40);
        let b_gate = matrix(2, 2, 50);
        let initial_recurrent_state = matrix(4, 3, 55);
        let final_recurrent_state = matrix(4, 3, 56);
        let output = matrix(2, 6, 60);
        let a_log_weight = vec![M31::from(70u32), M31::from(71u32)];
        let dt_bias = vec![M31::from(80u32), M31::from(81u32)];
        let inputs = Qwen35DeltaRecurrenceInputs {
            query: &query,
            key: &key,
            projected_value: &projected_value,
            a_gate: &a_gate,
            b_gate: &b_gate,
            a_log_weight: &a_log_weight,
            dt_bias: &dt_bias,
            initial_recurrent_state: &initial_recurrent_state,
            final_recurrent_state: &final_recurrent_state,
            output: &output,
            state_rows: 2,
            value_head_dim: 3,
            mode: Qwen35DeltaRecurrenceMode::RecurrentDecode,
        };

        let witness = qwen35_delta_recurrence_trace_binding_witness(&inputs).unwrap();
        verify_qwen35_delta_recurrence_trace_binding_witness(&inputs, &witness).unwrap();
        assert_eq!(witness.rows.len(), 24);
        assert_eq!(witness.rows[0].token_idx, 0);
        assert_eq!(witness.rows[0].state_row_idx, 0);
        assert_eq!(witness.rows[0].qk_col_idx, 0);
        assert_eq!(witness.rows[0].value_col_idx, 0);
        assert_eq!(witness.rows[0].query, query.get(0, 0));
        assert_eq!(
            witness.rows[0].initial_recurrent_state,
            initial_recurrent_state.get(0, 0)
        );
        assert_eq!(
            witness.rows[0].final_recurrent_state,
            final_recurrent_state.get(0, 0)
        );

        let mut swapped = witness.clone();
        swapped.rows.swap(1, 2);
        swapped.witness_hash =
            qwen35_delta_recurrence_trace_binding_witness_hash(&inputs, &swapped.rows).unwrap();
        let swap_err =
            verify_qwen35_delta_recurrence_trace_binding_witness(&inputs, &swapped).unwrap_err();
        assert!(swap_err.contains("expected token"));

        let mut tampered = witness.clone();
        tampered.rows[0].final_recurrent_state += M31::from(1u32);
        tampered.witness_hash =
            qwen35_delta_recurrence_trace_binding_witness_hash(&inputs, &tampered.rows).unwrap();
        let state_err =
            verify_qwen35_delta_recurrence_trace_binding_witness(&inputs, &tampered).unwrap_err();
        assert!(state_err.contains("active tensors/state"));

        let mut hash_tampered = witness.clone();
        hash_tampered.witness_hash += FieldElement::ONE;
        let hash_err =
            verify_qwen35_delta_recurrence_trace_binding_witness(&inputs, &hash_tampered)
                .unwrap_err();
        assert!(hash_err.contains("witness hash mismatch"));
    }

    #[test]
    fn qwen35_delta_recurrence_trace_binding_air_constraints_hold_on_trace() {
        use num_traits::Zero;
        use stwo::core::pcs::TreeVec;
        use stwo_constraint_framework::assert_constraints_on_trace;

        let query = matrix(2, 4, 10);
        let key = matrix(2, 4, 20);
        let projected_value = matrix(2, 6, 30);
        let a_gate = matrix(2, 2, 40);
        let b_gate = matrix(2, 2, 50);
        let initial_recurrent_state = matrix(4, 3, 55);
        let final_recurrent_state = matrix(4, 3, 56);
        let output = matrix(2, 6, 60);
        let a_log_weight = vec![M31::from(70u32), M31::from(71u32)];
        let dt_bias = vec![M31::from(80u32), M31::from(81u32)];
        let inputs = Qwen35DeltaRecurrenceInputs {
            query: &query,
            key: &key,
            projected_value: &projected_value,
            a_gate: &a_gate,
            b_gate: &b_gate,
            a_log_weight: &a_log_weight,
            dt_bias: &dt_bias,
            initial_recurrent_state: &initial_recurrent_state,
            final_recurrent_state: &final_recurrent_state,
            output: &output,
            state_rows: 2,
            value_head_dim: 3,
            mode: Qwen35DeltaRecurrenceMode::RecurrentDecode,
        };
        let trace = qwen35_delta_recurrence_trace_binding_trace(&inputs).unwrap();
        let preprocessed = trace.preprocessed.iter().collect::<Vec<_>>();
        let execution = trace.execution.iter().collect::<Vec<_>>();
        let trees = TreeVec::new(vec![preprocessed, execution]);
        let eval = Qwen35DeltaRecurrenceTraceBindingEval {
            log_n_rows: trace.log_size,
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
    fn qwen35_delta_recurrence_trace_binding_air_rejects_swapped_execution_rows() {
        use num_traits::Zero;
        use stwo::core::pcs::TreeVec;
        use stwo_constraint_framework::assert_constraints_on_trace;

        let query = matrix(2, 4, 10);
        let key = matrix(2, 4, 20);
        let projected_value = matrix(2, 6, 30);
        let a_gate = matrix(2, 2, 40);
        let b_gate = matrix(2, 2, 50);
        let initial_recurrent_state = matrix(4, 3, 55);
        let final_recurrent_state = matrix(4, 3, 56);
        let output = matrix(2, 6, 60);
        let a_log_weight = vec![M31::from(70u32), M31::from(71u32)];
        let dt_bias = vec![M31::from(80u32), M31::from(81u32)];
        let inputs = Qwen35DeltaRecurrenceInputs {
            query: &query,
            key: &key,
            projected_value: &projected_value,
            a_gate: &a_gate,
            b_gate: &b_gate,
            a_log_weight: &a_log_weight,
            dt_bias: &dt_bias,
            initial_recurrent_state: &initial_recurrent_state,
            final_recurrent_state: &final_recurrent_state,
            output: &output,
            state_rows: 2,
            value_head_dim: 3,
            mode: Qwen35DeltaRecurrenceMode::RecurrentDecode,
        };
        let mut trace = qwen35_delta_recurrence_trace_binding_trace(&inputs).unwrap();
        for col in &mut trace.execution {
            col.swap(1, 2);
        }
        let preprocessed = trace.preprocessed.iter().collect::<Vec<_>>();
        let execution = trace.execution.iter().collect::<Vec<_>>();
        let trees = TreeVec::new(vec![preprocessed, execution]);
        let eval = Qwen35DeltaRecurrenceTraceBindingEval {
            log_n_rows: trace.log_size,
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
    fn qwen35_delta_recurrence_trace_binding_air_proves_and_verifies_standalone() {
        let query = matrix(1, 4, 10);
        let key = matrix(1, 4, 20);
        let projected_value = matrix(1, 6, 30);
        let a_gate = matrix(1, 2, 40);
        let b_gate = matrix(1, 2, 50);
        let initial_recurrent_state = matrix(4, 3, 55);
        let final_recurrent_state = matrix(4, 3, 56);
        let output = matrix(1, 6, 60);
        let a_log_weight = vec![M31::from(70u32), M31::from(71u32)];
        let dt_bias = vec![M31::from(80u32), M31::from(81u32)];
        let inputs = Qwen35DeltaRecurrenceInputs {
            query: &query,
            key: &key,
            projected_value: &projected_value,
            a_gate: &a_gate,
            b_gate: &b_gate,
            a_log_weight: &a_log_weight,
            dt_bias: &dt_bias,
            initial_recurrent_state: &initial_recurrent_state,
            final_recurrent_state: &final_recurrent_state,
            output: &output,
            state_rows: 2,
            value_head_dim: 3,
            mode: Qwen35DeltaRecurrenceMode::RecurrentDecode,
        };

        let proof = prove_qwen35_delta_recurrence_trace_binding_air(3, &inputs).unwrap();
        assert_eq!(proof.n_real_rows, 12);
        verify_qwen35_delta_recurrence_trace_binding_air(&proof).unwrap();

        let wrong_statement = qwen35_delta_recurrence_statement(4, &inputs)
            .unwrap()
            .statement_hash;
        let err = verify_qwen35_delta_recurrence_trace_binding_air_with_statement_hash(
            &proof,
            wrong_statement,
        )
        .unwrap_err();
        assert!(err.to_string().contains("statement hash mismatch"));
    }

    #[test]
    fn qwen35_delta_recurrence_air_spec_matches_repeated_qk_contract() {
        let spec = qwen35_delta_recurrence_air_spec(
            128,
            4096,
            4096,
            4096,
            32,
            128,
            Qwen35DeltaRecurrenceMode::ChunkPrefill,
            false,
            true,
        )
        .unwrap();

        assert_eq!(spec.upstream, QWEN35_DELTA_RECURRENCE_UPSTREAM);
        assert_eq!(spec.mode.as_str(), "chunk-prefill");
        assert_eq!(spec.qk_head_dim, 128);
        assert!(spec.uses_qk_l2norm);
        assert!(spec.query_scale_is_inverse_sqrt_head_dim);
        assert!(spec.beta_is_sigmoid_b);
        assert!(spec.decay_is_neg_exp_a_log_times_softplus_a_plus_dt_bias);
        assert!(spec.state_update_is_gated_delta_rule);
        assert!(spec.output_is_pre_norm_attended_value);
    }

    #[test]
    fn qwen35_delta_recurrence_air_spec_rejects_unrepeated_key_width() {
        let err = qwen35_delta_recurrence_air_spec(
            1,
            4096,
            2048,
            4096,
            32,
            128,
            Qwen35DeltaRecurrenceMode::RecurrentDecode,
            true,
            true,
        )
        .unwrap_err();
        assert!(err.contains("matching widths"));
    }
}
