//! Qwen3.5-MoE execution/proof plan.
//!
//! This module is intentionally separate from the legacy flat HuggingFace
//! MatMul graph. Qwen3.5-MoE is a hybrid architecture: most layers use
//! GatedDeltaNet linear attention, every fourth layer uses gated full
//! attention, and every layer has packed routed experts plus a shared expert.
//! A proof over this model must follow that structure.

use crate::compiler::hf_loader::HfConfig;
use crate::components::matmul::M31Matrix;
use starknet_ff::FieldElement;
use std::collections::{HashMap, HashSet};
use stwo::core::fields::m31::M31;

const DOMAIN_QWEN35_CONTRACT: u64 = 0x51333543; // "Q35C"
const DOMAIN_QWEN35_LAYER: u64 = 0x5133354C; // "Q35L"
const DOMAIN_QWEN35_GDN_STAGE: u64 = 0x51334744; // "Q3GD"
const DOMAIN_QWEN35_DEPTHWISE_CONV1D: u64 = 0x51334443; // "Q3DC"
const DOMAIN_QWEN35_DELTA_RECURRENCE: u64 = 0x51334452; // "Q3DR"
const DOMAIN_QWEN35_NORM_AND_Z_GATE: u64 = 0x51334e5a; // "Q3NZ"
const DOMAIN_QWEN35_TRACE_BINDING: u64 = 0x51335442; // "Q3TB"
const DOMAIN_QWEN35_DEPTHWISE_CONV1D_STATEMENT: u64 = 0x513344_53544d; // "Q3D_STM"
const DOMAIN_QWEN35_DELTA_RECURRENCE_STATEMENT: u64 = 0x513352_53544d; // "Q3R_STM"
const DOMAIN_QWEN35_NORM_AND_Z_GATE_STATEMENT: u64 = 0x51334e_53544d; // "Q3N_STM"
const DOMAIN_QWEN35_TYPED_LEDGER: u64 = 0x5133544c; // "Q3TL"
const DOMAIN_QWEN35_DELTA_RECURRENCE_NONLINEAR_AGGREGATE: u64 = 0x513352_4e4147; // "Q3R_NAG"
const DOMAIN_QWEN35_CONVERSATION_STATE_LEDGER: u64 = 0x51334353; // "Q3CS"
const DOMAIN_QWEN35_ACTIVE_TYPED_SPAN_RECEIPT: u64 = 0x513341_545350; // "Q3A_TSP"
const DOMAIN_QWEN35_ACTIVE_CONVERSATION_RECEIPT: u64 = 0x513341_434f4e; // "Q3A_CON"
const DOMAIN_QWEN35_ACTIVE_CONVERSATION_STATEMENT: u64 = 0x513341_435354; // "Q3A_CST"
const DOMAIN_QWEN35_ACTIVE_BATCH_ARTIFACT: u64 = 0x513341_424154; // "Q3A_BAT"
const DOMAIN_QWEN35_RECURRENT_STATE_ROOT: u64 = 0x513353_52544f; // "Q3S_RTO"
const DOMAIN_QWEN35_TYPED_WITNESS_MANIFEST: u64 = 0x5133574d; // "Q3WM"
const KIND_GATED_DELTA_NET: u64 = 1;
const KIND_GATED_FULL_ATTENTION: u64 = 2;

fn qwen35_hash_str(value: &str) -> FieldElement {
    starknet_crypto::poseidon_hash_many(
        &value
            .as_bytes()
            .iter()
            .map(|byte| FieldElement::from(*byte as u64))
            .collect::<Vec<_>>(),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35AttentionKind {
    GatedDeltaNet,
    GatedFullAttention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35ProofObligation {
    InputRmsNorm,
    PostAttentionRmsNorm,
    GatedDeltaNet {
        qkv_rows: usize,
        z_rows: usize,
        state_rows: usize,
        conv_kernel: usize,
    },
    GatedFullAttention {
        q_rows_with_gate: usize,
        q_rows: usize,
        kv_rows: usize,
    },
    RouterTopK {
        num_experts: usize,
        top_k: usize,
    },
    PackedExpertBank {
        num_experts: usize,
        routed_ff: usize,
    },
    SharedExpert {
        shared_ff: usize,
    },
    SharedExpertGate,
    ResidualAdd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35TensorRole {
    TokenEmbedding,
    FinalNorm,
    LmHead,
    InputRmsNorm,
    PostAttentionRmsNorm,
    LinearAttentionALog,
    LinearAttentionDtBias,
    LinearAttentionConv1d,
    LinearAttentionInProjA,
    LinearAttentionInProjB,
    LinearAttentionInProjQkv,
    LinearAttentionInProjZ,
    LinearAttentionNorm,
    LinearAttentionOutProj,
    FullAttentionQProj,
    FullAttentionKProj,
    FullAttentionVProj,
    FullAttentionOProj,
    FullAttentionQNorm,
    FullAttentionKNorm,
    MoeRouter,
    MoePackedGateUp,
    MoePackedDown,
    SharedExpertGateProj,
    SharedExpertUpProj,
    SharedExpertDownProj,
    SharedExpertGate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35TensorContractEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub role: Qwen35TensorRole,
    pub layer_idx: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Qwen35ProofComponent {
    TokenEmbedding,
    InputRmsNorm,
    GatedDeltaNet,
    GatedFullAttention,
    AttentionResidualAdd,
    PostAttentionRmsNorm,
    RouterTopK,
    PackedExpertBank,
    SharedExpert,
    SharedExpertGate,
    MlpResidualAdd,
    FinalNorm,
    LmHead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35ComponentStatus {
    GenericAvailable,
    DedicatedMissing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35StageStatus {
    GenericAvailable,
    DedicatedAirAvailableIntegrationMissing,
    DedicatedMissing,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35ProofStep {
    pub step_idx: usize,
    pub layer_idx: Option<usize>,
    pub component: Qwen35ProofComponent,
    pub status: Qwen35ComponentStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35ExecutionPlan {
    pub contract_hash: FieldElement,
    pub steps: Vec<Qwen35ProofStep>,
}

impl Qwen35ProofComponent {
    pub fn label(self) -> &'static str {
        match self {
            Qwen35ProofComponent::TokenEmbedding => "TokenEmbedding",
            Qwen35ProofComponent::InputRmsNorm => "InputRmsNorm",
            Qwen35ProofComponent::GatedDeltaNet => "GatedDeltaNet",
            Qwen35ProofComponent::GatedFullAttention => "GatedFullAttention",
            Qwen35ProofComponent::AttentionResidualAdd => "AttentionResidualAdd",
            Qwen35ProofComponent::PostAttentionRmsNorm => "PostAttentionRmsNorm",
            Qwen35ProofComponent::RouterTopK => "RouterTopK",
            Qwen35ProofComponent::PackedExpertBank => "PackedExpertBank",
            Qwen35ProofComponent::SharedExpert => "SharedExpert",
            Qwen35ProofComponent::SharedExpertGate => "SharedExpertGate",
            Qwen35ProofComponent::MlpResidualAdd => "MlpResidualAdd",
            Qwen35ProofComponent::FinalNorm => "FinalNorm",
            Qwen35ProofComponent::LmHead => "LmHead",
        }
    }
}

impl Qwen35ExecutionPlan {
    pub fn total_steps(&self) -> usize {
        self.steps.len()
    }

    pub fn count_status(&self, status: Qwen35ComponentStatus) -> usize {
        self.steps
            .iter()
            .filter(|step| step.status == status)
            .count()
    }

    pub fn component_count(&self, component: Qwen35ProofComponent) -> usize {
        self.steps
            .iter()
            .filter(|step| step.component == component)
            .count()
    }

    pub fn missing_component_counts(&self) -> Vec<(Qwen35ProofComponent, usize)> {
        let mut counts = std::collections::BTreeMap::<Qwen35ProofComponent, usize>::new();
        for step in &self.steps {
            if step.status == Qwen35ComponentStatus::DedicatedMissing {
                *counts.entry(step.component).or_insert(0) += 1;
            }
        }
        counts.into_iter().collect()
    }

    pub fn production_ready(&self) -> bool {
        self.count_status(Qwen35ComponentStatus::DedicatedMissing) == 0
    }

    pub fn readiness_summary(&self) -> String {
        let missing = self.missing_component_counts();
        if missing.is_empty() {
            return format!(
                "production-ready typed Qwen3.5 proof plan: {} steps bound to contract_hash=0x{:x}",
                self.total_steps(),
                self.contract_hash,
            );
        }

        let missing_parts = missing
            .iter()
            .map(|(component, count)| format!("{}={count}", component.label()))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "typed Qwen3.5 proof plan is not production-ready: {} total steps, {} generic-ready, {} missing dedicated; missing: {}; contract_hash=0x{:x}",
            self.total_steps(),
            self.count_status(Qwen35ComponentStatus::GenericAvailable),
            self.count_status(Qwen35ComponentStatus::DedicatedMissing),
            missing_parts,
            self.contract_hash,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35LayerPlan {
    pub layer_idx: usize,
    pub attention: Qwen35AttentionKind,
    pub obligations: Vec<Qwen35ProofObligation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35Tensor2DShape {
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35Tensor3DShape {
    pub outer: usize,
    pub middle: usize,
    pub inner: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35GatedFullAttentionContract {
    pub layer_idx: usize,
    pub seq_len: usize,
    pub input: Qwen35Tensor2DShape,
    pub q_proj_weight: Qwen35Tensor2DShape,
    pub q_proj_output: Qwen35Tensor2DShape,
    pub query: Qwen35Tensor2DShape,
    pub output_gate: Qwen35Tensor2DShape,
    pub query_heads: Qwen35Tensor3DShape,
    pub q_norm_weight: usize,
    pub k_proj_weight: Qwen35Tensor2DShape,
    pub v_proj_weight: Qwen35Tensor2DShape,
    pub key: Qwen35Tensor2DShape,
    pub value: Qwen35Tensor2DShape,
    pub key_heads: Qwen35Tensor3DShape,
    pub value_heads: Qwen35Tensor3DShape,
    pub k_norm_weight: usize,
    pub query_groups_per_kv_head: usize,
    pub attention_context: Qwen35Tensor2DShape,
    pub gated_context: Qwen35Tensor2DShape,
    pub o_proj_weight: Qwen35Tensor2DShape,
    pub output: Qwen35Tensor2DShape,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35MoeContract {
    pub layer_idx: usize,
    pub seq_len: usize,
    pub input: Qwen35Tensor2DShape,
    pub router_weight: Qwen35Tensor2DShape,
    pub router_logits: Qwen35Tensor2DShape,
    pub selected_expert_ids: Qwen35Tensor2DShape,
    pub routing_weights: Qwen35Tensor2DShape,
    pub packed_gate_up_weight: Qwen35Tensor3DShape,
    pub expert_gate: Qwen35Tensor3DShape,
    pub expert_up: Qwen35Tensor3DShape,
    pub expert_hidden: Qwen35Tensor3DShape,
    pub packed_down_weight: Qwen35Tensor3DShape,
    pub expert_output: Qwen35Tensor3DShape,
    pub routed_output: Qwen35Tensor2DShape,
    pub shared_gate_weight: Qwen35Tensor2DShape,
    pub shared_up_weight: Qwen35Tensor2DShape,
    pub shared_down_weight: Qwen35Tensor2DShape,
    pub shared_expert_gate_weight: Qwen35Tensor2DShape,
    pub shared_hidden: Qwen35Tensor2DShape,
    pub shared_output: Qwen35Tensor2DShape,
    pub output: Qwen35Tensor2DShape,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35GatedDeltaNetContract {
    pub layer_idx: usize,
    pub seq_len: usize,
    pub input: Qwen35Tensor2DShape,
    pub in_proj_qkv_weight: Qwen35Tensor2DShape,
    pub qkv_projected: Qwen35Tensor2DShape,
    pub conv1d_weight: Qwen35Tensor3DShape,
    pub qkv_after_conv: Qwen35Tensor2DShape,
    pub query: Qwen35Tensor2DShape,
    pub key: Qwen35Tensor2DShape,
    pub projected_value: Qwen35Tensor2DShape,
    pub in_proj_z_weight: Qwen35Tensor2DShape,
    pub z_gate: Qwen35Tensor2DShape,
    pub in_proj_a_weight: Qwen35Tensor2DShape,
    pub in_proj_b_weight: Qwen35Tensor2DShape,
    pub a_gate: Qwen35Tensor2DShape,
    pub b_gate: Qwen35Tensor2DShape,
    pub a_log_weight: usize,
    pub dt_bias: usize,
    pub norm_weight: usize,
    pub linear_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub recurrent_state_rows: usize,
    pub attended_value: Qwen35Tensor2DShape,
    pub o_proj_weight: Qwen35Tensor2DShape,
    pub output: Qwen35Tensor2DShape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35TensorShape {
    Vector(usize),
    Matrix(Qwen35Tensor2DShape),
    Tensor3D(Qwen35Tensor3DShape),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35StageTensor {
    pub name: &'static str,
    pub shape: Qwen35TensorShape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Qwen35GatedDeltaNetStageKind {
    QkvProjection,
    DepthwiseConv1d,
    QkvSplit,
    ZProjection,
    ABProjection,
    DeltaRecurrence,
    NormAndZGate,
    OutputProjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35TraceRootRole {
    ProducerActivation,
    ModelWeight,
    ConsumerActivation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35GatedDeltaNetStageContract {
    pub stage_idx: usize,
    pub kind: Qwen35GatedDeltaNetStageKind,
    pub status: Qwen35StageStatus,
    pub relation: &'static str,
    pub inputs: Vec<Qwen35StageTensor>,
    pub outputs: Vec<Qwen35StageTensor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35TraceRootContract {
    pub name: &'static str,
    pub role: Qwen35TraceRootRole,
    pub shape: Qwen35TensorShape,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DepthwiseConv1dTraceBindingContract {
    pub layer_idx: usize,
    pub seq_len: usize,
    pub channels: usize,
    pub kernel: usize,
    pub stage_idx: usize,
    pub producer_stage_idx: usize,
    pub consumer_stage_idx: usize,
    pub input_root: Qwen35TraceRootContract,
    pub weight_root: Qwen35TraceRootContract,
    pub output_root: Qwen35TraceRootContract,
    pub stage_contract_hash: FieldElement,
    pub air_contract_hash: FieldElement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DeltaRecurrenceTraceBindingContract {
    pub layer_idx: usize,
    pub seq_len: usize,
    pub query_width: usize,
    pub key_width: usize,
    pub value_width: usize,
    pub state_rows: usize,
    pub value_head_dim: usize,
    pub stage_idx: usize,
    pub qkv_split_stage_idx: usize,
    pub ab_projection_stage_idx: usize,
    pub consumer_stage_idx: usize,
    pub query_root: Qwen35TraceRootContract,
    pub key_root: Qwen35TraceRootContract,
    pub projected_value_root: Qwen35TraceRootContract,
    pub a_gate_root: Qwen35TraceRootContract,
    pub b_gate_root: Qwen35TraceRootContract,
    pub a_log_weight_root: Qwen35TraceRootContract,
    pub dt_bias_root: Qwen35TraceRootContract,
    pub output_root: Qwen35TraceRootContract,
    pub stage_contract_hash: FieldElement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35NormAndZGateTraceBindingContract {
    pub layer_idx: usize,
    pub seq_len: usize,
    pub width: usize,
    pub norm_width: usize,
    pub stage_idx: usize,
    pub delta_recurrence_stage_idx: usize,
    pub z_projection_stage_idx: usize,
    pub consumer_stage_idx: usize,
    pub attended_value_root: Qwen35TraceRootContract,
    pub norm_weight_root: Qwen35TraceRootContract,
    pub z_gate_root: Qwen35TraceRootContract,
    pub output_root: Qwen35TraceRootContract,
    pub stage_contract_hash: FieldElement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Qwen35TypedWitnessRootKind {
    Activation,
    ModelWeight,
    RecurrentState,
    LookupTable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35TypedWitnessRoot {
    pub layer_idx: usize,
    pub statement_kind: Qwen35TypedProofStatementKind,
    pub stage_idx: usize,
    pub name: String,
    pub kind: Qwen35TypedWitnessRootKind,
    pub shape: Qwen35TensorShape,
    pub source: String,
    pub trace_root_contract_hash: FieldElement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35TypedWitnessLayerManifest {
    pub layer_idx: usize,
    pub seq_len: usize,
    pub depthwise_conv1d_trace_binding_hash: FieldElement,
    pub delta_recurrence_trace_binding_hash: FieldElement,
    pub norm_and_z_gate_trace_binding_hash: FieldElement,
    pub roots: Vec<Qwen35TypedWitnessRoot>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35TypedWitnessManifest {
    pub architecture_contract_hash: FieldElement,
    pub seq_len: usize,
    pub layers: Vec<Qwen35TypedWitnessLayerManifest>,
    pub manifest_hash: FieldElement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35TypedWitnessCommitment {
    pub root_hash: FieldElement,
    pub commitment: FieldElement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35TypedWitnessCommitmentSet {
    pub manifest_hash: FieldElement,
    pub seq_len: usize,
    pub commitments: Vec<Qwen35TypedWitnessCommitment>,
    pub commitment_set_hash: FieldElement,
}

#[derive(Debug, Clone)]
pub enum Qwen35TypedWitnessCapturedValue {
    Matrix(M31Matrix),
    Vector(Vec<M31>),
    LookupTable { table_log_size: u32 },
}

#[derive(Debug, Clone)]
pub struct Qwen35TypedWitnessCapturedRoot {
    pub root_hash: FieldElement,
    pub value: Qwen35TypedWitnessCapturedValue,
}

#[derive(Debug, Clone, Default)]
pub struct Qwen35TypedWitnessCapture {
    pub roots: Vec<Qwen35TypedWitnessCapturedRoot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Qwen35TypedWitnessSourceKind {
    Runtime,
    Safetensors,
    ConversationState,
    Statement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35TypedWitnessSourceRequirement {
    pub source: String,
    pub source_kind: Qwen35TypedWitnessSourceKind,
    pub root_hashes: Vec<FieldElement>,
    pub layer_indices: Vec<usize>,
    pub shape: Qwen35TensorShape,
}

#[derive(Debug, Clone)]
pub struct Qwen35TypedWitnessCapturedSource {
    pub source: String,
    pub value: Qwen35TypedWitnessCapturedValue,
}

#[derive(Debug, Clone, Default)]
pub struct Qwen35TypedWitnessSourceCapture {
    pub sources: Vec<Qwen35TypedWitnessCapturedSource>,
}

#[derive(Debug, Clone, Default)]
pub struct Qwen35TypedWitnessSourceInventory {
    pub runtime: HashMap<String, Qwen35TypedWitnessCapturedValue>,
    pub safetensors: HashMap<String, Qwen35TypedWitnessCapturedValue>,
    pub conversation_state: HashMap<String, Qwen35TypedWitnessCapturedValue>,
    pub statement: HashMap<String, Qwen35TypedWitnessCapturedValue>,
}

#[derive(Debug, Clone)]
pub struct Qwen35TypedWitnessInventoryRecorder {
    manifest: Qwen35TypedWitnessManifest,
    inventory: Qwen35TypedWitnessSourceInventory,
}

#[derive(Debug, Clone)]
pub struct Qwen35GatedDeltaNetRuntimeLayerTrace {
    pub layer_idx: usize,
    pub qkv_projected: M31Matrix,
    pub qkv_after_conv: M31Matrix,
    pub query: M31Matrix,
    pub key: M31Matrix,
    pub projected_value: M31Matrix,
    pub a_gate: M31Matrix,
    pub b_gate: M31Matrix,
    pub attended_value: M31Matrix,
    pub z_gate: M31Matrix,
    pub gated_value: M31Matrix,
    pub initial_recurrent_state: M31Matrix,
    pub final_recurrent_state: M31Matrix,
    pub delta_recurrence_transform: Option<Qwen35DeltaRecurrenceRuntimeTransformTrace>,
    pub norm_and_z_gate_rsqrt_table_log_size: u32,
}

#[derive(Debug, Clone)]
pub struct Qwen35DeltaRecurrenceRuntimeTransformTrace {
    pub scaled_query: M31Matrix,
    pub normalized_key: M31Matrix,
    pub decay: M31Matrix,
    pub beta: M31Matrix,
    pub q_norm_table_log_size: u32,
    pub k_norm_table_log_size: u32,
    pub beta_sigmoid_table_log_size: u32,
    pub decay_table_log_size: u32,
    pub query_norm_post_scale: M31,
    pub key_norm_post_scale: M31,
}

#[derive(Debug, Clone)]
pub struct Qwen35TypedRuntimeTrace {
    pub seq_len: usize,
    pub layers: Vec<Qwen35GatedDeltaNetRuntimeLayerTrace>,
}

#[derive(Debug, Clone, Default)]
pub struct Qwen35TypedRuntimeStatementSet {
    pub depthwise_conv1d:
        Vec<crate::components::qwen35_depthwise_conv1d::Qwen35DepthwiseConv1dStatement>,
    pub delta_recurrence:
        Vec<crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceStatement>,
    pub delta_recurrence_arithmetic:
        Vec<crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceArithmeticStatement>,
    pub delta_recurrence_transform:
        Vec<crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceTransformStatement>,
    pub delta_recurrence_q_norm:
        Vec<crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceNormStatement>,
    pub delta_recurrence_k_norm:
        Vec<crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceNormStatement>,
    pub delta_recurrence_beta_sigmoid:
        Vec<crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceBetaSigmoidStatement>,
    pub delta_recurrence_decay:
        Vec<crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceDecayStatement>,
    pub norm_and_z_gate:
        Vec<crate::components::qwen35_norm_and_z_gate::Qwen35NormAndZGateStatement>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35TypedProofStatementKind {
    DepthwiseConv1d,
    DeltaRecurrence,
    NormAndZGate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35TypedProofStatement {
    pub kind: Qwen35TypedProofStatementKind,
    pub layer_idx: usize,
    pub seq_len: usize,
    pub stage_idx: usize,
    pub trace_binding_hash: FieldElement,
    pub statement_hash: FieldElement,
    pub transform_statement_hash: Option<FieldElement>,
    pub transform_nonlinear_statement_hash: Option<FieldElement>,
    pub transform_nonlinear_q_norm_statement_hash: Option<FieldElement>,
    pub transform_nonlinear_k_norm_statement_hash: Option<FieldElement>,
    pub transform_nonlinear_beta_sigmoid_statement_hash: Option<FieldElement>,
    pub transform_nonlinear_decay_statement_hash: Option<FieldElement>,
    pub arithmetic_statement_hash: Option<FieldElement>,
    pub initial_recurrent_state_commitment: Option<FieldElement>,
    pub final_recurrent_state_commitment: Option<FieldElement>,
}

impl Qwen35TypedProofStatement {
    pub fn delta_recurrence_nonlinear_aggregate_hash(&self) -> Option<FieldElement> {
        if self.kind != Qwen35TypedProofStatementKind::DeltaRecurrence {
            return None;
        }
        Some(starknet_crypto::poseidon_hash_many(&[
            FieldElement::from(DOMAIN_QWEN35_DELTA_RECURRENCE_NONLINEAR_AGGREGATE),
            FieldElement::from(self.layer_idx as u64),
            FieldElement::from(self.seq_len as u64),
            FieldElement::from(self.stage_idx as u64),
            self.statement_hash,
            self.transform_statement_hash?,
            self.arithmetic_statement_hash?,
            self.transform_nonlinear_q_norm_statement_hash?,
            self.transform_nonlinear_k_norm_statement_hash?,
            self.transform_nonlinear_beta_sigmoid_statement_hash?,
            self.transform_nonlinear_decay_statement_hash?,
        ]))
    }

    pub fn has_valid_delta_recurrence_coverage(&self) -> bool {
        self.kind == Qwen35TypedProofStatementKind::DeltaRecurrence
            && self.transform_nonlinear_statement_hash.is_some()
            && self.transform_nonlinear_statement_hash
                == self.delta_recurrence_nonlinear_aggregate_hash()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35TypedProofLedger {
    pub architecture_contract_hash: FieldElement,
    pub seq_len: usize,
    pub expected_depthwise_conv1d_statements: usize,
    pub expected_delta_recurrence_statements: usize,
    pub expected_norm_and_z_gate_statements: usize,
    pub statements: Vec<Qwen35TypedProofStatement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35LayerRecurrentStateCommitment {
    pub layer_idx: usize,
    pub stage_idx: usize,
    pub initial_recurrent_state_commitment: FieldElement,
    pub final_recurrent_state_commitment: FieldElement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35ConversationStateSpan {
    pub span_idx: usize,
    pub seq_len: usize,
    pub typed_ledger_hash: FieldElement,
    pub layer_states: Vec<Qwen35LayerRecurrentStateCommitment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35ConversationStateLedger {
    pub architecture_contract_hash: FieldElement,
    pub expected_delta_recurrence_layers: usize,
    pub spans: Vec<Qwen35ConversationStateSpan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35ActiveTypedSpanReceipt {
    pub span_idx: usize,
    pub witness_commitment_set: Qwen35TypedWitnessCommitmentSet,
    pub typed_ledger: Qwen35TypedProofLedger,
    pub conversation_span: Qwen35ConversationStateSpan,
    pub receipt_hash: FieldElement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35ActiveConversationReceipt {
    pub architecture_contract_hash: FieldElement,
    pub span_receipts: Vec<Qwen35ActiveTypedSpanReceipt>,
    pub conversation_ledger: Qwen35ConversationStateLedger,
    pub receipt_hash: FieldElement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35ActiveConversationStatement {
    pub architecture_contract_hash: FieldElement,
    pub weight_super_root: FieldElement,
    pub receipt_hash: FieldElement,
    pub conversation: crate::conversation_statement::ConversationTraceStatement,
    pub steps: Vec<crate::conversation_statement::GenerationStepStatement>,
    pub span_receipt_hashes: Vec<FieldElement>,
    pub actions: Vec<crate::conversation_statement::ConversationActionStatement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35ActiveConversationBatchArtifact {
    pub canonical_statement: crate::conversation_statement::ConversationBatchStatement,
    pub active_statement_root: FieldElement,
    pub active_receipt_root: FieldElement,
    pub artifact_hash: FieldElement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35GatedDeltaNetStageReadiness {
    pub total_stages: usize,
    pub generic_ready_stages: usize,
    pub dedicated_air_available_stages: usize,
    pub missing_dedicated_stages: usize,
    pub dedicated_air_stage_counts: Vec<(Qwen35GatedDeltaNetStageKind, usize)>,
    pub missing_stage_counts: Vec<(Qwen35GatedDeltaNetStageKind, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DepthwiseConv1dAirContract {
    pub layer_idx: usize,
    pub seq_len: usize,
    pub channels: usize,
    pub kernel: usize,
    pub input: Qwen35Tensor2DShape,
    pub weight: Qwen35Tensor3DShape,
    pub output: Qwen35Tensor2DShape,
    pub tap_offsets: Vec<isize>,
    pub logical_trace_rows: usize,
    pub deterministic_columns: usize,
    pub witness_columns: usize,
    pub total_columns: usize,
    pub arithmetic_constraints_per_row: usize,
    pub row_binding_constraints_per_row: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DepthwiseConv1dAirReadiness {
    pub seq_len: usize,
    pub layers: usize,
    pub channels_per_layer: usize,
    pub kernel: usize,
    pub logical_rows_per_layer: usize,
    pub total_logical_rows: usize,
    pub columns_per_layer: usize,
    pub arithmetic_constraints_per_row: usize,
    pub row_binding_constraints_per_row: usize,
    pub aggregate_contract_hash: FieldElement,
    pub aggregate_trace_binding_hash: FieldElement,
}

impl Qwen35TensorShape {
    fn hash_tag(self) -> u64 {
        match self {
            Qwen35TensorShape::Vector(_) => 1,
            Qwen35TensorShape::Matrix(_) => 2,
            Qwen35TensorShape::Tensor3D(_) => 3,
        }
    }

    fn dimensions(self) -> [usize; 3] {
        match self {
            Qwen35TensorShape::Vector(n) => [n, 0, 0],
            Qwen35TensorShape::Matrix(shape) => [shape.rows, shape.cols, 0],
            Qwen35TensorShape::Tensor3D(shape) => [shape.outer, shape.middle, shape.inner],
        }
    }
}

impl Qwen35GatedDeltaNetStageKind {
    pub fn label(self) -> &'static str {
        match self {
            Qwen35GatedDeltaNetStageKind::QkvProjection => "QkvProjection",
            Qwen35GatedDeltaNetStageKind::DepthwiseConv1d => "DepthwiseConv1d",
            Qwen35GatedDeltaNetStageKind::QkvSplit => "QkvSplit",
            Qwen35GatedDeltaNetStageKind::ZProjection => "ZProjection",
            Qwen35GatedDeltaNetStageKind::ABProjection => "ABProjection",
            Qwen35GatedDeltaNetStageKind::DeltaRecurrence => "DeltaRecurrence",
            Qwen35GatedDeltaNetStageKind::NormAndZGate => "NormAndZGate",
            Qwen35GatedDeltaNetStageKind::OutputProjection => "OutputProjection",
        }
    }

    fn hash_tag(self) -> u64 {
        match self {
            Qwen35GatedDeltaNetStageKind::QkvProjection => 1,
            Qwen35GatedDeltaNetStageKind::DepthwiseConv1d => 2,
            Qwen35GatedDeltaNetStageKind::QkvSplit => 3,
            Qwen35GatedDeltaNetStageKind::ZProjection => 4,
            Qwen35GatedDeltaNetStageKind::ABProjection => 5,
            Qwen35GatedDeltaNetStageKind::DeltaRecurrence => 6,
            Qwen35GatedDeltaNetStageKind::NormAndZGate => 7,
            Qwen35GatedDeltaNetStageKind::OutputProjection => 8,
        }
    }
}

impl Qwen35TraceRootRole {
    pub fn label(self) -> &'static str {
        match self {
            Qwen35TraceRootRole::ProducerActivation => "producer-activation",
            Qwen35TraceRootRole::ModelWeight => "model-weight",
            Qwen35TraceRootRole::ConsumerActivation => "consumer-activation",
        }
    }

    fn hash_tag(self) -> u64 {
        match self {
            Qwen35TraceRootRole::ProducerActivation => 1,
            Qwen35TraceRootRole::ModelWeight => 2,
            Qwen35TraceRootRole::ConsumerActivation => 3,
        }
    }
}

impl Qwen35TypedProofStatementKind {
    pub fn label(self) -> &'static str {
        match self {
            Qwen35TypedProofStatementKind::DepthwiseConv1d => "DepthwiseConv1d",
            Qwen35TypedProofStatementKind::DeltaRecurrence => "DeltaRecurrence",
            Qwen35TypedProofStatementKind::NormAndZGate => "NormAndZGate",
        }
    }

    fn hash_tag(self) -> u64 {
        match self {
            Qwen35TypedProofStatementKind::DepthwiseConv1d => 1,
            Qwen35TypedProofStatementKind::DeltaRecurrence => 2,
            Qwen35TypedProofStatementKind::NormAndZGate => 3,
        }
    }
}

impl Qwen35StageStatus {
    pub fn label(self) -> &'static str {
        match self {
            Qwen35StageStatus::GenericAvailable => "generic-ready",
            Qwen35StageStatus::DedicatedAirAvailableIntegrationMissing => {
                "dedicated-air-available-integration-missing"
            }
            Qwen35StageStatus::DedicatedMissing => "dedicated-missing",
        }
    }

    fn hash_tag(self) -> u64 {
        match self {
            Qwen35StageStatus::GenericAvailable => 1,
            Qwen35StageStatus::DedicatedAirAvailableIntegrationMissing => 2,
            Qwen35StageStatus::DedicatedMissing => 3,
        }
    }
}

impl Qwen35TraceRootContract {
    fn contract_hash(&self) -> FieldElement {
        let dims = self.shape.dimensions();
        starknet_crypto::poseidon_hash_many(&[
            FieldElement::from(DOMAIN_QWEN35_TRACE_BINDING),
            FieldElement::from(self.role.hash_tag()),
            FieldElement::from(self.shape.hash_tag()),
            FieldElement::from(dims[0] as u64),
            FieldElement::from(dims[1] as u64),
            FieldElement::from(dims[2] as u64),
            qwen35_hash_str(self.name),
        ])
    }
}

impl Qwen35TypedWitnessRootKind {
    pub fn label(self) -> &'static str {
        match self {
            Qwen35TypedWitnessRootKind::Activation => "activation",
            Qwen35TypedWitnessRootKind::ModelWeight => "model-weight",
            Qwen35TypedWitnessRootKind::RecurrentState => "recurrent-state",
            Qwen35TypedWitnessRootKind::LookupTable => "lookup-table",
        }
    }

    fn hash_tag(self) -> u64 {
        match self {
            Qwen35TypedWitnessRootKind::Activation => 1,
            Qwen35TypedWitnessRootKind::ModelWeight => 2,
            Qwen35TypedWitnessRootKind::RecurrentState => 3,
            Qwen35TypedWitnessRootKind::LookupTable => 4,
        }
    }
}

impl Qwen35TypedWitnessSourceKind {
    pub fn label(self) -> &'static str {
        match self {
            Qwen35TypedWitnessSourceKind::Runtime => "runtime",
            Qwen35TypedWitnessSourceKind::Safetensors => "safetensors",
            Qwen35TypedWitnessSourceKind::ConversationState => "conversation-state",
            Qwen35TypedWitnessSourceKind::Statement => "statement",
        }
    }
}

impl Qwen35TypedWitnessRoot {
    pub fn source_kind(&self) -> Result<Qwen35TypedWitnessSourceKind, String> {
        qwen35_typed_witness_source_kind(&self.source)
    }

    pub fn root_hash(&self) -> FieldElement {
        let dims = self.shape.dimensions();
        starknet_crypto::poseidon_hash_many(&[
            FieldElement::from(DOMAIN_QWEN35_TYPED_WITNESS_MANIFEST),
            FieldElement::from(self.layer_idx as u64),
            FieldElement::from(self.statement_kind.hash_tag()),
            FieldElement::from(self.stage_idx as u64),
            FieldElement::from(self.kind.hash_tag()),
            FieldElement::from(self.shape.hash_tag()),
            FieldElement::from(dims[0] as u64),
            FieldElement::from(dims[1] as u64),
            FieldElement::from(dims[2] as u64),
            qwen35_hash_str(&self.name),
            qwen35_hash_str(&self.source),
            self.trace_root_contract_hash,
        ])
    }

    pub fn commitment_from_capture(
        &self,
        value: &Qwen35TypedWitnessCapturedValue,
    ) -> Result<FieldElement, String> {
        qwen35_validate_captured_value_shape(self, value)?;
        match (self.statement_kind, self.name.as_str(), value) {
            (
                Qwen35TypedProofStatementKind::DepthwiseConv1d,
                "qkv_projected",
                Qwen35TypedWitnessCapturedValue::Matrix(matrix),
            ) => Ok(crate::components::qwen35_depthwise_conv1d::qwen35_depthwise_conv1d_input_commitment(matrix)),
            (
                Qwen35TypedProofStatementKind::DepthwiseConv1d,
                "conv1d_weight",
                Qwen35TypedWitnessCapturedValue::Vector(weights),
            ) => {
                let Qwen35TensorShape::Tensor3D(shape) = self.shape else {
                    return Err("DepthwiseConv1D conv1d_weight root is not Tensor3D".to_string());
                };
                crate::components::qwen35_depthwise_conv1d::qwen35_depthwise_conv1d_weight_commitment(
                    weights,
                    shape.outer,
                    shape.inner,
                )
            }
            (
                Qwen35TypedProofStatementKind::DepthwiseConv1d,
                "qkv_after_conv",
                Qwen35TypedWitnessCapturedValue::Matrix(matrix),
            ) => Ok(crate::components::qwen35_depthwise_conv1d::qwen35_depthwise_conv1d_output_commitment(matrix)),
            (
                Qwen35TypedProofStatementKind::DeltaRecurrence,
                "query",
                Qwen35TypedWitnessCapturedValue::Matrix(matrix),
            ) => Ok(crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_query_commitment(matrix)),
            (
                Qwen35TypedProofStatementKind::DeltaRecurrence,
                "key",
                Qwen35TypedWitnessCapturedValue::Matrix(matrix),
            ) => Ok(crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_key_commitment(matrix)),
            (
                Qwen35TypedProofStatementKind::DeltaRecurrence,
                "projected_value",
                Qwen35TypedWitnessCapturedValue::Matrix(matrix),
            ) => Ok(crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_projected_value_commitment(matrix)),
            (
                Qwen35TypedProofStatementKind::DeltaRecurrence,
                "a_gate",
                Qwen35TypedWitnessCapturedValue::Matrix(matrix),
            ) => Ok(crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_a_gate_commitment(matrix)),
            (
                Qwen35TypedProofStatementKind::DeltaRecurrence,
                "b_gate",
                Qwen35TypedWitnessCapturedValue::Matrix(matrix),
            ) => Ok(crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_b_gate_commitment(matrix)),
            (
                Qwen35TypedProofStatementKind::DeltaRecurrence,
                "a_log_weight",
                Qwen35TypedWitnessCapturedValue::Vector(values),
            ) => Ok(crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_a_log_weight_commitment(values)),
            (
                Qwen35TypedProofStatementKind::DeltaRecurrence,
                "dt_bias",
                Qwen35TypedWitnessCapturedValue::Vector(values),
            ) => Ok(crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_dt_bias_commitment(values)),
            (
                Qwen35TypedProofStatementKind::DeltaRecurrence,
                "initial_recurrent_state",
                Qwen35TypedWitnessCapturedValue::Matrix(matrix),
            ) => Ok(crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_initial_state_commitment(matrix)),
            (
                Qwen35TypedProofStatementKind::DeltaRecurrence,
                "final_recurrent_state",
                Qwen35TypedWitnessCapturedValue::Matrix(matrix),
            ) => Ok(crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_final_state_commitment(matrix)),
            (
                Qwen35TypedProofStatementKind::DeltaRecurrence,
                "attended_value",
                Qwen35TypedWitnessCapturedValue::Matrix(matrix),
            ) => Ok(crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_output_commitment(matrix)),
            (
                Qwen35TypedProofStatementKind::NormAndZGate,
                "attended_value",
                Qwen35TypedWitnessCapturedValue::Matrix(matrix),
            ) => Ok(crate::components::qwen35_norm_and_z_gate::qwen35_norm_and_z_gate_attended_value_commitment(matrix)),
            (
                Qwen35TypedProofStatementKind::NormAndZGate,
                "norm_weight",
                Qwen35TypedWitnessCapturedValue::Vector(values),
            ) => Ok(crate::components::qwen35_norm_and_z_gate::qwen35_norm_and_z_gate_norm_weight_commitment(values)),
            (
                Qwen35TypedProofStatementKind::NormAndZGate,
                "z_gate",
                Qwen35TypedWitnessCapturedValue::Matrix(matrix),
            ) => Ok(crate::components::qwen35_norm_and_z_gate::qwen35_norm_and_z_gate_z_gate_commitment(matrix)),
            (
                Qwen35TypedProofStatementKind::NormAndZGate,
                "gated_value",
                Qwen35TypedWitnessCapturedValue::Matrix(matrix),
            ) => Ok(crate::components::qwen35_norm_and_z_gate::qwen35_norm_and_z_gate_output_commitment(matrix)),
            (
                Qwen35TypedProofStatementKind::NormAndZGate,
                "rsqrt_table_commitment",
                Qwen35TypedWitnessCapturedValue::LookupTable { table_log_size },
            ) => Ok(crate::components::qwen35_norm_and_z_gate::qwen35_norm_and_z_gate_table_commitment(*table_log_size)),
            _ => Err(format!(
                "typed witness root {} ({}) cannot be committed from provided captured value",
                self.name,
                self.statement_kind.label()
            )),
        }
    }
}

impl Qwen35TypedWitnessLayerManifest {
    pub fn root_count_by_kind(&self, kind: Qwen35TypedWitnessRootKind) -> usize {
        self.roots.iter().filter(|root| root.kind == kind).count()
    }

    pub fn layer_hash(&self) -> FieldElement {
        let mut felts = vec![
            FieldElement::from(DOMAIN_QWEN35_TYPED_WITNESS_MANIFEST),
            FieldElement::from(self.layer_idx as u64),
            FieldElement::from(self.seq_len as u64),
            self.depthwise_conv1d_trace_binding_hash,
            self.delta_recurrence_trace_binding_hash,
            self.norm_and_z_gate_trace_binding_hash,
            FieldElement::from(self.roots.len() as u64),
        ];
        for root in &self.roots {
            felts.push(root.root_hash());
        }
        starknet_crypto::poseidon_hash_many(&felts)
    }
}

impl Qwen35TypedWitnessManifest {
    pub fn root_count_by_kind(&self, kind: Qwen35TypedWitnessRootKind) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.root_count_by_kind(kind))
            .sum()
    }

    pub fn total_roots(&self) -> usize {
        self.layers.iter().map(|layer| layer.roots.len()).sum()
    }

    pub fn required_root_hashes(&self) -> Vec<FieldElement> {
        self.layers
            .iter()
            .flat_map(|layer| layer.roots.iter().map(|root| root.root_hash()))
            .collect()
    }

    pub fn source_requirements(&self) -> Result<Vec<Qwen35TypedWitnessSourceRequirement>, String> {
        let mut requirements = Vec::new();
        let mut source_indices: HashMap<&str, usize> = HashMap::new();
        for root in self.layers.iter().flat_map(|layer| layer.roots.iter()) {
            let root_hash = root.root_hash();
            if let Some(existing_idx) = source_indices.get(root.source.as_str()).copied() {
                let requirement: &mut Qwen35TypedWitnessSourceRequirement =
                    &mut requirements[existing_idx];
                if requirement.shape != root.shape {
                    return Err(format!(
                        "typed witness source {} has conflicting shapes {:?} and {:?}",
                        root.source, requirement.shape, root.shape
                    ));
                }
                requirement.root_hashes.push(root_hash);
                if !requirement.layer_indices.contains(&root.layer_idx) {
                    requirement.layer_indices.push(root.layer_idx);
                }
                continue;
            }

            source_indices.insert(root.source.as_str(), requirements.len());
            requirements.push(Qwen35TypedWitnessSourceRequirement {
                source: root.source.clone(),
                source_kind: root.source_kind()?,
                root_hashes: vec![root_hash],
                layer_indices: vec![root.layer_idx],
                shape: root.shape,
            });
        }
        Ok(requirements)
    }

    pub fn build_commitment_set(
        &self,
        commitments: &[(FieldElement, FieldElement)],
    ) -> Result<Qwen35TypedWitnessCommitmentSet, String> {
        let required = self.required_root_hashes();
        let mut seen = HashSet::new();
        for (root_hash, commitment) in commitments {
            if *commitment == FieldElement::ZERO {
                return Err(format!(
                    "typed witness root 0x{root_hash:x} has zero commitment"
                ));
            }
            if !seen.insert(*root_hash) {
                return Err(format!(
                    "duplicate typed witness commitment for root 0x{root_hash:x}"
                ));
            }
            if !required.contains(root_hash) {
                return Err(format!(
                    "typed witness commitment root 0x{root_hash:x} is not required by manifest 0x{:x}",
                    self.manifest_hash
                ));
            }
        }

        let mut ordered = Vec::with_capacity(required.len());
        for root_hash in &required {
            let Some((_, commitment)) = commitments
                .iter()
                .find(|(candidate, _)| candidate == root_hash)
            else {
                return Err(format!(
                    "missing typed witness commitment for root 0x{root_hash:x}"
                ));
            };
            ordered.push(Qwen35TypedWitnessCommitment {
                root_hash: *root_hash,
                commitment: *commitment,
            });
        }

        let commitment_set_hash = Qwen35TypedWitnessCommitmentSet::compute_hash(
            self.manifest_hash,
            self.seq_len,
            &ordered,
        );
        Ok(Qwen35TypedWitnessCommitmentSet {
            manifest_hash: self.manifest_hash,
            seq_len: self.seq_len,
            commitments: ordered,
            commitment_set_hash,
        })
    }

    pub fn build_commitment_set_from_capture(
        &self,
        capture: &Qwen35TypedWitnessCapture,
    ) -> Result<Qwen35TypedWitnessCommitmentSet, String> {
        let captured = capture.root_map()?;
        let mut commitments = Vec::with_capacity(self.total_roots());
        let required_roots = self
            .layers
            .iter()
            .flat_map(|layer| layer.roots.iter())
            .collect::<Vec<_>>();

        for root in required_roots {
            let root_hash = root.root_hash();
            let value = captured.get(&root_hash).ok_or_else(|| {
                format!(
                    "missing captured value for typed witness root {} 0x{root_hash:x}",
                    root.name
                )
            })?;
            commitments.push((root_hash, root.commitment_from_capture(value)?));
        }

        if captured.len() != commitments.len() {
            for extra_root_hash in captured.keys() {
                if !commitments
                    .iter()
                    .any(|(root_hash, _)| root_hash == extra_root_hash)
                {
                    return Err(format!(
                        "captured typed witness root 0x{extra_root_hash:x} is not required by manifest 0x{:x}",
                        self.manifest_hash
                    ));
                }
            }
        }

        self.build_commitment_set(&commitments)
    }

    pub fn validate_capture(
        &self,
        capture: &Qwen35TypedWitnessCapture,
    ) -> Result<Qwen35TypedWitnessCommitmentSet, String> {
        let commitment_set = self.build_commitment_set_from_capture(capture)?;
        self.validate_commitment_set(&commitment_set)?;
        Ok(commitment_set)
    }

    pub fn build_capture_from_source_capture(
        &self,
        source_capture: &Qwen35TypedWitnessSourceCapture,
    ) -> Result<Qwen35TypedWitnessCapture, String> {
        let captured_sources = source_capture.source_map()?;
        let requirements = self.source_requirements()?;
        let mut expected_sources = HashSet::new();
        for requirement in &requirements {
            expected_sources.insert(requirement.source.as_str());
        }

        for source in captured_sources.keys() {
            if !expected_sources.contains(*source) {
                return Err(format!(
                    "captured typed witness source {source} is not required by manifest 0x{:x}",
                    self.manifest_hash
                ));
            }
        }

        let mut capture = Qwen35TypedWitnessCapture::new();
        for root in self.layers.iter().flat_map(|layer| layer.roots.iter()) {
            let value = captured_sources.get(root.source.as_str()).ok_or_else(|| {
                format!(
                    "missing captured typed witness source {} for root {}",
                    root.source, root.name
                )
            })?;
            qwen35_validate_captured_value_shape(root, value)?;
            capture.roots.push(Qwen35TypedWitnessCapturedRoot {
                root_hash: root.root_hash(),
                value: (*value).clone(),
            });
        }
        Ok(capture)
    }

    pub fn validate_source_capture(
        &self,
        source_capture: &Qwen35TypedWitnessSourceCapture,
    ) -> Result<Qwen35TypedWitnessCommitmentSet, String> {
        let capture = self.build_capture_from_source_capture(source_capture)?;
        self.validate_capture(&capture)
    }

    pub fn build_source_capture_from_inventory(
        &self,
        inventory: &Qwen35TypedWitnessSourceInventory,
    ) -> Result<Qwen35TypedWitnessSourceCapture, String> {
        let requirements = self.source_requirements()?;
        let mut capture = Qwen35TypedWitnessSourceCapture::new();
        let mut expected_runtime = HashSet::new();
        let mut expected_safetensors = HashSet::new();
        let mut expected_conversation_state = HashSet::new();
        let mut expected_statement = HashSet::new();
        for requirement in &requirements {
            let body = requirement.source_body().to_string();
            match requirement.source_kind {
                Qwen35TypedWitnessSourceKind::Runtime => {
                    expected_runtime.insert(body);
                }
                Qwen35TypedWitnessSourceKind::Safetensors => {
                    expected_safetensors.insert(body);
                }
                Qwen35TypedWitnessSourceKind::ConversationState => {
                    expected_conversation_state.insert(body);
                }
                Qwen35TypedWitnessSourceKind::Statement => {
                    expected_statement.insert(body);
                }
            }
            let value = inventory
                .get_requirement_value(requirement)?
                .ok_or_else(|| {
                    format!(
                        "missing {} typed witness inventory value {}",
                        requirement.source_kind.label(),
                        requirement.source_body()
                    )
                })?
                .clone();
            capture.sources.push(Qwen35TypedWitnessCapturedSource {
                source: requirement.source.clone(),
                value,
            });
        }
        qwen35_reject_extra_inventory_values(
            "runtime",
            &inventory.runtime,
            &expected_runtime,
            self.manifest_hash,
        )?;
        qwen35_reject_extra_inventory_values(
            "safetensors",
            &inventory.safetensors,
            &expected_safetensors,
            self.manifest_hash,
        )?;
        qwen35_reject_extra_inventory_values(
            "conversation-state",
            &inventory.conversation_state,
            &expected_conversation_state,
            self.manifest_hash,
        )?;
        qwen35_reject_extra_inventory_values(
            "statement",
            &inventory.statement,
            &expected_statement,
            self.manifest_hash,
        )?;
        Ok(capture)
    }

    pub fn validate_source_inventory(
        &self,
        inventory: &Qwen35TypedWitnessSourceInventory,
    ) -> Result<Qwen35TypedWitnessCommitmentSet, String> {
        let source_capture = self.build_source_capture_from_inventory(inventory)?;
        self.validate_source_capture(&source_capture)
    }

    pub fn validate_commitment_set(
        &self,
        commitment_set: &Qwen35TypedWitnessCommitmentSet,
    ) -> Result<(), String> {
        if commitment_set.manifest_hash != self.manifest_hash {
            return Err(format!(
                "typed witness commitment set manifest hash 0x{:x} != expected 0x{:x}",
                commitment_set.manifest_hash, self.manifest_hash
            ));
        }
        if commitment_set.seq_len != self.seq_len {
            return Err(format!(
                "typed witness commitment set seq_len {} != manifest seq_len {}",
                commitment_set.seq_len, self.seq_len
            ));
        }
        let rebuilt = self.build_commitment_set(
            &commitment_set
                .commitments
                .iter()
                .map(|entry| (entry.root_hash, entry.commitment))
                .collect::<Vec<_>>(),
        )?;
        if rebuilt.commitment_set_hash != commitment_set.commitment_set_hash {
            return Err(format!(
                "typed witness commitment set hash 0x{:x} != expected 0x{:x}",
                commitment_set.commitment_set_hash, rebuilt.commitment_set_hash
            ));
        }
        Ok(())
    }

    fn compute_hash(
        architecture_contract_hash: FieldElement,
        seq_len: usize,
        layers: &[Qwen35TypedWitnessLayerManifest],
    ) -> FieldElement {
        let mut felts = vec![
            FieldElement::from(DOMAIN_QWEN35_TYPED_WITNESS_MANIFEST),
            architecture_contract_hash,
            FieldElement::from(seq_len as u64),
            FieldElement::from(layers.len() as u64),
        ];
        for layer in layers {
            felts.push(layer.layer_hash());
        }
        starknet_crypto::poseidon_hash_many(&felts)
    }
}

impl Qwen35TypedWitnessCommitmentSet {
    fn compute_hash(
        manifest_hash: FieldElement,
        seq_len: usize,
        commitments: &[Qwen35TypedWitnessCommitment],
    ) -> FieldElement {
        let mut felts = vec![
            FieldElement::from(DOMAIN_QWEN35_TYPED_WITNESS_MANIFEST),
            manifest_hash,
            FieldElement::from(seq_len as u64),
            FieldElement::from(commitments.len() as u64),
        ];
        for entry in commitments {
            felts.push(entry.root_hash);
            felts.push(entry.commitment);
        }
        starknet_crypto::poseidon_hash_many(&felts)
    }
}

impl Qwen35TypedWitnessCapture {
    pub fn new() -> Self {
        Self { roots: Vec::new() }
    }

    pub fn insert_matrix(&mut self, root_hash: FieldElement, matrix: M31Matrix) {
        self.roots.push(Qwen35TypedWitnessCapturedRoot {
            root_hash,
            value: Qwen35TypedWitnessCapturedValue::Matrix(matrix),
        });
    }

    pub fn insert_vector(&mut self, root_hash: FieldElement, values: Vec<M31>) {
        self.roots.push(Qwen35TypedWitnessCapturedRoot {
            root_hash,
            value: Qwen35TypedWitnessCapturedValue::Vector(values),
        });
    }

    pub fn insert_lookup_table(&mut self, root_hash: FieldElement, table_log_size: u32) {
        self.roots.push(Qwen35TypedWitnessCapturedRoot {
            root_hash,
            value: Qwen35TypedWitnessCapturedValue::LookupTable { table_log_size },
        });
    }

    fn root_map(&self) -> Result<HashMap<FieldElement, &Qwen35TypedWitnessCapturedValue>, String> {
        let mut map = HashMap::new();
        for captured in &self.roots {
            if captured.root_hash == FieldElement::ZERO {
                return Err("captured typed witness root hash is zero".to_string());
            }
            if map.insert(captured.root_hash, &captured.value).is_some() {
                return Err(format!(
                    "duplicate captured typed witness root 0x{:x}",
                    captured.root_hash
                ));
            }
        }
        Ok(map)
    }
}

impl Qwen35TypedWitnessSourceCapture {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    pub fn insert_matrix(&mut self, source: impl Into<String>, matrix: M31Matrix) {
        self.sources.push(Qwen35TypedWitnessCapturedSource {
            source: source.into(),
            value: Qwen35TypedWitnessCapturedValue::Matrix(matrix),
        });
    }

    pub fn insert_vector(&mut self, source: impl Into<String>, values: Vec<M31>) {
        self.sources.push(Qwen35TypedWitnessCapturedSource {
            source: source.into(),
            value: Qwen35TypedWitnessCapturedValue::Vector(values),
        });
    }

    pub fn insert_lookup_table(&mut self, source: impl Into<String>, table_log_size: u32) {
        self.sources.push(Qwen35TypedWitnessCapturedSource {
            source: source.into(),
            value: Qwen35TypedWitnessCapturedValue::LookupTable { table_log_size },
        });
    }

    fn source_map(&self) -> Result<HashMap<&str, &Qwen35TypedWitnessCapturedValue>, String> {
        let mut map = HashMap::new();
        for captured in &self.sources {
            if captured.source.is_empty() {
                return Err("captured typed witness source is empty".to_string());
            }
            qwen35_typed_witness_source_kind(&captured.source)?;
            if map
                .insert(captured.source.as_str(), &captured.value)
                .is_some()
            {
                return Err(format!(
                    "duplicate captured typed witness source {}",
                    captured.source
                ));
            }
        }
        Ok(map)
    }
}

impl Qwen35TypedWitnessSourceRequirement {
    pub fn source_body(&self) -> &str {
        qwen35_typed_witness_source_body(&self.source).unwrap_or(self.source.as_str())
    }
}

impl Qwen35TypedWitnessSourceInventory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_runtime_matrix(&mut self, name: impl Into<String>, matrix: M31Matrix) {
        self.insert_runtime_value(name, Qwen35TypedWitnessCapturedValue::Matrix(matrix));
    }

    pub fn insert_runtime_vector(&mut self, name: impl Into<String>, values: Vec<M31>) {
        self.insert_runtime_value(name, Qwen35TypedWitnessCapturedValue::Vector(values));
    }

    pub fn insert_safetensors_vector(&mut self, name: impl Into<String>, values: Vec<M31>) {
        self.insert_safetensors_value(name, Qwen35TypedWitnessCapturedValue::Vector(values));
    }

    pub fn insert_conversation_state_matrix(&mut self, name: impl Into<String>, matrix: M31Matrix) {
        self.insert_conversation_state_value(name, Qwen35TypedWitnessCapturedValue::Matrix(matrix));
    }

    pub fn insert_statement_lookup_table(&mut self, name: impl Into<String>, table_log_size: u32) {
        self.insert_statement_value(
            name,
            Qwen35TypedWitnessCapturedValue::LookupTable { table_log_size },
        );
    }

    pub fn insert_runtime_value(
        &mut self,
        name: impl Into<String>,
        value: Qwen35TypedWitnessCapturedValue,
    ) {
        self.runtime.insert(name.into(), value);
    }

    pub fn insert_safetensors_value(
        &mut self,
        name: impl Into<String>,
        value: Qwen35TypedWitnessCapturedValue,
    ) {
        self.safetensors.insert(name.into(), value);
    }

    pub fn insert_conversation_state_value(
        &mut self,
        name: impl Into<String>,
        value: Qwen35TypedWitnessCapturedValue,
    ) {
        self.conversation_state.insert(name.into(), value);
    }

    pub fn insert_statement_value(
        &mut self,
        name: impl Into<String>,
        value: Qwen35TypedWitnessCapturedValue,
    ) {
        self.statement.insert(name.into(), value);
    }

    fn get_requirement_value(
        &self,
        requirement: &Qwen35TypedWitnessSourceRequirement,
    ) -> Result<Option<&Qwen35TypedWitnessCapturedValue>, String> {
        let body = qwen35_typed_witness_source_body(&requirement.source)?;
        let value = match requirement.source_kind {
            Qwen35TypedWitnessSourceKind::Runtime => self.runtime.get(body),
            Qwen35TypedWitnessSourceKind::Safetensors => self.safetensors.get(body),
            Qwen35TypedWitnessSourceKind::ConversationState => self.conversation_state.get(body),
            Qwen35TypedWitnessSourceKind::Statement => self.statement.get(body),
        };
        Ok(value)
    }
}

impl Qwen35TypedWitnessInventoryRecorder {
    pub fn new(manifest: Qwen35TypedWitnessManifest) -> Self {
        Self {
            manifest,
            inventory: Qwen35TypedWitnessSourceInventory::new(),
        }
    }

    pub fn with_inventory(
        manifest: Qwen35TypedWitnessManifest,
        inventory: Qwen35TypedWitnessSourceInventory,
    ) -> Self {
        Self {
            manifest,
            inventory,
        }
    }

    pub fn inventory(&self) -> &Qwen35TypedWitnessSourceInventory {
        &self.inventory
    }

    pub fn into_inventory(self) -> Qwen35TypedWitnessSourceInventory {
        self.inventory
    }

    pub fn validate(&self) -> Result<Qwen35TypedWitnessCommitmentSet, String> {
        self.manifest.validate_source_inventory(&self.inventory)
    }

    pub fn finish(self) -> Result<Qwen35TypedWitnessCommitmentSet, String> {
        self.manifest.validate_source_inventory(&self.inventory)
    }

    pub fn record_qkv_projected(
        &mut self,
        layer_idx: usize,
        matrix: M31Matrix,
    ) -> Result<(), String> {
        self.record_runtime_matrix(layer_idx, "qkv_projected", matrix)
    }

    pub fn record_qkv_after_conv(
        &mut self,
        layer_idx: usize,
        matrix: M31Matrix,
    ) -> Result<(), String> {
        self.record_runtime_matrix(layer_idx, "qkv_after_conv", matrix)
    }

    pub fn record_query(&mut self, layer_idx: usize, matrix: M31Matrix) -> Result<(), String> {
        self.record_runtime_matrix(layer_idx, "query", matrix)
    }

    pub fn record_key(&mut self, layer_idx: usize, matrix: M31Matrix) -> Result<(), String> {
        self.record_runtime_matrix(layer_idx, "key", matrix)
    }

    pub fn record_projected_value(
        &mut self,
        layer_idx: usize,
        matrix: M31Matrix,
    ) -> Result<(), String> {
        self.record_runtime_matrix(layer_idx, "projected_value", matrix)
    }

    pub fn record_a_gate(&mut self, layer_idx: usize, matrix: M31Matrix) -> Result<(), String> {
        self.record_runtime_matrix(layer_idx, "a_gate", matrix)
    }

    pub fn record_b_gate(&mut self, layer_idx: usize, matrix: M31Matrix) -> Result<(), String> {
        self.record_runtime_matrix(layer_idx, "b_gate", matrix)
    }

    pub fn record_attended_value(
        &mut self,
        layer_idx: usize,
        matrix: M31Matrix,
    ) -> Result<(), String> {
        self.record_runtime_matrix(layer_idx, "attended_value", matrix)
    }

    pub fn record_z_gate(&mut self, layer_idx: usize, matrix: M31Matrix) -> Result<(), String> {
        self.record_runtime_matrix(layer_idx, "z_gate", matrix)
    }

    pub fn record_gated_value(
        &mut self,
        layer_idx: usize,
        matrix: M31Matrix,
    ) -> Result<(), String> {
        self.record_runtime_matrix(layer_idx, "gated_value", matrix)
    }

    pub fn record_conv1d_weight(
        &mut self,
        layer_idx: usize,
        values: Vec<M31>,
    ) -> Result<(), String> {
        self.record_safetensors_vector(layer_idx, "conv1d.weight", values)
    }

    pub fn record_a_log_weight(
        &mut self,
        layer_idx: usize,
        values: Vec<M31>,
    ) -> Result<(), String> {
        self.record_safetensors_vector(layer_idx, "A_log", values)
    }

    pub fn record_dt_bias(&mut self, layer_idx: usize, values: Vec<M31>) -> Result<(), String> {
        self.record_safetensors_vector(layer_idx, "dt_bias", values)
    }

    pub fn record_norm_weight(&mut self, layer_idx: usize, values: Vec<M31>) -> Result<(), String> {
        self.record_safetensors_vector(layer_idx, "norm.weight", values)
    }

    pub fn record_initial_recurrent_state(
        &mut self,
        layer_idx: usize,
        matrix: M31Matrix,
    ) -> Result<(), String> {
        self.record_conversation_state_matrix(layer_idx, "initial_recurrent_state", matrix)
    }

    pub fn record_final_recurrent_state(
        &mut self,
        layer_idx: usize,
        matrix: M31Matrix,
    ) -> Result<(), String> {
        self.record_conversation_state_matrix(layer_idx, "final_recurrent_state", matrix)
    }

    pub fn record_norm_and_z_gate_rsqrt_table(
        &mut self,
        layer_idx: usize,
        table_log_size: u32,
    ) -> Result<(), String> {
        let body = qwen35_linear_attn_state_source_body(layer_idx, "norm_and_z_gate_rsqrt_table");
        self.record_value(
            Qwen35TypedWitnessSourceKind::Statement,
            body,
            Qwen35TypedWitnessCapturedValue::LookupTable { table_log_size },
        )
    }

    fn record_runtime_matrix(
        &mut self,
        layer_idx: usize,
        name: &str,
        matrix: M31Matrix,
    ) -> Result<(), String> {
        let body = qwen35_linear_attn_tensor_source_body(layer_idx, name);
        self.record_value(
            Qwen35TypedWitnessSourceKind::Runtime,
            body,
            Qwen35TypedWitnessCapturedValue::Matrix(matrix),
        )
    }

    fn record_safetensors_vector(
        &mut self,
        layer_idx: usize,
        name: &str,
        values: Vec<M31>,
    ) -> Result<(), String> {
        let body = qwen35_linear_attn_tensor_source_body(layer_idx, name);
        self.record_value(
            Qwen35TypedWitnessSourceKind::Safetensors,
            body,
            Qwen35TypedWitnessCapturedValue::Vector(values),
        )
    }

    fn record_conversation_state_matrix(
        &mut self,
        layer_idx: usize,
        name: &str,
        matrix: M31Matrix,
    ) -> Result<(), String> {
        let body = qwen35_linear_attn_state_source_body(layer_idx, name);
        self.record_value(
            Qwen35TypedWitnessSourceKind::ConversationState,
            body,
            Qwen35TypedWitnessCapturedValue::Matrix(matrix),
        )
    }

    fn record_value(
        &mut self,
        source_kind: Qwen35TypedWitnessSourceKind,
        body: String,
        value: Qwen35TypedWitnessCapturedValue,
    ) -> Result<(), String> {
        let source = format!("{}:{body}", source_kind.label());
        let requirements = self.manifest.source_requirements()?;
        let requirement = requirements
            .iter()
            .find(|candidate| candidate.source == source)
            .ok_or_else(|| {
                format!(
                    "{} typed witness source {body} is not required by manifest 0x{:x}",
                    source_kind.label(),
                    self.manifest.manifest_hash
                )
            })?;
        qwen35_validate_captured_value_shape_for_source(
            &requirement.source,
            requirement.shape,
            &value,
        )?;
        let namespace = match source_kind {
            Qwen35TypedWitnessSourceKind::Runtime => &mut self.inventory.runtime,
            Qwen35TypedWitnessSourceKind::Safetensors => &mut self.inventory.safetensors,
            Qwen35TypedWitnessSourceKind::ConversationState => {
                &mut self.inventory.conversation_state
            }
            Qwen35TypedWitnessSourceKind::Statement => &mut self.inventory.statement,
        };
        if namespace.contains_key(&body) {
            return Err(format!(
                "duplicate {} typed witness recorder value {body}",
                source_kind.label()
            ));
        }
        namespace.insert(body, value);
        Ok(())
    }
}

impl Qwen35GatedDeltaNetRuntimeLayerTrace {
    pub fn validate_against_contract(
        &self,
        contract: &Qwen35GatedDeltaNetContract,
    ) -> Result<(), String> {
        if self.layer_idx != contract.layer_idx {
            return Err(format!(
                "typed runtime trace layer {} != GatedDeltaNet contract layer {}",
                self.layer_idx, contract.layer_idx
            ));
        }
        qwen35_validate_matrix_shape(
            &format!("layer {} qkv_projected", self.layer_idx),
            &self.qkv_projected,
            contract.qkv_projected,
        )?;
        qwen35_validate_matrix_shape(
            &format!("layer {} qkv_after_conv", self.layer_idx),
            &self.qkv_after_conv,
            contract.qkv_after_conv,
        )?;
        qwen35_validate_matrix_shape(
            &format!("layer {} query", self.layer_idx),
            &self.query,
            contract.query,
        )?;
        qwen35_validate_matrix_shape(
            &format!("layer {} key", self.layer_idx),
            &self.key,
            contract.key,
        )?;
        qwen35_validate_matrix_shape(
            &format!("layer {} projected_value", self.layer_idx),
            &self.projected_value,
            contract.projected_value,
        )?;
        qwen35_validate_matrix_shape(
            &format!("layer {} a_gate", self.layer_idx),
            &self.a_gate,
            contract.a_gate,
        )?;
        qwen35_validate_matrix_shape(
            &format!("layer {} b_gate", self.layer_idx),
            &self.b_gate,
            contract.b_gate,
        )?;
        qwen35_validate_matrix_shape(
            &format!("layer {} attended_value", self.layer_idx),
            &self.attended_value,
            contract.attended_value,
        )?;
        qwen35_validate_matrix_shape(
            &format!("layer {} z_gate", self.layer_idx),
            &self.z_gate,
            contract.z_gate,
        )?;
        qwen35_validate_matrix_shape(
            &format!("layer {} gated_value", self.layer_idx),
            &self.gated_value,
            contract.attended_value,
        )?;

        let qk_head_dim = contract
            .query
            .cols
            .checked_div(contract.recurrent_state_rows)
            .ok_or_else(|| format!("layer {} recurrent_state_rows is zero", contract.layer_idx))?;
        let recurrent_state_shape = Qwen35Tensor2DShape {
            rows: contract.recurrent_state_rows * qk_head_dim,
            cols: contract.linear_value_head_dim,
        };
        qwen35_validate_matrix_shape(
            &format!("layer {} initial_recurrent_state", self.layer_idx),
            &self.initial_recurrent_state,
            recurrent_state_shape,
        )?;
        qwen35_validate_matrix_shape(
            &format!("layer {} final_recurrent_state", self.layer_idx),
            &self.final_recurrent_state,
            recurrent_state_shape,
        )?;
        if let Some(transform) = &self.delta_recurrence_transform {
            qwen35_validate_matrix_shape(
                &format!("layer {} scaled_query", self.layer_idx),
                &transform.scaled_query,
                contract.query,
            )?;
            qwen35_validate_matrix_shape(
                &format!("layer {} normalized_key", self.layer_idx),
                &transform.normalized_key,
                contract.key,
            )?;
            qwen35_validate_matrix_shape(
                &format!("layer {} decay", self.layer_idx),
                &transform.decay,
                contract.a_gate,
            )?;
            qwen35_validate_matrix_shape(
                &format!("layer {} beta", self.layer_idx),
                &transform.beta,
                contract.b_gate,
            )?;
            if transform.q_norm_table_log_size == 0
                || transform.k_norm_table_log_size == 0
                || transform.beta_sigmoid_table_log_size == 0
                || transform.decay_table_log_size == 0
            {
                return Err(format!(
                    "layer {} DeltaRecurrence transform table log sizes must be non-zero",
                    self.layer_idx
                ));
            }
        }
        if self.norm_and_z_gate_rsqrt_table_log_size == 0 {
            return Err(format!(
                "layer {} norm_and_z_gate_rsqrt_table_log_size must be non-zero",
                self.layer_idx
            ));
        }
        Ok(())
    }

    pub fn record_into(
        &self,
        recorder: &mut Qwen35TypedWitnessInventoryRecorder,
    ) -> Result<(), String> {
        recorder.record_qkv_projected(self.layer_idx, self.qkv_projected.clone())?;
        recorder.record_qkv_after_conv(self.layer_idx, self.qkv_after_conv.clone())?;
        recorder.record_query(self.layer_idx, self.query.clone())?;
        recorder.record_key(self.layer_idx, self.key.clone())?;
        recorder.record_projected_value(self.layer_idx, self.projected_value.clone())?;
        recorder.record_a_gate(self.layer_idx, self.a_gate.clone())?;
        recorder.record_b_gate(self.layer_idx, self.b_gate.clone())?;
        recorder.record_attended_value(self.layer_idx, self.attended_value.clone())?;
        recorder.record_z_gate(self.layer_idx, self.z_gate.clone())?;
        recorder.record_gated_value(self.layer_idx, self.gated_value.clone())?;
        recorder
            .record_initial_recurrent_state(self.layer_idx, self.initial_recurrent_state.clone())?;
        recorder
            .record_final_recurrent_state(self.layer_idx, self.final_recurrent_state.clone())?;
        recorder.record_norm_and_z_gate_rsqrt_table(
            self.layer_idx,
            self.norm_and_z_gate_rsqrt_table_log_size,
        )?;
        Ok(())
    }
}

impl Qwen35TypedRuntimeTrace {
    pub fn new(seq_len: usize, layers: Vec<Qwen35GatedDeltaNetRuntimeLayerTrace>) -> Self {
        Self { seq_len, layers }
    }

    pub fn validate_layer_set(&self, plan: &Qwen35ProofPlan) -> Result<(), String> {
        if self.seq_len == 0 {
            return Err("typed runtime trace seq_len must be non-zero".to_string());
        }

        let expected = plan
            .layers
            .iter()
            .filter(|layer| layer.attention == Qwen35AttentionKind::GatedDeltaNet)
            .map(|layer| layer.layer_idx)
            .collect::<HashSet<_>>();
        let mut seen = HashSet::new();
        for layer in &self.layers {
            if !expected.contains(&layer.layer_idx) {
                return Err(format!(
                    "typed runtime trace layer {} is not a GatedDeltaNet layer in the Qwen3.5 plan",
                    layer.layer_idx
                ));
            }
            if !seen.insert(layer.layer_idx) {
                return Err(format!(
                    "duplicate typed runtime trace layer {}",
                    layer.layer_idx
                ));
            }
        }
        for layer_idx in expected {
            if !seen.contains(&layer_idx) {
                return Err(format!("missing typed runtime trace layer {layer_idx}"));
            }
        }
        Ok(())
    }

    pub fn validate_against_plan(&self, plan: &Qwen35ProofPlan) -> Result<(), String> {
        self.validate_layer_set(plan)?;
        for layer in &self.layers {
            let contract = plan.gated_delta_net_contract(layer.layer_idx, self.seq_len)?;
            layer.validate_against_contract(&contract)?;
        }
        Ok(())
    }

    pub fn record_into(
        &self,
        recorder: &mut Qwen35TypedWitnessInventoryRecorder,
    ) -> Result<(), String> {
        for layer in &self.layers {
            layer.record_into(recorder)?;
        }
        Ok(())
    }

    pub fn build_statement_set(
        &self,
        plan: &Qwen35ProofPlan,
        safetensors_inventory: &Qwen35TypedWitnessSourceInventory,
    ) -> Result<Qwen35TypedRuntimeStatementSet, String> {
        self.validate_against_plan(plan)?;
        let mut layers = self.layers.iter().collect::<Vec<_>>();
        layers.sort_by_key(|layer| layer.layer_idx);

        let mut statement_set = Qwen35TypedRuntimeStatementSet::default();
        for layer in layers {
            let contract = plan.gated_delta_net_contract(layer.layer_idx, self.seq_len)?;
            let conv1d_weight = qwen35_safetensors_vector(
                safetensors_inventory,
                &qwen35_linear_attn_tensor_source_body(layer.layer_idx, "conv1d.weight"),
            )?;
            let a_log_weight = qwen35_safetensors_vector(
                safetensors_inventory,
                &qwen35_linear_attn_tensor_source_body(layer.layer_idx, "A_log"),
            )?;
            let dt_bias = qwen35_safetensors_vector(
                safetensors_inventory,
                &qwen35_linear_attn_tensor_source_body(layer.layer_idx, "dt_bias"),
            )?;
            let norm_weight = qwen35_safetensors_vector(
                safetensors_inventory,
                &qwen35_linear_attn_tensor_source_body(layer.layer_idx, "norm.weight"),
            )?;

            let depthwise_statement =
                crate::components::qwen35_depthwise_conv1d::qwen35_depthwise_conv1d_statement(
                    layer.layer_idx,
                    &layer.qkv_projected,
                    conv1d_weight,
                    &layer.qkv_after_conv,
                    contract.conv1d_weight.inner,
                )?;
            contract
                .depthwise_conv1d_trace_binding_contract()?
                .validate_statement(
                    depthwise_statement.layer_idx,
                    depthwise_statement.seq_len,
                    depthwise_statement.channels,
                    depthwise_statement.kernel,
                    depthwise_statement.input_commitment,
                    depthwise_statement.weight_commitment,
                    depthwise_statement.output_commitment,
                    depthwise_statement.statement_hash,
                )?;
            statement_set.depthwise_conv1d.push(depthwise_statement);

            let mode = if self.seq_len == 1 {
                crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceMode::RecurrentDecode
            } else {
                crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceMode::ChunkPrefill
            };
            let delta_inputs =
                crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceInputs {
                    query: &layer.query,
                    key: &layer.key,
                    projected_value: &layer.projected_value,
                    a_gate: &layer.a_gate,
                    b_gate: &layer.b_gate,
                    a_log_weight,
                    dt_bias,
                    initial_recurrent_state: &layer.initial_recurrent_state,
                    final_recurrent_state: &layer.final_recurrent_state,
                    output: &layer.attended_value,
                    state_rows: contract.recurrent_state_rows,
                    value_head_dim: contract.linear_value_head_dim,
                    mode,
                };
            let delta_statement =
                crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_statement(
                    layer.layer_idx,
                    &delta_inputs,
                )?;
            contract
                .delta_recurrence_trace_binding_contract()?
                .validate_statement(
                    delta_statement.layer_idx,
                    delta_statement.seq_len,
                    delta_statement.query_width,
                    delta_statement.key_width,
                    delta_statement.value_width,
                    delta_statement.state_rows,
                    delta_statement.value_head_dim,
                    delta_statement.query_commitment,
                    delta_statement.key_commitment,
                    delta_statement.projected_value_commitment,
                    delta_statement.a_gate_commitment,
                    delta_statement.b_gate_commitment,
                    delta_statement.a_log_weight_commitment,
                    delta_statement.dt_bias_commitment,
                    delta_statement.mode.as_u64(),
                    delta_statement.air_spec_hash,
                    delta_statement.initial_recurrent_state_commitment,
                    delta_statement.final_recurrent_state_commitment,
                    delta_statement.output_commitment,
                    delta_statement.statement_hash,
                )?;
            statement_set.delta_recurrence.push(delta_statement.clone());

            if let Some(transform) = &layer.delta_recurrence_transform {
                let transform_inputs =
                    crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceTransformInputs {
                        query: &layer.query,
                        key: &layer.key,
                        a_gate: &layer.a_gate,
                        b_gate: &layer.b_gate,
                        a_log_weight,
                        dt_bias,
                        scaled_query: &transform.scaled_query,
                        normalized_key: &transform.normalized_key,
                        decay: &transform.decay,
                        beta: &transform.beta,
                        state_rows: contract.recurrent_state_rows,
                        value_head_dim: contract.linear_value_head_dim,
                        mode,
                    };
                let transform_statement =
                    crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_transform_statement(
                        layer.layer_idx,
                        &transform_inputs,
                    )?;

                let arithmetic_inputs =
                    crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceArithmeticInputs {
                        scaled_query: &transform.scaled_query,
                        normalized_key: &transform.normalized_key,
                        projected_value: &layer.projected_value,
                        decay: &transform.decay,
                        beta: &transform.beta,
                        initial_recurrent_state: &layer.initial_recurrent_state,
                        final_recurrent_state: &layer.final_recurrent_state,
                        output: &layer.attended_value,
                        state_rows: contract.recurrent_state_rows,
                        value_head_dim: contract.linear_value_head_dim,
                    };
                let arithmetic_statement =
                    crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_arithmetic_statement(
                        layer.layer_idx,
                        mode,
                        &arithmetic_inputs,
                    )?;
                qwen35_validate_delta_recurrence_transform_statement_bindings(
                    &delta_statement,
                    &arithmetic_statement,
                    &transform_statement,
                )?;

                let q_norm_statement =
                    crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_norm_statement(
                        crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceNormKind::Query,
                        layer.layer_idx,
                        &layer.query,
                        &transform.scaled_query,
                        contract.recurrent_state_rows,
                        transform.q_norm_table_log_size,
                        transform.query_norm_post_scale,
                    )?;
                if q_norm_statement.input_commitment != transform_statement.query_commitment
                    || q_norm_statement.output_commitment
                        != transform_statement.scaled_query_commitment
                {
                    return Err(format!(
                        "DeltaRecurrence query norm statement commitments do not match layer {} transform statement",
                        layer.layer_idx
                    ));
                }

                let k_norm_statement =
                    crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_norm_statement(
                        crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceNormKind::Key,
                        layer.layer_idx,
                        &layer.key,
                        &transform.normalized_key,
                        contract.recurrent_state_rows,
                        transform.k_norm_table_log_size,
                        transform.key_norm_post_scale,
                    )?;
                if k_norm_statement.input_commitment != transform_statement.key_commitment
                    || k_norm_statement.output_commitment
                        != transform_statement.normalized_key_commitment
                {
                    return Err(format!(
                        "DeltaRecurrence key norm statement commitments do not match layer {} transform statement",
                        layer.layer_idx
                    ));
                }

                let beta_statement =
                    crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_beta_sigmoid_statement(
                        layer.layer_idx,
                        &layer.b_gate,
                        &transform.beta,
                        transform.beta_sigmoid_table_log_size,
                    )?;
                if beta_statement.b_gate_commitment != transform_statement.b_gate_commitment
                    || beta_statement.beta_commitment != transform_statement.beta_commitment
                {
                    return Err(format!(
                        "DeltaRecurrence beta sigmoid statement commitments do not match layer {} transform statement",
                        layer.layer_idx
                    ));
                }

                let decay_statement =
                    crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_decay_statement(
                        layer.layer_idx,
                        &layer.a_gate,
                        a_log_weight,
                        dt_bias,
                        &transform.decay,
                        transform.decay_table_log_size,
                    )?;
                if decay_statement.a_gate_commitment != transform_statement.a_gate_commitment
                    || decay_statement.a_log_weight_commitment
                        != transform_statement.a_log_weight_commitment
                    || decay_statement.dt_bias_commitment != transform_statement.dt_bias_commitment
                    || decay_statement.decay_commitment != transform_statement.decay_commitment
                {
                    return Err(format!(
                        "DeltaRecurrence decay statement commitments do not match layer {} transform statement",
                        layer.layer_idx
                    ));
                }

                statement_set
                    .delta_recurrence_arithmetic
                    .push(arithmetic_statement);
                statement_set
                    .delta_recurrence_transform
                    .push(transform_statement);
                statement_set.delta_recurrence_q_norm.push(q_norm_statement);
                statement_set.delta_recurrence_k_norm.push(k_norm_statement);
                statement_set
                    .delta_recurrence_beta_sigmoid
                    .push(beta_statement);
                statement_set.delta_recurrence_decay.push(decay_statement);
            }

            let norm_statement =
                crate::components::qwen35_norm_and_z_gate::qwen35_norm_and_z_gate_statement(
                    layer.layer_idx,
                    &layer.attended_value,
                    norm_weight,
                    &layer.z_gate,
                    &layer.gated_value,
                    contract.norm_weight,
                    layer.norm_and_z_gate_rsqrt_table_log_size,
                )?;
            contract
                .norm_and_z_gate_trace_binding_contract()?
                .validate_statement(
                    norm_statement.layer_idx,
                    norm_statement.seq_len,
                    norm_statement.value_heads,
                    norm_statement.head_dim,
                    norm_statement.table_log_size,
                    norm_statement.trace_checksum,
                    norm_statement.table_commitment,
                    norm_statement.attended_value_commitment,
                    norm_statement.norm_weight_commitment,
                    norm_statement.z_gate_commitment,
                    norm_statement.output_commitment,
                    norm_statement.statement_hash,
                )?;
            statement_set.norm_and_z_gate.push(norm_statement);
        }
        Ok(statement_set)
    }

    pub fn build_typed_proof_ledger(
        &self,
        plan: &Qwen35ProofPlan,
        safetensors_inventory: &Qwen35TypedWitnessSourceInventory,
    ) -> Result<Qwen35TypedProofLedger, String> {
        let statement_set = self.build_statement_set(plan, safetensors_inventory)?;
        statement_set.build_typed_proof_ledger(plan, self.seq_len)
    }

    pub fn prove_active_typed_proof_ledger(
        &self,
        plan: &Qwen35ProofPlan,
        safetensors_inventory: &Qwen35TypedWitnessSourceInventory,
    ) -> Result<Qwen35TypedProofLedger, String> {
        use crate::components::qwen35_delta_recurrence::{
            prove_qwen35_delta_recurrence_arithmetic_air,
            prove_qwen35_delta_recurrence_beta_sigmoid_air,
            prove_qwen35_delta_recurrence_decay_air, prove_qwen35_delta_recurrence_norm_air,
            prove_qwen35_delta_recurrence_trace_binding_air,
            prove_qwen35_delta_recurrence_transform_binding_air,
            Qwen35DeltaRecurrenceArithmeticInputs, Qwen35DeltaRecurrenceInputs,
            Qwen35DeltaRecurrenceMode, Qwen35DeltaRecurrenceNormKind,
            Qwen35DeltaRecurrenceTransformInputs,
        };

        self.validate_against_plan(plan)?;
        let mut ledger = plan.typed_proof_ledger(self.seq_len);
        let mut layers = self.layers.iter().collect::<Vec<_>>();
        layers.sort_by_key(|layer| layer.layer_idx);

        for layer in layers {
            let contract = plan.gated_delta_net_contract(layer.layer_idx, self.seq_len)?;
            let conv1d_weight = qwen35_safetensors_vector(
                safetensors_inventory,
                &qwen35_linear_attn_tensor_source_body(layer.layer_idx, "conv1d.weight"),
            )?;
            let a_log_weight = qwen35_safetensors_vector(
                safetensors_inventory,
                &qwen35_linear_attn_tensor_source_body(layer.layer_idx, "A_log"),
            )?;
            let dt_bias = qwen35_safetensors_vector(
                safetensors_inventory,
                &qwen35_linear_attn_tensor_source_body(layer.layer_idx, "dt_bias"),
            )?;
            let norm_weight = qwen35_safetensors_vector(
                safetensors_inventory,
                &qwen35_linear_attn_tensor_source_body(layer.layer_idx, "norm.weight"),
            )?;

            let depthwise_proof =
                crate::components::qwen35_depthwise_conv1d::prove_qwen35_depthwise_conv1d_air_for_layer(
                    layer.layer_idx,
                    &layer.qkv_projected,
                    conv1d_weight,
                    contract.conv1d_weight.inner,
                )
                .map_err(|err| err.to_string())?;
            let runtime_depthwise_output =
                crate::components::qwen35_depthwise_conv1d::qwen35_depthwise_conv1d_output_commitment(
                    &layer.qkv_after_conv,
                );
            if depthwise_proof.statement.output_commitment != runtime_depthwise_output {
                return Err(format!(
                    "DepthwiseConv1D active proof output does not match typed runtime trace for layer {}",
                    layer.layer_idx
                ));
            }
            ledger.record_depthwise_conv1d_air_proof(
                &contract.depthwise_conv1d_trace_binding_contract()?,
                &depthwise_proof,
            )?;

            let mode = if self.seq_len == 1 {
                Qwen35DeltaRecurrenceMode::RecurrentDecode
            } else {
                Qwen35DeltaRecurrenceMode::ChunkPrefill
            };
            let transform = layer.delta_recurrence_transform.as_ref().ok_or_else(|| {
                format!(
                    "typed runtime trace layer {} missing DeltaRecurrence transform runtime trace",
                    layer.layer_idx
                )
            })?;
            let delta_inputs = Qwen35DeltaRecurrenceInputs {
                query: &layer.query,
                key: &layer.key,
                projected_value: &layer.projected_value,
                a_gate: &layer.a_gate,
                b_gate: &layer.b_gate,
                a_log_weight,
                dt_bias,
                initial_recurrent_state: &layer.initial_recurrent_state,
                final_recurrent_state: &layer.final_recurrent_state,
                output: &layer.attended_value,
                state_rows: contract.recurrent_state_rows,
                value_head_dim: contract.linear_value_head_dim,
                mode,
            };
            let trace_proof =
                prove_qwen35_delta_recurrence_trace_binding_air(layer.layer_idx, &delta_inputs)
                    .map_err(|err| err.to_string())?;

            let arithmetic_inputs = Qwen35DeltaRecurrenceArithmeticInputs {
                scaled_query: &transform.scaled_query,
                normalized_key: &transform.normalized_key,
                projected_value: &layer.projected_value,
                decay: &transform.decay,
                beta: &transform.beta,
                initial_recurrent_state: &layer.initial_recurrent_state,
                final_recurrent_state: &layer.final_recurrent_state,
                output: &layer.attended_value,
                state_rows: contract.recurrent_state_rows,
                value_head_dim: contract.linear_value_head_dim,
            };
            let arithmetic_proof = prove_qwen35_delta_recurrence_arithmetic_air(
                layer.layer_idx,
                mode,
                &arithmetic_inputs,
            )
            .map_err(|err| err.to_string())?;
            let delta_binding = contract.delta_recurrence_trace_binding_contract()?;
            ledger.record_delta_recurrence_active_air_proofs(
                &delta_binding,
                &trace_proof,
                &arithmetic_proof,
            )?;

            let transform_inputs = Qwen35DeltaRecurrenceTransformInputs {
                query: &layer.query,
                key: &layer.key,
                a_gate: &layer.a_gate,
                b_gate: &layer.b_gate,
                a_log_weight,
                dt_bias,
                scaled_query: &transform.scaled_query,
                normalized_key: &transform.normalized_key,
                decay: &transform.decay,
                beta: &transform.beta,
                state_rows: contract.recurrent_state_rows,
                value_head_dim: contract.linear_value_head_dim,
                mode,
            };
            let transform_proof = prove_qwen35_delta_recurrence_transform_binding_air(
                layer.layer_idx,
                &transform_inputs,
            )
            .map_err(|err| err.to_string())?;
            ledger.record_delta_recurrence_transform_binding_air_proof(
                &delta_binding,
                &trace_proof.statement,
                &arithmetic_proof.statement,
                &transform_proof,
            )?;

            let q_norm_proof = prove_qwen35_delta_recurrence_norm_air(
                Qwen35DeltaRecurrenceNormKind::Query,
                layer.layer_idx,
                &layer.query,
                &transform.scaled_query,
                contract.recurrent_state_rows,
                transform.q_norm_table_log_size,
                transform.query_norm_post_scale,
            )
            .map_err(|err| err.to_string())?;
            ledger.record_delta_recurrence_norm_air_proof(
                &delta_binding,
                &transform_proof.statement,
                &q_norm_proof,
            )?;

            let k_norm_proof = prove_qwen35_delta_recurrence_norm_air(
                Qwen35DeltaRecurrenceNormKind::Key,
                layer.layer_idx,
                &layer.key,
                &transform.normalized_key,
                contract.recurrent_state_rows,
                transform.k_norm_table_log_size,
                transform.key_norm_post_scale,
            )
            .map_err(|err| err.to_string())?;
            ledger.record_delta_recurrence_norm_air_proof(
                &delta_binding,
                &transform_proof.statement,
                &k_norm_proof,
            )?;

            let beta_proof = prove_qwen35_delta_recurrence_beta_sigmoid_air(
                layer.layer_idx,
                &layer.b_gate,
                &transform.beta,
                transform.beta_sigmoid_table_log_size,
            )
            .map_err(|err| err.to_string())?;
            ledger.record_delta_recurrence_beta_sigmoid_air_proof(
                &delta_binding,
                &transform_proof.statement,
                &beta_proof,
            )?;

            let decay_proof = prove_qwen35_delta_recurrence_decay_air(
                layer.layer_idx,
                &layer.a_gate,
                a_log_weight,
                dt_bias,
                &transform.decay,
                transform.decay_table_log_size,
            )
            .map_err(|err| err.to_string())?;
            ledger.record_delta_recurrence_decay_air_proof(
                &delta_binding,
                &transform_proof.statement,
                &decay_proof,
            )?;
            ledger.finalize_delta_recurrence_nonlinear_transform(
                &delta_binding,
                &transform_proof.statement,
            )?;

            let norm_proof =
                crate::components::qwen35_norm_and_z_gate::prove_qwen35_norm_and_z_gate_air(
                    layer.layer_idx,
                    &layer.attended_value,
                    norm_weight,
                    &layer.z_gate,
                    contract.norm_weight,
                    layer.norm_and_z_gate_rsqrt_table_log_size,
                )
                .map_err(|err| err.to_string())?;
            let runtime_norm_output =
                crate::components::qwen35_norm_and_z_gate::qwen35_norm_and_z_gate_output_commitment(
                    &layer.gated_value,
                );
            if norm_proof.statement.output_commitment != runtime_norm_output {
                return Err(format!(
                    "NormAndZGate active proof output does not match typed runtime trace for layer {}",
                    layer.layer_idx
                ));
            }
            ledger.record_norm_and_z_gate_air_proof(
                &contract.norm_and_z_gate_trace_binding_contract()?,
                &norm_proof,
            )?;
        }

        Ok(ledger)
    }

    pub fn prove_active_typed_span(
        &self,
        plan: &Qwen35ProofPlan,
        safetensors_inventory: &Qwen35TypedWitnessSourceInventory,
        span_idx: usize,
    ) -> Result<Qwen35ActiveTypedSpanReceipt, String> {
        let witness_commitment_set =
            self.finish_with_safetensors_inventory(plan, safetensors_inventory.clone())?;
        let typed_ledger = self.prove_active_typed_proof_ledger(plan, safetensors_inventory)?;
        typed_ledger.validate_production_ready()?;

        let mut conversation_ledger = Qwen35ConversationStateLedger::new(
            plan.architecture_contract_hash(),
            plan.linear_attention_layers(),
        );
        conversation_ledger.record_active_span(span_idx, &typed_ledger)?;
        let conversation_span = conversation_ledger
            .spans
            .first()
            .cloned()
            .ok_or_else(|| "active typed span receipt did not record a span".to_string())?;
        let receipt_hash = Qwen35ActiveTypedSpanReceipt::compute_hash(
            span_idx,
            witness_commitment_set.commitment_set_hash,
            typed_ledger.ledger_hash(),
            &conversation_span,
        );

        Ok(Qwen35ActiveTypedSpanReceipt {
            span_idx,
            witness_commitment_set,
            typed_ledger,
            conversation_span,
            receipt_hash,
        })
    }

    pub fn finish_with_safetensors_inventory(
        &self,
        plan: &Qwen35ProofPlan,
        safetensors_inventory: Qwen35TypedWitnessSourceInventory,
    ) -> Result<Qwen35TypedWitnessCommitmentSet, String> {
        self.validate_against_plan(plan)?;
        let manifest = plan.typed_witness_manifest(self.seq_len)?;
        let mut recorder =
            Qwen35TypedWitnessInventoryRecorder::with_inventory(manifest, safetensors_inventory);
        self.record_into(&mut recorder)?;
        recorder.finish()
    }
}

impl Qwen35TypedRuntimeStatementSet {
    pub fn delta_recurrence_transform_statement_coverage_complete(&self) -> bool {
        let expected = self.delta_recurrence.len();
        self.delta_recurrence_arithmetic.len() == expected
            && self.delta_recurrence_transform.len() == expected
            && self.delta_recurrence_q_norm.len() == expected
            && self.delta_recurrence_k_norm.len() == expected
            && self.delta_recurrence_beta_sigmoid.len() == expected
            && self.delta_recurrence_decay.len() == expected
    }

    pub fn build_typed_proof_ledger(
        &self,
        plan: &Qwen35ProofPlan,
        seq_len: usize,
    ) -> Result<Qwen35TypedProofLedger, String> {
        let mut ledger = plan.typed_proof_ledger(seq_len);
        for statement in &self.depthwise_conv1d {
            let binding = plan
                .gated_delta_net_contract(statement.layer_idx, seq_len)?
                .depthwise_conv1d_trace_binding_contract()?;
            ledger.record_depthwise_conv1d_proof(&binding, statement)?;
        }
        for statement in &self.delta_recurrence {
            let binding = plan
                .gated_delta_net_contract(statement.layer_idx, seq_len)?
                .delta_recurrence_trace_binding_contract()?;
            ledger.record_delta_recurrence_proof(&binding, statement)?;
        }
        for statement in &self.norm_and_z_gate {
            let binding = plan
                .gated_delta_net_contract(statement.layer_idx, seq_len)?
                .norm_and_z_gate_trace_binding_contract()?;
            ledger.record_norm_and_z_gate_proof(&binding, statement)?;
        }
        Ok(ledger)
    }
}

fn qwen35_validate_delta_recurrence_transform_statement_bindings(
    trace_statement: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceStatement,
    arithmetic_statement: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceArithmeticStatement,
    transform_statement: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceTransformStatement,
) -> Result<(), String> {
    if arithmetic_statement.layer_idx != trace_statement.layer_idx
        || transform_statement.layer_idx != trace_statement.layer_idx
    {
        return Err("DeltaRecurrence transform coverage layer mismatch".to_string());
    }
    if arithmetic_statement.seq_len != trace_statement.seq_len
        || transform_statement.seq_len != trace_statement.seq_len
        || arithmetic_statement.query_width != trace_statement.query_width
        || transform_statement.query_width != trace_statement.query_width
        || arithmetic_statement.key_width != trace_statement.key_width
        || transform_statement.key_width != trace_statement.key_width
        || arithmetic_statement.state_rows != trace_statement.state_rows
        || transform_statement.state_rows != trace_statement.state_rows
        || arithmetic_statement.value_head_dim != trace_statement.value_head_dim
        || transform_statement.value_head_dim != trace_statement.value_head_dim
    {
        return Err(
            "DeltaRecurrence transform coverage dimensions do not match trace statement"
                .to_string(),
        );
    }
    if arithmetic_statement.value_width != trace_statement.value_width {
        return Err("DeltaRecurrence arithmetic value width mismatch".to_string());
    }
    if arithmetic_statement.mode != trace_statement.mode
        || transform_statement.mode != trace_statement.mode
    {
        return Err("DeltaRecurrence transform coverage mode mismatch".to_string());
    }
    if arithmetic_statement.air_spec_hash != trace_statement.air_spec_hash
        || transform_statement.air_spec_hash != trace_statement.air_spec_hash
    {
        return Err("DeltaRecurrence transform coverage AIR spec hash mismatch".to_string());
    }
    if arithmetic_statement.projected_value_commitment != trace_statement.projected_value_commitment
        || arithmetic_statement.initial_recurrent_state_commitment
            != trace_statement.initial_recurrent_state_commitment
        || arithmetic_statement.final_recurrent_state_commitment
            != trace_statement.final_recurrent_state_commitment
        || arithmetic_statement.output_commitment != trace_statement.output_commitment
    {
        return Err(
            "DeltaRecurrence arithmetic coverage IO/state commitments do not match trace statement"
                .to_string(),
        );
    }
    if transform_statement.query_commitment != trace_statement.query_commitment
        || transform_statement.key_commitment != trace_statement.key_commitment
        || transform_statement.a_gate_commitment != trace_statement.a_gate_commitment
        || transform_statement.b_gate_commitment != trace_statement.b_gate_commitment
        || transform_statement.a_log_weight_commitment != trace_statement.a_log_weight_commitment
        || transform_statement.dt_bias_commitment != trace_statement.dt_bias_commitment
    {
        return Err(
            "DeltaRecurrence transform coverage source commitments do not match trace statement"
                .to_string(),
        );
    }
    if transform_statement.scaled_query_commitment != arithmetic_statement.scaled_query_commitment
        || transform_statement.normalized_key_commitment
            != arithmetic_statement.normalized_key_commitment
        || transform_statement.decay_commitment != arithmetic_statement.decay_commitment
        || transform_statement.beta_commitment != arithmetic_statement.beta_commitment
    {
        return Err(
            "DeltaRecurrence transform coverage target commitments do not match arithmetic statement"
                .to_string(),
        );
    }
    Ok(())
}

fn qwen35_reject_extra_inventory_values(
    label: &str,
    values: &HashMap<String, Qwen35TypedWitnessCapturedValue>,
    expected: &HashSet<String>,
    manifest_hash: FieldElement,
) -> Result<(), String> {
    for source in values.keys() {
        if !expected.contains(source) {
            return Err(format!(
                "extra {label} typed witness inventory value {source} is not required by manifest 0x{manifest_hash:x}"
            ));
        }
    }
    Ok(())
}

fn qwen35_linear_attn_tensor_source_body(layer_idx: usize, name: &str) -> String {
    format!("model.language_model.layers.{layer_idx}.linear_attn.{name}")
}

fn qwen35_linear_attn_state_source_body(layer_idx: usize, name: &str) -> String {
    format!("model.language_model.layers.{layer_idx}.linear_attn:{name}")
}

fn qwen35_safetensors_vector<'a>(
    inventory: &'a Qwen35TypedWitnessSourceInventory,
    body: &str,
) -> Result<&'a [M31], String> {
    match inventory.safetensors.get(body) {
        Some(Qwen35TypedWitnessCapturedValue::Vector(values)) => Ok(values),
        Some(_) => Err(format!(
            "safetensors typed runtime statement source {body} is not a vector"
        )),
        None => Err(format!(
            "missing safetensors typed runtime statement source {body}"
        )),
    }
}

fn qwen35_typed_witness_source_kind(source: &str) -> Result<Qwen35TypedWitnessSourceKind, String> {
    let Some((prefix, rest)) = source.split_once(':') else {
        return Err(format!("typed witness source {source} has no prefix"));
    };
    if rest.is_empty() {
        return Err(format!("typed witness source {source} has empty body"));
    }
    match prefix {
        "runtime" => Ok(Qwen35TypedWitnessSourceKind::Runtime),
        "safetensors" => Ok(Qwen35TypedWitnessSourceKind::Safetensors),
        "conversation-state" => Ok(Qwen35TypedWitnessSourceKind::ConversationState),
        "statement" => Ok(Qwen35TypedWitnessSourceKind::Statement),
        _ => Err(format!(
            "typed witness source {source} has unsupported prefix {prefix}"
        )),
    }
}

fn qwen35_typed_witness_source_body(source: &str) -> Result<&str, String> {
    let Some((_, rest)) = source.split_once(':') else {
        return Err(format!("typed witness source {source} has no prefix"));
    };
    if rest.is_empty() {
        return Err(format!("typed witness source {source} has empty body"));
    }
    Ok(rest)
}

fn qwen35_validate_captured_value_shape(
    root: &Qwen35TypedWitnessRoot,
    value: &Qwen35TypedWitnessCapturedValue,
) -> Result<(), String> {
    qwen35_validate_captured_value_matches_shape(
        &format!("typed witness root {}", root.name),
        root.shape,
        value,
    )
}

fn qwen35_validate_captured_value_shape_for_source(
    source: &str,
    shape: Qwen35TensorShape,
    value: &Qwen35TypedWitnessCapturedValue,
) -> Result<(), String> {
    qwen35_validate_captured_value_matches_shape(
        &format!("typed witness source {source}"),
        shape,
        value,
    )
}

fn qwen35_validate_captured_value_matches_shape(
    context: &str,
    shape: Qwen35TensorShape,
    value: &Qwen35TypedWitnessCapturedValue,
) -> Result<(), String> {
    match (shape, value) {
        (Qwen35TensorShape::Vector(expected), Qwen35TypedWitnessCapturedValue::Vector(values)) => {
            if values.len() != expected {
                return Err(format!(
                    "{context} vector length {} != expected {}",
                    values.len(),
                    expected
                ));
            }
        }
        (Qwen35TensorShape::Matrix(expected), Qwen35TypedWitnessCapturedValue::Matrix(matrix)) => {
            if matrix.rows != expected.rows || matrix.cols != expected.cols {
                return Err(format!(
                    "{context} matrix shape {}x{} != expected {}x{}",
                    matrix.rows, matrix.cols, expected.rows, expected.cols
                ));
            }
        }
        (
            Qwen35TensorShape::Tensor3D(expected),
            Qwen35TypedWitnessCapturedValue::Vector(values),
        ) => {
            let expected_len = expected.outer * expected.middle * expected.inner;
            if values.len() != expected_len {
                return Err(format!(
                    "{context} flattened tensor length {} != expected {}",
                    values.len(),
                    expected_len
                ));
            }
        }
        (Qwen35TensorShape::Vector(0), Qwen35TypedWitnessCapturedValue::LookupTable { .. }) => {}
        _ => {
            return Err(format!(
                "{context} captured value kind does not match {:?}",
                shape
            ));
        }
    }
    Ok(())
}

fn qwen35_validate_matrix_shape(
    context: &str,
    matrix: &M31Matrix,
    expected: Qwen35Tensor2DShape,
) -> Result<(), String> {
    if matrix.rows != expected.rows || matrix.cols != expected.cols {
        return Err(format!(
            "{context} matrix shape {}x{} != expected {}x{}",
            matrix.rows, matrix.cols, expected.rows, expected.cols
        ));
    }
    Ok(())
}

fn qwen35_witness_root_from_trace(
    layer_idx: usize,
    statement_kind: Qwen35TypedProofStatementKind,
    stage_idx: usize,
    root: &Qwen35TraceRootContract,
    source: String,
) -> Qwen35TypedWitnessRoot {
    let kind = match root.role {
        Qwen35TraceRootRole::ModelWeight => Qwen35TypedWitnessRootKind::ModelWeight,
        Qwen35TraceRootRole::ProducerActivation | Qwen35TraceRootRole::ConsumerActivation => {
            Qwen35TypedWitnessRootKind::Activation
        }
    };
    Qwen35TypedWitnessRoot {
        layer_idx,
        statement_kind,
        stage_idx,
        name: root.name.to_string(),
        kind,
        shape: root.shape,
        source,
        trace_root_contract_hash: root.contract_hash(),
    }
}

fn qwen35_synthetic_witness_root_hash(
    name: &str,
    kind: Qwen35TypedWitnessRootKind,
    shape: Qwen35TensorShape,
    source: &str,
) -> FieldElement {
    let dims = shape.dimensions();
    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(DOMAIN_QWEN35_TYPED_WITNESS_MANIFEST),
        FieldElement::from(kind.hash_tag()),
        FieldElement::from(shape.hash_tag()),
        FieldElement::from(dims[0] as u64),
        FieldElement::from(dims[1] as u64),
        FieldElement::from(dims[2] as u64),
        qwen35_hash_str(name),
        qwen35_hash_str(source),
    ])
}

fn qwen35_synthetic_witness_root(
    layer_idx: usize,
    statement_kind: Qwen35TypedProofStatementKind,
    stage_idx: usize,
    name: &str,
    kind: Qwen35TypedWitnessRootKind,
    shape: Qwen35TensorShape,
    source: String,
) -> Qwen35TypedWitnessRoot {
    let trace_root_contract_hash = qwen35_synthetic_witness_root_hash(name, kind, shape, &source);
    Qwen35TypedWitnessRoot {
        layer_idx,
        statement_kind,
        stage_idx,
        name: name.to_string(),
        kind,
        shape,
        source,
        trace_root_contract_hash,
    }
}

impl Qwen35GatedDeltaNetStageContract {
    pub fn contract_hash(&self, layer_idx: usize, seq_len: usize) -> FieldElement {
        let mut felts = vec![
            FieldElement::from(DOMAIN_QWEN35_GDN_STAGE),
            FieldElement::from(layer_idx as u64),
            FieldElement::from(seq_len as u64),
            FieldElement::from(self.stage_idx as u64),
            FieldElement::from(self.kind.hash_tag()),
            FieldElement::from(self.status.hash_tag()),
        ];
        for tensor in self.inputs.iter().chain(self.outputs.iter()) {
            let dims = tensor.shape.dimensions();
            felts.extend([
                FieldElement::from(tensor.shape.hash_tag()),
                FieldElement::from(dims[0] as u64),
                FieldElement::from(dims[1] as u64),
                FieldElement::from(dims[2] as u64),
            ]);
        }
        starknet_crypto::poseidon_hash_many(&felts)
    }
}

impl Qwen35DepthwiseConv1dTraceBindingContract {
    pub fn contract_hash(&self) -> FieldElement {
        starknet_crypto::poseidon_hash_many(&[
            FieldElement::from(DOMAIN_QWEN35_TRACE_BINDING),
            FieldElement::from(self.layer_idx as u64),
            FieldElement::from(self.seq_len as u64),
            FieldElement::from(self.channels as u64),
            FieldElement::from(self.kernel as u64),
            FieldElement::from(self.stage_idx as u64),
            FieldElement::from(self.producer_stage_idx as u64),
            FieldElement::from(self.consumer_stage_idx as u64),
            self.input_root.contract_hash(),
            self.weight_root.contract_hash(),
            self.output_root.contract_hash(),
            self.stage_contract_hash,
            self.air_contract_hash,
        ])
    }

    pub fn expected_statement_hash(
        &self,
        input_commitment: FieldElement,
        weight_commitment: FieldElement,
        output_commitment: FieldElement,
    ) -> FieldElement {
        starknet_crypto::poseidon_hash_many(&[
            FieldElement::from(DOMAIN_QWEN35_DEPTHWISE_CONV1D_STATEMENT),
            FieldElement::from(self.layer_idx as u64),
            FieldElement::from(self.seq_len as u64),
            FieldElement::from(self.channels as u64),
            FieldElement::from(self.kernel as u64),
            input_commitment,
            weight_commitment,
            output_commitment,
        ])
    }

    pub fn validate_statement(
        &self,
        statement_layer_idx: usize,
        statement_seq_len: usize,
        statement_channels: usize,
        statement_kernel: usize,
        input_commitment: FieldElement,
        weight_commitment: FieldElement,
        output_commitment: FieldElement,
        statement_hash: FieldElement,
    ) -> Result<(), String> {
        if statement_layer_idx != self.layer_idx {
            return Err(format!(
                "DepthwiseConv1D statement layer {} != contract layer {}",
                statement_layer_idx, self.layer_idx
            ));
        }
        if statement_seq_len != self.seq_len {
            return Err(format!(
                "DepthwiseConv1D statement seq_len {} != contract seq_len {}",
                statement_seq_len, self.seq_len
            ));
        }
        if statement_channels != self.channels {
            return Err(format!(
                "DepthwiseConv1D statement channels {} != contract channels {}",
                statement_channels, self.channels
            ));
        }
        if statement_kernel != self.kernel {
            return Err(format!(
                "DepthwiseConv1D statement kernel {} != contract kernel {}",
                statement_kernel, self.kernel
            ));
        }

        let expected_hash =
            self.expected_statement_hash(input_commitment, weight_commitment, output_commitment);
        if statement_hash != expected_hash {
            return Err(format!(
                "DepthwiseConv1D statement hash 0x{statement_hash:x} != expected 0x{expected_hash:x}"
            ));
        }

        Ok(())
    }
}

impl Qwen35DeltaRecurrenceTraceBindingContract {
    pub fn contract_hash(&self) -> FieldElement {
        starknet_crypto::poseidon_hash_many(&[
            FieldElement::from(DOMAIN_QWEN35_TRACE_BINDING),
            FieldElement::from(DOMAIN_QWEN35_DELTA_RECURRENCE),
            FieldElement::from(self.layer_idx as u64),
            FieldElement::from(self.seq_len as u64),
            FieldElement::from(self.query_width as u64),
            FieldElement::from(self.key_width as u64),
            FieldElement::from(self.value_width as u64),
            FieldElement::from(self.state_rows as u64),
            FieldElement::from(self.value_head_dim as u64),
            FieldElement::from(self.stage_idx as u64),
            FieldElement::from(self.qkv_split_stage_idx as u64),
            FieldElement::from(self.ab_projection_stage_idx as u64),
            FieldElement::from(self.consumer_stage_idx as u64),
            self.query_root.contract_hash(),
            self.key_root.contract_hash(),
            self.projected_value_root.contract_hash(),
            self.a_gate_root.contract_hash(),
            self.b_gate_root.contract_hash(),
            self.a_log_weight_root.contract_hash(),
            self.dt_bias_root.contract_hash(),
            self.output_root.contract_hash(),
            self.stage_contract_hash,
        ])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn expected_statement_hash(
        &self,
        query_commitment: FieldElement,
        key_commitment: FieldElement,
        projected_value_commitment: FieldElement,
        a_gate_commitment: FieldElement,
        b_gate_commitment: FieldElement,
        a_log_weight_commitment: FieldElement,
        dt_bias_commitment: FieldElement,
        mode_tag: u64,
        air_spec_hash: FieldElement,
        initial_recurrent_state_commitment: FieldElement,
        final_recurrent_state_commitment: FieldElement,
        output_commitment: FieldElement,
    ) -> FieldElement {
        starknet_crypto::poseidon_hash_many(&[
            FieldElement::from(DOMAIN_QWEN35_DELTA_RECURRENCE_STATEMENT),
            FieldElement::from(self.layer_idx as u64),
            FieldElement::from(self.seq_len as u64),
            FieldElement::from(self.query_width as u64),
            FieldElement::from(self.key_width as u64),
            FieldElement::from(self.value_width as u64),
            FieldElement::from(self.state_rows as u64),
            FieldElement::from(self.value_head_dim as u64),
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
    pub fn validate_statement(
        &self,
        statement_layer_idx: usize,
        statement_seq_len: usize,
        statement_query_width: usize,
        statement_key_width: usize,
        statement_value_width: usize,
        statement_state_rows: usize,
        statement_value_head_dim: usize,
        query_commitment: FieldElement,
        key_commitment: FieldElement,
        projected_value_commitment: FieldElement,
        a_gate_commitment: FieldElement,
        b_gate_commitment: FieldElement,
        a_log_weight_commitment: FieldElement,
        dt_bias_commitment: FieldElement,
        statement_mode_tag: u64,
        statement_air_spec_hash: FieldElement,
        initial_recurrent_state_commitment: FieldElement,
        final_recurrent_state_commitment: FieldElement,
        output_commitment: FieldElement,
        statement_hash: FieldElement,
    ) -> Result<(), String> {
        if statement_layer_idx != self.layer_idx {
            return Err(format!(
                "DeltaRecurrence statement layer {} != contract layer {}",
                statement_layer_idx, self.layer_idx
            ));
        }
        if statement_seq_len != self.seq_len {
            return Err(format!(
                "DeltaRecurrence statement seq_len {} != contract seq_len {}",
                statement_seq_len, self.seq_len
            ));
        }
        if statement_query_width != self.query_width
            || statement_key_width != self.key_width
            || statement_value_width != self.value_width
            || statement_state_rows != self.state_rows
            || statement_value_head_dim != self.value_head_dim
        {
            return Err("DeltaRecurrence statement dimensions do not match contract".to_string());
        }
        if !matches!(statement_mode_tag, 1 | 2) {
            return Err(format!(
                "DeltaRecurrence statement mode tag {} is not supported",
                statement_mode_tag
            ));
        }
        if statement_air_spec_hash == FieldElement::ZERO {
            return Err("DeltaRecurrence AIR spec hash must be non-zero".to_string());
        }

        let expected_hash = self.expected_statement_hash(
            query_commitment,
            key_commitment,
            projected_value_commitment,
            a_gate_commitment,
            b_gate_commitment,
            a_log_weight_commitment,
            dt_bias_commitment,
            statement_mode_tag,
            statement_air_spec_hash,
            initial_recurrent_state_commitment,
            final_recurrent_state_commitment,
            output_commitment,
        );
        if statement_hash != expected_hash {
            return Err(format!(
                "DeltaRecurrence statement hash 0x{statement_hash:x} != expected 0x{expected_hash:x}"
            ));
        }

        Ok(())
    }
}

impl Qwen35NormAndZGateTraceBindingContract {
    pub fn contract_hash(&self) -> FieldElement {
        starknet_crypto::poseidon_hash_many(&[
            FieldElement::from(DOMAIN_QWEN35_TRACE_BINDING),
            FieldElement::from(DOMAIN_QWEN35_NORM_AND_Z_GATE),
            FieldElement::from(self.layer_idx as u64),
            FieldElement::from(self.seq_len as u64),
            FieldElement::from(self.width as u64),
            FieldElement::from(self.norm_width as u64),
            FieldElement::from(self.stage_idx as u64),
            FieldElement::from(self.delta_recurrence_stage_idx as u64),
            FieldElement::from(self.z_projection_stage_idx as u64),
            FieldElement::from(self.consumer_stage_idx as u64),
            self.attended_value_root.contract_hash(),
            self.norm_weight_root.contract_hash(),
            self.z_gate_root.contract_hash(),
            self.output_root.contract_hash(),
            self.stage_contract_hash,
        ])
    }

    pub fn expected_statement_hash(
        &self,
        table_log_size: u32,
        trace_checksum: stwo::core::fields::m31::M31,
        table_commitment: FieldElement,
        attended_value_commitment: FieldElement,
        norm_weight_commitment: FieldElement,
        z_gate_commitment: FieldElement,
        output_commitment: FieldElement,
    ) -> FieldElement {
        let value_heads = self.width / self.norm_width;
        starknet_crypto::poseidon_hash_many(&[
            FieldElement::from(DOMAIN_QWEN35_NORM_AND_Z_GATE_STATEMENT),
            FieldElement::from(self.layer_idx as u64),
            FieldElement::from(self.seq_len as u64),
            FieldElement::from(value_heads as u64),
            FieldElement::from(self.norm_width as u64),
            FieldElement::from(table_log_size as u64),
            FieldElement::from(trace_checksum.0 as u64),
            table_commitment,
            attended_value_commitment,
            norm_weight_commitment,
            z_gate_commitment,
            output_commitment,
        ])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_statement(
        &self,
        statement_layer_idx: usize,
        statement_seq_len: usize,
        statement_value_heads: usize,
        statement_head_dim: usize,
        statement_table_log_size: u32,
        statement_trace_checksum: stwo::core::fields::m31::M31,
        statement_table_commitment: FieldElement,
        attended_value_commitment: FieldElement,
        norm_weight_commitment: FieldElement,
        z_gate_commitment: FieldElement,
        output_commitment: FieldElement,
        statement_hash: FieldElement,
    ) -> Result<(), String> {
        if statement_layer_idx != self.layer_idx {
            return Err(format!(
                "NormAndZGate statement layer {} != contract layer {}",
                statement_layer_idx, self.layer_idx
            ));
        }
        if statement_seq_len != self.seq_len {
            return Err(format!(
                "NormAndZGate statement seq_len {} != contract seq_len {}",
                statement_seq_len, self.seq_len
            ));
        }
        let expected_value_heads = self.width / self.norm_width;
        if statement_value_heads != expected_value_heads || statement_head_dim != self.norm_width {
            return Err("NormAndZGate statement dimensions do not match contract".to_string());
        }
        if statement_table_log_size == 0 || statement_table_commitment == FieldElement::ZERO {
            return Err("NormAndZGate statement table binding must be non-zero".to_string());
        }

        let expected_hash = self.expected_statement_hash(
            statement_table_log_size,
            statement_trace_checksum,
            statement_table_commitment,
            attended_value_commitment,
            norm_weight_commitment,
            z_gate_commitment,
            output_commitment,
        );
        if statement_hash != expected_hash {
            return Err(format!(
                "NormAndZGate statement hash 0x{statement_hash:x} != expected 0x{expected_hash:x}"
            ));
        }

        Ok(())
    }
}

impl Qwen35TypedProofLedger {
    pub fn new(
        architecture_contract_hash: FieldElement,
        seq_len: usize,
        expected_depthwise_conv1d_statements: usize,
        expected_delta_recurrence_statements: usize,
        expected_norm_and_z_gate_statements: usize,
    ) -> Self {
        Self {
            architecture_contract_hash,
            seq_len,
            expected_depthwise_conv1d_statements,
            expected_delta_recurrence_statements,
            expected_norm_and_z_gate_statements,
            statements: Vec::new(),
        }
    }

    pub fn record_depthwise_conv1d_statement(
        &mut self,
        binding: &Qwen35DepthwiseConv1dTraceBindingContract,
        statement_layer_idx: usize,
        statement_seq_len: usize,
        statement_channels: usize,
        statement_kernel: usize,
        input_commitment: FieldElement,
        weight_commitment: FieldElement,
        output_commitment: FieldElement,
        statement_hash: FieldElement,
    ) -> Result<(), String> {
        if binding.seq_len != self.seq_len {
            return Err(format!(
                "DepthwiseConv1D binding seq_len {} != ledger seq_len {}",
                binding.seq_len, self.seq_len
            ));
        }
        binding.validate_statement(
            statement_layer_idx,
            statement_seq_len,
            statement_channels,
            statement_kernel,
            input_commitment,
            weight_commitment,
            output_commitment,
            statement_hash,
        )?;
        let duplicate = self.statements.iter().any(|statement| {
            statement.kind == Qwen35TypedProofStatementKind::DepthwiseConv1d
                && statement.layer_idx == binding.layer_idx
                && statement.stage_idx == binding.stage_idx
        });
        if duplicate {
            return Err(format!(
                "duplicate DepthwiseConv1D statement for layer {} stage {}",
                binding.layer_idx, binding.stage_idx
            ));
        }

        self.statements.push(Qwen35TypedProofStatement {
            kind: Qwen35TypedProofStatementKind::DepthwiseConv1d,
            layer_idx: binding.layer_idx,
            seq_len: binding.seq_len,
            stage_idx: binding.stage_idx,
            trace_binding_hash: binding.contract_hash(),
            statement_hash,
            transform_statement_hash: None,
            transform_nonlinear_statement_hash: None,
            transform_nonlinear_q_norm_statement_hash: None,
            transform_nonlinear_k_norm_statement_hash: None,
            transform_nonlinear_beta_sigmoid_statement_hash: None,
            transform_nonlinear_decay_statement_hash: None,
            arithmetic_statement_hash: None,
            initial_recurrent_state_commitment: None,
            final_recurrent_state_commitment: None,
        });
        Ok(())
    }

    pub fn record_depthwise_conv1d_proof(
        &mut self,
        binding: &Qwen35DepthwiseConv1dTraceBindingContract,
        statement: &crate::components::qwen35_depthwise_conv1d::Qwen35DepthwiseConv1dStatement,
    ) -> Result<(), String> {
        self.record_depthwise_conv1d_statement(
            binding,
            statement.layer_idx,
            statement.seq_len,
            statement.channels,
            statement.kernel,
            statement.input_commitment,
            statement.weight_commitment,
            statement.output_commitment,
            statement.statement_hash,
        )
    }

    pub fn record_depthwise_conv1d_air_proof(
        &mut self,
        binding: &Qwen35DepthwiseConv1dTraceBindingContract,
        proof: &crate::components::qwen35_depthwise_conv1d::Qwen35DepthwiseConv1dProof<
            <stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel as stwo::core::channel::MerkleChannel>::H,
        >,
    ) -> Result<(), String> {
        crate::components::qwen35_depthwise_conv1d::verify_qwen35_depthwise_conv1d_air(proof)
            .map_err(|err| err.to_string())?;
        self.record_depthwise_conv1d_proof(binding, &proof.statement)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_delta_recurrence_statement(
        &mut self,
        binding: &Qwen35DeltaRecurrenceTraceBindingContract,
        statement_layer_idx: usize,
        statement_seq_len: usize,
        statement_query_width: usize,
        statement_key_width: usize,
        statement_value_width: usize,
        statement_state_rows: usize,
        statement_value_head_dim: usize,
        query_commitment: FieldElement,
        key_commitment: FieldElement,
        projected_value_commitment: FieldElement,
        a_gate_commitment: FieldElement,
        b_gate_commitment: FieldElement,
        a_log_weight_commitment: FieldElement,
        dt_bias_commitment: FieldElement,
        mode_tag: u64,
        air_spec_hash: FieldElement,
        initial_recurrent_state_commitment: FieldElement,
        final_recurrent_state_commitment: FieldElement,
        output_commitment: FieldElement,
        statement_hash: FieldElement,
    ) -> Result<(), String> {
        if binding.seq_len != self.seq_len {
            return Err(format!(
                "DeltaRecurrence binding seq_len {} != ledger seq_len {}",
                binding.seq_len, self.seq_len
            ));
        }
        binding.validate_statement(
            statement_layer_idx,
            statement_seq_len,
            statement_query_width,
            statement_key_width,
            statement_value_width,
            statement_state_rows,
            statement_value_head_dim,
            query_commitment,
            key_commitment,
            projected_value_commitment,
            a_gate_commitment,
            b_gate_commitment,
            a_log_weight_commitment,
            dt_bias_commitment,
            mode_tag,
            air_spec_hash,
            initial_recurrent_state_commitment,
            final_recurrent_state_commitment,
            output_commitment,
            statement_hash,
        )?;
        let duplicate = self.statements.iter().any(|statement| {
            statement.kind == Qwen35TypedProofStatementKind::DeltaRecurrence
                && statement.layer_idx == binding.layer_idx
                && statement.stage_idx == binding.stage_idx
        });
        if duplicate {
            return Err(format!(
                "duplicate DeltaRecurrence statement for layer {} stage {}",
                binding.layer_idx, binding.stage_idx
            ));
        }

        self.statements.push(Qwen35TypedProofStatement {
            kind: Qwen35TypedProofStatementKind::DeltaRecurrence,
            layer_idx: binding.layer_idx,
            seq_len: binding.seq_len,
            stage_idx: binding.stage_idx,
            trace_binding_hash: binding.contract_hash(),
            statement_hash,
            transform_statement_hash: None,
            transform_nonlinear_statement_hash: None,
            transform_nonlinear_q_norm_statement_hash: None,
            transform_nonlinear_k_norm_statement_hash: None,
            transform_nonlinear_beta_sigmoid_statement_hash: None,
            transform_nonlinear_decay_statement_hash: None,
            arithmetic_statement_hash: None,
            initial_recurrent_state_commitment: Some(initial_recurrent_state_commitment),
            final_recurrent_state_commitment: Some(final_recurrent_state_commitment),
        });
        Ok(())
    }

    pub fn record_delta_recurrence_proof(
        &mut self,
        binding: &Qwen35DeltaRecurrenceTraceBindingContract,
        statement: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceStatement,
    ) -> Result<(), String> {
        self.record_delta_recurrence_statement(
            binding,
            statement.layer_idx,
            statement.seq_len,
            statement.query_width,
            statement.key_width,
            statement.value_width,
            statement.state_rows,
            statement.value_head_dim,
            statement.query_commitment,
            statement.key_commitment,
            statement.projected_value_commitment,
            statement.a_gate_commitment,
            statement.b_gate_commitment,
            statement.a_log_weight_commitment,
            statement.dt_bias_commitment,
            statement.mode.as_u64(),
            statement.air_spec_hash,
            statement.initial_recurrent_state_commitment,
            statement.final_recurrent_state_commitment,
            statement.output_commitment,
            statement.statement_hash,
        )
    }

    pub fn record_norm_and_z_gate_proof(
        &mut self,
        binding: &Qwen35NormAndZGateTraceBindingContract,
        statement: &crate::components::qwen35_norm_and_z_gate::Qwen35NormAndZGateStatement,
    ) -> Result<(), String> {
        if binding.seq_len != self.seq_len {
            return Err(format!(
                "NormAndZGate binding seq_len {} != ledger seq_len {}",
                binding.seq_len, self.seq_len
            ));
        }
        binding.validate_statement(
            statement.layer_idx,
            statement.seq_len,
            statement.value_heads,
            statement.head_dim,
            statement.table_log_size,
            statement.trace_checksum,
            statement.table_commitment,
            statement.attended_value_commitment,
            statement.norm_weight_commitment,
            statement.z_gate_commitment,
            statement.output_commitment,
            statement.statement_hash,
        )?;
        if self.statements.iter().any(|recorded| {
            recorded.kind == Qwen35TypedProofStatementKind::NormAndZGate
                && recorded.layer_idx == binding.layer_idx
                && recorded.stage_idx == binding.stage_idx
        }) {
            return Err(format!(
                "duplicate NormAndZGate statement for layer {} stage {}",
                binding.layer_idx, binding.stage_idx
            ));
        }
        self.statements.push(Qwen35TypedProofStatement {
            kind: Qwen35TypedProofStatementKind::NormAndZGate,
            layer_idx: binding.layer_idx,
            seq_len: binding.seq_len,
            stage_idx: binding.stage_idx,
            trace_binding_hash: binding.contract_hash(),
            statement_hash: statement.statement_hash,
            transform_statement_hash: None,
            transform_nonlinear_statement_hash: None,
            transform_nonlinear_q_norm_statement_hash: None,
            transform_nonlinear_k_norm_statement_hash: None,
            transform_nonlinear_beta_sigmoid_statement_hash: None,
            transform_nonlinear_decay_statement_hash: None,
            arithmetic_statement_hash: None,
            initial_recurrent_state_commitment: None,
            final_recurrent_state_commitment: None,
        });
        Ok(())
    }

    pub fn record_norm_and_z_gate_air_proof(
        &mut self,
        binding: &Qwen35NormAndZGateTraceBindingContract,
        proof: &crate::components::qwen35_norm_and_z_gate::Qwen35NormAndZGateProof<
            <stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel as stwo::core::channel::MerkleChannel>::H,
        >,
    ) -> Result<(), String> {
        crate::components::qwen35_norm_and_z_gate::verify_qwen35_norm_and_z_gate_air(proof)
            .map_err(|err| err.to_string())?;
        self.record_norm_and_z_gate_proof(binding, &proof.statement)
    }

    pub fn record_delta_recurrence_trace_binding_witness(
        &mut self,
        binding: &Qwen35DeltaRecurrenceTraceBindingContract,
        inputs: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceInputs<'_>,
        witness: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceTraceBindingWitness,
    ) -> Result<(), String> {
        crate::components::qwen35_delta_recurrence::verify_qwen35_delta_recurrence_trace_binding_witness(
            inputs, witness,
        )?;
        let statement =
            crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_statement(
                binding.layer_idx,
                inputs,
            )?;
        self.record_delta_recurrence_proof(binding, &statement)
    }

    pub fn record_delta_recurrence_trace_binding_air_proof(
        &mut self,
        binding: &Qwen35DeltaRecurrenceTraceBindingContract,
        proof: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceTraceBindingProof<
            <stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel as stwo::core::channel::MerkleChannel>::H,
        >,
    ) -> Result<(), String> {
        crate::components::qwen35_delta_recurrence::verify_qwen35_delta_recurrence_trace_binding_air(
            proof,
        )
        .map_err(|err| err.to_string())?;
        self.record_delta_recurrence_proof(binding, &proof.statement)
    }

    pub fn record_delta_recurrence_active_air_proofs(
        &mut self,
        binding: &Qwen35DeltaRecurrenceTraceBindingContract,
        trace_proof: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceTraceBindingProof<
            <stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel as stwo::core::channel::MerkleChannel>::H,
        >,
        arithmetic_proof: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceArithmeticProof<
            <stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel as stwo::core::channel::MerkleChannel>::H,
        >,
    ) -> Result<(), String> {
        crate::components::qwen35_delta_recurrence::verify_qwen35_delta_recurrence_trace_binding_air(
            trace_proof,
        )
        .map_err(|err| err.to_string())?;
        crate::components::qwen35_delta_recurrence::verify_qwen35_delta_recurrence_arithmetic_air(
            arithmetic_proof,
        )
        .map_err(|err| err.to_string())?;

        let trace_statement = &trace_proof.statement;
        let arithmetic_statement = &arithmetic_proof.statement;
        if arithmetic_statement.layer_idx != trace_statement.layer_idx {
            return Err(format!(
                "DeltaRecurrence arithmetic statement layer {} != trace statement layer {}",
                arithmetic_statement.layer_idx, trace_statement.layer_idx
            ));
        }
        if arithmetic_statement.seq_len != trace_statement.seq_len
            || arithmetic_statement.query_width != trace_statement.query_width
            || arithmetic_statement.key_width != trace_statement.key_width
            || arithmetic_statement.value_width != trace_statement.value_width
            || arithmetic_statement.state_rows != trace_statement.state_rows
            || arithmetic_statement.value_head_dim != trace_statement.value_head_dim
        {
            return Err(
                "DeltaRecurrence arithmetic statement dimensions do not match trace statement"
                    .to_string(),
            );
        }
        if arithmetic_statement.mode != trace_statement.mode {
            return Err("DeltaRecurrence arithmetic statement mode mismatch".to_string());
        }
        if arithmetic_statement.air_spec_hash != trace_statement.air_spec_hash {
            return Err("DeltaRecurrence arithmetic AIR spec hash mismatch".to_string());
        }
        if arithmetic_statement.projected_value_commitment
            != trace_statement.projected_value_commitment
            || arithmetic_statement.initial_recurrent_state_commitment
                != trace_statement.initial_recurrent_state_commitment
            || arithmetic_statement.final_recurrent_state_commitment
                != trace_statement.final_recurrent_state_commitment
            || arithmetic_statement.output_commitment != trace_statement.output_commitment
        {
            return Err(
                "DeltaRecurrence arithmetic statement IO/state commitments do not match trace statement"
                    .to_string(),
            );
        }

        let expected_arithmetic_hash =
            crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_arithmetic_statement_hash(
                arithmetic_statement.layer_idx,
                arithmetic_statement.seq_len,
                arithmetic_statement.query_width,
                arithmetic_statement.key_width,
                arithmetic_statement.value_width,
                arithmetic_statement.state_rows,
                arithmetic_statement.value_head_dim,
                arithmetic_statement.mode.as_u64(),
                arithmetic_statement.air_spec_hash,
                arithmetic_statement.scaled_query_commitment,
                arithmetic_statement.normalized_key_commitment,
                arithmetic_statement.projected_value_commitment,
                arithmetic_statement.decay_commitment,
                arithmetic_statement.beta_commitment,
                arithmetic_statement.initial_recurrent_state_commitment,
                arithmetic_statement.final_recurrent_state_commitment,
                arithmetic_statement.output_commitment,
            );
        if arithmetic_statement.statement_hash != expected_arithmetic_hash {
            return Err(format!(
                "DeltaRecurrence arithmetic statement hash 0x{:x} != expected 0x{:x}",
                arithmetic_statement.statement_hash, expected_arithmetic_hash
            ));
        }

        self.record_delta_recurrence_proof(binding, trace_statement)?;
        let recorded = self
            .statements
            .iter_mut()
            .find(|statement| {
                statement.kind == Qwen35TypedProofStatementKind::DeltaRecurrence
                    && statement.layer_idx == binding.layer_idx
                    && statement.stage_idx == binding.stage_idx
            })
            .ok_or_else(|| "recorded DeltaRecurrence statement missing".to_string())?;
        recorded.arithmetic_statement_hash = Some(arithmetic_statement.statement_hash);
        Ok(())
    }

    pub fn record_delta_recurrence_transform_binding_witness(
        &mut self,
        binding: &Qwen35DeltaRecurrenceTraceBindingContract,
        trace_statement: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceStatement,
        arithmetic_statement: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceArithmeticStatement,
        transform_inputs: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceTransformInputs<'_>,
        transform_witness: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceTransformBindingWitness,
    ) -> Result<(), String> {
        crate::components::qwen35_delta_recurrence::verify_qwen35_delta_recurrence_transform_binding_witness(
            transform_inputs,
            transform_witness,
        )?;
        let transform_statement =
            crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_transform_statement(
                binding.layer_idx,
                transform_inputs,
            )?;
        self.record_delta_recurrence_transform_statement(
            binding,
            trace_statement,
            arithmetic_statement,
            &transform_statement,
        )
    }

    pub fn record_delta_recurrence_transform_binding_air_proof(
        &mut self,
        binding: &Qwen35DeltaRecurrenceTraceBindingContract,
        trace_statement: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceStatement,
        arithmetic_statement: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceArithmeticStatement,
        transform_proof: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceTransformBindingProof<
            <stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel as stwo::core::channel::MerkleChannel>::H,
        >,
    ) -> Result<(), String> {
        crate::components::qwen35_delta_recurrence::verify_qwen35_delta_recurrence_transform_binding_air(
            transform_proof,
        )
        .map_err(|err| err.to_string())?;
        self.record_delta_recurrence_transform_statement(
            binding,
            trace_statement,
            arithmetic_statement,
            &transform_proof.statement,
        )
    }

    fn record_delta_recurrence_transform_statement(
        &mut self,
        binding: &Qwen35DeltaRecurrenceTraceBindingContract,
        trace_statement: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceStatement,
        arithmetic_statement: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceArithmeticStatement,
        transform_statement: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceTransformStatement,
    ) -> Result<(), String> {
        if transform_statement.layer_idx != trace_statement.layer_idx
            || transform_statement.layer_idx != arithmetic_statement.layer_idx
        {
            return Err("DeltaRecurrence transform statement layer mismatch".to_string());
        }
        if transform_statement.seq_len != trace_statement.seq_len
            || transform_statement.seq_len != arithmetic_statement.seq_len
            || transform_statement.query_width != trace_statement.query_width
            || transform_statement.query_width != arithmetic_statement.query_width
            || transform_statement.key_width != trace_statement.key_width
            || transform_statement.key_width != arithmetic_statement.key_width
            || transform_statement.state_rows != trace_statement.state_rows
            || transform_statement.state_rows != arithmetic_statement.state_rows
            || transform_statement.value_head_dim != trace_statement.value_head_dim
            || transform_statement.value_head_dim != arithmetic_statement.value_head_dim
        {
            return Err(
                "DeltaRecurrence transform statement dimensions do not match trace/arithmetic statements"
                    .to_string(),
            );
        }
        if transform_statement.mode != trace_statement.mode
            || transform_statement.mode != arithmetic_statement.mode
        {
            return Err("DeltaRecurrence transform statement mode mismatch".to_string());
        }
        if transform_statement.air_spec_hash != trace_statement.air_spec_hash
            || transform_statement.air_spec_hash != arithmetic_statement.air_spec_hash
        {
            return Err("DeltaRecurrence transform AIR spec hash mismatch".to_string());
        }
        if transform_statement.query_commitment != trace_statement.query_commitment
            || transform_statement.key_commitment != trace_statement.key_commitment
            || transform_statement.a_gate_commitment != trace_statement.a_gate_commitment
            || transform_statement.b_gate_commitment != trace_statement.b_gate_commitment
            || transform_statement.a_log_weight_commitment
                != trace_statement.a_log_weight_commitment
            || transform_statement.dt_bias_commitment != trace_statement.dt_bias_commitment
        {
            return Err(
                "DeltaRecurrence transform statement source commitments do not match trace statement"
                    .to_string(),
            );
        }
        if transform_statement.scaled_query_commitment
            != arithmetic_statement.scaled_query_commitment
            || transform_statement.normalized_key_commitment
                != arithmetic_statement.normalized_key_commitment
            || transform_statement.decay_commitment != arithmetic_statement.decay_commitment
            || transform_statement.beta_commitment != arithmetic_statement.beta_commitment
        {
            return Err(
                "DeltaRecurrence transform statement target commitments do not match arithmetic statement"
                    .to_string(),
            );
        }

        let expected_transform_hash =
            crate::components::qwen35_delta_recurrence::qwen35_delta_recurrence_transform_statement_hash(
                transform_statement.layer_idx,
                transform_statement.seq_len,
                transform_statement.query_width,
                transform_statement.key_width,
                transform_statement.state_rows,
                transform_statement.value_head_dim,
                transform_statement.mode.as_u64(),
                transform_statement.air_spec_hash,
                transform_statement.query_commitment,
                transform_statement.key_commitment,
                transform_statement.a_gate_commitment,
                transform_statement.b_gate_commitment,
                transform_statement.a_log_weight_commitment,
                transform_statement.dt_bias_commitment,
                transform_statement.scaled_query_commitment,
                transform_statement.normalized_key_commitment,
                transform_statement.decay_commitment,
                transform_statement.beta_commitment,
            );
        if transform_statement.statement_hash != expected_transform_hash {
            return Err(format!(
                "DeltaRecurrence transform statement hash 0x{:x} != expected 0x{:x}",
                transform_statement.statement_hash, expected_transform_hash
            ));
        }

        let recorded = self
            .statements
            .iter_mut()
            .find(|statement| {
                statement.kind == Qwen35TypedProofStatementKind::DeltaRecurrence
                    && statement.layer_idx == binding.layer_idx
                    && statement.stage_idx == binding.stage_idx
            })
            .ok_or_else(|| {
                "recorded DeltaRecurrence statement missing before transform binding".to_string()
            })?;
        recorded.transform_statement_hash = Some(transform_statement.statement_hash);
        Ok(())
    }

    pub fn record_delta_recurrence_beta_sigmoid_air_proof(
        &mut self,
        binding: &Qwen35DeltaRecurrenceTraceBindingContract,
        transform_statement: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceTransformStatement,
        beta_proof: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceBetaSigmoidProof<
            <stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel as stwo::core::channel::MerkleChannel>::H,
        >,
    ) -> Result<(), String> {
        crate::components::qwen35_delta_recurrence::verify_qwen35_delta_recurrence_beta_sigmoid_air(
            beta_proof,
        )
        .map_err(|err| err.to_string())?;

        let beta_statement = &beta_proof.statement;
        if beta_statement.layer_idx != transform_statement.layer_idx
            || beta_statement.layer_idx != binding.layer_idx
        {
            return Err("DeltaRecurrence beta sigmoid statement layer mismatch".to_string());
        }
        if beta_statement.seq_len != transform_statement.seq_len
            || beta_statement.state_rows != transform_statement.state_rows
        {
            return Err(
                "DeltaRecurrence beta sigmoid dimensions do not match transform statement"
                    .to_string(),
            );
        }
        if beta_statement.b_gate_commitment != transform_statement.b_gate_commitment
            || beta_statement.beta_commitment != transform_statement.beta_commitment
        {
            return Err(
                "DeltaRecurrence beta sigmoid commitments do not match transform statement"
                    .to_string(),
            );
        }

        let recorded = self
            .statements
            .iter_mut()
            .find(|statement| {
                statement.kind == Qwen35TypedProofStatementKind::DeltaRecurrence
                    && statement.layer_idx == binding.layer_idx
                    && statement.stage_idx == binding.stage_idx
            })
            .ok_or_else(|| {
                "recorded DeltaRecurrence statement missing before beta sigmoid proof".to_string()
            })?;
        if recorded.transform_statement_hash != Some(transform_statement.statement_hash) {
            return Err(
                "DeltaRecurrence beta sigmoid proof requires the matching transform-binding proof"
                    .to_string(),
            );
        }
        recorded.transform_nonlinear_beta_sigmoid_statement_hash =
            Some(beta_statement.statement_hash);
        Ok(())
    }

    pub fn record_delta_recurrence_norm_air_proof(
        &mut self,
        binding: &Qwen35DeltaRecurrenceTraceBindingContract,
        transform_statement: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceTransformStatement,
        norm_proof: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceNormProof<
            <stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel as stwo::core::channel::MerkleChannel>::H,
        >,
    ) -> Result<(), String> {
        use crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceNormKind;

        crate::components::qwen35_delta_recurrence::verify_qwen35_delta_recurrence_norm_air(
            norm_proof,
        )
        .map_err(|err| err.to_string())?;

        let norm_statement = &norm_proof.statement;
        if norm_statement.layer_idx != transform_statement.layer_idx
            || norm_statement.layer_idx != binding.layer_idx
        {
            return Err(format!(
                "DeltaRecurrence {} norm statement layer mismatch",
                norm_statement.kind.as_str()
            ));
        }
        if norm_statement.seq_len != transform_statement.seq_len
            || norm_statement.state_rows != transform_statement.state_rows
            || norm_statement.qk_head_dim * norm_statement.state_rows
                != transform_statement.query_width
        {
            return Err(format!(
                "DeltaRecurrence {} norm dimensions do not match transform statement",
                norm_statement.kind.as_str()
            ));
        }

        match norm_statement.kind {
            Qwen35DeltaRecurrenceNormKind::Query => {
                if norm_statement.input_commitment != transform_statement.query_commitment
                    || norm_statement.output_commitment
                        != transform_statement.scaled_query_commitment
                {
                    return Err(
                        "DeltaRecurrence query norm commitments do not match transform statement"
                            .to_string(),
                    );
                }
            }
            Qwen35DeltaRecurrenceNormKind::Key => {
                if norm_statement.input_commitment != transform_statement.key_commitment
                    || norm_statement.output_commitment
                        != transform_statement.normalized_key_commitment
                {
                    return Err(
                        "DeltaRecurrence key norm commitments do not match transform statement"
                            .to_string(),
                    );
                }
            }
        }

        let recorded = self
            .statements
            .iter_mut()
            .find(|statement| {
                statement.kind == Qwen35TypedProofStatementKind::DeltaRecurrence
                    && statement.layer_idx == binding.layer_idx
                    && statement.stage_idx == binding.stage_idx
            })
            .ok_or_else(|| {
                "recorded DeltaRecurrence statement missing before Q/K norm proof".to_string()
            })?;
        if recorded.transform_statement_hash != Some(transform_statement.statement_hash) {
            return Err(
                "DeltaRecurrence Q/K norm proof requires the matching transform-binding proof"
                    .to_string(),
            );
        }

        match norm_statement.kind {
            Qwen35DeltaRecurrenceNormKind::Query => {
                recorded.transform_nonlinear_q_norm_statement_hash =
                    Some(norm_statement.statement_hash);
            }
            Qwen35DeltaRecurrenceNormKind::Key => {
                recorded.transform_nonlinear_k_norm_statement_hash =
                    Some(norm_statement.statement_hash);
            }
        }
        Ok(())
    }

    pub fn record_delta_recurrence_decay_air_proof(
        &mut self,
        binding: &Qwen35DeltaRecurrenceTraceBindingContract,
        transform_statement: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceTransformStatement,
        decay_proof: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceDecayProof<
            <stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel as stwo::core::channel::MerkleChannel>::H,
        >,
    ) -> Result<(), String> {
        crate::components::qwen35_delta_recurrence::verify_qwen35_delta_recurrence_decay_air(
            decay_proof,
        )
        .map_err(|err| err.to_string())?;

        let decay_statement = &decay_proof.statement;
        if decay_statement.layer_idx != transform_statement.layer_idx
            || decay_statement.layer_idx != binding.layer_idx
        {
            return Err("DeltaRecurrence decay statement layer mismatch".to_string());
        }
        if decay_statement.seq_len != transform_statement.seq_len
            || decay_statement.state_rows != transform_statement.state_rows
        {
            return Err(
                "DeltaRecurrence decay dimensions do not match transform statement".to_string(),
            );
        }
        if decay_statement.a_gate_commitment != transform_statement.a_gate_commitment
            || decay_statement.a_log_weight_commitment
                != transform_statement.a_log_weight_commitment
            || decay_statement.dt_bias_commitment != transform_statement.dt_bias_commitment
            || decay_statement.decay_commitment != transform_statement.decay_commitment
        {
            return Err(
                "DeltaRecurrence decay commitments do not match transform statement".to_string(),
            );
        }

        let recorded = self
            .statements
            .iter_mut()
            .find(|statement| {
                statement.kind == Qwen35TypedProofStatementKind::DeltaRecurrence
                    && statement.layer_idx == binding.layer_idx
                    && statement.stage_idx == binding.stage_idx
            })
            .ok_or_else(|| {
                "recorded DeltaRecurrence statement missing before decay proof".to_string()
            })?;
        if recorded.transform_statement_hash != Some(transform_statement.statement_hash) {
            return Err(
                "DeltaRecurrence decay proof requires the matching transform-binding proof"
                    .to_string(),
            );
        }
        recorded.transform_nonlinear_decay_statement_hash = Some(decay_statement.statement_hash);
        Ok(())
    }

    pub fn finalize_delta_recurrence_nonlinear_transform(
        &mut self,
        binding: &Qwen35DeltaRecurrenceTraceBindingContract,
        transform_statement: &crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceTransformStatement,
    ) -> Result<FieldElement, String> {
        let recorded = self
            .statements
            .iter_mut()
            .find(|statement| {
                statement.kind == Qwen35TypedProofStatementKind::DeltaRecurrence
                    && statement.layer_idx == binding.layer_idx
                    && statement.stage_idx == binding.stage_idx
            })
            .ok_or_else(|| {
                "recorded DeltaRecurrence statement missing before nonlinear finalization"
                    .to_string()
            })?;
        if recorded.transform_statement_hash != Some(transform_statement.statement_hash) {
            return Err(
                "DeltaRecurrence nonlinear finalization requires the matching transform-binding proof"
                    .to_string(),
            );
        }
        recorded.arithmetic_statement_hash.ok_or_else(|| {
            "DeltaRecurrence nonlinear finalization requires arithmetic proof coverage".to_string()
        })?;
        recorded
            .transform_nonlinear_q_norm_statement_hash
            .ok_or_else(|| {
                "DeltaRecurrence nonlinear finalization missing q-norm proof coverage".to_string()
            })?;
        recorded
            .transform_nonlinear_k_norm_statement_hash
            .ok_or_else(|| {
                "DeltaRecurrence nonlinear finalization missing k-norm proof coverage".to_string()
            })?;
        recorded
            .transform_nonlinear_beta_sigmoid_statement_hash
            .ok_or_else(|| {
                "DeltaRecurrence nonlinear finalization missing beta sigmoid proof coverage"
                    .to_string()
            })?;
        recorded
            .transform_nonlinear_decay_statement_hash
            .ok_or_else(|| {
                "DeltaRecurrence nonlinear finalization missing decay proof coverage".to_string()
            })?;

        let aggregate_hash = recorded
            .delta_recurrence_nonlinear_aggregate_hash()
            .ok_or_else(|| {
                "DeltaRecurrence nonlinear finalization could not compute aggregate hash"
                    .to_string()
            })?;
        recorded.transform_nonlinear_statement_hash = Some(aggregate_hash);
        Ok(aggregate_hash)
    }

    pub fn count_kind(&self, kind: Qwen35TypedProofStatementKind) -> usize {
        self.statements
            .iter()
            .filter(|statement| statement.kind == kind)
            .count()
    }

    pub fn depthwise_conv1d_coverage_complete(&self) -> bool {
        self.validate_depthwise_conv1d_coverage().is_ok()
    }

    pub fn delta_recurrence_coverage_complete(&self) -> bool {
        self.validate_delta_recurrence_coverage().is_ok()
    }

    pub fn norm_and_z_gate_coverage_complete(&self) -> bool {
        self.validate_norm_and_z_gate_coverage().is_ok()
    }

    pub fn validate_depthwise_conv1d_coverage(&self) -> Result<(), String> {
        let depthwise_statements = self
            .statements
            .iter()
            .filter(|statement| statement.kind == Qwen35TypedProofStatementKind::DepthwiseConv1d);
        let mut count = 0usize;
        let mut seen = HashSet::new();
        for statement in depthwise_statements {
            count += 1;
            if statement.seq_len != self.seq_len {
                return Err(format!(
                    "DepthwiseConv1D statement for layer {} stage {} has seq_len {}, expected {}",
                    statement.layer_idx, statement.stage_idx, statement.seq_len, self.seq_len
                ));
            }
            if !seen.insert((statement.layer_idx, statement.stage_idx)) {
                return Err(format!(
                    "duplicate DepthwiseConv1D statement for layer {} stage {}",
                    statement.layer_idx, statement.stage_idx
                ));
            }
        }
        if count != self.expected_depthwise_conv1d_statements {
            return Err(format!(
                "DepthwiseConv1D coverage has {} statements, expected {}",
                count, self.expected_depthwise_conv1d_statements
            ));
        }
        Ok(())
    }

    pub fn validate_norm_and_z_gate_coverage(&self) -> Result<(), String> {
        let norm_statements = self
            .statements
            .iter()
            .filter(|statement| statement.kind == Qwen35TypedProofStatementKind::NormAndZGate);
        let mut count = 0usize;
        let mut seen = HashSet::new();
        for statement in norm_statements {
            count += 1;
            if statement.seq_len != self.seq_len {
                return Err(format!(
                    "NormAndZGate statement for layer {} stage {} has seq_len {}, expected {}",
                    statement.layer_idx, statement.stage_idx, statement.seq_len, self.seq_len
                ));
            }
            if !seen.insert((statement.layer_idx, statement.stage_idx)) {
                return Err(format!(
                    "duplicate NormAndZGate statement for layer {} stage {}",
                    statement.layer_idx, statement.stage_idx
                ));
            }
        }
        if count != self.expected_norm_and_z_gate_statements {
            return Err(format!(
                "NormAndZGate coverage has {} statements, expected {}",
                count, self.expected_norm_and_z_gate_statements
            ));
        }
        Ok(())
    }

    pub fn validate_delta_recurrence_coverage(&self) -> Result<(), String> {
        let delta_statements = self
            .statements
            .iter()
            .filter(|statement| statement.kind == Qwen35TypedProofStatementKind::DeltaRecurrence);
        let mut count = 0usize;
        let mut seen = HashSet::new();
        for statement in delta_statements {
            count += 1;
            if statement.seq_len != self.seq_len {
                return Err(format!(
                    "DeltaRecurrence statement for layer {} stage {} has seq_len {}, expected {}",
                    statement.layer_idx, statement.stage_idx, statement.seq_len, self.seq_len
                ));
            }
            if !seen.insert((statement.layer_idx, statement.stage_idx)) {
                return Err(format!(
                    "duplicate DeltaRecurrence statement for layer {} stage {}",
                    statement.layer_idx, statement.stage_idx
                ));
            }
            if !statement.has_valid_delta_recurrence_coverage() {
                return Err(format!(
                    "DeltaRecurrence statement for layer {} stage {} is missing valid nonlinear aggregate coverage",
                    statement.layer_idx, statement.stage_idx
                ));
            }
        }
        if count != self.expected_delta_recurrence_statements {
            return Err(format!(
                "DeltaRecurrence coverage has {} statements, expected {}",
                count, self.expected_delta_recurrence_statements
            ));
        }
        Ok(())
    }

    pub fn validate_production_ready(&self) -> Result<(), String> {
        self.validate_depthwise_conv1d_coverage()?;
        self.validate_delta_recurrence_coverage()?;
        self.validate_norm_and_z_gate_coverage()?;
        Ok(())
    }

    pub fn production_ready(&self) -> bool {
        self.validate_production_ready().is_ok()
    }

    pub fn delta_recurrence_state_commitments(
        &self,
    ) -> Result<Vec<Qwen35LayerRecurrentStateCommitment>, String> {
        let mut states = Vec::new();
        for statement in &self.statements {
            if statement.kind != Qwen35TypedProofStatementKind::DeltaRecurrence {
                continue;
            }
            let initial = statement.initial_recurrent_state_commitment.ok_or_else(|| {
                format!(
                    "DeltaRecurrence statement for layer {} stage {} is missing initial recurrent state",
                    statement.layer_idx, statement.stage_idx
                )
            })?;
            let final_state = statement.final_recurrent_state_commitment.ok_or_else(|| {
                format!(
                    "DeltaRecurrence statement for layer {} stage {} is missing final recurrent state",
                    statement.layer_idx, statement.stage_idx
                )
            })?;
            states.push(Qwen35LayerRecurrentStateCommitment {
                layer_idx: statement.layer_idx,
                stage_idx: statement.stage_idx,
                initial_recurrent_state_commitment: initial,
                final_recurrent_state_commitment: final_state,
            });
        }
        states.sort_by_key(|state| (state.layer_idx, state.stage_idx));
        Ok(states)
    }

    pub fn ledger_hash(&self) -> FieldElement {
        let mut felts = vec![
            FieldElement::from(DOMAIN_QWEN35_TYPED_LEDGER),
            self.architecture_contract_hash,
            FieldElement::from(self.seq_len as u64),
            FieldElement::from(self.expected_depthwise_conv1d_statements as u64),
            FieldElement::from(self.expected_delta_recurrence_statements as u64),
            FieldElement::from(self.expected_norm_and_z_gate_statements as u64),
            FieldElement::from(self.statements.len() as u64),
        ];
        let mut statements: Vec<_> = self.statements.iter().collect();
        statements.sort_by_key(|statement| {
            (
                statement.kind.hash_tag(),
                statement.layer_idx,
                statement.stage_idx,
                statement.seq_len,
            )
        });
        for statement in statements {
            felts.extend([
                FieldElement::from(statement.kind.hash_tag()),
                FieldElement::from(statement.layer_idx as u64),
                FieldElement::from(statement.seq_len as u64),
                FieldElement::from(statement.stage_idx as u64),
                statement.trace_binding_hash,
                statement.statement_hash,
                FieldElement::from(statement.transform_statement_hash.is_some() as u64),
                statement
                    .transform_statement_hash
                    .unwrap_or(FieldElement::ZERO),
                FieldElement::from(statement.transform_nonlinear_statement_hash.is_some() as u64),
                statement
                    .transform_nonlinear_statement_hash
                    .unwrap_or(FieldElement::ZERO),
                FieldElement::from(
                    statement
                        .transform_nonlinear_q_norm_statement_hash
                        .is_some() as u64,
                ),
                statement
                    .transform_nonlinear_q_norm_statement_hash
                    .unwrap_or(FieldElement::ZERO),
                FieldElement::from(
                    statement
                        .transform_nonlinear_k_norm_statement_hash
                        .is_some() as u64,
                ),
                statement
                    .transform_nonlinear_k_norm_statement_hash
                    .unwrap_or(FieldElement::ZERO),
                FieldElement::from(
                    statement
                        .transform_nonlinear_beta_sigmoid_statement_hash
                        .is_some() as u64,
                ),
                statement
                    .transform_nonlinear_beta_sigmoid_statement_hash
                    .unwrap_or(FieldElement::ZERO),
                FieldElement::from(
                    statement.transform_nonlinear_decay_statement_hash.is_some() as u64
                ),
                statement
                    .transform_nonlinear_decay_statement_hash
                    .unwrap_or(FieldElement::ZERO),
                FieldElement::from(statement.arithmetic_statement_hash.is_some() as u64),
                statement
                    .arithmetic_statement_hash
                    .unwrap_or(FieldElement::ZERO),
                FieldElement::from(statement.initial_recurrent_state_commitment.is_some() as u64),
                statement
                    .initial_recurrent_state_commitment
                    .unwrap_or(FieldElement::ZERO),
                FieldElement::from(statement.final_recurrent_state_commitment.is_some() as u64),
                statement
                    .final_recurrent_state_commitment
                    .unwrap_or(FieldElement::ZERO),
            ]);
        }
        starknet_crypto::poseidon_hash_many(&felts)
    }
}

impl Qwen35ConversationStateLedger {
    pub fn new(
        architecture_contract_hash: FieldElement,
        expected_delta_recurrence_layers: usize,
    ) -> Self {
        Self {
            architecture_contract_hash,
            expected_delta_recurrence_layers,
            spans: Vec::new(),
        }
    }

    pub fn record_active_span(
        &mut self,
        span_idx: usize,
        typed_ledger: &Qwen35TypedProofLedger,
    ) -> Result<(), String> {
        if let Err(reason) = typed_ledger.validate_production_ready() {
            return Err(format!(
                "conversation state span {span_idx} requires a production-ready typed ledger: {reason}; DepthwiseConv1D {}/{}, DeltaRecurrence {}/{}, NormAndZGate {}/{}",
                typed_ledger.count_kind(Qwen35TypedProofStatementKind::DepthwiseConv1d),
                typed_ledger.expected_depthwise_conv1d_statements,
                typed_ledger.count_kind(Qwen35TypedProofStatementKind::DeltaRecurrence),
                typed_ledger.expected_delta_recurrence_statements,
                typed_ledger.count_kind(Qwen35TypedProofStatementKind::NormAndZGate),
                typed_ledger.expected_norm_and_z_gate_statements,
            ));
        }
        self.record_span(span_idx, typed_ledger)
    }

    fn record_span(
        &mut self,
        span_idx: usize,
        typed_ledger: &Qwen35TypedProofLedger,
    ) -> Result<(), String> {
        if typed_ledger.architecture_contract_hash != self.architecture_contract_hash {
            return Err("conversation state ledger architecture hash mismatch".to_string());
        }
        if self.spans.iter().any(|span| span.span_idx == span_idx) {
            return Err(format!("duplicate conversation state span {span_idx}"));
        }

        let mut layer_states = typed_ledger.delta_recurrence_state_commitments()?;
        Self::validate_span_states(self.expected_delta_recurrence_layers, &mut layer_states)?;
        self.spans.push(Qwen35ConversationStateSpan {
            span_idx,
            seq_len: typed_ledger.seq_len,
            typed_ledger_hash: typed_ledger.ledger_hash(),
            layer_states,
        });
        self.spans.sort_by_key(|span| span.span_idx);
        self.validate_continuity()
    }

    pub fn continuity_complete(&self) -> bool {
        self.validate_continuity().is_ok()
    }

    pub fn validate_continuity(&self) -> Result<(), String> {
        if self.spans.is_empty() {
            return Err("conversation state ledger has no spans".to_string());
        }

        for span in &self.spans {
            let mut layer_states = span.layer_states.clone();
            Self::validate_span_states(self.expected_delta_recurrence_layers, &mut layer_states)?;
        }

        for pair in self.spans.windows(2) {
            let previous = &pair[0];
            let next = &pair[1];
            if next.span_idx != previous.span_idx + 1 {
                return Err(format!(
                    "conversation state spans must be contiguous, got {} then {}",
                    previous.span_idx, next.span_idx
                ));
            }

            let mut previous_states = previous.layer_states.clone();
            let mut next_states = next.layer_states.clone();
            previous_states.sort_by_key(|state| (state.layer_idx, state.stage_idx));
            next_states.sort_by_key(|state| (state.layer_idx, state.stage_idx));

            for (previous_state, next_state) in previous_states.iter().zip(next_states.iter()) {
                if previous_state.layer_idx != next_state.layer_idx
                    || previous_state.stage_idx != next_state.stage_idx
                {
                    return Err(format!(
                        "conversation state layer set changed between spans {} and {}",
                        previous.span_idx, next.span_idx
                    ));
                }
                if previous_state.final_recurrent_state_commitment
                    != next_state.initial_recurrent_state_commitment
                {
                    return Err(format!(
                        "conversation recurrent state mismatch at layer {} stage {} between spans {} and {}",
                        previous_state.layer_idx,
                        previous_state.stage_idx,
                        previous.span_idx,
                        next.span_idx
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn ledger_hash(&self) -> FieldElement {
        let mut felts = vec![
            FieldElement::from(DOMAIN_QWEN35_CONVERSATION_STATE_LEDGER),
            self.architecture_contract_hash,
            FieldElement::from(self.expected_delta_recurrence_layers as u64),
            FieldElement::from(self.spans.len() as u64),
        ];
        for span in &self.spans {
            felts.extend([
                FieldElement::from(span.span_idx as u64),
                FieldElement::from(span.seq_len as u64),
                span.typed_ledger_hash,
                FieldElement::from(span.layer_states.len() as u64),
            ]);
            for state in &span.layer_states {
                felts.extend([
                    FieldElement::from(state.layer_idx as u64),
                    FieldElement::from(state.stage_idx as u64),
                    state.initial_recurrent_state_commitment,
                    state.final_recurrent_state_commitment,
                ]);
            }
        }
        starknet_crypto::poseidon_hash_many(&felts)
    }

    fn validate_span_states(
        expected_delta_recurrence_layers: usize,
        layer_states: &mut [Qwen35LayerRecurrentStateCommitment],
    ) -> Result<(), String> {
        if layer_states.len() != expected_delta_recurrence_layers {
            return Err(format!(
                "conversation state span has {} DeltaRecurrence states, expected {}",
                layer_states.len(),
                expected_delta_recurrence_layers
            ));
        }
        layer_states.sort_by_key(|state| (state.layer_idx, state.stage_idx));
        for pair in layer_states.windows(2) {
            if pair[0].layer_idx == pair[1].layer_idx && pair[0].stage_idx == pair[1].stage_idx {
                return Err(format!(
                    "duplicate conversation recurrent state for layer {} stage {}",
                    pair[0].layer_idx, pair[0].stage_idx
                ));
            }
        }
        Ok(())
    }
}

impl Qwen35ConversationStateSpan {
    fn recurrent_state_root(&self, domain: u64, final_state: bool) -> FieldElement {
        let mut states = self.layer_states.clone();
        states.sort_by_key(|state| (state.layer_idx, state.stage_idx));
        let mut felts = vec![
            FieldElement::from(domain),
            FieldElement::from(states.len() as u64),
        ];
        for state in states {
            felts.extend([
                FieldElement::from(state.layer_idx as u64),
                FieldElement::from(state.stage_idx as u64),
                if final_state {
                    state.final_recurrent_state_commitment
                } else {
                    state.initial_recurrent_state_commitment
                },
            ]);
        }
        starknet_crypto::poseidon_hash_many(&felts)
    }

    pub fn initial_recurrent_state_root(&self) -> FieldElement {
        self.recurrent_state_root(DOMAIN_QWEN35_RECURRENT_STATE_ROOT, false)
    }

    pub fn final_recurrent_state_root(&self) -> FieldElement {
        self.recurrent_state_root(DOMAIN_QWEN35_RECURRENT_STATE_ROOT, true)
    }
}

impl Qwen35ActiveTypedSpanReceipt {
    fn compute_hash(
        span_idx: usize,
        witness_commitment_set_hash: FieldElement,
        typed_ledger_hash: FieldElement,
        conversation_span: &Qwen35ConversationStateSpan,
    ) -> FieldElement {
        let mut felts = vec![
            FieldElement::from(DOMAIN_QWEN35_ACTIVE_TYPED_SPAN_RECEIPT),
            FieldElement::from(span_idx as u64),
            witness_commitment_set_hash,
            typed_ledger_hash,
            FieldElement::from(conversation_span.span_idx as u64),
            FieldElement::from(conversation_span.seq_len as u64),
            conversation_span.typed_ledger_hash,
            FieldElement::from(conversation_span.layer_states.len() as u64),
        ];
        for state in &conversation_span.layer_states {
            felts.extend([
                FieldElement::from(state.layer_idx as u64),
                FieldElement::from(state.stage_idx as u64),
                state.initial_recurrent_state_commitment,
                state.final_recurrent_state_commitment,
            ]);
        }
        starknet_crypto::poseidon_hash_many(&felts)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.conversation_span.span_idx != self.span_idx {
            return Err(format!(
                "active typed span receipt index {} != conversation span index {}",
                self.span_idx, self.conversation_span.span_idx
            ));
        }
        if self.witness_commitment_set.seq_len != self.typed_ledger.seq_len {
            return Err(format!(
                "active typed span receipt witness seq_len {} != typed ledger seq_len {}",
                self.witness_commitment_set.seq_len, self.typed_ledger.seq_len
            ));
        }
        if self.witness_commitment_set.manifest_hash == FieldElement::ZERO
            || self.witness_commitment_set.commitment_set_hash == FieldElement::ZERO
        {
            return Err(
                "active typed span receipt witness commitment set must be non-zero".to_string(),
            );
        }
        self.typed_ledger.validate_production_ready()?;
        let typed_ledger_hash = self.typed_ledger.ledger_hash();
        if self.conversation_span.typed_ledger_hash != typed_ledger_hash {
            return Err(format!(
                "active typed span receipt conversation ledger hash 0x{:x} != typed ledger hash 0x{:x}",
                self.conversation_span.typed_ledger_hash, typed_ledger_hash
            ));
        }
        let expected_hash = Self::compute_hash(
            self.span_idx,
            self.witness_commitment_set.commitment_set_hash,
            typed_ledger_hash,
            &self.conversation_span,
        );
        if self.receipt_hash != expected_hash {
            return Err(format!(
                "active typed span receipt hash 0x{:x} != expected 0x{:x}",
                self.receipt_hash, expected_hash
            ));
        }
        Ok(())
    }
}

impl Qwen35ActiveConversationReceipt {
    pub fn prove_from_runtime_traces(
        plan: &Qwen35ProofPlan,
        safetensors_inventory: &Qwen35TypedWitnessSourceInventory,
        traces: &[Qwen35TypedRuntimeTrace],
    ) -> Result<Self, String> {
        if traces.is_empty() {
            return Err("active Qwen3.5 conversation proof requires at least one span".to_string());
        }

        let mut span_receipts = Vec::with_capacity(traces.len());
        for (span_idx, trace) in traces.iter().enumerate() {
            span_receipts.push(trace.prove_active_typed_span(
                plan,
                safetensors_inventory,
                span_idx,
            )?);
        }
        Self::from_span_receipts(plan, span_receipts)
    }

    pub fn from_span_receipts(
        plan: &Qwen35ProofPlan,
        span_receipts: Vec<Qwen35ActiveTypedSpanReceipt>,
    ) -> Result<Self, String> {
        if span_receipts.is_empty() {
            return Err(
                "active Qwen3.5 conversation receipt requires at least one span".to_string(),
            );
        }

        let architecture_contract_hash = plan.architecture_contract_hash();
        let mut conversation_ledger = Qwen35ConversationStateLedger::new(
            architecture_contract_hash,
            plan.linear_attention_layers(),
        );
        Self::validate_uniform_span_witnesses(&span_receipts)?;

        for (expected_span_idx, receipt) in span_receipts.iter().enumerate() {
            if receipt.span_idx != expected_span_idx {
                return Err(format!(
                    "active Qwen3.5 conversation receipt spans must be contiguous from zero, got span {} at position {}",
                    receipt.span_idx, expected_span_idx
                ));
            }
            receipt.validate()?;
            if receipt.typed_ledger.architecture_contract_hash != architecture_contract_hash {
                return Err(format!(
                    "active Qwen3.5 conversation receipt span {} architecture hash mismatch",
                    receipt.span_idx
                ));
            }

            let manifest = plan.typed_witness_manifest(receipt.typed_ledger.seq_len)?;
            if receipt.witness_commitment_set.manifest_hash != manifest.manifest_hash {
                return Err(format!(
                    "active Qwen3.5 conversation receipt span {} witness manifest hash mismatch",
                    receipt.span_idx
                ));
            }
            if receipt.witness_commitment_set.commitments.len() != manifest.total_roots() {
                return Err(format!(
                    "active Qwen3.5 conversation receipt span {} has {} witness commitments, expected {}",
                    receipt.span_idx,
                    receipt.witness_commitment_set.commitments.len(),
                    manifest.total_roots()
                ));
            }

            conversation_ledger.record_active_span(receipt.span_idx, &receipt.typed_ledger)?;
            let recorded_span = conversation_ledger
                .spans
                .iter()
                .find(|span| span.span_idx == receipt.span_idx)
                .ok_or_else(|| {
                    format!(
                        "active Qwen3.5 conversation receipt failed to record span {}",
                        receipt.span_idx
                    )
                })?;
            if recorded_span != &receipt.conversation_span {
                return Err(format!(
                    "active Qwen3.5 conversation receipt span {} does not match rebuilt conversation state span",
                    receipt.span_idx
                ));
            }
        }

        let receipt_hash = Self::compute_hash(
            architecture_contract_hash,
            conversation_ledger.ledger_hash(),
            &span_receipts,
        );
        Ok(Self {
            architecture_contract_hash,
            span_receipts,
            conversation_ledger,
            receipt_hash,
        })
    }

    fn validate_uniform_span_witnesses(
        span_receipts: &[Qwen35ActiveTypedSpanReceipt],
    ) -> Result<FieldElement, String> {
        let first = span_receipts
            .first()
            .ok_or_else(|| "active Qwen3.5 conversation has no first span".to_string())?;
        let expected_root = first.witness_commitment_set.commitment_set_hash;
        let expected_manifest = first.witness_commitment_set.manifest_hash;

        for receipt in span_receipts {
            if receipt.witness_commitment_set.commitment_set_hash != expected_root {
                return Err(format!(
                    "active Qwen3.5 conversation span {} witness commitment root 0x{:x} != expected 0x{:x}",
                    receipt.span_idx,
                    receipt.witness_commitment_set.commitment_set_hash,
                    expected_root
                ));
            }
            if receipt.witness_commitment_set.manifest_hash != expected_manifest {
                return Err(format!(
                    "active Qwen3.5 conversation span {} witness manifest hash 0x{:x} != expected 0x{:x}",
                    receipt.span_idx,
                    receipt.witness_commitment_set.manifest_hash,
                    expected_manifest
                ));
            }
        }

        Ok(expected_root)
    }

    fn compute_hash(
        architecture_contract_hash: FieldElement,
        conversation_ledger_hash: FieldElement,
        span_receipts: &[Qwen35ActiveTypedSpanReceipt],
    ) -> FieldElement {
        let mut felts = vec![
            FieldElement::from(DOMAIN_QWEN35_ACTIVE_CONVERSATION_RECEIPT),
            architecture_contract_hash,
            conversation_ledger_hash,
            FieldElement::from(span_receipts.len() as u64),
        ];
        for receipt in span_receipts {
            felts.extend([
                FieldElement::from(receipt.span_idx as u64),
                receipt.witness_commitment_set.commitment_set_hash,
                receipt.typed_ledger.ledger_hash(),
                receipt.conversation_span.typed_ledger_hash,
                receipt.receipt_hash,
            ]);
        }
        starknet_crypto::poseidon_hash_many(&felts)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.architecture_contract_hash != self.conversation_ledger.architecture_contract_hash {
            return Err(
                "active Qwen3.5 conversation receipt architecture hash mismatch".to_string(),
            );
        }
        if self.span_receipts.is_empty() {
            return Err("active Qwen3.5 conversation receipt has no spans".to_string());
        }
        Self::validate_uniform_span_witnesses(&self.span_receipts)?;

        let mut rebuilt_ledger = Qwen35ConversationStateLedger::new(
            self.architecture_contract_hash,
            self.conversation_ledger.expected_delta_recurrence_layers,
        );
        for (expected_span_idx, receipt) in self.span_receipts.iter().enumerate() {
            if receipt.span_idx != expected_span_idx {
                return Err(format!(
                    "active Qwen3.5 conversation receipt spans must be contiguous from zero, got span {} at position {}",
                    receipt.span_idx, expected_span_idx
                ));
            }
            receipt.validate()?;
            if receipt.typed_ledger.architecture_contract_hash != self.architecture_contract_hash {
                return Err(format!(
                    "active Qwen3.5 conversation receipt span {} architecture hash mismatch",
                    receipt.span_idx
                ));
            }
            rebuilt_ledger.record_active_span(receipt.span_idx, &receipt.typed_ledger)?;
        }
        if rebuilt_ledger != self.conversation_ledger {
            return Err(
                "active Qwen3.5 conversation receipt ledger does not match span receipts"
                    .to_string(),
            );
        }

        let expected_hash = Self::compute_hash(
            self.architecture_contract_hash,
            self.conversation_ledger.ledger_hash(),
            &self.span_receipts,
        );
        if self.receipt_hash != expected_hash {
            return Err(format!(
                "active Qwen3.5 conversation receipt hash 0x{:x} != expected 0x{:x}",
                self.receipt_hash, expected_hash
            ));
        }
        Ok(())
    }

    pub fn weight_super_root(&self) -> Result<FieldElement, String> {
        self.validate()?;
        Self::validate_uniform_span_witnesses(&self.span_receipts)
    }

    pub fn generation_step_statements(
        &self,
        conversation_index: u64,
        first_global_step_index: u64,
        turn_index: u64,
        generated_token_ids: &[u64],
        io_commitments: &[FieldElement],
        sampling_commitments: &[FieldElement],
    ) -> Result<Vec<crate::conversation_statement::GenerationStepStatement>, String> {
        self.validate()?;
        let n_spans = self.span_receipts.len();
        if generated_token_ids.len() != n_spans {
            return Err(format!(
                "active Qwen3.5 conversation statement has {} generated token ids, expected {}",
                generated_token_ids.len(),
                n_spans
            ));
        }
        if io_commitments.len() != n_spans {
            return Err(format!(
                "active Qwen3.5 conversation statement has {} IO commitments, expected {}",
                io_commitments.len(),
                n_spans
            ));
        }
        if sampling_commitments.len() != n_spans {
            return Err(format!(
                "active Qwen3.5 conversation statement has {} sampling commitments, expected {}",
                sampling_commitments.len(),
                n_spans
            ));
        }

        Ok(self
            .span_receipts
            .iter()
            .enumerate()
            .map(|(offset, receipt)| {
                let conversation_span = &receipt.conversation_span;
                crate::conversation_statement::GenerationStepStatement {
                    global_step_index: first_global_step_index + offset as u64,
                    conversation_index,
                    turn_index,
                    token_index: offset as u64,
                    generated_token_id: generated_token_ids[offset],
                    io_commitment: io_commitments[offset],
                    sampling_commitment: sampling_commitments[offset],
                    prev_kv_commitment: conversation_span.initial_recurrent_state_root(),
                    kv_commitment: conversation_span.final_recurrent_state_root(),
                    recursive_proof_hash: receipt.receipt_hash,
                }
            })
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conversation_trace_statement(
        &self,
        conversation_index: u64,
        conversation_id_hash: FieldElement,
        prompt_commitment: FieldElement,
        transcript_commitment: FieldElement,
        action_root: FieldElement,
        n_turns: u64,
        n_prefill_tokens: u64,
        first_step_index: u64,
    ) -> Result<crate::conversation_statement::ConversationTraceStatement, String> {
        self.validate()?;
        let first_span = self
            .span_receipts
            .first()
            .ok_or_else(|| "active Qwen3.5 conversation has no first span".to_string())?;
        let last_span = self
            .span_receipts
            .last()
            .ok_or_else(|| "active Qwen3.5 conversation has no final span".to_string())?;

        Ok(crate::conversation_statement::ConversationTraceStatement {
            conversation_index,
            conversation_id_hash,
            prompt_commitment,
            transcript_commitment,
            action_root,
            initial_kv_commitment: first_span.conversation_span.initial_recurrent_state_root(),
            final_kv_commitment: last_span.conversation_span.final_recurrent_state_root(),
            n_turns,
            n_prefill_tokens,
            n_generated_tokens: self.span_receipts.len() as u64,
            first_step_index,
            n_steps: self.span_receipts.len() as u64,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn active_statement(
        &self,
        conversation_index: u64,
        first_global_step_index: u64,
        turn_index: u64,
        conversation_id_hash: FieldElement,
        prompt_commitment: FieldElement,
        transcript_commitment: FieldElement,
        actions: &[crate::conversation_statement::ConversationActionStatement],
        n_turns: u64,
        n_prefill_tokens: u64,
        generated_token_ids: &[u64],
        io_commitments: &[FieldElement],
        sampling_commitments: &[FieldElement],
    ) -> Result<Qwen35ActiveConversationStatement, String> {
        let steps = self.generation_step_statements(
            conversation_index,
            first_global_step_index,
            turn_index,
            generated_token_ids,
            io_commitments,
            sampling_commitments,
        )?;
        let scoped_actions = actions
            .iter()
            .filter(|action| action.conversation_index == conversation_index)
            .cloned()
            .collect::<Vec<_>>();
        let conversation = self.conversation_trace_statement(
            conversation_index,
            conversation_id_hash,
            prompt_commitment,
            transcript_commitment,
            crate::conversation_statement::action_root(&scoped_actions),
            n_turns,
            n_prefill_tokens,
            first_global_step_index,
        )?;

        let statement = Qwen35ActiveConversationStatement {
            architecture_contract_hash: self.architecture_contract_hash,
            weight_super_root: self.weight_super_root()?,
            receipt_hash: self.receipt_hash,
            conversation,
            steps,
            span_receipt_hashes: self
                .span_receipts
                .iter()
                .map(|receipt| receipt.receipt_hash)
                .collect(),
            actions: scoped_actions,
        };
        statement.validate()?;
        Ok(statement)
    }
}

impl Qwen35ActiveConversationStatement {
    pub fn to_felts(&self) -> Result<[FieldElement; 10], String> {
        self.validate()?;
        Ok([
            FieldElement::from(DOMAIN_QWEN35_ACTIVE_CONVERSATION_STATEMENT),
            self.architecture_contract_hash,
            self.weight_super_root,
            self.receipt_hash,
            self.conversation.commitment(),
            crate::conversation_statement::generation_root(&self.steps),
            qwen35_span_receipt_root(&self.span_receipt_hashes),
            crate::conversation_statement::action_root(&self.actions),
            FieldElement::from(self.steps.len() as u64),
            FieldElement::from(self.actions.len() as u64),
        ])
    }

    pub fn commitment(&self) -> Result<FieldElement, String> {
        Ok(starknet_crypto::poseidon_hash_many(&self.to_felts()?))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.architecture_contract_hash == FieldElement::ZERO {
            return Err("active Qwen3.5 statement architecture hash is zero".to_string());
        }
        if self.weight_super_root == FieldElement::ZERO {
            return Err("active Qwen3.5 statement weight root is zero".to_string());
        }
        if self.receipt_hash == FieldElement::ZERO {
            return Err("active Qwen3.5 statement receipt hash is zero".to_string());
        }
        if self.conversation.n_steps != self.steps.len() as u64 {
            return Err(format!(
                "active Qwen3.5 statement conversation {} has {} steps, expected {}",
                self.conversation.conversation_index,
                self.steps.len(),
                self.conversation.n_steps
            ));
        }
        if self.span_receipt_hashes.len() != self.steps.len() {
            return Err(format!(
                "active Qwen3.5 statement conversation {} has {} span receipt hashes, expected {}",
                self.conversation.conversation_index,
                self.span_receipt_hashes.len(),
                self.steps.len()
            ));
        }
        if self.conversation.n_generated_tokens != self.conversation.n_steps {
            return Err(format!(
                "active Qwen3.5 statement conversation {} generated token count {} != steps {}",
                self.conversation.conversation_index,
                self.conversation.n_generated_tokens,
                self.conversation.n_steps
            ));
        }
        let mut expected_prev = self.conversation.initial_kv_commitment;
        for (offset, step) in self.steps.iter().enumerate() {
            let expected_global = self.conversation.first_step_index + offset as u64;
            if step.recursive_proof_hash != self.span_receipt_hashes[offset] {
                return Err(format!(
                    "active Qwen3.5 statement conversation {} step {} recursive proof hash 0x{:x} != span receipt hash 0x{:x}",
                    self.conversation.conversation_index,
                    step.global_step_index,
                    step.recursive_proof_hash,
                    self.span_receipt_hashes[offset]
                ));
            }
            if step.global_step_index != expected_global {
                return Err(format!(
                    "active Qwen3.5 statement step global index {} != expected {}",
                    step.global_step_index, expected_global
                ));
            }
            if step.conversation_index != self.conversation.conversation_index {
                return Err(format!(
                    "active Qwen3.5 statement step conversation {} != expected {}",
                    step.conversation_index, self.conversation.conversation_index
                ));
            }
            if step.prev_kv_commitment != expected_prev {
                return Err(format!(
                    "active Qwen3.5 statement conversation {} step {} prev KV 0x{:x} != expected 0x{:x}",
                    self.conversation.conversation_index,
                    step.global_step_index,
                    step.prev_kv_commitment,
                    expected_prev
                ));
            }
            expected_prev = step.kv_commitment;
        }
        if expected_prev != self.conversation.final_kv_commitment {
            return Err(format!(
                "active Qwen3.5 statement conversation {} final KV 0x{:x} != expected 0x{:x}",
                self.conversation.conversation_index,
                expected_prev,
                self.conversation.final_kv_commitment
            ));
        }
        crate::conversation_statement::validate_conversation_actions(
            std::slice::from_ref(&self.conversation),
            &self.actions,
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}

impl Qwen35ActiveConversationBatchArtifact {
    fn compute_hash(
        canonical_statement_hash: FieldElement,
        active_statement_root: FieldElement,
        active_receipt_root: FieldElement,
    ) -> FieldElement {
        starknet_crypto::poseidon_hash_many(&[
            FieldElement::from(DOMAIN_QWEN35_ACTIVE_BATCH_ARTIFACT),
            canonical_statement_hash,
            active_statement_root,
            active_receipt_root,
        ])
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.active_statement_root == FieldElement::ZERO {
            return Err("active Qwen3.5 batch artifact statement root is zero".to_string());
        }
        if self.active_receipt_root == FieldElement::ZERO {
            return Err("active Qwen3.5 batch artifact receipt root is zero".to_string());
        }
        let expected = Self::compute_hash(
            self.canonical_statement.statement_hash(),
            self.active_statement_root,
            self.active_receipt_root,
        );
        if self.artifact_hash != expected {
            return Err(format!(
                "active Qwen3.5 batch artifact hash 0x{:x} != expected 0x{:x}",
                self.artifact_hash, expected
            ));
        }
        Ok(())
    }

    pub fn to_felts(&self) -> [FieldElement; 4] {
        [
            FieldElement::from(DOMAIN_QWEN35_ACTIVE_BATCH_ARTIFACT),
            self.canonical_statement.statement_hash(),
            self.active_statement_root,
            self.active_receipt_root,
        ]
    }

    pub fn public_output_hash(&self) -> FieldElement {
        starknet_crypto::poseidon_hash_many(&self.to_felts())
    }
}

pub fn qwen35_active_batch_artifact_felts(
    artifact: &Qwen35ActiveConversationBatchArtifact,
) -> Result<[FieldElement; 4], String> {
    artifact.validate()?;
    Ok(artifact.to_felts())
}

pub fn qwen35_active_conversation_statement_felts(
    active: &Qwen35ActiveConversationStatement,
) -> Result<[FieldElement; 10], String> {
    active.to_felts()
}

#[cfg(feature = "serde")]
fn qwen35_felt_hex(value: FieldElement) -> String {
    format!("0x{:x}", value)
}

#[cfg(feature = "serde")]
fn qwen35_u64_hex(value: u64) -> String {
    qwen35_felt_hex(FieldElement::from(value))
}

#[cfg(feature = "serde")]
pub fn qwen35_active_conversation_batch_artifact_json(
    artifact: &Qwen35ActiveConversationBatchArtifact,
    active_conversations: &[Qwen35ActiveConversationStatement],
) -> Result<serde_json::Value, String> {
    artifact.validate()?;
    let expected_statement_root = qwen35_active_statement_root(active_conversations)?;
    let expected_receipt_root = qwen35_active_receipt_root(active_conversations);
    if artifact.active_statement_root != expected_statement_root {
        return Err(format!(
            "active Qwen3.5 artifact statement root 0x{:x} != conversations root 0x{:x}",
            artifact.active_statement_root, expected_statement_root
        ));
    }
    if artifact.active_receipt_root != expected_receipt_root {
        return Err(format!(
            "active Qwen3.5 artifact receipt root 0x{:x} != conversations root 0x{:x}",
            artifact.active_receipt_root, expected_receipt_root
        ));
    }

    let active_statement_felts = active_conversations
        .iter()
        .map(|active| {
            active.to_felts().map(|felts| {
                felts
                    .iter()
                    .copied()
                    .map(qwen35_felt_hex)
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let active_conversation_json = active_conversations
        .iter()
        .zip(active_statement_felts.iter())
        .map(|(active, felts)| {
            serde_json::json!({
                "conversation_index": active.conversation.conversation_index,
                "active_statement_hash": qwen35_felt_hex(active.commitment().expect("validated active statement")),
                "active_statement_felts": felts,
                "active_receipt_hash": qwen35_felt_hex(active.receipt_hash),
                "span_receipt_hashes": active
                    .span_receipt_hashes
                    .iter()
                    .copied()
                    .map(qwen35_felt_hex)
                    .collect::<Vec<_>>(),
                "generation_step_count": active.steps.len(),
                "action_count": active.actions.len(),
            })
        })
        .collect::<Vec<_>>();
    let canonical_statement_felts = artifact
        .canonical_statement
        .to_felts()
        .iter()
        .copied()
        .map(qwen35_felt_hex)
        .collect::<Vec<_>>();
    let active_batch_felts = artifact
        .to_felts()
        .iter()
        .copied()
        .map(qwen35_felt_hex)
        .collect::<Vec<_>>();

    Ok(serde_json::json!({
        "schema": "obelyzk.qwen35_active_conversation_batch_artifact.v1",
        "verifier": "qwen35-active-conversation-statement-verifier",
        "scope": "qwen35_active_conversation_receipt_bound_statement",
        "statement_hash": qwen35_felt_hex(artifact.artifact_hash),
        "expected_cairo_output_hash": qwen35_felt_hex(artifact.artifact_hash),
        "active_batch_felts": active_batch_felts,
        "canonical_statement_hash": qwen35_felt_hex(artifact.canonical_statement.statement_hash()),
        "canonical_statement_felts": canonical_statement_felts,
        "active_statement_root": qwen35_felt_hex(artifact.active_statement_root),
        "active_receipt_root": qwen35_felt_hex(artifact.active_receipt_root),
        "n_active_conversations": active_conversations.len(),
        "active_conversations": active_conversation_json,
        "canonical_batch": {
            "version": artifact.canonical_statement.version,
            "model_id": qwen35_felt_hex(artifact.canonical_statement.model_id),
            "verifier_program_hash": qwen35_felt_hex(artifact.canonical_statement.verifier_program_hash),
            "circuit_hash": qwen35_felt_hex(artifact.canonical_statement.circuit_hash),
            "weight_super_root": qwen35_felt_hex(artifact.canonical_statement.weight_super_root),
            "policy_commitment": qwen35_felt_hex(artifact.canonical_statement.policy_commitment),
            "tokenizer_config_hash": qwen35_felt_hex(artifact.canonical_statement.tokenizer_config_hash),
            "hades_commitment": qwen35_felt_hex(artifact.canonical_statement.hades_commitment),
            "conversation_root": qwen35_felt_hex(artifact.canonical_statement.conversation_root),
            "generation_root": qwen35_felt_hex(artifact.canonical_statement.generation_root),
            "action_root": qwen35_felt_hex(artifact.canonical_statement.action_root),
            "initial_kv_root": qwen35_felt_hex(artifact.canonical_statement.initial_kv_root),
            "final_kv_root": qwen35_felt_hex(artifact.canonical_statement.final_kv_root),
            "n_conversations": artifact.canonical_statement.n_conversations,
            "n_steps": artifact.canonical_statement.n_steps,
            "n_prefill_tokens": artifact.canonical_statement.n_prefill_tokens,
            "n_generated_tokens": artifact.canonical_statement.n_generated_tokens,
            "security_bits": artifact.canonical_statement.security_bits,
        },
    }))
}

#[cfg(feature = "serde")]
pub fn qwen35_active_conversation_verifier_args_json_from_str(
    json: &str,
) -> Result<serde_json::Value, String> {
    let document: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("invalid active Qwen3.5 JSON: {e}"))?;
    qwen35_active_conversation_verifier_args_json_from_value(&document)
}

#[cfg(feature = "serde")]
pub fn qwen35_active_conversation_verifier_args_json_from_value(
    document: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let model_id = qwen35_json_required_felt(document, "model_id")?;
    let verifier_program_hash =
        qwen35_json_optional_felt(document, "verifier_program_hash")?.unwrap_or(FieldElement::ZERO);
    let policy_commitment = qwen35_json_required_felt(document, "policy_commitment")?;
    let tokenizer_config_hash =
        qwen35_json_optional_felt(document, "tokenizer_config_hash")?.unwrap_or(FieldElement::ZERO);
    let hades_commitment =
        qwen35_json_optional_felt(document, "hades_commitment")?.unwrap_or(FieldElement::ZERO);
    let security_bits = qwen35_json_optional_u64(document, "security_bits")?
        .unwrap_or(crate::conversation_statement::PRODUCTION_SECURITY_BITS);
    let active_documents = document
        .get("active_conversations")
        .and_then(|value| value.as_array())
        .ok_or_else(|| "active Qwen3.5 JSON missing active_conversations array".to_string())?;
    if active_documents.is_empty() {
        return Err("active Qwen3.5 JSON requires at least one active conversation".to_string());
    }

    let active_conversations = active_documents
        .iter()
        .enumerate()
        .map(|(idx, active)| qwen35_active_statement_from_json(idx, active))
        .collect::<Result<Vec<_>, _>>()?;
    let artifact = qwen35_build_active_conversation_batch_artifact(
        model_id,
        verifier_program_hash,
        policy_commitment,
        tokenizer_config_hash,
        hades_commitment,
        &active_conversations,
        security_bits,
    )?;
    qwen35_active_conversation_verifier_args_json(&artifact, &active_conversations)
}

#[cfg(feature = "serde")]
pub fn qwen35_active_conversation_verifier_args_json(
    artifact: &Qwen35ActiveConversationBatchArtifact,
    active_conversations: &[Qwen35ActiveConversationStatement],
) -> Result<serde_json::Value, String> {
    artifact.validate()?;
    let expected_statement_root = qwen35_active_statement_root(active_conversations)?;
    let expected_receipt_root = qwen35_active_receipt_root(active_conversations);
    if artifact.active_statement_root != expected_statement_root {
        return Err(format!(
            "active Qwen3.5 artifact statement root 0x{:x} != conversations root 0x{:x}",
            artifact.active_statement_root, expected_statement_root
        ));
    }
    if artifact.active_receipt_root != expected_receipt_root {
        return Err(format!(
            "active Qwen3.5 artifact receipt root 0x{:x} != conversations root 0x{:x}",
            artifact.active_receipt_root, expected_receipt_root
        ));
    }

    let statement = &artifact.canonical_statement;
    let mut conversations = Vec::with_capacity(active_conversations.len());
    let mut steps = Vec::new();
    let mut actions = Vec::new();
    for active in active_conversations {
        active.validate()?;
        conversations.push(active.conversation.clone());
        steps.extend(active.steps.iter().cloned());
        actions.extend(active.actions.iter().cloned());
    }

    let mut args = vec![
        qwen35_felt_hex(statement.model_id),
        qwen35_felt_hex(statement.verifier_program_hash),
        qwen35_felt_hex(statement.circuit_hash),
        qwen35_felt_hex(statement.weight_super_root),
        qwen35_felt_hex(statement.policy_commitment),
        qwen35_felt_hex(statement.tokenizer_config_hash),
        qwen35_felt_hex(statement.hades_commitment),
        qwen35_u64_hex(statement.security_bits),
        qwen35_u64_hex(conversations.len() as u64),
        qwen35_u64_hex(steps.len() as u64),
        qwen35_u64_hex(actions.len() as u64),
    ];

    for conversation in &conversations {
        args.extend([
            qwen35_u64_hex(conversation.conversation_index),
            qwen35_felt_hex(conversation.conversation_id_hash),
            qwen35_felt_hex(conversation.prompt_commitment),
            qwen35_felt_hex(conversation.transcript_commitment),
            qwen35_felt_hex(conversation.action_root),
            qwen35_felt_hex(conversation.initial_kv_commitment),
            qwen35_felt_hex(conversation.final_kv_commitment),
            qwen35_u64_hex(conversation.n_turns),
            qwen35_u64_hex(conversation.n_prefill_tokens),
            qwen35_u64_hex(conversation.n_generated_tokens),
            qwen35_u64_hex(conversation.first_step_index),
            qwen35_u64_hex(conversation.n_steps),
        ]);
    }

    for step in &steps {
        args.extend([
            qwen35_u64_hex(step.global_step_index),
            qwen35_u64_hex(step.conversation_index),
            qwen35_u64_hex(step.turn_index),
            qwen35_u64_hex(step.token_index),
            qwen35_u64_hex(step.generated_token_id),
            qwen35_felt_hex(step.io_commitment),
            qwen35_felt_hex(step.sampling_commitment),
            qwen35_felt_hex(step.prev_kv_commitment),
            qwen35_felt_hex(step.kv_commitment),
            qwen35_felt_hex(step.recursive_proof_hash),
        ]);
    }

    for action in &actions {
        args.extend([
            qwen35_u64_hex(action.conversation_index),
            qwen35_u64_hex(action.turn_index),
            qwen35_u64_hex(action.action_index),
            qwen35_felt_hex(action.action_kind_hash),
            qwen35_felt_hex(action.tool_name_hash),
            qwen35_felt_hex(action.input_commitment),
            qwen35_felt_hex(action.output_commitment),
            qwen35_felt_hex(action.policy_commitment),
        ]);
    }

    for active in active_conversations {
        args.extend([
            qwen35_felt_hex(active.architecture_contract_hash),
            qwen35_felt_hex(active.weight_super_root),
            qwen35_felt_hex(active.receipt_hash),
        ]);
        args.extend(
            active
                .span_receipt_hashes
                .iter()
                .copied()
                .map(qwen35_felt_hex),
        );
    }

    let mut serialized_args = Vec::with_capacity(args.len() + 1);
    serialized_args.push(qwen35_u64_hex(args.len() as u64));
    serialized_args.extend(args);
    Ok(serde_json::Value::Array(
        serialized_args
            .into_iter()
            .map(serde_json::Value::String)
            .collect(),
    ))
}

#[cfg(feature = "serde")]
pub fn qwen35_active_conversation_batch_artifact_json_from_str(
    json: &str,
) -> Result<serde_json::Value, String> {
    let document: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("invalid active Qwen3.5 JSON: {e}"))?;
    qwen35_active_conversation_batch_artifact_json_from_value(&document)
}

#[cfg(feature = "serde")]
pub fn qwen35_active_conversation_batch_artifact_json_from_value(
    document: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let model_id = qwen35_json_required_felt(document, "model_id")?;
    let verifier_program_hash =
        qwen35_json_optional_felt(document, "verifier_program_hash")?.unwrap_or(FieldElement::ZERO);
    let policy_commitment = qwen35_json_required_felt(document, "policy_commitment")?;
    let tokenizer_config_hash =
        qwen35_json_optional_felt(document, "tokenizer_config_hash")?.unwrap_or(FieldElement::ZERO);
    let hades_commitment =
        qwen35_json_optional_felt(document, "hades_commitment")?.unwrap_or(FieldElement::ZERO);
    let security_bits = qwen35_json_optional_u64(document, "security_bits")?
        .unwrap_or(crate::conversation_statement::PRODUCTION_SECURITY_BITS);
    let active_documents = document
        .get("active_conversations")
        .and_then(|value| value.as_array())
        .ok_or_else(|| "active Qwen3.5 JSON missing active_conversations array".to_string())?;
    if active_documents.is_empty() {
        return Err("active Qwen3.5 JSON requires at least one active conversation".to_string());
    }

    let active_conversations = active_documents
        .iter()
        .enumerate()
        .map(|(idx, active)| qwen35_active_statement_from_json(idx, active))
        .collect::<Result<Vec<_>, _>>()?;
    let artifact = qwen35_build_active_conversation_batch_artifact(
        model_id,
        verifier_program_hash,
        policy_commitment,
        tokenizer_config_hash,
        hades_commitment,
        &active_conversations,
        security_bits,
    )?;
    qwen35_active_conversation_batch_artifact_json(&artifact, &active_conversations)
}

#[cfg(feature = "serde")]
fn qwen35_active_statement_from_json(
    idx: usize,
    value: &serde_json::Value,
) -> Result<Qwen35ActiveConversationStatement, String> {
    let architecture_contract_hash =
        qwen35_json_required_felt(value, "architecture_contract_hash")?;
    let weight_super_root = qwen35_json_required_felt(value, "weight_super_root")?;
    let receipt_hash = qwen35_json_required_felt(value, "receipt_hash")?;
    let conversation_value = value.get("conversation").ok_or_else(|| {
        format!("active Qwen3.5 conversation document {idx} missing conversation")
    })?;
    let conversation = qwen35_conversation_trace_from_json(conversation_value)?;
    let steps = qwen35_json_required_array(value, "steps")?
        .iter()
        .enumerate()
        .map(|(step_idx, step)| qwen35_generation_step_from_json(step_idx, step))
        .collect::<Result<Vec<_>, _>>()?;
    let span_receipt_hashes = qwen35_json_required_array(value, "span_receipt_hashes")?
        .iter()
        .enumerate()
        .map(|(span_idx, felt)| {
            qwen35_parse_json_felt(
                felt,
                &format!("active_conversations[{idx}].span_receipt_hashes[{span_idx}]"),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let actions = value
        .get("actions")
        .and_then(|actions| actions.as_array())
        .map(|actions| {
            actions
                .iter()
                .enumerate()
                .map(|(action_idx, action)| qwen35_action_from_json(action_idx, action))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();

    let active = Qwen35ActiveConversationStatement {
        architecture_contract_hash,
        weight_super_root,
        receipt_hash,
        conversation,
        steps,
        span_receipt_hashes,
        actions,
    };
    active.validate()?;
    Ok(active)
}

#[cfg(feature = "serde")]
fn qwen35_conversation_trace_from_json(
    value: &serde_json::Value,
) -> Result<crate::conversation_statement::ConversationTraceStatement, String> {
    Ok(crate::conversation_statement::ConversationTraceStatement {
        conversation_index: qwen35_json_required_u64(value, "conversation_index")?,
        conversation_id_hash: qwen35_json_required_felt(value, "conversation_id_hash")?,
        prompt_commitment: qwen35_json_required_felt(value, "prompt_commitment")?,
        transcript_commitment: qwen35_json_required_felt(value, "transcript_commitment")?,
        action_root: qwen35_json_required_felt(value, "action_root")?,
        initial_kv_commitment: qwen35_json_required_felt(value, "initial_kv_commitment")?,
        final_kv_commitment: qwen35_json_required_felt(value, "final_kv_commitment")?,
        n_turns: qwen35_json_required_u64(value, "n_turns")?,
        n_prefill_tokens: qwen35_json_required_u64(value, "n_prefill_tokens")?,
        n_generated_tokens: qwen35_json_required_u64(value, "n_generated_tokens")?,
        first_step_index: qwen35_json_required_u64(value, "first_step_index")?,
        n_steps: qwen35_json_required_u64(value, "n_steps")?,
    })
}

#[cfg(feature = "serde")]
fn qwen35_generation_step_from_json(
    idx: usize,
    value: &serde_json::Value,
) -> Result<crate::conversation_statement::GenerationStepStatement, String> {
    Ok(crate::conversation_statement::GenerationStepStatement {
        global_step_index: qwen35_json_required_u64(value, "global_step_index")
            .map_err(|e| format!("step {idx}: {e}"))?,
        conversation_index: qwen35_json_required_u64(value, "conversation_index")
            .map_err(|e| format!("step {idx}: {e}"))?,
        turn_index: qwen35_json_required_u64(value, "turn_index")
            .map_err(|e| format!("step {idx}: {e}"))?,
        token_index: qwen35_json_required_u64(value, "token_index")
            .map_err(|e| format!("step {idx}: {e}"))?,
        generated_token_id: qwen35_json_required_u64(value, "generated_token_id")
            .map_err(|e| format!("step {idx}: {e}"))?,
        io_commitment: qwen35_json_required_felt(value, "io_commitment")
            .map_err(|e| format!("step {idx}: {e}"))?,
        sampling_commitment: qwen35_json_required_felt(value, "sampling_commitment")
            .map_err(|e| format!("step {idx}: {e}"))?,
        prev_kv_commitment: qwen35_json_required_felt(value, "prev_kv_commitment")
            .map_err(|e| format!("step {idx}: {e}"))?,
        kv_commitment: qwen35_json_required_felt(value, "kv_commitment")
            .map_err(|e| format!("step {idx}: {e}"))?,
        recursive_proof_hash: qwen35_json_required_felt(value, "recursive_proof_hash")
            .map_err(|e| format!("step {idx}: {e}"))?,
    })
}

#[cfg(feature = "serde")]
fn qwen35_action_from_json(
    idx: usize,
    value: &serde_json::Value,
) -> Result<crate::conversation_statement::ConversationActionStatement, String> {
    Ok(crate::conversation_statement::ConversationActionStatement {
        conversation_index: qwen35_json_required_u64(value, "conversation_index")
            .map_err(|e| format!("action {idx}: {e}"))?,
        turn_index: qwen35_json_required_u64(value, "turn_index")
            .map_err(|e| format!("action {idx}: {e}"))?,
        action_index: qwen35_json_required_u64(value, "action_index")
            .map_err(|e| format!("action {idx}: {e}"))?,
        action_kind_hash: qwen35_json_required_felt(value, "action_kind_hash")
            .map_err(|e| format!("action {idx}: {e}"))?,
        tool_name_hash: qwen35_json_required_felt(value, "tool_name_hash")
            .map_err(|e| format!("action {idx}: {e}"))?,
        input_commitment: qwen35_json_required_felt(value, "input_commitment")
            .map_err(|e| format!("action {idx}: {e}"))?,
        output_commitment: qwen35_json_required_felt(value, "output_commitment")
            .map_err(|e| format!("action {idx}: {e}"))?,
        policy_commitment: qwen35_json_required_felt(value, "policy_commitment")
            .map_err(|e| format!("action {idx}: {e}"))?,
    })
}

#[cfg(feature = "serde")]
fn qwen35_json_required_array<'a>(
    value: &'a serde_json::Value,
    field: &str,
) -> Result<&'a Vec<serde_json::Value>, String> {
    value
        .get(field)
        .and_then(|value| value.as_array())
        .ok_or_else(|| format!("active Qwen3.5 JSON missing {field} array"))
}

#[cfg(feature = "serde")]
fn qwen35_json_required_felt(
    value: &serde_json::Value,
    field: &str,
) -> Result<FieldElement, String> {
    let Some(felt) = value.get(field) else {
        return Err(format!("active Qwen3.5 JSON missing {field}"));
    };
    qwen35_parse_json_felt(felt, field)
}

#[cfg(feature = "serde")]
fn qwen35_json_optional_felt(
    value: &serde_json::Value,
    field: &str,
) -> Result<Option<FieldElement>, String> {
    value
        .get(field)
        .map(|felt| qwen35_parse_json_felt(felt, field))
        .transpose()
}

#[cfg(feature = "serde")]
fn qwen35_json_required_u64(value: &serde_json::Value, field: &str) -> Result<u64, String> {
    let Some(number) = value.get(field) else {
        return Err(format!("active Qwen3.5 JSON missing {field}"));
    };
    qwen35_parse_json_u64(number, field)
}

#[cfg(feature = "serde")]
fn qwen35_json_optional_u64(value: &serde_json::Value, field: &str) -> Result<Option<u64>, String> {
    value
        .get(field)
        .map(|number| qwen35_parse_json_u64(number, field))
        .transpose()
}

#[cfg(feature = "serde")]
fn qwen35_parse_json_u64(value: &serde_json::Value, field: &str) -> Result<u64, String> {
    if let Some(number) = value.as_u64() {
        return Ok(number);
    }
    if let Some(text) = value.as_str() {
        return text
            .parse::<u64>()
            .map_err(|e| format!("active Qwen3.5 JSON field {field} is not a u64: {e}"));
    }
    Err(format!(
        "active Qwen3.5 JSON field {field} must be a u64 or decimal string"
    ))
}

#[cfg(feature = "serde")]
fn qwen35_parse_json_felt(value: &serde_json::Value, field: &str) -> Result<FieldElement, String> {
    let Some(text) = value.as_str() else {
        return Err(format!(
            "active Qwen3.5 JSON field {field} must be a felt hex string"
        ));
    };
    let normalized = text
        .strip_prefix("0x")
        .or_else(|| text.strip_prefix("0X"))
        .unwrap_or(text);
    FieldElement::from_hex_be(normalized)
        .map_err(|e| format!("active Qwen3.5 JSON field {field} has invalid felt hex {text}: {e}"))
}

fn qwen35_active_receipt_root(
    active_conversations: &[Qwen35ActiveConversationStatement],
) -> FieldElement {
    let mut felts = vec![
        FieldElement::from(DOMAIN_QWEN35_ACTIVE_CONVERSATION_RECEIPT),
        FieldElement::from(active_conversations.len() as u64),
    ];
    for active in active_conversations {
        felts.extend([
            FieldElement::from(active.conversation.conversation_index),
            active.receipt_hash,
        ]);
    }
    starknet_crypto::poseidon_hash_many(&felts)
}

fn qwen35_span_receipt_root(span_receipt_hashes: &[FieldElement]) -> FieldElement {
    let mut felts = vec![
        FieldElement::from(DOMAIN_QWEN35_ACTIVE_TYPED_SPAN_RECEIPT),
        FieldElement::from(span_receipt_hashes.len() as u64),
    ];
    felts.extend(span_receipt_hashes.iter().copied());
    starknet_crypto::poseidon_hash_many(&felts)
}

fn qwen35_active_statement_root(
    active_conversations: &[Qwen35ActiveConversationStatement],
) -> Result<FieldElement, String> {
    let mut felts = vec![
        FieldElement::from(DOMAIN_QWEN35_ACTIVE_CONVERSATION_STATEMENT),
        FieldElement::from(active_conversations.len() as u64),
    ];
    for active in active_conversations {
        felts.push(active.commitment()?);
    }
    Ok(starknet_crypto::poseidon_hash_many(&felts))
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_build_active_conversation_batch_statement(
    model_id: FieldElement,
    verifier_program_hash: FieldElement,
    policy_commitment: FieldElement,
    tokenizer_config_hash: FieldElement,
    hades_commitment: FieldElement,
    active_conversations: &[Qwen35ActiveConversationStatement],
    security_bits: u64,
) -> Result<crate::conversation_statement::ConversationBatchStatement, String> {
    Ok(qwen35_build_active_conversation_batch_artifact(
        model_id,
        verifier_program_hash,
        policy_commitment,
        tokenizer_config_hash,
        hades_commitment,
        active_conversations,
        security_bits,
    )?
    .canonical_statement)
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_build_active_conversation_batch_artifact(
    model_id: FieldElement,
    verifier_program_hash: FieldElement,
    policy_commitment: FieldElement,
    tokenizer_config_hash: FieldElement,
    hades_commitment: FieldElement,
    active_conversations: &[Qwen35ActiveConversationStatement],
    security_bits: u64,
) -> Result<Qwen35ActiveConversationBatchArtifact, String> {
    if active_conversations.is_empty() {
        return Err(
            "active Qwen3.5 batch statement requires at least one conversation".to_string(),
        );
    }

    let architecture_contract_hash = active_conversations[0].architecture_contract_hash;
    let weight_super_root = active_conversations[0].weight_super_root;
    let mut conversations = Vec::with_capacity(active_conversations.len());
    let mut steps = Vec::new();
    let mut actions = Vec::new();

    for (expected_index, active) in active_conversations.iter().enumerate() {
        active.validate()?;
        if active.architecture_contract_hash != architecture_contract_hash {
            return Err(format!(
                "active Qwen3.5 conversation {} architecture hash 0x{:x} != expected 0x{:x}",
                active.conversation.conversation_index,
                active.architecture_contract_hash,
                architecture_contract_hash
            ));
        }
        if active.weight_super_root != weight_super_root {
            return Err(format!(
                "active Qwen3.5 conversation {} weight root 0x{:x} != expected 0x{:x}",
                active.conversation.conversation_index, active.weight_super_root, weight_super_root
            ));
        }
        if active.conversation.conversation_index != expected_index as u64 {
            return Err(format!(
                "active Qwen3.5 conversation index {} != expected {}",
                active.conversation.conversation_index, expected_index
            ));
        }

        conversations.push(active.conversation.clone());
        steps.extend(active.steps.iter().cloned());
        actions.extend(active.actions.iter().cloned());
    }

    let canonical_statement = crate::conversation_statement::build_conversation_batch_statement(
        model_id,
        verifier_program_hash,
        architecture_contract_hash,
        weight_super_root,
        policy_commitment,
        tokenizer_config_hash,
        hades_commitment,
        &conversations,
        &steps,
        &actions,
        security_bits,
    )
    .map_err(|e| e.to_string())?;
    let active_statement_root = qwen35_active_statement_root(active_conversations)?;
    let active_receipt_root = qwen35_active_receipt_root(active_conversations);
    let artifact_hash = Qwen35ActiveConversationBatchArtifact::compute_hash(
        canonical_statement.statement_hash(),
        active_statement_root,
        active_receipt_root,
    );

    let artifact = Qwen35ActiveConversationBatchArtifact {
        canonical_statement,
        active_statement_root,
        active_receipt_root,
        artifact_hash,
    };
    artifact.validate()?;
    Ok(artifact)
}

impl Qwen35GatedDeltaNetContract {
    pub fn depthwise_conv1d_air_contract(&self) -> Qwen35DepthwiseConv1dAirContract {
        let kernel = self.conv1d_weight.inner;
        let channels = self.conv1d_weight.outer;
        let logical_trace_rows = self.seq_len * channels;
        let deterministic_columns = 2 + kernel; // token_idx, channel_idx, valid_tap[k].
        let witness_columns = 1 + 4 * kernel; // output, input_tap[k], weight_tap[k], masked_tap[k], product_tap[k].
        let total_columns = deterministic_columns + witness_columns;
        let arithmetic_constraints_per_row = 2 * kernel + 1; // mask, product, output=sum(products).
        let row_binding_constraints_per_row = 2 + kernel; // token/channel schedule plus causal valid taps.
        let tap_offsets = (0..kernel)
            .map(|tap| tap as isize + 1 - kernel as isize)
            .collect();

        Qwen35DepthwiseConv1dAirContract {
            layer_idx: self.layer_idx,
            seq_len: self.seq_len,
            channels,
            kernel,
            input: self.qkv_projected,
            weight: self.conv1d_weight,
            output: self.qkv_after_conv,
            tap_offsets,
            logical_trace_rows,
            deterministic_columns,
            witness_columns,
            total_columns,
            arithmetic_constraints_per_row,
            row_binding_constraints_per_row,
        }
    }

    pub fn typed_witness_layer_manifest(&self) -> Result<Qwen35TypedWitnessLayerManifest, String> {
        use Qwen35TensorShape::{Matrix, Vector};
        use Qwen35TypedProofStatementKind::{DeltaRecurrence, DepthwiseConv1d, NormAndZGate};
        use Qwen35TypedWitnessRootKind::{LookupTable, RecurrentState};

        let depthwise = self.depthwise_conv1d_trace_binding_contract()?;
        let delta = self.delta_recurrence_trace_binding_contract()?;
        let norm = self.norm_and_z_gate_trace_binding_contract()?;
        let layer_prefix = format!("model.language_model.layers.{}", self.layer_idx);
        let linear_attn = format!("{layer_prefix}.linear_attn");

        let mut roots = vec![
            qwen35_witness_root_from_trace(
                self.layer_idx,
                DepthwiseConv1d,
                depthwise.producer_stage_idx,
                &depthwise.input_root,
                format!("runtime:{linear_attn}.qkv_projected"),
            ),
            qwen35_witness_root_from_trace(
                self.layer_idx,
                DepthwiseConv1d,
                depthwise.stage_idx,
                &depthwise.weight_root,
                format!("safetensors:{linear_attn}.conv1d.weight"),
            ),
            qwen35_witness_root_from_trace(
                self.layer_idx,
                DepthwiseConv1d,
                depthwise.consumer_stage_idx,
                &depthwise.output_root,
                format!("runtime:{linear_attn}.qkv_after_conv"),
            ),
            qwen35_witness_root_from_trace(
                self.layer_idx,
                DeltaRecurrence,
                delta.qkv_split_stage_idx,
                &delta.query_root,
                format!("runtime:{linear_attn}.query"),
            ),
            qwen35_witness_root_from_trace(
                self.layer_idx,
                DeltaRecurrence,
                delta.qkv_split_stage_idx,
                &delta.key_root,
                format!("runtime:{linear_attn}.key"),
            ),
            qwen35_witness_root_from_trace(
                self.layer_idx,
                DeltaRecurrence,
                delta.qkv_split_stage_idx,
                &delta.projected_value_root,
                format!("runtime:{linear_attn}.projected_value"),
            ),
            qwen35_witness_root_from_trace(
                self.layer_idx,
                DeltaRecurrence,
                delta.ab_projection_stage_idx,
                &delta.a_gate_root,
                format!("runtime:{linear_attn}.a_gate"),
            ),
            qwen35_witness_root_from_trace(
                self.layer_idx,
                DeltaRecurrence,
                delta.ab_projection_stage_idx,
                &delta.b_gate_root,
                format!("runtime:{linear_attn}.b_gate"),
            ),
            qwen35_witness_root_from_trace(
                self.layer_idx,
                DeltaRecurrence,
                delta.stage_idx,
                &delta.a_log_weight_root,
                format!("safetensors:{linear_attn}.A_log"),
            ),
            qwen35_witness_root_from_trace(
                self.layer_idx,
                DeltaRecurrence,
                delta.stage_idx,
                &delta.dt_bias_root,
                format!("safetensors:{linear_attn}.dt_bias"),
            ),
            qwen35_witness_root_from_trace(
                self.layer_idx,
                DeltaRecurrence,
                delta.stage_idx,
                &delta.output_root,
                format!("runtime:{linear_attn}.attended_value"),
            ),
            qwen35_witness_root_from_trace(
                self.layer_idx,
                NormAndZGate,
                norm.delta_recurrence_stage_idx,
                &norm.attended_value_root,
                format!("runtime:{linear_attn}.attended_value"),
            ),
            qwen35_witness_root_from_trace(
                self.layer_idx,
                NormAndZGate,
                norm.stage_idx,
                &norm.norm_weight_root,
                format!("safetensors:{linear_attn}.norm.weight"),
            ),
            qwen35_witness_root_from_trace(
                self.layer_idx,
                NormAndZGate,
                norm.z_projection_stage_idx,
                &norm.z_gate_root,
                format!("runtime:{linear_attn}.z_gate"),
            ),
            qwen35_witness_root_from_trace(
                self.layer_idx,
                NormAndZGate,
                norm.consumer_stage_idx,
                &norm.output_root,
                format!("runtime:{linear_attn}.gated_value"),
            ),
        ];

        let qk_head_dim = delta.query_width / delta.state_rows;
        let recurrent_state_shape = Matrix(Qwen35Tensor2DShape {
            rows: delta.state_rows * qk_head_dim,
            cols: delta.value_head_dim,
        });
        roots.push(qwen35_synthetic_witness_root(
            self.layer_idx,
            DeltaRecurrence,
            delta.stage_idx,
            "initial_recurrent_state",
            RecurrentState,
            recurrent_state_shape,
            format!("conversation-state:{linear_attn}:initial_recurrent_state"),
        ));
        roots.push(qwen35_synthetic_witness_root(
            self.layer_idx,
            DeltaRecurrence,
            delta.stage_idx,
            "final_recurrent_state",
            RecurrentState,
            recurrent_state_shape,
            format!("conversation-state:{linear_attn}:final_recurrent_state"),
        ));
        roots.push(qwen35_synthetic_witness_root(
            self.layer_idx,
            NormAndZGate,
            norm.stage_idx,
            "rsqrt_table_commitment",
            LookupTable,
            Vector(0),
            format!("statement:{linear_attn}:norm_and_z_gate_rsqrt_table"),
        ));

        let mut seen = HashSet::new();
        for root in &roots {
            if !seen.insert(root.root_hash()) {
                return Err(format!(
                    "duplicate typed witness root in layer {}: {}",
                    self.layer_idx, root.name
                ));
            }
        }

        Ok(Qwen35TypedWitnessLayerManifest {
            layer_idx: self.layer_idx,
            seq_len: self.seq_len,
            depthwise_conv1d_trace_binding_hash: depthwise.contract_hash(),
            delta_recurrence_trace_binding_hash: delta.contract_hash(),
            norm_and_z_gate_trace_binding_hash: norm.contract_hash(),
            roots,
        })
    }

    pub fn depthwise_conv1d_trace_binding_contract(
        &self,
    ) -> Result<Qwen35DepthwiseConv1dTraceBindingContract, String> {
        use Qwen35GatedDeltaNetStageKind::{DepthwiseConv1d, QkvProjection, QkvSplit};
        use Qwen35TensorShape::{Matrix, Tensor3D};

        let stages = self.stage_contracts();
        let producer = stages
            .iter()
            .find(|stage| stage.kind == QkvProjection)
            .ok_or_else(|| "missing QkvProjection stage".to_string())?;
        let depthwise = stages
            .iter()
            .find(|stage| stage.kind == DepthwiseConv1d)
            .ok_or_else(|| "missing DepthwiseConv1d stage".to_string())?;
        let consumer = stages
            .iter()
            .find(|stage| stage.kind == QkvSplit)
            .ok_or_else(|| "missing QkvSplit stage".to_string())?;

        let producer_output = producer
            .outputs
            .iter()
            .find(|tensor| tensor.name == "qkv_projected")
            .ok_or_else(|| "QkvProjection does not output qkv_projected".to_string())?;
        let depthwise_input = depthwise
            .inputs
            .iter()
            .find(|tensor| tensor.name == "qkv_projected")
            .ok_or_else(|| "DepthwiseConv1d does not input qkv_projected".to_string())?;
        let depthwise_weight = depthwise
            .inputs
            .iter()
            .find(|tensor| tensor.name == "conv1d_weight")
            .ok_or_else(|| "DepthwiseConv1d does not input conv1d_weight".to_string())?;
        let depthwise_output = depthwise
            .outputs
            .iter()
            .find(|tensor| tensor.name == "qkv_after_conv")
            .ok_or_else(|| "DepthwiseConv1d does not output qkv_after_conv".to_string())?;
        let consumer_input = consumer
            .inputs
            .iter()
            .find(|tensor| tensor.name == "qkv_after_conv")
            .ok_or_else(|| "QkvSplit does not input qkv_after_conv".to_string())?;

        if producer_output.shape != depthwise_input.shape {
            return Err(
                "QkvProjection output root does not match DepthwiseConv1d input root".to_string(),
            );
        }
        if depthwise_output.shape != consumer_input.shape {
            return Err(
                "DepthwiseConv1d output root does not match QkvSplit input root".to_string(),
            );
        }
        if depthwise_input.shape != Matrix(self.qkv_projected)
            || depthwise_weight.shape != Tensor3D(self.conv1d_weight)
            || depthwise_output.shape != Matrix(self.qkv_after_conv)
        {
            return Err(
                "DepthwiseConv1d trace binding shapes do not match layer contract".to_string(),
            );
        }

        let air_contract = self.depthwise_conv1d_air_contract();
        Ok(Qwen35DepthwiseConv1dTraceBindingContract {
            layer_idx: self.layer_idx,
            seq_len: self.seq_len,
            channels: air_contract.channels,
            kernel: air_contract.kernel,
            stage_idx: depthwise.stage_idx,
            producer_stage_idx: producer.stage_idx,
            consumer_stage_idx: consumer.stage_idx,
            input_root: Qwen35TraceRootContract {
                name: "qkv_projected",
                role: Qwen35TraceRootRole::ProducerActivation,
                shape: Matrix(self.qkv_projected),
            },
            weight_root: Qwen35TraceRootContract {
                name: "conv1d_weight",
                role: Qwen35TraceRootRole::ModelWeight,
                shape: Tensor3D(self.conv1d_weight),
            },
            output_root: Qwen35TraceRootContract {
                name: "qkv_after_conv",
                role: Qwen35TraceRootRole::ConsumerActivation,
                shape: Matrix(self.qkv_after_conv),
            },
            stage_contract_hash: depthwise.contract_hash(self.layer_idx, self.seq_len),
            air_contract_hash: air_contract.contract_hash(),
        })
    }

    pub fn delta_recurrence_trace_binding_contract(
        &self,
    ) -> Result<Qwen35DeltaRecurrenceTraceBindingContract, String> {
        use Qwen35GatedDeltaNetStageKind::{ABProjection, DeltaRecurrence, NormAndZGate, QkvSplit};
        use Qwen35TensorShape::{Matrix, Vector};

        let stages = self.stage_contracts();
        let qkv_split = stages
            .iter()
            .find(|stage| stage.kind == QkvSplit)
            .ok_or_else(|| "missing QkvSplit stage".to_string())?;
        let ab_projection = stages
            .iter()
            .find(|stage| stage.kind == ABProjection)
            .ok_or_else(|| "missing ABProjection stage".to_string())?;
        let recurrence = stages
            .iter()
            .find(|stage| stage.kind == DeltaRecurrence)
            .ok_or_else(|| "missing DeltaRecurrence stage".to_string())?;
        let consumer = stages
            .iter()
            .find(|stage| stage.kind == NormAndZGate)
            .ok_or_else(|| "missing NormAndZGate stage".to_string())?;

        let qkv_query = qkv_split
            .outputs
            .iter()
            .find(|tensor| tensor.name == "query")
            .ok_or_else(|| "QkvSplit does not output query".to_string())?;
        let qkv_key = qkv_split
            .outputs
            .iter()
            .find(|tensor| tensor.name == "key")
            .ok_or_else(|| "QkvSplit does not output key".to_string())?;
        let qkv_value = qkv_split
            .outputs
            .iter()
            .find(|tensor| tensor.name == "projected_value")
            .ok_or_else(|| "QkvSplit does not output projected_value".to_string())?;
        let ab_a = ab_projection
            .outputs
            .iter()
            .find(|tensor| tensor.name == "a_gate")
            .ok_or_else(|| "ABProjection does not output a_gate".to_string())?;
        let ab_b = ab_projection
            .outputs
            .iter()
            .find(|tensor| tensor.name == "b_gate")
            .ok_or_else(|| "ABProjection does not output b_gate".to_string())?;
        let recurrence_query = recurrence
            .inputs
            .iter()
            .find(|tensor| tensor.name == "query")
            .ok_or_else(|| "DeltaRecurrence does not input query".to_string())?;
        let recurrence_key = recurrence
            .inputs
            .iter()
            .find(|tensor| tensor.name == "key")
            .ok_or_else(|| "DeltaRecurrence does not input key".to_string())?;
        let recurrence_value = recurrence
            .inputs
            .iter()
            .find(|tensor| tensor.name == "projected_value")
            .ok_or_else(|| "DeltaRecurrence does not input projected_value".to_string())?;
        let recurrence_a = recurrence
            .inputs
            .iter()
            .find(|tensor| tensor.name == "a_gate")
            .ok_or_else(|| "DeltaRecurrence does not input a_gate".to_string())?;
        let recurrence_b = recurrence
            .inputs
            .iter()
            .find(|tensor| tensor.name == "b_gate")
            .ok_or_else(|| "DeltaRecurrence does not input b_gate".to_string())?;
        let recurrence_a_log = recurrence
            .inputs
            .iter()
            .find(|tensor| tensor.name == "a_log_weight")
            .ok_or_else(|| "DeltaRecurrence does not input a_log_weight".to_string())?;
        let recurrence_dt_bias = recurrence
            .inputs
            .iter()
            .find(|tensor| tensor.name == "dt_bias")
            .ok_or_else(|| "DeltaRecurrence does not input dt_bias".to_string())?;
        let recurrence_output = recurrence
            .outputs
            .iter()
            .find(|tensor| tensor.name == "attended_value")
            .ok_or_else(|| "DeltaRecurrence does not output attended_value".to_string())?;
        let consumer_input = consumer
            .inputs
            .iter()
            .find(|tensor| tensor.name == "attended_value")
            .ok_or_else(|| "NormAndZGate does not input attended_value".to_string())?;

        if qkv_query.shape != recurrence_query.shape
            || qkv_key.shape != recurrence_key.shape
            || qkv_value.shape != recurrence_value.shape
            || ab_a.shape != recurrence_a.shape
            || ab_b.shape != recurrence_b.shape
            || recurrence_output.shape != consumer_input.shape
        {
            return Err("DeltaRecurrence producer/consumer roots do not match".to_string());
        }
        if recurrence_query.shape != Matrix(self.query)
            || recurrence_key.shape != Matrix(self.key)
            || recurrence_value.shape != Matrix(self.projected_value)
            || recurrence_a.shape != Matrix(self.a_gate)
            || recurrence_b.shape != Matrix(self.b_gate)
            || recurrence_a_log.shape != Vector(self.a_log_weight)
            || recurrence_dt_bias.shape != Vector(self.dt_bias)
            || recurrence_output.shape != Matrix(self.attended_value)
        {
            return Err(
                "DeltaRecurrence trace binding shapes do not match layer contract".to_string(),
            );
        }

        Ok(Qwen35DeltaRecurrenceTraceBindingContract {
            layer_idx: self.layer_idx,
            seq_len: self.seq_len,
            query_width: self.query.cols,
            key_width: self.key.cols,
            value_width: self.projected_value.cols,
            state_rows: self.recurrent_state_rows,
            value_head_dim: self.linear_value_head_dim,
            stage_idx: recurrence.stage_idx,
            qkv_split_stage_idx: qkv_split.stage_idx,
            ab_projection_stage_idx: ab_projection.stage_idx,
            consumer_stage_idx: consumer.stage_idx,
            query_root: Qwen35TraceRootContract {
                name: "query",
                role: Qwen35TraceRootRole::ProducerActivation,
                shape: Matrix(self.query),
            },
            key_root: Qwen35TraceRootContract {
                name: "key",
                role: Qwen35TraceRootRole::ProducerActivation,
                shape: Matrix(self.key),
            },
            projected_value_root: Qwen35TraceRootContract {
                name: "projected_value",
                role: Qwen35TraceRootRole::ProducerActivation,
                shape: Matrix(self.projected_value),
            },
            a_gate_root: Qwen35TraceRootContract {
                name: "a_gate",
                role: Qwen35TraceRootRole::ProducerActivation,
                shape: Matrix(self.a_gate),
            },
            b_gate_root: Qwen35TraceRootContract {
                name: "b_gate",
                role: Qwen35TraceRootRole::ProducerActivation,
                shape: Matrix(self.b_gate),
            },
            a_log_weight_root: Qwen35TraceRootContract {
                name: "a_log_weight",
                role: Qwen35TraceRootRole::ModelWeight,
                shape: Vector(self.a_log_weight),
            },
            dt_bias_root: Qwen35TraceRootContract {
                name: "dt_bias",
                role: Qwen35TraceRootRole::ModelWeight,
                shape: Vector(self.dt_bias),
            },
            output_root: Qwen35TraceRootContract {
                name: "attended_value",
                role: Qwen35TraceRootRole::ConsumerActivation,
                shape: Matrix(self.attended_value),
            },
            stage_contract_hash: recurrence.contract_hash(self.layer_idx, self.seq_len),
        })
    }

    pub fn norm_and_z_gate_trace_binding_contract(
        &self,
    ) -> Result<Qwen35NormAndZGateTraceBindingContract, String> {
        use Qwen35GatedDeltaNetStageKind::{
            DeltaRecurrence, NormAndZGate, OutputProjection, ZProjection,
        };
        use Qwen35TensorShape::{Matrix, Vector};

        let stages = self.stage_contracts();
        let delta_recurrence = stages
            .iter()
            .find(|stage| stage.kind == DeltaRecurrence)
            .ok_or_else(|| "missing DeltaRecurrence stage".to_string())?;
        let z_projection = stages
            .iter()
            .find(|stage| stage.kind == ZProjection)
            .ok_or_else(|| "missing ZProjection stage".to_string())?;
        let norm_and_z_gate = stages
            .iter()
            .find(|stage| stage.kind == NormAndZGate)
            .ok_or_else(|| "missing NormAndZGate stage".to_string())?;
        let output_projection = stages
            .iter()
            .find(|stage| stage.kind == OutputProjection)
            .ok_or_else(|| "missing OutputProjection stage".to_string())?;

        let recurrence_output = delta_recurrence
            .outputs
            .iter()
            .find(|tensor| tensor.name == "attended_value")
            .ok_or_else(|| "DeltaRecurrence does not output attended_value".to_string())?;
        let norm_input = norm_and_z_gate
            .inputs
            .iter()
            .find(|tensor| tensor.name == "attended_value")
            .ok_or_else(|| "NormAndZGate does not input attended_value".to_string())?;
        let norm_weight = norm_and_z_gate
            .inputs
            .iter()
            .find(|tensor| tensor.name == "norm_weight")
            .ok_or_else(|| "NormAndZGate does not input norm_weight".to_string())?;
        let z_output = z_projection
            .outputs
            .iter()
            .find(|tensor| tensor.name == "z_gate")
            .ok_or_else(|| "ZProjection does not output z_gate".to_string())?;
        let z_input = norm_and_z_gate
            .inputs
            .iter()
            .find(|tensor| tensor.name == "z_gate")
            .ok_or_else(|| "NormAndZGate does not input z_gate".to_string())?;
        let norm_output = norm_and_z_gate
            .outputs
            .iter()
            .find(|tensor| tensor.name == "gated_value")
            .ok_or_else(|| "NormAndZGate does not output gated_value".to_string())?;
        let consumer_input = output_projection
            .inputs
            .iter()
            .find(|tensor| tensor.name == "gated_value")
            .ok_or_else(|| "OutputProjection does not input gated_value".to_string())?;

        if recurrence_output.shape != norm_input.shape
            || z_output.shape != z_input.shape
            || norm_output.shape != consumer_input.shape
        {
            return Err("NormAndZGate producer/consumer roots do not match".to_string());
        }
        if norm_input.shape != Matrix(self.attended_value)
            || norm_weight.shape != Vector(self.norm_weight)
            || z_input.shape != Matrix(self.z_gate)
            || norm_output.shape != Matrix(self.attended_value)
        {
            return Err(
                "NormAndZGate trace binding shapes do not match layer contract".to_string(),
            );
        }

        Ok(Qwen35NormAndZGateTraceBindingContract {
            layer_idx: self.layer_idx,
            seq_len: self.seq_len,
            width: self.attended_value.cols,
            norm_width: self.norm_weight,
            stage_idx: norm_and_z_gate.stage_idx,
            delta_recurrence_stage_idx: delta_recurrence.stage_idx,
            z_projection_stage_idx: z_projection.stage_idx,
            consumer_stage_idx: output_projection.stage_idx,
            attended_value_root: Qwen35TraceRootContract {
                name: "attended_value",
                role: Qwen35TraceRootRole::ProducerActivation,
                shape: Matrix(self.attended_value),
            },
            norm_weight_root: Qwen35TraceRootContract {
                name: "norm_weight",
                role: Qwen35TraceRootRole::ModelWeight,
                shape: Vector(self.norm_weight),
            },
            z_gate_root: Qwen35TraceRootContract {
                name: "z_gate",
                role: Qwen35TraceRootRole::ProducerActivation,
                shape: Matrix(self.z_gate),
            },
            output_root: Qwen35TraceRootContract {
                name: "gated_value",
                role: Qwen35TraceRootRole::ConsumerActivation,
                shape: Matrix(self.attended_value),
            },
            stage_contract_hash: norm_and_z_gate.contract_hash(self.layer_idx, self.seq_len),
        })
    }

    pub fn stage_status_counts(&self) -> (usize, usize, usize) {
        self.stage_contracts().iter().fold(
            (0usize, 0usize, 0usize),
            |(generic, dedicated_air, missing), stage| match stage.status {
                Qwen35StageStatus::GenericAvailable => (generic + 1, dedicated_air, missing),
                Qwen35StageStatus::DedicatedAirAvailableIntegrationMissing => {
                    (generic, dedicated_air + 1, missing)
                }
                Qwen35StageStatus::DedicatedMissing => (generic, dedicated_air, missing + 1),
            },
        )
    }

    pub fn stage_contracts(&self) -> Vec<Qwen35GatedDeltaNetStageContract> {
        use Qwen35GatedDeltaNetStageKind::*;
        use Qwen35StageStatus::{
            DedicatedAirAvailableIntegrationMissing, DedicatedMissing, GenericAvailable,
        };
        use Qwen35TensorShape::{Matrix, Tensor3D, Vector};

        vec![
            Qwen35GatedDeltaNetStageContract {
                stage_idx: 0,
                kind: QkvProjection,
                status: GenericAvailable,
                relation: "qkv_projected = input * transpose(in_proj_qkv_weight)",
                inputs: vec![
                    Qwen35StageTensor {
                        name: "input",
                        shape: Matrix(self.input),
                    },
                    Qwen35StageTensor {
                        name: "in_proj_qkv_weight",
                        shape: Matrix(self.in_proj_qkv_weight),
                    },
                ],
                outputs: vec![Qwen35StageTensor {
                    name: "qkv_projected",
                    shape: Matrix(self.qkv_projected),
                }],
            },
            Qwen35GatedDeltaNetStageContract {
                stage_idx: 1,
                kind: DepthwiseConv1d,
                status: DedicatedAirAvailableIntegrationMissing,
                relation: "qkv_after_conv = depthwise_conv1d(qkv_projected, conv1d_weight)",
                inputs: vec![
                    Qwen35StageTensor {
                        name: "qkv_projected",
                        shape: Matrix(self.qkv_projected),
                    },
                    Qwen35StageTensor {
                        name: "conv1d_weight",
                        shape: Tensor3D(self.conv1d_weight),
                    },
                ],
                outputs: vec![Qwen35StageTensor {
                    name: "qkv_after_conv",
                    shape: Matrix(self.qkv_after_conv),
                }],
            },
            Qwen35GatedDeltaNetStageContract {
                stage_idx: 2,
                kind: QkvSplit,
                status: GenericAvailable,
                relation:
                    "qkv_after_conv splits into raw q,k,value; q,k repeat to value-head count",
                inputs: vec![Qwen35StageTensor {
                    name: "qkv_after_conv",
                    shape: Matrix(self.qkv_after_conv),
                }],
                outputs: vec![
                    Qwen35StageTensor {
                        name: "query",
                        shape: Matrix(self.query),
                    },
                    Qwen35StageTensor {
                        name: "key",
                        shape: Matrix(self.key),
                    },
                    Qwen35StageTensor {
                        name: "projected_value",
                        shape: Matrix(self.projected_value),
                    },
                ],
            },
            Qwen35GatedDeltaNetStageContract {
                stage_idx: 3,
                kind: ZProjection,
                status: GenericAvailable,
                relation: "z_gate = input * transpose(in_proj_z_weight)",
                inputs: vec![
                    Qwen35StageTensor {
                        name: "input",
                        shape: Matrix(self.input),
                    },
                    Qwen35StageTensor {
                        name: "in_proj_z_weight",
                        shape: Matrix(self.in_proj_z_weight),
                    },
                ],
                outputs: vec![Qwen35StageTensor {
                    name: "z_gate",
                    shape: Matrix(self.z_gate),
                }],
            },
            Qwen35GatedDeltaNetStageContract {
                stage_idx: 4,
                kind: ABProjection,
                status: GenericAvailable,
                relation: "a_gate,b_gate = input * transpose(in_proj_a_weight/in_proj_b_weight)",
                inputs: vec![
                    Qwen35StageTensor {
                        name: "input",
                        shape: Matrix(self.input),
                    },
                    Qwen35StageTensor {
                        name: "in_proj_a_weight",
                        shape: Matrix(self.in_proj_a_weight),
                    },
                    Qwen35StageTensor {
                        name: "in_proj_b_weight",
                        shape: Matrix(self.in_proj_b_weight),
                    },
                ],
                outputs: vec![
                    Qwen35StageTensor {
                        name: "a_gate",
                        shape: Matrix(self.a_gate),
                    },
                    Qwen35StageTensor {
                        name: "b_gate",
                        shape: Matrix(self.b_gate),
                    },
                ],
            },
            Qwen35GatedDeltaNetStageContract {
                stage_idx: 5,
                kind: DeltaRecurrence,
                status: DedicatedMissing,
                relation:
                    "attended_value follows the gated delta recurrent update over q,k,v,a,b,a_log,dt_bias",
                inputs: vec![
                    Qwen35StageTensor {
                        name: "query",
                        shape: Matrix(self.query),
                    },
                    Qwen35StageTensor {
                        name: "key",
                        shape: Matrix(self.key),
                    },
                    Qwen35StageTensor {
                        name: "projected_value",
                        shape: Matrix(self.projected_value),
                    },
                    Qwen35StageTensor {
                        name: "a_gate",
                        shape: Matrix(self.a_gate),
                    },
                    Qwen35StageTensor {
                        name: "b_gate",
                        shape: Matrix(self.b_gate),
                    },
                    Qwen35StageTensor {
                        name: "a_log_weight",
                        shape: Vector(self.a_log_weight),
                    },
                    Qwen35StageTensor {
                        name: "dt_bias",
                        shape: Vector(self.dt_bias),
                    },
                ],
                outputs: vec![Qwen35StageTensor {
                    name: "attended_value",
                    shape: Matrix(self.attended_value),
                }],
            },
            Qwen35GatedDeltaNetStageContract {
                stage_idx: 6,
                kind: NormAndZGate,
                status: DedicatedMissing,
                relation: "gated_value = norm(attended_value, norm_weight) * z_gate",
                inputs: vec![
                    Qwen35StageTensor {
                        name: "attended_value",
                        shape: Matrix(self.attended_value),
                    },
                    Qwen35StageTensor {
                        name: "norm_weight",
                        shape: Vector(self.norm_weight),
                    },
                    Qwen35StageTensor {
                        name: "z_gate",
                        shape: Matrix(self.z_gate),
                    },
                ],
                outputs: vec![Qwen35StageTensor {
                    name: "gated_value",
                    shape: Matrix(self.attended_value),
                }],
            },
            Qwen35GatedDeltaNetStageContract {
                stage_idx: 7,
                kind: OutputProjection,
                status: GenericAvailable,
                relation: "output = gated_value * transpose(o_proj_weight)",
                inputs: vec![
                    Qwen35StageTensor {
                        name: "gated_value",
                        shape: Matrix(self.attended_value),
                    },
                    Qwen35StageTensor {
                        name: "o_proj_weight",
                        shape: Matrix(self.o_proj_weight),
                    },
                ],
                outputs: vec![Qwen35StageTensor {
                    name: "output",
                    shape: Matrix(self.output),
                }],
            },
        ]
    }

    pub fn stage_contract_hash(&self) -> FieldElement {
        let mut felts = vec![
            FieldElement::from(DOMAIN_QWEN35_GDN_STAGE),
            FieldElement::from(self.layer_idx as u64),
            FieldElement::from(self.seq_len as u64),
        ];
        for stage in self.stage_contracts() {
            felts.push(stage.contract_hash(self.layer_idx, self.seq_len));
        }
        starknet_crypto::poseidon_hash_many(&felts)
    }
}

impl Qwen35DepthwiseConv1dAirContract {
    pub fn contract_hash(&self) -> FieldElement {
        let mut felts = vec![
            FieldElement::from(DOMAIN_QWEN35_DEPTHWISE_CONV1D),
            FieldElement::from(self.layer_idx as u64),
            FieldElement::from(self.seq_len as u64),
            FieldElement::from(self.channels as u64),
            FieldElement::from(self.kernel as u64),
            FieldElement::from(self.logical_trace_rows as u64),
            FieldElement::from(self.deterministic_columns as u64),
            FieldElement::from(self.witness_columns as u64),
            FieldElement::from(self.total_columns as u64),
            FieldElement::from(self.arithmetic_constraints_per_row as u64),
            FieldElement::from(self.row_binding_constraints_per_row as u64),
        ];
        for shape in [
            Qwen35TensorShape::Matrix(self.input),
            Qwen35TensorShape::Tensor3D(self.weight),
            Qwen35TensorShape::Matrix(self.output),
        ] {
            let dims = shape.dimensions();
            felts.extend([
                FieldElement::from(shape.hash_tag()),
                FieldElement::from(dims[0] as u64),
                FieldElement::from(dims[1] as u64),
                FieldElement::from(dims[2] as u64),
            ]);
        }
        for offset in &self.tap_offsets {
            let shifted = *offset + self.kernel as isize;
            felts.push(FieldElement::from(shifted as u64));
        }
        starknet_crypto::poseidon_hash_many(&felts)
    }

    pub fn relation(&self) -> String {
        format!(
            "for each token t and channel c: output[t,c] = sum_{{tap=0..{}}} valid(t,tap) * input[t + tap + 1 - kernel,c] * weight[c,0,tap]",
            self.kernel.saturating_sub(1),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35ProofPlan {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_rows: usize,
    pub linear_value_rows: usize,
    pub linear_qkv_rows: usize,
    pub linear_state_rows: usize,
    pub linear_conv_kernel_dim: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub routed_ff: usize,
    pub shared_ff: usize,
    pub layers: Vec<Qwen35LayerPlan>,
}

impl Qwen35ProofPlan {
    pub fn from_hf_config(cfg: &HfConfig) -> Result<Self, String> {
        if cfg.model_type != "qwen3_5_moe" {
            return Err(format!("expected qwen3_5_moe, got {}", cfg.model_type));
        }
        if cfg.layer_types.len() != cfg.num_hidden_layers {
            return Err(format!(
                "layer_types length {} != num_hidden_layers {}",
                cfg.layer_types.len(),
                cfg.num_hidden_layers
            ));
        }

        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let routed_ff = cfg.moe_intermediate_size.unwrap_or(cfg.intermediate_size);
        let shared_ff = cfg
            .shared_expert_intermediate_size
            .unwrap_or(cfg.intermediate_size);
        let linear_key_head_dim = cfg.linear_key_head_dim.unwrap_or(0);
        let linear_value_head_dim = cfg.linear_value_head_dim.unwrap_or(0);
        let linear_num_key_heads = cfg.linear_num_key_heads.unwrap_or(0);
        let linear_num_value_heads = cfg.linear_num_value_heads.unwrap_or(0);
        let linear_key_rows = linear_num_key_heads * linear_key_head_dim;
        let linear_value_rows = linear_num_value_heads * linear_value_head_dim;
        let linear_state_rows = linear_num_value_heads;
        let linear_conv_kernel_dim = cfg.linear_conv_kernel_dim.unwrap_or(0);

        if cfg.num_experts == 0 || cfg.num_experts_per_tok == 0 {
            return Err("Qwen3.5-MoE requires non-zero num_experts/top_k".to_string());
        }
        if linear_key_rows == 0 || linear_value_rows == 0 || linear_state_rows == 0 {
            return Err("Qwen3.5 linear attention dimensions are missing".to_string());
        }
        if linear_conv_kernel_dim == 0 {
            return Err("Qwen3.5 linear_conv_kernel_dim is missing".to_string());
        }

        let linear_qkv_rows = 2 * linear_key_rows + linear_value_rows;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (layer_idx, kind) in cfg.layer_types.iter().enumerate() {
            let attention = match kind.as_str() {
                "linear_attention" => Qwen35AttentionKind::GatedDeltaNet,
                "full_attention" => Qwen35AttentionKind::GatedFullAttention,
                other => {
                    return Err(format!("unsupported layer_types[{layer_idx}]={other}"));
                }
            };

            let mut obligations = Vec::new();
            obligations.push(Qwen35ProofObligation::InputRmsNorm);
            match attention {
                Qwen35AttentionKind::GatedDeltaNet => {
                    obligations.push(Qwen35ProofObligation::GatedDeltaNet {
                        qkv_rows: linear_qkv_rows,
                        z_rows: linear_value_rows,
                        state_rows: linear_state_rows,
                        conv_kernel: linear_conv_kernel_dim,
                    });
                }
                Qwen35AttentionKind::GatedFullAttention => {
                    obligations.push(Qwen35ProofObligation::GatedFullAttention {
                        q_rows_with_gate: 2 * q_dim,
                        q_rows: q_dim,
                        kv_rows: kv_dim,
                    });
                }
            }
            obligations.push(Qwen35ProofObligation::ResidualAdd);
            obligations.push(Qwen35ProofObligation::PostAttentionRmsNorm);
            obligations.push(Qwen35ProofObligation::RouterTopK {
                num_experts: cfg.num_experts,
                top_k: cfg.num_experts_per_tok,
            });
            obligations.push(Qwen35ProofObligation::PackedExpertBank {
                num_experts: cfg.num_experts,
                routed_ff,
            });
            obligations.push(Qwen35ProofObligation::SharedExpert { shared_ff });
            obligations.push(Qwen35ProofObligation::SharedExpertGate);
            obligations.push(Qwen35ProofObligation::ResidualAdd);

            layers.push(Qwen35LayerPlan {
                layer_idx,
                attention,
                obligations,
            });
        }

        Ok(Self {
            hidden_size: cfg.hidden_size,
            vocab_size: cfg.vocab_size,
            num_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            q_dim,
            kv_dim,
            linear_key_head_dim,
            linear_value_head_dim,
            linear_num_key_heads,
            linear_num_value_heads,
            linear_key_rows,
            linear_value_rows,
            linear_qkv_rows,
            linear_state_rows,
            linear_conv_kernel_dim,
            num_experts: cfg.num_experts,
            top_k: cfg.num_experts_per_tok,
            routed_ff,
            shared_ff,
            layers,
        })
    }

    pub fn linear_attention_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| layer.attention == Qwen35AttentionKind::GatedDeltaNet)
            .count()
    }

    pub fn full_attention_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| layer.attention == Qwen35AttentionKind::GatedFullAttention)
            .count()
    }

    pub fn expected_language_tensor_count(&self) -> usize {
        let top_level = 3;
        let per_layer_common = 9 * self.num_layers;
        let linear_attention = 9 * self.linear_attention_layers();
        let full_attention = 6 * self.full_attention_layers();
        top_level + per_layer_common + linear_attention + full_attention
    }

    pub fn gated_delta_net_stage_readiness(
        &self,
        seq_len: usize,
    ) -> Result<Qwen35GatedDeltaNetStageReadiness, String> {
        let mut total_stages = 0usize;
        let mut generic_ready_stages = 0usize;
        let mut dedicated_air_available_stages = 0usize;
        let mut missing_dedicated_stages = 0usize;
        let mut dedicated_air_stage_counts =
            std::collections::BTreeMap::<Qwen35GatedDeltaNetStageKind, usize>::new();
        let mut missing_stage_counts =
            std::collections::BTreeMap::<Qwen35GatedDeltaNetStageKind, usize>::new();

        for layer in &self.layers {
            if layer.attention != Qwen35AttentionKind::GatedDeltaNet {
                continue;
            }
            let contract = self.gated_delta_net_contract(layer.layer_idx, seq_len)?;
            for stage in contract.stage_contracts() {
                total_stages += 1;
                match stage.status {
                    Qwen35StageStatus::GenericAvailable => generic_ready_stages += 1,
                    Qwen35StageStatus::DedicatedAirAvailableIntegrationMissing => {
                        dedicated_air_available_stages += 1;
                        *dedicated_air_stage_counts.entry(stage.kind).or_insert(0) += 1;
                    }
                    Qwen35StageStatus::DedicatedMissing => {
                        missing_dedicated_stages += 1;
                        *missing_stage_counts.entry(stage.kind).or_insert(0) += 1;
                    }
                }
            }
        }

        Ok(Qwen35GatedDeltaNetStageReadiness {
            total_stages,
            generic_ready_stages,
            dedicated_air_available_stages,
            missing_dedicated_stages,
            dedicated_air_stage_counts: dedicated_air_stage_counts.into_iter().collect(),
            missing_stage_counts: missing_stage_counts.into_iter().collect(),
        })
    }

    pub fn depthwise_conv1d_air_readiness(
        &self,
        seq_len: usize,
    ) -> Result<Qwen35DepthwiseConv1dAirReadiness, String> {
        let mut contracts = Vec::new();
        for layer in &self.layers {
            if layer.attention != Qwen35AttentionKind::GatedDeltaNet {
                continue;
            }
            contracts.push(
                self.gated_delta_net_contract(layer.layer_idx, seq_len)?
                    .depthwise_conv1d_air_contract(),
            );
        }
        let first = contracts
            .first()
            .ok_or_else(|| "missing GatedDeltaNet layers".to_string())?;
        for contract in &contracts {
            if contract.channels != first.channels
                || contract.kernel != first.kernel
                || contract.total_columns != first.total_columns
                || contract.arithmetic_constraints_per_row != first.arithmetic_constraints_per_row
                || contract.row_binding_constraints_per_row != first.row_binding_constraints_per_row
            {
                return Err("inconsistent DepthwiseConv1D AIR contract across layers".to_string());
            }
        }

        let mut felts = vec![
            FieldElement::from(DOMAIN_QWEN35_DEPTHWISE_CONV1D),
            FieldElement::from(seq_len as u64),
            FieldElement::from(contracts.len() as u64),
        ];
        let mut trace_binding_felts = vec![
            FieldElement::from(DOMAIN_QWEN35_TRACE_BINDING),
            FieldElement::from(seq_len as u64),
            FieldElement::from(contracts.len() as u64),
        ];
        let mut total_logical_rows = 0usize;
        for contract in &contracts {
            total_logical_rows += contract.logical_trace_rows;
            felts.push(contract.contract_hash());
        }
        for layer in &self.layers {
            if layer.attention != Qwen35AttentionKind::GatedDeltaNet {
                continue;
            }
            trace_binding_felts.push(
                self.gated_delta_net_contract(layer.layer_idx, seq_len)?
                    .depthwise_conv1d_trace_binding_contract()?
                    .contract_hash(),
            );
        }

        Ok(Qwen35DepthwiseConv1dAirReadiness {
            seq_len,
            layers: contracts.len(),
            channels_per_layer: first.channels,
            kernel: first.kernel,
            logical_rows_per_layer: first.logical_trace_rows,
            total_logical_rows,
            columns_per_layer: first.total_columns,
            arithmetic_constraints_per_row: first.arithmetic_constraints_per_row,
            row_binding_constraints_per_row: first.row_binding_constraints_per_row,
            aggregate_contract_hash: starknet_crypto::poseidon_hash_many(&felts),
            aggregate_trace_binding_hash: starknet_crypto::poseidon_hash_many(&trace_binding_felts),
        })
    }

    pub fn typed_proof_ledger(&self, seq_len: usize) -> Qwen35TypedProofLedger {
        Qwen35TypedProofLedger::new(
            self.architecture_contract_hash(),
            seq_len,
            self.linear_attention_layers(),
            self.linear_attention_layers(),
            self.linear_attention_layers(),
        )
    }

    pub fn typed_witness_manifest(
        &self,
        seq_len: usize,
    ) -> Result<Qwen35TypedWitnessManifest, String> {
        if seq_len == 0 {
            return Err("seq_len must be non-zero".to_string());
        }

        let mut layers = Vec::new();
        for layer in &self.layers {
            if layer.attention != Qwen35AttentionKind::GatedDeltaNet {
                continue;
            }
            layers.push(
                self.gated_delta_net_contract(layer.layer_idx, seq_len)?
                    .typed_witness_layer_manifest()?,
            );
        }

        let architecture_contract_hash = self.architecture_contract_hash();
        let manifest_hash =
            Qwen35TypedWitnessManifest::compute_hash(architecture_contract_hash, seq_len, &layers);

        Ok(Qwen35TypedWitnessManifest {
            architecture_contract_hash,
            seq_len,
            layers,
            manifest_hash,
        })
    }

    pub fn execution_plan(&self) -> Qwen35ExecutionPlan {
        let mut steps = Vec::with_capacity(3 + 7 * self.num_layers);
        let push = |steps: &mut Vec<Qwen35ProofStep>,
                    layer_idx: Option<usize>,
                    component: Qwen35ProofComponent,
                    status: Qwen35ComponentStatus| {
            let step_idx = steps.len();
            steps.push(Qwen35ProofStep {
                step_idx,
                layer_idx,
                component,
                status,
            });
        };

        push(
            &mut steps,
            None,
            Qwen35ProofComponent::TokenEmbedding,
            Qwen35ComponentStatus::GenericAvailable,
        );

        for layer in &self.layers {
            let layer_idx = Some(layer.layer_idx);
            push(
                &mut steps,
                layer_idx,
                Qwen35ProofComponent::InputRmsNorm,
                Qwen35ComponentStatus::GenericAvailable,
            );
            match layer.attention {
                Qwen35AttentionKind::GatedDeltaNet => push(
                    &mut steps,
                    layer_idx,
                    Qwen35ProofComponent::GatedDeltaNet,
                    Qwen35ComponentStatus::DedicatedMissing,
                ),
                Qwen35AttentionKind::GatedFullAttention => push(
                    &mut steps,
                    layer_idx,
                    Qwen35ProofComponent::GatedFullAttention,
                    Qwen35ComponentStatus::DedicatedMissing,
                ),
            }
            push(
                &mut steps,
                layer_idx,
                Qwen35ProofComponent::AttentionResidualAdd,
                Qwen35ComponentStatus::GenericAvailable,
            );
            push(
                &mut steps,
                layer_idx,
                Qwen35ProofComponent::PostAttentionRmsNorm,
                Qwen35ComponentStatus::GenericAvailable,
            );
            push(
                &mut steps,
                layer_idx,
                Qwen35ProofComponent::RouterTopK,
                Qwen35ComponentStatus::GenericAvailable,
            );
            push(
                &mut steps,
                layer_idx,
                Qwen35ProofComponent::PackedExpertBank,
                Qwen35ComponentStatus::DedicatedMissing,
            );
            push(
                &mut steps,
                layer_idx,
                Qwen35ProofComponent::SharedExpert,
                Qwen35ComponentStatus::DedicatedMissing,
            );
            push(
                &mut steps,
                layer_idx,
                Qwen35ProofComponent::SharedExpertGate,
                Qwen35ComponentStatus::DedicatedMissing,
            );
            push(
                &mut steps,
                layer_idx,
                Qwen35ProofComponent::MlpResidualAdd,
                Qwen35ComponentStatus::GenericAvailable,
            );
        }

        push(
            &mut steps,
            None,
            Qwen35ProofComponent::FinalNorm,
            Qwen35ComponentStatus::GenericAvailable,
        );
        push(
            &mut steps,
            None,
            Qwen35ProofComponent::LmHead,
            Qwen35ComponentStatus::GenericAvailable,
        );

        Qwen35ExecutionPlan {
            contract_hash: self.architecture_contract_hash(),
            steps,
        }
    }

    pub fn expected_tensor_contract(&self) -> Vec<Qwen35TensorContractEntry> {
        let mut expected = Vec::with_capacity(self.expected_language_tensor_count());
        expected.push(Qwen35TensorContractEntry {
            name: "model.language_model.embed_tokens.weight".to_string(),
            shape: vec![self.vocab_size, self.hidden_size],
            role: Qwen35TensorRole::TokenEmbedding,
            layer_idx: None,
        });
        expected.push(Qwen35TensorContractEntry {
            name: "model.language_model.norm.weight".to_string(),
            shape: vec![self.hidden_size],
            role: Qwen35TensorRole::FinalNorm,
            layer_idx: None,
        });
        expected.push(Qwen35TensorContractEntry {
            name: "lm_head.weight".to_string(),
            shape: vec![self.vocab_size, self.hidden_size],
            role: Qwen35TensorRole::LmHead,
            layer_idx: None,
        });

        for layer in &self.layers {
            let layer_idx = layer.layer_idx;
            let prefix = format!("model.language_model.layers.{layer_idx}");
            expected.extend([
                Qwen35TensorContractEntry {
                    name: format!("{prefix}.input_layernorm.weight"),
                    shape: vec![self.hidden_size],
                    role: Qwen35TensorRole::InputRmsNorm,
                    layer_idx: Some(layer_idx),
                },
                Qwen35TensorContractEntry {
                    name: format!("{prefix}.post_attention_layernorm.weight"),
                    shape: vec![self.hidden_size],
                    role: Qwen35TensorRole::PostAttentionRmsNorm,
                    layer_idx: Some(layer_idx),
                },
                Qwen35TensorContractEntry {
                    name: format!("{prefix}.mlp.gate.weight"),
                    shape: vec![self.num_experts, self.hidden_size],
                    role: Qwen35TensorRole::MoeRouter,
                    layer_idx: Some(layer_idx),
                },
                Qwen35TensorContractEntry {
                    name: format!("{prefix}.mlp.experts.gate_up_proj"),
                    shape: vec![self.num_experts, 2 * self.routed_ff, self.hidden_size],
                    role: Qwen35TensorRole::MoePackedGateUp,
                    layer_idx: Some(layer_idx),
                },
                Qwen35TensorContractEntry {
                    name: format!("{prefix}.mlp.experts.down_proj"),
                    shape: vec![self.num_experts, self.hidden_size, self.routed_ff],
                    role: Qwen35TensorRole::MoePackedDown,
                    layer_idx: Some(layer_idx),
                },
                Qwen35TensorContractEntry {
                    name: format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
                    shape: vec![self.shared_ff, self.hidden_size],
                    role: Qwen35TensorRole::SharedExpertGateProj,
                    layer_idx: Some(layer_idx),
                },
                Qwen35TensorContractEntry {
                    name: format!("{prefix}.mlp.shared_expert.up_proj.weight"),
                    shape: vec![self.shared_ff, self.hidden_size],
                    role: Qwen35TensorRole::SharedExpertUpProj,
                    layer_idx: Some(layer_idx),
                },
                Qwen35TensorContractEntry {
                    name: format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                    shape: vec![self.hidden_size, self.shared_ff],
                    role: Qwen35TensorRole::SharedExpertDownProj,
                    layer_idx: Some(layer_idx),
                },
                Qwen35TensorContractEntry {
                    name: format!("{prefix}.mlp.shared_expert_gate.weight"),
                    shape: vec![1, self.hidden_size],
                    role: Qwen35TensorRole::SharedExpertGate,
                    layer_idx: Some(layer_idx),
                },
            ]);

            match layer.attention {
                Qwen35AttentionKind::GatedDeltaNet => {
                    let attn = format!("{prefix}.linear_attn");
                    let linear_value_head_dim = self.linear_value_rows / self.linear_state_rows;
                    expected.extend([
                        Qwen35TensorContractEntry {
                            name: format!("{attn}.A_log"),
                            shape: vec![self.linear_state_rows],
                            role: Qwen35TensorRole::LinearAttentionALog,
                            layer_idx: Some(layer_idx),
                        },
                        Qwen35TensorContractEntry {
                            name: format!("{attn}.dt_bias"),
                            shape: vec![self.linear_state_rows],
                            role: Qwen35TensorRole::LinearAttentionDtBias,
                            layer_idx: Some(layer_idx),
                        },
                        Qwen35TensorContractEntry {
                            name: format!("{attn}.conv1d.weight"),
                            shape: vec![self.linear_qkv_rows, 1, self.linear_conv_kernel_dim],
                            role: Qwen35TensorRole::LinearAttentionConv1d,
                            layer_idx: Some(layer_idx),
                        },
                        Qwen35TensorContractEntry {
                            name: format!("{attn}.in_proj_a.weight"),
                            shape: vec![self.linear_state_rows, self.hidden_size],
                            role: Qwen35TensorRole::LinearAttentionInProjA,
                            layer_idx: Some(layer_idx),
                        },
                        Qwen35TensorContractEntry {
                            name: format!("{attn}.in_proj_b.weight"),
                            shape: vec![self.linear_state_rows, self.hidden_size],
                            role: Qwen35TensorRole::LinearAttentionInProjB,
                            layer_idx: Some(layer_idx),
                        },
                        Qwen35TensorContractEntry {
                            name: format!("{attn}.in_proj_qkv.weight"),
                            shape: vec![self.linear_qkv_rows, self.hidden_size],
                            role: Qwen35TensorRole::LinearAttentionInProjQkv,
                            layer_idx: Some(layer_idx),
                        },
                        Qwen35TensorContractEntry {
                            name: format!("{attn}.in_proj_z.weight"),
                            shape: vec![self.linear_value_rows, self.hidden_size],
                            role: Qwen35TensorRole::LinearAttentionInProjZ,
                            layer_idx: Some(layer_idx),
                        },
                        Qwen35TensorContractEntry {
                            name: format!("{attn}.norm.weight"),
                            shape: vec![linear_value_head_dim],
                            role: Qwen35TensorRole::LinearAttentionNorm,
                            layer_idx: Some(layer_idx),
                        },
                        Qwen35TensorContractEntry {
                            name: format!("{attn}.out_proj.weight"),
                            shape: vec![self.hidden_size, self.linear_value_rows],
                            role: Qwen35TensorRole::LinearAttentionOutProj,
                            layer_idx: Some(layer_idx),
                        },
                    ]);
                }
                Qwen35AttentionKind::GatedFullAttention => {
                    let attn = format!("{prefix}.self_attn");
                    expected.extend([
                        Qwen35TensorContractEntry {
                            name: format!("{attn}.q_proj.weight"),
                            shape: vec![2 * self.q_dim, self.hidden_size],
                            role: Qwen35TensorRole::FullAttentionQProj,
                            layer_idx: Some(layer_idx),
                        },
                        Qwen35TensorContractEntry {
                            name: format!("{attn}.k_proj.weight"),
                            shape: vec![self.kv_dim, self.hidden_size],
                            role: Qwen35TensorRole::FullAttentionKProj,
                            layer_idx: Some(layer_idx),
                        },
                        Qwen35TensorContractEntry {
                            name: format!("{attn}.v_proj.weight"),
                            shape: vec![self.kv_dim, self.hidden_size],
                            role: Qwen35TensorRole::FullAttentionVProj,
                            layer_idx: Some(layer_idx),
                        },
                        Qwen35TensorContractEntry {
                            name: format!("{attn}.o_proj.weight"),
                            shape: vec![self.hidden_size, self.q_dim],
                            role: Qwen35TensorRole::FullAttentionOProj,
                            layer_idx: Some(layer_idx),
                        },
                        Qwen35TensorContractEntry {
                            name: format!("{attn}.q_norm.weight"),
                            shape: vec![self.head_dim],
                            role: Qwen35TensorRole::FullAttentionQNorm,
                            layer_idx: Some(layer_idx),
                        },
                        Qwen35TensorContractEntry {
                            name: format!("{attn}.k_norm.weight"),
                            shape: vec![self.head_dim],
                            role: Qwen35TensorRole::FullAttentionKNorm,
                            layer_idx: Some(layer_idx),
                        },
                    ]);
                }
            }
        }

        expected
    }

    pub fn summary(&self) -> String {
        format!(
            "{} layers: {} GatedDeltaNet, {} gated full attention; hidden={}, vocab={}, q_dim={}, kv_dim={}, MoE={} top{}, routed_ff={}, shared_ff={}, expected_language_tensors={}",
            self.num_layers,
            self.linear_attention_layers(),
            self.full_attention_layers(),
            self.hidden_size,
            self.vocab_size,
            self.q_dim,
            self.kv_dim,
            self.num_experts,
            self.top_k,
            self.routed_ff,
            self.shared_ff,
            self.expected_language_tensor_count(),
        )
    }

    pub fn architecture_contract_hash(&self) -> FieldElement {
        let mut felts = vec![
            FieldElement::from(DOMAIN_QWEN35_CONTRACT),
            FieldElement::from(self.hidden_size as u64),
            FieldElement::from(self.vocab_size as u64),
            FieldElement::from(self.num_layers as u64),
            FieldElement::from(self.num_attention_heads as u64),
            FieldElement::from(self.num_key_value_heads as u64),
            FieldElement::from(self.head_dim as u64),
            FieldElement::from(self.q_dim as u64),
            FieldElement::from(self.kv_dim as u64),
            FieldElement::from(self.linear_key_head_dim as u64),
            FieldElement::from(self.linear_value_head_dim as u64),
            FieldElement::from(self.linear_num_key_heads as u64),
            FieldElement::from(self.linear_num_value_heads as u64),
            FieldElement::from(self.linear_key_rows as u64),
            FieldElement::from(self.linear_value_rows as u64),
            FieldElement::from(self.linear_qkv_rows as u64),
            FieldElement::from(self.linear_state_rows as u64),
            FieldElement::from(self.linear_conv_kernel_dim as u64),
            FieldElement::from(self.num_experts as u64),
            FieldElement::from(self.top_k as u64),
            FieldElement::from(self.routed_ff as u64),
            FieldElement::from(self.shared_ff as u64),
        ];

        for layer in &self.layers {
            felts.push(FieldElement::from(DOMAIN_QWEN35_LAYER));
            felts.push(FieldElement::from(layer.layer_idx as u64));
            felts.push(FieldElement::from(match layer.attention {
                Qwen35AttentionKind::GatedDeltaNet => KIND_GATED_DELTA_NET,
                Qwen35AttentionKind::GatedFullAttention => KIND_GATED_FULL_ATTENTION,
            }));
            for obligation in &layer.obligations {
                match *obligation {
                    Qwen35ProofObligation::InputRmsNorm => {
                        felts.extend([
                            FieldElement::from(1u64),
                            FieldElement::from(self.hidden_size as u64),
                        ]);
                    }
                    Qwen35ProofObligation::PostAttentionRmsNorm => {
                        felts.extend([
                            FieldElement::from(2u64),
                            FieldElement::from(self.hidden_size as u64),
                        ]);
                    }
                    Qwen35ProofObligation::GatedDeltaNet {
                        qkv_rows,
                        z_rows,
                        state_rows,
                        conv_kernel,
                    } => {
                        felts.extend([
                            FieldElement::from(3u64),
                            FieldElement::from(qkv_rows as u64),
                            FieldElement::from(z_rows as u64),
                            FieldElement::from(state_rows as u64),
                            FieldElement::from(conv_kernel as u64),
                        ]);
                    }
                    Qwen35ProofObligation::GatedFullAttention {
                        q_rows_with_gate,
                        q_rows,
                        kv_rows,
                    } => {
                        felts.extend([
                            FieldElement::from(4u64),
                            FieldElement::from(q_rows_with_gate as u64),
                            FieldElement::from(q_rows as u64),
                            FieldElement::from(kv_rows as u64),
                        ]);
                    }
                    Qwen35ProofObligation::RouterTopK { num_experts, top_k } => {
                        felts.extend([
                            FieldElement::from(5u64),
                            FieldElement::from(num_experts as u64),
                            FieldElement::from(top_k as u64),
                        ]);
                    }
                    Qwen35ProofObligation::PackedExpertBank {
                        num_experts,
                        routed_ff,
                    } => {
                        felts.extend([
                            FieldElement::from(6u64),
                            FieldElement::from(num_experts as u64),
                            FieldElement::from(routed_ff as u64),
                        ]);
                    }
                    Qwen35ProofObligation::SharedExpert { shared_ff } => {
                        felts.extend([
                            FieldElement::from(7u64),
                            FieldElement::from(shared_ff as u64),
                        ]);
                    }
                    Qwen35ProofObligation::SharedExpertGate => {
                        felts.push(FieldElement::from(8u64));
                    }
                    Qwen35ProofObligation::ResidualAdd => {
                        felts.extend([
                            FieldElement::from(9u64),
                            FieldElement::from(self.hidden_size as u64),
                        ]);
                    }
                }
            }
        }

        starknet_crypto::poseidon_hash_many(&felts)
    }

    pub fn execution_contract_summary(&self) -> Result<String, String> {
        let first_linear = self
            .layers
            .iter()
            .find(|layer| layer.attention == Qwen35AttentionKind::GatedDeltaNet)
            .ok_or_else(|| "missing GatedDeltaNet layer".to_string())?
            .layer_idx;
        let first_full = self
            .layers
            .iter()
            .find(|layer| layer.attention == Qwen35AttentionKind::GatedFullAttention)
            .ok_or_else(|| "missing gated full attention layer".to_string())?
            .layer_idx;

        let linear = self.gated_delta_net_contract(first_linear, 1)?;
        let full = self.gated_full_attention_contract(first_full, 1)?;
        let moe = self.moe_contract(first_linear, 1)?;

        Ok(format!(
            "contract_hash=0x{:x}; layer{} GatedDeltaNet qkv=[{},{}] conv=[{},1,{}] state={} out=[{},{}]; layer{} full_attn q_proj=[{},{}] kv=[{},{}] q_heads={}x{} kv_heads={}x{} gate={}; MoE router=[{},{}] packed_gate_up=[{},{},{}] packed_down=[{},{},{}] shared_ff={}",
            self.architecture_contract_hash(),
            first_linear,
            linear.in_proj_qkv_weight.rows,
            linear.in_proj_qkv_weight.cols,
            linear.conv1d_weight.outer,
            linear.conv1d_weight.inner,
            linear.recurrent_state_rows,
            linear.o_proj_weight.rows,
            linear.o_proj_weight.cols,
            first_full,
            full.q_proj_weight.rows,
            full.q_proj_weight.cols,
            full.k_proj_weight.rows,
            full.k_proj_weight.cols,
            full.query_heads.middle,
            full.query_heads.inner,
            full.key_heads.middle,
            full.key_heads.inner,
            full.output_gate.cols,
            moe.router_weight.rows,
            moe.router_weight.cols,
            moe.packed_gate_up_weight.outer,
            moe.packed_gate_up_weight.middle,
            moe.packed_gate_up_weight.inner,
            moe.packed_down_weight.outer,
            moe.packed_down_weight.middle,
            moe.packed_down_weight.inner,
            self.shared_ff,
        ))
    }

    pub fn gated_full_attention_contract(
        &self,
        layer_idx: usize,
        seq_len: usize,
    ) -> Result<Qwen35GatedFullAttentionContract, String> {
        if seq_len == 0 {
            return Err("seq_len must be non-zero".to_string());
        }
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| format!("layer_idx {layer_idx} out of range"))?;
        if layer.attention != Qwen35AttentionKind::GatedFullAttention {
            return Err(format!(
                "layer {layer_idx} is {:?}, not gated full attention",
                layer.attention
            ));
        }
        if self.num_key_value_heads == 0 || self.num_attention_heads % self.num_key_value_heads != 0
        {
            return Err(format!(
                "invalid GQA shape: {} query heads, {} KV heads",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }

        let q_rows_with_gate = 2 * self.q_dim;
        Ok(Qwen35GatedFullAttentionContract {
            layer_idx,
            seq_len,
            input: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.hidden_size,
            },
            q_proj_weight: Qwen35Tensor2DShape {
                rows: q_rows_with_gate,
                cols: self.hidden_size,
            },
            q_proj_output: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: q_rows_with_gate,
            },
            query: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.q_dim,
            },
            output_gate: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.q_dim,
            },
            query_heads: Qwen35Tensor3DShape {
                outer: seq_len,
                middle: self.num_attention_heads,
                inner: self.head_dim,
            },
            q_norm_weight: self.head_dim,
            k_proj_weight: Qwen35Tensor2DShape {
                rows: self.kv_dim,
                cols: self.hidden_size,
            },
            v_proj_weight: Qwen35Tensor2DShape {
                rows: self.kv_dim,
                cols: self.hidden_size,
            },
            key: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.kv_dim,
            },
            value: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.kv_dim,
            },
            key_heads: Qwen35Tensor3DShape {
                outer: seq_len,
                middle: self.num_key_value_heads,
                inner: self.head_dim,
            },
            value_heads: Qwen35Tensor3DShape {
                outer: seq_len,
                middle: self.num_key_value_heads,
                inner: self.head_dim,
            },
            k_norm_weight: self.head_dim,
            query_groups_per_kv_head: self.num_attention_heads / self.num_key_value_heads,
            attention_context: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.q_dim,
            },
            gated_context: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.q_dim,
            },
            o_proj_weight: Qwen35Tensor2DShape {
                rows: self.hidden_size,
                cols: self.q_dim,
            },
            output: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.hidden_size,
            },
        })
    }

    pub fn moe_contract(
        &self,
        layer_idx: usize,
        seq_len: usize,
    ) -> Result<Qwen35MoeContract, String> {
        if seq_len == 0 {
            return Err("seq_len must be non-zero".to_string());
        }
        if layer_idx >= self.num_layers {
            return Err(format!("layer_idx {layer_idx} out of range"));
        }
        if self.top_k == 0 || self.top_k > self.num_experts {
            return Err(format!(
                "invalid MoE routing shape: top_k={}, num_experts={}",
                self.top_k, self.num_experts
            ));
        }

        Ok(Qwen35MoeContract {
            layer_idx,
            seq_len,
            input: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.hidden_size,
            },
            router_weight: Qwen35Tensor2DShape {
                rows: self.num_experts,
                cols: self.hidden_size,
            },
            router_logits: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.num_experts,
            },
            selected_expert_ids: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.top_k,
            },
            routing_weights: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.top_k,
            },
            packed_gate_up_weight: Qwen35Tensor3DShape {
                outer: self.num_experts,
                middle: 2 * self.routed_ff,
                inner: self.hidden_size,
            },
            expert_gate: Qwen35Tensor3DShape {
                outer: seq_len,
                middle: self.top_k,
                inner: self.routed_ff,
            },
            expert_up: Qwen35Tensor3DShape {
                outer: seq_len,
                middle: self.top_k,
                inner: self.routed_ff,
            },
            expert_hidden: Qwen35Tensor3DShape {
                outer: seq_len,
                middle: self.top_k,
                inner: self.routed_ff,
            },
            packed_down_weight: Qwen35Tensor3DShape {
                outer: self.num_experts,
                middle: self.hidden_size,
                inner: self.routed_ff,
            },
            expert_output: Qwen35Tensor3DShape {
                outer: seq_len,
                middle: self.top_k,
                inner: self.hidden_size,
            },
            routed_output: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.hidden_size,
            },
            shared_gate_weight: Qwen35Tensor2DShape {
                rows: self.shared_ff,
                cols: self.hidden_size,
            },
            shared_up_weight: Qwen35Tensor2DShape {
                rows: self.shared_ff,
                cols: self.hidden_size,
            },
            shared_down_weight: Qwen35Tensor2DShape {
                rows: self.hidden_size,
                cols: self.shared_ff,
            },
            shared_expert_gate_weight: Qwen35Tensor2DShape {
                rows: 1,
                cols: self.hidden_size,
            },
            shared_hidden: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.shared_ff,
            },
            shared_output: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.hidden_size,
            },
            output: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.hidden_size,
            },
        })
    }

    pub fn gated_delta_net_contract(
        &self,
        layer_idx: usize,
        seq_len: usize,
    ) -> Result<Qwen35GatedDeltaNetContract, String> {
        if seq_len == 0 {
            return Err("seq_len must be non-zero".to_string());
        }
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| format!("layer_idx {layer_idx} out of range"))?;
        if layer.attention != Qwen35AttentionKind::GatedDeltaNet {
            return Err(format!(
                "layer {layer_idx} is {:?}, not GatedDeltaNet",
                layer.attention
            ));
        }
        if self.linear_state_rows == 0 || self.linear_value_rows % self.linear_state_rows != 0 {
            return Err(format!(
                "invalid linear value shape: rows={}, state_rows={}",
                self.linear_value_rows, self.linear_state_rows
            ));
        }
        if self.linear_num_key_heads == 0
            || self.linear_num_value_heads == 0
            || self.linear_num_value_heads % self.linear_num_key_heads != 0
        {
            return Err(format!(
                "invalid linear q/k head repeat: key_heads={}, value_heads={}",
                self.linear_num_key_heads, self.linear_num_value_heads
            ));
        }
        if self.linear_qkv_rows != 2 * self.linear_key_rows + self.linear_value_rows {
            return Err(format!(
                "invalid qkv rows: qkv_rows={}, expected 2*key_rows+value_rows={}",
                self.linear_qkv_rows,
                2 * self.linear_key_rows + self.linear_value_rows
            ));
        }
        let repeated_qk_rows = self.linear_num_value_heads * self.linear_key_head_dim;
        let projected_value_rows = self
            .linear_qkv_rows
            .checked_sub(2 * self.linear_key_rows)
            .ok_or_else(|| {
                format!(
                    "invalid qkv split: qkv_rows={}, key_rows={}",
                    self.linear_qkv_rows, self.linear_key_rows
                )
            })?;

        Ok(Qwen35GatedDeltaNetContract {
            layer_idx,
            seq_len,
            input: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.hidden_size,
            },
            in_proj_qkv_weight: Qwen35Tensor2DShape {
                rows: self.linear_qkv_rows,
                cols: self.hidden_size,
            },
            qkv_projected: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.linear_qkv_rows,
            },
            conv1d_weight: Qwen35Tensor3DShape {
                outer: self.linear_qkv_rows,
                middle: 1,
                inner: self.linear_conv_kernel_dim,
            },
            qkv_after_conv: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.linear_qkv_rows,
            },
            query: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: repeated_qk_rows,
            },
            key: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: repeated_qk_rows,
            },
            projected_value: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: projected_value_rows,
            },
            in_proj_z_weight: Qwen35Tensor2DShape {
                rows: self.linear_value_rows,
                cols: self.hidden_size,
            },
            z_gate: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.linear_value_rows,
            },
            in_proj_a_weight: Qwen35Tensor2DShape {
                rows: self.linear_state_rows,
                cols: self.hidden_size,
            },
            in_proj_b_weight: Qwen35Tensor2DShape {
                rows: self.linear_state_rows,
                cols: self.hidden_size,
            },
            a_gate: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.linear_state_rows,
            },
            b_gate: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.linear_state_rows,
            },
            a_log_weight: self.linear_state_rows,
            dt_bias: self.linear_state_rows,
            norm_weight: self.linear_value_rows / self.linear_state_rows,
            linear_value_heads: self.linear_state_rows,
            linear_value_head_dim: self.linear_value_rows / self.linear_state_rows,
            recurrent_state_rows: self.linear_state_rows,
            attended_value: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.linear_value_rows,
            },
            o_proj_weight: Qwen35Tensor2DShape {
                rows: self.hidden_size,
                cols: self.linear_value_rows,
            },
            output: Qwen35Tensor2DShape {
                rows: seq_len,
                cols: self.hidden_size,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen35_test_config() -> HfConfig {
        let mut layer_types = Vec::new();
        for idx in 0..40 {
            if idx % 4 == 3 {
                layer_types.push("full_attention".to_string());
            } else {
                layer_types.push("linear_attention".to_string());
            }
        }

        HfConfig {
            model_type: "qwen3_5_moe".to_string(),
            hidden_size: 2048,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            intermediate_size: 512,
            num_hidden_layers: 40,
            vocab_size: 248320,
            hidden_act: "silu".to_string(),
            max_position_embeddings: 262144,
            head_dim: 256,
            num_experts: 256,
            num_experts_per_tok: 8,
            layer_types,
            moe_intermediate_size: Some(512),
            shared_expert_intermediate_size: Some(512),
            linear_key_head_dim: Some(128),
            linear_value_head_dim: Some(128),
            linear_num_key_heads: Some(16),
            linear_num_value_heads: Some(32),
            linear_conv_kernel_dim: Some(4),
            attn_output_gate: true,
        }
    }

    fn qwen35_small_linear_test_config() -> HfConfig {
        HfConfig {
            model_type: "qwen3_5_moe".to_string(),
            hidden_size: 4,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: 8,
            num_hidden_layers: 1,
            vocab_size: 32,
            hidden_act: "silu".to_string(),
            max_position_embeddings: 16,
            head_dim: 4,
            num_experts: 2,
            num_experts_per_tok: 1,
            layer_types: vec!["linear_attention".to_string()],
            moe_intermediate_size: Some(8),
            shared_expert_intermediate_size: Some(8),
            linear_key_head_dim: Some(2),
            linear_value_head_dim: Some(3),
            linear_num_key_heads: Some(1),
            linear_num_value_heads: Some(2),
            linear_conv_kernel_dim: Some(3),
            attn_output_gate: true,
        }
    }

    fn m31_vector(len: usize, seed: u32) -> Vec<M31> {
        (0..len)
            .map(|idx| M31::from(seed + idx as u32 + 1))
            .collect()
    }

    fn m31_matrix(shape: Qwen35Tensor2DShape, seed: u32) -> M31Matrix {
        let mut matrix = M31Matrix::new(shape.rows, shape.cols);
        for idx in 0..matrix.data.len() {
            matrix.data[idx] = M31::from(seed + idx as u32 + 1);
        }
        matrix
    }

    fn m31_zero_vector(len: usize) -> Vec<M31> {
        vec![M31::from(0u32); len]
    }

    fn m31_zero_matrix(shape: Qwen35Tensor2DShape) -> M31Matrix {
        M31Matrix::new(shape.rows, shape.cols)
    }

    fn qwen35_delta_norm_output(
        input: &M31Matrix,
        state_rows: usize,
        table_log_size: u32,
        post_scale: M31,
    ) -> M31Matrix {
        let qk_head_dim = input.cols / state_rows;
        let table = crate::components::rmsnorm::build_rsqrt_table(table_log_size);
        let mut output = M31Matrix::new(input.rows, input.cols);
        for token_idx in 0..input.rows {
            for state_row_idx in 0..state_rows {
                let mut sum_sq = M31::from(0u32);
                for qk_idx in 0..qk_head_dim {
                    let col_idx = state_row_idx * qk_head_dim + qk_idx;
                    let value = input.get(token_idx, col_idx);
                    sum_sq += value * value;
                }
                let rsqrt = table.lookup(sum_sq).unwrap();
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

    fn qwen35_delta_decay_output(
        a_gate: &M31Matrix,
        a_log_weight: &[M31],
        dt_bias: &[M31],
        table_log_size: u32,
    ) -> M31Matrix {
        let softplus_table = crate::gadgets::lookup_table::PrecomputedTable::build(
            crate::gadgets::lookup_table::activations::softplus_approx,
            table_log_size,
        );
        let exp_table = crate::gadgets::lookup_table::PrecomputedTable::build(
            crate::gadgets::lookup_table::activations::softmax_exp,
            table_log_size,
        );
        let decay_table = crate::gadgets::lookup_table::PrecomputedTable::build(
            |x| crate::gadgets::lookup_table::activations::softmax_exp(M31::from(0u32) - x),
            table_log_size,
        );
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

    fn qwen35_delta_beta_output(b_gate: &M31Matrix) -> M31Matrix {
        let mut beta = M31Matrix::new(b_gate.rows, b_gate.cols);
        for row in 0..b_gate.rows {
            for col in 0..b_gate.cols {
                beta.set(
                    row,
                    col,
                    crate::gadgets::lookup_table::activations::sigmoid_approx(b_gate.get(row, col)),
                );
            }
        }
        beta
    }

    fn qwen35_runtime_trace_for_contract(
        contract: &Qwen35GatedDeltaNetContract,
        seed: u32,
    ) -> Qwen35GatedDeltaNetRuntimeLayerTrace {
        let qk_head_dim = contract.query.cols / contract.recurrent_state_rows;
        let state_rows = contract.recurrent_state_rows * qk_head_dim;
        let state_shape = Qwen35Tensor2DShape {
            rows: state_rows,
            cols: contract.linear_value_head_dim,
        };
        Qwen35GatedDeltaNetRuntimeLayerTrace {
            layer_idx: contract.layer_idx,
            qkv_projected: m31_matrix(contract.qkv_projected, seed + 10),
            qkv_after_conv: m31_matrix(contract.qkv_after_conv, seed + 20),
            query: m31_matrix(contract.query, seed + 30),
            key: m31_matrix(contract.key, seed + 40),
            projected_value: m31_matrix(contract.projected_value, seed + 50),
            a_gate: m31_matrix(contract.a_gate, seed + 60),
            b_gate: m31_matrix(contract.b_gate, seed + 70),
            attended_value: M31Matrix::new(
                contract.attended_value.rows,
                contract.attended_value.cols,
            ),
            z_gate: m31_matrix(contract.z_gate, seed + 90),
            gated_value: M31Matrix::new(contract.attended_value.rows, contract.attended_value.cols),
            initial_recurrent_state: m31_matrix(state_shape, seed + 110),
            final_recurrent_state: m31_matrix(state_shape, seed + 120),
            delta_recurrence_transform: None,
            norm_and_z_gate_rsqrt_table_log_size: 16,
        }
    }

    fn qwen35_active_runtime_trace_for_contract(
        contract: &Qwen35GatedDeltaNetContract,
        a_log_weight: &[M31],
        dt_bias: &[M31],
    ) -> Qwen35GatedDeltaNetRuntimeLayerTrace {
        use crate::components::qwen35_delta_recurrence::{
            qwen35_delta_recurrence_arithmetic_witness, Qwen35DeltaRecurrenceArithmeticInputs,
        };

        let qk_head_dim = contract.query.cols / contract.recurrent_state_rows;
        let state_shape = Qwen35Tensor2DShape {
            rows: contract.recurrent_state_rows * qk_head_dim,
            cols: contract.linear_value_head_dim,
        };
        let query = m31_zero_matrix(contract.query);
        let key = m31_zero_matrix(contract.key);
        let projected_value = m31_zero_matrix(contract.projected_value);
        let a_gate = m31_zero_matrix(contract.a_gate);
        let b_gate = m31_zero_matrix(contract.b_gate);
        let initial_recurrent_state = m31_zero_matrix(state_shape);
        let placeholder_final_state = m31_zero_matrix(state_shape);
        let placeholder_output = m31_zero_matrix(contract.attended_value);
        let scaled_query =
            qwen35_delta_norm_output(&query, contract.recurrent_state_rows, 16, M31::from(1u32));
        let normalized_key =
            qwen35_delta_norm_output(&key, contract.recurrent_state_rows, 16, M31::from(1u32));
        let decay = qwen35_delta_decay_output(&a_gate, a_log_weight, dt_bias, 16);
        let beta = qwen35_delta_beta_output(&b_gate);
        let arithmetic_inputs = Qwen35DeltaRecurrenceArithmeticInputs {
            scaled_query: &scaled_query,
            normalized_key: &normalized_key,
            projected_value: &projected_value,
            decay: &decay,
            beta: &beta,
            initial_recurrent_state: &initial_recurrent_state,
            final_recurrent_state: &placeholder_final_state,
            output: &placeholder_output,
            state_rows: contract.recurrent_state_rows,
            value_head_dim: contract.linear_value_head_dim,
        };
        let arithmetic_witness = qwen35_delta_recurrence_arithmetic_witness(&arithmetic_inputs)
            .expect("active test trace must produce a DeltaRecurrence arithmetic witness");

        Qwen35GatedDeltaNetRuntimeLayerTrace {
            layer_idx: contract.layer_idx,
            qkv_projected: m31_zero_matrix(contract.qkv_projected),
            qkv_after_conv: m31_zero_matrix(contract.qkv_after_conv),
            query,
            key,
            projected_value,
            a_gate,
            b_gate,
            attended_value: arithmetic_witness.output,
            z_gate: m31_zero_matrix(contract.z_gate),
            gated_value: m31_zero_matrix(contract.attended_value),
            initial_recurrent_state,
            final_recurrent_state: arithmetic_witness.final_recurrent_state,
            delta_recurrence_transform: Some(Qwen35DeltaRecurrenceRuntimeTransformTrace {
                scaled_query,
                normalized_key,
                decay,
                beta,
                q_norm_table_log_size: 16,
                k_norm_table_log_size: 16,
                beta_sigmoid_table_log_size: 16,
                decay_table_log_size: 16,
                query_norm_post_scale: M31::from(1u32),
                key_norm_post_scale: M31::from(1u32),
            }),
            norm_and_z_gate_rsqrt_table_log_size: 16,
        }
    }

    #[test]
    fn qwen35_plan_matches_downloaded_model_contract() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();

        assert_eq!(plan.num_layers, 40);
        assert_eq!(plan.linear_attention_layers(), 30);
        assert_eq!(plan.full_attention_layers(), 10);
        assert_eq!(plan.q_dim, 4096);
        assert_eq!(plan.kv_dim, 512);
        assert_eq!(plan.linear_key_head_dim, 128);
        assert_eq!(plan.linear_value_head_dim, 128);
        assert_eq!(plan.linear_num_key_heads, 16);
        assert_eq!(plan.linear_num_value_heads, 32);
        assert_eq!(plan.linear_key_rows, 2048);
        assert_eq!(plan.linear_value_rows, 4096);
        assert_eq!(plan.linear_qkv_rows, 8192);
        assert_eq!(plan.linear_state_rows, 32);
        assert_eq!(plan.expected_language_tensor_count(), 693);

        let full = &plan.layers[3];
        assert_eq!(full.attention, Qwen35AttentionKind::GatedFullAttention);
        assert!(full
            .obligations
            .contains(&Qwen35ProofObligation::GatedFullAttention {
                q_rows_with_gate: 8192,
                q_rows: 4096,
                kv_rows: 512,
            },));

        let linear = &plan.layers[0];
        assert_eq!(linear.attention, Qwen35AttentionKind::GatedDeltaNet);
        assert!(linear
            .obligations
            .contains(&Qwen35ProofObligation::GatedDeltaNet {
                qkv_rows: 8192,
                z_rows: 4096,
                state_rows: 32,
                conv_kernel: 4,
            }));
    }

    #[test]
    fn qwen35_execution_contract_hash_is_deterministic_and_shape_sensitive() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let same = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        assert_eq!(
            plan.architecture_contract_hash(),
            same.architecture_contract_hash()
        );

        let mut changed_cfg = cfg;
        changed_cfg.layer_types.swap(0, 3);
        let changed = Qwen35ProofPlan::from_hf_config(&changed_cfg).unwrap();
        assert_ne!(
            plan.architecture_contract_hash(),
            changed.architecture_contract_hash()
        );

        let mut same_rows_different_head_layout = qwen35_test_config();
        same_rows_different_head_layout.linear_num_key_heads = Some(32);
        same_rows_different_head_layout.linear_key_head_dim = Some(64);
        let changed_layout =
            Qwen35ProofPlan::from_hf_config(&same_rows_different_head_layout).unwrap();
        assert_eq!(plan.linear_key_rows, changed_layout.linear_key_rows);
        assert_eq!(plan.linear_qkv_rows, changed_layout.linear_qkv_rows);
        assert_ne!(
            plan.architecture_contract_hash(),
            changed_layout.architecture_contract_hash()
        );

        let summary = plan.execution_contract_summary().unwrap();
        assert!(summary.contains("contract_hash=0x"));
        assert!(summary.contains("GatedDeltaNet"));
        assert!(summary.contains("full_attn"));
        assert!(summary.contains("packed_gate_up=[256,1024,2048]"));
    }

    #[test]
    fn qwen35_tensor_contract_covers_all_named_model_weights() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let tensors = plan.expected_tensor_contract();

        assert_eq!(tensors.len(), plan.expected_language_tensor_count());
        assert!(tensors.iter().any(|entry| {
            entry.name == "model.language_model.embed_tokens.weight"
                && entry.shape == vec![248320, 2048]
                && entry.role == Qwen35TensorRole::TokenEmbedding
                && entry.layer_idx.is_none()
        }));
        assert!(tensors.iter().any(|entry| {
            entry.name == "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
                && entry.shape == vec![8192, 2048]
                && entry.role == Qwen35TensorRole::LinearAttentionInProjQkv
                && entry.layer_idx == Some(0)
        }));
        assert!(tensors.iter().any(|entry| {
            entry.name == "model.language_model.layers.3.self_attn.q_proj.weight"
                && entry.shape == vec![8192, 2048]
                && entry.role == Qwen35TensorRole::FullAttentionQProj
                && entry.layer_idx == Some(3)
        }));
        assert!(tensors.iter().any(|entry| {
            entry.name == "model.language_model.layers.39.mlp.experts.gate_up_proj"
                && entry.shape == vec![256, 1024, 2048]
                && entry.role == Qwen35TensorRole::MoePackedGateUp
                && entry.layer_idx == Some(39)
        }));
    }

    #[test]
    fn typed_witness_manifest_names_real_qwen35_gated_delta_roots() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let manifest = plan.typed_witness_manifest(13).unwrap();

        assert_eq!(manifest.seq_len, 13);
        assert_eq!(manifest.layers.len(), 30);
        assert_eq!(manifest.total_roots(), 30 * 18);
        assert_eq!(
            manifest.root_count_by_kind(Qwen35TypedWitnessRootKind::Activation),
            30 * 11
        );
        assert_eq!(
            manifest.root_count_by_kind(Qwen35TypedWitnessRootKind::ModelWeight),
            30 * 4
        );
        assert_eq!(
            manifest.root_count_by_kind(Qwen35TypedWitnessRootKind::RecurrentState),
            30 * 2
        );
        assert_eq!(
            manifest.root_count_by_kind(Qwen35TypedWitnessRootKind::LookupTable),
            30
        );

        let layer0 = &manifest.layers[0];
        assert_eq!(layer0.layer_idx, 0);
        assert_eq!(layer0.seq_len, 13);
        assert_ne!(
            layer0.depthwise_conv1d_trace_binding_hash,
            FieldElement::ZERO
        );
        assert_ne!(
            layer0.delta_recurrence_trace_binding_hash,
            FieldElement::ZERO
        );
        assert_ne!(
            layer0.norm_and_z_gate_trace_binding_hash,
            FieldElement::ZERO
        );
        assert!(layer0.roots.iter().any(|root| {
            root.statement_kind == Qwen35TypedProofStatementKind::DepthwiseConv1d
                && root.name == "conv1d_weight"
                && root.kind == Qwen35TypedWitnessRootKind::ModelWeight
                && root.shape
                    == Qwen35TensorShape::Tensor3D(Qwen35Tensor3DShape {
                        outer: 8192,
                        middle: 1,
                        inner: 4,
                    })
                && root.source
                    == "safetensors:model.language_model.layers.0.linear_attn.conv1d.weight"
        }));
        assert!(layer0.roots.iter().any(|root| {
            root.statement_kind == Qwen35TypedProofStatementKind::DeltaRecurrence
                && root.name == "initial_recurrent_state"
                && root.kind == Qwen35TypedWitnessRootKind::RecurrentState
                && root.shape
                    == Qwen35TensorShape::Matrix(Qwen35Tensor2DShape {
                        rows: 4096,
                        cols: 128,
                    })
        }));
        assert!(layer0.roots.iter().any(|root| {
            root.statement_kind == Qwen35TypedProofStatementKind::NormAndZGate
                && root.name == "rsqrt_table_commitment"
                && root.kind == Qwen35TypedWitnessRootKind::LookupTable
        }));
    }

    #[test]
    fn typed_witness_manifest_hash_is_sequence_and_shape_bound() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let seq13 = plan.typed_witness_manifest(13).unwrap();
        let seq14 = plan.typed_witness_manifest(14).unwrap();
        assert_ne!(seq13.manifest_hash, seq14.manifest_hash);
        assert_ne!(seq13.layers[0].layer_hash(), seq14.layers[0].layer_hash());

        let mut changed_cfg = qwen35_test_config();
        changed_cfg.linear_conv_kernel_dim = Some(8);
        let changed_plan = Qwen35ProofPlan::from_hf_config(&changed_cfg).unwrap();
        let changed = changed_plan.typed_witness_manifest(13).unwrap();
        assert_ne!(seq13.manifest_hash, changed.manifest_hash);

        let mut seen = HashSet::new();
        for root in &seq13.layers[0].roots {
            assert!(
                seen.insert(root.root_hash()),
                "duplicate root {}",
                root.name
            );
        }
    }

    #[test]
    fn typed_witness_commitment_set_requires_every_manifest_root() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let manifest = plan.typed_witness_manifest(2).unwrap();
        let commitments = manifest
            .required_root_hashes()
            .into_iter()
            .enumerate()
            .map(|(idx, root_hash)| (root_hash, FieldElement::from((idx + 1) as u64)))
            .collect::<Vec<_>>();

        let set = manifest.build_commitment_set(&commitments).unwrap();
        assert_eq!(set.manifest_hash, manifest.manifest_hash);
        assert_eq!(set.seq_len, 2);
        assert_eq!(set.commitments.len(), manifest.total_roots());
        assert_ne!(set.commitment_set_hash, FieldElement::ZERO);
        manifest.validate_commitment_set(&set).unwrap();

        let missing = &commitments[..commitments.len() - 1];
        let err = manifest.build_commitment_set(missing).unwrap_err();
        assert!(err.contains("missing typed witness commitment"));

        let mut duplicate = commitments.clone();
        duplicate.push(commitments[0]);
        let err = manifest.build_commitment_set(&duplicate).unwrap_err();
        assert!(err.contains("duplicate typed witness commitment"));

        let mut zero = commitments.clone();
        zero[0].1 = FieldElement::ZERO;
        let err = manifest.build_commitment_set(&zero).unwrap_err();
        assert!(err.contains("zero commitment"));

        let mut tampered = set.clone();
        tampered.commitments[0].commitment += FieldElement::ONE;
        assert!(manifest.validate_commitment_set(&tampered).is_err());
    }

    fn captured_value_for_root(
        root: &Qwen35TypedWitnessRoot,
        seed: u32,
    ) -> Qwen35TypedWitnessCapturedValue {
        match root.shape {
            Qwen35TensorShape::Vector(0)
                if root.kind == Qwen35TypedWitnessRootKind::LookupTable =>
            {
                Qwen35TypedWitnessCapturedValue::LookupTable { table_log_size: 4 }
            }
            Qwen35TensorShape::Vector(len) => Qwen35TypedWitnessCapturedValue::Vector(
                (0..len)
                    .map(|idx| M31::from(seed + idx as u32 + 1))
                    .collect(),
            ),
            Qwen35TensorShape::Matrix(shape) => {
                let mut matrix = M31Matrix::new(shape.rows, shape.cols);
                for idx in 0..matrix.data.len() {
                    matrix.data[idx] = M31::from(seed + idx as u32 + 1);
                }
                Qwen35TypedWitnessCapturedValue::Matrix(matrix)
            }
            Qwen35TensorShape::Tensor3D(shape) => {
                let len = shape.outer * shape.middle * shape.inner;
                Qwen35TypedWitnessCapturedValue::Vector(
                    (0..len)
                        .map(|idx| M31::from(seed + idx as u32 + 1))
                        .collect(),
                )
            }
        }
    }

    #[test]
    fn typed_witness_capture_builds_component_domain_commitment_set() {
        let root = |statement_kind: Qwen35TypedProofStatementKind,
                    stage_idx: usize,
                    name: &str,
                    kind: Qwen35TypedWitnessRootKind,
                    shape: Qwen35TensorShape| {
            Qwen35TypedWitnessRoot {
                layer_idx: 0,
                statement_kind,
                stage_idx,
                name: name.to_string(),
                kind,
                shape,
                source: format!("test:{stage_idx}:{name}:{}", statement_kind.label()),
                trace_root_contract_hash: FieldElement::from(10_000u64 + stage_idx as u64),
            }
        };
        let depthwise = Qwen35TypedProofStatementKind::DepthwiseConv1d;
        let delta = Qwen35TypedProofStatementKind::DeltaRecurrence;
        let norm = Qwen35TypedProofStatementKind::NormAndZGate;
        let activation = Qwen35TypedWitnessRootKind::Activation;
        let model_weight = Qwen35TypedWitnessRootKind::ModelWeight;
        let recurrent_state = Qwen35TypedWitnessRootKind::RecurrentState;
        let lookup_table = Qwen35TypedWitnessRootKind::LookupTable;
        let m24 = Qwen35TensorShape::Matrix(Qwen35Tensor2DShape { rows: 2, cols: 4 });
        let m26 = Qwen35TensorShape::Matrix(Qwen35Tensor2DShape { rows: 2, cols: 6 });
        let m22 = Qwen35TensorShape::Matrix(Qwen35Tensor2DShape { rows: 2, cols: 2 });
        let state = Qwen35TensorShape::Matrix(Qwen35Tensor2DShape { rows: 4, cols: 3 });
        let layer = Qwen35TypedWitnessLayerManifest {
            layer_idx: 0,
            seq_len: 2,
            depthwise_conv1d_trace_binding_hash: FieldElement::from(101u64),
            delta_recurrence_trace_binding_hash: FieldElement::from(102u64),
            norm_and_z_gate_trace_binding_hash: FieldElement::from(103u64),
            roots: vec![
                root(depthwise, 0, "qkv_projected", activation, m24),
                root(
                    depthwise,
                    1,
                    "conv1d_weight",
                    model_weight,
                    Qwen35TensorShape::Tensor3D(Qwen35Tensor3DShape {
                        outer: 4,
                        middle: 1,
                        inner: 3,
                    }),
                ),
                root(depthwise, 2, "qkv_after_conv", activation, m24),
                root(delta, 3, "query", activation, m24),
                root(delta, 3, "key", activation, m24),
                root(delta, 3, "projected_value", activation, m26),
                root(delta, 4, "a_gate", activation, m22),
                root(delta, 4, "b_gate", activation, m22),
                root(
                    delta,
                    5,
                    "a_log_weight",
                    model_weight,
                    Qwen35TensorShape::Vector(2),
                ),
                root(
                    delta,
                    5,
                    "dt_bias",
                    model_weight,
                    Qwen35TensorShape::Vector(2),
                ),
                root(delta, 5, "attended_value", activation, m26),
                root(delta, 5, "initial_recurrent_state", recurrent_state, state),
                root(delta, 5, "final_recurrent_state", recurrent_state, state),
                root(norm, 5, "attended_value", activation, m26),
                root(
                    norm,
                    6,
                    "norm_weight",
                    model_weight,
                    Qwen35TensorShape::Vector(3),
                ),
                root(norm, 4, "z_gate", activation, m26),
                root(norm, 7, "gated_value", activation, m26),
                root(
                    norm,
                    6,
                    "rsqrt_table_commitment",
                    lookup_table,
                    Qwen35TensorShape::Vector(0),
                ),
            ],
        };
        let architecture_contract_hash = FieldElement::from(55u64);
        let manifest_hash = Qwen35TypedWitnessManifest::compute_hash(
            architecture_contract_hash,
            2,
            &[layer.clone()],
        );
        let manifest = Qwen35TypedWitnessManifest {
            architecture_contract_hash,
            seq_len: 2,
            layers: vec![layer],
            manifest_hash,
        };

        let mut capture = Qwen35TypedWitnessCapture::new();
        for (idx, root) in manifest.layers[0].roots.iter().enumerate() {
            let root_hash = root.root_hash();
            match captured_value_for_root(root, idx as u32 + 11) {
                Qwen35TypedWitnessCapturedValue::Matrix(matrix) => {
                    capture.insert_matrix(root_hash, matrix)
                }
                Qwen35TypedWitnessCapturedValue::Vector(values) => {
                    capture.insert_vector(root_hash, values)
                }
                Qwen35TypedWitnessCapturedValue::LookupTable { table_log_size } => {
                    capture.insert_lookup_table(root_hash, table_log_size)
                }
            }
        }

        let commitment_set = manifest.validate_capture(&capture).unwrap();
        assert_eq!(commitment_set.commitments.len(), manifest.total_roots());
        assert_ne!(commitment_set.commitment_set_hash, FieldElement::ZERO);

        let mut missing = capture.clone();
        missing.roots.pop();
        let err = manifest.validate_capture(&missing).unwrap_err();
        assert!(err.contains("missing captured value"));

        let mut duplicate = capture.clone();
        duplicate.roots.push(duplicate.roots[0].clone());
        let err = manifest.validate_capture(&duplicate).unwrap_err();
        assert!(err.contains("duplicate captured typed witness root"));

        let mut wrong_shape = capture.clone();
        let matrix_root_idx = wrong_shape
            .roots
            .iter()
            .position(|entry| matches!(entry.value, Qwen35TypedWitnessCapturedValue::Matrix(_)))
            .unwrap();
        wrong_shape.roots[matrix_root_idx].value =
            Qwen35TypedWitnessCapturedValue::Matrix(M31Matrix::new(1, 1));
        let err = manifest.validate_capture(&wrong_shape).unwrap_err();
        assert!(err.contains("matrix shape"));

        let mut extra = capture;
        extra.insert_lookup_table(FieldElement::from(999_999u64), 4);
        let err = manifest.validate_capture(&extra).unwrap_err();
        assert!(err.contains("not required by manifest"));
    }

    #[test]
    fn typed_witness_source_capture_fans_out_shared_runtime_sources() {
        let root = |statement_kind: Qwen35TypedProofStatementKind,
                    stage_idx: usize,
                    name: &str,
                    kind: Qwen35TypedWitnessRootKind,
                    shape: Qwen35TensorShape,
                    source: &str| {
            Qwen35TypedWitnessRoot {
                layer_idx: 0,
                statement_kind,
                stage_idx,
                name: name.to_string(),
                kind,
                shape,
                source: source.to_string(),
                trace_root_contract_hash: starknet_crypto::poseidon_hash_many(&[
                    FieldElement::from(42u64),
                    FieldElement::from(stage_idx as u64),
                    qwen35_hash_str(name),
                    qwen35_hash_str(source),
                ]),
            }
        };

        let delta = Qwen35TypedProofStatementKind::DeltaRecurrence;
        let norm = Qwen35TypedProofStatementKind::NormAndZGate;
        let activation = Qwen35TypedWitnessRootKind::Activation;
        let model_weight = Qwen35TypedWitnessRootKind::ModelWeight;
        let recurrent_state = Qwen35TypedWitnessRootKind::RecurrentState;
        let lookup_table = Qwen35TypedWitnessRootKind::LookupTable;
        let attended = Qwen35TensorShape::Matrix(Qwen35Tensor2DShape { rows: 2, cols: 6 });
        let state = Qwen35TensorShape::Matrix(Qwen35Tensor2DShape { rows: 4, cols: 3 });
        let shared_attended_source =
            "runtime:model.language_model.layers.0.linear_attn.attended_value";
        let layer = Qwen35TypedWitnessLayerManifest {
            layer_idx: 0,
            seq_len: 2,
            depthwise_conv1d_trace_binding_hash: FieldElement::from(101u64),
            delta_recurrence_trace_binding_hash: FieldElement::from(102u64),
            norm_and_z_gate_trace_binding_hash: FieldElement::from(103u64),
            roots: vec![
                root(
                    delta,
                    5,
                    "attended_value",
                    activation,
                    attended,
                    shared_attended_source,
                ),
                root(
                    norm,
                    5,
                    "attended_value",
                    activation,
                    attended,
                    shared_attended_source,
                ),
                root(
                    delta,
                    5,
                    "a_log_weight",
                    model_weight,
                    Qwen35TensorShape::Vector(2),
                    "safetensors:model.language_model.layers.0.linear_attn.A_log",
                ),
                root(
                    delta,
                    5,
                    "initial_recurrent_state",
                    recurrent_state,
                    state,
                    "conversation-state:model.language_model.layers.0.linear_attn:initial_recurrent_state",
                ),
                root(
                    norm,
                    6,
                    "rsqrt_table_commitment",
                    lookup_table,
                    Qwen35TensorShape::Vector(0),
                    "statement:model.language_model.layers.0.linear_attn:norm_and_z_gate_rsqrt_table",
                ),
            ],
        };
        let architecture_contract_hash = FieldElement::from(55u64);
        let manifest_hash = Qwen35TypedWitnessManifest::compute_hash(
            architecture_contract_hash,
            2,
            &[layer.clone()],
        );
        let manifest = Qwen35TypedWitnessManifest {
            architecture_contract_hash,
            seq_len: 2,
            layers: vec![layer],
            manifest_hash,
        };

        let requirements = manifest.source_requirements().unwrap();
        assert_eq!(requirements.len(), 4);
        let shared_requirement = requirements
            .iter()
            .find(|req| req.source == shared_attended_source)
            .unwrap();
        assert_eq!(
            shared_requirement.source_kind,
            Qwen35TypedWitnessSourceKind::Runtime
        );
        assert_eq!(shared_requirement.root_hashes.len(), 2);

        let mut attended_matrix = M31Matrix::new(2, 6);
        for (idx, value) in attended_matrix.data.iter_mut().enumerate() {
            *value = M31::from(idx as u32 + 7);
        }
        let mut state_matrix = M31Matrix::new(4, 3);
        for (idx, value) in state_matrix.data.iter_mut().enumerate() {
            *value = M31::from(idx as u32 + 19);
        }

        let mut source_capture = Qwen35TypedWitnessSourceCapture::new();
        source_capture.insert_matrix(shared_attended_source, attended_matrix.clone());
        source_capture.insert_vector(
            "safetensors:model.language_model.layers.0.linear_attn.A_log",
            vec![M31::from(3), M31::from(4)],
        );
        source_capture.insert_matrix(
            "conversation-state:model.language_model.layers.0.linear_attn:initial_recurrent_state",
            state_matrix.clone(),
        );
        source_capture.insert_lookup_table(
            "statement:model.language_model.layers.0.linear_attn:norm_and_z_gate_rsqrt_table",
            4,
        );

        let root_capture = manifest
            .build_capture_from_source_capture(&source_capture)
            .unwrap();
        assert_eq!(root_capture.roots.len(), manifest.total_roots());
        let commitment_set = manifest.validate_source_capture(&source_capture).unwrap();
        assert_eq!(commitment_set.commitments.len(), manifest.total_roots());

        let mut inventory = Qwen35TypedWitnessSourceInventory::new();
        inventory.insert_runtime_matrix(
            "model.language_model.layers.0.linear_attn.attended_value",
            attended_matrix.clone(),
        );
        inventory.insert_safetensors_vector(
            "model.language_model.layers.0.linear_attn.A_log",
            vec![M31::from(3), M31::from(4)],
        );
        inventory.insert_conversation_state_matrix(
            "model.language_model.layers.0.linear_attn:initial_recurrent_state",
            state_matrix.clone(),
        );
        inventory.insert_statement_lookup_table(
            "model.language_model.layers.0.linear_attn:norm_and_z_gate_rsqrt_table",
            4,
        );
        let inventory_capture = manifest
            .build_source_capture_from_inventory(&inventory)
            .unwrap();
        assert_eq!(inventory_capture.sources.len(), requirements.len());
        let inventory_commitment_set = manifest.validate_source_inventory(&inventory).unwrap();
        assert_eq!(
            inventory_commitment_set.commitment_set_hash,
            commitment_set.commitment_set_hash
        );

        let mut recorder = Qwen35TypedWitnessInventoryRecorder::new(manifest.clone());
        recorder
            .record_attended_value(0, attended_matrix.clone())
            .unwrap();
        recorder
            .record_a_log_weight(0, vec![M31::from(3), M31::from(4)])
            .unwrap();
        recorder
            .record_initial_recurrent_state(0, state_matrix.clone())
            .unwrap();
        recorder.record_norm_and_z_gate_rsqrt_table(0, 4).unwrap();
        let recorder_commitment_set = recorder.validate().unwrap();
        assert_eq!(
            recorder_commitment_set.commitment_set_hash,
            commitment_set.commitment_set_hash
        );
        let duplicate_err = recorder
            .record_attended_value(0, attended_matrix.clone())
            .unwrap_err();
        assert!(duplicate_err.contains("duplicate runtime typed witness recorder value"));

        let mut wrong_layer_recorder = Qwen35TypedWitnessInventoryRecorder::new(manifest.clone());
        let wrong_layer_err = wrong_layer_recorder
            .record_attended_value(1, attended_matrix.clone())
            .unwrap_err();
        assert!(wrong_layer_err.contains("is not required by manifest"));

        let mut wrong_shape_recorder = Qwen35TypedWitnessInventoryRecorder::new(manifest.clone());
        let wrong_shape_err = wrong_shape_recorder
            .record_attended_value(0, M31Matrix::new(1, 1))
            .unwrap_err();
        assert!(wrong_shape_err.contains("matrix shape"));

        let mut extra_inventory = inventory.clone();
        extra_inventory.insert_runtime_matrix(
            "model.language_model.layers.0.linear_attn.extra",
            M31Matrix::new(2, 6),
        );
        let err = manifest
            .validate_source_inventory(&extra_inventory)
            .unwrap_err();
        assert!(err.contains("extra runtime typed witness inventory value"));

        let mut seeded_inventory = Qwen35TypedWitnessSourceInventory::new();
        seeded_inventory.insert_safetensors_vector(
            "model.language_model.layers.0.linear_attn.A_log",
            vec![M31::from(3), M31::from(4)],
        );
        let mut seeded_recorder =
            Qwen35TypedWitnessInventoryRecorder::with_inventory(manifest.clone(), seeded_inventory);
        seeded_recorder
            .record_attended_value(0, attended_matrix.clone())
            .unwrap();
        seeded_recorder
            .record_initial_recurrent_state(0, state_matrix.clone())
            .unwrap();
        seeded_recorder
            .record_norm_and_z_gate_rsqrt_table(0, 4)
            .unwrap();
        let seeded_commitment_set = seeded_recorder.validate().unwrap();
        assert_eq!(
            seeded_commitment_set.commitment_set_hash,
            commitment_set.commitment_set_hash
        );
        let duplicate_seeded_err = seeded_recorder
            .record_a_log_weight(0, vec![M31::from(3), M31::from(4)])
            .unwrap_err();
        assert!(duplicate_seeded_err.contains("duplicate safetensors typed witness recorder value"));

        let mut wrong_namespace = inventory.clone();
        let attended_value = wrong_namespace
            .runtime
            .remove("model.language_model.layers.0.linear_attn.attended_value")
            .unwrap();
        wrong_namespace.safetensors.insert(
            "model.language_model.layers.0.linear_attn.attended_value".into(),
            attended_value,
        );
        let err = manifest
            .validate_source_inventory(&wrong_namespace)
            .unwrap_err();
        assert!(err.contains("missing runtime typed witness inventory value"));

        let mut missing = source_capture.clone();
        missing.sources.pop();
        let err = manifest.validate_source_capture(&missing).unwrap_err();
        assert!(err.contains("missing captured typed witness source"));

        let mut duplicate = source_capture.clone();
        duplicate.sources.push(duplicate.sources[0].clone());
        let err = manifest.validate_source_capture(&duplicate).unwrap_err();
        assert!(err.contains("duplicate captured typed witness source"));

        let mut wrong_shape = source_capture.clone();
        wrong_shape.sources[0].value =
            Qwen35TypedWitnessCapturedValue::Matrix(M31Matrix::new(1, 1));
        let err = manifest.validate_source_capture(&wrong_shape).unwrap_err();
        assert!(err.contains("matrix shape"));

        let mut extra = source_capture;
        extra.insert_vector(
            "runtime:model.language_model.layers.0.linear_attn.extra",
            vec![],
        );
        let err = manifest.validate_source_capture(&extra).unwrap_err();
        assert!(err.contains("not required by manifest"));
    }

    #[test]
    fn typed_runtime_trace_finishes_with_preloaded_safetensors_inventory() {
        let cfg = qwen35_small_linear_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let seq_len = 2;
        let contract = plan.gated_delta_net_contract(0, seq_len).unwrap();
        let manifest = plan.typed_witness_manifest(seq_len).unwrap();

        let mut safetensors = Qwen35TypedWitnessSourceInventory::new();
        let conv1d_weight = m31_vector(
            contract.conv1d_weight.outer
                * contract.conv1d_weight.middle
                * contract.conv1d_weight.inner,
            200,
        );
        let a_log_weight = m31_vector(contract.a_log_weight, 300);
        let dt_bias = m31_vector(contract.dt_bias, 400);
        let norm_weight = m31_vector(contract.norm_weight, 500);
        safetensors.insert_safetensors_vector(
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            conv1d_weight,
        );
        safetensors.insert_safetensors_vector(
            "model.language_model.layers.0.linear_attn.A_log",
            a_log_weight.clone(),
        );
        safetensors.insert_safetensors_vector(
            "model.language_model.layers.0.linear_attn.dt_bias",
            dt_bias.clone(),
        );
        safetensors.insert_safetensors_vector(
            "model.language_model.layers.0.linear_attn.norm.weight",
            norm_weight,
        );

        let mut layer_trace = qwen35_runtime_trace_for_contract(&contract, 10);
        layer_trace.delta_recurrence_transform = Some(Qwen35DeltaRecurrenceRuntimeTransformTrace {
            scaled_query: qwen35_delta_norm_output(
                &layer_trace.query,
                contract.recurrent_state_rows,
                16,
                M31::from(1u32),
            ),
            normalized_key: qwen35_delta_norm_output(
                &layer_trace.key,
                contract.recurrent_state_rows,
                16,
                M31::from(1u32),
            ),
            decay: qwen35_delta_decay_output(&layer_trace.a_gate, &a_log_weight, &dt_bias, 16),
            beta: qwen35_delta_beta_output(&layer_trace.b_gate),
            q_norm_table_log_size: 16,
            k_norm_table_log_size: 16,
            beta_sigmoid_table_log_size: 16,
            decay_table_log_size: 16,
            query_norm_post_scale: M31::from(1u32),
            key_norm_post_scale: M31::from(1u32),
        });
        let trace = Qwen35TypedRuntimeTrace::new(seq_len, vec![layer_trace]);
        trace.validate_against_plan(&plan).unwrap();
        let commitment_set = trace
            .finish_with_safetensors_inventory(&plan, safetensors.clone())
            .unwrap();
        assert_eq!(commitment_set.manifest_hash, manifest.manifest_hash);
        assert_eq!(commitment_set.seq_len, seq_len);
        assert_eq!(commitment_set.commitments.len(), manifest.total_roots());

        let statement_set = trace.build_statement_set(&plan, &safetensors).unwrap();
        assert_eq!(statement_set.depthwise_conv1d.len(), 1);
        assert_eq!(statement_set.delta_recurrence.len(), 1);
        assert_eq!(statement_set.delta_recurrence_arithmetic.len(), 1);
        assert_eq!(statement_set.delta_recurrence_transform.len(), 1);
        assert_eq!(statement_set.delta_recurrence_q_norm.len(), 1);
        assert_eq!(statement_set.delta_recurrence_k_norm.len(), 1);
        assert_eq!(statement_set.delta_recurrence_beta_sigmoid.len(), 1);
        assert_eq!(statement_set.delta_recurrence_decay.len(), 1);
        assert!(statement_set.delta_recurrence_transform_statement_coverage_complete());
        assert_eq!(statement_set.norm_and_z_gate.len(), 1);
        let ledger = statement_set
            .build_typed_proof_ledger(&plan, seq_len)
            .unwrap();
        assert!(ledger.depthwise_conv1d_coverage_complete());
        assert!(ledger.norm_and_z_gate_coverage_complete());
        let delta_err = ledger.validate_delta_recurrence_coverage().unwrap_err();
        assert!(delta_err.contains("missing valid nonlinear aggregate coverage"));
        assert!(!ledger.production_ready());
        let runtime_ledger = trace.build_typed_proof_ledger(&plan, &safetensors).unwrap();
        assert_eq!(runtime_ledger.ledger_hash(), ledger.ledger_hash());

        let mut recorder = Qwen35TypedWitnessInventoryRecorder::with_inventory(
            manifest.clone(),
            safetensors.clone(),
        );
        trace.record_into(&mut recorder).unwrap();
        let recorder_commitment_set = recorder.finish().unwrap();
        assert_eq!(
            recorder_commitment_set.commitment_set_hash,
            commitment_set.commitment_set_hash
        );

        let missing = Qwen35TypedRuntimeTrace::new(seq_len, vec![]);
        let err = missing
            .finish_with_safetensors_inventory(&plan, safetensors.clone())
            .unwrap_err();
        assert!(err.contains("missing typed runtime trace layer 0"));

        let duplicate = Qwen35TypedRuntimeTrace::new(
            seq_len,
            vec![trace.layers[0].clone(), trace.layers[0].clone()],
        );
        let err = duplicate
            .finish_with_safetensors_inventory(&plan, safetensors.clone())
            .unwrap_err();
        assert!(err.contains("duplicate typed runtime trace layer 0"));

        let mut missing_weight = safetensors.clone();
        missing_weight
            .safetensors
            .remove("model.language_model.layers.0.linear_attn.conv1d.weight");
        let err = trace
            .build_statement_set(&plan, &missing_weight)
            .unwrap_err();
        assert!(err.contains("missing safetensors typed runtime statement source"));

        let mut tampered_beta_trace = trace.clone();
        tampered_beta_trace.layers[0]
            .delta_recurrence_transform
            .as_mut()
            .unwrap()
            .beta
            .data[0] += M31::from(1u32);
        let err = tampered_beta_trace
            .build_statement_set(&plan, &safetensors)
            .unwrap_err();
        assert!(err.contains("DeltaRecurrence beta sigmoid output mismatch"));

        let mut tampered_trace = trace.clone();
        tampered_trace.layers[0].gated_value.data[0] = M31::from(999u32);
        let err = tampered_trace
            .build_statement_set(&plan, &safetensors)
            .unwrap_err();
        assert!(err.contains("NormAndZGate output mismatch"));

        let mut wrong_layer = trace.layers[0].clone();
        wrong_layer.layer_idx = 1;
        let err = Qwen35TypedRuntimeTrace::new(seq_len, vec![wrong_layer])
            .finish_with_safetensors_inventory(&plan, safetensors.clone())
            .unwrap_err();
        assert!(err.contains("is not a GatedDeltaNet layer"));

        let mut wrong_shape = trace.layers[0].clone();
        wrong_shape.qkv_projected = M31Matrix::new(1, 1);
        let err = Qwen35TypedRuntimeTrace::new(seq_len, vec![wrong_shape])
            .validate_against_plan(&plan)
            .unwrap_err();
        assert!(err.contains("layer 0 qkv_projected matrix shape"));

        let mut wrong_state_shape = trace.layers[0].clone();
        wrong_state_shape.final_recurrent_state = M31Matrix::new(1, 1);
        let err = Qwen35TypedRuntimeTrace::new(seq_len, vec![wrong_state_shape])
            .finish_with_safetensors_inventory(&plan, safetensors.clone())
            .unwrap_err();
        assert!(err.contains("layer 0 final_recurrent_state matrix shape"));

        let mut zero_table = trace.layers[0].clone();
        zero_table.norm_and_z_gate_rsqrt_table_log_size = 0;
        let err = Qwen35TypedRuntimeTrace::new(seq_len, vec![zero_table])
            .finish_with_safetensors_inventory(&plan, safetensors)
            .unwrap_err();
        assert!(err.contains("norm_and_z_gate_rsqrt_table_log_size must be non-zero"));
    }

    #[test]
    fn typed_runtime_trace_proves_active_air_ledger_from_runtime_values() {
        let cfg = qwen35_small_linear_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let seq_len = 2;
        let contract = plan.gated_delta_net_contract(0, seq_len).unwrap();

        let conv1d_weight = m31_zero_vector(
            contract.conv1d_weight.outer
                * contract.conv1d_weight.middle
                * contract.conv1d_weight.inner,
        );
        let a_log_weight = m31_zero_vector(contract.a_log_weight);
        let dt_bias = m31_zero_vector(contract.dt_bias);
        let norm_weight = m31_zero_vector(contract.norm_weight);
        let mut safetensors = Qwen35TypedWitnessSourceInventory::new();
        safetensors.insert_safetensors_vector(
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            conv1d_weight,
        );
        safetensors.insert_safetensors_vector(
            "model.language_model.layers.0.linear_attn.A_log",
            a_log_weight.clone(),
        );
        safetensors.insert_safetensors_vector(
            "model.language_model.layers.0.linear_attn.dt_bias",
            dt_bias.clone(),
        );
        safetensors.insert_safetensors_vector(
            "model.language_model.layers.0.linear_attn.norm.weight",
            norm_weight,
        );

        let layer_trace =
            qwen35_active_runtime_trace_for_contract(&contract, &a_log_weight, &dt_bias);
        let trace = Qwen35TypedRuntimeTrace::new(seq_len, vec![layer_trace]);
        let statement_ledger = trace.build_typed_proof_ledger(&plan, &safetensors).unwrap();
        assert!(!statement_ledger.delta_recurrence_coverage_complete());
        assert!(!statement_ledger.production_ready());

        let active_receipt = trace
            .prove_active_typed_span(&plan, &safetensors, 0)
            .unwrap();
        active_receipt.validate().unwrap();
        assert_eq!(active_receipt.span_idx, 0);
        assert_ne!(active_receipt.receipt_hash, FieldElement::ZERO);
        assert_eq!(
            active_receipt.conversation_span.typed_ledger_hash,
            active_receipt.typed_ledger.ledger_hash()
        );
        assert_eq!(
            active_receipt.witness_commitment_set.manifest_hash,
            plan.typed_witness_manifest(seq_len).unwrap().manifest_hash
        );
        let active_ledger = &active_receipt.typed_ledger;
        assert!(active_ledger.depthwise_conv1d_coverage_complete());
        assert!(active_ledger.delta_recurrence_coverage_complete());
        assert!(active_ledger.norm_and_z_gate_coverage_complete());
        assert!(active_ledger.production_ready());

        let active_conversation = Qwen35ActiveConversationReceipt::from_span_receipts(
            &plan,
            vec![active_receipt.clone()],
        )
        .unwrap();
        active_conversation.validate().unwrap();
        assert_eq!(active_conversation.span_receipts.len(), 1);
        assert_ne!(active_conversation.receipt_hash, FieldElement::ZERO);

        let previous_final_state =
            active_receipt.conversation_span.layer_states[0].final_recurrent_state_commitment;
        let mut span1_ledger = active_ledger.clone();
        span1_ledger
            .statements
            .iter_mut()
            .find(|statement| statement.kind == Qwen35TypedProofStatementKind::DeltaRecurrence)
            .unwrap()
            .initial_recurrent_state_commitment = Some(previous_final_state);
        let mut span1_conversation_ledger = Qwen35ConversationStateLedger::new(
            plan.architecture_contract_hash(),
            plan.linear_attention_layers(),
        );
        span1_conversation_ledger
            .record_active_span(1, &span1_ledger)
            .unwrap();
        let span1 = span1_conversation_ledger.spans.first().unwrap().clone();
        let span1_receipt_hash = Qwen35ActiveTypedSpanReceipt::compute_hash(
            1,
            active_receipt.witness_commitment_set.commitment_set_hash,
            span1_ledger.ledger_hash(),
            &span1,
        );
        let span1_receipt = Qwen35ActiveTypedSpanReceipt {
            span_idx: 1,
            witness_commitment_set: active_receipt.witness_commitment_set.clone(),
            typed_ledger: span1_ledger,
            conversation_span: span1,
            receipt_hash: span1_receipt_hash,
        };
        let mut mixed_weight_span1_receipt = span1_receipt.clone();
        mixed_weight_span1_receipt
            .witness_commitment_set
            .commitment_set_hash = FieldElement::from(777_002u64);
        mixed_weight_span1_receipt.receipt_hash = Qwen35ActiveTypedSpanReceipt::compute_hash(
            1,
            mixed_weight_span1_receipt
                .witness_commitment_set
                .commitment_set_hash,
            mixed_weight_span1_receipt.typed_ledger.ledger_hash(),
            &mixed_weight_span1_receipt.conversation_span,
        );
        let err = Qwen35ActiveConversationReceipt::from_span_receipts(
            &plan,
            vec![active_receipt.clone(), mixed_weight_span1_receipt],
        )
        .unwrap_err();
        assert!(err.contains("witness commitment root"));

        let two_span_conversation = Qwen35ActiveConversationReceipt::from_span_receipts(
            &plan,
            vec![active_receipt.clone(), span1_receipt.clone()],
        )
        .unwrap();
        two_span_conversation.validate().unwrap();
        assert_eq!(two_span_conversation.span_receipts.len(), 2);
        assert!(two_span_conversation
            .conversation_ledger
            .continuity_complete());
        let qwen35_steps = two_span_conversation
            .generation_step_statements(
                0,
                0,
                0,
                &[101, 102],
                &[FieldElement::from(10_001u64), FieldElement::from(10_002u64)],
                &[FieldElement::ZERO, FieldElement::ZERO],
            )
            .unwrap();
        assert_eq!(qwen35_steps.len(), 2);
        assert_eq!(
            qwen35_steps[0].recursive_proof_hash,
            active_receipt.receipt_hash
        );
        assert_eq!(
            qwen35_steps[1].recursive_proof_hash,
            span1_receipt.receipt_hash
        );
        assert_eq!(
            qwen35_steps[0].kv_commitment,
            qwen35_steps[1].prev_kv_commitment
        );
        let qwen35_conversation_trace = two_span_conversation
            .conversation_trace_statement(
                0,
                crate::conversation_statement::text_commitment(0x434944, "qwen35-active-test"),
                FieldElement::from(20_001u64),
                FieldElement::from(20_002u64),
                crate::conversation_statement::action_root(&[]),
                1,
                2,
                0,
            )
            .unwrap();
        assert_eq!(
            qwen35_conversation_trace.initial_kv_commitment,
            qwen35_steps[0].prev_kv_commitment
        );
        assert_eq!(
            qwen35_conversation_trace.final_kv_commitment,
            qwen35_steps[1].kv_commitment
        );
        let qwen35_statement = crate::conversation_statement::build_conversation_batch_statement(
            FieldElement::from(30_001u64),
            FieldElement::ZERO,
            plan.architecture_contract_hash(),
            active_receipt.witness_commitment_set.commitment_set_hash,
            FieldElement::from(30_002u64),
            FieldElement::ZERO,
            FieldElement::ZERO,
            &[qwen35_conversation_trace.clone()],
            &qwen35_steps,
            &[],
            crate::conversation_statement::PRODUCTION_SECURITY_BITS,
        )
        .unwrap();
        assert_ne!(qwen35_statement.statement_hash(), FieldElement::ZERO);

        let qwen35_active_statement0 = two_span_conversation
            .active_statement(
                0,
                0,
                0,
                crate::conversation_statement::text_commitment(0x434944, "qwen35-active-test-0"),
                FieldElement::from(20_001u64),
                FieldElement::from(20_002u64),
                &[],
                1,
                2,
                &[101, 102],
                &[FieldElement::from(10_001u64), FieldElement::from(10_002u64)],
                &[FieldElement::ZERO, FieldElement::ZERO],
            )
            .unwrap();
        qwen35_active_statement0.validate().unwrap();
        let qwen35_active_statement1 = active_conversation
            .active_statement(
                1,
                2,
                0,
                crate::conversation_statement::text_commitment(0x434944, "qwen35-active-test-1"),
                FieldElement::from(20_101u64),
                FieldElement::from(20_102u64),
                &[],
                1,
                0,
                &[201],
                &[FieldElement::from(10_101u64)],
                &[FieldElement::ZERO],
            )
            .unwrap();
        qwen35_active_statement1.validate().unwrap();
        let mut broken_span_receipt_statement = qwen35_active_statement0.clone();
        broken_span_receipt_statement.span_receipt_hashes[1] = FieldElement::from(888_001u64);
        let err = broken_span_receipt_statement.validate().unwrap_err();
        assert!(err.contains("span receipt hash"));

        let qwen35_active_artifact = qwen35_build_active_conversation_batch_artifact(
            FieldElement::from(30_001u64),
            FieldElement::ZERO,
            FieldElement::from(30_002u64),
            FieldElement::ZERO,
            FieldElement::ZERO,
            &[
                qwen35_active_statement0.clone(),
                qwen35_active_statement1.clone(),
            ],
            crate::conversation_statement::PRODUCTION_SECURITY_BITS,
        )
        .unwrap();
        qwen35_active_artifact.validate().unwrap();
        assert_ne!(
            qwen35_active_artifact.canonical_statement.statement_hash(),
            FieldElement::ZERO
        );
        assert_ne!(
            qwen35_active_artifact.active_statement_root,
            FieldElement::ZERO
        );
        assert_ne!(
            qwen35_active_artifact.active_receipt_root,
            FieldElement::ZERO
        );
        assert_ne!(qwen35_active_artifact.artifact_hash, FieldElement::ZERO);
        assert_eq!(
            qwen35_active_artifact.public_output_hash(),
            qwen35_active_artifact.artifact_hash
        );
        let active_batch_felts =
            qwen35_active_batch_artifact_felts(&qwen35_active_artifact).unwrap();
        assert_eq!(
            active_batch_felts[0],
            FieldElement::from(DOMAIN_QWEN35_ACTIVE_BATCH_ARTIFACT)
        );
        assert_eq!(
            active_batch_felts[1],
            qwen35_active_artifact.canonical_statement.statement_hash()
        );
        let active_statement_felts =
            qwen35_active_conversation_statement_felts(&qwen35_active_statement0).unwrap();
        assert_eq!(
            active_statement_felts[0],
            FieldElement::from(DOMAIN_QWEN35_ACTIVE_CONVERSATION_STATEMENT)
        );
        assert_eq!(
            starknet_crypto::poseidon_hash_many(&active_statement_felts),
            qwen35_active_statement0.commitment().unwrap()
        );
        let active_artifact_json = qwen35_active_conversation_batch_artifact_json(
            &qwen35_active_artifact,
            &[
                qwen35_active_statement0.clone(),
                qwen35_active_statement1.clone(),
            ],
        )
        .unwrap();
        assert_eq!(
            active_artifact_json["schema"],
            "obelyzk.qwen35_active_conversation_batch_artifact.v1"
        );
        assert_eq!(
            active_artifact_json["statement_hash"],
            format!("0x{:x}", qwen35_active_artifact.artifact_hash)
        );
        assert_eq!(
            active_artifact_json["canonical_statement_hash"],
            format!(
                "0x{:x}",
                qwen35_active_artifact.canonical_statement.statement_hash()
            )
        );
        assert_eq!(
            active_artifact_json["active_conversations"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        let felt_json = |felt: FieldElement| serde_json::Value::String(format!("0x{:x}", felt));
        let conversation_json =
            |conversation: &crate::conversation_statement::ConversationTraceStatement| {
                serde_json::json!({
                    "conversation_index": conversation.conversation_index,
                    "conversation_id_hash": felt_json(conversation.conversation_id_hash),
                    "prompt_commitment": felt_json(conversation.prompt_commitment),
                    "transcript_commitment": felt_json(conversation.transcript_commitment),
                    "action_root": felt_json(conversation.action_root),
                    "initial_kv_commitment": felt_json(conversation.initial_kv_commitment),
                    "final_kv_commitment": felt_json(conversation.final_kv_commitment),
                    "n_turns": conversation.n_turns,
                    "n_prefill_tokens": conversation.n_prefill_tokens,
                    "n_generated_tokens": conversation.n_generated_tokens,
                    "first_step_index": conversation.first_step_index,
                    "n_steps": conversation.n_steps,
                })
            };
        let step_json = |step: &crate::conversation_statement::GenerationStepStatement| {
            serde_json::json!({
                "global_step_index": step.global_step_index,
                "conversation_index": step.conversation_index,
                "turn_index": step.turn_index,
                "token_index": step.token_index,
                "generated_token_id": step.generated_token_id,
                "io_commitment": felt_json(step.io_commitment),
                "sampling_commitment": felt_json(step.sampling_commitment),
                "prev_kv_commitment": felt_json(step.prev_kv_commitment),
                "kv_commitment": felt_json(step.kv_commitment),
                "recursive_proof_hash": felt_json(step.recursive_proof_hash),
            })
        };
        let active_statement_json = |active: &Qwen35ActiveConversationStatement| {
            serde_json::json!({
                "architecture_contract_hash": felt_json(active.architecture_contract_hash),
                "weight_super_root": felt_json(active.weight_super_root),
                "receipt_hash": felt_json(active.receipt_hash),
                "conversation": conversation_json(&active.conversation),
                "steps": active.steps.iter().map(step_json).collect::<Vec<_>>(),
                "span_receipt_hashes": active.span_receipt_hashes
                    .iter()
                    .copied()
                    .map(felt_json)
                    .collect::<Vec<_>>(),
                "actions": [],
            })
        };
        let active_artifact_input = serde_json::json!({
            "model_id": "0x7531",
            "verifier_program_hash": "0x0",
            "policy_commitment": "0x7532",
            "tokenizer_config_hash": "0x0",
            "hades_commitment": "0x0",
            "security_bits": crate::conversation_statement::PRODUCTION_SECURITY_BITS,
            "active_conversations": [
                active_statement_json(&qwen35_active_statement0),
                active_statement_json(&qwen35_active_statement1),
            ],
        });
        let active_artifact_from_json = qwen35_active_conversation_batch_artifact_json_from_str(
            &active_artifact_input.to_string(),
        )
        .unwrap();
        assert_eq!(
            active_artifact_from_json["statement_hash"],
            active_artifact_json["statement_hash"]
        );
        assert_eq!(
            active_artifact_from_json["active_batch_felts"],
            active_artifact_json["active_batch_felts"]
        );

        let mut tampered_artifact = qwen35_active_artifact.clone();
        tampered_artifact.active_receipt_root = FieldElement::from(888_002u64);
        let err = tampered_artifact.validate().unwrap_err();
        assert!(err.contains("batch artifact hash"));
        let err = qwen35_active_conversation_batch_artifact_json(
            &tampered_artifact,
            &[
                qwen35_active_statement0.clone(),
                qwen35_active_statement1.clone(),
            ],
        )
        .unwrap_err();
        assert!(err.contains("batch artifact hash"));

        let mut relabelled_receipt_statement = qwen35_active_statement1.clone();
        relabelled_receipt_statement.receipt_hash = FieldElement::from(888_003u64);
        let relabelled_receipt_artifact = qwen35_build_active_conversation_batch_artifact(
            FieldElement::from(30_001u64),
            FieldElement::ZERO,
            FieldElement::from(30_002u64),
            FieldElement::ZERO,
            FieldElement::ZERO,
            &[
                qwen35_active_statement0.clone(),
                relabelled_receipt_statement.clone(),
            ],
            crate::conversation_statement::PRODUCTION_SECURITY_BITS,
        )
        .unwrap();
        assert_eq!(
            relabelled_receipt_artifact
                .canonical_statement
                .statement_hash(),
            qwen35_active_artifact.canonical_statement.statement_hash()
        );
        assert_ne!(
            relabelled_receipt_artifact.active_receipt_root,
            qwen35_active_artifact.active_receipt_root
        );
        assert_ne!(
            relabelled_receipt_artifact.artifact_hash,
            qwen35_active_artifact.artifact_hash
        );
        let err = qwen35_active_conversation_batch_artifact_json(
            &qwen35_active_artifact,
            &[
                qwen35_active_statement0.clone(),
                relabelled_receipt_statement,
            ],
        )
        .unwrap_err();
        assert!(err.contains("root"));

        let qwen35_active_batch = qwen35_build_active_conversation_batch_statement(
            FieldElement::from(30_001u64),
            FieldElement::ZERO,
            FieldElement::from(30_002u64),
            FieldElement::ZERO,
            FieldElement::ZERO,
            &[
                qwen35_active_statement0.clone(),
                qwen35_active_statement1.clone(),
            ],
            crate::conversation_statement::PRODUCTION_SECURITY_BITS,
        )
        .unwrap();
        assert_ne!(qwen35_active_batch.statement_hash(), FieldElement::ZERO);
        assert_eq!(qwen35_active_batch.n_conversations, 2);
        assert_eq!(qwen35_active_batch.n_generated_tokens, 3);
        assert_eq!(
            qwen35_active_batch.circuit_hash,
            plan.architecture_contract_hash()
        );
        assert_eq!(
            qwen35_active_batch.weight_super_root,
            active_receipt.witness_commitment_set.commitment_set_hash
        );

        let mut wrong_weight_statement = qwen35_active_statement1.clone();
        wrong_weight_statement.weight_super_root = FieldElement::from(777_001u64);
        let err = qwen35_build_active_conversation_batch_statement(
            FieldElement::from(30_001u64),
            FieldElement::ZERO,
            FieldElement::from(30_002u64),
            FieldElement::ZERO,
            FieldElement::ZERO,
            &[qwen35_active_statement0.clone(), wrong_weight_statement],
            crate::conversation_statement::PRODUCTION_SECURITY_BITS,
        )
        .unwrap_err();
        assert!(err.contains("weight root"));

        let mut broken_qwen35_steps = qwen35_steps.clone();
        broken_qwen35_steps[1].prev_kv_commitment = FieldElement::from(999_001u64);
        let err = crate::conversation_statement::build_conversation_batch_statement(
            FieldElement::from(30_001u64),
            FieldElement::ZERO,
            plan.architecture_contract_hash(),
            active_receipt.witness_commitment_set.commitment_set_hash,
            FieldElement::from(30_002u64),
            FieldElement::ZERO,
            FieldElement::ZERO,
            &[qwen35_conversation_trace],
            &broken_qwen35_steps,
            &[],
            crate::conversation_statement::PRODUCTION_SECURITY_BITS,
        )
        .unwrap_err();
        assert!(err.to_string().contains("prev KV mismatch"));

        let mut non_contiguous_receipt = span1_receipt.clone();
        non_contiguous_receipt.span_idx = 2;
        non_contiguous_receipt.conversation_span.span_idx = 2;
        non_contiguous_receipt.receipt_hash = Qwen35ActiveTypedSpanReceipt::compute_hash(
            2,
            non_contiguous_receipt
                .witness_commitment_set
                .commitment_set_hash,
            non_contiguous_receipt.typed_ledger.ledger_hash(),
            &non_contiguous_receipt.conversation_span,
        );
        let err = Qwen35ActiveConversationReceipt::from_span_receipts(
            &plan,
            vec![active_receipt.clone(), non_contiguous_receipt],
        )
        .unwrap_err();
        assert!(err.contains("spans must be contiguous"));

        let mut state_mismatch_ledger = active_ledger.clone();
        state_mismatch_ledger
            .statements
            .iter_mut()
            .find(|statement| statement.kind == Qwen35TypedProofStatementKind::DeltaRecurrence)
            .unwrap()
            .initial_recurrent_state_commitment = Some(FieldElement::from(888_888u64));
        let mut mismatch_conversation_ledger = Qwen35ConversationStateLedger::new(
            plan.architecture_contract_hash(),
            plan.linear_attention_layers(),
        );
        mismatch_conversation_ledger
            .record_active_span(1, &state_mismatch_ledger)
            .unwrap();
        let mismatch_span = mismatch_conversation_ledger.spans.first().unwrap().clone();
        let mismatch_receipt = Qwen35ActiveTypedSpanReceipt {
            span_idx: 1,
            witness_commitment_set: active_receipt.witness_commitment_set.clone(),
            typed_ledger: state_mismatch_ledger,
            receipt_hash: Qwen35ActiveTypedSpanReceipt::compute_hash(
                1,
                active_receipt.witness_commitment_set.commitment_set_hash,
                mismatch_span.typed_ledger_hash,
                &mismatch_span,
            ),
            conversation_span: mismatch_span,
        };
        let err = Qwen35ActiveConversationReceipt::from_span_receipts(
            &plan,
            vec![active_receipt.clone(), mismatch_receipt],
        )
        .unwrap_err();
        assert!(err.contains("conversation recurrent state mismatch"));

        let delta_statement = active_ledger
            .statements
            .iter()
            .find(|statement| statement.kind == Qwen35TypedProofStatementKind::DeltaRecurrence)
            .unwrap();
        assert!(delta_statement.arithmetic_statement_hash.is_some());
        assert!(delta_statement.transform_statement_hash.is_some());
        assert!(delta_statement
            .transform_nonlinear_q_norm_statement_hash
            .is_some());
        assert!(delta_statement
            .transform_nonlinear_k_norm_statement_hash
            .is_some());
        assert!(delta_statement
            .transform_nonlinear_beta_sigmoid_statement_hash
            .is_some());
        assert!(delta_statement
            .transform_nonlinear_decay_statement_hash
            .is_some());
        assert!(delta_statement.transform_nonlinear_statement_hash.is_some());

        let mut stale_receipt = active_receipt.clone();
        stale_receipt.receipt_hash = FieldElement::from(123_456u64);
        let err = stale_receipt.validate().unwrap_err();
        assert!(err.contains("active typed span receipt hash"));

        let mut relabeled_receipt = active_receipt.clone();
        relabeled_receipt.conversation_span.typed_ledger_hash = FieldElement::from(654_321u64);
        let err = relabeled_receipt.validate().unwrap_err();
        assert!(err.contains("conversation ledger hash"));

        let mut bad_conversation = active_conversation.clone();
        bad_conversation.receipt_hash = FieldElement::from(777_777u64);
        let err = bad_conversation.validate().unwrap_err();
        assert!(err.contains("active Qwen3.5 conversation receipt hash"));

        let mut missing_transform = trace.clone();
        missing_transform.layers[0].delta_recurrence_transform = None;
        let err = missing_transform
            .prove_active_typed_span(&plan, &safetensors, 0)
            .unwrap_err();
        assert!(err.contains("missing DeltaRecurrence transform runtime trace"));

        let mut tampered_depthwise_output = trace.clone();
        tampered_depthwise_output.layers[0].qkv_after_conv.data[0] = M31::from(1u32);
        let err = tampered_depthwise_output
            .prove_active_typed_span(&plan, &safetensors, 0)
            .unwrap_err();
        assert!(err.contains("DepthwiseConv1D active proof output"));
    }

    #[test]
    fn qwen35_execution_plan_reports_missing_dedicated_components() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let execution = plan.execution_plan();

        assert_eq!(execution.total_steps(), 363);
        assert_eq!(
            execution.count_status(Qwen35ComponentStatus::GenericAvailable),
            203
        );
        assert_eq!(
            execution.count_status(Qwen35ComponentStatus::DedicatedMissing),
            160
        );
        assert_eq!(
            execution.component_count(Qwen35ProofComponent::GatedDeltaNet),
            30
        );
        assert_eq!(
            execution.component_count(Qwen35ProofComponent::GatedFullAttention),
            10
        );
        assert_eq!(
            execution.component_count(Qwen35ProofComponent::PackedExpertBank),
            40
        );
        assert!(!execution.production_ready());

        let summary = execution.readiness_summary();
        assert!(summary.contains("363 total steps"));
        assert!(summary.contains("203 generic-ready"));
        assert!(summary.contains("160 missing dedicated"));
        assert!(summary.contains("GatedDeltaNet=30"));
        assert!(summary.contains("GatedFullAttention=10"));
        assert!(summary.contains("PackedExpertBank=40"));
        assert!(summary.contains("SharedExpert=40"));
        assert!(summary.contains("SharedExpertGate=40"));
    }

    #[test]
    fn gated_full_attention_contract_matches_qwen35_shapes() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let contract = plan.gated_full_attention_contract(3, 17).unwrap();

        assert_eq!(contract.layer_idx, 3);
        assert_eq!(
            contract.input,
            Qwen35Tensor2DShape {
                rows: 17,
                cols: 2048
            }
        );
        assert_eq!(
            contract.q_proj_weight,
            Qwen35Tensor2DShape {
                rows: 8192,
                cols: 2048
            }
        );
        assert_eq!(
            contract.q_proj_output,
            Qwen35Tensor2DShape {
                rows: 17,
                cols: 8192
            }
        );
        assert_eq!(
            contract.query,
            Qwen35Tensor2DShape {
                rows: 17,
                cols: 4096
            }
        );
        assert_eq!(
            contract.output_gate,
            Qwen35Tensor2DShape {
                rows: 17,
                cols: 4096
            }
        );
        assert_eq!(
            contract.query_heads,
            Qwen35Tensor3DShape {
                outer: 17,
                middle: 16,
                inner: 256
            }
        );
        assert_eq!(contract.q_norm_weight, 256);
        assert_eq!(
            contract.k_proj_weight,
            Qwen35Tensor2DShape {
                rows: 512,
                cols: 2048
            }
        );
        assert_eq!(
            contract.v_proj_weight,
            Qwen35Tensor2DShape {
                rows: 512,
                cols: 2048
            }
        );
        assert_eq!(
            contract.key,
            Qwen35Tensor2DShape {
                rows: 17,
                cols: 512
            }
        );
        assert_eq!(
            contract.value,
            Qwen35Tensor2DShape {
                rows: 17,
                cols: 512
            }
        );
        assert_eq!(
            contract.key_heads,
            Qwen35Tensor3DShape {
                outer: 17,
                middle: 2,
                inner: 256
            }
        );
        assert_eq!(
            contract.value_heads,
            Qwen35Tensor3DShape {
                outer: 17,
                middle: 2,
                inner: 256
            }
        );
        assert_eq!(contract.k_norm_weight, 256);
        assert_eq!(contract.query_groups_per_kv_head, 8);
        assert_eq!(
            contract.attention_context,
            Qwen35Tensor2DShape {
                rows: 17,
                cols: 4096
            }
        );
        assert_eq!(
            contract.gated_context,
            Qwen35Tensor2DShape {
                rows: 17,
                cols: 4096
            }
        );
        assert_eq!(
            contract.o_proj_weight,
            Qwen35Tensor2DShape {
                rows: 2048,
                cols: 4096
            }
        );
        assert_eq!(
            contract.output,
            Qwen35Tensor2DShape {
                rows: 17,
                cols: 2048
            }
        );
    }

    #[test]
    fn gated_full_attention_contract_rejects_linear_layers() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let err = plan.gated_full_attention_contract(0, 1).unwrap_err();
        assert!(err.contains("not gated full attention"));
    }

    #[test]
    fn moe_contract_matches_qwen35_packed_expert_shapes() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let contract = plan.moe_contract(0, 9).unwrap();

        assert_eq!(contract.layer_idx, 0);
        assert_eq!(
            contract.input,
            Qwen35Tensor2DShape {
                rows: 9,
                cols: 2048
            }
        );
        assert_eq!(
            contract.router_weight,
            Qwen35Tensor2DShape {
                rows: 256,
                cols: 2048
            }
        );
        assert_eq!(
            contract.router_logits,
            Qwen35Tensor2DShape { rows: 9, cols: 256 }
        );
        assert_eq!(
            contract.selected_expert_ids,
            Qwen35Tensor2DShape { rows: 9, cols: 8 }
        );
        assert_eq!(
            contract.routing_weights,
            Qwen35Tensor2DShape { rows: 9, cols: 8 }
        );
        assert_eq!(
            contract.packed_gate_up_weight,
            Qwen35Tensor3DShape {
                outer: 256,
                middle: 1024,
                inner: 2048
            }
        );
        assert_eq!(
            contract.expert_gate,
            Qwen35Tensor3DShape {
                outer: 9,
                middle: 8,
                inner: 512
            }
        );
        assert_eq!(contract.expert_gate, contract.expert_up);
        assert_eq!(contract.expert_gate, contract.expert_hidden);
        assert_eq!(
            contract.packed_down_weight,
            Qwen35Tensor3DShape {
                outer: 256,
                middle: 2048,
                inner: 512
            }
        );
        assert_eq!(
            contract.expert_output,
            Qwen35Tensor3DShape {
                outer: 9,
                middle: 8,
                inner: 2048
            }
        );
        assert_eq!(
            contract.routed_output,
            Qwen35Tensor2DShape {
                rows: 9,
                cols: 2048
            }
        );
        assert_eq!(
            contract.shared_gate_weight,
            Qwen35Tensor2DShape {
                rows: 512,
                cols: 2048
            }
        );
        assert_eq!(contract.shared_gate_weight, contract.shared_up_weight);
        assert_eq!(
            contract.shared_down_weight,
            Qwen35Tensor2DShape {
                rows: 2048,
                cols: 512
            }
        );
        assert_eq!(
            contract.shared_expert_gate_weight,
            Qwen35Tensor2DShape {
                rows: 1,
                cols: 2048
            }
        );
        assert_eq!(
            contract.shared_hidden,
            Qwen35Tensor2DShape { rows: 9, cols: 512 }
        );
        assert_eq!(
            contract.shared_output,
            Qwen35Tensor2DShape {
                rows: 9,
                cols: 2048
            }
        );
        assert_eq!(
            contract.output,
            Qwen35Tensor2DShape {
                rows: 9,
                cols: 2048
            }
        );
    }

    #[test]
    fn gated_delta_net_contract_matches_qwen35_linear_attention_shapes() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let contract = plan.gated_delta_net_contract(0, 13).unwrap();

        assert_eq!(contract.layer_idx, 0);
        assert_eq!(
            contract.input,
            Qwen35Tensor2DShape {
                rows: 13,
                cols: 2048
            }
        );
        assert_eq!(
            contract.in_proj_qkv_weight,
            Qwen35Tensor2DShape {
                rows: 8192,
                cols: 2048
            }
        );
        assert_eq!(
            contract.qkv_projected,
            Qwen35Tensor2DShape {
                rows: 13,
                cols: 8192
            }
        );
        assert_eq!(
            contract.conv1d_weight,
            Qwen35Tensor3DShape {
                outer: 8192,
                middle: 1,
                inner: 4
            }
        );
        assert_eq!(contract.qkv_projected, contract.qkv_after_conv);
        assert_eq!(
            contract.query,
            Qwen35Tensor2DShape {
                rows: 13,
                cols: 4096
            }
        );
        assert_eq!(
            contract.key,
            Qwen35Tensor2DShape {
                rows: 13,
                cols: 4096
            }
        );
        assert_eq!(
            contract.projected_value,
            Qwen35Tensor2DShape {
                rows: 13,
                cols: 4096
            }
        );
        assert_eq!(
            contract.in_proj_z_weight,
            Qwen35Tensor2DShape {
                rows: 4096,
                cols: 2048
            }
        );
        assert_eq!(
            contract.z_gate,
            Qwen35Tensor2DShape {
                rows: 13,
                cols: 4096
            }
        );
        assert_eq!(
            contract.in_proj_a_weight,
            Qwen35Tensor2DShape {
                rows: 32,
                cols: 2048
            }
        );
        assert_eq!(contract.in_proj_a_weight, contract.in_proj_b_weight);
        assert_eq!(contract.a_gate, Qwen35Tensor2DShape { rows: 13, cols: 32 });
        assert_eq!(contract.a_gate, contract.b_gate);
        assert_eq!(contract.a_log_weight, 32);
        assert_eq!(contract.dt_bias, 32);
        assert_eq!(contract.norm_weight, 128);
        assert_eq!(contract.linear_value_heads, 32);
        assert_eq!(contract.linear_value_head_dim, 128);
        assert_eq!(contract.recurrent_state_rows, 32);
        assert_eq!(
            contract.attended_value,
            Qwen35Tensor2DShape {
                rows: 13,
                cols: 4096
            }
        );
        assert_eq!(
            contract.o_proj_weight,
            Qwen35Tensor2DShape {
                rows: 2048,
                cols: 4096
            }
        );
        assert_eq!(
            contract.output,
            Qwen35Tensor2DShape {
                rows: 13,
                cols: 2048
            }
        );
    }

    #[test]
    fn gated_delta_net_stage_contract_enumerates_required_relations() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let contract = plan.gated_delta_net_contract(0, 13).unwrap();
        let stages = contract.stage_contracts();

        assert_eq!(stages.len(), 8);
        assert_eq!(contract.stage_status_counts(), (5, 1, 2));
        assert_eq!(stages[0].kind, Qwen35GatedDeltaNetStageKind::QkvProjection);
        assert_eq!(stages[0].status, Qwen35StageStatus::GenericAvailable);
        assert_eq!(
            stages[0].outputs,
            vec![Qwen35StageTensor {
                name: "qkv_projected",
                shape: Qwen35TensorShape::Matrix(Qwen35Tensor2DShape {
                    rows: 13,
                    cols: 8192
                })
            }]
        );
        assert_eq!(
            stages[1].inputs[1],
            Qwen35StageTensor {
                name: "conv1d_weight",
                shape: Qwen35TensorShape::Tensor3D(Qwen35Tensor3DShape {
                    outer: 8192,
                    middle: 1,
                    inner: 4
                })
            }
        );
        assert_eq!(
            stages[1].status,
            Qwen35StageStatus::DedicatedAirAvailableIntegrationMissing
        );
        assert_eq!(
            stages[5].kind,
            Qwen35GatedDeltaNetStageKind::DeltaRecurrence
        );
        assert_eq!(stages[5].status, Qwen35StageStatus::DedicatedMissing);
        assert!(stages[5].relation.contains("recurrent update"));
        assert_eq!(stages[6].status, Qwen35StageStatus::DedicatedMissing);
        assert_eq!(
            stages[6].outputs[0],
            Qwen35StageTensor {
                name: "gated_value",
                shape: Qwen35TensorShape::Matrix(Qwen35Tensor2DShape {
                    rows: 13,
                    cols: 4096
                })
            }
        );
        assert_eq!(
            stages[7].outputs[0],
            Qwen35StageTensor {
                name: "output",
                shape: Qwen35TensorShape::Matrix(Qwen35Tensor2DShape {
                    rows: 13,
                    cols: 2048
                })
            }
        );
    }

    #[test]
    fn gated_delta_net_stage_readiness_aggregates_all_linear_layers() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let readiness = plan.gated_delta_net_stage_readiness(1).unwrap();

        assert_eq!(readiness.total_stages, 240);
        assert_eq!(readiness.generic_ready_stages, 150);
        assert_eq!(readiness.dedicated_air_available_stages, 30);
        assert_eq!(readiness.missing_dedicated_stages, 60);
        assert_eq!(
            readiness.dedicated_air_stage_counts,
            vec![(Qwen35GatedDeltaNetStageKind::DepthwiseConv1d, 30)]
        );
        assert_eq!(
            readiness.missing_stage_counts,
            vec![
                (Qwen35GatedDeltaNetStageKind::DeltaRecurrence, 30),
                (Qwen35GatedDeltaNetStageKind::NormAndZGate, 30),
            ]
        );
    }

    #[test]
    fn depthwise_conv1d_air_contract_matches_qwen35_shape_and_constraints() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let contract = plan
            .gated_delta_net_contract(0, 13)
            .unwrap()
            .depthwise_conv1d_air_contract();

        assert_eq!(contract.layer_idx, 0);
        assert_eq!(contract.seq_len, 13);
        assert_eq!(contract.channels, 8192);
        assert_eq!(contract.kernel, 4);
        assert_eq!(contract.tap_offsets, vec![-3, -2, -1, 0]);
        assert_eq!(contract.logical_trace_rows, 13 * 8192);
        assert_eq!(contract.deterministic_columns, 6);
        assert_eq!(contract.witness_columns, 17);
        assert_eq!(contract.total_columns, 23);
        assert_eq!(contract.arithmetic_constraints_per_row, 9);
        assert_eq!(contract.row_binding_constraints_per_row, 6);
        assert_eq!(
            contract.input,
            Qwen35Tensor2DShape {
                rows: 13,
                cols: 8192
            }
        );
        assert_eq!(
            contract.weight,
            Qwen35Tensor3DShape {
                outer: 8192,
                middle: 1,
                inner: 4
            }
        );
        assert_eq!(contract.input, contract.output);
        assert!(contract.relation().contains("valid(t,tap)"));
    }

    #[test]
    fn depthwise_conv1d_air_readiness_aggregates_all_linear_layers() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let seq1 = plan.depthwise_conv1d_air_readiness(1).unwrap();
        let seq2 = plan.depthwise_conv1d_air_readiness(2).unwrap();

        assert_eq!(seq1.layers, 30);
        assert_eq!(seq1.channels_per_layer, 8192);
        assert_eq!(seq1.kernel, 4);
        assert_eq!(seq1.logical_rows_per_layer, 8192);
        assert_eq!(seq1.total_logical_rows, 30 * 8192);
        assert_eq!(seq1.columns_per_layer, 23);
        assert_eq!(seq1.arithmetic_constraints_per_row, 9);
        assert_eq!(seq1.row_binding_constraints_per_row, 6);
        assert_eq!(seq2.logical_rows_per_layer, 2 * 8192);
        assert_ne!(seq1.aggregate_contract_hash, seq2.aggregate_contract_hash);
        assert_ne!(
            seq1.aggregate_trace_binding_hash,
            seq2.aggregate_trace_binding_hash
        );
    }

    #[test]
    fn depthwise_conv1d_trace_binding_contract_links_producer_and_consumer_roots() {
        use crate::components::qwen35_depthwise_conv1d::qwen35_depthwise_conv1d_statement_hash;

        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let binding = plan
            .gated_delta_net_contract(0, 13)
            .unwrap()
            .depthwise_conv1d_trace_binding_contract()
            .unwrap();

        assert_eq!(binding.layer_idx, 0);
        assert_eq!(binding.seq_len, 13);
        assert_eq!(binding.channels, 8192);
        assert_eq!(binding.kernel, 4);
        assert_eq!(binding.stage_idx, 1);
        assert_eq!(binding.producer_stage_idx, 0);
        assert_eq!(binding.consumer_stage_idx, 2);
        assert_eq!(binding.input_root.name, "qkv_projected");
        assert_eq!(
            binding.input_root.role,
            Qwen35TraceRootRole::ProducerActivation
        );
        assert_eq!(binding.weight_root.name, "conv1d_weight");
        assert_eq!(binding.weight_root.role, Qwen35TraceRootRole::ModelWeight);
        assert_eq!(binding.output_root.name, "qkv_after_conv");
        assert_eq!(
            binding.output_root.role,
            Qwen35TraceRootRole::ConsumerActivation
        );

        let input_commitment = FieldElement::from(11u64);
        let weight_commitment = FieldElement::from(22u64);
        let output_commitment = FieldElement::from(33u64);
        assert_eq!(
            binding
                .expected_statement_hash(input_commitment, weight_commitment, output_commitment,),
            qwen35_depthwise_conv1d_statement_hash(
                0,
                13,
                8192,
                4,
                input_commitment,
                weight_commitment,
                output_commitment,
            )
        );
        binding
            .validate_statement(
                0,
                13,
                8192,
                4,
                input_commitment,
                weight_commitment,
                output_commitment,
                qwen35_depthwise_conv1d_statement_hash(
                    0,
                    13,
                    8192,
                    4,
                    input_commitment,
                    weight_commitment,
                    output_commitment,
                ),
            )
            .unwrap();
        assert!(binding
            .validate_statement(
                1,
                13,
                8192,
                4,
                input_commitment,
                weight_commitment,
                output_commitment,
                qwen35_depthwise_conv1d_statement_hash(
                    1,
                    13,
                    8192,
                    4,
                    input_commitment,
                    weight_commitment,
                    output_commitment,
                ),
            )
            .unwrap_err()
            .contains("statement layer"));
        assert!(binding
            .validate_statement(
                0,
                13,
                8192,
                4,
                input_commitment,
                weight_commitment,
                output_commitment,
                qwen35_depthwise_conv1d_statement_hash(
                    0,
                    13,
                    8192,
                    4,
                    input_commitment,
                    weight_commitment,
                    FieldElement::from(34u64),
                ),
            )
            .unwrap_err()
            .contains("statement hash"));

        let seq14_binding = plan
            .gated_delta_net_contract(0, 14)
            .unwrap()
            .depthwise_conv1d_trace_binding_contract()
            .unwrap();
        let layer1_binding = plan
            .gated_delta_net_contract(1, 13)
            .unwrap()
            .depthwise_conv1d_trace_binding_contract()
            .unwrap();
        assert_ne!(binding.contract_hash(), seq14_binding.contract_hash());
        assert_ne!(binding.contract_hash(), layer1_binding.contract_hash());
    }

    #[test]
    fn typed_proof_ledger_consumes_depthwise_statements_from_active_proofs() {
        use crate::components::qwen35_depthwise_conv1d::Qwen35DepthwiseConv1dStatement;

        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let mut ledger = plan.typed_proof_ledger(1);
        assert_eq!(ledger.expected_depthwise_conv1d_statements, 30);
        assert_eq!(ledger.expected_delta_recurrence_statements, 30);
        assert_eq!(ledger.expected_norm_and_z_gate_statements, 30);
        assert_eq!(
            ledger.count_kind(Qwen35TypedProofStatementKind::DepthwiseConv1d),
            0
        );
        assert!(!ledger.depthwise_conv1d_coverage_complete());
        assert!(!ledger.delta_recurrence_coverage_complete());
        assert!(!ledger.norm_and_z_gate_coverage_complete());
        assert!(!ledger.production_ready());
        let empty_hash = ledger.ledger_hash();

        let binding = plan
            .gated_delta_net_contract(0, 1)
            .unwrap()
            .depthwise_conv1d_trace_binding_contract()
            .unwrap();
        let statement = Qwen35DepthwiseConv1dStatement {
            layer_idx: binding.layer_idx,
            seq_len: binding.seq_len,
            channels: binding.channels,
            kernel: binding.kernel,
            input_commitment: FieldElement::from(101u64),
            weight_commitment: FieldElement::from(102u64),
            output_commitment: FieldElement::from(103u64),
            statement_hash: binding.expected_statement_hash(
                FieldElement::from(101u64),
                FieldElement::from(102u64),
                FieldElement::from(103u64),
            ),
        };

        ledger
            .record_depthwise_conv1d_proof(&binding, &statement)
            .unwrap();
        assert_eq!(
            ledger.count_kind(Qwen35TypedProofStatementKind::DepthwiseConv1d),
            1
        );
        assert_ne!(ledger.ledger_hash(), empty_hash);
        assert!(ledger
            .record_depthwise_conv1d_proof(&binding, &statement)
            .unwrap_err()
            .contains("duplicate"));

        let mut relabeled = statement.clone();
        relabeled.layer_idx = 1;
        relabeled.statement_hash = binding.expected_statement_hash(
            relabeled.input_commitment,
            relabeled.weight_commitment,
            relabeled.output_commitment,
        );
        let layer_err = plan
            .typed_proof_ledger(1)
            .record_depthwise_conv1d_proof(&binding, &relabeled)
            .unwrap_err();
        assert!(layer_err.contains("statement layer"));

        let mut bad_hash = statement.clone();
        bad_hash.statement_hash += FieldElement::ONE;
        let hash_err = plan
            .typed_proof_ledger(1)
            .record_depthwise_conv1d_proof(&binding, &bad_hash)
            .unwrap_err();
        assert!(hash_err.contains("statement hash"));
    }

    #[test]
    fn typed_proof_ledger_requires_all_depthwise_delta_and_norm_linear_layers() {
        use crate::components::qwen35_depthwise_conv1d::Qwen35DepthwiseConv1dStatement;
        use crate::components::qwen35_norm_and_z_gate::Qwen35NormAndZGateStatement;

        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let mut ledger = plan.typed_proof_ledger(1);

        for layer in &plan.layers {
            if layer.attention != Qwen35AttentionKind::GatedDeltaNet {
                continue;
            }
            let binding = plan
                .gated_delta_net_contract(layer.layer_idx, 1)
                .unwrap()
                .depthwise_conv1d_trace_binding_contract()
                .unwrap();
            let input_commitment = FieldElement::from(10_000u64 + layer.layer_idx as u64);
            let weight_commitment = FieldElement::from(20_000u64 + layer.layer_idx as u64);
            let output_commitment = FieldElement::from(30_000u64 + layer.layer_idx as u64);
            let statement = Qwen35DepthwiseConv1dStatement {
                layer_idx: binding.layer_idx,
                seq_len: binding.seq_len,
                channels: binding.channels,
                kernel: binding.kernel,
                input_commitment,
                weight_commitment,
                output_commitment,
                statement_hash: binding.expected_statement_hash(
                    input_commitment,
                    weight_commitment,
                    output_commitment,
                ),
            };
            ledger
                .record_depthwise_conv1d_proof(&binding, &statement)
                .unwrap();
        }

        assert_eq!(
            ledger.count_kind(Qwen35TypedProofStatementKind::DepthwiseConv1d),
            30
        );
        assert!(ledger.depthwise_conv1d_coverage_complete());
        assert!(!ledger.delta_recurrence_coverage_complete());
        assert!(!ledger.norm_and_z_gate_coverage_complete());
        assert!(!ledger.production_ready());

        for layer in &plan.layers {
            if layer.attention != Qwen35AttentionKind::GatedDeltaNet {
                continue;
            }
            let binding = plan
                .gated_delta_net_contract(layer.layer_idx, 1)
                .unwrap()
                .delta_recurrence_trace_binding_contract()
                .unwrap();
            let query_commitment = FieldElement::from(40_000u64 + layer.layer_idx as u64);
            let key_commitment = FieldElement::from(50_000u64 + layer.layer_idx as u64);
            let projected_value_commitment = FieldElement::from(60_000u64 + layer.layer_idx as u64);
            let a_gate_commitment = FieldElement::from(70_000u64 + layer.layer_idx as u64);
            let b_gate_commitment = FieldElement::from(80_000u64 + layer.layer_idx as u64);
            let a_log_weight_commitment = FieldElement::from(90_000u64 + layer.layer_idx as u64);
            let dt_bias_commitment = FieldElement::from(100_000u64 + layer.layer_idx as u64);
            let mode_tag = 1u64;
            let air_spec_hash = FieldElement::from(105_000u64 + layer.layer_idx as u64);
            let initial_recurrent_state_commitment =
                FieldElement::from(106_000u64 + layer.layer_idx as u64);
            let final_recurrent_state_commitment =
                FieldElement::from(107_000u64 + layer.layer_idx as u64);
            let output_commitment = FieldElement::from(110_000u64 + layer.layer_idx as u64);
            let statement_hash = binding.expected_statement_hash(
                query_commitment,
                key_commitment,
                projected_value_commitment,
                a_gate_commitment,
                b_gate_commitment,
                a_log_weight_commitment,
                dt_bias_commitment,
                mode_tag,
                air_spec_hash,
                initial_recurrent_state_commitment,
                final_recurrent_state_commitment,
                output_commitment,
            );
            ledger
                .record_delta_recurrence_statement(
                    &binding,
                    binding.layer_idx,
                    binding.seq_len,
                    binding.query_width,
                    binding.key_width,
                    binding.value_width,
                    binding.state_rows,
                    binding.value_head_dim,
                    query_commitment,
                    key_commitment,
                    projected_value_commitment,
                    a_gate_commitment,
                    b_gate_commitment,
                    a_log_weight_commitment,
                    dt_bias_commitment,
                    mode_tag,
                    air_spec_hash,
                    initial_recurrent_state_commitment,
                    final_recurrent_state_commitment,
                    output_commitment,
                    statement_hash,
                )
                .unwrap();
        }

        assert_eq!(
            ledger.count_kind(Qwen35TypedProofStatementKind::DeltaRecurrence),
            30
        );
        assert!(!ledger.delta_recurrence_coverage_complete());
        assert!(!ledger.production_ready());

        for statement in &mut ledger.statements {
            if statement.kind == Qwen35TypedProofStatementKind::DeltaRecurrence {
                statement.transform_statement_hash =
                    Some(statement.statement_hash + FieldElement::from(500_000u64));
                statement.transform_nonlinear_q_norm_statement_hash =
                    Some(statement.statement_hash + FieldElement::from(751_000u64));
                statement.transform_nonlinear_k_norm_statement_hash =
                    Some(statement.statement_hash + FieldElement::from(752_000u64));
                statement.transform_nonlinear_beta_sigmoid_statement_hash =
                    Some(statement.statement_hash + FieldElement::from(753_000u64));
                statement.transform_nonlinear_decay_statement_hash =
                    Some(statement.statement_hash + FieldElement::from(754_000u64));
                statement.arithmetic_statement_hash =
                    Some(statement.statement_hash + FieldElement::from(1_000_000u64));
                let aggregate_hash = statement
                    .delta_recurrence_nonlinear_aggregate_hash()
                    .unwrap();
                statement.transform_nonlinear_statement_hash = Some(aggregate_hash);
            }
        }
        assert!(ledger.delta_recurrence_coverage_complete());
        assert!(!ledger.norm_and_z_gate_coverage_complete());
        assert!(!ledger.production_ready());

        for layer in &plan.layers {
            if layer.attention != Qwen35AttentionKind::GatedDeltaNet {
                continue;
            }
            let binding = plan
                .gated_delta_net_contract(layer.layer_idx, 1)
                .unwrap()
                .norm_and_z_gate_trace_binding_contract()
                .unwrap();
            let layer_offset = layer.layer_idx as u64;
            let table_log_size = 16u32;
            let trace_checksum =
                stwo::core::fields::m31::M31::from(120_000u32 + layer.layer_idx as u32);
            let table_commitment = FieldElement::from(121_000u64 + layer_offset);
            let attended_value_commitment = FieldElement::from(122_000u64 + layer_offset);
            let norm_weight_commitment = FieldElement::from(123_000u64 + layer_offset);
            let z_gate_commitment = FieldElement::from(124_000u64 + layer_offset);
            let output_commitment = FieldElement::from(125_000u64 + layer_offset);
            let statement = Qwen35NormAndZGateStatement {
                layer_idx: binding.layer_idx,
                seq_len: binding.seq_len,
                value_heads: binding.width / binding.norm_width,
                head_dim: binding.norm_width,
                table_log_size,
                trace_checksum,
                table_commitment,
                attended_value_commitment,
                norm_weight_commitment,
                z_gate_commitment,
                output_commitment,
                statement_hash: binding.expected_statement_hash(
                    table_log_size,
                    trace_checksum,
                    table_commitment,
                    attended_value_commitment,
                    norm_weight_commitment,
                    z_gate_commitment,
                    output_commitment,
                ),
            };
            ledger
                .record_norm_and_z_gate_proof(&binding, &statement)
                .unwrap();
        }

        assert_eq!(
            ledger.count_kind(Qwen35TypedProofStatementKind::NormAndZGate),
            30
        );
        assert!(ledger.norm_and_z_gate_coverage_complete());
        assert!(ledger.production_ready());

        let mut reordered = ledger.clone();
        reordered.statements.reverse();
        assert!(reordered.production_ready());
        assert_eq!(ledger.ledger_hash(), reordered.ledger_hash());

        let mut duplicate_depthwise = ledger.clone();
        let duplicate = duplicate_depthwise
            .statements
            .iter()
            .find(|statement| statement.kind == Qwen35TypedProofStatementKind::DepthwiseConv1d)
            .unwrap()
            .clone();
        duplicate_depthwise.statements.push(duplicate);
        assert!(!duplicate_depthwise.depthwise_conv1d_coverage_complete());
        assert!(!duplicate_depthwise.production_ready());
        assert!(duplicate_depthwise
            .validate_production_ready()
            .unwrap_err()
            .contains("duplicate DepthwiseConv1D statement"));

        let mut stale_aggregate = ledger.clone();
        let delta = stale_aggregate
            .statements
            .iter_mut()
            .find(|statement| statement.kind == Qwen35TypedProofStatementKind::DeltaRecurrence)
            .unwrap();
        delta.transform_nonlinear_beta_sigmoid_statement_hash =
            Some(delta.statement_hash + FieldElement::from(8_000_000u64));
        assert!(!stale_aggregate.delta_recurrence_coverage_complete());
        assert!(!stale_aggregate.production_ready());

        let mut extra_invalid_delta = ledger.clone();
        let mut extra = extra_invalid_delta
            .statements
            .iter()
            .find(|statement| statement.kind == Qwen35TypedProofStatementKind::DeltaRecurrence)
            .unwrap()
            .clone();
        extra.stage_idx += 10_000;
        extra.transform_nonlinear_statement_hash = None;
        extra_invalid_delta.statements.push(extra);
        assert!(!extra_invalid_delta.delta_recurrence_coverage_complete());
        assert!(!extra_invalid_delta.production_ready());

        let mut duplicate_delta = ledger.clone();
        let duplicate = duplicate_delta
            .statements
            .iter()
            .find(|statement| statement.kind == Qwen35TypedProofStatementKind::DeltaRecurrence)
            .unwrap()
            .clone();
        duplicate_delta.statements.push(duplicate);
        assert!(!duplicate_delta.delta_recurrence_coverage_complete());
        assert!(!duplicate_delta.production_ready());
        assert!(duplicate_delta
            .validate_production_ready()
            .unwrap_err()
            .contains("duplicate DeltaRecurrence statement"));

        let mut duplicate_norm = ledger.clone();
        let duplicate = duplicate_norm
            .statements
            .iter()
            .find(|statement| statement.kind == Qwen35TypedProofStatementKind::NormAndZGate)
            .unwrap()
            .clone();
        duplicate_norm.statements.push(duplicate);
        assert!(!duplicate_norm.norm_and_z_gate_coverage_complete());
        assert!(!duplicate_norm.production_ready());
        assert!(duplicate_norm
            .validate_production_ready()
            .unwrap_err()
            .contains("duplicate NormAndZGate statement"));
    }

    #[test]
    fn delta_recurrence_trace_binding_is_statement_bound_and_relabel_safe() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let binding = plan
            .gated_delta_net_contract(0, 13)
            .unwrap()
            .delta_recurrence_trace_binding_contract()
            .unwrap();

        assert_eq!(binding.layer_idx, 0);
        assert_eq!(binding.seq_len, 13);
        assert_eq!(binding.stage_idx, 5);
        assert_eq!(binding.qkv_split_stage_idx, 2);
        assert_eq!(binding.ab_projection_stage_idx, 4);
        assert_eq!(binding.consumer_stage_idx, 6);
        assert_eq!(
            binding.query_width,
            plan.linear_num_value_heads * plan.linear_key_head_dim
        );
        assert_eq!(binding.key_width, binding.query_width);
        assert_eq!(binding.value_width, plan.linear_value_rows);
        assert_eq!(binding.state_rows, plan.linear_state_rows);
        assert_eq!(
            binding.value_head_dim,
            plan.linear_value_rows / plan.linear_state_rows
        );
        assert_eq!(binding.query_root.name, "query");
        assert_eq!(binding.output_root.name, "attended_value");

        let query_commitment = FieldElement::from(41u64);
        let key_commitment = FieldElement::from(42u64);
        let projected_value_commitment = FieldElement::from(43u64);
        let a_gate_commitment = FieldElement::from(44u64);
        let b_gate_commitment = FieldElement::from(45u64);
        let a_log_weight_commitment = FieldElement::from(46u64);
        let dt_bias_commitment = FieldElement::from(47u64);
        let mode_tag = 1u64;
        let air_spec_hash = FieldElement::from(48u64);
        let initial_recurrent_state_commitment = FieldElement::from(49u64);
        let final_recurrent_state_commitment = FieldElement::from(50u64);
        let output_commitment = FieldElement::from(51u64);
        let statement_hash = binding.expected_statement_hash(
            query_commitment,
            key_commitment,
            projected_value_commitment,
            a_gate_commitment,
            b_gate_commitment,
            a_log_weight_commitment,
            dt_bias_commitment,
            mode_tag,
            air_spec_hash,
            initial_recurrent_state_commitment,
            final_recurrent_state_commitment,
            output_commitment,
        );
        binding
            .validate_statement(
                0,
                13,
                binding.query_width,
                binding.key_width,
                binding.value_width,
                binding.state_rows,
                binding.value_head_dim,
                query_commitment,
                key_commitment,
                projected_value_commitment,
                a_gate_commitment,
                b_gate_commitment,
                a_log_weight_commitment,
                dt_bias_commitment,
                mode_tag,
                air_spec_hash,
                initial_recurrent_state_commitment,
                final_recurrent_state_commitment,
                output_commitment,
                statement_hash,
            )
            .unwrap();

        assert!(binding
            .validate_statement(
                1,
                13,
                binding.query_width,
                binding.key_width,
                binding.value_width,
                binding.state_rows,
                binding.value_head_dim,
                query_commitment,
                key_commitment,
                projected_value_commitment,
                a_gate_commitment,
                b_gate_commitment,
                a_log_weight_commitment,
                dt_bias_commitment,
                mode_tag,
                air_spec_hash,
                initial_recurrent_state_commitment,
                final_recurrent_state_commitment,
                output_commitment,
                statement_hash,
            )
            .unwrap_err()
            .contains("statement layer"));
        assert!(binding
            .validate_statement(
                0,
                13,
                binding.query_width,
                binding.key_width,
                binding.value_width,
                binding.state_rows,
                binding.value_head_dim,
                query_commitment,
                key_commitment,
                projected_value_commitment,
                a_gate_commitment,
                b_gate_commitment,
                a_log_weight_commitment,
                dt_bias_commitment,
                mode_tag,
                air_spec_hash,
                initial_recurrent_state_commitment,
                final_recurrent_state_commitment,
                output_commitment + FieldElement::ONE,
                statement_hash,
            )
            .unwrap_err()
            .contains("statement hash"));

        let seq14_binding = plan
            .gated_delta_net_contract(0, 14)
            .unwrap()
            .delta_recurrence_trace_binding_contract()
            .unwrap();
        let layer1_binding = plan
            .gated_delta_net_contract(1, 13)
            .unwrap()
            .delta_recurrence_trace_binding_contract()
            .unwrap();
        assert_ne!(binding.contract_hash(), seq14_binding.contract_hash());
        assert_ne!(binding.contract_hash(), layer1_binding.contract_hash());
    }

    #[test]
    fn norm_and_z_gate_trace_binding_is_statement_bound_and_relabel_safe() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let binding = plan
            .gated_delta_net_contract(0, 13)
            .unwrap()
            .norm_and_z_gate_trace_binding_contract()
            .unwrap();

        assert_eq!(binding.layer_idx, 0);
        assert_eq!(binding.seq_len, 13);
        assert_eq!(binding.width, 4096);
        assert_eq!(binding.norm_width, 128);
        assert_eq!(binding.stage_idx, 6);
        assert_eq!(binding.delta_recurrence_stage_idx, 5);
        assert_eq!(binding.z_projection_stage_idx, 3);
        assert_eq!(binding.consumer_stage_idx, 7);
        assert_eq!(binding.attended_value_root.name, "attended_value");
        assert_eq!(
            binding.attended_value_root.role,
            Qwen35TraceRootRole::ProducerActivation
        );
        assert_eq!(binding.norm_weight_root.name, "norm_weight");
        assert_eq!(
            binding.norm_weight_root.role,
            Qwen35TraceRootRole::ModelWeight
        );
        assert_eq!(binding.z_gate_root.name, "z_gate");
        assert_eq!(
            binding.z_gate_root.role,
            Qwen35TraceRootRole::ProducerActivation
        );
        assert_eq!(binding.output_root.name, "gated_value");
        assert_eq!(
            binding.output_root.role,
            Qwen35TraceRootRole::ConsumerActivation
        );

        let attended_value_commitment = FieldElement::from(61u64);
        let norm_weight_commitment = FieldElement::from(62u64);
        let z_gate_commitment = FieldElement::from(63u64);
        let output_commitment = FieldElement::from(64u64);
        let table_log_size = 16u32;
        let trace_checksum = stwo::core::fields::m31::M31::from(65u32);
        let table_commitment = FieldElement::from(66u64);
        let statement_hash = binding.expected_statement_hash(
            table_log_size,
            trace_checksum,
            table_commitment,
            attended_value_commitment,
            norm_weight_commitment,
            z_gate_commitment,
            output_commitment,
        );

        binding
            .validate_statement(
                0,
                13,
                binding.width / binding.norm_width,
                binding.norm_width,
                table_log_size,
                trace_checksum,
                table_commitment,
                attended_value_commitment,
                norm_weight_commitment,
                z_gate_commitment,
                output_commitment,
                statement_hash,
            )
            .unwrap();
        assert!(binding
            .validate_statement(
                1,
                13,
                binding.width / binding.norm_width,
                binding.norm_width,
                table_log_size,
                trace_checksum,
                table_commitment,
                attended_value_commitment,
                norm_weight_commitment,
                z_gate_commitment,
                output_commitment,
                statement_hash,
            )
            .unwrap_err()
            .contains("statement layer"));
        assert!(binding
            .validate_statement(
                0,
                13,
                binding.width / binding.norm_width,
                binding.norm_width,
                table_log_size,
                trace_checksum,
                table_commitment,
                attended_value_commitment,
                norm_weight_commitment,
                z_gate_commitment,
                output_commitment + FieldElement::ONE,
                statement_hash,
            )
            .unwrap_err()
            .contains("statement hash"));

        let seq14_binding = plan
            .gated_delta_net_contract(0, 14)
            .unwrap()
            .norm_and_z_gate_trace_binding_contract()
            .unwrap();
        let layer1_binding = plan
            .gated_delta_net_contract(1, 13)
            .unwrap()
            .norm_and_z_gate_trace_binding_contract()
            .unwrap();
        assert_ne!(binding.contract_hash(), seq14_binding.contract_hash());
        assert_ne!(binding.contract_hash(), layer1_binding.contract_hash());
    }

    #[test]
    fn typed_proof_ledger_consumes_norm_and_z_gate_active_air_proof() {
        use crate::components::matmul::M31Matrix;
        use crate::components::qwen35_norm_and_z_gate::prove_qwen35_norm_and_z_gate_air;
        use stwo::core::fields::m31::M31;

        fn filled(rows: usize, cols: usize, value: u32) -> M31Matrix {
            M31Matrix {
                rows,
                cols,
                data: vec![M31::from(value); rows * cols],
            }
        }

        fn matrix_root(
            name: &'static str,
            role: Qwen35TraceRootRole,
            rows: usize,
            cols: usize,
        ) -> Qwen35TraceRootContract {
            Qwen35TraceRootContract {
                name,
                role,
                shape: Qwen35TensorShape::Matrix(Qwen35Tensor2DShape { rows, cols }),
            }
        }

        let attended_value = filled(2, 4, 1);
        let norm_weight = vec![M31::from(2u32), M31::from(3u32)];
        let z_gate = filled(2, 4, 4);
        let proof =
            prove_qwen35_norm_and_z_gate_air(3, &attended_value, &norm_weight, &z_gate, 2, 4)
                .unwrap();
        let binding = Qwen35NormAndZGateTraceBindingContract {
            layer_idx: 3,
            seq_len: 2,
            width: 4,
            norm_width: 2,
            stage_idx: 6,
            delta_recurrence_stage_idx: 5,
            z_projection_stage_idx: 3,
            consumer_stage_idx: 7,
            attended_value_root: matrix_root(
                "attended_value",
                Qwen35TraceRootRole::ProducerActivation,
                2,
                4,
            ),
            norm_weight_root: Qwen35TraceRootContract {
                name: "norm_weight",
                role: Qwen35TraceRootRole::ModelWeight,
                shape: Qwen35TensorShape::Vector(2),
            },
            z_gate_root: matrix_root("z_gate", Qwen35TraceRootRole::ProducerActivation, 2, 4),
            output_root: matrix_root("gated_value", Qwen35TraceRootRole::ConsumerActivation, 2, 4),
            stage_contract_hash: FieldElement::from(99u64),
        };

        let mut ledger = Qwen35TypedProofLedger::new(FieldElement::from(7u64), 2, 0, 0, 1);
        ledger
            .record_norm_and_z_gate_air_proof(&binding, &proof)
            .unwrap();
        assert_eq!(
            ledger.count_kind(Qwen35TypedProofStatementKind::NormAndZGate),
            1
        );
        assert!(ledger.norm_and_z_gate_coverage_complete());
        assert!(ledger.production_ready());

        let duplicate_err = ledger
            .record_norm_and_z_gate_air_proof(&binding, &proof)
            .unwrap_err();
        assert!(duplicate_err.contains("duplicate NormAndZGate"));

        let mut relabeled_binding = binding.clone();
        relabeled_binding.layer_idx = 4;
        let relabel_err = Qwen35TypedProofLedger::new(FieldElement::from(7u64), 2, 0, 0, 1)
            .record_norm_and_z_gate_air_proof(&relabeled_binding, &proof)
            .unwrap_err();
        assert!(relabel_err.contains("statement layer"));
    }

    #[test]
    fn typed_proof_ledger_consumes_delta_recurrence_statements_from_active_proofs() {
        use crate::components::qwen35_delta_recurrence::{
            Qwen35DeltaRecurrenceMode, Qwen35DeltaRecurrenceStatement,
        };

        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let binding = plan
            .gated_delta_net_contract(0, 1)
            .unwrap()
            .delta_recurrence_trace_binding_contract()
            .unwrap();
        let mut ledger = plan.typed_proof_ledger(1);
        let statement = Qwen35DeltaRecurrenceStatement {
            layer_idx: binding.layer_idx,
            seq_len: binding.seq_len,
            query_width: binding.query_width,
            key_width: binding.key_width,
            value_width: binding.value_width,
            state_rows: binding.state_rows,
            value_head_dim: binding.value_head_dim,
            mode: Qwen35DeltaRecurrenceMode::RecurrentDecode,
            air_spec_hash: FieldElement::from(200u64),
            query_commitment: FieldElement::from(201u64),
            key_commitment: FieldElement::from(202u64),
            projected_value_commitment: FieldElement::from(203u64),
            a_gate_commitment: FieldElement::from(204u64),
            b_gate_commitment: FieldElement::from(205u64),
            a_log_weight_commitment: FieldElement::from(206u64),
            dt_bias_commitment: FieldElement::from(207u64),
            initial_recurrent_state_commitment: FieldElement::from(208u64),
            final_recurrent_state_commitment: FieldElement::from(209u64),
            output_commitment: FieldElement::from(210u64),
            statement_hash: binding.expected_statement_hash(
                FieldElement::from(201u64),
                FieldElement::from(202u64),
                FieldElement::from(203u64),
                FieldElement::from(204u64),
                FieldElement::from(205u64),
                FieldElement::from(206u64),
                FieldElement::from(207u64),
                Qwen35DeltaRecurrenceMode::RecurrentDecode.as_u64(),
                FieldElement::from(200u64),
                FieldElement::from(208u64),
                FieldElement::from(209u64),
                FieldElement::from(210u64),
            ),
        };

        ledger
            .record_delta_recurrence_proof(&binding, &statement)
            .unwrap();
        assert_eq!(
            ledger.count_kind(Qwen35TypedProofStatementKind::DeltaRecurrence),
            1
        );
        assert!(ledger
            .record_delta_recurrence_proof(&binding, &statement)
            .unwrap_err()
            .contains("duplicate"));

        let mut relabeled = statement.clone();
        relabeled.layer_idx = 1;
        relabeled.statement_hash = binding.expected_statement_hash(
            relabeled.query_commitment,
            relabeled.key_commitment,
            relabeled.projected_value_commitment,
            relabeled.a_gate_commitment,
            relabeled.b_gate_commitment,
            relabeled.a_log_weight_commitment,
            relabeled.dt_bias_commitment,
            relabeled.mode.as_u64(),
            relabeled.air_spec_hash,
            relabeled.initial_recurrent_state_commitment,
            relabeled.final_recurrent_state_commitment,
            relabeled.output_commitment,
        );
        let layer_err = plan
            .typed_proof_ledger(1)
            .record_delta_recurrence_proof(&binding, &relabeled)
            .unwrap_err();
        assert!(layer_err.contains("statement layer"));

        let mut bad_hash = statement.clone();
        bad_hash.statement_hash += FieldElement::ONE;
        let hash_err = plan
            .typed_proof_ledger(1)
            .record_delta_recurrence_proof(&binding, &bad_hash)
            .unwrap_err();
        assert!(hash_err.contains("statement hash"));
    }

    #[test]
    fn typed_proof_ledger_consumes_delta_recurrence_trace_binding_witness() {
        use crate::components::matmul::M31Matrix;
        use crate::components::qwen35_delta_recurrence::{
            prove_qwen35_delta_recurrence_arithmetic_air,
            prove_qwen35_delta_recurrence_beta_sigmoid_air,
            prove_qwen35_delta_recurrence_decay_air, prove_qwen35_delta_recurrence_norm_air,
            prove_qwen35_delta_recurrence_trace_binding_air,
            prove_qwen35_delta_recurrence_transform_binding_air,
            qwen35_delta_recurrence_arithmetic_witness,
            qwen35_delta_recurrence_trace_binding_witness,
            qwen35_delta_recurrence_transform_binding_witness,
            Qwen35DeltaRecurrenceArithmeticInputs, Qwen35DeltaRecurrenceInputs,
            Qwen35DeltaRecurrenceMode, Qwen35DeltaRecurrenceNormKind,
            Qwen35DeltaRecurrenceTransformInputs,
        };
        use stwo::core::fields::m31::M31;

        fn matrix(rows: usize, cols: usize, seed: u32) -> M31Matrix {
            let mut matrix = M31Matrix::new(rows, cols);
            for row in 0..rows {
                for col in 0..cols {
                    matrix.set(row, col, M31::from(seed + (row * cols + col) as u32));
                }
            }
            matrix
        }

        fn norm_output(
            input: &M31Matrix,
            state_rows: usize,
            table_log_size: u32,
            post_scale: M31,
        ) -> M31Matrix {
            let qk_head_dim = input.cols / state_rows;
            let table = crate::components::rmsnorm::build_rsqrt_table(table_log_size);
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

        fn decay_output(
            a_gate: &M31Matrix,
            a_log_weight: &[M31],
            dt_bias: &[M31],
            table_log_size: u32,
        ) -> M31Matrix {
            let softplus_table = crate::gadgets::lookup_table::PrecomputedTable::build(
                crate::gadgets::lookup_table::activations::softplus_approx,
                table_log_size,
            );
            let exp_table = crate::gadgets::lookup_table::PrecomputedTable::build(
                crate::gadgets::lookup_table::activations::softmax_exp,
                table_log_size,
            );
            let decay_table = crate::gadgets::lookup_table::PrecomputedTable::build(
                |x| crate::gadgets::lookup_table::activations::softmax_exp(M31::from(0u32) - x),
                table_log_size,
            );
            let mut decay = M31Matrix::new(a_gate.rows, a_gate.cols);
            for token_idx in 0..a_gate.rows {
                for state_row_idx in 0..a_gate.cols {
                    let a_sum = a_gate.get(token_idx, state_row_idx) + dt_bias[state_row_idx];
                    let softplus = softplus_table.lookup(a_sum).unwrap();
                    let exp_a_log = exp_table.lookup(a_log_weight[state_row_idx]).unwrap();
                    let product =
                        M31::from(((exp_a_log.0 as u64 * softplus.0 as u64) >> 16) as u32);
                    decay.set(
                        token_idx,
                        state_row_idx,
                        decay_table.lookup(product).unwrap(),
                    );
                }
            }
            decay
        }

        fn root(name: &'static str, rows: usize, cols: usize) -> Qwen35TraceRootContract {
            Qwen35TraceRootContract {
                name,
                role: Qwen35TraceRootRole::ProducerActivation,
                shape: Qwen35TensorShape::Matrix(Qwen35Tensor2DShape { rows, cols }),
            }
        }

        let binding = Qwen35DeltaRecurrenceTraceBindingContract {
            layer_idx: 0,
            seq_len: 2,
            query_width: 4,
            key_width: 4,
            value_width: 6,
            state_rows: 2,
            value_head_dim: 3,
            stage_idx: 5,
            qkv_split_stage_idx: 2,
            ab_projection_stage_idx: 4,
            consumer_stage_idx: 6,
            query_root: root("query", 2, 4),
            key_root: root("key", 2, 4),
            projected_value_root: root("projected_value", 2, 6),
            a_gate_root: root("a_gate", 2, 2),
            b_gate_root: root("b_gate", 2, 2),
            a_log_weight_root: Qwen35TraceRootContract {
                name: "a_log_weight",
                role: Qwen35TraceRootRole::ModelWeight,
                shape: Qwen35TensorShape::Vector(2),
            },
            dt_bias_root: Qwen35TraceRootContract {
                name: "dt_bias",
                role: Qwen35TraceRootRole::ModelWeight,
                shape: Qwen35TensorShape::Vector(2),
            },
            output_root: Qwen35TraceRootContract {
                name: "attended_value",
                role: Qwen35TraceRootRole::ConsumerActivation,
                shape: Qwen35TensorShape::Matrix(Qwen35Tensor2DShape { rows: 2, cols: 6 }),
            },
            stage_contract_hash: FieldElement::from(99u64),
        };
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
        let mut ledger = Qwen35TypedProofLedger::new(FieldElement::from(7u64), 2, 0, 1, 0);
        ledger
            .record_delta_recurrence_trace_binding_witness(&binding, &inputs, &witness)
            .unwrap();
        assert_eq!(
            ledger.count_kind(Qwen35TypedProofStatementKind::DeltaRecurrence),
            1
        );

        let mut swapped = witness.clone();
        swapped.rows.swap(1, 2);
        let err = Qwen35TypedProofLedger::new(FieldElement::from(7u64), 2, 0, 1, 0)
            .record_delta_recurrence_trace_binding_witness(&binding, &inputs, &swapped)
            .unwrap_err();
        assert!(err.contains("expected token") || err.contains("witness hash mismatch"));

        let proof =
            prove_qwen35_delta_recurrence_trace_binding_air(binding.layer_idx, &inputs).unwrap();
        let mut proof_ledger = Qwen35TypedProofLedger::new(FieldElement::from(7u64), 2, 0, 1, 0);
        proof_ledger
            .record_delta_recurrence_trace_binding_air_proof(&binding, &proof)
            .unwrap();
        assert_eq!(
            proof_ledger.count_kind(Qwen35TypedProofStatementKind::DeltaRecurrence),
            1
        );

        let mut wrong_binding = binding.clone();
        wrong_binding.layer_idx = 1;
        let err = Qwen35TypedProofLedger::new(FieldElement::from(7u64), 2, 0, 1, 0)
            .record_delta_recurrence_trace_binding_air_proof(&wrong_binding, &proof)
            .unwrap_err();
        assert!(err.contains("statement layer"));

        let norm_table_log_size = 12;
        let norm_post_scale = M31::from(1u32);
        let scaled_query = norm_output(&query, 2, norm_table_log_size, norm_post_scale);
        let normalized_key = norm_output(&key, 2, norm_table_log_size, M31::from(1u32));
        let arithmetic_projected_value = matrix(2, 6, 11);
        let decay_table_log_size = 16;
        let decay = decay_output(&a_gate, &a_log_weight, &dt_bias, decay_table_log_size);
        let mut beta = M31Matrix::new(2, 2);
        for row in 0..2 {
            for col in 0..2 {
                beta.set(
                    row,
                    col,
                    crate::gadgets::lookup_table::activations::sigmoid_approx(b_gate.get(row, col)),
                );
            }
        }
        let arithmetic_initial_state = matrix(4, 3, 13);
        let placeholder_final_state = matrix(4, 3, 0);
        let placeholder_output = matrix(2, 6, 0);
        let draft_arithmetic_inputs = Qwen35DeltaRecurrenceArithmeticInputs {
            scaled_query: &scaled_query,
            normalized_key: &normalized_key,
            projected_value: &arithmetic_projected_value,
            decay: &decay,
            beta: &beta,
            initial_recurrent_state: &arithmetic_initial_state,
            final_recurrent_state: &placeholder_final_state,
            output: &placeholder_output,
            state_rows: 2,
            value_head_dim: 3,
        };
        let arithmetic_witness =
            qwen35_delta_recurrence_arithmetic_witness(&draft_arithmetic_inputs).unwrap();
        let arithmetic_final_state = arithmetic_witness.final_recurrent_state.clone();
        let arithmetic_output = arithmetic_witness.output.clone();
        let arithmetic_inputs = Qwen35DeltaRecurrenceArithmeticInputs {
            final_recurrent_state: &arithmetic_final_state,
            output: &arithmetic_output,
            ..draft_arithmetic_inputs
        };
        let trace_inputs = Qwen35DeltaRecurrenceInputs {
            query: &query,
            key: &key,
            projected_value: &arithmetic_projected_value,
            a_gate: &a_gate,
            b_gate: &b_gate,
            a_log_weight: &a_log_weight,
            dt_bias: &dt_bias,
            initial_recurrent_state: &arithmetic_initial_state,
            final_recurrent_state: &arithmetic_final_state,
            output: &arithmetic_output,
            state_rows: 2,
            value_head_dim: 3,
            mode: Qwen35DeltaRecurrenceMode::RecurrentDecode,
        };
        let active_trace_proof =
            prove_qwen35_delta_recurrence_trace_binding_air(binding.layer_idx, &trace_inputs)
                .unwrap();
        let active_arithmetic_proof = prove_qwen35_delta_recurrence_arithmetic_air(
            binding.layer_idx,
            Qwen35DeltaRecurrenceMode::RecurrentDecode,
            &arithmetic_inputs,
        )
        .unwrap();
        let mut active_ledger = Qwen35TypedProofLedger::new(FieldElement::from(7u64), 2, 0, 1, 0);
        active_ledger
            .record_delta_recurrence_active_air_proofs(
                &binding,
                &active_trace_proof,
                &active_arithmetic_proof,
            )
            .unwrap();
        assert!(!active_ledger.delta_recurrence_coverage_complete());
        assert!(!active_ledger.production_ready());
        assert!(active_ledger.statements[0]
            .arithmetic_statement_hash
            .is_some());
        assert!(active_ledger.statements[0]
            .transform_statement_hash
            .is_none());

        let transform_inputs = Qwen35DeltaRecurrenceTransformInputs {
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
        let transform_witness =
            qwen35_delta_recurrence_transform_binding_witness(&transform_inputs).unwrap();
        let transform_proof = prove_qwen35_delta_recurrence_transform_binding_air(
            binding.layer_idx,
            &transform_inputs,
        )
        .unwrap();
        active_ledger
            .record_delta_recurrence_transform_binding_air_proof(
                &binding,
                &active_trace_proof.statement,
                &active_arithmetic_proof.statement,
                &transform_proof,
            )
            .unwrap();
        assert!(!active_ledger.delta_recurrence_coverage_complete());
        assert!(!active_ledger.production_ready());
        assert!(active_ledger.statements[0]
            .transform_statement_hash
            .is_some());
        assert!(active_ledger.statements[0]
            .transform_nonlinear_statement_hash
            .is_none());
        assert!(active_ledger.statements[0]
            .transform_nonlinear_q_norm_statement_hash
            .is_none());
        assert!(active_ledger.statements[0]
            .transform_nonlinear_k_norm_statement_hash
            .is_none());
        assert!(active_ledger.statements[0]
            .transform_nonlinear_beta_sigmoid_statement_hash
            .is_none());

        let q_norm_proof = prove_qwen35_delta_recurrence_norm_air(
            Qwen35DeltaRecurrenceNormKind::Query,
            binding.layer_idx,
            &query,
            &scaled_query,
            2,
            norm_table_log_size,
            norm_post_scale,
        )
        .unwrap();
        active_ledger
            .record_delta_recurrence_norm_air_proof(
                &binding,
                &transform_proof.statement,
                &q_norm_proof,
            )
            .unwrap();
        assert_eq!(
            active_ledger.statements[0].transform_nonlinear_q_norm_statement_hash,
            Some(q_norm_proof.statement.statement_hash)
        );
        assert!(active_ledger.statements[0]
            .transform_nonlinear_k_norm_statement_hash
            .is_none());

        let k_norm_proof = prove_qwen35_delta_recurrence_norm_air(
            Qwen35DeltaRecurrenceNormKind::Key,
            binding.layer_idx,
            &key,
            &normalized_key,
            2,
            norm_table_log_size,
            M31::from(1u32),
        )
        .unwrap();
        active_ledger
            .record_delta_recurrence_norm_air_proof(
                &binding,
                &transform_proof.statement,
                &k_norm_proof,
            )
            .unwrap();
        assert_eq!(
            active_ledger.statements[0].transform_nonlinear_k_norm_statement_hash,
            Some(k_norm_proof.statement.statement_hash)
        );

        let beta_sigmoid_proof =
            prove_qwen35_delta_recurrence_beta_sigmoid_air(binding.layer_idx, &b_gate, &beta, 6)
                .unwrap();
        active_ledger
            .record_delta_recurrence_beta_sigmoid_air_proof(
                &binding,
                &transform_proof.statement,
                &beta_sigmoid_proof,
            )
            .unwrap();
        assert!(!active_ledger.delta_recurrence_coverage_complete());
        assert!(!active_ledger.production_ready());
        assert_eq!(
            active_ledger.statements[0].transform_nonlinear_beta_sigmoid_statement_hash,
            Some(beta_sigmoid_proof.statement.statement_hash)
        );
        assert!(active_ledger.statements[0]
            .transform_nonlinear_decay_statement_hash
            .is_none());
        let finalize_err = active_ledger
            .finalize_delta_recurrence_nonlinear_transform(&binding, &transform_proof.statement)
            .unwrap_err();
        assert!(finalize_err.contains("decay"));

        let decay_proof = prove_qwen35_delta_recurrence_decay_air(
            binding.layer_idx,
            &a_gate,
            &a_log_weight,
            &dt_bias,
            &decay,
            decay_table_log_size,
        )
        .unwrap();
        let mut wrong_decay_commitment_statement = transform_proof.statement.clone();
        wrong_decay_commitment_statement.decay_commitment = FieldElement::from(123_456u64);
        let err = active_ledger
            .record_delta_recurrence_decay_air_proof(
                &binding,
                &wrong_decay_commitment_statement,
                &decay_proof,
            )
            .unwrap_err();
        assert!(err.contains("decay commitments"));

        let mut unrecorded_transform_statement = transform_proof.statement.clone();
        unrecorded_transform_statement.statement_hash = FieldElement::from(654_321u64);
        let err = active_ledger
            .record_delta_recurrence_decay_air_proof(
                &binding,
                &unrecorded_transform_statement,
                &decay_proof,
            )
            .unwrap_err();
        assert!(err.contains("matching transform-binding proof"));

        active_ledger
            .record_delta_recurrence_decay_air_proof(
                &binding,
                &transform_proof.statement,
                &decay_proof,
            )
            .unwrap();
        assert_eq!(
            active_ledger.statements[0].transform_nonlinear_decay_statement_hash,
            Some(decay_proof.statement.statement_hash)
        );
        let aggregate_hash = active_ledger
            .finalize_delta_recurrence_nonlinear_transform(&binding, &transform_proof.statement)
            .unwrap();
        assert_eq!(
            active_ledger.statements[0].transform_nonlinear_statement_hash,
            Some(aggregate_hash)
        );
        assert!(active_ledger.delta_recurrence_coverage_complete());
        assert!(active_ledger.production_ready());

        let mut witness_ledger = Qwen35TypedProofLedger::new(FieldElement::from(7u64), 2, 0, 1, 0);
        witness_ledger
            .record_delta_recurrence_active_air_proofs(
                &binding,
                &active_trace_proof,
                &active_arithmetic_proof,
            )
            .unwrap();
        witness_ledger
            .record_delta_recurrence_transform_binding_witness(
                &binding,
                &active_trace_proof.statement,
                &active_arithmetic_proof.statement,
                &transform_inputs,
                &transform_witness,
            )
            .unwrap();
        assert!(!witness_ledger.delta_recurrence_coverage_complete());

        let bad_scaled_query = matrix(2, 4, 102);
        let bad_transform_inputs = Qwen35DeltaRecurrenceTransformInputs {
            scaled_query: &bad_scaled_query,
            ..transform_inputs.clone()
        };
        let bad_transform_proof = prove_qwen35_delta_recurrence_transform_binding_air(
            binding.layer_idx,
            &bad_transform_inputs,
        )
        .unwrap();
        let mut bad_transform_ledger =
            Qwen35TypedProofLedger::new(FieldElement::from(7u64), 2, 0, 1, 0);
        bad_transform_ledger
            .record_delta_recurrence_active_air_proofs(
                &binding,
                &active_trace_proof,
                &active_arithmetic_proof,
            )
            .unwrap();
        let err = bad_transform_ledger
            .record_delta_recurrence_transform_binding_air_proof(
                &binding,
                &active_trace_proof.statement,
                &active_arithmetic_proof.statement,
                &bad_transform_proof,
            )
            .unwrap_err();
        assert!(err.contains("target commitments"));

        let bad_projected_value = matrix(2, 6, 12);
        let bad_draft_arithmetic_inputs = Qwen35DeltaRecurrenceArithmeticInputs {
            projected_value: &bad_projected_value,
            ..arithmetic_inputs.clone()
        };
        let bad_witness =
            qwen35_delta_recurrence_arithmetic_witness(&bad_draft_arithmetic_inputs).unwrap();
        let bad_final_state = bad_witness.final_recurrent_state.clone();
        let bad_output = bad_witness.output.clone();
        let bad_arithmetic_inputs = Qwen35DeltaRecurrenceArithmeticInputs {
            final_recurrent_state: &bad_final_state,
            output: &bad_output,
            ..bad_draft_arithmetic_inputs
        };
        let bad_arithmetic_proof = prove_qwen35_delta_recurrence_arithmetic_air(
            binding.layer_idx,
            Qwen35DeltaRecurrenceMode::RecurrentDecode,
            &bad_arithmetic_inputs,
        )
        .unwrap();
        let err = Qwen35TypedProofLedger::new(FieldElement::from(7u64), 2, 0, 1, 0)
            .record_delta_recurrence_active_air_proofs(
                &binding,
                &active_trace_proof,
                &bad_arithmetic_proof,
            )
            .unwrap_err();
        assert!(err.contains("IO/state commitments"));
    }

    #[test]
    fn conversation_state_ledger_rejects_spliced_recurrent_state_between_spans() {
        use crate::components::qwen35_delta_recurrence::Qwen35DeltaRecurrenceMode;
        use crate::components::qwen35_depthwise_conv1d::Qwen35DepthwiseConv1dStatement;
        use crate::components::qwen35_norm_and_z_gate::Qwen35NormAndZGateStatement;

        fn span_ledger(
            plan: &Qwen35ProofPlan,
            seq_len: usize,
            initial_base: u64,
            final_base: u64,
            tampered_initial_layer: Option<usize>,
            include_depthwise: bool,
        ) -> Qwen35TypedProofLedger {
            let mut ledger = plan.typed_proof_ledger(seq_len);
            for layer in &plan.layers {
                if layer.attention != Qwen35AttentionKind::GatedDeltaNet {
                    continue;
                }
                if include_depthwise {
                    let binding = plan
                        .gated_delta_net_contract(layer.layer_idx, seq_len)
                        .unwrap()
                        .depthwise_conv1d_trace_binding_contract()
                        .unwrap();
                    let layer_offset = layer.layer_idx as u64;
                    let input_commitment = FieldElement::from(1_000 + layer_offset);
                    let weight_commitment = FieldElement::from(2_000 + layer_offset);
                    let output_commitment = FieldElement::from(3_000 + layer_offset);
                    let statement = Qwen35DepthwiseConv1dStatement {
                        layer_idx: binding.layer_idx,
                        seq_len: binding.seq_len,
                        channels: binding.channels,
                        kernel: binding.kernel,
                        input_commitment,
                        weight_commitment,
                        output_commitment,
                        statement_hash: binding.expected_statement_hash(
                            input_commitment,
                            weight_commitment,
                            output_commitment,
                        ),
                    };
                    ledger
                        .record_depthwise_conv1d_proof(&binding, &statement)
                        .unwrap();
                }
                let binding = plan
                    .gated_delta_net_contract(layer.layer_idx, seq_len)
                    .unwrap()
                    .delta_recurrence_trace_binding_contract()
                    .unwrap();
                let layer_offset = layer.layer_idx as u64;
                let query = FieldElement::from(10_000 + layer_offset);
                let key = FieldElement::from(11_000 + layer_offset);
                let projected_value = FieldElement::from(12_000 + layer_offset);
                let a_gate = FieldElement::from(13_000 + layer_offset);
                let b_gate = FieldElement::from(14_000 + layer_offset);
                let a_log_weight = FieldElement::from(15_000 + layer_offset);
                let dt_bias = FieldElement::from(16_000 + layer_offset);
                let air_spec_hash = FieldElement::from(17_000 + layer_offset);
                let initial_recurrent_state = if tampered_initial_layer == Some(layer.layer_idx) {
                    FieldElement::from(99_999u64)
                } else {
                    FieldElement::from(initial_base + layer_offset)
                };
                let final_recurrent_state = FieldElement::from(final_base + layer_offset);
                let output = FieldElement::from(18_000 + layer_offset);
                let statement_hash = binding.expected_statement_hash(
                    query,
                    key,
                    projected_value,
                    a_gate,
                    b_gate,
                    a_log_weight,
                    dt_bias,
                    Qwen35DeltaRecurrenceMode::RecurrentDecode.as_u64(),
                    air_spec_hash,
                    initial_recurrent_state,
                    final_recurrent_state,
                    output,
                );
                ledger
                    .record_delta_recurrence_statement(
                        &binding,
                        binding.layer_idx,
                        binding.seq_len,
                        binding.query_width,
                        binding.key_width,
                        binding.value_width,
                        binding.state_rows,
                        binding.value_head_dim,
                        query,
                        key,
                        projected_value,
                        a_gate,
                        b_gate,
                        a_log_weight,
                        dt_bias,
                        Qwen35DeltaRecurrenceMode::RecurrentDecode.as_u64(),
                        air_spec_hash,
                        initial_recurrent_state,
                        final_recurrent_state,
                        output,
                        statement_hash,
                    )
                    .unwrap();
                ledger
                    .statements
                    .last_mut()
                    .unwrap()
                    .transform_statement_hash =
                    Some(statement_hash + FieldElement::from(500_000u64));
                ledger
                    .statements
                    .last_mut()
                    .unwrap()
                    .transform_nonlinear_q_norm_statement_hash =
                    Some(statement_hash + FieldElement::from(751_000u64));
                ledger
                    .statements
                    .last_mut()
                    .unwrap()
                    .transform_nonlinear_k_norm_statement_hash =
                    Some(statement_hash + FieldElement::from(752_000u64));
                ledger
                    .statements
                    .last_mut()
                    .unwrap()
                    .transform_nonlinear_beta_sigmoid_statement_hash =
                    Some(statement_hash + FieldElement::from(753_000u64));
                ledger
                    .statements
                    .last_mut()
                    .unwrap()
                    .transform_nonlinear_decay_statement_hash =
                    Some(statement_hash + FieldElement::from(754_000u64));
                ledger
                    .statements
                    .last_mut()
                    .unwrap()
                    .arithmetic_statement_hash =
                    Some(statement_hash + FieldElement::from(1_000_000u64));
                let aggregate_hash = ledger
                    .statements
                    .last()
                    .unwrap()
                    .delta_recurrence_nonlinear_aggregate_hash()
                    .unwrap();
                ledger
                    .statements
                    .last_mut()
                    .unwrap()
                    .transform_nonlinear_statement_hash = Some(aggregate_hash);

                let binding = plan
                    .gated_delta_net_contract(layer.layer_idx, seq_len)
                    .unwrap()
                    .norm_and_z_gate_trace_binding_contract()
                    .unwrap();
                let table_log_size = 16u32;
                let trace_checksum =
                    stwo::core::fields::m31::M31::from(120_000u32 + layer.layer_idx as u32);
                let table_commitment = FieldElement::from(121_000 + layer_offset);
                let attended_value_commitment = FieldElement::from(122_000 + layer_offset);
                let norm_weight_commitment = FieldElement::from(123_000 + layer_offset);
                let z_gate_commitment = FieldElement::from(124_000 + layer_offset);
                let output_commitment = FieldElement::from(125_000 + layer_offset);
                let statement = Qwen35NormAndZGateStatement {
                    layer_idx: binding.layer_idx,
                    seq_len: binding.seq_len,
                    value_heads: binding.width / binding.norm_width,
                    head_dim: binding.norm_width,
                    table_log_size,
                    trace_checksum,
                    table_commitment,
                    attended_value_commitment,
                    norm_weight_commitment,
                    z_gate_commitment,
                    output_commitment,
                    statement_hash: binding.expected_statement_hash(
                        table_log_size,
                        trace_checksum,
                        table_commitment,
                        attended_value_commitment,
                        norm_weight_commitment,
                        z_gate_commitment,
                        output_commitment,
                    ),
                };
                ledger
                    .record_norm_and_z_gate_proof(&binding, &statement)
                    .unwrap();
            }
            ledger
        }

        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let partial_span0 = span_ledger(&plan, 1, 1_000, 2_000, None, false);
        let span0 = span_ledger(&plan, 1, 1_000, 2_000, None, true);
        let span1 = span_ledger(&plan, 1, 2_000, 3_000, None, true);
        let bad_span1 = span_ledger(&plan, 1, 2_000, 3_000, Some(1), true);

        let partial_err = Qwen35ConversationStateLedger::new(
            plan.architecture_contract_hash(),
            plan.linear_attention_layers(),
        )
        .record_active_span(0, &partial_span0)
        .unwrap_err();
        assert!(partial_err.contains("production-ready typed ledger"));

        let mut good_chain = Qwen35ConversationStateLedger::new(
            plan.architecture_contract_hash(),
            plan.linear_attention_layers(),
        );
        good_chain.record_active_span(0, &span0).unwrap();
        good_chain.record_active_span(1, &span1).unwrap();
        assert!(good_chain.continuity_complete());

        let mut bad_chain = Qwen35ConversationStateLedger::new(
            plan.architecture_contract_hash(),
            plan.linear_attention_layers(),
        );
        bad_chain.record_active_span(0, &span0).unwrap();
        let err = bad_chain.record_active_span(1, &bad_span1).unwrap_err();
        assert!(err.contains("conversation recurrent state mismatch"));
        assert_ne!(good_chain.ledger_hash(), bad_chain.ledger_hash());
    }

    #[test]
    fn gated_delta_net_stage_contract_hash_is_layer_and_sequence_sensitive() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let layer0_seq13 = plan.gated_delta_net_contract(0, 13).unwrap();
        let layer0_seq14 = plan.gated_delta_net_contract(0, 14).unwrap();
        let layer1_seq13 = plan.gated_delta_net_contract(1, 13).unwrap();

        assert_ne!(
            layer0_seq13.stage_contract_hash(),
            layer0_seq14.stage_contract_hash()
        );
        assert_ne!(
            layer0_seq13.stage_contract_hash(),
            layer1_seq13.stage_contract_hash()
        );
        assert_eq!(
            layer0_seq13.stage_contract_hash(),
            plan.gated_delta_net_contract(0, 13)
                .unwrap()
                .stage_contract_hash()
        );
    }

    #[test]
    fn gated_delta_net_contract_rejects_full_attention_layers() {
        let cfg = qwen35_test_config();
        let plan = Qwen35ProofPlan::from_hf_config(&cfg).unwrap();
        let err = plan.gated_delta_net_contract(3, 1).unwrap_err();
        assert!(err.contains("not GatedDeltaNet"));
    }
}
