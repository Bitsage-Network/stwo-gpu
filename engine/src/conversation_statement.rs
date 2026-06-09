//! Canonical statements for conversation/action proving.
//!
//! The STARK-in-STARK path should answer one question:
//! "what exact model, transcript, generated tokens, actions, and KV-cache
//! transitions did this recursive proof attest?"  This module defines that
//! statement as ordered Poseidon commitments over felt252 values.

use core::fmt;

use starknet_ff::FieldElement;

const DOMAIN_TEXT: u64 = 0x54585431; // "TXT1"
const DOMAIN_SEQ: u64 = 0x53455131; // "SEQ1"
const DOMAIN_STEP: u64 = 0x53544550; // "STEP"
const DOMAIN_ACTION: u64 = 0x4143544E; // "ACTN"
const DOMAIN_CONVERSATION: u64 = 0x434F4E56; // "CONV"
const DOMAIN_BATCH: u64 = 0x43424154; // "CBAT"
const DOMAIN_ML_RECEIPT: u64 = 0x4D4C5243; // "MLRC"
const DOMAIN_WEIGHT_GROUPS: u64 = 0x57475254; // "WGRT"
const GROUPED_BINDING_MARKER: u64 = 0x47525044; // "GRPD"

/// Version of the conversation/action statement schema.
pub const CONVERSATION_STATEMENT_VERSION: u64 = 1;

/// Security level expected for production STARK-in-STARK conversation proofs.
pub const PRODUCTION_SECURITY_BITS: u64 = 160;

/// One generated-token step in a conversation.
///
/// `global_step_index` fixes the step's position across the entire batch.
/// `conversation_index`, `turn_index`, and `token_index` fix the semantic
/// position inside a conversation.  The KV fields are what make row swapping
/// detectable before a proof is accepted as a coherent generation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GenerationStepStatement {
    pub global_step_index: u64,
    pub conversation_index: u64,
    pub turn_index: u64,
    pub token_index: u64,
    pub generated_token_id: u64,
    pub io_commitment: FieldElement,
    pub sampling_commitment: FieldElement,
    pub prev_kv_commitment: FieldElement,
    pub kv_commitment: FieldElement,
    pub recursive_proof_hash: FieldElement,
}

impl GenerationStepStatement {
    pub fn commitment(&self) -> FieldElement {
        starknet_crypto::poseidon_hash_many(&[
            FieldElement::from(DOMAIN_STEP),
            FieldElement::from(self.global_step_index),
            FieldElement::from(self.conversation_index),
            FieldElement::from(self.turn_index),
            FieldElement::from(self.token_index),
            FieldElement::from(self.generated_token_id),
            self.io_commitment,
            self.sampling_commitment,
            self.prev_kv_commitment,
            self.kv_commitment,
            self.recursive_proof_hash,
        ])
    }
}

/// One externally visible action/tool call bound into the conversation proof.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConversationActionStatement {
    pub conversation_index: u64,
    pub turn_index: u64,
    pub action_index: u64,
    pub action_kind_hash: FieldElement,
    pub tool_name_hash: FieldElement,
    pub input_commitment: FieldElement,
    pub output_commitment: FieldElement,
    pub policy_commitment: FieldElement,
}

impl ConversationActionStatement {
    pub fn commitment(&self) -> FieldElement {
        starknet_crypto::poseidon_hash_many(&[
            FieldElement::from(DOMAIN_ACTION),
            FieldElement::from(self.conversation_index),
            FieldElement::from(self.turn_index),
            FieldElement::from(self.action_index),
            self.action_kind_hash,
            self.tool_name_hash,
            self.input_commitment,
            self.output_commitment,
            self.policy_commitment,
        ])
    }
}

/// Per-conversation transcript and KV summary.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConversationTraceStatement {
    pub conversation_index: u64,
    pub conversation_id_hash: FieldElement,
    pub prompt_commitment: FieldElement,
    pub transcript_commitment: FieldElement,
    pub action_root: FieldElement,
    pub initial_kv_commitment: FieldElement,
    pub final_kv_commitment: FieldElement,
    pub n_turns: u64,
    pub n_prefill_tokens: u64,
    pub n_generated_tokens: u64,
    pub first_step_index: u64,
    pub n_steps: u64,
}

impl ConversationTraceStatement {
    pub fn commitment(&self) -> FieldElement {
        starknet_crypto::poseidon_hash_many(&[
            FieldElement::from(DOMAIN_CONVERSATION),
            FieldElement::from(self.conversation_index),
            self.conversation_id_hash,
            self.prompt_commitment,
            self.transcript_commitment,
            self.action_root,
            self.initial_kv_commitment,
            self.final_kv_commitment,
            FieldElement::from(self.n_turns),
            FieldElement::from(self.n_prefill_tokens),
            FieldElement::from(self.n_generated_tokens),
            FieldElement::from(self.first_step_index),
            FieldElement::from(self.n_steps),
        ])
    }
}

/// Batch-level statement emitted as the Cairo verifier output hash.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConversationBatchStatement {
    pub version: u64,
    pub model_id: FieldElement,
    pub verifier_program_hash: FieldElement,
    pub circuit_hash: FieldElement,
    pub weight_super_root: FieldElement,
    pub policy_commitment: FieldElement,
    pub tokenizer_config_hash: FieldElement,
    pub hades_commitment: FieldElement,
    pub conversation_root: FieldElement,
    pub generation_root: FieldElement,
    pub action_root: FieldElement,
    pub initial_kv_root: FieldElement,
    pub final_kv_root: FieldElement,
    pub n_conversations: u64,
    pub n_steps: u64,
    pub n_prefill_tokens: u64,
    pub n_generated_tokens: u64,
    pub security_bits: u64,
}

impl ConversationBatchStatement {
    /// The exact felt sequence that must be emitted by the recursive Cairo
    /// verifier, then packed into `VerificationOutput.output_hash`.
    pub fn to_felts(&self) -> [FieldElement; 19] {
        [
            FieldElement::from(DOMAIN_BATCH),
            FieldElement::from(self.version),
            self.model_id,
            self.verifier_program_hash,
            self.circuit_hash,
            self.weight_super_root,
            self.policy_commitment,
            self.tokenizer_config_hash,
            self.hades_commitment,
            self.conversation_root,
            self.generation_root,
            self.action_root,
            self.initial_kv_root,
            self.final_kv_root,
            FieldElement::from(self.n_conversations),
            FieldElement::from(self.n_steps),
            FieldElement::from(self.n_prefill_tokens),
            FieldElement::from(self.n_generated_tokens),
            FieldElement::from(self.security_bits),
        ]
    }

    /// Canonical statement hash.  For STARK-in-STARK this is the expected
    /// `output_hash` of the Cairo verifier execution.
    pub fn statement_hash(&self) -> FieldElement {
        starknet_crypto::poseidon_hash_many(&self.to_felts())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversationStatementError {
    EmptyConversations,
    StepRangeGap {
        conversation_index: u64,
        expected_first_step_index: u64,
        actual_first_step_index: u64,
    },
    GeneratedTokenCountMismatch {
        conversation_index: u64,
        expected: u64,
        actual: u64,
    },
    UncoveredGenerationSteps {
        covered: u64,
        total: u64,
    },
    StepIndexOutOfRange {
        conversation_index: u64,
        step_index: u64,
    },
    StepConversationMismatch {
        expected: u64,
        actual: u64,
        step_index: u64,
    },
    KvContinuityMismatch {
        conversation_index: u64,
        step_index: u64,
        expected_prev: FieldElement,
        actual_prev: FieldElement,
    },
    FinalKvMismatch {
        conversation_index: u64,
        expected_final: FieldElement,
        actual_final: FieldElement,
    },
    ActionIndexMismatch {
        conversation_index: u64,
        expected: u64,
        actual: u64,
    },
    ConversationActionRootMismatch {
        conversation_index: u64,
        expected: FieldElement,
        actual: FieldElement,
    },
    ConversationIndexMismatch {
        expected: u64,
        actual: u64,
    },
    StepGlobalIndexMismatch {
        expected: u64,
        actual: u64,
    },
    ActionConversationOutOfRange {
        conversation_index: u64,
    },
}

impl fmt::Display for ConversationStatementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyConversations => write!(f, "conversation statement has no conversations"),
            Self::StepRangeGap {
                conversation_index,
                expected_first_step_index,
                actual_first_step_index,
            } => write!(
                f,
                "conversation {conversation_index} step range starts at {actual_first_step_index}, expected {expected_first_step_index}",
            ),
            Self::GeneratedTokenCountMismatch {
                conversation_index,
                expected,
                actual,
            } => write!(
                f,
                "conversation {conversation_index} generated token count is {actual}, expected {expected}",
            ),
            Self::UncoveredGenerationSteps { covered, total } => write!(
                f,
                "conversation ranges cover {covered} generation steps, but statement has {total}",
            ),
            Self::StepIndexOutOfRange { conversation_index, step_index } => write!(
                f,
                "conversation {conversation_index} references missing step {step_index}",
            ),
            Self::StepConversationMismatch { expected, actual, step_index } => write!(
                f,
                "step {step_index} belongs to conversation {actual}, expected {expected}",
            ),
            Self::KvContinuityMismatch {
                conversation_index,
                step_index,
                expected_prev,
                actual_prev,
            } => write!(
                f,
                "conversation {conversation_index} step {step_index} prev KV mismatch: expected 0x{expected_prev:x}, got 0x{actual_prev:x}",
            ),
            Self::FinalKvMismatch {
                conversation_index,
                expected_final,
                actual_final,
            } => write!(
                f,
                "conversation {conversation_index} final KV mismatch: expected 0x{expected_final:x}, got 0x{actual_final:x}",
            ),
            Self::ActionIndexMismatch {
                conversation_index,
                expected,
                actual,
            } => write!(
                f,
                "conversation {conversation_index} action index is {actual}, expected {expected}",
            ),
            Self::ConversationActionRootMismatch {
                conversation_index,
                expected,
                actual,
            } => write!(
                f,
                "conversation {conversation_index} action root mismatch: expected 0x{expected:x}, got 0x{actual:x}",
            ),
            Self::ConversationIndexMismatch { expected, actual } => write!(
                f,
                "conversation row has index {actual}, expected contiguous index {expected}",
            ),
            Self::StepGlobalIndexMismatch { expected, actual } => write!(
                f,
                "generation step row has global_step_index {actual}, expected {expected}",
            ),
            Self::ActionConversationOutOfRange { conversation_index } => write!(
                f,
                "action references missing conversation index {conversation_index}",
            ),
        }
    }
}

impl std::error::Error for ConversationStatementError {}

/// Poseidon commitment to UTF-8 text with a caller-provided domain tag.
pub fn text_commitment(domain: u64, text: &str) -> FieldElement {
    let mut felts = vec![
        FieldElement::from(DOMAIN_TEXT),
        FieldElement::from(domain),
        FieldElement::from(text.as_bytes().len() as u64),
    ];
    for chunk in text.as_bytes().chunks(31) {
        let mut buf = [0u8; 32];
        buf[32 - chunk.len()..].copy_from_slice(chunk);
        felts.push(FieldElement::from_bytes_be(&buf).unwrap_or(FieldElement::ZERO));
    }
    starknet_crypto::poseidon_hash_many(&felts)
}

/// Ordered commitment to an already-hashed sequence.
pub fn sequence_commitment(domain: u64, commitments: &[FieldElement]) -> FieldElement {
    let mut acc = starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(DOMAIN_SEQ),
        FieldElement::from(domain),
        FieldElement::from(commitments.len() as u64),
    ]);
    for (index, commitment) in commitments.iter().enumerate() {
        acc = starknet_crypto::poseidon_hash_many(&[
            FieldElement::from(DOMAIN_SEQ),
            FieldElement::from(domain),
            acc,
            FieldElement::from(index as u64),
            *commitment,
        ]);
    }
    acc
}

pub fn generation_root(steps: &[GenerationStepStatement]) -> FieldElement {
    let commitments: Vec<_> = steps
        .iter()
        .map(GenerationStepStatement::commitment)
        .collect();
    sequence_commitment(DOMAIN_STEP, &commitments)
}

pub fn action_root(actions: &[ConversationActionStatement]) -> FieldElement {
    let commitments: Vec<_> = actions
        .iter()
        .map(ConversationActionStatement::commitment)
        .collect();
    sequence_commitment(DOMAIN_ACTION, &commitments)
}

pub fn conversation_root(conversations: &[ConversationTraceStatement]) -> FieldElement {
    let commitments: Vec<_> = conversations
        .iter()
        .map(ConversationTraceStatement::commitment)
        .collect();
    sequence_commitment(DOMAIN_CONVERSATION, &commitments)
}

pub fn kv_root(values: &[FieldElement]) -> FieldElement {
    sequence_commitment(0x4B565254, values) // "KVRT"
}

/// Hash of the Cairo `MLVerificationOutput` returned by a verified inner ML proof.
///
/// In the strict STARK-in-STARK conversation verifier, each generation step's
/// `recursive_proof_hash` field must equal this receipt hash. The outer Cairo
/// proof then attests the verifier execution, not just an opaque proof id.
pub fn ml_verification_receipt_hash(
    model_id: FieldElement,
    io_commitment: FieldElement,
    weight_commitment: FieldElement,
    num_layers: u64,
    num_matmuls: u64,
    verified: bool,
) -> FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(DOMAIN_ML_RECEIPT),
        model_id,
        io_commitment,
        weight_commitment,
        FieldElement::from(num_layers),
        FieldElement::from(num_matmuls),
        FieldElement::from(u64::from(verified)),
    ])
}

/// Validate that each conversation references a contiguous KV chain.
pub fn validate_conversation_batch(
    conversations: &[ConversationTraceStatement],
    steps: &[GenerationStepStatement],
) -> Result<(), ConversationStatementError> {
    if conversations.is_empty() {
        return Err(ConversationStatementError::EmptyConversations);
    }

    let mut covered_steps = 0;
    for (expected_conversation_index, conversation) in conversations.iter().enumerate() {
        if conversation.conversation_index != expected_conversation_index as u64 {
            return Err(ConversationStatementError::ConversationIndexMismatch {
                expected: expected_conversation_index as u64,
                actual: conversation.conversation_index,
            });
        }
        if conversation.first_step_index != covered_steps {
            return Err(ConversationStatementError::StepRangeGap {
                conversation_index: conversation.conversation_index,
                expected_first_step_index: covered_steps,
                actual_first_step_index: conversation.first_step_index,
            });
        }
        if conversation.n_generated_tokens != conversation.n_steps {
            return Err(ConversationStatementError::GeneratedTokenCountMismatch {
                conversation_index: conversation.conversation_index,
                expected: conversation.n_steps,
                actual: conversation.n_generated_tokens,
            });
        }

        let mut expected_prev = conversation.initial_kv_commitment;
        for local_offset in 0..conversation.n_steps {
            let step_pos = conversation.first_step_index + local_offset;
            let step = steps.get(step_pos as usize).ok_or(
                ConversationStatementError::StepIndexOutOfRange {
                    conversation_index: conversation.conversation_index,
                    step_index: step_pos,
                },
            )?;
            if step.global_step_index != step_pos {
                return Err(ConversationStatementError::StepGlobalIndexMismatch {
                    expected: step_pos,
                    actual: step.global_step_index,
                });
            }
            if step.conversation_index != conversation.conversation_index {
                return Err(ConversationStatementError::StepConversationMismatch {
                    expected: conversation.conversation_index,
                    actual: step.conversation_index,
                    step_index: step_pos,
                });
            }
            if step.prev_kv_commitment != expected_prev {
                return Err(ConversationStatementError::KvContinuityMismatch {
                    conversation_index: conversation.conversation_index,
                    step_index: step_pos,
                    expected_prev,
                    actual_prev: step.prev_kv_commitment,
                });
            }
            expected_prev = step.kv_commitment;
        }

        if expected_prev != conversation.final_kv_commitment {
            return Err(ConversationStatementError::FinalKvMismatch {
                conversation_index: conversation.conversation_index,
                expected_final: conversation.final_kv_commitment,
                actual_final: expected_prev,
            });
        }

        covered_steps += conversation.n_steps;
    }

    if covered_steps != steps.len() as u64 {
        return Err(ConversationStatementError::UncoveredGenerationSteps {
            covered: covered_steps,
            total: steps.len() as u64,
        });
    }

    Ok(())
}

pub fn validate_conversation_actions(
    conversations: &[ConversationTraceStatement],
    actions: &[ConversationActionStatement],
) -> Result<(), ConversationStatementError> {
    for action in actions {
        if !conversations
            .iter()
            .any(|conversation| conversation.conversation_index == action.conversation_index)
        {
            return Err(ConversationStatementError::ActionConversationOutOfRange {
                conversation_index: action.conversation_index,
            });
        }
    }

    for conversation in conversations {
        let mut conv_actions = Vec::new();
        for action in actions
            .iter()
            .filter(|action| action.conversation_index == conversation.conversation_index)
        {
            let expected = conv_actions.len() as u64;
            if action.action_index != expected {
                return Err(ConversationStatementError::ActionIndexMismatch {
                    conversation_index: conversation.conversation_index,
                    expected,
                    actual: action.action_index,
                });
            }
            conv_actions.push(action.clone());
        }

        let expected = action_root(&conv_actions);
        if conversation.action_root != expected {
            return Err(ConversationStatementError::ConversationActionRootMismatch {
                conversation_index: conversation.conversation_index,
                expected,
                actual: conversation.action_root,
            });
        }
    }

    Ok(())
}

/// Build a batch statement after validating KV continuity.
#[allow(clippy::too_many_arguments)]
pub fn build_conversation_batch_statement(
    model_id: FieldElement,
    verifier_program_hash: FieldElement,
    circuit_hash: FieldElement,
    weight_super_root: FieldElement,
    policy_commitment: FieldElement,
    tokenizer_config_hash: FieldElement,
    hades_commitment: FieldElement,
    conversations: &[ConversationTraceStatement],
    steps: &[GenerationStepStatement],
    actions: &[ConversationActionStatement],
    security_bits: u64,
) -> Result<ConversationBatchStatement, ConversationStatementError> {
    validate_conversation_batch(conversations, steps)?;
    validate_conversation_actions(conversations, actions)?;

    let initial_kvs: Vec<_> = conversations
        .iter()
        .map(|c| c.initial_kv_commitment)
        .collect();
    let final_kvs: Vec<_> = conversations
        .iter()
        .map(|c| c.final_kv_commitment)
        .collect();
    let n_prefill_tokens = conversations.iter().map(|c| c.n_prefill_tokens).sum();
    let n_generated_tokens = conversations.iter().map(|c| c.n_generated_tokens).sum();

    Ok(ConversationBatchStatement {
        version: CONVERSATION_STATEMENT_VERSION,
        model_id,
        verifier_program_hash,
        circuit_hash,
        weight_super_root,
        policy_commitment,
        tokenizer_config_hash,
        hades_commitment,
        conversation_root: conversation_root(conversations),
        generation_root: generation_root(steps),
        action_root: action_root(actions),
        initial_kv_root: kv_root(&initial_kvs),
        final_kv_root: kv_root(&final_kvs),
        n_conversations: conversations.len() as u64,
        n_steps: steps.len() as u64,
        n_prefill_tokens,
        n_generated_tokens,
        security_bits,
    })
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ConversationStatementDocument {
    pub model_id: String,
    #[serde(default)]
    pub verifier_program_hash: Option<String>,
    pub circuit_hash: String,
    #[serde(default)]
    pub execution_contract_hash: Option<String>,
    pub weight_super_root: String,
    pub policy_commitment: String,
    #[serde(default)]
    pub tokenizer_config_hash: Option<String>,
    #[serde(default)]
    pub hades_commitment: Option<String>,
    #[serde(default = "default_security_bits")]
    pub security_bits: u64,
    pub conversations: Vec<ConversationDocument>,
    #[serde(default)]
    pub steps: Vec<GenerationStepDocument>,
    #[serde(default)]
    pub actions: Vec<ActionDocument>,
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ConversationDocument {
    pub conversation_id: String,
    #[serde(default)]
    pub conversation_id_hash: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub prompt_commitment: Option<String>,
    #[serde(default)]
    pub transcript: Option<String>,
    #[serde(default)]
    pub transcript_commitment: Option<String>,
    pub initial_kv_commitment: String,
    pub final_kv_commitment: String,
    #[serde(default)]
    pub n_turns: Option<u64>,
    #[serde(default)]
    pub n_prefill_tokens: Option<u64>,
    #[serde(default)]
    pub n_generated_tokens: Option<u64>,
    #[serde(default)]
    pub first_step_index: Option<u64>,
    #[serde(default)]
    pub n_steps: Option<u64>,
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct GenerationStepDocument {
    pub global_step_index: u64,
    pub conversation_index: u64,
    pub turn_index: u64,
    pub token_index: u64,
    pub generated_token_id: u64,
    pub io_commitment: String,
    #[serde(default)]
    pub sampling_commitment: Option<String>,
    pub prev_kv_commitment: String,
    pub kv_commitment: String,
    #[serde(default)]
    pub ml_receipt_hash: Option<String>,
    #[serde(default)]
    pub recursive_proof_hash: Option<String>,
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ActionDocument {
    pub conversation_index: u64,
    pub turn_index: u64,
    pub action_index: u64,
    #[serde(default)]
    pub action_kind: Option<String>,
    #[serde(default)]
    pub action_kind_hash: Option<String>,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub tool_name_hash: Option<String>,
    pub input_commitment: String,
    pub output_commitment: String,
    #[serde(default)]
    pub policy_commitment: Option<String>,
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConversationStatementArtifact {
    pub schema: String,
    pub statement_hash: String,
    pub expected_cairo_output_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_contract_hash: Option<String>,
    pub statement_felts: Vec<String>,
    pub cairo_args: Vec<String>,
    pub batch: ConversationBatchStatementJson,
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConversationBatchStatementJson {
    pub version: u64,
    pub model_id: String,
    pub verifier_program_hash: String,
    pub circuit_hash: String,
    pub weight_super_root: String,
    pub policy_commitment: String,
    pub tokenizer_config_hash: String,
    pub hades_commitment: String,
    pub conversation_root: String,
    pub generation_root: String,
    pub action_root: String,
    pub initial_kv_root: String,
    pub final_kv_root: String,
    pub n_conversations: u64,
    pub n_steps: u64,
    pub n_prefill_tokens: u64,
    pub n_generated_tokens: u64,
    pub security_bits: u64,
}

#[cfg(feature = "serde")]
fn default_security_bits() -> u64 {
    PRODUCTION_SECURITY_BITS
}

#[cfg(feature = "serde")]
pub fn build_conversation_statement_artifact_from_json_str(
    json: &str,
) -> Result<ConversationStatementArtifact, String> {
    let document: ConversationStatementDocument =
        serde_json::from_str(json).map_err(|e| format!("invalid statement JSON: {e}"))?;
    build_conversation_statement_artifact(&document)
}

#[cfg(feature = "serde")]
pub fn build_conversation_statement_artifact(
    document: &ConversationStatementDocument,
) -> Result<ConversationStatementArtifact, String> {
    if document.conversations.is_empty() {
        return Err("statement document must include at least one conversation".to_string());
    }

    let model_id = parse_felt_hex(&document.model_id, "model_id")?;
    let verifier_program_hash = parse_optional_felt_hex(
        document.verifier_program_hash.as_deref(),
        "verifier_program_hash",
    )?;
    let circuit_hash = parse_felt_hex(&document.circuit_hash, "circuit_hash")?;
    let execution_contract_hash = match document.execution_contract_hash.as_deref() {
        Some(value) => {
            let parsed = parse_felt_hex(value, "execution_contract_hash")?;
            if parsed != circuit_hash {
                return Err(format!(
                    "execution_contract_hash {} must equal circuit_hash {}; put the Qwen3.5 architecture contract hash in circuit_hash for this statement schema",
                    felt_hex(parsed),
                    felt_hex(circuit_hash),
                ));
            }
            Some(parsed)
        }
        None => None,
    };
    let weight_super_root = parse_felt_hex(&document.weight_super_root, "weight_super_root")?;
    let policy_commitment = parse_felt_hex(&document.policy_commitment, "policy_commitment")?;
    let tokenizer_config_hash = parse_optional_felt_hex(
        document.tokenizer_config_hash.as_deref(),
        "tokenizer_config_hash",
    )?;
    let hades_commitment =
        parse_optional_felt_hex(document.hades_commitment.as_deref(), "hades_commitment")?;

    let mut steps = Vec::with_capacity(document.steps.len());
    for (expected_index, step) in document.steps.iter().enumerate() {
        if step.global_step_index != expected_index as u64 {
            return Err(format!(
                "steps must be ordered by contiguous global_step_index: got {} at row {}",
                step.global_step_index, expected_index,
            ));
        }
        steps.push(GenerationStepStatement {
            global_step_index: step.global_step_index,
            conversation_index: step.conversation_index,
            turn_index: step.turn_index,
            token_index: step.token_index,
            generated_token_id: step.generated_token_id,
            io_commitment: parse_felt_hex(&step.io_commitment, "step.io_commitment")?,
            sampling_commitment: parse_optional_felt_hex(
                step.sampling_commitment.as_deref(),
                "step.sampling_commitment",
            )?,
            prev_kv_commitment: parse_felt_hex(
                &step.prev_kv_commitment,
                "step.prev_kv_commitment",
            )?,
            kv_commitment: parse_felt_hex(&step.kv_commitment, "step.kv_commitment")?,
            recursive_proof_hash: match (
                step.ml_receipt_hash.as_deref(),
                step.recursive_proof_hash.as_deref(),
            ) {
                (Some(value), _) => parse_felt_hex(value, "step.ml_receipt_hash")?,
                (None, Some(value)) => parse_felt_hex(value, "step.recursive_proof_hash")?,
                (None, None) => {
                    return Err(
                        "step must include ml_receipt_hash or recursive_proof_hash".to_string()
                    );
                }
            },
        });
    }

    let mut actions = Vec::with_capacity(document.actions.len());
    for action in &document.actions {
        let action_kind_hash = match action.action_kind_hash.as_deref() {
            Some(value) => parse_felt_hex(value, "action.action_kind_hash")?,
            None => text_commitment(0x4143544B, action.action_kind.as_deref().unwrap_or("")),
        };
        let tool_name_hash = match action.tool_name_hash.as_deref() {
            Some(value) => parse_felt_hex(value, "action.tool_name_hash")?,
            None => text_commitment(0x544F4F4C, action.tool_name.as_deref().unwrap_or("")),
        };
        actions.push(ConversationActionStatement {
            conversation_index: action.conversation_index,
            turn_index: action.turn_index,
            action_index: action.action_index,
            action_kind_hash,
            tool_name_hash,
            input_commitment: parse_felt_hex(&action.input_commitment, "action.input_commitment")?,
            output_commitment: parse_felt_hex(
                &action.output_commitment,
                "action.output_commitment",
            )?,
            policy_commitment: parse_optional_felt_hex(
                action.policy_commitment.as_deref(),
                "action.policy_commitment",
            )?
            .or_else_nonzero(policy_commitment),
        });
    }

    let mut conversations = Vec::with_capacity(document.conversations.len());
    for (index, conversation) in document.conversations.iter().enumerate() {
        let conversation_index = index as u64;
        let conversation_id_hash = match conversation.conversation_id_hash.as_deref() {
            Some(value) => parse_felt_hex(value, "conversation.conversation_id_hash")?,
            None => text_commitment(0x434944, &conversation.conversation_id),
        };
        let prompt_commitment = match conversation.prompt_commitment.as_deref() {
            Some(value) => parse_felt_hex(value, "conversation.prompt_commitment")?,
            None => text_commitment(0x50524D54, conversation.prompt.as_deref().unwrap_or("")),
        };
        let transcript_commitment = match conversation.transcript_commitment.as_deref() {
            Some(value) => parse_felt_hex(value, "conversation.transcript_commitment")?,
            None => text_commitment(0x5452414E, conversation.transcript.as_deref().unwrap_or("")),
        };

        let step_positions: Vec<u64> = steps
            .iter()
            .filter(|step| step.conversation_index == conversation_index)
            .map(|step| step.global_step_index)
            .collect();
        let first_step_index = conversation
            .first_step_index
            .or_else(|| step_positions.first().copied())
            .unwrap_or(0);
        let n_steps = conversation.n_steps.unwrap_or(step_positions.len() as u64);
        let n_generated_tokens = conversation.n_generated_tokens.unwrap_or_else(|| {
            steps
                .iter()
                .filter(|step| step.conversation_index == conversation_index)
                .count() as u64
        });
        let n_turns = conversation.n_turns.unwrap_or_else(|| {
            let max_turn = steps
                .iter()
                .filter(|step| step.conversation_index == conversation_index)
                .map(|step| step.turn_index)
                .max();
            max_turn.map_or(0, |turn| turn + 1)
        });
        let conv_actions: Vec<_> = actions
            .iter()
            .filter(|action| action.conversation_index == conversation_index)
            .cloned()
            .collect();

        conversations.push(ConversationTraceStatement {
            conversation_index,
            conversation_id_hash,
            prompt_commitment,
            transcript_commitment,
            action_root: action_root(&conv_actions),
            initial_kv_commitment: parse_felt_hex(
                &conversation.initial_kv_commitment,
                "conversation.initial_kv_commitment",
            )?,
            final_kv_commitment: parse_felt_hex(
                &conversation.final_kv_commitment,
                "conversation.final_kv_commitment",
            )?,
            n_turns,
            n_prefill_tokens: conversation.n_prefill_tokens.unwrap_or(0),
            n_generated_tokens,
            first_step_index,
            n_steps,
        });
    }

    let statement = build_conversation_batch_statement(
        model_id,
        verifier_program_hash,
        circuit_hash,
        weight_super_root,
        policy_commitment,
        tokenizer_config_hash,
        hades_commitment,
        &conversations,
        &steps,
        &actions,
        document.security_bits,
    )
    .map_err(|e| e.to_string())?;
    let statement_felts = statement.to_felts();
    let statement_hash = statement.statement_hash();

    Ok(ConversationStatementArtifact {
        schema: "obelyzk.conversation_statement_artifact.v1".to_string(),
        statement_hash: felt_hex(statement_hash),
        expected_cairo_output_hash: felt_hex(statement_hash),
        execution_contract_hash: execution_contract_hash.map(felt_hex),
        statement_felts: statement_felts.iter().copied().map(felt_hex).collect(),
        cairo_args: cairo_verifier_args(&statement, &conversations, &steps, &actions),
        batch: ConversationBatchStatementJson {
            version: statement.version,
            model_id: felt_hex(statement.model_id),
            verifier_program_hash: felt_hex(statement.verifier_program_hash),
            circuit_hash: felt_hex(statement.circuit_hash),
            weight_super_root: felt_hex(statement.weight_super_root),
            policy_commitment: felt_hex(statement.policy_commitment),
            tokenizer_config_hash: felt_hex(statement.tokenizer_config_hash),
            hades_commitment: felt_hex(statement.hades_commitment),
            conversation_root: felt_hex(statement.conversation_root),
            generation_root: felt_hex(statement.generation_root),
            action_root: felt_hex(statement.action_root),
            initial_kv_root: felt_hex(statement.initial_kv_root),
            final_kv_root: felt_hex(statement.final_kv_root),
            n_conversations: statement.n_conversations,
            n_steps: statement.n_steps,
            n_prefill_tokens: statement.n_prefill_tokens,
            n_generated_tokens: statement.n_generated_tokens,
            security_bits: statement.security_bits,
        },
    })
}

#[cfg(feature = "serde")]
fn cairo_verifier_args(
    statement: &ConversationBatchStatement,
    conversations: &[ConversationTraceStatement],
    steps: &[GenerationStepStatement],
    actions: &[ConversationActionStatement],
) -> Vec<String> {
    let mut args = vec![
        felt_hex(statement.model_id),
        felt_hex(statement.verifier_program_hash),
        felt_hex(statement.circuit_hash),
        felt_hex(statement.weight_super_root),
        felt_hex(statement.policy_commitment),
        felt_hex(statement.tokenizer_config_hash),
        felt_hex(statement.hades_commitment),
        statement.security_bits.to_string(),
        conversations.len().to_string(),
        steps.len().to_string(),
        actions.len().to_string(),
    ];

    for conversation in conversations {
        args.extend([
            conversation.conversation_index.to_string(),
            felt_hex(conversation.conversation_id_hash),
            felt_hex(conversation.prompt_commitment),
            felt_hex(conversation.transcript_commitment),
            felt_hex(conversation.action_root),
            felt_hex(conversation.initial_kv_commitment),
            felt_hex(conversation.final_kv_commitment),
            conversation.n_turns.to_string(),
            conversation.n_prefill_tokens.to_string(),
            conversation.n_generated_tokens.to_string(),
            conversation.first_step_index.to_string(),
            conversation.n_steps.to_string(),
        ]);
    }

    for step in steps {
        args.extend([
            step.global_step_index.to_string(),
            step.conversation_index.to_string(),
            step.turn_index.to_string(),
            step.token_index.to_string(),
            step.generated_token_id.to_string(),
            felt_hex(step.io_commitment),
            felt_hex(step.sampling_commitment),
            felt_hex(step.prev_kv_commitment),
            felt_hex(step.kv_commitment),
            felt_hex(step.recursive_proof_hash),
        ]);
    }

    for action in actions {
        args.extend([
            action.conversation_index.to_string(),
            action.turn_index.to_string(),
            action.action_index.to_string(),
            felt_hex(action.action_kind_hash),
            felt_hex(action.tool_name_hash),
            felt_hex(action.input_commitment),
            felt_hex(action.output_commitment),
            felt_hex(action.policy_commitment),
        ]);
    }

    args
}

/// One generation step's full GKR witness in the flattened shape expected by
/// `conversation-gkr-statement-verifier`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationGkrStepProofArgs {
    pub raw_io_data: Vec<FieldElement>,
    pub circuit_depth: u32,
    pub num_layers: u32,
    /// Expected number of weight claims emitted by Cairo after the main walk
    /// plus deferred weighted branches. Used only for Rust-side receipt checks.
    pub num_matmuls: u32,
    pub matmul_dims: Vec<u32>,
    pub dequantize_bits: Vec<u64>,
    pub proof_data: Vec<FieldElement>,
    /// Main-walk weight commitments only. The Cairo verifier reads deferred
    /// branch commitments from `proof_data` and appends them internally.
    pub weight_commitments: Vec<FieldElement>,
    pub weight_binding_root: FieldElement,
    pub weight_binding_mode: u32,
    pub weight_binding_data: Vec<FieldElement>,
    pub packed: bool,
    pub double_packed: bool,
    pub has_kv_cache: bool,
}

impl ConversationGkrStepProofArgs {
    /// Build strict GKR recursive arguments from a native proof.
    ///
    /// This intentionally supports only production mode-4 aggregated oracle
    /// binding today. Single and grouped binding proofs are accepted; RLC-only
    /// artifacts remain rejected.
    pub fn from_gkr_proof(
        proof: &crate::gkr::GKRProof,
        raw_io_data: Vec<FieldElement>,
        circuit_depth: u32,
        matmul_dims: Vec<u32>,
        dequantize_bits: Vec<u64>,
        packed: bool,
        double_packed: bool,
    ) -> Result<Self, String> {
        use crate::gkr::types::WeightOpeningTranscriptMode;

        if proof.weight_opening_transcript_mode
            != WeightOpeningTranscriptMode::AggregatedOracleSumcheck
        {
            return Err(format!(
                "conversation GKR recursion requires AggregatedOracleSumcheck binding, got {:?}",
                proof.weight_opening_transcript_mode
            ));
        }
        let mut proof_data = Vec::new();
        if double_packed {
            crate::cairo_serde::serialize_gkr_proof_data_only_double_packed(proof, &mut proof_data);
        } else if packed {
            crate::cairo_serde::serialize_gkr_proof_data_only_packed(proof, &mut proof_data);
        } else {
            crate::cairo_serde::serialize_gkr_proof_data_only(proof, &mut proof_data);
        }

        let mut weight_binding_data = Vec::new();
        if let Some(binding) = proof.aggregated_binding.as_ref() {
            crate::cairo_serde::serialize_aggregated_binding_proof_packed(
                binding,
                &mut weight_binding_data,
            );
        } else if !proof.binding_groups.is_empty() {
            weight_binding_data.push(FieldElement::from(GROUPED_BINDING_MARKER));
            weight_binding_data.push(FieldElement::from(proof.binding_groups.len() as u64));
            for group in &proof.binding_groups {
                crate::cairo_serde::serialize_aggregated_binding_proof_packed(
                    group,
                    &mut weight_binding_data,
                );
            }
        } else {
            return Err(
                "conversation GKR recursion requires full aggregated binding proof data"
                    .to_string(),
            );
        }

        Ok(Self {
            raw_io_data,
            circuit_depth,
            num_layers: proof.layer_proofs.len() as u32,
            num_matmuls: gkr_num_weight_claims(proof) as u32,
            matmul_dims,
            dequantize_bits,
            proof_data,
            weight_commitments: proof.weight_commitments.clone(),
            weight_binding_root: gkr_weight_binding_root(proof)?,
            weight_binding_mode: 4,
            weight_binding_data,
            packed,
            double_packed,
            has_kv_cache: proof.kv_cache_commitment.is_some(),
        })
    }

    pub fn io_commitment(&self) -> FieldElement {
        starknet_crypto::poseidon_hash_many(&self.raw_io_data)
    }
}

pub fn gkr_num_weight_claims(proof: &crate::gkr::GKRProof) -> u64 {
    let deferred_weight_claims = proof
        .deferred_proofs
        .iter()
        .filter(|deferred| deferred.has_weights())
        .count();
    (proof.weight_claims.len() + deferred_weight_claims) as u64
}

pub fn gkr_weight_binding_root(proof: &crate::gkr::GKRProof) -> Result<FieldElement, String> {
    if let Some(binding) = proof.aggregated_binding.as_ref() {
        Ok(binding.super_root.root)
    } else if !proof.binding_groups.is_empty() {
        let roots: Vec<_> = proof
            .binding_groups
            .iter()
            .map(|group| group.super_root.root)
            .collect();
        Ok(sequence_commitment(DOMAIN_WEIGHT_GROUPS, &roots))
    } else {
        Err("GKR proof has no full aggregated binding root".to_string())
    }
}

pub fn gkr_layer_tag(layer: &crate::gkr::types::LayerProof) -> u32 {
    use crate::gkr::types::LayerProof;

    match layer {
        LayerProof::MatMul { .. } => 0,
        LayerProof::Add { .. } => 1,
        LayerProof::Mul { .. } => 2,
        LayerProof::Activation { .. } => 3,
        LayerProof::LayerNorm { .. } => 4,
        LayerProof::Attention { .. } => 5,
        LayerProof::Dequantize { .. } => 6,
        LayerProof::MatMulDualSimd { .. } => 7,
        LayerProof::RMSNorm { .. } => 8,
        LayerProof::Quantize { .. } => 9,
        LayerProof::Embedding { .. } => 10,
        LayerProof::AttentionDecode { .. } => 11,
        LayerProof::TopK { .. } => 12,
    }
}

pub fn gkr_circuit_hash(circuit_depth: u32, proof: &crate::gkr::GKRProof) -> FieldElement {
    let mut tags_hash = FieldElement::ZERO;
    for layer in &proof.layer_proofs {
        tags_hash = starknet_crypto::poseidon_hash_many(&[
            tags_hash,
            FieldElement::from(gkr_layer_tag(layer) as u64),
        ]);
    }
    starknet_crypto::poseidon_hash_many(&[FieldElement::from(circuit_depth as u64), tags_hash])
}

/// Flatten a full conversation + per-token GKR witness for
/// `conversation-gkr-statement-verifier`.
pub fn conversation_gkr_verifier_args(
    statement: &ConversationBatchStatement,
    conversations: &[ConversationTraceStatement],
    steps: &[GenerationStepStatement],
    actions: &[ConversationActionStatement],
    gkr_proofs: &[ConversationGkrStepProofArgs],
) -> Result<Vec<String>, String> {
    if gkr_proofs.len() != steps.len() {
        return Err(format!(
            "GKR proof count {} does not match step count {}",
            gkr_proofs.len(),
            steps.len()
        ));
    }

    let mut args = vec![
        felt_hex(statement.model_id),
        felt_hex(statement.verifier_program_hash),
        felt_hex(statement.circuit_hash),
        felt_hex(statement.weight_super_root),
        felt_hex(statement.policy_commitment),
        felt_hex(statement.tokenizer_config_hash),
        felt_hex(statement.hades_commitment),
        statement.security_bits.to_string(),
    ];

    push_conversations(&mut args, conversations);
    push_steps(&mut args, steps);
    push_actions(&mut args, actions);

    args.push(gkr_proofs.len().to_string());
    for (idx, proof) in gkr_proofs.iter().enumerate() {
        let io_commitment = proof.io_commitment();
        if io_commitment != steps[idx].io_commitment {
            return Err(format!(
                "step {idx} raw IO commitment does not match statement"
            ));
        }
        if proof.weight_binding_root != statement.weight_super_root {
            return Err(format!(
                "step {idx} weight binding root does not match statement"
            ));
        }
        let receipt_hash = ml_verification_receipt_hash(
            statement.model_id,
            io_commitment,
            statement.weight_super_root,
            proof.num_layers as u64,
            proof.num_matmuls as u64,
            true,
        );
        if receipt_hash != steps[idx].recursive_proof_hash {
            return Err(format!(
                "step {idx} GKR receipt hash does not match statement"
            ));
        }
        push_felt_array(&mut args, &proof.raw_io_data);
        args.push(proof.circuit_depth.to_string());
        args.push(proof.num_layers.to_string());
        push_u32_array(&mut args, &proof.matmul_dims);
        push_u64_array(&mut args, &proof.dequantize_bits);
        push_felt_array(&mut args, &proof.proof_data);
        push_felt_array(&mut args, &proof.weight_commitments);
        args.push(proof.weight_binding_mode.to_string());
        push_felt_array(&mut args, &proof.weight_binding_data);
        args.push(u8::from(proof.packed).to_string());
        args.push(u8::from(proof.double_packed).to_string());
        args.push(u8::from(proof.has_kv_cache).to_string());
    }

    Ok(args)
}

fn push_conversations(args: &mut Vec<String>, conversations: &[ConversationTraceStatement]) {
    args.push(conversations.len().to_string());
    for conversation in conversations {
        args.extend([
            conversation.conversation_index.to_string(),
            felt_hex(conversation.conversation_id_hash),
            felt_hex(conversation.prompt_commitment),
            felt_hex(conversation.transcript_commitment),
            felt_hex(conversation.action_root),
            felt_hex(conversation.initial_kv_commitment),
            felt_hex(conversation.final_kv_commitment),
            conversation.n_turns.to_string(),
            conversation.n_prefill_tokens.to_string(),
            conversation.n_generated_tokens.to_string(),
            conversation.first_step_index.to_string(),
            conversation.n_steps.to_string(),
        ]);
    }
}

fn push_steps(args: &mut Vec<String>, steps: &[GenerationStepStatement]) {
    args.push(steps.len().to_string());
    for step in steps {
        args.extend([
            step.global_step_index.to_string(),
            step.conversation_index.to_string(),
            step.turn_index.to_string(),
            step.token_index.to_string(),
            step.generated_token_id.to_string(),
            felt_hex(step.io_commitment),
            felt_hex(step.sampling_commitment),
            felt_hex(step.prev_kv_commitment),
            felt_hex(step.kv_commitment),
            felt_hex(step.recursive_proof_hash),
        ]);
    }
}

fn push_actions(args: &mut Vec<String>, actions: &[ConversationActionStatement]) {
    args.push(actions.len().to_string());
    for action in actions {
        args.extend([
            action.conversation_index.to_string(),
            action.turn_index.to_string(),
            action.action_index.to_string(),
            felt_hex(action.action_kind_hash),
            felt_hex(action.tool_name_hash),
            felt_hex(action.input_commitment),
            felt_hex(action.output_commitment),
            felt_hex(action.policy_commitment),
        ]);
    }
}

fn push_felt_array(args: &mut Vec<String>, values: &[FieldElement]) {
    args.push(values.len().to_string());
    args.extend(values.iter().copied().map(felt_hex));
}

fn push_u32_array(args: &mut Vec<String>, values: &[u32]) {
    args.push(values.len().to_string());
    args.extend(values.iter().map(u32::to_string));
}

fn push_u64_array(args: &mut Vec<String>, values: &[u64]) {
    args.push(values.len().to_string());
    args.extend(values.iter().map(u64::to_string));
}

#[cfg(feature = "serde")]
trait FeltDefaultExt {
    fn or_else_nonzero(self, fallback: FieldElement) -> FieldElement;
}

#[cfg(feature = "serde")]
impl FeltDefaultExt for FieldElement {
    fn or_else_nonzero(self, fallback: FieldElement) -> FieldElement {
        if self == FieldElement::ZERO {
            fallback
        } else {
            self
        }
    }
}

#[cfg(feature = "serde")]
fn parse_optional_felt_hex(value: Option<&str>, field_name: &str) -> Result<FieldElement, String> {
    value.map_or(Ok(FieldElement::ZERO), |value| {
        parse_felt_hex(value, field_name)
    })
}

#[cfg(feature = "serde")]
fn parse_felt_hex(value: &str, field_name: &str) -> Result<FieldElement, String> {
    let normalized = value
        .trim()
        .strip_prefix("0x")
        .or_else(|| value.trim().strip_prefix("0X"))
        .unwrap_or(value.trim());
    FieldElement::from_hex_be(normalized)
        .map_err(|e| format!("invalid felt for {field_name}: {value}: {e}"))
}

fn felt_hex(value: FieldElement) -> String {
    format!("0x{:x}", value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fe(value: u64) -> FieldElement {
        FieldElement::from(value)
    }

    fn step(
        global: u64,
        conversation: u64,
        token: u64,
        prev: u64,
        next: u64,
    ) -> GenerationStepStatement {
        GenerationStepStatement {
            global_step_index: global,
            conversation_index: conversation,
            turn_index: 0,
            token_index: token,
            generated_token_id: 1000 + token,
            io_commitment: fe(10_000 + global),
            sampling_commitment: fe(20_000 + global),
            prev_kv_commitment: fe(prev),
            kv_commitment: fe(next),
            recursive_proof_hash: fe(30_000 + global),
        }
    }

    fn action(conversation: u64, action_index: u64) -> ConversationActionStatement {
        ConversationActionStatement {
            conversation_index: conversation,
            turn_index: 0,
            action_index,
            action_kind_hash: text_commitment(0x4B494E44, "tool_call"),
            tool_name_hash: text_commitment(0x544F4F4C, "search"),
            input_commitment: fe(40_000 + action_index),
            output_commitment: fe(50_000 + action_index),
            policy_commitment: fe(60_000),
        }
    }

    fn conversations() -> Vec<ConversationTraceStatement> {
        vec![
            ConversationTraceStatement {
                conversation_index: 0,
                conversation_id_hash: text_commitment(0x434944, "conversation-a"),
                prompt_commitment: text_commitment(0x50524D54, "prompt-a"),
                transcript_commitment: text_commitment(0x5452414E, "transcript-a"),
                action_root: action_root(&[action(0, 0)]),
                initial_kv_commitment: fe(1),
                final_kv_commitment: fe(3),
                n_turns: 1,
                n_prefill_tokens: 4,
                n_generated_tokens: 2,
                first_step_index: 0,
                n_steps: 2,
            },
            ConversationTraceStatement {
                conversation_index: 1,
                conversation_id_hash: text_commitment(0x434944, "conversation-b"),
                prompt_commitment: text_commitment(0x50524D54, "prompt-b"),
                transcript_commitment: text_commitment(0x5452414E, "transcript-b"),
                action_root: action_root(&[action(1, 0)]),
                initial_kv_commitment: fe(7),
                final_kv_commitment: fe(9),
                n_turns: 1,
                n_prefill_tokens: 5,
                n_generated_tokens: 2,
                first_step_index: 2,
                n_steps: 2,
            },
        ]
    }

    fn steps() -> Vec<GenerationStepStatement> {
        vec![
            step(0, 0, 0, 1, 2),
            step(1, 0, 1, 2, 3),
            step(2, 1, 0, 7, 8),
            step(3, 1, 1, 8, 9),
        ]
    }

    fn statement(
        conversations: &[ConversationTraceStatement],
        steps: &[GenerationStepStatement],
        actions: &[ConversationActionStatement],
    ) -> ConversationBatchStatement {
        build_conversation_batch_statement(
            fe(1),
            fe(2),
            fe(3),
            fe(4),
            fe(5),
            fe(6),
            fe(7),
            conversations,
            steps,
            actions,
            PRODUCTION_SECURITY_BITS,
        )
        .unwrap()
    }

    #[test]
    fn gkr_witness_args_bind_raw_io_and_receipt_per_step() {
        let raw_io = vec![fe(1), fe(1), fe(1), fe(42), fe(1), fe(1), fe(1), fe(43)];
        let io_commitment = starknet_crypto::poseidon_hash_many(&raw_io);
        let model_id = fe(1);
        let weight_super_root = fe(4);
        let receipt =
            ml_verification_receipt_hash(model_id, io_commitment, weight_super_root, 2, 1, true);
        let steps = vec![GenerationStepStatement {
            global_step_index: 0,
            conversation_index: 0,
            turn_index: 0,
            token_index: 0,
            generated_token_id: 1000,
            io_commitment,
            sampling_commitment: fe(20_000),
            prev_kv_commitment: fe(1),
            kv_commitment: fe(2),
            recursive_proof_hash: receipt,
        }];
        let conversations = vec![ConversationTraceStatement {
            conversation_index: 0,
            conversation_id_hash: text_commitment(0x434944, "conversation-a"),
            prompt_commitment: text_commitment(0x50524D54, "prompt-a"),
            transcript_commitment: text_commitment(0x5452414E, "transcript-a"),
            action_root: action_root(&[]),
            initial_kv_commitment: fe(1),
            final_kv_commitment: fe(2),
            n_turns: 1,
            n_prefill_tokens: 4,
            n_generated_tokens: 1,
            first_step_index: 0,
            n_steps: 1,
        }];
        let statement = build_conversation_batch_statement(
            model_id,
            fe(2),
            fe(3),
            weight_super_root,
            fe(5),
            fe(6),
            fe(7),
            &conversations,
            &steps,
            &[],
            PRODUCTION_SECURITY_BITS,
        )
        .unwrap();
        let gkr = ConversationGkrStepProofArgs {
            raw_io_data: raw_io.clone(),
            circuit_depth: 3,
            num_layers: 2,
            num_matmuls: 1,
            matmul_dims: vec![1, 1, 1],
            dequantize_bits: vec![],
            proof_data: vec![fe(9), fe(10)],
            weight_commitments: vec![fe(11)],
            weight_binding_root: weight_super_root,
            weight_binding_mode: 4,
            weight_binding_data: vec![fe(12), fe(13), fe(14)],
            packed: true,
            double_packed: false,
            has_kv_cache: true,
        };

        let args =
            conversation_gkr_verifier_args(&statement, &conversations, &steps, &[], &[gkr.clone()])
                .unwrap();
        assert_eq!(
            args[8], "1",
            "conversations array length is serialized first"
        );
        assert!(args.iter().any(|arg| arg == &felt_hex(io_commitment)));

        let mut bad_root = gkr.clone();
        bad_root.weight_binding_root += FieldElement::ONE;
        let err =
            conversation_gkr_verifier_args(&statement, &conversations, &steps, &[], &[bad_root])
                .unwrap_err();
        assert!(err.contains("weight binding root"));

        let mut bad_gkr = gkr;
        bad_gkr.raw_io_data[3] += FieldElement::ONE;
        let err =
            conversation_gkr_verifier_args(&statement, &conversations, &steps, &[], &[bad_gkr])
                .unwrap_err();
        assert!(err.contains("raw IO commitment"));
    }

    #[test]
    fn statement_binds_multiple_conversations_generations_and_actions() {
        let conversations = conversations();
        let steps = steps();
        let actions = vec![action(0, 0), action(1, 0)];

        let base = statement(&conversations, &steps, &actions).statement_hash();

        let mut changed_token = steps.clone();
        changed_token[2].generated_token_id += 1;
        assert_ne!(
            base,
            statement(&conversations, &changed_token, &actions).statement_hash()
        );

        let mut changed_action = actions.clone();
        changed_action[1].output_commitment += FieldElement::ONE;
        let mut changed_action_conversations = conversations.clone();
        changed_action_conversations[1].action_root = action_root(&[changed_action[1].clone()]);
        assert_ne!(
            base,
            statement(&changed_action_conversations, &steps, &changed_action).statement_hash()
        );

        let mut changed_final = conversations.clone();
        changed_final[1].final_kv_commitment += FieldElement::ONE;
        assert!(build_conversation_batch_statement(
            fe(1),
            fe(2),
            fe(3),
            fe(4),
            fe(5),
            fe(6),
            fe(7),
            &changed_final,
            &steps,
            &actions,
            PRODUCTION_SECURITY_BITS,
        )
        .is_err());
    }

    #[test]
    fn validation_rejects_swapped_or_dropped_generation_rows() {
        let conversations = conversations();
        let mut swapped = steps();
        swapped.swap(0, 1);
        assert!(matches!(
            validate_conversation_batch(&conversations, &swapped),
            Err(ConversationStatementError::StepGlobalIndexMismatch { .. })
        ));

        let dropped = vec![step(0, 0, 0, 1, 2), step(1, 0, 1, 2, 3)];
        assert!(matches!(
            validate_conversation_batch(&conversations, &dropped),
            Err(ConversationStatementError::StepIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn validation_rejects_relabelled_conversation_and_step_indices() {
        let mut convs = conversations();
        let generation_steps = steps();
        convs[1].conversation_index = 7;
        assert!(matches!(
            validate_conversation_batch(&convs, &generation_steps),
            Err(ConversationStatementError::ConversationIndexMismatch { .. })
        ));

        let convs = conversations();
        let mut generation_steps = steps();
        generation_steps[2].global_step_index = 99;
        assert!(matches!(
            validate_conversation_batch(&convs, &generation_steps),
            Err(ConversationStatementError::StepGlobalIndexMismatch { .. })
        ));
    }

    #[test]
    fn validation_rejects_relabelled_counts_and_actions() {
        let mut relabelled_conversations = conversations();
        let steps = steps();
        relabelled_conversations[0].n_generated_tokens = 1;
        assert!(matches!(
            validate_conversation_batch(&relabelled_conversations, &steps),
            Err(ConversationStatementError::GeneratedTokenCountMismatch { .. })
        ));

        let valid_conversations = conversations();
        let bad_actions = vec![action(0, 1), action(1, 0)];
        assert!(matches!(
            validate_conversation_actions(&valid_conversations, &bad_actions),
            Err(ConversationStatementError::ActionIndexMismatch { .. })
        ));

        let mut bad_root = conversations();
        bad_root[0].action_root = FieldElement::ZERO;
        assert!(matches!(
            validate_conversation_actions(&bad_root, &[action(0, 0), action(1, 0)]),
            Err(ConversationStatementError::ConversationActionRootMismatch { .. })
        ));

        assert!(matches!(
            validate_conversation_actions(&valid_conversations, &[action(0, 0), action(9, 0)]),
            Err(ConversationStatementError::ActionConversationOutOfRange { .. })
        ));
    }

    #[test]
    fn statement_hash_is_exact_cairo_output_hash_target() {
        let conversations = conversations();
        let steps = steps();
        let actions = vec![action(0, 0), action(1, 0)];
        let stmt = statement(&conversations, &steps, &actions);

        assert_eq!(
            stmt.statement_hash(),
            starknet_crypto::poseidon_hash_many(&stmt.to_felts())
        );
        assert_eq!(
            stmt.to_felts()[1],
            FieldElement::from(CONVERSATION_STATEMENT_VERSION)
        );
        assert_eq!(
            stmt.to_felts()[18],
            FieldElement::from(PRODUCTION_SECURITY_BITS)
        );
    }

    #[test]
    fn ml_receipt_hash_binds_inner_verifier_output() {
        let base = ml_verification_receipt_hash(fe(1), fe(2), fe(3), 8, 12, true);
        assert_ne!(
            base,
            ml_verification_receipt_hash(fe(1), fe(9), fe(3), 8, 12, true)
        );
        assert_ne!(
            base,
            ml_verification_receipt_hash(fe(1), fe(2), fe(3), 8, 13, true)
        );
        assert_ne!(
            base,
            ml_verification_receipt_hash(fe(1), fe(2), fe(3), 8, 12, false)
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn json_artifact_builds_cairo_args_for_multiple_conversations() {
        let json = r#"
        {
          "model_id": "0x1",
          "verifier_program_hash": "0x2",
          "circuit_hash": "0x3",
          "execution_contract_hash": "0x03",
          "weight_super_root": "0x4",
          "policy_commitment": "0x5",
          "tokenizer_config_hash": "0x6",
          "hades_commitment": "0x7",
          "security_bits": 160,
          "conversations": [
            {
              "conversation_id": "conversation-a",
              "prompt": "user asks a",
              "transcript": "assistant answers a",
              "initial_kv_commitment": "0x1",
              "final_kv_commitment": "0x3",
              "n_prefill_tokens": 4
            },
            {
              "conversation_id": "conversation-b",
              "prompt": "user asks b",
              "transcript": "assistant answers b",
              "initial_kv_commitment": "0x7",
              "final_kv_commitment": "0x9",
              "n_prefill_tokens": 5
            }
          ],
          "steps": [
            {
              "global_step_index": 0,
              "conversation_index": 0,
              "turn_index": 0,
              "token_index": 0,
              "generated_token_id": 100,
              "io_commitment": "0x100",
              "prev_kv_commitment": "0x1",
              "kv_commitment": "0x2",
              "recursive_proof_hash": "0x1000"
            },
            {
              "global_step_index": 1,
              "conversation_index": 0,
              "turn_index": 0,
              "token_index": 1,
              "generated_token_id": 101,
              "io_commitment": "0x101",
              "prev_kv_commitment": "0x2",
              "kv_commitment": "0x3",
              "recursive_proof_hash": "0x1001"
            },
            {
              "global_step_index": 2,
              "conversation_index": 1,
              "turn_index": 0,
              "token_index": 0,
              "generated_token_id": 200,
              "io_commitment": "0x200",
              "prev_kv_commitment": "0x7",
              "kv_commitment": "0x8",
              "recursive_proof_hash": "0x2000"
            },
            {
              "global_step_index": 3,
              "conversation_index": 1,
              "turn_index": 0,
              "token_index": 1,
              "generated_token_id": 201,
              "io_commitment": "0x201",
              "prev_kv_commitment": "0x8",
              "kv_commitment": "0x9",
              "recursive_proof_hash": "0x2001"
            }
          ],
          "actions": [
            {
              "conversation_index": 0,
              "turn_index": 0,
              "action_index": 0,
              "action_kind": "tool_call",
              "tool_name": "search",
              "input_commitment": "0x300",
              "output_commitment": "0x301"
            }
          ]
        }
        "#;

        let artifact = build_conversation_statement_artifact_from_json_str(json).unwrap();
        assert_eq!(artifact.batch.n_conversations, 2);
        assert_eq!(artifact.batch.n_steps, 4);
        assert_eq!(artifact.batch.n_generated_tokens, 4);
        assert_eq!(artifact.execution_contract_hash.as_deref(), Some("0x3"));
        assert_eq!(artifact.statement_felts.len(), 19);
        assert_eq!(artifact.cairo_args.len(), 11 + 2 * 12 + 4 * 10 + 8);
        assert_eq!(artifact.statement_hash, artifact.expected_cairo_output_hash);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn json_artifact_rejects_execution_contract_mismatch() {
        let json = r#"
        {
          "model_id": "0x1",
          "circuit_hash": "0x3",
          "execution_contract_hash": "0x4",
          "weight_super_root": "0x4",
          "policy_commitment": "0x5",
          "conversations": [
            {
              "conversation_id": "conversation-a",
              "initial_kv_commitment": "0x1",
              "final_kv_commitment": "0x2"
            }
          ],
          "steps": [
            {
              "global_step_index": 0,
              "conversation_index": 0,
              "turn_index": 0,
              "token_index": 0,
              "generated_token_id": 100,
              "io_commitment": "0x100",
              "prev_kv_commitment": "0x1",
              "kv_commitment": "0x2",
              "recursive_proof_hash": "0x1000"
            }
          ]
        }
        "#;

        let err = build_conversation_statement_artifact_from_json_str(json).unwrap_err();
        assert!(err.contains("execution_contract_hash"));
        assert!(err.contains("must equal circuit_hash"));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn json_artifact_rejects_broken_kv_chain() {
        let json = r#"
        {
          "model_id": "0x1",
          "circuit_hash": "0x3",
          "weight_super_root": "0x4",
          "policy_commitment": "0x5",
          "conversations": [
            {
              "conversation_id": "conversation-a",
              "initial_kv_commitment": "0x1",
              "final_kv_commitment": "0x3"
            }
          ],
          "steps": [
            {
              "global_step_index": 0,
              "conversation_index": 0,
              "turn_index": 0,
              "token_index": 0,
              "generated_token_id": 100,
              "io_commitment": "0x100",
              "prev_kv_commitment": "0x2",
              "kv_commitment": "0x3",
              "recursive_proof_hash": "0x1000"
            }
          ]
        }
        "#;

        let err = build_conversation_statement_artifact_from_json_str(json).unwrap_err();
        assert!(err.contains("prev KV mismatch"));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn json_artifact_accepts_ml_receipt_hash_without_legacy_proof_hash() {
        let json = r#"
        {
          "model_id": "0x1",
          "circuit_hash": "0x3",
          "weight_super_root": "0x4",
          "policy_commitment": "0x5",
          "conversations": [
            {
              "conversation_id": "conversation-a",
              "initial_kv_commitment": "0x1",
              "final_kv_commitment": "0x2"
            }
          ],
          "steps": [
            {
              "global_step_index": 0,
              "conversation_index": 0,
              "turn_index": 0,
              "token_index": 0,
              "generated_token_id": 100,
              "io_commitment": "0x100",
              "prev_kv_commitment": "0x1",
              "kv_commitment": "0x2",
              "ml_receipt_hash": "0xabc"
            }
          ]
        }
        "#;

        let artifact = build_conversation_statement_artifact_from_json_str(json).unwrap();
        assert_eq!(artifact.batch.n_steps, 1);
    }
}
