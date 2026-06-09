# Conversation ML Statement Verifier

Strict STARK-in-STARK executable for production conversation proofs.

Unlike `conversation-statement-verifier`, this program does not only bind
precomputed step receipts. It accepts one Cairo `MLProof` per generation step,
calls `obelysk_ml_air::verify_ml()` inside the same Cairo execution, checks the
returned `MLVerificationOutput` against the step row, and then emits the
canonical 19-felt conversation statement.

For each step:

```text
verify_ml_v2(ml_proof) must return:
  verified          = true
  model_id          = witness.model_id
  io_commitment     = step.io_commitment
  weight_commitment = witness.weight_super_root

step.ml_receipt_hash must equal:
  poseidon_hash_span([
    "MLRC",
    model_id,
    io_commitment,
    weight_commitment,
    num_layers,
    num_matmuls,
    verified_as_felt
  ])
```

The final output is still:

```text
poseidon_hash_span(statement_felts) == statement_hash
```

That means `verify_conversation_stwo` can verify the same statement hash, but
the proven Cairo execution now includes full per-step ML verifier execution.

Current integration note: this executable now uses the versioned
`obelysk_ml_air::MLProofV2` schema. V2 deliberately fails closed on proof
sections that are not fully verified in Cairo yet: batched matmul needs exact
Poseidon transcript wiring, attention needs its softmax STARK payload, and
RMSNorm/layernorm/add/mul/embedding/quantize/dequantize need their unified AIR
evaluators wired into the Cairo verifier. This is the production-safe boundary:
accepted steps are verified by the recursive Cairo execution; unsupported real
Qwen/full-model sections are rejected until implemented.
