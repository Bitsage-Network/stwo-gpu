# Conversation Statement Verifier

Standalone Cairo executable for STARK-in-STARK conversation/action binding.

It consumes the flat `cairo_args` emitted by:

```bash
cargo run --bin prove-model --features cli -- statement \
  --input conversation_statement.json \
  --output conversation_statement.artifact.json \
  --args-output conversation_statement.args.json
```

The executable recomputes ordered conversation, generation, action, and KV roots,
checks KV continuity for every conversation, and returns the canonical 19
statement felts. With `poseidon_outputs_packing`, the on-chain STWO verifier
observes:

```text
output_hash = poseidon_hash_span(statement_felts)
```

That `output_hash` must equal `conversation_statement.artifact.json::statement_hash`
when calling `verify_conversation_stwo`.

End-to-end proof command:

```bash
INPUT=conversation_statement.json OUT_DIR=target/conversation-proof \
  ./engine/scripts/prove_conversation_statement.sh
```

This builds the artifact, compiles this Cairo executable, and runs:

```bash
cairo-prove prove ... --recursive-160 --arguments-file conversation_statement.args.json
```

The produced STWO Cairo proof is the trustless STARK-in-STARK proof of the
statement verifier execution. The on-chain verifier should be called through
`verify_conversation_stwo_with_statement` with the proof plus the artifact's
canonical 19 `statement_felts`:

```text
expected_program_hash = artifact.statement_felts[3]
statement_hash        = artifact.statement_hash
model_id              = artifact.statement_felts[2]
statement_felts       = artifact.statement_felts
```

Production note: this executable verifies the conversation statement only. The
strict path that also executes one inner ML verifier per generated token lives
in `conversation-ml-statement-verifier`. New production JSON should use
`ml_receipt_hash` for each step; `recursive_proof_hash` remains accepted only
for legacy artifacts.
