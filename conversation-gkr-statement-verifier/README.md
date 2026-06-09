# Conversation GKR Statement Verifier

Strict STARK-in-STARK executable for real full-model GKR conversation proofs.

This verifier is the production-oriented recursive target for Qwen-style
conversation proving. For every generated step it:

1. Recomputes the full uncompressed `io_commitment` from `raw_io_data`.
2. Seeds the Poseidon Fiat-Shamir channel exactly like the GKR prover.
3. Evaluates the output MLE to create the initial GKR claim.
4. Runs `elo_cairo_verifier::model_verifier::verify_gkr_model_with_trace_dp`.
5. Verifies the final input MLE against the raw input.
6. Verifies aggregated weight binding against `weight_super_root`.
7. Checks the step `ml_receipt_hash`, then emits the canonical 19 statement felts.

Unsupported paths fail closed: non-aggregated weight binding, trailing proof
data, bad IO, bad circuit tags, bad input/output MLEs, or bad weight roots all
panic inside the recursive Cairo execution.

The decode prover emits the strict artifact and argument file when run with
`--decode --recursive`:

```text
<basename>.conversation_gkr.artifact.json
<basename>.conversation_gkr.args.json
```

Run the recursive proof over this verifier with:

```bash
ARTIFACT=target/decode/<basename>.conversation_gkr.artifact.json \
  ./engine/scripts/prove_conversation_gkr_statement.sh
```

This compiles `conversation_gkr_statement_verifier`, runs `cairo-prove prove
--recursive-160 --arguments-file <basename>.conversation_gkr.args.json`, and
writes `<basename>.conversation_gkr.proof.json`. The default proof format is
`cairo-serde`, a JSON array of felt252 strings ready for calldata.

The resulting STWO Cairo proof attests the execution of the strict verifier:
the public output hash must equal
`conversation_gkr.artifact.json::statement_hash`.

Submit the proof through the strict statement-bound verifier entrypoint:

```bash
STARKNET_PRIVATE_KEY=... STARKNET_ACCOUNT=... CONTRACT=<general_stwo_verifier> \
  node engine/scripts/submit_conversation_stwo.mjs \
    target/decode/<basename>.conversation_gkr.artifact.json \
    target/decode/<basename>.conversation_gkr.proof.json
```

This calls `verify_conversation_stwo_with_statement`, so the contract recomputes
the 19-felt statement hash and rejects caller-side relabeling of model,
program, and security metadata.
