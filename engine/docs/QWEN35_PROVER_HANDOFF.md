# Qwen3.5 Prover Handoff

Last updated: May 18, 2026

This document captures the current state of the Qwen3.5/Qwen3.5-35B proving work so the session can be resumed without reconstructing context from chat history.

## Current Objective

Make the prover production-grade for real LLM conversations, not only one-pass or one-token proofs.

The target production shape is:

1. Run real multi-token generations and actions.
2. Produce active per-token typed receipts.
3. Bind those receipts to a canonical conversation/action statement.
4. Prove the statement verifier execution with STWO Cairo at 160-bit security.
5. Eventually submit the final recursive proof on-chain as the public trust anchor.

The immediate H100 target is `Qwen/Qwen3.5-35B-A3B`, with the FP8 variant as the default setup target because it is the practical first H100 path.

## Architecture Direction

We are not replacing the general ZKML prover with a one-off Qwen prover.

The intended production architecture is:

- Shared prover core: graph loading, tensor commitments, GKR/STARK proving, recursive Cairo proving, serialization, public input binding, on-chain submission.
- Component library: matmul, RMSNorm, RoPE, attention, MLP/SwiGLU, MoE routing, recurrence, convolution, normalization, lookup/logup components.
- Model-family adapters: thin architecture descriptions that map a HuggingFace/GGUF model to a typed proof plan.

Qwen3.5 needs deeper adapter/component work because it is a hybrid MoE architecture with `GatedDeltaNet`, depthwise convolution, recurrent state, nonlinear transforms, and specialized normalization/gating. Standard Llama/Gemma-style dense transformer adapters should be much thinner.

## Starknet Team Findings Being Addressed

The current work is driven by these priority items:

1. Full verifier execution must be attested by a real recursive proof path, not only by digest-chain consistency.
2. Hades/LogUp provider and consumer must be active in the same proof and have negative tests for swapped rows.
3. Full felt252 IO commitments must stay intact through recursive API and calldata.
4. Security claims must be consistent. Current path is 160-bit: `pow=20`, `log_blowup=5`, `queries=28`.
5. Performance work comes after soundness: slim Hades AIR partial rounds and remove production diagnostics/scans.

## Current Honest Status

Done or materially advanced:

- Active Qwen3.5 conversation artifact exists in Rust.
- Active statement JSON ingestion exists for multi-conversation/action documents.
- `prove-model statement --qwen35-active` emits:
  - full active artifact JSON
  - length-prefixed Cairo argument array for the active statement verifier
- `qwen35-active-statement-verifier` Cairo crate exists.
- The active Cairo verifier now rebuilds the statement instead of passing through four felts.
- A one-shot H100 setup script exists: `scripts/setup_h100_qwen35b.sh`.
- Focused Qwen3.5 Rust tests pass locally.

Still not done:

- The real model forward/decode path does not yet emit production active Qwen3.5 runtime traces for a real Qwen3.5-35B generation.
- Hades LogUp is not yet fully production-wired into the same active proof path.
- Full STARK-in-STARK verification of the complete model verifier is not finished for Qwen3.5 active receipts.
- Local Mac CPU did not finish the 160-bit STWO proof generation cleanly; H100 validation is next.

## Key Files Added or Changed

### Active Qwen3.5 Rust logic

- `src/compiler/qwen35.rs`

Important additions include:

- `Qwen35ActiveConversationStatement`
- `Qwen35ActiveConversationBatchArtifact`
- active statement roots and active receipt roots
- JSON ingestion for active conversations/actions/steps
- length-prefixed Cairo verifier args generation
- active typed span/conversation receipt validation

### CLI entrypoint

- `src/bin/prove_model.rs`

The `statement` subcommand now accepts:

```bash
prove-model statement --qwen35-active \
  --input qwen35_active_statement.json \
  --output qwen35_active_statement.artifact.json \
  --args-output qwen35_active_statement.args.json
```

### Active Cairo verifier

- `../qwen35-active-statement-verifier/Scarb.toml`
- `../qwen35-active-statement-verifier/src/lib.cairo`
- `../qwen35-active-statement-verifier/README.md`

The Cairo executable consumes a length-prefixed `Array<felt252>` witness and rebuilds:

- canonical conversation/action statement hash
- conversation root
- generation root
- action root
- initial/final KV roots
- per-conversation active statement hashes
- active statement root
- active receipt root

It returns:

```text
[DOMAIN_QWEN35_ACTIVE_BATCH_ARTIFACT,
 canonical_statement_hash,
 active_statement_root,
 active_receipt_root]
```

Under STWO Cairo output packing, the public output hash is the active artifact hash.

### Scripts

- `scripts/prove_qwen35_active_conversation_statement.sh`
- `scripts/setup_h100_qwen35b.sh`
- existing: `scripts/run_h100_qwen35b_full.sh`
- existing: `scripts/download_model.sh`

`setup_h100_qwen35b.sh` is the new repeatable H100 bootstrap.

## Local Verification Already Run

From `libs/engine`:

```bash
cargo fmt
```

Passed.

```bash
cargo check --features cli,serde --bin prove-model
```

Passed.

```bash
cargo test qwen35 --features cli,model-loading,safetensors,serde --lib
```

Passed:

```text
57 passed; 0 failed
```

From `libs/qwen35-active-statement-verifier`:

```bash
scarb check
scarb build
```

Passed.

Script syntax:

```bash
bash -n scripts/prove_qwen35_active_conversation_statement.sh
bash -n scripts/setup_h100_qwen35b.sh
```

Passed.

## Local Active Statement Smoke

The synthetic smoke input used locally was written under:

```text
/private/tmp/obelyzk-active-smoke/input.json
```

Regenerating the artifact and Cairo args:

```bash
cd libs/engine
cargo run --features cli,serde --bin prove-model -- statement \
  --qwen35-active \
  --input /private/tmp/obelyzk-active-smoke/input.json \
  --output /private/tmp/obelyzk-active-smoke/artifact.json \
  --args-output /private/tmp/obelyzk-active-smoke/args.json
```

Observed output:

```text
Active Qwen3.5 artifact written: /private/tmp/obelyzk-active-smoke/artifact.json
  statement_hash: 0x650982a19a8e2422fd04d818b12e50941dd1bc71dcd741d4c3a2be6248b57de
  active_batch_felts: 4
  active_verifier_args: 38
```

The args file is length-prefixed for `main(input: Array<felt252>)`:

```text
args_len = 38
args_prefix = 0x25
first_payload_felt = 0x1001
```

## Local Cairo/STWO Proof Attempt

Command shape:

```bash
cd libs/engine
cargo run --manifest-path ../stark-cairo/cairo-prove/Cargo.toml -- \
  prove ../qwen35-active-statement-verifier/target/dev/qwen35_active_statement_verifier.executable.json \
  /private/tmp/obelyzk-active-smoke/proof.json \
  --proof-format cairo-serde \
  --recursive-160 \
  --arguments-file /private/tmp/obelyzk-active-smoke/args.json
```

Important finding:

- Before length-prefixing, Cairo VM failed immediately because `Array<felt252>` serialization was wrong.
- After length-prefixing, Cairo VM execution succeeded and STWO proving started.
- On the Mac CPU, proof generation exited with code `-1` before writing a proof file.

Conclusion: the statement format and Cairo execution path are valid; the next meaningful proof-generation check should happen on H100.

## H100 Setup Script

New script:

```bash
cd libs/engine
bash scripts/setup_h100_qwen35b.sh
```

Default target:

```text
qwen3.5-35b-a3b-fp8
```

Full model target:

```bash
bash scripts/setup_h100_qwen35b.sh --model qwen3.5-35b-a3b
```

Optional small Qwen smoke after setup:

```bash
bash scripts/setup_h100_qwen35b.sh \
  --run-qwen-smoke \
  --decode-steps 2 \
  --prefill-len 32
```

The script performs:

1. system dependency install
2. Rust nightly setup
3. Scarb setup
4. H100/CUDA verification
5. repo clone/update
6. CUDA build of `prove-model`
7. `cairo-prove` build
8. Cairo verifier builds
9. Qwen3.5 model download
10. Qwen3.5 readiness and validation
11. active recursive statement smoke
12. optional small Qwen model-path smoke

## H100 Next Steps

When the GPU is restarted:

1. Confirm SSH works:

```bash
brev shell pale-yellow-trout
```

2. On the GPU, check basics:

```bash
nvidia-smi
df -h
free -h
```

3. Sync or clone this repo/branch with the new script and active verifier changes.

4. Run setup:

```bash
cd ~/bitsage-network/libs/engine
bash scripts/setup_h100_qwen35b.sh --model qwen3.5-35b-a3b-fp8
```

5. If setup passes, run a small model-path smoke:

```bash
bash scripts/setup_h100_qwen35b.sh \
  --skip-deps \
  --skip-repo \
  --skip-build \
  --skip-model \
  --skip-active-smoke \
  --run-qwen-smoke \
  --decode-steps 2 \
  --prefill-len 32
```

6. Production-style run:

```bash
bash scripts/run_h100_qwen35b_full.sh \
  --model-dir ~/.obelyzk/models/qwen3.5-35b-a3b-fp8 \
  --layers all \
  --decode-steps 16 \
  --prefill-len 128
```

Do not claim production readiness from a `--layers 1` run. It is diagnostic only.

## What To Measure On H100

Capture these values in the run artifact directory:

- GPU model and driver/CUDA versions
- disk and RAM availability
- model variant: BF16/FP8/int4
- model load time
- readiness/contract hash
- token generation throughput
- proof time per token
- active typed component proof time
- recursive Cairo proof time
- proof size in felts
- peak VRAM
- failure location if any

## Production Gaps To Resume After H100 Smoke

1. Real runtime trace emission:
   - Wire actual Qwen3.5 decode/forward activations into `Qwen35TypedRuntimeTrace`.
   - Stop relying on synthetic test traces for active receipt validation.

2. Active component coverage:
   - Ensure DepthwiseConv1D, DeltaRecurrence transform/nonlinear/arithmetic, and NormAndZGate are consumed from the real active Qwen3.5 path.

3. Hades LogUp:
   - Wire provider and consumer into the same active proof.
   - Add/keep negative test where Hades rows are swapped independently of chain rows.

4. Full STARK-in-STARK:
   - Current active sidecar proves statement reconstruction and receipt binding.
   - Full model verifier execution still needs to be proven inside the recursive path for complete trustless composition.

5. Performance:
   - Slim Hades AIR partial rounds.
   - Remove unconditional diagnostics and duplicate row scans from production prover path.
   - Only optimize after soundness and statement binding are fully active.

## Worktree Notes

The repo is dirty and contains unrelated changes. Do not reset or revert broad files.

Relevant current work includes:

```text
libs/engine/src/compiler/qwen35.rs
libs/engine/src/bin/prove_model.rs
libs/engine/scripts/prove_qwen35_active_conversation_statement.sh
libs/engine/scripts/setup_h100_qwen35b.sh
libs/qwen35-active-statement-verifier/
libs/engine/docs/QWEN35_PROVER_HANDOFF.md
```

Before committing, inspect the full diff carefully because `src/bin/prove_model.rs` already had substantial changes in the working tree.

