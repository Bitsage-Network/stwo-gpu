# Qwen3.5 Active Statement Verifier

This Cairo executable verifies the active Qwen3.5 conversation batch artifact
boundary. It consumes the length-prefixed flat witness emitted by
`prove-model statement --qwen35-active --args-output ...`, rebuilds the
canonical conversation/action statement, rebuilds the active statement and
receipt roots, and returns the four public active batch felts.

With `cairo-prove --recursive-160`, the public output hash is:

`Poseidon([DOMAIN_QWEN35_ACTIVE_BATCH_ARTIFACT, canonical_statement_hash, active_statement_root, active_receipt_root])`

That hash is the Qwen3.5 active artifact hash. The canonical conversation
statement remains reusable by generic conversation verifiers, while this active
sidecar binds the statement to active typed Qwen3.5 receipts, per-token span
receipt hashes, KV continuity, action roots, model/policy metadata, and the
active receipt root inside the Cairo execution being proven.
