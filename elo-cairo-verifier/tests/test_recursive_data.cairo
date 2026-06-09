// Small metadata stub for recursive verifier tests.
//
// The real 160-bit Hades+LogUp fixture generated on this machine is 56,545
// felts. Embedding it directly in a Cairo test source overflows Sierra test
// compilation, so the production fixture is generated externally by:
//
//   OBELYZK_FIXTURE_OUT=/private/tmp/test_recursive_data.cairo \
//     cargo test --features cli --lib \
//     recursive::prover::tests::dump_tiny_recursive_160_cairo_fixture \
//     -- --ignored --nocapture

pub fn tiny_model_id() -> felt252 {
    0x54494e595f524543555253495645
}

pub fn tiny_circuit_hash() -> felt252 {
    0xd4a38a02676a648f2e7f6278ad2d995
}

pub fn tiny_weight_root() -> felt252 {
    0x789abc
}

pub fn tiny_io_commitment() -> felt252 {
    0x123456789abcdef
}

pub fn tiny_policy_commitment() -> felt252 {
    0
}

pub fn tiny_level1_proof_hash() -> felt252 {
    0
}

pub fn tiny_statement_hash() -> felt252 {
    0xfeedcafe
}

pub fn tiny_n_layers() -> u32 {
    1
}

pub fn tiny_n_matmuls() -> u32 {
    1
}

pub fn tiny_hidden_size() -> u32 {
    4
}

pub fn tiny_num_transformer_blocks() -> u32 {
    1
}

pub fn tiny_expected_n_poseidon_perms() -> u32 {
    18
}

pub fn tiny_trace_log_size() -> u32 {
    11
}

pub fn tiny_total_felts() -> u32 {
    56545
}

pub fn tiny_calldata() -> Array<felt252> {
    array![]
}
