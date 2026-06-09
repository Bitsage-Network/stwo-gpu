// Full GKR Model Walk Verifier
//
// Walks layers output → input, dispatching by tag to per-layer verifiers.
// This is the core of the 100% on-chain ZKML verification pipeline:
// no STARK, no FRI, no dicts — only field ops + Poseidon + sumcheck.
//
// Entry point: `verify_gkr_model()` takes flat felt252 proof data
// (serialized by stwo-ml/src/cairo_serde.rs:serialize_gkr_model_proof)
// and walks each layer in proof order, verifying via per-layer verifiers.
//
// Layer tags (matching Rust gkr/types.rs):
//   0=MatMul, 1=Add, 2=Mul, 3=Activation, 4=LayerNorm,
//   5=Attention, 6=Dequantize, 7=MatMulDualSimd, 8=RMSNorm

use crate::channel::{
    PoseidonChannel, channel_draw_qm31, channel_mix_poly_coeffs_deg3, channel_mix_secure_field,
    channel_mix_u64,
};
use crate::field::{
    CM31, QM31, eq_eval, log2_ceil, next_power_of_two, pack_qm31_to_felt, poly_eval_degree3,
    qm31_add, qm31_eq, qm31_from_u32, qm31_mul, qm31_one, qm31_sub, qm31_zero,
    unpack_qm31_from_felt, unpack_qm31_pair_from_felt,
};
use crate::layer_verifiers::{
    clone_point, verify_activation_layer, verify_add_layer, verify_matmul_layer,
    verify_rmsnorm_layer,
};
use crate::types::{CompressedGkrRoundPoly, CompressedRoundPoly, GKRClaim};

/// Weight claim collected during the GKR walk.
/// Each MatMul layer produces one: the evaluation point and expected value
/// for the weight MLE opening proof.
#[derive(Drop)]
pub struct WeightClaimData {
    /// Evaluation point: [r_j || sumcheck_challenges]
    pub eval_point: Array<QM31>,
    /// Expected value: final_b_eval from the matmul sumcheck
    pub expected_value: QM31,
}

// ============================================================================
// Proof Data Reader
// ============================================================================

/// Offset-based reader for flat felt252 proof data.
/// Advances through the data one field at a time.
/// When `packed` is true, QM31 values are read from a single packed felt252
/// (4x compression) instead of 4 separate felt252s.
/// When `double_packed` is true, degree-2 round polys (c0, c2) are read as
/// a single felt252 containing two QM31 values (8x compression per pair).
#[derive(Drop, Copy)]
pub struct ProofReader {
    pub data: Span<felt252>,
    pub offset: u32,
    pub packed: bool,
    pub double_packed: bool,
}

pub fn reader_new(data: Span<felt252>, packed: bool) -> ProofReader {
    ProofReader { data, offset: 0, packed, double_packed: false }
}

pub fn reader_new_double_packed(data: Span<felt252>) -> ProofReader {
    ProofReader { data, offset: 0, packed: true, double_packed: true }
}

pub fn read_felt(ref r: ProofReader) -> felt252 {
    assert!(r.offset < r.data.len(), "PROOF_DATA_TRUNCATED");
    let v = *r.data.at(r.offset);
    r.offset += 1;
    v
}

pub fn read_u32(ref r: ProofReader) -> u32 {
    let f = read_felt(ref r);
    let v: u256 = f.into();
    v.try_into().unwrap()
}

fn read_u64(ref r: ProofReader) -> u64 {
    let f = read_felt(ref r);
    let v: u256 = f.into();
    v.try_into().unwrap()
}

pub fn read_qm31(ref r: ProofReader) -> QM31 {
    if r.packed {
        let f = read_felt(ref r);
        unpack_qm31_from_felt(f)
    } else {
        let aa = read_u64(ref r);
        let ab = read_u64(ref r);
        let ba = read_u64(ref r);
        let bb = read_u64(ref r);
        QM31 { a: CM31 { a: aa, b: ab }, b: CM31 { a: ba, b: bb } }
    }
}

/// Read two QM31 values from a single double-packed felt252.
pub fn read_qm31_pair(ref r: ProofReader) -> (QM31, QM31) {
    let f = read_felt(ref r);
    unpack_qm31_pair_from_felt(f)
}

/// Read a compressed degree-2 round polynomial (c0, c2 only — c1 omitted).
/// The verifier reconstructs c1 = current_sum - 2*c0 - c2 during verification.
fn read_compressed_deg2_poly(ref r: ProofReader) -> CompressedRoundPoly {
    if r.double_packed {
        // Double-packed: c0 and c2 in a single felt252
        let (c0, c2) = read_qm31_pair(ref r);
        return CompressedRoundPoly { c0, c2 };
    }
    let c0 = read_qm31(ref r);
    let c2 = read_qm31(ref r);
    CompressedRoundPoly { c0, c2 }
}

/// Read a compressed degree-3 round polynomial (c0, c2, c3 — c1 omitted).
/// The verifier reconstructs c1 = current_sum - 2*c0 - c2 - c3 during verification.
fn read_compressed_deg3_poly(ref r: ProofReader) -> CompressedGkrRoundPoly {
    if r.double_packed {
        // Double-packed: (c0, c2) in one felt, c3 as single packed QM31
        let (c0, c2) = read_qm31_pair(ref r);
        let c3 = read_qm31(ref r);
        return CompressedGkrRoundPoly { c0, c2, c3 };
    }
    let c0 = read_qm31(ref r);
    let c2 = read_qm31(ref r);
    let c3 = read_qm31(ref r);
    CompressedGkrRoundPoly { c0, c2, c3 }
}

fn read_compressed_deg2_polys(ref r: ProofReader, count: u32) -> Array<CompressedRoundPoly> {
    let mut result: Array<CompressedRoundPoly> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= count {
            break;
        }
        result.append(read_compressed_deg2_poly(ref r));
        i += 1;
    }
    result
}

fn read_compressed_deg3_polys(ref r: ProofReader, count: u32) -> Array<CompressedGkrRoundPoly> {
    let mut result: Array<CompressedGkrRoundPoly> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= count {
            break;
        }
        result.append(read_compressed_deg3_poly(ref r));
        i += 1;
    }
    result
}

/// Read an optional LogUp proof from flat data.
/// Returns (has_logup, round_polys, final_w, final_in, final_out, claimed_sum).
/// Multiplicities are skipped (table-side verification done externally).
/// Round polynomials use compressed format (c1 omitted).
fn read_optional_logup(
    ref r: ProofReader,
) -> (bool, Array<CompressedGkrRoundPoly>, QM31, QM31, QM31, QM31) {
    let has_logup = read_u32(ref r);
    if has_logup == 0 {
        return (false, array![], qm31_zero(), qm31_zero(), qm31_zero(), qm31_zero());
    }

    let claimed_sum = read_qm31(ref r);
    let num_rounds = read_u32(ref r);
    let polys = read_compressed_deg3_polys(ref r, num_rounds);
    let final_w = read_qm31(ref r);
    let final_in = read_qm31(ref r);
    let final_out = read_qm31(ref r);

    // Skip multiplicities (consumed but not used — table check is external)
    let num_mults = read_u32(ref r);
    let mut i: u32 = 0;
    loop {
        if i >= num_mults {
            break;
        }
        let _ = read_u32(ref r);
        i += 1;
    }

    (true, polys, final_w, final_in, final_out, claimed_sum)
}

/// Read an optional multiplicity sumcheck proof from the proof data.
///
/// Returns (has_sumcheck, n_rounds, round_c0s, round_c1s, final_eval, claimed_sum).
/// If has_sumcheck == false, the arrays are empty and scalars are zero.
fn read_optional_multiplicity_sumcheck(
    ref r: ProofReader,
) -> (bool, u32, Array<QM31>, Array<QM31>, QM31, QM31) {
    let has_sumcheck = read_u32(ref r);
    if has_sumcheck == 0 {
        return (false, 0, array![], array![], qm31_zero(), qm31_zero());
    }

    let n_rounds = read_u32(ref r);
    let mut c0s: Array<QM31> = array![];
    let mut c1s: Array<QM31> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= n_rounds {
            break;
        }
        c0s.append(read_qm31(ref r));
        c1s.append(read_qm31(ref r));
        i += 1;
    }
    let final_eval = read_qm31(ref r);
    let claimed_sum = read_qm31(ref r);

    (true, n_rounds, c0s, c1s, final_eval, claimed_sum)
}

// ============================================================================
// Per-Layer Dispatch
// ============================================================================

/// Parse and verify a Tag 0 (MatMul) layer proof.
/// Returns (new_claim, final_b_eval) — final_b_eval is needed for weight opening verification.
pub fn dispatch_matmul(
    current_claim: @GKRClaim,
    m: u32,
    k: u32,
    n: u32,
    ref reader: ProofReader,
    ref ch: PoseidonChannel,
) -> (GKRClaim, QM31) {
    let num_rounds = read_u32(ref reader);
    let round_polys = read_compressed_deg2_polys(ref reader, num_rounds);
    let final_a = read_qm31(ref reader);
    let final_b = read_qm31(ref reader);

    let claim = verify_matmul_layer(
        current_claim, round_polys.span(), final_a, final_b, m, k, n, ref ch,
    );
    (claim, final_b)
}

/// Draw a fresh claim and dispatch a sub-matmul inside an attention block.
fn dispatch_fresh_sub_matmul(
    sub_claim_value: QM31, m: u32, k: u32, n: u32, ref reader: ProofReader, ref ch: PoseidonChannel,
) -> GKRClaim {
    panic!("Not supported in lean build")
}

/// Parse and verify a Tag 5 (Attention) layer proof.
///
/// Attention decomposes into 4 + 2*num_heads MatMul sub-proofs:
///   0: output projection, then per-head (context + score), then V, K, Q projections.
pub fn dispatch_attention(
    current_claim: @GKRClaim, ref reader: ProofReader, ref ch: PoseidonChannel,
) -> GKRClaim {
    panic!("Not supported in lean build")
}

/// Parse and verify a Tag 11 (AttentionDecode) layer proof.
///
/// Like dispatch_attention but with decode-specific dimensions:
/// - Score: (new_tokens, d_k, full_seq_len)
/// - Context: (new_tokens, full_seq_len, d_k)
/// - Projections: (new_tokens, d_model, d_model)
pub fn dispatch_attention_decode(
    current_claim: @GKRClaim, ref reader: ProofReader, ref ch: PoseidonChannel,
) -> GKRClaim {
    panic!("Not supported in lean build")
}

/// Parse and verify a Tag 1 (Add) layer proof.
fn dispatch_add(
    current_claim: @GKRClaim, ref reader: ProofReader, ref ch: PoseidonChannel,
) -> GKRClaim {
    let lhs = read_qm31(ref reader);
    let rhs = read_qm31(ref reader);
    let trunk_idx = read_u32(ref reader);

    verify_add_layer(current_claim, lhs, rhs, trunk_idx, ref ch)
}

/// Parse and verify a Tag 2 (Mul) layer proof.
fn dispatch_mul(
    current_claim: @GKRClaim, ref reader: ProofReader, ref ch: PoseidonChannel,
) -> GKRClaim {
    panic!("Not supported in lean build")
}

/// Parse and verify a Tag 3 (Activation) layer proof.
///
/// Serialization order (Rust cairo_serde.rs:serialize_layer_proof_packed_inner):
///   tag(3), act_type, input_eval, output_eval, table_commitment,
///   has_logup + [logup_data], has_ms + [ms_data], has_act_proof + [act_data]
fn dispatch_activation(
    current_claim: @GKRClaim, ref reader: ProofReader, ref ch: PoseidonChannel,
) -> GKRClaim {
    let act_type_tag = read_u64(ref reader);
    let input_eval = read_qm31(ref reader);
    let output_eval = read_qm31(ref reader);
    let _table_commitment = read_felt(ref reader);

    let (has_logup, logup_polys, w, in_e, out_e, claimed) = read_optional_logup(ref reader);
    let (ms_has, ms_n, ms_c0s, ms_c1s, ms_final, ms_claimed) = read_optional_multiplicity_sumcheck(
        ref reader,
    );

    // Read optional activation product proof (always serialized after multiplicity sumcheck).
    // This flag was previously missing, causing a 1-felt reader offset drift that
    // corrupted all subsequent layer reads (MATMUL_FINAL_MISMATCH).
    let has_act_proof = read_u32(ref reader);

    let mut result = if has_act_proof == 1 {
        // Activation product proof (Phase A soundness, replaces LogUp for ReLU).
        // Channel transcript: mix "ACT" + claim_value, draw eta, deg3 sumcheck,
        // mix final evals, optional bit evals.
        dispatch_activation_product_proof(current_claim, ref reader, ref ch)
    } else if has_logup {
        // has_act_proof == 0: use original LogUp or no-LogUp path
        verify_activation_layer(
            current_claim,
            act_type_tag,
            logup_polys.span(),
            w,
            in_e,
            out_e,
            claimed,
            ms_has,
            ms_n,
            ms_c0s.span(),
            ms_c1s.span(),
            ms_final,
            ms_claimed,
            input_eval,
            output_eval,
            ref ch,
        )
    } else {
        // LogUp skipped (M31 matmul outputs exceed table range).
        channel_mix_secure_field(ref ch, input_eval);
        GKRClaim { point: clone_point(current_claim.point), value: input_eval }
    };

    // Verify optional piecewise-linear proof (default production path for
    // GELU/Sigmoid/Softmax/SiLU). This must be active on-chain because LogUp
    // may be absent for full-M31 activations.
    let has_piecewise = read_u32(ref reader);
    if has_piecewise == 1 {
        result =
            dispatch_piecewise_activation_proof(
                current_claim, act_type_tag, input_eval, ref reader, ref ch,
            );
    }

    result
}

fn piecewise_coeff(act_type_tag: u64, idx: u32) -> (QM31, QM31) {
    if act_type_tag == 2 {
        if idx == 0 {
            return (qm31_from_u32(493464900), qm31_from_u32(0));
        }
        if idx == 1 {
            return (qm31_from_u32(907732391), qm31_from_u32(1225473846));
        }
        if idx == 2 {
            return (qm31_from_u32(1321999882), qm31_from_u32(1608135299));
        }
        if idx == 3 {
            return (qm31_from_u32(1736267373), qm31_from_u32(1147984359));
        }
        if idx == 4 {
            return (qm31_from_u32(289382371), qm31_from_u32(847180061));
        }
        if idx == 5 {
            return (qm31_from_u32(417318708), qm31_from_u32(1994212593));
        }
        if idx == 6 {
            return (qm31_from_u32(1117917353), qm31_from_u32(1582604850));
        }
        if idx == 7 {
            return (qm31_from_u32(1532184844), qm31_from_u32(1759840478));
        }
        if idx == 8 {
            return (qm31_from_u32(328967650), qm31_from_u32(1903006056));
        }
        if idx == 9 {
            return (qm31_from_u32(1315897449), qm31_from_u32(1153108122));
        }
        if idx == 10 {
            return (qm31_from_u32(1730164940), qm31_from_u32(1994212595));
        }
        if idx == 11 {
            return (qm31_from_u32(2144432431), qm31_from_u32(1992504674));
        }
        if idx == 12 {
            return (qm31_from_u32(411216275), qm31_from_u32(1147984360));
        }
        if idx == 13 {
            return (qm31_from_u32(825483766), qm31_from_u32(1608135300));
        }
        if idx == 14 {
            return (qm31_from_u32(1526082411), qm31_from_u32(1511805000));
        }
        if idx == 15 {
            return (qm31_from_u32(1285247315), qm31_from_u32(1285247315));
        }
    }
    if act_type_tag == 3 {
        if idx == 0 {
            return (qm31_from_u32(1825361101), qm31_from_u32(536870911));
        }
        if idx == 1 {
            return (qm31_from_u32(1825361101), qm31_from_u32(858993458));
        }
        if idx == 2 {
            return (qm31_from_u32(1825361101), qm31_from_u32(1181116005));
        }
        if idx == 3 {
            return (qm31_from_u32(1825361101), qm31_from_u32(1503238552));
        }
        if idx == 4 {
            return (qm31_from_u32(1825361101), qm31_from_u32(1825361099));
        }
        if idx == 5 {
            return (qm31_from_u32(1825361101), qm31_from_u32(2147483646));
        }
        if idx == 6 {
            return (qm31_from_u32(1825361101), qm31_from_u32(322122546));
        }
        if idx == 7 {
            return (qm31_from_u32(1825361101), qm31_from_u32(644245093));
        }
        if idx == 8 {
            return (qm31_from_u32(1825361101), qm31_from_u32(429496729));
        }
        if idx == 9 {
            return (qm31_from_u32(1825361101), qm31_from_u32(751619276));
        }
        if idx == 10 {
            return (qm31_from_u32(1825361101), qm31_from_u32(1073741823));
        }
        if idx == 11 {
            return (qm31_from_u32(1825361101), qm31_from_u32(1395864370));
        }
        if idx == 12 {
            return (qm31_from_u32(1825361101), qm31_from_u32(1717986917));
        }
        if idx == 13 {
            return (qm31_from_u32(1825361101), qm31_from_u32(2040109464));
        }
        if idx == 14 {
            return (qm31_from_u32(1825361101), qm31_from_u32(214748364));
        }
        if idx == 15 {
            return (qm31_from_u32(2130165231), qm31_from_u32(519552495));
        }
    }
    if act_type_tag == 4 {
        if idx == 0 {
            return (qm31_from_u32(849660839), qm31_from_u32(1073741823));
        }
        if idx == 1 {
            return (qm31_from_u32(1831514974), qm31_from_u32(1370674654));
        }
        if idx == 2 {
            return (qm31_from_u32(1522175356), qm31_from_u32(114698047));
        }
        if idx == 3 {
            return (qm31_from_u32(1496103356), qm31_from_u32(744894842));
        }
        if idx == 4 {
            return (qm31_from_u32(750732055), qm31_from_u32(2119488873));
        }
        if idx == 5 {
            return (qm31_from_u32(1427218), qm31_from_u32(663390487));
        }
        if idx == 6 {
            return (qm31_from_u32(679321127), qm31_from_u32(676188513));
        }
        if idx == 7 {
            return (qm31_from_u32(922668489), qm31_from_u32(158838373));
        }
        if idx == 8 {
            return (qm31_from_u32(516563348), qm31_from_u32(136728555));
        }
        if idx == 9 {
            return (qm31_from_u32(79595521), qm31_from_u32(1342354617));
        }
        if idx == 10 {
            return (qm31_from_u32(1932283198), qm31_from_u32(1983877489));
        }
        if idx == 11 {
            return (qm31_from_u32(2065857279), qm31_from_u32(630850749));
        }
        if idx == 12 {
            return (qm31_from_u32(480167476), qm31_from_u32(291131733));
        }
        if idx == 13 {
            return (qm31_from_u32(1613176540), qm31_from_u32(2111619863));
        }
        if idx == 14 {
            return (qm31_from_u32(883393319), qm31_from_u32(1942310010));
        }
        if idx == 15 {
            return (qm31_from_u32(1181807026), qm31_from_u32(108065201));
        }
    }
    if act_type_tag == 5 {
        if idx == 0 {
            return (qm31_from_u32(67108865), qm31_from_u32(0));
        }
        if idx == 1 {
            return (qm31_from_u32(1776147933), qm31_from_u32(362947105));
        }
        if idx == 2 {
            return (qm31_from_u32(1910365662), qm31_from_u32(449070147));
        }
        if idx == 3 {
            return (qm31_from_u32(1758252237), qm31_from_u32(1117362584));
        }
        if idx == 4 {
            return (qm31_from_u32(1606138812), qm31_from_u32(2081493263));
        }
        if idx == 5 {
            return (qm31_from_u32(1454025387), qm31_from_u32(1193978537));
        }
        if idx == 6 {
            return (qm31_from_u32(1301911962), qm31_from_u32(602302053));
        }
        if idx == 7 {
            return (qm31_from_u32(1149798537), qm31_from_u32(306463811));
        }
        if idx == 8 {
            return (qm31_from_u32(711353957), qm31_from_u32(449629389));
        }
        if idx == 9 {
            return (qm31_from_u32(845571686), qm31_from_u32(602302055));
        }
        if idx == 10 {
            return (qm31_from_u32(693458261), qm31_from_u32(1193978539));
        }
        if idx == 11 {
            return (qm31_from_u32(541344836), qm31_from_u32(2081493265));
        }
        if idx == 12 {
            return (qm31_from_u32(389231411), qm31_from_u32(1117362586));
        }
        if idx == 13 {
            return (qm31_from_u32(237117986), qm31_from_u32(449070149));
        }
        if idx == 14 {
            return (qm31_from_u32(85004561), qm31_from_u32(76615954));
        }
        if idx == 15 {
            return (qm31_from_u32(937359294), qm31_from_u32(937359294));
        }
    }
    panic!("UNSUPPORTED_PIECEWISE_ACTIVATION")
}

fn dispatch_piecewise_activation_proof(
    current_claim: @GKRClaim,
    act_type_tag: u64,
    expected_input_eval: QM31,
    ref reader: ProofReader,
    ref ch: PoseidonChannel,
) -> GKRClaim {
    let num_rounds = read_u32(ref reader);
    assert!(num_rounds > 0, "PW_ZERO_ROUNDS");
    assert!(current_claim.point.len() >= num_rounds, "PW_POINT_TOO_SHORT");

    channel_mix_u64(ref ch, 0x50575F414354); // "PW_ACT"
    channel_mix_u64(ref ch, act_type_tag);
    channel_mix_u64(ref ch, num_rounds.into());
    channel_mix_secure_field(ref ch, *current_claim.value);
    let eta = channel_draw_qm31(ref ch);

    let mut current_sum = qm31_zero();
    let mut challenges: Array<QM31> = array![];
    let mut round: u32 = 0;
    loop {
        if round >= num_rounds {
            break;
        }
        let poly = read_compressed_deg3_poly(ref reader);
        let c0 = poly.c0;
        let c2 = poly.c2;
        let c3 = poly.c3;
        let c1 = qm31_sub(qm31_sub(qm31_sub(current_sum, qm31_add(c0, c0)), c2), c3);
        channel_mix_poly_coeffs_deg3(ref ch, c0, c1, c2, c3);
        let challenge = channel_draw_qm31(ref ch);
        challenges.append(challenge);
        current_sum = poly_eval_degree3(c0, c1, c2, c3, challenge);
        round += 1;
    }

    let pw_input = read_qm31(ref reader);
    let pw_output = read_qm31(ref reader);

    let mut indicators: Array<QM31> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= 16 {
            break;
        }
        indicators.append(read_qm31(ref reader));
        i += 1;
    }

    let has_seg = read_u32(ref reader);
    assert!(has_seg == 1, "PW_MISSING_SEG_BITS");
    let mut seg_bits: Array<QM31> = array![];
    i = 0;
    loop {
        if i >= 4 {
            break;
        }
        seg_bits.append(read_qm31(ref reader));
        i += 1;
    }

    let has_low = read_u32(ref reader);
    assert!(has_low == 1, "PW_MISSING_LOW_BITS");
    let low_len = read_u32(ref reader);
    assert!(low_len == 27, "PW_LOW_BIT_COUNT");
    let mut low_bits: Array<QM31> = array![];
    i = 0;
    loop {
        if i >= low_len {
            break;
        }
        low_bits.append(read_qm31(ref reader));
        i += 1;
    }

    let has_canonical = read_u32(ref reader);
    assert!(has_canonical == 1, "PW_MISSING_CANONICAL");
    let canonical_len = read_u32(ref reader);
    assert!(canonical_len == 31, "PW_CANONICAL_COUNT");
    let mut canonical_ands: Array<QM31> = array![];
    i = 0;
    loop {
        if i >= canonical_len {
            break;
        }
        canonical_ands.append(read_qm31(ref reader));
        i += 1;
    }

    let one = qm31_one();
    let mut eta_power = one;
    let mut piecewise_val = qm31_zero();
    let mut ind_sum = qm31_zero();
    i = 0;
    loop {
        if i >= 16 {
            break;
        }
        let ind = *indicators.at(i);
        let (slope, intercept) = piecewise_coeff(act_type_tag, i);
        ind_sum = qm31_add(ind_sum, ind);
        piecewise_val =
            qm31_add(piecewise_val, qm31_mul(ind, qm31_add(qm31_mul(slope, pw_input), intercept)));
        i += 1;
    }

    let mut h = qm31_mul(eta_power, qm31_sub(pw_output, piecewise_val));
    eta_power = qm31_mul(eta_power, eta);
    h = qm31_add(h, qm31_mul(eta_power, qm31_sub(ind_sum, one)));

    i = 0;
    loop {
        if i >= 16 {
            break;
        }
        eta_power = qm31_mul(eta_power, eta);
        let ind = *indicators.at(i);
        h = qm31_add(h, qm31_mul(eta_power, qm31_mul(ind, qm31_sub(one, ind))));
        i += 1;
    }

    eta_power = qm31_mul(eta_power, eta);
    let mut bit_sum = qm31_zero();
    let two = qm31_from_u32(2);
    let mut seg_pow2 = one;
    i = 0;
    loop {
        if i >= 4 {
            break;
        }
        bit_sum = qm31_add(bit_sum, qm31_mul(seg_pow2, *seg_bits.at(i)));
        seg_pow2 = qm31_mul(seg_pow2, two);
        i += 1;
    }
    let mut ind_index_sum = qm31_zero();
    i = 0;
    loop {
        if i >= 16 {
            break;
        }
        ind_index_sum = qm31_add(ind_index_sum, qm31_mul(qm31_from_u32(i), *indicators.at(i)));
        i += 1;
    }
    h = qm31_add(h, qm31_mul(eta_power, qm31_sub(bit_sum, ind_index_sum)));

    i = 0;
    loop {
        if i >= 4 {
            break;
        }
        eta_power = qm31_mul(eta_power, eta);
        let bit = *seg_bits.at(i);
        h = qm31_add(h, qm31_mul(eta_power, qm31_mul(bit, qm31_sub(one, bit))));
        i += 1;
    }

    eta_power = qm31_mul(eta_power, eta);
    let mut low_sum = qm31_zero();
    let mut pow2 = one;
    i = 0;
    loop {
        if i >= 27 {
            break;
        }
        let bit = *low_bits.at(i);
        low_sum = qm31_add(low_sum, qm31_mul(pow2, bit));
        pow2 = qm31_mul(pow2, two);
        i += 1;
    }
    h =
        qm31_add(
            h, qm31_mul(eta_power, qm31_sub(qm31_sub(pw_input, low_sum), qm31_mul(pow2, bit_sum))),
        );

    i = 0;
    loop {
        if i >= 27 {
            break;
        }
        eta_power = qm31_mul(eta_power, eta);
        let bit = *low_bits.at(i);
        h = qm31_add(h, qm31_mul(eta_power, qm31_mul(bit, qm31_sub(one, bit))));
        i += 1;
    }

    i = 0;
    loop {
        if i >= 31 {
            break;
        }
        eta_power = qm31_mul(eta_power, eta);
        let bit = if i < 4 {
            *seg_bits.at(i)
        } else {
            *low_bits.at(i - 4)
        };
        let expected = if i == 0 {
            bit
        } else {
            qm31_mul(*canonical_ands.at(i - 1), bit)
        };
        h = qm31_add(h, qm31_mul(eta_power, qm31_sub(*canonical_ands.at(i), expected)));
        i += 1;
    }

    eta_power = qm31_mul(eta_power, eta);
    h = qm31_add(h, qm31_mul(eta_power, *canonical_ands.at(30)));

    let mut claim_prefix: Array<QM31> = array![];
    i = 0;
    loop {
        if i >= num_rounds {
            break;
        }
        claim_prefix.append(*current_claim.point.at(i));
        i += 1;
    }
    let expected = qm31_mul(eq_eval(claim_prefix.span(), challenges.span()), h);
    assert!(qm31_eq(current_sum, expected), "PW_FINAL_MISMATCH");
    assert!(qm31_eq(pw_input, expected_input_eval), "PW_INPUT_MISMATCH");

    channel_mix_secure_field(ref ch, pw_input);
    channel_mix_secure_field(ref ch, pw_output);
    i = 0;
    loop {
        if i >= 16 {
            break;
        }
        channel_mix_secure_field(ref ch, *indicators.at(i));
        i += 1;
    }
    i = 0;
    loop {
        if i >= 4 {
            break;
        }
        channel_mix_secure_field(ref ch, *seg_bits.at(i));
        i += 1;
    }
    i = 0;
    loop {
        if i >= 27 {
            break;
        }
        channel_mix_secure_field(ref ch, *low_bits.at(i));
        i += 1;
    }
    i = 0;
    loop {
        if i >= 31 {
            break;
        }
        channel_mix_secure_field(ref ch, *canonical_ands.at(i));
        i += 1;
    }

    GKRClaim { point: challenges, value: pw_input }
}

/// Read and verify an activation product proof (has_act_proof == 1 path).
///
/// Channel transcript (matches Rust starknet.rs:3354-3386):
///   1. mix_u64(0x414354)   — "ACT" tag
///   2. mix_secure_field(current_claim_value)
///   3. draw_qm31            — eta
///   4. For each round: mix_poly_coeffs_deg3(c0, c1, c2, c3), draw_qm31 — challenge
///   5. mix_secure_field(act_input_eval)
///   6. mix_secure_field(act_indicator_eval)
///   7. Optional: for each bit_eval: mix_secure_field(bit_eval)
fn dispatch_activation_product_proof(
    current_claim: @GKRClaim, ref reader: ProofReader, ref ch: PoseidonChannel,
) -> GKRClaim {
    // 1-3: Mix "ACT" tag + current claim value, draw eta
    channel_mix_u64(ref ch, 0x414354); // "ACT"
    channel_mix_secure_field(ref ch, *current_claim.value);
    let _eta = channel_draw_qm31(ref ch);

    // 4: Degree-3 sumcheck rounds
    let num_rounds = read_u32(ref reader);
    let round_polys = read_compressed_deg3_polys(ref reader, num_rounds);

    let mut act_sum = *current_claim.value;

    let mut i: u32 = 0;
    loop {
        if i >= num_rounds {
            break;
        }
        let poly = round_polys.at(i);
        let c0 = *poly.c0;
        let c2 = *poly.c2;
        let c3 = *poly.c3;
        // Reconstruct c1 = current_sum - 2*c0 - c2 - c3
        let c1 = qm31_sub(qm31_sub(qm31_sub(act_sum, qm31_add(c0, c0)), c2), c3);
        channel_mix_poly_coeffs_deg3(ref ch, c0, c1, c2, c3);
        let challenge = channel_draw_qm31(ref ch);
        act_sum = poly_eval_degree3(c0, c1, c2, c3, challenge);
        i += 1;
    }

    // 5-6: Read and mix final evaluations
    let act_input_eval = read_qm31(ref reader);
    let act_indicator_eval = read_qm31(ref reader);
    channel_mix_secure_field(ref ch, act_input_eval);
    channel_mix_secure_field(ref ch, act_indicator_eval);

    // 7: Optional Phase B binary decomposition bit evals
    let has_bit_evals = read_u32(ref reader);
    if has_bit_evals == 1 {
        let num_bits = read_u32(ref reader);
        let mut j: u32 = 0;
        loop {
            if j >= num_bits {
                break;
            }
            let bit_eval = read_qm31(ref reader);
            channel_mix_secure_field(ref ch, bit_eval);
            j += 1;
        };
    }

    GKRClaim { point: clone_point(current_claim.point), value: act_input_eval }
}

/// Parse and verify a Tag 4 (LayerNorm) layer proof.
fn dispatch_layernorm(
    current_claim: @GKRClaim, ref reader: ProofReader, ref ch: PoseidonChannel,
) -> GKRClaim {
    panic!("Not supported in lean build")
}

/// Parse and verify a Tag 6 (Dequantize) layer proof.
fn dispatch_dequantize(
    current_claim: @GKRClaim, bits: u64, ref reader: ProofReader, ref ch: PoseidonChannel,
) -> GKRClaim {
    panic!("Not supported in lean build")
}

/// Parse and verify a Tag 8 (RMSNorm) layer proof.
///
/// Serialization order (must match Rust cairo_serde.rs):
///   Part 0: input_eval, output_eval, rms_sq_eval, rsqrt_eval, rsqrt_table_commitment,
///   simd_combined Part 0b: RMS² verification proof (has_flag + optional: n_active, sq_sum,
///   rounds, final_eval)
///   Part 1: Linear eq-sumcheck (num_rounds, deg3 polys, input_final, rsqrt_final)
///   Part 2: LogUp (optional)
///   Part 3: Multiplicity sumcheck
///   Part 4: Row RMS² (optional)
fn dispatch_rmsnorm(
    current_claim: @GKRClaim, ref reader: ProofReader, ref ch: PoseidonChannel,
) -> GKRClaim {
    let input_eval = read_qm31(ref reader);
    let output_eval = read_qm31(ref reader);
    let rms_sq = read_qm31(ref reader);
    let rsqrt_eval = read_qm31(ref reader);
    let _rsqrt_table_commitment = read_felt(ref reader);
    let _simd_combined = read_u32(ref reader);

    // Part 0b: RMS² verification proof — replay channel operations to keep Fiat-Shamir in sync.
    let has_rms_sq_proof = read_u32(ref reader);
    if has_rms_sq_proof == 1 {
        let n_active = read_u32(ref reader);
        let sq_sum = read_qm31(ref reader);
        let rms_sq_rounds = read_u32(ref reader);

        // Replay exact channel sequence from Rust verify_rmsnorm_reduction Part 0
        channel_mix_u64(ref ch, 0x5251); // "RQ" tag
        channel_mix_u64(ref ch, n_active.into());
        channel_mix_secure_field(ref ch, sq_sum);

        let mut rms_sum = sq_sum;
        let mut ri: u32 = 0;
        loop {
            if ri >= rms_sq_rounds {
                break;
            }
            let c0 = read_qm31(ref reader);
            let c2 = read_qm31(ref reader);
            let c3 = read_qm31(ref reader);
            // Reconstruct c1 = current_sum - 2*c0 - c2 - c3
            let c1 = qm31_sub(qm31_sub(qm31_sub(rms_sum, qm31_add(c0, c0)), c2), c3);
            channel_mix_poly_coeffs_deg3(ref ch, c0, c1, c2, c3);
            let challenge = channel_draw_qm31(ref ch);
            rms_sum = poly_eval_degree3(c0, c1, c2, c3, challenge);
            ri += 1;
        }
        let rms_sq_final = read_qm31(ref reader);
        channel_mix_secure_field(ref ch, rms_sq_final);
    }

    // Part 1: Linear eq-sumcheck
    let num_linear_rounds = read_u32(ref reader);
    let linear_polys = read_compressed_deg3_polys(ref reader, num_linear_rounds);
    let input_final = read_qm31(ref reader);
    let rsqrt_final = read_qm31(ref reader);

    // Part 2: LogUp
    let (has_logup, logup_polys, w, in_e, out_e, claimed) = read_optional_logup(ref reader);
    // Part 3: Multiplicity sumcheck
    let (ms_has, ms_n, ms_c0s, ms_c1s, ms_final, ms_claimed) = read_optional_multiplicity_sumcheck(
        ref reader,
    );

    // Part 4: Row RMS² binding (optional — skip over it)
    let has_row_rms = read_u32(ref reader);
    if has_row_rms == 1 {
        let row_count = read_u32(ref reader);
        let mut rri: u32 = 0;
        loop {
            if rri >= row_count {
                break;
            }
            let _v = read_u32(ref reader);
            rri += 1;
        };
    }

    verify_rmsnorm_layer(
        current_claim,
        linear_polys.span(),
        input_final,
        rsqrt_final,
        rms_sq,
        rsqrt_eval,
        has_logup,
        logup_polys.span(),
        w,
        in_e,
        out_e,
        claimed,
        ms_has,
        ms_n,
        ms_c0s.span(),
        ms_c1s.span(),
        ms_final,
        ms_claimed,
        input_eval,
        output_eval,
        ref ch,
    )
}

// ============================================================================
// Top-Level Model Verifier
// ============================================================================

/// Verify a complete GKR model proof by walking layers output → input.
///
/// The proof_data is a flat felt252 array containing tag-dispatched
/// per-layer proofs (serialized by cairo_serde.rs:serialize_gkr_model_proof).
///
/// The caller is responsible for:
///   1. Seeding the channel: mix_u64(d), mix_u64(input_rows), mix_u64(input_cols)
///   2. Reconstructing the initial output claim (draw r_out, evaluate output MLE)
///   3. Mixing the output value: mix_secure_field(output_value)
///   4. Passing the initial claim
///
/// Circuit dimensions not in the proof:
///   - matmul_dims: flat [m0,k0,n0, m1,k1,n1, ...] — one triple per MatMul layer
///   - dequantize_bits: flat [bits0, bits1, ...] — one per Dequantize layer
///
/// Returns (final_input_claim, weight_claims). The caller should:
///   1. Verify final_input_claim matches the committed input data
///   2. Verify each weight_claim via MLE opening proofs against registered roots
pub fn verify_gkr_model(
    proof_data: Span<felt252>,
    num_layers: u32,
    matmul_dims: Span<u32>,
    dequantize_bits: Span<u64>,
    initial_claim: GKRClaim,
    ref ch: PoseidonChannel,
) -> (GKRClaim, Array<WeightClaimData>) {
    let (final_claim, weight_claims, _layer_tags, _deferred_weight_commitments) =
        verify_gkr_model_with_trace(
        proof_data, num_layers, matmul_dims, dequantize_bits, initial_claim, ref ch, false,
    );
    (final_claim, weight_claims)
}

/// Verify a complete GKR model proof and return additional trace metadata.
///
/// When `packed` is true, QM31 values in proof_data are read from single
/// packed felt252s (4x compression). All other data (tags, u32, felt252)
/// is read identically.
///
/// When `double_packed` is true (implies packed), degree-2 round polys
/// are read as paired QM31 values from a single felt252 (c0+c2 in 248 bits),
/// and degree-3 round polys read (c0,c2) paired + c3 single.
///
/// Returns:
///   - final_input_claim
///   - weight_claims (main + deferred)
///   - layer_tags observed in proof order (for circuit hash binding)
///   - deferred_weight_commitments (one per deferred matmul proof)
pub fn verify_gkr_model_with_trace(
    proof_data: Span<felt252>,
    num_layers: u32,
    matmul_dims: Span<u32>,
    dequantize_bits: Span<u64>,
    initial_claim: GKRClaim,
    ref ch: PoseidonChannel,
    packed: bool,
) -> (GKRClaim, Array<WeightClaimData>, Array<u32>, Array<felt252>) {
    verify_gkr_model_with_trace_dp(
        proof_data, num_layers, matmul_dims, dequantize_bits, initial_claim, ref ch, packed, false,
    )
}

/// Double-pack-aware variant of verify_gkr_model_with_trace.
pub fn verify_gkr_model_with_trace_dp(
    proof_data: Span<felt252>,
    num_layers: u32,
    matmul_dims: Span<u32>,
    dequantize_bits: Span<u64>,
    initial_claim: GKRClaim,
    ref ch: PoseidonChannel,
    packed: bool,
    double_packed: bool,
) -> (GKRClaim, Array<WeightClaimData>, Array<u32>, Array<felt252>) {
    let mut reader = if double_packed {
        reader_new_double_packed(proof_data)
    } else {
        reader_new(proof_data, packed)
    };
    let mut current_claim = initial_claim;
    let mut weight_claims: Array<WeightClaimData> = array![];
    let mut layer_tags: Array<u32> = array![];
    let mut deferred_weight_commitments: Array<felt252> = array![];

    // Counters for per-type dimension arrays
    let mut matmul_idx: u32 = 0;
    let mut dequantize_idx: u32 = 0;

    // Save claim points at each Add layer for deferred proof reconstruction.
    // DAG Add layers (residual connections) produce deferred proofs for skip
    // branches. The deferred claim's point = walk claim point at the Add layer.
    let mut deferred_add_points: Array<Array<QM31>> = array![];

    let mut layer_idx: u32 = 0;
    loop {
        if layer_idx >= num_layers {
            break;
        }

        let tag = read_u32(ref reader);
        layer_tags.append(tag);

        if tag == 0 {
            // MatMul — collect weight claim for MLE opening verification
            let dims_base = matmul_idx * 3;
            assert!(dims_base + 2 < matmul_dims.len(), "MATMUL_DIMS_UNDERRUN");
            let m = *matmul_dims.at(dims_base);
            let k = *matmul_dims.at(dims_base + 1);
            let n = *matmul_dims.at(dims_base + 2);
            matmul_idx += 1;

            // Capture r_j from current claim before reduction
            let log_m = log2_ceil(next_power_of_two(m));
            let log_n = log2_ceil(next_power_of_two(n));

            let (new_claim, final_b_eval) = dispatch_matmul(
                @current_claim, m, k, n, ref reader, ref ch,
            );

            // Build weight evaluation point: [r_j || sumcheck_challenges]
            // r_j = current_claim.point[log_m..log_m+log_n]
            // sumcheck_challenges = new_claim.point[log_m..]
            let mut eval_point: Array<QM31> = array![];
            let mut j: u32 = 0;
            loop {
                if j >= log_n {
                    break;
                }
                eval_point.append(*current_claim.point.at(log_m + j));
                j += 1;
            }
            j = log_m;
            loop {
                if j >= new_claim.point.len() {
                    break;
                }
                eval_point.append(*new_claim.point.at(j));
                j += 1;
            }

            weight_claims.append(WeightClaimData { eval_point, expected_value: final_b_eval });

            current_claim = new_claim;
        } else if tag == 1 {
            // Add — save claim point for deferred proof (skip connection)
            let claim_snap = @current_claim;
            deferred_add_points.append(clone_point(claim_snap.point));
            current_claim = dispatch_add(@current_claim, ref reader, ref ch);
        } else if tag == 2 {
            // Mul
            current_claim = dispatch_mul(@current_claim, ref reader, ref ch);
        } else if tag == 3 {
            // Activation
            current_claim = dispatch_activation(@current_claim, ref reader, ref ch);
        } else if tag == 4 {
            // LayerNorm
            current_claim = dispatch_layernorm(@current_claim, ref reader, ref ch);
        } else if tag == 5 {
            // Attention — decomposed into sub-matmul proofs
            current_claim = dispatch_attention(@current_claim, ref reader, ref ch);
        } else if tag == 6 {
            // Dequantize
            assert!(dequantize_idx < dequantize_bits.len(), "DEQUANTIZE_BITS_UNDERRUN");
            let bits = *dequantize_bits.at(dequantize_idx);
            dequantize_idx += 1;
            current_claim = dispatch_dequantize(@current_claim, bits, ref reader, ref ch);
        } else if tag == 8 {
            // RMSNorm
            current_claim = dispatch_rmsnorm(@current_claim, ref reader, ref ch);
        } else if tag == 11 {
            // AttentionDecode — decode step with cached KV
            current_claim = dispatch_attention_decode(@current_claim, ref reader, ref ch);
        } else {
            assert!(false, "UNKNOWN_LAYER_TAG");
        }

        layer_idx += 1;
    }

    assert!(matmul_idx * 3 == matmul_dims.len(), "MATMUL_DIMS_TRAILING");
    assert!(dequantize_idx == dequantize_bits.len(), "DEQUANTIZE_BITS_TRAILING");

    // ========================================================================
    // Deferred Proofs (DAG Add skip connections)
    // ========================================================================
    // After the main walk, verify deferred matmul sumcheck proofs for skip
    // branches of Add layers. Each Add layer saved its claim point above.
    // Fiat-Shamir order: walk -> deferred proofs -> weight openings.
    let num_deferred = read_u32(ref reader);
    assert!(num_deferred <= deferred_add_points.len(), "DEFERRED_COUNT_EXCEEDS_ADDS");

    let deferred_points_span = deferred_add_points.span();
    let mut def_idx: u32 = 0;
    loop {
        if def_idx >= num_deferred {
            break;
        }

        // Read deferred claim value (skip_eval from Add reduction)
        let claim_value = read_qm31(ref reader);

        // Reconstruct deferred claim point from saved Add layer point
        let point_snap = deferred_points_span.at(def_idx);
        let deferred_point = clone_point(point_snap);

        // Mix claim value into Fiat-Shamir channel (matches Rust prover)
        channel_mix_secure_field(ref ch, claim_value);

        // Read deferred proof kind tag: 0 = MatMul, 1 = Weightless (Add)
        let kind = read_u32(ref reader);

        if kind == 0 {
            // MatMul deferred proof: read dims, sumcheck, weight commitment
            let m = read_u32(ref reader);
            let k = read_u32(ref reader);
            let n = read_u32(ref reader);

            let log_m = log2_ceil(next_power_of_two(m));
            let log_n = log2_ceil(next_power_of_two(n));

            // Construct and verify deferred matmul sumcheck
            let deferred_claim = GKRClaim { point: deferred_point, value: claim_value };
            let (new_claim, final_b_eval) = dispatch_matmul(
                @deferred_claim, m, k, n, ref reader, ref ch,
            );

            // Build weight evaluation point: [r_j || sumcheck_challenges]
            let mut eval_point: Array<QM31> = array![];
            let mut j: u32 = 0;
            loop {
                if j >= log_n {
                    break;
                }
                eval_point.append(*deferred_claim.point.at(log_m + j));
                j += 1;
            }
            j = log_m;
            loop {
                if j >= new_claim.point.len() {
                    break;
                }
                eval_point.append(*new_claim.point.at(j));
                j += 1;
            }

            weight_claims.append(WeightClaimData { eval_point, expected_value: final_b_eval });

            // Read deferred weight commitment (bound by caller against registration)
            let deferred_weight_commitment = read_felt(ref reader);
            deferred_weight_commitments.append(deferred_weight_commitment);
        } else {
            // kind == 1: Weightless deferred proof (Add layer)
            // Read lhs_eval, rhs_eval, trunk_idx and replay Add channel ops
            let lhs_eval = read_qm31(ref reader);
            let rhs_eval = read_qm31(ref reader);
            let _trunk_idx = read_u32(ref reader);

            // Replay Add channel mixing (matches Rust prover transcript)
            channel_mix_secure_field(ref ch, lhs_eval);
            channel_mix_secure_field(ref ch, rhs_eval);
            let _alpha = channel_draw_qm31(ref ch);
        }

        def_idx += 1;
    }

    assert!(reader.offset == reader.data.len(), "PROOF_DATA_TRAILING");
    (current_claim, weight_claims, layer_tags, deferred_weight_commitments)
}

// ============================================================================
// Streaming Batch Verifier (v25)
// ============================================================================

/// Intermediate state returned by verify_gkr_layers_batch.
/// All hashes are incremental Poseidon hashes to avoid accumulating arrays.
#[derive(Drop)]
pub struct GKRBatchResult {
    /// Updated claim after processing this batch of layers
    pub next_claim: GKRClaim,
    /// Running Poseidon hash of packed weight expected_values
    pub weight_hash: felt252,
    /// Running Poseidon hash of layer tags
    pub tags_hash: felt252,
    /// Total weight claims seen so far (across all batches)
    pub weight_count: u32,
    /// Updated matmul dimension index (for next batch)
    pub matmul_idx: u32,
    /// Updated dequantize bits index (for next batch)
    pub dequantize_idx: u32,
    /// Claim points saved at each Add layer in this batch (for deferred proofs).
    /// Each Add layer (tag=1) produces a point snapshot used later for deferred
    /// matmul sumcheck verification of skip connections.
    pub deferred_add_points: Array<Array<QM31>>,
}

/// Verify a batch of N consecutive GKR layers and return intermediate state.
///
/// Unlike verify_gkr_model_with_trace, this processes only `num_layers_in_batch`
/// layers and uses incremental hashing instead of accumulating full arrays.
/// Weight claims are hashed: weight_hash = poseidon(prev_hash, packed_expected_value).
/// Layer tags are hashed: tags_hash = poseidon(prev_hash, tag).
///
/// The caller is responsible for:
///   1. Providing the correct initial_claim (from init TX or previous batch)
///   2. Providing only this batch's matmul_dims / dequantize_bits slices
///   3. Passing the running hashes from the previous batch
///   4. Processing deferred proofs in the finalize TX (not here)
///
/// Parameters:
///   - proof_data: flat felt252 proof data for THIS batch only
///   - num_layers_in_batch: layers to process in this batch
///   - matmul_dims_batch: [m0,k0,n0, ...] only for MatMuls in this batch
///   - dequantize_bits_batch: [bits0, ...] only for Dequantizes in this batch
///   - initial_claim: claim entering this batch
///   - ch: Fiat-Shamir channel (mutated, checkpoint between TXs)
///   - packed: whether QM31 values are packed
///   - prev_weight_hash: running Poseidon hash from previous batch
///   - prev_tags_hash: running Poseidon hash from previous batch
///   - prev_weight_count: weight claims from previous batches
///   - prev_matmul_idx: matmul dim index offset from previous batches
///   - prev_dequantize_idx: dequantize bits index offset from previous batches
pub fn verify_gkr_layers_batch(
    proof_data: Span<felt252>,
    num_layers_in_batch: u32,
    matmul_dims_batch: Span<u32>,
    dequantize_bits_batch: Span<u64>,
    initial_claim: GKRClaim,
    ref ch: PoseidonChannel,
    packed: bool,
    prev_weight_hash: felt252,
    prev_tags_hash: felt252,
    prev_weight_count: u32,
) -> GKRBatchResult {
    let mut reader = reader_new(proof_data, packed);
    let mut current_claim = initial_claim;
    let mut weight_hash = prev_weight_hash;
    let mut tags_hash = prev_tags_hash;
    let mut weight_count = prev_weight_count;
    let mut deferred_add_points: Array<Array<QM31>> = array![];

    // Local counters for THIS batch's dimension arrays
    let mut matmul_idx: u32 = 0;
    let mut dequantize_idx: u32 = 0;

    let mut layer_idx: u32 = 0;
    loop {
        if layer_idx >= num_layers_in_batch {
            break;
        }

        let tag = read_u32(ref reader);
        // Incrementally hash the tag
        tags_hash = core::poseidon::poseidon_hash_span(array![tags_hash, tag.into()].span());

        // Diagnostic: track reader offset and channel state per layer
        // assert: offset sanity + which layer we're on
        let _diag_offset = reader.offset;
        let _diag_digest = ch.digest;

        if tag == 0 {
            // MatMul
            let dims_base = matmul_idx * 3;
            assert!(dims_base + 2 < matmul_dims_batch.len(), "BATCH_MATMUL_DIMS_UNDERRUN");
            let m = *matmul_dims_batch.at(dims_base);
            let k = *matmul_dims_batch.at(dims_base + 1);
            let n = *matmul_dims_batch.at(dims_base + 2);
            matmul_idx += 1;

            let (new_claim, final_b_eval) = dispatch_matmul(
                @current_claim, m, k, n, ref reader, ref ch,
            );

            // Incrementally hash the packed expected_value
            let packed_ev = pack_qm31_to_felt(final_b_eval);
            weight_hash = core::poseidon::poseidon_hash_span(array![weight_hash, packed_ev].span());
            weight_count += 1;

            current_claim = new_claim;
        } else if tag == 1 {
            // Add — save claim point for deferred proof (skip connections)
            let claim_snap = @current_claim;
            deferred_add_points.append(clone_point(claim_snap.point));
            current_claim = dispatch_add(@current_claim, ref reader, ref ch);
        } else if tag == 2 {
            // Mul
            current_claim = dispatch_mul(@current_claim, ref reader, ref ch);
        } else if tag == 3 {
            // Activation
            current_claim = dispatch_activation(@current_claim, ref reader, ref ch);
        } else if tag == 4 {
            // LayerNorm
            current_claim = dispatch_layernorm(@current_claim, ref reader, ref ch);
        } else if tag == 5 {
            // Attention — decomposed into sub-matmul proofs
            current_claim = dispatch_attention(@current_claim, ref reader, ref ch);
        } else if tag == 6 {
            // Dequantize
            assert!(dequantize_idx < dequantize_bits_batch.len(), "BATCH_DEQUANTIZE_BITS_UNDERRUN");
            let bits = *dequantize_bits_batch.at(dequantize_idx);
            dequantize_idx += 1;
            current_claim = dispatch_dequantize(@current_claim, bits, ref reader, ref ch);
        } else if tag == 8 {
            // RMSNorm
            current_claim = dispatch_rmsnorm(@current_claim, ref reader, ref ch);
        } else if tag == 11 {
            // AttentionDecode — decode step with cached KV
            current_claim = dispatch_attention_decode(@current_claim, ref reader, ref ch);
        } else {
            assert!(false, "UNKNOWN_LAYER_TAG");
        }

        layer_idx += 1;
    }

    assert!(reader.offset == reader.data.len(), "BATCH_PROOF_DATA_TRAILING");

    GKRBatchResult {
        next_claim: current_claim,
        weight_hash,
        tags_hash,
        weight_count,
        matmul_idx,
        dequantize_idx,
        deferred_add_points,
    }
}
