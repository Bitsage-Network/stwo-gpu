#!/usr/bin/env node
// Submit a strict STARK-in-STARK conversation/action proof.
//
// Required:
//   STARKNET_PRIVATE_KEY, STARKNET_ACCOUNT, CONTRACT
//   node scripts/submit_conversation_stwo.mjs <artifact.json> <proof.cairo-serde.json>
//
// The proof file must be generated with:
//   PROOF_FORMAT=cairo-serde ./engine/scripts/prove_conversation_gkr_statement.sh
//
// This calls GeneralStwoVerifier.verify_conversation_stwo_with_statement so the
// contract recomputes the 19-felt statement hash and rejects caller relabeling.

import { Account, RpcProvider, CallData } from "starknet";
import { readFileSync } from "fs";

const RPC = process.env.STARKNET_RPC || process.env.RPC_URL || "https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/demo";
const CONTRACT = process.env.CONTRACT || process.env.GENERAL_STWO_CONTRACT || process.env.STWO_CONTRACT;
const ADDR = process.env.STARKNET_ACCOUNT || process.env.ACCOUNT_ADDRESS;
const KEY = process.env.STARKNET_PRIVATE_KEY;
const REGISTER_PROGRAM = (process.env.REGISTER_PROGRAM || "1") !== "0";
const MIN_SECURITY_BITS = Number.parseInt(process.env.MIN_SECURITY_BITS || "160", 10);
const DRY_RUN = (process.env.DRY_RUN || "0") === "1";

const artifactPath = process.argv[2];
const proofPath = process.argv[3];

if (!CONTRACT || !ADDR || !KEY || !artifactPath || !proofPath) {
  console.error("Usage: STARKNET_PRIVATE_KEY=... STARKNET_ACCOUNT=... CONTRACT=... node scripts/submit_conversation_stwo.mjs <artifact.json> <proof.cairo-serde.json>");
  process.exit(1);
}

function asHex(value, name) {
  if (typeof value !== "string" || !value.startsWith("0x")) {
    throw new Error(`${name} must be a 0x-prefixed felt string`);
  }
  return "0x" + BigInt(value).toString(16);
}

function readJson(path, name) {
  try {
    return JSON.parse(readFileSync(path, "utf-8"));
  } catch (e) {
    throw new Error(`failed to read ${name} at ${path}: ${e.message}`);
  }
}

function readCairoSerdeProof(path) {
  const proof = readJson(path, "cairo-serde proof");
  if (!Array.isArray(proof)) {
    throw new Error("proof must be a cairo-serde JSON array of felts; regenerate with PROOF_FORMAT=cairo-serde");
  }
  if (proof.length === 0) {
    throw new Error("proof cairo-serde array is empty");
  }
  return proof.map((felt, index) => asHex(felt, `proof[${index}]`));
}

async function main() {
  const artifact = readJson(artifactPath, "conversation artifact");
  const proofFelts = readCairoSerdeProof(proofPath);
  const statementFelts = (artifact.statement_felts || []).map((felt, index) =>
    asHex(felt, `statement_felts[${index}]`),
  );
  if (statementFelts.length !== 19) {
    throw new Error(`artifact.statement_felts must contain 19 felts, got ${statementFelts.length}`);
  }

  const statementHash = asHex(artifact.statement_hash || artifact.expected_cairo_output_hash, "statement_hash");
  const expectedOutputHash = asHex(artifact.expected_cairo_output_hash || artifact.statement_hash, "expected_cairo_output_hash");
  if (statementHash !== expectedOutputHash) {
    throw new Error("statement_hash and expected_cairo_output_hash differ");
  }

  const modelId = statementFelts[2];
  const expectedProgramHash = statementFelts[3];
  if (BigInt(modelId) === 0n) throw new Error("statement model_id is zero");
  if (BigInt(expectedProgramHash) === 0n) throw new Error("statement verifier_program_hash is zero");
  if (BigInt(statementFelts[18]) < 160n) throw new Error("statement security_bits is below 160");

  console.log("RPC:              " + RPC);
  console.log("Contract:         " + CONTRACT);
  console.log("Account:          " + ADDR);
  console.log("Program hash:     " + expectedProgramHash);
  console.log("Model ID:         " + modelId);
  console.log("Statement hash:   " + statementHash);
  console.log("Statement felts:  " + statementFelts.length);
  console.log("Proof felts:      " + proofFelts.length);

  const calldata = [
    ...proofFelts,
    expectedProgramHash,
    statementHash,
    modelId,
    "0x" + statementFelts.length.toString(16),
    ...statementFelts,
  ];

  if (DRY_RUN) {
    console.log("Dry run:          no transaction submitted");
    console.log("Entrypoint:       verify_conversation_stwo_with_statement");
    console.log("Calldata felts:   " + calldata.length);
    console.log("RESULT_JSON:" + JSON.stringify({
      success: true,
      dry_run: true,
      contract: CONTRACT,
      entrypoint: "verify_conversation_stwo_with_statement",
      statement_hash: statementHash,
      proof_felts: proofFelts.length,
      calldata_felts: calldata.length,
    }));
    return;
  }

  const provider = new RpcProvider({ nodeUrl: RPC });
  const account = new Account(provider, ADDR, KEY);

  if (REGISTER_PROGRAM) {
    console.log("Register program: min_security_bits=" + MIN_SECURITY_BITS);
    const registerTx = await account.execute({
      contractAddress: CONTRACT,
      entrypoint: "register_program",
      calldata: CallData.compile({
        program_hash: expectedProgramHash,
        min_security_bits: MIN_SECURITY_BITS,
      }),
    });
    await provider.waitForTransaction(registerTx.transaction_hash);
    console.log("  registered tx:  " + registerTx.transaction_hash);
  }

  console.log("Submitting strict verify_conversation_stwo_with_statement...");
  const tx = await account.execute({
    contractAddress: CONTRACT,
    entrypoint: "verify_conversation_stwo_with_statement",
    calldata,
  });
  console.log("  verify tx:      " + tx.transaction_hash);

  const receipt = await provider.waitForTransaction(tx.transaction_hash);
  if (receipt.execution_status === "REVERTED") {
    const reason = (receipt.revert_reason || "").slice(0, 800);
    console.log("REVERTED: " + reason);
    console.log("RESULT_JSON:" + JSON.stringify({ success: false, tx_hash: tx.transaction_hash, error: reason }));
    process.exit(1);
  }

  const explorer = "https://sepolia.starkscan.co/tx/" + tx.transaction_hash;
  console.log("");
  console.log("================================================================");
  console.log("  STRICT CONVERSATION STARK-IN-STARK VERIFIED ON-CHAIN");
  console.log("================================================================");
  console.log("  TX:             " + tx.transaction_hash);
  console.log("  Statement hash: " + statementHash);
  console.log("  Proof felts:    " + proofFelts.length);
  console.log("  Explorer:       " + explorer);
  console.log("================================================================");
  console.log("RESULT_JSON:" + JSON.stringify({
    success: true,
    tx_hash: tx.transaction_hash,
    explorer_url: explorer,
    statement_hash: statementHash,
    proof_felts: proofFelts.length,
  }));
}

main().catch((e) => {
  console.error("Fatal:", (e.message || "").slice(0, 1000));
  console.log("RESULT_JSON:" + JSON.stringify({ success: false, error: (e.message || "").slice(0, 1000) }));
  process.exit(1);
});
