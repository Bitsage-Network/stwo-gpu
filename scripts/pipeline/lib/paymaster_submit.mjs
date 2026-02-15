#!/usr/bin/env node
// ═══════════════════════════════════════════════════════════════════════
// Obelysk Pipeline — Paymaster Submission Helper
// ═══════════════════════════════════════════════════════════════════════
//
// Uses starknet.js v8 native paymaster support (AVNU sponsored mode)
// to submit proofs gaslessly on Starknet.
//
// Commands:
//   setup   — Generate keypair + deploy agent account via factory
//   verify  — Submit proof via AVNU paymaster (gasless)
//   status  — Check account and verification status
//
// Usage:
//   node paymaster_submit.mjs setup --network sepolia
//   node paymaster_submit.mjs verify --proof proof.json --contract 0x... --model-id 0x1
//   node paymaster_submit.mjs status --contract 0x... --model-id 0x1
//
import {
  Account,
  RpcProvider,
  CallData,
  ETransactionVersion,
  ec,
  hash,
  num,
  byteArray,
} from "starknet";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "fs";
import { join } from "path";
import { homedir } from "os";

// ─── Constants ────────────────────────────────────────────────────────

const NETWORKS = {
  sepolia: {
    rpcPublic: "https://free-rpc.nethermind.io/sepolia-juno/",
    paymasterUrl: "https://sepolia.paymaster.avnu.fi",
    explorer: "https://sepolia.starkscan.co/tx/",
    factory: "0x2f69e566802910359b438ccdb3565dce304a7cc52edbf9fd246d6ad2cd89ce4",
    accountClassHash: "0x14d44fb938b43e5fbcec27894670cb94898d759e2ef30e7af70058b4da57e7f",
    identityRegistry: "0x72eb37b0389e570bf8b158ce7f0e1e3489de85ba43ab3876a0594df7231631",
  },
  mainnet: {
    rpcPublic: "https://free-rpc.nethermind.io/mainnet-juno/",
    paymasterUrl: "https://starknet.paymaster.avnu.fi",
    explorer: "https://starkscan.co/tx/",
    factory: "",
    accountClassHash: "",
    identityRegistry: "",
  },
};

const ACCOUNT_CONFIG_DIR = join(homedir(), ".obelysk", "starknet");
const ACCOUNT_CONFIG_FILE = join(ACCOUNT_CONFIG_DIR, "pipeline_account.json");

// ─── Argument Parsing ─────────────────────────────────────────────────

function parseArgs(argv) {
  const args = {};
  const positional = [];
  for (let i = 2; i < argv.length; i++) {
    if (argv[i].startsWith("--")) {
      const key = argv[i].slice(2);
      const next = argv[i + 1];
      if (next && !next.startsWith("--")) {
        args[key] = next;
        i++;
      } else {
        args[key] = true;
      }
    } else {
      positional.push(argv[i]);
    }
  }
  return { command: positional[0], ...args };
}

// ─── Helpers ──────────────────────────────────────────────────────────

function getProvider(network) {
  const rpcUrl =
    process.env.STARKNET_RPC ||
    (process.env.ALCHEMY_KEY
      ? `https://starknet-${network}.g.alchemy.com/starknet/version/rpc/v0_8/${process.env.ALCHEMY_KEY}`
      : NETWORKS[network].rpcPublic);
  return new RpcProvider({ nodeUrl: rpcUrl, batch: 0 });
}

function loadAccountConfig() {
  if (!existsSync(ACCOUNT_CONFIG_FILE)) return null;
  return JSON.parse(readFileSync(ACCOUNT_CONFIG_FILE, "utf-8"));
}

function saveAccountConfig(config) {
  mkdirSync(ACCOUNT_CONFIG_DIR, { recursive: true });
  writeFileSync(ACCOUNT_CONFIG_FILE, JSON.stringify(config, null, 2));
}

function getAccount(provider, privateKey, address) {
  return new Account({
    provider,
    address,
    signer: privateKey,
    transactionVersion: ETransactionVersion.V3,
  });
}

function jsonOutput(obj) {
  process.stdout.write(JSON.stringify(obj, null, 2) + "\n");
}

function die(msg) {
  process.stderr.write(`[ERR] ${msg}\n`);
  process.exit(1);
}

function info(msg) {
  process.stderr.write(`[INFO] ${msg}\n`);
}

// ─── Paymaster Execution ─────────────────────────────────────────────

async function executeViaPaymaster(account, calls) {
  const callsArray = Array.isArray(calls) ? calls : [calls];
  const feeDetails = { feeMode: { mode: "sponsored" } };

  info("Estimating paymaster fee...");
  const estimation = await account.estimatePaymasterTransactionFee(
    callsArray,
    feeDetails
  );

  info("Executing via AVNU paymaster (sponsored mode)...");
  const result = await account.executePaymasterTransaction(
    callsArray,
    feeDetails,
    estimation.suggested_max_fee_in_gas_token
  );

  return result.transaction_hash;
}

// ═══════════════════════════════════════════════════════════════════════
// Command: setup
// ═══════════════════════════════════════════════════════════════════════

async function cmdSetup(args) {
  const network = args.network || "sepolia";
  const net = NETWORKS[network];
  if (!net) die(`Unknown network: ${network}`);
  if (!net.factory) die(`Factory not deployed on ${network}`);

  const deployerKey = process.env.OBELYSK_DEPLOYER_KEY;
  if (!deployerKey) {
    die(
      "OBELYSK_DEPLOYER_KEY is required for account deployment.\n" +
        "This is the private key of the deployer that calls the factory."
    );
  }
  const deployerAddress = process.env.OBELYSK_DEPLOYER_ADDRESS;
  if (!deployerAddress) {
    die(
      "OBELYSK_DEPLOYER_ADDRESS is required for account deployment.\n" +
        "This is the address of the deployer account."
    );
  }

  const provider = getProvider(network);
  const deployer = getAccount(provider, deployerKey, deployerAddress);

  // Generate new keypair for the pipeline account
  info("Generating new Stark keypair...");
  const privateKey = num.toHex(ec.starkCurve.utils.randomPrivateKey());
  const publicKey = ec.starkCurve.getStarkKey(privateKey);
  info(`Public key: ${publicKey}`);

  // Build factory deploy_account call
  const salt = publicKey;
  const tokenUri = byteArray.byteArrayFromString(
    'data:application/json,{"name":"ObelyskPipeline","description":"Zero-config proof submission account","agentType":"prover"}'
  );

  const deployCall = {
    contractAddress: args.factory || net.factory,
    entrypoint: "deploy_account",
    calldata: CallData.compile({
      public_key: publicKey,
      salt,
      token_uri: tokenUri,
    }),
  };

  info("Deploying agent account via factory...");
  const txHash = await executeViaPaymaster(deployer, deployCall);
  info(`TX: ${txHash}`);

  info("Waiting for confirmation...");
  const receipt = await provider.waitForTransaction(txHash);
  const execStatus = receipt.execution_status ?? receipt.status ?? "unknown";
  if (execStatus === "REVERTED") {
    die(`Account deployment reverted: ${receipt.revert_reason || "unknown"}`);
  }

  // Parse AccountDeployed event
  let accountAddress = null;
  let agentId = null;
  const events = receipt.events || [];
  for (const event of events) {
    if (event.keys && event.keys.length >= 3 && event.data && event.data.length >= 3) {
      const possiblePubKey = event.keys[2];
      if (possiblePubKey && BigInt(possiblePubKey) === BigInt(publicKey)) {
        accountAddress = "0x" + BigInt(event.keys[1]).toString(16);
        const idLow = BigInt(event.data[0] || "0");
        const idHigh = BigInt(event.data[1] || "0");
        agentId = (idLow + (idHigh << 128n)).toString();
        break;
      }
    }
  }

  if (!accountAddress) {
    die("Could not parse AccountDeployed event from receipt");
  }

  // Save config
  const config = {
    address: accountAddress,
    privateKey,
    publicKey,
    agentId,
    network,
    factory: args.factory || net.factory,
    deployedAt: new Date().toISOString(),
    deployTxHash: txHash,
  };
  saveAccountConfig(config);
  info(`Account saved to ${ACCOUNT_CONFIG_FILE}`);

  jsonOutput({
    address: accountAddress,
    agentId,
    txHash,
    explorerUrl: `${net.explorer}${txHash}`,
  });
}

// ═══════════════════════════════════════════════════════════════════════
// Command: verify
// ═══════════════════════════════════════════════════════════════════════

async function cmdVerify(args) {
  const network = args.network || "sepolia";
  const net = NETWORKS[network];
  if (!net) die(`Unknown network: ${network}`);

  const proofPath = args.proof;
  const contract = args.contract;
  const modelId = args["model-id"] || "0x1";

  if (!proofPath) die("--proof is required");
  if (!contract) die("--contract is required");

  // Load or resolve account
  let privateKey, accountAddress;

  if (process.env.STARKNET_PRIVATE_KEY) {
    privateKey = process.env.STARKNET_PRIVATE_KEY;
    accountAddress = process.env.STARKNET_ACCOUNT_ADDRESS;
    if (!accountAddress) die("STARKNET_ACCOUNT_ADDRESS required when using STARKNET_PRIVATE_KEY");
    info("Using user-provided account");
  } else {
    const config = loadAccountConfig();
    if (!config) {
      die(
        "No account configured. Run 'setup' first or set STARKNET_PRIVATE_KEY.\n" +
          "  node paymaster_submit.mjs setup --network sepolia"
      );
    }
    privateKey = config.privateKey;
    accountAddress = config.address;
    info(`Using pipeline account: ${accountAddress}`);
  }

  const provider = getProvider(network);
  const account = getAccount(provider, privateKey, accountAddress);

  // Read proof file
  info(`Reading proof: ${proofPath}`);
  let proofData;
  try {
    proofData = JSON.parse(readFileSync(proofPath, "utf-8"));
  } catch (e) {
    die(`Failed to read proof file: ${e.message}`);
  }

  // Extract calldata
  let calldata = proofData.calldata || proofData.gkr_calldata;
  if (!calldata || !Array.isArray(calldata)) {
    die("Proof file missing 'calldata' or 'gkr_calldata' array");
  }

  // Build the verification call
  const verifyCall = {
    contractAddress: contract,
    entrypoint: "verify_model_gkr",
    calldata: CallData.compile([modelId, ...calldata.map(String)]),
  };

  info(`Submitting verify_model_gkr for model ${modelId}...`);
  info(`Contract: ${contract}`);
  info(`Calldata elements: ${calldata.length}`);

  let txHash;
  try {
    txHash = await executeViaPaymaster(account, verifyCall);
  } catch (e) {
    const msg = e.message || String(e);
    if (msg.includes("not eligible") || msg.includes("not supported")) {
      die(
        `Paymaster rejected transaction: ${msg}\n` +
          "This may mean:\n" +
          "  - The account is not deployed on-chain\n" +
          "  - The dApp is not registered with AVNU for sponsored mode\n" +
          "  - Daily gas limit exceeded\n" +
          "Try: STARKNET_PRIVATE_KEY=0x... ./04_verify_onchain.sh --submit --no-paymaster"
      );
    }
    throw e;
  }

  info(`TX submitted: ${txHash}`);
  info("Waiting for confirmation...");

  const receipt = await provider.waitForTransaction(txHash);
  const execStatus = receipt.execution_status ?? receipt.status ?? "unknown";

  if (execStatus === "REVERTED") {
    die(`TX reverted: ${receipt.revert_reason || "unknown reason"}`);
  }

  // Check verification status
  let isVerified = false;
  try {
    const result = await provider.callContract({
      contractAddress: contract,
      entrypoint: "get_verification_count",
      calldata: CallData.compile([modelId]),
    });
    const count = result.result ? BigInt(result.result[0] || "0") : 0n;
    isVerified = count > 0n;
  } catch {
    try {
      const result = await provider.callContract({
        contractAddress: contract,
        entrypoint: "is_proof_verified",
        calldata: CallData.compile([modelId]),
      });
      isVerified = result.result && BigInt(result.result[0] || "0") > 0n;
    } catch {
      info("Could not check verification status (contract may use different interface)");
    }
  }

  const explorerUrl = `${net.explorer}${txHash}`;
  info(`Explorer: ${explorerUrl}`);
  info(`Verified: ${isVerified}`);

  jsonOutput({
    txHash,
    explorerUrl,
    isVerified,
    gasSponsored: true,
    executionStatus: execStatus,
  });
}

// ═══════════════════════════════════════════════════════════════════════
// Command: status
// ═══════════════════════════════════════════════════════════════════════

async function cmdStatus(args) {
  const network = args.network || "sepolia";
  const net = NETWORKS[network];
  if (!net) die(`Unknown network: ${network}`);

  const contract = args.contract;
  const modelId = args["model-id"] || "0x1";

  const provider = getProvider(network);
  const config = loadAccountConfig();

  // Account status
  let accountStatus = { configured: false };
  if (config) {
    accountStatus = {
      configured: true,
      address: config.address,
      agentId: config.agentId,
      network: config.network,
      deployedAt: config.deployedAt,
    };

    // Check if deployed on-chain
    try {
      const classHash = await provider.getClassHashAt(config.address);
      accountStatus.deployedOnChain = !!classHash;
      accountStatus.classHash = classHash;
    } catch {
      accountStatus.deployedOnChain = false;
    }
  }

  // Verification status
  let verificationStatus = { checked: false };
  if (contract) {
    try {
      const result = await provider.callContract({
        contractAddress: contract,
        entrypoint: "get_verification_count",
        calldata: CallData.compile([modelId]),
      });
      const count = result.result ? Number(BigInt(result.result[0] || "0")) : 0;
      verificationStatus = {
        checked: true,
        contract,
        modelId,
        verificationCount: count,
        isVerified: count > 0,
      };
    } catch (e) {
      verificationStatus = { checked: true, error: e.message };
    }
  }

  jsonOutput({ account: accountStatus, verification: verificationStatus });
}

// ═══════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════

const args = parseArgs(process.argv);

try {
  switch (args.command) {
    case "setup":
      await cmdSetup(args);
      break;
    case "verify":
      await cmdVerify(args);
      break;
    case "status":
      await cmdStatus(args);
      break;
    default:
      process.stderr.write(
        "Usage: node paymaster_submit.mjs <command> [options]\n\n" +
          "Commands:\n" +
          "  setup    Generate keypair + deploy agent account via factory\n" +
          "  verify   Submit proof via AVNU paymaster (gasless)\n" +
          "  status   Check account and verification status\n\n" +
          "Examples:\n" +
          "  node paymaster_submit.mjs setup --network sepolia\n" +
          "  node paymaster_submit.mjs verify --proof proof.json --contract 0x... --model-id 0x1\n" +
          "  node paymaster_submit.mjs status --contract 0x... --model-id 0x1\n"
      );
      process.exit(1);
  }
} catch (e) {
  die(e.message || String(e));
}
