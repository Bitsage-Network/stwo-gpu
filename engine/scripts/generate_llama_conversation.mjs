#!/usr/bin/env node

import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import process from "node:process";

const DEFAULT_MODEL =
  ".models/llama-cache/Qwen_Qwen3-14B-GGUF_Qwen3-14B-Q4_K_M.gguf";

function usage() {
  console.error(`Usage:
  node libs/engine/scripts/generate_llama_conversation.mjs \\
    --output target/qwen-local/conversation.json \\
    --topic "ZKML production hardening" \\
    --turns 2 \\
    --max-tokens 64

Options:
  --model PATH         GGUF model path. Default: ${DEFAULT_MODEL}
  --base-url URL      Use an already-running llama.cpp server.
  --port N            Port for managed llama-server. Default: 18081
  --ctx-size N        llama.cpp context size. Default: 4096
  --request-timeout N Per-request timeout in seconds. Default: 60
  --topic TEXT        Conversation topic or first user question. Required.
  --turns N           Number of user/assistant turns. Default: 2
  --max-tokens N      Max response tokens per turn. Default: 64
  --temperature N     Sampling temperature. Default: 0.7
  --output PATH       Output conversation JSON. Required.`);
}

function parseArgs(argv) {
  const args = {
    model: DEFAULT_MODEL,
    baseUrl: null,
    port: 18081,
    ctxSize: 4096,
    requestTimeout: 60,
    topic: null,
    turns: 2,
    maxTokens: 64,
    temperature: 0.7,
    output: null,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    const val = argv[i + 1];
    switch (key) {
      case "--model":
        args.model = val;
        i += 1;
        break;
      case "--base-url":
        args.baseUrl = val?.replace(/\/$/, "");
        i += 1;
        break;
      case "--port":
        args.port = Number.parseInt(val, 10);
        i += 1;
        break;
      case "--ctx-size":
        args.ctxSize = Number.parseInt(val, 10);
        i += 1;
        break;
      case "--request-timeout":
        args.requestTimeout = Number.parseInt(val, 10);
        i += 1;
        break;
      case "--topic":
        args.topic = val;
        i += 1;
        break;
      case "--turns":
        args.turns = Number.parseInt(val, 10);
        i += 1;
        break;
      case "--max-tokens":
        args.maxTokens = Number.parseInt(val, 10);
        i += 1;
        break;
      case "--temperature":
        args.temperature = Number.parseFloat(val);
        i += 1;
        break;
      case "--output":
        args.output = val;
        i += 1;
        break;
      case "--help":
      case "-h":
        usage();
        process.exit(0);
      default:
        throw new Error(`unknown argument: ${key}`);
    }
  }

  if (!args.topic) throw new Error("missing --topic");
  if (!args.output) throw new Error("missing --output");
  if (!Number.isFinite(args.turns) || args.turns < 1) {
    throw new Error("--turns must be a positive integer");
  }
  if (!Number.isFinite(args.maxTokens) || args.maxTokens < 1) {
    throw new Error("--max-tokens must be a positive integer");
  }
  return args;
}

function firstQuestion(topic) {
  const text = topic.trim();
  const lower = text.toLowerCase();
  if (
    lower.endsWith("?") ||
    ["what ", "how ", "why ", "when ", "where ", "who ", "explain ", "describe "].some(
      (prefix) => lower.startsWith(prefix),
    )
  ) {
    return text;
  }
  return `What is ${text}, and why does it matter for production ZKML?`;
}

function followUpQuestion(topic, turnIndex) {
  const prompts = [
    `What is the most important bottleneck or risk in ${topic}, and how should it be measured?`,
    `What concrete evidence would prove that ${topic} is ready for production review?`,
    `What should be tested next for ${topic} before scaling to larger models?`,
  ];
  return prompts[(turnIndex - 1) % prompts.length];
}

async function sleep(ms) {
  await new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

async function waitForServer(baseUrl) {
  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${baseUrl}/health`);
      if (res.ok) return;
    } catch {
      // Server is still booting.
    }
    await sleep(500);
  }
  throw new Error(`llama-server did not become healthy at ${baseUrl}`);
}

function startServer(args) {
  const modelPath = resolve(args.model);
  const serverArgs = [
    "-m",
    modelPath,
    "--host",
    "127.0.0.1",
    "--port",
    String(args.port),
    "-c",
    String(args.ctxSize),
    "--jinja",
    "--reasoning",
    "off",
    "-np",
    "1",
  ];

  console.error(`[llama] starting llama-server for ${basename(modelPath)}`);
  const child = spawn("llama-server", serverArgs, {
    stdio: ["ignore", "ignore", "pipe"],
    env: {
      ...process.env,
      LLAMA_CACHE: resolve(".models/llama-cache"),
    },
  });

  child.stderr.on("data", (chunk) => {
    const text = String(chunk);
    for (const line of text.split(/\r?\n/)) {
      if (
        line.includes("error") ||
        line.includes("address") ||
        line.includes("listening") ||
        line.includes("loaded") ||
        line.includes("Metal") ||
        line.includes("exiting")
      ) {
        console.error(`[llama] ${line}`);
      }
    }
  });

  child.on("exit", (code, signal) => {
    if (code !== 0 && signal !== "SIGTERM") {
      console.error(`[llama] server exited: code=${code} signal=${signal ?? ""}`);
    }
  });

  return child;
}

async function postJson(url, body, timeoutSecs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutSecs * 1000);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await res.text();
    if (!res.ok) {
      throw new Error(`${url} failed with ${res.status}: ${text.slice(0, 500)}`);
    }
    return JSON.parse(text);
  } finally {
    clearTimeout(timer);
  }
}

async function tokenize(baseUrl, content, timeoutSecs) {
  const attempts = [
    { content, add_special: false },
    { content },
    { text: content },
  ];
  let lastError = null;
  for (const body of attempts) {
    try {
      const json = await postJson(`${baseUrl}/tokenize`, body, timeoutSecs);
      const tokens = json.tokens ?? json.content ?? json;
      if (Array.isArray(tokens)) return tokens.map((n) => Number(n));
    } catch (err) {
      lastError = err;
    }
  }
  throw new Error(`tokenization failed: ${lastError?.message ?? "unknown error"}`);
}

async function chat(baseUrl, messages, args) {
  const json = await postJson(`${baseUrl}/v1/chat/completions`, {
    model: "qwen3-14b-local",
    messages,
    max_tokens: args.maxTokens,
    temperature: args.temperature,
    stream: false,
  }, args.requestTimeout);
  const content = json.choices?.[0]?.message?.content;
  if (typeof content !== "string") {
    throw new Error(`unexpected chat response: ${JSON.stringify(json).slice(0, 500)}`);
  }
  return content.trim();
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const baseUrl = args.baseUrl ?? `http://127.0.0.1:${args.port}`;
  let server = null;

  try {
    if (!args.baseUrl) {
      server = startServer(args);
    }
    await waitForServer(baseUrl);

    const conversationId = `local_qwen_${new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14)}`;
    const system = {
      role: "system",
      content:
        "You are a concise technical assistant. Answer with concrete engineering detail.",
    };
    const messages = [system];
    const turns = [];

    for (let i = 0; i < args.turns; i += 1) {
      const question = i === 0 ? firstQuestion(args.topic) : followUpQuestion(args.topic, i);
      messages.push({ role: "user", content: question });

      const contextText = messages.map((m) => `${m.role}: ${m.content}`).join("\n");
      const fullContextTokens = await tokenize(baseUrl, contextText, args.requestTimeout);
      const lastTokenId = fullContextTokens.at(-1) ?? 0;

      const started = Date.now();
      const answer = await chat(baseUrl, messages, args);
      const generationTimeMs = Date.now() - started;
      const responseTokens = await tokenize(baseUrl, answer, args.requestTimeout);

      messages.push({ role: "assistant", content: answer });
      turns.push({
        turn_index: i,
        role: "user",
        content: question,
        full_context_tokens: fullContextTokens,
        last_token_id: lastTokenId,
        response: {
          content: answer,
          tokens: responseTokens,
          generation_time_ms: generationTimeMs,
        },
      });

      const tokPerSec =
        responseTokens.length > 0 ? responseTokens.length / (generationTimeMs / 1000) : 0;
      console.error(
        `[turn ${i}] ${responseTokens.length} tokens in ${generationTimeMs}ms (${tokPerSec.toFixed(1)} tok/s)`,
      );
    }

    const output = resolve(args.output);
    const conversation = {
      version: "1",
      conversation_id: conversationId,
      topic: args.topic,
      model_name: "Qwen3-14B-GGUF-Q4_K_M",
      model_dir: resolve(args.model),
      turns,
      metadata: {
        generated_at: new Date().toISOString(),
        total_turns: turns.length,
        temperature: args.temperature,
        max_new_tokens: args.maxTokens,
        generator: "obelysk-llama-local/0.1.0",
        base_url: baseUrl,
      },
    };

    await mkdir(dirname(output), { recursive: true });
    await writeFile(output, `${JSON.stringify(conversation, null, 2)}\n`);

    console.log(`CONVERSATION_FILE=${output}`);
    console.log(`CONVERSATION_ID=${conversationId}`);
    console.log(`CONVERSATION_TURNS=${turns.length}`);
    console.log(
      `CONVERSATION_RESPONSE_TOKENS=${turns.reduce((sum, t) => sum + t.response.tokens.length, 0)}`,
    );
  } finally {
    if (server) {
      server.kill("SIGTERM");
      await sleep(500);
    }
  }
}

main().catch((err) => {
  console.error(`Error: ${err.message}`);
  process.exit(1);
});
