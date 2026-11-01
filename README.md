# AI-Inference-Execution-CSharp

[![Tests](https://github.com/Glareone/AI-Inference-Execution-CSharp/actions/workflows/tests.yml/badge.svg)](https://github.com/Glareone/AI-Inference-Execution-CSharp/actions/workflows/tests.yml)

An investigation into how LLM inference engines work, implemented in C#/.NET 10.

## What This Project Is

A hands-on learning project: understand every piece of an LLM inference engine — model
loading, tokenization, attention mechanisms, KV-cache, sampling, serving — by building a
working version in C#. Not a production engine. Not a wrapper. A guided investigation where
every component is understood deeply enough to explain to anyone.

**Apple Silicon-centric.** This project targets M-series Macs running macOS. We have no plans to
support Windows or Intel. Hardware acceleration goes through Apple's stack, in two phases (see
[261101-apple-silicon-acceleration-strategy.md](docs/architecture/261101-apple-silicon-acceleration-strategy.md)):

- **Phase 1 — done.** Matrix-vector math runs through Apple Accelerate (`cblas_sgemv`) on the
  **CPU**. About 2.4–2.8x faster than before on this machine.
- **Phase 2 — not started.** **The engine does not use the GPU or Metal yet.** The plan is MLX
  (which runs on Metal) through its C API. Next step is a spike; see the
  [handoff doc](docs/investigation/phase2-mlx-handoff.md).

## How to test the project

Point `--model` at a local GGUF file (F16 or F32 — see [Test Model](#test-model)) and pass a prompt:

```sh
dotnet run --project src/InferenceEngine.Cli -- --model /path/to/SmolLM2-135M-Instruct-f16.gguf --prompt "What is machine learning? Explain briefly."
```

The answer streams straight to the console. Or copy `.env.example` to `.env`, set
`INFERENCE_MODEL`, and just run `dotnet run --project src/InferenceEngine.Cli`. Worth trying:
`--temperature 0.7` for varied output instead of the deterministic default, or
`--ban-words "Machine"` to see that exact token sequence excluded from generation (it bans the
tokenized sequence `"Machine"` encodes to, not the lexical word in every surrounding context —
see [260917-logits-processing.md](docs/architecture/260917-logits-processing.md)).

## What We're Investigating

GGUF loading, BPE tokenization, the transformer forward pass (GQA, RoPE, SwiGLU), a paged
KV-cache, sampling, and logits processing — each a real ADR in `docs/architecture/`, not just a
topic list.

**Supported today**: Llama-architecture GGUF (F32/F16 only), text prompts (ChatML or raw), and
temperature/top-k/top-p/seed control — plus banning specific token sequences (e.g. keeping a
competitor's name out of the output entirely), enforced on every decode step including the
default greedy path. See [Known limitations](#known-limitations) for what's not there yet.

## Architecture

Start with the [solution-layout ADR](docs/architecture/260811-solution-and-project-layout.md) —
project boundaries, what each project owns, why. The diagram below is the runtime pipeline that
layout supports: one prompt in, one streamed response out.


```mermaid
flowchart TB
    prompt["Prompt text"] --> encode

    subgraph tok["InferenceEngine.Tokenizers"]
        encode["Encode: text to token ids"]
        decode["Decode: token ids to text"]
    end

    subgraph mod["InferenceEngine.Models"]
        forward["Forward pass: embed, RoPE, GQA attention, SwiGLU, logits"]
    end

    subgraph eng["InferenceEngine.Engine"]
        kv["KV-cache: paged, head-major"]
        processor["Logits Processor: ban token sequences"]
        sampler["Sampler: temperature, top-k, top-p"]
    end

    encode --> forward
    forward --> kv
    kv --> forward
    forward --> processor --> sampler
    sampler -->|next token, loop| forward
    sampler -->|EOS or max tokens| decode
    decode --> output["Streamed text"]
```

## Project Structure

```
src/
  InferenceEngine.Core/         Shared abstractions (IModel, ITokenizer, etc.)
  InferenceEngine.Models/       GGUF loading, model config
  InferenceEngine.Tokenizers/   Tokenization (BPE, SentencePiece)
  InferenceEngine.Engine/       Generation loop, KV-cache, sampling
  InferenceEngine.Cli/          Console entry point
tests/
  InferenceEngine.{Core,Models,Tokenizers,Engine,Cli}.Tests/
                                 xUnit test project per src/ project (see each README.md)
docs/
  scenarios/                   Gherkin (.feature) specs, one per src/ project — the business
                                scenarios the tests and manual CLI verification cover
  architecture/                 ADRs in MADR format
  investigation/                Research notes, architecture traces
experiments/                    Benchmark data, reference measurements
```

See [the solution-layout ADR](docs/architecture/260811-solution-and-project-layout.md) for the
rationale behind this layout.

## Test Model

**SmolLM2-135M-Instruct**, Llama architecture, 30 layers, 576 hidden dim, GQA 3:1 (9 query / 3 KV
heads), 49K vocab. Small enough to iterate on, large enough to produce coherent text.

Two GGUF variants are in play, for different reasons:

- **F16** (`bartowski/SmolLM2-135M-Instruct-GGUF`, ~270 MB) — what *this* engine actually loads
  and runs. Our GGUF reader only materializes F32/F16 tensor data (see Status below), so this is
  the variant `--model` points at.
- **Q4_K_M** (101 MB) — installed via the dotLLM CLI and used only as the reference baseline
  (53.9 tok/s decode) in [experiments/reference-measurements-dotllm.md](experiments/reference-measurements-dotllm.md).
  Our engine cannot load this file yet — quantized tensors parse as metadata but throw on read.

## Claude Code Agents

This project uses AI-assisted development with structured agents:

| Agent | Purpose |
|-------|---------|
| `adr-author` | Write/review Architecture Decision Records (MADR format) |
| `csharp-dotnet` | C#/.NET implementation and code review |
| `test-writer-runner` | Write/update/run tests, one project per `src/` project, business-case documented |
| `researcher` | Web research on inference internals (algorithms, specs, papers) |
| `huggingface-explorer` | HuggingFace ecosystem (model formats, APIs, downloads) |
| `code-reader` | Read/analyze external repos (dotLLM, LLamaSharp) |

### MCP servers

| Server | Purpose |
|--------|---------|
| [context7](https://github.com/upstash/context7) | Current library/BCL docs instead of training data |
| [Serena](https://github.com/oraios/serena) | Symbol-level code navigation via the Roslyn language server |

Serena is configured per-project in [.mcp.json](.mcp.json) and launched with `uvx`, so the only
prerequisite is [uv](https://docs.astral.sh/uv/getting-started/installation/) — no global
install. It needs .NET 10+ for its C# language server, which this repo already requires.
Approve the server once when Claude Code prompts, then check it with `/mcp`.

## Status

**End-to-end generation works.** `dotnet run --project src/InferenceEngine.Cli -- --model
<path.gguf> --prompt "..."` loads a GGUF file, tokenizes, runs the full transformer forward pass,
samples, and streams generated text — on real weights, not a stub. Last verified: 2026-11-01.

> **Acceleration status: Phase 1 of 2.** Matrix-vector math is accelerated on the **CPU** via
> Apple Accelerate. **No GPU/Metal execution yet** — that is Phase 2 (MLX), not started. Details in
> the [acceleration ADR](docs/architecture/261101-apple-silicon-acceleration-strategy.md).

### What works today

- **Model loading**: hand-rolled GGUF v3 reader (`InferenceEngine.Models/Gguf/`), memory-mapped,
  metadata + tensor descriptors + F32/F16 tensor data.
- **Tokenization**: hand-rolled byte-level BPE (`InferenceEngine.Tokenizers/`) sourced from the
  GGUF file's own embedded vocab/merges — no external tokenizer file needed.
- **Transformer forward pass**: embeddings → RMSNorm → grouped-query attention with RoPE →
  SwiGLU FFN → LM head (`InferenceEngine.Models/Llama/`), built on
  `System.Numerics.Tensors.TensorPrimitives` — no hand-written SIMD/intrinsics, no hand-written
  `unsafe` — plus Apple Accelerate `cblas_sgemv` for the matrix-vector multiplies (CPU, macOS).
  The original managed loop stays as the test reference and the non-macOS fallback.
- **KV-cache**: block-paged, head-major (HND) layout — 32-token blocks, lazy allocation, the
  attention loop reordered to cut K/V memory traffic ~3x over a naive per-query-head walk. See the
  [KV-cache ADR](docs/architecture/260914-kv-cache.md) and its
  [benchmark](experiments/kv-layout-benchmark.md) (+4.7% at short context, −24% wall-clock at a
  1,900-token prompt vs. the original contiguous cache).
- **Logits processing**: `ILogitsProcessor`/`BannedSequenceLogitsProcessor` — bans literal token
  *sequences* from ever being generated, enforced on every decode step including the default
  greedy path. See the [logits-processing ADR](docs/architecture/260917-logits-processing.md).
- **Sampling**: composable temperature / top-k / top-p pipeline (`InferenceEngine.Engine/Sampling/`).
- **CLI**: streamed token output, `--stats`, `--debug-tokenize`, `--debug-logits`, `--ban-words`,
  `.env` config.

On this dev machine (Apple M4 Pro, Release build), SmolLM2-135M-Instruct-F16 generates at
**~82–95 tok/s decode** at short context with Accelerate (Phase 1), against **~34–37 tok/s** on
the managed loop it replaced (64 tokens, short prompt, 3–4 runs each, via the CLI `--stats` line).
Set `INFERENCE_MATVEC=managed` to run the old loop for an A/B comparison. Only the short prompt
was re-measured for Phase 1; the longer-context numbers in
[experiments/kv-layout-benchmark.md](experiments/kv-layout-benchmark.md) predate Accelerate.

### Test coverage

**150 test cases, 0 failing.** On macOS, with `INFERENCE_MODEL` set: 149 pass, 1 skip; without it:
145 pass, 5 skip. On Linux (the CI runner), the 23 Accelerate-only cases skip: 127 pass, 23 skip
with the model set; 123 pass, 27 skip without it. One xUnit v3 project per `src/` project:

| Project | Test cases (incl. `[Theory]` rows) |
|---|---|
| `InferenceEngine.Core.Tests` | 4 |
| `InferenceEngine.Tokenizers.Tests` | 8 |
| `InferenceEngine.Cli.Tests` | 27 (incl. process-level exit-code tests for bad `--model` paths) |
| `InferenceEngine.Models.Tests` | 67 (incl. a bit-exact FNV-1a golden-logit-hash regression test against the real model, pinned to the managed loop, and Accelerate-vs-managed tolerance and greedy-token tests) |
| `InferenceEngine.Engine.Tests` | 44 |

Run with `dotnet test` (do not pass `-nologo`: it breaks the Microsoft.Testing.Platform runner).
Four `Models.Tests` cases (the two golden-logit baselines and the two Accelerate end-to-end
generation tests) additionally need `INFERENCE_MODEL=<path.gguf>` set to a real GGUF file — they
skip cleanly, not fail, when it isn't. The 23 Accelerate-specific cases skip off macOS. See each test project's own `README.md` for the business scenarios covered.

### Architecture decisions

| ADR | Status |
|---|---|
| [Solution and project layout](docs/architecture/260811-solution-and-project-layout.md) | accepted |
| [Project challenges and how to address them](docs/architecture/260901-project-challenges-and-how-to-address-them.md) | accepted |
| [Paged + head-major KV-cache](docs/architecture/260914-kv-cache.md) | accepted |
| [Logits processing and sampling pipeline](docs/architecture/260917-logits-processing.md) | accepted |
| [Attention & transformer](docs/architecture/260917-attention-and-transformer.md) | accepted |
| [Apple Silicon acceleration](docs/architecture/261101-apple-silicon-acceleration-strategy.md) | proposed — **Phase 1 (Accelerate, CPU) implemented; Phase 2 (MLX/Metal GPU) not started** |
| Model format loading, tokenization | still `planned` placeholders — each is already implemented in code, the ADR write-up just hasn't caught up |
| HuggingFace model acquisition, performance baseline | still `planned` placeholders, and genuinely not started — models are fetched via the dotLLM CLI as a stopgap, not a real download path of our own, and there's no formal performance-baseline methodology yet beyond the ad hoc measurements in `experiments/` |

### Known limitations

- **Only F32/F16 GGUF tensors load.** Every quantized type (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, …)
  parses as metadata/shape but throws `NotSupportedException` on read. Dequantization math is
  already researched ([docs/investigation/gguf-format-research.md](docs/investigation/gguf-format-research.md))
  but not implemented.
- **Only `general.architecture == "llama"` loads** — Mistral/Qwen/Phi-family GGUF files are
  rejected today, even though they're structurally similar.
- **Single-token forward pass only** — prefill is a sequential loop of single-token calls, not a
  batched matmul. Fine for short prompts; for a long prompt this is the dominant cost (see the
  KV-cache benchmark: ~29s for a ~1,900-token prompt + 32 decode tokens on the 135M model).
- **Weights are copied, not zero-copy** — `LlamaWeights` materializes every tensor into a managed
  `float[]` at load, ~2x memory versus the on-disk F16 size.

### Path to running bigger HuggingFace models

Yes — **GGUF**, matching llama.cpp/HuggingFace's own local-inference ecosystem, not ONNX or
safetensors. Where each size class stands:

- **Tiny models (≤ ~500M params): already possible today**, no code changes needed — point
  `--model` at any **F16 (or F32) GGUF file of a Llama-architecture model** from HuggingFace (e.g.
  a `bartowski/*-GGUF` repo's `-f16.gguf` file, when one is published alongside the quantized
  variants). SmolLM2-135M-Instruct is the proof; SmolLM2-360M in F16 should work unmodified,
  memory and CPU speed permitting.
- **Medium models (roughly 1B–8B params): blocked mainly on one thing** — quantized tensor
  dequantization. TinyLlama-1.1B is the practical edge case here: an F16 GGUF of it would already
  load today (same "already possible" mechanism as the tiny models above, just at ~2.2 GB instead
  of a few hundred MB), but almost every GGUF repo on HuggingFace above a few hundred million
  parameters publishes Q4_K_M/Q5_K_M/Q6_K/Q8_0 as the practical download rather than F16; full
  F16/F32 files for a 7–8B model specifically are large (14–16 GB) and often not published at all.
  Q4_K/Q6_K/Q8_0 dequant math is already worked out in the GGUF research notes — implementing it
  (following the same research → ADR → implementation → benchmark cycle the KV-cache work just
  went through) is the natural next milestone, and the one that actually unlocks "download a
  model from HuggingFace and run it" for anything beyond the current tiny test model. Zero-copy
  mmap reads and
  multi-architecture support (Mistral/Qwen/Phi) would matter more at this size too, but are
  secondary to quantization.

This project has no sprint schedule or committed dates (see AGENTS.md's working style — small,
reviewable steps, not a roadmap with deadlines), so the honest answer to "when" is "whenever the
quantization ADR and implementation land," not a calendar date.

See [docs/investigation/status.md](docs/investigation/status.md) for the investigation-phase
history and [CHANGELOG.md](CHANGELOG.md) for the detailed change-by-change log.


## References & Inspiration

### Primary Reference

- **[dotLLM](https://github.com/kkokosa/dotLLM)** by Konrad Kokosa — a ground-up LLM
  inference engine in pure C#/.NET 10. Not a wrapper around llama.cpp. Reaches 66–88%
  of llama.cpp decode throughput on CPU. Supports Llama, Mistral, Phi, Qwen, DeepSeek.
  Our architecture is inspired by its layered design (Core → Models/Tokenizers → Engine).
  - Blog post: [Introducing dotLLM](https://kokosa.dev/blog/2026/dotllm/)
  - Foundational posts: [Logits, logprobs, and temperature](https://kokosa.dev/blog/2026/temperature/),
    [Visualizing logprobs](https://kokosa.dev/blog/2026/logprobs/)

### Other C# Projects Studied

- **[LLamaSharp](https://github.com/SciSharp/LLamaSharp)** — mature P/Invoke wrapper
  around llama.cpp. High-level API patterns (executors, context management). Not a learning
  tool for internals, but useful for API design reference.
- **[ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai)** — Microsoft's
  ONNX-based inference with generate loop. Different format (ONNX, not GGUF).
- **[Microsoft.ML.Tokenizers](https://www.nuget.org/packages/Microsoft.ML.Tokenizers)** —
  standalone .NET tokenizer library (BPE, SentencePiece, Tiktoken). Candidate for our
  tokenizer implementation.

### Background Reading

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — the C/C++ inference engine
  that defined the GGUF format and local LLM inference
- [GGUF format specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [Implementing BPE from Scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html) — Sebastian Raschka
- [HuggingFace Tokenizer Summary](https://huggingface.co/docs/transformers/tokenizer_summary)
- [KV-Cache Explained](https://www.emergentmind.com/topics/kv-cache)
- Konrad Kokosa, *Pro .NET Memory Management* (2nd ed.) — relevant for understanding
  GC-free inference and native memory patterns in .NET


## Project Docs

- [AGENTS.md](AGENTS.md) — ground rules for any AI coding assistant (stack, `unsafe` policy,
  ADR process)
- [CLAUDE.md](CLAUDE.md) — Claude Code–specific config (subagents, skills)
- [.github/workflows/tests.yml](.github/workflows/tests.yml) — CI: restores, builds, and runs
  `dotnet test` on every push to `main` and every PR
- **Architecture decisions** ([docs/architecture/](docs/architecture/),
  [MADR](https://adr.github.io/madr/) format) — see the
  [Architecture decisions](#architecture-decisions) table above for every ADR and its status.
- **Investigation notes** ([docs/investigation/](docs/investigation/)):
  - [overview.md](docs/investigation/overview.md) — goals, references, topic map
  - [status.md](docs/investigation/status.md) — live done/next tracker, including the Phase B
    implementation plan for logits processing
  - [dotllm-architecture-trace.md](docs/investigation/dotllm-architecture-trace.md) — dotLLM's own
    architecture, traced
  - [gguf-format-research.md](docs/investigation/gguf-format-research.md) — GGUF binary format,
    quantization types
  - [huggingface-ecosystem.md](docs/investigation/huggingface-ecosystem.md) — GGUF providers, Hub
    API, download options
  - [inference-engine-project-layouts.md](docs/investigation/inference-engine-project-layouts.md) —
    how other engines structure their projects
  - [kv-cache-research.md](docs/investigation/kv-cache-research.md) — the five eras of KV-cache
    design, this engine's choice
  - [phase2-mlx-handoff.md](docs/investigation/phase2-mlx-handoff.md) — start-here doc for the
    Phase 2 MLX/GPU spike (not started)
- **Business scenarios** ([docs/scenarios/](docs/scenarios/), Gherkin `.feature`, one per `src/`
  project): [Core](docs/scenarios/InferenceEngine.Core.feature),
  [Models](docs/scenarios/InferenceEngine.Models.feature),
  [Tokenizers](docs/scenarios/InferenceEngine.Tokenizers.feature),
  [Engine](docs/scenarios/InferenceEngine.Engine.feature),
  [Cli](docs/scenarios/InferenceEngine.Cli.feature)
- **Benchmarks** ([experiments/](experiments/)):
  [reference-measurements-dotllm.md](experiments/reference-measurements-dotllm.md),
  [kv-layout-benchmark.md](experiments/kv-layout-benchmark.md)
- [CHANGELOG.md](CHANGELOG.md) — notable changes, change by change
