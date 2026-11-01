# AGENTS.md

Ground rules for any AI coding assistant working in this repository (Claude Code, Codex,
Qwen Code, opencode, or otherwise). See [README.md](README.md) for the project's purpose
and inspiration.

## What this project is

A guided investigation into how LLM inference engines work, in C#/.NET — inspired by and
explicitly benchmarked against [dotLLM](https://github.com/kkokosa/dotLLM). We are not
trying to out-build dotLLM. We're building a smaller version to *understand* the pieces
(model loading, tokenization, attention, sampling, KV-cache, serving), reusing existing
libraries wherever that's reasonable.

**Apple Silicon-centric.** This project targets M-series Macs running macOS. We have no plans to
support Windows or Intel. Platform-specific choices (Apple Accelerate, MLX) are fine and
expected. Do not add cross-platform abstractions or non-Apple code paths for their own sake.
See [ADR 261101](docs/architecture/261101-apple-silicon-acceleration-strategy.md).

## Stack

- Target **.NET 10** (latest LTS as of writing) unless a specific dependency forces
  otherwise — note the reason in an ADR if so.
- Prefer well-maintained NuGet packages over reimplementing math/tokenization/model-parsing
  kernels. The point of this project is understanding the *architecture* of an inference
  engine, not re-deriving BLAS or a BPE tokenizer from first principles.
- Only reach for a hand-rolled implementation when no reasonable library exists, or when the
  point of the current work *is* to learn that specific piece by building it. Say which,
  explicitly, when you do it.
- For symbol navigation (find a declaration, find all references), prefer the Serena MCP
  server over text search if your assistant supports MCP — see [.mcp.json](.mcp.json).

## `unsafe` code policy

`unsafe` / raw pointers / unmanaged memory are allowed only in narrow, explicitly justified
cases (e.g. a tensor buffer on a proven hot path). Before writing `unsafe` code:

1. State *where* it's needed and *why* a managed alternative isn't good enough (with a
   number, if it's a perf claim — not a guess).
2. Get explicit confirmation from the user before landing it. Don't introduce `unsafe` blocks
   speculatively "in case we need the speed later."
3. Keep the `unsafe` surface as small as possible — wrap it behind a safe API.

**Approved exception:** `InferenceEngine.Models` sets `AllowUnsafeBlocks=true` in its own
`.csproj` (not in `Directory.Build.props`). The `LibraryImport` source generator requires it
(`SYSLIB1062`) for the Apple Accelerate / MLX P/Invoke declarations
([ADR 261101](docs/architecture/261101-apple-silicon-acceleration-strategy.md)). Every other
project stays `AllowUnsafeBlocks=false`. The flag does not allow hand-written `unsafe` blocks;
those still need the three steps above. See [.claude/rules/unsafe-policy.md](.claude/rules/unsafe-policy.md).

## Architecture decisions

Non-trivial design decisions (tokenizer library choice, GGUF loading strategy, tensor memory
model, KV-cache design, etc.) get an ADR in [docs/architecture/](docs/architecture/), using the
[MADR](https://adr.github.io/madr/) format (template at `docs/architecture/template.md`; ADR files
are named `YYMMDD-<slug>.md`, and a not-yet-written ADR may exist as a `planned-<slug>.md`
placeholder until it's completed). Every ADR ends with a "Decision Log" table (see the template) —
never skip it. Write or review ADRs with the `adr-author` agent if using Claude Code (see
[.claude/agents/](.claude/agents/)); otherwise just follow the template directly.

## Investigation workflow

Before implementing a component, research it first using the investigation agents
(`researcher`, `huggingface-explorer`, `code-reader`). Document findings in
`docs/investigation/`, then write an ADR (MADR format, in `docs/architecture/`) capturing both
the understanding and the implementation decision. This ensures we understand *why*
things are designed the way they are, not just *how* to call an API.

Reference materials and external projects:
- **[dotLLM](https://github.com/kkokosa/dotLLM)** — primary reference (pure C# inference engine)
- **[LLamaSharp](https://github.com/SciSharp/LLamaSharp)** — API design patterns
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — GGUF format origin, algorithms
- **[Microsoft.ML.Tokenizers](https://www.nuget.org/packages/Microsoft.ML.Tokenizers)** — tokenizer candidate
- **[Apple Accelerate](https://developer.apple.com/documentation/accelerate)** — CBLAS/vDSP, the Phase 1 CPU math path
- **[MLX](https://github.com/ml-explore/mlx)** / **[mlx-c](https://github.com/ml-explore/mlx-c)** — Apple's array framework and its C API, the Phase 2 GPU path

Test model: SmolLM2-135M-Instruct (bartowski Q4_K_M, 101 MB) — installed via dotLLM CLI.

## Working style

- Small, reviewable steps over big-bang implementations. Document the plan before writing
  a nontrivial chunk of code.
- Update [CHANGELOG.md](CHANGELOG.md) for user-visible or architecturally notable changes.
- Don't add abstractions, config flags, or error handling for cases that can't happen yet.
  This is an investigation project — prefer the simplest thing that demonstrates the concept
  correctly, and note follow-ups instead of speculatively building for them.
- **Never `git commit` (or `push`) without the user asking for that commit, every time** — not
  just once at the start of a multi-step task. Finish the work, leave it staged/unstaged in the
  working tree, and say what changed and why; let the user review the diff and explicitly say
  "commit this" before it happens. This applies to subagents too — when delegating work that
  touches files, don't instruct a subagent to commit on your own initiative; have it report back
  with what it changed instead, and commit (or ask the subagent to) only after the user confirms.
