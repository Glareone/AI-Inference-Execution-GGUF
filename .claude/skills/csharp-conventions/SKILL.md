---
name: csharp-conventions
description: Per-project conventions, ownership boundaries, and gotchas for the five InferenceEngine.* projects (Core, Models, Tokenizers, Engine, Cli), the KV-cache's paging/head-major contract, and the mandatory context7 library-check step. Use when writing, reviewing, or reasoning about C#/.NET code in this repository.
paths: "src/**/*.cs,tests/**/*.cs,**/*.csproj"
---

# C# project conventions — InferenceEngine

Reference for working in any `InferenceEngine.*` project. Read the project's own row below before
touching its code — each project has a narrow, deliberate responsibility (see the
[solution-layout ADR](../../../docs/architecture/260811-solution-and-project-layout.md)), and the
most common mistake is drifting outside it (e.g. `Cli` reaching into `Models` directly, or `Core`
picking up a dependency).

## `InferenceEngine.Core`

Leaf project — zero dependencies, everything else depends on it. Owns `IModel`, `ITokenizer`,
`IKvCache`, `ModelConfig`, `TokenizerData`.

- No logic belongs here, only contracts and value types. If you're tempted to write a method body
  beyond a trivial expression, it belongs in `Models` or `Engine` instead.
- A signature change here ripples into `Models`, `Engine`, and `Tokenizers` at minimum — grep for
  every call site before changing one, not just the ones you already know about.

## `InferenceEngine.Models`

Owns GGUF parsing (`Gguf/`), the Llama forward pass (`Llama/LlamaModel.cs`, `Llama/GqaAttention.cs`),
and math primitives (`Math/Ops.cs`).

- **Only NuGet dependency: `System.Numerics.Tensors`** (`TensorPrimitives`), pinned to `10.0.12` in
  `InferenceEngine.Models.csproj`. No custom SIMD/intrinsics, no hand-rolled kernels — if
  `TensorPrimitives` doesn't have the op you think you need, check context7 against that exact
  version before assuming you have to hand-roll it.
- **Single-token forward pass only** — there is no batched/multi-token path. Don't build one
  without confirming with the user first; it's a real architectural change that needs its own ADR.
- **Only F32/F16 GGUF tensors materialize.** Every quantized `GgmlType` (Q4_0, Q4_K, Q6_K, Q8_0,
  …) throws `NotSupportedException` from `GgufFile.ReadTensorAsF32` — they parse as metadata/shape
  but can't be read as data yet. Dequantization math for Q4_K/Q6_K/Q8_0 is already researched
  (`docs/investigation/gguf-format-research.md`) but unimplemented. Don't assume a Q4_K_M model
  "just works" — it won't, today.
- **Only `general.architecture == "llama"` loads** — anything else throws `NotSupportedException`
  from `LlamaModel.LoadFromGguf`.
- Weights are copied into managed `float[]` at load, not zero-copy over the mmap — ~2x memory vs.
  the on-disk F16 size. That's a documented, deliberate trade-off (see `LlamaWeights.cs`'s doc
  comment and the format-loading ADR), not a bug to silently fix mid-task.
- **Bit-identity constraint on the attention/KV write path.** Any change to loop order, RoPE
  application, or per-head math must reproduce the exact same floating-point operations in the
  exact same accumulation order as before — float addition isn't associative, so a "harmless"
  reorder can silently change output. Validate with the FNV-1a golden-logit-hash pattern already
  established in `tests/InferenceEngine.Models.Tests/Llama/{LogitHash,GoldenLogitBaselineTests,
  GqaAttentionTests}.cs` (see the [kv-cache ADR](../../../docs/architecture/260914-kv-cache.md) for
  why) before calling a hot-path change to attention/RoPE/KV done — reuse that established
  technique rather than inventing tolerance-based comparison for a new change.

## `InferenceEngine.Tokenizers`

Owns BPE tokenization sourced from the GGUF file's own embedded vocab/merges
(`GgufBpeTokenizer.cs`, `ByteLevelAlphabet.cs`) — hand-rolled, no NuGet tokenizer package yet,
despite `Microsoft.ML.Tokenizers` being named a candidate in AGENTS.md. The tokenization ADR
(`docs/architecture/planned-tokenization.md`) is still an unwritten placeholder. If you're about to
change tokenizer strategy or add a library dependency here, that decision needs an ADR first (hand
off to the `adr-author` agent) — don't just add the package.

## `InferenceEngine.Engine`

Owns orchestration: `InferenceSession` (the facade the CLI drives), the KV-cache implementation
(`KvBlockPool`, `PagedKvCache`), the sampling pipeline (`Sampling/`), chat prompt assembly
(`Prompting/`). The only project that knows about both `Models` and `Tokenizers` at once.

**KV-cache specifics** (full detail in the
[kv-cache ADR](../../../docs/architecture/260914-kv-cache.md)):

- 32-token blocks (power of two), head-major layout `[layer][kvHead][slot][headDim]`.
- `Reserve(position)` must be called before `KeySlot`/`ValueSlot`, or before reading a
  not-yet-assigned block — it throws `InvalidOperationException` otherwise, by design (a caller
  bug, not a cache bug to work around).
- Freed blocks are never zeroed. Stale data past `Length` is a deliberate, tested invariant —
  don't "defensively" zero it, that would mask a real bug if one ever reads past `Length`.
- `Reset()`/`Rollback()` are documented seams with **no production caller**. Don't wire them into
  `InferenceSession` speculatively — they're there for multi-turn reuse and speculative decoding,
  neither of which this engine does yet.
- The ADR names specific seams for ref-counting, copy-on-write, prefix-block-hashing,
  sliding-window eviction, and KV quantization — each scoped to one future method/field. Don't
  implement any of them without the user asking; "it seemed useful" isn't the bar here.

## `InferenceEngine.Cli`

Owns the process entry point only: arg/env parsing (`CliOptions`), console I/O. Deliberately does
**not** reference `Models` or `Tokenizers` directly — everything routes through the `Engine` facade.
Don't add a direct reference here even for a debug flag that looks like a shortcut; extend
`InferenceSession`/`Engine` instead.

## Cross-cutting

- `AllowUnsafeBlocks=false` repo-wide (`Directory.Build.props`). `unsafe` needs the
  justify-and-confirm step from AGENTS.md before it's written — every time, not just the first time.
- `TreatWarningsAsErrors=true` — a new compiler warning breaks the build. Fix the warning; don't
  reach for a suppressing pragma without saying why in the response.
- One xUnit v3 test project per `src/` project (`tests/InferenceEngine.<Name>.Tests/`), each with
  its own `README.md` documenting the business scenarios it covers by name — match that format
  when adding scenarios. Hand test writing/running off to `test-writer-runner`, which always runs
  `dotnet test` and confirms green before reporting done.

## Context7 — mandatory, not just for "investigating"

- Before writing code that calls `TensorPrimitives`/`System.Numerics.Tensors`, any other NuGet
  package, or a .NET BCL API you're not certain about: `mcp__context7__resolve-library-id` then
  `mcp__context7__query-docs`, every time. Training data can be stale or describe a different
  package version than what's actually pinned — check the real version in the `.csproj` first
  (e.g. `System.Numerics.Tensors` is `10.0.12`, not "whatever's current"), then verify against
  that exact version.
- Before recommending a **new** package — the "reuse a library" default from AGENTS.md's Stack
  policy — use context7 to confirm it's actually maintained and does what you think. Don't
  recommend from memory alone; a stale recommendation is worse than no recommendation.

## Serena — symbol navigation, not grep

- To find where a type, method, or field is declared or used, use `mcp__serena__find_symbol` /
  `find_referencing_symbols` / `get_symbols_overview` — not `Grep`. Grep matches text; Serena
  matches symbols through the Roslyn language server, so it won't confuse a comment, a string
  literal, or a same-named member on a different type. Keep `Grep` for non-symbol searches (a
  TODO, a literal, a `.csproj` property).
- After a large edit or a `.csproj`/solution change, if symbol results look stale, call
  `restart_language_server` before trusting the next query.
- Serena's editing/file-mutation tools (`replace_symbol_body`, `insert_after_symbol`,
  `create_text_file`, etc.) are off by design in the `claude-code` context (see `.mcp.json`) —
  edits still go through `Edit`/`Write` as before, not through Serena.
