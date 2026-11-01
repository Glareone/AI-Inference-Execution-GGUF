# Phase 2 handoff: MLX / Metal GPU spike

Written 2026-11-01 for a fresh session. Read this, then do the spike. Do not start the redesign.

## Where we are

- **Phase 1 is done.** `Ops.MatVec` calls Apple Accelerate `cblas_sgemv` on the **CPU**.
  Measured: ~34-37 tok/s before, ~82-95 tok/s after (SmolLM2-135M F16, M4 Pro, Release,
  64 tokens). Output text identical.
- **Phase 2 is not started. The engine does not use the GPU or Metal.**
- Decision and reasoning: [261101-apple-silicon-acceleration-strategy.md](../architecture/261101-apple-silicon-acceleration-strategy.md).
  Ollama/llama.cpp/MLX research is summarized there.
- 150 tests, 0 failing. On macOS: 5 skip without a model file, 1 skips with it. On Linux CI the 23
  Accelerate-only cases skip as well (127 pass / 23 skip with the model, 123 / 27 without).

## Goal of the next session

**A throwaway spike that answers one question: does MLX on the GPU beat Accelerate on the CPU
for our workload, and by how much?** Output is numbers plus a go/no-go recommendation. No
interface changes. No `IModel`/`IKvCache` edits. No product code in `src/`.

## Rules for this repo (easy to miss)

- Never `git commit`/`push` unless the user asks for that commit. Subagents too.
- ADRs: write or edit them with the `adr-author` agent. Code: `csharp-dotnet`. Tests:
  `test-writer-runner`. Library docs: context7 before calling any API.
- Apple Silicon only. No Windows/Intel code paths.
- `AllowUnsafeBlocks=true` is set in `InferenceEngine.Models` **only**
  (`.claude/rules/unsafe-policy.md`). A spike project that needs the flag for `LibraryImport`
  is another project: **ask the user first**, or keep the spike out of the solution.
- Run tests with plain `dotnet test`. **Do not pass `-nologo`**: in MTP mode it is forwarded to
  the test app and every project reports "Zero tests ran".
- CI is `ubuntu-latest`. Any MLX reference or test must be skipped or conditional off macOS, or
  restore/test breaks there.
- Caveman style for docs and reports (`.claude/skills/caveman`).

## Spike plan

1. **Pick bindings to try** (all from a research pass, **not verified by us**):
   - `Itexoft.MLX` NuGet (reported v0.31.2, .NET 10, bundles native dylibs, small download count).
   - `ManagedCode.MLXSharp` (reported v0.0.1, very early).
   - Raw `mlx-c` via our own `LibraryImport`. The C API has had breaking changes between
     releases (reported 0.5 -> 0.6): pin the version.
   - Verify package existence, versions, license, and arm64 dylib via NuGet and context7 first.
2. **Smoke test:** create two arrays, `matmul`, evaluate, read back, compare to a managed
   reference.
3. **Measure matrix-vector latency** (batch of 1, one token):
   - Shapes, SmolLM2: 576x576, 1536x576, 576x1536, 49152x576 (LM head).
   - Shapes, a ~1B model (Llama-3.2-1B-like: hidden 2048, FFN 8192, vocab 128256; **verify**
     against the real config before using).
   - Compare against `Ops.MatVec` (Accelerate) and `Ops.MatVecManaged` in the same harness.
   - F32 and F16 weights.
4. **Measure the realistic pattern, not one op.** MLX is lazy: ops record into a graph and run
   at `eval`. Chain one layer's worth of ops (or the full 211-matvec token) and call `eval`
   once. Timing a single op with an `eval` each hides the batching MLX is designed for. Always
   evaluate/sync **before** stopping the clock, or you time nothing.
5. **Report** a table, the binding choice, and a go/no-go.

Budget check (our reasoning, not measured): SmolLM2 has 30 layers x 7 matvecs + 1 LM head = 211
matvecs per token. At ~85 tok/s that is ~11.8 ms per token, so ~56 us per matvec on average.
GPU dispatch/sync overhead per op can eat that on a 135M model. Expect a larger model to be
where the GPU wins. Confirm, do not assume.

## What a redesign would touch (for the later ADR, not this session)

- `IModel.Forward(int tokenId, int position, IKvCache kvCache, bool needLogits) -> ReadOnlySpan<float>`
  (`src/InferenceEngine.Core/IModel.cs`). Can stay if the GPU model copies logits to a managed
  buffer (`VocabSize * 4` bytes per token).
- `IKvCache` (`src/InferenceEngine.Core/IKvCache.cs`) is the real blocker. It hands out CPU
  spans: `KeySlot`/`ValueSlot` (`Span<float>`), `KeyBlockForHead`/`ValueBlockForHead`
  (`ReadOnlySpan<float>`), plus `Reserve`, `Rollback`, `Reset`, `Length`, `Capacity`,
  `BlockSize`, `HeadDim`. A GPU-resident cache cannot honor span accessors the same way.
  Options: a separate GPU cache interface, or an MLX-owned cache that `IKvCache` only tracks.
  Unified memory may allow shared buffers (unverified).
- `GqaAttention` (`src/InferenceEngine.Models/Llama/GqaAttention.cs`) walks the cache block by
  block; MLX has its own attention primitive.
- `LlamaWeights` (`src/InferenceEngine.Models/Llama/LlamaWeights.cs`) holds `float[]`, F16
  expanded to F32 at load. An MLX path could load F16 directly.
- Backend selection point: `InferenceSession.Load` (`src/InferenceEngine.Engine/InferenceSession.cs`,
  calls `LlamaModel.LoadFromGguf`), `CreateKvCache` (builds `PagedKvCache` directly,
  `KvBlockSize = 32`), and three `_model.Forward` call sites.
- CPU path stays: it is the test oracle and the non-GPU backend. Every future model feature is
  written twice or deliberately CPU-only.
- Test shape for the GPU path: top-1 / greedy-token agreement and top-k overlap vs CPU, never
  bit-exact. See the Test plan in the ADR.

## Numbers and commands to reuse

Model: `~/.cache/inference-engine/models/SmolLM2-135M-Instruct-f16.gguf` (F16, 270 MB).

```sh
# speed (prints "generated N tok in Xs (Y tok/s)")
dotnet run -c Release --project src/InferenceEngine.Cli -- \
  --model ~/.cache/inference-engine/models/SmolLM2-135M-Instruct-f16.gguf \
  --prompt "What is machine learning? Explain briefly." --max-tokens 64 --stats

# same, CPU managed loop instead of Accelerate (A/B)
INFERENCE_MATVEC=managed dotnet run -c Release --project src/InferenceEngine.Cli -- ...

# full suite, with the golden + end-to-end tests actually running
INFERENCE_MODEL=$HOME/.cache/inference-engine/models/SmolLM2-135M-Instruct-f16.gguf dotnet test
```

Repeat each timing at least 3 times. Run-to-run noise was ~82-95 tok/s on Accelerate.

## Traps

- Golden logit hash tests are bit-exact and pinned to the managed loop with
  `Ops.ForceManagedMatVec()`. Do not "fix" them by re-baselining to Accelerate or GPU output:
  BLAS and GPU sum in different orders and differ by chip.
- Lazy evaluation moves errors and cost to `eval`. Profile with explicit sync points.
- Only the short prompt was benchmarked end to end. Long-prompt (multi-block KV) was not.
- CPU and GPU share memory bandwidth (~273 GB/s on M4 Pro, research pass). Decode is
  bandwidth-bound, so GPU gains on small models may be small.
- Our weights are F32/F16. Published MLX numbers are mostly Q4. Not like for like.
- Itexoft/MLX figures (versions, downloads, tok/s, ABI history) came from a web research pass.
  Treat as leads to verify.

## Hand back at the end

- The latency table (op, shape, dtype, ms, path), 3+ runs each.
- Which binding worked, with exact version and any native-lib or CI implications.
- Go/no-go for the redesign, with the number that justifies it.
- If go: open questions for the Phase 2 ADR (cache contract, weight loading, backend selection).
- Anything in this doc that turned out wrong.

## Prompt to start the session

> Read docs/investigation/phase2-mlx-handoff.md and docs/architecture/261101-apple-silicon-acceleration-strategy.md.
> Run the Phase 2 MLX spike described there. Throwaway code only, nothing in src/, no commits.
> Check with me before adding AllowUnsafeBlocks to any new project.
