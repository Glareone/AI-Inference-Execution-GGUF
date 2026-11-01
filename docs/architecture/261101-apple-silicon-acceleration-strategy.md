# Apple Silicon acceleration strategy: Accelerate now, MLX next

- Status: proposed
- Date: 2026-11-01

## Executive summary

**We are moving to MLX for GPU execution, in two phases. Today only Phase 1 (CPU) exists; the
engine does not use the GPU yet.**

- **Phase 1 - Accelerate (CPU).** Replace the `Ops.MatVec` loop with Apple Accelerate
  `cblas_sgemv` via P/Invoke. Contained in `Models`. No interface change.
- **Phase 2 - MLX (GPU).** Call MLX through its C API (`mlx-c`), via the `Itexoft.MLX` NuGet
  package or our own P/Invoke. C# keeps model loading, tokenization, KV-cache policy, sampling,
  and the generation loop. MLX runs tensor math. Needs an `IModel`/`IKvCache` redesign and its
  own ADR before code.
- **Why MLX, not raw Metal.** MLX supplies kernels, unified-memory handling, fusion, and
  command-buffer batching. Hand-writing that is months of GPU work.
- **Measured so far:** managed loop ~34-37 tok/s, Accelerate ~82-95 tok/s (SmolLM2-135M-F16, M4
  Pro; see Phase 1 results). Every other number here is from a research pass, not measured. A 135M
  model may gain little from a GPU. Phase 2 gates on a benchmark.

**Current status (2026-11-01)**

- **Phase 1 (Accelerate, CPU) is implemented and tested.** Go/no-go gate passed.
- **Phase 2 (MLX / Metal GPU) is NOT started. The engine does not use the GPU or Metal today.**
  Phase 1 accelerates CPU matrix-vector math only.
- Phase 2 next step: the spike (see Implementation plan). Handoff doc:
  [phase2-mlx-handoff.md](../investigation/phase2-mlx-handoff.md).

## Context and Problem Statement

Before Phase 1 the engine decoded at ~30-37 tok/s (measured). All math went through `TensorPrimitives`
(auto-vectorized to NEON) in `src/InferenceEngine.Models/Math/Ops.cs`. `MatVec` (`Ops.cs:25-31`)
is a loop of `TensorPrimitives.Dot`, one per output row. The forward pass is single-token, F32
([260917](260917-attention-and-transformer.md)).

[260901](260901-project-challenges-and-how-to-address-them.md) rated CUDA kernels "Skip" and chose
"reuse a math library, no custom kernels." This project targets Apple Silicon (M-series Macs,
macOS) only: no CUDA, but a GPU that shares memory with the CPU. **How do we use that GPU without
writing kernels and without giving up the parts of inference we built to understand?**

Scope:

- **Phase 1:** `InferenceEngine.Models` only. One new file, one changed method.
- **Phase 2:** `Core` contracts, `Models` (second `IModel`), `Engine` (cache, session init).
  Added as an alternative backend, not a rewrite.
- **Unchanged:** `Tokenizers`, sampling, generation loop, `Cli`, and the ownership split in
  [260811](260811-solution-and-project-layout.md).

## Decision Drivers

1. **Low-hanging fruit first.** Ship the cheapest, most contained win (Accelerate) before the
   expensive one (GPU redesign). Each step is small and reviewable.
2. **Leave the door open.** Phase 1 must not block Phase 2 MLX: no interface change, no choices
   that tie us to a CPU-only design.
3. **Keep the high-payoff pieces ours** (260901): forward-pass wiring, KV-cache policy, sampling,
   and the loop stay readable C#. Otherwise we are just wrapping a runtime.
4. **No hand-written kernels.** Highest effort, lowest payoff in 260901. Otherwise this becomes a
   GPU-programming project.
5. **`unsafe` only with justification** (AGENTS.md). Any interop that needs it is named and
   decided.
6. **Apple Silicon only.** M4 Pro, macOS. No Windows or Intel support goal.

## Considered Options

- **A - Accelerate (Phase 1) then MLX via C API (Phase 2).**
- **B - Accelerate only.** CPU-side, no GPU.
- **C - TorchSharp + MPS.** libtorch bindings, Metal Performance Shaders backend.
- **D - Raw Metal, hand-written kernels.** Our own MSL kernels called from C#.

Not applicable on Apple Silicon: **CUDA** (NVIDIA dropped macOS support in 2020), **OpenCL/ILGPU**
(OpenCL deprecated on macOS since 2018).

## Decision Outcome

Chosen option: **A - Accelerate in Phase 1, MLX in Phase 2.** Decisive reason: it is the only
option that reaches the Apple GPU with zero kernels from us, keeps orchestration in C#, and lets
Phase 1 ship alone (drivers 1-4).

Rejected: **B** stops short of the GPU. **C** wraps a ~500 MB native runtime (260901 rejects
that). **D** is months of MSL work (llama.cpp has 38 kernels).

**`unsafe` decision (user, 2026-11-01).** `AllowUnsafeBlocks=true` is set in
`InferenceEngine.Models.csproj` **only**. `Directory.Build.props` stays `false`, so `Core`,
`Engine`, `Tokenizers`, and `Cli` stay safe-only. We write **no `unsafe` blocks**. The flag exists
because the `LibraryImport` generator requires it (`SYSLIB1062`, verified via context7).

## How Ollama, llama.cpp, and MLX accelerate on Apple Silicon

Research pass (web), **not measured here**. Figures are community-reported.

- **Metal compute kernels.** llama.cpp (and Ollama on it) ships hand-written MSL kernels (matmul,
  flash attention, softmax, RoPE, dequant; 38 counted). MLX also uses Metal compute shaders, not
  MPS.
- **Unified memory.** CPU and GPU share pages through shared-storage `MTLBuffer`. No host-to-device
  copy of weights or KV.
- **Lazy evaluation and fusion (MLX).** Ops record a graph. `eval` fuses kernels and batches
  command buffers. llama.cpp builds a static graph per step.
- **Accelerate / matrix coprocessor.** Accelerate is the only supported route to it from .NET.
  M4 is reported to use SME, not AMX; sources conflict.
- **ANE not used** for LLM decode by any of these. Out of scope.
- **Reported results.** Ollama made MLX its default Apple Silicon backend in March 2026 (~2x
  throughput claimed). MLX reports ~80 tok/s on Qwen3-8B Q4 (M4 Pro). Other models, not ours.
- **Layering.** MLX = C++ core, `mlx-c` C API, language bindings. We bind at the C API.

## Consequences

Legend: 🟢 upside · 🟡 accepted trade-off · 🔴 downside.

### Phase 1 (Accelerate)

- `Math/Ops.cs`: `MatVec` calls `cblas_sgemv`.
- New `Math/AccelerateBlas.cs`: `LibraryImport` declaration.
- `InferenceEngine.Models.csproj`: `AllowUnsafeBlocks=true` (see Decision Outcome).
- `IModel`, `IKvCache`, `Engine`, `Cli`: no change. Accelerate reads the same `float[]` memory.
- The CPU `TensorPrimitives` loop stays as `Ops.MatVecManaged`: **test oracle, golden-hash path,
  and runtime fallback** when Accelerate is unavailable (see Deviations). It is not a
  Windows/Intel support goal.
- 🟡 The "no `unsafe`" claim in [260917](260917-attention-and-transformer.md) stops being true at
  the `Models` project-settings level. It stays true for hand-written code.
- 🟢 No hand-written pinning. Per context7 (`SpanMarshallers.md`, marked historical), the generator
  pins blittable `Span<float>` rather than copying. Tests confirm correctness with the generator.
  A no-copy check was not done: still unverified.
- 🟡 **Golden hash is not valid on the Accelerate path.** BLAS may accumulate in a different order
  than `TensorPrimitives.Dot`. The FNV-1a logit hash (`GoldenLogitBaselineTests.cs`) is bit-exact
  and stays pinned to the CPU path. On the Accelerate path, verify with float tolerance on
  `MatVec` and identical greedy tokens. This reverses 260914/260917 bit-identity for that path
  only.
- 🟢 **Speedup measured, Go/no-go gate PASSED.** See Phase 1 results. Caveat: one model, one
  machine. P/Invoke runs ~211 times per token on a 30-layer model (arithmetic, not measured).
- 🟢 **Library path verified.** `/System/Library/Frameworks/Accelerate.framework/Accelerate` loads
  and `cblas_sgemv` resolves. The plan's `libSystem.B.dylib` was never used or tested.
- 🟢 **CBLAS constants verified.** CblasRowMajor=101 and CblasNoTrans=111 with the classic 32-bit
  `int` entry point match the managed loop in tests.

#### Phase 1 results (measured: Apple M4 Pro, Release, SmolLM2-135M-Instruct-F16)

Prompt "What is machine learning? Explain briefly.", 64 tokens, CLI `--stats` line. Short prompt
only; the long prompt was not benchmarked.

| Path | tok/s runs | Range |
|------|------------|-------|
| Managed loop (before) | 36.9, 34.2, 33.6, 35.2 | ~34-37 |
| Accelerate `cblas_sgemv` | 82.2, 94.8, 84.6 | ~82-95 |

- Roughly 2.4-2.8x. Generated text identical on both paths (64 tokens).
- Tests: 32 greedy tokens identical, prefill top-1 identical, top-10 overlap >= 8.
- **Why faster is not measured.** Could be the tuned kernel, multithreading, or the matrix unit.
  Do not claim AMX/SME as the cause. AMX vs SME stays open.

#### Deviations from the Phase 1 plan

- **Not unconditional.** Plan said call Accelerate always and drop the OS check. Reality:
  `AccelerateBlas.IsAvailable` probes once (false off macOS) and `Ops.MatVec` falls back to
  `Ops.MatVecManaged`. CI runs on ubuntu-latest and tests must pass there. Not a Windows/Intel
  support goal.
- **Pinning the managed path.** Added `Ops.ForceManagedMatVec()` (AsyncLocal scope) and env var
  `INFERENCE_MATVEC=managed`. `GoldenLogitBaselineTests` pin the managed loop inside that scope, so
  the hash stays bit-exact and machine-independent. Accelerate correctness is covered by tolerance
  tests plus a greedy-token-equality test, not the hash.
- **Files:** `src/InferenceEngine.Models/Math/AccelerateBlas.cs` (new), `Ops.cs` (changed);
  tests `Math/AccelerateBlasTests.cs`, `Math/OpsMatVecTests.cs`, `Llama/AcceleratedGenerationTests.cs`
  (new). 147 tests: 146 pass + 1 skip with the model set; 142 pass + 5 skip without.
- **`global.json`** had lost its `"test": {"runner": "Microsoft.Testing.Platform"}` block (removed
  by commit 079ca5e). Restored.

### Phase 2 (MLX)

| Component | Impact |
|-----------|--------|
| `Core/IModel.cs` | `Forward` returns `ReadOnlySpan<float>`. Stays if MLX copies logits to a managed buffer (`VocabSize * 4` bytes/token), else a tensor handle. Phase 2 ADR picks |
| `Core/IKvCache.cs` | Span-returning members cannot be honored by a GPU-resident cache. Needs a new contract or an MLX-owned cache that `IKvCache` only tracks |
| `Models/Llama/*` | New MLX-composed `IModel`. CPU `LlamaModel` stays |
| `Engine/InferenceSession.cs` | Backend selection, MLX device/stream init |
| Build | New dependency (`Itexoft.MLX` or raw `mlx-c`), macOS arm64 only |

- 🟡 [260914](260914-kv-cache.md) (CPU spans, block tiles) stays correct for the CPU backend. Its
  bandwidth analysis does not carry to a GPU cache. Phase 2 ADR says what it keeps.
- 🟡 **Dual path is permanent.** The CPU `IModel` remains the test oracle. New model features are
  written twice or kept CPU-only.
- 🟡 **Tolerance/top-k comparison replaces bit-exact** for MLX. GPU float math differs.
- 🟡 **135M may not benefit.** Launch and sync cost can dominate tiny matmuls. Validate on SmolLM2
  *and* a larger model (1B-8B) before committing the redesign. A "200+ tok/s" guess from planning
  has no source.
- 🟡 **Quantization gap.** We load F32/F16 only. MLX headline numbers are Q4.

### Open risks

- 🔴 **`Itexoft.MLX` is young** (research pass: v0.31.2, .NET 10, ~2.4K downloads, one maintainer).
  Not verified by us.
- 🔴 **`mlx-c` ABI has had breaking changes** (research pass). Pin the version. Raw P/Invoke moves
  ABI tracking to us.
- 🔴 **MLX `unsafe` and lifetime surface.** Whether `Itexoft.MLX` needs `AllowUnsafeBlocks` is
  unknown. Handles need deterministic disposal.
- 🟡 **Lazy evaluation hides errors and timings until `eval`.** Profile with explicit sync points.
- 🟡 **AMX vs SME unresolved.** Affects how we explain Phase 1 results, not whether it works.

### Relationship to earlier ADRs

Narrowly supersedes 260901 "CUDA kernels: Skip" (GPU via *someone else's* kernels is now in scope)
and the "no `unsafe`" stance of 260901/260917 (project-level flag in `Models` only). Unchanged:
build-vs-reuse verdicts, the learning goal, the five-project layout, CPU path as oracle.

## Implementation plan

### Phase 1 - Accelerate (DONE except step 5)

1. **DONE.** **Spike (no product code).** Confirm the library path resolves `cblas_sgemv`. Measure
   `MatVec` before/after on real shapes. Go/no-go on the number.
2. **DONE.** **Set** `AllowUnsafeBlocks=true` in `InferenceEngine.Models.csproj` only.
3. **DONE.** **Create** `src/InferenceEngine.Models/Math/AccelerateBlas.cs`:

   ```csharp
   [LibraryImport("/System/Library/Frameworks/Accelerate.framework/Accelerate")]
   private static partial void cblas_sgemv(
       int order, int transA, int m, int n, float alpha,
       ReadOnlySpan<float> a, int lda,
       ReadOnlySpan<float> x, int incX,
       float beta, Span<float> y, int incY);
   ```

   Parameter order matches Apple's `cblas_sgemv`, checked via context7. The span-for-pointer
   mapping is ours. Call row-major, no-transpose, `m = outDim`, `n = inDim`, `lda = inDim`,
   `alpha = 1`, `beta = 0`. Use the classic 32-bit `int` entry point, not the ILP64 variants.
   Both confirmed (see Phase 1 results).
4. **DONE.** **Modify** `Ops.MatVec` to call Accelerate. Keep the loop (`MatVecManaged`) for tests,
   the golden hash, and fallback.
5. **NOT DONE.** **Update** the `Ops.MatVec` doc comment and the "Numerics" note in 260917
   (Decision Log row, separate commit). Still open.
6. **DONE (short prompt only).** **Benchmark** (`experiments/`). Numbers in Phase 1 results and the
   Decision Log. Long prompt not run.

No change to `Core`, `Engine`, `Cli`, other `.csproj` files.

### Phase 2 - MLX (outline; own ADR first)

1. **Spike:** run one `matmul` via Itexoft.MLX and raw `mlx-c`. Pick the binding. Measure per-op
   latency at SmolLM2 and larger-model shapes.
2. **Decide contracts:** `IModel.Forward` shape and the GPU KV-cache contract.
3. **Implement** an MLX-backed `IModel`, selected by `InferenceSession`.
4. **Validate** against the CPU path, then benchmark SmolLM2 and a larger model.
5. **Go/no-go** on keeping the redesign, from step 4 numbers.

## Test plan

### Phase 1 (tests done; long-prompt benchmark not run)

- **Unit (done):** Accelerate `MatVec` equals the `TensorPrimitives.Dot` loop within float
  tolerance on several shapes (include non-SIMD-width multiples and real model shapes).
- **Unit (done):** symbol resolves; hand-computed 2x3 case correct; result written in place.
- **Integration (done):** same greedy tokens as the CPU path (32 greedy tokens, prefill top-1,
  top-10 overlap >= 8). Golden FNV-1a hash stays pinned to the managed path.
- **Benchmark:** tok/s before/after. Short prompt done. Long prompt not run.
- Follow `test-writer-runner` conventions: `tests/InferenceEngine.Models.Tests`, business case
  documented, `dotnet test` green.

### Phase 2 (outline)

- Binding smoke test: create array, `matmul`, `add`, read back.
- MLX and CPU `Forward` agree on greedy tokens and top-k.
- GPU KV cache correct across positions, rollback, reset.
- Performance gate: MLX must beat CPU on the target model.
- Lifetime test: no MLX resource leak across repeated sessions.

## Pros and Cons of the Options

### A - Accelerate then MLX (chosen)

- 🟢 Reaches the Apple GPU with zero kernels from us.
- 🟢 Phase 1 ships alone and stays as oracle.
- 🔴 Phase 2 crosses every layer and doubles the backend surface.
- 🔴 Young binding, unstable C ABI.

### B - Accelerate only

- 🟢 Smallest change, no GPU dependency.
- 🔴 Ceiling is CPU, F32, single-token.

### C - TorchSharp + MPS

- 🟢 Mature tensor API.
- 🔴 ~500 MB libtorch hides the inference stack.

### D - Raw Metal, hand-written kernels

- 🟢 Maximum control.
- 🔴 Months of MSL work; a different subject than inference.

## More Information

- [260901](260901-project-challenges-and-how-to-address-them.md) - build-vs-reuse rules.
- [260917](260917-attention-and-transformer.md) - forward pass, `Ops`, golden-hash discipline.
- [260914](260914-kv-cache.md) - CPU KV design Phase 2 must reconcile.
- **Verified via context7:** `cblas_sgemv` parameter order; `LibraryImport` span pinning;
  `SYSLIB1062`.
- **Verified by running code (M4 Pro):** Accelerate framework path loads, `cblas_sgemv` resolves;
  CBLAS constants 101/111 with the 32-bit `int` entry point; generator span pinning gives correct
  results.
- **Not verified (research pass or memory):** tok/s figures other than the Phase 1 results
  (including MLX/Ollama numbers); span pinning is no-copy; `Itexoft.MLX` details; `mlx-c` ABI
  history; Ollama MLX default and ~2x claim; 38-kernel count; AMX vs SME on M4; why Accelerate is
  faster.

## Decision Log

| Date       | Change                                                                                                                                                                                                              | By                 |
|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| 2026-11-01 | Initial proposal                                                                                                                                                                                                    | Aleksei Kolesnikov |
| 2026-11-01 | Simplified: merged scope and problem statement, numbered drivers (added low-hanging-fruit and keep-door-open), options limited to Apple Silicon (CUDA, OpenCL/ILGPU dropped to one line), compressed research and Consequences | Aleksei Kolesnikov |
| 2026-11-01 | `unsafe` scope decided: `AllowUnsafeBlocks=true` in `InferenceEngine.Models.csproj` only; `Directory.Build.props` stays `false`; no hand-written `unsafe`; CPU path kept as test oracle only, not a cross-platform fallback | Aleksei Kolesnikov |
| 2026-11-01 | Phase 1 implemented. Measured (M4 Pro, SmolLM2-135M-F16, short prompt, 64 tokens): managed 36.9/34.2/33.6/35.2 tok/s, Accelerate 82.2/94.8/84.6 tok/s (~2.4-2.8x). Text identical. Go/no-go gate passed. Cause of speedup unmeasured; AMX vs SME open. Library path, CBLAS constants verified. Phase 2 not started. Step 5 (260917 note) and long-prompt benchmark not done | Aleksei Kolesnikov |
| 2026-11-01 | Deviations from Phase 1 plan: `AccelerateBlas.IsAvailable` probe plus `Ops.MatVecManaged` fallback (CI is ubuntu-latest) instead of unconditional call; added `Ops.ForceManagedMatVec()` and `INFERENCE_MATVEC=managed`, golden hash tests pin the managed path; restored `"test"` runner block in `global.json` (lost in 079ca5e). 147 tests: 146 pass + 1 skip (model set), 142 pass + 5 skip (no model) | Aleksei Kolesnikov |
