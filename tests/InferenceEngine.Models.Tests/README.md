# InferenceEngine.Models.Tests

Tests `src/InferenceEngine.Models`: the hand-rolled GGUF file reader and the transformer math
primitives it feeds.

## Business scenarios covered

### `Gguf/GgufFile` (`GgufFileTests.cs`, fixtures via `GgufFileBuilder.cs`)

GGUF has no maintained NuGet parser, so this reader is hand-rolled (see the format-loading ADR).
Fixtures are synthetic GGUF v3 files built byte-for-byte with `BinaryWriter` in
`GgufFileBuilder`, not a downloaded model, so every case is small and deterministic.

- Scalar metadata (string/u32/f32) round-trips exactly; a missing key falls back to the caller's
  supplied default instead of throwing.
- Array metadata (string arrays and int32 arrays) round-trips exactly.
- An F32 tensor's bytes come back unchanged.
- An F16 tensor is dequantized to F32 within floating-point tolerance.
- A quantized type this POC doesn't support (Q4_0, ggml type 2) parses as metadata/shape but
  throws `NotSupportedException` on `ReadTensorAsF32` instead of silently returning wrong data.
- An unknown tensor name throws `KeyNotFoundException`.
- Two adjacent tensors placed after a header whose length isn't itself a multiple of the default
  32-byte alignment read back independently and correctly — proving the data-section alignment
  padding is computed correctly and neither tensor's bytes leak into the other's.
- A `general.alignment` metadata override (64 instead of the 32 default) is honored, not just
  parsed.

### `Math/Ops` (`OpsTests.cs`)

The attention/FFN math primitives, checked against hand-computed values rather than mirrored
implementation logic:

- `RmsNorm` on `x=[3,4]`, `weight=[1,1]`, `eps=0` matches `invRms = 1/sqrt(12.5)`.
- `MatVec` on a small hand-written row-major `[out,in]` matrix matches manual dot products.
- `Rope` at `position=0` is an exact no-op for any `freqBase` (the rotation angle is zero), and
  at a non-zero position rotates by the expected angle.
- `Softmax` output sums to 1 and matches the standard `[1,2,3]` softmax constants.
- `SwiGlu` satisfies `Silu(0) = 0` and matches a hand-computed value at `gate=1, up=1`.

Note: `Ops` lives in a namespace literally named `Math` (shadowing `System.Math` once imported),
so `OpsTests.cs` only ever uses `MathF` for scalar math to avoid the ambiguity.

### `Math/AccelerateBlas` (`AccelerateBlasTests.cs`) — Apple Accelerate matvec

On Apple Silicon the matrix-vector product runs through Accelerate `cblas_sgemv`
([ADR 261101](../../docs/architecture/261101-apple-silicon-acceleration-strategy.md)). All tests
except the off-macOS probe check skip with a clear reason where Accelerate is unavailable (the
Linux CI runner).

- On macOS the framework loads and the symbol resolves (`IsAvailable`); off macOS it is false.
- A hand-computed 2x3 product is correct.
- Results match the managed loop within a float tolerance that scales with `inDim` (not
  bit-equal; BLAS sums in a different order) across odd shapes and the real SmolLM2 shapes,
  including the 49152x576 LM head.
- A destination pre-filled with NaN is fully overwritten (beta = 0 never reads old contents).
- Only the first `outDim` destination elements are written, and only the first `outDim*inDim`
  weight / `inDim` input elements are read; sliced spans with a non-zero offset work.
- Too-short spans and zero or negative dimensions throw `ArgumentOutOfRangeException` before any
  native call and leave the destination untouched.

### `Math/Ops.MatVec` dispatch (`OpsMatVecTests.cs`) — runs on every OS

- `MatVec` and `MatVecManaged` give the hand-computed product; `MatVecManaged` matches a plain
  double-precision triple loop.
- Zero `outDim` or `inDim` never reaches BLAS and agrees with the managed loop (empty dot = 0).
- `Ops.ForceManagedMatVec()` makes `MatVec` bit-identical to the managed loop inside its scope,
  and is lifted on dispose. It is per execution context: a parallel task outside the scope is
  unaffected.

### `Llama/GqaAttentionTests` — attention-loop equivalence

`GqaAttention.Attend` (extracted from `LlamaModel.Forward`'s attention block per the
[kv-cache ADR](../../docs/architecture/260914-kv-cache.md)) must compute exactly the same
floating-point values, in exactly the same accumulation order, as an independent transcription of
the pre-extraction attention math — not just "look equivalent". The reference is transcribed
directly from the original `LlamaModel.cs` attention block, not by calling `GqaAttention`, so a
bug introduced during extraction or the KV-head-outermost loop reorder can't hide from both sides
of the comparison at once. Parameterized over context lengths (`1, 31, 32, 33, 64, 65, 96, 129`)
chosen to straddle 32-token block boundaries.

### `Llama/GoldenLogitBaselineTests` — golden KV-cache logit baseline

A frozen-in-time correctness oracle, originally captured against the pre-rewrite `SimpleKvCache`
implementation before the KV-cache rewrite in `docs/architecture/260914-kv-cache.md` began:
prefilling the real SmolLM2-135M-Instruct GGUF model with a fixed short prompt and a fixed long
prompt (long enough to span several 32-token cache blocks) always produces the identical
next-token logit distribution, bit-for-bit. The hardcoded `Fnv1a` hashes held unchanged across
every one of the rewrite's six implementation commits (loop reorder, RoPE-table hoist, head-major
layout, block paging, session wiring) — proof the rewrite restructured *where* the floating-point
operations happen, never what they compute. The top-10 `(tokenId, logit)` pairs are captured
alongside for human debuggability if a hash ever mismatches. `LogitHash.Fnv1a` (hashing raw
IEEE-754 bit patterns, not decimal text) is written as a small reusable static method so any
future rewrite can compute the same hash for comparison.

Requires the real 270 MB model and is slow (a few seconds of real prefill), so it skips cleanly —
not a failure — unless `INFERENCE_MODEL` is set to an existing GGUF file path:

```sh
INFERENCE_MODEL=~/.cache/inference-engine/models/SmolLM2-135M-Instruct-f16.gguf dotnet test ...
```

(with `~` already expanded by the shell). This is the only test in this project that references
`InferenceEngine.Engine` in addition to `InferenceEngine.Models` — it needs
`InferenceSession.PrefillTopLogits` (the exact code path `InferenceEngine.Cli`'s debug-logits
flag drives) because the chat-template wrapping that produces the fixed 164-token long prompt is
internal to `Engine` and can't be reconstructed from `Models` alone.

### `Llama/AcceleratedGenerationTests` — Accelerate does not change what the model says

With the real SmolLM2-135M model (skips unless `INFERENCE_MODEL` is set), greedy generation of
32 tokens from the short prompt yields the same token ids with Accelerate as with the managed
loop, and prefill agrees on the top-1 token and shares at least 8 of the top-10 ids (float drift
may reorder near-ties). The bit-exact golden hash tests pin the managed loop via
`ForceManagedMatVec()`.
