# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Apple Accelerate matvec (Phase 1 of the acceleration ADR).** `Ops.MatVec` now calls
  `cblas_sgemv` through `AccelerateBlas` (CPU, macOS). About 82–95 tok/s vs 34–37 tok/s on
  SmolLM2-135M F16, M4 Pro. **No GPU/Metal yet; Phase 2 (MLX) is not started.** The old loop
  remains as `Ops.MatVecManaged` (test reference, non-macOS fallback);
  `INFERENCE_MATVEC=managed` switches to it. Golden-hash tests pin the managed loop. Handoff for
  Phase 2: [phase2-mlx-handoff.md](docs/investigation/phase2-mlx-handoff.md).

- Fix: a missing `--model` file (or directory) now prints a one-line error and exits with code 1
  instead of crashing with an unhandled `FileNotFoundException` stack trace and exit code 134.
  Covered by new process-level tests in `InferenceEngine.Cli.Tests/ProgramTests.cs`.
- Fix: `global.json` regained its `Microsoft.Testing.Platform` test-runner block (dropped by an
  earlier commit), so plain `dotnet test` works again on the .NET 10 SDK.
- [Apple Silicon acceleration ADR](docs/architecture/261101-apple-silicon-acceleration-strategy.md)
  (proposed): Accelerate `cblas_sgemv` for CPU matvec (Phase 1), MLX (via mlx-c, Metal-backed) for GPU tensor compute
  (Phase 2). Project is now stated as Apple Silicon-centric (no Windows/Intel plans).
- `AllowUnsafeBlocks=true` in `InferenceEngine.Models` only, for `LibraryImport` P/Invoke; rule
  in `.claude/rules/unsafe-policy.md`. All other projects stay safe-only.
- Project scaffolding: README with project goal and inspiration (dotLLM by Konrad Kokosa),
  AGENTS.md / CLAUDE.md ground rules, ADR template (`architecture/`), and Claude Code
  subagents (`adr-writer-reviewer`, `csharp-dotnet`).
- [Solution-layout ADR](docs/architecture/260811-solution-and-project-layout.md): solution/project layout.
- [Project-challenges ADR](docs/architecture/260901-project-challenges-and-how-to-address-them.md):
  the problems each layer solves, the build-vs-reuse strategy per challenge, the fetch-once/mmap
  model-weight dependency, a runtime-call diagram, and how other engines are structured.
- Seven placeholder ADRs (`architecture/planned-*.md`, `Status: planned`) — one per component
  challenge from the project-challenges ADR: format loading, tokenization, attention & transformer,
  KV-cache, sampling pipeline, HuggingFace acquisition, performance baseline. Each is renamed to
  `YYMMDD-<slug>.md` and filled in when its round comes.
- `src/` scaffolding per the solution-layout ADR — `InferenceEngine.{Core,Models,Tokenizers,Engine}` class
  libraries and a runnable `InferenceEngine.Cli` console app (net10.0, no functionality yet).
- Rider run/debug configuration for `InferenceEngine.Cli` (`launchSettings.json` +
  `.idea/runConfigurations/`).
- `adr-writer-reviewer` and `csharp-dotnet` agents now consult context7 for current library
  docs before evaluating/using a package, instead of relying on training-data memory.
- Three investigation agents: `researcher`, `huggingface-explorer`, `code-reader` — for
  structured research into inference internals before implementation.
- `investigation/` directory with research documentation:
  - `overview.md` — investigation goals, references, topic map
  - `dotllm-architecture-trace.md` — full code-path trace of dotLLM's architecture
    (GGUF loading → tokenization → generation loop → sampling)
  - `huggingface-ecosystem.md` — GGUF model providers, Hub API, download options, tokenizer embedding
  - `gguf-format-research.md` — binary format spec, quantization types (Q4_0 through Q6_K),
    K-quant super-block structure, memory-mapping, .NET implementation notes
  - `inference-engine-project-layouts.md` — how other engines are structured (dotLLM, Jlama,
    llama3.java, LLamaSharp, ONNX Runtime GenAI, LM-Kit.NET), with sources — reference for the
    solution-layout comparison
- `experiments/` directory with benchmark data:
  - `reference-measurements-dotllm.md` — dotLLM v0.1.0-preview.3 baseline: 53.9 tok/s decode
    on SmolLM2-135M Q4_K_M, GGUF metadata dump, tensor quantization analysis
- README.md expanded with investigation topics, references, project structure, agent descriptions.
- **Bare-minimum end-to-end inference POC**: `InferenceEngine.Cli` now loads a real GGUF model
  and streams generated tokens, running SmolLM2-135M-Instruct (f16) locally.
  - `InferenceEngine.Core`: `ModelConfig`, `IModel`, `ITokenizer`, `IKvCache`, `TokenizerData` contracts.
  - `InferenceEngine.Models`: hand-rolled GGUF v3 reader (`Gguf/`) — no maintained GGUF-parsing
    library exists on NuGet, so format parsing had to be built rather than reused (see the
    format-loading ADR follow-up); Llama forward pass (`Llama/`) — RMSNorm, GQA attention with
    interleaved-pair RoPE, SwiGLU FFN, tied LM head — built on `System.Numerics.Tensors`
    (`Math/Ops.cs`), no custom kernels, no `unsafe`. Supports F32/F16 tensors only; quantized
    types parse correctly as metadata but throw on dequantization.
  - `InferenceEngine.Tokenizers`: hand-rolled byte-level BPE (`GgufBpeTokenizer`) built from a
    model's embedded vocab/merges, since GGUF doesn't embed the pre-tokenizer's splitting regex.
    Implements the `smollm` pre-tokenizer variant (matched against llama.cpp's `unicode.cpp`).
  - `InferenceEngine.Engine`: `SimpleKvCache` (contiguous per-layer arrays), a composable
    `Sampler` (temperature → top-k → top-p, with a greedy short-circuit), and the
    `InferenceSession` facade (ChatML wrapping, prefill/decode loop, streaming generation).
  - `InferenceEngine.Cli`: `--model`, `--prompt`, `--max-tokens`, `--temperature`, `--top-k`,
    `--top-p`, `--seed`, `--raw`, `--stats`, `--debug-tokenize`, `--debug-logits`.
  - Verified: extracted `ModelConfig` matches the recorded dotLLM baseline exactly; tokenizer
    round-trips exactly; top-1 next-token prediction after a test prompt is semantically
    correct; 64-token greedy generation is coherent and factually correct, at ~38 tok/s decode
    (f16, vs. dotLLM's 53.9 tok/s on Q4_K_M — expected, given ~2.6x the memory traffic per token
    and no fused/quantized kernels).
  - Explicitly out of scope for this POC (see the plan): HuggingFace download (`--model` takes
    a local path only), batched prefill (single-token loop), the model's Jinja2 chat template
    (hard-coded ChatML string instead), multi-threading, and HTTP serving.
- `Directory.Build.props`: `TreatWarningsAsErrors` enabled.
- `.claude/settings.json`: checked-in project permissions policy (read-only allowlist for the
  model cache and dotLLM reference install; narrow, pre-approved `dotnet`/`brew` inspection
  commands; `WebFetch` allowed for github.com, huggingface.co, raw.githubusercontent.com).
- **Automated test suite**: one xUnit v3 (Microsoft Testing Platform) test project per `src/`
  project — `tests/InferenceEngine.{Core,Models,Tokenizers,Engine,Cli}.Tests` — 51 tests, all
  passing, each documenting the business scenario it protects (see each project's README.md).
  `global.json` now selects the .NET 10 SDK's native MTP `dotnet test` runner. Highlights:
  a synthetic-GGUF-file builder exercises the hand-rolled reader (metadata, F32/F16 tensors,
  alignment, unsupported quant types) without needing the real 258 MB model; `Ops.cs`'s math
  (RmsNorm/MatVec/RoPE/Softmax/SwiGLU) is checked against hand-computed values; the BPE
  tokenizer's merge-rank ordering and digit pre-tokenization are verified directly; the sampler
  (greedy/top-k/top-p, seeded reproducibility) and `ChatMlTemplate`'s turn ordering are covered.
  Coverage on these hand-rolled files: `Ops.cs` 100%, `GgufFile.cs` 86%, `GgufBpeTokenizer.cs`
  92–100%, `SamplingPipeline.cs` 98.4%, `ChatMlTemplate.cs` 100%.
  Found and documented (not silently fixed) a real gap: `TokenizerData`'s `string[]` fields give
  it reference-based, not value-based, record equality.
  One minimal production seam: `CliOptions.Load` gained an optional `loadDotEnv` parameter
  (default `true`, so production behavior is unchanged) so tests can exercise the flags/env-var
  precedence logic without touching the filesystem or a stray real `.env`.
  New `test-writer-runner` Claude Code agent (`.claude/agents/`) owns writing and running these
  going forward — one test project per `src/` project, business-case-documented, always run
  before being reported done.
- `docs/scenarios/`: one Gherkin (`.feature`) file per `src/` project, documenting in plain
  Given/When/Then form the business scenarios the automated tests and manual CLI verification
  cover — plain specification files, not wired to a BDD execution framework.
- `Llama/GoldenLogitBaselineTests` (`tests/InferenceEngine.Models.Tests/`): a golden correctness
  oracle for the upcoming KV-cache rewrite. Prefills the real SmolLM2-135M-Instruct GGUF model
  (via `InferenceSession.PrefillTopLogits`, the same path `--debug-logits` drives) for a short and
  a multi-block-spanning long prompt, and hashes the full logit vector's raw IEEE-754 bit patterns
  (`LogitHash.Fnv1a`, reusable by a later end-to-end golden test) with the hardcoded golden hashes
  captured against the pre-rewrite `SimpleKvCache`. Skips cleanly (not a failure) unless
  `INFERENCE_MODEL` points at an existing GGUF file.
- `docs/investigation/kv-cache-research.md`: the five eras of KV-cache design (contiguous →
  PagedAttention → prefix caching → heterogeneous/quantized → distributed), why no .NET library
  (LLamaSharp, ONNX Runtime GenAI, TorchSharp) exposes its KV cache for learning purposes, the
  NHD-vs-HND layout analysis, this machine's measured (not assumed) cache-line/page/L2 facts, and
  the quantitative finding driving the implementation: reordering the attention loop over KV heads
  cuts K/V DRAM traffic 3x (283 MB → 94 MB/token at context 2048) with zero new types, while block
  paging alone buys a single-sequence engine no measurable speedup.
- [KV-cache ADR](docs/architecture/260914-kv-cache.md) (`260914-kv-cache.md`, replacing the
  `planned-kv-cache.md` placeholder): chosen design is block-paged, head-major (HND) layout,
  single sequence, landed as three separately-measured changes (loop reorder, RoPE-table hoist,
  head-major layout + paging) rather than one bundled change.
- **Paged, head-major KV-cache rewrite**, per the ADR above:
  - `IKvCache` (`InferenceEngine.Core`) now exposes per-KV-head accessors — `KeySlot`/`ValueSlot`
    (write, one head's `headDim` floats) and `KeyBlockForHead`/`ValueBlockForHead` (read, a
    contiguous `[count, headDim]` tile) — plus `Reserve`/`Reset`/`Rollback` lifecycle methods,
    replacing the old per-layer `Key`/`Value` row accessors that a head-major layout can't support.
  - `KvBlockPool` + `PagedKvCache` (`InferenceEngine.Engine`, replacing `SimpleKvCache`): physical
    KV storage in 32-token blocks, allocated lazily via `Reserve(position)` as generation advances
    instead of exact-fit up front; freed blocks retain their arrays so `Reset()`/`Rollback()` are
    allocation-free. `InferenceSession` now sizes both KV caches to the model's full `MaxSeqLen`
    (cheap — an `int[]` block table, not the KV data) instead of `promptLength + maxNewTokens`.
  - `GqaAttention` (`InferenceEngine.Models`), extracted from `LlamaModel.Forward`: the attention
    loop now iterates KV heads outermost and the `groupSize` query heads sharing each one innermost,
    reading each KV-head tile from the cache once and reusing it in L1 across the query heads that
    share it, instead of re-reading it once per query head.
  - `Ops.Rope` split into `RopeTable` (once per `Forward` call — the rotation table depends only on
    position, not layer or head) + `RopeHead` (per head), cutting ~5,660 redundant
    `MathF.Pow`/`Cos`/`Sin` calls per decode token.
  - `LlamaModel`'s constructor now rejects a `NumAttentionHeads` not evenly divisible by
    `NumKvHeads` — previously an inexact ratio silently truncated the query-head group size and
    read another head's memory (latent bug, unreachable by SmolLM2-135M's exact 9:3 ratio).
  - Verified bit-identical against `GoldenLogitBaselineTests`' pre-rewrite hashes at every one of
    the six implementation commits — every step restructures *where* the same floating-point
    operations happen, never what they compute.
- `experiments/kv-layout-benchmark.md`: before/after measurement across the six-commit rewrite.
  +4.7% throughput at short context (noise-level, as predicted — KV traffic is a small fraction of
  per-token traffic there); −24.2% wall-clock time at a 1,906-token prompt, closely matching the
  ADR's independent ~23% traffic-reduction estimate. Includes the reproduction fixture
  (`kv-benchmark-long-prompt.txt`) and the measured hardware facts (128 B cache line, 4 MiB L2) the
  block-size choice depends on.
- [Logits processing and sampling pipeline ADR](docs/architecture/260917-logits-processing.md)
  (`260917-logits-processing.md`, replacing the `planned-sampling-pipeline.md` placeholder): draws
  an explicit four-way line between the tokenizer, the raw logits data crossing the
  `Models → Engine` boundary, a new deterministic *logits processor* phase, and the existing
  probabilistic *sampler*. Chosen design (Option B): an internal split inside `Engine` — a new
  `ILogitsProcessor` interface distinct from `ISamplerStep`, with `BannedSequenceLogitsProcessor`
  (HuggingFace `NoBadWordsLogitsProcessor`-style sequence-prefix matching) as its first
  implementation, and `SamplingPipeline.Sample` running all registered processors unconditionally
  before the greedy/stochastic branch splits — the current greedy path bypasses `ISamplerStep`
  entirely, which a ban must not. Rejected a new `InferenceEngine.LogitsProcessing` project as
  disproportionate ceremony for this round's content, backed by a survey showing dotLLM,
  HuggingFace transformers, vLLM, and llama.cpp all treat this as a phase split within one
  pipeline, never a package boundary. Names repetition penalty, forced-token bias, and
  grammar/schema-constrained decoding as future seams, not built now.
- Polished [260811-solution-and-project-layout.md](docs/architecture/260811-solution-and-project-layout.md):
  labeled its existing project-reference diagram explicitly as a C4 Container diagram, and noted
  the new sampler/logits-processor split inside `Engine`'s row without adding a new project row.
- Polished [260901-project-challenges-and-how-to-address-them.md](docs/architecture/260901-project-challenges-and-how-to-address-them.md):
  fixed a dead link (KV-cache row pointed at the retired `planned-kv-cache.md`; now points at
  `260914-kv-cache.md`) and repointed the Sampling/logits row's follow-up-ADR link from the
  retired `planned-sampling-pipeline.md` to `260917-logits-processing.md`.
- Retired `docs/architecture/planned-sampling-pipeline.md` (placeholder, `Status: planned`),
  superseded by `260917-logits-processing.md` — same pattern as the KV-cache round retiring
  `planned-kv-cache.md`.
- CI: `.github/workflows/tests.yml` restores, builds, and runs `dotnet test` on every push to
  `main` and every pull request (public repo, so this is free on GitHub Actions — unlimited
  minutes, no billing setup needed). The two golden-hash tests skip cleanly in CI, by design —
  they need a real ~270 MB GGUF model file that isn't checked into the repo. Added a status badge
  to the top of `README.md`.
- **Logits processing, Phase B**: implements the design from
  [260917-logits-processing.md](docs/architecture/260917-logits-processing.md).
  - `ILogitsProcessor` (`InferenceEngine.Engine.Sampling`, new): one method,
    `Apply(Span<float> logits, ReadOnlySpan<int> generatedTokenIds)`, documented to never mask
    EOS. `BannedSequenceLogitsProcessor` (new) implements it: a length-1 banned sequence is always
    masked to `float.NegativeInfinity`; a longer sequence is masked only when the tail of
    `generatedTokenIds` matches its prefix, HuggingFace `NoBadWordsLogitsProcessor`-style. Its
    `FromWords(ITokenizer, IReadOnlyList<string>)` factory tokenizes each banned word once, at
    session-configuration time, and drops any word that encodes to zero tokens or to a single
    EOS token.
  - `SamplingPipeline`'s constructor now also takes `IReadOnlyList<ILogitsProcessor> processors`
    and `int eosTokenId`; `Sample` gains a `ReadOnlySpan<int> generatedTokenIds` parameter and
    runs every processor unconditionally, before the greedy/stochastic branch splits — previously
    the greedy path (`Temperature: 0`, the default) skipped `ISamplerStep`s entirely, so a ban
    could only ever reach the non-default stochastic path. Greedy now also copies logits into its
    scratch buffer when at least one processor is registered (unchanged, zero-copy, otherwise).
    Both paths now return `eosTokenId` directly if every logit is `float.NegativeInfinity` after
    processors run — a defensive backstop against picking an arbitrary (possibly banned) index;
    on the stochastic path this also closes a real bug, not just a hypothetical one:
    `Exp(-inf - -inf)` is `NaN`, and NaN comparisons are always `false`, so the pre-fix
    cumulative-probability loop fell through to `working.Length - 1`.
  - `GenerationOptions` gains `IReadOnlyList<string>? BannedWords = null`.
  - `InferenceSession.GenerateCore` builds the processor list from `options.BannedWords` (empty
    when null/empty), constructs `SamplingPipeline` with it and `_tokenizer.EosTokenId`, and
    tracks generated-token history across the decode loop to pass into `Sample`.
  - `InferenceEngine.Cli`: new `--ban-words "W1,W2"` flag / `INFERENCE_BAN_WORDS` environment
    variable, comma-split with **no** trimming (a trailing space in an entry, e.g. `"EPAM "`, is
    a meaningfully different literal from `"EPAM"`, not incidental whitespace) — same
    flag-beats-env precedence as every other option. `Program.cs`'s `GenerationOptions`
    construction switched from positional to named arguments while here, since a positional
    record with every parameter defaulted silently reassigns values if a future field lands out
    of order.
  - 14 new tests (113 total, all passing): `BannedSequenceLogitsProcessor`'s masking rules (always
    masked for a length-1 sequence, prefix-matched for a longer one, independent sequences not
    interfering, a sequence longer than history not throwing, EOS never masked either directly or
    as a completing token, `FromWords` dropping an EOS-only word), the critical greedy-path ban
    regression, the all-masked-EOS-fallback on both the greedy and stochastic paths (the latter
    constructed so it actually exercises the NaN path, not just "EOS happens to be the only
    finite logit"), and `CliOptions` coverage for `--ban-words`/`INFERENCE_BAN_WORDS` parsing,
    precedence, and the off-by-default (`null`) case. `GoldenLogitBaselineTests`' hashes are
    unchanged (banning is opt-in) — verified against the real model.
- CodeRabbit review of the CI workflow and Phase B docs/tests:
  - `.github/workflows/tests.yml` — added `permissions: contents: read` at the workflow level (a
    push run otherwise inherits repo/org defaults, which may be broader) and
    `persist-credentials: false` on the checkout step; pinned `actions/checkout`/
    `actions/setup-dotnet` to immutable commit SHAs (verified via `git ls-remote` against the
    upstream repos, not from memory) instead of the mutable `@v4` tag, with the human-readable
    version kept in a trailing comment.
  - `README.md`'s `--ban-words "Machine"` example reworded — it bans the exact token *sequence*
    `"Machine"` encodes to, not the lexical word across every surrounding tokenization context;
    the original phrasing overstated the guarantee.
  - Trimmed `SamplingPipelineTests.cs`'s class doc comment to the two facts not obvious from the
    test names/assertions themselves (processors must run on the greedy path since `_steps` is
    skipped there; the pipeline must not mutate a reused caller buffer) — removed the restated
    per-test outcome descriptions.
  - Same Docstring Coverage pre-merge check as previous rounds (`6ae9b9b` and the KV-cache round),
    same resolution: the generic 80% threshold isn't a policy this repo opted into and conflicts
    with `.coderabbit.yaml`'s own `**/*.cs` rule ("no WHAT comments, only WHY when non-obvious").
    Added `<inheritdoc/>` to `BannedSequenceLogitsProcessor.Apply` (implements the
    already-documented `ILogitsProcessor.Apply` contract) instead of restating it, and genuine WHY
    docs where they were actually missing: `BannedSequenceLogitsProcessor.FromWords` (why it drops
    zero-token/EOS-only words) and `SamplingPipeline.Sample` (why processors must run
    unconditionally before the greedy/stochastic split, and the all-masked-EOS fallback). Left
    self-explanatory or already-documented members alone — not chasing the percentage.

CodeRabbit review findings on the POC PR:

- `CliOptions.Load` no longer throws `IndexOutOfRangeException`/`FormatException` for a flag
  missing its value or a malformed numeric flag/environment-variable value — both now raise a
  clear `ArgumentException` naming the offending flag or variable.
- `Program.cs` now catches `ArgumentException` around model loading and generation, not just
  around CLI option parsing, so an `InferenceSession.Generate` validation failure (see below)
  reports a clean one-line error and exit code 1 instead of an unhandled-exception stack trace.
- `InferenceSession.Generate` now validates `MaxNewTokens >= 0`, a non-empty encoded prompt, and
  the requested sequence length against `ModelConfig.MaxSeqLen` — and validates *eagerly*, before
  returning, rather than only once the caller starts enumerating (it was refactored from a single
  iterator method into a plain validating wrapper around a private iterator, since code before a
  `yield` in an iterator method doesn't run until the first `MoveNext`). Previously, a negative
  `--max-tokens` or `--raw` with an empty prompt could throw a confusing exception deep inside
  `SimpleKvCache`, or silently sample from an empty logits span.
- Streamed generation could show a UTF-8 replacement character when a single multi-byte
  character's bytes were split across two generated tokens, because each token was decoded to
  text independently. `InferenceSession` now feeds each token's raw bytes (`ITokenizer` gained
  `GetTokenBytes`) through a new `IncrementalUtf8Decoder`, which buffers an incomplete character
  across calls the way `System.Text.Decoder` is designed to.

Second round of CodeRabbit review findings:

- `GgufFile`: a file-declared count (metadata KV count, tensor count, tensor dimension count,
  string length, array length) exceeding the file's own size now throws `InvalidDataException`
  before any allocation sized to it — a small malformed or truncated file could otherwise trigger
  an excessive allocation. Separately, reading a metadata string now uses `Stream.ReadExactly`
  instead of `BinaryReader.ReadBytes`, which silently returns a shorter-than-requested array on a
  truncated file instead of throwing.
- `GgufTensorDescriptor.ElementCount`: the dimension-count multiplication is now `checked`, so
  dimensions that would overflow `long` throw `OverflowException` instead of silently wrapping to
  a smaller, incorrect (but plausible-looking) tensor size.
- `GgufBpeTokenizer`: a merge-list entry that isn't a space-separated pair now throws a clear
  `InvalidDataException` naming the entry, instead of `IndexOutOfRangeException` from `Split`.
- `InferenceSession.Load` now validates that the model's vocab size (`llama.vocab_size`) matches
  the tokenizer's vocabulary length (`tokenizer.ggml.tokens`) — both are read independently from
  the same GGUF file, and if they disagree, a sampled id could be out of range for the tokenizer.
  Caught at load time with a clear cause instead of an obscure exception mid-generation.
- `LlamaModel.Forward` now rejects a `position` outside `[0, MaxSeqLen)` — its attention scratch
  buffers are sized to `MaxSeqLen` — and `InferenceSession.PrefillTopLogits` (the `--debug-logits`
  path, which had no such guard) now rejects a prompt longer than `MaxSeqLen` before allocating
  the KV-cache, matching the guard already added to `Generate`.
- `TokenizerData.UnknownTokenId`: removed — read from GGUF metadata but never consumed anywhere.
- `Ops.Rope`: the per-pair rotation angle (`cos`/`sin`) depended only on position and pair index,
  not on which head was being rotated, but was recomputed (`MathF.Pow`/`Cos`/`Sin`) once per head
  per pair. Now computed once per pair and reused across all heads — fewer redundant transcendental
  calls on the decode hot path, same result.
- Removed several XML doc comments across the sampling types that restated what the code already
  says (pure "what", no non-obvious "why"), per this project's comment policy.

10 new tests across Models/Tokenizers/Engine (71 total, all passing): implausible GGUF counts,
truncated-string reads, tensor dimension overflow, a malformed merge entry, the vocab-size
consistency check, and the `PrefillTopLogits` sequence-length guard. Re-verified end-to-end
against the real model after the `Rope` refactor and the new `GgufFile` bounds checks — output
and load time are unchanged. Clean build (0 warnings/errors, Debug + Release, wiped `bin`/`obj`).

CodeRabbit's Docstring Coverage pre-merge check (80% threshold) flagged 70 touched functions
across 24 files — a generic platform default this repo never opted into, and one that pulls
against the comment policy already declared in `.coderabbit.yaml` ("no WHAT comments, only
WHY"). Rather than pad functions with restated-behavior summaries to hit a number, added
`<inheritdoc/>` on the handful of public methods implementing an already-documented interface
member (`LlamaModel.Forward`, and `GgufBpeTokenizer`'s `DecodeToken`/`GetTokenBytes`/
`EosTokenId`/`TryGetId`) plus two contract-level docs on `ITokenizer.EosTokenId`/`TryGetId` that
were genuinely missing, and three inline WHY comments for non-obvious behavior (`GgufFile.Open`'s
dispose-on-parse-failure, `GgufTensorDescriptor.ElementCount`'s `checked` overflow guard,
`LlamaModel.LoadFromGguf`'s config fallback defaults). The threshold itself is unaddressed by
design — it doesn't fit this project's documented style.

Third round of CodeRabbit review findings:

- `CliOptions`: numeric flags/env vars (`--temperature`, `--top-p`, etc.) now parse with
  `CultureInfo.InvariantCulture` instead of the current culture — on a comma-decimal locale,
  `float.TryParse` without it can silently misparse or reject valid values like `0.7`.
- `DotEnvLoader`: a line with an empty key (e.g. a stray `=value`) is now skipped instead of
  calling `Environment.GetEnvironmentVariable`/`SetEnvironmentVariable` with an empty name, which
  throws `ArgumentException` and — since it happens inside the `.env` line loop — previously
  aborted every entry after it in the same file, not just the bad line.
- `Program.cs`'s error handling now catches `InvalidDataException` and `NotSupportedException`
  alongside `ArgumentException`. `InferenceSession.Load` can throw either (a malformed GGUF file,
  a model/tokenizer vocab-size mismatch, or an unsupported GGUF version/architecture/tokenizer/
  pre-tokenizer) but only `ArgumentException` was caught, so these reported as unhandled-exception
  stack traces instead of a clean one-line error.
- `InferenceSession.Generate`'s context-length check now compares via subtraction
  (`options.MaxNewTokens > Config.MaxSeqLen - promptIds.Count`) instead of addition
  (`promptIds.Count + options.MaxNewTokens > Config.MaxSeqLen`) — the addition can overflow for
  a large `MaxNewTokens` (e.g. `int.MaxValue`) and wrap past the check instead of failing it.
- `InferenceSession.GenerateCore` no longer runs a forward pass after yielding the very last
  token of a `MaxNewTokens`-bounded generation — that pass's logits and KV-cache write were
  never read by anything, since the loop exits right after. Skips the most expensive op in the
  loop for every token-limit completion (not needed for the EOS-triggered stop, which already
  exited before reaching it).
- Markdown lint fixes on `.claude/agents/test-writer-runner.md` (MD041 top-level heading after
  front matter, MD040 language on the coverage-command fence).
- Trimmed a handful of XML/inline comments that restated behavior instead of explaining
  non-obvious rationale (`ITokenizer.GetTokenBytes`, `IChatPromptStep`, `Timed`, `ConsoleOutput`,
  the `FakeModel` test double) — same "WHY not WHAT" policy as the previous round.

5 new tests (76 total, all passing): invariant-culture rejection of a comma-decimal value, the
empty-`.env`-key line no longer aborting the file, the overflow-safe context-length check
(`MaxNewTokens: int.MaxValue`), and a `FakeModel.ForwardCallCount` assertion proving the final
forward pass is actually skipped. Re-verified end-to-end: normal generation output unchanged, a
non-GGUF file now reports a clean error instead of a stack trace, and a comma-decimal
`--temperature` is cleanly rejected. Clean build (0 warnings/errors, Debug + Release, wiped
`bin`/`obj`).

CodeRabbit review findings on the KV-cache-rewrite PR:

- `PagedKvCache.Rollback` now rejects a `toPosition` outside `[0, Length]` with
  `ArgumentOutOfRangeException` instead of silently corrupting state — a negative target froze
  `Length` at a nonsensical negative value, and a target past `Length` marked never-written
  positions as resident. Two new tests cover both directions; `IKvCache.Rollback`'s doc comment
  now states the contract.
- `PagedKvCacheTests`' rollback test wrote and re-read a sentinel at a position inside the
  partially-retained block instead of only re-reading a default-valued position — the previous
  assertion (`0f == 0f`) would have passed even if that block had been wrongly freed.
- `docs/investigation/status.md` no longer contradicts itself: the CLI-is-a-stub and
  generation-loop-is-future-work bullets were still there next to the new KV-cache-done entry.
  Updated to match the actual end-to-end state.
- `experiments/kv-layout-benchmark.md`'s observation "the loop reorder is the whole story" is not
  something the before/after commits can establish — they bundle the loop reorder, head-major
  layout, and paging together, so no single change's share of the measured speedup is isolated.
  Reworded to say what the measurement actually shows (consistent with a bandwidth-bound change)
  versus what it can't (attribution to one of the three).
- `README.md`'s "tiny models" bound (≤ ~500M params) listed TinyLlama-1.1B as an example, which is
  more than double that bound. Moved it to the medium-model bullet, where it actually belongs, and
  noted it's the boundary case an F16 GGUF would already load.
- Two markdownlint MD040 fixes (missing fence language) and one WHAT-vs-WHY trim on
  `LogitHash`'s class doc, matching this repo's established comment policy.
- Same Docstring Coverage pre-merge check as `6ae9b9b`, same resolution: the 80% threshold is a
  generic default this repo hasn't opted into, and it conflicts with `.coderabbit.yaml`'s own
  `**/*.cs` policy ("no WHAT comments, only WHY when non-obvious"). Added `<inheritdoc/>` to every
  `PagedKvCache` member implementing an already-documented `IKvCache` member (`BlockSize`,
  `HeadDim`, `Capacity`, `Length`, `Reserve`, `KeySlot`, `ValueSlot`, `KeyBlockForHead`,
  `ValueBlockForHead`, `Reset`, `Rollback`) instead of restating their docs — the actual gap was
  that the interface's existing documentation wasn't surfaced on the implementation, not that the
  behavior was undocumented. Left the trivial one-line accessors on `KvBlockPool`
  (`KeyStore`/`ValueStore`/`HeadDim`/`NumLayers`) undocumented, same as before — self-explanatory,
  not a real gap.

Verified: `dotnet build` (0 warnings/errors) and `dotnet test` (99/99, up from 97 — the two new
`Rollback` validation tests) both pass.

### Changed

- Moved `architecture/`, `investigation/`, and `scenarios/` under a new `docs/` folder
  (`docs/architecture/`, `docs/investigation/`, `docs/scenarios/`), via `git mv` to preserve
  history. `experiments/` stays at the repo root (not part of this move). Updated every
  cross-reference: README, AGENTS.md, CLAUDE.md, CHANGELOG, `.coderabbit.yaml`'s path
  filters/instructions, the `adr-writer-reviewer`/`csharp-dotnet` agent definitions, and the
  one relative link (`planned-performance-baseline.md` → `experiments/`) whose depth changed
  because `architecture/` moved but `experiments/` didn't.
- ADR file naming convention switched from sequential `NNNN-title.md` to `YYMMDD-<slug>.md`
  (chronological by date prefix). Renamed `0001-solution-and-project-layout.md` →
  `260811-solution-and-project-layout.md` and `0000-template.md` → `template.md`; updated
  references in README, AGENTS.md, the `adr-writer-reviewer`/`csharp-dotnet` agents, and the
  investigation notes.
- Polished the solution-layout ADR: added a project-reference diagram (Mermaid) and a
  "which project is responsible for what" table; corrected dotLLM's project count (10 → ~17);
  kept it focused on structure, with the engineering challenges moved out (see below).
- Renamed the `adr-writer-reviewer` Claude Code agent to `adr-author` (via `git mv`, preserving
  history) — the old name's "writer" and "reviewer" both stay true of the role, but "author"
  covers both without the split. Updated every cross-reference: README, AGENTS.md, CLAUDE.md,
  and the `code-reader`/`csharp-dotnet`/`huggingface-explorer`/`researcher` agent definitions.
  Also hardened the agent's own instructions: writing or amending an ADR without ending it in a
  filled-in "Decision Log" table is now called out as incomplete, not just implied by the
  template, and the review checklist explicitly flags a missing/stale one as a finding.
- `AGENTS.md`'s "Working style" and the `csharp-dotnet` agent definition now state explicitly:
  never `git commit`/`push` without the user asking for that specific commit — approving a plan
  or a multi-step task is not standing approval to commit along the way, and this applies to
  work delegated to a subagent too.
