
# KV-Cache Design Research

Research backing the [KV-cache ADR](../architecture/260914-kv-cache.md). Question: given the
educational goal and a single-sequence CPU engine, how does a modern KV cache actually work, and
which pieces of that design are worth building versus worth only understanding?

## Why there's no library to reuse

Checked every C#/.NET inference option in `AGENTS.md`'s reference list:

| Library | Where the KV cache lives | Usable as a learning reference? |
|---|---|---|
| **LLamaSharp** | Inside llama.cpp, via P/Invoke | No — opaque native state |
| **ONNX Runtime GenAI** | Inside native `onnxruntime-genai` | No — opaque native state |
| **TorchSharp** | Doesn't provide one — exposes libtorch tensors only | N/A — you'd build it yourself anyway |
| **dotLLM** (our primary reference) | Hand-built in C#: block-allocated, ref-counted, copy-on-write, Q8_0/Q4_0 quantized, hash-based prefix cache | Yes — the existence proof that this is worth building by hand in .NET |

Conclusion: there is nothing to adopt as a dependency. The two wrapper libraries hide exactly the
thing this project exists to understand, and the one pure-C# engine we benchmark against (dotLLM)
proves a hand-built cache is the normal approach even outside C/C++. So: build it, mirroring the
two reference designs below.

## The five eras of KV-cache design

Summarized from Modular's "Five Eras of KVCache" retrospective, cross-checked against the vLLM
PagedAttention paper and the llama.cpp paged-KV design discussion.

1. **Contiguous per-sequence (2017).** One tensor per sequence, preallocated to `max_seq_len`.
   Simple, but storage is `2 × layers × kv_heads × head_dim × max_seq_len` regardless of actual
   length — most sequences are shorter than their allocation, and fragmentation across
   variable-length requests caps batch size. **This is exactly what `SimpleKvCache` is today**
   (except it sizes to `prompt + maxNewTokens` rather than the model's true `max_seq_len`, which
   sidesteps the worst of the waste at the cost of being unable to grow past that exact figure).

2. **PagedAttention (vLLM, 2023).** Borrows OS virtual-memory paging: fixed-size blocks (pages)
   allocated on demand from a free pool, addressed per sequence through a **block table** (logical
   block index → physical block id). Kills fragmentation, lets many sequences share one pool.
   vLLM's own numbers: within the concurrency a contiguous cache can handle, paged trails it by only
   ~3% throughput — the win comes entirely from *not* pre-committing memory, which lets far more
   sequences run concurrently before OOM. The llama.cpp equivalent (in-progress, discussion #21961)
   reports 26 sequences OOM for the unified/contiguous cache vs. 247 for the paged one at the same
   memory budget, 2.5× aggregate throughput.

3. **Prefix caching / sharing.** Once storage is block-addressed, identical blocks (e.g. a shared
   system prompt) can be deduplicated: hash each full block's token content, keep a global
   `hash → physical block` map, reference-count blocks, and copy-on-write on first divergent write.
   vLLM's `KVCacheBlock` carries `block_id`, `block_hash`, `ref_cnt`, and free-list pointers for
   this. llama.cpp's unified cache does the same thing structurally via `seq_cp` aliasing cells
   across sequences without duplicating data.

4. **Heterogeneous / quantized caches (2024).** Real deployments need more than one cache shape at
   once — speculative-decoding draft vs. target caches, vision-encoder embeddings, sliding-window
   variants, and **quantized KV** (typically int8/Q8_0 with a per-token scale factor: ~4× memory
   cut, reported near-lossless in 2026 papers). Alongside quantization, eviction/compression
   policies trade cache size for accuracy: StreamingLLM (keep a few "attention sink" tokens + a
   sliding window — cheap, but degrades outside narrow task types), H2O (track cumulative attention
   mass, keep "heavy hitters" — more accurate, costs more to compute), SnapKV (score prefix
   importance from an observation window at prefill time). No single eviction policy dominates
   across tasks in 2026 survey results; they're presented as tunable options, not a solved problem.

5. **Distributed / unified hybrid (2025+).** Multi-node caches spilling hot pages to CPU RAM and
   cold pages to SSD/network storage, plus attempts to unify heterogeneous cache shapes behind one
   virtual-memory-style allocator (CUDA VMM remapping, or huge pages sized as an LCM of shapes).
   Not relevant here — single-process, single-sequence, CPU only.

**Where this project sits:** era 1 today, moving to era 2 (paged) plus the layout lesson below.
Era 3 (prefix caching) and era 4 (quantization/eviction) are designed for — see the ADR's
Consequences — but not built in this pass; era 5 doesn't apply to a CPU POC.

## The layout axis: NHD vs. HND — orthogonal to paging, and the bigger win here

Independent of *whether* the cache is paged, there's a data-layout choice inside each block/tensor:
lay tokens out contiguously per position (`[position, kv_head, head_dim]`, "NHD" in vLLM/TRT-LLM
naming) or contiguously per head (`[kv_head, position, head_dim]`, "HND"). TRT-LLM's native
`KVCacheManager` uses HND; other stacks default to NHD. Neither is universally correct — the choice
follows which matmul shape the attention kernel wants contiguous, and a portable format either picks
one (and someone's kernel pays a permute) or defines a neutral form (and every handoff pays a
conversion).

For a **decode step**, attention reads the *entire* K/V history and produces one output token — it
is memory-bandwidth-bound (arithmetic intensity is low: one dot product per cached position, not a
tile of matmuls). Our current attention loop (`LlamaModel.cs:122-144`) reads under an NHD-equivalent
layout: `kvDim = numKvHeads * headDim = 192` floats (768 B) between one token's K/V and the next, but
each query head's dot product only touches its own `headDim = 64` floats (256 B) of that — a
strided walk that uses 1/3 of every cache line group it touches, repeated separately for each of the
9 query heads even though only 3 distinct KV-head rows exist (GQA 3:1). Two independent fixes:

- **Reorder the loop** to iterate per KV head first, and the `groupSize` (3) query heads sharing it
  second — so each KV-head row is pulled from DRAM once and reused 3× from L1, instead of pulled 3
  separate times. This needs no layout change at all and is the largest win available (§ below).
- **Switch to HND layout** so that reuse is over a *contiguous* run of tokens per head rather than a
  strided one — turning the stride into a sequential stream, which is the best case for hardware
  prefetch.

## Hardware facts this project's block-size argument depends on

Measured on the actual development machine via `sysctl`, not assumed:

| Fact | Value |
|---|---|
| CPU | Apple M4 Pro |
| Cache line size | **128 B** (not the common assumption of 64 B — halves any "wasted bandwidth" arithmetic based on 64 B lines) |
| Page size | 16 KiB |
| L1 data cache | 64 KiB per core |
| L2 cache | 4 MiB (shared per core cluster) |
| RAM | 48 GB |

Consequences for this design: `headDim = 64` floats = 256 B = exactly 2 cache lines, so a KV-head
tile at a 32-token block size is 8 KiB — long enough to amortize per-tile overhead, short enough
(≤2 pages) to stay TLB-friendly. And critically: **a single layer's K+V at a 2048-token context is
~3 MiB — it very nearly fits inside this CPU's 4 MiB L2.** Any microbenchmark that simulates only
one layer will show the layout change doing almost nothing, because the "memory-bound" walk never
actually leaves L2. A benchmark has to cycle through all 30 layers (~90 MiB at ctx 2048) to
reproduce the real engine's cold-cache behavior — noted as a trap to avoid in the ADR.

## Quantitative estimate: which change matters more

At context length 2048, per decode token, across all 30 layers, K+V DRAM traffic:

- **Today** (9 separate reads per KV-head row, one per query head): `2 × 9 × 2048 × 256 B × 30 ≈ 283 MB`
- **After reordering the loop only** (3 reads per KV-head row — matches GQA's actual sharing): `2 × 3 × 2048 × 256 B × 30 ≈ 94 MB`

Against the model's weight-read traffic per token (F32 weights, ~537 MB — this engine has no
quantized runtime path yet), that reordering alone cuts total per-token memory traffic from ~820 MB
to ~631 MB, a 23% reduction, **before any layout or paging change**. This is why the implementation
plan lands the loop reorder as its own commit, measured independently of the layout and paging
changes that follow it.

## Sources

- [Modular — The Five Eras of KVCache](https://www.modular.com/blog/the-five-eras-of-kvcache)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM paper)](https://arxiv.org/pdf/2309.06180)
- [vLLM — Automatic Prefix Caching design docs](https://docs.vllm.ai/en/stable/design/prefix_caching)
- [llama.cpp — Paged KV cache and scheduler, Phase 1 design discussion #21961](https://github.com/ggml-org/llama.cpp/discussions/21961)
- [llama.cpp — Memory Management and KV Cache (DeepWiki)](https://deepwiki.com/ggml-org/llama.cpp/3.6-memory-management-and-kv-cache)
- [dotLLM — Introducing dotLLM blog post](https://kokosa.dev/blog/2026/dotllm/)
- [A Survey on LLM Acceleration based on KV Cache Management](https://arxiv.org/pdf/2412.19442)
- [SnapKV: LLM Knows What You Are Looking For Before Generation](https://arxiv.org/pdf/2404.14469)
- [KVQuant: Towards 10M Context Length LLM Inference with KV Cache Quantization](https://arxiv.org/pdf/2401.18079)
- [head-major (HND) KV cache layout PR discussion, torch-spyre](https://github.com/torch-spyre/spyre-inference/pull/870)
