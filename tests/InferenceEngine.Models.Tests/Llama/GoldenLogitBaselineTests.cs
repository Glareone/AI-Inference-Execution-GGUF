using InferenceEngine.Engine;
using InferenceEngine.Models.Math;

namespace InferenceEngine.Models.Tests.Llama;

/// <summary>
/// Golden baseline captured against the CURRENT (pre-rewrite) <c>SimpleKvCache</c> implementation,
/// before the KV-cache rewrite described in <c>docs/architecture/260914-kv-cache.md</c> begins.
/// Business guarantee under test: prefilling the real SmolLM2-135M-Instruct GGUF model with a
/// fixed prompt always produces the identical next-token logit distribution, bit-for-bit — a
/// short prompt (well within one future 32-token cache block) and a long prompt (spanning several
/// future cache blocks) are both covered, because a KV-cache rewrite is exactly the kind of change
/// that could silently corrupt results only once a prompt crosses a block boundary. Every later
/// step of the rewrite must reproduce these exact hashes; any difference, however small, is a
/// regression this test exists to catch.
///
/// Skips (via <c>Assert.SkipUnless</c>, not a failure) when <c>INFERENCE_MODEL</c> isn't set to
/// an existing GGUF file, since most machines running the suite won't have the 270 MB model
/// downloaded. Run locally with:
/// <c>INFERENCE_MODEL=~/.cache/inference-engine/models/SmolLM2-135M-Instruct-f16.gguf dotnet test ...</c>
/// (with <c>~</c> already expanded by the shell).
/// </summary>
public sealed class GoldenLogitBaselineTests(GoldenModelFixture fixture) : IClassFixture<GoldenModelFixture>
{
    private const string ShortPrompt = "What is machine learning? Explain briefly.";

    private const string LongPrompt =
        "Explain, in careful and precise technical detail suitable for a software engineer new " +
        "to the topic, how a transformer language model generates text one token at a time, " +
        "covering tokenization of the input text into subword units, the embedding lookup that " +
        "turns token ids into vectors, the role of positional information via rotary embeddings, " +
        "how self-attention lets each position attend to all previous positions through query key " +
        "and value projections, why grouped-query attention shares key and value heads across " +
        "multiple query heads to save memory, how a feed-forward network with a gating " +
        "nonlinearity processes each position independently after attention, and finally how the " +
        "model produces a probability distribution over the vocabulary for the next token.";

    // Golden hashes captured from a real run of this test against the pre-rewrite SimpleKvCache
    // implementation (commit on KV-adjustments before the rewrite lands), model
    // SmolLM2-135M-Instruct-f16.gguf. Every later KV-cache rewrite step must reproduce these
    // exactly, bit-for-bit — see docs/architecture/260914-kv-cache.md.
    private const ulong ShortPromptExpectedHash = 0xB07058A15AA1650AUL;
    private const ulong LongPromptExpectedHash = 0xECC1669FE5C7DB32UL;

    [Fact]
    public void Prefill_OnShortPrompt_ProducesGoldenLogitHash()
    {
        Assert.SkipUnless(fixture.Session is not null, fixture.SkipReason);
        var session = fixture.Session!;

        var (hash, top10, tokenCount) = CapturePrefillLogits(session, ShortPrompt);

        Assert.True(
            hash == ShortPromptExpectedHash,
            $"Short-prompt logit hash changed: expected 0x{ShortPromptExpectedHash:X16}, got " +
            $"0x{hash:X16} ({tokenCount} prompt tokens). Top-10: {FormatTop10(top10)}");
    }

    [Fact]
    public void Prefill_OnLongMultiBlockPrompt_ProducesGoldenLogitHash()
    {
        Assert.SkipUnless(fixture.Session is not null, fixture.SkipReason);
        var session = fixture.Session!;

        var (hash, top10, tokenCount) = CapturePrefillLogits(session, LongPrompt);

        // Deliberately spans multiple future 32-token KV-cache blocks, not just one, so the
        // golden baseline exercises cross-block prefill behavior too.
        Assert.True(
            tokenCount > 32,
            $"Long prompt encoded to only {tokenCount} tokens after chat templating — expected " +
            "it to span multiple 32-token cache blocks.");

        Assert.True(
            hash == LongPromptExpectedHash,
            $"Long-prompt logit hash changed: expected 0x{LongPromptExpectedHash:X16}, got " +
            $"0x{hash:X16} ({tokenCount} prompt tokens). Top-10: {FormatTop10(top10)}");
    }

    /// <summary>
    /// Drives prefill through <see cref="InferenceSession.PrefillTopLogits"/> — the same code
    /// path <c>InferenceEngine.Cli</c>'s <c>--debug-logits</c> flag uses — asking for every
    /// vocabulary entry (not just the top few) so the full logit vector can be reassembled in
    /// token-id order and hashed.
    /// </summary>
    private static (ulong Hash, (int Id, string Text, float Logit)[] Top10, int TokenCount) CapturePrefillLogits(
        InferenceSession session, string prompt)
    {
        // The hashes are bit-exact, so pin the managed MatVec loop. Accelerate may sum in a
        // different order (and differently per chip); its correctness is covered by
        // AcceleratedGenerationTests (same greedy tokens) instead. ADR 261101.
        using var _ = Ops.ForceManagedMatVec();

        var promptIds = session.Tokenize(prompt, raw: false);
        var vocabSize = session.Config.VocabSize;

        var rankedByLogitDescending = session.PrefillTopLogits(promptIds, vocabSize);
        Assert.Equal(vocabSize, rankedByLogitDescending.Length);

        var logitsByTokenId = new float[vocabSize];
        foreach (var (id, _, logit) in rankedByLogitDescending)
        {
            logitsByTokenId[id] = logit;
        }

        var hash = LogitHash.Fnv1a(logitsByTokenId);
        var top10 = rankedByLogitDescending.Take(10).ToArray();
        return (hash, top10, promptIds.Count);
    }

    private static string FormatTop10((int Id, string Text, float Logit)[] top10) =>
        string.Join(", ", top10.Select(t => $"[{t.Id} '{t.Text.Replace("\n", "\\n")}' {t.Logit:F3}]"));
}
