using InferenceEngine.Engine;
using InferenceEngine.Engine.Config;
using InferenceEngine.Models.Math;

namespace InferenceEngine.Models.Tests.Llama;

/// <summary>
/// Business case: switching matrix-vector math to Apple Accelerate (ADR 261101) changes float
/// summation order, so logits differ in the last bits. That must not change what the model says:
/// greedy generation from a fixed prompt on the real SmolLM2-135M model must produce the same token
/// ids with Accelerate as with the managed loop, and the prefill ranking must agree on the top
/// token. Skips unless <c>INFERENCE_MODEL</c> points at the GGUF file (see <see cref="GoldenModelFixture"/>).
/// On non-macOS both paths are the managed loop, so the test passes trivially there.
/// </summary>
public sealed class AcceleratedGenerationTests(GoldenModelFixture fixture) : IClassFixture<GoldenModelFixture>
{
    private const string ShortPrompt = "What is machine learning? Explain briefly.";
    private const int NewTokens = 32;

    [Fact]
    public void GreedyGeneration_WithAccelerate_MatchesManagedLoopTokenForToken()
    {
        Assert.SkipUnless(fixture.Session is not null, fixture.SkipReason);
        var session = fixture.Session!;
        var options = new GenerationOptions(MaxNewTokens: NewTokens, Temperature: 0f);

        var accelerated = session.Generate(ShortPrompt, options).Select(t => t.Id).ToList();
        List<int> managed;
        using (Ops.ForceManagedMatVec())
        {
            // ToList inside the scope: Generate is lazy.
            managed = session.Generate(ShortPrompt, options).Select(t => t.Id).ToList();
        }

        Assert.NotEmpty(managed);
        var firstDiff = FirstDivergence(accelerated, managed);
        Assert.True(
            firstDiff < 0,
            $"Greedy sequences diverge at generated index {firstDiff}. " +
            $"Accelerate: [{string.Join(",", accelerated)}] Managed: [{string.Join(",", managed)}]");
    }

    [Fact]
    public void Prefill_WithAccelerate_AgreesWithManagedLoopOnTopToken()
    {
        Assert.SkipUnless(fixture.Session is not null, fixture.SkipReason);
        var session = fixture.Session!;
        var promptIds = session.Tokenize(ShortPrompt, raw: false);

        var accelerated = session.PrefillTopLogits(promptIds, 10);
        (int Id, string Text, float Logit)[] managed;
        using (Ops.ForceManagedMatVec())
        {
            managed = session.PrefillTopLogits(promptIds, 10);
        }

        Assert.Equal(managed[0].Id, accelerated[0].Id);

        // Top-10 ids need not match exactly: float drift can reorder or swap near-tied logits at
        // the edge of the top 10. Requiring >= 8 shared ids still catches a broken matmul.
        var overlap = accelerated.Select(t => t.Id).Intersect(managed.Select(t => t.Id)).Count();
        Assert.True(overlap >= 8, $"Only {overlap}/10 top ids shared between Accelerate and managed paths.");
    }

    private static int FirstDivergence(List<int> a, List<int> b)
    {
        var n = System.Math.Min(a.Count, b.Count);
        for (var i = 0; i < n; i++)
        {
            if (a[i] != b[i])
            {
                return i;
            }
        }

        return a.Count == b.Count ? -1 : n;
    }
}
