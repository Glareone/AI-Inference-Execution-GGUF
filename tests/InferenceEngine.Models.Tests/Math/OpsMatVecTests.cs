using InferenceEngine.Models.Math;

namespace InferenceEngine.Models.Tests.Math;

/// <summary>
/// Business case: <c>Ops.MatVec</c> is the one matrix-vector entry point the model calls. Whatever
/// backend it picks (Accelerate on a Mac, the managed loop elsewhere) it must return the
/// mathematically correct product, and tests must be able to pin the managed loop (for bit-exact
/// golden hashes) without disturbing tests running in parallel. All tests here run on every OS.
/// </summary>
/// <remarks>Uses only <see cref="MathF"/>; <c>Math</c> is shadowed by the project namespace.</remarks>
public class OpsMatVecTests
{
    private static readonly float[] Weight = [1f, 2f, 3f, 4f, 5f, 6f];
    private static readonly float[] X = [1f, 0f, -1f];

    [Fact]
    public void MatVec_OnHandComputedCase_ReturnsCorrectProduct()
    {
        var destination = new float[2];

        Ops.MatVec(Weight, 2, 3, X, destination);

        Assert.Equal(-2f, destination[0], 1e-6f);
        Assert.Equal(-2f, destination[1], 1e-6f);
    }

    [Fact]
    public void MatVecManaged_OnHandComputedCase_ReturnsCorrectProduct()
    {
        var destination = new float[2];

        Ops.MatVecManaged(Weight, 2, 3, X, destination);

        Assert.Equal(-2f, destination[0], 1e-6f);
        Assert.Equal(-2f, destination[1], 1e-6f);
    }

    [Fact]
    public void MatVecManaged_OnRandomMatrix_MatchesPlainScalarTripleLoop()
    {
        const int outDim = 13, inDim = 37;
        var weight = AccelerateBlasTests.RandomData(outDim * inDim, seed: 7);
        var x = AccelerateBlasTests.RandomData(inDim, seed: 8);
        var actual = new float[outDim];

        Ops.MatVecManaged(weight, outDim, inDim, x, actual);

        for (var o = 0; o < outDim; o++)
        {
            double sum = 0;
            for (var i = 0; i < inDim; i++)
            {
                sum += (double)weight[o * inDim + i] * x[i];
            }

            Assert.Equal((float)sum, actual[o], 1e-5f);
        }
    }

    [Fact]
    public void MatVec_WithZeroOutDim_DoesNotThrowAndTouchesNothing()
    {
        var destination = new float[] { 5f, 5f };

        Ops.MatVec(Weight, 0, 3, X, destination);

        Assert.Equal(new float[] { 5f, 5f }, destination);
    }

    [Fact]
    public void MatVec_WithZeroInDim_AgreesWithManagedLoop()
    {
        // An empty dot product is 0 in the managed loop; the dispatcher must not send this to BLAS.
        var viaOps = new float[] { 5f, 5f };
        var viaManaged = new float[] { 5f, 5f };

        Ops.MatVec(ReadOnlySpan<float>.Empty, 2, 0, ReadOnlySpan<float>.Empty, viaOps);
        Ops.MatVecManaged(ReadOnlySpan<float>.Empty, 2, 0, ReadOnlySpan<float>.Empty, viaManaged);

        Assert.Equal(viaManaged, viaOps);
        Assert.Equal(new float[] { 0f, 0f }, viaOps);
    }

    [Fact]
    public void ForceManagedMatVec_InsideScope_IsBitIdenticalToManagedLoop()
    {
        const int outDim = 64, inDim = 100;
        var weight = AccelerateBlasTests.RandomData(outDim * inDim, seed: 3);
        var x = AccelerateBlasTests.RandomData(inDim, seed: 4);
        var expected = new float[outDim];
        var actual = new float[outDim];
        Ops.MatVecManaged(weight, outDim, inDim, x, expected);

        using (Ops.ForceManagedMatVec())
        {
            Ops.MatVec(weight, outDim, inDim, x, actual);
        }

        Assert.Equal(expected, actual); // exact, not tolerance
    }

    [Fact]
    public void ForceManagedMatVec_AfterDispose_StillComputesCorrectResult()
    {
        // On macOS the scope being lifted switches back to Accelerate (bits may differ), so only
        // the value is asserted.
        var destination = new float[2];
        Ops.ForceManagedMatVec().Dispose();

        Ops.MatVec(Weight, 2, 3, X, destination);

        Assert.Equal(-2f, destination[0], 1e-6f);
        Assert.Equal(-2f, destination[1], 1e-6f);
    }

    [Fact]
    public async Task ForceManagedMatVec_InOneTask_DoesNotAffectConcurrentTaskOutsideScope()
    {
        const int outDim = 256, inDim = 256;
        var weight = AccelerateBlasTests.RandomData(outDim * inDim, seed: 5);
        var x = AccelerateBlasTests.RandomData(inDim, seed: 6);
        var managed = new float[outDim];
        Ops.MatVecManaged(weight, outDim, inDim, x, managed);
        using var bothStarted = new Barrier(2);

        var inside = Task.Run(() =>
        {
            using var _ = Ops.ForceManagedMatVec();
            bothStarted.SignalAndWait();
            var results = new List<float[]>();
            for (var n = 0; n < 50; n++)
            {
                var d = new float[outDim];
                Ops.MatVec(weight, outDim, inDim, x, d);
                results.Add(d);
            }

            return results;
        });
        var outside = Task.Run(() =>
        {
            bothStarted.SignalAndWait();
            var last = new float[outDim];
            for (var n = 0; n < 50; n++)
            {
                Ops.MatVec(weight, outDim, inDim, x, last);
            }

            return last;
        });

        var insideResults = await inside;
        var outsideResult = await outside;

        Assert.All(insideResults, r => Assert.Equal(managed, r)); // bit-identical
        for (var o = 0; o < outDim; o++)
        {
            Assert.Equal(managed[o], outsideResult[o], 1e-4f); // may be Accelerate; value-correct
        }
    }
}
