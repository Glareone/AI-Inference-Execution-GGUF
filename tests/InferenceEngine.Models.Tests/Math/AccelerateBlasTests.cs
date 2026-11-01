using InferenceEngine.Models.Math;

namespace InferenceEngine.Models.Tests.Math;

/// <summary>
/// Business case: on Apple Silicon the matrix-vector product runs through Apple Accelerate
/// (<c>cblas_sgemv</c>, ADR 261101) for speed. It must give the same answer as the portable managed
/// loop (up to float summation-order noise), must never read or write outside the spans it was
/// given (native code has no bounds checks), and must fail clearly when called with bad sizes.
/// Tests that need Accelerate skip with a clear reason on Linux/Windows CI; the cross-platform
/// <c>Ops</c> tests live in <see cref="OpsMatVecTests"/>.
/// </summary>
/// <remarks>Uses only <see cref="MathF"/>; <c>Math</c> is shadowed by the project namespace.</remarks>
public class AccelerateBlasTests
{
    private const string SkipReason = "Apple Accelerate is only available on macOS.";

    private static void RequireAccelerate() =>
        Assert.SkipUnless(OperatingSystem.IsMacOS(), SkipReason);

    /// <summary>Deterministic pseudo-random floats in [-1, 1].</summary>
    internal static float[] RandomData(int length, int seed)
    {
        var rng = new Random(seed);
        var data = new float[length];
        for (var i = 0; i < data.Length; i++)
        {
            data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        return data;
    }

    [Fact]
    public void IsAvailable_OnMacOS_IsTrue()
    {
        // Proves the framework loaded and cblas_sgemv resolved (ADR 261101 required this check).
        RequireAccelerate();

        Assert.True(AccelerateBlas.IsAvailable);
    }

    [Fact]
    public void IsAvailable_OffMacOS_IsFalse()
    {
        Assert.SkipWhen(OperatingSystem.IsMacOS(), "Only meaningful off macOS.");

        Assert.False(AccelerateBlas.IsAvailable);
    }

    [Fact]
    public void MatVec_OnSmallHandComputedCase_ReturnsDotProducts()
    {
        RequireAccelerate();
        // [2,3] row-major: row0 = [1,2,3], row1 = [4,5,6]; x = [1,0,-1] -> [-2, -2].
        float[] weight = [1f, 2f, 3f, 4f, 5f, 6f];
        float[] x = [1f, 0f, -1f];
        var destination = new float[2];

        AccelerateBlas.MatVec(weight, 2, 3, x, destination);

        Assert.Equal(-2f, destination[0], 1e-6f);
        Assert.Equal(-2f, destination[1], 1e-6f);
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(1, 64)]
    [InlineData(64, 1)]
    [InlineData(4, 4)]
    [InlineData(7, 3)]
    [InlineData(17, 33)]
    [InlineData(576, 576)]
    [InlineData(1536, 576)]
    [InlineData(576, 1536)]
    [InlineData(49152, 576)]
    public void MatVec_OnVariousShapes_MatchesManagedLoopWithinFloatTolerance(int outDim, int inDim)
    {
        RequireAccelerate();
        var weight = RandomData(outDim * inDim, seed: 1);
        var x = RandomData(inDim, seed: 2);
        var expected = new float[outDim];
        var actual = new float[outDim];

        Ops.MatVecManaged(weight, outDim, inDim, x, expected);
        AccelerateBlas.MatVec(weight, outDim, inDim, x, actual);

        for (var o = 0; o < outDim; o++)
        {
            // Each float sum of inDim terms carries error <= ~inDim * 2^-24 * sum|w*x| (worst
            // case); two different summation orders differ by at most twice that. Using the
            // per-row sum of |w*x| makes the bound scale with inDim and with cancellation.
            var absSum = 0f;
            for (var i = 0; i < inDim; i++)
            {
                absSum += MathF.Abs(weight[o * inDim + i] * x[i]);
            }

            var tolerance = 2f * inDim * 5.96e-8f * absSum + 1e-6f;
            Assert.True(
                MathF.Abs(expected[o] - actual[o]) <= tolerance,
                $"Row {o} of [{outDim}x{inDim}]: managed {expected[o]:R}, accelerate {actual[o]:R}, tol {tolerance:E3}");
        }
    }

    [Fact]
    public void MatVec_WithGarbageInDestination_OverwritesEveryElement()
    {
        // beta = 0 must mean "never read old y": NaN garbage must not leak into the result.
        RequireAccelerate();
        float[] weight = [1f, 2f, 3f, 4f, 5f, 6f];
        float[] x = [1f, 1f, 1f];
        float[] destination = [float.NaN, float.PositiveInfinity];

        AccelerateBlas.MatVec(weight, 2, 3, x, destination);

        Assert.Equal(6f, destination[0], 1e-6f);
        Assert.Equal(15f, destination[1], 1e-6f);
    }

    [Fact]
    public void MatVec_WithLargerDestination_WritesOnlyFirstOutDimElements()
    {
        RequireAccelerate();
        float[] weight = [1f, 2f, 3f, 4f, 5f, 6f];
        float[] x = [1f, 1f, 1f];
        float[] destination = [0f, 0f, 111f, 222f, float.NaN];

        AccelerateBlas.MatVec(weight, 2, 3, x, destination);

        Assert.Equal(6f, destination[0], 1e-6f);
        Assert.Equal(15f, destination[1], 1e-6f);
        Assert.Equal(111f, destination[2]);
        Assert.Equal(222f, destination[3]);
        Assert.True(float.IsNaN(destination[4]));
    }

    [Fact]
    public void MatVec_WithLargerWeightAndInput_IgnoresTrailingElements()
    {
        // Trailing poison would change the result if read.
        RequireAccelerate();
        float[] weight = [1f, 2f, 3f, 4f, 5f, 6f, 1e30f, float.NaN];
        float[] x = [1f, 1f, 1f, 1e30f, float.NaN];
        var destination = new float[2];

        AccelerateBlas.MatVec(weight, 2, 3, x, destination);

        Assert.Equal(6f, destination[0], 1e-6f);
        Assert.Equal(15f, destination[1], 1e-6f);
    }

    [Fact]
    public void MatVec_OnSlicedWeightWithNonZeroOffset_UsesTheSlice()
    {
        // LlamaWeights hands out per-layer slices of one big array.
        RequireAccelerate();
        float[] big = [9f, 9f, 9f, 1f, 2f, 3f, 4f, 5f, 6f, 9f];
        float[] bigX = [7f, 1f, 1f, 1f];
        var bigDest = new float[] { -1f, 0f, 0f, -1f };

        AccelerateBlas.MatVec(big.AsSpan(3, 6), 2, 3, bigX.AsSpan(1, 3), bigDest.AsSpan(1, 2));

        Assert.Equal(-1f, bigDest[0]);
        Assert.Equal(6f, bigDest[1], 1e-6f);
        Assert.Equal(15f, bigDest[2], 1e-6f);
        Assert.Equal(-1f, bigDest[3]);
    }

    [Theory]
    [InlineData(5, 2, 3, 3, 2, "weight")]
    [InlineData(6, 2, 3, 2, 2, "x")]
    [InlineData(6, 2, 3, 3, 1, "destination")]
    [InlineData(6, 0, 3, 3, 2, "outDim")]
    [InlineData(6, 2, 0, 3, 2, "inDim")]
    [InlineData(6, -1, 3, 3, 2, "outDim")]
    [InlineData(6, 2, -3, 3, 2, "inDim")]
    public void MatVec_WithInvalidSizes_ThrowsAndLeavesDestinationUntouched(
        int weightLength, int outDim, int inDim, int xLength, int destLength, string expectedParam)
    {
        RequireAccelerate();
        var weight = new float[weightLength];
        var x = new float[xLength];
        var destination = new float[destLength];
        Array.Fill(destination, 42f);

        var ex = Assert.Throws<ArgumentOutOfRangeException>(
            () => AccelerateBlas.MatVec(weight, outDim, inDim, x, destination));

        Assert.Equal(expectedParam, ex.ParamName);
        Assert.All(destination, v => Assert.Equal(42f, v));
    }
}
