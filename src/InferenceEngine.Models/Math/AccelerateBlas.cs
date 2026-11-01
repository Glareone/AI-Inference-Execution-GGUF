using System.Runtime.InteropServices;

namespace InferenceEngine.Models.Math;

/// <summary>
/// Safe wrapper over Apple Accelerate's CBLAS <c>cblas_sgemv</c>. The only place in the
/// codebase that crosses into native code for CPU math (ADR 261101). Callers pass spans; the
/// <c>LibraryImport</c> generator pins them for the duration of the call, so no <c>unsafe</c>
/// code is written here. <c>AllowUnsafeBlocks</c> is on in this project only because that
/// generator requires it (SYSLIB1062).
/// </summary>
internal static partial class AccelerateBlas
{
    private const string AccelerateLib = "/System/Library/Frameworks/Accelerate.framework/Accelerate";

    // Standard CBLAS enum values (cblas.h): CblasRowMajor = 101, CblasNoTrans = 111.
    private const int CblasRowMajor = 101;
    private const int CblasNoTrans = 111;

    /// <summary>
    /// True when the Accelerate framework loaded and <c>cblas_sgemv</c> resolved and returned the
    /// right answer for a 1x1 probe. False off macOS (e.g. the Linux CI runner) or if the symbol is
    /// missing. Decided once, so a missing library costs one failed probe, not one per call.
    /// </summary>
    public static bool IsAvailable { get; } = Probe();

    /// <summary><c>destination = weight * x</c> for a row-major <c>[outDim, inDim]</c> matrix.</summary>
    /// <exception cref="InvalidOperationException">Accelerate is not available on this machine.</exception>
    /// <exception cref="ArgumentException">A span is shorter than the dimensions require.</exception>
    public static void MatVec(ReadOnlySpan<float> weight, int outDim, int inDim, ReadOnlySpan<float> x, Span<float> destination)
    {
        if (!IsAvailable)
        {
            throw new InvalidOperationException("Apple Accelerate is not available on this machine.");
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inDim);

        // Native code does not bounds-check. A short span here would read or write past the end
        // of the managed array, so check before the call.
        ArgumentOutOfRangeException.ThrowIfLessThan(weight.Length, checked(outDim * inDim), nameof(weight));
        ArgumentOutOfRangeException.ThrowIfLessThan(x.Length, inDim, nameof(x));
        ArgumentOutOfRangeException.ThrowIfLessThan(destination.Length, outDim, nameof(destination));

        // y = 1 * A * x + 0 * y. A is row-major [outDim, inDim], so lda = inDim. beta = 0 means
        // BLAS never reads the old contents of y.
        cblas_sgemv(CblasRowMajor, CblasNoTrans, outDim, inDim, 1f, weight, inDim, x, 1, 0f, destination, 1);
    }

    private static bool Probe()
    {
        if (!OperatingSystem.IsMacOS())
        {
            return false;
        }

        try
        {
            ReadOnlySpan<float> a = [2f];
            ReadOnlySpan<float> x = [3f];
            Span<float> y = [0f];
            cblas_sgemv(CblasRowMajor, CblasNoTrans, 1, 1, 1f, a, 1, x, 1, 0f, y, 1);
            return y[0] == 6f;
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException)
        {
            return false;
        }
    }

    [LibraryImport(AccelerateLib)]
    private static partial void cblas_sgemv(
        int order, int transA, int m, int n, float alpha,
        ReadOnlySpan<float> a, int lda,
        ReadOnlySpan<float> x, int incX,
        float beta, Span<float> y, int incY);
}
