using System.Numerics.Tensors;

namespace InferenceEngine.Models.Math;

/// <summary>
/// Thin, readable wrappers over <see cref="TensorPrimitives"/> for the handful of operations
/// the Llama forward pass needs. Kept separate from <c>LlamaModel</c> so the layer-wiring code
/// (embed -> norm -> attend -> ffn) reads like the architecture, not like SIMD plumbing.
/// </summary>
internal static class Ops
{
    public static void RmsNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> weight, float eps, Span<float> destination)
    {
        var meanSquare = TensorPrimitives.SumOfSquares(x) / x.Length;
        var invRms = 1f / MathF.Sqrt(meanSquare + eps);
        TensorPrimitives.Multiply(x, invRms, destination);
        TensorPrimitives.Multiply(destination, weight, destination);
    }

    /// <summary>
    /// Environment variable that forces the managed <see cref="MatVecManaged"/> loop when set to
    /// <c>managed</c>. For A/B benchmarking and for reproducing the bit-exact golden-hash path;
    /// not a feature flag.
    /// </summary>
    public const string MatVecBackendEnvVar = "INFERENCE_MATVEC";

    private static readonly bool UseAccelerate =
        AccelerateBlas.IsAvailable
        && !string.Equals(Environment.GetEnvironmentVariable(MatVecBackendEnvVar), "managed", StringComparison.OrdinalIgnoreCase);

    // Per-execution-context override so one test can pin the managed loop while others, running
    // in parallel, keep using Accelerate. Only tests set it.
    private static readonly AsyncLocal<bool> ForceManaged = new();

    /// <summary>
    /// Makes <see cref="MatVec"/> use <see cref="MatVecManaged"/> for the current execution
    /// context until the returned scope is disposed. The golden logit hash is bit-exact, and
    /// BLAS may sum in a different order on a different chip, so the golden tests pin the managed
    /// loop instead of pinning one machine's Accelerate output.
    /// </summary>
    public static IDisposable ForceManagedMatVec()
    {
        ForceManaged.Value = true;
        return new ManagedScope();
    }

    private sealed class ManagedScope : IDisposable
    {
        public void Dispose() => ForceManaged.Value = false;
    }

    /// <summary>
    /// Multiplies a row-major <c>[outDim, inDim]</c> weight matrix by <paramref name="x"/>.
    /// GGUF stores weights with the input dimension contiguous (<c>ne[0]</c>), which is
    /// exactly this layout — no transpose needed.
    /// </summary>
    /// <remarks>
    /// Uses Apple Accelerate <c>cblas_sgemv</c> when available (ADR 261101), else
    /// <see cref="MatVecManaged"/>. The two can differ in the last float bits because BLAS may sum
    /// in a different order.
    /// </remarks>
    public static void MatVec(ReadOnlySpan<float> weight, int outDim, int inDim, ReadOnlySpan<float> x, Span<float> destination)
    {
        if (UseAccelerate && !ForceManaged.Value && outDim > 0 && inDim > 0)
        {
            AccelerateBlas.MatVec(weight, outDim, inDim, x, destination);
            return;
        }

        MatVecManaged(weight, outDim, inDim, x, destination);
    }

    /// <summary>
    /// The original loop: one <see cref="TensorPrimitives.Dot"/> per output row. Always available.
    /// It is the reference the Accelerate path is tested against.
    /// </summary>
    public static void MatVecManaged(ReadOnlySpan<float> weight, int outDim, int inDim, ReadOnlySpan<float> x, Span<float> destination)
    {
        for (var o = 0; o < outDim; o++)
        {
            destination[o] = TensorPrimitives.Dot(weight.Slice(o * inDim, inDim), x);
        }
    }

    /// <summary>
    /// Interleaved-pair ("normal" / GPT-J-style) rotary embedding, applied in place to every
    /// head packed into <paramref name="vec"/>. GGUF llama weights are permuted at conversion
    /// time (see <c>convert_hf_to_gguf.py</c>) specifically so this layout — not HF's
    /// half-split rotation — is the one that reproduces the original model's behavior.
    /// </summary>
    /// <remarks>
    /// Thin wrapper over <see cref="RopeTable"/> + <see cref="RopeHead"/>, kept for callers (and
    /// tests) that don't need to hoist the table themselves. Per-call-site hot paths that apply
    /// RoPE to multiple heads/slots at the same position should call <see cref="RopeTable"/> once
    /// and <see cref="RopeHead"/> per head instead, since the table depends only on
    /// (position, freqBase, headDim) — not on the head.
    /// </remarks>
    public static void Rope(Span<float> vec, int numHeads, int headDim, int position, float freqBase)
    {
        var half = headDim / 2;
        Span<float> cosCache = half <= 128 ? stackalloc float[half] : new float[half];
        Span<float> sinCache = half <= 128 ? stackalloc float[half] : new float[half];
        RopeTable(headDim, position, freqBase, cosCache, sinCache);

        for (var h = 0; h < numHeads; h++)
        {
            RopeHead(vec.Slice(h * headDim, headDim), cosCache, sinCache);
        }
    }

    /// <summary>
    /// Fills <paramref name="cos"/>/<paramref name="sin"/> (each <c>headDim/2</c> long) with the
    /// per-pair rotation angle's cos/sin at <paramref name="position"/> — depends only on
    /// (position, freqBase, headDim), not on any one head, so a caller applying RoPE to several
    /// heads at the same position computes this once and reuses it via <see cref="RopeHead"/>
    /// instead of recomputing the same <see cref="MathF.Pow"/>/<see cref="MathF.Cos"/>/
    /// <see cref="MathF.Sin"/> calls per head.
    /// </summary>
    public static void RopeTable(int headDim, int position, float freqBase, Span<float> cos, Span<float> sin)
    {
        var half = headDim / 2;
        for (var i = 0; i < half; i++)
        {
            var theta = position * MathF.Pow(freqBase, -2f * i / headDim);
            cos[i] = MathF.Cos(theta);
            sin[i] = MathF.Sin(theta);
        }
    }

    /// <summary>
    /// Applies the interleaved-pair rotation to one head (<c>headDim</c> floats, in place) using
    /// a table already filled by <see cref="RopeTable"/>.
    /// </summary>
    public static void RopeHead(Span<float> head, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin)
    {
        var half = cos.Length;
        for (var i = 0; i < half; i++)
        {
            var c = cos[i];
            var s = sin[i];
            var i0 = 2 * i;
            var i1 = i0 + 1;
            var x0 = head[i0];
            var x1 = head[i1];
            head[i0] = (x0 * c) - (x1 * s);
            head[i1] = (x0 * s) + (x1 * c);
        }
    }

    public static void Softmax(ReadOnlySpan<float> x, Span<float> destination) => TensorPrimitives.SoftMax(x, destination);

    /// <summary>SwiGLU: <c>Silu(gate) * up</c>, where <c>Silu(x) = x * sigmoid(x)</c>.</summary>
    public static void SwiGlu(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> sigmoidScratch, Span<float> destination)
    {
        TensorPrimitives.Sigmoid(gate, sigmoidScratch);
        TensorPrimitives.Multiply(gate, sigmoidScratch, destination);
        TensorPrimitives.Multiply(destination, up, destination);
    }
}
