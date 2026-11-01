# `unsafe` scope

- `Directory.Build.props` sets `AllowUnsafeBlocks=false`. This applies to `Core`, `Engine`,
  `Tokenizers`, `Cli`, and every test project. Keep them safe-only.
- **One exception: `InferenceEngine.Models`.** Its `.csproj` sets `AllowUnsafeBlocks=true`.
  Reason: the `LibraryImport` source generator needs it (`SYSLIB1062`) for the Apple
  Accelerate / MLX P/Invoke declarations. Decided in
  [ADR 261101](../../docs/architecture/261101-apple-silicon-acceleration-strategy.md).
- The flag is not a license. Do not write `unsafe` blocks or raw-pointer code, even in `Models`,
  without the justify-and-confirm steps in AGENTS.md.
- Keep interop declarations in one place (`Models/Math/AccelerateBlas.cs` and its MLX
  successor). Wrap them behind a safe `Span<float>` API. Callers never see the P/Invoke.
- Do not copy the flag into another project. If another project needs interop, stop and ask.
