using System.Diagnostics;
using InferenceEngine.Cli.Config;

namespace InferenceEngine.Cli.Tests;

/// <summary>
/// Business case: a user who mistypes the model path (or forgets it) must get a one-line error
/// and exit code 1 — not a .NET stack trace and an abort code. <c>Program.cs</c> is top-level
/// statements, so the only honest way to check its exit code and stderr is to run the built CLI
/// as a child process. These tests never load a real model: they fail before any weights are read.
/// </summary>
public class ProgramTests
{
    private sealed record CliResult(int ExitCode, string Stdout, string Stderr);

    [Fact]
    public async Task Run_WithNonexistentModelFile_PrintsOneLineErrorAndExitsWithCode1()
    {
        var missing = Path.Combine(Path.GetTempPath(), $"no-such-model-{Guid.NewGuid():N}.gguf");

        var result = await RunCliAsync("--model", missing, "--prompt", "hi");

        Assert.Equal(1, result.ExitCode);
        Assert.Contains("Could not find file", result.Stderr);
        Assert.Contains(missing, result.Stderr);
        AssertNoStackTrace(result);
    }

    [Fact]
    public async Task Run_WithModelPathInMissingDirectory_PrintsOneLineErrorAndExitsWithCode1()
    {
        var missing = Path.Combine(Path.GetTempPath(), $"no-such-dir-{Guid.NewGuid():N}", "model.gguf");

        var result = await RunCliAsync("--model", missing, "--prompt", "hi");

        Assert.Equal(1, result.ExitCode);
        Assert.Contains("Could not find a part of the path", result.Stderr);
        AssertNoStackTrace(result);
    }

    [Fact]
    public async Task Run_WithoutModelOption_PrintsUsageAndExitsWithCode1()
    {
        var result = await RunCliAsync("--prompt", "hi");

        Assert.Equal(1, result.ExitCode);
        Assert.Contains(CliOptions.UsageText.Split('\n')[0].Trim(), result.Stderr);
        AssertNoStackTrace(result);
    }

    private static void AssertNoStackTrace(CliResult result)
    {
        Assert.DoesNotContain("Unhandled exception", result.Stderr);
        Assert.DoesNotContain("   at ", result.Stderr);
    }

    /// <summary>
    /// Runs the CLI assembly the test project already references (same build output directory, so
    /// its runtimeconfig is next to it). Uses an empty temp working directory and strips every
    /// <c>INFERENCE_*</c> variable so a developer's real <c>.env</c> or shell environment can't
    /// supply a model path and change the outcome.
    /// </summary>
    private static async Task<CliResult> RunCliAsync(params string[] args)
    {
        var cliDll = typeof(CliOptions).Assembly.Location;
        var workDir = Directory.CreateTempSubdirectory("cli-test-").FullName;
        try
        {
            var psi = new ProcessStartInfo("dotnet")
            {
                WorkingDirectory = workDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            psi.ArgumentList.Add(cliDll);
            foreach (var arg in args)
            {
                psi.ArgumentList.Add(arg);
            }

            foreach (var key in psi.Environment.Keys.Where(k => k.StartsWith("INFERENCE_", StringComparison.Ordinal)).ToList())
            {
                psi.Environment.Remove(key);
            }

            using var process = Process.Start(psi)!;
            var stdout = process.StandardOutput.ReadToEndAsync();
            var stderr = process.StandardError.ReadToEndAsync();

            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(60));
            await process.WaitForExitAsync(timeout.Token);

            return new CliResult(process.ExitCode, await stdout, await stderr);
        }
        finally
        {
            Directory.Delete(workDir, recursive: true);
        }
    }
}
