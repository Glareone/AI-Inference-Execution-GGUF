# InferenceEngine.Cli.Tests

Tests `src/InferenceEngine.Cli`: configuration loading, the console output boundary, and the
elapsed-time helper.

## Business scenarios covered

### `Config/CliOptions` (`Config/CliOptionsTests.cs`)

Configuration can come from a `.env` file, real environment variables (`INFERENCE_*` prefix), or
command-line flags, with flags always winning over environment, which wins over `.env` defaults.

- Running without `--model` (and without `INFERENCE_MODEL`) is rejected with a clear
  `ArgumentException` before any model-loading work would start.
- An environment variable is used when no flag is given; a flag overrides an environment variable
  when both are present.
- `--ban-words`/`INFERENCE_BAN_WORDS` follow the same flag-beats-env precedence, comma-split with
  **no** trimming — `"EPAM,EPAM ,E"` parses to three entries, the middle one keeping its trailing
  space, since a trailing space is a meaningfully different literal to ban, not incidental
  whitespace. With neither the flag nor the environment variable set, `BannedWords` is `null` —
  pinning banning as opt-in and fully off by default.
- Full three-tier precedence in one scenario: a `.env` file, a real environment variable, and a
  command-line flag are all present at once — the flag wins for the field it sets, the
  environment variable wins over `.env` for the field only it and `.env` set, and `.env` supplies
  the default for the field nothing else sets.
- `CliOptions.Load` takes a `loadDotEnv` parameter (seam added for these tests — see below) so a
  test can prove a `.env` file sitting in the working directory is never read when the caller
  opts out.

Every test saves and restores the exact `INFERENCE_*` environment variable keys `CliOptions`
reads, and never assumes they start out unset — a real `.env` or shell environment could exist on
the machine running these tests.

### `Config/DotEnvLoader` (`Config/DotEnvLoaderTests.cs`)

File-based tests using a unique temp `.env` path and a unique, GUID-suffixed environment variable
name per test, so nothing collides with a developer's real environment.

- `KEY=value` lines set the corresponding environment variable.
- Comments (`#`) and blank lines are ignored.
- Quoted values have their surrounding quotes stripped.
- An already-set environment variable is never overridden by the `.env` file.
- Loading a `.env` file that doesn't exist does nothing and does not throw.

### `IO/IOutput`, `IO/ConsoleOutput` (`IO/ConsoleOutputTests.cs`)

Thin by nature — `ConsoleOutput` is a direct pass-through to `Console`. A minimal test redirects
`Console.Out`/`Console.Error` to a `StringWriter` and confirms `Write`/`WriteLine`/`Error` reach
the expected stream.

### `Diagnostics/Timed` (`Diagnostics/TimedTests.cs`)

Minimal: confirms the wrapped action's result is returned unchanged and the reported elapsed time
is never negative.

### `Program` exit codes (`ProgramTests.cs`)

`Program.cs` is top-level statements, so these tests run the built CLI as a child process (empty
temp working directory, every `INFERENCE_*` variable stripped so a real `.env` or shell can't
supply a model). None of them load real weights.

- A `--model` path that doesn't exist prints a one-line `Could not find file '...'` message to
  stderr and exits with code 1 — no stack trace, no abort code.
- A `--model` path inside a missing directory does the same (`Could not find a part of the path`).
- Omitting `--model` prints the usage text and exits with code 1.

## Production-code seam added

`CliOptions.Load` gained an optional `bool loadDotEnv = true` parameter so tests can exercise the
flags/environment-merge logic without touching the filesystem or a stray root `.env`. The
production default (`Program.cs` calls `CliOptions.Load(args)` with no second argument) is
unchanged.

## Not covered here

`Program.cs`'s top-level orchestration is **not** unit-tested in this project — a top-level
`Main` statement isn't invokable from a test. It was verified manually via CLI runs, documented in
`CHANGELOG.md`. This project covers only the extracted, testable pieces (`CliOptions`,
`DotEnvLoader`, `ConsoleOutput`, `Timed`).
