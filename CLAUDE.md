# CLAUDE.md

This project's ground rules (stack, `unsafe` policy, ADR process, working style) live in
[AGENTS.md](AGENTS.md), written to be portable across AI coding assistants — read that first.

## Claude Code specifics

- Subagents live in [.claude/agents/](.claude/agents/):
  - `adr-author` — write or review ADRs in `docs/architecture/` (MADR format). Use this instead
    of writing ADRs freehand. Every ADR it writes or touches must end with a filled-in
    "Decision Log" table.
  - `csharp-dotnet` — C#/.NET implementation and review, enforcing the stack and `unsafe`
    policy from AGENTS.md. Preloads the `csharp-conventions` skill (below) for per-project
    detail. Never commits/pushes on its own initiative.
  - `test-writer-runner` — write, update, and run tests for any `InferenceEngine.*` project
    (one test project per `src/` project, business-case-documented). Use after implementing or
    changing behavior; always runs `dotnet test` and confirms it passes.
  - `researcher` — web research on inference internals (algorithms, specs, papers). Use
    before making implementation decisions.
  - `huggingface-explorer` — HuggingFace ecosystem (model formats, APIs, downloads).
  - `code-reader` — read and analyze external repos (dotLLM, LLamaSharp) to understand
    architecture and patterns. Not for writing code in our project.
- All agents use context7 MCP for current library documentation.
- Skills live in [.claude/skills/](.claude/skills/):
  - `csharp-conventions` — per-project ownership/gotchas for `Core`/`Models`/`Tokenizers`/
    `Engine`/`Cli`, the KV-cache contract, and the context7 requirement. Preloaded into
    `csharp-dotnet`; also auto-discoverable when working with `.cs`/`.csproj` files generally.
  - `caveman` — concise, specific writing: short sentences, no hedging, no filler. Preloaded into
    every subagent, so ADRs, comments, commit messages, and reports back stay tight.
- Rules live in [.claude/rules/](.claude/rules/):
  - `unsafe-policy` — every project is safe-only except `InferenceEngine.Models`, which sets
    `AllowUnsafeBlocks=true` for `LibraryImport` P/Invoke (Accelerate/MLX). Never copy the flag
    to another project.
- This project is Apple Silicon-centric (see AGENTS.md). Do not add Windows/Intel code paths.
- No hooks yet; configure them in `.claude/settings.json` as real, recurring needs show up —
  don't scaffold empty ones speculatively.
