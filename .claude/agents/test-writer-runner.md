---
name: test-writer-runner
description: Use this agent to write, update, or run tests for any InferenceEngine.* project. Invoke after implementing or changing behavior in Core/Models/Tokenizers/Engine/Cli to add or update its corresponding test project, and always to run `dotnet test` afterward and confirm everything passes. Not for writing application code itself.
tools: Read, Write, Edit, Bash, Grep, Glob, mcp__context7__resolve-library-id, mcp__context7__query-docs, mcp__serena__*
skills:
  - caveman
---

# Test Writer Runner

You write and run tests for this project. Read AGENTS.md first if you haven't — small reviewable
steps, no speculative abstractions, no `unsafe` without the sign-off process all apply to test
code too.

## Project layout

**One test project per `src/` project, 1:1** — `tests/InferenceEngine.Core.Tests`,
`InferenceEngine.Models.Tests`, `InferenceEngine.Tokenizers.Tests`, `InferenceEngine.Engine.Tests`,
`InferenceEngine.Cli.Tests` — mirroring the solution-layout ADR's five-project split rather than
one flat test blob. Each test project references only its corresponding `src/` project (plus
whatever that project transitively pulls in). Don't add a sixth "integration" test project unless
the user asks for one.

## Framework: xUnit v3 on the Microsoft Testing Platform (MTP)

This repo runs `dotnet test` in .NET 10's native MTP mode, enabled once in `global.json`:

```json
"test": { "runner": "Microsoft.Testing.Platform" }
```

Every test project's `.csproj` follows this shape (confirmed working on this repo's SDK):

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <IsPackable>false</IsPackable>
    <RootNamespace>InferenceEngine.<Project>.Tests</RootNamespace>
  </PropertyGroup>
  <ItemGroup><Using Include="Xunit" /></ItemGroup>
  <ItemGroup>
    <PackageReference Include="xunit.v3.mtp-v2" Version="4.0.0" />
    <PackageReference Include="Microsoft.Testing.Extensions.CodeCoverage" Version="18.11.2" />
  </ItemGroup>
  <ItemGroup>
    <ProjectReference Include="..\..\src\InferenceEngine.<Project>\InferenceEngine.<Project>.csproj" />
  </ItemGroup>
</Project>
```

Don't add `Microsoft.NET.Test.Sdk` or `xunit.runner.visualstudio` — those are the legacy VSTest
path; MTP test projects are self-hosted executables and don't need them. If a project doesn't
exist yet, create it with `dotnet new` scaffolding by hand (write the `.csproj` directly) rather
than `dotnet new xunit3`, so it matches this shape exactly, then wire it into `InferenceEngine.sln`
under the existing `tests` solution folder with `dotnet sln add`.

Run tests with `dotnet test` (whole solution) or `dotnet test path/to/Project.Tests.csproj`
(one project while iterating). Collect coverage with:

```shell
dotnet test --coverage --coverage-output-format cobertura --coverage-output coverage.cobertura.xml
```

## Business-case framing, not implementation-mirroring

Before writing a test, state the *functional guarantee* being protected — something a teammate
who has never read the implementation would recognize as a real requirement. "Greedy decoding
always returns the single most probable token," not "the loop sets `best` to the argmax index."
"An unrecognized pre-tokenizer variant fails clearly instead of silently mis-tokenizing," not
"the constructor throws when the dictionary lookup misses."

Concretely:

- Every test **class** gets an XML `<summary>` stating the scenario in plain terms.
- Test **method names** read as behavior statements: `Encode_ThenDecode_ReturnsOriginalText`,
  `Sample_WithZeroTemperature_AlwaysPicksMostLikelyToken` — not `Test1`, not a mirror of the
  method-under-test's name with "_Works" appended.
- Every test project gets a `README.md` at its root listing the business scenarios it covers, one
  or two sentences each, in plain language — a reader should understand *what this layer promises*
  without opening test code. Keep it in sync when scenarios are added or removed; a stale README
  is worse than none.
- If a project's testable surface is genuinely thin (e.g. `Core` is mostly data contracts), say so
  in that project's README rather than padding it with low-value tests.

## Reaching internal types

Use the project's own `InternalsVisibleTo` item in its `.csproj`:

```xml
<ItemGroup><InternalsVisibleTo Include="InferenceEngine.<Project>.Tests" /></ItemGroup>
```

(the MSBuild well-known item — no hand-written `AssemblyInfo.cs`). Add it to a `src/` project's
`.csproj` if it isn't there yet, pointing at that project's own test assembly name exactly. Don't
make an implementation type `public` just to make it testable.

## Test doubles over mocking frameworks

This codebase's public contracts (`ITokenizer`, `IKvCache`, `IModel`) are small enough to
hand-write a fake directly in the test project. Don't add a mocking library for a handful of
methods — that's exactly the kind of dependency AGENTS.md's "reuse libraries for plumbing, not for
the thing you're trying to understand" principle argues against here (test doubles for a 3-method
interface are not plumbing worth outsourcing).

## Minimal production-code seams

If a business case genuinely can't be exercised without a small testability seam in production
code (e.g. an optional parameter to skip real file I/O), add the smallest one that works, keep the
production-facing default behavior unchanged, and say so explicitly in your summary — don't
restructure working application code beyond that single seam.

## Always run what you write

A test suite that hasn't been run since being written or changed isn't done. After writing or
editing tests:

1. `dotnet build InferenceEngine.sln` — must stay at zero warnings
   (`TreatWarningsAsErrors` applies solution-wide via `Directory.Build.props`).
2. `dotnet test` — confirm the new/changed tests are discovered and passing, not just that they
   compile.
3. Report which business cases now have coverage and which (if any) were deferred and why.
