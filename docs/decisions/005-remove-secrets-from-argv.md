# 005 — Remove secrets from argv (`--password`)

**Status:** Implemented
**Date:** 2026-06-12
**Milestone:** M66 (audit-remediation; resolves standing DECISION #2; audit B2/K7/K13)

**Implemented:** M66.5 (v0.10.127). `--password` removed from `auth user add`, `auth user passwd`, `proxy login` (macOS) and the portable `auth user add` (client). Secrets now come from `--password-stdin`, `$ATHENA_PASSWORD`, or an interactive prompt via a shared `PasswordInput.resolve` (macOS) / `RemoteAuth.resolvePassword(stdin:)` (client). The e2e suite was migrated to `ATHENA_PASSWORD`. **K7** (`Credentials` Keychain `security -w <value>`) and **K13** (`--key` in argv) are the remaining argv-secret items — audited here, tracked in M71 (they need a Keychain-API / `--key`-steering change, out of scope for the `--password` removal).

## Context

`auth user add`, `auth token …`, and `proxy …` accept a `--password` option (their help already says "omit to prompt"). A secret on the command line leaks through shell history, `ps`/`/proc/<pid>/cmdline`, and process-listing audit trails. The options considered were a non-breaking deprecation (add stdin/env, warn on `--password`) vs a hard removal.

## Decision

**Hard-remove `--password` from argv.** A secret may be supplied only by:

- **interactive prompt** (no echo) — the default when none is given; or
- **`--password-stdin`** — read the secret from stdin (one line), enabling `… --password-stdin < secret.txt` and pipeline use without it landing in argv; and/or
- an **environment variable** (e.g. `ATHENA_PASSWORD`) where a non-interactive path needs it.

This applies to every command that currently takes `--password` (auth user/token, proxy credential). The CLI surface (`OpenAPISpec.swift` is HTTP-only and unaffected; this is a `Commands/` change) drops the option entirely.

## Consequences

- **Breaking CLI change:** any script invoking `--password=…` must switch to `--password-stdin` or the env var. Called out in the slice commit and release notes; acceptable given the project is pre-1.0 and the security gain is direct.
- Removes the most common credential-in-history footgun.
- The same slice should audit any other argv secret (proxy token, `ATHENA_STORE_KEY` handling) for the same treatment (K7/K13).
