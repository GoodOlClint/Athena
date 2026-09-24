# Athena

Single native macOS/MLX daemon providing LLM chat, text embeddings, and audio/video transcription/diarization/speaker-embeddings — all governed by one Metal memory budget. Passive oracle: answers inbound requests only.

## Architecture

- Swift package (`Package.swift`, `Sources/`) targeting macOS on Apple Silicon. Build requires full Xcode 26.5+ / Swift 6.3 (Command-Line Tools alone cannot compile the MLX Metal shaders; mlx-swift 0.31.5+ ships a Swift 6.3 manifest, so an older Xcode fails at dependency resolution).
- HTTP daemon on `127.0.0.1:7447` by default. Bearer-token RBAC; auth disabled in loopback dev mode.
- Modules: `AthenaLLM`, `AthenaEmbedding`, `AthenaTranscription` (each in `Sources/`).
- Cross-platform client CLI lives under `clients/` (Swift package, builds on Linux/Windows).
- Rust shim in `rust-shim/` for components not yet Swift-native.

## Canonical pipelines

- **All HTTP routes are defined in `Sources/athena/Server/OpenAPISpec.swift`.** Do not add or modify a route without updating that spec in the same edit. The daemon serves it verbatim at `GET /openapi.json`.
- **All errors return the envelope `{"error":{"message","type","code"}}`.** Never invent ad-hoc error shapes.
- **Outbound network is forbidden except model-weight fetches from Hugging Face** (and the opt-in remote-syslog sink). The "passive oracle" rule is binding — no result webhooks, no billing callbacks, no telemetry pings.
- **Native DTOs for the `/api/*` dialect live in `Sources/athena/Server/NativeAPIDTO.swift`.** Do not duplicate `/v1/*` OpenAI shapes there.

## Public surface

Athena is a passive oracle. Consumers interact only via HTTP.

**Self-describing**: `GET /openapi.json` returns the full OpenAPI 3.0.3 spec for both dialects. Always reachable, no auth required. For static reading from another repo, load `Sources/athena/Server/OpenAPISpec.swift`.

**Two HTTP dialects**:

| Surface | Shape | Use for |
|---|---|---|
| `/v1/*` | Inference + data plane | OpenAI-compatible drop-ins **and** Athena-native extensions under the `/v1` namespace (see the compatibility rule below). |
| `/api/*` | Athena native (control) | Daemon **control plane** — model-store, RBAC, allowlist, lifecycle, audit, usage, logs, cache. **NOT inference** — `/api/chat`+`/api/embed` are deprecated (ADR 013); new inference features go to `/v1` only. |

**`/v1` compatibility rule (binding).** `/v1/*` is **not uniformly
OpenAI-compatible** — it is the inference + data surface, and only a subset is a
literal OpenAI drop-in. **Whenever a new `/v1/*` surface is added that is not
strictly OpenAI-compatible (no equivalent OpenAI endpoint, or an
OpenAI-shaped response over a non-OpenAI route), it MUST be marked as
Athena-native** in the same edit: (a) note it in the introducing ADR, (b) say so
in its `OpenAPISpec.swift` operation `description`, and (c) tag the endpoint in
the list below. A consumer must be able to tell drop-in OpenAI from native
extension without reading the code. Reusing an OpenAI *response shape* (e.g.
verbose_json) for consumer convenience does **not** make a native route
OpenAI-compatible — the route's existence/semantics are what count.

**Stable `/v1/*` endpoints** (canonical list in `OpenAPISpec.swift`; **[oai]** =
OpenAI-compatible drop-in, **[native]** = Athena extension under `/v1`):

- `POST /v1/chat/completions` **[oai]** — streaming SSE, tool calls, JSON-schema structured output, `stop`/`seed`/`top_p`
- `POST /v1/chat/completions/count_tokens` **[native]** — exact pre-flight `{prompt_tokens}` for a chat body (same template/tokenizer as the request path, ADR 042); no OpenAI equivalent; no generation, no inference gate, not metered
- `POST /v1/messages/count_tokens` **[anthropic]** — the Messages-dialect analogue (`{input_tokens}`), same decoder + same counting core, so the two dialects report the same number for the same conversation (ADR 042 §4(a), deferral lifted 2026-07-25 for dialect parity)
- `POST /v1/embeddings` **[oai]**
- `POST /v1/audio/transcriptions` **[oai]** — word timestamps, SRT/VTT
- `POST /v1/video/transcriptions` **[native]** — demux a video's audio track → transcription (ADR 022); no OpenAI equivalent
- `POST /v1/audio/diarizations` **[native]** — no OpenAI equivalent
- `POST /v1/audio/embeddings` (speaker embeddings) **[native]** — no OpenAI equivalent
- `GET /v1/models`, `GET /v1/models/{id}` **[oai]** — plus two **Athena-native extension fields** (ADR 042): `context_length` (what the checkpoint advertises) + `max_prompt_tokens` (the ADR 030 prefill ceiling this daemon enforces), both omitted-when-nil
- _(ADR 025: `/v1/queue*` removed v0.10.203 — model lifecycle ops stream SSE on `POST /api/models/{pull,convert,prune}`; `/v1/vectors*` + `/v1/store*` removed v0.10.201.)_

**Native `/api/*`**: model-store and RBAC admin. Surface defined in `OpenAPISpec.swift` and `NativeAPIDTO.swift`.

**Auth**: `Authorization: Bearer <token>`. Each route requires a single RBAC permission. Loopback dev mode (no seeded users) opens every route.

**Diagnostics**: macOS unified log, subsystem `athena`. Off-box log shipping is operator-side — see `docs/logging.md`, `docs/logging-shipping.md`.

## Dependencies (consumed by this repo)

- Hugging Face — model weight fetches only. No other outbound dependencies.
- **MLX substrate (`GoodOlClint/mlx-swift-lm`) — pin an immutable `integration-YYYY-MM-DD` TAG, never a bare commit hash and never the `integration` branch.** That branch is force-pushed on every rebuild, which orphans whatever commit a consumer pinned; because SPM fetches refs (an orphaned commit is reachable only by explicit SHA), a bare-hash pin breaks every COLD build while CI stays green on cache warmth alone. This is the fork's documented consumer contract (`~/Source/mlx/CLAUDE.md` "Discipline") and it is written from Athena's own 2026-07-31 outage — pin `751aaed` went unreachable mid-session, recovered by tagging the orphan as `integration-2026-07-07` (#86). A substrate bump means moving to a newer dated tag, never re-pointing at a branch — and it is a measurement exercise, not a version edit: follow [`docs/substrate-bump-runbook.md`](docs/substrate-bump-runbook.md) (blast-radius inventory, mlx-swift floor coupling, stale-checkout/cache poison, the gate battery incl. the #64 parity gate, per-pin ADR 028 re-stamping).

## ADRs

@docs/decisions/LEDGER.md

Open the linked ADR file for a decision's substance — the ledger line is only a pointer.
Read `docs/decisions/` before any architectural change — particularly anything that would touch the passive-oracle rule, the OpenAPI spec, or the Metal memory governor.

## Build / run / test

```sh
./deploy/build.sh Release           # xcodebuild → .build/xcode/.../Release/athena
./deploy/test.sh                    # unit tests
swift format --in-place --configuration .swift-format --recursive Sources Tests clients Package.swift
                                    # style gate (CI lints with swift-format pinned at 6.3.3)
./deploy/e2e-rbac.sh                # RBAC end-to-end
athena load                         # run daemon in foreground on loopback (no auth)
curl http://127.0.0.1:7447/healthz  # liveness
curl http://127.0.0.1:7447/openapi.json  # self-describing surface
```

For production install (boot-time launchd, TLS, bearer auth, WebUI), see `docs/quickstart.md`.

## Agent pushes go through the GitHub MCP (operator decision 2026-08-31)

Agent-authored branches are pushed via the `github` MCP tools (`create_branch` + `push_files`, committing as the `goodolclint-claude`/`goodolclint-codex` App), **not** local `git push` over the operator's SSH key. Local git stays for everything else — worktrees, branches, local commits, diffs; only the push itself goes through the API. Workflow files under `.github/workflows/` push through the same API path — the App token does carry `workflows` permission (commit `87c9f8f` on branch `issues-172` pushed a workflow file via `push_files` and verified; operator ruled 2026-09-01, reaffirmed 2026-09-23). The real gate on a workflow-touching change is the merge, not the push: it needs operator PR approval, same as any other change reaching main. `push_files` creates its own commit from full file contents, so the commit message is passed to the tool and no `Co-Authored-By` trailer is needed (the App is the author). Because the content is re-uploaded inline, every API push is verified before the PR: commit the identical change locally, `git fetch`, and `git diff <local-commit> origin/<branch> --` must be empty — a mangled byte fails loudly here instead of merging. Deletions and renames: `push_files` cannot express a removal, so a diff that deletes a file uses `delete_file` (one path per call; it makes its own App-authored commit on the same branch), and a rename is `push_files` of the new path plus `delete_file` of the old. The same byte-verification gate applies — commit the identical deletion locally and the diff against `origin/<branch>` must be empty; a deletion the push failed to carry shows up loudly as the file still present (sequence verified 2026-08-31: `push_files` + `delete_file` on one branch round-trips to exact tree parity with the base). A mode-bearing rename (an executable, a symlink) is the known gap: `push_files` writes the destination as a regular 100644 file, the byte-verify catches the mode diff, and that case falls back to operator-attended local `git push`.

## Versioning (ADR 040 S8 — in force since v0.11.0)

**Versions are release events, not commit events.** `Athena.appVersion` (`Sources/athena/Athena.swift`, SSOT — also stamped into `/openapi.json`) bumps only when a release is cut: **patch** = fix-only release, **minor** = feature release, **major/1.0** = operator-reserved. Slices land as plain commits with NO version bump and NO per-slice tag (the pre-publication bump-every-slice ritual is retired). Annotated `v*` tags mark releases and drive the release pipeline. Do not bump `appVersion` in a feature commit — bump it in the release commit the operator approves (`/ship`).

## Coordinator and workers

This project runs one long-lived coordinator session, **Athena Scheduling Coordinator**, that launches background workers from kickoff files and independently verifies each worker's results.
The coordinator asks the operator before every worker launch — there is no standing launch permission.
Workers push through the GitHub MCP the same as any other agent, workflow files included (see "Agent pushes go through the GitHub MCP" above).
