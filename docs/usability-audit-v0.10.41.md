# Athena usability audit — v0.10.41

Friction surfaces walked as a new operator: install banner, `athena --help`,
every subcommand's help, `athena.toml`, error envelopes, log routing,
`/healthz` + `/api/*`, `/ui`, README, `/openapi.json`. Code-quality and the
M43 punch-list are deliberately out of scope; this is the operator-facing
axis.

For each finding: persona → what they see → what's missing → smallest fix
shape.

## Blockers

### 1. TOML edits silently don't reach the running install
*Persona:* any operator changing config after install.

The launchd plist hard-codes every config value as `ProgramArguments` at
install time
([Sources/AthenaDeploy/LaunchdPlist.swift:30-118](../Sources/AthenaDeploy/LaunchdPlist.swift#L30-L118)).
After install, `athena config set budget_bytes 36000000000` writes to
`/usr/local/etc/athena/athena.toml` — but the plist still ships the old
`--budget-bytes`. `athena start` just `launchctl bootstrap`s that same
plist
([Sources/athena/Commands/DaemonLifecycle.swift:96-114](../Sources/athena/Commands/DaemonLifecycle.swift#L96-L114)).
The only TOML key the running daemon re-reads on its own is
`kv_compression`
([Sources/athena/Commands/Load.swift:335-338](../Sources/athena/Commands/Load.swift#L335-L338)).
Nothing in `athena config set`'s output, `athena --help`, or the TOML
preamble tells the operator that edits require `sudo athena install` to
take effect.

**Fix shape.** `athena config set` either refuses with "run `sudo athena
install` to apply" or re-renders the plist and `launchctl bootstrap`s it;
either way the rule has to surface.

### 2. First-time install + default config produces an LLM-less daemon
*Persona:* first-timer.

The synthesized `athena.toml` has `model =` commented out
([deploy/athena.toml:43](../deploy/athena.toml#L43)) and the README's "Get
started" path is `athena load`. The first `/v1/chat/completions` returns
`400 model_not_available` with `available: []`
([Sources/AthenaCore/AthenaError.swift:100-103](../Sources/AthenaCore/AthenaError.swift#L100-L103)).
The quickstart mentions `athena pull` + `athena default` in step 5, but as
a parenthetical after the curl example — anyone who copy-pastes the curl
hits the empty-allowlist error first.

**Fix shape.** Install banner prints "no LLM in store — `athena pull
<id>` then `athena default <id>` before chatting"; `model_not_available`
with an empty `available` list adds a remediation hint pointing at `pull`.

### 3. `athena allowlist` is undiscoverable and unrecoverable offline
*Persona:* existing operator swapping a model.

`athena --help` shows ~35 subcommands flat
([Sources/athena/Athena.swift:25-34](../Sources/athena/Athena.swift#L25-L34));
`allowlist` sits between `resident` and `start` with abstract "Manage the
persistent per-module model allowlist (M42)" — the milestone code is
meaningless to outsiders. README + quickstart never mention `allowlist`.
Worse, `AllowlistCmd` lives in the client subset
([clients/Sources/AthenaClient/RemoteAllowlist.swift:136-146](../clients/Sources/AthenaClient/RemoteAllowlist.swift#L136-L146))
— it always speaks HTTP, with no offline-store path. An operator whose
daemon is down or who has no token cannot edit the allowlist at all.

**Fix shape.** Add a local `--data-dir` path symmetric to `athena auth
user passwd`; strip the "M42" tag; document `allowlist` in quickstart's
"Next steps."

### 4. Allowlist add succeeds, then first request hangs with no signal
*Persona:* existing operator.

`athena allowlist add --module textEmbedding --id new/model` prints
`added textEmbedding:new/model` and exits 0 — no hint the model isn't on
disk. The next `/v1/embeddings` triggers a synchronous HF download on the
request thread (M43 punch-list #2). `/healthz` keeps reporting 200,
`request_timeout_secs` is off by default
([deploy/athena.toml:213-220](../deploy/athena.toml#L213-L220)), and no
log line surfaces "downloading." The operator sees: client transport
timeout, no Retry-After, `athena status` healthy.

**Fix shape (complementing M43.2).** `allowlist add` warns when the id
isn't in the local store and points at `athena pull <id>`; ship a
non-zero `request_timeout_secs` default so unbounded hang becomes a 504.

### 5. Auth-deny envelopes carry no remediation
*Persona:* operator hitting a 401/403.

Missing/invalid bearer returns `{"error":{"message":"missing or invalid
bearer token","type":"auth_error","code":"unauthorized"}}`
([Sources/athena/Server/Auth.swift:390-392](../Sources/athena/Server/Auth.swift#L390-L392));
insufficient permissions returns `"insufficient permissions"`
([Sources/athena/Server/Auth.swift:396-397](../Sources/athena/Server/Auth.swift#L396-L397)).
Neither names `athena auth login`, `ATHENA_KEY`, or `/ui/login`. CLI
clients dump raw JSON on 401
([clients/Sources/AthenaClient/RemoteAllowlist.swift:34-37](../clients/Sources/AthenaClient/RemoteAllowlist.swift#L34-L37))
— no localized "set `ATHENA_KEY` or run `athena auth login`" hint at the
client edge either.

**Fix shape.** Add a `hint` field to auth-error envelopes; the CLI's
`fail()` helper renders it.

## Annoying

### 6. Log routing is split four ways; `athena logs` default mis-fires
*Persona:* operator debugging.

Three on-disk paths plus the unified log:

- `<log_dir>/athena.err.log`, `<log_dir>/athena.out.log` (launchd-managed)
- `<data-dir>/athena.log` (when started via `athena start` user-context)
- `log stream --subsystem athena` (notice+ only persists)
- optional remote syslog

`athena logs` defaults to `source: .err`
([Sources/athena/Commands/Logs.swift:21](../Sources/athena/Commands/Logs.swift#L21))
and dies with "is the daemon installed?" if the operator started via
`athena start`. The error is wrong: the daemon *is* running, just under a
different lifecycle.

**Fix shape.** Default `--source` to "whichever log file exists right
now"; the dead-end error names the other candidates.

### 7. `/healthz` has no in-flight, queue depth, or last-request signal
*Persona:* operator debugging a hang.

`GovernorSnapshot` carries module state + reservedBytes + freeBytes
([Sources/AthenaCore/MemoryGovernor.swift:286-303](../Sources/AthenaCore/MemoryGovernor.swift#L286-L303))
— nothing about in-flight count, last activity, queue depth, or current
job. Combined with M43 #1 (state=loaded lies after eviction), `/healthz`
is doubly useless for live diagnosis. `/metrics` Prometheus does carry
the counters, but it's admin-gated and no doc says "if a request hangs,
scrape /metrics."

**Fix shape.** Add `inflight`, `queue_depth`, `last_request_at` to the
`/healthz` snapshot.

### 8. `/ui` has no allowlist page — post-M42 surface is CLI-only on web
*Persona:* operator using the WebUI.

`/ui` covers dashboard, config, models (store ops), daemon, users, roles,
tokens
([Sources/athena/Server/AthenaServer.swift:169-337](../Sources/athena/Server/AthenaServer.swift#L169-L337))
— no `/ui/allowlist`. An operator who learned Athena via the WebUI
cannot discover or edit the per-module allowlist there, even though the
M42 API is wired server-side.

**Fix shape.** Add `/ui/allowlist` reusing the existing handlers.

### 9. `athena --help` is a 35-verb flat list with milestone codes leaking
*Persona:* first-timer.

Subcommand declaration order
([Sources/athena/Athena.swift:25-34](../Sources/athena/Athena.swift#L25-L34))
interleaves local model-store ops (`pull`/`convert`/`verify`/`prune`/
`cp`), HTTP client verbs (`run`/`queue`/`vectors`/`store`/`usage`/
`audit`), daemon lifecycle (`load`/`start`/`stop`/`status`), and admin
(`auth`/`hf`/`proxy`/`allowlist`). Abstracts reference M-numbers:
"Manage the persistent per-module model allowlist (M42)", "(M41.1)
rebind a module's slot" in `--module` help.

**Fix shape.** ArgumentParser supports grouped subcommands — group as
**Daemon / Models / RBAC / Client / Diagnostics**; strip M-codes.

### 10. CLI-only options not surfaced anywhere
*Persona:* operator who reads `athena.toml` looking for every knob.

The flat TOML is the documented surface
([README.md:79-84](../README.md#L79-L84)). But:

- `--prompt-cache-cap-bytes` exists only on `athena load`
  ([Sources/athena/Commands/Load.swift:49-54](../Sources/athena/Commands/Load.swift#L49-L54))
  — not in `AthenaConfig`, `knownKeys`, the TOML, or the plist.
- `--llm-model` / `--embedding-model` / `--whisper-model` /
  `--diarization-model` / `--speaker-embedding-model`
  ([Sources/athena/Commands/Load.swift:70-132](../Sources/athena/Commands/Load.swift#L70-L132))
  are CLI-only AND first-boot-seed-only (DB wins thereafter — M43 #6).
  The operator who edits `athena.toml`'s `model =` line a month later
  sees no effect and no warning.

**Fix shape.** Either add these to TOML/`config set`, or document the
CLI-only set in `athena.toml`'s header and have `athena doctor` report
"the running daemon was launched with N CLI-only flags not visible in
this TOML."

### 11. `sudo athena start` with no install runs anyway, on stderr only
*Persona:* operator hitting an error.

[Sources/athena/Commands/DaemonLifecycle.swift:117-123](../Sources/athena/Commands/DaemonLifecycle.swift#L117-L123)
writes "warning: running as root but no installed launchd plist …
falling back to a root-owned Process() daemon" to stderr and falls
through. The daemon comes up — broken in the same metallib-path way that
drove v0.10.38–41 — but `athena status` says healthy. M43 #3 covers the
code side; on the usability side, the warning is hidden by `2>/dev/null`
in any script wrapper, and there's no stdout signal.

**Fix shape.** Print the warning to stdout too (preferred: refuse, per
M43 #3).

## Nits

### 12. Reinstall banner offers password-reset path but not token-reset
*Persona:* reinstall flow.

Fresh-DB install banner prints the seeded admin password and token in a
box; non-fresh
([Sources/athena/Commands/Install.swift:349-357](../Sources/athena/Commands/Install.swift#L349-L357))
prints "auth: existing accounts kept (no admin seeded)" with a one-liner
about offline `auth user passwd`. No equivalent token-recovery hint —
operator who's lost both bearer and password gets only the password path.

**Fix shape.** Append "mint a fresh admin token offline: `athena auth
token add --user admin --data-dir <…>`" to the reinstall banner.

### 13. `athena init` only pulls the *default* aux set
*Persona:* operator using M41/M42 multi-model surface.

[Sources/athena/Commands/Init.swift:36-51](../Sources/athena/Commands/Init.swift#L36-L51)
hard-codes `Load.defaultEmbeddingModel` etc.; it doesn't read the DB
allowlist. After adding a second embedding model via `athena allowlist
add`, operator must remember to `athena pull <id>` manually — `athena
init` won't fetch it.

**Fix shape.** `athena init --from-allowlist` queries the running
daemon's `/api/models/resident` (or local DB) and pulls every allowed
id.

## CLI-only options (no TOML/DB equivalent)

The audit's add-on question: which `athena load` / `athena start` flags
have no `athena config set` analog?

| Flag | TOML key | DB? | Notes |
|---|---|---|---|
| `--budget-bytes` | `budget_bytes` | — | In TOML, but plist freezes value at install (finding #1). |
| `--prompt-cache-cap-bytes` | — | — | CLI-only, undocumented in TOML. |
| `--llm-model` (repeatable) | `model` (single) | yes (M42) | TOML seeds DB on first boot only; later TOML edits ignored (finding #10). |
| `--embedding-model` (rep) | — | yes (M42) | Same: CLI flag seeds DB once. |
| `--whisper-model` (rep) | — | yes (M42) | Same. |
| `--diarization-model` (rep) | — | yes (M42) | Same. |
| `--speaker-embedding-model` (rep) | — | yes (M42) | Same. |
| `--module / --id / --key` | — | — | M41.1 rebind verb, not daemon config. |

Every other `athena load` flag has a TOML key
([Sources/AthenaDeploy/AthenaConfig.swift:7-111](../Sources/AthenaDeploy/AthenaConfig.swift#L7-L111))
that the plist generator passes through
([Sources/AthenaDeploy/LaunchdPlist.swift:30-118](../Sources/AthenaDeploy/LaunchdPlist.swift#L30-L118))
— subject to finding #1.
