# Athena logging — operator cheatsheet

Athena's diagnostic surface is the **macOS unified log**
(`subsystem == "athena"`). There are no log files to tail under
launchd; everything goes through `os.Logger` and is queried with
`log stream` / `log show`. Foreground invocations (interactive
`athena load` or `athena start` without install) also write to your
terminal's **stderr** in ISO 8601 UTC format so you can see what
the daemon's doing live.

This doc is the recipe sheet. The Swift-side details are in
[../Sources/athena/Logging/AthenaLogging.swift](../Sources/athena/Logging/AthenaLogging.swift); the design rationale
lives in `docs/logging-audit.md`.

## Where do my logs go?

| You ran | Stdout | Stderr | Unified log |
|---|---|---|---|
| `athena load` (interactive) | `started athena daemon` once | log lines (ISO timestamp) | yes (always) |
| `athena start` no-install | `started athena daemon (pid…)` once | log lines (until terminal closes) | yes |
| `athena install` + `sudo athena start` | `/dev/null` | `athena.err.log` crash-dump only | yes (sole diagnostic) |

`athena.err.log` exists only to catch fatalError / NIO precondition /
MLX-Metal panics that fire after the Logger is torn down. Don't grep
it for normal events — they aren't there.

## The four operator workflows

### 1. "What's Athena doing right now?"

```sh
athena logs --follow                            # convenience wrapper
# or, equivalent raw:
/usr/bin/log stream --predicate 'subsystem == "athena"' --style compact
```

### 2. Last hour of activity, one component only

```sh
athena logs --category daemon --since 1h
# or:
/usr/bin/log show --last 1h --style syslog \
  --predicate 'subsystem == "athena" AND category == "daemon"'
```

Categories Athena emits under:

| Category | What it carries |
|---|---|
| `daemon` | HTTP routes, request lifecycle, governor, RBAC denials, startup |
| `audit` | M30 audit_log write failures + retention notices (M30 audit trail itself is the SQLite `audit_log` table, not this stream) |
| `model.llm` | LLM inference (per-model rebind, generation) |
| `model.embedding` | text-embedding module |
| `model.transcription` | Whisper port |
| `model.diarization` | Sortformer (including the verbose port-debug emissions, at `.debug`) |
| `model.speakerEmbedding` | WeSpeaker |

### 3. Merge several components into one timeline

```sh
athena logs --category daemon --category model.llm --since 30m
# or:
/usr/bin/log show --last 30m --style syslog \
  --predicate 'subsystem == "athena" AND \
               category IN { "daemon", "model.llm" }'
```

The persisted unified-log clock is shared across categories, so the
merged view is in true time order without any client-side sort.

### 4. "What happened in request X?"

Every Logger emission inside a request task hierarchy carries
`req=<uuid>` (and `principal=<resolved>` when auth is on).
Cross-handler correlation:

```sh
athena logs --since 1h | grep 'req=01234abc'
# or grep the unified log directly:
/usr/bin/log show --last 1h --style syslog \
  --predicate 'subsystem == "athena"' | grep 'req=01234abc'
```

The req-id is bound by `AuthMiddleware` once the principal resolves.
Auth-deny paths return before the bind — those events are forensic
records in the SQLite `audit_log` table (see `athena audit`), not
operational logs.

You can also filter by Swift call site:

```sh
athena logs --since 1h | grep 'function=handleChatCompletions'
```

The `function=` field is present on every line.

## Key lines to look for

Every request and model load now leaves a one-line, `key=value` trail
(all carry `req=`/`function=`, and `principal=` when auth is on):

- **Model load** (category `model.<class>`): names the model, its real
  footprint, and load time —
  `model llm loaded Qwen3.5-27b-4bit (15.55GB) in 1.2s`
  (a failed load reports `load failed after <t>: <reason>`). The bytes
  are the reconciled real footprint, which can differ sharply from the
  `loading (estimate …)` line that precedes it.
- **LLM decode** (category `daemon`): the periodic
  `decode heartbeat … tokens_per_sec=… rss=… phys_footprint=…
  mlx_active=… mlx_cache=…` while a generation is in flight.
  `phys_footprint` is the Activity-Monitor "Memory" number (counts the
  GPU KV/prompt-cache buffers `rss` misses).
- **Per-request summaries** (category `model.<class>`), one per request:
  `embeddings done model=… inputs=… vectors=… prompt_tokens=… elapsed_ms=…`,
  `transcription done segments=… audio_secs=… lang=… elapsed_ms=…`,
  `diarization done method=… speakers=… turns=… elapsed_ms=…`,
  `speaker-embeddings done model=… segments=… elapsed_ms=…`.

So `athena logs --since 1h | grep ' done '` is a quick per-request
latency/throughput view across the non-LLM surfaces, and
`grep 'loaded '` shows what loaded, how big, and how long it took.

## Live verbosity (without a restart)

The OS, not the daemon, gates how much detail `log show` captures.
Crank it up at runtime:

```sh
sudo /usr/bin/log config --mode "level:debug" --subsystem athena
# triage…
sudo /usr/bin/log config --reset --subsystem athena
```

When debug is active, `log show --info --debug` surfaces the
memory-only entries too:

```sh
sudo log config --mode "level:debug" --subsystem athena
/usr/bin/log show --last 5m --info --debug --style syslog \
  --predicate 'subsystem == "athena" AND category == "model.diarization"'
```

## The persistence gotcha

`os.Logger` has five tiers. Only three persist by default:

| swift-log level | OSLogType | Persisted by `log show`? |
|---|---|---|
| `.critical` | `.fault` | yes |
| `.error` | `.error` | yes |
| `.warning`, `.notice` | `.default` | **yes** |
| `.info` | `.info` | memory-only (use `--info`) |
| `.debug`, `.trace` | `.debug` | memory-only (use `--debug`) |

Athena's startup line and queue-event emissions are at `.notice` so
they survive a later `log show` without flags. Hot-path
diagnostic-only events are `.debug`; surface them with the
`log config` recipe above or pass `--debug` to `log show`.

The `--log-level` CLI flag / `log_level` config key gates only the
foreground stdout handler — useful when running `athena load`
interactively and you want a quiet terminal. Under launchd it's
inert; the OS-side `log config` mode is the gate.

## Stdout vs. stderr in foreground

`athena start` writes its `started athena daemon (pid …) on …`
message to **stdout**. The daemon's diagnostic logs flow to
**stderr**. So:

```sh
athena start | jq           # operator's pipe — clean (logs go to terminal)
athena start 2>/dev/null    # silence the logs, keep the started-message
athena start > /tmp/out.log 2> /tmp/err.log   # split sinks
athena start 2>&1 | tee log # merge, save, and watch
```

`athena load` (direct invocation) puts everything on stderr; the
"started…" line doesn't apply.

## A note on `/usr/bin/log`

In some shells the bare `log` command is shadowed by a builtin
(notably under nix or homebrew-overrides). The cheatsheet uses
`/usr/bin/log` explicitly so the recipes work everywhere.

## Reading logs through the daemon (remote / RBAC-gated)

`athena logs` is a CLIENT of `GET /api/logs` (one-shot) and
`GET /api/logs/stream` (SSE), both gated on `daemon.admin` and
documented in [openapi.json](../Sources/athena/Server/OpenAPISpec.swift). So:

- **Local box**: `athena logs --since 1h --category daemon` —
  goes through the API. Same shape as `log show` underneath, but the
  flag surface is **controlled by Athena**, not by whatever the
  `/usr/bin/log` binary's current version supports. RBAC enforced.
- **Remote daemon**: `athena --host studio.example --port 7447 logs
  --follow` — talks to that daemon's `/api/logs/stream` SSE. No SSH,
  no host-local `log` command, no operator account on the daemon box.
- **Daemon is DOWN, need to triage**: `athena logs --offline` shells
  out to local `/usr/bin/log` directly — bypasses the daemon (so RBAC
  doesn't apply, and remote `--host` is ignored). For "what crashed"
  triage scenarios only.

Roles that can read the API: `admin`. Non-admin requests get `403`.
Auth-off (loopback + no keys file) leaves the endpoint open, mirroring
the rest of the daemon's admin surface.

## Off-box log shipping

To ship Athena's unified-log entries to a SIEM or central log host,
two paths are supported:

1. **Pull via `/api/logs`** — collectors that can talk HTTP/JSON or
   SSE just hit the daemon. Same RBAC story as `athena logs`.
2. **Tap the macOS unified log locally** — for FluentBit / vector.dev
   / syslog-ng running on the daemon box. Tested recipes are in
   [docs/logging-shipping.md](logging-shipping.md).

Athena itself emits nothing outbound by design (the passive-oracle
contract; see `MEMORY.md`).
