# Athena logging audit — v0.10.48

Walked every diagnostic emission path in `Sources/` (and `deploy/`) and
matched it against the M10 centralized-logging spine
([Sources/athena/Logging/AthenaLogging.swift](../Sources/athena/Logging/AthenaLogging.swift)).
The spine itself is sound: one `LoggingSystem.bootstrap` multiplexes
`StreamLogHandler` (stdout) + `OSUnifiedLogHandler` (os.Logger with
subsystem `athena`) + an opt-in RFC5424 UDP syslog sink, all gated on a
single `log_level`. This audit catches the leaks AROUND that spine —
specifically the print-and-stderr writes whose lines land in the
launchd-captured `athena.{out,err}.log` files without a timestamp, plus
shallow level-discipline and metadata gaps.

Code changes are explicitly out of scope; this is a written audit + work
items.

## Summary

- **Three direct `print()` / stderr writes** in daemon-side code land in
  `athena.{out,err}.log` UNTIMESTAMPED. Two are bootstrap-edge (before
  Logger is alive) and acceptable; one
  ([Load.swift:515-519](../Sources/athena/Commands/Load.swift#L515-L519))
  is redundant with the Logger.notice four lines later and should just be
  deleted. **This is the most likely source of the missing-timestamp
  observation.**
- **`AthenaTranscription/Sortformer/Sortformer.swift` has 7 raw
  `print()`s** gated on a `verbose: Bool = false` flag that the daemon
  never sets — latent, but they're the only diagnostic stdout writes left
  in a model module and they bypass Logger entirely. P2 cleanup, not a
  live leak.
- **Timestamp format is INCONSISTENT across sinks.** `StreamLogHandler`
  emits `2026-05-27T14:30:45-0700` (second precision, local offset);
  `SyslogLogHandler` emits `2026-05-27T21:30:45.123Z` (millisecond, UTC);
  `os.Logger` adds its own µs-precision timestamp at OS level. A grep
  across all three sinks for a single event produces three different
  timestamps. Nothing in the spine timestamps lines; both timestamps come
  from the handlers themselves.
- **Level discipline is acceptable**: zero silent `catch {}` blocks, zero
  per-request Logger constructions, audit and ops logs cleanly separated
  (M30 audit_log is its own SQLite trail, not co-mingled with diagnostic
  emissions). The one real instrumentation gap is missing
  request-id/principal metadata on the operational logs (`Logger`
  `MetadataProvider` declared but never set), so triaging a queue or
  inference failure means cross-referencing audit + log by wall clock.
- **Doctor has check #7 (log_dir exists) and #14 (audit retention)** but
  no check that the daemon's log files are actually receiving fresh
  writes, no recent-error-spike probe, and no syslog reachability probe
  when `syslog_remote` is configured.

## Log sink inventory

Six sinks, each fed differently. The columns are: WHERE the line ends up,
WHAT writes it, and WHO supplies the timestamp on each emitted line.

| Sink | Path / endpoint | Writer | Timestamp source | Format |
|---|---|---|---|---|
| Launchd stdout | `{log_dir}/athena.out.log` | launchd captures the daemon process's stdout via `StandardOutPath` ([LaunchdPlist.swift:128](../Sources/AthenaDeploy/LaunchdPlist.swift#L128)) | Whichever writer emitted the line | Mixed — see below |
| Launchd stderr | `{log_dir}/athena.err.log` | `StandardErrorPath` ([LaunchdPlist.swift:129](../Sources/AthenaDeploy/LaunchdPlist.swift#L129)) | Same | Same |
| User-context start | `{data-dir}/athena.log` | `athena start` opens the file as both `Process.standardOutput` and `.standardError` ([DaemonLifecycle.swift:137-176](../Sources/athena/Commands/DaemonLifecycle.swift#L137-L176)) | Same | Same |
| Unified log (os.Logger) | `log show --predicate 'subsystem == "athena"'` | `OSUnifiedLogHandler` ([AthenaLogging.swift:218-277](../Sources/athena/Logging/AthenaLogging.swift#L218-L277)) | OS (µs precision, applied by the kernel at emit) | Apple-defined; respect `--style {syslog\|json\|compact}` |
| Remote syslog (opt-in) | `udp://host[:port]` (config `syslog_remote`) | `SyslogLogHandler` ([AthenaLogging.swift:146-214](../Sources/athena/Logging/AthenaLogging.swift#L146-L214)) | Handler-supplied: ISO8601 UTC with fractional seconds | RFC5424 `<PRI>1 ts host athena pid msgid - text` |
| Audit trail (SQLite) | `{data-dir}/athena.sqlite` table `audit_log` | `AthenaStore.addAudit()` via `AthenaServer.audit()` ([AthenaServer.swift:3325-3343](../Sources/athena/Server/AthenaServer.swift#L3325-L3343)) | `ts REAL` column, written at INSERT (`Date().timeIntervalSince1970`) | Per-row: principal, action, target, result, detail |

The three file sinks are all "whatever-the-writer-emitted." That writer is
one of:

- **`StreamLogHandler`** — `2026-05-27T14:30:45-0700 notice athena.daemon:
  [source] message` (second precision, local TZ, no fractional seconds).
  Format defined at
  [swift-log/Sources/Logging/Logging.swift:2077](../.build/checkouts/swift-log/Sources/Logging/Logging.swift#L2077).
- **Raw `print()` / `FileHandle.standardError.write`** — no timestamp at
  all. The launchd/start file just gets the bytes the daemon wrote.
- **Swift runtime / NIO / MLX C-side crashes** — `fatalError`, Metal
  failures, NIO precondition fails. These dump to stderr untimestamped at
  process death, into `athena.err.log`.

## Logger emission sites (the part that's correct)

Three files construct `Logger`s, all using the `AthenaLogLabel` constants
([Sources/AthenaCore/ModuleID.swift:29](../Sources/AthenaCore/ModuleID.swift#L29)):

| File | Logger | Label / category | Emissions |
|---|---|---|---|
| [RequestQueue.swift:38](../Sources/athena/Server/RequestQueue.swift#L38) | `log` (private instance) | `athena.daemon` → category `daemon` | 3× `.info`, 2× `.notice`, 1× `.warning` |
| [AthenaServer.swift:4114](../Sources/athena/Server/AthenaServer.swift#L4114) | `log` (private static) | `athena.daemon` | governed-request error classifier (`classified()`, `.warning`) |
| [AthenaServer.swift:4115](../Sources/athena/Server/AthenaServer.swift#L4115) | `auditLog` (private static) | `athena.audit` → category `audit` | M30 audit table write failures + retention notices |
| [Load.swift:369, 392, 395, 445, 523, 547, 553](../Sources/athena/Commands/Load.swift#L369) | per-call construction | mix of `daemon` and `model(id)` | 5× `.notice`, 1× `.warning` — startup/CLI only |
| [AthenaServer.swift:648, 698, 1507](../Sources/athena/Server/AthenaServer.swift#L648) | per-call construction | `daemon` | preload + TLS + auth startup logs (one-shot) |

Hummingbird `Application` is constructed with `serverName: "athena"`
([AthenaServer.swift:673](../Sources/athena/Server/AthenaServer.swift#L673)),
which makes its internal logger label `"athena"`. Per
`AthenaLog.category(forLabel:)` rules
([AthenaLogging.swift:26-32](../Sources/athena/Logging/AthenaLogging.swift#L26-L32)),
that label (no `athena.` prefix) collapses to category `daemon`. So
Hummingbird/NIO logs route through the bootstrapped multiplex correctly.

**Per-request constructions:** none. The three `Logger(label:…)` calls at
AthenaServer.swift:648/698/1507 are all bootstrap (preload Task, TLS
serverBuilder, auth-load callback) — fire-once at startup, not hot-path.

**MetadataProvider:** declared in both `OSUnifiedLogHandler` and
`SyslogLogHandler`, never set anywhere in the bootstrap or downstream. No
log line carries request-id, principal, model-id, or queue-id as
structured metadata — the data is interpolated into the message string
when present (e.g. RequestQueue's `kind=… id=…`), unstructured.

## Findings

### F1 — `Load.swift:515-519` raw print duplicates the very next Logger line (P1)

[Load.swift:515-519](../Sources/athena/Commands/Load.swift#L515-L519)
emits a bare `print("athena: engine=… model=… budget=… listen=…")` to
stdout. Four lines later
([Load.swift:523-528](../Sources/athena/Commands/Load.swift#L523-L528))
`Logging.Logger(label: AthenaLog.daemonLabel).notice("athena daemon up
— engine=… listen=… budget=…")` emits substantively the same content via
the Logger.

Because the `print` runs in the daemon process (the spawned `athena
load`), launchd captures it into `athena.out.log` UNTIMESTAMPED, while the
Logger line that follows is timestamped via `StreamLogHandler`. Operators
tailing the file see one untimestamped line followed by a near-identical
timestamped one — the most plausible source of the "log file missing
timestamps" observation.

**Fix:** delete lines 515-519. The Logger.notice at 523-528 already
emits the same info to stdout (multiplexed), the unified log (`notice`
persists to `log show`), and remote syslog when configured.

### F2 — Two pre-bootstrap stderr writes are untimestamped (P3 — acceptable)

[AthenaLogging.swift:53-58](../Sources/athena/Logging/AthenaLogging.swift#L53-L58)
and
[Load.swift:293-297](../Sources/athena/Commands/Load.swift#L293-L297) both
write directly to stderr to warn about bad CLI args (`--syslog-remote
udp://…` only, `--log-level` invalid). Both fire BEFORE
`LoggingSystem.bootstrap` returns, so they can't route through Logger.
They land untimestamped in `athena.err.log`.

These are correct as written — the alternative is a buffered "queue and
flush later" dance that's not worth it for two warnings. Optional
hardening: prefix each line with a manually-formatted ISO8601 timestamp so
the entries don't visually float in the file.

### F3 — `Sortformer.swift` has 7 untimestamped `print()` calls gated on `verbose: false` (P2)

[Sortformer.swift:633-637, 665-666, 928, 992](../Sources/AthenaTranscription/Sortformer/Sortformer.swift#L633-L637)
contains `if verbose { print(...) }` for audio length, feature shape,
segments-found, processing time, and streaming chunk progress. The
`verbose:` parameter defaults to `false` and the diarization module
([Sources/AthenaModels/MLXDiarizationModule.swift](../Sources/AthenaModels/MLXDiarizationModule.swift))
never sets it, so in production these never fire — but they're the only
diagnostic stdout calls left in any model module and the wiring is one
parameter flip from spamming `athena.out.log` with untimestamped lines.

**Fix:** convert to `Logger(label: AthenaLogLabel.model("sortformer"))`
at `.debug` — the existing log-level gate then naturally hides them in
production. Drop the `verbose` parameter; it's vestigial port-debug
scaffolding from the Sortformer vendoring (M4.3).

### F4 — Three different timestamp formats across three sinks (P2)

The same log call produces:

- stdout / file: `2026-05-27T14:30:45-0700` (second precision, local TZ)
- unified log: `2026-05-27 14:30:45.123456-0700` (Apple format, µs)
- syslog UDP: `2026-05-27T21:30:45.123Z` (ms precision, UTC)

Correlation across sinks works (events are clearly the "same one") but
the seconds-precision local-TZ stdout format is the weakest link: it
can't disambiguate two events in the same second, and mixing offsets with
UTC syslog is a known triage tax.

**Fix shape:** replace `StreamLogHandler` with a tiny in-spine handler
that emits `<ISO8601 UTC with .SSS> <level> <category>: <message>` —
matches the SyslogLogHandler format, eliminates the local-TZ drift, gains
ms precision. ~30 LOC; no API change. The unified-log format is set by
Apple and not worth fighting.

### F5 — No request-id / principal metadata on operational logs (P2)

`Logger.MetadataProvider` is plumbed through both handlers
([AthenaLogging.swift:151, 222](../Sources/athena/Logging/AthenaLogging.swift#L151))
but never set. Every error log either has no caller context
([AthenaServer.swift:4121-4125](../Sources/athena/Server/AthenaServer.swift#L4121-L4125),
the governed-error classifier) or interpolates an id manually into the
message (RequestQueue's `id=…`). The audit trail carries principal +
action richly, but it's a SEPARATE table — you can't grep the audit_log
for the millisecond an inference timeout fired in `athena.err.log` and
get a clean answer.

The audit-write failure path is the sharpest instance:
[AthenaServer.swift:3335-3336](../Sources/athena/Server/AthenaServer.swift#L3335-L3336)
logs `action=\(action)` but drops the principal — if a constraint
violation prevents an audit row from being written for user X, we lose
which user it was.

**Fix shape:** add a task-local `RequestContext` set by an early router
middleware (request id + principal), expose it as a
`Logger.MetadataProvider`, and ALL log lines on that request automatically
gain `req=` / `principal=` keys. Out-of-scope for a no-code audit; track
as a workitem.

### F6 — `OSUnifiedLogHandler` emits everything with `privacy: .public` (P3 — by design, document it)

[AthenaLogging.swift:259-261](../Sources/athena/Logging/AthenaLogging.swift#L259-L261)
unconditionally renders the message as `\(text, privacy: .public)`. The
comment notes "Server logs are not sensitive — keep them readable in
Console rather than redacted as `<private>`." That's correct for ops
diagnostics, but worth pinning down before any future log line ever
interpolates user prompts or tokens — once a secret reaches that
`os.Logger.log`, it WILL appear in plaintext in Console / sysdiagnose.

**Fix:** no code change. Add one line to the doc comment forbidding
interpolation of user prompt content / bearer-token strings / Keychain
secrets into the message field. Already true in practice — the audit
trail keeps `target=u:foo` / `t:hash[:8]` and never the raw token.

### F7 — Doctor checks no logging health (P3)

`athena doctor` has 16 numbered checks. #7 confirms `log_dir` exists
([Doctor.swift:129-136](../Sources/athena/Commands/Doctor.swift#L129-L136))
and #14 inspects the audit table size / retention
([Doctor.swift:433-458](../Sources/athena/Commands/Doctor.swift#L433-L458)).
Nothing checks:

- whether `athena.out.log` / `athena.err.log` have been written to in
  the last N seconds (a hung daemon shows as "still listening" but with
  a stale log)
- syslog UDP reachability when `syslog_remote` is configured (a typo'd
  hostname silently swallows logs forever — `SyslogSender.send` is
  fire-and-forget by design)
- recent error/warning spike (cheap: `log show --last 5m --predicate
  'subsystem == "athena" AND messageType >= error'` count)
- log file size / unbounded growth (no rotation is shipped; an operator
  with `auditRetentionDays` set might assume the same applies to the
  text logs)

**Fix shape:** three additional doctor checks (#17–19), each ≤30 LOC.
Track as a workitem.

### F8 — No log rotation is provided (P2)

`athena.out.log` / `athena.err.log` / `athena.log` are append-only with no
rotation, no size cap, no compression. The audit table has
`audit_retention_days`; the file logs have nothing equivalent. A long-
running daemon's `.out.log` grows without bound. README and quickstart
don't document this (or point at `newsyslog.conf`).

**Fix shape:** ship a `newsyslog.d` snippet alongside the install pkg
that rotates `{log_dir}/athena.*.log` weekly with 4 generations and
gzip; document the file in `at-rest.md`'s neighbor. No daemon code
needed.

## macOS unified-logging primer

For an operator coming from Windows / linux journald, here's the minimum
you need to know about how `os.Logger` works on macOS, and how Athena
uses it.

**Subsystem and category.** Every `os.Logger` is constructed with
`subsystem` (a string, conventionally reverse-DNS like `com.acme.foo`
but Apple accepts anything) and `category` (a subcomponent label).
Athena's subsystem is literally `"athena"`
([AthenaLogging.swift:17](../Sources/athena/Logging/AthenaLogging.swift#L17))
— short, matches the binary name, no reverse-DNS. Categories are derived
from the swift-log label: `athena.daemon` → `daemon`, `athena.audit` →
`audit`, `athena.model.qwen3-2b-mtp` → `model.qwen3-2b-mtp`, anything
else → `daemon`.

**Levels.** os.Logger has five tiers, distinct from swift-log's seven:
`debug`, `info`, `default` (often spelled "notice"), `error`, `fault`.
Athena maps `Logger.Level` → `OSLogType` at
[AthenaLogging.swift:268-276](../Sources/athena/Logging/AthenaLogging.swift#L268-L276):
trace/debug → `.debug`, info → `.info`, notice/warning → `.default`,
error → `.error`, critical → `.fault`.

**Persistence is the gotcha.** Only `.default` / `.error` / `.fault`
persist to disk in the macOS unified log. `.debug` and `.info` are
**memory-only** by default — they show up live in `log stream` but
disappear from a later `log show` unless you pass `--info` / `--debug`.
This is why Athena's "daemon up" line is `.notice` (mapped to
`.default`), so it survives `log show` after the fact.

**Querying.** A few recipes:

```
# live tail of every Athena log
log stream --predicate 'subsystem == "athena"' --level=info

# everything in the last hour, persisted only (notice+)
log show --last 1h --predicate 'subsystem == "athena"'

# everything in the last hour, INCLUDING in-memory info/debug
log show --last 1h --info --debug --predicate 'subsystem == "athena"'

# just the audit category
log stream --predicate 'subsystem == "athena" AND category == "audit"'

# just one model's logs
log show --last 30m --predicate \
  'subsystem == "athena" AND category == "model.qwen3-2b-mtp"'

# errors and faults only, JSON for grep-friendliness
log show --last 1h --style json --predicate \
  'subsystem == "athena" AND messageType >= error'
```

**Format.** `log show` defaults to a wide human-readable format; pass
`--style syslog` for RFC3164-ish single-line, `--style compact` for
narrow terminals, `--style json` for tooling. Apple-controlled — Athena
can't change it. The shell built-in `log` may be shadowed in a
non-default PATH; use `/usr/bin/log` when in doubt (see the
`project_unified-log-verification` memory).

**Where Apple's docs live.** `man log`, `man os_log` (the C-side), and
Apple's "Generating Log Messages from Your Code" developer doc.

## CMTrace recommendation — decline (with a `log show` recipe instead)

CMTrace-format lines look like:

```
<![LOG[athena daemon up — engine=mlx]LOG]!><time="14:30:45.123+00"
 date="05-27-2026" component="daemon" context="" type="1" thread="1"
 file="Load.swift:523">
```

CMTrace's value proposition is "I have a tool (CMTrace.exe, OneTrace,
LogViewer) that grouhps + colours + filters this format". Athena's
operators are on macOS, so:

- **CMTrace.exe is Windows-only.** Running it under WINE on macOS for an
  Athena log is a long detour from `Console.app` / `log show`, both of
  which already do every CMTrace UX win (per-line severity, component
  filtering, live tail) natively against the unified log.
- **No CMTrace consumer exists on the user's stack.** The only data
  point for "I want CMTrace" is operator familiarity — real, but solved
  cheaper by documenting the `log show` recipes above (a one-page
  cheatsheet beats a custom sink with its own rotation, escaping, and
  drift risk).
- **Adding a side-channel CMTrace file** would mean a fourth log sink
  with its own format-drift and rotation problem (already an issue —
  see F8), AND it'd duplicate every entry currently going to the unified
  log.

**Recommendation: decline.** Ship the `log show` cheatsheet (a
docs-only diff, ~50 lines, lives next to `quickstart.md`). If a future
operator has a CMTrace-consuming workflow that they can't move off,
revisit with a concrete consumer named.

## Proposed work items

Each is a milestone-slice candidate; pick a number when scheduling.

- **MX.1 — delete the redundant `print` in Load.swift.** F1. ~5 LOC
  diff, single file. Quick win, almost certainly the missing-timestamp
  observation.

- **MX.2 — convert Sortformer.verbose to Logger.debug.** F3. ~10 LOC,
  drops the dead `verbose:` parameter, removes the last `print()`
  calls in any non-CLI module.

- **MX.3 — uniform-timestamp StreamLogHandler replacement.** F4.
  ~30 LOC custom `LogHandler` emitting ISO8601 UTC with ms precision
  matching the syslog format. Replaces the swift-log default in the
  bootstrap multiplex. No external API change.

- **MX.4 — request-scoped Logger.MetadataProvider.** F5. Add a task-
  local `RequestContext { reqId, principal }` set by an early router
  middleware, expose it as the metadata provider on bootstrap. All
  daemon log lines under that request auto-gain `req=` / `principal=`
  — AND `function=` (plus `file=` / `line=` when cheap) so the MX.8
  filter-by-function recipe lands on structured metadata rather than
  substring-matching `eventMessage`. swift-log already supplies
  `function`/`file`/`line` to the handler `log(...)` call site; the
  metadata provider just needs to surface them as merged keys (or
  the handlers can fold them in alongside the provider output before
  emit). ~80 LOC, touches the bootstrap + an auth/early-middleware
  site + one helper. Lift principal into the audit-write-failure log
  line at AthenaServer.swift:3336 in the same slice.

- **MX.5 — privacy-policy comment in OSUnifiedLogHandler.** F6.
  One-line doc-comment update forbidding interpolation of secrets /
  user prompt content into the message field. Pure docs.

- **MX.6 — log-health doctor checks.** F7. Three new numbered checks:
  (a) stdout/stderr log was written within the last N seconds; (b)
  syslog UDP reachability when configured; (c) error spike in last
  5 min via `log show --predicate 'subsystem == "athena" AND
  messageType >= error'`. ~80 LOC against `Doctor.swift`.

- **MX.7 — log rotation snippet.** F8. Ship a `newsyslog.d/athena.conf`
  drop-in with the install pkg (weekly, 4 generations, gzip), reference
  it from the quickstart. No daemon code; install path only.

- **MX.8 — `docs/logging.md` operator cheatsheet.** CMTrace decline
  follow-up. This is the deliverable that makes
  unified-log-as-primary defensible; if it's a stub, the
  per-component-file split (see below, deferred) becomes mandatory.
  Spec the three concrete operator workflows as one-line copy-paste
  recipes, then a "Where do my logs go?" decision tree (installed
  vs. start-launched, persisted vs. memory-only). Recipes:

  ```
  # (a) tail one component, live
  log stream --predicate 'subsystem == "athena" AND \
    category == "daemon"' --style compact

  # (b) merge N components, last hour, persisted only
  log show --last 1h --predicate 'subsystem == "athena" AND \
    category IN {"daemon", "audit"}' --style syslog

  # (c) filter merged view by function (MX.4 metadata)
  log show --last 1h --predicate 'subsystem == "athena" AND \
    eventMessage CONTAINS "function=handleChatCompletions"' \
    --style syslog
  ```

  Caveats to call out by name in the doc:
  - **`.info` / `.debug` are memory-only by default** — `log show`
    will silently omit them unless you pass `--info` / `--debug`.
    Recipes (b) and (c) above land only on `notice`+ without those
    flags. This is the single most common gotcha for operators new
    to the unified log.
  - **The shell `log` builtin may be shadowed** in some PATHs (zsh
    plugins, Nix profiles, custom wrappers). Use `/usr/bin/log`
    explicitly when in doubt — same lesson recorded in the
    `project_unified-log-verification` memory.
  - **MX.4 metadata format** — once the provider is wired, the
    function field appears in the message as ` function=<Name>`
    (StreamLogHandler sort-and-append, see
    [AthenaLogging.swift:198-205](../Sources/athena/Logging/AthenaLogging.swift#L198-L205)
    for the equivalent SyslogLogHandler rendering). Recipe (c) above
    is written for that form; update the doc when the metadata-key
    surface lands so the recipe and the code stay in sync.

  Pure docs. No daemon changes.

**Explicitly deferred — per-component log-file split.** Splitting
`athena.out.log` into `athena-<category>.log` files (one file per
swift-log category) is a viable future alternative if MX.8 proves
insufficient in practice — operators who can't internalize predicate
syntax get `tail -F athena-audit.log` instead. The cost is real (a
custom file-rotating handler, multiplied N-files-worth of fd / disk
overhead, plus the file-level rotation that MX.7 already has to
solve once). Don't start it unless the unified-log cheatsheet
demonstrably fails to land — pick this up as MX.8b if there's
evidence.
