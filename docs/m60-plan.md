# M60 — Unattended-operation robustness + throughput-decay legibility

Status: **in progress.** M60.1 shipped to the working tree (v0.10.100, uncommitted).
Milestone number provisional (latest shipped = M59).

## 1. Origin — the the consuming application throughput-decay investigation (2026-06-02)

the consuming application (an OpenAI-shaped consumer on `:7447/v1`) reported Athena's LLM
throughput "cratering" / "freezing" under sustained load: the first eval run of
a session is clean, a second back-to-back run degrades until normal extractions
cross the client's 540 s `inference_timeout` and 504, and a daemon restart
"fixes" it. A full day of live instrumentation on the M5 Max host
(`deploy/integration/soak-throughput-decay.sh` + a `sudo powermetrics`
GPU-clock logger + a 2 s `/healthz` poller) produced a **confirmed root cause**
and a set of secondary fragilities.

### Root cause — macOS system sleep (NOT a daemon bug)
Athena holds **no power assertion**. Left unattended, the display sleeps
(`displaysleep` 2 min battery / 10 min AC), `powerd` drops its built-in
*"prevent sleep while display is on"* assertion, and ~minutes later the Mac hits
the **system-sleep idle timer** and **suspends the whole process mid-generation**.
Reproduced deterministically: `pmset displaysleepnow` during a live decode →
decode held ~31 tok/s for ~3 min, then **both** the `/healthz` poller and the
independent `powermetrics` logger showed an **identical 50 s gap** (full
suspend), decode → 0, snapping back to ~40 only when the machine was woken.
That 50 s gap is the overnight *"nothing in the logs then suddenly something"*
freeze. **"Restart fixes it" was a red herring — restarting meant touching the
machine, which woke it.** The real the consuming application eval A/B (single-pass then
multipass over the 7-file `mini-nocsv` corpus) **completed cleanly while the
machine stayed awake** — the workload is fine; sleep was the cause.

### Secondary fragilities (real, but not the overnight crater)
1. **Thermal slope** — GPU boost 1619 MHz throttles to <900 under sustained
   load, but decode is **bandwidth-bound**: ~45 % clock loss = only ~10–15 %
   tok/s loss. **Prefill** (compute-bound) takes the full hit. Heat was a minor
   contributor, not the crater.
2. **Budget = deadline by construction** — 16384 max_tokens ÷ ~31 tok/s ≈ 522 s,
   sitting on the 540 s wall before any throttle. Output-length, not throughput,
   is what tips long generations over.
3. **Thinking burns the budget** — the interleaved-thinking template emits
   thousands of reasoning tokens (e.g. ~4100 think tokens for a 124-token JSON
   answer); a cap-hit triggers the consuming application's recovery cascade re-running at a
   higher max_tokens. *(Mostly model-template + client concerns, advisory.)*
4. **M59 pressure-relief never fires during sustained decode** — the pool is
   shed only at model-load admission, so it stays pinned at its count-cap even
   when phys exceeds the budget (a synthetic unique-docs run reached 96 GB phys
   while the relief hook never ran).
5. **No abort on client disconnect** — Athena decodes an in-flight request to
   completion even after the client disconnects, so a timed-out-then-retried doc
   leaves the original generation burning the GPU while the retry piles on.

## 2. Scope decision

M60 is **"make Athena safe to run unattended as an appliance, and legible while
it does."** The power assertion (M60.2) is the fix that closes the reported
issue; the rest is operator legibility (M60.1, .3), an explicitly feasibility-
gated thermal-management spike (M60.4), and the two server-side fragilities that
are squarely Athena's (M60.5 disconnect-abort, M60.6 pressure-relief). The
deadline/thinking items (§1.2–1.3) are **advisory to the consuming application** (client
timeout + max_tokens, model template) and are documented, not built here.

## 3. Slice plan

### M60.1 — `/healthz` operator legibility — **SHIPPED v0.10.100 (uncommitted)**
- `HealthResponse` gains `thermalState` (`ProcessInfo.thermalState` →
  `nominal`/`fair`/`serious`/`critical`, **no sudo** — the throttle signal a
  client polls to back off), `lastDecodeTokensPerSec` (recorded by the
  `collectMetered` heartbeat each decode tick → `AthenaMetrics`), and
  `mlxActiveBytes`/`mlxCacheBytes`.
- Validated via foreground `load` + curl. OpenAPI `/healthz` description updated;
  drift-guard unaffected (response schema is free-form `type: object`).
- **Done.** Folds into the M60 commit.

### M60.2 — Power assertion (THE FIX) — **SHIPPED to working tree v0.10.101 (uncommitted)**
- New `AthenaCore/PowerAssertion.swift`: lock-guarded `@unchecked Sendable`
  wrapper over `IOPMAssertionCreateWithName(kIOPMAssertionTypePreventUserIdleSystemSleep…)`.
- Held for the lifetime of `serve` (`Load.run()`): `acquire()` before
  `server.run()`, `release()` in a `defer` (and on graceful SIGTERM drain).
  Startup logs whether it was acquired.
- `/healthz` gains `powerAssertionHeld: Bool` so the protection is observable.
- **Acceptance:** `pmset -g assertions` lists athena's
  `PreventUserIdleSystemSleep`; `/healthz` reports `powerAssertionHeld: true`;
  re-running the `pmset displaysleepnow` test shows **no suspend gap** and decode
  continues across display sleep.
- Docs: a "running unattended" note (this assertion + the `pmset`/`caffeinate`
  context) in the operator docs.

### M60.3 — Sudoless GPU telemetry on `/healthz` (IOReport) — **SHIPPED v0.10.102 (uncommitted)**
Built as a standalone MIT package `AppleSiliconMetrics` (`~/Source/AppleSiliconMetrics`,
validated vs `powermetrics`), consumed by Athena via a local path-dep. `/healthz`
gains `gpuClockMHz` + `gpuActiveResidency` from a background-sampled,
lock-guarded `GPUTelemetryProbe` (nil-safe). Temp deferred to the package's
later milestones. *(Original IOReport-in-daemon plan below superseded by the
package approach.)*


- The honest gap M60.1 left: actual **GPU clock (MHz)** and **die temp (°C)**
  have no public API and the daemon runs as a non-root service user (so
  `powermetrics` is out). The sudoless path is the **private `IOReport`
  framework** (how `mactop`/`macmon` read GPU active-residency → frequency) plus
  SMC/IOReport temp.
- New `AthenaCore/GPUTelemetry.swift`: minimal IOReport subscription over the
  GPU performance-state channel; compute the active frequency from residency ×
  the per-state frequencies (read once from IORegistry `voltage-states`). Temp
  via SMC sensor keys **if** a clean read is reachable without root (probe
  first; otherwise ship clock-only and document temp as `powermetrics`-only).
- `/healthz` gains `gpuClockMHz` (and `gpuDieTempC` if feasible).
- **Risk:** private framework, version-fragile, unsafe C interop. Validate live
  on this M5 against the `powermetrics` logger (we can, since we're on the host).
- **Acceptance:** `/healthz` `gpuClockMHz` tracks the `powermetrics` logger
  within tolerance during a decode; degrades gracefully (field omitted / `null`)
  if IOReport is unavailable.

### M60.4 — Proactive thermal/fan management — **FEASIBILITY-GATED, likely DEFERRED**
- Original idea ([[athena-fan-management-idea]]): pre-spin the fan at inference
  start instead of riding the reactive curve. **Value dropped sharply** now that
  the root cause is sleep, not heat (the thermal slope costs only ~10–15 %
  decode). And Apple-Silicon fan **writes** are non-public, **root-only** SMC
  key writes (`F*Tg`/forced-mode), undocumented and OS-version-fragile.
- This slice is a **spike, not a commitment**: probe whether fan RPM can be
  *read* (SMC/IOReport) and whether a bounded *write* even succeeds on M5 via a
  privileged helper. If reads work and writes are stable → a future root-gated
  capability (privilege boundary like install/start, see
  [[reference_athena-sudo-commands]]). If not → **document and drop**, recommend
  macOS **High Power mode** (raises the sustained power cap — the supported
  proxy) for batch runs.
- **Default expectation: deferred** with the spike findings recorded.

### M60.5 — Abort in-flight decode on cancellation — **SHIPPED v0.10.104 (uncommitted)**
The synchronous decode loops (Speculative, GuidedGreedy, SpeculativeSampling,
GuidedSubstrate) now poll a cancel flag bridged through the shared
`HeartbeatCounter` TaskLocal: a `withTaskCancellationHandler` in `collectMetered`
flips it on task cancellation, and the loops `break`, freeing the GPU.
**Validated (deadline path):** with `request_timeout_secs=15`, a request that
would run ~120 s returns 504 at 15 s and the GPU drops to idle (338 MHz) within
~4 s — confirmed via `asmetrics`, vs. the old behavior of decoding to maxTokens.
**Known limitation:** a pure client *disconnect* with NO server timeout does NOT
abort — Hummingbird doesn't cancel a mid-compute handler task — so set
`request_timeout_secs` (the deadline path is what cancels). The original
(superseded) plan below.


- Today a disconnected client's generation runs to completion on the GPU
  (observed: a killed soak client left the daemon decoding ~4.5k tokens). Under
  a retry cascade that compounds load. Wire request-cancellation
  (Hummingbird/NIO channel-closed → cancel the `collectMetered` task →
  cooperative check in the decode loops, which already poll `Task.isCancelled`
  for the heartbeat) so an abandoned generation stops.
- **Acceptance:** disconnecting mid-generation drops `inflight` to 0 and frees
  the GPU within ~1 decode step; bit-identical contract for *completed* requests
  unchanged.

### M60.6 — M59 pressure-relief during sustained operation — **SHIPPED v0.10.105 (uncommitted)**
New `MemoryGovernor.relievePromptCachePressureIfNeeded()` sheds the prompt-prefix
KV pool when **phys_footprint** (not the RSS probe, which under-counts the GPU
buffers — M55) exceeds the 90% high-water mark; called after every metered
generation in `collectMetered`. Prompt-cache only — never evicts a loaded module
mid-serving. **Validated:** same temp=0 generation + prompt-cache engagement, at
normal budget the pool retains 0.57 GB / 1 entry (relief idle), at a 20 GB budget
(phys ~19.5 GB > 18 GB high-water) the pool is shed to 0 / 0 entries, no OOM.


- Make the prompt-cache pool eviction **pressure-aware continuously**, not only
  at load-admission: when process phys / governor budget is exceeded during
  decode, shed idle (refcount-0) prefix entries even if under the static
  count/byte caps. Drive from the governor's periodic reconcile or a
  request-completion hook.
- **Acceptance:** under a sustained loop that pushes phys over budget, the pool
  sheds idle entries and phys returns under budget without a model unload;
  in-use entries (refcount > 0) survive.

## 4. Config / touchpoints
- M60.2 power assertion: on by default for `serve` (it's an appliance); a
  `[power] prevent_idle_sleep` TOML key (default `true`) to opt out for a
  desktop dev box that *should* sleep. 5-touchpoint wire like `kvCompression`.
- M60.3/M60.4 telemetry: read-only, no config.

## 5. Validation
- M60.2 is the gate: the `pmset displaysleepnow` reproduction must show **no
  suspend gap** with the assertion held (it deterministically showed a 50 s gap
  without it). Plus the live the consuming application eval A/B left unattended (display
  asleep) must complete without the freeze.
- M60.3 validated live against `powermetrics` on the M5 host.
- e2e: extend the gate to assert `/healthz` exposes the new fields and
  `powerAssertionHeld: true` on a normally-started daemon.

## 6. Out of scope (advisory to the consuming application, not Athena slices)
- Lower `max_completion_tokens` / raise the client `timeout` so a full-length
  generation has margin under the 540 s wall (§1.2).
- Cap or disable the model's interleaved **thinking** (§1.3) — the single
  highest-leverage latency lever, but a model-template/client choice.
- Stop the recovery cascade from re-burning 540 s on a doc that hit the cap.
