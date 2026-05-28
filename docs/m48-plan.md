# M48 — Heartbeat truth + structured speculative widening

Planning doc captured 2026-05-28 from the the consuming application mini-corpus
hang (transcript file, 14 KB source, ~19k prompt tokens, request
wedged >15 min, daemon process alive on PID 69798).

## Status

In progress. Three of four slices shipped, paused for operator
confirmation before the fourth (behavior-changing) slice.

| Slice | Tag | Date |
|---|---|---|
| M48.1 — heartbeat TaskLocal binding fix | v0.10.69 | 2026-05-28 |
| M48.2 — dispatch-decision debug log | v0.10.70 | 2026-05-28 |
| M48.4 — prefill chunk granularity in heartbeat | v0.10.71 | 2026-05-28 |
| M48.3 — widen speculative to engage under temp>0+schema | — | PAUSED |

- ✅ **M48.1** — v0.10.69 — `collectMetered` refactored to take a
  builder thunk so the AsyncStream's internal Task spawns inside
  `DecodeProgress.$counter.withValue(...)`. Pre-fix, every
  structured-path `incrementToken()` was a no-op because the
  TaskLocal binding was established AFTER the Task spawn. New
  `DecodeProgressTaskLocalTests` pins both sides of the contract.
- ✅ **M48.2** — v0.10.70 — One `.debug` log line emitted at the
  start of every `MLXLLMModule.runSpeculative` declaring the chosen
  internal path (`speculative-greedy | speculative-sampling |
  guided-greedy-or-substrate | substrate-stream`). Operator surfaces
  via `sudo log config --mode "level:debug" --subsystem athena` +
  `athena logs -f`. AthenaLLM picks up swift-log directly (was
  already transitive via Hummingbird/NIO).
- ✅ **M48.4** — v0.10.71 — `DecodeProgressCounter` extended with
  `recordPrefillChunk(completed:total:)` (default no-op).
  `HeartbeatCounter` records it; the heartbeat log line includes
  `prefill=N/M` when present. Three native prefill loops
  (`GuidedGreedy`, `SpeculativeGeneration`,
  `SpeculativeSamplingGenerate`) publish per chunk. The substrate
  path doesn't expose per-chunk callbacks so its heartbeat omits
  the field (correctly says "don't know").

**PAUSED before M48.3**: behavior-changing slice. Awaiting operator
confirmation from a v0.10.71 rerun of the the consuming application mini-corpus
so we can see (via the new dispatch log) which path the transcript
request actually takes, and (via the now-honest heartbeat) whether
it was decoding slowly all along or genuinely wedged. The right
M48.3 shape depends on what we see — possibly the simpler answer is
the consuming application sets `speculative: true` per request rather than
Athena widening the eligibility gate.

## Trigger

the consuming application mini-corpus run on v0.10.68: 5 of 9 ingest files
completed cleanly (apple-notes 53-64s, court-doc 70-89s, witness 185s),
then the **13.7 KB transcript hung past 10 min** with the daemon
process still alive, modules still loaded, `inflight: 1`, no log
errors. Heartbeat showing `tokens=0 tokens_per_sec=0.0` indefinitely.

## Finding #1 — Heartbeat lies for the structured paths

The M46.8 `DecodeProgress.counter` TaskLocal is bound at
[AthenaServer.swift:1144](../Sources/athena/Server/AthenaServer.swift#L1144)
INSIDE `collectMetered`, wrapping the `for await event in events`
loop. But `events` is an `AsyncStream` returned by
[MLXLLMModule.generateMetered](../Sources/AthenaLLM/MLXLLMModule.swift#L379)
which spawns its own `Task { ... }` inside the stream initializer —
*before* the TaskLocal binding exists.

```swift
// MLXLLMModule.swift:379
AsyncStream { continuation in
    let task = Task {          // ← spawned BEFORE withValue runs
        ...                    //   so DecodeProgress.counter is nil
        let stream = try await self.beginGeneration(...)
        for await event in stream { continuation.yield(...) }
    }
}
```

```swift
// AthenaServer.swift:1144 — in the caller
await DecodeProgress.$counter.withValue(counter) {
    for await event in events {   // ← TaskLocal in scope HERE only,
        ...                       //   not in the already-spawned Task
    }
}
```

Structured tasks inherit TaskLocals from the spawning context. The
decode task's spawning context (the AsyncStream initializer's closure)
runs *before* the withValue, so its TaskLocal table has no entry for
`counter`. Every `DecodeProgress.counter?.incrementToken()` call
inside `GuidedGreedy.generate` / `SpeculativeGeneration.generate` /
`GuidedSubstrate.generate` / `SpeculativeSamplingGenerate.generate`
is a no-op.

The M46.8 comment at AthenaServer.swift:1136-1142 asserting
"TaskLocal propagates across `await` and into `container.perform { ... }`
actor calls" is correct in isolation — TaskLocal DOES propagate to
child tasks created **inside** the withValue scope, and across `await`
within that scope. The comment is wrong about THIS layout because the
relevant task was spawned outside the scope entirely.

**Consequence**: the heartbeat reports `tokens=0 tokens_per_sec=0.0`
for every structured request even when the decode is making real
progress. The 900+ seconds of `tokens=0` in the transcript hang
**does not prove the loop is stuck** — the sample profile proves it
*is* iterating (worker thread spending 83% of CPU time at
[GuidedGreedy.swift:80](../Sources/AthenaLLM/GuidedGreedy.swift#L80),
the decode-loop `asyncEval(backbone)`, with the MLX scheduler at the
top of the stack and Metal kernel dispatch underneath).

The substrate-streamed (non-Guide) path is unaffected — it increments
the counter directly in the for-await event drain at
AthenaServer.swift:1149 (inside the withValue scope).

## Finding #2 — the consuming application's transcript request didn't engage M47

The captured worker thread call stack (Thread_39367498) shows:
```
closure #4 in MLXLLMModule.runSpeculative(...)  MLXLLMModule.swift:593
  → specialized static GuidedGreedy.generate(...)   GuidedGreedy.swift:80
```

`MLXLLMModule.swift:593` is the `else if guide != nil` fallback in
the structured-output dispatch — i.e. the path taken when
`greedyEligible` is false. From line 475:

```swift
let greedyEligible = effectiveSpec && effectiveTemp == 0
```

So the request lands in `GuidedGreedy.generate` (M47-UNAWARE) when
either `effectiveSpec == false` (no `speculative: true` opt-in) OR
`effectiveTemp > 0` (any non-zero temperature with a schema). M47.2's
Guide-masked-draft fix is in `SpeculativeGeneration.generate` —
this path doesn't engage it.

So the structured short-note wins we measured (~9.1 tok/s vs ~1.6
tok/s) must have come from a request shape that DID enter
`SpeculativeGeneration`. The transcript request shape did not. Need
to instrument the dispatch decision so the consuming application (and we) can see
which path each request takes without resorting to a process sample.

## Finding #3 — GuidedGreedy at long context is bottlenecked on MLX dispatch

From the same sample (5 s window, ~3636 samples on the worker
thread):

| Worker frame | Samples | % |
|---|---|---|
| asyncEval at GuidedGreedy.swift:80 | 3015 | 83% |
| └─ scheduler::wait_for_one | 2669 | 73% |
| └─ eval_impl → gpu::eval → kernel dispatch | 282 | 8% |
| decoder.pick at GuidedGreedy.swift:67 | 513 | 14% |
| logitsAndHidden at GuidedGreedy.swift:78 | 103 | 3% |

The CPU thread is overwhelmingly stuck in
`mlx::core::scheduler::Scheduler::wait_for_one` — the GPU command
queue is back-pressuring the CPU dispatcher. Inside the dispatch
phase, time is split across QuantizedMatmul kernel setup,
CommandEncoder construction, kernel cache lookups, and Metal binding
calls. The MLX scheduler thread itself is parked on
`pthread_cond_wait` (working as designed — wakes when the GPU stream
has room).

This is "GPU saturated, CPU dispatching as fast as MLX allows" — i.e.
the runtime is doing what it can, the workload is just expensive at
this prompt+context length on a 27B 4-bit MTP model. The most direct
path to faster wall time on this workload is **getting the request
INTO the speculative path** (Finding #2), which amortizes ~2 tokens
per backbone forward — roughly 2× before any acceptance-rate gain.
Deeper MLX-side improvements (batched kernel submission, persistent
command encoders) are out of scope for M48.

## Sequencing

### M48.1 — heartbeat truth

Bind `DecodeProgress.counter` BEFORE spawning the decode task, not
after. Two equivalent shapes:

**(A)** Pass the counter explicitly into `generateMetered` and have
that function own the `withValue(counter)` scope:

```swift
public nonisolated func generateMetered(
    ..., counter: any DecodeProgressCounter
) -> AsyncStream<GenChunk> {
    AsyncStream { continuation in
        let task = Task {
            await DecodeProgress.$counter.withValue(counter) {
                ...
            }
        }
    }
}
```

**(B)** Read the TaskLocal value at the call site, capture it in the
Task closure (`let captured = DecodeProgress.counter`), re-bind
inside the spawned task. Same effect, no signature change.

Recommend **(A)** — explicit, harder to break, signature change is
internal (`generateMetered` callers are all in this repo).

Test (gated, real model): bind a `RecordingCounter`, run a structured
request, assert `counter.tokens > 0` at the end. Today this test would
fail; after the fix it passes. Pure observability change — no
behavior shift on the model output.

### M48.2 — dispatch-decision log

One-line debug log at the start of `runSpeculative` that records
which branch was taken:

```swift
Self.log.debug("""
    dispatch path=\(path) temp=\(effectiveTemp) \
    spec=\(effectiveSpec) schema=\(schemaJSON != nil)
    """, metadata: ["function": "runSpeculative"])
```

Where `path` is one of `speculative-greedy | speculative-sampling |
guided-greedy | guided-substrate | substrate-stream`. Lets the
operator (and the consuming application) see at a glance which path each request
took. Closes the loop on "did M47.2 engage?" without needing a
process sample.

### M48.3 — widen speculative to engage under temp > 0 + schema

Today's `greedyEligible` gate forces structured requests with
`temperature > 0` into `GuidedGreedy.generate`. But under a tight
schema the Guide masks the sampling distribution down to a small set
of tokens — and at the M40 sampling-speculative path's distributional
identity, sampling speculative + Guide collapses to greedy-of-the-
valid-set anyway. The temp argument has effectively no influence on
output token choice once the mask is applied, so refusing to speed up
the path is leaving real wall-time on the floor.

Proposal: relax `greedyEligible` so a structured request engages the
speculative greedy loop regardless of temp value (since the Guide
collapses any sampling to the masked argmax). This is a behavior
change worth landing carefully — the bit-identical-greedy contract
test from M47.1 protects the output sequence; what changes is the
*request shape* that qualifies for the fast path.

Risk: a request that today returns sampled JSON might shift to
greedy-of-allowed-set JSON. Behavior change is observable. Worth a
dedicated note in the release tag.

Gate: same `StructuredSpeculativeParityTests` (M47.1) extended with a
`temperature: 0.7` case asserting the result is still a valid JSON
matching the schema. The bit-identical contract holds against itself
within the masked region.

### M48.4 — prefill heartbeat granularity (small, drive-by)

Add a prefill-chunks counter to the heartbeat output:
`prefill_chunks=14/38 elapsed=27s tokens=0`. Distinguishes
"stuck in prefill" from "stuck just-after-prefill" the next time we
have to diagnose a hang. Tiny change. Free once M48.1 lands and the
heartbeat is actually trustworthy.

## Definition of done

- [ ] Structured-path heartbeat reports real per-token progress
      (gated test asserts `counter.tokens > 0` at end of a
      structured decode).
- [ ] One dispatch-decision log line per generate request, visible
      via `athena logs -f`.
- [ ] Structured + temp>0 requests engage the speculative-greedy
      path; bit-identical-greedy contract test still holds.
- [ ] the consuming application mini-corpus 9-file run with `--no-speculative=
      false` (i.e. default) emits real `tokens_per_sec` heartbeats
      throughout; transcript file either completes or fails fast on
      `request_timeout_secs`.
- [ ] `MEMORY.md` updated with `project_m48-heartbeat-and-widening.md`
      entry.

## Out of scope (deferred)

- MLX-side scheduler/dispatch batching (Finding #3 wedge: ~73% in
  `wait_for_one`). Architectural; needs upstream changes.
- A real `generationTokensPerSec` rolling window on `/healthz`
  (the M46.5 deferral — even more meaningful now that M48.1 makes
  the per-request signal real).
- Prefill heartbeat fires per chunk only when the loop calls
  `clearCache` (current cadence) — finer-grained tracking would need
  the chunked prefill loop in each generate function to publish to a
  separate TaskLocal. Defer until someone hits a prefill-side hang.

## Evidence on file

- `/tmp/m48-evidence-sample.txt` — 5 s sample of PID 69798 while
  wedged. Worker thread Thread_39367498 shows 83% at
  GuidedGreedy.swift:80 (asyncEval), 14% at GuidedGreedy.swift:67
  (decoder.pick + materialization), 3% at GuidedGreedy.swift:78
  (logitsAndHidden). MLX scheduler thread parked on
  pthread_cond_wait.
- `athena status` snapshot: `inflight: 1`, LLM resident 29 GB, total
  resident 66 GB, free 36 GB. `lastRequestAt: 1779943754.749804`
  (May 27, 2026 22:09 UTC; sample taken May 28 00:00 UTC ⇒ request
  was ~2 hours old at sample time).
- `athena logs -f` showed continuous heartbeat at
  `tokens=0 tokens_per_sec=0.0` covering elapsed 889 s → 915 s
  before manual cancel.
