# M52 — Decode-path completeness brief (M52.A / M52.B / M52.C)

Planning doc captured 2026-05-28. Consolidates the three larger
tracked-deferred items from the C cluster of the deferred-items
review:

- **M52.A** — M20 #909 Metal perf kernels (TurboQuant decode)
- **M52.B** — M21 calibrated-trig scoring (TriAttention eviction)
- **M52.C** — M2.2 sampled MTP speculative residual (lift greedy-only)

Each could become its own M-number when actually scheduled (M52,
M53, M54 respectively). They're consolidated here because they
share the speculative + decode code area and the same "tracked-
deferred, correctness-first" discipline that M20/M21/M2 set up.
Splitting only when one of them is ready to ship.

## Status

All three NOT STARTED. Each has been "tracked deferred" since its
parent program shipped — they're known follow-ups, not surprises.
The deferrals were correctness-first decisions: M20 shipped the
KV-codec correctness without the perf kernels because the codec
matters more than the speedup; M21 shipped norm-only eviction
because the calibrated version needs offline stats-capture
infrastructure; M2.2 shipped greedy-only speculative because the
sampled-residual case is harder to verify than the greedy bit-
identical case.

The shared discipline (named in M21's memory): "DO NOT treat as a
silent descope."

## Why brief all three together

They:
- Share the speculative + decode code path
- Share the substrate-delta touchpoint (mlx-swift-lm fork)
- Share the test-discipline pattern (bit-identical or
  distributionally-identical contract per M47.1)
- Have wildly different size profiles (M52.A is weeks of Metal
  kernel work; M52.B is weeks of scoring + stats-capture infra;
  M52.C is days of MTP loop work)
- Have INDEPENDENT design surfaces — they don't naturally ship
  together

So this doc is the consolidated context-and-sequence overview;
each gets its own implementation plan (`m52a-plan.md` /
`m52b-plan.md` / `m52c-plan.md`) WHEN actually scheduled.

---

## M52.A — M20 #909 perf kernels for TurboQuant decode

### Status

TRACKED-DEFERRED. M20 (v0.9.65-68, 2026-05-23) shipped the
TurboQuant KV-cache codec with **correct** decode that reconstructs
full K/V on read (host-side / standard MLX matmul). The reference
implementation in #909 has **1024-thread + 2-pass Metal decode
kernels** that fuse the dequantize + attention into one pass and
parallelize across the head-dim differently than the standard MLX
quantized matmul.

The Athena M20 memory called this out:

> #909 perf kernels = deferred follow-up. STOP — M21=TriAttention
> is separate.
> ...
> 1024-thread + 2-pass decode kernels are a large standalone Metal
> effort = a TRACKED DEFERRED perf follow-up (correctness-first;
> identical precedent to M2's "Release/multi-token speedup =
> future tuning, not M2"). The codec is correct & functional
> without them.

### Why it matters

TurboQuant trades KV memory for one decode-pass overhead. The
default decode reconstructs full K/V every step — extra GPU work
on the hot path. The #909 fused-kernel design eliminates that
overhead. On long-context decodes (>=16k tokens), the difference
is meaningful — potentially 1.3-1.5× tok/s vs current TurboQuant
decode.

For the consuming application's workload (~14k prompt + variable output),
TurboQuant isn't typically engaged today (default `kv_compression
= none`). So the immediate operational win is small. The strategic
case is "TurboQuant becomes the default once #909 lands" — which
opens 2× context length at the same memory budget.

### Scope

- Vendor / port the #909 1024-thread + 2-pass decode kernels into
  Athena's substrate fork at `~/Source/mlx/mlx-swift-lm`.
- Wire them into `TurboQuant`-mode KV cache decode in
  `Sources/AthenaModels/.../TurboQuantCache.swift`.
- Gate behind the existing `kv_compression = turboquant` knob —
  no new operator-visible config.
- The bit-identical contract for greedy decode (M47.1 + M20.4)
  must hold across the kernel swap.

### Substrate delta

This adds substrate code, so it grows the M20 "tracked substrate
delta" — already 3 unpushed commits ahead of upstream
(`a32e72e/29823e4/e1f78f8`). M52.A adds a 4th. The
[release-distribution program]
(../../.internal/projects/-Users-goodolclint-Source-Athena/memory/project_release-distribution.md)
already calls these out as the #1 blocker; M52.A increases that
blocker's weight.

### Risks

- Metal kernel correctness is the hard part. Reference impl
  (#909) is open-source; behaviorally correct against the
  reference is the gate.
- M52.A interacts with M21's TriAttention eviction code path
  (also touches KV decode). Need to verify TriAttention + #909
  TurboQuant kernels co-exist; might require an additional
  config gate.
- Effort: probably 2-3 weeks of focused Metal work + 1 week
  validation. Real program.

### Dependencies

- Substrate fork access (already in place at
  `~/Source/mlx/mlx-swift-lm`)
- The M20.4 `TurboQuantE2ETests` heavy regression gate
- Apple Metal toolchain (already required for Athena build)

---

## M52.B — M21 calibrated-trig scoring for TriAttention eviction

### Status

TRACKED-DEFERRED. M21 (v0.9.69-72, 2026-05-25) shipped TriAttention
with **norm-only** eviction scoring (score = `‖k‖`, no stats, no
positions). The reference implementation has a more sophisticated
calibrated-trig scoring that uses pre-RoPE Q-center stats captured
offline during a calibration run.

From the M21 memory:

> Calibration = norm-only NOW, calibrated-trig DEFERRED (tracked).
> ...
> Calibrated trig scoring (needs from-architecture pre-RoPE
> Q-center stats capture against vendored Athena modules, storing
> stats beside the model, ModelConvert-shaped) = a TRACKED
> deferred follow-up. Mirrors M20's deferred-perf-kernel discipline
> (correctness-first, ship-small). **Do NOT treat as a silent
> descope.**

### Why it matters

Norm-only eviction is fast and works (M21.4 e2e validation
passed), but the calibrated version has measurably better quality
preservation under aggressive eviction. The quality delta widens
as the eviction budget tightens — at relaxed budgets (cache
length ≥ context / 2), the difference is marginal; at aggressive
budgets (cache length ≤ context / 4), calibrated-trig is
noticeably better.

For Athena's typical workload (long-prompt structured extraction
without aggressive eviction), the delta is small. For the case
where TriAttention would actually run at aggressive budget — long
multi-turn chat over hours of conversation — calibrated-trig is
the right answer.

### Scope

Three parts:

1. **Calibration infrastructure**: a one-time per-model
   calibration run that captures pre-RoPE Q-center stats from a
   representative prompt corpus. Stored beside the model, similar
   shape to `ModelConvert`'s metadata.
2. **Calibrated-trig scorer**: replace / supplement the existing
   norm-only scorer with the trig-based variant. Per-attention-
   layer; uses the captured stats + position info during decode.
3. **Knob extension**: extend `kv_compression = triattention` to
   accept a calibration mode (`triattention-norm` vs
   `triattention-trig`) OR auto-detect from the presence of
   calibration metadata. Backward-compatible default = norm-only.

### Risks

- Calibration corpus selection biases the scoring — needs design
  thought about what represents "representative."
- The calibration tool is a new ModelConvert-shaped artifact in
  the project. Adds operator surface (`athena calibrate`?) and
  documentation.
- The norm-only path stays in production indefinitely as the
  default. Calibrated-trig is opt-in.
- Effort: probably 3-4 weeks (calibration infra is the gnarly
  part, not the scoring itself).

### Dependencies

- M21.1-M21.4 TriAttention substrate (already shipped)
- A representative calibration corpus (selection is its own
  design question)

---

## M52.C — M2.2 sampled MTP speculative residual

### Status

TRACKED-DEFERRED. M2.2 (the original MTP speculative implementation,
v0.5.x) shipped greedy-only speculative because the bit-identical
contract for greedy is easier to verify than the distributionally-
identical contract for sampling. M40 (v0.10.27-29) added sampling-
speculative via Leviathan/Chen sampling but for the
non-Guide path. The **residual sampling** case — the per-token
sampling correction when an MTP draft is rejected mid-decode — is
the unfinished sub-piece.

### Why it matters

Today's M40 sampling-speculative does Leviathan/Chen acceptance:
on reject, it discards the draft and samples from the residual
distribution (`P_target - P_draft` clamped to positive). The
sampling-speculative path correctly engages and is distributionally
identical to non-speculative sampling at the same (temp, top_p,
seed).

The residual-sampling correction is what M2.2's deferral originally
flagged: under tight schemas, the residual collapses to near-zero
probability mass across the valid set, and the resulting sampled
token may not be valid under the Guide. M48.3 sidestepped this by
gating sampling-speculative to no-schema (the `schemaJSON == nil`
precondition that M51 / M47.3 also touches).

So M52.C is partly already done (M40 lifted the temp gate) and
partly still tied to M51 / M47.3 (the schema gate).

### Scope

Reconcile and document:

1. What's done: M40 sampling-speculative for the unstructured
   case.
2. What M51 will land: symmetric Guide-mask in
   SpeculativeSamplingGenerate's draft picks.
3. What M52.C adds beyond M51: the operator-facing flag to relax
   `samplingEligible`'s no-schema precondition, with the
   contract (residual sampling under a Guide collapses to masked
   argmax of the valid set, distributionally identical to greedy
   speculative + Guide).

In other words: M52.C is the **operator visibility + documentation
+ precondition relaxation** layer on top of M51's mechanical fix.

### Risks

- The contract "sampling under a Guide collapses to argmax" is
  intuitive but should be proven with a dedicated test (extension
  of the M47.1 parity test).
- Releasing the precondition is an observable behavior change for
  the rare consumer who sends `temperature > 0 + schema +
  speculative: true` and expects "no speculative engaged" today.
  Likely no real-world consumer hits this; the consuming application's
  behavior matches the contract either way.
- Effort: small (~1 day) IF M51 has landed.

### Dependencies

- M51 (symmetric Guide-mask) — landed first
- M47.1 parity test framework — already in place

---

## Sequencing recommendation

These are decoupled — no hard ordering. The sane decision points:

1. **If the consuming application / production workloads stay similar** —
   structured extraction, long prompts, moderate output — none of
   M52.A/B/C is urgent. M50 (clearCache audit) and the M35
   commercial-readiness loose ends (#7/#8/#9/#4a) are higher
   priority because they affect reliability + commercial posture.

2. **If long-context workloads grow** (e.g. multi-turn agents, hour-
   long transcription contexts) — M52.A becomes most valuable
   (TurboQuant default opens 2× context budget) and M52.B becomes
   valuable in parallel.

3. **If a consumer asks for sampled structured output** (rare but
   plausible) — M51 + M52.C ship together as a small bundle.

## Definition of done per slice

- M52.A: TurboQuant decode runs through the #909 kernels; bit-
  identical greedy contract holds; existing TurboQuantE2ETests
  pass + benchmark shows the expected speedup.
- M52.B: Calibrated-trig scorer available as
  `kv_compression = triattention-trig`; calibration tool documented;
  quality benchmark shows the expected improvement at aggressive
  eviction budgets.
- M52.C: Sampling-speculative engages under a Guide; M47.1-style
  parity test passes for the temp>0 + schema + speculative case;
  documentation reflects the new eligibility.

## Out of scope (all slices)

- Implementing any of M52.A/B/C without an operational trigger.
  These are "ready to scope when needed," not "must ship soon."
- Cross-slice abstraction (e.g. unifying TurboQuant + TriAttention
  + clearCache patterns into a single KV-management layer). Each
  slice stays independent.
- Tier B auto-pull (M41 deferred), multi-resident embedding pool
  (M39 deferred) — those are separate workstreams covered
  elsewhere in the deferred-items review.
