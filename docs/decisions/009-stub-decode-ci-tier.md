# 009 — Stub decode CI tier: pure-Swift control-flow seams, not a fake MLX device

**Status:** Accepted — Implemented (M70.2/.3, v0.10.151+). All decision-algebra seams + unit tests landed (`DecodeLoopControl`, `GuidedMask`/`additiveMask` (its `maskedArgmax` is a test-only mirror, not the production seam — see L5), ~~`PrefixKVCache`~~, ~~speculative-sampling~~, structured-schema). The numeric tier stays env-gated (`ATHENA_RUN_MODEL_TESTS`) by design — `MLXArray` cannot evaluate under `swift test`. (M70.3 test-coverage tail ND14/ND15 still open — see `docs/backlog-hitlist.md` #8.) **Two seams struck (#76).** `PrefixKVCache` — the M59 **in-RAM** prefix cache that the ADR 027 disk stack was built on, not a member of it — was deleted by publication S0 (`997d36fd`). `speculative-sampling` (the `SpeculativeSampling` seam) was deleted by #41/#48 (`9d1c5442`) after S0 orphaned it. Neither is in `Sources/` or `Tests/`, so this line claimed two landed seams that are not in the tree. Struck rather than silently dropped, so the list does not read as if the tier always had three seams. **One Consequences deliverable was never built (#101):** `StubDecodeHarness` — see Consequences. That is a different failure from the two struck seams above, which existed and were later removed, and it is called out here because a reader who stops at this Status line would otherwise take "all seams + unit tests landed" at face value. **The L5 seam name was corrected (#136):** this line and the L5 row named `maskedArgmax` as the landed seam; `additiveMask` is the production seam, and `maskedArgmax` is a test-only mirror with no production caller.
**Date:** 2026-06-13
**Milestone:** M70 (audit-remediation; resolves the L-cross-cutting "stub-model CI
tier" keystone + L1/L2/L5/L7/L8, NC4/NC5/NC6 — "CI blindness")

## Context

> **Deliberate retention (#76, 2026-08-01).** This section names `GuidedGreedy`,
> `SpeculativeGeneration`, `SpeculativeSampling` and the prompt-prefix KV cache.
> All four are gone: the first two and `PrefixKVCache` went in publication S0
> (`997d36fd`), and `SpeculativeSampling` followed in #41/#48 (`9d1c5442`).
> It is **kept as written**:
> a Context section records the state the decision was taken in, dated
> 2026-06-13, and editing it to match today's tree would falsify why the tier
> was built rather than correct anything. The live **structured** decode path is
> `GuidedSubstrate` (a plain request routes to `beginGeneration` via
> `DecodeDispatch.route`); the surviving per-invariant record is the table below,
> where the retired rows are struck. Read this section as history, the table as
> current. (The alternative — striking the names here too — was considered and
> rejected: it makes the Context unreadable and duplicates the table's job.)

ADR 008 made the daemon's *server* boundary unit-testable by extracting an
MLX-free `AthenaServerKit`. The *inference* boundary — the decode loops
(`GuidedGreedy`, `GuidedSubstrate`, `SpeculativeGeneration`,
`SpeculativeSampling`), the structured-output guide/vocab, the prompt-prefix KV
cache, and the vector cosine path — is still validated only by env-gated manual
`xcodebuild`/Release-binary runs (`ATHENA_RUN_MODEL_TESTS=1`) plus the host-bound
`e2e-rbac.sh`. The audit calls this out as the dominant remaining test-debt
cluster (baseline L1–L11; re-audit NC4/NC5/NC6). A regression that broadened a
prompt-cache scope key (cross-principal KV leak), inverted LRU eviction
(unbounded growth → OOM), dropped a decode loop's cancellation early-break
(M60.5 wedge), corrupted the structured token→byte mapping, or broke same-seed
sampling reproducibility would ship green.

The task as originally framed was a "fake/stub model + decode loop that runs the
control flow in pure Swift with no Metal kernels." **An empirical probe settled
the shape of that.** Under `swift test` / `./deploy/test.sh`, even a bare
`argMax(MLXArray([...])).item()` — no model weights — aborts the process with:

```
MLX error: Failed to load the default metallib. library not found …
  at .../mlx-swift/Source/Cmlx/mlx-c/mlx/c/array.cpp:232
```

The default `metallib` is a resource that `xcodebuild` bundles and SPM's test
bundle never locates; MLX requires it to materialize **any** array, on CPU or
GPU. So **no `MLXArray` can be evaluated in the CI tier at all.** The four decode
loops thread `MLXArray` end-to-end (`model.logitsAndHidden` → `MLXArray` logits →
`decoder.pick(slice)` → `argMax` → `.item`), so they cannot run there as written.

Two structural options to get past that:

- **(a)** Extract each decode path's MLX-free *decision logic* (operating on
  `[Int]`/`[Float]`/`[UInt8]`/`Bool`) into static seams co-located with the
  loops, test those + the seams that are already pure, and provide a pure-Swift
  scripted-logits/stub-counter harness. The `MLXArray` math stays in `AthenaLLM`,
  byte-unchanged; the loop bodies call the extracted seam where they decided
  inline before.
- **(b)** Abstract the hot path behind a tensor protocol so the *real* loops run
  on a pure-Swift logit source in CI.

## Decision

**Option (a).** The stub decode tier is a **control-flow / decision-algebra
tier, not a numeric tier.** It follows the same extract-pure-seam pattern this
program already used for ~~`rankTopK`~~ (deleted with the vector store by
`c6cab9d6`, ADR 025 S1+S3)/`lengthBuckets`/`AthenaMetrics.percentile`/
`AthenaProxy.describe`: the seams that decide *what* the loop does are pulled
into MLX-free Swift and pinned by `./deploy/test.sh`; the `MLXArray` numerics
that compute the values stay in the MLX-linked targets and remain the province
of the env-gated manual tier + `e2e-rbac.sh`.

Option (b) was rejected: it touches the **bit-identical-greedy decode path** —
the most load-bearing, highest-risk surface in the codebase (the M20/M21/M59
contracts) — purely for test reachability, which violates the M70 rule that this
milestone *re-arms invariants as tests, it does not modify them*. A fake MLX
device is not even available as a fallback: MLX won't initialize without the
metallib, so (b) would still require a non-MLX tensor type threaded through the
hot path.

Co-location, not a new target (per the lengthBuckets/~~rankTopK~~/~~`SpeculativeSampling`~~
precedent — `SpeculativeSampling` was deleted by #41/#48 and `rankTopK` by `c6cab9d6`,
so the precedent now stands on `lengthBuckets` **alone among the three named
here** (`Sources/AthenaEmbedding/MLXEmbeddingModule.swift` + its test); it
lives in an MLX-linked target but never calls MLX in-body, so it
runs fine under `swift test`. That is a narrower class than the four seams
listed under Decision above: `AthenaMetrics.percentile` and
`AthenaProxy.describe` are both alive, but they live in MLX-free targets
(`AthenaServerKit`/`AthenaCore`), so they were never precedents for
co-locating a seam *inside* an MLX-linked target): the new seams live beside their loops in
`AthenaLLM`, except the cross-loop cancellation predicate, which lives in
`AthenaCore` beside `DecodeProgress` (all four loops already import it). No
`Package.swift` change.

### What each invariant gets

Every invariant splits into a **CI-covered mechanism half** and a
**gated-numeric half**, and BOTH are documented (no silent caps — a green CI
tier must not read as "the numerics are covered"):

| Invariant | CI (this tier) | Gated / manual (unchanged) |
|---|---|---|
| ~~**L2** acceptance-rate floor~~ | ~~`SpeculativeStats` observer + accept/reject algebra~~ — **row retired (#47)**: publication S0 removed the vendored decode loop that published to the observer, leaving no in-tree publisher. Both columns are deleted (`SpeculativeStatsTests` and the gated `testStructuredSpeculativeAcceptanceRate`). ADR 032 S4 speculative stats ride the substrate's aggregate `GenerateCompletionInfo` counts instead, which the per-iteration observer never bridged to. | ~~real-model accept rate~~ |
| ~~**L7 / C1** seeded sampling~~ | ~~`SpeculativeSampling` distribution/RNG/tie-break + multi-seed property (already MLX-free)~~ — **row retired (#41/#48, struck by #76)**: publication S0 orphaned the vendored speculative loop that consumed it, and `9d1c5442` then deleted `SpeculativeSampling.swift`/`SpeculativeAcceptance.swift` with their two test files. `grep -rw SpeculativeSampling Sources/ Tests/` returns nothing. Left unstruck, this claimed CI coverage of seeded-sampling reproducibility — the invariant the Context names as "same-seed sampling reproducibility would ship green". The MTP speculative path (ADR 032) uses the substrate's own sampler and verify step; it does not restore this seam. | ~~real-model sampled tokens~~ |
| **NC6** StructuredVocab | pure `build(vocabSize:eos:decode:)` core — C12 eos-sentinel, UTF-8 byte map | real tokenizer |
| ~~**NC6** GuidedDecoder~~ | ~~IDLE→ENFORCING phase machine via a real `byteVocab` guide (`commit`/`forceEnforce`/idleBudget/jsonStart)~~ — **row retired (#49)**: publication S0 deleted the in-closure guided-greedy loop this pinned, orphaning `GuidedDecoder`; the live structured path is `GuidedSubstrate`, covered by the **L5** row below. The tier no longer owes this invariant. | ~~`pick` (MLX argmax)~~ |
| **L5** schema mask | `additiveMask(allowed:vocab:)` — the production seam — plus `maskedArgmax([Float],[UInt8])`, a test-only mirror of the caller's mask-add+argmax math with no production caller (`GuidedMask.swift`'s own split-of-duties note) — scripted off-schema logits — **row corrected (#136)**: this row credited `maskedArgmax` as the production CI seam; `additiveMask` is, and `maskedArgmax` is a test-only mirror that never had a production caller. | the `MLXArray` math around `additiveMask` (`GuidedSubstrate`: `logits + MLXArray(add)`; the argmax runs in the substrate's `ArgMaxSampler`) |
| ~~**NC4** prefix cache~~ | ~~`scopeKey`/`commonPrefixLength`/eviction policy over scalar descriptors~~ — **row retired (#76)**: publication S0 deleted `PrefixKVCache` and the whole disk-snapshot stack it anchored (ADR 027 Status — Qwen3.5-fused-path-exclusive, inert once S0 moved Qwen3.5 onto the substrate generate loop; the capability that would restore this row is the upstream prompt-cache seam, mlx-tracker **#24**; #36 is the disk-snapshot layer built on top of it). Neither `scopeKey` nor `commonPrefixLength` exists anywhere in `Sources/` or `Tests/`. Left unstruck between two struck rows, this asserted live CI coverage of the very invariant the Context names as a cross-principal KV leak — a **false coverage claim**, which is the failure this ADR opens by forbidding, inverted. The tier no longer owes this invariant; there is no cache to scope. | ~~KV-tensor clone/restore bit-identity~~ |
| **NC5** cancellation | `DecodeLoopControl.isCancelled()` — the predicate the decode loop consults, pinned by `DecodeLoopControlTests` — **row corrected, not retired (#76)**: the seam is live, but this row named `shouldStop`, which has never existed, and "the 4 loops", of which one survives (`GuidedSubstrate.swift:91`; S0 deleted the other three). Corrected rather than struck, since a real seam with real tests still pins the invariant. | the live disconnect bridge (M68.4 A8, e2e) |
| **L1** greedy parity | ~~accept/commit algebra: committed-seq == greedy-seq property~~ — **no longer pinned (#47)**: the algebra it described lived in the vendored decode loop publication S0 removed. | ~~real-model bit-identity (`ATHENA_RUN_MODEL_TESTS`)~~ — **gone (#47)**: `testStructuredGreedyParityAcrossSpeculative` stopped discriminating post-S0 (both arms carry a schema, so `DecodeDispatch` routes both to `GuidedSubstrate` and the `speculative` flag is decode-irrelevant) and was deleted rather than left as false coverage. **Gate restored non-vacuously by #64** (with the #91 substrate bump): `MTPSpeculativeParityTests.testUnstructuredGreedyParityAcrossSpeculative` (`ATHENA_RUN_MODEL_TESTS=1`) — no schema, so both arms route to `beginGeneration` and `speculative: true` provably takes the ADR 032 MTP-drafter overload (the arm must report proposed draft tokens > 0, or the test fails rather than passing vacuously). |
| **M48.3** temperature inertness under a Guide | **empty — no seam was extracted, and the row says so rather than omitting the invariant.** The invariant is *structural*, not merely unobservable: `GuidedSubstrate.generate` takes **no temperature parameter at all**, and constructs `ArgMaxSampler()` inline. Note this is a weaker claim than "un-CI-able" — an eval-free factory seam asserting the sampler's type would be constructible, since `ArgMaxSampler()` touches no MLX op on construction. It was not built; end-to-end inertness is what the gated test buys. `DecodeDispatch.route` already pins the routing, and L5 covers the mask seam. | `GuidedTemperatureInertnessTests.testTemperatureIsInertUnderGuide` (`ATHENA_RUN_MODEL_TESTS=1`) — a schema request at temperature 0.1 must emit the byte-identical sequence a temperature-0 request does. **This gate was never absent and was never in this table**: it landed as `testStructuredGreedyParityTempIneretUnderGuide` with M48.3 itself (`033f1b15`, 2026-05-28) and has been in the tree continuously since; #47 renamed it in place and rehomed it out of the class whose *other two* tests it deleted. The row is a long-standing omission being corrected, not a record of something added back. Its arms **differed** in the `speculative` flag until #67 pinned both to `false` and removed the parameter that allowed it; `DecodeDispatch.route(hasSchema:hasLogprobSink:)` never took `speculative` at all, so the flag was decode-irrelevant here — but it was not load-irrelevant (`MLXLLMModule.loadMTPDrafterIfEligible` guards on it, so the temp>0 arm attempted a drafter load the temp-0 arm did not), which is why #67 pinned rather than merely documented it. Distinct from both struck rows above: L1 was bit-identity across `speculative`, L2 was the acceptance-rate floor. |
| **L8** stub vectors | model-distinguishable stub embedding (stub-only behavior tweak) | real embedding numerics |
| ~~**NF3** TriAttention~~ | ~~geometry/eviction already CI-tested (NF1/NF9)~~ — **row retired (#76)**: publication S0 (`997d36fd`) deleted `TriAttentionCacheTests` and `TriAttentionE2ETests`, and no `NF1`/`NF9` reference survives in `Sources/` or `Tests/`. The surviving `KVCompressionTests` pins the `kv_compression` **knob resolution** (default/TOML/env precedence, case-folding, fail-closed on an unknown value, the `kvScheme` accessor) — not TriAttention geometry or eviction, which is what this row claimed. TriAttention now rides the substrate's `GenerateParameters.kvScheme` (ADR 028 S0 amendment). | ~~compress/`gatedDeltaOps` numeric parity stays gated~~ |

## Consequences

- The new seams are MLX-free static functions/structs co-located with their
  loops; they compile and link into the MLX-linked targets and, being MLX-free
  in body, run under `./deploy/test.sh` exactly as `lengthBuckets`
  already does (and, when this was written, ~~`rankTopK`~~ and
  ~~`SpeculativeSampling`~~ — both since deleted, by `c6cab9d6` and #41/#48
  respectively).
  ~~`AthenaCoreTests` gains a small `StubDecodeHarness` (scripted
  logits/tokens + a controllable `DecodeProgressCounter` whose `isCancelled`
  can flip).~~ **NOT BUILT.** Unlike the struck seams above, this one was never
  removed — it never existed. The durable check is path-scoped:
  `git log -S"StubDecodeHarness" --all -- Sources/ Tests/ clients/` returns
  nothing, and at the object level every blob in the repository containing the
  string is a revision of this ADR. (A path-LESS search is not the check: it
  also matches this ADR's own revisions, so it grows whenever a revision's diff changes
  the number of occurrences of the string — `git log -S` is the pickaxe, and a
  commit that merely reworks the surrounding prose does not appear — and what
  it returns depends on clone shape, since a branch-only revision is invisible
  to a clone lacking that ref. Which commits it happens to list is deliberately
  not enumerated here: that enumeration has been written twice and was wrong
  both times.)
  The seams it was meant to drive
  did land, each pinned directly by its own test, so nothing is uncovered by
  its absence; this line described a deliverable of its own Decision that was
  not built. Recorded rather than struck silently, because "promised and never
  built" is a different failure from "built and later deleted".
- The extractions are **behavior-preserving refactors**: each loop calls the
  seam where it made the decision inline before, so the bit-identical-greedy /
  structured-output / TurboQuant / TriAttention contracts are unchanged.
  (**Deliberate retention, same basis as the Context banner (#76):** TurboQuant
  was hard-removed by ADR 028 in v0.10.217, months after this refactor. The
  sentence records what M70.2 preserved at the time and is left as written;
  it is not a claim that TurboQuant exists today.) The
  `e2e-rbac.sh` gate (561/0) + the real-model smokes prove no drift on every
  slice.
- The numeric half of each invariant (real-model bit-identity, cosine scoring,
  `gatedDeltaOps` parity) stays env-gated; this is recorded per item in the
  tracker and in the table above so a green `deploy/test.sh` is never mistaken
  for numeric coverage. The two tiers are complementary: CI covers the
  *mechanism*, the manual/e2e tier covers the *numbers*.
- One genuine production change is in scope and isolated: **L8** makes the
  `--engine stub` embedding vectors model-distinguishable (stub-only — never the
  real model path), gated by e2e.
- `AthenaServerKit` (ADR 008) and `AthenaDecodeKit`-style new targets are NOT
  introduced — the co-location precedent keeps `Package.swift` stable. NB4
  (Commands/ testability) is its own slice with its own target relocation
  (`Engine`/`KVCompression` → MLX-free target), tracked separately per ADR 008's
  Consequences; it does not depend on this tier.
