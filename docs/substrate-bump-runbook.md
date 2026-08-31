# Substrate bump runbook

How to move Athena's `mlx-swift-lm` pin forward. Written from the `integration-2026-07-07` → `integration-2026-07-31` bump (#91 / PR #93, 2026-08-01), which hit every failure class below at least once. A bump is a **measurement exercise, not a version edit**: the version edit is two lines; everything else here is what makes the result trustworthy.

The worked example behind each step: issue #91 (blast-radius inventory), PR #93 (execution, 4 review rounds), ADR 044 (trait decision), ADR 028 (bit-identity re-scoping), issues #92/#95 (what the gates surfaced).

## 0. Preconditions

- **Target must be an immutable dated tag** (`integration-YYYY-MM-DD`), never the `integration` branch and never a bare commit SHA. The branch is force-pushed on every rebuild, which orphans whatever a consumer pinned; SPM fetches refs, so an orphaned SHA breaks every cold build while CI stays green on cache warmth (#86, the founding incident). This is the fork's consumer contract and Athena's AGENTS.md "Dependencies" rule.
- Have the substrate checkout (`~/Source/mlx/mlx-swift-lm`) fetched with tags so old→new diffs are cheap.

## 1. Measure the blast radius before touching anything

Diff `old-tag..new-tag` restricted to what Athena actually links — `Libraries/MLXLLM`, `MLXVLM`, `MLXLMCommon`, `MLXEmbedders`, `MLXHuggingFace`, and `Package.swift` — and inventory four classes:

1. **API signature changes** Athena calls (e.g. `GenerateParameters.prefillStepSize` went `Int`(512) → `Int?`(nil) — silently rechunking the serve path until #88 pinned it). Grep Athena's `Sources/` for each changed symbol.
2. **Behavioral changes** on paths Athena rides (e.g. `TokenIterator.next()` gaining a `clearCache()` cadence perturbs ADR 023's accounting; cancellation-order flips move metering edges).
3. **Numeric changes** (mask fills, pooler rewrites, quantization) — these are what the heavy gates exist for; note which model families they touch.
4. **Manifest/structural changes**: new traits, new vendored code, tools-version bumps, dependency floor moves. Each of these is a *decision* Athena must make deliberately (see ADR 044 for the trait template), and traits specifically trigger the cache poison in §4.

Write the inventory into the bump issue (the #91 table is the format). Anything in class 1–3 that lands without a line in that table is the next incident.

## 2. The mlx-swift coupling — trust nothing but a cold build

- **The substrate's own manifest floor is not evidence.** At 5b892140 it declared `.upToNextMinor(from: "0.31.4")` while calling `MLXArray.maskFill` / `DType.greatestFiniteMagnitudeArray`, which land in 0.31.5 — a floor it does not itself compile against. Determine the real floor by building cold, then **raise Athena's floor in `Package.swift`** to the version you verified against. Never rely on `Package.resolved` alone: a partially-restored `.build` can resolve inside the stale range and fail with missing-member errors that read like a substrate bug (that is exactly how the first bump attempt died).
- **Check mlx-swift's `swift-tools-version`.** 0.31.5+ ships a 6.3 manifest, which imposes a *toolchain* floor (Swift 6.3 ⇒ Xcode 26.5+, verified; 26.3/Swift 6.2.4 fails, 26.4 unverified) that is invisible in Athena's own `swift-tools-version: 6.1` header. The failure on an old toolchain is a **resolution error naming mlx-swift's tools version**, before any build. If the floor moved: bump the CI runner image (`ci.yml` unit + seed **and** `release.yml`, together — the shipped binary's toolchain moves with it) and the three prerequisites docs (README, `docs/quickstart.md`, CONTRIBUTING).

## 3. Apply the pin

- `Package.swift`: the `revision:` tag on both nothing else — plus whatever floors §2 demands, plus `traits:` decisions from §1 class 4. Keep the pin comment honest (what the tag is, why the floor).
- `swift package resolve`, then verify `Package.resolved` records the tag under `"branch"` with the expected SHA under `"revision"`, and the mlx-swift version you intend.
- `deploy/build.sh Release` prints the resolved pins into the build log (the ship ritual copies them into the tag annotation) — confirm the guard emits the new pin.

## 4. The stale-checkout poison (any manifest-structure change, traits especially)

SwiftPM validates dependency-level settings (e.g. `traits: []`) against the manifest of the **currently checked-out** dependency *before* re-resolving. Any cached checkout of the previous pin therefore hard-fails resolution with `Disabled default traits … that declares no traits` (or the analogous error for whatever structural feature moved):

- **Local `swift test`/`swift build`**: `rm -rf .build/checkouts .build/workspace-state.json`, re-resolve.
- **Local xcodebuild**: `rm -rf .build/xcode/SourcePackages`.
- **CI**: `restore-keys` prefix-restores a stale `.build` from ANY sibling key, so the whole cache namespace is poisoned — but since #96 the resolve step **self-heals the re-resolvable case**: the `unit` job (and `seed`, only on the cache-miss runs where its lookup-only probe lets it resolve at all — an exact hit skips resolve entirely) resolves explicitly and, on failure, emits a `::warning::`, clears `.build`, and resolves exactly once more, paying one cold rebuild with no hand-authored salt bump. A salt bump is **still required in exactly two cases**: **(a) a mis-keyed cache entry** (the #56 story) — it resolves cleanly, so the self-heal never fires and no warning ever appears; the symptom is a full recompile on every run despite an exact cache hit; **(b) re-resolvable poison whose cache key does not rotate** — a `Package.swift`-only dependency-shape change with `Package.resolved` unchanged is an exact key hit, so the poisoned entry is never replaced: every `unit` run re-heals and pays the full cold build indefinitely, and the recurring `::warning::` from the self-heal step is the signal to bump (the signal covers only (b); (a) never warns). When bumping: **bump the SPM salt** (`spm-vN` → `spm-vN+1`, all six occurrences across unit + seed, byte-identical), **in the bump PR itself**: a separate salt-bump PR re-poisons the new namespace, because main's post-merge seed still builds the pre-bump tree and a failed build never overwrites its save (the #94 lesson — its review is the full analysis).

## 5. Gates — all of them, in this order

| Gate | Command | What it proves |
|---|---|---|
| Unit tier | `./deploy/test.sh` | Compiles + decision seams hold against the new pin (787 tests, MLX-free — proves **no numerics**) |
| Release build | `./deploy/build.sh Release` | Metal shaders compile; pin guard emits provenance |
| Cold resolve | `swift package resolve --cache-path <fresh> --scratch-path <fresh>` from a copied manifest | The tag is fetchable with zero cache — the #86 failure mode; the local rehearsal of what `cold-resolve.yml` enforces on CI (weekly, on dispatch, and on any PR touching `Package.swift`/`Package.resolved`) |
| e2e tool-calling | `deploy/e2e-tool-choice-auto.sh` | ADR 034/035 vs. any `ToolCallProcessor`/tool-parsing changes |
| e2e count-tokens | `deploy/e2e-count-tokens.sh` | ADR 042 count==usage parity through the new `container.prepare` |
| Vision smoke | red-PNG `image_url` chat against a VLM checkpoint → expect the color | ADR 010 vs. any vision-tower/pooler rewrite |
| **MTP parity gate (#64)** | `TEST_RUNNER_ATHENA_RUN_MODEL_TESTS=1 xcodebuild test -scheme athena-Package -destination platform=macOS -derivedDataPath .build/xcode -only-testing:AthenaCoreTests/MTPSpeculativeParityTests` | Temp-0 speculative == greedy byte-for-byte **with the drafter provably engaged** (fails, not skips, if it proposed nothing) |
| Anything §1 named | model-family-specific | The blast radius is the test plan |

Heavy gates need xcodebuild (metallib; ADR 009 — they crash under bare `swift test`) and real checkpoints in `~/.athena/models`. **Gate boundary to know:** the parity gate *skips* (with a stated reason) when the drafter fails to *pair* — a bump that breaks pairing shows up as a skip a human must notice, not a red test. #92 is the live example: Qwen3.5 `-mtp` checkpoints have never paired (`language_model.mtp.*` vs the bare `mtp.*` sanitize filter), which the gate's first run exposed after the old "verification" had been vacuously green.

## 6. Docs owed in the same PR

- **ADR 028's bit-identity claim is per-pin evidence.** It does not survive a bump. Re-run the parity gate at the new pin and re-stamp the ADR (revision, model pair, proposed/accepted counts) — or explicitly re-scope the claim to the revision it was measured on. Never leave it silently pointing at the old pin.
- A new ADR for any §1 class-4 decision (trait, vendored surface) — ADR 044 is the template, including its measured honesty boundary (xcodebuild ignores `.when(traits:)` at build planning; `nm -gU athena | grep -c xgrammar` tripwire per Xcode major).
- AGENTS.md: ADR index entry; the Dependencies bullet if the discipline itself changed.
- Toolchain-floor docs (§2) if the floor moved. State only **measured** versions — "26.4+" was flagged in review precisely because the evidence bracketed 26.3-bad/26.5-good without testing 26.4.

## 7. Delivery

One PR, base main, `Closes` the bump issue and every issue it resolves (grep the body for closing keywords next to issues that must stay open — GitHub ignores negation). The PR's CI run gives you a **genuine on-CI cold-resolve demonstration**: `cold-resolve.yml` triggers on any change to `Package.swift`/`Package.resolved`, and resolves with no cache and no `.build`, so a bump PR always exercises the new pin against the real remote. (The `unit` job's own run proves nothing about reachability — `restore-keys` silently serves the pins from a restored mirror, which is exactly how #86 stayed green.) Expect the review loop to attack the claims; every number in the PR body should have a log behind it.

## Failure signatures → cause

| Symptom | Cause | Fix |
|---|---|---|
| `Disabled default traits … declares no traits` | Stale checkout of a pre-traits pin (local or restored cache) | §4 |
| `'mlx-swift' >= X contains incompatible tools version` | Toolchain older than mlx-swift's manifest | §2 (Xcode/runner floor) |
| Missing-member errors on MLX types (`maskFill`, …) | mlx-swift resolved below the real floor (manifest floor too low + warm `.build`) | §2 (raise the manifest floor) |
| Cold build fails; `unit` green on cache warmth | Unreachable/orphaned pin — the warm `unit` cache masks it; `cold-resolve.yml`'s `Resolve pins with no cache` job reddens on its next run (weekly, on dispatch, and on any PR touching `Package.swift`/`Package.resolved` — a direct-to-main manifest push gets no run until the weekly cron) | §0 (tag pin); `.github/workflows/cold-resolve.yml` |
| `keyNotFound` at MTP drafter load | Checkpoint key-prefix vs sanitize filter mismatch — may be pre-existing, diff the load chain across pins before blaming the bump | #92 |
| Parity gate skipped | Drafter didn't pair — investigate before reading the run as green | §5 boundary |
