# ADR 024 — in-memory data protection against a co-resident adversary

**Status:** **Accepted — T1 + T2 committed** (hardening program, M80; ADR 024
tier-scope decision resolved with the operator at Phase 0). **T3 (encrypt idle
prompt-cache KV at-rest-in-RAM) deferred** as a stretch, gated on whether the
target deployment runs the prompt cache with sensitive idle entries. **T1
shipped v0.10.198** (process lockdown: Hardened Runtime + no `get-task-allow`,
notarization hook); **T2 shipped v0.10.200** (`ProcessHardening.swift`: core
dumps off unconditionally via `setrlimit(RLIMIT_CORE,0)`; opt-in
`ptrace(PT_DENY_ATTACH)` behind `deny_debugger_attach` / `ATHENA_DENY_DEBUGGER`;
best-effort `secureZero`/`memset_s` of the decoded request PCM on every
audio/video path. **`mlock` deferred, honestly** — the sensitive data is the
GB-scale weights/KV pool itself, and pinning it fights the unified-memory
governor budget (ADR 011/023); a broad `mlockall` is rejected for that reason,
and there is no clean *small* region to pin today, so the seam exists unused).
This ADR captures the threat model, the hardware ceiling, and the defense ladder
*before* a consumer request ("confidential KV cache" / "Private Cloud
Compute-style protection") lands, so we do not get pushed into re-litigating an
unbuildable NVIDIA-Confidential-Computing port. Motivated by a consumer floating
PCC/NVIDIA-CC as a model for protecting the M59 prompt-prefix cache in memory.

**Phase-0 Spike B (2026-06-20) validated T1 on the real binary:** re-signing
`athena` with the Hardened Runtime + `get-task-allow=false` flipped a
debugger/`debugserver` attach from "reaches the task" (baseline adhoc:
*attached to process, but could not pause execution*) to **`error: attach
failed: Not allowed to attach to process`** (AMFI task-port denial), and MLX's
`default.metallib` still loaded under the Hardened Runtime (a 27B model loaded
and `/v1/chat/completions` returned) — clearing the named M43.3 risk. The binary
is effectively statically linked (one weak `/usr/lib/swift` platform dylib), so
library validation is a non-issue and no `disable-library-validation` /
`allow-jit` entitlement is granted. Notarization could not be exercised locally
(no signing identity) and remains an operator-side release step
(`NOTARIZE=1` + `CODESIGN_IDENTITY` + `NOTARYTOOL_PROFILE` in `deploy/build.sh`).

## Context

The M59 prompt-prefix cache (and, more broadly, the resident model weights and
live KV) sit in plaintext in DRAM while the daemon serves. The original framing
of the consumer's ask was *confidential computing* — protect data-in-use from the
**infrastructure operator** (a cloud threat). That framing does **not** fit
Athena (a local daemon; the operator is the machine owner) and rests on two
premises that do not hold on our platform:

1. **NVIDIA Confidential Computing's primitive does not exist on Apple Silicon.**
   NVIDIA CC (Hopper+) encrypts VRAM with keys the host can't read and emits a
   GPU attestation report. Apple Silicon is **unified memory** — CPU and GPU
   share one physical DRAM pool (exactly what MLX exploits for zero-copy) — and
   exposes **no** public API for confidential/encrypted GPU memory or GPU
   attestation for general Metal compute. A literal CC port is off the table at
   the hardware level, not merely unimplemented.

2. **Apple Private Cloud Compute's guarantees are a whole-system posture, not an
   ML-framework feature, and whether it uses MLX is unconfirmed + irrelevant.**
   PCC's in-memory protection comes from Apple **owning the kernel**: signed and
   externally-inspectable OS images, secure boot rooted in hardware, no
   persistent state, no operator runtime/shell access, and hardware attestation
   of the boot chain that the client verifies before sending data. None of that
   is inheritable by a third-party userland daemon, and it has nothing to do with
   the inference framework.

**The correct threat model for Athena is a co-resident malicious process** — user-
level malware, or root-level malware on an otherwise-intact machine — that
scrapes the daemon's address space (`task_for_pid()` → `mach_vm_read()`) while it
serves. "The owner is trusted" is a false dichotomy: an owner's machine can be
infected. Data-in-use against a same-host software adversary is a legitimate,
and on macOS a meaningfully defensible, target.

**Current baseline is the bottom of the ladder.** `deploy/build.sh` does a plain
`xcodebuild` with default (ad-hoc/dev) signing — **no `--options runtime`, no
`.entitlements`, no notarization.** A dev-signed binary effectively carries
`get-task-allow`, so any process that can call `task_for_pid` (and root trivially)
can attach and read the entire address space — KV cache, weights, decrypted
secrets. The consumer's concern is the current state, not a hypothetical.

A known cost is already on record: M43.3 found that **hardened-runtime spawn
broke MLX's metallib bundle lookup** ("Failed to load the default metallib"). So
moving up the ladder carries a real, named engineering cost — it is not a flag
flip.

## The access path (the lever)

To read another process's memory a co-resident attacker needs
`task_for_pid()` → `mach_vm_read()`. macOS/AMFI gates that on **both** the
*caller's* entitlement (`com.apple.security.cs.debugger`) **and** the *target's*
`get-task-allow` flag. That gate is the lever the defense ladder pulls.

## Decision

**T1 + T2 committed; T3 deferred.** The tier-scope decision was resolved with the
operator at the hardening-program Phase-0 gate: build **T1** (process lockdown)
and **T2** (side-channel hygiene) now — the high-value, bounded work — and treat
**T3** (encrypt idle prompt-cache KV at-rest-in-RAM) as a stretch, gated on
whether the target deployment runs the prompt cache with sensitive idle entries.
This ADR records the threat model, the defense ladder, and the binding honesty
boundary. (The original framing left tier selection open pending the consumer's
exact adversary — co-resident non-root vs. root-but-SIP-intact vs. "prove I'm
running untampered Athena"; the operator chose the co-resident-scraper lockdown
as the committed baseline.)

### The defense ladder (what each tier buys, smallest blast radius first)

**Tier 1 — process lockdown.** Build with **Hardened Runtime + notarized +
without `get-task-allow` + without the debugger entitlement.** The single biggest
win and the direct answer to "lock the pool from another process":
- **Non-root malware: blocked** — cannot obtain our task port.
- **Root malware on a SIP-intact machine: substantially raised** — AMFI still
  requires the *caller* to hold an Apple-granted debugger entitlement against a
  target that permits it; a hardened binary without `get-task-allow` can't be
  attached even by root unless AMFI is disabled (boot-arg → SIP off → **reboot**,
  very loud) or via a kernel/AMFI exploit.
- **Cost:** must first resolve the M43.3 hardened-runtime metallib-bundling
  breakage.

**Tier 2 — shrink the plaintext footprint and close side channels.**
- `setrlimit(RLIMIT_CORE, 0)` — a crash can't dump the cache to `/cores`.
- Selective `mlock` of sensitive regions — keep them out of swap (already
  encrypted on macOS) and the hibernation image. Caveat: GB-scale KV under
  unified memory fights the governor budget (ADR 011/023), so this is small
  selected regions, not the whole pool.
- `ptrace(PT_DENY_ATTACH)` — cheap defense-in-depth (kernel-bypassable).
- Best-effort zeroize-on-evict for buffers we own — same ADR-023 honesty
  boundary (MLX's pool may hand the page back before we touch it).

**Tier 3 — encrypt the *idle* cache at-rest-in-RAM (most responsive to the
co-resident threat).** The prompt-prefix cache is mostly **cold entries on
idle-TTL waiting for reuse** — long dwell, high value, the prime scraping target.
Keep idle/retained KV entries **encrypted in RAM** (AES-GCM via CryptoKit, key
wrapped by the Secure Enclave), decrypt only the slice being restored into a live
decode. Collapses the plaintext window from "the whole pool, indefinitely" to
"the single entry currently decoding." Cost: per-hit crypto on cache
restore; complexity.

### Honesty boundary (binding)

**The active working set is irreducibly plaintext in DRAM, and a kernel/SIP-off
adversary can read it.** Specifically:

- **Resident model weights and the KV slice under live computation must be
  plaintext** — Metal kernels operate on plaintext `MLXArray`s; there is **no
  encrypted-compute path on Metal.** Tier 3 shrinks the window; it does not close
  it. Bulk AES of GB-scale data runs in-process, so the symmetric key is briefly
  in RAM (the SEP can wrap/gate the key but cannot practically do the bulk
  crypto). Tier 3 is "shrink the window + raise the bar," **not** a guarantee.
- **Apple Silicon DRAM is encrypted by the memory controller, but transparently**
  — that defeats **physical/cold-boot** attacks, not a software read through the
  kernel. Irrelevant to the co-resident-software adversary.
- **Against a kernel-level / SIP-disabled-root adversary the plaintext working
  set is readable**, and no userland daemon can prevent it — the same exposure
  the keychain's in-use secrets carry. PCC closes this only by owning the kernel;
  we do not. We state this residual; we do not pretend to close it.

Net achievable posture (if the full ladder is built): from "trivially scrapable"
to **"non-root can't; SIP-intact root can't without a loud reboot or kernel
exploit; idle cache is ciphertext; only the active slice + weights remain
plaintext."**

### Rejected / out of scope

- **A literal NVIDIA-Confidential-Computing port** (encrypted GPU memory + GPU
  attestation) — rejected; the hardware primitive does not exist on Apple Silicon
  (unified memory, no public confidential-Metal API).
- **"Be like PCC by using MLX"** — rejected on a false premise; PCC's guarantees
  are an OS/boot/attestation posture we cannot inherit as userland, independent of
  the ML framework.
- **Moving the working set into a TEE / Secure Enclave** — infeasible; the SEP is
  a small key-management coprocessor and cannot run MLX or hold a GB-scale KV
  cache.
- **Defending against a kernel-level / SIP-off adversary** — out of scope; see the
  honesty boundary.

### Adjacent, separable work (not this ADR's in-memory question)

- **Build attestation** — if a consumer's real ask is "prove I'm talking to
  untampered Athena," that is a release-engineering / supply-chain feature (signed
  + inspectable build manifest, exposed binary measurement), the only
  PCC-shaped piece portable to userland. It attests the **code**, not the memory;
  track separately if requested.
- **At-rest** — if M59.5 (disk-backed prompt cache, currently deferred) ever
  lands, the SQLCipher path (M34) is the one place we *can* give a real
  cryptographic guarantee, because it is our file, not MLX's pool.

## Consequences (if/when tiers are adopted)

- Tier 1 alone closes the largest current gap (dev-signed = trivially
  debuggable) and is a prerequisite for a credible co-resident posture; it costs
  the M43.3 metallib-bundling fix and a notarization step in `deploy/build.sh`.
- Tiers interact with the governor (ADR 011/023): `mlock` pins physical pages
  against the budget; Tier 3 adds CPU on cache hit/restore. Sizing must respect
  the Metal memory budget.
- All tier **decision logic** (what to lock/encrypt/evict, when) stays MLX-free
  and unit-pinned (ADR 008/009); crypto is CryptoKit + SEP. MLX numerics
  unchanged.

### Validation (when built)

- MLX-free decision logic (eviction/encryption/lock policy) → unit tests
  (ADR 008/009).
- Tier 1: assert a non-root, debugger-entitled process **cannot** `task_for_pid`
  the notarized hardened binary; assert metallib still loads (M43.3 regression).
- Tier 3: round-trip an idle cache entry through encrypt → evict-window →
  decrypt-on-restore with bit-identical KV, and assert idle entries are
  ciphertext in a memory snapshot.

Plan + slices to follow on approval (separate plan doc), once a tier is chosen.

## Verification (2026-06-20) — MLX has no encrypted-compute primitive; PCC's confidentiality is hardware, not MLX

Primary-source check of the rejected premises and the honesty boundary:

- **MLX itself: no encrypted compute / no encrypted `MLXArray`.** Against
  `ml-explore/mlx` `main`: a recursive tree scan returns **zero** paths matching
  `crypt|secure|enclave|confidential|attest|sgx|tdx|sev`; code search for
  `encrypt` and issue/PR search for `encrypted`/`confidential`/`enclave` return
  **zero** hits; the allocator is the plain `mlx/backend/metal/allocator.cpp`
  (no encrypted-buffer variant); no `SECURITY.md`. `MLXArray` remains a lazy
  plain-buffer array — the "no encrypted-compute path on Metal" boundary holds.
- **WWDC 2026 reaffirms premise 2 from the opposite direction.** Apple's PCC
  expansion (June 2026) runs Apple Foundation Models on **Google Cloud / NVIDIA
  GPUs**, with confidentiality from **NVIDIA Confidential Computing (Blackwell) +
  Intel TDX + Google Titan** — **MLX is not cited as a confidentiality
  mechanism**, and PCC isn't even on Apple Silicon in this deployment. Confirms
  confidential compute = a hardware + attestation layer, not an ML-framework
  feature. (Apple Security Research, *Expanding Private Cloud Compute*.)
- **New, non-changing wrinkle:** MLX now has a **CUDA backend**
  (`mlx/backend/cuda/allocator.cpp`) — compute *portability* to NVIDIA, **not**
  encrypted compute. It does not give MLX a confidentiality primitive.

Net: the ADR's rejected premises and honesty boundary stand, now anchored to a
dated source check. Re-verify if a future MLX release adds a security/crypto
module or a protected-buffer allocator.
