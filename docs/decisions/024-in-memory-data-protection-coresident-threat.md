# ADR 024 — in-memory data protection against a co-resident adversary

**Status (amended publication S0, 2026-07-07):** **T1 + T2 retained; T3 REMOVED.** T3 (encrypt idle prompt-cache KV at-rest-in-RAM, `IdleKVCipher`) rode the M59 `PrefixKVCache`, which was **Qwen3.5-vendored-path-exclusive**; the S0 de-vendor moved Qwen3.5 onto the substrate generate loop (no idle-prompt-cache seam yet), so T3 was inert and was deleted with the rest of the prompt-cache stack (opt-in, off by default). Capability tracked for upstream re-implementation: **mlx-tracker #37** (idle KV-at-rest encryption), gated on #24 (the prompt-cache seam). **T1 (hardened runtime / no `get-task-allow` / notarization) and T2 (`ProcessHardening`: `RLIMIT_CORE=0`, `PT_DENY_ATTACH`, PCM zeroize) are UNAFFECTED** — not prompt-cache-coupled, still shipped. Original acceptance record follows.

_Original:_ **Accepted — T1 + T2 + T3 ALL SHIPPED** (hardening program, M80–M81;
ADR 024 tier-scope decision resolved with the operator at Phase 0, T3 gate
cleared + shipped M81). **T3 (encrypt idle prompt-cache KV at-rest-in-RAM)** was
deferred as a stretch, gated on whether the target deployment runs the prompt
cache with sensitive idle entries; **that gate cleared** — the deployment handles
HIPAA (PHI) and/or PCI (PAN) workloads whose reused prefixes sit in the M59 idle
cache. **T3 shipped v0.10.206–208** (plan: `docs/confidential-kv-cache-plan.md`;
usage: `docs/confidential-kv-cache.md`): S1 v0.10.206 = `IdleKVCipher`
(AES-256-GCM, swift-crypto) + the MLX-free `KVFrame` wire format + the
`KVByteCodec` MLXArray bridge (the serialize seam a future M59.5 reuses); S2
v0.10.207 = `PrefixKVCache` holds idle entries as ciphertext-at-rest-always
(seal-on-store, secureZero the transient plaintext, decrypt just-in-time on a hit
into the working clone, key rotates when the pool drains), behind
`prompt_cache_encrypt_idle` (off by default); S3 v0.10.208 = the **bit-identical
gate re-run with encryption ON PASSED** on the real Qwen3.5-27B-4bit-mtp —
warm-prefix-reuse over an AES-256-GCM-sealed entry (HIT at L=2868, B=2560,
2560/2886 tokens reused) produced **byte-identical** output to a cold prefill,
and ran *faster* end-to-end (37.0 vs 24.4 tok/s for the cold-prefill request)
since prefill collapsed 2886→326 tokens and the per-hit decrypt added no
measurable stall — confirming the seal→open→decode round-trip is lossless and the
restore comfortably beats a cold re-prefill. **T1 shipped v0.10.198** (process
lockdown: Hardened Runtime + no
`get-task-allow`, notarization hook); **T2 shipped v0.10.200**
(`ProcessHardening.swift`: core
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

> **Amendment (2026-06-22) — SEP envelope key for *persisted* KV (triggers the
> re-open condition below; see the new section "Amendment: persisted-KV key model
> (ADR 027)" at the end of this file).** T3's "RAM-only key, NO Secure Enclave"
> decision was **explicitly scoped to a process-ephemeral key** and recorded the
> condition *"Re-open SEP only if a future M59.5 persists the key."* **ADR 027
> (disk-backed KV snapshots) persists the key — that condition is now met.** The
> amendment adopts a **DEK/KEK envelope** with a **swappable KEK**
> (passphrase/keyfile now → SEP-bound later), reopening SEP **for the persisted
> path only**. The in-RAM T3 decision (RAM-only key for the idle pool) is
> **unchanged**.

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

**T1 + T2 committed (M80); T3 accepted (M81, gate cleared).** The tier-scope
decision was resolved with the operator at the hardening-program Phase-0 gate:
build **T1** (process lockdown) and **T2** (side-channel hygiene) first — the
high-value, bounded work — and treat **T3** (encrypt idle prompt-cache KV
at-rest-in-RAM) as a stretch, gated on whether the target deployment runs the
prompt cache with sensitive idle entries. **That T3 gate is now cleared** (M81):
the deployment handles HIPAA/PCI workloads whose reused prefixes sit in the idle
cache, so T3 is accepted with the refinements recorded under "Tier 3" below
(swap-led justification, RAM-only key, ciphertext-at-rest-always, independent
`prompt_cache_encrypt_idle` toggle). Plan: `docs/confidential-kv-cache-plan.md`.
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
co-resident threat). SHIPPED M81 v0.10.206–208 (bit-identical gate PASSED with
encryption on); plan `docs/confidential-kv-cache-plan.md`, usage
`docs/confidential-kv-cache.md`.** The prompt-prefix
cache is mostly **cold entries on idle-TTL waiting for reuse** — long dwell, high
value, the prime scraping target. Keep idle/retained KV entries **encrypted in
RAM** (AES-256-GCM via CryptoKit), decrypt only the slice being restored into a
live decode. Collapses the plaintext window from "the whole pool, indefinitely"
to "the single entry currently decoding." Cost: per-hit crypto on cache restore;
a net-new KV serialize/deserialize seam; complexity.

As-accepted refinements (M81):

- **Justification leads with swap, not the in-RAM scraper.** T1 already blocks a
  non-root `task_for_pid`, so T3's strongest *T1-independent* value is **swap /
  compressed-memory leakage**: macOS can compress and swap the long-dwell idle
  pool under pressure, and if host swap-encryption / FileVault is off, plaintext
  PHI/PAN KV could hit disk — outside any process-memory adversary and outside
  T1's scope. T3 ensures **only ciphertext is ever swappable**, which is the
  clean compliance story ("encrypted at rest, including volatile caches and
  anything that can reach swap"). This is *why T3 and not `mlock`* for the idle
  pool — encrypted pages stay swappable/reclaimable (the governor keeps its
  budget) while never exposing plaintext to swap, whereas pinning GB-scale KV
  fights the unified-memory budget (ADR 011/023). Secondary value:
  residency-window collapse (a momentary core-dump/debugger/`task_for_pid`-
  regression read catches only the one decoding entry) and defense-in-depth /
  attestation (survives a T1 misconfig; an auditor can be told the idle cache is
  AES-256-GCM).
- **Key model = RAM-only, NO Secure Enclave (narrows this ADR's original "key
  wrapped by the Secure Enclave" sketch).** A random per-process
  `SymmetricKey(.bits256)`, rotated/zeroed on full pool flush. SEP-wrapping is
  **rejected** for a process-ephemeral key: the key is necessarily in our RAM
  during every bulk seal/open, so against the co-resident/kernel threat SEP
  changes nothing (a kernel adversary reads the unwrapped key regardless; a
  non-root one is already blocked by T1), and its only value would be an auditor
  "keys in Secure Enclave" checkbox that is partly theater for a RAM-only key.
  Re-open SEP only if a future M59.5 *persists* the key.
- **Ciphertext-at-rest-always (a refinement of "encrypt-on-evict").** Because the
  cache is clone-on-hit (the entry's own tensors are never the working buffer —
  `acquire` clones them into a fresh working cache under the lock), an `Entry`
  holds **only ciphertext from `store` to evict**; `acquire` decrypts
  just-in-time into the working clone (the irreducibly-plaintext active entry)
  and wipes the transient buffer. This strictly dominates literal refcount-gated
  encryption (same guarantee, simpler invariant, no wider plaintext window).
- **Latency posture = "must beat cold."** No fixed SLA; the invariant is
  restore < cold re-prefill (hardware AES ~GB/s vs ~190 s prefill wins
  comfortably; measured on a ~760 MB entry, numbers recorded on completion), with
  a guard that prefers cold if a restore would ever cost more than re-prefilling.
- **Toggle = independent `prompt_cache_encrypt_idle`** (off by default, mirroring
  `encrypt_store`), not a broader "confidential mode" umbrella.
- **Bit-identical gate is sacred:** AES-GCM is lossless, so warm-with-encryption
  output must stay byte-identical to cold — re-run `e2e-m59-prefix-cache.sh` with
  encryption on. A decrypt that can't reproduce the exact bytes is a correctness
  bug, not a degraded mode.
- **Honesty boundary unchanged** (see below): T3 shrinks the window to the active
  decoding entry; the key + active working set + weights remain plaintext and a
  kernel/SIP-off adversary reads them. The transient plaintext wipe is
  best-effort, but the at-rest/swap guarantee is designed not to depend on it.

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

## Amendment: persisted-KV key model (ADR 027) — 2026-06-22

**Status:** **Accepted — envelope + keyfile KEK SHIPPED v0.10.212–215** (ADR 027
S1–S3b); the **SEP-bound KEK (`kek_type=sep`) is DEFERRED** to S6, gated on the
headless-launchd SEP operability spike (the blob header reserves the slot for a
drop-in swap). Scope: the **at-rest, on-disk** KV key **only**. The T3 in-RAM
idle-pool decision (RAM-only `SymmetricKey`, no SEP) is **unchanged** — this
amendment governs a *different* surface (a file on disk), created by ADR 027
(`docs/decisions/027-disk-kv-snapshots.md`).

### Why re-open SEP now

T3 rejected SEP for a **process-ephemeral** key and recorded the explicit gate:
*"Re-open SEP only if a future M59.5 persists the key."* ADR 027 persists KV (and
therefore a key) to disk for **resume-across-restart**. A RAM-only key is, by
construction, gone after restart — so it **cannot** decrypt a disk blob on the next
boot. Persistence forces a persisted key, which forces the SEP question back open.

### Decision — DEK/KEK envelope with a swappable KEK

Do **not** encrypt the KV blob directly under a persisted key (that path is the
"hardware-locked-forever / key-on-disk" trap). Instead:

- **DEK (data key):** a random per-blob `SymmetricKey(.bits256)`; AES-256-GCM over
  the `KVByteCodec` bytes — i.e. the **existing `IdleKVCipher` cipher, reused
  verbatim**. The DEK never persists in the clear.
- **KEK (key-encrypting key):** wraps the DEK; the **wrapped-DEK is stored in the
  blob header** (ADR 027 §4). The KEK is **swappable**, recorded by a header
  `kek_type`:
  - **`keyfile` / `passphrase` (ship first).** A high-entropy operator keyfile
    (HKDF-derived wrapping key) or passphrase (KDF-derived). Cross-restart works
    immediately; **encrypted-at-rest**, but the key material lives on the host —
    defends disk theft / backups / another local user, **not** an adversary holding
    the key.
  - **`sep` (follow-up slice).** A Secure-Enclave-backed **P-256** key
    (non-extractable) wraps the DEK via key-agreement/ECIES. The DEK is unwrapped
    *inside* the enclave path; the wrapping key never leaves the chip. Drop-in: a
    new `kek_type` + a `KEKProvider` implementation, **no blob-format change**.
- **Multi-node (deferred, reachable):** because SEP keys are **asymmetric**, a
  future cluster wraps the DEK **once per peer** (ECIES to each node's SEP *public*
  key — multi-recipient), or re-wraps online when a node joins. **No SEP key ever
  leaves any chip**, and the blob is **not hardware-locked-forever** — provided the
  DEK/KEK split exists now. (Research: internal SSD-streaming / KV-snapshot notes
  §3a.1.)

### New at-rest secret class — threat notes

- The persisted artifact is **{AES-256-GCM ciphertext blob + a wrapped-DEK in its
  header}**. Confidentiality reduces to the **KEK**, and the header's `kek_type`
  states the posture honestly — we never claim SEP-grade protection for a keyfile
  KEK.
- **In-RAM boundary is unchanged.** During a restore the DEK and the decrypted KV
  are briefly plaintext in RAM — exactly the T3 window, governed by the existing
  honesty boundary above. This amendment adds **nothing** to the in-RAM exposure;
  it only ensures the **disk** copy is never plaintext.
- **Operability (open spike, ADR 027 plan):** a headless launchd daemon must access
  the SEP KEK **without biometric/user-presence** (device-bound,
  `kSecAttrAccessibleAfterFirstUnlock`-class) → the Mac must be unlocked once
  post-boot; key-loss/recovery if the data dir is restored to a *different* Mac is a
  named operability question, not a blocker for the `keyfile`/`passphrase` ship.

### Honesty boundary (unchanged, restated for the disk surface)

The envelope protects **data at rest on disk**. It does **not** create
encrypted-compute, does **not** change the irreducibly-plaintext active working set,
and does **not** defend a kernel/SIP-off adversary who can read the unwrapped DEK
from RAM during a restore. SEP raises the bar for the *persisted key* (non-extractable,
hardware-bound); it does not move the in-RAM ceiling.

### Validation (on implementation)

- Envelope **DEK/KEK wrap→unwrap** round-trip, tamper-reject, and **`kek_type`
  dispatch** — MLX-free unit tests (ADR 008/009).
- `keyfile`/`passphrase` KEK: wrong-key ⇒ clean refusal, never a partial/garbage
  restore.
- `sep` KEK (when built): unwrap requires the enclave; assert the wrapping key is
  non-extractable and the daemon path works headless under launchd.
