# Multi-node and embedded deployment — research note

**Status:** Research only. No decision, no chosen architecture, no implementation plan.
This note maps the decision space from the repo's *actual current state* (v0.10.208) so a
later ADR/change-plan can be written against ground truth rather than guesswork.

**Scope:** Two deployment shapes the project does not support today —
(1) **multi-node** (the governor's budget spanning more than one Mac) and
(2) **embedded** (multiple independent products on one host contending for one physical
Metal pool) — plus the cross-cutting question of whether these are the same problem at
different transport scopes.

**Method:** current code read directly and cited (`file:line`); external prior art cited
inline; fact-in-code separated from inference; assumptions and gaps marked explicitly.
Nothing here amends an ADR — if any option below is pursued it must go through the
brownfield change gate (CLAUDE.md) and reconcile with ADR 011/013/024/025.

---

## 1. Current-state findings (ground truth)

Everything below is **single-process, single-host** unless stated. That is not an
omission to be fixed in passing; it is the binding thesis (ADR 011: "the unified Metal
memory governor is Athena's reason to exist", `docs/decisions/011-unified-memory-governor-as-thesis.md`).

### 1.1 Memory governor / pool-budget management

**Fact-in-code.**
- The governor is a single Swift `actor` — `public actor MemoryGovernor`
  ([MemoryGovernor.swift:108](../Sources/AthenaCore/MemoryGovernor.swift#L108)). All state
  is private actor-isolated instance state; the `actor` keyword is the *only* concurrency
  primitive guarding admission. There are no locks, no shared file, no RPC.
- Budget is **one number** for one host: `totalBudgetBytes`
  ([MemoryGovernor.swift:148](../Sources/AthenaCore/MemoryGovernor.swift#L148)), defaulting
  to `0.75 × ProcessInfo.processInfo.physicalMemory`
  ([GovernorConfig.swift:32](../Sources/AthenaCore/GovernorConfig.swift#L32)), overridable
  by `--budget-bytes`/TOML. It maps to the process-global Metal pool via
  `MLX.Memory.memoryLimit` / `MLX.Memory.cacheLimit` set once at serve bring-up
  ([Load.swift](../Sources/athena/Commands/Load.swift), the `mlx_cache_limit_bytes` seam
  from ADR 023 / `docs/governor-truth-plan.md`).
- "Tenants" are a **fixed enum of 5 slots** — `llm, transcription, textEmbedding,
  diarization, speakerEmbedding`
  ([ModuleID.swift:6](../Sources/AthenaCore/ModuleID.swift#L6)) — each holding at most one
  resident model, registered at startup with an evictability flag (LLM `evictable:false`,
  the rest `evictable:true`,
  [Load.swift:697](../Sources/athena/Commands/Load.swift#L697)).
- Admission API: `beginLoadIfNeeded` (non-blocking, →503),
  `ensureLoaded` (blocking, coalesces concurrent callers for the *same* slot onto one load
  via `inFlight`/`inFlightToken`), `awaitLoad`/`peekLoad` (ADR 015 block-until-ready),
  `unload`, `releaseSlot`, `snapshot`
  ([MemoryGovernor.swift:283-830](../Sources/AthenaCore/MemoryGovernor.swift#L283)).
- Arbitration is **budget-based, not a queue**: when `residentBytes + estimate > budget`,
  LRU-evict an *evictable* slot; on memory-pressure (90% high-water by `phys_footprint`),
  shed the prompt-cache pool first, then evict
  ([MemoryGovernor.swift:480-739](../Sources/AthenaCore/MemoryGovernor.swift#L480)).
- The governor is **MLX-free by construction** — it never imports MLX; it takes injected
  closures (`MemoryProbe`, `UnloadHook`, `EventHook`, `PromptCachePoolProbe`,
  `PromptCacheReliefHook`,
  [MemoryGovernor.swift:124-146](../Sources/AthenaCore/MemoryGovernor.swift#L124)) wired at
  the call site ([Load.swift:512](../Sources/athena/Commands/Load.swift#L512)). The probe
  is a *witness* of real resident bytes, reconciled into `learnedFootprint` per slot
  ([MemoryGovernor.swift:198](../Sources/AthenaCore/MemoryGovernor.swift#L198)).

**Absent (stated plainly).** No lease, no remote/distributed budget, no cross-process
sharing, no node/peer/cluster concept. The actor's reach is exactly one process. The ADR
023 honesty boundary already forbids per-tenant attribution of the *reclaimable* MLX cache
(one global untagged pool); multi-node would inherit that constraint and stack on top of it.

### 1.2 Worker queue / scheduler / concurrency arbitration

**Fact-in-code.**
- The async job queue (`jobs` table, `/v1/queue*`) was **deleted**, not deprecated
  (ADR 025 S2, v0.10.203). Confirmed: no `enqueue`/`jobs`/`/v1/queue` live code remains;
  only removal comments. Model lifecycle ops (`pull`/`convert`/`prune`) now run
  **synchronously with SSE progress** on `POST /api/models/{pull,convert,prune}`
  ([AthenaServer.swift:505](../Sources/athena/Server/AthenaServer.swift#L505),
  `performModelOp`/`streamModelOp` ~2839-3005).
- Concurrent inference is **not serialized into a scheduler**. Same-slot concurrent loads
  coalesce; different slots load in parallel; once a decode starts it runs to completion
  or cancellation. **There is no priority, fairness, FIFO, or preemption** — only the OS
  scheduler plus LRU eviction ordering.
- The only admission gates besides the governor are in-process and per-principal:
  `RateLimiter` (token bucket, 429 + Retry-After) and `ConcurrencyLimiter` (global +
  per-principal in-flight caps, 429)
  ([RateLimit.swift](../Sources/AthenaServerKit/RateLimit.swift), both actors, in-memory
  only, exempt `/healthz`/`/metrics`/`/ui`).

**Inference (mine).** The system is intentionally a *budget arbiter, not a scheduler*.
Any multi-node or multi-client design that wants fairness/priority across requestors would
be adding a genuinely new subsystem, not extending an existing one. That is a feature gap,
not a regression to repair.

### 1.3 Daemon lifecycle, IPC, config, model-store, auth

**Fact-in-code.**
- Lifecycle: user-context daemon is a forked `athena load` child tracked by a pidfile with
  SIGTERM→(5s)→SIGKILL; system daemon is `launchctl bootstrap system <plist>` with
  `RunAtLoad:true`/`KeepAlive:true`
  ([DaemonLifecycle.swift:62-281](../Sources/athena/Commands/DaemonLifecycle.swift#L62),
  [LaunchdPlist.swift](../Sources/AthenaDeploy/LaunchdPlist.swift)). Graceful in-flight
  drain on SIGTERM via Hummingbird `runService()`
  ([AthenaServer.swift:672](../Sources/athena/Server/AthenaServer.swift#L672)).
- **IPC: HTTP-loopback is the *only* surface.** `127.0.0.1:7447` (Hummingbird/NIO), no
  unix socket, no XPC, no mach ports, no shared memory, no inter-daemon channel. Passive
  oracle: no outbound except HF weight fetches (CLAUDE.md; binding ADR rule).
- Config: flat TOML, `AthenaConfig` ([AthenaConfig.swift](../Sources/AthenaDeploy/AthenaConfig.swift)),
  `ATHENA_CONFIG` override, CLI > env > TOML precedence. Port/host are config keys.
- Model-store: store entries are **symlinks into the HF cache** under `~/.athena/models`;
  `convert` produces real dirs. Selection is store-backed per-module (ADR 026, no
  allowlist table); unknown id → `400 model_not_available`, ambiguous → `400 ambiguous_model`.
- Persistence: one SQLite `AthenaStore` actor, post-ADR-025/026 tables are **auth/audit/usage
  only**; loopback + no-creds → no DB at all (`StoreMode.swift`, ADR 025 S4).
- Auth: bearer-token RBAC, SHA-256 hashes, bootstrap keys (env/file) + managed DB tokens,
  constant-time compare, per-route single permission
  ([Auth.swift](../Sources/AthenaServerKit/Auth.swift)). **Loopback + no seeded creds opens
  every route** (dev mode); non-loopback + no creds fails closed.

**Absent (stated plainly).** No node identity, peer discovery, cluster membership, leader,
heartbeat, or replication anywhere in the tree. One daemon per install is the whole model.

### 1.4 Inference-backend boundary (coupling to MLX/Metal)

**Fact-in-code.**
- The package splits cleanly into **MLX-free** (`AthenaCore`, `AthenaServerKit`,
  `AthenaDeploy`, `AthenaClient`) and **MLX-bound** (`AthenaLLM`, `AthenaTranscription`,
  `AthenaEmbedding`, `AthenaModels`, the `athena` executable) targets
  ([Package.swift](../Package.swift)). ADR 008/009 enforce this seam for testability.
- The decoupling protocol is `InferenceModule`
  ([InferenceModule.swift:17](../Sources/AthenaCore/InferenceModule.swift#L17)):
  `id`, `residentBytes`, `memoryEstimate()`, `load(reservation:)`, `unload()`.
  **`infer` is deliberately *not* in the protocol** — each modality exposes its own typed
  API (`LLMModule.generate`, `DiarizationModule.diarize`, …) and the server holds typed
  handles ([AthenaServer.swift:22-29](../Sources/athena/Server/AthenaServer.swift#L22)).
- **No RPC/serialization boundary exists between the HTTP handler and the compute.** The
  handler `await`s the in-process module actor, which directly calls the MLX substrate
  (`ModelContainer.perform { … }`) over shared in-process `MLXArray` buffers. Global MLX
  state is set process-wide (`MLX.Memory.memoryLimit/cacheLimit`, `clearCache()` ~every 256
  tokens). The only FFI is the Rust structured-output sieve (one-shot per request, not a
  streaming backend).
- The stub-decode CI tier (ADR 009) proves the seam works: decision algebra (e.g.
  `GovernorMemory.resolveCacheLimit`) lives MLX-free and unit-pinned; the MLX side effect
  is applied separately in the executable.

**Inference (mine).** The protocol seam is real and load/unload/budget already flow through
it — but it stops at the modality boundary, not at a transport boundary. Today
"swap the backend" means "reimplement five typed module protocols and replace direct
`MLXArray` access," and out-of-process anything would require introducing the serialization
boundary that does not currently exist. The governor thesis (ADR 011: "never compose at the
inference layer") is the reason this boundary was never built.

### 1.5 Packaging / embedding seams

**Fact-in-code.**
- Products: one executable `athena`; public libraries `AthenaCore` and `AthenaClient`;
  `AthenaServerKit`/`AthenaStore` exist as targets but are not exported products
  ([Package.swift](../Package.swift)).
- `clients/` is a **standalone cross-platform SwiftPM package** (Linux/Windows; deps =
  `swift-argument-parser` only, zero MLX) exporting `AthenaClient` — a pure HTTP-over-bearer
  thin client (`DaemonOptions`, `HTTPClient.send`, `Remote*` wrappers,
  [clients/Package.swift](../clients/Package.swift)). It is already a usable SDK shape, but
  **HTTP-only**: there is no in-process linking path to the inference modules (they are
  private targets holding Metal state under a singleton governor).
- Build/dist: `xcodebuild` (required for Metal shaders), single executable + resource
  bundles, Hardened Runtime + no `get-task-allow` + optional notarization
  ([deploy/build.sh](../deploy/build.sh), [deploy/athena.entitlements](../deploy/athena.entitlements),
  ADR 024 T1). Install lays out `libexec/athena/athena` (+ metallib bundle), an exec
  *wrapper* (not symlink) at `bin/athena`, config at `etc/athena/`. **No .pkg/brew today.**
- Wire protocol: hand-authored OpenAPI 3.0.3 served at `/openapi.json`, version-stamped
  from the single `Athena.appVersion` ([Athena.swift](../Sources/athena/Athena.swift),
  [OpenAPISpec.swift](../Sources/athena/Server/OpenAPISpec.swift)). **The daemon version
  *is* the protocol version** — there is no independent wire-protocol version and no `/v2`.

**Absent (stated plainly).** No shared-singleton-service abstraction beyond "one daemon on
a port." No in-process embedding story. A second product on the same host must speak HTTP
to the daemon today.

---

## 2. Problem framing

Two pressures push past the single-process governor:

- **Multi-node:** one model (or one workload mix) too large or too slow for one Mac; the
  operator owns several Macs and wants them to behave as one budgeted resource. The
  governor's "one number for one Metal pool" assumption breaks: there are now N pools, N
  processes, N failure domains.
- **Embedded:** several *independent* products on one Mac each want Athena's inference but
  must not each spin up an uncoordinated MLX allocator — that is precisely the
  "two uncoordinated allocators on one Metal pool defeats the governor" failure ADR 011
  forbids. The pressure is for *one* governor that many local clients share.

Both are, at heart, **the governor's authority extending across a boundary it currently
does not cross** — a host boundary in one case, a process boundary in the other. Whether
that makes them one problem is the cross-cutting question (§5). Neither is a localized fix;
both are new subsystems and both touch the passive-oracle, OpenAPI-surface, and
data-at-rest invariants — i.e. both are brownfield-gate changes, not patches.

---

## 3. Decision space — Area 1: Multi-node

Options and tradeoffs only; no recommendation.

### 3.1 Cross-node budget coordination model

The current governor is a single actor owning one number (§1.1). Spanning nodes needs a
model for *who decides admission*:

- **(A) Centralized arbiter.** One node runs the authoritative governor; others run thin
  "resource agents" reporting capacity and requesting leases. *Pros:* admission stays a
  single serialization point — closest to today's actor semantics, easiest to reason about
  for floors/eviction. *Cons:* arbiter is a SPOF and a latency hop on every cold-load;
  needs failover (§3.3); the arbiter's view of remote `phys_footprint` is only as truthful
  as the agents' probes (and ADR 023 already documents that local probe truthfulness is
  hard).
- **(B) Replicated/eventually-consistent state.** Each node runs a governor; budget state
  is gossiped/replicated. *Pros:* no SPOF, admission is local-fast. *Cons:* two nodes can
  admit against the same headroom during a partition (overcommit → Metal OOM, which today
  is a local 503 but cross-node becomes a correctness bug); needs conflict resolution and a
  consistency model the current code has no analog for.
- **(C) Partitioned/no-shared-budget.** Each node owns its budget independently; a router
  places work by static capability. *Pros:* zero new consensus; reuses the existing
  per-process governor verbatim per node. *Cons:* not actually a shared budget — it is load
  balancing; defeats the "behave as one resource" goal for single large models.

**Tradeoff axis:** consistency vs. availability vs. latency-on-admission, against the
backdrop that admission today is a *synchronous actor call* and any cross-node model adds a
network round-trip to the cold-load path (ADR 015's `cold_load_wait_secs` budget would have
to absorb it).

### 3.2 Parallelism strategy (what "multi-node" actually buys)

Distinct goals, distinct mechanisms, distinct support status:

- **Pipeline parallelism** (layers split across nodes; activations flow rank0→rankN). Serves
  **capacity** (run a model that doesn't fit on one Mac). This is exactly what MLX's
  distributed *ring* backend does today (§4). Current code supports *none* of it: the model
  is loaded whole into one `ModelContainer` per process (§1.4); there is no layer-sharding,
  no cross-node activation transport, no rank concept.
- **Tensor parallelism** (each layer's matmuls split across nodes). Serves **single-stream
  throughput/latency** but is communication-heavy (per-layer all-reduce) — viable only with
  very low-latency interconnect (JACCL/RDMA, §4). Not supported; would require
  substrate-level integration with MLX collectives, far below the `InferenceModule` seam.
- **Expert parallelism** (MoE experts placed across nodes). Serves **capacity** for MoE
  models specifically. Athena has a Gemma4 MoE path (ADR 002) but expert routing is
  in-process via `SwitchGLU`; cross-node expert dispatch is absent and, per the M63.5
  notes, even *in-process* Gemma4 expert routing has been a substrate limitation.
- **Replica/data parallelism** (whole model on each node, route whole requests). Serves
  **capacity for concurrency** (more simultaneous requests), not bigger models or faster
  single streams. This is the *only* strategy the current code could approach without
  touching the inference layer — it is §3.1(C) plus a router.

**Key point for framing:** "multi-node" is not one feature. Capacity-for-big-models
(pipeline/expert/tensor) lives *below* Athena's protocol seam and is owned by MLX/substrate;
capacity-for-concurrency (replicas) lives *above* it and is owned by routing/governor. The
two have almost nothing in common in this codebase.

### 3.3 Coordinator/governor role assignment & failover

Single-host has no election problem — the daemon either runs or doesn't (launchd restarts
it, §1.3). Multi-node introduces: who is the arbiter (3.1A), what happens when it dies
mid-lease, how a replacement reconstructs outstanding leases, and how nodes agree on
membership. None of this exists. Prior art is standard (Raft/Paxos for the arbiter log,
lease-based leadership, or an external coordinator) — see §4 — but every option imports a
dependency and a failure mode the passive-oracle daemon has never had. Open: does a
home/small-cluster deployment (the realistic Apple-Silicon case, ≤4–8 Macs by Thunderbolt
topology limits, §4) even want consensus, or is a static-leader-with-manual-failover
acceptable? Unresolved here.

### 3.4 Transport-agnostic budget/lease protocol — required guarantees

If budget is leased across a boundary, the protocol must define at minimum:

- **Crash-safety / reclamation:** a node that dies holding a lease must have it reclaimed
  (TTL + heartbeat, or arbiter-side liveness). Today eviction is synchronous and local
  ([MemoryGovernor.swift:716](../Sources/AthenaCore/MemoryGovernor.swift#L716)); a remote
  holder crashing has no analog.
- **Floors / non-eviction:** the LLM slot is `evictable:false` today
  ([Load.swift:697](../Sources/athena/Commands/Load.swift#L697)) — a lease protocol needs a
  way to express "this reservation is a floor, never reclaim it" across nodes.
- **Truthful accounting across the wire:** ADR 023's local truthfulness problem
  (`phys_footprint` vs estimates vs untagged MLX cache) becomes a *distributed* truthfulness
  problem; the arbiter trusts remote probes it cannot verify.
- **Idempotency / ordering:** lease grant/renew/release must tolerate retransmission
  (the client already has a bounded-retry `HTTPClient`, §1.5, but that is request-level not
  lease-level).

**Open questions:** consistency model (linearizable leases vs. lease-with-grace); does the
protocol ride the existing HTTP surface (simplest, but admission-on-HTTP adds latency) or a
new low-latency channel; how floors compose with cross-node eviction.

### 3.5 Trust/security boundary changes (enumerate; do not solve)

Today the boundary is loopback + bearer RBAC, with loopback-no-creds opening everything
(§1.3) and a documented non-loopback plaintext warn-only posture (ADR 004). A mesh changes,
at minimum:
- Inter-node traffic leaves loopback → ADR 004's plaintext posture and ADR 024's
  co-resident threat model both need re-examination (now there is on-wire data and remote
  peers).
- Node-to-node *mutual* authentication (not just client→daemon bearer) — a new principal
  type (peer/node identity) the auth model has no slot for.
- The ADR 024 in-memory data-protection story assumed one trusted process; a peer that can
  request leases or receive activations is a new actor in that threat model.
- Audit/usage are per-principal and per-host today (§1.3); cross-node attribution is
  undefined.
This note enumerates these; it does not resolve them.

### 3.6 Heterogeneous nodes

Apple-Silicon clusters are naturally asymmetric (e.g. an M3 Ultra 512 GB beside an
M-series laptop). Implications: budget is per-node and unequal (the 0.75×RAM default
already differs per machine, §1.1); pipeline parallelism is bottlenecked by the slowest
rank and the smallest memory; placement must be capability-aware (which node can even hold
which slot). The current single-number budget has no notion of "this budget is one of
several unequal ones." Memory asymmetry interacts with §3.2: pipeline stage sizing and
expert placement both want to weight by per-node capacity.

---

## 4. Relevant prior art — multi-node

This is the most important external context for Area 1: **the Apple-Silicon multi-node
substrate now exists below Athena**, which sharpens the framing (Athena would *consume* it,
not invent it).

- **MLX distributed.** `mlx.core.distributed` provides collective primitives
  (`all_sum`, etc.) over Ethernet or Thunderbolt, with a **ring backend** doing pipeline
  parallelism (each node holds a slice of layers; activations flow rank0→rankN). This is the
  capacity-for-big-models path (§3.2) and it is MLX-native, i.e. it would integrate at the
  *substrate* level, below Athena's `InferenceModule` seam.
  ([MLX distributed docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html))
- **JACCL + RDMA over Thunderbolt (shipped macOS Tahoe 26.2 / WWDC 2026).** Apple's
  open-source collective communication library ("Jack and Angelos' CCL") runs MLX collectives
  over RDMA on Thunderbolt 5: ~50–60 Gbps, sub-50 µs (single-digit µs RDMA) latency, ~3×
  inference speedup on four Macs; demoed running a 1T-param model across four M3 Ultras at
  28+ tok/s. **Limitation: fully-connected topology only** — a Thunderbolt cable between
  every pair of nodes, no switch support yet, which caps practical cluster size and dictates
  physical layout. JACCL is general (not ML-only), so a lease/coordination channel could in
  principle ride it.
  ([Apple TN3205: RDMA over Thunderbolt](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt),
  [byteiota: MLX+JACCL](https://byteiota.com/mlx-jaccl-thunderbolt-distributed-training/),
  [WWDC 2026 session 233](https://developer.apple.com/videos/play/wwdc2026/233/))
- **Topology constraint → coordination scale.** The fully-connected-Thunderbolt limit means
  realistic clusters are small (≤4, practically ≤8) directly-cabled Macs. That bears on
  §3.3: consensus machinery sized for hundreds of nodes is likely overkill; the leader-election
  literature (Raft/Paxos, lease-based leadership, external coordinators like
  etcd/ZooKeeper-style) is the menu, but the deployment reality is closer to "a few trusted
  boxes in one room" than a datacenter.
- **Adjacent systems for contrast (not Apple-native):** `exo`, `llama.cpp` RPC, and Ray-style
  placement show the replica/router and pipeline-shard patterns (§3.2) in other ecosystems;
  useful as design references, not as drop-ins (they don't speak the Metal governor's budget).

**Assumption flagged:** the WWDC/byteiota performance numbers are vendor/secondary-source
figures, not independently reproduced here; treat them as order-of-magnitude.

---

## 5. Cross-cutting: are multi-node and embedded the same problem?

**The case for "same problem, different transport scope":** Both are "the governor's
admission authority must cross a boundary it doesn't cross today." In both, a *requestor*
(remote node / local client) asks an *authority* (the governor) for a slice of one budgeted
resource, and the authority must lease it crash-safely with floors and reclamation (§3.4).
If you squint, embedded is the **degenerate single-host case** of the multi-node lease
protocol where the "network" is loopback/IPC and there is exactly one budget. A lease/agent
protocol designed once could, in principle, serve both — the embedded client and the remote
node agent are both "thin requestors of a remote governor." The thin HTTP client
(`AthenaClient`, §1.5) is already the embryonic form of that requestor.

**The case for "genuinely distinct":**
- **What is scarce differs.** Embedded contends for **one physical Metal pool** that
  already has a truthful (if imperfect, ADR 023) owner — the problem is *sharing one
  governor*, not reconciling many. Multi-node contends for **N pools** — the problem is
  *federating many governors*, which is the harder consistency problem (§3.1) embedded does
  not have.
- **The interesting parallelism lives in different layers.** Embedded never needs pipeline/
  tensor/expert parallelism — there is one GPU. Multi-node's value *is* that parallelism,
  and it lives *below* Athena's seam in MLX/JACCL (§3.2, §4), where Athena is a consumer not
  an author. Embedded's value lives *above* the seam (process/client coordination), which
  Athena would own end-to-end.
- **Failure domains differ.** Embedded failover is "the daemon restarts" (launchd already
  owns it, §1.3) — no election. Multi-node introduces arbiter election, partitions, and
  remote-crash reclamation (§3.3) that have no embedded analog.
- **Trust differs.** Embedded co-resident clients are governed by ADR 024's existing
  same-host threat model; multi-node adds on-wire data and remote peer identity (§3.5) —
  a strictly larger boundary.

**Synthesis (no verdict given).** The *lease/admission protocol* may be a shared
abstraction (one requestor↔authority contract, transport-agnostic per §3.4). The
*topology, parallelism, consistency, and failover* around it are not shared: embedded is
"many requestors, one authority, one pool"; multi-node is "federate many authorities and
their pools, and exploit cross-pool parallelism that lives below us." Whether to build one
protocol with two transports or two designs is itself an open decision — explicitly left
unresolved.

---

## 6. Decision space — Area 2: Embedded

### 6.1 Coordination model for co-resident products

The binding constraint is ADR 011: two uncoordinated MLX allocators on one Metal pool
defeat the governor, so N products may **not** each embed their own engine. Options:

- **(A) Thin client → shared singleton governor (out-of-process).** Products link a small
  client (today's `AthenaClient`, §1.5) and speak to one daemon that owns the only MLX
  allocator. *Pros:* preserves the single-governor invariant exactly; already mostly built
  (HTTP loopback); the daemon is the one truthful budget owner. *Cons:* HTTP/loopback hop
  per request (latency, serialization of `MLXArray` results to JSON/bytes); needs the
  daemon's lifecycle to be owned by *someone* (§6.2); no in-process zero-copy path.
- **(B) In-process shared governor library.** Products link `AthenaCore` + the modules and
  share one `MemoryGovernor` instance. *Pros:* zero-copy, no transport. *Cons:* the modules
  are private targets with process-global MLX state (§1.4); "shared instance across
  independently-shipped products in one process" only works if they are *one* process —
  which independent products are not. Effectively impossible without merging the products.
- **(C) Out-of-process engine, fast local IPC (not HTTP).** Like (A) but over a unix
  socket/XPC/shared-memory ring instead of HTTP, to cut the latency/serialization cost.
  *Pros:* keeps single-governor invariant, lowers per-call cost. *Cons:* a transport that
  does not exist today (§1.3 — HTTP-loopback is the only surface); large-tensor results
  still need a copy unless shared memory is used (and the passive-oracle/ADR-024 surfaces
  would need re-examination for a shared-memory channel).

### 6.2 Governor-as-shared-service lifecycle

Today the daemon's lifecycle is owned by launchd (system) or a pidfile-tracked fork (user)
(§1.3). For a shared singleton that *outlives any single client*, the questions are: who
spawns it (first client lazily? launchd always-on? an installer?), who owns restart, and
how it survives all clients exiting. **Platform mechanisms available:** `launchd`
on-demand/`KeepAlive` (already used), launchd socket-activation (not used today), or a
client-spawns-and-detaches pattern (already used for the user daemon fork). None of these
is wired for "many independent clients, one survivor service" — but launchd is the obvious
substrate and the install path already places a LaunchDaemon plist.

### 6.3 Versioned client/governor wire protocol & version skew

Today **the daemon version is the protocol version** (single `Athena.appVersion`, no `/v2`,
§1.5). Independently-shipped embedders break this assumption: product X may ship against
daemon 0.10.x while product Y expects 0.11.x, against one running daemon. Open questions:
introduce an explicit wire-protocol version (decoupled from `appVersion`) and a negotiation
handshake; define a compatibility window and deprecation policy; decide whether `/openapi.json`
(self-describing, already always-reachable) is the negotiation mechanism. The OpenAPI surface
is an asset here — a client can discover capabilities — but there is no versioned-contract
discipline beyond "read the spec."

### 6.4 Multi-tenant isolation across mutually-untrusted co-resident clients

The governor has **no per-tenant budget cap today** — all 5 slots share one number with LRU
eviction (§1.1), and the LLM slot is a hard floor for *everyone*, not per-client. For
mutually-untrusted embedders this is insufficient: a noisy client can evict another's
working set or exhaust the budget. Needed concepts (none exist): per-client ceilings,
per-client floors/reservations, and fairness. Observability is also a leak surface — through
the *shared resource* a client can infer another's activity (eviction timing, latency,
`/healthz` global numbers, cache hit/miss). ADR 024's co-resident threat model is the right
home for that analysis but currently assumes the daemon's *own* memory, not cross-client
inference via shared-governor side channels. RBAC (§1.3) authenticates clients but does not
*isolate resource consumption* between them.

### 6.5 Dev/prod multi-daemon & zero-downtime deploy

The codebase already supports multiple daemons by binding different ports (config, §1.3) —
a dev daemon and a prod daemon coexist trivially because each owns its own process and
budget. That is *not* the embedded shared-singleton case (§6.1) — it is the opposite (full
isolation). It is relevant to **zero-downtime deploy**: today there is graceful SIGTERM
drain ([AthenaServer.swift:672](../Sources/athena/Server/AthenaServer.swift#L672)) and
block-until-ready cold-load (ADR 015), but no handoff between an old and new daemon
instance. A blue/green model (start new daemon on a second port, drain old, flip clients)
is possible with today's primitives *only if* clients can be repointed — which the thin
client's configurable host/port (§1.5) allows, but nothing orchestrates. Load-time admission
(ADR 015) and graceful drain (ADR 033/M33) are the building blocks; the orchestration is
absent.

### 6.6 Prior art — embedded shared-resource coordination

- **launchd as the singleton/lifecycle owner** (on-demand, KeepAlive, socket activation) —
  the canonical macOS mechanism for "one service many local clients depend on"; Athena
  already uses a subset.
- **GPU/accelerator arbitration daemons:** the general pattern of "one privileged process
  owns the device, clients lease via IPC" (analogous to display servers, audio servers like
  coreaudiod, or NVIDIA MPS for CUDA) — direct conceptual prior art for §6.1(A/C).
- **Versioned local IPC with negotiation:** XPC service versioning, gRPC/Cap'n-Proto schema
  evolution, and OpenAPI-driven client generation — references for §6.3.
- **In-process vs out-of-process plugin isolation:** the broad "host one untrusted tenant
  per process vs many in one" literature informs §6.4 (the safe answer is usually process
  isolation, which points back to out-of-process §6.1(A/C)).

---

## 7. Open questions deliberately left unresolved

Each is tagged with what further investigation it needs.

1. **Where does multi-node value actually come from for this project's users?**
   Capacity-for-big-models (pipeline/expert, substrate-owned) vs. capacity-for-concurrency
   (replicas, governor-owned)? — *Needs:* operator intent interview; the two answers lead to
   entirely different designs (§3.2).
2. **Is a federated *budget* even wanted, or just placement/routing?** §3.1(C) vs (A/B). —
   *Needs:* decision on whether "one model bigger than one Mac" is in scope; if not, most of
   §3.1–3.4 collapses.
3. **Does the lease protocol ride HTTP, JACCL, or a new channel, and what consistency model?**
   — *Needs:* a latency budget (admission is on the cold-load path, ADR 015) and a JACCL
   integration spike to know if it can carry non-tensor control traffic.
4. **One protocol two transports, or two designs?** (§5) — *Needs:* a concrete lease-protocol
   sketch tested against both the embedded (loopback, one pool) and multi-node (N pools,
   partition) cases to see if one abstraction survives both.
5. **Who owns the shared-singleton lifecycle and how does it survive client churn?** (§6.2)
   — *Needs:* a launchd socket-activation spike; a decision on lazy-spawn vs always-on.
6. **What isolation primitives does multi-tenant embedding require** (per-client ceilings/
   floors/fairness) and **where do they live** — governor core (MLX-free, unit-pinnable per
   ADR 008/009) vs server edge? (§6.4) — *Needs:* a threat-model extension of ADR 024 to
   cover cross-client side channels via the shared governor.
7. **Decouple wire-protocol version from `appVersion`?** (§6.3) — *Needs:* a compatibility
   policy decision; cheap to start (add a version field) but expensive to retrofit later.
8. **How do the passive-oracle (no outbound), ADR 004 (plaintext posture), and ADR 024
   (in-memory protection) invariants restate themselves once there are peers and on-wire
   data?** (§3.5) — *Needs:* explicit ADR amendments before any mesh code; this is a gate,
   not an afterthought.

---

## 8. Risks / unknowns

- **Thesis collision.** Both areas press directly on ADR 011 ("never compose at the
  inference layer") and the passive-oracle rule. Multi-node *activations on the wire* and
  embedded *shared-memory tensors* each risk becoming "a second uncoordinated path." Any
  design must show it *extends* the single-governor invariant, not forks it — otherwise it
  is a defect by the repo's own rules. **High risk, design-level.**
- **Truthfulness debt compounds.** ADR 023 shows local accounting is already imperfect
  (`phys_footprint` vs estimates vs untagged MLX cache). A distributed arbiter trusting
  remote probes inherits and multiplies this; embedded per-client attribution faces the same
  untagged-cache wall. **Medium-high.**
- **Substrate ownership of the hard part.** The capacity win (pipeline/tensor/expert) is
  MLX/JACCL's, not Athena's (§4). If that is the goal, Athena's role shrinks to a
  control/governor plane over MLX-distributed — a different product shape than today's
  monolith, and dependent on substrate APIs that are new (macOS 26.2) and topology-limited
  (fully-connected Thunderbolt). **Unknown: how stable/extensible those APIs are for a
  long-lived daemon.**
- **Latency on the admission path.** Admission is a synchronous actor call today; any
  cross-boundary lease adds a round-trip to cold-load (ADR 015's budget). Embedded
  HTTP-loopback adds serialization cost to every inference result. **Medium; measurable —
  needs a spike, not speculation.**
- **Security surface growth.** Peer identity, on-wire encryption, cross-client isolation,
  and shared-resource side channels are all *new* boundaries (§3.5, §6.4) on a daemon whose
  entire current posture assumes loopback + one trusted process. **High; gated on ADR
  amendments.**
- **Information genuinely missing (not filled here):** real operator cluster size/intent;
  whether JACCL can carry control/lease traffic; measured loopback-IPC overhead for
  large-tensor results; whether MLX-distributed's ring backend is reachable from Swift at
  Athena's integration point (the substrate work in `AthenaModels` is Swift-MLX, but the
  distributed primitives' Swift surface was not verified in this note). Each is a spike, and
  is named as such above rather than assumed.

---

## 9. Assumptions made explicit

- The 5-slot model, single-number budget, HTTP-only IPC, and "daemon version = protocol
  version" facts are read from current code (§1) and assumed stable as of v0.10.208.
- External performance/availability figures for MLX-distributed/JACCL (§4) are from Apple
  WWDC material and secondary sources, **not** reproduced here.
- "Embedded" is interpreted as *multiple independent products on one host sharing one Metal
  pool* (per the task), not "Athena running on an embedded/edge device."
- This note assumes the binding ADRs (011/013/024/025/026) remain in force; any option that
  contradicts them is flagged as requiring an amendment, never silently assumed away.
