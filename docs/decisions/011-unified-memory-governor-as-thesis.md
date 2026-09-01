# 011 — The unified-memory governor is Athena's reason to exist; everything else is a tenant or a tax

**Status:** Accepted — positioning / strategy (no code in this ADR)
**Date:** 2026-06-17
**Milestone:** pre-M72 strategy; sets the next-milestone priority (governor accounting truthfulness)

> **Amended (ADR 025, M80):** the **vector store and the async queue were removed** (v0.10.201 / v0.10.203). They are no longer governor tenants — they were always the weakest of the set (CPU bookkeeping, no Metal budget, a client can self-host them), so the multi-tenant story **narrows to audio / embeddings / vision / video**. The Metal-governor thesis is unaffected: those modalities are exactly the ones that contend for the one Metal budget. Read the queue/vectors references below as historical context, not the current tenant set.

> **Amended (2026-07-04, CUDA-port audit):** the thesis is **unified-memory-general, not Metal-specific**. Metal is the Apple instance of unified memory; **CUDA UMA (GB10/DGX Spark, Jetson Thor/Orin) is the same shape**, and the governor ports without bending the thesis because MLX's `CudaAllocator` implements the **identical memory API the governor is built on** (`activeMemory`/`cacheMemory`/`cacheLimit`/`memoryLimit`/`clearCache`) and a GB10-class box is one coherent pool. Read "Metal budget" throughout as "the one unified-memory budget" — "one Metal budget" → "one CUDA budget" on a coherent-pool CUDA box (`docs/cuda-port-audit-2026-07-04.md`). This generalization is **node-local**: heterogeneous **sharding** of a single model across a Metal node and a CUDA node is a separate, harder problem it does **not** imply — that is blocked below the thesis (MLX's backend-homogeneous collectives + the interconnect), see `docs/multi-node-and-embedded-deployment-research.md` §3.7. GH200/GB200's two physical pools do not transfer 1:1 (governor denominator becomes the GPU pool).

## Context

Triggered by a "build vs. switch" question: with LM Studio installed for vision, the operator observed that LM Studio is more polished on the overlap (chat UX, model discovery, **download reliability** — Athena forks `swift-huggingface` just to reach parity here) and asked whether Athena is still worth developing.

Findings that framed the decision:

- **LM Studio is closed-source proprietary freeware** (Element Labs), free for personal and commercial use. The "community developed / more developers" intuition is a *vendor team you cannot direct*, not a contributor commons. Depending on it is procurement of a maintained commodity layer, not collaboration.
- **The two products are different categories.** LM Studio is a single-user desktop **client** that loads a model (plus an OpenAI/Anthropic-compatible server, embeddings, RAG, MLX vision, and a headless mode `llmster`). It has **no audio pipeline and no job queue at all**, and no multi-tenant governance (RBAC/auth/audit/quotas). Athena is a headless multi-modal **backend** (LLM + embeddings + transcription + diarization + speaker embeddings + vector store + queue) under one Metal budget, with multi-tenant hardening. On the narrow overlap LM Studio wins and always will (paid team, that is the whole product); a solo+Claude effort cannot out-polish it there.
- **The async queue is not an independent moat** — a client can implement its own queue. Its only irreplaceable part is the *governed serial worker*, which exists **because** one daemon owns the budget. The queue is a surface on top of the governor, not a pillar.
- **The real moat is the unified Metal memory budget.** Apple Silicon shares one physical pool across CPU/GPU/ANE. N separate processes (LM Studio + a Whisper tool + an embedder) have N independent allocators, none aware of the others → over-provision or OOM/thrash. A single daemon that sees the *real* working set across all modalities can make **global eviction decisions** ("drop the idle 27B so the diarization job fits") that are **structurally impossible** from inside any one app. This is the one categorical (not merely relative) advantage in the whole comparison, and it is the reason Athena must be a long-lived daemon rather than a library or per-invocation tool.

Athena's purpose to the operator is mixed and all consistent with the backend role: a stable compute daemon for a downstream client (coding), the backend a consuming product was adapted onto, and a possible future commercial / supporting product. Real workloads in use include **audio (transcribe/diarize/speaker)** and embeddings — both of which LM Studio cannot serve — so "switch to LM Studio" was never on the table for the workloads that matter; it only ever applied to personal chat-with-vision, a smaller and separable question.

## Decision

**Athena's reason to exist is the unified-memory governor: one daemon that owns the Metal budget and multiplexes heterogeneous accelerated workloads on a fixed-RAM Apple Silicon box so it never over-commits or OOMs. Audio, embeddings, vision, vectors, and the queue are not independent "features" — they are the first tenants proving the governor across modalities. The chat GUI, model-discovery UX, and download-reliability-as-a-feature are undifferentiated *tax*: maintain at parity-enough for the backend role, do not invest to beat a desktop client there.**

Positioning, one line:

> Athena is the unified-memory governor for Apple Silicon. LM Studio is a client that loads a model; Athena is the broker that decides what gets to be resident at all. (LM Studio : Athena :: a SQLite GUI : a multi-tenant Postgres server.)

Coexistence, not competition: LM Studio remains the operator's personal chat-with-vision client; Athena is the governed backend behind the operator's own products (a downstream client and other consumers). **Do not compose at the inference layer** — running LM Studio's process alongside Athena's puts two uncoordinated allocators on one Metal pool with no shared governor, destroying the single most differentiated capability. Inference stays in-process.

Two operative calls follow from making the governor the thesis:

1. **Governor accounting truthfulness is the next milestone — promoted above feature work.** The moat is only as good as the numbers it decides on, and today the accounting is spongy: the heartbeat RSS probe *undercounts* GPU memory (phys_footprint observed at 63 GB while the heartbeat reported 35), Metal-OOM still reaches request time as a 503, and non-decode phases (embedding) are invisible because heartbeats fire only on decode. If global eviction is the crown jewel, trustworthy cross-modality Metal/wired accounting is the load-bearing wall and must come before further tenants. (Builds on the M5/M43 governor-truth and M5.5 RSS-probe work; this milestone closes the cross-modality gap.)

2. **The future GPU-compute extension's governance model is left open.** The governor concept generalizes from MLX inference to arbitrary accelerated workloads (Metal compute, ANE) — the abstraction is a reservation-based allocator over one physical pool. But the governor can only evict what it can *make* free: in-process tenants vs. an external cooperative reserve/use/release protocol is a real fork, and macOS gives no hook to force an uncooperative third-party process to release Metal memory. This ADR records the fork as **explicitly unresolved**; decide it when a concrete second compute workload actually arrives, not speculatively.

## Consequences

- **Freeze investment in the overlap.** Chat GUI, model-discovery polish, and download-UX-as-a-feature are maintained, not improved. Effort spent matching LM Studio there does not pay rent. This is also the bus-factor mitigation: shrink surface, don't add developers.
- **Vision work continues** — it is needed for a consumer-adjacent item and fits the tenant model. Build it to "solid backend capability," not "beat LM Studio's chat UX."
- **Next milestone is governor accounting**, not a new tenant. See call (1).
- **The moat is invisible in single-user desktop use.** Its value scales with `(workload concurrency × modality mix × closeness to the RAM ceiling)` — i.e. it shows up when a downstream client + another consumer hammer one Mac, not in personal chat. Do not expect to *feel* it personally; measure it under concurrent multi-modal load.
- **Tripwire to retire Athena (the honest counter-case):** if audio + multi-tenant + near-ceiling concurrency usage ever go to zero and consumers could live on a plain OpenAI-compatible chat+embeddings endpoint, the governor stops earning its maintenance cost and Athena should be retired for LM Studio + a small embeddings sidecar. Not the situation today; revisit if those usages lapse.
- **Commercial framing** (if pursued): Athena is B2B/infrastructure — the self-hostable governed compute substrate behind the operator's products — not a consumer chat app. It does not compete with free LM Studio; it occupies a position nothing else on Mac holds.
