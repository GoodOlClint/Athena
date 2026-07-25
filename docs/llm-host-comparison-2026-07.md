# LLM host comparison & feature-gap analysis — 2026-07-07

Deep-research run comparing Athena against popular self-hosted LLM serving hosts (Ollama, LM Studio, vLLM, llama.cpp server, SGLang, mlx-lm, MLC-LLM, vllm-mlx), focused on ease-of-use, killer features, and performance techniques worth adopting. 23 sources fetched, 94 claims extracted, top 25 adversarially verified (3 votes each): 24 confirmed, 1 refuted. Athena's ADR non-goals (chat GUI, OpenAI platform tail, CoreML/ANE, vector DB, job queue) were excluded from recommendations by construction.

## Headline

The landscape splits into GPU-datacenter engines (vLLM, SGLang) that win on throughput via PagedAttention/RadixAttention + continuous batching, and Apple-Silicon runtimes (Ollama, LM Studio, llama.cpp, mlx-lm, MLC-LLM, vllm-mlx) that trade throughput for ease-of-use and single-box operation. Athena already matches or exceeds peers on its differentiating axis — the unified Metal memory governor, encrypted disk KV, MTP speculative decoding, and just-landed continuous batching. The real gaps are a small set of KV-cache efficiency techniques now proven on Apple Silicon, plus one API-surface expectation (`/v1/responses`).

**The most important single finding is competitive, not technical: [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx) (EuroMLSys '26, arXiv 2601.19139) is a near-complete blueprint of Athena's roadmap** — continuous batching + OpenAI + Anthropic APIs in one native MLX server, with paged KV cache, trie-based prefix caching, an SSD-tiered cache spill (~1100-line `ssd_cache.py`, SQLite-indexed, LRU), and MCP tool calling. Self-reported 21–87% higher throughput than llama.cpp (Qwen3-0.6B → Nemotron-30B), up to 525 tok/s on M4 Max, 4.3× aggregate at 16 concurrent. What it does NOT have: a unified cross-modality memory governor, audio/diarization/video tenants, RBAC/TLS/audit, or any of the hardening program. The governor thesis (ADR 011) remains the durable moat — but vllm-mlx is the reference implementation to study for paged/prefix/SSD KV.

## Verified findings (all survived 3-vote adversarial verification)

| # | Finding | Confidence | Athena status |
|---|---|---|---|
| 1 | Continuous batching is the highest-value perf lever on MLX: 3.7–4.3× aggregate at 16 concurrent (vllm-mlx); LM Studio shipped it Jan/Feb 2026 (llama.cpp slots v0.4.0, MLX engine v0.4.2) — now table-stakes | High (3-0, merged from 3 claims) | **Already built** (ADR 039, default-off). Matches our ~3× at N=8 spike. |
| 2 | Automatic cross-request prefix caching (vLLM BlockHash chains / SGLang RadixAttention radix-tree LRU + longest-prefix-match scheduling; 75–95% hit rates at 60%+ prefix overlap). vllm-mlx ports it to MLX (trie `prefix_cache.py`) | High (3-0, merged from 4) | **Gap.** Our M59 prompt-prefix cache is per-sequential-request; no automatic reuse across distinct concurrent requests. |
| 3 | Q8 KV-cache quantization halves KV memory vs FP16 for <0.5pp quality loss (M5 Max/MLX benchmark: 70B @64K, 20.8→10.4 GB, MMLU −0.2pp, needle −0.4pp); recommended production default for long context. Q4 costs 8–14pp on needle — pressure-only | Medium (3-0, single blog-tier source for decimals; consensus corroborated) | **Gap.** Substrate support unverified (upstream #230 added `kvScheme: String?`; TurboQuant removed by ADR 028). |
| 4 | Paged KV cache is vLLM's foundational technique (KV waste 60–80% → <4%) and strongest for 64K–128K contexts (MLC-LLM), but on Apple unified memory the benefit appears only at higher concurrency. llama.cpp's own vLLM-style proposal (discussion #21961) is unmerged Phase-1 design | High (3-0, merged from 6) | **Correctly parked.** Our Phase-2 spike measured ~1.0× at N≤8; ceiling real only at N≥32. Peer evidence confirms the park. |
| 5 | vLLM V1 scheduler mixes compute-bound prefill + bandwidth-bound decode in one step (chunked prefill, up to 1.7×) — directly exploits the ADR 038 finding that batch-1 decode leaves the GPU ~95% compute-idle | High (3-0) | **Gap** (future BatchGenerator extension). |
| 6 | `POST /v1/responses` is now an expected OpenAI-compat surface — Codex drives it exclusively; LM Studio implements it | High (3-0) | **Gap.** Fits the ADR 036 adapter seam. |
| 7 | One-command HF pull (`ollama run`, `llama-server -hf <repo>`) is the ease-of-use baseline | High (3-0, merged from 3) | **Met/exceeded** — `athena pull --check` pre-download loadability preflight is ahead of every peer. |
| 8 | vllm-mlx = closest direct competitor (see headline) | High (3-0, merged from 2) | Study as reference impl; governor stays the moat. |
| 9 | Multi-machine distributed inference (mlx-lm `mx.distributed` Ring backend — 27B across 4× M3 Ultra; vLLM tensor/pipeline parallelism) is a parallelism axis Athena doesn't address | High (3-0) | Out of scope for a single-operator LAN appliance; flag only if deployment target shifts. |

Refuted (excluded): "LM Studio model switching requires manual unload/GPU-layer reconfig vs Ollama" — 0-3 against; outdated UX complaint.

## Recommendations (value / effort)

1. **Enable/tune continuous batching when the trigger fires** — high value, low effort (already built). No new work; the ADR 038/039 gate (`gateWaiters ≥ 2` on /metrics) is exactly right. Peer data says the multiplier shrinks with model size (8B-class ~2.6×; our 26B-A4B MoE likely less) — the existing evidence-gated posture is validated, not challenged.
2. **Automatic cross-request prefix caching** — high value, moderate effort. The biggest verified gap. Strongest for Claude Code/agent loops re-sending large stable system prompts — precisely Athena's real workload since ADR 036. Builds on the existing M59 prefix cache + ADR 039 per-sequence KV ledger. Open design question flagged by the research: how shared KV blocks interact with encrypted-idle-KV (ADR 024), disk snapshots (ADR 027), and per-sequence admission accounting (ADR 039).
3. **Q8 KV-cache quantization** — high value, low-moderate effort *if* the substrate supports it. Directly extends the governor thesis: half the KV bytes = more tenants/longer contexts per Metal budget. First step is a substrate check (upstream `kvScheme` seam from #230), not Athena code.
4. **`/v1/responses` adapter** — medium value, moderate effort. Legal under ADR 036 (decode/encode over the same engine, like `/v1/messages`), but per our own precedent (Anthropic adapter justified by real Claude Code) it should wait for a **named driver** — i.e. build it when someone actually points Codex (or another Responses-only client) at Athena. Do not let it pull in the platform tail (assistants/batches/files remain non-goals).
5. **Chunked-prefill mixing in the batch scheduler** — medium value, moderate effort. Natural follow-on when the BatchGenerator is next extended; not before multi-request contention is real.
6. **Paged KV: keep parked** — the research independently corroborates our Phase-2 park (mlx-tracker #13). Re-open only at N≥32/multi-tenant, per the existing tripwire.
7. **No action: pull UX, distributed inference** — pull UX already meets/exceeds the bar; multi-box contradicts the single-Metal-budget thesis.

## Caveats

- vllm-mlx throughput figures are self-reported preprint benchmarks by the framework's own authors, biggest multipliers on the smallest models; gains shrink toward Athena's actual 26B-class serving size.
- Q8 KV decimal deltas are single-source blog-tier; the <0.5pp-for-Q8 consensus is multi-source.
- Fast-moving snapshot: LM Studio's MLX continuous batching is ~5 months old, vllm-mlx is weeks-to-months old, llama.cpp paged-KV is unmerged design — re-check within a quarter.
- 3 of 75 verifier votes ran without the safety-classifier review pass (infra note, not a content issue); their verdicts agreed with the other voters on the same claims.

## Open questions from the run

1. Does the vendored MLX-Swift substrate support KV-cache quantization natively (post-#230 `kvScheme`), or does Q8-KV need an upstream port?
2. What is the real batching multiplier for gemma-4-26B-A4B MoE at LAN-realistic N=2–8 (vs the small-model 3.7–4.3× figures)?
3. Is there a named consumer for `/v1/responses`, or is it speculative?
4. Does cross-request KV sharing break the ADR 039 per-sequence admission ledger / ADR 024 encrypted-idle-KV design?

## Sources (23 fetched; key ones)

- Primary: [vllm-mlx repo](https://github.com/waybarrios/vllm-mlx) · [arXiv 2601.19139 (vllm-mlx paper)](https://arxiv.org/html/2601.19139v1) · [vLLM: Anatomy of vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) · [arXiv 2511.05502 (Apple-Silicon runtime benchmark)](https://arxiv.org/pdf/2511.05502) · [mlx-lm](https://github.com/ml-explore/mlx-lm) · [LM Studio OpenAI-compat docs](https://lmstudio.ai/docs/developer/openai-compat)
- Forum/primary-adjacent: [llama.cpp paged-KV Phase-1 discussion #21961](https://github.com/ggml-org/llama.cpp/discussions/21961)
- Blog-tier (claims traced to primary literature: vLLM SOSP'23 arXiv 2309.06180, SGLang arXiv 2312.07104): TensorFoundry, particula.tech, codersera, glukhov.org, contracollective (Q8 KV), Medium/hannecke, itsfoss, makeuseof, hybrid-llm, corsair.

Full run artifacts: workflow `wf_c8d8254a-2d9` (106 agents, 0 errors, ~14 min); journal at the session transcript dir.
