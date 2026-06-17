# Athena — Manual Integration Runbook (two-node, real model)

This is the **human-driven, host-bound** test tier. It covers what the
automated stub gate (`deploy/e2e-rbac.sh`, 208/208) and the unit suite
**cannot**: real MLX inference, real governor memory/OOM, the genuine
off-box remote path, and the zero-JS WebUI console in a real browser.

It is **not** CI. Setup/teardown is scripted; the steps below are a
fixed checklist a human performs and records.

## Topology

| Node | Role | Prep |
|---|---|---|
| **Mac Studio** | Real-model daemon host (Metal, external SSD store, `--engine mlx`) | `deploy/integration/studio-setup.sh` |
| **MacBook** | Second node: portable `ath` client + a real browser | `source deploy/integration/macbook-env.sh <studio> 7447` |

## Prerequisites

- Studio: `athena` built (xcodebuild — MLX needs full Xcode), the
  external SSD mounted, a **pinned small real model** chosen.
- `TEST_MODEL` + `MODEL_STORE` (SSD path) exported on the Studio.
- A trusted LAN/VPN between the boxes. **No TLS:** the bearer token
  and the browser session cookie cross the LAN in plaintext — never
  run this over untrusted Wi-Fi; front with a TLS proxy for anything
  beyond a lab.

## Bring-up

**On the Studio:**
```
TEST_MODEL=mlx-community/<small-model> \
MODEL_STORE=/Volumes/<SSD>/athena/models \
deploy/integration/studio-setup.sh
```
Copy the printed **admin bearer token** (shown once) and the LAN IP.

For scenario **G** (TurboQuant) or **H** (TriAttention) you can bring
the host up already in the compressed posture by adding
`KV_COMPRESSION=turboquant` (or `=triattention`) to that invocation;
the G/H steps instead toggle it live via `athena config set` / the
`ATHENA_KV_COMPRESSION` env, each followed by a daemon restart
(`kv_compression` is resolved once at start).

**On the MacBook:**
```
source deploy/integration/macbook-env.sh <studio-lan-ip> 7447
# paste the token when prompted
ath_ping        # expect: healthz JSON + governor snapshot

# For the raw-curl rows (A3–A7, B-notes, E) export these on whichever
# box runs curl (the MacBook is fine):
export B="http://<studio-lan-ip>:7447"
export TOK="<admin-bearer-token>"
```

Record the build/tag (`ath --version`), model, date, tester in the
**Sign-off** block at the bottom. Tick each row PASS/FAIL with notes.
`results/` holds dated copies of completed runbooks (release trail).

Priority: **P0** = release-blocking, **P1** = should-pass.

---

## A — Single-node real model  (Studio)  · P0

| # | Action | Expected | Result |
|---|---|---|---|
| A1 | `athena show "$TEST_MODEL" --model-store "$MODEL_STORE"` | model present, health ok | ☐ |
| A2 | MacBook: `ath run "$TEST_MODEL" "say hello in one word"` | coherent real completion | ☐ |
| A3 | `curl -s -H "Authorization: Bearer $TOK" $B/v1/chat/completions -d '{"model":"…","messages":[{"role":"user","content":"hi"}]}'` | OpenAI-shaped, real text | ☐ |
| A4 | same for `POST $B/api/chat` | Athena-native `{content,…}`, real text | ☐ |
| A5 | `POST $B/v1/embeddings` `{"input":"hi"}` | vector, correct dim | ☐ |
| A6 | `POST $B/v1/audio/transcriptions` (fixed WAV, multipart) | sane transcript | ☐ |
| A7 | `POST $B/v1/audio/diarizations` (fixed WAV) | speaker turns | ☐ |
| A8 | `ath ps` / `curl $B/healthz` after A2–A7 | governor shows **real** resident bytes for `.llm` | ☐ |

## B — Two-node remote client  (MacBook → Studio)  · P0

`ath` is the portable client off-box (`isRemote` true ⇒ every verb
drives the remote `/api` over HTTP).

| # | Action | Expected | Result |
|---|---|---|---|
| B1 | `ath status` | remote posture (model, listen, auth on) | ☐ |
| B2 | `ath list` ; `ath show "$TEST_MODEL"` | store listing / detail over the wire | ☐ |
| B3 | `ath pull <small-hf-id> --follow` | 202 → live SSE progress → done | ☐ |
| B4 | `ath convert <hf-id> --follow` | job runs, SSE progress | ☐ |
| B5 | `ath cp "$TEST_MODEL" itest-tmp` ; `ath default itest-tmp` ; `ath rm itest-tmp` | copy/default/rm reflected in `ath list` | ☐ |
| B6 | `ath queue ls` ; `ath queue get <id>` | jobs visible to owner | ☐ |
| B7 | Make a **member** token (`ath auth token add --user <member>`), retry B3 with `ATHENA_KEY=<member-tok>` | **403** (member ∌ model.write) | ☐ |
| B8 | `ath vectors …` upsert/query ; `ath store stats` | vector + store ops over HTTP | ☐ |

## C — WebUI console in a real browser  (MacBook → Studio)  · P0

Open `ath_web` (`http://<studio>:7447/ui`) in Safari/Chrome.

| # | Action | Expected | Result |
|---|---|---|---|
| C1 | Visit `/ui` unauthenticated | redirected to `/ui/login` (303) | ☐ |
| C2 | Sign in as admin | dashboard renders; nav shows dashboard/models/daemon/users/config | ☐ |
| C3 | Watch the dashboard ~10 s | live poll updates (governor/metrics/queue) | ☐ |
| C4 | Models → Pull a small HF id | **live progress** (SSE), then model appears | ☐ |
| C5 | Models → Delete a copy | JS confirm() prompt; deletes only on confirm | ☐ |
| C6 | Daemon → Warm, then Unload | confirm() on Unload; posture panel updates | ☐ |
| C7 | Users → create user, grant then revoke a role, delete user | each reflected in the table | ☐ |
| C8 | Users → mint a token | secret shown **once**, in a copyable field | ☐ |
| C9 | DevTools: strip/alter the `X-CSRF-Token` on any mutation, replay | **403** (CSRF) | ☐ |
| C10 | Log out, then hit `/ui` Back | session invalid → `/ui/login` | ☐ |
| C11 | (If a non-admin account exists) sign in as it | `/ui*` → 303 (daemonAdmin entry gate) | ☐ |

## D — Lifecycle / ops  (Studio)  · P0 / P1

| # | Action | Expected | Result |
|---|---|---|---|
| D1 | `sudo athena install` then `athena start` (launchd path) | daemon runs as a launchd service | ☐ |
| D2 | reboot the Studio | daemon auto-starts at boot | ☐ |
| D3 | `athena status` ; `athena doctor` | running; doctor posture incl. auth check ok | ☐ |
| D4 | `athena logs --source err -f` during a request | request logged; no secrets in logs | ☐ |
| D5 | Console → Config: change a value, restart, re-check | change applied after restart | ☐ |
| D6 | Lockout drill: `athena auth user passwd admin` (offline) | password reset without a token | ☐ |
| D7 | `athena uninstall` (no `--purge`) | launchd removed; data/config kept | ☐ (P1) |
| D8 | `log show --last 5m --predicate 'subsystem == "athena"'` after activity | recent entries present; M45.1 unified-log is the sole diagnostic surface | ☐ (P1) |

## E — Security negatives (by hand)  · P0

| # | Action | Expected | Result |
|---|---|---|---|
| E1 | `ath status` with a wrong/empty token | 401/403, no daemon info | ☐ |
| E2 | Restart the daemon, reuse an old browser cookie | session invalid (per-process secret) | ☐ |
| E3 | Two users submit queue jobs; each `ath queue get` the other's id | non-owner → **404** (not 403) | ☐ |
| E4 | Stop daemon; set `listen_host` non-loopback, remove all creds; start | **refuses to start** (fail-safe) | ☐ |
| E5 | `chmod 0644` the auth keys file (if used); restart | startup **warns** group/other-readable | ☐ (P1) |

## F — Resilience / chaos  · P1

| # | Action | Expected | Result |
|---|---|---|---|
| F1 | `kill` `athenad` mid stream (during A3) | client errors cleanly; no corruption | ☐ |
| F2 | `kill` mid queue job; restart | job state/owner intact on restart | ☐ |
| F3 | Fill the store volume; `ath store export` | graceful error, daemon survives | ☐ |
| F4 | Governor stress: 1 model + concurrent requests until budget hit | **503 `metal_oom`** (not a crash); eviction/reconcile/prompt-cache-cap visible in `/healthz` | ☐ |

## G — KV-cache compression / TurboQuant (Studio + MacBook) · P0*

Exercises the `kv_compression` knob (M20) on the **real model** — the
codec changes KV-cache numerics under genuine MLX inference and the
governor's prompt-cache accounting, which the stub gate and the
single-node automated `TurboQuantE2ETests` cannot cover off-box.
TurboQuant is **off by default**; bring the host up in the TurboQuant
posture with `KV_COMPRESSION=turboquant … studio-setup.sh`, or per G1
toggle it live. `kv_compression` is resolved **once at daemon start**,
so every change needs a daemon restart. `*`P0 **only for a release
that touches the kv_compression / KV-codec path** — otherwise N/A
(opt-in feature, default off; do not block unrelated releases).

| # | Action | Expected | Result |
|---|---|---|---|
| G1 | Studio: `athena config set kv_compression turboquant` → `athena stop` → `athena start …` ; `curl $B/healthz` | daemon healthy with TurboQuant active | ☐ |
| G2 | MacBook: `ath run "$TEST_MODEL" "name three primary colors"` (daemon from G1) | coherent real completion (no degenerate single-token/char loops) | ☐ |
| G3 | Studio: keep TOML `kv_compression = none`; restart daemon with `ATHENA_KV_COMPRESSION=turboquant` in its env ; check `athena logs --source start` | env wins over TOML — TurboQuant active (precedence env > TOML) | ☐ |
| G4 | Studio: `athena config set kv_compression bogus` (also try an unknown like `snapkv`) → restart | daemon **refuses to start**; clear "unrecognized kv_compression" error; **no silent fallback to none** (fail-closed). `triattention` is now a valid value — see scenario H | ☐ |
| G5 | Restore `kv_compression = none`, restart; `ath run "$TEST_MODEL" "<fixed long-ish prompt>"`, note output; repeat with `turboquant` (G1) | both coherent; outputs need **not** match (numerics differ by design) — neither degenerates | ☐ (P1) |
| G6 | At a low `--prompt-cache-cap-bytes`, send a context that **503s `prompt_cache_cap_exceeded`** under `none`; restart with `turboquant`, resend | same context now **admitted** (per-token KV 256→64 KiB accounting); cap still enforced for an even larger context | ☐ (P1) |

## H — KV-cache compression / TriAttention (Studio + MacBook) · P0*

Exercises the `kv_compression = triattention` value (M21) on the
**real model**. TriAttention is **token EVICTION**, not quantization:
it drops low-`‖k‖` decode tokens once the cache exceeds its budget
(norm-only mode; calibrated trig scoring is a deferred follow-up).
Off by default; bring the host up with
`KV_COMPRESSION=triattention … studio-setup.sh`, or per H1 toggle it
live. Resolved **once at daemon start** — every change needs a
restart. Unlike TurboQuant, eviction does **not** lower the
per-token KV figure (the prefill is pinned full-precision), so the
prompt-cache-cap accounting is unchanged from `none`. `*`P0 **only
for a release that touches the kv_compression / KV path** — else N/A.

| # | Action | Expected | Result |
|---|---|---|---|
| H1 | Studio: `athena config set kv_compression triattention` → `athena stop` → `athena start …` ; `curl $B/healthz` | daemon healthy, starts cleanly (triattention is a valid value) | ☐ |
| H2 | MacBook: `ath run "$TEST_MODEL" "name three primary colors"` (daemon from H1) | coherent real completion (no degenerate loops) — the evicting cache is a correct drop-in even when the budget is not hit | ☐ |
| H3 | Studio: TOML `kv_compression = none`; restart with `ATHENA_KV_COMPRESSION=triattention` in env; `athena logs --source start` | env wins over TOML — TriAttention active (precedence env > TOML) | ☐ |
| H4 | MacBook: with speculative/MTP enabled on the daemon AND `kv_compression=triattention` (H1), run a fixed greedy (temp 0) prompt; repeat with `kv_compression=none` + speculative | **identical** output both runs — eviction is inert on the MTP/speculative path (it cannot un-mix GDN/Mamba recurrent state); bit-identical greedy preserved | ☐ |
| H5 | MacBook: send a **long** context/generation (enough decode tokens to exceed the eviction budget) under `triattention`; watch `/healthz` governor memory across the run; repeat under `none` | both coherent; under `triattention` resident KV stays **bounded** (eviction caps growth) while `none` grows unbounded — neither degenerates; outputs need **not** match | ☐ (P1) |

## I — Multi-architecture support (Studio + MacBook) · P0*

Exercises a **non-Qwen** architecture (M23) on the real model path —
Llama / Gemma / Mistral / Phi / … load through the substrate factory,
not the vendored Qwen3.5 model. The single-node automated
`MultiArchE2ETests` covers load + structured output; this checks the
off-box client + the daemon warnings. Bring the host up with a non-Qwen
`TEST_MODEL` (e.g. `mlx-community/Llama-3.2-1B-Instruct-4bit`). `*`P0
**only for a release that touches the model-load / structured-output /
kv_compression path** — otherwise N/A.

| # | Action | Expected | Result |
|---|---|---|---|
| I1 | Studio: `athena show "$TEST_MODEL"` (non-Qwen) | `type:` matches (e.g. `llama`); support line = "validated substrate arch: guided structured output + TurboQuant (no MTP / TriAttention)" (M23 fork D) | ☐ |
| I2 | MacBook: `ath run "$TEST_MODEL" "name three primary colors"` | coherent real completion from the substrate stream | ☐ |
| I3 | `curl $B/v1/chat/completions` with `response_format` json_schema (integer field) | **schema-valid JSON** — the substrate-path guided decoder enforces it; before M23 this was unconstrained prose (M23 fork A) | ☐ |
| I4 | `curl $B/v1/chat/completions` with `tools` + a forced `tool_choice` | a `tool_calls` response with valid JSON arguments (guided on the substrate path) | ☐ |
| I5 | Studio: start with `kv_compression=triattention` on the non-Qwen model; `athena logs --source start` | startup **warns** "inert for model type … running uncompressed"; generation still works (M23 fork B) | ☐ |
| I6 | Studio: restart with `kv_compression=turboquant` on the non-Qwen model | **no** inert warning; coherent output (TurboQuant applies to any arch) | ☐ |
| I7 | (control) point `TEST_MODEL` at the Qwen3.5 model; redo I2–I3 | unchanged — vendored path, MTP/structured intact (no regression) | ☐ (P1) |

---

## J — Vision input / VLM chat (Studio + MacBook) · P0*

Exercises **image input** (M71) on the real model path — a vision
checkpoint (`vision_config` present, e.g. a `gemma-4-*-it` unified model)
loads through the substrate **VLMModelFactory** (`MLXVLM.Gemma4`), not the
text `LLMModelFactory`. The automated stub gate (phase 2.3) covers only the
wire protocol + passive-oracle 400s; the stub is non-vision, so the actual
image→text generation can only be checked here. Bring the host up with a
vision `TEST_MODEL`. `*`P0 **only for a release that touches the
vision/VLM load or chat-image path** — otherwise N/A.

`IMG` below = a small base64 data: URL, e.g.
`IMG="data:image/png;base64,$(base64 -i some.png | tr -d '\n')"`.

| # | Action | Expected | Result |
|---|---|---|---|
| J1 | Studio: `athena show "$TEST_MODEL"` (vision model) | loads; `type:` is the gemma4 family; daemon comes up with the model resident | ☐ |
| J2 | `curl $B/v1/chat/completions` with `content:[{type:text,text:"describe this image"},{type:image_url,image_url:{url:"$IMG"}}]` | a coherent **description of the image** (the VLM vision path ran) | ☐ |
| J3 | Same as J2 but `url` = `https://…/x.png` | **400** `invalid_image` — passive-oracle, no outbound fetch | ☐ |
| J4 | Plain text chat (no image parts) to the same vision model | normal completion — text path still works through the VLM container | ☐ |
| J5 | (control) point `TEST_MODEL` at a TEXT model; redo J2 | **400** `vision_not_supported` — a text-only model refuses image input | ☐ |
| J6 | Studio: `athena logs` during J2 | governor counts ONE resident copy (no double-load); decode heartbeats fire | ☐ |

> NOTE (M71.2): a vision checkpoint loads via the VLM path, which **disables
> DFlash for that model** (the drafter seam is bound to the text
> `MLXLLM.Gemma4`). Qwen3.5-MTP / structured output are unaffected. Re-wiring
> DFlash onto the VLM's inner text backbone is a deferred follow-on.

> **VALIDATED 2026-06-17 (v0.10.160)** on `mlx-community/gemma-4-e2b-it-4bit`
> (`model_type: gemma4_audio`, full `vision_tower`, 3.58 GB): J2 returned an
> accurate description of a red-square test image ("solid red square … the rest
> white"); J3 remote URL → 400 `invalid_image`; J4 text on the VLM container →
> normal completion; J5 image → a text model (Qwen) → 400 `vision_not_supported`.
>
> **Substrate VLM support is the gemma-4 VLM family only** (`gemma4` /
> `gemma4_audio`, full `vision_tower`). The **omni** family
> (`gemma4_unified_audio`, e.g. `gemma-4-12B-it`) has a *different, minimal*
> vision arch (`vision_embedder` / `embed_vision`, no SigLIP blocks) the
> substrate does **not** implement — it fails to load with `keyNotFound
> vision_tower.patch_embedder.input_proj.weight`. Serving the omni family is a
> separate, larger substrate-port milestone, not M71 wiring.

---

## Teardown

```
# Studio
deploy/integration/studio-setup.sh --teardown            # stop
deploy/integration/studio-setup.sh --teardown --purge    # + creds
# (or, if launchd path D1 was used)  sudo athena uninstall [--purge]
```

## Sign-off

| Field | Value |
|---|---|
| Date | |
| Tester | |
| Build / tag (`ath --version`) | |
| Pinned model | |
| Studio macOS / hardware | |
| P0 result (A–E + F4) | ☐ all PASS / ☐ blocked |
| G result (kv_compression / TurboQuant) | ☐ N/A (release untouched) / ☐ G1–G4 PASS / ☐ blocked |
| H result (kv_compression / TriAttention) | ☐ N/A (release untouched) / ☐ H1–H4 PASS / ☐ blocked |
| I result (multi-architecture) | ☐ N/A (release untouched) / ☐ I1–I6 PASS / ☐ blocked |
| P1 result | |
| Notes / filed issues | |

> Release rule: a version tag is **not** shipped until every **P0**
> row (A, B, C, D1–D6, E, F4) is PASS on the two boxes — **plus
> G1–G4 (TurboQuant) and/or H1–H4 (TriAttention) for any release
> that touches the `kv_compression` / KV path, and I1–I6 (multi-arch)
> for any release that touches the model-load / structured-output path**
> (else G/H/I are N/A).
> Archive this completed file to
> `results/<tag>-<date>.md`. The automated stub gate stays the
> per-PR gate; this is the pre-release gate.
