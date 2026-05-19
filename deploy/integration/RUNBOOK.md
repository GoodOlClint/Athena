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
| D8 | (opt-in) set `syslog_remote`, restart, watch collector | logs only egress; nothing else leaves | ☐ (P1) |

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
| P1 result | |
| Notes / filed issues | |

> Release rule: a version tag is **not** shipped until every **P0**
> row (A, B, C, D1–D6, E, F4) is PASS on the two boxes. Archive this
> completed file to `results/<tag>-<date>.md`. The automated stub
> gate stays the per-PR gate; this is the pre-release gate.
