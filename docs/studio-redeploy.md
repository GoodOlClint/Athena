# Redeploying / updating the Mac Studio Athena appliance

How to push a new Athena build to the headless **Mac Studio** inference appliance
(`ssh studio` → 172.16.100.118, M4 Max 48 GiB). Distilled from the 2026-06-30
deploys (initial install + the v0.10.230 streaming-tool-call update).

## Why it's not just `git pull && build` on the studio

The studio has **Command-Line-Tools only, no Xcode** → it **cannot build MLX**
(the Metal shaders need full `xcodebuild`). So you **build on a dev Mac with full
Xcode and ship the pre-built binary + its resource bundles**. There is no
`athena update` command yet; **`athena install` is the upgrade path** (it has an
M38 version guard that classifies fresh / reinstall / upgrade / downgrade).

The deployable unit is the binary **plus all 6 `*.bundle` resources** next to it
(notably `mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` — MLX loads
it relative to the executable; and `athena_AthenaCore.bundle` holds the ADR-032
drafter map). `athena install` copies all of them into `/usr/local/libexec/athena`.

## Update procedure (existing install → new build)

On the **dev Mac** (full Xcode):

```sh
cd ~/Source/Athena
./deploy/build.sh Release                 # → .build/xcode/.../Release/{athena, *.bundle}
PD=.build/xcode/Build/Products/Release

# Ship the binary + ALL bundles. Use BASIC rsync flags — macOS ships old
# rsync 2.6.9, which rejects --info=progress2. -L follows symlinks.
rsync -aL "$PD/athena" "$PD"/*.bundle studio:~/athena-build/

# Upgrade in place, REUSING the installed config (preserves budget/model/bind +
# the seeded admin account/token — no re-seed when userCount>0). install restarts
# the daemon (launchctl kickstart); the boot launchd service auto-loads the new
# binary.
ssh studio 'cd ~/athena-build && sudo ./athena install --from . \
  --config /usr/local/etc/athena/athena.toml'
```

Verify remotely from the dev Mac:

```sh
ssh studio 'curl -s http://127.0.0.1:7447/openapi.json | python3 -c "import sys,json;print(json.load(sys.stdin)[\"info\"][\"version\"])"'
TOKEN=...   # studio admin bearer (athena auth login --host 172.16.100.118 --port 7447)
curl -s http://172.16.100.118:7447/healthz -o /dev/null -w "%{http_code}\n"
curl -s http://172.16.100.118:7447/v1/chat/completions -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":20}'
# + any fix-specific check (e.g. for v0.10.230, the streaming-tools curl below).
```

The **model store is untouched** by an update — no need to re-copy the ~26 GiB
MoE unless you're changing models.

## Gotchas (the things that bit us)

- **No Xcode on the studio** → never try to build there. Ship the binary.
- **`/usr/local/bin` may not exist** on Apple Silicon (Homebrew is `/opt/homebrew`).
  Fixed in v0.10.229+ (install creates it); for an OLDER binary, first
  `ssh studio 'sudo mkdir -p /usr/local/bin'`.
- **Old rsync**: `rsync -aL …` only — no `--info=progress2`.
- **`~/athena-build/` survives reboots** (home dir); the session scratch under
  `/private/tmp` does NOT (recreate the token file after a reboot).
- **Signing**: the binary is adhoc + Hardened-Runtime signed; it runs fine
  cross-machine over rsync/ssh (no quarantine xattr is set).
- **Auth**: non-loopback bind ⇒ bearer required. The install seeds an admin token
  ONCE (fresh store only). Cache it: `athena auth login --host 172.16.100.118
  --port 7447`. Lost it? mint offline on the studio:
  `sudo athena auth token add --user admin --data-dir /Users/goodolclint/.athena`.
- **Sleep**: studio is `pmset sleep 0` + the daemon holds a power assertion; keep
  it that way (headless box must not idle-sleep).

## Fresh install (clean machine) — extra steps vs an update

1. `ssh studio 'sudo mkdir -p /usr/local/bin'` (only if on a pre-0.10.229 binary).
2. Push a config: `scp deploy/athena.toml studio:~/athena-build/athena.toml` and
   edit it (LAN bind `0.0.0.0`, `budget_bytes` ~83% of RAM for headless,
   `model=`, `preload=true`, a safe `max_prompt_tokens` — ~8192 on 48 GiB,
   `speculative=false` for the MoE). Install with `--config athena.toml`.
3. **Save the seeded admin password + token** (printed once).
4. Get the model into the store: `rsync -aL ~/.athena/models/<model>/
   studio:~/.athena/models/<model>/` (LAN, ~26 GiB) **or** `athena pull <id>` on
   the studio. Restart the daemon so `preload` warms it.

## Studio facts (current)

- Daemon: launchd `me.goodolclint.athena` (RunAtLoad+KeepAlive), config
  `/usr/local/etc/athena/athena.toml`, binary `/usr/local/libexec/athena/athena`,
  launcher `/usr/local/bin/athena`, store `~/.athena/models`, data `~/.athena`.
- Serving `gemma-4-26b-a4b-it-8bit` on `http://172.16.100.118:7447` (LAN, auth-on),
  budget ~43 GB, FileVault OFF (remote reboot safe).
