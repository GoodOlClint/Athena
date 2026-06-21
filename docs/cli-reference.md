# Athena CLI Reference

`athena` is the single unified command for the Project the platform inference appliance
(passive oracle). One binary carries three concerns:

1. **Local daemon lifecycle** — `load`/`start`/`stop`/`restart`/`status`.
2. **Apple-host operator ops** — `install`/`pull`/`convert`/`allowlist`/… (macOS only).
3. **HTTP client verbs** — `run`/`auth`/`cache`, which drive
   a **local OR remote** daemon. On Linux/Windows only this portable client subset
   ships (no local daemon to manage).

Default port is **7447** (Athena's own port, not Ollama's 11434). The default
subcommand is `load`.

> Generated from the v0.10.10x command tree. Abstracts taken from source; run
> `athena <cmd> --help` for the authoritative, version-matched option list.

---

## Cross-cutting options (client verbs)

Verbs that talk to a daemon (`run`, `ps`, `cache`,
`auth …` when remote, etc.) share `DaemonOptions`:

| Option | Default | Notes |
|--------|---------|-------|
| `--host <host>` | `127.0.0.1` | Daemon host. A non-loopback host ⇒ "remote" mode. |
| `--port <port>` | `7447` | Daemon port. |
| `--key <key>` | — | Bearer key. Falls back to `ATHENA_KEY` env, then Keychain. |

`--data-dir <path>` (default: configured / `~/.athena`) appears on the offline
auth/allowlist verbs that touch the local SQLite store directly.

---

## Quickstart: add an API key

A bearer token (the "API key") always belongs to a **user**. Two steps:

```bash
# 1. Create a user (prompts for password, no echo). Default role: member.
athena auth user add myapp --role member

# 2. Mint a bearer token for that user — printed ONCE, stored only as a hash.
athena auth token add --user myapp --label "the consuming application" --ttl 90d
```

Use it:

```
Authorization: Bearer <printed-key>
```

Scope a token to a role subset with repeatable `--role`; omit `--ttl` for no
expiry (still subject to `token_max_age_days`).

> **Disabling auth** is not a flag — auth is *presence-driven*. It enables itself
> whenever any credential exists (bootstrap env key, `auth_keys_file`, or any DB
> user/token) and only runs open on a **loopback** bind. To run open: remove all
> credentials and bind `127.0.0.1`. A non-loopback bind with zero credentials
> refuses to start (fail-safe).

---

## Models & store (operator, macOS)

| Command | Abstract |
|---------|----------|
| `load` (alias `serve`, default) | Run the governed HTTP inference surface (foreground). |
| `init` | Pull the default auxiliary models (embeddings, transcription, diarization, speaker-embeddings) into the store. |
| `install` | Install Athena as a boot-time launchd system daemon. |
| `list` (alias `ls`) | List models available in the local model store. |
| `ps` | Show governed module state from a running daemon. |
| `pull` | Download a model (HF id) into the local store. `--check` is a config-only dry run reporting whether Athena can load it (modality + loadability), downloading nothing — see [model-support.md](model-support.md). |
| `convert` | Convert an HF model into the local MLX-format store (optionally quantize). Redirects non-quantizable modalities (embedding/transcription/diarization/speaker) to `pull`. |
| `verify` | Check a stored model's **integrity** (offline). Complements `pull --check` (model **support**) — see [model-support.md](model-support.md). |
| `prune` | Remove broken/dangling models from the store. |
| `cp` | Alias or copy a stored model under a new name. |
| `default` | Show or set the default served model. |
| `rm` | Remove a model from the local model store. |
| `show` | Show a model's config and size. |
| `unload` | Release a module's slot and free its budget. |
| `resident` | Show every module's resident model slot. |

### `load` (serve) — key options

```bash
athena load \
  --host 127.0.0.1 --port 7447 \
  --model <name-or-path> \           # default served model
  --llm-model <id> [--llm-model …] \ # repeatable; FIRST is default, set selectable per-request
  --engine mlx \                     # mlx (real) | stub (no model)
  --temperature 0 --max-tokens 1024 \
  --speculative \                    # MTP speculative decode (needs MTP head)
  --budget-bytes <N> \               # global memory budget (default 75% RAM)
  --prompt-cache-cap-bytes <N>       # per-request KV/prompt-cache cap (default ¼ budget)
```

---

## Allowlist

`allowlist` — Manage the persistent per-module model allowlist (SQLite-backed;
`/api/models/allow` CRUD has the daemon-side equivalent).

---

## Daemon lifecycle

| Command | Abstract |
|---------|----------|
| `start` | Start the daemon. Installed (root): launchd-managed; uninstalled: foreground-attached (Ctrl-C to stop). |
| `stop` | Stop the daemon process (user pidfile, else system launchd). |
| `restart` | Re-bootstrap the system LaunchDaemon (re-reads the plist). |
| `status` | Daemon health / readiness. |
| `doctor` | Diagnose host + config posture (TLS, FileVault, rate-limit, audit, …). |
| `logs` | Tail / stream daemon + model logs (wraps `/api/logs[/stream]`). |
| `uninstall` | Remove the installed launchd system daemon. |

---

## Config

`config` — Inspect and edit the daemon TOML config.

| Subcommand | Abstract |
|------------|----------|
| `config path` | Print the resolved config path. |
| `config show` | Print the raw config file. |
| `config get <key>` | Print one effective value (parsed). |
| `config set <key> <value>` | Set one scalar in place (keeps comments/layout). |

```bash
athena config set listen 127.0.0.1:7447
athena config get listen
```

---

## Auth / RBAC

`auth` — Manage RBAC users, roles, and bearer tokens. Keys are stored
hash-only (SHA-256); constant-time compare.

| Subcommand | Abstract |
|------------|----------|
| `auth user add <username>` | Create an account. `--role` (default member), `--force` (replace existing). Password: prompt (no echo), or `--password-stdin`, or `$ATHENA_PASSWORD` — never on argv (ADR 005). |
| `auth user list` | List accounts and roles. |
| `auth user passwd <username>` | Offline password reset (local-only; keeps roles/tokens). |
| `auth user rm <username>` | Delete an account (cascades roles + tokens). |
| `auth role list` | List the role → permission catalog. |
| `auth role grant <role> <user>` | Grant ROLE to USER. |
| `auth role revoke <role> <user>` | Revoke ROLE from USER. |
| `auth token add --user <u>` | Mint a bearer token for a user (shown once). `--role`*, `--label`, `--ttl`. |
| `auth token rotate <prefix>` | Revoke + reissue a token (M36.2). |
| `auth list` | List tokens: user, scope, hash prefix (no secrets). |
| `auth rm <prefix>` | Remove tokens whose hash hex starts with PREFIX (≥ 6 chars). |

`--ttl` accepts `30d` / `12h` / `90m` / `3600s` / bare seconds. The last admin
user cannot be removed (`guardLastAdmin`).

---

## Outbound credentials (Keychain)

| Command | Abstract |
|---------|----------|
| `hf login` / `logout` / `status` | Manage the Hugging Face token (Keychain). |
| `proxy login` / `logout` / `status` | Manage egress-proxy credentials (Keychain). |

Precedence is env > TOML. Proxy auth is Basic-only.

---

<!-- ADR 025 S2 — the async request queue and the `athena queue` CLI were
     removed. Model lifecycle ops (`pull`/`convert`/`prune`) run synchronously
     and stream SSE progress; there are no jobs to inspect. -->

---

## Prompt-prefix cache (client)

`cache` — Inspect or flush the prompt-prefix KV cache (M59).

| Subcommand | Abstract |
|------------|----------|
| `cache prompt` | Show prompt-prefix cache stats (entries, bytes, hits). |
| `cache flush` | Flush the prompt-prefix cache (admin-only, audited). |

---

## Metering & audit (client)

| Command | Abstract |
|---------|----------|
| `usage` | Per-principal usage counters (`GET /api/usage`); `--include-usage` on streams. |
| `audit` | Query the append-only audit log (`GET /api/audit`, admin-only, filterable). |

---

## Notes

- **Verb overload, not namespaces.** The same verb (e.g. `run`, `pull`) targets a
  local or off-box daemon depending on `--host`/`DaemonOptions.isRemote` — there is
  no `local`/`remote` command split.
- **Passive oracle.** The only outbound surface is opt-in remote syslog; no webhooks.
- **macOS build** requires `xcodebuild` (full Xcode) for the MLX Metal shaders;
  `swift build` cannot build them.
