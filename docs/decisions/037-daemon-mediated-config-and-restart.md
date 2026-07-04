# 037 — Daemon-mediated config + sudoless restart (sudo only for install/uninstall)

**Status:** **Accepted** (operator-approved 2026-07-03). Implemented in 3 slices (static plist → config API → restart API).
**Date:** 2026-07-03
**Milestone:** usability audit 2026-07-02 §6 remediation (WP-E)
**Relates:** ADR 013 (`/api/*` is the control plane), ADR 025/026 (store = auth/audit/usage; loopback writes nothing), ADR 024 (in-memory data-protection — informs the deny-list), ADR 029 (inference execution gate — the restart drain composes with it).

## Context

Changing the daemon's configuration or restarting it currently requires `sudo`, and the daemon's *own* config-write surfaces are broken by the same ownership. This is a security-boundary change, so it goes through the brownfield gate.

**Inventory (verified against source):**
- Explicit root gates that are *correct* (the launchd/install boundary): `install` / `uninstall`.
- Explicit root gates that are the problem: `start` / `stop` / `restart` for the system daemon (`DaemonLifecycle.swift:107/268/308`) and `config set` apply (`ConfigCmd.swift:110`).
- Implicitly broken without sudo: the TOML write in `config set --no-apply` and `athena default` (the installed `/usr/local/etc/athena/athena.toml` is `root:wheel 0644`), **and the daemon's own config-write surfaces** — WebUI `POST /ui/api/config` and `POST /api/models/default` both call `ConfigEditor.setScalarThrowing` on that root-owned file and fail `writeFailed` on any installed daemon.

**Structural root cause:** `LaunchdPlist.dictionary` (`AthenaDeploy/LaunchdPlist.swift:38–149`) **freezes ~30 TOML values into the plist's `ProgramArguments`**, so any config change requires a plist re-render + `bootout`/`bootstrap` = root. Only a few keys are read live from TOML at boot. Meanwhile the shipped plist already has `KeepAlive=true` (verified live), so a daemon self-exit → launchd relaunch works on every existing install with no plist change.

The daemon-mediated path the operator wants already exists in code (`ConfigEditor`); install ownership defeats it.

## Decision

Move config + restart onto the daemon's own control plane (ADR 013, `/api/*`), so `sudo` is required **only** for `install` / `uninstall`.

**1. Enabler — static plist.** `LaunchdPlist.dictionary` emits only `label` / service user / executable path / `ATHENA_CONFIG` env (the config-file path). `athena load --background` reads the **full TOML at boot** (explicit CLI flags still win). A config change is then a TOML edit + restart — never a plist re-render. Removes the freeze that made config a root operation.

**2. Config — chown + `GET`/`PUT /api/config`.**
- `chown` the installed TOML to the service user at install time (one line next to `Install.swift:325`), so the daemon (and its WebUI editor) can write it.
- Add `GET /api/config` and `PUT /api/config` to the control plane: `daemon.admin`-gated, audited, wrapping the existing hardened `ConfigEditor`; `OpenAPISpec.swift` updated in the same edit (canonical-pipeline rule). Un-breaks the WebUI editor (`POST /ui/api/config`) and `POST /api/models/default`.
- `athena config set` / `athena default` repoint at the API when a daemon is reachable; the sudo path stays as a fallback for the daemon-down case.

**3. Restart — `POST /api/admin/restart`.** `daemon.admin`-gated, audited. Drains in-flight requests (composes with the ADR 029 inference-execution gate) then `exit(0)`s; `KeepAlive` relaunches (the ~10s `ThrottleInterval` delay is acceptable). `athena restart` as non-root calls it; the sudo `bootout`/`bootstrap` path remains as a fallback.

**4. Security decision (binding) — API config deny-list.** Config takeover ≈ daemon takeover, and loopback dev mode has **no auth** so the route is open to any local process there. The following keys are **not settable via `PUT /api/config`** (edit the TOML + sudo-restart instead): `auth_keys_file`, `tls_cert`, `tls_key`, `encrypt_store`, `data_dir`, `deny_debugger_attach`. A denied key returns a cause-naming `400` (standard error envelope). The deny-list decision is MLX-free and unit-pinned (ADR 008/009).

**End state:** `sudo` required only for `install` / `uninstall`.

## Rejected alternatives

- **Status quo (freeze config in the plist).** Every config change stays a root plist re-render; the daemon's own editor stays broken. This is the defect.
- **A setuid/SMJobBless privileged helper** to write the root-owned TOML. Adds a privileged attack surface and install complexity to solve a problem that a chown + a `daemon.admin`-gated route already solves. The daemon is the natural mediator; it already authenticates and audits.
- **Making the whole config settable via API (no deny-list).** Rejected on the ADR 024 threat model: auth/TLS/encryption/data-dir/debugger keys are exactly the ones whose remote settability turns config-write into daemon-takeover, and loopback mode has no auth. These stay TOML-plus-sudo.
- **Dropping the sudo fallback entirely.** Rejected: when the daemon is *down*, the operator still needs to edit config and (re)start it; the CLI must degrade to the direct TOML/launchd path.

## Consequences

- **Migration for existing installs:** the plist must be re-rendered once (to the static form) on the next `install`/upgrade. Until then, an old freeze-style plist keeps working (config still applies via the API on the running daemon; only boot-time defaults come from the frozen args). Note the one-time re-render in the install/upgrade path.
- The restart route depends on `KeepAlive=true` — already true on every shipped install (verified), so no plist change is needed to make sudoless restart work today.
- The WebUI config editor and `POST /api/models/default` start working on an installed daemon (they were silently broken).
- New audited control-plane routes (`/api/config` GET/PUT, `/api/admin/restart`) — two more `daemon.admin` surfaces; non-admin tokens get `403` + audit rows.
- Honesty boundary: a local process in loopback dev mode (no auth) can call `PUT /api/config` — the deny-list is what bounds the blast radius there; it does not make loopback multi-tenant-safe (it never was).

## Definition of done (from audit §6, on approval)

- Non-root `athena config set max_tokens 4096` exits 0 and takes effect after non-root `athena restart` (daemon PID changes, `/healthz` green ≤30s).
- WebUI config save works on an installed daemon.
- Non-admin token → `403` + audit rows on both new routes; deny-listed keys → `400` via API.
- An e2e proves config-set → restart → value-effective **without sudo**; `deploy/e2e-rbac.sh` green.

## Implementation slices (each its own commit + tag, appVersion bump in the slice; only after acceptance)

1. **Static plist + full-TOML-at-boot** (enabler; `LaunchdPlist` + `athena load --background`; unit-pin the plist shape).
2. **Chown + `GET`/`PUT /api/config`** (control-plane routes + deny-list, spec same-edit; repoint `athena config set`/`default`; un-break WebUI).
3. **`POST /api/admin/restart`** (drain + `exit(0)`; repoint `athena restart`; e2e config-set→restart→effective).
