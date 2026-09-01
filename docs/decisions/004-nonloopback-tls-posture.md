# 004 — Non-loopback TLS posture: warn-only, not fail-closed

**Status:** Implemented
**Date:** 2026-06-12
**Milestone:** M65 (audit-remediation; resolves standing DECISION #1; audit A2/K1/K8/A12/A3)

**Implemented across:** A12 `Secure` cookie — M65.2 (v0.10.118); doctor TLS posture finding — M28.2; the loud non-loopback-plaintext **startup warning** (`Load.swift`) and the **A3 peer-IP login limiter** (`AppRequestContext.remoteAddress` → `loginLimiter`, no XFF trust) — M65.6 (v0.10.122). Client `https` (K1/K8) stays deferred per the ruling below; A2/K1/K8 remain advisory (warn + doctor), not fail-closed.

## Context

An auth-on daemon bound to a non-loopback address today serves **plaintext HTTP** silently — bearer tokens and the session cookie can cross the wire in clear on a misconfigured deployment. In-daemon TLS exists (M28, opt-in `tls_cert`/`tls_key`) and a blessed reverse-proxy guide exists, but nothing *requires* either. The portable client has no `https` support. Related: the `/ui/login` rate-limiter (audit **A3**) needs a client-IP source, and "which IP do we trust" is the same proxy/trust question.

The two postures considered:

- **Fail-closed:** non-loopback + auth-on **refuses to start** without TLS unless an explicit `--insecure` flag is passed. Strongest guarantee; a breaking change for existing plaintext-behind-proxy deployments.
- **Warn-only:** start anyway, but emit a loud startup warning and a `doctor` finding. Lower friction; relies on the operator heeding the warning.

## Decision

**Warn-only.** A non-loopback, auth-on daemon serving plaintext starts, but:

- logs a loud `warning`-level startup line (subsystem `athena`) stating that tokens and the session cookie are exposed and recommending `tls_cert`/`tls_key` or a TLS-terminating reverse proxy;
- surfaces the same as a `doctor` posture check.

No `--insecure` gate and no start refusal. Client `https` (K1/K8) is **deferred** — not part of this ruling; the portable client stays http-only for now.

**A3 (login limiter)** is unblocked under this ruling and keys on the **TCP peer address only** (via a `RemoteAddressRequestContext`). We do **not** trust `X-Forwarded-For` — without an enforced trusted proxy, XFF is spoofable. Limitation, documented: behind a reverse proxy every client shares the proxy's bucket; an operator who needs per-client login throttling should rate-limit at the proxy. `A12` (Secure cookie when the daemon serves TLS) already shipped in M65.2.

## Consequences

- No deployment breaks; the guarantee is advisory (warning + doctor), not enforced.
- Revisit fail-closed + `--insecure` + client `https` as a future hardening milestone if the appliance moves toward untrusted-network exposure.
- A3 implementation requires introducing a custom `RemoteAddressRequestContext` (peer address plumbing) — modest structural change shared by any future per-IP control.
- A2/K1/K8 remain open (downgraded to warn); tracked, not closed.

## Amendment (WP5, 2026-07-01) — `/healthz` + `/openapi.json` unauth is accepted posture

The 2026-07-01 audit flagged that `/healthz` and `/openapi.json` are reachable **without auth on a non-loopback bind** (`Auth.swift`), and `/healthz` exposes model ids, memory footprint, GPU clock, and thermal state.

**Decision: accepted, not gated** (same warn-only spirit as the TLS ruling above).

- `/openapi.json` unauth is a **deliberate, documented contract** — the daemon is self-describing (`CLAUDE.md` "Self-describing: always reachable, no auth required"); a consumer must be able to read the surface without a token.
- `/healthz` carries **operational telemetry** (liveness + resident model ids + memory/GPU/thermal), not secrets. Unauthenticated liveness is the standard contract for a health endpoint; gating it would break trivial LAN monitoring of the passive-oracle appliance, whose threat model is a **trusted LAN** (the studio deployment). The bytes it exposes are reconnaissance-grade at worst.
- **Tripwire:** if the appliance is ever exposed to an untrusted network, gate the `/healthz` *detail* behind `.metricsRead` (keep a bare `200` liveness open) — the same "revisit on untrusted-network exposure" trigger as the TLS posture.

No code change: this records the standing posture so it is explicit rather than implicit.
