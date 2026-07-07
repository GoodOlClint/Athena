# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** — do not open a public issue.

Use GitHub's private vulnerability reporting (the **Security** tab → **Report a vulnerability**) on this repository. Include the affected version (`athena --version`), a description, and a reproduction if you have one. You'll get an acknowledgement, and a fix or mitigation will be coordinated before any public disclosure.

## Security posture

Athena is a **passive oracle** by design, which shapes its threat model:

- It **answers inbound requests only** and never initiates outbound connections, except to fetch model weights from Hugging Face (and an opt-in remote-syslog sink). There are no result, billing, or telemetry callbacks.
- It **binds `127.0.0.1` by default**, with authentication disabled in that loopback dev mode. Any non-loopback deployment must enable bearer-token auth and TLS (see `docs/quickstart.md` and `docs/reverse-proxy.md`).
- Authentication is bearer-token RBAC: each token maps to a user with roles, and each route requires a single permission. Tokens are stored hashed at rest.
- In-memory data protection (process hardening, idle KV-cache encryption) and at-rest store encryption are documented in `docs/at-rest.md`, `docs/confidential-kv-cache.md`, and the ADRs under `docs/decisions/`. The honesty boundary is stated there: the active working set is irreducibly plaintext in RAM, and a kernel/root adversary is out of scope.

## Supported versions

Athena is pre-1.0 and ships from a single active line. Security fixes land on the latest release; there is no back-port branch. Run the newest tagged release.
