---
name: Bug report
about: Something in the daemon or CLI misbehaves
title: ""
labels: bug
---

**What happened**
A clear description of the bug and what you expected instead.

**Reproduction**
Steps, and the request if it's an API bug (method, path, and a minimal body — redact secrets/tokens).

**Environment**
- `athena --version`:
- `athena doctor` output (redact anything sensitive):
- Model id (the `model` you requested, and `athena ls` if relevant):
- macOS + hardware (e.g. macOS 15.x, M4 Max):

**Daemon state**
- `curl http://127.0.0.1:7447/healthz` output:
- Relevant logs — macOS unified log, subsystem `athena` (see `docs/logging.md`):

**Anything else**
Notes, hypotheses, or related issues.
