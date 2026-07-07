---
name: Feature request
about: Propose a capability or change
title: ""
labels: enhancement
---

**The problem**
What are you trying to do that Athena doesn't support today?

**Proposed direction**
What you'd like to see. If it touches the API surface, the memory governor, the passive-oracle rule, or the OpenAPI spec, please skim `docs/decisions/` first — the trade-off may already be recorded in an ADR.

**Alternatives considered**
Other approaches, and why they fall short.

**Scope check**
- Does this require any outbound network call? (Athena is a passive oracle — model-weight fetches and the opt-in remote-syslog sink are the only exceptions.)
- Is it inference (`/v1/*`) or daemon control (`/api/*`)?
