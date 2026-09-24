# Architecture Decision Records — Athena

One file per decision: `NNN-title.md` (e.g. `001-passive-oracle.md`).

Each ADR should answer:
- **Context** — what prompted the decision
- **Decision** — what was chosen
- **Rejected alternatives** — what was considered and why not
- **Consequences** — what this commits the project to
- **Status** — a `**Status:**` line near the top, kept current as the decision ships (`LEDGER.md`'s status column is generated from this line, not from a separate record)

`LEDGER.md` is a one-line-per-ADR index, and AGENTS.md imports it (`@docs/decisions/LEDGER.md`), so the index auto-loads into Claude's context at session start. Substance stays in the ADR files, one hop away — the ledger line is a pointer, not a summary. A new ADR needs its own line added to `LEDGER.md` in the same commit — nothing else adds it automatically.
