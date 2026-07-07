<!-- Keep PRs small and focused: one concern each. -->

## What this changes
A short description of the change and why.

## Test evidence
- [ ] `./deploy/test.sh` passes (unit tier)
- [ ] Relevant e2e run if applicable (`deploy/e2e-*.sh` — name which, and the result)
- [ ] For a bug fix: a test that fails before the change and passes after

## Design / ADR discipline
- [ ] No new outbound network call (passive-oracle rule); or explain the model-weight/remote-syslog exception used
- [ ] Route changes update `Sources/athena/Server/OpenAPISpec.swift` in this PR
- [ ] Errors use the `{"error":{"message","type","code"}}` envelope
- [ ] A new architectural decision (if any) is recorded as an ADR under `docs/decisions/` in this PR
- [ ] I read the relevant ADR(s) before changing the governor / API surface / passive-oracle rule

## Notes for the reviewer
Anything non-obvious, trade-offs, or follow-ups.
