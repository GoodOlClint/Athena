# Upload limits + `Expect: 100-continue` — change plan

Implements [ADR 017](decisions/017-upload-limits-and-expect-continue.md). House style:
small, test-pinned slices; each slice is one commit + annotated semver tag pushed to
`origin/main`; `Athena.appVersion` bumped **in** the slice commit; pre-commit pipeline
Tests → Security → Quality → Refactor.

## Problem recap (from the investigation)

- **Hang:** the daemon never sends `100 Continue`, so any client that uses `Expect:
  100-continue` (which clients do *for large bodies*) hangs until its own timeout. Reproduced
  for 1 MB **and** 30 MB. This is the user-visible bug.
- **Cap:** oversized bodies return `400` with a leaked `NIOTooManyBytesError` instead of a
  clean `413`; the 25 MiB cap is fixed and undocumented.

## Decisions locked (operator review)

- Split caps: `max_audio_upload_bytes` (default **100 MiB = 104857600**) for `/v1/audio/*`;
  `max_request_body_bytes` (default **4 MiB = 4194304**) for JSON routes. Both configurable.
- `0` ⇒ config-parse error (not "unlimited").
- Over-cap ⇒ `413` + `code: "payload_too_large"`, clean envelope, no internal type names.

## Sequencing

Ordered so the **highest-value, lowest-risk** fixes land first and each slice is independently
shippable. The risky interim-`100`-write is isolated last.

### Slice 1 — Configurable, modality-scoped caps (no behavior change at defaults yet for audio)

- `AthenaConfig`: add `maxAudioUploadBytes: Int?`, `maxRequestBodyBytes: Int?`
  ([AthenaConfig.swift](../Sources/AthenaDeploy/AthenaConfig.swift)); parse `max_audio_upload_bytes`
  / `max_request_body_bytes` (reject `0`/negative with a clear error).
- `ConfigEditor` get/set cases for both keys (both files:
  [athena/Commands/ConfigEditor.swift](../Sources/athena/Commands/ConfigEditor.swift),
  [AthenaDeploy/ConfigEditor.swift](../Sources/AthenaDeploy/ConfigEditor.swift)).
- `Load`: CLI flags + wiring into the server (defaults 104857600 / 4194304)
  ([Load.swift](../Sources/athena/Commands/Load.swift)).
- `AthenaServer`: `var maxAudioUploadBytes = 104_857_600`, `var maxRequestBodyBytes = 4_194_304`
  (memberwise-init params, matching `coldLoadWaitSecs`); replace the literal `25 * 1024 * 1024`
  at lines 1702/1866/2034 with `maxAudioUploadBytes`, and the audio-route `4 * 1024 * 1024`
  collect sites with `maxRequestBodyBytes`. (Audit each `collect(upTo:)` site and route it to
  the right knob.)
- **Tests:** `AthenaDeployTests` — TOML parse of both keys, `0`→error, round-trip via
  `ConfigEditor`. **Bar:** `swift test` green.
- Net effect of this slice: audio cap rises to 100 MiB; oversized still returns the *old*
  leaked-400 (fixed in Slice 2). Ships value (bigger files work over loopback for non-`Expect`
  clients) with minimal risk.

### Slice 2 — Clean `413` on over-cap (replace leaked 400)

- Map the `collect(upTo:)` overflow (`NIOTooManyBytesError`) to `413` with
  `{"error":{"message":"file exceeds the N-byte upload limit","type":"invalid_request_error",
  "code":"payload_too_large"}}` at the three audio handlers and the JSON handlers. No internal
  type names in `message`.
- Add an up-front `Content-Length` check at the top of each handler: if present and > cap →
  `413` immediately (before touching the body). This is the fast path and is reused by Slice 3.
- Put the decision (`contentLength?, cap → .reject413 | .proceed`) in **`AthenaServerKit`** as a
  pure function; unit-pin it (`AthenaServerKitTests`).
- **Tests:** unit (decision algebra) + `e2e` over-cap upload asserts `413` + `payload_too_large`
  + no `NIOTooManyBytesError` substring. **Bar:** `./deploy/test.sh` + `e2e` green.

### Slice 3 — `Expect: 100-continue` handler (the actual hang fix) — isolated, low risk

Research (ADR 017 sources) simplified this from the original "risky" framing to a **stateless
~15-line handler**, because the over-cap rejection is handled entirely by Slice 2 (app-side),
not by this handler.

- New `ExpectContinueHandler` in `AthenaServerKit`: a `ChannelDuplexHandler & RemovableChannelHandler`
  (`InboundIn/Out = HTTPRequestPart`, `OutboundIn/Out = HTTPResponsePart`). On inbound `.head`
  carrying `Expect: 100-continue` → `context.writeAndFlush(wrapOutboundOut(.head(HTTPResponse(status:
  .continue))))`, then `context.fireChannelRead(data)` to forward the head unchanged. Every other
  part: straight passthrough. **No cap awareness, no body draining, no short-circuit.**
- Wire via `HTTP1Channel.Configuration(additionalChannelHandlers: { [ExpectContinueHandler()] })`
  into **both** `serverBuilder` branches: `.http1(configuration:)` and the `.tls(base:)` builder
  (`.tls(_ base: HTTPServerBuilder = .http1(configuration:), tlsConfiguration:)`)
  ([AthenaServer.swift:829-847](../Sources/athena/Server/AthenaServer.swift#L829)). One config
  object, both transports.
- **Why low risk (confirmed locally):** the server codec passes informational heads through
  unconditionally ([`HTTP1ToHTTPCodec.swift:117`]); `withPipeliningAssistance:false` +
  `withErrorHandling:false` mean the fragile NIO state machines aren't in the pipeline; our
  interim write doesn't traverse `HTTPConnectionStateHandler`.
- **Tests:** `NIOEmbeddedChannel` unit test in `AthenaServerKitTests` — feed a `.head` with
  `Expect: 100-continue`, assert a `100 Continue` head is written outbound AND the head is
  forwarded inbound; feed a `.head` without it, assert no write + forwarded. Plus `e2e` with the
  repro client (sends `Expect`, waits): (a) small body → `100` then normal `200`/400, (b) valid
  large (e.g. 50 MiB) body → `100`, streams, succeeds, (c) over-cap body → `413` (via Slice 2).
  **Bar:** the exact hang repro returns < 1 s.

### Slice 4 — Surface in OpenAPI + docs

- `OpenAPISpec.swift`: add a reusable `PayloadTooLarge` response component; add `413` +
  documented max size to the three audio routes and `413` to the JSON routes; stamp the cap in
  the route `description`. Keep the version stamp; keep the drift-guard green.
- `docs/reverse-proxy.md`: note `client_max_body_size` / `proxy_request_buffering` must be
  raised to match `max_audio_upload_bytes`; the daemon's 413 still fires for direct clients.
- Document the worst-case transient memory: `max_audio_upload_bytes × max_concurrency`.
- **Tests:** drift-guard (spec↔routes) green; `openapi.json` served reflects the caps.

## Test bar (overall)

- `./deploy/test.sh` (unit) green; `./deploy/e2e-rbac.sh` unaffected (no auth/RBAC change).
- New `e2e` cases: oversized→413, `Expect`-small→ok, `Expect`-oversized→fast-413,
  `Expect`-valid-large→ok. The original hang repro must return < 1 s.
- Every slice bumps `appVersion`; `--version` / OpenAPI / install marker stay consistent.

## Out of scope (explicit)

- Streaming-to-disk decode (avoid buffering the whole upload) — noted follow-up.
- A video-input route / `max_video_upload_bytes` — no surface exists yet; the knob pattern
  accommodates it later (ADR 017 consequences).
- Per-route or per-principal upload caps — single audio + single JSON knob for v1.
- Touching the governor / Metal budget — these are ingress limits, not tenants.

## Open question for review

- **Default audio cap = 100 MiB**: confirm the worst-case transient (100 MiB × `max_concurrency`)
  is acceptable for the target hardware, or whether `max_concurrency` should ship with a default
  when `max_audio_upload_bytes` is large. (Today concurrency caps are opt-in/off.)
