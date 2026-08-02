# Athena quickstart

From a fresh checkout to your first authenticated chat request. Athena is
a single native macOS/MLX daemon — a **passive oracle** that only answers
inbound requests (see the [README](../README.md)).

## 1. Requirements

- macOS on Apple Silicon.
- A **full Xcode** install (not just the Command-Line Tools), **Xcode 26.4
  or newer (Swift 6.3)** — MLX builds Metal shaders that the CLT cannot,
  and `mlx-swift` 0.31.5+ ships a Swift 6.3 manifest: an older Xcode fails
  at dependency *resolution* with an error naming mlx-swift's tools
  version, before any build starts.

## 2. Build

```sh
./deploy/build.sh Release
```

The binary lands at `.build/xcode/Build/Products/Release/athena` (plus
the `athenad` launcher). Put it on your `PATH`, or call it by path.

## 3. Run the daemon

You have two ways to run Athena. Pick one.

### a. Dev — foreground, loopback

```sh
athena load
```

This serves the governed HTTP surface in the foreground on
`127.0.0.1:7447`. With **no users seeded and a loopback bind, auth is
off** — convenient for local development. (Bind to a non-loopback address
with auth off and the daemon refuses to start.)

### b. Production — boot-time system daemon

```sh
sudo athena install --config deploy/athena.toml
```

`install` registers a launchd system daemon, and on a **fresh** store it
seeds an `admin` account and one admin bearer token, printing both
**once**:

```
  ┌──────────────────────────────────────────┐
  │ admin account created (role: admin)       │
  │   username: admin                         │
  │   password: ……                           │
  └──────────────────────────────────────────┘

  admin bearer token (SAVE NOW — shown once):
    sk-athena-……
```

Save the token — it is never shown again. Manage the daemon with `athena
start` / `stop` / `status`, and tail logs with `athena logs -f`.

## 4. Seed users and tokens (manual)

If you ran the dev daemon, or want more accounts, create them with the
CLI. Seeding any user turns auth **on**.

```sh
# Create an admin and a member account (prompts for a password).
athena auth user add admin --role admin
athena auth user add alice --role member

# Mint a bearer token (printed once).
athena auth token add --user alice --label laptop
```

A token can be scoped to a subset of its user's roles with repeated
`--role` flags. Tokens are stored hash-only — the secret never rests on
disk.

## 5. First requests

Two endpoints are always open (no token), so you can confirm the daemon
is up and explore the API:

```sh
curl http://127.0.0.1:7447/healthz
curl http://127.0.0.1:7447/openapi.json | head
```

`/openapi.json` is the full OpenAPI 3.0.3 reference — point Swagger UI or
an SDK generator at it.

Now a chat completion. Make sure a model is available (`athena ls` lists
the store; `athena default` shows the served model; `athena pull
<hf-id>` downloads one). Then, OpenAI-style:

```sh
TOKEN=sk-athena-……   # the token from step 3 or 4

curl http://127.0.0.1:7447/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{ "role": "user", "content": "Say hello in one word." }]
  }'
```

Add `"stream": true` for an SSE stream terminated by `data: [DONE]`.

The CLI wraps all of this — `athena run <model> "your prompt"` against a
running daemon, local or remote.

## 6. WebUI

Athena ships a browser control console at **`/ui`**:

```
http://127.0.0.1:7447/ui
```

Log in with an admin account (from step 3 or 4). The console is
RBAC-aware: it surfaces monitoring, config, the model store, and
user/role/token management according to the logged-in user's roles.

## 7. TLS

Bearer tokens and the WebUI session cookie travel in plaintext, so a
non-loopback deployment needs TLS. Two options — pick one, never both:

1. **In-daemon TLS.** Point both keys at PEM files in `athena.toml` and
   Athena terminates HTTPS itself:

   ```toml
   tls_cert = "/usr/local/etc/athena/tls/fullchain.pem"
   tls_key  = "/usr/local/etc/athena/tls/privkey.pem"
   ```

   Setting only one is a hard startup error (fail-closed). `athena
   doctor` reports cert/key existence, expiry, and permissions.

2. **Reverse proxy.** Keep the daemon on loopback HTTP and let
   nginx/Caddy terminate TLS — best for managed certificates
   (Let's Encrypt) or fronting many services. See
   [reverse-proxy.md](reverse-proxy.md).

## Next steps

- **Models** — `athena pull <hf-id>`, `athena convert <hf-id>` (MLX
  conversion; pass `--q-bits 4` to also quantize), `athena ls`,
  `athena default <name>`, `athena init` (pull the auxiliary
  embedding/transcription/diarization models).
- **Config** — `athena config get|set <key>`; see
  [deploy/athena.toml](../deploy/athena.toml) for every key.
- **Diagnostics** — `athena doctor` (read-only environment + TLS + rate
  limit posture).
- **Operations** — rate limits, concurrency caps, usage metering, and the
  audit trail are all in `athena.toml` and under `/api/*`.
