# Fronting Athena with a TLS reverse proxy

Athena is a **passive oracle**: the daemon only answers inbound requests
and never initiates outbound connections (except fetching model weights
and the opt-in remote-syslog sink). That makes it a clean fit behind a
reverse proxy — the proxy terminates TLS for external clients while the
daemon stays plaintext on the loopback interface.

You have two ways to serve HTTPS. Pick one:

1. **In-daemon TLS** — point `tls_cert` / `tls_key` at PEM files in
   `athena.toml` and Athena terminates HTTPS itself. Simplest for a
   single-box appliance; no extra software. See the *Security* section
   of `deploy/athena.toml`.
2. **Reverse proxy** (this guide) — keep the daemon on loopback HTTP and
   let nginx / Caddy terminate TLS. Best when you already run a proxy,
   want managed certificates (Let's Encrypt), or terminate many services
   on one host.

Do **not** do both at once: if a proxy terminates TLS, the daemon should
speak plaintext HTTP on loopback (leave `tls_cert`/`tls_key` unset).

---

## Topology

```
client ──HTTPS:443──> reverse proxy ──HTTP──> 127.0.0.1:7447 (athenad)
        (TLS here)                   (loopback, plaintext but local)
```

Bind the daemon to loopback in `athena.toml`:

```toml
listen_host = "127.0.0.1"
listen_port = 7447
```

The loopback hop is unencrypted, but it never leaves the host. External
traffic is encrypted by the proxy.

### Security: seed auth credentials

Athena's fail-safe gate refuses to start **only** when it is bound to a
*non-loopback* address with no credentials. Behind a proxy the daemon is
bound to `127.0.0.1`, so that gate sees loopback and stays open by
default. **Anyone who can reach the proxy would then be unauthenticated.**
Always seed credentials before exposing the proxy:

```sh
athena auth user add admin --role admin       # WebUI login
athena auth add inference                      # bearer token (printed once)
```

`athena doctor` reports the auth posture; with a loopback bind it will
say "auth: disabled — loopback open (dev)" until you add credentials —
that warning is the reminder to do so.

---

## Daemon-derived constraints the proxy must honor

These come from the daemon's own limits — keep the proxy in sync:

- **Request body size (ADR 017):** JSON bodies (chat/embed) accept up to
  `max_request_body_bytes` (**default 4 MiB**); audio endpoints
  (`/v1/audio/*`) up to `max_audio_upload_bytes` (**default 100 MiB**, up
  from the old 25 MB). Set the proxy's max body to **at least the audio
  cap** (e.g. `client_max_body_size 100m`) or large uploads get a
  proxy-level 413 before they ever reach the daemon. If you also enable
  request buffering (nginx `proxy_request_buffering on`, the default), size
  the proxy's temp/buffer storage for the cap too. Over-cap at the daemon
  is a clean `413 payload_too_large`; worst-case transient daemon memory ≈
  `max_audio_upload_bytes × in-flight audio requests`, so bound concurrency
  (`--max-concurrency`) if you raise the cap much further.
- **Streaming:** SSE chat streams (`stream: true`) and the model-op
  progress streams (`POST /api/models/{pull,convert,prune}`, ADR 025) hold
  the connection open — the latter for the full duration of a multi-GB pull
  or a convert. Disable response buffering and use generous read timeouts,
  or clients see truncated/stalled streams.
- **Cold-load waits (ADR 015):** a request for a not-yet-resident model
  now BLOCKS until the model loads (up to `cold_load_wait_secs`, default
  120 s) instead of returning 503 immediately. Streamed requests emit
  `: loading` SSE keep-alive comments during the wait, so the buffering-off
  + generous-read-timeout settings above already cover them. **Non-streaming**
  cold-loads send no bytes until the load finishes, so set
  `proxy_read_timeout` (and the equivalent on other proxies) **≥
  `cold_load_wait_secs`** or a slow first load trips a proxy idle-timeout
  before the daemon's own fallback. The `1h` below is comfortably above the
  default budget.
- **Authorization header:** bearer auth and the WebUI session cookie ride
  on the request — pass `Authorization` and `Cookie` through untouched.
- **Rate / concurrency limits:** the daemon does its own per-principal
  rate limiting (`rate_limit`/`rate_burst`) and in-flight concurrency caps
  (`max_concurrency`/`max_concurrency_per_principal`), keyed by the
  authenticated principal — so you generally don't need to duplicate them
  at the proxy. When over a limit it returns **HTTP 429** with a
  `Retry-After` header; pass that response through unbuffered so clients
  can back off. (These limits are opt-in and only enforced when auth is
  enabled — see `deploy/athena.toml`.)

---

## nginx

```nginx
server {
    listen 443 ssl;
    server_name athena.example.com;

    ssl_certificate     /etc/letsencrypt/live/athena.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/athena.example.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;

    # Audio uploads go up to max_audio_upload_bytes (default 100 MiB; see
    # daemon limits above). Match or exceed the daemon cap.
    client_max_body_size 100m;

    location / {
        proxy_pass http://127.0.0.1:7447;

        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Streaming: SSE chat + model-op progress SSE. Turn off
        # buffering so tokens flush to the client immediately, and allow
        # long-lived connections.
        proxy_buffering    off;
        proxy_cache        off;
        proxy_read_timeout 1h;
        proxy_send_timeout 1h;
        proxy_http_version 1.1;
    }
}

# Optional: redirect plaintext to HTTPS.
server {
    listen 80;
    server_name athena.example.com;
    return 301 https://$host$request_uri;
}
```

---

## Caddy

Caddy provisions and renews certificates automatically (Let's Encrypt /
ZeroSSL) when `server_name` is a public domain.

```caddyfile
athena.example.com {
    # Audio uploads go up to max_audio_upload_bytes (default 100 MiB).
    request_body {
        max_size 100MB
    }

    reverse_proxy 127.0.0.1:7447 {
        # Caddy streams responses by default and preserves the
        # Authorization header — no extra config needed for SSE or auth.
        flush_interval -1
    }
}
```

For a private host with no public DNS, use Caddy's internal CA:

```caddyfile
https://athena.internal {
    tls internal
    request_body {
        max_size 100MB
    }
    reverse_proxy 127.0.0.1:7447 {
        flush_interval -1
    }
}
```

(Clients must then trust Caddy's local root, or pass `--cacert` /
`-k` for testing.)

---

## Verifying

```sh
# Health (should be 200):
curl https://athena.example.com/healthz

# Authenticated chat through the proxy:
curl https://athena.example.com/v1/chat/completions \
  -H "Authorization: Bearer sk-athena-…" \
  -H "Content-Type: application/json" \
  -d '{"model":"…","messages":[{"role":"user","content":"hi"}]}'

# Streaming flushes incrementally (no buffering):
curl -N https://athena.example.com/v1/chat/completions \
  -H "Authorization: Bearer sk-athena-…" \
  -H "Content-Type: application/json" \
  -d '{"model":"…","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

If `-N` (no buffer) shows tokens arriving in chunks, streaming is wired
correctly through the proxy.
