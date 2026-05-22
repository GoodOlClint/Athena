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

- **Request body size:** chat/embed accept up to **4 MB**; audio
  endpoints (`/v1/audio/*`) up to **25 MB**. Set the proxy's max body to
  at least **25 MB** or large uploads get a proxy-level 413.
- **Streaming + long-poll:** SSE chat streams (`stream: true`), the
  queue long-poll, and `/v1/queue` SSE hold the connection open. Disable
  response buffering and use generous read timeouts, or clients see
  truncated/stalled streams.
- **Authorization header:** bearer auth and the WebUI session cookie ride
  on the request — pass `Authorization` and `Cookie` through untouched.

---

## nginx

```nginx
server {
    listen 443 ssl;
    server_name athena.example.com;

    ssl_certificate     /etc/letsencrypt/live/athena.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/athena.example.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;

    # Audio uploads go up to 25 MB (see daemon limits above).
    client_max_body_size 25m;

    location / {
        proxy_pass http://127.0.0.1:7447;

        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Streaming: SSE chat, queue long-poll, queue SSE. Turn off
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
    # Audio uploads go up to 25 MB (see daemon limits above).
    request_body {
        max_size 25MB
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
        max_size 25MB
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
