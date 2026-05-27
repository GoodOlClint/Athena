# Shipping Athena's logs off-box

Athena emits no outbound traffic by design (the passive-oracle
contract). M45.1 dropped the in-daemon RFC5424 UDP shipper that
existed briefly — it was a strictly inferior reinvention of what
purpose-built collectors already do, and operators on macOS already
have those collectors. This doc gives tested recipes for the three
most common.

All three read from the macOS unified log filtered to
`subsystem == "athena"`. None of them need root once installed; they
just need read access to the unified log (default for the running
user when invoked interactively, or specific entitlements when run
under launchd).

## FluentBit (recommended)

FluentBit 2.x+ ships a native `macos_unified_log` input. Filter to
Athena's subsystem and forward via the output of your choice.

```ini
# /usr/local/etc/fluent-bit/fluent-bit.conf
[INPUT]
    Name           macos_unified_log
    Tag            athena.*
    # `log show`-style predicate.
    Predicate      subsystem == "athena"
    # Include info+debug (default is notice+ only).
    Level          debug
    # Streaming mode (vs. batched historical).
    Stream         on

[FILTER]
    Name           parser
    Match          athena.*
    Key_Name       message
    # The Athena-emitted message body uses `key=value` pairs (req=,
    # principal=, function=, category=, …). The `logfmt` parser
    # extracts them into structured fields the output stage can route.
    Parser         logfmt
    Reserve_Data   On

[OUTPUT]
    Name           opensearch
    Match          athena.*
    Host           your-siem.internal
    Port           9200
    Index          athena-logs
    Suppress_Type_Name  On
```

Verify the input works before wiring an output:

```sh
fluent-bit -i macos_unified_log -p predicate='subsystem == "athena"' -o stdout
```

Trade-off note: FluentBit's `macos_unified_log` input was added in
2.1.x and is still listed as experimental on some platform builds.
If you see CPU pressure or dropped records under load, fall back to
the vector.dev recipe — vector.dev's Apple log integration has had
more bake time.

## vector.dev

vector.dev's `apple_unified_log` source reads the same `os.Logger`
records as FluentBit. The transform stage can split out the
Athena-emitted `key=value` fields the same way.

```toml
# /usr/local/etc/vector/vector.toml
[sources.athena]
type = "apple_unified_log"
# Same predicate language as `log show`.
predicate = 'subsystem == "athena"'
include_info = true
include_debug = false  # flip on for triage windows

[transforms.parsed]
type = "remap"
inputs = ["athena"]
source = '''
. = parse_logfmt!(.message)
'''

[sinks.siem]
type = "elasticsearch"
inputs = ["parsed"]
endpoints = ["https://your-siem.internal:9200"]
mode = "bulk"
bulk.index = "athena-logs"
```

## syslog-ng / rsyslog (via `log stream` pipe)

If your collector doesn't speak macOS unified log natively, run a
tiny launchd helper that pipes `log stream --style ndjson` into a
named-pipe / unix socket / forward target. This is more brittle than
the FluentBit/vector.dev paths — prefer those — but it works for
locked-in syslog deployments.

```sh
# /usr/local/bin/athena-log-relay.sh
#!/bin/zsh
exec /usr/bin/log stream --style ndjson \
  --predicate 'subsystem == "athena"' \
  --info --debug \
  | nc -u syslog.internal 514
```

Run under launchd with `KeepAlive` so it restarts if the collector
or the network blinks. Set the plist's `StandardErrorPath` to a
small rotating file so collector failures are noticed.

## What gets shipped

Each entry includes the standard macOS unified-log envelope (host,
pid, timestamp, level, subsystem, category) plus the Athena message
body — which itself carries M45.3's `req=<uuid>`,
`principal=<resolved>`, `function=<call-site>` fields when the
emission was inside a request task hierarchy. After a `logfmt`
parser on the collector side, those become first-class queryable
fields in your SIEM.

## What does NOT get shipped

The M30 SQLite **audit_log** table is NOT in the unified log
stream. That's the forensic trail of who-did-what (M30 design); to
get audit events off-box, query the `/api/audit` endpoint (see
[../openapi.json](../Sources/AthenaDeploy/openapi.json)) and push the rows wherever you need them.
Don't try to derive an audit trail from the unified-log entries —
they're operational, not append-only-immutable.
