/// The portable client's own notion of the default daemon endpoint
/// port. Same value as the daemon's `GovernorConfig.defaultPort`
/// (7447 — Athena's myth-derived port; see the port-contract note),
/// duplicated here so `AthenaClient` carries NO dependency on the
/// Apple-only `AthenaCore`/`Darwin` graph and stays Linux-portable
/// (M14.3). The daemon remains the source of truth for the server
/// side; this is just the client's default to dial.
public let athenaDefaultPort = 7447
