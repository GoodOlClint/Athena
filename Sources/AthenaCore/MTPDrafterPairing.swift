import Foundation

/// ADR 032 — pair a generative target with its MTP speculative **drafter**
/// (Gemma 4 `gemma4_assistant`). The drafter is a separate checkpoint the
/// target's `config.json` does not advertise (verified: Gemma 4 carries no
/// `mtp_*`/`assistant_*` field, unlike Qwen's `mtp_num_hidden_layers`), so
/// pairing cannot be sniffed from metadata. It is resolved instead from an
/// explicit config key over a seeded, operator-overridable default map.
///
/// Resolution order (first hit wins): the `mtp_drafter` config key (explicit
/// override) > the default-pairing map (`Resources/mtp-drafters.toml`, overlaid
/// by `<data_dir>/mtp-drafters.toml`) > none (the `speculative` knob is inert).
///
/// The pure `resolve`/`parse` are MLX-free and unit-pinned (ADR 008/009); only
/// `defaultMap` touches the filesystem.
public enum MTPDrafterPairing {
    /// The drafter store id paired to `targetID`, or nil when none is configured.
    /// `explicit` is the `mtp_drafter` config key; a non-blank value always wins.
    /// Otherwise the target's basename (org prefix stripped, case-insensitive)
    /// is looked up in `defaults`, so a locally-converted target with the same
    /// basename still pairs.
    public static func resolve(
        targetID: String, explicit: String?, defaults: [String: String]
    ) -> String? {
        if let explicit = explicit?.trimmingCharacters(in: .whitespaces),
            !explicit.isEmpty
        {
            return explicit
        }
        return defaults[basename(targetID).lowercased()]
    }

    /// Drop an `org/name` prefix (and any path), leaving the bare model name.
    public static func basename(_ id: String) -> String {
        id.split(separator: "/").last.map(String.init) ?? id
    }

    /// Parse a `[drafters]` table of `"key" = "value"` lines into a map keyed by
    /// lowercased basename. Foundation-only (AthenaCore links no TOML lib); this
    /// one fixed shape needs no general parser. Comments (`#`), blank lines, and
    /// section headers (`[...]`) are ignored; quotes/whitespace are trimmed.
    public static func parse(_ text: String) -> [String: String] {
        var map: [String: String] = [:]
        for rawLine in text.split(
            separator: "\n", omittingEmptySubsequences: false)
        {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") || line.hasPrefix("[") {
                continue
            }
            guard let eq = line.firstIndex(of: "=") else { continue }
            let key = unquote(String(line[..<eq]))
            let value = unquote(String(line[line.index(after: eq)...]))
            guard !key.isEmpty, !value.isEmpty else { continue }
            map[basename(key).lowercased()] = value
        }
        return map
    }

    private static func unquote(_ s: String) -> String {
        var t = s.trimmingCharacters(in: .whitespaces)
        if t.count >= 2, t.hasPrefix("\""), t.hasSuffix("\"") {
            t.removeFirst()
            t.removeLast()
        }
        return t
    }

    /// The merged default map: the bundled seed (`Resources/mtp-drafters.toml`)
    /// overlaid by an optional operator override at `<data_dir>/mtp-drafters.toml`
    /// (override entries win). Either missing degrades to whatever loaded.
    public static func defaultMap(dataDir: URL?) -> [String: String] {
        var map: [String: String] = [:]
        if let seed = Bundle.module.url(
            forResource: "mtp-drafters", withExtension: "toml"),
            let text = try? String(contentsOf: seed, encoding: .utf8)
        {
            map = parse(text)
        }
        if let dataDir {
            let override = dataDir.appendingPathComponent("mtp-drafters.toml")
            if let text = try? String(contentsOf: override, encoding: .utf8) {
                for (k, v) in parse(text) { map[k] = v }
            }
        }
        return map
    }
}
