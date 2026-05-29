import Foundation

/// M49.5 — pre-compile complexity check for the structured-output JSON
/// schema. The outlines-core DFA compile (`oc_index_from_schema`) builds
/// an FSM whose state space scales combinatorially with nested unbounded
/// arrays nested inside a bounded outer array (a `maxItems` outer with
/// per-item `array` properties that lack `maxItems`). A real-world
/// 17 KB extraction schema with 13 such inner arrays inside a single
/// `events.maxItems=30` outer compiled to ~60 GB of rust-shim heap on
/// 2026-05-29 and pushed the daemon into 200 GB+ on a 128 GB box.
///
/// This analyzer counts the offending shapes WITHOUT invoking
/// outlines-core, so a pathological request can be refused with a
/// classified 400 (`schema_too_complex`) before the compile begins.
///
/// The check is deliberately structural, not size-based: a 1 KB schema
/// with the same shape would blow up the same way; a 50 KB schema of
/// flat fields compiles fast. The operator can lift the ceiling via the
/// `structured_max_unbounded_subarrays` TOML key, accepting the memory
/// risk.
public enum SchemaComplexity {

    /// Structural metrics computed from a JSON Schema string. All counts
    /// are global (across `$defs` and inline subschemas), since
    /// outlines-core inlines `$ref` during compile.
    public struct Analysis: Sendable, Equatable {
        /// `array` types declared with `maxItems`. ≥1 of these alongside
        /// any `unboundedInnerArrays` is the pathology.
        public var boundedArrays: Int
        /// `array` types declared WITHOUT `maxItems` that sit inside a
        /// bounded outer array (i.e. inside the `items` subtree of an
        /// `array` with `maxItems`). These are the combinatorial drivers
        /// — each one multiplies the outer's per-position state.
        public var unboundedInnerArrays: Int
        /// `array` types declared WITHOUT `maxItems` at the top level
        /// (not nested inside a bounded outer). Inert for the pathology
        /// but reported for context.
        public var unboundedTopLevelArrays: Int
        /// Names of `$defs` entries reached from the unbounded inner
        /// arrays — used to make the error message actionable (the
        /// caller can be told which definitions to add `maxItems` to).
        public var unboundedInnerArrayPaths: [String]
        /// Total schema bytes (UTF-8). Coarse, included for the error
        /// message; not a gate by itself.
        public var schemaBytes: Int

        public init(
            boundedArrays: Int = 0,
            unboundedInnerArrays: Int = 0,
            unboundedTopLevelArrays: Int = 0,
            unboundedInnerArrayPaths: [String] = [],
            schemaBytes: Int = 0
        ) {
            self.boundedArrays = boundedArrays
            self.unboundedInnerArrays = unboundedInnerArrays
            self.unboundedTopLevelArrays = unboundedTopLevelArrays
            self.unboundedInnerArrayPaths = unboundedInnerArrayPaths
            self.schemaBytes = schemaBytes
        }
    }

    /// Walk `schemaJSON` and compute the structural metrics. Returns nil
    /// if the JSON does not parse as an object — the caller should let
    /// outlines-core surface that failure itself (the gate is only for
    /// well-formed-but-pathological schemas).
    public static func analyze(_ schemaJSON: String) -> Analysis? {
        guard let data = schemaJSON.data(using: .utf8),
            let root = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else { return nil }

        var a = Analysis()
        a.schemaBytes = schemaJSON.utf8.count

        // `$defs` are inlined by outlines-core during compile, so we
        // resolve `$ref` against them while walking. A pathological
        // shape lives in EITHER the inline tree or a referenced `$def`.
        let defs = (root["$defs"] as? [String: Any]) ?? [:]
        var visiting: Set<String> = []  // cycle guard for $ref

        walk(
            root, path: "#", insideBoundedOuter: false,
            defs: defs, visiting: &visiting, into: &a)
        return a
    }

    /// Recurse the schema tree. `insideBoundedOuter` becomes true once
    /// we descend into the `items` subtree of an `array` declared with
    /// `maxItems`; from that point on, every `array` we hit without
    /// `maxItems` is an unbounded INNER array (the pathology driver).
    private static func walk(
        _ node: Any, path: String, insideBoundedOuter: Bool,
        defs: [String: Any], visiting: inout Set<String>,
        into a: inout Analysis
    ) {
        // Object subschema — the only shape we care to inspect.
        guard let obj = node as? [String: Any] else { return }

        // `$ref` follow with cycle guard. `$defs/Name` form — the only
        // form Pydantic/dataclass-style schemas emit and the only one
        // we promise to resolve. External `$ref`s fall through silently
        // (analyzed as opaque ⇒ no count contribution).
        if let ref = obj["$ref"] as? String,
            ref.hasPrefix("#/$defs/")
        {
            let name = String(ref.dropFirst("#/$defs/".count))
            if !visiting.contains(name),
                let resolved = defs[name]
            {
                visiting.insert(name)
                walk(
                    resolved,
                    path: "\(path) → $defs/\(name)",
                    insideBoundedOuter: insideBoundedOuter,
                    defs: defs, visiting: &visiting, into: &a)
                visiting.remove(name)
            }
            return
        }

        // Composition keywords — recurse each branch under the SAME
        // bounded-outer flag (anyOf inside a bounded outer is still
        // inside it).
        for key in ["anyOf", "oneOf", "allOf"] {
            if let branches = obj[key] as? [Any] {
                for (i, br) in branches.enumerated() {
                    walk(
                        br, path: "\(path)/\(key)[\(i)]",
                        insideBoundedOuter: insideBoundedOuter,
                        defs: defs, visiting: &visiting, into: &a)
                }
            }
        }

        // Type-tagged schemas (the only shapes outlines-core's compiler
        // cares about for combinatorial scaling).
        let type = obj["type"] as? String
        switch type {
        case "array":
            let hasMaxItems = obj["maxItems"] != nil
            if hasMaxItems {
                a.boundedArrays += 1
            } else if insideBoundedOuter {
                a.unboundedInnerArrays += 1
                a.unboundedInnerArrayPaths.append(path)
            } else {
                a.unboundedTopLevelArrays += 1
            }
            // Descend into `items`. Once we cross a bounded outer the
            // flag stays true for the whole subtree (per-position state
            // multiplies through every nested unbounded array).
            if let items = obj["items"] {
                walk(
                    items, path: "\(path)/items",
                    insideBoundedOuter: insideBoundedOuter || hasMaxItems,
                    defs: defs, visiting: &visiting, into: &a)
            }
        case "object":
            if let props = obj["properties"] as? [String: Any] {
                for (k, v) in props {
                    walk(
                        v, path: "\(path)/properties/\(k)",
                        insideBoundedOuter: insideBoundedOuter,
                        defs: defs, visiting: &visiting, into: &a)
                }
            }
        default:
            // Leaf types (string, number, integer, boolean, null, enum,
            // const) don't contribute to the pathology. No recursion.
            break
        }
    }

    /// The default unbounded-inner-arrays ceiling. Picked to be one
    /// step below the smallest known pathological schema (the consuming application's
    /// extraction schema at 13). An operator who needs to lift the
    /// ceiling can set `structured_max_unbounded_subarrays` in TOML.
    public static let defaultMaxUnboundedInnerArrays = 5

    /// Build an actionable error message for a schema that exceeds
    /// `max`. Lists the offending paths so the caller knows WHERE to
    /// add `maxItems` (or to remove `maxItems` from the outer array).
    public static func tooComplexReason(
        _ analysis: Analysis, max: Int
    ) -> String {
        let pathSample = analysis.unboundedInnerArrayPaths
            .prefix(8).joined(separator: ", ")
        let elision =
            analysis.unboundedInnerArrayPaths.count > 8 ? ", ..." : ""
        return
            "schema has \(analysis.unboundedInnerArrays) unbounded inner "
            + "array(s) nested inside \(analysis.boundedArrays) "
            + "`maxItems`-bounded outer array(s) (limit: \(max)). "
            + "The outlines-core DFA compile scales combinatorially in "
            + "this shape and can use tens of GB of memory. Fix by ONE "
            + "of: (1) add `maxItems` to the inner array types at: "
            + "\(pathSample)\(elision); OR (2) remove `maxItems` from "
            + "the outer array and validate the count post-decode; OR "
            + "(3) operator can raise "
            + "`structured_max_unbounded_subarrays` in the TOML config "
            + "and accept the memory risk."
    }
}
