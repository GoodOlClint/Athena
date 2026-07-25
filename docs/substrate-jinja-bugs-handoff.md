# Handoff: two substrate Jinja tool-schema rendering bugs

**Discovered:** 2026-06-30, validating real Claude Code against `/v1/messages`
(ADR 036 S3). **Status in Athena:** both **worked around in userland** (v0.10.236)
so Athena is unblocked; the **root cause is upstream and still latent**.

## TL;DR

Both bugs are in **`huggingface/swift-jinja`** — `Value.init(any:) throws` in
[`Sources/Jinja/Value.swift`](https://github.com/huggingface/swift-jinja/blob/main/Sources/Jinja/Value.swift)
(the Foundation→Jinja value bridge the substrate's chat-template renderer calls).
They fire when a tool's JSON-Schema is lowered to a Foundation dict and rendered
into a model's chat template. They are **dialect-independent** — they hit the
OpenAI `/v1/chat/completions` path exactly as hard as the Anthropic
`/v1/messages` path, because both lower tool specs through the same code. Our
e2e tools simply never carried the triggering shapes; Claude Code's 25 real tools
do.

- **Repos:** dep `huggingface/swift-jinja`; consumed by the substrate
  `~/Source/mlx/mlx-swift-lm` (SwiftPM path dep, see `Package.swift:71`).
- **Athena workarounds (so don't be fooled into thinking it's fixed upstream):**
  `Sources/AthenaStructured/StructuredSchema.swift` — `foundationValue()` and
  `guardingTypelessSchemaNodes()`.

The current `Value.init(any:)` switch handles `Value`, `nil`, `String`, **`Int`**,
`Double`, `Float`, `Bool`, `[Any?]`, `[String: Any?]`, `Macro`, and throws on
everything else. Both bugs are gaps in that switch.

---

## Bug 1 — `Int64` is not bridged (only `Int`)

**Symptom:** any tool whose JSON-Schema carries an integer literal
(`minimum`/`maximum`/`exclusiveMinimum`/`minLength`/`maxLength`/`minItems`/…) →
chat-template render throws → daemon returns `500 module_load_failed`
`runtime("Cannot convert value of type Int64 to Jinja Value")`.

**Root cause:** `Value.init(any:)` has `case let int as Int:` but **no `Int64`
case** (Value.swift ~line 45). On 64-bit, an `Int64` boxed as `Any` does **not**
satisfy `as? Int`, so it reaches the `default` and throws.

**Why it reached Jinja as `Int64`:** Athena's `JSONValue.foundationValue()`
lowered a JSON integer to `Int64` (to preserve >2^53 schema constants on the
JSON round-trip). Any pipeline that produces `Int64`/`Int32`/`UInt…` in a
template context hits this.

**Minimal repro (against any Athena daemon with an LLM resident):**
```sh
curl -s -o /dev/null -w '%{http_code}\n' -X POST $URL/v1/messages \
  -H 'Content-Type: application/json' -d '{"model":"<id>","max_tokens":8,
  "tools":[{"name":"t","input_schema":{"type":"object","properties":{
  "n":{"type":"integer","minimum":0}}}}],
  "messages":[{"role":"user","content":"hi"}]}'
# 500 before the fix; the `minimum:0` is the Int64. Drop it → 200.
```

**Athena workaround (v0.10.236):** `foundationValue()` lowers `.integer` to
`Int(i)` (lossless on 64-bit); the `.integer` case still keeps `Int64` for the
JSON round-trip.

**Proper upstream fix (swift-jinja):** add integer-width cases to
`Value.init(any:)`:
```swift
case let int as Int:   self = .int(int)
case let int as Int64: self = .int(Int(int))   // + Int32/UInt/UInt64 as needed
```
(Or a single `case let n as any BinaryInteger: self = .int(Int(n))`.) This makes
the bridge robust for **any** caller, not just Athena's tool path.

---

## Bug 2 — a typeless schema node crashes the render

**Symptom:** a tool property that is **exactly** `{"description": <string>}` with
**no `type`** (an "any-type" param — e.g. Claude Code's `Workflow.args`) → render
throws → `500` `runtime("Cannot convert value of type Optional<Any> to Jinja
Value")`.

**Root cause:** the gemma-4 chat template accesses a key (`type`) that is absent
on that node; the access yields an `Optional<Any>` (`.some` wrapping an optional)
that `Value.init(any:)` does **not** flatten — `case nil` only catches a literal
`nil`, and an `Optional<Any>.some(...)` of an otherwise-unhandled shape falls to
`default` and throws. (Contrast: a typed node renders fine, and a *multi-key*
typeless node renders fine — only the bare single-key described node trips it,
which is why a broad "add a type to every typeless node" pass **regresses**
working tools. See the workaround note.)

**Minimal repro:**
```sh
# 500:
... "properties":{"args":{"description":"anything"}} ...
# 200 (add a type):
... "properties":{"args":{"description":"anything","type":"string"}} ...
```

**Athena workaround (v0.10.236):** `guardingTypelessSchemaNodes()` defaults
**only** the exact single-key `{"description": <string>}` shape to
`type:"string"`. Narrow by construction: that shape crashes in isolation, so no
currently-loading tool can contain one ⇒ the transform provably touches only
already-broken nodes. A broader typeless pass was tried and **reverted** — it
regressed `Agent`/`Bash`/`AskUserQuestion`/`Monitor` (which carry multi-key
typeless nodes that render fine).

**Proper upstream fix (swift-jinja, preferred):** make `Value.init(any:)` treat
an unhandled value that is itself an `Optional` as `.null` instead of throwing —
i.e. before the `default` throw, recursively unwrap and map a `.none` (at any
nesting) to `.null`. Sketch:
```swift
// Unwrap `Any?` of `Any?` … down to a concrete value or nil, using Mirror:
func unwrap(_ v: Any?) -> Any? {
    guard let v else { return nil }
    let m = Mirror(reflecting: v)
    if m.displayStyle == .optional {
        return m.children.isEmpty ? nil : unwrap(m.children.first!.value)
    }
    return v
}
// At the top of init(any:): switch on `unwrap(value)`, so a missing template
// member (which arrives as Optional<Any>.some(nil) / nested optional) becomes
// `.null` rather than hitting `default`.
```
The principle: accessing a missing template member should degrade to Jinja
null/Undefined, never a hard `runtime` throw. This is the real robustness fix and
covers other "missing key" template paths, not just typeless tool props.

**Alternative (substrate-side, if swift-jinja can't change):** in
`mlx-swift-lm`, normalize the tool-spec dict the template receives so every
described schema node has a renderable `type` — but that re-creates Athena's
narrow guard at the substrate layer and is strictly worse than fixing the bridge.

---

## How to verify a fix

With a fixed swift-jinja / substrate, **remove** Athena's two workarounds
(`foundationValue()` `Int(i)` → `i`; drop the `guardingTypelessSchemaNodes()`
calls in `OpenAIDTO.swift` + `AnthropicDTO.swift`) and confirm:

1. `deploy/e2e-anthropic-messages.sh` → 5/5.
2. `deploy/e2e-tool-choice-auto.sh` → PASS (OpenAI path).
3. The per-tool sweep over Claude Code's 25 tools → all 200. (Capture a current
   Claude Code body once via `ANTHROPIC_BASE_URL`; each tool individually as
   `{"tools":[<tool>],"messages":[{"role":"user","content":"hi"}]}`.)
4. A real `claude -p "..."` turn completes (see `docs/claude-code.md`).

Until upstream lands, the Athena workarounds stay — they are cheap, narrow, and
regression-pinned by the e2e suites.

---

## Related (separate — NOT one of the two above)

A tool **`description`** field breaks **forced** tool generation
(`tool_choice:required`/`any`) on `gemma-4-26b-a4b-it-8bit` for **both** dialects:
the model emits a nameless `{"arguments":{…}}` and runs to `max_tokens` instead of
a clean `{"name":…,"arguments":…}`. This is a **generation/template behavior**
(not a Jinja-bridge crash), so it needs its own investigation — likely the gemma-4
tool-rendering template interacting with the forcing Guide. The proven
`deploy/e2e-tool-choice-auto.sh` sidesteps it by declaring tools **without** a
description. Tracked here so it isn't lost.
