import Foundation

/// OpenAI `tool_choice` modes. `absent` (no field) maps to `.auto` at the DTO
/// boundary — they are the same behavior.
public enum ToolChoiceMode: Equatable, Sendable {
    case none
    case auto
    case required
    case named(String)
}

/// The decision derived from `tool_choice` + the declared tool names:
/// which tools (if any) to FORCE via the structured-output Guide, and whether
/// to advertise the tool menu to the chat template.
///
/// `forcedToolNames` empty ⇒ no Guide forcing (the model decides freely; a tool
/// call, if any, is detected by the substrate's native handler — ADR 034).
/// `advertiseMenu` false ⇒ the template renders no tools, so the model cannot
/// call one (`none`, or no tools declared).
public struct ToolChoiceResolution: Equatable, Sendable {
    public let forcedToolNames: [String]
    public let advertiseMenu: Bool
    public init(forcedToolNames: [String], advertiseMenu: Bool) {
        self.forcedToolNames = forcedToolNames
        self.advertiseMenu = advertiseMenu
    }
}

/// Resolve `tool_choice` to a force/advertise decision (ADR 034). Pure +
/// MLX-free so it is unit-pinned without the executable (ADR 008/009).
///
/// - `none` / no tools ⇒ no force, no menu (the model cannot call a tool).
/// - `auto` (the OpenAI default) ⇒ no force, menu advertised — the model
///   decides text-vs-tool each turn. This is the fix: `auto` is NOT forced.
/// - `required` ⇒ force all declared tools (a `oneOf` over the menu).
/// - `named(x)` present in the menu ⇒ force exactly that one; a name that
///   matches nothing falls through to `auto` (no force, menu advertised),
///   preserving the prior forced-name-miss behavior.
public func resolveToolChoice(
    mode: ToolChoiceMode, toolNames: [String]
) -> ToolChoiceResolution {
    guard !toolNames.isEmpty else {
        return ToolChoiceResolution(forcedToolNames: [], advertiseMenu: false)
    }
    switch mode {
    case .none:
        return ToolChoiceResolution(forcedToolNames: [], advertiseMenu: false)
    case .auto:
        return ToolChoiceResolution(forcedToolNames: [], advertiseMenu: true)
    case .required:
        return ToolChoiceResolution(
            forcedToolNames: toolNames, advertiseMenu: true)
    case .named(let n):
        return toolNames.contains(n)
            ? ToolChoiceResolution(forcedToolNames: [n], advertiseMenu: true)
            : ToolChoiceResolution(forcedToolNames: [], advertiseMenu: true)
    }
}
