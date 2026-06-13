import Foundation

/// The inference engine selector (`--engine` / `engine` config key): the
/// real MLX modules vs the model-free governed stubs (the CI/e2e surface).
///
/// NB4 (M70.1b): relocated here from the `athena` executable (`Commands/
/// Load.swift`) so `ConfigEditor`'s `engine ∈ Engine.allCases` validation can
/// move to the MLX-free `AthenaDeploy` and be unit-tested without dragging the
/// executable's MLX/Metal graph into the test bundle (ADR 008 follow-on). The
/// `ExpressibleByArgument` conformance (which needs ArgumentParser) stays in
/// the executable as an extension on `Load`.
public enum Engine: String, CaseIterable, Sendable {
    case mlx
    case stub
}
