import ArgumentParser
import Darwin
import Foundation

/// `athena hf` — manage the Hugging Face access token used for
/// gated/private model downloads (`pull`, `convert`, and `serve`'s
/// on-demand fetches). Stored in the same keyed Keychain as the
/// daemon bearer key (M13); never written to disk in plaintext.
struct Hf: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "hf",
        abstract: "Manage the Hugging Face token (Keychain).",
        subcommands: [
            HfLogin.self, HfLogout.self, HfStatus.self,
        ])
}

struct HfLogin: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "login",
        abstract:
            "Store a Hugging Face token (macOS Keychain).")
    @Option(help: "Token (omit to read stdin / prompt, no echo).")
    var token: String?

    func run() async throws {
        let t: String
        if let token, !token.isEmpty {
            t = token
        } else if isatty(0) == 0 {
            t = (readLine() ?? "").trimmingCharacters(
                in: .whitespacesAndNewlines)
        } else {
            t = String(cString: getpass("hugging face token: "))
        }
        guard !t.isEmpty else {
            FailableExit.die("error: empty token")
        }
        do {
            try HFAuth.store(t)
        } catch {
            FailableExit.die("error: \(error)")
        }
        print("stored Hugging Face token (not echoed)")
    }
}

struct HfLogout: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "logout",
        abstract: "Remove the stored Hugging Face token.")
    func run() async throws {
        print(
            HFAuth.remove()
                ? "removed stored Hugging Face token"
                : "no stored Hugging Face token")
    }
}

struct HfStatus: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "status",
        abstract:
            "Report how the HF token resolves (no secret).")
    func run() async throws {
        print("hugging face token source: \(HFAuth.source())")
    }
}
