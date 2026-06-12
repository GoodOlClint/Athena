import AthenaClient
import Foundation

/// Secret input that NEVER touches argv (ADR 005 / audit B2). A password on
/// the command line leaks through shell history, `ps`, and
/// `/proc/<pid>/cmdline`; `--password` is removed entirely. A secret is
/// supplied by exactly one of, in precedence order:
///
///   1. `--password-stdin` ⇒ read ONE line from stdin (file/pipeline use,
///      e.g. `… --password-stdin < secret.txt`);
///   2. `$ATHENA_PASSWORD` ⇒ a non-interactive escape hatch for scripts;
///   3. an interactive no-echo prompt (confirmed when minting a NEW secret).
enum PasswordInput {
    static let envVar = "ATHENA_PASSWORD"

    /// Resolve a secret. `stdin` ⇒ honor `--password-stdin`. `confirmNew`
    /// ⇒ the interactive prompt asks twice and must match (for a new
    /// password); the stdin/env paths are single-source and never confirm.
    static func resolve(
        stdin: Bool, confirmNew: Bool, prompt: String = "password: "
    ) -> String {
        if stdin {
            var line = readLine(strippingNewline: true) ?? ""
            if line.hasSuffix("\r") { line.removeLast() }  // CRLF pipes
            return line
        }
        if let env = ProcessInfo.processInfo.environment[envVar],
            !env.isEmpty
        {
            return env
        }
        let a = String(cString: getpass(prompt))
        if confirmNew {
            let b = String(cString: getpass("confirm:  "))
            guard a == b else {
                FailableExit.die("error: passwords do not match")
            }
        }
        return a
    }
}
