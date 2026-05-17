import Darwin
import Foundation

/// Process-environment helpers. Substrate-agnostic (AthenaCore stays
/// dependency-free).
public enum AthenaEnv {
    /// The home directory whose `~/.athena` we should use.
    ///
    /// Under `sudo` (euid 0 with `SUDO_USER` set), `homeDirectoryFor
    /// CurrentUser` resolves to root's `/var/root` — so `athena
    /// doctor`/CLI run with sudo would look at the wrong store. Honor
    /// the invoking user instead. A daemon launched by launchd runs as
    /// its real (non-root) service user, so the normal path is used.
    public static func userHome() -> URL {
        if geteuid() == 0,
            let sudoUser = ProcessInfo.processInfo
                .environment["SUDO_USER"],
            !sudoUser.isEmpty, sudoUser != "root",
            let home = homeDirectory(ofUser: sudoUser)
        {
            return home
        }
        return FileManager.default.homeDirectoryForCurrentUser
    }

    /// `~user` from the passwd database (not `/Users/<name>`
    /// guesswork — honors relocated/network homes).
    public static func homeDirectory(ofUser user: String) -> URL? {
        guard let pw = getpwnam(user), let dir = pw.pointee.pw_dir
        else { return nil }
        return URL(
            fileURLWithPath: String(cString: dir), isDirectory: true)
    }
}
