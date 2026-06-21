import Foundation

/// Role-based access control catalog (M15). Pure value types, no
/// storage/transport coupling — the store keeps role *names* as
/// opaque strings; the server resolves a caller's roles to this
/// permission set and gates each route on a required `Permission`.
///
/// Subjects are unified: a bearer token belongs to a user; the user
/// holds roles; a token may further *narrow* (never widen) to a
/// subset of those roles. The catalog is closed and code-defined
/// (adding a capability is a deliberate, reviewed change here).

/// A `resource.action` capability. `CaseIterable` so `admin` = all.
public enum Permission: String, Sendable, CaseIterable, Hashable,
    Comparable, Codable
{
    case modelRead = "model.read"
    case modelWrite = "model.write"
    case usersRead = "users.read"
    case usersAdmin = "users.admin"
    case tokensAdmin = "tokens.admin"
    case inference = "inference"
    case metricsRead = "metrics.read"
    case daemonAdmin = "daemon.admin"

    public static func < (a: Permission, b: Permission) -> Bool {
        a.rawValue < b.rawValue
    }
}

public enum RBAC {
    /// Built-in role → permission bundle. `admin` is every
    /// permission (so a new `Permission` case is admin-granted
    /// automatically). `readonly` is exactly the `*.read` set.
    public static let catalog: [String: Set<Permission>] = [
        "admin": Set(Permission.allCases),
        "operator": [
            .modelRead, .modelWrite, .inference,
            .metricsRead,
        ],
        "member": [.inference],
        "readonly": [
            .modelRead, .usersRead, .metricsRead,
        ],
    ]

    /// Stable, sorted list of defined role names (for help/CLI).
    public static let roleNames: [String] = catalog.keys.sorted()

    public static func isValidRole(_ role: String) -> Bool {
        catalog[role] != nil
    }

    /// Effective permissions = union over the named roles. Unknown
    /// role names contribute nothing (fail-closed).
    public static func permissions(
        forRoles roles: some Sequence<String>
    ) -> Set<Permission> {
        roles.reduce(into: Set<Permission>()) {
            $0.formUnion(catalog[$1] ?? [])
        }
    }

    /// A token may scope its user's roles to a subset. The effective
    /// set is the intersection of the user's role-perms with the
    /// token's scoped role-perms (nil scope ⇒ inherit the user's
    /// full set; a scope can only NARROW, never widen).
    public static func effectivePermissions(
        userRoles: some Sequence<String>,
        tokenScopedRoles: [String]?
    ) -> Set<Permission> {
        let userPerms = permissions(forRoles: userRoles)
        guard let scoped = tokenScopedRoles else { return userPerms }
        return userPerms.intersection(permissions(forRoles: scoped))
    }

    /// Privilege-escalation guard: a grantor may assign `role` only
    /// if they ALREADY hold every permission that role confers
    /// (you cannot grant power you do not have). Unknown role ⇒
    /// refused.
    public static func canGrant(
        role: String, grantorPermissions: Set<Permission>
    ) -> Bool {
        guard let needed = catalog[role] else { return false }
        return needed.isSubset(of: grantorPermissions)
    }
}
