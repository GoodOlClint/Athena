import XCTest

@testable import AthenaCore

/// M15.1 — RBAC catalog: bundle membership, scoping (narrow-only),
/// and the privilege-escalation guard. Security-critical, so the
/// adversarial cases are explicit.
final class RBACTests: XCTestCase {

    func testAdminIsEveryPermission() {
        XCTAssertEqual(
            RBAC.catalog["admin"], Set(Permission.allCases))
    }

    func testRoleBundlesPerSpec() {
        XCTAssertEqual(
            RBAC.catalog["member"],
            [.inference, .queueSubmit])
        XCTAssertEqual(
            RBAC.catalog["operator"],
            [
                .modelRead, .modelWrite, .inference,
                .queueSubmit, .metricsRead,
            ])
        // operator must NOT have user/token/daemon admin.
        let op = RBAC.catalog["operator"]!
        for p: Permission in [
            .usersAdmin, .tokensAdmin, .daemonAdmin,
            .usersRead,
        ] {
            XCTAssertFalse(op.contains(p), "operator leaked \(p)")
        }
        // readonly = exactly the *.read perms.
        XCTAssertEqual(
            RBAC.catalog["readonly"],
            [.modelRead, .usersRead, .metricsRead])
    }

    func testUnknownRoleContributesNothing() {
        XCTAssertFalse(RBAC.isValidRole("superuser"))
        XCTAssertEqual(
            RBAC.permissions(forRoles: ["member", "bogus"]),
            RBAC.catalog["member"])
    }

    func testTokenScopeNarrowsNeverWidens() {
        // admin user, token scoped to member ⇒ only member perms.
        let eff = RBAC.effectivePermissions(
            userRoles: ["admin"], tokenScopedRoles: ["member"])
        XCTAssertEqual(eff, RBAC.catalog["member"])
        // scope naming a role the user lacks cannot grant it:
        // user=member, token scoped to admin ⇒ still only the
        // intersection (member), NOT admin.
        let eff2 = RBAC.effectivePermissions(
            userRoles: ["member"], tokenScopedRoles: ["admin"])
        XCTAssertEqual(eff2, RBAC.catalog["member"])
        // nil scope ⇒ inherit the user's full set.
        XCTAssertEqual(
            RBAC.effectivePermissions(
                userRoles: ["operator"], tokenScopedRoles: nil),
            RBAC.catalog["operator"])
    }

    func testCanGrantEscalationGuard() {
        let adminPerms = Set(Permission.allCases)
        let memberPerms = RBAC.catalog["member"]!
        // admin can grant anything.
        XCTAssertTrue(
            RBAC.canGrant(
                role: "operator", grantorPermissions: adminPerms))
        XCTAssertTrue(
            RBAC.canGrant(
                role: "admin", grantorPermissions: adminPerms))
        // a member CANNOT grant operator/admin (lacks those perms).
        XCTAssertFalse(
            RBAC.canGrant(
                role: "operator", grantorPermissions: memberPerms))
        XCTAssertFalse(
            RBAC.canGrant(
                role: "admin", grantorPermissions: memberPerms))
        // a member CAN grant member (holds exactly those perms).
        XCTAssertTrue(
            RBAC.canGrant(
                role: "member", grantorPermissions: memberPerms))
        // unknown role is always refused.
        XCTAssertFalse(
            RBAC.canGrant(
                role: "root", grantorPermissions: adminPerms))
    }

    // MARK: - M70.3 NE8 — explicitly-empty token scope is the empty set

    /// A token whose scope is an EXPLICITLY empty role list must confer NO
    /// permissions (it is the deauthorized-token boundary): the effective set
    /// is the intersection of the user's perms with `permissions(forRoles:[])`
    /// == ∅. A nil scope, by contrast, inherits the user's full set. Pinning
    /// this guards against a future `[] coalesced with nil` refactor silently
    /// widening an emptied token back to the user's full permissions.
    func testEmptyTokenScopeYieldsNoPermissions() {
        let empty = RBAC.effectivePermissions(
            userRoles: ["admin"], tokenScopedRoles: [])
        XCTAssertTrue(
            empty.isEmpty,
            "an explicitly empty scope narrows to ∅, never the user's perms")
        // nil scope inherits the user's full set (the contrast case).
        let inherited = RBAC.effectivePermissions(
            userRoles: ["admin"], tokenScopedRoles: nil)
        XCTAssertEqual(inherited, RBAC.permissions(forRoles: ["admin"]))
        XCTAssertFalse(inherited.isEmpty)
    }
}
