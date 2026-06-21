import CSQLCipher
import Foundation

/// A queued/async job row.
public struct JobRow: Sendable {
    public let id: String
    public let kind: String
    /// queued | running | done | error | canceled
    public let status: String
    public let request: Data
    public let result: Data?
    public let error: String?
    public let created: Double
    public let updated: Double
    /// Submitting principal (M12.6). nil = legacy/unowned (pre-
    /// ownership rows, or auth disabled) — readable without scoping.
    public let owner: String?
}

/// NA5 (M69.1) — a blob-free job summary for list/dashboard views that
/// only need the metadata (never the request/result BLOBs). Backs the /ui
/// dashboard's recent-jobs panel so polling it doesn't materialize every
/// row's payload.
public struct JobSummary: Sendable {
    public let id: String
    public let kind: String
    public let status: String
    public let created: Double
    public let updated: Double
    public let owner: String?
}

/// One principal's cumulative token usage (M27.2).
public struct UsageRow: Sendable, Equatable {
    public let principal: String
    public let requests: Int
    public let promptTokens: Int
    public let completionTokens: Int
    /// Wall-clock of the last metered request (epoch seconds).
    public let updated: Double
    public var totalTokens: Int { promptTokens + completionTokens }
    public init(
        principal: String, requests: Int, promptTokens: Int,
        completionTokens: Int, updated: Double
    ) {
        self.principal = principal
        self.requests = requests
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.updated = updated
    }
}

/// One append-only audit-log entry (M30): a security/admin mutation —
/// who did it, to what, and the outcome. Rows are inserted, never
/// updated, so the trail is tamper-evident at the application layer.
/// One row of the M42 model allowlist: (module, id) is unique;
/// exactly one row per `module` carries `isDefault == true`.
public struct ModelAllowlistRow: Sendable, Equatable {
    public let module: String
    public let id: String
    public let isDefault: Bool
    public let declared: Double
    public init(
        module: String, id: String, isDefault: Bool, declared: Double
    ) {
        self.module = module
        self.id = id
        self.isDefault = isDefault
        self.declared = declared
    }
}

public struct AuditRow: Sendable, Equatable {
    public let id: Int
    /// Epoch seconds of the recorded event.
    public let ts: Double
    /// The acting subject: `u:<user>` / `t:<hash>` / `xenos`
    /// (auth-off loopback operator).
    public let principal: String
    /// Stable machine action key, e.g. `user.create`, `role.grant`.
    public let action: String
    /// What was acted on (username / role / model name …); nil when
    /// the action has no single subject.
    public let target: String?
    /// Outcome: `ok` (mutation applied) or `denied` (authorization
    /// guard refused it).
    public let result: String
    /// Optional human context (e.g. the denial reason).
    public let detail: String?
    public init(
        id: Int, ts: Double, principal: String, action: String,
        target: String?, result: String, detail: String?
    ) {
        self.id = id
        self.ts = ts
        self.principal = principal
        self.action = action
        self.target = target
        self.result = result
        self.detail = detail
    }
}

/// One embedded SQLite store backing auth/audit/usage and the async
/// request queue (M7). Zero new dependency — system `SQLite3`.
/// Actor-isolated: SQLite access is single-threaded here.
public actor AthenaStore {
    public enum StoreError: Error, CustomStringConvertible {
        case open(String)
        case sql(String)
        case encryption(String)
        public var description: String {
            switch self {
            case .open(let s): return "store open: \(s)"
            case .sql(let s): return "store sql: \(s)"
            case .encryption(let s): return "store encryption: \(s)"
            }
        }
    }

    /// SQLite plaintext file magic (first 16 bytes incl. the NUL). An
    /// at-rest-encrypted (SQLCipher-keyed) file does NOT start with this —
    /// its header pages are ciphertext. Used to decide whether a store
    /// needs the one-time plaintext→encrypted migration (M34.3b).
    public nonisolated static let plaintextMagic = Data(
        "SQLite format 3\u{0}".utf8)

    /// True if `path` exists and begins with the standard SQLite magic
    /// (i.e. it is an UNENCRYPTED database). A missing file or a keyed
    /// (ciphertext-header) file returns false.
    public nonisolated static func isPlaintextDatabase(
        at path: URL
    ) -> Bool {
        guard let h = try? FileHandle(forReadingFrom: path) else {
            return false
        }
        defer { try? h.close() }
        let head = (try? h.read(upToCount: plaintextMagic.count)) ?? Data()
        return head == plaintextMagic
    }

    // nonisolated(unsafe): the actor serialises all real access; only
    // `deinit` (no live refs) reads it outside isolation, to close.
    private nonisolated(unsafe) var db: OpaquePointer?
    /// The backing SQLite file. Immutable + Sendable ⇒ readable off-actor.
    public nonisolated let dbPath: URL
    // SQLite asks the binding to copy (buffer is freed after the call).
    private static let transient = unsafeBitCast(
        -1, to: sqlite3_destructor_type.self)

    /// Open (creating if needed) the SQLite store. When `key` is a
    /// non-empty passphrase the connection is keyed via SQLCipher
    /// (`sqlite3_key`) BEFORE any other statement, so the database is
    /// transparently AES-256 encrypted at rest (M34.3b). A wrong key on
    /// an existing encrypted store, or a key against a plaintext store,
    /// fails fast with `StoreError.encryption` (no silent corruption).
    /// `key == nil`/empty ⇒ a standard plaintext store (default).
    public init(path: URL, key: String? = nil) throws {
        self.dbPath = path
        try FileManager.default.createDirectory(
            at: path.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        guard
            sqlite3_open_v2(
                path.path, &db,
                SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE
                    | SQLITE_OPEN_FULLMUTEX, nil) == SQLITE_OK
        else {
            throw StoreError.open(String(cString: sqlite3_errmsg(db)))
        }
        if let key, !key.isEmpty {
            // Key MUST precede any other statement on the connection.
            var bytes = Array(key.utf8)
            // H15 (M66.1): SQLCipher derives and copies its own key from
            // these bytes during `sqlite3_key`; zero our copy afterwards so
            // the raw material doesn't linger on the heap for the process
            // lifetime. The `defer` fires on every exit (success + throws).
            defer { for i in bytes.indices { bytes[i] = 0 } }
            guard sqlite3_key(db, bytes, Int32(bytes.count)) == SQLITE_OK
            else {
                throw StoreError.encryption(
                    String(cString: sqlite3_errmsg(db)))
            }
            // Probe: the first read forces SQLCipher to derive the key and
            // decrypt page 1. A wrong key (or a plaintext file opened with
            // a key) fails here — surface it as a clear encryption error
            // rather than a confusing CREATE TABLE failure later.
            guard
                Self.exec0(db, "SELECT count(*) FROM sqlite_master;")
                    == SQLITE_OK
            else {
                throw StoreError.encryption(
                    "cannot open keyed store — wrong key, or the file is "
                        + "not SQLCipher-encrypted")
            }
        }
        try Self.exec(db, "PRAGMA journal_mode=WAL;")
        try Self.exec(
            db,
            """
            CREATE TABLE IF NOT EXISTS jobs(
              id TEXT PRIMARY KEY, kind TEXT NOT NULL,
              status TEXT NOT NULL, request BLOB NOT NULL,
              result BLOB, error TEXT,
              created REAL NOT NULL, updated REAL NOT NULL,
              owner TEXT);
            CREATE TABLE IF NOT EXISTS auth_users(
              username TEXT PRIMARY KEY, salt BLOB NOT NULL,
              hash BLOB NOT NULL, iters INTEGER NOT NULL,
              created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS auth_user_roles(
              username TEXT NOT NULL, role TEXT NOT NULL,
              PRIMARY KEY(username, role));
            CREATE TABLE IF NOT EXISTS auth_tokens(
              hash BLOB PRIMARY KEY, username TEXT NOT NULL,
              scoped_roles TEXT, label TEXT,
              created REAL NOT NULL, expires REAL);
            CREATE TABLE IF NOT EXISTS usage_counters(
              principal TEXT PRIMARY KEY,
              requests INTEGER NOT NULL DEFAULT 0,
              prompt_tokens INTEGER NOT NULL DEFAULT 0,
              completion_tokens INTEGER NOT NULL DEFAULT 0,
              updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS audit_log(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts REAL NOT NULL, principal TEXT NOT NULL,
              action TEXT NOT NULL, target TEXT,
              result TEXT NOT NULL, detail TEXT);
            CREATE INDEX IF NOT EXISTS audit_ts ON audit_log(ts);
            CREATE INDEX IF NOT EXISTS jobs_status_created
              ON jobs(status, created);
            CREATE TABLE IF NOT EXISTS model_allowlist(
              module TEXT NOT NULL, id TEXT NOT NULL,
              is_default INTEGER NOT NULL DEFAULT 0,
              declared REAL NOT NULL,
              PRIMARY KEY(module, id));
            """)
        // Migration for stores created before M12.6 (no IF NOT
        // EXISTS for columns; the dup-column error is expected and
        // ignored on already-migrated DBs).
        try? Self.exec(db, "ALTER TABLE jobs ADD COLUMN owner TEXT;")
        // M36.1: tokens gained an optional expiry. Existing rows get
        // NULL expires (no expiry) — a NULL token never expires, so
        // pre-migration tokens keep working (fail-safe / backward-
        // compatible). Only tokens minted with a TTL carry a timestamp.
        try? Self.exec(db, "ALTER TABLE auth_tokens ADD COLUMN expires REAL;")
    }

    deinit { sqlite3_close(db) }

    /// Explicitly close the SQLite connection now, rather than waiting for
    /// `deinit`. Lets a caller release the file (and its WAL/SHM locks)
    /// deterministically — e.g. before a `migrateToEncrypted` swap. The
    /// connection must not be used afterwards; `deinit`'s close becomes a
    /// no-op (`sqlite3_close(nil)`).
    public func close() {
        sqlite3_close(db)
        db = nil
    }

    /// One-time, in-place migration of a PLAINTEXT store at `path` to a
    /// SQLCipher-encrypted one keyed with `key` (M34.3b). Uses SQLCipher's
    /// `sqlcipher_export` into an attached encrypted database, then
    /// atomically swaps it in (deleting the plaintext file and its stale
    /// WAL/SHM sidecars). Idempotent at the call site: callers guard with
    /// `isPlaintextDatabase(at:)`, so an already-encrypted store is never
    /// re-migrated. Throws (leaving the original untouched) on any failure.
    public nonisolated static func migrateToEncrypted(
        at path: URL, key: String
    ) throws {
        // Keep the values safe to inline into the ATTACH literal.
        guard !key.contains("'") else {
            throw StoreError.encryption(
                "encryption key must not contain a single quote")
        }
        let enc = path.appendingPathExtension("enc-migrate")
        guard !enc.path.contains("'") else {
            throw StoreError.encryption(
                "data dir path must not contain a single quote")
        }
        let fm = FileManager.default
        try? fm.removeItem(at: enc)  // clear any interrupted prior attempt

        var db: OpaquePointer?
        // ATTACH inherits the main connection's open flags, so CREATE is
        // required here for the new encrypted file to be created.
        guard
            sqlite3_open_v2(
                path.path, &db,
                SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nil)
                == SQLITE_OK
        else {
            let m = String(cString: sqlite3_errmsg(db))
            sqlite3_close(db)
            throw StoreError.encryption("migration open: \(m)")
        }
        do {
            try exec(
                db,
                "ATTACH DATABASE '\(enc.path)' AS encrypted KEY '\(key)';")
            try exec(db, "SELECT sqlcipher_export('encrypted');")
            try exec(db, "DETACH DATABASE encrypted;")
        } catch {
            sqlite3_close(db)
            try? fm.removeItem(at: enc)
            throw StoreError.encryption("migration export: \(error)")
        }
        sqlite3_close(db)

        // NH1 (M66.1) — recoverable swap. The plaintext db is moved ASIDE
        // to a `.migrate-bak` (NOT deleted) before the encrypted copy takes
        // its place, and the backup is removed only once the move
        // succeeds. A crash or a failed `moveItem` therefore never destroys
        // data: the encrypted copy survives at `enc-migrate` and/or the
        // plaintext at `.migrate-bak`, and `recoverInterruptedMigration`
        // (run at startup before the plaintext probe) finishes or rolls
        // back the swap. The plaintext WAL/SHM are checkpointed on the
        // close above, so removing them here loses nothing.
        let bak = path.appendingPathExtension("migrate-bak")
        try? fm.removeItem(at: bak)  // clear any interrupted prior attempt
        do {
            try fm.moveItem(at: path, to: bak)  // plaintext aside, not gone
        } catch {
            try? fm.removeItem(at: enc)
            throw StoreError.encryption("migration backup: \(error)")
        }
        for ext in ["-wal", "-shm"] {
            try? fm.removeItem(at: URL(fileURLWithPath: path.path + ext))
        }
        do {
            try fm.moveItem(at: enc, to: path)
        } catch {
            // Move failed — restore the plaintext and abort. No data lost.
            try? fm.moveItem(at: bak, to: path)
            try? fm.removeItem(at: enc)
            throw StoreError.encryption("migration swap: \(error)")
        }
        try? fm.removeItem(at: bak)  // success — drop the plaintext backup
    }

    /// NH1 (M66.1) — finish or roll back a `migrateToEncrypted` that a
    /// crash interrupted mid-swap, so the store is never left without a
    /// usable database. Run at startup BEFORE `isPlaintextDatabase`. The
    /// three recoverable states (see the swap above):
    ///   - `path` missing, `enc-migrate` present → the plaintext was moved
    ///     to `.migrate-bak` but the encrypted copy hadn't landed: complete
    ///     the swap by moving `enc-migrate` into place (its content equals
    ///     the backup's), then drop the backup.
    ///   - `path` present, `enc-migrate` present → a stale encrypted copy
    ///     from a crash before the swap began: discard it; the normal
    ///     migration (if still plaintext) re-runs cleanly.
    ///   - `path` present, `.migrate-bak` present → the swap completed but
    ///     the backup deletion didn't: drop the orphaned plaintext backup.
    public nonisolated static func recoverInterruptedMigration(at path: URL)
        throws
    {
        let fm = FileManager.default
        let enc = path.appendingPathExtension("enc-migrate")
        let bak = path.appendingPathExtension("migrate-bak")
        let pathExists = fm.fileExists(atPath: path.path)
        if fm.fileExists(atPath: enc.path) {
            if !pathExists {
                do {
                    try fm.moveItem(at: enc, to: path)
                } catch {
                    throw StoreError.encryption(
                        "migration recovery (complete swap): \(error)")
                }
                try? fm.removeItem(at: bak)
            } else {
                // path already in place — the orphan is stale.
                try? fm.removeItem(at: enc)
            }
        }
        if fm.fileExists(atPath: bak.path),
            fm.fileExists(atPath: path.path)
        {
            try? fm.removeItem(at: bak)
        }
    }

    // MARK: SQLite helpers

    // Nonisolated statics (take `db`) so the actor initializer can use
    // them too; isolated methods pass `self.db`.
    private static func exec(
        _ db: OpaquePointer?, _ sql: String
    ) throws {
        var err: UnsafeMutablePointer<CChar>?
        if sqlite3_exec(db, sql, nil, nil, &err) != SQLITE_OK {
            let m = err.map { String(cString: $0) } ?? "exec failed"
            sqlite3_free(err)
            throw StoreError.sql(m)
        }
    }

    /// Non-throwing exec returning the raw SQLite status — for the keyed-
    /// open probe where we want to inspect the code, not throw a `.sql`.
    private static func exec0(
        _ db: OpaquePointer?, _ sql: String
    ) -> Int32 {
        sqlite3_exec(db, sql, nil, nil, nil)
    }

    /// Run `body` inside a single `BEGIN IMMEDIATE … COMMIT` write
    /// transaction (M66.1): a throw mid-`body` ROLLBACKs so a multi-
    /// statement mutation can't leave a half-applied state (e.g. the
    /// allowlist default clear-then-set landing with zero defaults, or a
    /// user delete cascading partway). Actor-isolated, so the single
    /// connection has no concurrent writer — this adds atomicity, not
    /// locking. Not re-entrant: SQLite rejects a nested BEGIN, so a `body`
    /// must not itself call `withTransaction`.
    private func withTransaction<T>(_ body: () throws -> T) throws -> T {
        try Self.exec(db, "BEGIN IMMEDIATE;")
        let result: T
        do {
            result = try body()
        } catch {
            try? Self.exec(db, "ROLLBACK;")
            throw error
        }
        do {
            try Self.exec(db, "COMMIT;")
        } catch {
            try? Self.exec(db, "ROLLBACK;")
            throw error
        }
        return result
    }

    private static func prepared(
        _ db: OpaquePointer?, _ sql: String
    ) throws -> OpaquePointer? {
        var st: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK
        else { throw StoreError.sql(String(cString: sqlite3_errmsg(db))) }
        return st
    }

    private static func bindBlob(
        _ st: OpaquePointer?, _ i: Int32, _ d: Data
    ) {
        d.withUnsafeBytes {
            _ = sqlite3_bind_blob(
                st, i, $0.baseAddress, Int32(d.count), transient)
        }
    }

    // MARK: Jobs

    public func insertJob(
        id: String, kind: String, request: Data, owner: String?
    ) throws {
        let now = Date().timeIntervalSince1970
        let st = try Self.prepared(db,
            "INSERT INTO jobs(id,kind,status,request,result,error,"
                + "created,updated,owner) "
                + "VALUES(?,?,'queued',?,NULL,NULL,?,?,?);")
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, id, -1, Self.transient)
        sqlite3_bind_text(st, 2, kind, -1, Self.transient)
        Self.bindBlob(st, 3, request)
        sqlite3_bind_double(st, 4, now)
        sqlite3_bind_double(st, 5, now)
        if let owner {
            sqlite3_bind_text(st, 6, owner, -1, Self.transient)
        } else { sqlite3_bind_null(st, 6) }
        guard sqlite3_step(st) == SQLITE_DONE else {
            throw StoreError.sql(String(cString: sqlite3_errmsg(db)))
        }
    }

    public func updateJob(
        id: String, status: String, result: Data?, error: String?
    ) throws {
        let st = try Self.prepared(db,
            "UPDATE jobs SET status=?,result=?,error=?,updated=? "
                + "WHERE id=?;")
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, status, -1, Self.transient)
        if let r = result { Self.bindBlob(st, 2, r) } else {
            sqlite3_bind_null(st, 2)
        }
        if let e = error {
            sqlite3_bind_text(st, 3, e, -1, Self.transient)
        } else { sqlite3_bind_null(st, 3) }
        sqlite3_bind_double(st, 4, Date().timeIntervalSince1970)
        sqlite3_bind_text(st, 5, id, -1, Self.transient)
        guard sqlite3_step(st) == SQLITE_DONE else {
            throw StoreError.sql(String(cString: sqlite3_errmsg(db)))
        }
    }

    /// A23 (M68.4) — atomically cancel a job ONLY if it is still `queued`,
    /// in ONE conditional UPDATE. The previous `getJob`-check-then-`updateJob`
    /// straddled two `await`s, so the serial worker could pick the job up
    /// (queued → running) in between and a running/done job would be clobbered
    /// back to `canceled`. The `WHERE … AND status='queued'` guard makes the
    /// check-and-act a single SQLite statement; returns true iff a queued row
    /// was actually transitioned.
    public func cancelQueuedJob(id: String) throws -> Bool {
        let st = try Self.prepared(db,
            "UPDATE jobs SET status='canceled',updated=? "
                + "WHERE id=? AND status='queued';")
        defer { sqlite3_finalize(st) }
        sqlite3_bind_double(st, 1, Date().timeIntervalSince1970)
        sqlite3_bind_text(st, 2, id, -1, Self.transient)
        guard sqlite3_step(st) == SQLITE_DONE else {
            throw StoreError.sql(String(cString: sqlite3_errmsg(db)))
        }
        return sqlite3_changes(db) > 0
    }

    /// Zero-length SQLite blobs return a NULL pointer — read safely.
    private static func blob(_ st: OpaquePointer?, _ i: Int32) -> Data {
        guard let p = sqlite3_column_blob(st, i) else { return Data() }
        return Data(bytes: p, count: Int(sqlite3_column_bytes(st, i)))
    }

    private func rowToJob(_ st: OpaquePointer?) -> JobRow {
        let result: Data? = sqlite3_column_type(st, 4) == SQLITE_NULL
            ? nil : Self.blob(st, 4)
        let err: String? = sqlite3_column_type(st, 5) == SQLITE_NULL
            ? nil : String(cString: sqlite3_column_text(st, 5))
        let owner: String? =
            sqlite3_column_type(st, 8) == SQLITE_NULL
            ? nil : String(cString: sqlite3_column_text(st, 8))
        return JobRow(
            id: String(cString: sqlite3_column_text(st, 0)),
            kind: String(cString: sqlite3_column_text(st, 1)),
            status: String(cString: sqlite3_column_text(st, 2)),
            request: Self.blob(st, 3),
            result: result, error: err,
            created: sqlite3_column_double(st, 6),
            updated: sqlite3_column_double(st, 7),
            owner: owner)
    }

    private static let jobCols =
        "id,kind,status,request,result,error,created,updated,owner"

    /// One job by id. `owner` (M65.6 / audit H6) is an optional
    /// defense-in-depth filter: when non-nil, the row matches only if its
    /// `owner` column equals it (a NULL/ownerless row never matches, so a
    /// scoped caller can't read pre-auth jobs — same outcome the server's
    /// `canAccess` enforces, pushed down a layer). nil = unfiltered (admin
    /// / worker / auth-off), preserving every existing caller.
    public func getJob(id: String, owner: String? = nil) -> JobRow? {
        let sql =
            owner == nil
            ? "SELECT \(Self.jobCols) FROM jobs WHERE id=?;"
            : "SELECT \(Self.jobCols) FROM jobs WHERE id=? AND owner=?;"
        guard let st = try? Self.prepared(db, sql) else { return nil }
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, id, -1, Self.transient)
        if let owner {
            sqlite3_bind_text(st, 2, owner, -1, Self.transient)
        }
        return sqlite3_step(st) == SQLITE_ROW ? rowToJob(st) : nil
    }

    /// Jobs, optionally filtered by status and/or `owner`, oldest first.
    /// `owner` is the H6 defense-in-depth scope (see `getJob`); nil leaves
    /// the listing unfiltered.
    public func listJobs(status: String? = nil, owner: String? = nil)
        -> [JobRow]
    {
        var conds: [String] = []
        if status != nil { conds.append("status=?") }
        if owner != nil { conds.append("owner=?") }
        var sql = "SELECT \(Self.jobCols) FROM jobs"
        if !conds.isEmpty {
            sql += " WHERE " + conds.joined(separator: " AND ")
        }
        sql += " ORDER BY created;"
        guard let st = try? Self.prepared(db, sql) else { return [] }
        defer { sqlite3_finalize(st) }
        var idx: Int32 = 1
        if let s = status {
            sqlite3_bind_text(st, idx, s, -1, Self.transient)
            idx += 1
        }
        if let o = owner {
            sqlite3_bind_text(st, idx, o, -1, Self.transient)
            idx += 1
        }
        var out: [JobRow] = []
        while sqlite3_step(st) == SQLITE_ROW { out.append(rowToJob(st)) }
        return out
    }

    /// A4/A24 (M69.1) — the single oldest job still needing work (queued or
    /// running, FIFO by `created`), fetched in ONE indexed `LIMIT 1` query
    /// rather than materializing EVERY row's request/result BLOBs just to pick
    /// the head (the worker drain ran `listJobs().first { … }`). The
    /// `ORDER BY created ASC` is explicit, so FIFO no longer rides on an
    /// unstated default ordering, and `jobs_status_created` (H9) serves it.
    public func nextPendingJob() -> JobRow? {
        let sql =
            "SELECT \(Self.jobCols) FROM jobs "
            + "WHERE status IN ('queued','running') "
            + "ORDER BY created ASC LIMIT 1;"
        guard let st = try? Self.prepared(db, sql) else { return nil }
        defer { sqlite3_finalize(st) }
        return sqlite3_step(st) == SQLITE_ROW ? rowToJob(st) : nil
    }

    /// A15 (M69.1) — count of `queued` rows via `COUNT(*)` (no blob
    /// materialization). Backs the `/healthz` queue depth. Uses
    /// `jobs_status_created` (H9).
    public func queuedJobCount() -> Int {
        guard
            let st = try? Self.prepared(db,
                "SELECT COUNT(*) FROM jobs WHERE status='queued';")
        else { return 0 }
        defer { sqlite3_finalize(st) }
        return sqlite3_step(st) == SQLITE_ROW
            ? Int(sqlite3_column_int(st, 0)) : 0
    }

    /// NA5 (M69.1) — the most recent `limit` job summaries (blob-free),
    /// newest first. Backs the /ui dashboard's recent-jobs panel so a polled
    /// `/ui/api/state` doesn't load every row's request/result BLOBs to show
    /// the last few. `limit <= 0` ⇒ empty.
    public func recentJobSummaries(limit: Int) -> [JobSummary] {
        guard limit > 0 else { return [] }
        let sql =
            "SELECT id,kind,status,created,updated,owner FROM jobs "
            + "ORDER BY created DESC LIMIT ?;"
        guard let st = try? Self.prepared(db, sql) else { return [] }
        defer { sqlite3_finalize(st) }
        sqlite3_bind_int(st, 1, Int32(limit))
        var out: [JobSummary] = []
        while sqlite3_step(st) == SQLITE_ROW {
            let owner: String? =
                sqlite3_column_type(st, 5) == SQLITE_NULL
                ? nil : String(cString: sqlite3_column_text(st, 5))
            out.append(
                JobSummary(
                    id: String(cString: sqlite3_column_text(st, 0)),
                    kind: String(cString: sqlite3_column_text(st, 1)),
                    status: String(cString: sqlite3_column_text(st, 2)),
                    created: sqlite3_column_double(st, 3),
                    updated: sqlite3_column_double(st, 4),
                    owner: owner))
        }
        return out
    }

    @discardableResult
    public func deleteJob(id: String) -> Bool {
        guard let st = try? Self.prepared(db,"DELETE FROM jobs WHERE id=?;")
        else { return false }
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, id, -1, Self.transient)
        return sqlite3_step(st) == SQLITE_DONE
            && sqlite3_changes(db) > 0
    }

    public func jobCount() -> Int {
        guard let st = try? Self.prepared(db,"SELECT COUNT(*) FROM jobs;")
        else { return 0 }
        defer { sqlite3_finalize(st) }
        return sqlite3_step(st) == SQLITE_ROW
            ? Int(sqlite3_column_int(st, 0)) : 0
    }

    /// Replace a job's stored `request` blob with empty bytes (M34.2
    /// content opt-out). The prompt is no longer needed once the job
    /// finishes; clearing it keeps the inference INPUT off disk while the
    /// result (the output the client polls for) stays, bounded by the
    /// queue TTL. `request` is NOT NULL, so we store zero-length bytes.
    public func clearJobRequest(id: String) throws {
        let st = try Self.prepared(db,
            "UPDATE jobs SET request=? WHERE id=?;")
        defer { sqlite3_finalize(st) }
        Self.bindBlob(st, 1, Data())
        sqlite3_bind_text(st, 2, id, -1, Self.transient)
        guard sqlite3_step(st) == SQLITE_DONE else {
            throw StoreError.sql(String(cString: sqlite3_errmsg(db)))
        }
    }

    /// Terminal job statuses — work is finished and the row is just a
    /// retained result. Only these are eligible for retention pruning;
    /// `queued`/`running` rows are pending work and never auto-deleted.
    static let terminalJobStatuses = "'done','error','canceled'"

    /// Delete terminal (done/error/canceled) jobs whose `updated`
    /// (completion) time is older than `cutoff` (epoch seconds). Returns
    /// the number removed. Backs the M34.1 queue-result TTL. Pending
    /// (queued/running) jobs are never touched regardless of age.
    @discardableResult
    public func pruneJobs(olderThan cutoff: Double) throws -> Int {
        let st = try Self.prepared(db,
            "DELETE FROM jobs WHERE status IN "
                + "(\(Self.terminalJobStatuses)) AND updated < ?;")
        defer { sqlite3_finalize(st) }
        sqlite3_bind_double(st, 1, cutoff)
        guard sqlite3_step(st) == SQLITE_DONE else {
            throw StoreError.sql(String(cString: sqlite3_errmsg(db)))
        }
        return Int(sqlite3_changes(db))
    }

    /// Count of terminal (done/error/canceled) job rows — the retained
    /// RESULTS the row cap governs (H13).
    private func terminalJobCount() -> Int {
        guard let st = try? Self.prepared(db,
            "SELECT COUNT(*) FROM jobs WHERE status IN "
                + "(\(Self.terminalJobStatuses));")
        else { return 0 }
        defer { sqlite3_finalize(st) }
        return sqlite3_step(st) == SQLITE_ROW
            ? Int(sqlite3_column_int(st, 0)) : 0
    }

    /// Bound the retained RESULT rows to `maxRows` by deleting the OLDEST
    /// terminal jobs first (by completion time). Returns the number
    /// removed. Backs the M34.1 queue-result row cap. Pending
    /// (queued/running) rows are live work — never deleted AND never
    /// counted toward the cap.
    ///
    /// H13 (M66.1): the excess is computed against the TERMINAL count, not
    /// `jobCount()` (all rows). The old total-based excess over-pruned
    /// whenever a pending backlog was present — e.g. maxRows 60 with 50
    /// pending + 50 terminal gave excess 40 and destroyed 40 good results,
    /// even though the 50 terminal results were already within a 60-result
    /// budget. Now a large pending queue no longer evicts finished results.
    @discardableResult
    public func trimJobs(maxRows: Int) throws -> Int {
        let excess = terminalJobCount() - maxRows
        guard maxRows > 0, excess > 0 else { return 0 }
        let st = try Self.prepared(db,
            "DELETE FROM jobs WHERE id IN ("
                + "SELECT id FROM jobs WHERE status IN "
                + "(\(Self.terminalJobStatuses)) "
                + "ORDER BY updated ASC LIMIT ?);")
        defer { sqlite3_finalize(st) }
        sqlite3_bind_int(st, 1, Int32(excess))
        guard sqlite3_step(st) == SQLITE_DONE else {
            throw StoreError.sql(String(cString: sqlite3_errmsg(db)))
        }
        return Int(sqlite3_changes(db))
    }

    // MARK: Usage metering (M27.2) — cumulative per-principal token
    // counters. Persisted so usage survives restarts and is retrievable
    // locally (pull only; the passive oracle never pushes usage out).

    /// Add one request's token counts to `principal`'s running totals,
    /// creating the row on first use. Token columns are INTEGER (64-bit)
    /// so cumulative counts don't overflow over an appliance's lifetime.
    public func addUsage(
        principal: String, promptTokens: Int, completionTokens: Int
    ) throws {
        let st = try Self.prepared(db,
            "INSERT INTO usage_counters"
                + "(principal,requests,prompt_tokens,completion_tokens,"
                + "updated) VALUES(?,1,?,?,?) "
                + "ON CONFLICT(principal) DO UPDATE SET "
                + "requests=requests+1,"
                + "prompt_tokens=prompt_tokens+excluded.prompt_tokens,"
                + "completion_tokens=completion_tokens"
                + "+excluded.completion_tokens,"
                + "updated=excluded.updated;")
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, principal, -1, Self.transient)
        sqlite3_bind_int64(st, 2, Int64(promptTokens))
        sqlite3_bind_int64(st, 3, Int64(completionTokens))
        sqlite3_bind_double(st, 4, Date().timeIntervalSince1970)
        guard sqlite3_step(st) == SQLITE_DONE else {
            throw StoreError.sql(String(cString: sqlite3_errmsg(db)))
        }
    }

    private func rowToUsage(_ st: OpaquePointer?) -> UsageRow {
        UsageRow(
            principal: String(cString: sqlite3_column_text(st, 0)),
            requests: Int(sqlite3_column_int64(st, 1)),
            promptTokens: Int(sqlite3_column_int64(st, 2)),
            completionTokens: Int(sqlite3_column_int64(st, 3)),
            updated: sqlite3_column_double(st, 4))
    }

    private static let usageCols =
        "principal,requests,prompt_tokens,completion_tokens,updated"

    /// One principal's cumulative usage, or nil if it has none yet.
    public func usage(principal: String) -> UsageRow? {
        guard let st = try? Self.prepared(db,
            "SELECT \(Self.usageCols) FROM usage_counters "
                + "WHERE principal=?;")
        else { return nil }
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, principal, -1, Self.transient)
        return sqlite3_step(st) == SQLITE_ROW ? rowToUsage(st) : nil
    }

    /// Every principal's usage, highest total tokens first (admin view).
    /// `principal` (M65.6 / audit H6) is an optional defense-in-depth
    /// filter: non-nil scopes the result to that one principal's row, so a
    /// non-admin caller can't read the whole table even if a future handler
    /// forgets to branch. nil = unfiltered (the admin view), preserving the
    /// existing caller.
    public func allUsage(principal: String? = nil) -> [UsageRow] {
        let order =
            " ORDER BY (prompt_tokens + completion_tokens) DESC, "
            + "principal;"
        let sql =
            principal == nil
            ? "SELECT \(Self.usageCols) FROM usage_counters" + order
            : "SELECT \(Self.usageCols) FROM usage_counters "
                + "WHERE principal=?" + order
        guard let st = try? Self.prepared(db, sql) else { return [] }
        defer { sqlite3_finalize(st) }
        if let principal {
            sqlite3_bind_text(st, 1, principal, -1, Self.transient)
        }
        var out: [UsageRow] = []
        while sqlite3_step(st) == SQLITE_ROW { out.append(rowToUsage(st)) }
        return out
    }

    // MARK: Audit log (M30) — append-only security/admin trail.

    /// Append one audit entry. Throwing so the caller can decide
    /// (the server treats a failed audit write as non-fatal, like
    /// usage metering — an audit hiccup must never sink the mutation
    /// that already happened).
    public func addAudit(
        principal: String, action: String, target: String?,
        result: String, detail: String? = nil
    ) throws {
        let st = try Self.prepared(db,
            "INSERT INTO audit_log"
                + "(ts,principal,action,target,result,detail) "
                + "VALUES(?,?,?,?,?,?);")
        defer { sqlite3_finalize(st) }
        sqlite3_bind_double(st, 1, Date().timeIntervalSince1970)
        sqlite3_bind_text(st, 2, principal, -1, Self.transient)
        sqlite3_bind_text(st, 3, action, -1, Self.transient)
        if let target {
            sqlite3_bind_text(st, 4, target, -1, Self.transient)
        } else { sqlite3_bind_null(st, 4) }
        sqlite3_bind_text(st, 5, result, -1, Self.transient)
        if let detail {
            sqlite3_bind_text(st, 6, detail, -1, Self.transient)
        } else { sqlite3_bind_null(st, 6) }
        guard sqlite3_step(st) == SQLITE_DONE else {
            throw StoreError.sql(String(cString: sqlite3_errmsg(db)))
        }
    }

    private static let auditCols =
        "id,ts,principal,action,target,result,detail"

    private func rowToAudit(_ st: OpaquePointer?) -> AuditRow {
        AuditRow(
            id: Int(sqlite3_column_int64(st, 0)),
            ts: sqlite3_column_double(st, 1),
            principal: String(cString: sqlite3_column_text(st, 2)),
            action: String(cString: sqlite3_column_text(st, 3)),
            target: sqlite3_column_type(st, 4) == SQLITE_NULL
                ? nil : String(cString: sqlite3_column_text(st, 4)),
            result: String(cString: sqlite3_column_text(st, 5)),
            detail: sqlite3_column_type(st, 6) == SQLITE_NULL
                ? nil : String(cString: sqlite3_column_text(st, 6)))
    }

    /// Most-recent-first audit entries, optionally narrowed by
    /// principal, action prefix, and a lower time bound (epoch
    /// seconds). `limit` caps the result page.
    public func listAudit(
        principal: String? = nil, action: String? = nil,
        since: Double? = nil, limit: Int = 100
    ) -> [AuditRow] {
        var sql = "SELECT \(Self.auditCols) FROM audit_log WHERE 1=1"
        if principal != nil { sql += " AND principal=?" }
        if action != nil { sql += " AND action=?" }
        if since != nil { sql += " AND ts>=?" }
        sql += " ORDER BY id DESC LIMIT ?;"
        guard let st = try? Self.prepared(db, sql) else { return [] }
        defer { sqlite3_finalize(st) }
        var idx: Int32 = 1
        if let principal {
            sqlite3_bind_text(st, idx, principal, -1, Self.transient)
            idx += 1
        }
        if let action {
            sqlite3_bind_text(st, idx, action, -1, Self.transient)
            idx += 1
        }
        if let since {
            sqlite3_bind_double(st, idx, since)
            idx += 1
        }
        sqlite3_bind_int(st, idx, Int32(max(1, limit)))
        var out: [AuditRow] = []
        while sqlite3_step(st) == SQLITE_ROW {
            out.append(rowToAudit(st))
        }
        return out
    }

    /// Total audit rows (tests / retention accounting).
    public func auditCount() -> Int {
        guard let st = try? Self.prepared(db,
            "SELECT COUNT(*) FROM audit_log;")
        else { return 0 }
        defer { sqlite3_finalize(st) }
        return sqlite3_step(st) == SQLITE_ROW
            ? Int(sqlite3_column_int64(st, 0)) : 0
    }

    /// Delete audit rows older than `cutoff` (epoch seconds). Returns
    /// the number removed. Backs the M30.3 age-based retention bound.
    @discardableResult
    public func pruneAudit(olderThan cutoff: Double) throws -> Int {
        let st = try Self.prepared(db,
            "DELETE FROM audit_log WHERE ts < ?;")
        defer { sqlite3_finalize(st) }
        sqlite3_bind_double(st, 1, cutoff)
        guard sqlite3_step(st) == SQLITE_DONE else {
            throw StoreError.sql(String(cString: sqlite3_errmsg(db)))
        }
        return Int(sqlite3_changes(db))
    }

    // MARK: Model allowlist (M42 — operator-declared per-module
    // selectable set, persisted across daemon restarts; one row per
    // (module, id), exactly one row per module marked `is_default`).

    public func listModelAllowlist(module: String? = nil)
        -> [ModelAllowlistRow]
    {
        let sql: String
        if module != nil {
            sql =
                "SELECT module,id,is_default,declared "
                + "FROM model_allowlist WHERE module=? "
                + "ORDER BY is_default DESC, declared ASC;"
        } else {
            sql =
                "SELECT module,id,is_default,declared "
                + "FROM model_allowlist "
                + "ORDER BY module, is_default DESC, declared ASC;"
        }
        guard let st = try? Self.prepared(db, sql) else { return [] }
        defer { sqlite3_finalize(st) }
        if let module {
            sqlite3_bind_text(st, 1, module, -1, Self.transient)
        }
        var out: [ModelAllowlistRow] = []
        while sqlite3_step(st) == SQLITE_ROW {
            out.append(
                ModelAllowlistRow(
                    module: String(cString: sqlite3_column_text(st, 0)),
                    id: String(cString: sqlite3_column_text(st, 1)),
                    isDefault: sqlite3_column_int(st, 2) != 0,
                    declared: sqlite3_column_double(st, 3)))
        }
        return out
    }

    /// Returns nil when the module's allowlist is empty.
    public func defaultModelAllowlistID(module: String) -> String? {
        let rows = listModelAllowlist(module: module)
        if let d = rows.first(where: { $0.isDefault }) { return d.id }
        return rows.first?.id
    }

    public func modelAllowlistCount(module: String) -> Int {
        guard let st = try? Self.prepared(db,
            "SELECT COUNT(*) FROM model_allowlist WHERE module=?;")
        else { return 0 }
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, module, -1, Self.transient)
        return sqlite3_step(st) == SQLITE_ROW
            ? Int(sqlite3_column_int64(st, 0)) : 0
    }

    /// Idempotent: INSERT OR IGNORE under (module, id). When
    /// `isDefault: true`, clears the prior default and sets this row.
    public func addModelAllowlist(
        module: String, id: String, isDefault: Bool = false
    ) throws {
        let ins = try Self.prepared(db,
            "INSERT OR IGNORE INTO model_allowlist"
                + "(module,id,is_default,declared) "
                + "VALUES(?,?,?,?);")
        defer { sqlite3_finalize(ins) }
        sqlite3_bind_text(ins, 1, module, -1, Self.transient)
        sqlite3_bind_text(ins, 2, id, -1, Self.transient)
        sqlite3_bind_int(ins, 3, isDefault ? 1 : 0)
        sqlite3_bind_double(ins, 4, Date().timeIntervalSince1970)
        guard sqlite3_step(ins) == SQLITE_DONE else {
            throw StoreError.sql(String(cString: sqlite3_errmsg(db)))
        }
        if isDefault {
            try setModelAllowlistDefault(module: module, id: id)
        }
    }

    @discardableResult
    public func removeModelAllowlist(module: String, id: String)
        -> Bool
    {
        guard let st = try? Self.prepared(db,
            "DELETE FROM model_allowlist WHERE module=? AND id=?;")
        else { return false }
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, module, -1, Self.transient)
        sqlite3_bind_text(st, 2, id, -1, Self.transient)
        guard sqlite3_step(st) == SQLITE_DONE else { return false }
        return sqlite3_changes(db) > 0
    }

    /// Exactly-one-default-per-module enforced by the two-step swap:
    /// clear all defaults in this module, then mark the chosen row.
    /// Throws when `id` is not in the module's allowlist (so a default
    /// can't drift to an unknown id).
    public func setModelAllowlistDefault(
        module: String, id: String
    ) throws {
        let exists = listModelAllowlist(module: module)
            .contains { $0.id == id }
        guard exists else {
            throw StoreError.sql(
                "model_allowlist: '\(id)' is not in the \(module) "
                + "allowlist")
        }
        // H2 (M66.1): the clear-then-set is one atomic swap. Without the
        // transaction, a throw/crash between the two UPDATEs leaves the
        // module with ZERO defaults (the clear committed, the set didn't),
        // which breaks default-model resolution. BEGIN IMMEDIATE …
        // COMMIT — or ROLLBACK — keeps exactly one default per module.
        try withTransaction {
            let clr = try Self.prepared(db,
                "UPDATE model_allowlist SET is_default=0 WHERE module=?;")
            defer { sqlite3_finalize(clr) }
            sqlite3_bind_text(clr, 1, module, -1, Self.transient)
            guard sqlite3_step(clr) == SQLITE_DONE else {
                throw StoreError.sql(String(cString: sqlite3_errmsg(db)))
            }
            let set = try Self.prepared(db,
                "UPDATE model_allowlist SET is_default=1 "
                    + "WHERE module=? AND id=?;")
            defer { sqlite3_finalize(set) }
            sqlite3_bind_text(set, 1, module, -1, Self.transient)
            sqlite3_bind_text(set, 2, id, -1, Self.transient)
            guard sqlite3_step(set) == SQLITE_DONE else {
                throw StoreError.sql(String(cString: sqlite3_errmsg(db)))
            }
        }
    }

    // MARK: Auth — users (PBKDF2 salt/hash computed by the caller;
    // this layer only stores opaque bytes), per-user role grants, and
    // token hashes bound to a user with an optional scoped-role
    // narrowing. Role names are opaque strings here — the RBAC
    // catalog (AthenaCore) is the sole authority on what they mean.

    public struct UserRow: Sendable {
        public let salt: Data
        public let hash: Data
        public let iters: Int
    }

    /// A managed token: the owning user plus an optional scoped-role
    /// subset (nil ⇒ inherit the user's full role set; non-nil ⇒
    /// narrow to the intersection — never widen).
    public struct TokenRow: Sendable {
        public let username: String
        public let scopedRoles: [String]?
        /// Mint time (epoch). Drives the global `token_max_age_days` cap.
        public let created: Double
        /// Per-token expiry (epoch), or nil ⇒ never expires.
        public let expires: Double?
    }

    public func putUser(
        username: String, salt: Data, hash: Data, iters: Int
    ) throws {
        let st = try Self.prepared(db,
            "INSERT OR REPLACE INTO auth_users"
                + "(username,salt,hash,iters,created) "
                + "VALUES(?,?,?,?,?);")
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, username, -1, Self.transient)
        Self.bindBlob(st, 2, salt)
        Self.bindBlob(st, 3, hash)
        sqlite3_bind_int(st, 4, Int32(iters))
        sqlite3_bind_double(st, 5, Date().timeIntervalSince1970)
        guard sqlite3_step(st) == SQLITE_DONE else {
            throw StoreError.sql(String(cString: sqlite3_errmsg(db)))
        }
    }

    public func getUser(username: String) -> UserRow? {
        guard let st = try? Self.prepared(db,
            "SELECT salt,hash,iters FROM auth_users WHERE username=?;")
        else { return nil }
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, username, -1, Self.transient)
        guard sqlite3_step(st) == SQLITE_ROW else { return nil }
        return UserRow(
            salt: Self.blob(st, 0), hash: Self.blob(st, 1),
            iters: Int(sqlite3_column_int(st, 2)))
    }

    public func listUsers() -> [String] {
        guard let st = try? Self.prepared(db,
            "SELECT username FROM auth_users ORDER BY username;")
        else { return [] }
        defer { sqlite3_finalize(st) }
        var out: [String] = []
        while sqlite3_step(st) == SQLITE_ROW {
            out.append(String(cString: sqlite3_column_text(st, 0)))
        }
        return out
    }

    /// Delete a user and cascade: their role grants and every token
    /// they own go too (no orphaned grants/tokens). Returns whether
    /// the user row existed.
    @discardableResult
    /// Delete a user and cascade its role grants + tokens. H11 (M66.1):
    /// all three DELETEs run in ONE transaction (was three autonomous
    /// statements, the first two swallowing errors) so a mid-cascade
    /// failure rolls back instead of orphaning role/token rows under a
    /// deleted user — and the error is surfaced (throws) rather than
    /// dropped. Returns whether a user row existed.
    public func deleteUser(username: String) throws -> Bool {
        try withTransaction {
            for sql in [
                "DELETE FROM auth_user_roles WHERE username=?;",
                "DELETE FROM auth_tokens WHERE username=?;",
            ] {
                let st = try Self.prepared(db, sql)
                defer { sqlite3_finalize(st) }
                sqlite3_bind_text(st, 1, username, -1, Self.transient)
                guard sqlite3_step(st) == SQLITE_DONE else {
                    throw StoreError.sql(
                        String(cString: sqlite3_errmsg(db)))
                }
            }
            let st = try Self.prepared(db,
                "DELETE FROM auth_users WHERE username=?;")
            defer { sqlite3_finalize(st) }
            sqlite3_bind_text(st, 1, username, -1, Self.transient)
            guard sqlite3_step(st) == SQLITE_DONE else {
                throw StoreError.sql(String(cString: sqlite3_errmsg(db)))
            }
            return sqlite3_changes(db) > 0
        }
    }

    public func userCount() -> Int {
        guard let st = try? Self.prepared(db,
            "SELECT COUNT(*) FROM auth_users;")
        else { return 0 }
        defer { sqlite3_finalize(st) }
        return sqlite3_step(st) == SQLITE_ROW
            ? Int(sqlite3_column_int(st, 0)) : 0
    }

    // MARK: Auth — role grants (opaque role-name strings)

    public func grantRole(username: String, role: String) throws {
        let st = try Self.prepared(db,
            "INSERT OR IGNORE INTO auth_user_roles"
                + "(username,role) VALUES(?,?);")
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, username, -1, Self.transient)
        sqlite3_bind_text(st, 2, role, -1, Self.transient)
        guard sqlite3_step(st) == SQLITE_DONE else {
            throw StoreError.sql(String(cString: sqlite3_errmsg(db)))
        }
    }

    @discardableResult
    public func revokeRole(username: String, role: String) -> Bool {
        guard let st = try? Self.prepared(db,
            "DELETE FROM auth_user_roles WHERE username=? AND role=?;")
        else { return false }
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, username, -1, Self.transient)
        sqlite3_bind_text(st, 2, role, -1, Self.transient)
        return sqlite3_step(st) == SQLITE_DONE
            && sqlite3_changes(db) > 0
    }

    public func rolesForUser(username: String) -> [String] {
        guard let st = try? Self.prepared(db,
            "SELECT role FROM auth_user_roles WHERE username=? "
                + "ORDER BY role;")
        else { return [] }
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, username, -1, Self.transient)
        var out: [String] = []
        while sqlite3_step(st) == SQLITE_ROW {
            out.append(String(cString: sqlite3_column_text(st, 0)))
        }
        return out
    }

    /// Usernames holding a given role (drives last-admin protection).
    /// Usernames holding `role`. NB11 (M66.2): THROWS on a prepare/step
    /// failure instead of returning `[]`, so a transient query error is
    /// distinguishable from "genuinely no users with this role". The
    /// last-admin guards depend on that distinction — an empty result must
    /// mean empty, never "the query broke" (which would silently bypass
    /// the protection and allow the final admin to be stripped).
    public func usersWithRole(_ role: String) throws -> [String] {
        let st = try Self.prepared(db,
            "SELECT username FROM auth_user_roles WHERE role=? "
                + "ORDER BY username;")
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, role, -1, Self.transient)
        var out: [String] = []
        while sqlite3_step(st) == SQLITE_ROW {
            out.append(String(cString: sqlite3_column_text(st, 0)))
        }
        return out
    }

    // MARK: Auth — token hashes (bound to a user; optional scope)

    /// CSV is safe: role names are RBAC catalog identifiers
    /// (`[a-z]+`), never contain a comma. nil/empty ⇒ NULL (inherit).
    private static func encodeScope(_ roles: [String]?) -> String? {
        guard let roles, !roles.isEmpty else { return nil }
        return roles.joined(separator: ",")
    }
    private static func decodeScope(_ csv: String?) -> [String]? {
        guard let csv, !csv.isEmpty else { return nil }
        return csv.split(separator: ",").map(String.init)
    }

    public func putToken(
        hash: Data, username: String, scopedRoles: [String]?,
        label: String?, expires: Double? = nil
    ) throws {
        let st = try Self.prepared(db,
            "INSERT OR REPLACE INTO auth_tokens"
                + "(hash,username,scoped_roles,label,created,expires) "
                + "VALUES(?,?,?,?,?,?);")
        defer { sqlite3_finalize(st) }
        Self.bindBlob(st, 1, hash)
        sqlite3_bind_text(st, 2, username, -1, Self.transient)
        if let scope = Self.encodeScope(scopedRoles) {
            sqlite3_bind_text(st, 3, scope, -1, Self.transient)
        } else { sqlite3_bind_null(st, 3) }
        if let label {
            sqlite3_bind_text(st, 4, label, -1, Self.transient)
        } else { sqlite3_bind_null(st, 4) }
        sqlite3_bind_double(st, 5, Date().timeIntervalSince1970)
        if let expires {
            sqlite3_bind_double(st, 6, expires)
        } else { sqlite3_bind_null(st, 6) }
        guard sqlite3_step(st) == SQLITE_DONE else {
            throw StoreError.sql(String(cString: sqlite3_errmsg(db)))
        }
    }

    /// The owning user + scoped roles for an exact token-hash
    /// (SHA-256 bytes). Indexed PK lookup — the presented value is
    /// already a hash, so a byte-probing timing attack is infeasible.
    public func tokenPrincipal(hash: Data) -> TokenRow? {
        guard let st = try? Self.prepared(db,
            "SELECT username,scoped_roles,created,expires FROM "
                + "auth_tokens WHERE hash=?;")
        else { return nil }
        defer { sqlite3_finalize(st) }
        Self.bindBlob(st, 1, hash)
        guard sqlite3_step(st) == SQLITE_ROW else { return nil }
        let user = String(cString: sqlite3_column_text(st, 0))
        let scope =
            sqlite3_column_type(st, 1) == SQLITE_NULL
            ? nil : String(cString: sqlite3_column_text(st, 1))
        let created = sqlite3_column_double(st, 2)
        let expires =
            sqlite3_column_type(st, 3) == SQLITE_NULL
            ? nil : sqlite3_column_double(st, 3)
        return TokenRow(
            username: user, scopedRoles: Self.decodeScope(scope),
            created: created, expires: expires)
    }

    /// Number of hex chars of the token-hash shown for display/matching —
    /// enough to be an unambiguous handle, far short of the full digest.
    public static let tokenHashPrefixLen = 12

    /// Token listing for DISPLAY. H12 (M66.2): returns only a 12-hex
    /// `hashPrefix`, never the full SHA-256 digest — the at-rest credential
    /// digest no longer leaves the store for a listing. The rm/rotate
    /// paths, which legitimately need the full hash to delete a row, use
    /// `tokensMatchingHashPrefix` instead.
    public func listTokens() -> [(hashPrefix: String, username: String,
        scoped: [String]?, label: String?, expires: Double?)]
    {
        rowsForTokenQuery(
            "SELECT hash,username,scoped_roles,label,expires FROM "
                + "auth_tokens ORDER BY created;",
            prefix: nil
        ).map {
            (String($0.hex.prefix(Self.tokenHashPrefixLen)),
                $0.username, $0.scoped, $0.label, $0.expires)
        }
    }

    /// Tokens whose lowercase hash-hex starts with `prefix` (H12 / M66.2),
    /// WITH the full hash — for the credential-mutation paths (`auth rm` /
    /// `auth token rotate`) that must reconstruct the row's hash to delete
    /// it. Confines the full digest to that path; display goes through
    /// `listTokens`. `prefix` is matched case-insensitively and is
    /// expected to be hex (callers validate), so no LIKE metacharacters.
    public func tokensMatchingHashPrefix(_ prefix: String)
        -> [(hex: String, username: String, scoped: [String]?,
            label: String?, expires: Double?)]
    {
        rowsForTokenQuery(
            "SELECT hash,username,scoped_roles,label,expires FROM "
                + "auth_tokens WHERE lower(hex(hash)) LIKE ? "
                + "ORDER BY created;",
            prefix: prefix.lowercased() + "%")
    }

    /// Shared row reader for the token queries above. When `prefix` is
    /// non-nil it is bound to the single `?` (a `LIKE` pattern).
    private func rowsForTokenQuery(_ sql: String, prefix: String?)
        -> [(hex: String, username: String, scoped: [String]?,
            label: String?, expires: Double?)]
    {
        guard let st = try? Self.prepared(db, sql) else { return [] }
        defer { sqlite3_finalize(st) }
        if let prefix {
            sqlite3_bind_text(st, 1, prefix, -1, Self.transient)
        }
        var out: [(String, String, [String]?, String?, Double?)] = []
        while sqlite3_step(st) == SQLITE_ROW {
            let h = Self.blob(st, 0)
                .map { String(format: "%02x", $0) }.joined()
            let user = String(cString: sqlite3_column_text(st, 1))
            let scope =
                sqlite3_column_type(st, 2) == SQLITE_NULL
                ? nil : String(cString: sqlite3_column_text(st, 2))
            let label =
                sqlite3_column_type(st, 3) == SQLITE_NULL
                ? nil : String(cString: sqlite3_column_text(st, 3))
            let expires =
                sqlite3_column_type(st, 4) == SQLITE_NULL
                ? nil : sqlite3_column_double(st, 4)
            out.append(
                (h, user, Self.decodeScope(scope), label, expires))
        }
        return out
    }

    @discardableResult
    public func deleteToken(hash: Data) -> Bool {
        guard let st = try? Self.prepared(db,
            "DELETE FROM auth_tokens WHERE hash=?;")
        else { return false }
        defer { sqlite3_finalize(st) }
        Self.bindBlob(st, 1, hash)
        return sqlite3_step(st) == SQLITE_DONE
            && sqlite3_changes(db) > 0
    }

    public func tokenCount() -> Int {
        guard let st = try? Self.prepared(db,
            "SELECT COUNT(*) FROM auth_tokens;")
        else { return 0 }
        defer { sqlite3_finalize(st) }
        return sqlite3_step(st) == SQLITE_ROW
            ? Int(sqlite3_column_int(st, 0)) : 0
    }

    // MARK: Backup

    /// Write a fully self-contained, defragmented snapshot of the live
    /// DB to `dest` via `VACUUM INTO` — checkpoints the WAL into the
    /// copy, so it is safe while the daemon keeps serving. `dest` must
    /// not already exist (SQLite refuses to overwrite); the caller
    /// removes any stale file first.
    public func backup(to dest: URL) throws {
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.createDirectory(
            at: dest.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let st = try Self.prepared(db, "VACUUM INTO ?;")
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, dest.path, -1, Self.transient)
        guard sqlite3_step(st) == SQLITE_DONE else {
            throw StoreError.sql(String(cString: sqlite3_errmsg(db)))
        }
    }
}
