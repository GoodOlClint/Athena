import Foundation
import SQLite3

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
}

/// One embedded SQLite store backing the built-in vector DB and the
/// async request queue (M7). Zero new dependency — system `SQLite3`.
/// Actor-isolated: SQLite access is single-threaded here.
public actor AthenaStore {
    public enum StoreError: Error, CustomStringConvertible {
        case open(String)
        case sql(String)
        public var description: String {
            switch self {
            case .open(let s): return "store open: \(s)"
            case .sql(let s): return "store sql: \(s)"
            }
        }
    }

    // nonisolated(unsafe): the actor serialises all real access; only
    // `deinit` (no live refs) reads it outside isolation, to close.
    private nonisolated(unsafe) var db: OpaquePointer?
    // SQLite asks the binding to copy (buffer is freed after the call).
    private static let transient = unsafeBitCast(
        -1, to: sqlite3_destructor_type.self)

    public init(path: URL) throws {
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
        try Self.exec(db, "PRAGMA journal_mode=WAL;")
        try Self.exec(
            db,
            """
            CREATE TABLE IF NOT EXISTS vectors(
              id TEXT PRIMARY KEY, dim INTEGER NOT NULL,
              vec BLOB NOT NULL, metadata BLOB);
            CREATE TABLE IF NOT EXISTS jobs(
              id TEXT PRIMARY KEY, kind TEXT NOT NULL,
              status TEXT NOT NULL, request BLOB NOT NULL,
              result BLOB, error TEXT,
              created REAL NOT NULL, updated REAL NOT NULL);
            """)
    }

    deinit { sqlite3_close(db) }

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

    // MARK: Vectors

    public func putVector(
        id: String, vector: [Float], metadata: Data?
    ) throws {
        let st = try Self.prepared(db,
            "INSERT OR REPLACE INTO vectors(id,dim,vec,metadata) "
                + "VALUES(?,?,?,?);")
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, id, -1, Self.transient)
        sqlite3_bind_int(st, 2, Int32(vector.count))
        Self.bindBlob(st, 3, vector.withUnsafeBytes { Data($0) })
        if let m = metadata { Self.bindBlob(st, 4, m) } else {
            sqlite3_bind_null(st, 4)
        }
        guard sqlite3_step(st) == SQLITE_DONE else {
            throw StoreError.sql(String(cString: sqlite3_errmsg(db)))
        }
    }

    private func readVector(_ st: OpaquePointer?) -> [Float] {
        guard let p = sqlite3_column_blob(st, 1) else { return [] }
        let n = Int(sqlite3_column_bytes(st, 1))
        return Data(bytes: p, count: n).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
    }

    private func readMeta(_ st: OpaquePointer?, _ i: Int32) -> Data? {
        guard let p = sqlite3_column_blob(st, i) else { return nil }
        return Data(bytes: p, count: Int(sqlite3_column_bytes(st, i)))
    }

    public func getVector(
        id: String
    ) -> (vector: [Float], metadata: Data?)? {
        guard
            let st = try? Self.prepared(db,
                "SELECT id,vec,metadata FROM vectors WHERE id=?;")
        else { return nil }
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, id, -1, Self.transient)
        guard sqlite3_step(st) == SQLITE_ROW else { return nil }
        return (readVector(st), readMeta(st, 2))
    }

    @discardableResult
    public func deleteVector(id: String) -> Bool {
        guard let st = try? Self.prepared(db,"DELETE FROM vectors WHERE id=?;")
        else { return false }
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, id, -1, Self.transient)
        return sqlite3_step(st) == SQLITE_DONE
            && sqlite3_changes(db) > 0
    }

    /// All vectors (for loading the resident MLX query matrix, M7.2).
    public func allVectors()
        -> [(id: String, vector: [Float], metadata: Data?)]
    {
        guard
            let st = try? Self.prepared(db,
                "SELECT id,vec,metadata FROM vectors ORDER BY id;")
        else { return [] }
        defer { sqlite3_finalize(st) }
        var out: [(String, [Float], Data?)] = []
        while sqlite3_step(st) == SQLITE_ROW {
            let id = String(cString: sqlite3_column_text(st, 0))
            out.append((id, readVector(st), readMeta(st, 2)))
        }
        return out
    }

    public func vectorCount() -> Int {
        guard let st = try? Self.prepared(db,"SELECT COUNT(*) FROM vectors;")
        else { return 0 }
        defer { sqlite3_finalize(st) }
        return sqlite3_step(st) == SQLITE_ROW
            ? Int(sqlite3_column_int(st, 0)) : 0
    }

    // MARK: Jobs

    public func insertJob(
        id: String, kind: String, request: Data
    ) throws {
        let now = Date().timeIntervalSince1970
        let st = try Self.prepared(db,
            "INSERT INTO jobs(id,kind,status,request,result,error,"
                + "created,updated) VALUES(?,?,'queued',?,NULL,NULL,?,?);")
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, id, -1, Self.transient)
        sqlite3_bind_text(st, 2, kind, -1, Self.transient)
        Self.bindBlob(st, 3, request)
        sqlite3_bind_double(st, 4, now)
        sqlite3_bind_double(st, 5, now)
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
        return JobRow(
            id: String(cString: sqlite3_column_text(st, 0)),
            kind: String(cString: sqlite3_column_text(st, 1)),
            status: String(cString: sqlite3_column_text(st, 2)),
            request: Self.blob(st, 3),
            result: result, error: err,
            created: sqlite3_column_double(st, 6),
            updated: sqlite3_column_double(st, 7))
    }

    private static let jobCols =
        "id,kind,status,request,result,error,created,updated"

    public func getJob(id: String) -> JobRow? {
        guard
            let st = try? Self.prepared(db,
                "SELECT \(Self.jobCols) FROM jobs WHERE id=?;")
        else { return nil }
        defer { sqlite3_finalize(st) }
        sqlite3_bind_text(st, 1, id, -1, Self.transient)
        return sqlite3_step(st) == SQLITE_ROW ? rowToJob(st) : nil
    }

    /// Jobs, optionally filtered by status, oldest first.
    public func listJobs(status: String? = nil) -> [JobRow] {
        let sql =
            status == nil
            ? "SELECT \(Self.jobCols) FROM jobs ORDER BY created;"
            : "SELECT \(Self.jobCols) FROM jobs WHERE status=? "
                + "ORDER BY created;"
        guard let st = try? Self.prepared(db,sql) else { return [] }
        defer { sqlite3_finalize(st) }
        if let s = status {
            sqlite3_bind_text(st, 1, s, -1, Self.transient)
        }
        var out: [JobRow] = []
        while sqlite3_step(st) == SQLITE_ROW { out.append(rowToJob(st)) }
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
}
