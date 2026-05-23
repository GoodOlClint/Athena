# CSQLCipher — vendored SQLCipher amalgamation

`AthenaStore`'s SQLite engine. This is the **SQLCipher** amalgamation
(an API-compatible SQLite fork that adds transparent, page-level
AES-256 encryption via `sqlite3_key` / `PRAGMA key`) compiled in with
the **CommonCrypto** provider so there is **no OpenSSL dependency** —
the appliance stays a single self-contained binary on Apple crypto.

Without a key, SQLCipher behaves as vanilla SQLite and reads/writes the
standard on-disk format, so this vends in inert: the at-rest encryption
is opt-in (M34.3b `encrypt_store`), and a plaintext store is unchanged.

## Provenance / how to regenerate

- Upstream: https://github.com/sqlcipher/sqlcipher — tag **v4.6.1**
  (bundles SQLite 3.46.1).
- Source tarball
  `https://github.com/sqlcipher/sqlcipher/archive/refs/tags/v4.6.1.tar.gz`
  SHA-256 `d8f9afcbc2f4b55e316ca4ada4425daf3d0b4aab25f45e11a802ae422b9f53a3`
  — verify before regenerating.
- The amalgamation (`sqlite3.c` + `sqlite3.h`) is a build artifact,
  generated from that tag:

  ```sh
  ./configure --with-crypto-lib=commoncrypto --enable-tempstore=yes \
    CFLAGS="-DSQLITE_HAS_CODEC -DSQLCIPHER_CRYPTO_CC" \
    LDFLAGS="-framework Security -framework Foundation"
  make sqlite3.c        # → sqlite3.c, sqlite3.h
  ```

  Copy `sqlite3.c` to this directory and `sqlite3.h` to `include/`. The
  crypto provider (`crypto_cc.c`) is inlined into the amalgamation, so no
  other source files are needed. Do NOT vendor the configure-generated
  `sqlite_cfg.h`: it is keyed to the build host's OS and forces APIs
  newer than our deployment target (e.g. `strchrnul`, macOS 15.4+) — the
  build settings below supply the few macros we want directly instead.

## Build settings (see Package.swift)

- `SQLITE_HAS_CODEC`, `SQLCIPHER_CRYPTO_CC` — enable the codec on the
  CommonCrypto backend.
- `NDEBUG` — the production setting; compiles out the amalgamation's
  internal asserts (which reference SQLITE_DEBUG-only helpers). SwiftPM
  C targets don't define it even under `-Os`, so we set it explicitly.
- `SQLITE_TEMP_STORE=2` — keep temp tables/sorter spill in memory so an
  encrypted store never spills plaintext to a temp file on disk.
- `SQLITE_THREADSAFE=1`, `HAVE_USLEEP=1`.
- Links `Security.framework` (SecRandomCopyBytes); CommonCrypto rides in
  libSystem.
