#ifndef CSQLCIPHER_H
#define CSQLCIPHER_H

/* Define the codec macro BEFORE sqlite3.h so the SQLCipher key API
 * (sqlite3_key / sqlite3_rekey, guarded by SQLITE_HAS_CODEC) is part of
 * the module that Swift importers see. The amalgamation's IMPLEMENTATION
 * of these symbols is enabled separately via the target cSettings define
 * (Package.swift), which apply to compiling sqlite3.c — that define does
 * not reach module consumers, hence this header. */
#define SQLITE_HAS_CODEC 1
#include "sqlite3.h"

#endif /* CSQLCIPHER_H */
