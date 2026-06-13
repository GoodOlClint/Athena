//! Athena structured-output shim.
//!
//! A C-ABI `staticlib` over `llguidance` (guidance-ai): a lazy/incremental
//! grammar-constrained decoding engine. Compile a JSON schema into a
//! grammar once per model+schema, then walk it per request, asking for the
//! per-step allowed-token mask and advancing on committed tokens.
//!
//! This replaces the previous `outlines-core` full-DFA precompile (M53).
//! outlines built the entire schema→regex→DFA over the full vocabulary
//! BEFORE the first token, so a `maxItems`-bounded outer array over a large
//! item subschema unrolled into a state×vocab cross-product (~150 GB of
//! heap on a 128 GB box). llguidance parses incrementally — bounded
//! repetition is a counter in the parser, not unrolled states — so the
//! same schema constructs in <0.3 s at <0.1 GB.
//!
//! ABI shape is unchanged so the Swift `AthenaStructured` surface barely
//! moves. Mapping onto llguidance:
//!   - `OcVocab`  — the `TokEnv` (token trie built from the model's own
//!     id→bytes) plus the per-model `ParserFactory` (builds the vocab
//!     slicer once; this is the ~0.24 s, schema-independent cost). Cache
//!     it per model on the Swift side.
//!   - `OcIndex`  — the schema (held as a parsed JSON value) + a shared
//!     handle on the factory. Cheap to build; per request is fine.
//!   - `OcGuide`  — a fresh, stateful `TokenParser` for one request.
//!
//! The previous engine's rollback ring is GONE: Athena only ever advances
//! the guide on COMMITTED tokens (monotonic), so rollback was dead code.
//! `oc_guide_rollback`/`oc_guide_allowed_rollback` remain in the ABI as
//! no-ops for compatibility. The regex `Index` path is likewise unused and
//! now returns an error.
//!
//! Error convention: fallible calls return NULL / -1 / false and stash a
//! message in a thread-local; the caller reads `oc_last_error`.
//!
//! ## Hardening (M65.1 — audit G1/G2/G3/G6/G10)
//!
//! The schema string crosses the FFI from a *remote* caller (`/v1`
//! `response_format.json_schema`), and the token arrays come from the
//! model tokenizer. Both are validated before they reach llguidance:
//!   - **G1** every `extern "C"` entry runs inside [`ffi_guard`], which
//!     `catch_unwind`s the body. A panic in llguidance (or anywhere) is
//!     turned into the normal NULL/-1/false error return instead of
//!     unwinding across the C ABI — which is UB and aborts the process
//!     (remote DoS). The crate stays `panic = "unwind"` (no `abort`) so the
//!     catch is live; see [`ffi_guard`].
//!   - **G2/G6** [`build_words`] caps the token count and `max_id+1` dense
//!     allocation (a `0xFFFF_FFFF` id would otherwise reserve ~4 GB) and
//!     caps each token's byte length before the unchecked pointer read.
//!   - **G3** [`validate_schema`] caps raw schema bytes, nesting depth, and
//!     repetition bounds before compile.
//!   - **G10** `id < size` is enforced before every dense index, a
//!     too-small mask buffer fails loud instead of silently truncating, and
//!     [`set_err`] is interior-NUL-safe and panic-reentrancy-safe.

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::os::raw::c_int;
use std::sync::Arc;

use llguidance::api::TopLevelGrammar;
use llguidance::{ParserFactory, TokenParser};
use toktrie::{ApproximateTokEnv, SimpleVob, TokEnv, TokRxInfo, TokTrie};

// ---- input caps (G2/G3/G6) ------------------------------------------
//
// All comfortably above any real workload, so legitimate callers never
// hit them; they exist purely to bound a hostile or corrupt input.

/// Max vocabulary size — bounds the `max_id+1` dense `Vec<Vec<u8>>` in
/// [`build_words`]. ~16× the largest real tokenizer (~262 K for Gemma);
/// the worst-case dense alloc is `MAX_VOCAB * size_of::<Vec<u8>>()` ≈ 96 MB,
/// vs the ~4 GB a 32-bit-max id would otherwise reserve (G2).
const MAX_VOCAB: usize = 1 << 22; // 4,194,304

/// Max decoded bytes for a single token. Real BPE merges are well under
/// 100 bytes; this only rejects an absurd caller-supplied length before the
/// raw-pointer read (G6).
const MAX_TOKEN_BYTES: usize = 4096;

/// Max raw schema length in bytes (G3). Real schemas are a few KB; 1 MiB is
/// far past any legitimate `response_format`.
const MAX_SCHEMA_BYTES: usize = 1 << 20; // 1 MiB

/// Max JSON nesting depth for a schema (G3). Deep nesting risks stack
/// blow-up in the grammar compiler; serde already caps at 128, this is
/// stricter and explicit.
const MAX_SCHEMA_DEPTH: usize = 64;

/// Max value for a repetition bound key (`maxItems`, `minItems`, etc.) in a
/// schema (G3). 100 K bounded repetitions is already absurd for structured
/// LLM output; rejecting larger keeps the parser's repetition counters and
/// any derived grammar state bounded.
const MAX_REPETITION: u64 = 100_000;

thread_local! {
    static LAST_ERR: RefCell<CString> = RefCell::new(CString::default());
}

/// Stash a message for the next `oc_last_error`. Interior-NUL-safe (a NUL in
/// an llguidance message would otherwise truncate the C string or make
/// `CString::new` fail and silently drop the whole message) and reentrancy-
/// safe (`try_borrow_mut` so a panic-during-panic that re-enters here can't
/// double-panic into a process abort) — G10.
fn set_err(msg: impl Into<Vec<u8>>) {
    let mut bytes = msg.into();
    for b in bytes.iter_mut() {
        if *b == 0 {
            *b = b'?';
        }
    }
    let c = CString::new(bytes).unwrap_or_default();
    LAST_ERR.with(|e| {
        if let Ok(mut slot) = e.try_borrow_mut() {
            *slot = c;
        }
    });
}

unsafe fn cstr<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    CStr::from_ptr(p).to_str().ok()
}

// ---- panic boundary (G1) --------------------------------------------

/// Run an FFI entry body inside `catch_unwind` so a panic NEVER unwinds
/// across the C ABI. Unwinding past `extern "C"` is undefined behaviour and
/// in practice aborts the whole daemon — a remote DoS, since the schema a
/// `/v1` caller supplies is what reaches llguidance. On a caught panic we
/// stash a message and return `default` (the function's normal failure
/// sentinel: NULL / -1 / false / 0), so the caller sees an ordinary error
/// and the daemon stays up. Requires `panic = "unwind"` (the default — we
/// deliberately do NOT set `panic = "abort"`, which would make the catch a
/// no-op and re-expose the abort).
/// The body is wrapped in `AssertUnwindSafe`: most handles carry an
/// `llguidance::TokenParser` (an `Arc<dyn BiasComputer>` inside), which is
/// not `UnwindSafe`. The assertion is sound here because a caught panic is
/// reported to the caller as an ordinary failure (NULL/-1/false) and the
/// handle is discarded on that path — we never observe post-panic state.
fn ffi_guard<F, R>(label: &str, default: R, f: F) -> R
where
    F: FnOnce() -> R,
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(payload) => {
            set_err(format!(
                "{label}: panic caught at FFI boundary: {}",
                panic_message(payload)
            ));
            default
        }
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "unknown panic".to_string()
}

// ---- schema validation (G3) -----------------------------------------

fn is_repetition_key(k: &str) -> bool {
    matches!(
        k,
        "maxItems"
            | "minItems"
            | "maxLength"
            | "minLength"
            | "maxProperties"
            | "minProperties"
            | "maxContains"
            | "minContains"
    )
}

/// Parse and bound a caller schema before it reaches the grammar compiler:
/// raw byte length, nesting depth, and repetition-bound magnitude (G3). The
/// depth/repetition walk is iterative (explicit stack) so the validator
/// itself can't be made to overflow by a deep input.
fn validate_schema(json: &str) -> Result<serde_json::Value, String> {
    if json.len() > MAX_SCHEMA_BYTES {
        return Err(format!(
            "schema too large: {} bytes > {} cap",
            json.len(),
            MAX_SCHEMA_BYTES
        ));
    }
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("invalid JSON schema: {e}"))?;

    let mut stack: Vec<(&serde_json::Value, usize)> = vec![(&value, 1)];
    while let Some((v, depth)) = stack.pop() {
        if depth > MAX_SCHEMA_DEPTH {
            return Err(format!("schema nesting too deep: > {MAX_SCHEMA_DEPTH}"));
        }
        match v {
            serde_json::Value::Object(map) => {
                for (k, child) in map {
                    if is_repetition_key(k) {
                        if let Some(n) = child.as_u64() {
                            if n > MAX_REPETITION {
                                return Err(format!(
                                    "schema repetition bound {k}={n} > {MAX_REPETITION} cap"
                                ));
                            }
                        }
                    }
                    stack.push((child, depth + 1));
                }
            }
            serde_json::Value::Array(arr) => {
                for child in arr {
                    stack.push((child, depth + 1));
                }
            }
            _ => {}
        }
    }
    Ok(value)
}

// Athena caches one `OcVocab` per model and calls `oc_index_from_schema` /
// `oc_guide_new` (which clone the factory `Arc` and `create_parser`) from
// concurrent request handlers. That is only sound if the shared factory is
// `Send + Sync`; assert it at compile time so a future llguidance bump that
// regresses this fails the build rather than racing at runtime.
#[allow(dead_code)]
fn _assert_factory_thread_safe() {
    fn is_send_sync<T: Send + Sync>() {}
    is_send_sync::<Arc<ParserFactory>>();
}

// ---- opaque handles -------------------------------------------------

/// Per-model vocabulary + parser factory. The factory build (the vocab
/// slicer) is the only non-trivial cost (~0.24 s) and is schema-
/// independent — cache one of these per loaded model.
pub struct OcVocab {
    factory: Arc<ParserFactory>,
    /// Total token count in the trie (incl. eos); mask length is
    /// `ceil(vocab_size / 8)`.
    vocab_size: usize,
}

/// A compiled schema bound to a model's factory. Cheap to build; holds the
/// parsed schema so each guide gets a fresh parser.
pub struct OcIndex {
    factory: Arc<ParserFactory>,
    grammar: TopLevelGrammar,
    vocab_size: usize,
}

/// A stateful per-request walker over an `OcIndex`. Caches the mask for the
/// current parser state so an `allowed_mask` immediately followed by an
/// `advance` of a token taken from that mask costs a single compute.
pub struct OcGuide {
    parser: TokenParser,
    vocab_size: usize,
    /// Mask valid for the CURRENT parser state; invalidated on every
    /// committed token. `None` ⇒ recompute on next use.
    cached_mask: Option<SimpleVob>,
}

impl OcGuide {
    fn mask_len(&self) -> usize {
        self.vocab_size.div_ceil(8)
    }

    /// Ensure `cached_mask` holds the mask for the current state.
    fn ensure_mask(&mut self) -> bool {
        if self.cached_mask.is_none() {
            match self.parser.compute_mask() {
                Ok(m) => self.cached_mask = Some(m),
                Err(e) => {
                    set_err(format!("compute_mask: {e}"));
                    return false;
                }
            }
        }
        true
    }

    /// Fill `buf` (one bit per token, bit `i` of byte `i>>3`) with the
    /// tokens allowed from the current state. A buffer shorter than
    /// `mask_len` fails LOUD (returns false) rather than writing a
    /// truncated mask — a short mask reads as "high token ids disallowed",
    /// which silently under-constrains decoding and could let through a
    /// token the schema forbids (G10).
    fn allowed_mask(&mut self, buf: &mut [u8]) -> bool {
        let need = self.mask_len();
        if buf.len() < need {
            set_err(format!(
                "allowed_mask: buffer {} < required {} bytes",
                buf.len(),
                need
            ));
            return false;
        }
        for b in buf.iter_mut() {
            *b = 0;
        }
        if !self.ensure_mask() {
            return false;
        }
        let mask = self.cached_mask.as_ref().unwrap();
        // `write_to` asserts `buf.len() <= words*4`; clamp to both the
        // required byte length and the mask's word capacity (the mask may be
        // shorter than `need` when the vocab isn't a multiple of the word
        // size — the trailing bytes stay zero = disallowed, which is safe).
        let cap = mask.as_slice().len() * 4;
        let n = need.min(cap);
        mask.write_to(&mut buf[..n]);
        true
    }

    /// Advance on `token`. Returns false (no mutation) when the token has
    /// no transition from the current state — this gives the IDLE-probe
    /// caller the "failed advance doesn't change state" guarantee, since
    /// `consume_token` would otherwise error on a disallowed token.
    fn advance(&mut self, token: u32) -> bool {
        if token as usize >= self.vocab_size {
            // G9: an out-of-range token id is a CALLER contract violation,
            // distinct from a token the schema merely disallows. Record it
            // on the error channel so it is not silently conflated with a
            // normal guide rejection (the disallowed case below returns a
            // clean `false` with NO error set, so the two are separable).
            set_err(format!(
                "advance: token {} out of range (vocab_size {})",
                token, self.vocab_size
            ));
            return false;
        }
        if !self.ensure_mask() {
            return false; // ensure_mask sets its own error on failure
        }
        let allowed = self.cached_mask.as_ref().unwrap().is_allowed(token);
        if !allowed {
            // Legitimate schema rejection: a clean false, error channel
            // left untouched (G9) — the IDLE-probe caller relies on this.
            return false;
        }
        match self.parser.consume_token(token) {
            Ok(_) => {
                self.cached_mask = None; // state changed
                true
            }
            Err(e) => {
                set_err(format!("consume_token: {e}"));
                false
            }
        }
    }

    fn is_final(&mut self) -> bool {
        self.parser.is_accepting()
    }
}

// ---- vocab / factory construction -----------------------------------

/// Build the `words` table (token id → bytes) for the trie. Non-eos tokens
/// carry the model's own decoded bytes; the eos id and any gap slots are
/// marked with a leading `0xFF`, toktrie's convention for a control token
/// that is never matched as document content.
///
/// Hardened (G2/G6/G10): the token count, the `max_id+1` dense size, and
/// each token's byte length are all capped before any allocation or
/// unchecked pointer read, and every dense index is `id < size` guarded.
unsafe fn build_words(
    ids: *const u32,
    byte_ptrs: *const *const u8,
    byte_lens: *const usize,
    n: usize,
    eos: u32,
) -> Option<Vec<Vec<u8>>> {
    if ids.is_null() || byte_ptrs.is_null() || byte_lens.is_null() {
        set_err("build_words: null array");
        return None;
    }
    // G6: bound the caller-supplied token count before any per-token read.
    if n > MAX_VOCAB {
        set_err(format!(
            "build_words: token count {n} exceeds {MAX_VOCAB} cap"
        ));
        return None;
    }
    let mut max_id = eos;
    for i in 0..n {
        max_id = max_id.max(*ids.add(i));
    }
    let size = max_id as usize + 1;
    // G2: a hostile/corrupt id (e.g. 0xFFFF_FFFF) would otherwise reserve a
    // multi-GB dense table — OOM DoS. Reject before the allocation.
    if size > MAX_VOCAB {
        set_err(format!(
            "build_words: max token id {max_id} exceeds {} vocab cap",
            MAX_VOCAB - 1
        ));
        return None;
    }
    let mut words: Vec<Vec<u8>> = vec![Vec::new(); size];
    let mut filled = vec![false; size];

    for i in 0..n {
        let id = *ids.add(i) as usize;
        // G10: guaranteed by `size = max_id+1`, but never index OOB even if
        // that invariant is ever broken.
        if id >= size {
            continue;
        }
        if id == eos as usize {
            continue; // eos is a control token, set below
        }
        let p = *byte_ptrs.add(i);
        let l = *byte_lens.add(i);
        if p.is_null() {
            continue;
        }
        // G6: cap the caller-supplied byte length before the unchecked read.
        if l > MAX_TOKEN_BYTES {
            set_err(format!(
                "build_words: token {i} byte length {l} exceeds {MAX_TOKEN_BYTES} cap"
            ));
            return None;
        }
        words[id] = std::slice::from_raw_parts(p, l).to_vec();
        filled[id] = true;
    }

    // eos + any unfilled slot → unique 0xFF-marked control tokens (never
    // valid document content, never zero-length).
    for (i, slot) in words.iter_mut().enumerate() {
        if i == eos as usize || !filled[i] {
            let mut b = vec![0xFFu8];
            b.extend_from_slice(&(i as u32).to_le_bytes());
            *slot = b;
        }
    }
    Some(words)
}

unsafe fn build_factory(
    ids: *const u32,
    byte_ptrs: *const *const u8,
    byte_lens: *const usize,
    n: usize,
    eos: u32,
) -> Option<(Arc<ParserFactory>, usize)> {
    let words = build_words(ids, byte_ptrs, byte_lens, n, eos)?;
    let vocab_size = words.len();
    let info = TokRxInfo::new(vocab_size as u32, eos);
    let trie = TokTrie::from(&info, &words);
    let tok_env: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
    match ParserFactory::new_simple(&tok_env) {
        Ok(f) => Some((Arc::new(f), vocab_size)),
        Err(e) => {
            set_err(format!("ParserFactory::new_simple: {e}"));
            None
        }
    }
}

// ---- misc -----------------------------------------------------------

#[no_mangle]
pub extern "C" fn oc_version() -> *const c_char {
    ffi_guard("oc_version", std::ptr::null(), || {
        c"llguidance-1.7.5".as_ptr()
    })
}

#[no_mangle]
pub unsafe extern "C" fn oc_last_error(buf: *mut c_char, len: usize) -> usize {
    ffi_guard("oc_last_error", 0, move || {
        LAST_ERR.with(|e| {
            let e = e.borrow();
            let bytes = e.as_bytes_with_nul();
            if !buf.is_null() && len > 0 {
                let n = bytes.len().min(len);
                std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, n);
                if n == len {
                    *buf.add(len - 1) = 0;
                }
            }
            bytes.len()
        })
    })
}

// ---- vocabulary -----------------------------------------------------

/// Build a per-model vocabulary + parser factory from parallel arrays of
/// token ids and their decoded UTF-8/byte forms (the caller — Athena's
/// swift-transformers tokenizer — produces the byte mapping so it matches
/// the model exactly). `eos` is registered as the stop token.
#[no_mangle]
pub unsafe extern "C" fn oc_vocab_new_from_tokens(
    ids: *const u32,
    byte_ptrs: *const *const u8,
    byte_lens: *const usize,
    n: usize,
    eos: u32,
) -> *mut OcVocab {
    ffi_guard(
        "oc_vocab_new_from_tokens",
        std::ptr::null_mut(),
        move || match build_factory(ids, byte_ptrs, byte_lens, n, eos) {
            Some((factory, vocab_size)) => {
                Box::into_raw(Box::new(OcVocab { factory, vocab_size }))
            }
            None => std::ptr::null_mut(),
        },
    )
}

#[no_mangle]
pub unsafe extern "C" fn oc_vocab_free(v: *mut OcVocab) {
    ffi_guard("oc_vocab_free", (), move || {
        if !v.is_null() {
            drop(Box::from_raw(v));
        }
    })
}

// ---- index ----------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn oc_index_from_regex(
    _regex: *const c_char,
    _v: *const OcVocab,
) -> *mut OcIndex {
    // The regex path is unused by Athena (all constraints are JSON
    // schema); llguidance is grammar-based and we do not expose a raw
    // regex compile. Fail explicitly rather than silently misbehave.
    ffi_guard("oc_index_from_regex", std::ptr::null_mut(), || {
        set_err("oc_index_from_regex: unsupported (use oc_index_from_schema)");
        std::ptr::null_mut()
    })
}

#[no_mangle]
pub unsafe extern "C" fn oc_index_from_schema(
    json: *const c_char,
    _whitespace: *const c_char,
    v: *const OcVocab,
) -> *mut OcIndex {
    ffi_guard("oc_index_from_schema", std::ptr::null_mut(), move || {
        if v.is_null() {
            set_err("oc_index_from_schema: null vocabulary");
            return std::ptr::null_mut();
        }
        let Some(json) = cstr(json) else {
            set_err("oc_index_from_schema: bad json string");
            return std::ptr::null_mut();
        };
        // G3: bound size/depth/repetition before the grammar compiler sees it.
        let value = match validate_schema(json) {
            Ok(v) => v,
            Err(e) => {
                set_err(format!("oc_index_from_schema: {e}"));
                return std::ptr::null_mut();
            }
        };
        let vocab = &*v;
        let grammar = TopLevelGrammar::from_json_schema(value);
        Box::into_raw(Box::new(OcIndex {
            factory: vocab.factory.clone(),
            grammar,
            vocab_size: vocab.vocab_size,
        }))
    })
}

#[no_mangle]
pub unsafe extern "C" fn oc_index_free(i: *mut OcIndex) {
    ffi_guard("oc_index_free", (), move || {
        if !i.is_null() {
            drop(Box::from_raw(i));
        }
    })
}

// ---- guide ----------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn oc_guide_new(i: *const OcIndex) -> *mut OcGuide {
    ffi_guard("oc_guide_new", std::ptr::null_mut(), move || {
        if i.is_null() {
            set_err("oc_guide_new: null index");
            return std::ptr::null_mut();
        }
        let index = &*i;
        let mut parser = match index.factory.create_parser(index.grammar.clone()) {
            Ok(p) => p,
            Err(e) => {
                set_err(format!("oc_guide_new: create_parser: {e}"));
                return std::ptr::null_mut();
            }
        };
        parser.start_without_prompt();
        Box::into_raw(Box::new(OcGuide {
            parser,
            vocab_size: index.vocab_size,
            cached_mask: None,
        }))
    })
}

#[no_mangle]
pub unsafe extern "C" fn oc_guide_free(g: *mut OcGuide) {
    ffi_guard("oc_guide_free", (), move || {
        if !g.is_null() {
            drop(Box::from_raw(g));
        }
    })
}

/// Required mask length in bytes for this guide's vocabulary.
#[no_mangle]
pub unsafe extern "C" fn oc_guide_mask_len(g: *const OcGuide) -> usize {
    ffi_guard("oc_guide_mask_len", 0, move || {
        if g.is_null() {
            return 0;
        }
        (*g).mask_len()
    })
}

/// 0 on success; -1 if buffer too small (call `oc_guide_mask_len`) or on a
/// parser error (see `oc_last_error`). Takes `*mut` because computing the
/// mask mutates (and caches) parser state.
#[no_mangle]
pub unsafe extern "C" fn oc_guide_allowed_mask(g: *mut OcGuide, buf: *mut u8, len: usize) -> c_int {
    ffi_guard("oc_guide_allowed_mask", -1, move || {
        if g.is_null() || buf.is_null() {
            return -1;
        }
        let g = &mut *g;
        if len < g.mask_len() {
            return -1;
        }
        if g.allowed_mask(std::slice::from_raw_parts_mut(buf, len)) {
            0
        } else {
            -1
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn oc_guide_advance(g: *mut OcGuide, token: u32) -> bool {
    ffi_guard("oc_guide_advance", false, move || {
        if g.is_null() {
            return false;
        }
        (*g).advance(token)
    })
}

/// Takes `*mut` because `is_accepting` mutates parser state.
#[no_mangle]
pub unsafe extern "C" fn oc_guide_is_final(g: *mut OcGuide) -> bool {
    ffi_guard("oc_guide_is_final", false, move || {
        !g.is_null() && (*g).is_final()
    })
}

// Rollback is dead code (the guide only ever advances on committed
// tokens). Kept in the ABI as no-ops for compatibility.

#[no_mangle]
pub unsafe extern "C" fn oc_guide_allowed_rollback(_g: *const OcGuide) -> usize {
    ffi_guard("oc_guide_allowed_rollback", 0, || 0)
}

#[no_mangle]
pub unsafe extern "C" fn oc_guide_rollback(_g: *mut OcGuide, n: usize) -> bool {
    // Only the no-op (n == 0) "rollback" succeeds.
    ffi_guard("oc_guide_rollback", false, move || n == 0)
}

// ---- tests ----------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A 256-single-byte vocab + an eos token at id 256.
    fn byte_vocab() -> (Arc<ParserFactory>, usize, u32) {
        let mut words: Vec<Vec<u8>> = (0u16..256).map(|b| vec![b as u8]).collect();
        let eos = words.len() as u32; // 256
        words.push(vec![0xFFu8, b'e', b'o', b's']);
        let info = TokRxInfo::new(words.len() as u32, eos);
        let trie = TokTrie::from(&info, &words);
        let tok_env: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        let factory = Arc::new(ParserFactory::new_simple(&tok_env).expect("factory"));
        (factory, words.len(), eos)
    }

    fn guide_for(schema: &str) -> (OcGuide, u32) {
        let (factory, vocab_size, eos) = byte_vocab();
        let value: serde_json::Value = serde_json::from_str(schema).unwrap();
        let grammar = TopLevelGrammar::from_json_schema(value);
        let mut parser = factory.create_parser(grammar).expect("parser");
        parser.start_without_prompt();
        (
            OcGuide {
                parser,
                vocab_size,
                cached_mask: None,
            },
            eos,
        )
    }

    #[test]
    fn integer_schema_masks_and_advances() {
        let (mut g, _eos) = guide_for(r#"{"type":"integer"}"#);
        assert!(!g.is_final(), "no digits yet: not accepting");

        // The opening mask must allow at least one digit and must NOT allow
        // a letter like 'a'.
        let mut mask = vec![0u8; g.mask_len()];
        assert!(g.allowed_mask(&mut mask));
        let allowed = |t: u8| (mask[(t as usize) >> 3] >> (t & 7)) & 1 == 1;
        assert!(allowed(b'4'), "digit allowed at start");
        assert!(!allowed(b'a'), "letter not allowed for an integer");

        assert!(g.advance(b'4' as u32), "'4' accepted");
        assert!(g.advance(b'2' as u32), "'2' accepted");
        assert!(g.is_final(), "\"42\" is a complete integer");
        // A letter has no transition and must not mutate state.
        assert!(!g.advance(b'a' as u32), "letter rejected, no transition");
        assert!(g.is_final(), "state unchanged after rejected advance");
    }

    /// G9: `advance` must distinguish an out-of-range token id (a caller
    /// contract violation → error channel set) from a token the schema
    /// merely disallows (clean false, no error).
    #[test]
    fn advance_distinguishes_out_of_range_from_disallowed_g9() {
        let (mut g, _eos) = guide_for(r#"{"type":"integer"}"#);
        // In-range but disallowed ('a' for an integer): reject WITHOUT
        // recording an out-of-range error.
        set_err("");
        assert!(!g.advance(b'a' as u32), "letter disallowed");
        assert!(
            !StructuredShimErr::last().contains("out of range"),
            "a disallowed token must not set the out-of-range error"
        );
        // Out-of-range id (>= vocab_size): a caller bug → distinct error.
        assert!(!g.advance(g.vocab_size as u32), "out-of-range rejected");
        assert!(
            StructuredShimErr::last().contains("out of range"),
            "out-of-range token sets the distinct error channel (G9)"
        );
    }

    #[test]
    fn object_schema_accepts_valid_doc_and_rejects_invalid() {
        let schema = r#"{"type":"object","properties":{"n":{"type":"integer"}},
                "required":["n"],"additionalProperties":false}"#;

        // Every byte of a valid document must be permitted by the mask in
        // turn, and the parser must end in an accepting state. eos must
        // then be allowed (so the loop can stop).
        let (mut g, eos) = guide_for(schema);
        let mut mask = vec![0u8; g.mask_len()];
        for &b in br#"{"n":7}"# {
            assert!(g.allowed_mask(&mut mask));
            let allowed = |t: u32| (mask[(t as usize) >> 3] >> (t & 7)) & 1 == 1;
            assert!(allowed(b as u32), "byte {:?} permitted in sequence", b as char);
            assert!(g.advance(b as u32), "advance {:?}", b as char);
        }
        assert!(g.is_final(), "complete object is accepting");
        assert!(g.allowed_mask(&mut mask));
        assert!(
            (mask[(eos as usize) >> 3] >> (eos & 7)) & 1 == 1,
            "eos allowed once the object is complete"
        );

        // An additional property is forbidden: after `{"n":7` the only
        // structural continuations are more digits or `}` — never `,`.
        let (mut g2, _eos) = guide_for(schema);
        for &b in br#"{"n":7"# {
            assert!(g2.advance(b as u32), "prefix advance {:?}", b as char);
        }
        assert!(!g2.advance(b',' as u32), "additionalProperties:false rejects a comma");
    }

    #[test]
    fn regex_path_is_unsupported() {
        let (factory, vocab_size, _eos) = byte_vocab();
        let v = OcVocab {
            factory,
            vocab_size,
        };
        let p = unsafe { oc_index_from_regex(c"[0-9]+".as_ptr(), &v as *const _) };
        assert!(p.is_null(), "regex path returns null");
    }

    // ---- M65.1 hardening fixtures (G1/G2/G3/G6/G10) -----------------

    /// Build a real `OcVocab` for the FFI-level hostile-schema tests.
    fn make_vocab() -> *mut OcVocab {
        let (factory, vocab_size, _eos) = byte_vocab();
        Box::into_raw(Box::new(OcVocab { factory, vocab_size }))
    }

    /// G1: a panic inside an FFI body must be caught and converted to the
    /// function's failure sentinel — never unwind across the C ABI.
    #[test]
    fn ffi_guard_catches_panic_and_sets_error() {
        let r: i32 = ffi_guard("unit_panic", -1, || panic!("boom from the schema"));
        assert_eq!(r, -1, "panic returns the default sentinel");
        assert!(
            StructuredShimErr::last().contains("panic caught at FFI boundary"),
            "panic message stashed for oc_last_error"
        );
        assert!(
            StructuredShimErr::last().contains("boom from the schema"),
            "original panic payload surfaced"
        );
    }

    /// Test helper mirroring the Swift `oc_last_error` read.
    struct StructuredShimErr;
    impl StructuredShimErr {
        fn last() -> String {
            LAST_ERR.with(|e| e.borrow().to_string_lossy().into_owned())
        }
    }

    /// G2: a token id past the vocab cap must be rejected before the dense
    /// `max_id+1` allocation, not OOM the process.
    #[test]
    fn oversized_token_id_rejected() {
        let ids: [u32; 1] = [0xFFFF_FFFF];
        let bytes: [u8; 1] = [b'a'];
        let byte_ptrs: [*const u8; 1] = [bytes.as_ptr()];
        let byte_lens: [usize; 1] = [1];
        let v = unsafe {
            oc_vocab_new_from_tokens(ids.as_ptr(), byte_ptrs.as_ptr(), byte_lens.as_ptr(), 1, 0)
        };
        assert!(v.is_null(), "0xFFFFFFFF id rejected, no 4GB alloc");
        assert!(StructuredShimErr::last().contains("vocab cap"));
    }

    /// G6: an absurd per-token byte length must be rejected before the
    /// unchecked `from_raw_parts` read.
    #[test]
    fn oversized_token_bytes_rejected() {
        let ids: [u32; 1] = [0];
        let bytes: [u8; 1] = [b'a'];
        let byte_ptrs: [*const u8; 1] = [bytes.as_ptr()];
        let byte_lens: [usize; 1] = [usize::MAX]; // hostile length
        let v = unsafe {
            oc_vocab_new_from_tokens(ids.as_ptr(), byte_ptrs.as_ptr(), byte_lens.as_ptr(), 1, 1)
        };
        assert!(v.is_null(), "absurd byte length rejected");
        assert!(StructuredShimErr::last().contains("byte length"));
    }

    /// G3: schema bytes / depth / repetition all rejected before compile,
    /// each via the normal NULL return (never a crash).
    #[test]
    fn hostile_schemas_rejected_before_compile() {
        let v = make_vocab();

        // (a) oversized raw schema
        let big = format!(r#"{{"type":"string","description":"{}"}}"#, "x".repeat(MAX_SCHEMA_BYTES + 16));
        let p = unsafe { oc_index_from_schema(CString::new(big).unwrap().as_ptr(), std::ptr::null(), v) };
        assert!(p.is_null(), "oversized schema rejected");
        assert!(StructuredShimErr::last().contains("schema too large"));

        // (b) pathologically deep nesting
        let mut deep = String::new();
        let layers = MAX_SCHEMA_DEPTH + 8;
        for _ in 0..layers {
            deep.push_str(r#"{"type":"object","properties":{"a":"#);
        }
        deep.push_str(r#"{"type":"integer"}"#);
        for _ in 0..layers {
            deep.push_str("}}");
        }
        let p = unsafe { oc_index_from_schema(CString::new(deep).unwrap().as_ptr(), std::ptr::null(), v) };
        assert!(p.is_null(), "deep schema rejected (depth or serde recursion)");

        // (c) huge repetition bound
        let rep = format!(r#"{{"type":"array","items":{{"type":"integer"}},"maxItems":{}}}"#, MAX_REPETITION + 1);
        let p = unsafe { oc_index_from_schema(CString::new(rep).unwrap().as_ptr(), std::ptr::null(), v) };
        assert!(p.is_null(), "huge maxItems rejected");
        assert!(StructuredShimErr::last().contains("repetition bound"));

        // A normal schema still compiles fine through the same path.
        let ok = r#"{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}"#;
        let p = unsafe { oc_index_from_schema(CString::new(ok).unwrap().as_ptr(), std::ptr::null(), v) };
        assert!(!p.is_null(), "valid schema still compiles");
        unsafe {
            oc_index_free(p);
            oc_vocab_free(v);
        }
    }

    /// G10: a mask buffer shorter than `mask_len` fails loud rather than
    /// writing a truncated (under-constraining) mask.
    #[test]
    fn short_mask_buffer_fails_loud() {
        let (mut g, _eos) = guide_for(r#"{"type":"integer"}"#);
        let need = g.mask_len();
        let mut too_short = vec![0u8; need - 1];
        assert!(!g.allowed_mask(&mut too_short), "short buffer rejected");
        assert!(StructuredShimErr::last().contains("buffer"));
    }

    /// G10: `set_err` survives an interior NUL in the message (it would
    /// otherwise truncate or be dropped entirely).
    #[test]
    fn set_err_is_interior_nul_safe() {
        set_err("before\0after");
        let got = StructuredShimErr::last();
        assert!(got.starts_with("before"), "message not truncated at NUL: {got:?}");
        assert!(got.contains("after"), "tail after the NUL survives: {got:?}");
    }
}
