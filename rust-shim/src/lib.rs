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

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::os::raw::c_int;
use std::sync::Arc;

use llguidance::api::TopLevelGrammar;
use llguidance::{ParserFactory, TokenParser};
use toktrie::{ApproximateTokEnv, SimpleVob, TokEnv, TokRxInfo, TokTrie};

thread_local! {
    static LAST_ERR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_err(msg: impl Into<Vec<u8>>) {
    let c = CString::new(msg).unwrap_or_default();
    LAST_ERR.with(|e| *e.borrow_mut() = c);
}

unsafe fn cstr<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    CStr::from_ptr(p).to_str().ok()
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
    /// tokens allowed from the current state.
    fn allowed_mask(&mut self, buf: &mut [u8]) -> bool {
        for b in buf.iter_mut() {
            *b = 0;
        }
        if !self.ensure_mask() {
            return false;
        }
        let mask = self.cached_mask.as_ref().unwrap();
        // `write_to` asserts `buf.len() <= words*4`; clamp to both the
        // requested byte length and the mask's capacity.
        let cap = mask.as_slice().len() * 4;
        let n = buf.len().min(self.mask_len()).min(cap);
        mask.write_to(&mut buf[..n]);
        true
    }

    /// Advance on `token`. Returns false (no mutation) when the token has
    /// no transition from the current state — this gives the IDLE-probe
    /// caller the "failed advance doesn't change state" guarantee, since
    /// `consume_token` would otherwise error on a disallowed token.
    fn advance(&mut self, token: u32) -> bool {
        if token as usize >= self.vocab_size {
            return false;
        }
        if !self.ensure_mask() {
            return false;
        }
        let allowed = self.cached_mask.as_ref().unwrap().is_allowed(token);
        if !allowed {
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
    let mut max_id = eos;
    for i in 0..n {
        max_id = max_id.max(*ids.add(i));
    }
    let size = max_id as usize + 1;
    let mut words: Vec<Vec<u8>> = vec![Vec::new(); size];
    let mut filled = vec![false; size];

    for i in 0..n {
        let id = *ids.add(i) as usize;
        if id == eos as usize {
            continue; // eos is a control token, set below
        }
        let p = *byte_ptrs.add(i);
        let l = *byte_lens.add(i);
        if p.is_null() {
            continue;
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
    c"llguidance-1.7.5".as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn oc_last_error(buf: *mut c_char, len: usize) -> usize {
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
    match build_factory(ids, byte_ptrs, byte_lens, n, eos) {
        Some((factory, vocab_size)) => Box::into_raw(Box::new(OcVocab { factory, vocab_size })),
        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn oc_vocab_free(v: *mut OcVocab) {
    if !v.is_null() {
        drop(Box::from_raw(v));
    }
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
    set_err("oc_index_from_regex: unsupported (use oc_index_from_schema)");
    std::ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn oc_index_from_schema(
    json: *const c_char,
    _whitespace: *const c_char,
    v: *const OcVocab,
) -> *mut OcIndex {
    if v.is_null() {
        set_err("oc_index_from_schema: null vocabulary");
        return std::ptr::null_mut();
    }
    let Some(json) = cstr(json) else {
        set_err("oc_index_from_schema: bad json string");
        return std::ptr::null_mut();
    };
    let value: serde_json::Value = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(e) => {
            set_err(format!("oc_index_from_schema: invalid JSON schema: {e}"));
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
}

#[no_mangle]
pub unsafe extern "C" fn oc_index_free(i: *mut OcIndex) {
    if !i.is_null() {
        drop(Box::from_raw(i));
    }
}

// ---- guide ----------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn oc_guide_new(i: *const OcIndex) -> *mut OcGuide {
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
}

#[no_mangle]
pub unsafe extern "C" fn oc_guide_free(g: *mut OcGuide) {
    if !g.is_null() {
        drop(Box::from_raw(g));
    }
}

/// Required mask length in bytes for this guide's vocabulary.
#[no_mangle]
pub unsafe extern "C" fn oc_guide_mask_len(g: *const OcGuide) -> usize {
    if g.is_null() {
        return 0;
    }
    (*g).mask_len()
}

/// 0 on success; -1 if buffer too small (call `oc_guide_mask_len`) or on a
/// parser error (see `oc_last_error`). Takes `*mut` because computing the
/// mask mutates (and caches) parser state.
#[no_mangle]
pub unsafe extern "C" fn oc_guide_allowed_mask(g: *mut OcGuide, buf: *mut u8, len: usize) -> c_int {
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
}

#[no_mangle]
pub unsafe extern "C" fn oc_guide_advance(g: *mut OcGuide, token: u32) -> bool {
    if g.is_null() {
        return false;
    }
    (*g).advance(token)
}

/// Takes `*mut` because `is_accepting` mutates parser state.
#[no_mangle]
pub unsafe extern "C" fn oc_guide_is_final(g: *mut OcGuide) -> bool {
    !g.is_null() && (*g).is_final()
}

// Rollback is dead code (the guide only ever advances on committed
// tokens). Kept in the ABI as no-ops for compatibility.

#[no_mangle]
pub unsafe extern "C" fn oc_guide_allowed_rollback(_g: *const OcGuide) -> usize {
    0
}

#[no_mangle]
pub unsafe extern "C" fn oc_guide_rollback(_g: *mut OcGuide, n: usize) -> bool {
    // Only the no-op (n == 0) "rollback" succeeds.
    n == 0
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
}
