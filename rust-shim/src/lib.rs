//! Athena structured-output shim.
//!
//! A C-ABI `staticlib` over `outlines-core` 0.2.14: compile a
//! JSON-schema / regex / choice into a DFA `Index`, then walk it with a
//! ported Guide (state + 32-entry rollback ring — outlines-core's own
//! Guide is pyo3-gated, so it is reimplemented here from the public
//! `Index` primitives; NO pyo3, NO regex→DFA reimplementation).
//!
//! Error convention: fallible calls return NULL / -1 / false and stash a
//! message in a thread-local; the caller reads `oc_last_error`.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::ffi::{c_char, CStr, CString};
use std::os::raw::c_int;

use outlines_core::json_schema;
use outlines_core::prelude::*;

const MAX_ROLLBACK: usize = 32;

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

// ---- opaque handles -------------------------------------------------

pub struct OcVocab(Vocabulary);
pub struct OcIndex(Index);

/// Ported outlines-core Guide: current DFA state + a bounded rollback
/// ring of prior states (cap = `MAX_ROLLBACK`, evict-oldest).
pub struct OcGuide {
    index: Index,
    state: StateId,
    ring: VecDeque<StateId>,
}

impl OcGuide {
    fn new(index: Index) -> Self {
        let state = index.initial_state();
        Self {
            index,
            state,
            ring: VecDeque::with_capacity(MAX_ROLLBACK),
        }
    }

    /// Advance on `token`. Returns false (no mutation) when the token has
    /// no transition from the current state.
    fn advance(&mut self, token: TokenId) -> bool {
        match self.index.next_state(&self.state, &token) {
            Some(next) => {
                if self.ring.len() == MAX_ROLLBACK {
                    self.ring.pop_front();
                }
                self.ring.push_back(self.state);
                self.state = next;
                true
            }
            None => false,
        }
    }

    fn allowed_rollback(&self) -> usize {
        self.ring.len()
    }

    fn rollback(&mut self, n: usize) -> bool {
        if n == 0 {
            return true;
        }
        if n > self.ring.len() {
            return false;
        }
        for _ in 0..n {
            self.state = self.ring.pop_back().unwrap();
        }
        true
    }

    fn is_final(&self) -> bool {
        self.index.is_final_state(&self.state)
    }

    /// Fill `buf` (one bit per token, bit `i` of byte `i>>3`) with the
    /// tokens allowed from the current state. `buf` must be
    /// `ceil(vocab_size/8)` bytes.
    fn allowed_mask(&self, buf: &mut [u8]) {
        buf.iter_mut().for_each(|b| *b = 0);
        if let Some(iter) = self.index.allowed_tokens_iter(&self.state) {
            for &t in iter {
                let t = t as usize;
                buf[t >> 3] |= 1u8 << (t & 7);
            }
        }
    }

    fn mask_len(&self) -> usize {
        self.index.vocab_size().div_ceil(8)
    }
}

// ---- misc -----------------------------------------------------------

#[no_mangle]
pub extern "C" fn oc_version() -> *const c_char {
    c"0.2.14".as_ptr()
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

/// Build a `Vocabulary` from parallel arrays of token ids and their
/// decoded UTF-8/byte forms (the caller — Athena's swift-transformers
/// tokenizer — produces the byte mapping so it matches the model
/// exactly; the EOS id is skipped, outlines-core adds it itself).
#[no_mangle]
pub unsafe extern "C" fn oc_vocab_new_from_tokens(
    ids: *const u32,
    byte_ptrs: *const *const u8,
    byte_lens: *const usize,
    n: usize,
    eos: u32,
) -> *mut OcVocab {
    if ids.is_null() || byte_ptrs.is_null() || byte_lens.is_null() {
        set_err("oc_vocab_new_from_tokens: null array");
        return std::ptr::null_mut();
    }
    let mut vocab = Vocabulary::new(eos);
    for i in 0..n {
        let id = *ids.add(i);
        if id == eos {
            continue;
        }
        let p = *byte_ptrs.add(i);
        let l = *byte_lens.add(i);
        if p.is_null() {
            continue;
        }
        let bytes = std::slice::from_raw_parts(p, l).to_vec();
        if let Err(e) = vocab.try_insert(bytes, id) {
            set_err(format!("vocab.try_insert(id={id}): {e}"));
            return std::ptr::null_mut();
        }
    }
    Box::into_raw(Box::new(OcVocab(vocab)))
}

#[no_mangle]
pub unsafe extern "C" fn oc_vocab_free(v: *mut OcVocab) {
    if !v.is_null() {
        drop(Box::from_raw(v));
    }
}

// ---- index ----------------------------------------------------------

unsafe fn build_index(regex: &str, v: *const OcVocab) -> *mut OcIndex {
    if v.is_null() {
        set_err("index: null vocabulary");
        return std::ptr::null_mut();
    }
    match Index::new(regex, &(*v).0) {
        Ok(idx) => Box::into_raw(Box::new(OcIndex(idx))),
        Err(e) => {
            set_err(format!("Index::new: {e}"));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn oc_index_from_regex(
    regex: *const c_char,
    v: *const OcVocab,
) -> *mut OcIndex {
    let Some(regex) = cstr(regex) else {
        set_err("oc_index_from_regex: bad regex string");
        return std::ptr::null_mut();
    };
    build_index(regex, v)
}

#[no_mangle]
pub unsafe extern "C" fn oc_index_from_schema(
    json: *const c_char,
    whitespace: *const c_char,
    v: *const OcVocab,
) -> *mut OcIndex {
    let Some(json) = cstr(json) else {
        set_err("oc_index_from_schema: bad json string");
        return std::ptr::null_mut();
    };
    let ws = cstr(whitespace);
    match json_schema::regex_from_str(json, ws, None) {
        Ok(regex) => build_index(&regex, v),
        Err(e) => {
            set_err(format!("json_schema::regex_from_str: {e}"));
            std::ptr::null_mut()
        }
    }
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
    Box::into_raw(Box::new(OcGuide::new((*i).0.clone())))
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

/// 0 on success; -1 if buffer too small (call `oc_guide_mask_len`).
#[no_mangle]
pub unsafe extern "C" fn oc_guide_allowed_mask(
    g: *const OcGuide,
    buf: *mut u8,
    len: usize,
) -> c_int {
    if g.is_null() || buf.is_null() {
        return -1;
    }
    let g = &*g;
    if len < g.mask_len() {
        return -1;
    }
    g.allowed_mask(std::slice::from_raw_parts_mut(buf, len));
    0
}

#[no_mangle]
pub unsafe extern "C" fn oc_guide_advance(g: *mut OcGuide, token: u32) -> bool {
    if g.is_null() {
        return false;
    }
    (*g).advance(token)
}

#[no_mangle]
pub unsafe extern "C" fn oc_guide_is_final(g: *const OcGuide) -> bool {
    !g.is_null() && (*g).is_final()
}

#[no_mangle]
pub unsafe extern "C" fn oc_guide_allowed_rollback(g: *const OcGuide) -> usize {
    if g.is_null() {
        0
    } else {
        (*g).allowed_rollback()
    }
}

#[no_mangle]
pub unsafe extern "C" fn oc_guide_rollback(g: *mut OcGuide, n: usize) -> bool {
    !g.is_null() && (*g).rollback(n)
}

// ---- tests ----------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Vocabulary of single-char byte tokens '0'..'9' (ids 0..9),
    /// eos id = 10.
    fn digit_vocab() -> Vocabulary {
        let mut v = Vocabulary::new(10);
        for d in 0u32..10 {
            v.try_insert(vec![b'0' + d as u8], d).unwrap();
        }
        v
    }

    #[test]
    fn regex_index_walk_and_rollback() {
        let v = digit_vocab();
        let index = Index::new("[0-9][0-9]", &v).expect("index");
        let mut g = OcGuide::new(index);

        assert!(!g.is_final(), "0 digits: not final");
        // mask at start: some digit tokens allowed
        let mut mask = vec![0u8; g.mask_len()];
        g.allowed_mask(&mut mask);
        assert!(mask.iter().any(|b| *b != 0), "start has allowed tokens");
        assert_eq!(g.allowed_rollback(), 0);

        assert!(g.advance(3), "digit 3 accepted");
        assert!(!g.is_final(), "1 digit: not final ([0-9][0-9])");
        assert_eq!(g.allowed_rollback(), 1);

        assert!(g.advance(7), "digit 7 accepted");
        assert!(g.is_final(), "2 digits: final");
        assert_eq!(g.allowed_rollback(), 2);

        // rollback both digits → back to start, not final
        assert!(g.rollback(2));
        assert!(!g.is_final());
        assert_eq!(g.allowed_rollback(), 0);
        assert!(!g.rollback(1), "cannot roll back past recorded history");
    }

    #[test]
    fn json_schema_compiles_to_index() {
        let v = digit_vocab();
        let regex = json_schema::regex_from_str(r#"{"type":"integer"}"#, None, None)
            .expect("schema→regex");
        assert!(!regex.is_empty());
        // Index builds over the (tiny) digit vocab without error.
        Index::new(&regex, &v).expect("index from schema regex");
    }
}
