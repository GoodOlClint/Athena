/* Athena structured-output shim — C ABI over outlines-core 0.2.14.
 * Hand-written (no cbindgen dep). Mirrors rust-shim/src/lib.rs.
 *
 * Ownership: oc_*_new return owned opaque handles freed by the matching
 * oc_*_free; NULL/-1/false signal failure — call oc_last_error.
 * Strings from oc_version are static (do not free). */
#ifndef ATHENA_STRUCTURED_H
#define ATHENA_STRUCTURED_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcVocab OcVocab;
typedef struct OcIndex OcIndex;
typedef struct OcGuide OcGuide;

const char *oc_version(void);
size_t oc_last_error(char *buf, size_t len);

/* Vocabulary from parallel arrays; the EOS id is skipped (outlines-core
 * adds it). Caller owns the input buffers for the call's duration. */
OcVocab *oc_vocab_new_from_tokens(const uint32_t *ids,
                                  const uint8_t *const *byte_ptrs,
                                  const size_t *byte_lens, size_t n,
                                  uint32_t eos);
void oc_vocab_free(OcVocab *v);

OcIndex *oc_index_from_regex(const char *regex, const OcVocab *v);
/* whitespace may be NULL (outlines-core default). */
OcIndex *oc_index_from_schema(const char *json, const char *whitespace,
                              const OcVocab *v);
void oc_index_free(OcIndex *i);

OcGuide *oc_guide_new(const OcIndex *i);
void oc_guide_free(OcGuide *g);

size_t oc_guide_mask_len(const OcGuide *g);          /* ceil(vocab/8) */
int oc_guide_allowed_mask(const OcGuide *g, uint8_t *buf, size_t len);
bool oc_guide_advance(OcGuide *g, uint32_t token);   /* false: no transition */
bool oc_guide_is_final(const OcGuide *g);
size_t oc_guide_allowed_rollback(const OcGuide *g);
bool oc_guide_rollback(OcGuide *g, size_t n);        /* false: n too large */

#ifdef __cplusplus
}
#endif
#endif
