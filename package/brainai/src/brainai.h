#ifndef BRAINAI_H
#define BRAINAI_H

#include <stddef.h>
#include <stdint.h>

/* Special token ids of brain-ai v1 (fixed by its tokenizer). */
enum {
	TOK_UNK = 0, TOK_BOS = 1, TOK_EOS = 2,
	TOK_SYS = 3, TOK_USER = 4, TOK_BOT = 5, TOK_CALL = 6, TOK_RESULT = 7,
	TOK_END = 8, TOK_BYTE0 = 9,
};

/* engine.c */
typedef struct model model_t;
model_t *model_load(const char *path);
int model_vocab(const model_t *m);
void model_reset(model_t *m);
size_t model_state_size(const model_t *m);
void model_state_save(const model_t *m, void *dst);
void model_state_load(model_t *m, const void *src);
/* Feed one token; returns logits (vocab floats) when want_logits. */
const float *model_step(model_t *m, int token, int want_logits);

/* tokenizer.c */
typedef struct tokenizer tokenizer_t;
tokenizer_t *tok_load(const char *path);
int tok_encode(const tokenizer_t *t, const char *text, int *ids, int max);
/* Raw bytes of a piece (byte-fallback pieces give one byte, U+2581 -> ' '). */
int tok_piece(const tokenizer_t *t, int id, char *buf, int max);

/* text.c */
int romaji_to_kana(const char *in, char *out, size_t max);
int calc_eval(const char *expr, double *result, int degrees);
void format_number(double v, char *buf, size_t max);
/* Find an arithmetic expression in a user message; writes it and its value. */
int igc_find(const char *text, char *expr, size_t exmax, char *result, size_t max);

#endif
