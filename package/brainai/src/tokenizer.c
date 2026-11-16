/* SentencePiece unigram encoder (Viterbi) with byte fallback. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "brainai.h"

typedef struct {
	const char *s;
	float score;
	unsigned char type, len;
} piece_t;

struct tokenizer {
	int n, maxlen, hsize;
	float unk_score;
	piece_t *p;
	int *hash; /* open addressing, -1 = empty */
	char *blob;
};

static unsigned hstr(const char *s, int n)
{
	unsigned h = 2166136261u;
	while (n--)
		h = (h ^ (unsigned char)*s++) * 16777619u;
	return h;
}

static int lookup(const tokenizer_t *t, const char *s, int n)
{
	unsigned i = hstr(s, n) & (t->hsize - 1);
	for (;; i = (i + 1) & (t->hsize - 1)) {
		int id = t->hash[i];
		if (id < 0)
			return -1;
		if (t->p[id].len == n && !memcmp(t->p[id].s, s, n))
			return id;
	}
}

tokenizer_t *tok_load(const char *path)
{
	FILE *f = fopen(path, "rb");
	tokenizer_t *t;
	long len, off;
	int i;
	float minsc = 0;
	if (!f)
		return NULL;
	fseek(f, 0, SEEK_END);
	len = ftell(f);
	fseek(f, 0, SEEK_SET);
	t = calloc(1, sizeof(*t));
	t->blob = malloc(len);
	if (!t->blob || fread(t->blob, 1, len, f) != (size_t)len || memcmp(t->blob, "BRTK", 4)) {
		fclose(f);
		return NULL;
	}
	fclose(f);
	memcpy(&t->n, t->blob + 4, 4);
	t->p = calloc(t->n, sizeof(piece_t));
	for (t->hsize = 1; t->hsize < 2 * t->n; t->hsize <<= 1)
		;
	t->hash = malloc(sizeof(int) * t->hsize);
	memset(t->hash, 0xff, sizeof(int) * t->hsize);
	off = 8;
	for (i = 0; i < t->n; i++) {
		piece_t *p = &t->p[i];
		if (off + 6 > len)
			return NULL;
		memcpy(&p->score, t->blob + off, 4);
		p->type = t->blob[off + 4];
		p->len = t->blob[off + 5];
		p->s = t->blob + off + 6;
		off += 6 + p->len;
		if (p->type == 1 || p->type == 4) {
			unsigned h = hstr(p->s, p->len) & (t->hsize - 1);
			while (t->hash[h] >= 0)
				h = (h + 1) & (t->hsize - 1);
			t->hash[h] = i;
			if (p->len > t->maxlen)
				t->maxlen = p->len;
			if (p->type == 1 && p->score < minsc)
				minsc = p->score;
		}
	}
	t->unk_score = minsc - 10.0f;
	return t;
}

/* Spaces become U+2581 (no dummy prefix, identity normalisation). */
int tok_encode(const tokenizer_t *t, const char *text, int *ids, int max)
{
	int n = 0, i, s, L, out = 0;
	const char *c;
	char *b;
	float *best;
	int *from, *tokid, *stack;
	for (c = text; *c; c++)
		n += (*c == ' ') ? 3 : 1;
	b = malloc(n + 1);
	for (c = text, i = 0; *c; c++) {
		if (*c == ' ') {
			b[i++] = (char)0xe2; b[i++] = (char)0x96; b[i++] = (char)0x81;
		} else {
			b[i++] = *c;
		}
	}
	best = malloc(sizeof(float) * (n + 1));
	from = malloc(sizeof(int) * (n + 1));
	tokid = malloc(sizeof(int) * (n + 1));
	stack = malloc(sizeof(int) * (n + 1));
	for (i = 0; i <= n; i++)
		best[i] = -1e30f;
	best[0] = 0;
	for (s = 0; s < n; s++) {
		if (best[s] < -1e29f || ((unsigned char)b[s] & 0xc0) == 0x80)
			continue;
		for (L = 1; L <= t->maxlen && s + L <= n; L++) {
			int id = lookup(t, b + s, L);
			float sc;
			if (id < 0)
				continue;
			sc = best[s] + (t->p[id].type == 1 ? t->p[id].score : 0.0f);
			if (sc > best[s + L]) {
				best[s + L] = sc;
				from[s + L] = s;
				tokid[s + L] = id;
			}
		}
		/* unknown character -> its UTF-8 bytes */
		for (L = 1; s + L < n && ((unsigned char)b[s + L] & 0xc0) == 0x80; L++)
			;
		if (best[s] + t->unk_score > best[s + L]) {
			best[s + L] = best[s] + t->unk_score;
			from[s + L] = s;
			tokid[s + L] = -1;
		}
	}
	/* backtrack into stack (reverse order) */
	{
		int sp = 0, pos = n;
		while (pos > 0) {
			stack[sp++] = pos;
			pos = from[pos];
		}
		while (sp-- > 0) {
			int e = stack[sp], st = from[e];
			if (tokid[e] >= 0) {
				if (out < max)
					ids[out++] = tokid[e];
			} else {
				for (i = st; i < e; i++)
					if (out < max)
						ids[out++] = TOK_BYTE0 + (unsigned char)b[i];
			}
		}
	}
	free(b); free(best); free(from); free(tokid); free(stack);
	return out;
}

int tok_piece(const tokenizer_t *t, int id, char *buf, int max)
{
	const piece_t *p;
	int i, o = 0;
	if (id < 0 || id >= t->n || max < 1)
		return 0;
	p = &t->p[id];
	if (p->type == 6) { /* <0xNN> */
		buf[0] = (char)strtol(p->s + 3, NULL, 16);
		return 1;
	}
	if (p->type != 1)
		return 0;
	for (i = 0; i < p->len && o < max; i++) {
		if (i + 2 < p->len && (unsigned char)p->s[i] == 0xe2 &&
		    (unsigned char)p->s[i + 1] == 0x96 && (unsigned char)p->s[i + 2] == 0x81) {
			buf[o++] = ' ';
			i += 2;
		} else {
			buf[o++] = p->s[i];
		}
	}
	return o;
}
