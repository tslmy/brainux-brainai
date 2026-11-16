/*
 * brainai inference engine: Mamba-2 + BitNet b1.58 (ternary) language model.
 *
 * This file turns one token into the scores ("logits") for the next token.
 * It is written for the ARM926EJ-S CPU (central processing unit) in SHARP
 * Brain, which has no FPU (floating-point unit), no vector unit and a 16 KiB
 * data cache.  docs/how-it-works.md explains every trick below with
 * diagrams; the comments here point at the matching section.
 *
 * The four main ideas:
 *  - Ternary matrices (weights -1, 0, +1) are multiplied with a LUT (lookup
 *    table): for each group of four int8 inputs, all 81 possible sums are
 *    precomputed, and each packed weight byte is the index of its answer.
 *    One table load replaces four multiply-adds.  (Article sections 4, 6.)
 *  - The weights are reordered at load time so the tables stay in the cache
 *    and one 32-bit load serves four rows.  (Section 6.)
 *  - The LM (language model) head scores the vocabulary as int8 x int8 dot
 *    products on the CPU's DSP (digital signal processing) multiply
 *    instructions.  (Section 7.)
 *  - Per-element work inside a layer (convolution, SiLU, the SSM recurrence,
 *    the gated norm) runs in 32-bit fixed point, because emulated float was
 *    the biggest cost.  Only a few values per layer (one decay rate per
 *    head, the per-token scales) still use float.  (Section 8.)
 *
 * Glossary of acronyms used in this file:
 *   SSM   state space model: the recurrent core of each Mamba-2 layer
 *   SiLU  sigmoid linear unit: SiLU(x) = x * sigmoid(x)
 *   RMS   root mean square: sqrt(mean(x^2)), used by the normalisation
 *   QN    fixed-point number with N fractional bits: value = integer / 2^N
 */
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "brainai.h"

/*
 * Optional per-stage timing, compiled in only with -DBRAINAI_PROF (see
 * bench.c).  PT(i) adds the time since the previous PT/PSTART to slot i.
 * In a normal build both macros expand to nothing.
 */
#ifdef BRAINAI_PROF
#include <time.h>
double brainai_prof[10];
static double prof_last;
static double pnow(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#define PT(i) do { double t_ = pnow(); brainai_prof[i] += t_ - prof_last; prof_last = t_; } while (0)
#define PSTART() (prof_last = pnow())
#else
#define PT(i) do { } while (0)
#define PSTART() do { } while (0)
#endif

/* Fractional bits of the SSM state S and of the C vector (Q20). */
#define FX 20
/*
 * Column groups (of 4 inputs each) per ternary block, see repack_ternary().
 * 32 groups = 128 inputs, whose 32 tables take 32 * 81 * 2 = 5,184 bytes:
 * small enough to stay in the 16 KiB data cache.
 */
#ifndef GB
#define GB 32
#endif

/* Everything one layer needs.  Pointers point into the loaded file. */
typedef struct {
	float *norm;            /* RMS-norm weights before the input matrix (Dm) */
	float in_scale;         /* the input matrix's weight scale s_W */
	const uint8_t *in_w;    /* input matrix, Din x Dm ternary, repacked */
	float *conv_w, *conv_b; /* depthwise convolution, Dconv x K and Dconv */
	float *dt_bias;         /* per-head bias added to the step size dt (H) */
	float *A, *D;           /* per-head decay rate A (negative) and skip D */
	float *gnorm;           /* gated RMS-norm weights (E) */
	int32_t *conv_wq, *conv_bq, *gnorm_q; /* fixed-point copies: Q20, Q16, Q16 */
	float out_scale;        /* the output matrix's weight scale */
	const uint8_t *out_w;   /* output matrix, Dm x E ternary, repacked */
} layer_t;

/*
 * The whole model plus its recurrent state and scratch buffers.
 *
 * Dimension names (values for brain-ai v1 preview):
 *   V = 9501 vocabulary     Dm = 384 width between layers   L = 12 layers
 *   E = 768 inner width     N = 32 state size               K = 4 conv width
 *   H = 16 heads            P = 48 channels per head (H * P = E)
 *   Din   = 2E + 2N + H = 1616  rows of the input matrix: z, x, B, C, dt
 *   Dconv = E + 2N      = 832   channels that go through the convolution
 */
struct model {
	uint32_t V, Dm, L, E, N, H, P, K, Din, Dconv;
	uint8_t *blob;           /* the whole model.bin file, kept in memory */
	const float *emb_scale;  /* one scale per vocabulary row (V) */
	const int8_t *emb;       /* V x Dm embedding, stored as e + 128 after load */
	layer_t *layers;
	float *norm_f;           /* final RMS-norm weights (Dm) */
	/* recurrent state: everything the model remembers between tokens */
	int32_t *conv_state; /* L * Dconv * (K-1): last K-1 inputs per channel, Q16 */
	int32_t *ssm;        /* L * H * P * N: the SSM state S, Q20 */
	/* scratch buffers, reused by every layer */
	float *x, *hn;          /* x: the vector between layers; hn: norm output */
	int32_t *zx, *y, *xbc;  /* input-matrix output, SSM output, conv output (Q16) */
	uint16_t *sig;          /* sigmoid table, SIG_N + 1 entries, Q16 */
	int8_t *q;              /* int8 activations fed to a ternary matrix */
	int32_t *q2;            /* q packed as int16 pairs (0,2) (1,3) for the LM head */
	int16_t *lut;           /* lookup tables for one column block (81 per group) */
	int32_t *acc;           /* integer results of a ternary matrix */
	float *logits;          /* scores for every vocabulary entry (V) */
	int32_t *cfx;           /* C vector in Q20 */
	int16_t *b16;           /* B vector shrunk to 16 bits for SMULWB */
};

/* calloc that exits instead of returning NULL; memory starts zeroed. */
static void *xmalloc(size_t n)
{
	void *p = calloc(1, n);
	if (!p) {
		fprintf(stderr, "brainai: out of memory\n");
		exit(1);
	}
	return p;
}

/* Reads a whole file into memory; returns NULL on any error. */
static uint8_t *read_file(const char *path, size_t *len)
{
	FILE *f = fopen(path, "rb");
	uint8_t *buf;
	long n;
	if (!f)
		return NULL;
	fseek(f, 0, SEEK_END);
	n = ftell(f);
	fseek(f, 0, SEEK_SET);
	buf = xmalloc(n + 1);
	if (fread(buf, 1, n, f) != (size_t)n) {
		fclose(f);
		free(buf);
		return NULL;
	}
	fclose(f);
	*len = n;
	return buf;
}

static void repack_ternary(uint8_t *w, uint32_t rows, uint32_t G);
/* The sigmoid table covers [-16, 16) in SIG_N steps of 1/32. */
#define SIG_N 1024

/*
 * Loads model.bin (written by tools/convert.py) and prepares it for fast
 * inference.  The file layout is described in the article, section 3:
 *
 *   "BRAI", 9 x u32 header (version, V, Dm, L, E, N, H, P, K)
 *   embedding: V float scales, V x Dm int8
 *   per layer: norm, in_scale, in_w, conv_w, conv_b, dt_bias, A, D, gnorm,
 *              out_scale, out_w
 *   final norm
 *
 * Instead of copying, most fields point straight into the loaded file.
 * Then, once per run:
 *   - the ternary matrices are reordered in place (repack_ternary);
 *   - fixed-point copies of the small per-layer tensors are made;
 *   - the sigmoid table is built;
 *   - the embedding bytes are made unsigned for the LM head.
 */
model_t *model_load(const char *path)
{
	size_t len, off;
	uint32_t hdr[9], l, i;
	model_t *m;
	uint8_t *b = read_file(path, &len);
	if (!b || len < 40 || memcmp(b, "BRAI", 4)) {
		fprintf(stderr, "brainai: cannot read model %s\n", path);
		return NULL;
	}
	memcpy(hdr, b + 4, sizeof(hdr));
	if (hdr[0] != 1) {
		fprintf(stderr, "brainai: unsupported model version %u\n", hdr[0]);
		return NULL;
	}
	m = xmalloc(sizeof(*m));
	m->blob = b;
	m->V = hdr[1]; m->Dm = hdr[2]; m->L = hdr[3]; m->E = hdr[4];
	m->N = hdr[5]; m->H = hdr[6]; m->P = hdr[7]; m->K = hdr[8];
	m->Din = 2 * m->E + 2 * m->N + m->H;
	m->Dconv = m->E + 2 * m->N;
	off = 40;
	/* TAKE: point `ptr` at the next `count` bytes of the file and skip them. */
#define TAKE(ptr, type, count) do { ptr = (type)(b + off); off += (count); } while (0)
	TAKE(m->emb_scale, const float *, 4 * m->V);
	TAKE(m->emb, const int8_t *, (size_t)m->V * m->Dm);
	m->layers = xmalloc(m->L * sizeof(layer_t));
	for (l = 0; l < m->L; l++) {
		layer_t *ly = &m->layers[l];
		TAKE(ly->norm, float *, 4 * m->Dm);
		memcpy(&ly->in_scale, b + off, 4); off += 4;
		/* ternary: 4 weights per byte, so rows * cols / 4 bytes */
		TAKE(ly->in_w, const uint8_t *, (size_t)m->Din * m->Dm / 4);
		TAKE(ly->conv_w, float *, 4 * m->Dconv * m->K);
		TAKE(ly->conv_b, float *, 4 * m->Dconv);
		TAKE(ly->dt_bias, float *, 4 * m->H);
		TAKE(ly->A, float *, 4 * m->H);
		TAKE(ly->D, float *, 4 * m->H);
		TAKE(ly->gnorm, float *, 4 * m->E);
		memcpy(&ly->out_scale, b + off, 4); off += 4;
		TAKE(ly->out_w, const uint8_t *, (size_t)m->Dm * m->E / 4);
	}
	TAKE(m->norm_f, float *, 4 * m->Dm);
#undef TAKE
	if (off != len) {
		fprintf(stderr, "brainai: model size mismatch (%zu != %zu)\n", off, len);
		return NULL;
	}
	for (l = 0; l < m->L; l++) {
		layer_t *ly = &m->layers[l];
		/* reorder both matrices for the cache-friendly lookup (section 6) */
		repack_ternary((uint8_t *)ly->in_w, m->Din, m->Dm / 4); /* in place */
		repack_ternary((uint8_t *)ly->out_w, m->Dm, m->E / 4);
		/* fixed-point copies: multiply by 2^20 or 2^16 and round */
		ly->conv_wq = xmalloc(4 * m->Dconv * m->K);
		ly->conv_bq = xmalloc(4 * m->Dconv);
		ly->gnorm_q = xmalloc(4 * m->E);
		for (i = 0; i < m->Dconv * m->K; i++)
			ly->conv_wq[i] = (int32_t)lrintf(ly->conv_w[i] * 1048576.0f); /* Q20 */
		for (i = 0; i < m->Dconv; i++)
			ly->conv_bq[i] = (int32_t)lrintf(ly->conv_b[i] * 65536.0f); /* Q16 */
		for (i = 0; i < m->E; i++)
			ly->gnorm_q[i] = (int32_t)lrintf(ly->gnorm[i] * 65536.0f); /* Q16 */
	}
	/*
	 * Sigmoid table: entry i holds sigmoid(-16 + i/32) in Q16, for
	 * i = 0 .. 1024.  sigmoid_q16() interpolates between neighbours.
	 */
	m->sig = xmalloc(2 * (SIG_N + 1));
	for (i = 0; i <= SIG_N; i++) {
		float x = -16.0f + i * (1.0f / 32.0f);
		long v = lrintf(65536.0f / (1.0f + expf(-x)));
		m->sig[i] = (uint16_t)(v > 65535 ? 65535 : v);
	}
	/*
	 * Embedding bytes become unsigned (e + 128) for the DSP LM head.
	 * Flipping the top bit (XOR, exclusive or, with 0x80) maps -128..127 to
	 * 0..255 in that order.  model_step() and the LM head undo the offset.
	 */
	{
		uint8_t *e = (uint8_t *)m->emb;
		size_t i, n = (size_t)m->V * m->Dm;
		for (i = 0; i < n; i++)
			e[i] ^= 0x80;
	}
	m->conv_state = xmalloc(sizeof(int32_t) * m->L * m->Dconv * (m->K - 1));
	m->ssm = xmalloc(sizeof(int32_t) * m->L * m->H * m->P * m->N);
	m->x = xmalloc(4 * m->Dm);
	m->hn = xmalloc(4 * m->E);
	m->zx = xmalloc(4 * m->Din);
	m->y = xmalloc(4 * m->E);
	m->xbc = xmalloc(4 * m->Dconv);
	m->q = xmalloc(m->E > m->Dm ? m->E : m->Dm);
	m->lut = xmalloc(2 * 81 * GB); /* one block's tables: GB groups x 81 x int16 */
	m->q2 = xmalloc(2 * (m->E > m->Dm ? m->E : m->Dm));
	m->acc = xmalloc(4 * m->Din);
	m->logits = xmalloc(4 * m->V);
	m->b16 = xmalloc(2 * m->N);
	m->cfx = xmalloc(4 * m->N);
	return m;
}

int model_vocab(const model_t *m) { return m->V; }

/* Forget the conversation: zero the convolution history and SSM state. */
void model_reset(model_t *m)
{
	memset(m->conv_state, 0, sizeof(int32_t) * m->L * m->Dconv * (m->K - 1));
	memset(m->ssm, 0, sizeof(int32_t) * m->L * m->H * m->P * m->N);
}

/*
 * Snapshot support: the model's entire memory of a conversation is these
 * two arrays, so saving and restoring them is enough.
 */
size_t model_state_size(const model_t *m)
{
	return sizeof(int32_t) * m->L * m->Dconv * (m->K - 1) +
	       sizeof(int32_t) * m->L * m->H * m->P * m->N;
}

void model_state_save(const model_t *m, void *dst)
{
	size_t a = sizeof(int32_t) * m->L * m->Dconv * (m->K - 1);
	memcpy(dst, m->conv_state, a);
	memcpy((char *)dst + a, m->ssm, sizeof(int32_t) * m->L * m->H * m->P * m->N);
}

void model_state_load(model_t *m, const void *src)
{
	size_t a = sizeof(int32_t) * m->L * m->Dconv * (m->K - 1);
	memcpy(m->conv_state, src, a);
	memcpy(m->ssm, (const char *)src + a, sizeof(int32_t) * m->L * m->H * m->P * m->N);
}

/*
 * Per-token absmax int8 quantisation, as BitNet b1.58 does for activations:
 *
 *   q[i] = round(127 * x[i] / max|x|)
 *
 * Returns the scale s_q = max|x| / 127, so that x[i] ~= q[i] * s_q.
 */
static float quantize(const float *x, int8_t *q, int n)
{
	float amax = 1e-6f, s;
	int i;
	for (i = 0; i < n; i++) {
		float a = fabsf(x[i]);
		if (a > amax)
			amax = a;
	}
	s = 127.0f / amax;
	for (i = 0; i < n; i++) {
		float v = x[i] * s;
		/* round half away from zero */
		q[i] = (int8_t)(v >= 0 ? (int)(v + 0.5f) : -(int)(-v + 0.5f));
	}
	return amax / 127.0f;
}

/*
 * Reorders a ternary matrix in place for ternary_matvec() (article section 6).
 *
 * Input (from convert.py): row-major, G bytes per row, byte g of row r holds
 * the base-3 code c = d0 + 3*d1 + 9*d2 + 27*d3 of weights W[r][4g..4g+3],
 * with d = weight + 1.
 *
 * Output: column blocks of GB groups; inside a block, rows in quads; for each
 * group, one 32-bit word with the four rows' codes:
 *
 *   block 0 (groups 0..31)                          block 1 (groups 32..63)
 *   [r0 r1 r2 r3][r0 r1 r2 r3] ... [r4 r5 r6 r7]...  [r0 r1 r2 r3] ...
 *    group 0      group 1           group 0
 *   └──────── rows 0..3 ────────┘ └─ rows 4..7 ─┘
 *
 * Each code is also doubled (2c, at most 160) so it is a byte offset into
 * the int16 table: the lookup needs no index arithmetic.  Rows must be a
 * multiple of 4.
 */
static void repack_ternary(uint8_t *w, uint32_t rows, uint32_t G)
{
	uint8_t *src = xmalloc((size_t)rows * G), *d = w;
	memcpy(src, w, (size_t)rows * G);
	uint32_t g0, r, g, k;
	for (g0 = 0; g0 < G; g0 += GB) {                 /* each column block */
		uint32_t gb = G - g0 < GB ? G - g0 : GB;    /* last block may be short */
		for (r = 0; r < rows; r += 4)               /* each quad of rows */
			for (g = 0; g < gb; g++)            /* each group in the block */
				for (k = 0; k < 4; k++)     /* the quad's four rows */
					*d++ = (uint8_t)(2 * src[(size_t)(r + k) * G + g0 + g]);
	}
	free(src);
}

/*
 * Builds the 81-entry table for each of `groups` groups of four inputs:
 *
 *   t[c] = sum over k of (d_k - 1) * q[k],   c = d0 + 3*d1 + 9*d2 + 27*d3
 *
 * i.e. the dot product of the four inputs with every possible ternary weight
 * pattern.  Splitting c = i + 9*j into a low half i (digits d0, d1) and a
 * high half j (digits d2, d3) means only 9 + 9 partial sums are computed;
 * each entry is then lo[i] + hi.  Nested loops avoid the slow % and / that a
 * flat loop over c would need.
 */
static void build_lut(int16_t *t, const int8_t *q, uint32_t groups)
{
	uint32_t g;
	int i, j;
	for (g = 0; g < groups; g++, q += 4, t += 81) {
		int16_t lo[9], hi;
		for (i = 0; i < 9; i++) /* i = d0 + 3*d1 */
			lo[i] = (int16_t)((i % 3 - 1) * q[0] + (i / 3 - 1) * q[1]);
		for (j = 0; j < 9; j++) { /* j = d2 + 3*d3 */
			hi = (int16_t)((j % 3 - 1) * q[2] + (j / 3 - 1) * q[3]);
			for (i = 0; i < 9; i++)
				t[j * 9 + i] = lo[i] + hi;
		}
	}
}

/*
 * LK(t, w): the int16 table entry at byte offset w from t.  Because the
 * repacked codes are already byte offsets (2c), this compiles to a single
 * "ldrsh rd, [t, w]" instruction.
 */
#define LK(t, w) (*(const int16_t *)((const char *)(t) + (w)))

/*
 * out[r] = sum over c of W[r][c] * q[c], for the repacked layout above.
 *
 * For each column block: build the block's tables (they fit in the cache),
 * then walk every quad of rows.  Each 32-bit word holds one group's codes
 * for four rows, so one load feeds four lookups into the same table.
 * Results of later blocks are added to those of earlier ones.
 */
static void ternary_matvec(int16_t *lut, const int8_t *q, uint32_t cols,
			   const uint8_t *w, uint32_t rows, int32_t *out)
{
	uint32_t G = cols / 4, g0, r, g;
	const uint32_t *wp = (const uint32_t *)w;
	for (g0 = 0; g0 < G; g0 += GB) {
		uint32_t gb = G - g0 < GB ? G - g0 : GB;
		build_lut(lut, q + 4 * g0, gb);
		for (r = 0; r < rows; r += 4) {
			const char *t = (const char *)lut;
			int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0; /* four rows' sums */
			uint32_t mk = 0xff;
			/* keep the mask in a register: "and rX, mk, v, lsr #8" is one
			 * instruction, an immediate mask after a shift is two */
			__asm__("" : "+r"(mk));
			for (g = 0; g < gb; g++, t += 162) { /* 162 bytes = one table */
				uint32_t v = *wp++;          /* codes of rows r..r+3 */
				s0 += LK(t, v & mk);
				s1 += LK(t, (v >> 8) & mk);
				s2 += LK(t, (v >> 16) & mk);
				s3 += LK(t, v >> 24);
			}
			if (g0) { /* not the first block: accumulate */
				out[r] += s0; out[r + 1] += s1; out[r + 2] += s2; out[r + 3] += s3;
			} else {
				out[r] = s0; out[r + 1] = s1; out[r + 2] = s2; out[r + 3] = s3;
			}
		}
	}
}

/*
 * Dot product for the LM head (article section 7): the sum over n words of
 * the four unsigned bytes in row[] times int8 q, where q2 holds q as signed
 * int16 pairs (q0,q2) and (q1,q3) per word.
 *
 * A word [b3 b2 b1 b0] is split with masks into [0 b2 0 b0] and [0 b3 0 b1].
 * SMLABB multiplies the bottom 16-bit halves of two registers and adds to an
 * accumulator; SMLATT does the same with the top halves.  So four
 * single-cycle instructions handle four bytes, instead of four byte loads
 * plus four multiply-adds.  The caller subtracts the +128 offset.
 *
 * __ARM_FEATURE_DSP is defined by the compiler when these instructions exist
 * (ARMv5TE and later); the second version is a portable fallback that gives
 * the same result, used for host builds.
 */
#if defined(__ARM_FEATURE_DSP)
#include <arm_acle.h>
static inline int32_t dot_u8_q(const uint32_t *row, const int32_t *q2, uint32_t n)
{
	int32_t a = 0, b = 0;
	uint32_t j;
	for (j = 0; j < n; j++, q2 += 2) {
		uint32_t w = row[j];
		int32_t u02 = (int32_t)(w & 0x00ff00ffu), u13 = (int32_t)((w >> 8) & 0x00ff00ffu);
		a = __smlabb(u02, q2[0], a); /* b0 * q0 */
		b = __smlatt(u02, q2[0], b); /* b2 * q2 */
		a = __smlabb(u13, q2[1], a); /* b1 * q1 */
		b = __smlatt(u13, q2[1], b); /* b3 * q3 */
	}
	return a + b;
}
#else
static inline int32_t dot_u8_q(const uint32_t *row, const int32_t *q2, uint32_t n)
{
	int32_t a = 0;
	uint32_t j;
	for (j = 0; j < n; j++, q2 += 2) {
		uint32_t w = row[j];
		a += (int32_t)(w & 0xff) * (int16_t)q2[0] + (int32_t)((w >> 16) & 0xff) * (int16_t)(q2[0] >> 16);
		a += (int32_t)((w >> 8) & 0xff) * (int16_t)q2[1] + (int32_t)(w >> 24) * (int16_t)(q2[1] >> 16);
	}
	return a;
}
#endif

/*
 * RMS norm times weights, followed by absmax int8 quantisation.
 *
 * RMS norm would compute u[i] = w[i] * x[i] / r with r = sqrt(mean(x^2)).
 * But quantisation divides by max|u| anyway, so r cancels out of q[] and
 * only enters the returned scale (article section 8, "the square root that
 * cancels").  That saves a division per element.
 */
static float norm_quantize(float *tmp, int8_t *q, const float *x, const float *w, int n)
{
	float ss = 0.0f, inv;
	int i;
	for (i = 0; i < n; i++) {
		ss += x[i] * x[i];
		tmp[i] = x[i] * w[i];
	}
	inv = 1.0f / sqrtf(ss / n + 1e-5f); /* 1 / r, with the model's epsilon */
	return quantize(tmp, q, n) * inv;
}

/*
 * Fixed point for the per-layer element-wise work (article section 8).
 * Soft float is the main cost on this CPU, so: activations in Q16 (value =
 * integer / 2^16), convolution weights in Q20, SSM state in Q20, decay in
 * Q31.  Multiplying Qa by Qb gives Q(a+b); shifting right by s gives Q(a-s).
 */
#define SIG_LO (-16 * 65536) /* table starts at x = -16, in Q16 */
#define SIG_STEP 11          /* table step 1/32 = 2^-5 = 2^11 in Q16 */

/*
 * (a * (int16_t)b) >> 16 in one SMULWB instruction ("signed multiply word
 * by bottom halfword").  GCC's arm_acle.h does not provide it, hence the
 * inline assembly.
 */
static inline int32_t smulwb(int32_t a, int32_t b)
{
#if defined(__ARM_FEATURE_DSP)
	int32_t r;
	__asm__("smulwb %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
	return r;
#else
	return (int32_t)(((int64_t)a * (int16_t)b) >> 16);
#endif
}

/* (a * b) >> sh with a 64-bit intermediate, so nothing overflows. */
static inline int32_t mulsh(int32_t a, int32_t b, int sh)
{
	return (int32_t)(((int64_t)a * b) >> sh);
}

/*
 * sigmoid(x) for x in Q16, result in Q16 (0 .. 65536).
 *
 * u = x + 16 (in Q16) is the distance from the table start; its top bits
 * (u >> 11) are the entry index i, its low 11 bits the fraction between
 * entries i and i+1, used for linear interpolation.  Beyond the table,
 * sigmoid is 0 or 1 to within Q16 precision.
 */
static inline int32_t sigmoid_q16(const uint16_t *t, int32_t x)
{
	uint32_t u, i, f;
	if (x <= SIG_LO)
		return 0;
	u = (uint32_t)(x - SIG_LO);
	i = u >> SIG_STEP;
	if (i >= SIG_N)
		return 65536;
	f = u & ((1u << SIG_STEP) - 1);
	return t[i] + (int32_t)(((int32_t)(t[i + 1] - t[i]) * (int32_t)f) >> SIG_STEP);
}

/* SiLU(x) = x * sigmoid(x); Q16 * Q16 >> 16 = Q16. */
static inline int32_t silu_q16(const uint16_t *t, int32_t x)
{
	return mulsh(x, sigmoid_q16(t, x), 16);
}

/*
 * Turns a float scale s into an integer multiplier and shift, so that
 * per-element scaling needs no float:
 *
 *   s = f * 2^e with 0.5 <= |f| < 1 (frexpf),  mul = f * 2^30,  sh = 14 - e
 *   (a * mul) >> sh = a * f * 2^(30 - 14 + e) = a * s * 2^16
 *
 * i.e. the result is a * s in Q16.
 */
static void q16_mult(float s, int32_t *mul, int *sh)
{
	int e;
	float f = frexpf(s, &e); /* s = f * 2^e, 0.5 <= |f| < 1 */
	*mul = (int32_t)(f * (float)(1 << 30));
	*sh = 14 - e;
	if (*sh > 62)
		*mul = 0, *sh = 0; /* s is so small the result is 0 */
	if (*sh < 0)
		*mul <<= -*sh, *sh = 0; /* |s| >= 2^14 never happens in practice */
}

/*
 * One Mamba-2 layer for one token (article section 2 has the diagram).
 * Reads and updates m->x; updates this layer's convolution history and SSM
 * state.
 */
static void layer_step(model_t *m, uint32_t l)
{
	layer_t *ly = &m->layers[l];
	const uint32_t E = m->E, N = m->N, H = m->H, P = m->P, K = m->K;
	/*
	 * zx holds the input matrix's output, Din = 2E + 2N + H values:
	 *   zx[0 .. E)            gate z
	 *   zx[E .. 2E + 2N)      x', B, C: the Dconv channels for the conv
	 *   zx[2E + 2N .. Din)    dt, one per head
	 * After the convolution, xbc holds x' (E), B (N), C (N), all Q16.
	 */
	int32_t *zx = m->zx, *xbc = m->xbc, *y = m->y;
	const int32_t *dt = zx + 2 * E + 2 * N;
	int32_t *cs = m->conv_state + (size_t)l * m->Dconv * (K - 1);
	int32_t *ssm = m->ssm + (size_t)l * H * P * N;
	const uint16_t *sig = m->sig;
	float s;
	int32_t mul, amax;
	int sh;
	int64_t ss;
	uint32_t i, k, h, p, n;

	/* 1. Normalise x, round to int8, multiply by the input matrix. */
	s = norm_quantize(m->hn, m->q, m->x, ly->norm, m->Dm) * ly->in_scale;
	PT(0);
	ternary_matvec(m->lut, m->q, m->Dm, ly->in_w, m->Din, m->acc);
	PT(1);
	/* integer sums -> real values in Q16: acc * s * 2^16 */
	q16_mult(s, &mul, &sh);
	for (i = 0; i < m->Din; i++)
		zx[i] = (int32_t)(((int64_t)m->acc[i] * mul) >> sh);

	/*
	 * 2. Causal depthwise convolution over [x', B, C]: each channel mixes its
	 * current value with its previous K-1 = 3 values, then SiLU.
	 *   v = w[3]*cur + w[0]*h[0] + w[1]*h[1] + w[2]*h[2]   (Q20 * Q16 = Q36)
	 * The history shifts by one, oldest value dropped.
	 */
	for (i = 0; i < m->Dconv; i++) {
		const int32_t *w = ly->conv_wq + i * K;
		int32_t *hist = cs + i * (K - 1);
		int32_t cur = zx[E + i];
		int64_t v = (int64_t)w[K - 1] * cur;
		for (k = 0; k < K - 1; k++)
			v += (int64_t)w[k] * hist[k];
		for (k = 0; k + 1 < K - 1; k++)
			hist[k] = hist[k + 1];
		hist[K - 2] = cur;
		xbc[i] = silu_q16(sig, (int32_t)(v >> 20) + ly->conv_bq[i]); /* Q36 >> 20 = Q16 */
	}
	for (n = 0; n < N; n++) /* C: Q16 -> Q20, to match the state */
		m->cfx[n] = xbc[E + N + n] * 16;
	PT(2);

	/*
	 * 3. SSM state update (article sections 2 and 8).  Per head h, channel p
	 * and state index n:
	 *   delta  = softplus(dt[h] + dt_bias[h])
	 *   decay  = exp(delta * A[h])              (0 < decay < 1, since A < 0)
	 *   S[n]   = decay * S[n] + delta * x * B[n]
	 *   y      = sum over n of S[n] * C[n]  +  D[h] * x
	 */
	{
		/* B as int16 (Q16 >> sb) so delta*x*B is one 32x16 SMULWB.  sb is
		 * the smallest shift that makes every B[n] fit in 16 bits. */
		int32_t bmax = 1, sb = 0;
		for (n = 0; n < N; n++) {
			int32_t a = xbc[E + n] < 0 ? -xbc[E + n] : xbc[E + n];
			if (a > bmax)
				bmax = a;
		}
		while ((bmax >> sb) > 32767)
			sb++;
		for (n = 0; n < N; n++)
			m->b16[n] = (int16_t)(xbc[E + n] >> sb);
		for (h = 0; h < H; h++) {
			/* per head, in float: only 16 of these per layer */
			float d = dt[h] * (1.0f / 65536.0f) + ly->dt_bias[h];
			float dtv = d > 20.0f ? d : log1pf(expf(d)); /* softplus */
			float da = expf(dtv * ly->A[h]) * 2147483648.0f;
			int32_t dA = da >= 2147483647.0f ? 2147483647 : (int32_t)da; /* Q31 */
			int32_t dtq = (int32_t)(dtv * 65536.0f), Dq = (int32_t)(ly->D[h] * 65536.0f);
			for (p = 0; p < P; p++) {
				uint32_t c = h * P + p;
				int32_t xv = xbc[c];
				int32_t dtx = mulsh(dtq, xv, 12); /* Q16 * Q16 >> 12 = Q20 */
				int32_t *st = ssm + (size_t)c * N;
				const int16_t *b16 = m->b16;
				const int32_t *cf = m->cfx;
				int64_t acc = 0;
				for (n = 0; n < N; n++) {
					/* decay * S: high word of Q20 * Q31 is Q19, << 1 -> Q20 */
					int32_t hi = (int32_t)(((int64_t)st[n] * dA) >> 32);
					/* delta*x*B: Q20 * Q(16-sb) >> 16, << sb -> Q20 */
					int32_t v = (smulwb(dtx, b16[n]) << sb) + (hi << 1);
					st[n] = v;
					acc += (int64_t)v * cf[n]; /* Q20 * Q20 = Q40 */
				}
				/* Q40 >> 24 = Q16, plus the skip term D * x in Q16 */
				y[c] = (int32_t)(acc >> (2 * FX - 16)) + mulsh(Dq, xv, 16);
			}
		}
	}

	PT(3);
	/*
	 * 4. Gated RMSNorm (norm_before_gate = false), then int8 quantisation:
	 *   v[i] = y[i] * SiLU(z[i]) * gnorm[i] / rms(y * SiLU(z))
	 * The RMS factor is common to all elements, so it cancels in the absmax
	 * quantisation and only enters the output scale (section 8).  So the
	 * loop only collects the sum of squares (ss) and the largest |v| (amax).
	 */
	ss = 0;
	amax = 1;
	for (i = 0; i < E; i++) {
		int32_t v = mulsh(y[i], silu_q16(sig, zx[i]), 16), a;
		ss += (int64_t)v * v;                 /* Q32 */
		v = mulsh(v, ly->gnorm_q[i], 16);
		y[i] = v;
		a = v < 0 ? -v : v;
		if (a > amax)
			amax = a;
	}
	{
		/* q[i] = round(127 * v[i] / amax), with 127 / amax as a 32.32
		 * fixed-point reciprocal R so the loop multiplies instead of divides */
		int64_t R = ((int64_t)127 << 32) / amax;
		float inv = 1.0f / sqrtf((float)ss * (1.0f / 4294967296.0f) / E + 1e-5f);
		for (i = 0; i < E; i++) {
			int32_t v = y[i];
			int32_t qv = (int32_t)(((int64_t)(v < 0 ? -v : v) * R + ((int64_t)1 << 31)) >> 32);
			m->q[i] = (int8_t)(v < 0 ? -qv : qv);
		}
		/* the real scale: (1 / rms) * (amax in real units) / 127 * s_W */
		s = inv * ((float)amax * (1.0f / 65536.0f)) * (1.0f / 127.0f) * ly->out_scale;
	}
	PT(4);
	/* 5. Output matrix, then add the result back onto x (residual). */
	ternary_matvec(m->lut, m->q, E, ly->out_w, m->Dm, m->acc);
	PT(5);
	for (i = 0; i < m->Dm; i++)
		m->x[i] += m->acc[i] * s;
	PT(4);
}

/*
 * Feeds one token through the model.  Returns the V logits when want_logits
 * is set, NULL otherwise (prompt tokens before the last one don't need the
 * expensive LM head).
 */
const float *model_step(model_t *m, int token, int want_logits)
{
	uint32_t i, l, v;
	const uint8_t *e = (const uint8_t *)m->emb + (size_t)token * m->Dm;
	float es = m->emb_scale[token], s;
	int32_t qsum = 0, bias;
	PSTART();
	/* embedding lookup; bytes are stored as e + 128 (see model_load) */
	for (i = 0; i < m->Dm; i++)
		m->x[i] = ((int)e[i] - 128) * es;
	for (l = 0; l < m->L; l++)
		layer_step(m, l);
	if (!want_logits)
		return NULL;
	PT(7);
	/*
	 * LM head (article section 7): logit[v] = s_v * s_q * sum e[v][i] * q[i].
	 * First pack q as int16 pairs (q0,q2) and (q1,q3) per four values, as
	 * dot_u8_q() expects.  Since the table stores e + 128,
	 *   sum (e + 128) * q = sum e * q + 128 * sum q,
	 * so `bias` = 128 * sum q is subtracted once per row.
	 */
	s = norm_quantize(m->hn, m->q, m->x, m->norm_f, m->Dm);
	for (i = 0; i < m->Dm; i += 4) {
		const int8_t *q = m->q + i;
		m->q2[i / 2] = (int32_t)((uint16_t)q[0] | ((uint32_t)(uint16_t)q[2] << 16));
		m->q2[i / 2 + 1] = (int32_t)((uint16_t)q[1] | ((uint32_t)(uint16_t)q[3] << 16));
		qsum += q[0] + q[1] + q[2] + q[3];
	}
	bias = 128 * qsum;
	for (v = 0; v < m->V; v++) {
		const uint32_t *row = (const uint32_t *)((const uint8_t *)m->emb + (size_t)v * m->Dm);
		m->logits[v] = (float)(dot_u8_q(row, m->q2, m->Dm / 4) - bias) * s * m->emb_scale[v];
	}
	PT(6);
	return m->logits;
}
