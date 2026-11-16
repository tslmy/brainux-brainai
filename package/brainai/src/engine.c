/*
 * brainai inference engine: Mamba-2 + BitNet b1.58 (ternary) language model.
 *
 * Written for the ARM926EJ-S in SHARP Brain (no FPU, 16 KiB D-cache):
 *  - ternary projections use a lookup table per group of four int8
 *    activations (81 pre-summed entries), so one table load replaces four
 *    multiply-adds;
 *  - the SSM recurrence runs in 32-bit fixed point;
 *  - the tied embedding / LM head is int8 x int8;
 *  - only small element-wise work (norms, SiLU, conv) uses (soft) float.
 */
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "brainai.h"

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

#define FX 20 /* Q11.20 for SSM activations and state */
#define FXA 30
#ifndef GB
#define GB 32 /* column groups per ternary block, see repack_ternary */
#endif /* Q1.30 for the per-step decay */

typedef struct {
	float *norm;
	float in_scale;
	const uint8_t *in_w;
	float *conv_w, *conv_b, *dt_bias, *A, *D, *gnorm;
	int32_t *conv_wq, *conv_bq, *gnorm_q; /* Q20, Q16, Q16 */
	float out_scale;
	const uint8_t *out_w;
} layer_t;

struct model {
	uint32_t V, Dm, L, E, N, H, P, K, Din, Dconv;
	uint8_t *blob;
	const float *emb_scale;
	const int8_t *emb;
	layer_t *layers;
	float *norm_f;
	/* recurrent state */
	int32_t *conv_state; /* L * Dconv * (K-1), Q16 */
	int32_t *ssm; /* L * H * P * N */
	/* scratch */
	float *x, *hn;
	int32_t *zx, *y, *xbc; /* Q16 */
	uint16_t *sig; /* sigmoid table, Q16 */
	int8_t *q;
	int32_t *q2; /* q packed as int16 pairs (0,2) (1,3) for the LM head */
	int16_t *lut;
	int32_t *acc;
	float *logits;
	int32_t *cfx;
	int16_t *b16;
};

static void *xmalloc(size_t n)
{
	void *p = calloc(1, n);
	if (!p) {
		fprintf(stderr, "brainai: out of memory\n");
		exit(1);
	}
	return p;
}

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
#define SIG_N 1024

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
#define TAKE(ptr, type, count) do { ptr = (type)(b + off); off += (count); } while (0)
	TAKE(m->emb_scale, const float *, 4 * m->V);
	TAKE(m->emb, const int8_t *, (size_t)m->V * m->Dm);
	m->layers = xmalloc(m->L * sizeof(layer_t));
	for (l = 0; l < m->L; l++) {
		layer_t *ly = &m->layers[l];
		TAKE(ly->norm, float *, 4 * m->Dm);
		memcpy(&ly->in_scale, b + off, 4); off += 4;
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
		repack_ternary((uint8_t *)ly->in_w, m->Din, m->Dm / 4); /* in place */
		repack_ternary((uint8_t *)ly->out_w, m->Dm, m->E / 4);
		ly->conv_wq = xmalloc(4 * m->Dconv * m->K);
		ly->conv_bq = xmalloc(4 * m->Dconv);
		ly->gnorm_q = xmalloc(4 * m->E);
		for (i = 0; i < m->Dconv * m->K; i++)
			ly->conv_wq[i] = (int32_t)lrintf(ly->conv_w[i] * 1048576.0f);
		for (i = 0; i < m->Dconv; i++)
			ly->conv_bq[i] = (int32_t)lrintf(ly->conv_b[i] * 65536.0f);
		for (i = 0; i < m->E; i++)
			ly->gnorm_q[i] = (int32_t)lrintf(ly->gnorm[i] * 65536.0f);
	}
	m->sig = xmalloc(2 * (SIG_N + 1));
	for (i = 0; i <= SIG_N; i++) {
		float x = -16.0f + i * (1.0f / 32.0f);
		long v = lrintf(65536.0f / (1.0f + expf(-x)));
		m->sig[i] = (uint16_t)(v > 65535 ? 65535 : v);
	}
	/* embedding bytes become unsigned (e + 128) for the DSP LM head */
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
	m->lut = xmalloc(2 * 81 * GB);
	m->q2 = xmalloc(2 * (m->E > m->Dm ? m->E : m->Dm));
	m->acc = xmalloc(4 * m->Din);
	m->logits = xmalloc(4 * m->V);
	m->b16 = xmalloc(2 * m->N);
	m->cfx = xmalloc(4 * m->N);
	return m;
}

int model_vocab(const model_t *m) { return m->V; }

void model_reset(model_t *m)
{
	memset(m->conv_state, 0, sizeof(int32_t) * m->L * m->Dconv * (m->K - 1));
	memset(m->ssm, 0, sizeof(int32_t) * m->L * m->H * m->P * m->N);
}

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

/* Per-token absmax int8 quantisation, as in BitNet b1.58 activations. */
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
		q[i] = (int8_t)(v >= 0 ? (int)(v + 0.5f) : -(int)(-v + 0.5f));
	}
	return amax / 127.0f;
}

/*
 * Ternary weights are repacked at load time into column blocks of GB groups
 * (4 columns each).  Inside a block, rows go in quads; for each group a quad
 * stores one 32-bit word holding that group's code for its four rows.  Every
 * base-3 code is doubled so it is a byte offset into an int16 table.  Each
 * block gets its own table (81 entries per group), small enough to stay in
 * the 16 KiB D-cache while the weights stream past, and the four lookups in
 * a word share one table base.  Rows must be a multiple of 4.
 */
static void repack_ternary(uint8_t *w, uint32_t rows, uint32_t G)
{
	uint8_t *src = xmalloc((size_t)rows * G), *d = w;
	memcpy(src, w, (size_t)rows * G);
	uint32_t g0, r, g, k;
	for (g0 = 0; g0 < G; g0 += GB) {
		uint32_t gb = G - g0 < GB ? G - g0 : GB;
		for (r = 0; r < rows; r += 4)
			for (g = 0; g < gb; g++)
				for (k = 0; k < 4; k++)
					*d++ = (uint8_t)(2 * src[(size_t)(r + k) * G + g0 + g]);
	}
	free(src);
}

static void build_lut(int16_t *t, const int8_t *q, uint32_t groups)
{
	uint32_t g;
	int i, j;
	for (g = 0; g < groups; g++, q += 4, t += 81) {
		int16_t lo[9], hi;
		for (i = 0; i < 9; i++)
			lo[i] = (int16_t)((i % 3 - 1) * q[0] + (i / 3 - 1) * q[1]);
		for (j = 0; j < 9; j++) {
			hi = (int16_t)((j % 3 - 1) * q[2] + (j / 3 - 1) * q[3]);
			for (i = 0; i < 9; i++)
				t[j * 9 + i] = lo[i] + hi;
		}
	}
}

#define LK(t, w) (*(const int16_t *)((const char *)(t) + (w)))

/* out[r] = sum_c W[r][c] * q[c] over the repacked layout above. */
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
			int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
			uint32_t mk = 0xff;
			/* keep the mask in a register: "and rX, mk, v, lsr #8" is one
			 * instruction, an immediate mask after a shift is two */
			__asm__("" : "+r"(mk));
			for (g = 0; g < gb; g++, t += 162) {
				uint32_t v = *wp++;
				s0 += LK(t, v & mk);
				s1 += LK(t, (v >> 8) & mk);
				s2 += LK(t, (v >> 16) & mk);
				s3 += LK(t, v >> 24);
			}
			if (g0) {
				out[r] += s0; out[r + 1] += s1; out[r + 2] += s2; out[r + 3] += s3;
			} else {
				out[r] = s0; out[r + 1] = s1; out[r + 2] = s2; out[r + 3] = s3;
			}
		}
	}
}

/*
 * sum over n words of the four unsigned bytes in row[] times int8 q, where
 * q2 holds q as signed int16 pairs (q0,q2) and (q1,q3) per word.  On ARMv5TE
 * this is four SMLAxy per word instead of four byte loads plus MLAs.
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
		a = __smlabb(u02, q2[0], a);
		b = __smlatt(u02, q2[0], b);
		a = __smlabb(u13, q2[1], a);
		b = __smlatt(u13, q2[1], b);
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
 * rmsnorm(x) * w followed by absmax int8 quantisation.  The RMS factor is
 * common to every element, so it cancels in q and only enters the scale.
 */
static float norm_quantize(float *tmp, int8_t *q, const float *x, const float *w, int n)
{
	float ss = 0.0f, inv;
	int i;
	for (i = 0; i < n; i++) {
		ss += x[i] * x[i];
		tmp[i] = x[i] * w[i];
	}
	inv = 1.0f / sqrtf(ss / n + 1e-5f);
	return quantize(tmp, q, n) * inv;
}

/*
 * Fixed point for the per-layer element-wise work (soft float is the cost on
 * this CPU): activations in Q16, conv weights in Q20, SSM state in Q20.
 */
#define SIG_LO (-16 * 65536)
#define SIG_STEP 11 /* table step 2^-5 in Q16 */

/* (a * (int16_t)b) >> 16, one SMULWB on ARMv5TE */
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

static inline int32_t mulsh(int32_t a, int32_t b, int sh)
{
	return (int32_t)(((int64_t)a * b) >> sh);
}

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

static inline int32_t silu_q16(const uint16_t *t, int32_t x)
{
	return mulsh(x, sigmoid_q16(t, x), 16);
}

/* s as a multiplier and shift so that (a * mul) >> sh == a * s * 2^16 */
static void q16_mult(float s, int32_t *mul, int *sh)
{
	int e;
	float f = frexpf(s, &e); /* s = f * 2^e, 0.5 <= |f| < 1 */
	*mul = (int32_t)(f * (float)(1 << 30));
	*sh = 14 - e;
	if (*sh > 62)
		*mul = 0, *sh = 0;
	if (*sh < 0)
		*mul <<= -*sh, *sh = 0; /* |s| >= 2^14 never happens in practice */
}

static void layer_step(model_t *m, uint32_t l)
{
	layer_t *ly = &m->layers[l];
	const uint32_t E = m->E, N = m->N, H = m->H, P = m->P, K = m->K;
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

	s = norm_quantize(m->hn, m->q, m->x, ly->norm, m->Dm) * ly->in_scale;
	PT(0);
	ternary_matvec(m->lut, m->q, m->Dm, ly->in_w, m->Din, m->acc);
	PT(1);
	q16_mult(s, &mul, &sh);
	for (i = 0; i < m->Din; i++)
		zx[i] = (int32_t)(((int64_t)m->acc[i] * mul) >> sh);

	/* causal depthwise conv over [x, B, C] with a K-1 sample history */
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
		xbc[i] = silu_q16(sig, (int32_t)(v >> 20) + ly->conv_bq[i]);
	}
	for (n = 0; n < N; n++) /* Q16 -> Q20 */
		m->cfx[n] = xbc[E + N + n] * 16;
	PT(2);

	/* B as int16 (Q16 >> sb) so dtx * B is one 32x16 multiply */
	{
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
			float d = dt[h] * (1.0f / 65536.0f) + ly->dt_bias[h];
			float dtv = d > 20.0f ? d : log1pf(expf(d)); /* softplus */
			float da = expf(dtv * ly->A[h]) * 2147483648.0f;
			int32_t dA = da >= 2147483647.0f ? 2147483647 : (int32_t)da; /* Q31 */
			int32_t dtq = (int32_t)(dtv * 65536.0f), Dq = (int32_t)(ly->D[h] * 65536.0f);
			for (p = 0; p < P; p++) {
				uint32_t c = h * P + p;
				int32_t xv = xbc[c];
				int32_t dtx = mulsh(dtq, xv, 12); /* Q20 */
				int32_t *st = ssm + (size_t)c * N;
				const int16_t *b16 = m->b16;
				const int32_t *cf = m->cfx;
				int64_t acc = 0;
				for (n = 0; n < N; n++) {
					int32_t hi = (int32_t)(((int64_t)st[n] * dA) >> 32);
					int32_t v = (smulwb(dtx, b16[n]) << sb) + (hi << 1);
					st[n] = v;
					acc += (int64_t)v * cf[n];
				}
				y[c] = (int32_t)(acc >> (2 * FX - 16)) + mulsh(Dq, xv, 16);
			}
		}
	}

	PT(3);
	/*
	 * Gated RMSNorm (norm_before_gate = false), then int8 quantisation.  The
	 * RMS factor is common to all elements, so it cancels in the absmax
	 * quantisation and only enters the output scale.
	 */
	ss = 0;
	amax = 1;
	for (i = 0; i < E; i++) {
		int32_t v = mulsh(y[i], silu_q16(sig, zx[i]), 16), a;
		ss += (int64_t)v * v;
		v = mulsh(v, ly->gnorm_q[i], 16);
		y[i] = v;
		a = v < 0 ? -v : v;
		if (a > amax)
			amax = a;
	}
	{
		int64_t R = ((int64_t)127 << 32) / amax;
		float inv = 1.0f / sqrtf((float)ss * (1.0f / 4294967296.0f) / E + 1e-5f);
		for (i = 0; i < E; i++) {
			int32_t v = y[i];
			int32_t qv = (int32_t)(((int64_t)(v < 0 ? -v : v) * R + ((int64_t)1 << 31)) >> 32);
			m->q[i] = (int8_t)(v < 0 ? -qv : qv);
		}
		s = inv * ((float)amax * (1.0f / 65536.0f)) * (1.0f / 127.0f) * ly->out_scale;
	}
	PT(4);
	ternary_matvec(m->lut, m->q, E, ly->out_w, m->Dm, m->acc);
	PT(5);
	for (i = 0; i < m->Dm; i++)
		m->x[i] += m->acc[i] * s;
	PT(4);
}

const float *model_step(model_t *m, int token, int want_logits)
{
	uint32_t i, l, v;
	const uint8_t *e = (const uint8_t *)m->emb + (size_t)token * m->Dm;
	float es = m->emb_scale[token], s;
	int32_t qsum = 0, bias;
	PSTART();
	for (i = 0; i < m->Dm; i++)
		m->x[i] = ((int)e[i] - 128) * es;
	for (l = 0; l < m->L; l++)
		layer_step(m, l);
	if (!want_logits)
		return NULL;
	PT(7);
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
