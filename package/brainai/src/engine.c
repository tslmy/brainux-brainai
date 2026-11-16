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

#define FX 20 /* Q11.20 for SSM activations and state */
#define FXA 30 /* Q1.30 for the per-step decay */

typedef struct {
	float *norm;
	float in_scale;
	const uint8_t *in_w;
	float *conv_w, *conv_b, *dt_bias, *A, *D, *gnorm;
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
	float *conv_state; /* L * Dconv * (K-1) */
	int32_t *ssm; /* L * H * P * N */
	/* scratch */
	float *x, *hn, *zx, *y, *xbc;
	int8_t *q;
	int16_t *lut;
	int32_t *acc;
	float *logits;
	int32_t *bfx, *cfx;
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

model_t *model_load(const char *path)
{
	size_t len, off;
	uint32_t hdr[9], l;
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
	m->conv_state = xmalloc(sizeof(float) * m->L * m->Dconv * (m->K - 1));
	m->ssm = xmalloc(sizeof(int32_t) * m->L * m->H * m->P * m->N);
	m->x = xmalloc(4 * m->Dm);
	m->hn = xmalloc(4 * m->E);
	m->zx = xmalloc(4 * m->Din);
	m->y = xmalloc(4 * m->E);
	m->xbc = xmalloc(4 * m->Dconv);
	m->q = xmalloc(m->E > m->Dm ? m->E : m->Dm);
	m->lut = xmalloc(2 * 81 * (m->E / 4));
	m->acc = xmalloc(4 * m->Din);
	m->logits = xmalloc(4 * m->V);
	m->bfx = xmalloc(4 * m->N);
	m->cfx = xmalloc(4 * m->N);
	return m;
}

int model_vocab(const model_t *m) { return m->V; }

void model_reset(model_t *m)
{
	memset(m->conv_state, 0, sizeof(float) * m->L * m->Dconv * (m->K - 1));
	memset(m->ssm, 0, sizeof(int32_t) * m->L * m->H * m->P * m->N);
}

size_t model_state_size(const model_t *m)
{
	return sizeof(float) * m->L * m->Dconv * (m->K - 1) +
	       sizeof(int32_t) * m->L * m->H * m->P * m->N;
}

void model_state_save(const model_t *m, void *dst)
{
	size_t a = sizeof(float) * m->L * m->Dconv * (m->K - 1);
	memcpy(dst, m->conv_state, a);
	memcpy((char *)dst + a, m->ssm, sizeof(int32_t) * m->L * m->H * m->P * m->N);
}

void model_state_load(model_t *m, const void *src)
{
	size_t a = sizeof(float) * m->L * m->Dconv * (m->K - 1);
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

/* out[r] = sum_c W[r][c] * q[c], W ternary packed base-3, four per byte. */
static void ternary_matvec(int16_t *lut, const int8_t *q, int cols,
			   const uint8_t *w, int rows, int32_t *out)
{
	int G = cols / 4, g, r, i;
	for (g = 0; g < G; g++) {
		int16_t lo[9], hi[9], *t = lut + g * 81;
		int a = q[4 * g], b = q[4 * g + 1], c = q[4 * g + 2], d = q[4 * g + 3];
		for (i = 0; i < 9; i++) {
			lo[i] = (int16_t)((i % 3 - 1) * a + (i / 3 - 1) * b);
			hi[i] = (int16_t)((i % 3 - 1) * c + (i / 3 - 1) * d);
		}
		for (i = 0; i < 81; i++)
			t[i] = lo[i % 9] + hi[i / 9];
	}
	for (r = 0; r < rows; r++) {
		const uint8_t *wr = w + (size_t)r * G;
		const int16_t *t = lut;
		int32_t s0 = 0, s1 = 0;
		for (g = 0; g + 1 < G; g += 2, t += 162) {
			s0 += t[wr[g]];
			s1 += t[81 + wr[g + 1]];
		}
		if (g < G)
			s0 += t[wr[g]];
		out[r] = s0 + s1;
	}
}

static void rmsnorm(float *o, const float *x, const float *w, int n)
{
	float ss = 0.0f, s;
	int i;
	for (i = 0; i < n; i++)
		ss += x[i] * x[i];
	s = 1.0f / sqrtf(ss / n + 1e-5f);
	for (i = 0; i < n; i++)
		o[i] = x[i] * s * w[i];
}

static inline float silu(float v) { return v / (1.0f + expf(-v)); }

static inline int32_t tofx(float v)
{
	float s = v * (float)(1 << FX);
	if (s > 2147483000.0f) return 2147483000;
	if (s < -2147483000.0f) return -2147483000;
	return (int32_t)s;
}

static void layer_step(model_t *m, uint32_t l)
{
	layer_t *ly = &m->layers[l];
	const uint32_t E = m->E, N = m->N, H = m->H, P = m->P, K = m->K;
	float *z = m->zx, *xbc = m->xbc, *dt = m->zx + 2 * E + 2 * N;
	float *cs = m->conv_state + (size_t)l * m->Dconv * (K - 1);
	int32_t *ssm = m->ssm + (size_t)l * H * P * N;
	float s;
	uint32_t i, k, h, p, n;

	rmsnorm(m->hn, m->x, ly->norm, m->Dm);
	s = quantize(m->hn, m->q, m->Dm) * ly->in_scale;
	ternary_matvec(m->lut, m->q, m->Dm, ly->in_w, m->Din, m->acc);
	for (i = 0; i < m->Din; i++)
		m->zx[i] = m->acc[i] * s;

	/* causal depthwise conv over [x, B, C] with a K-1 sample history */
	for (i = 0; i < m->Dconv; i++) {
		const float *w = ly->conv_w + i * K;
		float *hist = cs + i * (K - 1);
		float cur = m->zx[E + i], v = ly->conv_b[i];
		for (k = 0; k < K - 1; k++)
			v += w[k] * hist[k];
		v += w[K - 1] * cur;
		for (k = 0; k + 1 < K - 1; k++)
			hist[k] = hist[k + 1];
		hist[K - 2] = cur;
		xbc[i] = silu(v);
	}
	for (n = 0; n < N; n++) {
		m->bfx[n] = tofx(xbc[E + n]);
		m->cfx[n] = tofx(xbc[E + N + n]);
	}

	for (h = 0; h < H; h++) {
		float d = dt[h] + ly->dt_bias[h];
		float dtv = d > 20.0f ? d : log1pf(expf(d)); /* softplus */
		int32_t dA = (int32_t)(expf(dtv * ly->A[h]) * (float)(1 << FXA));
		float Dh = ly->D[h];
		for (p = 0; p < P; p++) {
			uint32_t c = h * P + p;
			float xv = xbc[c];
			int32_t dtx = tofx(dtv * xv);
			int32_t *st = ssm + (size_t)c * N;
			int64_t acc = 0;
			for (n = 0; n < N; n++) {
				int32_t v = (int32_t)(((int64_t)st[n] * dA) >> FXA) +
					    (int32_t)(((int64_t)dtx * m->bfx[n]) >> FX);
				st[n] = v;
				acc += (int64_t)v * m->cfx[n];
			}
			m->y[c] = (float)acc * (1.0f / ((float)(1 << FX) * (float)(1 << FX))) +
				  Dh * xv;
		}
	}

	/* gated RMSNorm (norm_before_gate = false), then out_proj + residual */
	for (i = 0; i < E; i++)
		m->y[i] *= silu(z[i]);
	rmsnorm(m->y, m->y, ly->gnorm, E);
	s = quantize(m->y, m->q, E) * ly->out_scale;
	ternary_matvec(m->lut, m->q, E, ly->out_w, m->Dm, m->acc);
	for (i = 0; i < m->Dm; i++)
		m->x[i] += m->acc[i] * s;
}

const float *model_step(model_t *m, int token, int want_logits)
{
	uint32_t i, l, v;
	const int8_t *e = m->emb + (size_t)token * m->Dm;
	float es = m->emb_scale[token], s;
	for (i = 0; i < m->Dm; i++)
		m->x[i] = e[i] * es;
	for (l = 0; l < m->L; l++)
		layer_step(m, l);
	if (!want_logits)
		return NULL;
	rmsnorm(m->hn, m->x, m->norm_f, m->Dm);
	s = quantize(m->hn, m->q, m->Dm);
	for (v = 0; v < m->V; v++) {
		const int8_t *row = m->emb + (size_t)v * m->Dm;
		const int8_t *q = m->q;
		int32_t a0 = 0, a1 = 0;
		for (i = 0; i < m->Dm; i += 4) {
			a0 += row[i] * q[i] + row[i + 1] * q[i + 1];
			a1 += row[i + 2] * q[i + 2] + row[i + 3] * q[i + 3];
		}
		m->logits[v] = (float)(a0 + a1) * s * m->emb_scale[v];
	}
	return m->logits;
}
