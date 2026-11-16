/*
 * bench: time model_step on the device, per stage when built with
 * -DBRAINAI_PROF.  Usage: bench [model.bin] [steps]
 * Prints a checksum of the final logits so optimisations can be checked
 * for bit-exactness.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "brainai.h"

#ifdef BRAINAI_PROF
extern double brainai_prof[10];
static const char *names[10] = { "norm+quant in", "matvec in rows", "conv+silu",
				 "ssm", "gate+norm+quant+resid", "matvec out rows",
				 "lm head", "(layers->head)", "lut build", "" };
#endif

static double now(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv)
{
	const char *path = argc > 1 ? argv[1] : BRAINAI_DATADIR "/model.bin";
	int steps = argc > 2 ? atoi(argv[2]) : 32, i;
	model_t *m = model_load(path);
	const float *lg = NULL;
	double t0, t1, sum = 0;
	int tok = TOK_BOS, best = 0;
	if (!m)
		return 1;
	t0 = now();
	for (i = 0; i < steps; i++) {
		int v, V = model_vocab(m);
		lg = model_step(m, tok, 1);
		best = 0;
		for (v = 1; v < V; v++)
			if (lg[v] > lg[best])
				best = v;
		tok = best < 9 ? 9 + i : best; /* greedy, never feed a special */
	}
	t1 = now();
	for (i = 0; i < model_vocab(m); i++)
		sum += lg[i] * (double)((i % 97) + 1);
	printf("%d steps: %.1f ms/token (%.2f tok/s), last=%d checksum=%.6f\n", steps,
	       (t1 - t0) * 1e3 / steps, steps / (t1 - t0), best, sum);
	{
		char line[128];
		FILE *f = fopen("/proc/self/status", "r");
		while (f && fgets(line, sizeof(line), f))
			if (!strncmp(line, "VmHWM:", 6))
				printf("peak RSS:%s", line + 6);
		if (f)
			fclose(f);
	}
#ifdef BRAINAI_PROF
	for (i = 0; i < 9; i++)
		printf("  %-24s %7.2f ms/token\n", names[i], brainai_prof[i] * 1e3 / steps);
#endif
	return 0;
}
