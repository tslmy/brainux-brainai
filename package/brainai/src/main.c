/*
 * brainai: offline Japanese chat with brain-ai v1 preview on Brainux.
 *
 * This file is the app around the engine:
 *   - parses the command line and loads the tokenizer and model;
 *   - on the Brain's own screen, restarts itself inside fbterm, a
 *     framebuffer terminal that can draw kanji (run_in_fbterm());
 *   - talks to you through the terminal UI (ui_tty.c);
 *   - runs the chat loop: encode your message, feed it to the model, then
 *     sample the reply token by token (sample());
 *   - runs the tools the model calls (calc, time, memo, search);
 *   - stands in for the original app's exact-arithmetic unit, "IGC".
 *
 * The conversation format the model was trained on, as token ids:
 *   <user> message <end> <bot> reply <end>
 *   reply may contain  <call> tool: argument <end>  after which the app
 *   feeds  <result> tool output <end> <bot>  and the model continues.
 */
#include <linux/kd.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "brainai.h"
#include "ui.h"

#ifndef BRAINAI_DATADIR
#define BRAINAI_DATADIR "/usr/share/brainai"
#endif

#define MAX_PROMPT 1024 /* longest message, in bytes and in tokens */
#define MAX_CALLS 3     /* tool calls allowed per reply */

static model_t *M;
static tokenizer_t *T;
static const ui_t *U;
/* sampling settings, changeable with -t, -r and /temp */
static float temperature = 0.7f, top_p = 0.9f, rep_penalty = 1.1f;
static int top_k = 40, max_new = 300, show_stats = 0;
static uint32_t rng = 2463534242u; /* random state; reseeded from the clock */
static unsigned char *digit_tok; /* 1: starts with a digit, 2: with '.' or ',' */
static int recent[64], nrecent;  /* the reply's last 64 tokens */
static unsigned char *is_recent; /* V entries, set only inside sample() */

/* Wall-clock time in seconds, for the --stats line. */
static double now(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* Uniform random float in [0, 1), from a xorshift32 generator. */
static float frand(void)
{
	rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
	return (rng >> 8) * (1.0f / 16777216.0f);
}

/*
 * Picks the next token from the logits.
 *
 *  1. Skip tokens that must never be generated here (special tokens, and
 *     digits right after a forced IGC number when mask_digits is set).
 *  2. Repetition penalty: tokens used in the last 64 become less likely
 *     (positive logits divided by rep_penalty, negative ones multiplied).
 *     is_recent marks them so each check is one array read.
 *  3. Keep the top_k (at most 64) best tokens in a sorted list.
 *  4. With temperature 0, return the best one (greedy).  Otherwise turn the
 *     kept logits into probabilities, p_i ~ exp((logit_i - max) / temperature),
 *     cut the tail once the running total reaches top_p of the whole
 *     ("nucleus" sampling), and draw at random.
 */
static int sample(const float *lg, int mask_digits)
{
	int V = model_vocab(M), i, j, n = 0;
	int idx[64];
	float val[64], sum = 0, cum = 0, r, mx;
	int k = top_k < 64 ? top_k : 64;
	if (temperature <= 0)
		k = 1;
	if (!is_recent)
		is_recent = calloc(V, 1);
	for (j = 0; j < nrecent; j++)
		is_recent[recent[j]] = 1;
	for (i = 0; i < V; i++) {
		float v = lg[i];
		if (i == TOK_UNK || i == TOK_BOS || i == TOK_SYS || i == TOK_USER || i == TOK_BOT ||
		    i == TOK_RESULT || (mask_digits && digit_tok[i]))
			continue;
		if (is_recent[i])
			v = v > 0 ? v / rep_penalty : v * rep_penalty;
		/* insertion into the sorted top-k list (val descending) */
		if (n == k && v <= val[n - 1])
			continue;
		if (n < k)
			n++;
		for (j = n - 1; j > 0 && val[j - 1] < v; j--) {
			val[j] = val[j - 1];
			idx[j] = idx[j - 1];
		}
		val[j] = v;
		idx[j] = i;
	}
	for (j = 0; j < nrecent; j++) /* clear the marks for next time */
		is_recent[recent[j]] = 0;
	if (n == 1 || temperature <= 0)
		return idx[0];
	mx = val[0];
	for (i = 0; i < n; i++) {
		val[i] = expf((val[i] - mx) / temperature);
		sum += val[i];
	}
	for (i = 0; i < n; i++) { /* top-p cut */
		cum += val[i];
		if (cum >= top_p * sum) {
			n = i + 1;
			sum = cum;
			break;
		}
	}
	r = frand() * sum;
	for (i = 0; i < n; i++) {
		r -= val[i];
		if (r <= 0)
			return idx[i];
	}
	return idx[n - 1];
}

/*
 * Feeds tokens; logits of the last one are returned.  Only the last token
 * asks for logits, which skips the expensive LM (language model) head for
 * the rest of the prompt.
 */
static const float *feed(const int *ids, int n)
{
	const float *lg = NULL;
	int i;
	for (i = 0; i < n; i++)
		lg = model_step(M, ids[i], i == n - 1);
	return lg;
}

/* Feeds [pre] + encoded text + [post1] + [post2]; a negative id is skipped. */
static const float *feed_text(int pre, const char *text, int post1, int post2)
{
	int ids[MAX_PROMPT + 4], n = 0;
	if (pre >= 0)
		ids[n++] = pre;
	n += tok_encode(T, text, ids + n, MAX_PROMPT);
	if (post1 >= 0)
		ids[n++] = post1;
	if (post2 >= 0)
		ids[n++] = post2;
	return feed(ids, n);
}

static void memo_path(char *buf, size_t max)
{
	const char *home = getenv("HOME");
	snprintf(buf, max, "%s/.brainai-memo.txt", home ? home : "/root");
}

/*
 * Runs one tool call written by the model, "name: argument", and writes the
 * result text (in Japanese, as the model expects) into res.
 *   calc    evaluate an arithmetic expression (angles in degrees)
 *   time    current date and time
 *   memo    with an argument: append it to ~/.brainai-memo.txt;
 *           without: read all memos back
 *   search  look the argument up in an optional search.tsv
 */
static void run_tool(const char *call, char *res, size_t max)
{
	const char *colon = strchr(call, ':');
	char name[32] = "", arg[512] = "";
	size_t n = colon ? (size_t)(colon - call) : strlen(call);
	if (n >= sizeof(name))
		n = sizeof(name) - 1;
	memcpy(name, call, n);
	name[n] = 0;
	if (colon) {
		colon++;
		while (*colon == ' ')
			colon++;
		snprintf(arg, sizeof(arg), "%s", colon);
	}
	if (!strcmp(name, "calc")) {
		double v;
		if (calc_eval(arg, &v, 1) == 0)
			format_number(v, res, max);
		else
			snprintf(res, max, "エラー: 計算できません");
	} else if (!strcmp(name, "time")) {
		static const char *const wd[] = {"日", "月", "火", "水", "木", "金", "土"};
		time_t t = time(NULL);
		struct tm tm;
		localtime_r(&t, &tm);
		snprintf(res, max, "%d年%d月%d日(%s) %d時%02d分", tm.tm_year + 1900, tm.tm_mon + 1,
			 tm.tm_mday, wd[tm.tm_wday], tm.tm_hour, tm.tm_min);
	} else if (!strcmp(name, "memo")) {
		char path[256];
		FILE *f;
		memo_path(path, sizeof(path));
		if (*arg) {
			f = fopen(path, "a");
			if (f) {
				fprintf(f, "%s\n", arg);
				fclose(f);
				snprintf(res, max, "メモしました: %s", arg);
			} else {
				snprintf(res, max, "エラー: メモを保存できません");
			}
		} else {
			/* join all memo lines with "、" */
			size_t o = 0;
			char line[256];
			f = fopen(path, "r");
			res[0] = 0;
			while (f && fgets(line, sizeof(line), f)) {
				line[strcspn(line, "\n")] = 0;
				o += snprintf(res + o, o < max ? max - o : 0, "%s%s", o ? "、" : "", line);
				if (o >= max)
					break;
			}
			if (f)
				fclose(f);
			if (!res[0])
				snprintf(res, max, "メモはありません");
		}
	} else if (!strcmp(name, "search")) {
		/* Optional offline encyclopedia: "keyword<TAB>text" lines. */
		FILE *f = fopen(BRAINAI_DATADIR "/search.tsv", "r");
		char line[1024];
		size_t al = strlen(arg);
		snprintf(res, max, "「%s」は見つかりませんでした", arg);
		while (f && al && fgets(line, sizeof(line), f)) {
			if (!strncmp(line, arg, al) && line[al] == '\t') {
				line[strcspn(line, "\n")] = 0;
				snprintf(res, max, "%s", line + al + 1);
				break;
			}
		}
		if (f)
			fclose(f);
	} else {
		snprintf(res, max, "エラー: 不明なツール %s", name);
	}
}

/*
 * Streams token text to the UI, holding back incomplete UTF-8.
 * A Japanese character is 3 bytes in UTF-8 (Unicode Transformation Format,
 * 8-bit), and byte-fallback tokens deliver it one byte at a time, so
 * printing must wait until a character is complete.
 */
typedef struct {
	char buf[64];  /* bytes not yet shown */
	int n, color;
	char tail[16]; /* last bytes shown, for the IGC trigger */
} out_t;

static void out_bytes(out_t *o, const char *s, int n)
{
	int i, cut;
	for (i = 0; i < n && o->n < (int)sizeof(o->buf) - 1; i++)
		o->buf[o->n++] = s[i];
	/*
	 * Find the last complete UTF-8 boundary: walk back over continuation
	 * bytes (10xxxxxx) to the lead byte, whose top bits say how long its
	 * character is (110x: 2, 1110: 3, 11110: 4).  If not all of it has
	 * arrived yet, show only what comes before it.
	 */
	cut = o->n;
	for (i = o->n - 1; i >= 0 && i >= o->n - 4; i--) {
		unsigned char c = o->buf[i];
		if ((c & 0xc0) == 0x80)
			continue;
		if (c >= 0xc0) {
			int need = c >= 0xf0 ? 4 : c >= 0xe0 ? 3 : 2;
			if (o->n - i < need)
				cut = i;
		}
		break;
	}
	if (cut > 0) {
		char tmp[96];
		int have = (int)strlen(o->tail), keep = (int)sizeof(o->tail) - 1, tot;
		memcpy(tmp, o->buf, cut);
		tmp[cut] = 0;
		U->write(tmp, o->color);
		/* remember the last bytes shown */
		memmove(tmp + have, tmp, cut + 1);
		memcpy(tmp, o->tail, have);
		tot = have + cut;
		strcpy(o->tail, tmp + (tot > keep ? tot - keep : 0));
		memmove(o->buf, o->buf + cut, o->n - cut);
		o->n -= cut;
	}
}

/*
 * True when the reply so far ends in a place where a number is expected,
 * e.g. "答えは" ("the answer is"), "=" or the very start of the reply.
 */
static int igc_trigger(const char *tail)
{
	static const char *const keys[] = {"=", "＝", "は", "答え", "結果", "約", ":", "：", "、"};
	size_t i, n = strlen(tail);
	while (n && tail[n - 1] == ' ')
		n--;
	if (n == 0)
		return 1;
	for (i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
		size_t k = strlen(keys[i]);
		if (n >= k && !memcmp(tail + n - k, keys[i], k))
			return 1;
	}
	return 0;
}

/*
 * Answers one user message: feeds it, then generates until the model ends
 * its turn, max_new tokens are produced, or the user cancels.
 */
static void reply(const char *user)
{
	char igc[64], igc_expr[256], call[512], res[1024], piece[64];
	/* if the message contains arithmetic, compute it exactly now */
	int igc_pending = igc_find(user, igc_expr, sizeof(igc_expr), igc, sizeof(igc));
	int in_call = 0, ncall = 0, mask_digits = 0, produced = 0, i, cl = 0;
	const float *lg;
	double t0, t1;
	out_t o;

	memset(&o, 0, sizeof(o));
	t0 = now();
	/* "prefill": <user> message <end> <bot> */
	lg = feed_text(TOK_USER, user, TOK_END, TOK_BOT);
	t1 = now();
	nrecent = 0;
	while (produced < max_new) {
		int t;
		if (U->cancelled()) {
			model_step(M, TOK_END, 0); /* close the turn so the state stays valid */
			U->write(" …", 2);
			break;
		}
		t = sample(lg, mask_digits);
		mask_digits = 0;
		if (t == TOK_EOS) {
			model_step(M, TOK_END, 0);
			break;
		}
		if (t == TOK_END && !in_call) { /* the model finished its reply */
			model_step(M, TOK_END, 0);
			break;
		}
		if (t == TOK_CALL && !in_call) { /* a tool call starts */
			in_call = 1;
			cl = 0;
			lg = model_step(M, t, 1);
			continue;
		}
		if (t == TOK_END && in_call) { /* the tool call is complete: run it */
			call[cl] = 0;
			model_step(M, TOK_END, 0);
			/* an explicit expression in the message beats the model's guess */
			if (igc_pending && !strncmp(call, "calc:", 5))
				snprintf(call, sizeof(call), "calc: %s", igc_expr);
			run_tool(call, res, sizeof(res));
			if (o.n) /* flush */
				out_bytes(&o, "", 0);
			U->write("\n", 0);
			{
				char line[1600];
				snprintf(line, sizeof(line), "[%s → %s]\n", call, res);
				U->write(line, 2);
			}
			o.tail[0] = 0;
			in_call = 0;
			igc_pending = 0; /* the tool result is authoritative */
			if (++ncall > MAX_CALLS)
				break;
			/* give the model the result: <result> output <end> <bot> */
			lg = feed_text(TOK_RESULT, res, TOK_END, TOK_BOT);
			continue;
		}
		if (in_call) { /* inside a tool call: collect its text, don't show it */
			int n = tok_piece(T, t, piece, sizeof(piece));
			if (cl + n < (int)sizeof(call)) {
				memcpy(call + cl, piece, n);
				cl += n;
			}
			lg = model_step(M, t, 1);
			produced++;
			continue;
		}
		if (igc_pending && digit_tok[t] == 1 && igc_trigger(o.tail)) {
			/* force the exactly computed number, then forbid more digits */
			int ids[32], n = tok_encode(T, igc, ids, 32);
			out_bytes(&o, igc, (int)strlen(igc));
			lg = feed(ids, n);
			produced += n;
			igc_pending = 0;
			mask_digits = 1;
			continue;
		}
		/* an ordinary token: show it and feed it back */
		{
			int n = tok_piece(T, t, piece, sizeof(piece));
			out_bytes(&o, piece, n);
		}
		if (nrecent < 64)
			recent[nrecent++] = t;
		else {
			memmove(recent, recent + 1, sizeof(int) * 63);
			recent[63] = t;
		}
		lg = model_step(M, t, 1);
		produced++;
	}
	U->write("\n", 0);
	if (show_stats) {
		char s[128];
		double t2 = now();
		snprintf(s, sizeof(s), "[prefill %.2fs, %d tokens in %.2fs = %.2f tok/s]\n", t1 - t0,
			 produced, t2 - t1, produced / (t2 - t1 + 1e-9));
		U->write(s, 2);
	}
	(void)i;
}

static void help(void)
{
	U->write("/reset 会話をリセット  /kana ローマ字変換の切替  /temp 温度  /stats 速度表示  /quit 終了\n"
		 "[ ]で囲んだ部分はかなに変換されません (例: [123*45]wa?)\n", 2);
}

/*
 * The Linux text console can only draw a few hundred glyphs, so no kanji.
 * When brainai is started on it (stdin is a VT, a virtual terminal, which
 * answers the KDGETMODE ioctl), it runs itself again inside fbterm, which
 * draws text on the framebuffer with a CJK (Chinese, Japanese, Korean)
 * font, and turns romaji input on (-k), since the Brain's keyboard has no
 * Japanese input method.  Inside fbterm, stdin is a pseudo-terminal, which
 * does not answer KDGETMODE, so this happens only once.  Returns only if
 * fbterm cannot be started; brainai then carries on without kanji.
 */
static void run_in_fbterm(int argc, char **argv)
{
	char *av[64], self[256];
	int mode, i, n = 0;
	ssize_t len;

	if (ioctl(0, KDGETMODE, &mode) < 0 || argc > 60)
		return;
	len = readlink("/proc/self/exe", self, sizeof(self) - 1);
	if (len <= 0)
		return;
	self[len] = 0;
	av[n++] = "fbterm";
	av[n++] = "--font-names=WenQuanYi Zen Hei Mono";
	av[n++] = "--font-size=16";
	av[n++] = "--";
	av[n++] = self;
	av[n++] = "-k";
	for (i = 1; i < argc; i++)
		av[n++] = argv[i];
	av[n] = NULL;
	execvp(av[0], av);
	fprintf(stderr, "brainai: fbterm not found, Japanese text will not show here\n");
	sleep(2);
}

int main(int argc, char **argv)
{
	const char *dir = BRAINAI_DATADIR, *prompt = NULL;
	char path[512], line[MAX_PROMPT], conv[MAX_PROMPT * 3];
	int i, kana = -1, force_tty = 0, V;

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-d") && i + 1 < argc) dir = argv[++i];
		else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
		else if (!strcmp(argv[i], "-t") && i + 1 < argc) temperature = atof(argv[++i]);
		else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_new = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-s") && i + 1 < argc) rng = (uint32_t)strtoul(argv[++i], NULL, 0);
		else if (!strcmp(argv[i], "-r") && i + 1 < argc) rep_penalty = atof(argv[++i]);
		else if (!strcmp(argv[i], "-k")) kana = 1;
		else if (!strcmp(argv[i], "--tty")) force_tty = 1;
		else if (!strcmp(argv[i], "--stats")) show_stats = 1;
		else {
			fprintf(stderr,
				"usage: brainai [-d datadir] [-p prompt] [-t temp] [-n max_tokens] [-s seed]\n"
				"               [-r repetition_penalty] [-k] [--tty] [--stats]\n"
				"  -p    answer one prompt and exit\n"
				"  -k    convert romaji input to hiragana\n"
				"  --tty do not start fbterm on the Brain's console\n");
			return 2;
		}
	}
	if (!prompt && !force_tty)
		run_in_fbterm(argc, argv);

	/* no -s given: seed from the time and process id */
	if (rng == 2463534242u)
		rng ^= (uint32_t)time(NULL) ^ ((uint32_t)getpid() << 16);
	if (!rng)
		rng = 1; /* xorshift never leaves 0 */

	snprintf(path, sizeof(path), "%s/tok.bin", dir);
	T = tok_load(path);
	if (!T) {
		fprintf(stderr, "brainai: cannot load %s\n", path);
		return 1;
	}
	snprintf(path, sizeof(path), "%s/model.bin", dir);
	M = model_load(path);
	if (!M)
		return 1;
	/* mark tokens that start with a digit (1) or '.' / ',' (2), for IGC */
	V = model_vocab(M);
	digit_tok = calloc(V, 1);
	for (i = 0; i < V; i++) {
		char b[64];
		int n = tok_piece(T, i, b, sizeof(b));
		digit_tok[i] = n > 0 && b[0] >= '0' && b[0] <= '9' ? 1 :
			       n > 0 && (b[0] == '.' || b[0] == ',') ? 2 : 0;
	}

	/* line editing with live kana, except for one-shot -p answers */
	U = ui_tty_open(!prompt);
	if (kana < 0)
		kana = U->wants_kana;

	if (prompt) {
		reply(prompt);
		U->close();
		return 0;
	}
	U->write("brain-ai v1 preview (Brainux)  /help でヘルプ\n", 2);
	for (;;) {
		const char *msg = line;
		if (U->readline(line, sizeof(line), &kana) < 0)
			break;
		if (!line[0])
			continue;
		/* slash commands */
		if (!strcmp(line, "/quit") || !strcmp(line, "/exit"))
			break;
		if (!strcmp(line, "/help")) { help(); continue; }
		if (!strcmp(line, "/reset")) {
			model_reset(M);
			U->write("[リセットしました]\n", 2);
			continue;
		}
		if (!strcmp(line, "/kana")) {
			kana = !kana;
			U->write(kana ? "[ローマ字→かな: ON]\n" : "[ローマ字→かな: OFF]\n", 2);
			continue;
		}
		if (!strcmp(line, "/stats")) { show_stats = !show_stats; continue; }
		if (!strncmp(line, "/temp", 5)) {
			char s[64];
			if (line[5])
				temperature = atof(line + 5);
			snprintf(s, sizeof(s), "[temperature %.2f]\n", temperature);
			U->write(s, 2);
			continue;
		}
		/* romaji -> hiragana, showing the result in terminal mode */
		if (kana && romaji_to_kana(line, conv, sizeof(conv)) == 0) {
			msg = conv;
			if (strcmp(conv, line) && U->echo_conversion) {
				U->write("→ ", 2);
				U->write(conv, 1);
				U->write("\n", 0);
			}
		}
		reply(msg);
	}
	U->close();
	return 0;
}
