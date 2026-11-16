/* Romaji input, the calculator tool and the arithmetic helper ("IGC-lite"). */
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "brainai.h"

static const char *const romaji[][2] = {
	{"kya","きゃ"},{"kyu","きゅ"},{"kyo","きょ"},{"sha","しゃ"},{"shi","し"},{"shu","しゅ"},{"she","しぇ"},{"sho","しょ"},
	{"sya","しゃ"},{"syu","しゅ"},{"syo","しょ"},{"cha","ちゃ"},{"chi","ち"},{"chu","ちゅ"},{"che","ちぇ"},{"cho","ちょ"},
	{"tya","ちゃ"},{"tyu","ちゅ"},{"tyo","ちょ"},{"nya","にゃ"},{"nyu","にゅ"},{"nyo","にょ"},{"hya","ひゃ"},{"hyu","ひゅ"},
	{"hyo","ひょ"},{"mya","みゃ"},{"myu","みゅ"},{"myo","みょ"},{"rya","りゃ"},{"ryu","りゅ"},{"ryo","りょ"},{"gya","ぎゃ"},
	{"gyu","ぎゅ"},{"gyo","ぎょ"},{"jya","じゃ"},{"jyu","じゅ"},{"jyo","じょ"},{"zya","じゃ"},{"zyu","じゅ"},{"zyo","じょ"},
	{"bya","びゃ"},{"byu","びゅ"},{"byo","びょ"},{"pya","ぴゃ"},{"pyu","ぴゅ"},{"pyo","ぴょ"},{"tsu","つ"},{"dya","ぢゃ"},
	{"dyu","ぢゅ"},{"dyo","ぢょ"},{"thi","てぃ"},{"dhi","でぃ"},{"xtu","っ"},{"ltu","っ"},{"xya","ゃ"},{"xyu","ゅ"},{"xyo","ょ"},
	{"lya","ゃ"},{"lyu","ゅ"},{"lyo","ょ"},
	{"ka","か"},{"ki","き"},{"ku","く"},{"ke","け"},{"ko","こ"},{"sa","さ"},{"si","し"},{"su","す"},{"se","せ"},{"so","そ"},
	{"ta","た"},{"ti","ち"},{"tu","つ"},{"te","て"},{"to","と"},{"na","な"},{"ni","に"},{"nu","ぬ"},{"ne","ね"},{"no","の"},
	{"ha","は"},{"hi","ひ"},{"hu","ふ"},{"fu","ふ"},{"he","へ"},{"ho","ほ"},{"ma","ま"},{"mi","み"},{"mu","む"},{"me","め"},
	{"mo","も"},{"ya","や"},{"yu","ゆ"},{"yo","よ"},{"ra","ら"},{"ri","り"},{"ru","る"},{"re","れ"},{"ro","ろ"},{"wa","わ"},
	{"wo","を"},{"ga","が"},{"gi","ぎ"},{"gu","ぐ"},{"ge","げ"},{"go","ご"},{"za","ざ"},{"zi","じ"},{"ji","じ"},{"zu","ず"},
	{"ze","ぜ"},{"zo","ぞ"},{"ja","じゃ"},{"ju","じゅ"},{"je","じぇ"},{"jo","じょ"},{"da","だ"},{"di","ぢ"},{"du","づ"},
	{"de","で"},{"do","ど"},{"ba","ば"},{"bi","び"},{"bu","ぶ"},{"be","べ"},{"bo","ぼ"},{"pa","ぱ"},{"pi","ぴ"},{"pu","ぷ"},
	{"pe","ぺ"},{"po","ぽ"},{"fa","ふぁ"},{"fi","ふぃ"},{"fe","ふぇ"},{"fo","ふぉ"},{"va","ゔぁ"},{"vi","ゔぃ"},{"vu","ゔ"},
	{"ve","ゔぇ"},{"vo","ゔぉ"},{"xa","ぁ"},{"xi","ぃ"},{"xu","ぅ"},{"xe","ぇ"},{"xo","ぉ"},{"la","ぁ"},{"li","ぃ"},{"lu","ぅ"},
	{"le","ぇ"},{"lo","ぉ"},{"ye","いぇ"},{"wi","うぃ"},{"we","うぇ"},
	{"a","あ"},{"i","い"},{"u","う"},{"e","え"},{"o","お"},
	{"-","ー"},{",","、"},{".","。"},{"?","？"},{"!","！"},{"~","〜"},
};

static int is_vowel(int c) { return c && strchr("aiueo", c) != NULL; }

static int put(char *out, size_t *o, size_t max, const char *s, size_t n)
{
	if (*o + n >= max)
		return -1;
	memcpy(out + *o, s, n);
	*o += n;
	return 0;
}

/* Converts ASCII romaji to hiragana; text inside [ ] and non-ASCII is kept. */
int romaji_to_kana(const char *in, char *out, size_t max)
{
	size_t o = 0, k;
	int raw = 0;
	while (*in) {
		int c = tolower((unsigned char)in[0]), d = tolower((unsigned char)in[1]);
		if (*in == '[' && !raw) { raw = 1; in++; continue; }
		if (*in == ']' && raw) { raw = 0; in++; continue; }
		if (raw || (unsigned char)*in >= 0x80 || isdigit(c) || c == ' ') {
			if (put(out, &o, max, in, 1)) return -1;
			in++;
			continue;
		}
		if (c == 'n' && d == 'n') {
			if (put(out, &o, max, "ん", 3)) return -1;
			in += 2;
			continue;
		}
		if (c == 'n' && d != 'y' && !is_vowel(d)) {
			if (put(out, &o, max, "ん", 3)) return -1;
			in++;
			continue;
		}
		if (isalpha(c) && c == d && !is_vowel(c)) {
			if (put(out, &o, max, "っ", 3)) return -1;
			in++;
			continue;
		}
		for (k = 0; k < sizeof(romaji) / sizeof(romaji[0]); k++) {
			size_t n = strlen(romaji[k][0]);
			if (!strncasecmp(in, romaji[k][0], n))
				break;
		}
		if (k < sizeof(romaji) / sizeof(romaji[0])) {
			if (put(out, &o, max, romaji[k][1], strlen(romaji[k][1]))) return -1;
			in += strlen(romaji[k][0]);
		} else {
			if (put(out, &o, max, in, 1)) return -1;
			in++;
		}
	}
	out[o] = 0;
	return 0;
}

/* --- calculator ------------------------------------------------------- */

/* Full-width ASCII, ×, ÷, −, ・, °/度 folded into a plain ASCII expression. */
static void normalize(const char *in, char *out, size_t max)
{
	const unsigned char *s = (const unsigned char *)in;
	size_t o = 0;
	while (*s && o + 4 < max) {
		if (s[0] == 0xef && (s[1] == 0xbc || s[1] == 0xbd) && s[2] >= 0x80) {
			unsigned cp = ((s[0] & 0x0f) << 12) | ((s[1] & 0x3f) << 6) | (s[2] & 0x3f);
			if (cp >= 0xff01 && cp <= 0xff5e) {
				out[o++] = (char)(cp - 0xff01 + 0x21);
				s += 3;
				continue;
			}
		}
		if (!memcmp(s, "×", 2)) { out[o++] = '*'; s += 2; continue; }
		if (!memcmp(s, "÷", 2)) { out[o++] = '/'; s += 2; continue; }
		if (!memcmp(s, "°", 2)) { s += 2; continue; }
		if (!memcmp(s, "−", 3)) { out[o++] = '-'; s += 3; continue; }
		if (!memcmp(s, "度", 3)) { s += 3; continue; }
		if (!memcmp(s, "√", 3)) { memcpy(out + o, "sqrt", 4); o += 4; s += 3; continue; }
		if (!memcmp(s, "π", 2)) { memcpy(out + o, "pi", 2); o += 2; s += 2; continue; }
		out[o++] = (char)*s++;
	}
	out[o] = 0;
}

typedef struct { const char *p; int deg, err; } calc_t;

static double c_expr(calc_t *c);

static void skip(calc_t *c) { while (*c->p == ' ') c->p++; }

static double c_primary(calc_t *c)
{
	static const char *const fn[] = {"sqrt","sin","cos","tan","asin","acos","atan","log","ln","exp","abs"};
	double v;
	size_t i;
	skip(c);
	if (*c->p == '(') {
		c->p++;
		v = c_expr(c);
		skip(c);
		if (*c->p == ')') c->p++; else c->err = 1;
		return v;
	}
	if (isdigit((unsigned char)*c->p) || *c->p == '.') {
		char *e;
		v = strtod(c->p, &e);
		c->p = e;
		while (*c->p == ',' && isdigit((unsigned char)c->p[1]) && isdigit((unsigned char)c->p[2]) &&
		       isdigit((unsigned char)c->p[3])) { /* 1,234 */
			v = v * 1000 + strtod(c->p + 1, &e);
			c->p = e;
		}
		return v;
	}
	if (!strncmp(c->p, "pi", 2)) { c->p += 2; return M_PI; }
	for (i = 0; i < sizeof(fn) / sizeof(fn[0]); i++) {
		size_t n = strlen(fn[i]);
		if (!strncmp(c->p, fn[i], n) && !isalpha((unsigned char)c->p[n])) {
			double a, r = c->deg ? M_PI / 180.0 : 1.0;
			c->p += n;
			skip(c);
			if (*c->p == '(') a = c_primary(c);
			else a = c_primary(c);
			switch (i) {
			case 0: return sqrt(a);
			case 1: return sin(a * r);
			case 2: return cos(a * r);
			case 3: return tan(a * r);
			case 4: return asin(a) / r;
			case 5: return acos(a) / r;
			case 6: return atan(a) / r;
			case 7: return log10(a);
			case 8: return log(a);
			case 9: return exp(a);
			default: return fabs(a);
			}
		}
	}
	if (*c->p == 'e' && !isalpha((unsigned char)c->p[1])) { c->p++; return M_E; }
	c->err = 1;
	return 0;
}

static double c_unary(calc_t *c)
{
	double v;
	skip(c);
	if (*c->p == '-') { c->p++; return -c_unary(c); }
	if (*c->p == '+') { c->p++; return c_unary(c); }
	v = c_primary(c);
	skip(c);
	while (*c->p == '%' || *c->p == '!') {
		if (*c->p == '%') v /= 100.0;
		else { double f = 1; int k; for (k = 2; k <= (int)v && k < 171; k++) f *= k; v = f; }
		c->p++;
		skip(c);
	}
	return v;
}

static double c_power(calc_t *c)
{
	double b = c_unary(c);
	skip(c);
	if (*c->p == '^' || (c->p[0] == '*' && c->p[1] == '*')) {
		c->p += (*c->p == '^') ? 1 : 2;
		return pow(b, c_power(c));
	}
	return b;
}

static double c_term(calc_t *c)
{
	double v = c_power(c);
	for (;;) {
		skip(c);
		if (*c->p == '*' && c->p[1] != '*') { c->p++; v *= c_power(c); }
		else if (*c->p == '/') { c->p++; v /= c_power(c); }
		else if (*c->p == '(' || isalpha((unsigned char)*c->p)) v *= c_power(c); /* 2pi, 3(4) */
		else return v;
		if (c->err) return v;
	}
}

static double c_expr(calc_t *c)
{
	double v = c_term(c);
	for (;;) {
		skip(c);
		if (*c->p == '+') { c->p++; v += c_term(c); }
		else if (*c->p == '-') { c->p++; v -= c_term(c); }
		else return v;
		if (c->err) return v;
	}
}

int calc_eval(const char *expr, double *result, int degrees)
{
	char buf[256], *e;
	calc_t c;
	normalize(expr, buf, sizeof(buf));
	for (e = buf; *e; e++)
		*e = (char)tolower((unsigned char)*e);
	/* drop trailing "=", "?" and similar */
	while (e > buf && strchr(" =?。", e[-1]))
		*--e = 0;
	c.p = buf; c.deg = degrees; c.err = 0;
	*result = c_expr(&c);
	skip(&c);
	if (c.err || *c.p || isnan(*result) || isinf(*result))
		return -1;
	return 0;
}

void format_number(double v, char *buf, size_t max)
{
	if (fabs(v) < 1e15 && fabs(v - floor(v + 0.5)) < 1e-9 * (fabs(v) > 1 ? fabs(v) : 1)) {
		snprintf(buf, max, "%.0f", floor(v + 0.5));
	} else {
		char *e;
		snprintf(buf, max, "%.10g", v);
		if (strchr(buf, '.') && !strchr(buf, 'e')) {
			for (e = buf + strlen(buf) - 1; *e == '0'; e--)
				*e = 0;
			if (*e == '.')
				*e = 0;
		}
	}
	if (!strcmp(buf, "-0"))
		strcpy(buf, "0");
}

/*
 * The distributed app contains "IGC", a deterministic arithmetic unit that is
 * not part of the published weights. As a stand-in, find the longest run in
 * the message that looks like arithmetic (digits with at least one binary
 * operator between them) and compute it exactly; the chat loop then forces
 * that number into the reply where the model starts writing digits.
 */
int igc_find(const char *text, char *expr_out, size_t exmax, char *result, size_t max)
{
	char buf[512];
	const char *s;
	int best = 0;
	normalize(text, buf, sizeof(buf));
	for (s = buf; *s; s++) {
		const char *e;
		int ops = 0, digits = 0, len;
		char expr[256];
		double v;
		if (!isdigit((unsigned char)*s) && *s != '(' && *s != '-')
			continue;
		for (e = s; *e && strchr("0123456789.()+-*/^% ", *e); e++) {
			if (strchr("+-*/^", *e) && e > s && (isdigit((unsigned char)e[-1]) || e[-1] == ')' || e[-1] == ' ' || e[-1] == '%'))
				ops++;
			if (isdigit((unsigned char)*e))
				digits++;
		}
		while (e > s && strchr(" +-*/^(", e[-1]))
			e--;
		len = (int)(e - s);
		if (ops == 0 || digits < 2 || len <= best || len >= (int)sizeof(expr))
			continue;
		memcpy(expr, s, len);
		expr[len] = 0;
		if (calc_eval(expr, &v, 1) == 0) {
			best = len;
			format_number(v, result, max);
			snprintf(expr_out, exmax, "%s", expr);
		}
	}
	return best > 0;
}
