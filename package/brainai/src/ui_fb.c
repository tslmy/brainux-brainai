/*
 * Framebuffer front end for the Brain's own screen and keyboard.
 *
 * The Linux text console cannot draw kanji, so brainai draws text itself
 * with a 16-px Unifont subset while the VT (virtual terminal) is in
 * KD_GRAPHICS mode. Keyboard input still arrives on the VT (raw termios,
 * i.e. one byte per key press, no line editing or echo by the kernel).
 * Romaji typed at the prompt is shown already converted to hiragana;
 * Ctrl-K toggles the conversion.
 *
 * The framebuffer (/dev/fb0) is the screen's memory: mapped into this
 * process with mmap(), writing a pixel value at (x, y) makes it appear.
 * The screen is treated as a grid of 8 x 16-pixel cells; a kanji or kana
 * glyph is two cells wide, an ASCII (American Standard Code for Information
 * Interchange) letter one.
 *
 * font.bin (from tools/mkfont.py): "BRFN", u32 n, then n x (u32 code point,
 * u32 width in pixels) sorted by code point, then n 32-byte bitmaps
 * (16 rows x 16 bits, most significant bit = leftmost pixel).
 */
#include <fcntl.h>
#include <linux/fb.h>
#include <linux/kd.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

#include "brainai.h"
#include "ui.h"

#ifndef BRAINAI_DATADIR
#define BRAINAI_DATADIR "/usr/share/brainai"
#endif

#define CW 8  /* cell width in pixels */
#define CH 16 /* cell height in pixels */

/* All framebuffer UI state. */
static struct {
	uint8_t *fb;                   /* mapped screen memory, visible page */
	size_t fblen;
	int xres, yres, bpp, stride;   /* pixels, bits per pixel, bytes per row */
	struct fb_var_screeninfo var;  /* pixel format (colour bit positions) */
	int cols, rows, cx, cy, scrolled; /* grid size, cursor cell, scroll count */
	uint32_t color[3], bg;         /* text colours (see ui.h) and background */
	/* font */
	uint32_t nglyph, *cp, *gw;     /* glyph count, code points, widths */
	const uint8_t *bits;           /* 32 bytes of bitmap per glyph */
	struct termios saved;          /* terminal settings to restore on exit */
	int vt_ok;                     /* the VT is in graphics mode */
} F;

/*
 * Converts 0xRRGGBB into the screen's pixel format, e.g. RGB565 (5 bits red,
 * 6 green, 5 blue) on a 16-bit screen: each channel keeps its top bits and
 * moves to the position the driver reports.
 */
static uint32_t pack(uint32_t rgb)
{
	uint32_t r = (rgb >> 16) & 255, g = (rgb >> 8) & 255, b = rgb & 255;
	return ((r >> (8 - F.var.red.length)) << F.var.red.offset) |
	       ((g >> (8 - F.var.green.length)) << F.var.green.offset) |
	       ((b >> (8 - F.var.blue.length)) << F.var.blue.offset);
}

static inline void pset(uint8_t *row, int x, uint32_t c)
{
	if (F.bpp == 16)
		((uint16_t *)row)[x] = (uint16_t)c;
	else
		((uint32_t *)row)[x] = c;
}

static void fill(int x, int y, int w, int h, uint32_t c)
{
	int i, j;
	for (j = y; j < y + h && j < F.yres; j++) {
		uint8_t *row = F.fb + (size_t)j * F.stride;
		for (i = x; i < x + w && i < F.xres; i++)
			pset(row, i, c);
	}
}

/* Index of the glyph for code point c (binary search), or -1. */
static int glyph(uint32_t c)
{
	int lo = 0, hi = (int)F.nglyph - 1;
	while (lo <= hi) {
		int mid = (lo + hi) / 2;
		if (F.cp[mid] == c)
			return mid;
		if (F.cp[mid] < c)
			lo = mid + 1;
		else
			hi = mid - 1;
	}
	return -1;
}

/* Moves every text row up by one and clears the bottom row. */
static void scroll(void)
{
	size_t line = (size_t)F.stride * CH;
	memmove(F.fb, F.fb + line, (size_t)F.stride * (F.rows - 1) * CH);
	fill(0, (F.rows - 1) * CH, F.xres, CH, F.bg);
	F.scrolled++;
}

static void newline(void)
{
	F.cx = 0;
	if (++F.cy >= F.rows) {
		scroll();
		F.cy = F.rows - 1;
	}
}

/*
 * Draws one character at the cursor and advances it, wrapping and
 * scrolling as needed.  Missing CJK (Chinese, Japanese, Korean) glyphs show
 * as □, other missing ones as '?'.
 */
static void putcp(uint32_t c, uint32_t fg)
{
	int g, w, x, y, x0, y0;
	if (c == '\n') {
		newline();
		return;
	}
	if (c == '\t') {
		do
			putcp(' ', fg);
		while (F.cx % 4);
		return;
	}
	if (c < 0x20)
		return;
	g = glyph(c);
	if (g < 0)
		g = glyph(c >= 0x2e80 ? 0x25a1 : '?');
	w = g >= 0 ? (int)F.gw[g] / CW : 1;
	if (F.cx + w > F.cols)
		newline();
	x0 = F.cx * CW;
	y0 = F.cy * CH;
	for (y = 0; y < CH; y++) {
		uint8_t *row = F.fb + (size_t)(y0 + y) * F.stride;
		unsigned bits = g >= 0 ? (F.bits[g * 32 + 2 * y] << 8) | F.bits[g * 32 + 2 * y + 1] : 0;
		for (x = 0; x < w * CW; x++)
			pset(row, x0 + x, (bits & (0x8000u >> x)) ? fg : F.bg);
	}
	F.cx += w;
}

/*
 * Decodes one UTF-8 (Unicode Transformation Format, 8-bit) character at s
 * into *c and returns a pointer past it.  The lead byte's top bits give the
 * length (0xxxxxxx: 1 byte, 110xxxxx: 2, 1110xxxx: 3, 11110xxx: 4); each
 * following byte contributes 6 bits.
 */
static const char *utf8(const char *s, uint32_t *c)
{
	unsigned char b = (unsigned char)*s;
	int n = b < 0x80 ? 0 : b >= 0xf0 ? 3 : b >= 0xe0 ? 2 : b >= 0xc0 ? 1 : 0;
	*c = n ? (b & (0x3f >> n)) : b;
	s++;
	while (n-- && ((unsigned char)*s & 0xc0) == 0x80)
		*c = (*c << 6) | (*s++ & 0x3f);
	return s;
}

static void fb_write(const char *s, int color)
{
	uint32_t c;
	while (*s) {
		s = utf8(s, &c);
		putcp(c, F.color[color]);
	}
}

static void cursor(int on)
{
	if (F.cx < F.cols)
		fill(F.cx * CW, F.cy * CH + CH - 2, CW, 2, on ? F.color[1] : F.bg);
}

/*
 * Reads one key byte.  timeout_ms < 0 waits forever; otherwise returns -1
 * if nothing arrives in time.  Returns -2 at end of input.
 */
static int readbyte(int timeout_ms)
{
	unsigned char b;
	if (timeout_ms >= 0) {
		fd_set s;
		struct timeval tv = {timeout_ms / 1000, (timeout_ms % 1000) * 1000};
		FD_ZERO(&s);
		FD_SET(0, &s);
		if (select(1, &s, NULL, NULL, &tv) <= 0)
			return -1;
	}
	if (read(0, &b, 1) != 1)
		return -2;
	return b;
}

/*
 * Redraws the input line from scratch after every key press: the prompt,
 * then the typed text (converted to kana when enabled).  (sx, sy) is where
 * the line starts; it moves up if drawing scrolled the screen.  last_rows
 * remembers how many rows the previous drawing used, so they are cleared.
 */
static void render_input(int *sx, int *sy, int *last_rows, const char *raw, int kana)
{
	char disp[4096];
	int r, before;
	/* clear what the previous rendering used */
	for (r = *sy; r < *sy + *last_rows && r < F.rows; r++)
		fill(0, r * CH, F.xres, CH, F.bg);
	F.cx = *sx;
	F.cy = *sy;
	before = F.scrolled;
	fb_write(kana ? "あ> " : "A> ", 2);
	if (!kana || romaji_to_kana(raw, disp, sizeof(disp)))
		snprintf(disp, sizeof(disp), "%s", raw);
	fb_write(disp, 1);
	if (F.cx >= F.cols)
		newline();
	*sy -= F.scrolled - before;
	if (*sy < 0)
		*sy = 0;
	*last_rows = F.cy - *sy + 1;
	cursor(1);
}

/*
 * Line editor for the Brain's keyboard: Enter submits, Backspace deletes a
 * whole UTF-8 character, Ctrl-C clears the line, Ctrl-D on an empty line
 * quits, Ctrl-K toggles kana, Ctrl-L clears the screen.
 */
static int fb_readline(char *buf, size_t max, int *kana)
{
	size_t n = 0;
	int sx, sy, last = 1;
	if (F.cx)
		newline();
	sx = F.cx;
	sy = F.cy;
	buf[0] = 0;
	render_input(&sx, &sy, &last, buf, *kana);
	for (;;) {
		int c = readbyte(-1);
		if (c == -2)
			return -1;
		if (c == '\r' || c == '\n') {
			cursor(0);
			newline();
			return 0;
		}
		if (c == 4 && n == 0) /* Ctrl-D */
			return -1;
		if (c == 0x7f || c == 8) {
			while (n > 0 && ((unsigned char)buf[n - 1] & 0xc0) == 0x80)
				n--;
			if (n > 0)
				n--;
			buf[n] = 0;
		} else if (c == 3) { /* Ctrl-C clears the line */
			n = 0;
			buf[0] = 0;
		} else if (c == 11) { /* Ctrl-K toggles romaji conversion */
			*kana = !*kana;
		} else if (c == 12) { /* Ctrl-L clears the screen */
			fill(0, 0, F.xres, F.yres, F.bg);
			sx = 0;
			sy = 0;
			last = 1;
		} else if (c == 27) { /* swallow escape sequences (arrows etc.) */
			int d = readbyte(30);
			if (d == '[' || d == 'O')
				while ((d = readbyte(30)) >= 0 && !(d >= 0x40 && d <= 0x7e))
					;
			continue;
		} else if (c >= 0x20 && n + 1 < max) {
			buf[n++] = (char)c;
			buf[n] = 0;
		} else {
			continue;
		}
		render_input(&sx, &sy, &last, buf, *kana);
	}
}

/* Polled between tokens: true if Esc or Ctrl-C was pressed. */
static int fb_cancelled(void)
{
	int c;
	while ((c = readbyte(0)) >= 0)
		if (c == 27 || c == 3)
			return 1;
	return 0;
}

/*
 * Puts the VT back into text mode with its old settings.  Called at exit and
 * from fatal signals, so a crash never leaves the console unusable.
 */
static void restore(void)
{
	if (F.vt_ok) {
		ioctl(0, KDSETMODE, KD_TEXT);
		tcsetattr(0, TCSANOW, &F.saved);
		F.vt_ok = 0;
	}
}

static void on_signal(int sig)
{
	restore();
	signal(sig, SIG_DFL);
	raise(sig);
}

static void fb_close(void)
{
	if (F.fb)
		fill(0, 0, F.xres, F.yres, 0);
	restore();
}

static const ui_t fb_ui = {fb_write, fb_readline, fb_cancelled, fb_close, 1, 0};

static int load_font(void)
{
	FILE *f = fopen(BRAINAI_DATADIR "/font.bin", "rb");
	uint8_t *d;
	long len;
	uint32_t i;
	if (!f)
		return -1;
	fseek(f, 0, SEEK_END);
	len = ftell(f);
	fseek(f, 0, SEEK_SET);
	d = malloc(len);
	if (!d || fread(d, 1, len, f) != (size_t)len || len < 8 || memcmp(d, "BRFN", 4)) {
		fclose(f);
		return -1;
	}
	fclose(f);
	memcpy(&F.nglyph, d + 4, 4);
	if ((long)(8 + F.nglyph * 40) != len)
		return -1;
	F.cp = malloc(4 * F.nglyph);
	F.gw = malloc(4 * F.nglyph);
	for (i = 0; i < F.nglyph; i++) {
		memcpy(&F.cp[i], d + 8 + 8 * i, 4);
		memcpy(&F.gw[i], d + 12 + 8 * i, 4);
	}
	F.bits = d + 8 + 8 * F.nglyph;
	return 0;
}

/*
 * Starts the framebuffer UI.  Returns NULL (so main.c falls back to the
 * terminal UI) unless stdin is a Linux VT, font.bin loads, and /dev/fb0 has
 * 16 or 32 bits per pixel.
 */
const ui_t *ui_fb_open(void)
{
	struct fb_fix_screeninfo fix;
	struct termios t;
	int mode, fd;
	if (!isatty(0) || ioctl(0, KDGETMODE, &mode) < 0)
		return NULL; /* not a virtual terminal (SSH, serial, pipe) */
	if (load_font() < 0)
		return NULL;
	fd = open("/dev/fb0", O_RDWR);
	if (fd < 0)
		return NULL;
	if (ioctl(fd, FBIOGET_VSCREENINFO, &F.var) < 0 || ioctl(fd, FBIOGET_FSCREENINFO, &fix) < 0 ||
	    (F.var.bits_per_pixel != 16 && F.var.bits_per_pixel != 32)) {
		close(fd);
		return NULL;
	}
	F.xres = F.var.xres;
	F.yres = F.var.yres;
	F.bpp = F.var.bits_per_pixel;
	F.stride = fix.line_length;
	F.fblen = fix.smem_len;
	F.fb = mmap(NULL, F.fblen, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	close(fd);
	if (F.fb == MAP_FAILED) {
		F.fb = NULL;
		return NULL;
	}
	F.fb += (size_t)F.var.yoffset * F.stride;
	F.cols = F.xres / CW;
	F.rows = F.yres / CH;
	F.bg = pack(0x000000);
	F.color[0] = pack(0xf0f0f0);
	F.color[1] = pack(0x7fd4ff);
	F.color[2] = pack(0x9a9a9a);

	tcgetattr(0, &F.saved);
	t = F.saved;
	t.c_lflag &= ~(ICANON | ECHO | ISIG | IEXTEN);
	t.c_iflag &= ~(IXON | ICRNL);
	t.c_cc[VMIN] = 1;
	t.c_cc[VTIME] = 0;
	tcsetattr(0, TCSANOW, &t);
	ioctl(0, KDSETMODE, KD_GRAPHICS);
	F.vt_ok = 1;
	atexit(restore);
	signal(SIGTERM, on_signal);
	signal(SIGHUP, on_signal);
	signal(SIGSEGV, on_signal);
	signal(SIGABRT, on_signal);

	fill(0, 0, F.xres, F.yres, F.bg);
	F.cx = F.cy = 0;
	fb_write("Esc: 回答を止める  Ctrl-K: かな切替  Ctrl-D: 終了\n", 2);
	return &fb_ui;
}
