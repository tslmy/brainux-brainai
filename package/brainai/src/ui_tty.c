/*
 * Terminal front end: the only UI.  It runs over SSH (Secure Shell), a
 * serial console, USB (Universal Serial Bus) networking, and on the Brain's
 * own screen inside fbterm (main.c starts fbterm there, because the Linux
 * text console cannot draw kanji).  The terminal itself draws the text.
 *
 * When the input is a terminal, brainai edits the line itself so that
 * romaji typed at the prompt can be shown already converted to hiragana.
 * For that the terminal is put in non-canonical mode (termios without
 * ICANON): the kernel hands over every key press at once and does no
 * echo or line editing of its own.  After each key the line is redrawn
 * with ANSI (American National Standards Institute) escape codes.
 * Piped input (`echo ... | brainai`) is read line by line as before.
 */
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

#include "brainai.h"
#include "ui.h"

static volatile sig_atomic_t stop; /* Ctrl-C while a reply runs (piped input) */
static int color_ok;               /* stdout is a terminal, so colours work */
static int edit_ok;                /* stdin is a terminal we edit ourselves */
static struct termios saved;       /* settings to restore on exit */

static void on_int(int sig) { (void)sig; stop = 1; }

/* Colour 0 plain, 1 cyan (user text), 2 yellow (info and tool output). */
static void tty_write(const char *s, int color)
{
	static const char *const esc[] = {"", "\033[36m", "\033[33m"};
	if (color_ok && color)
		fputs(esc[color], stdout);
	fputs(s, stdout);
	if (color_ok && color)
		fputs("\033[0m", stdout);
	fflush(stdout);
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

/* Screen columns a UTF-8 string takes: ASCII 1, everything else (kana,
 * kanji, full-width punctuation) 2.  Close enough for brainai's text. */
static int columns(const char *s)
{
	int w = 0;
	for (; *s; s++) {
		unsigned char c = (unsigned char)*s;
		if (c < 0x80)
			w++;
		else if (c >= 0xc0) /* first byte of a multi-byte character */
			w += 2;
	}
	return w;
}

/*
 * Redraws the input line: go back to where it started (*row lines up),
 * clear to the end of the screen, print the prompt and the text (as kana
 * when enabled), then remember how many lines down the cursor ended.
 */
static void render(const char *raw, int kana, int *row)
{
	char disp[4096];
	const char *prompt = kana ? "あ> " : "> ";
	struct winsize ws;
	int cols = ioctl(1, TIOCGWINSZ, &ws) == 0 && ws.ws_col ? ws.ws_col : 80;
	int w;

	if (*row)
		printf("\033[%dA", *row); /* cursor up */
	fputs("\r\033[J", stdout);        /* column 0, clear to end of screen */
	if (!kana || romaji_to_kana(raw, disp, sizeof(disp)))
		snprintf(disp, sizeof(disp), "%s", raw);
	tty_write(prompt, 2);
	tty_write(disp, 1);
	/* A full last line leaves the cursor on that line, not the next. */
	w = columns(prompt) + columns(disp);
	*row = w ? (w - 1) / cols : 0;
}

/*
 * Line editor: Enter submits, Backspace deletes a whole UTF-8 character,
 * Ctrl-C clears the line, Ctrl-D on an empty line quits, Ctrl-K toggles
 * kana, Ctrl-L clears the screen.  Arrow keys are ignored.
 */
static int edit_line(char *buf, size_t max, int *kana)
{
	size_t n = 0;
	int row = 0;
	buf[0] = 0;
	render(buf, *kana, &row);
	for (;;) {
		int c = readbyte(-1);
		if (c == -2)
			return -1;
		if (c == '\r' || c == '\n') {
			tty_write("\n", 0);
			return 0;
		}
		if (c == 4 && n == 0) { /* Ctrl-D */
			tty_write("\n", 0);
			return -1;
		}
		if (c == 0x7f || c == 8) {
			while (n > 0 && ((unsigned char)buf[n - 1] & 0xc0) == 0x80)
				n--; /* skip continuation bytes 10xxxxxx */
			if (n > 0)
				n--;
			buf[n] = 0;
		} else if (c == 3) { /* Ctrl-C clears the line */
			n = 0;
			buf[0] = 0;
		} else if (c == 11) { /* Ctrl-K toggles romaji conversion */
			*kana = !*kana;
		} else if (c == 12) { /* Ctrl-L clears the screen */
			fputs("\033[H\033[2J", stdout);
			row = 0;
		} else if (c == 27) { /* swallow escape sequences (arrows etc.) */
			int d = readbyte(30);
			if (d == '[' || d == 'O')
				while ((d = readbyte(30)) >= 0 && !(d >= 0x40 && d <= 0x7e))
					;
			continue;
		} else if (c >= 0x20 && n + 1 < max) {
			buf[n++] = (char)c; /* UTF-8 bytes arrive one by one */
			buf[n] = 0;
		} else {
			continue;
		}
		render(buf, *kana, &row);
	}
}

/*
 * Piped input: plain fgets.  Ctrl-C quits at the prompt (default signal
 * action) but only stops the current reply while the model is answering.
 */
static int read_line(char *buf, size_t max, int *kana)
{
	struct sigaction sa;
	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = SIG_DFL;
	sigaction(SIGINT, &sa, NULL);
	if (isatty(0))
		tty_write(*kana ? "あ> " : "> ", 2);
	if (!fgets(buf, (int)max, stdin))
		return -1;
	buf[strcspn(buf, "\r\n")] = 0;
	stop = 0;
	sa.sa_handler = on_int;
	sigaction(SIGINT, &sa, NULL);
	return 0;
}

static int tty_readline(char *buf, size_t max, int *kana)
{
	return edit_ok ? edit_line(buf, max, kana) : read_line(buf, max, kana);
}

/* Polled between tokens: Esc or Ctrl-C stops the reply. */
static int tty_cancelled(void)
{
	int c;
	if (!edit_ok)
		return stop;
	while ((c = readbyte(0)) >= 0)
		if (c == 27 || c == 3)
			return 1;
	return 0;
}

/* Puts the terminal back as it was.  Also runs from fatal signals, so a
 * crash never leaves the shell without echo. */
static void restore(void)
{
	if (edit_ok) {
		tcsetattr(0, TCSANOW, &saved);
		edit_ok = 0;
	}
}

static void on_fatal(int sig)
{
	restore();
	signal(sig, SIG_DFL);
	raise(sig);
}

static void tty_close(void) { restore(); }

/* wants_kana = 0 (type Japanese directly; -k turns it on);
 * echo_conversion: show "→ かな" after a piped line was converted. */
static ui_t tty_ui = {tty_write, tty_readline, tty_cancelled, tty_close, 0, 1};

const ui_t *ui_tty_open(int interactive)
{
	color_ok = isatty(1);
	if (interactive && isatty(0) && isatty(1) && tcgetattr(0, &saved) == 0) {
		struct termios t = saved;
		static const int sigs[] = {SIGTERM, SIGHUP, SIGQUIT, SIGSEGV, SIGABRT};
		size_t i;
		t.c_lflag &= ~(ICANON | ECHO | ISIG | IEXTEN); /* keys reach us raw */
		t.c_iflag &= ~(IXON | ICRNL); /* Ctrl-S/Ctrl-Q and CR as plain keys */
		t.c_cc[VMIN] = 1;
		t.c_cc[VTIME] = 0;
		for (i = 0; i < sizeof(sigs) / sizeof(sigs[0]); i++)
			signal(sigs[i], on_fatal);
		tcsetattr(0, TCSANOW, &t);
		edit_ok = 1;
		tty_ui.echo_conversion = 0; /* the kana are already on screen */
	}
	return &tty_ui;
}
