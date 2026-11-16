/* Plain UTF-8 terminal front end (SSH, serial, USB networking). */
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "ui.h"

static volatile sig_atomic_t stop;
static int color_ok;

static void on_int(int sig) { (void)sig; stop = 1; }

static void tty_write(const char *s, int color)
{
	static const char *const esc[] = {"", "\033[36m", "\033[2m"};
	if (color_ok && color)
		fputs(esc[color], stdout);
	fputs(s, stdout);
	if (color_ok && color)
		fputs("\033[0m", stdout);
	fflush(stdout);
}

static int tty_readline(char *buf, size_t max, int *kana)
{
	struct sigaction sa;
	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = SIG_DFL; /* Ctrl-C at the prompt quits */
	sigaction(SIGINT, &sa, NULL);
	if (isatty(0))
		tty_write(*kana ? "あ> " : "> ", 1);
	if (!fgets(buf, (int)max, stdin))
		return -1;
	buf[strcspn(buf, "\r\n")] = 0;
	stop = 0;
	sa.sa_handler = on_int; /* Ctrl-C while answering stops the reply */
	sigaction(SIGINT, &sa, NULL);
	return 0;
}

static int tty_cancelled(void) { return stop; }
static void tty_close(void) {}

static const ui_t tty_ui = {tty_write, tty_readline, tty_cancelled, tty_close, 0, 1};

const ui_t *ui_tty_open(void)
{
	color_ok = isatty(1);
	return &tty_ui;
}
