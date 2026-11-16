#ifndef BRAINAI_UI_H
#define BRAINAI_UI_H

#include <stddef.h>

/* colours: 0 = assistant text, 1 = user text, 2 = info / tool output */
typedef struct {
	void (*write)(const char *utf8, int color);
	/* Reads one line (without newline); returns -1 on end of input. */
	int (*readline)(char *buf, size_t max, int *kana);
	/* Non-zero once the user asked to stop the current reply. */
	int (*cancelled)(void);
	void (*close)(void);
	int wants_kana;      /* default for romaji conversion */
	int echo_conversion; /* print the converted kana after input */
} ui_t;

const ui_t *ui_tty_open(void);
const ui_t *ui_fb_open(void); /* NULL when not on a framebuffer console */

#endif
