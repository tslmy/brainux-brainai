# brainai: build with the Buildroot cross toolchain (CC=...) or natively.
CC ?= cc
CFLAGS ?= -O2
CFLAGS += -Wall -Wextra -std=gnu99
DATADIR ?= /usr/share/brainai
OBJS = engine.o tokenizer.o text.o main.o ui_tty.o ui_fb.o

brainai: $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) -lm

%.o: %.c brainai.h ui.h
	$(CC) $(CFLAGS) -DBRAINAI_DATADIR='"$(DATADIR)"' -c -o $@ $<

clean:
	rm -f brainai $(OBJS)

.PHONY: clean
