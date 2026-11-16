# brainai: brain-ai on Buildroot Brainux

Runs [brain-ai v1 preview](https://huggingface.co/yamada2wasabi/brain-ai-v1-preview)
(Mamba-2 + BitNet b1.58, 14.7M parameters, Japanese) on SHARP Brain. The
upstream app is a closed Windows CE binary, so `package/brainai/src` is a new
inference engine written for the ARM926EJ-S (no FPU):

- ternary projections use a 81-entry lookup table per group of four int8
  activations (the trick smolmaxxing uses on small ARM cores);
- the SSM recurrence is 32-bit fixed point; the tied LM head is int8 x int8;
- activations are int8-quantised per token, as in the model's training.

Its greedy output matches a float reference implementation of the published weights.

## Build

```sh
make BR2_EXTERNAL=/work/buildroot-external/cm762:/work/buildroot-external/brainai docker-buildroot-full
```

`brainai-data` downloads the weights from a pinned Hugging Face revision and
the GNU Unifont JP BDF, then converts them with standard-library Python
(`src/tools/convert.py`, `src/tools/mkfont.py`) into `/usr/share/brainai/`
(`model.bin` 6.7 MB, `tok.bin`, `font.bin`).

## Use

- On the Brain's console, run `brainai`. The Linux console cannot draw kanji, so
  brainai draws on `/dev/fb0` itself with a Unifont subset. Romaji typed at the
  prompt appears as hiragana (`[ ]` keeps text as-is, Ctrl-K toggles), Esc
  stops a reply and Ctrl-D quits.
- Over SSH (Wi-Fi or USB `192.168.7.2`), `brainai` is a plain UTF-8 chat; type
  Japanese directly, or use `-k` for romaji. `brainai -p "質問"` answers once.
- Tools the model calls: `calc` (local calculator, degrees), `time`, `memo`
  (`~/.brainai-memo.txt`), `search` (optional `/usr/share/brainai/search.tsv`
  with `keyword<TAB>text` lines; there is no bundled encyclopedia).
- The upstream app's built-in arithmetic unit (IGC) is not in the published
  weights. As a stand-in, an expression in the message (e.g. `123*45`) is
  computed exactly and used both for `calc` calls and for the number in the
  reply.

Licences: engine MIT; model CC BY-SA 4.0; Unifont OFL-1.1 / GPL-2.0+ with the
font embedding exception.
