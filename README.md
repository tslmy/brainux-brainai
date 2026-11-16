# brainai: brain-ai on Buildroot Brainux

Runs [brain-ai v1 preview](https://huggingface.co/yamada2wasabi/brain-ai-v1-preview)
(Mamba-2 + BitNet b1.58, 14.7M parameters, Japanese) on SHARP Brain. The
upstream app is a closed Windows CE binary, so `package/brainai/src` is a new
inference engine written for the ARM926EJ-S (no FPU):

- ternary projections use an 81-entry lookup table per group of four int8
  activations (T-MAC style, as smolmaxxing does on small ARM cores). Weights are
  repacked at load time so each table block stays in the 16 KiB D-cache and
  four rows share one table base;
- everything element-wise inside a layer (conv, SiLU, SSM recurrence, gated
  RMSNorm) is 32-bit fixed point with a sigmoid table, since soft float was
  the biggest cost; the tied LM head is int8 x int8 on the DSP `SMLAxy`
  instructions;
- activations are int8-quantised per token, as in the model's training.

## Speed

Measured on a PW-SH1 (i.MX283, ARM926EJ-S at 454 MHz, DRAM reads about
170 MB/s) with `bench` (24 greedy steps, logits every step):

| Version | ms/token | tok/s |
|---|---|---|
| first version (float element-wise, row-major LUT) | 331 | 3.0 |
| repacked weights, cache-sized LUT blocks, DSP LM head (bit-exact) | 257 | 3.9 |
| + fixed-point element-wise work, SMULWB in the SSM | 169 | 5.9 |

Per token after these changes: ternary matvecs 68 ms, LM head 49 ms
(9501 x 384 int8, 22 ms of it is just reading 3.6 MB), SSM 27 ms, conv 14 ms,
the rest about 10 ms. Against the bit-exact build, the fixed-point build picks
the same top token on 40 of 41 prompt positions (mean KL 0.002 nats).
`make bench` in `src/` builds the benchmark; add `-DBRAINAI_PROF` to CFLAGS
for the per-stage split.

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
