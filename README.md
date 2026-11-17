# brainux-brainai

Offline Japanese chat on SHARP Brain under [Brainux](https://github.com/brain-hackers/buildbrain)
Linux, using the [brain-ai v1 preview](https://huggingface.co/yamada2wasabi/brain-ai-v1-preview)
model (Mamba-2 + BitNet b1.58, 14.7M parameters) by yamada2wasabi.

The model's author ships their own app for the Brain's stock Windows CE
system. This repository is for Brains running Linux instead: a new inference
engine in C for the ARM926EJ-S (no FPU), packaged as an optional Buildroot
external tree. Without this tree in `BR2_EXTERNAL`, a Brainux build is
unchanged.

## Contents

- `external.desc`, `Config.in`, `external.mk`: Buildroot external tree
  metadata (`BR2_EXTERNAL_BRAINAI_PATH`).
- `package/brainai/`: the engine and app (`brainai`), built from `src/`.
- `package/brainai-data/`: downloads the weights from a pinned Hugging Face
  revision, then converts them with standard-library Python
  (`src/tools/convert.py`) into `/usr/share/brainai/` (`model.bin` 6.7 MB,
  `tok.bin`).
- `BR2_PACKAGE_BRAINAI_FBTERM` (on by default): Buildroot's own
  [fbterm](https://code.google.com/archive/p/fbterm/) and WenQuanYi Zen Hei
  font packages, so kanji show on the Brain's screen. fbterm is C++, so this
  turns on C++ in Buildroot's toolchain.

## Engine

[docs/how-it-works.md](docs/how-it-works.md) walks through the engine and
its optimisations with diagrams and formulas. In short:

- Ternary projections use an 81-entry lookup table per group of four int8
  activations (T-MAC style, as [smolmaxxing](https://github.com/hyperionaidev/smolmaxxing)
  does on small ARM cores). Weights are repacked at load time so each table
  block stays in the 16 KiB D-cache and four rows share one table base.
- Everything element-wise inside a layer (conv, SiLU, SSM recurrence, gated
  RMSNorm) is 32-bit fixed point with a sigmoid table, since soft float was
  the biggest cost. The tied LM head is int8 x int8 on the DSP `SMLAxy`
  instructions.
- Activations are int8-quantised per token, as in the model's training.

## Speed

Measured on a PW-SA3 (i.MX283, ARM926EJ-S at 454 MHz, DRAM reads about
170 MB/s) with `bench`:

| Version | ms/token | tok/s |
|---|---|---|
| first version (float element-wise, row-major LUT) | 331 | 3.0 |
| repacked weights, cache-sized LUT blocks, DSP LM head (bit-exact) | 257 | 3.9 |
| + fixed-point element-wise work, SMULWB in the SSM | 163 | 6.1 |

In the app, `brainai --tty --stats -t 0 -n 64 -p "日本の首都はどこですか？"`
generates at 5.9 tok/s (median of three runs; prompt processing 1.4 s), with
8.6 MB peak RSS. Per token: ternary matvecs 68 ms, LM head 49 ms (9501 x 384
int8, 22 ms of it is just reading 3.6 MB), SSM 27 ms, conv 14 ms, the rest
about 10 ms. Against the bit-exact build, the fixed-point build picks the same
top token on 40 of 41 prompt positions (mean KL 0.002 nats).

`make bench` in `package/brainai/src/` builds the benchmark
(`bench [model.bin] [steps]`); add `-DBRAINAI_PROF` to `CFLAGS` for the
per-stage split.

## Build an image

Use a [buildbrain](https://github.com/brain-hackers/buildbrain) checkout as
the build host. These commands leave its tracked files unchanged. On macOS,
Docker Desktop must be running.

```sh
BRAINUX=/path/to/buildbrain
BRAINAI_ADDON=/path/to/brainux-brainai
make -C "$BRAINUX" docker-buildroot-build
```

The top-level `docker-buildroot-full` target does not forward external trees
into its container, so run the `buildroot_rootfs` target inside it with
`BR2_EXTERNAL` set. The tree is mounted at `/work/os-buildroot/brainai-addon`,
which is only a container mount point:

```sh
BUILD_TMP=$(mktemp -d)
ADDON_MOUNT="$BRAINUX/os-buildroot/brainai-addon"
mkdir -p "$BUILD_TMP/output" "$BUILD_TMP/rootfs" "$ADDON_MOUNT"

docker run --rm --platform linux/amd64 --privileged \
  -v "$BRAINUX":/work \
  -v "$BRAINAI_ADDON":/work/os-buildroot/brainai-addon:ro \
  -v "$BUILD_TMP/output":/work/buildroot_output \
  -v "$BUILD_TMP/rootfs":/work/buildroot_rootfs \
  -w /work buildbrain-buildroot:local \
  bash -lc 'make BR2_EXTERNAL=/work/os-buildroot/brainai-addon buildroot_rootfs'

rmdir "$ADDON_MOUNT"
```

Other add-ons, such as [brainux-cm762](https://github.com/tslmy/brainux-cm762)
for Wi-Fi, can be combined by listing their trees in the same colon-separated
`BR2_EXTERNAL`. Then assemble the SD image with buildbrain's image builder
(the 256 MiB size fits the 6.7 MB model comfortably):

```sh
make -C "$BRAINUX" docker-build
docker run --rm --platform linux/amd64 --privileged \
  -v "$BRAINUX":/work \
  -v "$BUILD_TMP/rootfs":/work/buildroot_rootfs:ro \
  -w /work buildbrain-builder:local \
  bash -lc 'make -C nkbin_maker clean all && \
    ./image/build_image.sh buildroot_rootfs sd_buildroot_brainai.img 256'
rm -rf "$BUILD_TMP"
```

The image is written to `"$BRAINUX/image/sd_buildroot_brainai.img"`.

## Use

- On the Brain's console, run `brainai`. The Linux console cannot draw kanji,
  so brainai restarts itself inside fbterm, a framebuffer terminal, with
  romaji input on: romaji typed at the prompt appears as hiragana as you type
  (`[ ]` keeps text as-is, Ctrl-K toggles). Esc stops a reply and Ctrl-D
  quits. `--tty` skips fbterm.
- Over SSH, `brainai` is the same chat in your own terminal; type Japanese
  directly, or use `-k` for romaji. `brainai -p "質問"` answers once.
- Tools the model calls: `calc` (local calculator, degrees), `time`, `memo`
  (`~/.brainai-memo.txt`), `search` (optional `/usr/share/brainai/search.tsv`
  with `keyword<TAB>text` lines; there is no bundled encyclopedia).
- The upstream app's built-in arithmetic unit (IGC) is not in the published
  weights. As a stand-in, an expression in the message (e.g. `123*45`) is
  computed exactly and used both for `calc` calls and for the number in the
  reply.

## Licence

The code in this repository is MIT (`LICENSE`). The model (CC BY-SA 4.0) is
downloaded at build time and keeps its own licence; fbterm and the font come
from Buildroot's packages under theirs. See [THIRD-PARTY.md](THIRD-PARTY.md).

Parts of this code were written with AI assistance.
