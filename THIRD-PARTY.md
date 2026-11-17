# Third-party material

This repository contains only original source code (MIT, see `LICENSE`). The
build downloads the following at pinned versions and converts them into files
that end up on the device image. Those files keep their own licences.

## brain-ai v1 preview (model weights and tokenizer)

- Source: <https://huggingface.co/yamada2wasabi/brain-ai-v1-preview>
  (revision `19f84a22ebd742cbde5c90095e1f76ebffff295a`)
- Author: yamada2wasabi
- Licence: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- On the image: `/usr/share/brainai/model.bin` and `tok.bin`, produced by
  `package/brainai/src/tools/convert.py` (ternary weights repacked, embedding
  re-encoded). These are adaptations of the model and are distributed under
  CC BY-SA 4.0, with credit to the original author.

The author's own brain-ai app for SHARP Brain's Windows CE system is a
separate program. No code from it is used here.

## fbterm and WenQuanYi Zen Hei

With `BR2_PACKAGE_BRAINAI_FBTERM`, the image also gets these unmodified
Buildroot packages, which Buildroot downloads and records in its usual
licence manifest (`make legal-info`):

- fbterm 1.7.0, GPL-2.0-or-later
- WenQuanYi Zen Hei 0.9.45, GPL-2.0 with the font embedding exception
