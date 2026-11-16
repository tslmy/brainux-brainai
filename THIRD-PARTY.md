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

## GNU Unifont (Japanese)

- Source: <https://unifoundry.com/unifont/> (`unifont_jp-17.0.04.bdf.gz`,
  downloaded from ftp.gnu.org)
- Licence: SIL Open Font License 1.1, or GPL-2.0-or-later with the GNU font
  embedding exception
- On the image: `/usr/share/brainai/font.bin`, a subset produced by
  `package/brainai/src/tools/mkfont.py`, distributed under the same licence.
