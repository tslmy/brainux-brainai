#!/usr/bin/env python3
"""Subset a 16-px BDF font (GNU Unifont JP) into brainai's font.bin.

    mkfont.py unifont_jp.bdf[.gz] tok.bin OUT [extra-text-files...]

Keeps ASCII, kana, CJK/full-width punctuation, every character that occurs
in the tokenizer vocabulary and every character in the extra files (UI
strings). Output: "BRFN", u32 n, n x (u32 codepoint, u32 width), then n
32-byte bitmaps (16 rows x 16 bits, MSB = leftmost pixel).
"""
import gzip, struct, sys


def wanted(tok_path, extra):
    cps = set(range(0x20, 0x7F)) | set(range(0x3000, 0x3100)) | set(range(0xFF01, 0xFF9F))
    cps |= set(range(0x2010, 0x2070)) | set(range(0x2190, 0x2200)) | set(range(0x2500, 0x25A0))
    cps |= {0x25A1, 0x25CB, 0x25CF, 0x2605, 0x2606, 0xA5, 0xB0, 0xB1, 0xD7, 0xF7, 0x3C0, 0xFFFD}
    data = open(tok_path, "rb").read()
    n = struct.unpack("<I", data[4:8])[0]
    off = 8
    for _ in range(n):
        _, typ, ln = struct.unpack("<fBB", data[off:off + 6])
        if typ == 1:
            cps |= {ord(c) for c in data[off + 6:off + 6 + ln].decode("utf-8", "replace")}
        off += 6 + ln
    for path in extra:
        cps |= {ord(c) for c in open(path, encoding="utf-8").read()}
    return cps


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    bdf, tok, out = sys.argv[1:4]
    keep = wanted(tok, sys.argv[4:])
    opener = gzip.open if bdf.endswith(".gz") else open
    glyphs = {}
    cp = width = None
    rows = None
    with opener(bdf, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            w = line.split()
            if not w:
                continue
            if w[0] == "ENCODING":
                cp = int(w[1])
            elif w[0] == "BBX":
                width, height = int(w[1]), int(w[2])
            elif w[0] == "BITMAP":
                rows = []
            elif w[0] == "ENDCHAR":
                if cp in keep and width in (8, 16) and len(rows) == 16:
                    bits = bytearray()
                    for r in rows:
                        v = int(r, 16)
                        if width == 8:
                            v <<= 8
                        bits += struct.pack(">H", v)
                    glyphs[cp] = (width, bytes(bits))
                rows = None
            elif rows is not None:
                rows.append(w[0])
    with open(out, "wb") as f:
        f.write(b"BRFN" + struct.pack("<I", len(glyphs)))
        for c in sorted(glyphs):
            f.write(struct.pack("<II", c, glyphs[c][0]))
        for c in sorted(glyphs):
            f.write(glyphs[c][1])
    print(f"font.bin: {len(glyphs)} glyphs ({len(keep - set(glyphs))} wanted but missing)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
