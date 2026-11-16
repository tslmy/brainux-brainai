#!/usr/bin/env python3
"""Convert the brain-ai v1 preview Hugging Face release into brainai's files.

    convert.py model.safetensors tokenizer.model OUTDIR

Writes OUTDIR/model.bin (ternary weights packed base-3, four per byte; int8
embedding) and OUTDIR/tok.bin (SentencePiece unigram vocabulary). Python
standard library only, so it runs as a Buildroot host step.
"""
import array, json, math, os, struct, sys

MAGIC, VERSION = b"BRAI", 1


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        blob = f.read()
    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if meta["dtype"] != "F32":
            sys.exit(f"{name}: unsupported dtype {meta['dtype']}")
        s, e = meta["data_offsets"]
        a = array.array("f")
        a.frombytes(blob[s:e])
        if sys.byteorder != "little":
            a.byteswap()
        tensors[name] = (meta["shape"], a)
    return tensors


def f32(values):
    a = array.array("f", values)
    if sys.byteorder != "little":
        a.byteswap()
    return a.tobytes()


def ternary(name, shape, a):
    """Pack a BitNet b1.58 weight (scale * {-1,0,1}) as scale + base-3 bytes.

    Each weight w becomes a digit d = w + 1 (0, 1 or 2). Four digits of one
    row make one byte, c = d0 + 3*d1 + 9*d2 + 27*d3 (0..80), because
    3**4 = 81 fits in a byte. The engine uses c directly as an index into
    its 81-entry lookup table (docs/how-it-works.md, section 4).
    """
    rows, cols = shape
    if cols % 4:
        sys.exit(f"{name}: columns not a multiple of 4")
    scale = max(abs(min(a)), abs(max(a)))
    if scale == 0:
        sys.exit(f"{name}: all-zero weight")
    tol = scale * 1e-3
    out = bytearray(rows * cols // 4)
    j = 0
    for i in range(0, len(a), 4):
        b = 0
        for k, w in enumerate(a[i:i + 4]):
            if abs(w) <= tol:
                d = 1
            elif abs(abs(w) - scale) <= tol:
                d = 2 if w > 0 else 0
            else:
                sys.exit(f"{name}: value {w} is not ternary (scale {scale})")
            b += d * (1, 3, 9, 27)[k]
        out[j] = b
        j += 1
    return f32([scale]) + bytes(out)


def int8_rows(shape, a):
    """Quantise a matrix to int8 with one scale per row: v ~= q * scale,
    scale = max|row| / 127. Used for the embedding, which is also the
    LM (language model) head."""
    rows, cols = shape
    scales, q = [], bytearray(rows * cols)
    for r in range(rows):
        row = a[r * cols:(r + 1) * cols]
        m = max(abs(min(row)), abs(max(row))) or 1.0
        s = m / 127.0
        scales.append(s)
        base = r * cols
        for c, v in enumerate(row):
            q[base + c] = int(round(v / s)) & 0xFF
    return f32(scales) + bytes(q)


def convert_model(tensors, cfg, out):
    """Write model.bin in the order engine.c's model_load() reads it.

    A_log is stored as A = -exp(A_log), the per-head decay rate the engine
    uses directly. Everything that is not ternary or the embedding stays
    float32."""
    V, D, L = cfg["vocab_size"], cfg["hidden_size"], cfg["num_hidden_layers"]
    N, H, P, K = cfg["state_size"], cfg["num_heads"], cfg["head_dim"], cfg["conv_kernel"]
    E = cfg["expand"] * D
    assert H * P == E and cfg["n_groups"] == 1 and cfg["rms_norm"]
    t = lambda n: tensors[n][1]
    sh = lambda n: tensors[n][0]
    parts = [MAGIC, struct.pack("<9I", VERSION, V, D, L, E, N, H, P, K)]
    parts.append(int8_rows(sh("backbone.embeddings.weight"), t("backbone.embeddings.weight")))
    for l in range(L):
        p = f"backbone.layers.{l}."
        m = p + "mixer."
        assert sh(m + "in_proj.weight") == [2 * E + 2 * N + H, D]
        parts.append(f32(t(p + "norm.weight")))
        parts.append(ternary(m + "in_proj.weight", sh(m + "in_proj.weight"), t(m + "in_proj.weight")))
        parts.append(f32(t(m + "conv1d.weight")))
        parts.append(f32(t(m + "conv1d.bias")))
        parts.append(f32(t(m + "dt_bias")))
        parts.append(f32([-math.exp(x) for x in t(m + "A_log")]))
        parts.append(f32(t(m + "D")))
        parts.append(f32(t(m + "norm.weight")))
        parts.append(ternary(m + "out_proj.weight", sh(m + "out_proj.weight"), t(m + "out_proj.weight")))
        print(f"layer {l} packed", file=sys.stderr)
    parts.append(f32(t("backbone.norm_f.weight")))
    with open(out, "wb") as f:
        for b in parts:
            f.write(b)


# tokenizer.model is a protobuf (Google's binary serialisation format).
# Rather than depend on the protobuf library, the two helpers below decode
# the wire format directly: each field is a "key" varint (field number << 3 |
# wire type) followed by a varint (type 0), a length-prefixed byte string
# (type 2), or 4 or 8 raw bytes (types 5 and 1).


def pb_varint(b, i):
    """Decode a varint (7 bits per byte, low bits first, high bit = "more")
    at b[i]; return (value, next index)."""
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def pb_fields(b):
    """Yield (field number, value) for every field of one protobuf message."""
    i = 0
    while i < len(b):
        k, i = pb_varint(b, i)
        f, wt = k >> 3, k & 7
        if wt == 0:
            v, i = pb_varint(b, i)
        elif wt == 2:
            n, i = pb_varint(b, i)
            v, i = b[i:i + n], i + n
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        else:
            sys.exit("tokenizer.model: unsupported protobuf wire type")
        yield f, v


def convert_tokenizer(path, out):
    """Write tok.bin: "BRTK", u32 count, then per piece f32 score, u8 type,
    u8 length and the piece's UTF-8 bytes. In SentencePiece's ModelProto,
    field 1 is a piece (1 = text, 2 = score, 3 = type), field 2 the trainer
    settings and field 3 the normaliser settings."""
    pieces, trainer, normalizer = [], {}, {}
    for f, v in pb_fields(open(path, "rb").read()):
        if f == 1:
            p = {"piece": b"", "score": 0.0, "type": 1}
            for f2, v2 in pb_fields(v):
                if f2 == 1:
                    p["piece"] = v2
                elif f2 == 2:
                    p["score"] = struct.unpack("<f", v2)[0]
                elif f2 == 3:
                    p["type"] = v2
            pieces.append(p)
        elif f == 2:
            trainer = dict(pb_fields(v))
        elif f == 3:
            normalizer = dict(pb_fields(v))
    # brainai implements exactly this configuration: unigram, byte fallback,
    # identity normalisation, no dummy prefix, whitespace kept as U+2581.
    if trainer.get(3, 1) != 1 or trainer.get(35) != 1:
        sys.exit("tokenizer.model: expected a unigram model with byte fallback")
    if normalizer.get(1) != b"identity" or normalizer.get(3, 1) or normalizer.get(4, 1):
        sys.exit("tokenizer.model: unexpected normaliser settings")
    with open(out, "wb") as f:
        f.write(b"BRTK" + struct.pack("<I", len(pieces)))
        for p in pieces:
            if len(p["piece"]) > 255:
                sys.exit("tokenizer.model: piece too long")
            f.write(struct.pack("<fBB", p["score"], p["type"], len(p["piece"])) + p["piece"])


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    st, spm, outdir = sys.argv[1:]
    cfg_path = os.path.join(os.path.dirname(st), "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {
        "vocab_size": 9501, "hidden_size": 384, "num_hidden_layers": 12,
        "state_size": 32, "expand": 2, "head_dim": 48, "num_heads": 16,
        "n_groups": 1, "conv_kernel": 4, "rms_norm": True}
    os.makedirs(outdir, exist_ok=True)
    convert_tokenizer(spm, os.path.join(outdir, "tok.bin"))
    convert_model(load_safetensors(st), cfg, os.path.join(outdir, "model.bin"))


if __name__ == "__main__":
    main()
