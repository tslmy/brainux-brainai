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
    """Pack a BitNet b1.58 weight (scale * {-1,0,1}) as scale + base-3 bytes."""
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


def pb_varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def pb_fields(b):
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
