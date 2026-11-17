# Running a language model on an electronic dictionary

*How brainux-brainai runs brain-ai, a 14.7-million-parameter Japanese
language model, on a SHARP Brain electronic dictionary at 6 tokens per
second.*

SHARP Brain is a line of Japanese electronic dictionaries. Under the hood,
they are small ARM computers, and the [Brainux](https://github.com/brain-hackers/buildbrain)
project runs Linux on them. In late 2026, the developer yamada2wasabi
released [brain-ai](https://huggingface.co/yamada2wasabi/brain-ai-v1-preview),
a tiny Japanese chat model designed for exactly this kind of device, together
with an app for the Brain's stock Windows CE (Compact Edition) system. That
app is closed source and does not run on Linux.

This article describes the engine we wrote to run the same model on Brainux:
first how it works at all, then how we made it twice as fast. Every number
below was measured on a PW-SA3.

## Contents

1. [The hardware](#1-the-hardware)
2. [The model](#2-the-model)
3. [Getting the model onto the device](#3-getting-the-model-onto-the-device)
4. [Ternary weights: multiplying without multiplying](#4-ternary-weights-multiplying-without-multiplying)
5. [Measure first](#5-measure-first)
6. [Rearranging the weights for the cache](#6-rearranging-the-weights-for-the-cache)
7. [Scoring the vocabulary with DSP (digital signal processing) instructions](#7-scoring-the-vocabulary-with-dsp-instructions)
8. [Replacing floating point with integers](#8-replacing-floating-point-with-integers)
9. [Did it change the answers?](#9-did-it-change-the-answers)
10. [The app around the engine](#10-the-app-around-the-engine)
11. [Results and what is left](#11-results-and-what-is-left)

## 1. The hardware

The PW-SA3 is built around an NXP i.MX283 SoC (system on a chip):

| Part | What it means for us |
|---|---|
| ARM926EJ-S core, 454 MHz, ARMv5TE instruction set | One instruction at a time, no out-of-order execution |
| No FPU (floating-point unit) | Every `float` operation is emulated in software, costing tens of instructions |
| No SIMD (single instruction, multiple data) unit, no NEON | No vector maths; one number at a time |
| A few DSP instructions | 16-bit × 16-bit and 32-bit × 16-bit multiplies in a single instruction |
| 16 KiB data cache, blocking | A cache miss stalls the whole CPU (central processing unit) until memory answers |
| 128 MB of RAM (random-access memory), 115 MB visible to Linux | Plenty for a 6.7 MB model, but slow |

We measured memory bandwidth with a small read loop:

```
  data size     read speed
  ─────────     ──────────
     8 KiB      518 MB/s   ← fits in the 16 KiB cache
    64 KiB      192 MB/s   ← spills out of the cache
   DRAM         168 MB/s   ← DRAM (dynamic RAM), i.e. main memory
```

Two lessons fall out of this table, and they drive every optimisation later:

- **Avoid `float`.** Without an FPU, a single `a * b` on floats is a
  function call.
- **Keep the working set under 16 KiB.** Anything that does not fit is read
  at a third of the speed.

## 2. The model

brain-ai v1 preview combines two ideas.

**Mamba-2 instead of a Transformer.** A Transformer looks back at every
previous token at each step, so its memory and time grow with the
conversation. Mamba-2 is an SSM (state space model): each layer keeps a
fixed-size *state* and updates it once per token. The cost per token is
constant, which is exactly what a small device wants.

**BitNet b1.58 weights.** In the big matrices, every weight is −1, 0 or +1
(that is $\log_2 3 \approx 1.58$ bits, hence the name), times one scale per
matrix. The inputs to those matrices are rounded to 8-bit integers (int8)
per token. A matrix–vector product therefore needs only additions and
subtractions of small integers.

The sizes:

| Symbol | Value | Meaning |
|---|---|---|
| $V$ | 9,501 | vocabulary size (tokens) |
| $D$ | 384 | width of the vector passed between layers |
| $L$ | 12 | layers |
| $E$ | 768 | inner width of each layer ($2D$) |
| $H \times P$ | 16 × 48 | 16 heads of 48 channels ($= E$) |
| $N$ | 32 | state size per channel |
| $K$ | 4 | convolution width (tokens) |

Two terms appear in the diagram below. SiLU (sigmoid linear unit) is the
activation function $\mathrm{SiLU}(x) = x \cdot \sigma(x)$, where
$\sigma(x) = 1 / (1 + e^{-x})$ is the sigmoid. The LM (language model) head
is the last step, which scores every vocabulary word; it reuses the embedding
table, which is why the model is called "tied".

One step, for one token, looks like this:

```
 token id
    │
    ▼
 ┌────────────────────┐
 │ embedding lookup   │  row `token` of a 9,501 × 384 table
 └─────────┬──────────┘
           │ x  (384 numbers)
           ▼
 ╔═══════════════════════════════════════════════════════════╗
 ║  layer 1 … 12                                             ║
 ║                                                           ║
 ║   x ──► normalise ──► round to int8 ──► ternary matrix    ║
 ║                                         384 → 1,616       ║
 ║            ┌──────────────┬───────────┬────────┬──────┐   ║
 ║            ▼              ▼           ▼        ▼      ▼   ║
 ║          gate z      signal x'       B        C     step  ║
 ║          (768)        (768)        (32)     (32)  dt (16) ║
 ║            │              └─────┬─────┴────────┘      │   ║
 ║            │                    ▼                     │   ║
 ║            │        4-token convolution + SiLU        │   ║
 ║            │                    │                     │   ║
 ║            │                    ▼                     │   ║
 ║            │         SSM state update  ◄──────────────┘   ║
 ║            │         (16 × 48 × 32 numbers kept)          ║
 ║            │                    │ y (768)                 ║
 ║            └──────► y · SiLU(z), normalise, round to int8 ║
 ║                                 │                         ║
 ║                                 ▼                         ║
 ║                     ternary matrix 768 → 384              ║
 ║                                 │                         ║
 ║   x  ◄──────────── x + result ──┘                         ║
 ╚═══════════════════════════════════════════════════════════╝
           │
           ▼
 ┌────────────────────┐
 │ normalise, score   │  dot product with all 9,501 rows of
 │ every vocabulary   │  the same embedding table
 │ word ("LM head")   │
 └─────────┬──────────┘
           ▼
   9,501 scores ("logits") → pick the next token
```

The heart of the layer is the SSM state update. For each head $h$, channel
$p$ and state index $n$:

$$
\begin{aligned}
\Delta_h &= \mathrm{softplus}(dt_h + b_h) = \ln\!\left(1 + e^{dt_h + b_h}\right) \\
\bar{A}_h &= e^{\Delta_h A_h} \qquad (A_h < 0,\ \text{so } 0 < \bar{A}_h < 1) \\
S_{h,p,n} &\leftarrow \bar{A}_h \, S_{h,p,n} + \Delta_h \, x_{h,p} \, B_n \\
y_{h,p} &= \sum_{n} S_{h,p,n} \, C_n + D_h \, x_{h,p}
\end{aligned}
$$

In words: the old state fades by a factor $\bar{A}_h$, the new input is added
in through $B$, and the output is read back out through $C$. The state
$S$ is the model's whole memory of the conversation: 16 × 48 × 32 =
24,576 numbers per layer, regardless of how long you chat.

## 3. Getting the model onto the device

The Hugging Face release stores everything as 32-bit floats in a
`safetensors` file. A converter script (`tools/convert.py`, standard-library
Python only, so it runs inside the Buildroot build) rewrites it into a
compact `model.bin`:

```
 model.bin
 ┌──────────────────────────────────────────────────────────┐
 │ "BRAI"  version  V D L E N H P K          (40 bytes)     │
 ├──────────────────────────────────────────────────────────┤
 │ embedding: 9,501 row scales (float)                      │
 │            9,501 × 384 int8 values         3.65 MB       │
 ├──────────────────────────────────────────────────────────┤
 │ layer 1:  norm weights (float)                           │
 │           input matrix: scale + 1,616 × 384 ternary      │
 │                         packed 4 per byte  155 KB        │
 │           conv weights, biases, dt bias, A, D (float)    │
 │           gate-norm weights (float)                      │
 │           output matrix: scale + 384 × 768 ternary       │
 │                          packed 4 per byte  74 KB        │
 ├──────────────────────────────────────────────────────────┤
 │ layer 2 … layer 12  (same layout)                        │
 ├──────────────────────────────────────────────────────────┤
 │ final norm weights (float)                               │
 └──────────────────────────────────────────────────────────┘
                                         total 6.7 MB
```

**Packing ternary weights.** A weight $w \in \{-1, 0, +1\}$ becomes a digit
$d = w + 1 \in \{0, 1, 2\}$. Four digits make a number in base 3:

$$
c = d_0 + 3 d_1 + 9 d_2 + 27 d_3, \qquad 0 \le c \le 80
$$

Since $3^4 = 81 \le 256$, four weights fit in one byte. (Two bits per weight
would also fit four in a byte, but base 3 gives us something more useful: a
table index, as the next section shows.)

```
 weights   +1   0  −1  +1
 digits     2   1   0   2
 code      2·1 + 1·3 + 0·9 + 2·27 = 59   → one byte: 59
```

**Quantising the embedding.** Each of the 9,501 embedding rows gets its own
scale $s_v = \max_i |e_{v,i}| / 127$, and each value is stored as
$\mathrm{round}(e_{v,i} / s_v)$, an int8.

The tokenizer file (`tokenizer.model`, a protobuf, Google's binary
serialisation format) is parsed by hand into `tok.bin`.

## 4. Ternary weights: multiplying without multiplying

A matrix–vector product computes, for each output row $r$:

$$
\mathrm{out}_r = \sum_{c} W_{r,c} \, q_c
$$

With $W_{r,c} \in \{-1, 0, +1\}$, each term is $+q_c$, $0$ or $-q_c$.
Still, deciding which, one weight at a time, costs a few instructions per
weight.

The trick, used in the T-MAC paper (Wei et al., 2024, on table lookups for
low-bit models on edge CPUs) and in the smolmaxxing project, is to
take the inputs **four at a time**. Four inputs $q_0, q_1, q_2, q_3$ can
meet only $3^4 = 81$ different weight patterns, so we compute all 81
possible sums once and store them in a LUT (lookup table):

$$
T[c] = \sum_{k=0}^{3} (d_k - 1)\, q_k \qquad \text{for every } c = d_0 + 3d_1 + 9d_2 + 27d_3
$$

Then each packed weight byte *is* the index of the answer:

$$
\sum_{k=0}^{3} W_{r,4g+k} \, q_{4g+k} = T_g[\text{byte}_{r,g}]
$$

```
 inputs of group g:   q0 = 5   q1 = −2   q2 = 7   q3 = 1

 table T_g (81 entries, built once per token per group):
   c:     0     1     2     3   …    59   …    80
        ───   ───   ───   ───       ───        ───
        −11    −6    −1   −13   …    −1   …    11
                                      ▲
 row r's byte for group g: 59 ────────┘   (weights +1 0 −1 +1)
                     check: 5·1 + (−2)·0 + 7·(−1) + 1·1 = −1

 T[0]  is all weights −1:  −5 + 2 − 7 − 1 = −11
 T[80] is all weights +1:   5 − 2 + 7 + 1 =  11

 one table read replaces four multiply-adds
```

Building the 81 entries naively would cost 81 × 4 multiply-adds. Instead, the
engine splits the code into a low half $i = d_0 + 3d_1$ and a high half
$j = d_2 + 3d_3$, computes 9 partial sums for each half, and adds them:

$$
T[i + 9j] = \underbrace{(d_0 - 1) q_0 + (d_1 - 1) q_1}_{\text{lo}[i],\ 9 \text{ values}}
          + \underbrace{(d_2 - 1) q_2 + (d_3 - 1) q_3}_{\text{hi}[j],\ 9 \text{ values}}
$$

That is 18 small sums and 81 additions per group of four inputs. Every one
of the 1,616 rows then reuses the table, so building it is cheap by
comparison.

Finally, the integer result is turned back into a real number with two
scales: the matrix's weight scale $s_W$ and the input's rounding scale $s_q$:

$$
\mathrm{out}_r^{\text{real}} = s_W \cdot s_q \cdot \sum_c W_{r,c}\, q_c
$$

This version ran at **331 ms per token (3.0 tokens per second)**. It worked,
and it was slow.

## 5. Measure first

Before changing anything, we added a benchmark (`bench`) with an optional
per-stage timer (build with `-DBRAINAI_PROF`). It feeds tokens through the
model and reports the time spent in each part. The first profile:

```
 stage                              ms/token   share
 ─────                              ────────   ─────
 ternary matrices (incl. tables)       127     ████████████████████  38%
 LM head (scoring 9,501 words)          69     ███████████           21%
 convolution + SiLU                     51     ████████              15%
 gate + normalisation                   43     ███████               13%
 SSM state update                       34     █████                 10%
 other                                   7     █                      2%
                                       ───
                                       331
```

The surprise: the ternary maths we had designed for was barely a third of
the time. Small per-element work done in emulated floating point (the
convolution, SiLU, the gate, the normalisation) added up to more.

## 6. Rearranging the weights for the cache

**Problem 1: the tables did not fit in the cache.** The input matrix has 384
inputs, so 96 groups of four, so 96 tables of 81 entries. At 2 bytes per
entry that is 96 × 162 = 15.5 KiB, the entire data cache. As the weights
streamed through, they pushed the tables out, and every lookup risked a
miss.

**Fix: column blocks.** At load time, the engine reorders the weights in
place into blocks of 32 groups (128 columns). It builds the 32 tables for one
block (32 × 162 bytes ≈ 5 KiB, comfortably cached), runs every row over that
block, then moves on to the next block and adds to the running sums.

```
 original layout: each row is one long line of bytes

          group:   0    1    2   …   95
 row 0           [b ][b ][b ]   … [b ]
 row 1           [b ][b ][b ]   … [b ]
 …
 row 1615        [b ][b ][b ]   … [b ]

          all 96 tables needed for every row  →  15.5 KiB of tables

 repacked: three column blocks, processed one after another

          ┌─ block 0 ─┐┌─ block 1 ─┐┌─ block 2 ─┐
          groups 0–31   groups 32–63  groups 64–95
                │
                ▼
          build 32 tables (5 KiB, stays in cache)
          run all 1,616 rows over these 32 groups
          then block 1, then block 2, adding up
```

**Problem 2: one byte per load.** Reading one byte at a time from memory
wastes three quarters of each 32-bit load.

**Fix: four rows per word.** Inside a block, rows are taken four at a time.
For each group, the four rows' bytes sit next to each other in one 32-bit
word. One load fetches the codes for four rows, and all four lookups use the
same table:

```
 inside a block, rows r … r+3, groups g = 0 … 31:

  memory:  [ r  r+1 r+2 r+3 ] [ r  r+1 r+2 r+3 ] [ r  r+1 r+2 r+3 ] …
           └──── group 0 ────┘ └──── group 1 ────┘ └──── group 2 ────┘
                one 32-bit word     one word           one word

  for each word v:
     s0 += T[ v        & 0xff ]   ← row r
     s1 += T[ (v >> 8)  & 0xff ]   ← row r+1
     s2 += T[ (v >> 16) & 0xff ]   ← row r+2
     s3 += T[  v >> 24        ]   ← row r+3
     T  += next group's table
```

**Fix 3: store byte offsets, not indexes.** Table entries are 16-bit, so
entry $c$ lives at byte offset $2c$. The repacker stores $2c$ instead of $c$
(the largest code, 80, becomes 160, which still fits in a byte). The lookup
becomes a single ARM `ldrsh` instruction ("load signed halfword from base +
offset") with no index arithmetic.

**Two compiler details.** The mask `0xff` is kept in a register (GCC, the
GNU Compiler Collection, otherwise emits two instructions for
"shift then mask with a constant"), and the table builder uses nested loops
instead of computing `i % 3` and `i / 3` per entry, which are slow on a CPU
without a divide instruction.

Block size turned out not to matter once the four-rows-per-word layout was
in: 16, 32 and 48 groups per block all ran at about 257 ms per token. The
win came from fitting in the cache at all, not from tuning.

## 7. Scoring the vocabulary with DSP instructions

At the end of each step, the model scores all 9,501 words: a dot product
between the final vector $q$ (384 int8 values) and every embedding row
$e_v$:

$$
\mathrm{logit}_v = s_v \cdot s_q \cdot \sum_{i=0}^{383} e_{v,i}\, q_i
$$

That is 3.6 million multiply-adds per token, and reading 3.65 MB of
embedding table. At 168 MB/s, the reading alone takes about 22 ms.

The ARM926 has DSP instructions that treat a 32-bit register as two 16-bit
halves:

```
 SMLABB  acc += bottom16(a) × bottom16(b)
 SMLATT  acc += top16(a)    × top16(b)

 one instruction, one cycle, signed 16 × 16 → added to 32-bit acc
```

To feed them, we load **four embedding bytes as one 32-bit word** and split
it into two words, each holding two bytes in its 16-bit halves:

```
 word from the table:   [ b3 | b2 | b1 | b0 ]

 w & 0x00ff00ff       = [ 00 | b2 | 00 | b0 ]   halves: (b0, b2)
 (w >> 8) & 0x00ff00ff = [ 00 | b3 | 00 | b1 ]   halves: (b1, b3)

 q prepared once per token as 16-bit pairs:
   q2[0] = [ q2 | q0 ]
   q2[1] = [ q3 | q1 ]

 a += b0·q0  (SMLABB)      b += b2·q2  (SMLATT)
 a += b1·q1  (SMLABB)      b += b3·q3  (SMLATT)
```

Masking only works if the bytes are **unsigned**: a negative byte would need
its sign extended into the top 8 bits of its half. So at load time, every
embedding byte is stored as $e + 128$ (a range of 0 to 255; in practice an
XOR (exclusive or) with `0x80`). The extra $128$ is removed once per word, not per
multiply:

$$
\sum_i e_i\, q_i = \sum_i (e_i + 128)\, q_i \; - \; 128 \sum_i q_i
$$

The second term, $128 \sum_i q_i$, is the same for all 9,501 rows, so it is
computed once per token.

The same section also fixed the sampler: the repetition penalty used to
scan the last 64 tokens for each of the 9,501 words (608,000 comparisons).
Now it marks those 64 tokens in an array, checks the mark per word, and
clears the marks afterwards.

All of section 6 and this section are **bit-exact**: the logits were
identical before and after, which we checked with a checksum after every
change. Together they brought the time to **257 ms per token
(3.9 tokens/s)**.

## 8. Replacing floating point with integers

The biggest single win. Everything element-wise inside a layer now uses
32-bit fixed-point integers.

### Fixed point in one paragraph

A fixed-point number stores a real value $v$ as the integer
$\mathrm{round}(v \cdot 2^f)$. We write this as **Q$f$**: Q16 means "16
fractional bits", so the integer 98,304 in Q16 is $98{,}304 / 2^{16} = 1.5$.
The rules are simple:

$$
\begin{aligned}
\text{Q}a + \text{Q}a &= \text{Q}a \\
\text{Q}a \times \text{Q}b &= \text{Q}(a + b) \\
\text{Q}a \gg s &= \text{Q}(a - s)
\end{aligned}
$$

So multiplying two Q16 numbers gives Q32, and shifting right by 16 brings it
back to Q16. One integer multiply and one shift replace a software float
multiply that costs tens of instructions.

The engine uses:

| Quantity | Format | Range |
|---|---|---|
| activations ($z$, $x'$, $B$, $C$, $y$) | Q16 | ±32,768 |
| convolution weights | Q20 | ±2,048, finer steps for small weights |
| SSM state $S$ | Q20 | ±2,048 |
| decay $\bar{A}$ | Q31 | $[0, 1)$, as precise as 32 bits allow |

### From the matrix result to Q16

The ternary matrix produces an integer sum $a_r$. Its real value is
$a_r \cdot s$, where $s$ is a float scale that changes per token. To get Q16
without a float per element, the engine converts $s$ once per token into an
integer multiplier $m$ and shift $k$:

$$
s = f \cdot 2^{e}\ \ (0.5 \le |f| < 1), \qquad m = f \cdot 2^{30}, \qquad k = 14 - e
$$

$$
(a_r \cdot m) \gg k \;=\; a_r \cdot f \cdot 2^{30 - 14 + e} \;=\; a_r \cdot s \cdot 2^{16}
$$

That is one 64-bit multiply and shift per element.

### SiLU with a table

$\sigma(x)$ needs $e^{-x}$, expensive without an FPU. Instead, the engine
precomputes 1,025 values of $\sigma$ at load time, over $[-16, 16)$ in steps
of $1/32$, and interpolates linearly between neighbours:

```
 σ(x)
 1.0 ┤                          ●●●●●●●●●  (≥ 16: exactly 1)
     │                      ●●●
 0.5 ┤                 ●●●●            each ● is one table entry,
     │            ●●●                  1/32 apart
 0.0 ┤●●●●●●●●●●●                       (≤ −16: exactly 0)
     └──────────────────────────────── x
    −16              0               16

 between entries i and i+1:   σ ≈ t[i] + (t[i+1] − t[i]) · frac
```

For $x$ in Q16, the table index and the fraction come straight out of the
bits, because $2^{16} / 32 = 2^{11}$:

$$
u = x + 16 \cdot 2^{16}, \qquad i = u \gg 11, \qquad \mathrm{frac} = (u \bmod 2^{11}) / 2^{11}
$$

Then $\mathrm{SiLU}(x) = (x \cdot \sigma(x)) \gg 16$.

### The SSM update in integers

Recall the update $S \leftarrow \bar{A} S + \Delta\, x\, B$. Per layer there
are 16 × 48 × 32 = 24,576 of them, so this inner loop matters:

- $\bar{A}$ is computed once per head (16 per layer, still in float) and
  stored as Q31: $\bar{A}_{Q31} = \bar{A} \cdot 2^{31}$. Then
  $\bar{A} \cdot S = (S \cdot \bar{A}_{Q31}) \gg 31$. The engine takes the
  high 32 bits of the 64-bit product (that is, $\gg 32$) and shifts left by
  one, which the CPU does in one `SMULL` instruction plus a shift.
- $\Delta x$ is computed once per channel in Q20.
- $B$ is scaled down to fit in 16 bits: $B_{16} = B \gg s_b$, where $s_b$ is
  the smallest shift that makes every $B_n$ fit. Then
  $\Delta x \cdot B = (\mathrm{SMULWB}(\Delta x, B_{16})) \ll s_b$, where
  `SMULWB` is the DSP instruction "32-bit × bottom 16 bits, keep the top 32
  of the 48-bit result", i.e. $(a \cdot b) \gg 16$.

```
 per state element n (24,576 per layer):

   decay:  hi = (S[n] × Ā_Q31) >> 32          SMULL, high word
   input:  v  = SMULWB(Δx, B16[n]) << s_b     one DSP instruction
           S[n] = v + (hi << 1)
   output: acc += S[n] × C[n]                 64-bit multiply-add
```

### Normalisation: the square root that cancels

Both normalisation steps compute an RMS (root mean square) norm and then
immediately round the result to int8. Write $r = \sqrt{\tfrac{1}{n}\sum_i x_i^2}$
and $u_i = w_i x_i / r$. Rounding to int8 by the largest absolute value
gives:

$$
q_i = \mathrm{round}\!\left(127 \cdot \frac{u_i}{\max_j |u_j|}\right)
    = \mathrm{round}\!\left(127 \cdot \frac{w_i x_i / r}{\max_j |w_j x_j| / r}\right)
    = \mathrm{round}\!\left(127 \cdot \frac{w_i x_i}{\max_j |w_j x_j|}\right)
$$

The $r$ cancels. The int8 values do not depend on the RMS at all; only the
scale that goes with them does:

$$
s_q = \frac{\max_j |u_j|}{127} = \frac{\max_j |w_j x_j|}{127\, r}
$$

So the engine skips dividing each element by $r$, and computes one square
root per normalisation instead of one division per element.

### Result

Together these brought the time to **169 ms per token**, later measured at
**163 ms (6.1 tokens/s)** in the benchmark. This round is not bit-exact,
which the next section checks.

## 9. Did it change the answers?

Fixed point rounds differently from float, so we compared the new engine
with the bit-exact one on a 41-token prompt, position by position:

- **Top choice:** the same next token at 40 of 41 positions.
- **Whole distribution:** the mean KL (Kullback–Leibler) divergence between
  the two engines' next-token probabilities was 0.002 nats. KL divergence
  measures how different two probability distributions are; 0 means
  identical, and 0.002 is tiny.
- **On the device:** with greedy decoding (always taking the top token),
  four of five test prompts gave word-for-word identical replies. The fifth
  recommended a different book.

## 10. The app around the engine

The engine is about a quarter of the code. The rest makes it usable:

- **Tokenizer** (`tokenizer.c`). brain-ai uses a SentencePiece "unigram"
  vocabulary: each of the 9,501 pieces has a score, and the best split of a
  sentence is the one with the highest total score. The encoder finds it with
  the Viterbi algorithm (dynamic programming over byte positions), and falls
  back to raw UTF-8 (Unicode Transformation Format, 8-bit) bytes for
  characters not in the vocabulary.
- **Screen** (`main.c`). The Linux text console cannot draw kanji. Started
  on it, brainai runs itself again inside
  [fbterm](https://code.google.com/archive/p/fbterm/), a framebuffer terminal
  that draws text with FreeType and the WenQuanYi Zen Hei font. Both are
  ready-made Buildroot packages.
- **Input** (`ui_tty.c`, `text.c`). The app edits the input line itself, so
  romaji typed at the prompt is shown converted to hiragana as you type; text
  in `[ ]` is left alone.
- **Tools** (`main.c`). The model was trained to call tools by emitting a
  special token, a call such as `calc: 123*45`, and an end token. The app runs
  `calc`, `time`, `memo` and `search`, then feeds the result back in.
- **Exact arithmetic.** The original app contains an arithmetic unit it calls
  "IGC", which is not part of the published weights. As a stand-in, the app
  finds an arithmetic expression in your message, computes it exactly, and
  when the model starts writing digits after "は" or "=", it inserts the
  exact number instead. So `123*45は？` really answers 5535.
- **Over SSH (Secure Shell)**, the same terminal UI runs in your own
  terminal.

## 11. Results and what is left

| Version | ms/token | tokens/s |
|---|---|---|
| First working engine | 331 | 3.0 |
| Cache-sized blocks, four rows per word, DSP LM head (bit-exact) | 257 | 3.9 |
| Fixed-point layers | 163 | 6.1 |

In the app, the same prompt three times generated at 5.91, 5.91 and 5.88
tokens/s (the app also samples and prints), and reading the prompt dropped
from 3.2 s to 1.4 s. Peak memory use is 8.6 MB.

Where the remaining 163 ms go:

```
 ternary matrices   68 ms  ████████████████████████████████
 LM head            49 ms  ███████████████████████   (22 ms of it: reading 3.65 MB)
 SSM update         27 ms  █████████████
 convolution        14 ms  ███████
 everything else    10 ms  █████
```

Ideas not yet tried:

- **LM head:** process two vocabulary rows per pass, so each prepared $q$ pair
  is loaded once for two rows.
- **SSM:** keep the $\sum_n S_n C_n$ accumulation in 32 bits with the
  `SMLAWB` instruction instead of 64-bit multiply-adds.
- **Shortlist scoring:** score only likely candidate words instead of all
  9,501. Faster, but it changes the model's output, so it is a design choice
  rather than a free speedup.

The lesson matches what smolmaxxing found on much bigger ARM chips: on small
devices, how data moves through memory matters more than how much arithmetic
you do. On a CPU without an FPU, there is a second lesson: floating point is
not free just because it is "only a few elements".

---

*Source: [github.com/tslmy/brainux-brainai](https://github.com/tslmy/brainux-brainai).
Model: [brain-ai v1 preview](https://huggingface.co/yamada2wasabi/brain-ai-v1-preview)
by yamada2wasabi, CC BY-SA 4.0. Parts of this engine and article were
written with AI assistance.*
