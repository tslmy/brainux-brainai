/*
 * Interactive figures and symbol tooltips for the brainux-brainai explainer.
 * Plain browser JavaScript, no dependencies, no build step. Mounts on
 * <figure data-widget="…"> elements inside .brainai-explainer, and gives
 * every symbol in the (pre-rendered MathML) maths a hover/tap/focus tooltip.
 */
(() => {
  'use strict';

  // ───────────────────────────── glossary ─────────────────────────────
  // Keys are what a symbol looks like: "x", "s_W", "Abar_h", "out_r^real".
  // Lookups try the innermost [data-sec] first, then outer ones, then "*".
  const GLOSS = {
    '*': {
      V: '<b>V = 9,501</b>: the vocabulary size, the number of different tokens the model knows.',
      D: '<b>D = 384</b>: how many numbers are passed from one layer to the next.',
      L: '<b>L = 12</b>: the number of layers.',
      E: '<b>E = 768</b>: the inner width of each layer, twice D.',
      H: '<b>H = 16</b>: the number of heads. Each head has its own decay speed.',
      P: '<b>P = 48</b>: channels per head. H × P = 16 × 48 = 768 = E.',
      N: '<b>N = 32</b>: the size of the state kept for each channel.',
      K: '<b>K = 4</b>: the convolution looks at the last 4 tokens.',
      x: '<b>x</b>: the signal, a list of numbers being processed (here, one number of it).',
      'x^′': '<b>x′</b>: the “signal” part of the matrix output, 768 numbers that go through the convolution into the SSM.',
      z: '<b>z</b>: the “gate”, 768 numbers that decide how much of the SSM output passes through.',
      y: '<b>y</b>: the SSM output, read out of the state.',
      S: '<b>S</b>: the state, the layer’s memory of everything it has seen so far.',
      B: '<b>B</b>: input weights: how strongly the current token is written into each state number. Computed fresh from every token.',
      C: '<b>C</b>: output weights: how strongly each state number is read out. Computed fresh from every token.',
      Abar: '<b>Ā</b> (“A bar”): the decay factor between 0 and 1. Each token, the old state is multiplied by it, so it slowly fades.',
      'Δ': '<b>Δ</b> (delta): the step size. Large Δ: forget faster, take in more of the current token.',
      q: '<b>q</b>: the int8 inputs, numbers rounded to whole values between −127 and 127.',
      w: '<b>w</b>: one ternary weight, −1, 0 or +1.',
      W: '<b>W</b>: the ternary weight matrix. Every entry is −1, 0 or +1.',
      d: '<b>d</b>: a base-3 digit, d = w + 1, so 0, 1 or 2.',
      T: '<b>T</b>: the lookup table of 81 precomputed sums.',
      s: '<b>s</b>: a scale factor: the real value is the integer times s.',
      s_W: '<b>s<sub>W</sub></b>: the weight scale, one number per matrix. The real weights are s<sub>W</sub> × (−1, 0 or +1).',
      s_q: '<b>s<sub>q</sub></b>: the input scale: the real inputs are s<sub>q</sub> × q. Recomputed for every token.',
      s_v: '<b>s<sub>v</sub></b>: the scale of embedding row v: its largest absolute value divided by 127.',
      σ: '<b>σ</b> (sigma): the sigmoid, σ(x) = 1 / (1 + e<sup>−x</sup>). Squashes any number into the range 0 to 1.',
      SiLU: '<b>SiLU</b>: the activation x · σ(x). Positive values pass almost unchanged, negative ones shrink towards 0.',
      softplus: '<b>softplus</b>: ln(1 + e<sup>x</sup>), a smooth version of max(0, x). Always positive.',
      ln: '<b>ln</b>: the natural logarithm, the inverse of e<sup>x</sup>.',
      e: '<b>e</b> ≈ 2.718: Euler’s number. e<sup>x</sup> is the exponential function.',
      log: '<b>log<sub>2</sub></b>: logarithm base 2: how many bits it takes to tell the options apart.',
      log_2: '<b>log<sub>2</sub></b>: logarithm base 2: how many bits it takes to tell the options apart. log<sub>2</sub> 3 ≈ 1.58.',
      round: '<b>round</b>: to the nearest whole number.',
      max: '<b>max</b>: the largest value, over the index written below it.',
      'max_i': '<b>max<sub>i</sub></b>: the largest value over all positions i.',
      'max_j': '<b>max<sub>j</sub></b>: the largest value over all positions j.',
      '∑': '<b>Σ</b> (sum): add up the expression after it for every value of the index written below.',
      '≫': '<b>≫ s</b>: shift right by s bits, which divides by 2<sup>s</sup> and drops the remainder. One instruction.',
      '≪': '<b>≪ s</b>: shift left by s bits, which multiplies by 2<sup>s</sup>. One instruction.',
      '←': '<b>←</b>: “becomes”: the left side is replaced by the right side, computed from the old values.',
      '∈': '<b>∈</b>: “is one of”.',
      '≈': '<b>≈</b>: approximately equal.',
      '×': '<b>×</b>: multiplied by.',
      '⋅': '<b>·</b>: multiplied by.',
      '≤': '<b>≤</b>: less than or equal to.',
      '<': '<b>&lt;</b>: less than.',
      '/': '<b>/</b>: divided by.',
      '∣': '<b>| … |</b>: the absolute value, the number without its sign: |−3| = 3.',
      h: '<b>h</b>: the head index, 0 to 15.',
      p: '<b>p</b>: the channel index inside a head, 0 to 47.',
      n: '<b>n</b>: the state index, 0 to 31.',
      r: '<b>r</b>: the row index: which output number is being computed.',
      g: '<b>g</b>: the group index: which block of four inputs.',
      k: '<b>k</b>: position inside a group of four, 0 to 3.',
      v: '<b>v</b>: a vocabulary index, one of the 9,501 tokens.',
      i: '<b>i</b>: an index into a list of numbers.',
      j: '<b>j</b>: an index into a list of numbers.',
      t: '<b>t</b>: the token number, i.e. time.',
      GB: '<b>GB</b>: groups per block, 32 in the engine. Only this many tables have to stay in the cache at once.',
      // tooltips used by the figures
      x_t: '<b>x<sub>t</sub></b>: the input at token t.',
      S_t: '<b>S<sub>t</sub></b>: the state after token t.',
      y_t: '<b>y<sub>t</sub></b>: the output at token t, S<sub>t</sub> × C + D × x<sub>t</sub>.',
      dt: '<b>dt</b>: the raw step value the matrix produces for this head (plus a learned bias). Softplus turns it into Δ.',
      A: '<b>A</b>: the learned, negative decay rate of this head.',
      half: '<b>half-life</b>: after this many tokens with no input, the state has faded to half.',
      Dskip: '<b>D</b>: the skip weight: a bit of the input is passed straight to the output.',
      q_k: '<b>q<sub>k</sub></b>: input k of the group, an int8 value.',
      q_0: '<b>q<sub>0</sub></b>: the first input of the group of four (an int8 value).',
      q_1: '<b>q<sub>1</sub></b>: the second input of the group.',
      q_2: '<b>q<sub>2</sub></b>: the third input of the group.',
      q_3: '<b>q<sub>3</sub></b>: the fourth input of the group.',
      d_0: '<b>d<sub>0</sub></b>: base-3 digit of the first weight, w<sub>0</sub> + 1.',
      d_1: '<b>d<sub>1</sub></b>: base-3 digit of the second weight.',
      d_2: '<b>d<sub>2</sub></b>: base-3 digit of the third weight.',
      d_3: '<b>d<sub>3</sub></b>: base-3 digit of the fourth weight.',
      d_k: '<b>d<sub>k</sub></b>: the base-3 digit of weight k: 0 for −1, 1 for 0, 2 for +1.',
      lo: '<b>lo[i]</b>: the 9 possible partial sums of the first two inputs, (d<sub>0</sub>−1)q<sub>0</sub> + (d<sub>1</sub>−1)q<sub>1</sub>.',
      hi: '<b>hi[j]</b>: the 9 possible partial sums of the last two inputs, (d<sub>2</sub>−1)q<sub>2</sub> + (d<sub>3</sub>−1)q<sub>3</sub>.',
      c: '<b>c</b>: the base-3 code of four weights, d<sub>0</sub> + 3d<sub>1</sub> + 9d<sub>2</sub> + 27d<sub>3</sub>, between 0 and 80. Stored as one byte.',
      'T[c]': '<b>T[c]</b>: entry c of the table: the sum of the four inputs, each multiplied by its weight from pattern c.',
      word: '<b>v</b>: one 32-bit word loaded from memory: four bytes, one code each for rows r, r+1, r+2, r+3.',
      s0: '<b>s0 … s3</b>: running sums for the four rows that share this word.',
      vreal: '<b>v</b>: the real value to be stored.',
      f: '<b>f</b>: the number of fractional bits: Q<i>f</i> stores v as round(v · 2<sup>f</sup>).',
      nint: '<b>n</b>: the integer actually stored, round(v · 2<sup>f</sup>).',
      step: '<b>step</b>: the smallest difference Q<i>f</i> can represent, 2<sup>−f</sup>.',
      Δx: '<b>Δx</b>: step size times input for one channel, computed once per channel and stored in Q20.',
      B_16: '<b>B<sub>16</sub></b>: B shifted right by s<sub>b</sub> so that it fits in 16 bits, the half-register SMULWB reads.',
      s_b: '<b>s<sub>b</sub></b>: the smallest shift that makes every B<sub>n</sub> fit in 16 bits. Shifted back left at the end.',
      SMULWB: '<b>SMULWB</b>: ARM DSP instruction “signed multiply word by bottom half”: a 32-bit value times the low 16 bits of another, keeping the top 32 bits of the 48-bit product, i.e. (a · b) ≫ 16.',
      x_i: '<b>x<sub>i</sub></b>: the i-th real value before rounding.',
      q_i: '<b>q<sub>i</sub></b>: the i-th value after rounding: a whole number between −127 and 127.',
      λ: '<b>λ</b>: a common factor applied to every value. It changes the scale s, never the integers q<sub>i</sub>.',
      frac: '<b>frac</b>: how far x lies between table entry i and entry i+1, from 0 to 1.',
      u: '<b>u</b>: x moved up by 16 so it starts at 0 at the left end of the table, still in Q16.',
      'σ_t': '<b>t[i]</b>: table entry i, the sigmoid at −16 + i/32, computed once at load time.',
    },
    2: {
      x: '<b>x</b>: the vector of D = 384 numbers passed from layer to layer. In SiLU(x) and σ(x), any single number.',
      x_h: '<b>x<sub>h,p</sub></b>: the input for head h, channel p (one of the 768 signal numbers after the convolution).',
      'x_h,p': '<b>x<sub>h,p</sub></b>: the input for head h, channel p (one of the 768 signal numbers after the convolution).',
      'Δ_h': '<b>Δ<sub>h</sub></b>: the step size of head h for this token. Large Δ: forget faster and take in more of the current token.',
      'dt_h': '<b>dt<sub>h</sub></b>: the raw step value for head h, one of the 16 “step” outputs of the input matrix.',
      'b_h': '<b>b<sub>h</sub></b>: a learned bias for head h, fixed after training.',
      'A_h': '<b>A<sub>h</sub></b>: the learned decay rate of head h, always negative. Fixed after training.',
      'Abar_h': '<b>Ā<sub>h</sub></b>: the decay factor of head h for this token, between 0 and 1. The state is multiplied by it, so the past fades.',
      'S_h,p,n': '<b>S<sub>h,p,n</sub></b>: one number of the state: head h, channel p, state index n. The layer keeps 16 × 48 × 32 of them.',
      'B_n': '<b>B<sub>n</sub></b>: how strongly this token is written into state index n. 32 values, computed from the token.',
      'C_n': '<b>C<sub>n</sub></b>: how strongly state index n is read out. 32 values, computed from the token.',
      'y_h,p': '<b>y<sub>h,p</sub></b>: the output for head h, channel p.',
      'D_h': '<b>D<sub>h</sub></b>: the skip weight of head h: a bit of the input is passed straight to the output.',
      D: '<b>D = 384</b>: how many numbers are passed from one layer to the next.',
    },
    3: {
      w: '<b>w</b>: one ternary weight, −1, 0 or +1.',
      c: '<b>c</b>: the code of four weights, a number from 0 to 80 that fits in one byte.',
      d_0: '<b>d<sub>0</sub></b>: digit of the first weight (weight + 1).', d_1: '<b>d<sub>1</sub></b>: digit of the second weight.',
      d_2: '<b>d<sub>2</sub></b>: digit of the third weight.', d_3: '<b>d<sub>3</sub></b>: digit of the fourth weight.',
      'e_v,i': '<b>e<sub>v,i</sub></b>: number i of embedding row v, as a 32-bit float before conversion.',
      i: '<b>i</b>: the position in the row, 0 to 383.',
      x_i: '<b>x<sub>i</sub></b>: the i-th value before rounding.',
      s: '<b>s</b>: the scale, max|x<sub>i</sub>| / 127. Multiply a stored integer by s to get the value back.',
    },
    4: {
      out_r: '<b>out<sub>r</sub></b>: output number r of the matrix–vector product, still as an integer.',
      'out_r^real': '<b>out<sub>r</sub><sup>real</sup></b>: the same output as a real number, after applying both scales.',
      'W_r,c': '<b>W<sub>r,c</sub></b>: the weight in row r, column c: −1, 0 or +1.',
      'W_r,4g+k': '<b>W<sub>r,4g+k</sub></b>: the weight in row r, at input k of group g.',
      q_c: '<b>q<sub>c</sub></b>: input number c, an int8 value.',
      'q_4g+k': '<b>q<sub>4g+k</sub></b>: input k of group g.',
      c: '<b>c</b>: under Σ, the column index (which input). In T[c], the base-3 code of a four-weight pattern, 0 to 80.',
      T_g: '<b>T<sub>g</sub></b>: the 81-entry table for group g, rebuilt for every token because the inputs change.',
      'byte_r,g': '<b>byte<sub>r,g</sub></b>: the stored byte for row r, group g: the code of its four weights.',
      i: '<b>i</b>: the low half of the code, d<sub>0</sub> + 3d<sub>1</sub>, 0 to 8.',
      j: '<b>j</b>: the high half of the code, d<sub>2</sub> + 3d<sub>3</sub>, 0 to 8.',
    },
    6: { c: '<b>c</b>: the code, 0 to 80. Stored as 2c, the byte offset of its 16-bit table entry.' },
    7: {
      logit_v: '<b>logit<sub>v</sub></b>: the score of vocabulary word v. Higher means more likely next.',
      'e_v,i': '<b>e<sub>v,i</sub></b>: number i of embedding row v, an int8.',
      e_v: '<b>e<sub>v</sub></b>: embedding row v: 384 int8 values describing word v.',
      e_i: '<b>e<sub>i</sub></b>: one embedding byte (the row is left implicit).',
      q: '<b>q</b>: the final vector, 384 int8 values.',
      q_i: '<b>q<sub>i</sub></b>: number i of the final int8 vector.',
      i: '<b>i</b>: the position in the vector, 0 to 383.',
    },
    8.1: {
      v: '<b>v</b>: the real value we want to store.',
      f: '<b>f</b>: the number of fractional bits. Q<i>f</i> stores v as round(v · 2<sup>f</sup>).',
      Q: '<b>Q<i>f</i></b>: a fixed-point format with f fractional bits.',
      a: '<b>a</b>: a number of fractional bits.', b: '<b>b</b>: another number of fractional bits.',
      s: '<b>s</b>: a shift amount in bits.',
    },
    8.2: {
      a_r: '<b>a<sub>r</sub></b>: the integer sum the ternary matrix produced for row r.',
      s: '<b>s</b>: the real scale of this matrix result (s<sub>W</sub> · s<sub>q</sub>), one float per token.',
      f: '<b>f</b>: the mantissa of s, between 0.5 and 1 in size.',
      e: '<b>e</b>: the binary exponent of s, so s = f · 2<sup>e</sup>.',
      m: '<b>m</b>: the integer multiplier, f · 2<sup>30</sup>.',
      k: '<b>k</b>: the shift, 14 − e, chosen so the result lands in Q16.',
    },
    8.3: {
      x: '<b>x</b>: the input to σ, in Q16.',
      i: '<b>i</b>: the table index, the top bits of u.',
      mod: '<b>mod</b>: the remainder after division; here, the low 11 bits.',
      t: '<b>t[i]</b>: table entry i, the sigmoid at −16 + i/32.',
    },
    8.4: {
      Abar: '<b>Ā</b>: this head’s decay factor, between 0 and 1.',
      Abar_Q31: '<b>Ā<sub>Q31</sub></b>: Ā stored in Q31, i.e. Ā · 2<sup>31</sup> as a 32-bit integer.',
      x: '<b>x</b>: the input of this channel.',
      B_n: '<b>B<sub>n</sub></b>: input weight n of this token.',
      a: '<b>a</b>: the first operand, a full 32-bit register.',
      b: '<b>b</b>: the second operand; only its bottom 16 bits are used.',
    },
    8.5: {
      r: '<b>r</b>: the RMS (root mean square) of x: square every value, average, take the square root.',
      n: '<b>n</b>: the number of values, 384 or 768.',
      x_i: '<b>x<sub>i</sub></b>: value i before normalisation.',
      x_j: '<b>x<sub>j</sub></b>: value j before normalisation.',
      w_i: '<b>w<sub>i</sub></b>: the learned normalisation weight for position i.',
      w_j: '<b>w<sub>j</sub></b>: the learned normalisation weight for position j.',
      u_i: '<b>u<sub>i</sub></b>: value i after normalisation, w<sub>i</sub> x<sub>i</sub> / r.',
      u_j: '<b>u<sub>j</sub></b>: value j after normalisation.',
      q_i: '<b>q<sub>i</sub></b>: value i rounded to int8.',
    },
    11: {
      S_n: '<b>S<sub>n</sub></b>: state number n.', C_n: '<b>C<sub>n</sub></b>: output weight n.',
      q: '<b>q</b>: the final int8 vector, prepared as 16-bit pairs.',
    },
    9: {
      p_v: '<b>p<sub>v</sub></b>: the probability the first engine gives word v.',
      'p^′_v': '<b>p′<sub>v</sub></b>: the probability the other engine gives word v.',
      "p_v^′": '<b>p′<sub>v</sub></b>: the probability the other engine gives word v.',
      logit_v: '<b>logit<sub>v</sub></b>: the score of word v.', logit_u: '<b>logit<sub>u</sub></b>: the score of word u.',
      u: '<b>u</b>: runs over all 9,501 words in the sum.',
    },
  };

  const INVIS = /[⁡⁢⁣⁤\s]/g;
  const clean = (s) => s.replace(INVIS, '').replace(/′/g, '′');
  const missing = new Set();

  function chainFor(node) {
    const ids = [];
    let n = node.closest('[data-sec]');
    while (n) { ids.push(n.getAttribute('data-sec')); n = n.parentElement && n.parentElement.closest('[data-sec]'); }
    ids.push('*');
    return ids;
  }
  function look(chain, key) {
    for (const id of chain) { const g = GLOSS[id]; if (g && Object.prototype.hasOwnProperty.call(g, key)) return g[key]; }
    return null;
  }
  function mark(node, html) {
    node.setAttribute('data-tip', html);
    node.setAttribute('tabindex', '0');
    node.classList.add('sym');
  }

  // Key of a MathML node used as a base: "x", "Abar", "dt", "max".
  function keyOf(n) {
    const t = n.localName;
    if (t === 'mi' || t === 'mo' || t === 'mtext') return clean(n.textContent);
    if (t === 'mover' && n.children[1] && /[ˉ¯]/.test(n.children[1].textContent)) {
      const b = keyOf(n.children[0]); return b == null ? null : b + 'bar';
    }
    if (t === 'mrow' && n.children.length && [...n.children].every((c) => c.localName === 'mi' || (c.localName === 'mo' && clean(c.textContent) === ''))) {
      return clean(n.textContent);
    }
    return null;
  }

  function annotateMath(math) {
    const chain = chainFor(math);
    const walk = (n) => {
      for (const c of n.children) {
        const t = c.localName;
        if (t === 'annotation') continue;
        if (t === 'msub' || t === 'msup' || t === 'msubsup') {
          const b = keyOf(c.children[0]);
          if (b != null) {
            const a1 = clean(c.children[1].textContent);
            const cand = t === 'msub' ? [b + '_' + a1] : t === 'msup' ? [b + '^' + a1] : [b + '_' + a1 + '^' + clean(c.children[2].textContent), b + '_' + a1];
            const hit = cand.map((k) => look(chain, k)).find(Boolean);
            if (hit) { mark(c, hit); continue; }
          }
          walk(c);
          continue;
        }
        const k = keyOf(c);
        if (k != null && k !== '') {
          const v = look(chain, k);
          if (v) { mark(c, v); continue; }
          if (t !== 'mo' && t !== 'mtext' && !/^[\d.,]+$/.test(k)) missing.add(chain[0] + ':' + k);
          if (t === 'mi' || t === 'mrow') continue;
        }
        walk(c);
      }
    };
    walk(math);
  }

  function annotate(root) {
    root.querySelectorAll('math').forEach(annotateMath);
    root.querySelectorAll('[data-k]').forEach((el) => {
      const v = look(chainFor(el), el.getAttribute('data-k'));
      if (v) mark(el, v); else missing.add('widget:' + el.getAttribute('data-k'));
    });
  }

  // ───────────────────────────── tooltip ─────────────────────────────
  let tip, pinned = null, shown = null;
  function ensureTip() {
    if (tip) return;
    tip = document.createElement('div');
    tip.className = 'bx-tip'; tip.id = 'bx-tip'; tip.setAttribute('role', 'tooltip'); tip.hidden = true;
    document.body.appendChild(tip);
  }
  function show(el) {
    ensureTip();
    if (shown && shown !== el) { shown.classList.remove('is-open'); shown.removeAttribute('aria-describedby'); }
    shown = el;
    el.classList.add('is-open');
    el.setAttribute('aria-describedby', 'bx-tip');
    tip.innerHTML = el.getAttribute('data-tip');
    tip.hidden = false;
    const r = el.getBoundingClientRect();
    const tw = tip.offsetWidth, th = tip.offsetHeight, vw = document.documentElement.clientWidth;
    let left = r.left + r.width / 2 - tw / 2;
    left = Math.max(8, Math.min(left, vw - tw - 8));
    let top = r.top - th - 8;
    if (top < 8) top = r.bottom + 8;
    tip.style.left = left + window.scrollX + 'px';
    tip.style.top = top + window.scrollY + 'px';
  }
  function hide() {
    if (tip) tip.hidden = true;
    if (shown) { shown.classList.remove('is-open'); shown.removeAttribute('aria-describedby'); }
    shown = null; pinned = null;
  }
  function wireTooltips() {
    const target = (e) => (e.target && e.target.closest ? e.target.closest('[data-tip]') : null);
    document.addEventListener('pointerover', (e) => {
      if (e.pointerType !== 'mouse' || pinned) return;
      const el = target(e); if (el) show(el);
    });
    document.addEventListener('pointerout', (e) => {
      if (e.pointerType !== 'mouse' || pinned) return;
      const el = target(e);
      if (el && !(e.relatedTarget && el.contains(e.relatedTarget))) hide();
    });
    document.addEventListener('click', (e) => {
      const el = target(e);
      if (!el) { if (pinned) hide(); return; }
      if (pinned === el) { hide(); return; }
      show(el); pinned = el;
    });
    document.addEventListener('focusin', (e) => { const el = target(e); if (el && !pinned) show(el); });
    document.addEventListener('focusout', (e) => { if (!pinned && target(e)) hide(); });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') hide();
      if ((e.key === 'Enter' || e.key === ' ') && e.target.closest && e.target.closest('[data-tip]') === e.target && !/^(BUTTON|INPUT|A)$/.test(e.target.tagName)) {
        e.preventDefault(); if (pinned === e.target) hide(); else { show(e.target); pinned = e.target; }
      }
    });
    window.addEventListener('resize', hide);
  }

  // ───────────────────────────── helpers ─────────────────────────────
  const M = '−';
  const fmt = (x, d = 0) => {
    if (!isFinite(x)) return String(x);
    const s = Math.abs(x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    return (x < 0 && Number(s.replace(/,/g, '')) !== 0 ? M : '') + s;
  };
  const sgn = (w) => (w > 0 ? '+1' : w < 0 ? M + '1' : '0');
  const par = (x) => (x < 0 ? '(' + fmt(x) + ')' : fmt(x));
  const sym = (k, html) => `<var data-k="${k}">${html}</var>`;
  const h = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; };
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const NS = 'http://www.w3.org/2000/svg';
  function svgEl(tag, attrs = {}, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function slider(label, min, max, step, value, fmtFn) {
    const w = h(`<label class="bx-slider">${label}<input type="range" min="${min}" max="${max}" step="${step}" value="${value}"><output></output></label>`);
    const inp = w.querySelector('input'), out = w.querySelector('output');
    const sync = () => { out.textContent = fmtFn(Number(inp.value)); };
    inp.addEventListener('input', sync); sync();
    return { el: w, input: inp, get value() { return Number(inp.value); }, set value(v) { inp.value = v; sync(); } };
  }
  // A row of bit boxes, most significant first. classes(i) → extra class names for bit i.
  function bitRow(label, bits, classes, tipFn, total = bits.length) {
    const row = h(`<div class="bx-bitrow"><b>${label}</b><div class="bx-bits"></div></div>`);
    const box = row.querySelector('.bx-bits');
    for (let i = bits.length; i < total; i++) box.appendChild(h('<span class="pad"></span>'));
    for (let i = bits.length - 1; i >= 0; i--) {
      const c = classes(i);
      if (c === 'point') { box.appendChild(h('<span class="point">.</span>')); continue; }
      const s = document.createElement('span');
      s.textContent = bits[i]; s.className = (bits[i] === '1' ? 'one ' : '') + (c || '');
      if (tipFn) { s.setAttribute('data-tip', tipFn(i)); }
      box.appendChild(s);
    }
    return row;
  }
  const toBits = (big, n) => { // two's complement, bit i at index i
    const mod = 1n << BigInt(n); let v = ((big % mod) + mod) % mod; const out = [];
    for (let i = 0; i < n; i++) { out.push((v & 1n) ? '1' : '0'); v >>= 1n; }
    return out;
  };

  // ───────────────────────────── widgets ─────────────────────────────
  const WIDGETS = {};

  // 4 — the 81-entry ternary lookup table
  WIDGETS.lut = (fig) => {
    const q = [5, -2, 7, 1], w = [1, 0, -1, 1];
    const box = h('<div></div>');
    box.innerHTML = `
      <div class="bx-row"><span class="bx-label">Inputs of group ${sym('g', 'g')}</span>
        ${q.map((v, k) => `<label>${sym('q_' + k, `q<sub>${k}</sub>`)} = <input type="number" min="-127" max="127" step="1" value="${v}" data-q="${k}" aria-label="q${k}"></label>`).join(' ')}</div>
      <div class="bx-row bx-weights"><span class="bx-label">One row’s weights <small>(click to change)</small></span>
        ${w.map((v, k) => `<span class="bx-weightcol"><button type="button" data-w="${k}" aria-label="weight ${k}"></button><small>${sym('d_' + k, `d<sub>${k}</sub>`)} = <span data-d="${k}"></span></small></span>`).join(' ')}</div>
      <div class="bx-out" data-code></div>
      <div class="bx-lut-wrap"><table class="bx-lut"></table></div>
      <div class="bx-out" data-result></div>
      <p class="bx-note">Table cost per group: 9 ${sym('lo', 'lo')} + 9 ${sym('hi', 'hi')} partial sums + 81 additions, instead of 81 × 4 = 324 multiply-adds. Then 1,616 rows each need one lookup per group instead of four multiply-adds.</p>`;
    const pre = fig.querySelector('pre'); if (pre) pre.remove();
    fig.appendChild(box);
    const table = box.querySelector('table');
    const dig = (c) => [c % 3, ((c / 3) | 0) % 3, ((c / 9) | 0) % 3, ((c / 27) | 0) % 3];
    const pat = (a, b) => `${sgn(a - 1)}, ${sgn(b - 1)}`;

    function render() {
      const d = w.map((x) => x + 1);
      const c = d[0] + 3 * d[1] + 9 * d[2] + 27 * d[3];
      const lo = [], hi = [];
      for (let i = 0; i < 9; i++) lo.push((i % 3 - 1) * q[0] + (((i / 3) | 0) - 1) * q[1]);
      for (let j = 0; j < 9; j++) hi.push((j % 3 - 1) * q[2] + (((j / 3) | 0) - 1) * q[3]);
      const maxAbs = Math.max(1, ...lo.map((a) => Math.max(...hi.map((b) => Math.abs(a + b)))));
      box.querySelectorAll('[data-w]').forEach((b) => { const k = +b.dataset.w; b.textContent = sgn(w[k]); b.className = 'w' + w[k]; });
      box.querySelectorAll('[data-d]').forEach((s) => { s.textContent = d[+s.dataset.d]; });
      box.querySelector('[data-code]').innerHTML =
        `${sym('c', 'c')} = ${sym('d_0', 'd<sub>0</sub>')} + 3·${sym('d_1', 'd<sub>1</sub>')} + 9·${sym('d_2', 'd<sub>2</sub>')} + 27·${sym('d_3', 'd<sub>3</sub>')}` +
        ` = ${d[0]} + 3·${d[1]} + 9·${d[2]} + 27·${d[3]} = <b>${c}</b> &nbsp;→ one byte` +
        `<br>low half ${sym('i', 'i')} = ${d[0]} + 3·${d[1]} = ${c % 9}, high half ${sym('j', 'j')} = ${d[2]} + 3·${d[3]} = ${(c / 9) | 0}, and ${c} = ${c % 9} + 9·${(c / 9) | 0}`;
      const si = c % 9, sj = (c / 9) | 0;
      let html = `<tr><th class="corner">hi[j] ↓<br>lo[i] →</th>`;
      for (let i = 0; i < 9; i++) html += `<th class="${i === si ? 'hl' : ''}" title="lo[${i}]: weights of q0, q1 = ${pat(i % 3, (i / 3) | 0)}">${fmt(lo[i])}<small>i=${i}</small></th>`;
      html += '</tr>';
      for (let j = 0; j < 9; j++) {
        html += `<tr><th class="${j === sj ? 'hl' : ''}" title="hi[${j}]: weights of q2, q3 = ${pat(j % 3, (j / 3) | 0)}">${fmt(hi[j])}<small>j=${j}</small></th>`;
        for (let i = 0; i < 9; i++) {
          const v = lo[i] + hi[j], cc = i + 9 * j, a = Math.abs(v) / maxAbs;
          const col = v < 0 ? 'var(--bx-neg)' : 'var(--bx-pos)';
          const wd = dig(cc).map((x) => sgn(x - 1)).join(' ');
          html += `<td data-c="${cc}" class="${cc === c ? 'sel' : ''}" style="background:color-mix(in srgb, ${col} ${Math.round(10 + a * 45)}%, transparent)" title="T[${cc}] = ${fmt(v)}, weights ${wd}" aria-label="T[${cc}] = ${fmt(v)}"><small>${cc}</small>${fmt(v)}</td>`;
        }
        html += '</tr>';
      }
      table.innerHTML = html;
      const T = lo[si] + hi[sj];
      const direct = w.reduce((s, wk, k) => s + wk * q[k], 0);
      box.querySelector('[data-result]').innerHTML =
        `${sym('T[c]', 'T[' + c + ']')} = lo[${si}] + hi[${sj}] = ${par(lo[si])} + ${par(hi[sj])} = <b>${fmt(T)}</b>` +
        `<br>check: ${q.map((v, k) => `${par(v)}·${par(w[k])}`).join(' + ')} = ${fmt(direct)} <span class="${T === direct ? 'ok' : 'bad'}">${T === direct ? '✓' : '✗'}</span>`;
      annotate(box);
    }
    box.addEventListener('click', (e) => {
      const b = e.target.closest('[data-w]');
      if (b) { const k = +b.dataset.w; w[k] = w[k] === 1 ? -1 : w[k] + 1; render(); return; }
      const td = e.target.closest('td[data-c]');
      if (td) { dig(+td.dataset.c).forEach((x, k) => { w[k] = x - 1; }); render(); }
    });
    box.addEventListener('input', (e) => {
      const k = e.target.dataset.q; if (k == null) return;
      const v = Math.round(Number(e.target.value)); if (!isFinite(v)) return;
      q[+k] = clamp(v, -127, 127); render();
    });
    render();
  };

  // 6 — repacking the weights for the cache
  WIDGETS.repack = (fig) => {
    const R = 8, G = 12;
    let seed = 7; const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    const code = Array.from({ length: R }, () => Array.from({ length: G }, () => Math.floor(rnd() * 81)));
    let mode = 'repacked', gbDemo = 4, step = -1, timer = null;
    const box = h('<div></div>');
    box.innerHTML = `
      <div class="bx-row"><span class="bx-label">Memory order</span>
        <button type="button" data-mode="original">Original: row by row, 1 byte per load</button>
        <button type="button" data-mode="repacked">Repacked: blocks, 4 rows per word</button></div>
      <div class="bx-row" data-gbrow><span class="bx-label">Demo block size</span>
        ${[3, 4, 6, 12].map((g) => `<button type="button" data-gb="${g}">${g} groups</button>`).join('')}</div>
      <div class="bx-row"><button type="button" data-act="back" aria-label="Step back">◀</button>
        <button type="button" data-act="play">▶ Play</button>
        <button type="button" data-act="fwd" aria-label="Step forward">▶|</button>
        <button type="button" data-act="reset">Reset</button>
        <span class="bx-mono" data-count></span></div>
      <div data-matrix></div>
      <div class="bx-note">Weight bytes in memory order (each box is one load):</div>
      <div class="bx-strip" data-strip></div>
      <div class="bx-out" data-code></div>
      <hr style="border:0;border-top:1px solid var(--bx-rule);margin:1rem 0">
      <div class="bx-row"><span class="bx-label">Real input matrix</span><span>1,616 rows × 96 groups, tables of 81 × 2 bytes = 162 B</span></div>
      <div class="bx-row" data-real></div>
      <div class="bx-row"><span class="bx-label">Tables in cache</span><div class="bx-meter" data-meter-orig></div></div>
      <div class="bx-row"><span class="bx-label"></span><div class="bx-meter" data-meter-rep></div></div>
      <p class="bx-note" data-realnote></p>`;
    fig.appendChild(box);
    const gbSl = slider(`block size ${sym('GB', 'GB')} = `, 0, 5, 1, 3, (i) => [8, 16, 24, 32, 48, 96][i] + ' groups');
    box.querySelector('[data-real]').appendChild(gbSl.el);

    function loads() {
      const L = [];
      if (mode === 'original') {
        for (let r = 0; r < R; r++) for (let g = 0; g < G; g++) L.push({ cells: [[r, g]], g, live: [0, G - 1] });
      } else {
        for (let b0 = 0; b0 < G; b0 += gbDemo) for (let rq = 0; rq < R; rq += 4) for (let g = b0; g < b0 + gbDemo; g++)
          L.push({ cells: [0, 1, 2, 3].map((k) => [rq + k, g]), g, live: [b0, b0 + gbDemo - 1], rq });
      }
      return L;
    }
    function render() {
      const L = loads(); step = clamp(step, -1, L.length - 1);
      box.querySelectorAll('[data-mode]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
      box.querySelectorAll('[data-gb]').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.gb === gbDemo)));
      box.querySelector('[data-gbrow]').style.display = mode === 'repacked' ? '' : 'none';
      const done = new Set(), now = new Set();
      L.forEach((ld, k) => { if (k <= step) ld.cells.forEach(([r, g]) => (k === step ? now : done).add(r + ',' + g)); });
      const cur = L[step];
      const live = cur ? cur.live : null;
      const mx = box.querySelector('[data-matrix]');
      const blockStart = (g) => mode === 'repacked' && g > 0 && g % gbDemo === 0;
      let html = `<div class="bx-matrix" style="grid-template-columns:3.2rem repeat(${G}, minmax(0,1fr))"><div class="hdr">table</div>`;
      for (let g = 0; g < G; g++) {
        const isLive = live && g >= live[0] && g <= live[1];
        html += `<div class="tbl ${isLive ? 'live' : ''} ${cur && cur.g === g ? 'now' : ''} ${blockStart(g) ? 'blk-start' : ''}">T<sub>${g}</sub></div>`;
      }
      for (let r = 0; r < R; r++) {
        const qs = mode === 'repacked' && r > 0 && r % 4 === 0 ? 'quad-start' : '';
        html += `<div class="hdr ${qs}">row ${r}</div>`;
        for (let g = 0; g < G; g++) {
          const key = r + ',' + g;
          html += `<div class="cell ${now.has(key) ? 'now' : done.has(key) ? 'done' : ''} ${blockStart(g) ? 'blk-start' : ''} ${qs}">${code[r][g]}</div>`;
        }
      }
      mx.innerHTML = html + '</div>';
      const st = box.querySelector('[data-strip]');
      st.innerHTML = L.map((ld, k) => {
        const cls = k === step ? 'now' : k < step ? 'done' : '';
        return mode === 'original' ? `<span class="byte ${cls}"></span>` : `<span class="word ${cls}"><span></span><span></span><span></span><span></span></span>`;
      }).join('');
      box.querySelector('[data-count]').textContent = `load ${step + 1} of ${L.length}`;
      const out = box.querySelector('[data-code]');
      if (!cur) out.innerHTML = 'Press play or step to start.';
      else if (mode === 'original') {
        const [r, g] = cur.cells[0];
        out.innerHTML = `load 1 byte: row ${r}, group ${g}, code ${code[r][g]}<br>s += T<sub>${g}</sub>[${code[r][g]}] &nbsp; (this row needs all ${G} tables before the next row starts over)`;
      } else {
        const cs = cur.cells.map(([r, g]) => code[r][g]);
        const v = cs.reduce((a, c, k) => a + c * 2 ** (8 * k), 0);
        out.innerHTML = `load 1 word ${sym('word', 'v')} = 0x${v.toString(16).padStart(8, '0')}: group ${cur.g}, rows ${cur.rq}–${cur.rq + 3}<br>` +
          cs.map((c, k) => `${sym('s0', 's' + k)} += T<sub>${cur.g}</sub>[${k === 0 ? 'v &amp; 0xff' : k === 3 ? 'v &gt;&gt; 24' : `(v &gt;&gt; ${8 * k}) &amp; 0xff`}] = T<sub>${cur.g}</sub>[${c}]`).join('<br>') +
          `<br><span style="opacity:.7">(the engine stores 2c, the byte offset, so the lookup is one ldrsh)</span>`;
      }
      // real-matrix numbers
      const GB = [8, 16, 24, 32, 48, 96][gbSl.value];
      const cache = 16384, orig = 96 * 162, rep = GB * 162;
      const meter = (el, bytes, label) => {
        const pct = Math.min(100, (bytes / cache) * 100);
        el.innerHTML = `<div class="${bytes > cache * 0.6 ? 'over' : ''}" style="width:${pct}%"></div><span>${label}: ${fmt(bytes)} of 16,384 bytes</span>`;
      };
      meter(box.querySelector('[data-meter-orig]'), orig, 'original, 96 tables');
      meter(box.querySelector('[data-meter-rep]'), rep, `repacked, ${GB} tables`);
      box.querySelector('[data-realnote]').innerHTML =
        `Loads for one pass over the real matrix: 1,616 × 96 = 155,136 byte loads originally, 155,136 / 4 = 38,784 word loads repacked, in ${Math.ceil(96 / GB)} block${GB < 96 ? 's' : ''}. ` +
        (GB === 96 ? 'One block of 96 groups is the original problem again: the tables fill the cache.' : 'The rest of the cache is left for the streaming weights, so the tables are not pushed out.');
      annotate(box);
    }
    function stop() { clearInterval(timer); timer = null; box.querySelector('[data-act=play]').textContent = '▶ Play'; }
    box.addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      if (b.dataset.mode) { mode = b.dataset.mode; step = -1; stop(); }
      else if (b.dataset.gb) { gbDemo = +b.dataset.gb; step = -1; stop(); }
      else if (b.dataset.act === 'back') { stop(); step--; }
      else if (b.dataset.act === 'fwd') { stop(); step++; }
      else if (b.dataset.act === 'reset') { stop(); step = -1; }
      else if (b.dataset.act === 'play') {
        if (timer) { stop(); } else {
          if (step >= loads().length - 1) step = -1;
          b.textContent = '❚❚ Pause';
          timer = setInterval(() => { step++; if (step >= loads().length - 1) stop(); render(); }, mode === 'original' ? 120 : 380);
        }
      }
      render();
    });
    gbSl.input.addEventListener('input', render);
    render();
  };

  // 2 — the SSM recurrence for one state number
  WIDGETS.ssm = (fig) => {
    const T = 32;
    const x = new Array(T).fill(0); x[3] = 1; x[16] = 1; x[17] = -1;
    let hover = null;
    const softplus = (v) => Math.log1p(Math.exp(v));
    const box = h('<div></div>');
    box.innerHTML = `
      <div class="bx-row" data-sliders></div>
      <div class="bx-row"><span class="bx-label">Inputs</span>
        <button type="button" data-preset="pulse">one pulse</button>
        <button type="button" data-preset="steady">steady input</button>
        <button type="button" data-preset="mix">pulses</button>
        <button type="button" data-preset="clear">clear</button></div>
      <svg class="bx-svg" viewBox="0 0 640 235" role="img" aria-label="Input pulses and the state over 32 tokens"></svg>
      <div class="bx-legend"><span><i style="background:var(--bx-pos)"></i>inputs x (top) and state S</span><span><i style="background:var(--accent)"></i>output y (dashed)</span><span>token t runs left to right · click or tap the top row to place inputs</span></div>
      <div class="bx-out" data-read></div>
      <div class="bx-out" data-step></div>`;
    fig.appendChild(box);
    const sl = box.querySelector('[data-sliders]');
    const sDt = slider(`${sym('dt', 'dt')}`, -4, 4, 0.05, 0, (v) => fmt(v, 2));
    const sA = slider(`${sym('A', 'A')}`, -3, -0.05, 0.05, -0.5, (v) => fmt(v, 2));
    const sB = slider(`${sym('B', 'B')}`, -2, 2, 0.05, 1, (v) => fmt(v, 2));
    const sC = slider(`${sym('C', 'C')}`, -2, 2, 0.05, 1, (v) => fmt(v, 2));
    [sDt, sA, sB, sC].forEach((s) => { sl.appendChild(s.el); s.input.addEventListener('input', render); });
    const svg = box.querySelector('svg');
    const X0 = 40, X1 = 630, colW = (X1 - X0) / T;
    const inTop = 10, inH = 50, plTop = 80, plH = 150;

    function compute() {
      const D = softplus(sDt.value), Ab = Math.exp(D * sA.value), B = sB.value, C = sC.value;
      const S = []; let s = 0;
      for (let t = 0; t < T; t++) { s = Ab * s + D * x[t] * B; S.push(s); }
      return { D, Ab, B, C, S, y: S.map((v) => v * C) };
    }
    function render() {
      const m = compute();
      svg.innerHTML = '';
      svgEl('text', { x: 2, y: inTop + inH / 2 + 4 }, svg).textContent = 'x';
      
      svgEl('line', { x1: X0, x2: X1, y1: inTop + inH / 2, y2: inTop + inH / 2, class: 'axis' }, svg);
      const lim = Math.max(0.5, ...m.S.map(Math.abs), ...m.y.map(Math.abs)) * 1.1;
      const Y = (v) => plTop + plH / 2 - (v / lim) * (plH / 2);
      svgEl('line', { x1: X0, x2: X1, y1: Y(0), y2: Y(0), class: 'axis' }, svg);
      svgEl('text', { x: X0 - 4, y: Y(lim * 0.9) + 4, 'text-anchor': 'end' }, svg).textContent = fmt(lim * 0.9, 1);
      svgEl('text', { x: X0 - 4, y: Y(-lim * 0.9) + 4, 'text-anchor': 'end' }, svg).textContent = fmt(-lim * 0.9, 1);
      for (let t = 0; t < T; t++) {
        const cx = X0 + t * colW;
        if (hover === t) svgEl('rect', { x: cx, y: inTop - 4, width: colW, height: plTop + plH - inTop + 8, fill: 'var(--bx-hl)', opacity: 0.18 }, svg);
        const v = x[t];
        if (v) {
          const hh = (Math.abs(v) * inH) / 2 - 2;
          svgEl('rect', { x: cx + 3, y: v > 0 ? inTop + inH / 2 - hh : inTop + inH / 2, width: colW - 6, height: hh, rx: 2, fill: v > 0 ? 'var(--bx-pos)' : 'var(--bx-neg)' }, svg);
        } else svgEl('circle', { cx: cx + colW / 2, cy: inTop + inH / 2, r: 2, fill: 'var(--muted)' }, svg);
      }
      const path = (arr) => arr.map((v, t) => `${t ? 'L' : 'M'}${X0 + t * colW + colW / 2},${Y(v)}`).join('');
      svgEl('path', { d: path(m.y), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2, 'stroke-dasharray': '5 4' }, svg);
      svgEl('path', { d: path(m.S), fill: 'none', stroke: 'var(--bx-pos)', 'stroke-width': 2.5 }, svg);
      m.S.forEach((v, t) => svgEl('circle', { cx: X0 + t * colW + colW / 2, cy: Y(v), r: 2.5, fill: 'var(--bx-pos)' }, svg));
      for (let t = 0; t < T; t++) {
        const hit = svgEl('rect', { x: X0 + t * colW, y: 0, width: colW, height: plTop + plH, fill: 'transparent', 'data-t': t, style: 'cursor:pointer' }, svg);
        hit.setAttribute('aria-label', 'token ' + t);
      }
      const half = m.Ab > 0 && m.Ab < 1 ? Math.log(0.5) / Math.log(m.Ab) : Infinity;
      box.querySelector('[data-read]').innerHTML =
        `${sym('Δ', 'Δ')} = softplus(${fmt(sDt.value, 2)}) = ${fmt(m.D, 3)} &nbsp; ` +
        `${sym('Abar', 'Ā')} = e<sup>Δ·A</sup> = e<sup>${fmt(m.D * sA.value, 3)}</sup> = ${fmt(m.Ab, 3)} &nbsp; ` +
        `${sym('half', 'half-life')} ≈ ${isFinite(half) ? fmt(half, 1) + ' tokens' : '∞'}`;
      const t = hover == null ? 17 : hover;
      const prev = t ? m.S[t - 1] : 0;
      box.querySelector('[data-step]').innerHTML =
        `token ${t}: ${sym('S_t', 'S<sub>t</sub>')} = ${sym('Abar', 'Ā')}·${sym('S_t', 'S<sub>t−1</sub>')} + ${sym('Δ', 'Δ')}·${sym('x_t', 'x<sub>t</sub>')}·${sym('B', 'B')}` +
        ` = ${fmt(m.Ab, 3)}·${par(+prev.toFixed(3))} + ${fmt(m.D, 3)}·${par(x[t])}·${par(m.B)} = <b>${fmt(m.S[t], 3)}</b>` +
        `<br>${sym('y_t', 'y<sub>t</sub>')} = ${sym('S_t', 'S<sub>t</sub>')}·${sym('C', 'C')} = ${fmt(m.S[t], 3)}·${par(m.C)} = ${fmt(m.y[t], 3)}` +
        ` &nbsp;<span style="opacity:.7">(the D·x skip term is left out here)</span>`;
      annotate(box);
    }
    svg.addEventListener('pointermove', (e) => { const t = e.target.getAttribute && e.target.getAttribute('data-t'); if (t != null && +t !== hover) { hover = +t; render(); } });
    svg.addEventListener('pointerleave', () => { hover = null; render(); });
    svg.addEventListener('click', (e) => {
      const t = e.target.getAttribute && e.target.getAttribute('data-t'); if (t == null) return;
      x[+t] = x[+t] === 0 ? 1 : x[+t] === 1 ? -1 : 0; hover = +t; render();
    });
    box.addEventListener('click', (e) => {
      const p = e.target.closest('[data-preset]'); if (!p) return;
      x.fill(0);
      if (p.dataset.preset === 'pulse') x[3] = 1;
      if (p.dataset.preset === 'steady') for (let t = 4; t < 24; t++) x[t] = 1;
      if (p.dataset.preset === 'mix') { x[3] = 1; x[16] = 1; x[17] = -1; }
      render();
    });
    render();
  };

  // 3 / 8.5 — absmax int8 quantisation
  WIDGETS.absmax = (fig) => {
    const base0 = [0.42, -0.18, 0.07, -0.63, 0.31, 0.12, -0.05, 0.88, -0.27, 0.15, -0.41, 0.22];
    let base = base0.slice(), drag = null, range = 1;
    const box = h('<div></div>');
    box.innerHTML = `
      <div class="bx-row"><button type="button" data-act="outlier">Add an outlier</button>
        <button type="button" data-act="random">Random values</button>
        <button type="button" data-act="reset">Reset</button></div>
      <div class="bx-row" data-lam></div>
      <svg class="bx-svg" viewBox="0 0 640 236" role="img" aria-label="Twelve values, their int8 codes, and the values recovered from them"></svg>
      <div class="bx-legend"><span><i style="background:var(--bx-pos)"></i>real values x<sub>i</sub></span><span><i style="border:1.5px dashed var(--text)"></i>q<sub>i</sub> × s, what comes back</span><span>numbers underneath: the stored int8 q<sub>i</sub> · drag a bar to change it</span></div>
      <div class="bx-out" data-read></div>`;
    fig.appendChild(box);
    const lam = slider(`multiply every value by ${sym('λ', 'λ')}`, -2, 2, 0.01, 0, (v) => '× ' + fmt(2 ** v, 2));
    box.querySelector('[data-lam]').appendChild(lam.el);
    const svg = box.querySelector('svg');
    const n = base.length, X0 = 46, X1 = 630, cw = (X1 - X0) / n, top = 14, H = 190, mid = top + H / 2;
    function render() {
      const L = 2 ** lam.value;
      const xs = base.map((b) => b * L);
      const mx = Math.max(...xs.map(Math.abs)) || 1;
      const s = mx / 127;
      const q = xs.map((v) => Math.round(v / s));
      if (!drag) range = mx * 1.1;
      const Y = (v) => mid - (v / range) * (H / 2);
      svg.innerHTML = '';
      svgEl('line', { x1: X0, x2: X1, y1: mid, y2: mid, class: 'axis' }, svg);
      for (const sgn1 of [1, -1]) {
        svgEl('line', { x1: X0, x2: X1, y1: Y(sgn1 * mx), y2: Y(sgn1 * mx), stroke: 'var(--accent)', 'stroke-dasharray': '3 4', opacity: 0.7 }, svg);
        svgEl('text', { x: X0 - 4, y: Y(sgn1 * mx) + 4, 'text-anchor': 'end' }, svg).textContent = (sgn1 > 0 ? '+' : M) + '127';
      }
      svgEl('text', { x: X0 - 4, y: mid + 4, 'text-anchor': 'end' }, svg).textContent = '0';
      xs.forEach((v, i) => {
        const cx = X0 + i * cw;
        const y0 = Y(Math.max(v, 0)), hgt = Math.abs(Y(v) - Y(0));
        svgEl('rect', { x: cx + 6, y: y0, width: cw - 12, height: Math.max(hgt, 0.5), rx: 2, fill: v < 0 ? 'var(--bx-neg)' : 'var(--bx-pos)', opacity: 0.75 }, svg);
        const rv = q[i] * s, ry0 = Y(Math.max(rv, 0)), rh = Math.abs(Y(rv) - Y(0));
        svgEl('rect', { x: cx + 3, y: ry0, width: cw - 6, height: Math.max(rh, 0.5), fill: 'none', stroke: 'var(--text)', 'stroke-width': 1.2, 'stroke-dasharray': '2 2' }, svg);
        const t = svgEl('text', { x: cx + cw / 2, y: top + H + 22, 'text-anchor': 'middle', style: 'font-family:var(--bx-mono);fill:var(--text)' }, svg);
        t.textContent = fmt(q[i]);
        svgEl('rect', { x: cx, y: 0, width: cw, height: top + H + 8, fill: 'transparent', 'data-i': i, style: 'cursor:ns-resize' }, svg);
      });
      svgEl('text', { x: X0 - 4, y: top + H + 22, 'text-anchor': 'end' }, svg).textContent = 'q';
      const err = Math.max(...xs.map((v, i) => Math.abs(v - q[i] * s)));
      const zeros = q.filter((v) => v === 0).length;
      box.querySelector('[data-read]').innerHTML =
        `max |${sym('x_i', 'x<sub>i</sub>')}| = ${fmt(mx, 3)}, &nbsp;${sym('s', 's')} = ${fmt(mx, 3)} / 127 = ${s.toPrecision(3)}` +
        `<br>${sym('q_i', 'q<sub>i</sub>')} = round(${sym('x_i', 'x<sub>i</sub>')} / ${sym('s', 's')}): [${q.map((v) => fmt(v)).join(', ')}]` +
        `<br>largest rounding error ${err.toPrecision(2)} (at most s / 2 = ${(s / 2).toPrecision(2)})` +
        (zeros ? `, <span class="bad">${zeros} value${zeros > 1 ? 's' : ''} rounded to 0</span>` : '');
      annotate(box);
    }
    const valAt = (e) => {
      const r = svg.getBoundingClientRect(); const y = ((e.clientY - r.top) / r.height) * 236;
      return clamp(((mid - y) / (H / 2)) * range, -range, range) / 2 ** lam.value;
    };
    svg.addEventListener('pointerdown', (e) => {
      const i = e.target.getAttribute && e.target.getAttribute('data-i'); if (i == null) return;
      drag = +i; svg.setPointerCapture(e.pointerId); base[drag] = valAt(e); render();
    });
    svg.addEventListener('pointermove', (e) => { if (drag == null) return; base[drag] = valAt(e); render(); });
    const end = () => { if (drag != null) { drag = null; render(); } };
    svg.addEventListener('pointerup', end); svg.addEventListener('pointercancel', end);
    box.addEventListener('click', (e) => {
      const b = e.target.closest('[data-act]'); if (!b) return;
      if (b.dataset.act === 'outlier') { const mx = Math.max(...base.map(Math.abs)); base[5] = mx * 20; }
      if (b.dataset.act === 'random') base = base.map(() => +(Math.random() * 2 - 1).toFixed(2));
      if (b.dataset.act === 'reset') { base = base0.slice(); lam.value = 0; }
      render();
    });
    lam.input.addEventListener('input', render);
    render();
  };

  // 8.1 — one number in Qf
  WIDGETS.qformat = (fig) => {
    let v = 1.5, f = 16, v2 = -0.75;
    const box = h('<div></div>');
    box.innerHTML = `
      <div class="bx-row"><label>${sym('vreal', 'v')} = <input type="text" inputmode="decimal" value="1.5" data-v></label>
        <span data-fs></span>
        <button type="button" data-f="16">Q16</button><button type="button" data-f="20">Q20</button><button type="button" data-f="31">Q31</button></div>
      <div data-bits></div>
      <div class="bx-legend"><span><i style="background:var(--bx-neg)"></i>sign bit (worth −2<sup>31−f</sup>)</span><span><i style="background:color-mix(in srgb,var(--muted) 30%,transparent)"></i>whole-number bits</span><span><i style="background:var(--bx-hl)"></i>fractional bits</span><span>hover a bit to see its worth</span></div>
      <div class="bx-out" data-read></div>
      <div class="bx-row" style="margin-top:.8rem"><span class="bx-label">Multiply by</span><label>${sym('vreal', 'v')}<sub>2</sub> = <input type="text" inputmode="decimal" value="-0.75" data-v2></label></div>
      <div class="bx-out" data-mul></div>`;
    fig.appendChild(box);
    const fs = slider(`${sym('f', 'f')} =`, 0, 31, 1, f, (x) => 'Q' + x);
    box.querySelector('[data-fs]').appendChild(fs.el);
    const parse = (s) => Number(String(s).replace(M, '-').replace(/,/g, ''));
    function render() {
      f = fs.value;
      box.querySelectorAll('[data-f]').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.f === f)));
      const scale = 2 ** f;
      const raw = Math.round(v * scale);
      const lo = -(2 ** 31), hi = 2 ** 31 - 1;
      const over = raw < lo || raw > hi;
      const n = clamp(raw, lo, hi);
      const bits = toBits(BigInt(n), 32);
      const wrap = box.querySelector('[data-bits]'); wrap.innerHTML = '';
      const withPoint = [];
      // build a pseudo bit list with a point marker inserted after bit f
      const row = h('<div class="bx-bitrow"><b>stored n</b><div class="bx-bits"></div></div>');
      const bb = row.querySelector('.bx-bits');
      for (let i = 31; i >= 0; i--) {
        const s = document.createElement('span');
        s.textContent = bits[i];
        s.className = (bits[i] === '1' ? 'one ' : '') + (i === 31 ? 'sign' : i < f ? 'frac' : '');
        const worth = i === 31 ? -(2 ** (31 - f)) : 2 ** (i - f);
        s.setAttribute('data-tip', `<b>bit ${i}</b>: worth ${i === 31 ? '−' : ''}2<sup>${i === 31 ? 31 - f : i - f}</sup> = ${fmt(worth, Math.max(0, Math.min(10, f - i)))}${i === 31 ? ' (the sign bit of two’s complement)' : ''}`);
        bb.appendChild(s);
        if (i === f && f > 0) bb.appendChild(h('<span class="point" title="binary point">.</span>'));
      }
      void withPoint;
      wrap.appendChild(row);
      const back = n / scale;
      box.querySelector('[data-read]').innerHTML =
        `${sym('nint', 'n')} = round(${fmt(v, 6)} × 2<sup>${f}</sup>) = <b>${fmt(n)}</b>${over ? ' <span class="bad">(overflow: v is outside the range, clamped)</span>' : ''}` +
        `<br>back: ${fmt(n)} / 2<sup>${f}</sup> = ${back.toPrecision(10).replace(/\.?0+$/, '')} &nbsp;error ${(Math.abs(back - v)).toPrecision(2)}` +
        `<br>${sym('step', 'step')} 2<sup>−${f}</sup> = ${(2 ** -f).toPrecision(4)}, &nbsp;range ±${fmt(2 ** (31 - f), f > 21 ? 4 : 0)}`;
      const n2 = clamp(Math.round(v2 * scale), lo, hi);
      const prod = BigInt(n) * BigInt(n2);
      const back16 = prod >> BigInt(f);
      box.querySelector('[data-mul]').innerHTML =
        `${fmt(n)} × ${par(n2)} = ${fmt(Number(prod))} &nbsp;in Q${2 * f} (Q${f} × Q${f})` +
        `<br>≫ ${f} → ${fmt(Number(back16))} in Q${f} = ${(Number(back16) / scale).toPrecision(8).replace(/\.?0+$/, '')} &nbsp;(exact: ${(v * v2).toPrecision(8).replace(/\.?0+$/, '')})`;
      annotate(box);
    }
    box.addEventListener('input', (e) => {
      if (e.target.matches('[data-v]')) { const x = parse(e.target.value); if (isFinite(x)) v = x; }
      if (e.target.matches('[data-v2]')) { const x = parse(e.target.value); if (isFinite(x)) v2 = x; }
      render();
    });
    box.addEventListener('click', (e) => { const b = e.target.closest('[data-f]'); if (b) { fs.value = +b.dataset.f; render(); } });
    render();
  };

  // 8.4 — SMULWB
  WIDGETS.smulwb = (fig) => {
    let dx = 0.37, B = 3.2;
    const box = h('<div></div>');
    box.innerHTML = `
      <div class="bx-row"><label>${sym('Δx', 'Δx')} = <input type="text" inputmode="decimal" value="0.37" data-dx></label>
        <label>${sym('B', 'B')}<sub>n</sub> = <input type="text" inputmode="decimal" value="3.2" data-b></label></div>
      <div class="bx-out" data-prep></div>
      <div data-bits style="margin-top:.5rem"></div>
      <div class="bx-legend"><span><i style="background:color-mix(in srgb,var(--bx-pos) 40%,transparent)"></i>1 bits</span><span><i style="background:var(--bx-hl)"></i>bits that are used / kept</span><span style="opacity:.6">faded: ignored or dropped</span></div>
      <div class="bx-out" data-res></div>`;
    fig.appendChild(box);
    const parse = (s) => Number(String(s).replace(M, '-').replace(/,/g, ''));
    function render() {
      const a = Math.round(dx * 2 ** 20);
      const Bq = Math.round(B * 2 ** 16);
      let sb = 0; while ((Bq >> sb) > 32767 || (Bq >> sb) < -32768) sb++;
      const b16 = Bq >> sb;
      const p = BigInt(a) * BigInt(b16);
      const top = p >> 16n;
      const res = Number(top) * 2 ** sb;
      const exact = dx * B;
      box.querySelector('[data-prep]').innerHTML =
        `${sym('Δx', 'Δx')} in Q20: a = round(${fmt(dx, 4)} × 2<sup>20</sup>) = ${fmt(a)}` +
        `<br>${sym('B', 'B')} in Q16 = ${fmt(Bq)}; fits 16 bits after ${sym('s_b', 's<sub>b</sub>')} = ${sb} shift${sb === 1 ? '' : 's'}: ${sym('B_16', 'B<sub>16</sub>')} = ${fmt(Bq)} ≫ ${sb} = ${fmt(b16)}`;
      const wrap = box.querySelector('[data-bits]'); wrap.innerHTML = '';
      const aBits = toBits(BigInt(a), 32);
      wrap.appendChild(bitRow('a = Δx (Q20)', aBits, () => 'frac', (i) => `bit ${i} of a`, 48));
      // b register: bottom 16 bits = B16; top 16 bits are whatever else is in the register
      const bReg = toBits((BigInt(0x5a3c) << 16n) | (BigInt(b16) & 0xffffn), 32);
      wrap.appendChild(bitRow('b register', bReg, (i) => (i >= 16 ? 'off' : 'frac'), (i) => (i >= 16 ? `bit ${i}: top half, ignored by SMULWB (it may hold anything)` : `bit ${i} of B<sub>16</sub>`), 48));
      const pBits = toBits(p, 48);
      wrap.appendChild(bitRow('a × B16 (48 bits)', pBits, (i) => (i < 16 ? 'drop' : 'frac'), (i) => (i < 16 ? `bit ${i}: dropped, worth less than 2<sup>−20</sup> after the shift` : `bit ${i}: kept, becomes bit ${i - 16} of the result`)));
      box.querySelector('[data-res]').innerHTML =
        `${sym('SMULWB', 'SMULWB')}(a, ${sym('B_16', 'B<sub>16</sub>')}) = (${fmt(a)} × ${par(b16)}) ≫ 16 = ${fmt(Number(top))} &nbsp;<span style="opacity:.7">(Q20 × Q16 = Q36, ≫ 16 → Q20, but ${2 ** sb === 1 ? '' : '2<sup>' + sb + '</sup> times '}too small)</span>` +
        `<br>≪ ${sym('s_b', 's<sub>b</sub>')} → ${fmt(res)} in Q20 = <b>${(res / 2 ** 20).toPrecision(7)}</b>` +
        `<br>exact ${sym('Δx', 'Δx')} × ${sym('B', 'B')} = ${exact.toPrecision(7)}, &nbsp;error ${Math.abs(res / 2 ** 20 - exact).toPrecision(2)}`;
      annotate(box);
    }
    box.addEventListener('input', (e) => {
      const x = parse(e.target.value); if (!isFinite(x)) return;
      if (e.target.matches('[data-dx]')) dx = clamp(x, -2000, 2000);
      if (e.target.matches('[data-b]')) B = clamp(x, -30000, 30000);
      render();
    });
    render();
  };

  // 8.3 — the sigmoid table with linear interpolation
  WIDGETS.sigmoid = (fig) => {
    const tbl = Array.from({ length: 1025 }, (_, i) => 1 / (1 + Math.exp(-(-16 + i / 32))));
    const sig = (x) => 1 / (1 + Math.exp(-x));
    const box = h('<div></div>');
    box.innerHTML = `
      <div class="bx-row" data-sl></div>
      <svg class="bx-svg" viewBox="0 0 640 196" role="img" aria-label="The sigmoid curve, the table entries around x, and the interpolated value"></svg>
      <div class="bx-legend"><span>left: the whole curve</span><span>right: zoomed in on two table entries t[i], t[i+1] and the straight line between them</span><span><i style="background:var(--bx-hl)"></i>the interpolated value at x</span></div>
      <div data-bits></div>
      <div class="bx-legend"><span><i style="background:color-mix(in srgb,var(--bx-pos) 25%,transparent)"></i>top bits: table index i</span><span><i style="background:var(--bx-hl)"></i>low 11 bits: frac</span></div>
      <div class="bx-out" data-read></div>`;
    const pre = fig.querySelector('pre'); if (pre) pre.remove();
    fig.appendChild(box);
    const xs = slider(`${sym('x', 'x')} =`, -18, 18, 0.001, 1.3, (v) => fmt(v, 3));
    box.querySelector('[data-sl]').appendChild(xs.el);
    const svg = box.querySelector('svg');
    function render() {
      const x = xs.value, xq = Math.round(x * 65536), u = xq + 16 * 65536;
      let i, fr, val;
      if (u < 0) { i = 0; fr = 0; val = 0; } else if ((u >> 11) >= 1024) { i = 1024; fr = 0; val = 1; } else { i = u >> 11; fr = (u & 2047) / 2048; val = tbl[i] + (tbl[i + 1] - tbl[i]) * fr; }
      svg.innerHTML = '';
      // overview, left
      const ox = 10, ow = 250, oy = 20, oh = 150;
      const OX = (v) => ox + ((v + 18) / 36) * ow, OY = (s) => oy + oh - s * oh;
      svgEl('rect', { x: ox, y: oy, width: ow, height: oh, fill: 'none', class: 'axis' }, svg);
      let d = ''; for (let k = 0; k <= 200; k++) { const v = -18 + (36 * k) / 200; d += `${k ? 'L' : 'M'}${OX(v)},${OY(sig(v))}`; }
      svgEl('path', { d, fill: 'none', stroke: 'var(--bx-pos)', 'stroke-width': 2 }, svg);
      for (const b of [-16, 16]) svgEl('line', { x1: OX(b), x2: OX(b), y1: oy, y2: oy + oh, stroke: 'var(--muted)', 'stroke-dasharray': '2 3' }, svg);
      svgEl('circle', { cx: OX(x), cy: OY(val), r: 4, fill: 'var(--bx-hl)', stroke: 'var(--text)' }, svg);
      svgEl('text', { x: ox, y: oy + oh + 16 }, svg).textContent = '−18';
      svgEl('text', { x: ox + ow, y: oy + oh + 16, 'text-anchor': 'end' }, svg).textContent = '18';
      svgEl('text', { x: OX(-16) + 3, y: oy + 12 }, svg).textContent = 'table: −16 … 16';
      // zoom, right: one table step either side
      const zx = 300, zw = 330, zy = 20, zh = 150;
      const x0 = -16 + Math.max(0, Math.min(1023, i)) / 32 - 0.6 / 32, x1 = x0 + 2.2 / 32;
      const s0 = Math.min(sig(x0), sig(x1)), s1 = Math.max(sig(x0), sig(x1)), pad = Math.max(1e-6, (s1 - s0) * 0.15);
      const ZX = (v) => zx + ((v - x0) / (x1 - x0)) * zw, ZY = (s) => zy + zh - ((s - (s0 - pad)) / (s1 - s0 + 2 * pad)) * zh;
      svgEl('rect', { x: zx, y: zy, width: zw, height: zh, fill: 'none', class: 'axis' }, svg);
      d = ''; for (let k = 0; k <= 80; k++) { const v = x0 + ((x1 - x0) * k) / 80; d += `${k ? 'L' : 'M'}${ZX(v)},${ZY(sig(v))}`; }
      svgEl('path', { d, fill: 'none', stroke: 'var(--bx-pos)', 'stroke-width': 2, opacity: 0.6 }, svg);
      for (let k = -1; k <= 2; k++) {
        const ii = i + k; if (ii < 0 || ii > 1024) continue;
        const tx = -16 + ii / 32; if (tx < x0 - 1e-9 || tx > x1 + 1e-9) continue;
        svgEl('circle', { cx: ZX(tx), cy: ZY(tbl[ii]), r: k === 0 || k === 1 ? 5 : 3.5, fill: 'var(--text)' }, svg);
        svgEl('text', { x: ZX(tx), y: zy + zh + 16, 'text-anchor': 'middle' }, svg).textContent = `t[${ii}]`;
      }
      if (i < 1024 && u >= 0) svgEl('line', { x1: ZX(-16 + i / 32), y1: ZY(tbl[i]), x2: ZX(-16 + (i + 1) / 32), y2: ZY(tbl[i + 1]), stroke: 'var(--accent)', 'stroke-width': 1.5 }, svg);
      if (x >= x0 && x <= x1) svgEl('circle', { cx: ZX(x), cy: ZY(val), r: 5, fill: 'var(--bx-hl)', stroke: 'var(--text)' }, svg);
      // bits of u
      const wrap = box.querySelector('[data-bits]'); wrap.innerHTML = '';
      if (u >= 0 && (u >> 11) < 1024) {
        const ub = toBits(BigInt(u), 22);
        wrap.appendChild(bitRow('u (22 bits)', ub, (k) => (k >= 11 ? 'idx' : 'frac'), (k) => (k >= 11 ? `bit ${k}: part of the index i` : `bit ${k}: part of frac`)));
      }
      const exact = sig(x);
      box.querySelector('[data-read]').innerHTML =
        (u < 0 ? `x &lt; −16: σ is taken as exactly 0` : (u >> 11) >= 1024 ? `x ≥ 16: σ is taken as exactly 1` :
          `${sym('u', 'u')} = ${fmt(xq)} + 16·2<sup>16</sup> = ${fmt(u)}, &nbsp;${sym('i', 'i')} = u ≫ 11 = ${i}, &nbsp;${sym('frac', 'frac')} = ${u & 2047} / 2048 = ${fr.toFixed(4)}` +
          `<br>σ ≈ ${sym('σ_t', 't[' + i + ']')} + (t[${i + 1}] − t[${i}])·frac = ${tbl[i].toFixed(6)} + ${(tbl[i + 1] - tbl[i]).toFixed(6)}·${fr.toFixed(4)} = <b>${val.toFixed(6)}</b>`) +
        `<br>exact σ(${fmt(x, 3)}) = ${exact.toFixed(6)}, &nbsp;error ${Math.abs(val - exact).toExponential(1)} &nbsp;→ SiLU(x) = x·σ = ${fmt(x * val, 4)}`;
      annotate(box);
    }
    xs.input.addEventListener('input', render);
    render();
  };

  // ───────────────────────────── boot ─────────────────────────────
  function boot() {
    const roots = document.querySelectorAll('.brainai-explainer');
    if (!roots.length) return;
    wireTooltips();
    roots.forEach((root) => {
      root.querySelectorAll('figure[data-widget]').forEach((fig) => {
        const fn = WIDGETS[fig.getAttribute('data-widget')];
        if (!fn) return;
        try { fn(fig); } catch (err) { console.error('brainai-explainer widget', fig.getAttribute('data-widget'), err); }
      });
      annotate(root);
      // A display equation too wide for the text column may use the margin too.
      root.querySelectorAll('.katex-display').forEach((d) => { if (d.scrollWidth > d.clientWidth + 1) d.classList.add('wide'); });
    });
    if (missing.size) console.debug('brainai-explainer: symbols without a tooltip:', [...missing].join(' '));
    window.__bxMissing = missing;
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
