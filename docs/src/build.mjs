// Builds the interactive explainer from docs/src/article.html.
//   deno run -A docs/src/build.mjs
// Writes docs/index.html (standalone, for GitHub Pages) and
// docs/blog/brainai-how-it-works.md (a post for the author's Astro blog).
// The only dependency is KaTeX, fetched by Deno, used to turn TeX into
// MathML once, so the published page needs no maths library at all.
import katex from "npm:katex@0.16.11";

const here = new URL(".", import.meta.url);
const docs = new URL("../", here);
const read = (p) => Deno.readTextFileSync(new URL(p, here));

const TITLE = "Running a language model on an electronic dictionary";
const DESCRIPTION = "How brainux-brainai runs brain-ai, a 14.7-million-parameter Japanese language model, on a SHARP Brain electronic dictionary at 6 tokens per second.";
const DATE = "2026-11-17";

let body = read("article.html").replace(/^\s*<!--[\s\S]*?-->\s*/, "");
body = `<p class="lede"><em>${DESCRIPTION}</em></p>\n\n` + body;

// <sn>…</sn> → the blog's sidenote markup (same as its rehype-sidenotes plugin output)
let n = 0;
body = body.replace(/<sn>([\s\S]*?)<\/sn>/g, (_, note) => {
  n += 1;
  return `<span class="sidenote-wrap"><label for="sn-${n}" class="sidenote-number"></label><input type="checkbox" id="sn-${n}" class="sidenote-toggle"><span class="sidenote">${note.trim()}</span></span>`;
});

const tex = (src, displayMode) =>
  katex.renderToString(src.trim(), { output: "mathml", displayMode, throwOnError: true, strict: "ignore" });
body = body.replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => `<div class="katex-display">${tex(t, true)}</div>`);
body = body.replace(/\$([^$\n]+?)\$/g, (_, t) => tex(t, false));
// Keep inline maths together with the punctuation around it, so a line never
// starts with a comma.
body = body.replace(/(\(?)(<span class="katex"><math[\s\S]*?<\/math><\/span>)([,.;:)’”]*)/g,
  (m, pre, math, post) => (pre || post ? `<span class="bx-nowrap">${pre}${math}${post}</span>` : m));

const fill = (s) => s.replaceAll("{{TITLE}}", TITLE).replaceAll("{{DESCRIPTION}}", DESCRIPTION).replaceAll("{{DATE}}", DATE);
Deno.writeTextFileSync(new URL("index.html", docs), fill(read("template.html")).replace("{{BODY}}", body));

// Blog export. The whole body is one raw-HTML block for Markdown: no blank
// lines (an empty comment keeps empty lines inside <pre> intact).
const blogBody = body.replace(/\n[ \t]*(?=\n)/g, "\n<!---->");
const md = `---
title: ${JSON.stringify(TITLE)}
date: ${DATE}
description: ${JSON.stringify(DESCRIPTION)}
---
<link rel="stylesheet" href="/brainai/explainer.css">
<div class="brainai-explainer">
${blogBody}
</div>
<script src="/brainai/explainer.js" defer></script>
`;
Deno.mkdirSync(new URL("blog/", docs), { recursive: true });
Deno.writeTextFileSync(new URL("blog/brainai-how-it-works.md", docs), md);
console.log(`wrote docs/index.html and docs/blog/brainai-how-it-works.md (${n} side notes)`);
