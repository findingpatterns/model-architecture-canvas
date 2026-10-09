"""Muse Glimmer 30B (meta-models/Muse-Glimmer-30B): whole-model canvas + one-decoder-layer canvas.

Read top → bottom:
  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale per canvas)
  - boxes       = operations, all one size; parameter shares are the two bars at the top of model.canvas
  - layer rows  = one per layer, coloured by layer type from config.json (sliding+RoPE vs full+NoPE)
  - note-* ids  = annotations (dim, borderless); cap-* ids = borderless captions
Every number comes from config.json + safetensors headers (shapes.json) of the HF repo, the drafter's
config/shapes (meta-models/Muse-Glimmer-30B-assistant) and the transformers `muse_glimmer` modeling code.

usage: python3 muse_glimmer_model_canvas.py <shapes.json> <config.json> <asst_shapes.json> <asst_config.json> <models/muse-glimmer>
"""
import json
import math
import os
import re
import sys

S = json.load(open(sys.argv[1]))
CFG = json.load(open(sys.argv[2]))
AS = json.load(open(sys.argv[3]))
AC = json.load(open(sys.argv[4]))
OUT = sys.argv[5]
C, V = CFG["text_config"], CFG["vision_config"]

N, DIM, VOCAB, FF = C["num_hidden_layers"], C["hidden_size"], C["vocab_size"], C["intermediate_size"]
HQ, HKV, HD, WIN = C["num_attention_heads"], C["num_key_value_heads"], C["head_dim"], C["sliding_window"]
TYPES, THETA = C["layer_types"], C["layer_rope_theta"]
VD, VL, PS, PT, MS = V["hidden_size"], V["num_hidden_layers"], V["patch_size"], V["patch_temporal"], V["merge_size"]
PATCH_IN = PT * 3 * PS * PS
V_WIN = sum(t == "window_attention" for t in V["layer_types"])
TAPS = AC["target_layer_ids"]

# colours (JSON Canvas presets "1"–"6" or hex)
EMB, NORM, FFN, HEAD, ACT, ATT = "6", "3", "4", "2", "#64748b", "5"
VISION, DRAFT = "#a78bfa", "#f59e0b"
ROW_COLOR = {"sliding_attention": "#94a3b8", "full_attention": "5"}

OP_W, OP_H, GAP = 300, 56, 40
MIN_SEG = 16  # the viewer draws no box narrower than ~14px


def params_of(pattern, shapes=S):
    rx = re.compile(pattern)
    return sum(math.prod(s) for k, (_, s) in shapes.items() if rx.fullmatch(k))


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


LM = r"model\.language_model\."
P = {
    "layers": params_of(LM + r"layers\..*"),
    "layer": params_of(LM + r"layers\.0\..*"),
    "attn": params_of(LM + r"layers\.0\.self_attn\..*"),
    "mlp": params_of(LM + r"layers\.0\.mlp\..*"),
    "embed": params_of(LM + r"embed_tokens\.weight"),
    "head": params_of(r"lm_head\.weight"),
    "vit": params_of(r"model\.vision_tower\..*"),
    "patch": params_of(r"model\.vision_tower\.patch_embedder\..*"),
    "adapter": params_of(r"model\.vision_adapter\..*"),
    "vproj": params_of(r"model\.vision_projection\..*"),
    "all": params_of(r".*"),
    "draft": params_of(r".*", AS),
    "draft_fc": params_of(r"encoder\..*", AS),
}
assert all(params_of(LM + rf"layers\.{i}\..*") == P["layer"] for i in range(N)), "layers differ in size"


class Canvas:
    def __init__(self, kt, clip, lbl_w):
        self.nodes, self.edges, self.kt, self.clip, self.lbl_w, self.seq = [], [], kt, clip, lbl_w, 0

    def node(self, nid, x, y, w, h, text="", color=None, group=False):
        text = re.sub(r"(?<=[^\n])\n(?=[^\n-])", "  \n", text)  # markdown hard break; keeps paragraphs and lists
        d = dict(id=nid, type="group" if group else "text", x=round(x), y=round(y), width=round(w), height=round(h))
        d["label" if group else "text"] = text
        if color:
            d["color"] = color
        self.nodes.append(d)
        return nid

    def edge(self, a, b, sides=("bottom", "top"), label=None, color=None):
        d = dict(id=f"e{len(self.edges)}", fromNode=a, toNode=b, fromSide=sides[0], toSide=sides[1])
        if label:
            d["label"] = label
        if color:
            d["color"] = color
        self.edges.append(d)

    def uid(self, key):
        self.seq += 1
        return f"{key}{self.seq}"

    def save(self, name):
        json.dump(dict(nodes=self.nodes, edges=self.edges), open(os.path.join(OUT, name), "w"), indent=1, ensure_ascii=False)


class Flow:
    """A vertical chain of ops and tensors centred on cx; each item is linked to the previous one."""

    def __init__(self, cv, cx, y, last=None, labels_left=False, op_w=OP_W, lbl_w=None):
        self.cv, self.cx, self.y, self.last, self.labels_left, self.op_w = cv, cx, y, last, labels_left, op_w
        self.lbl_w = lbl_w or cv.lbl_w

    def _link(self, top, bottom):
        if self.last:
            self.cv.edge(self.last, top)
        self.last = bottom

    def op(self, text, color, params=0, note="", key="op", h=OP_H):
        if params:
            text += f"\n{fmt(params)}" + (f" · {note}" if note else "")
        nid = self.cv.node(self.cv.uid(key), self.cx - self.op_w / 2, self.y, self.op_w, h, text, color)
        self._link(nid, nid)
        self.y += h + GAP
        return nid

    def tensor(self, dim, label, lanes=1, key="t", left=None):
        cv = self.cv
        real = dim * cv.kt
        w = min(real, cv.clip)
        ids = [cv.node(cv.uid(key), self.cx - w / 2, self.y + j * 14, w, 9, "", ACT) for j in range(lanes)]
        clip = f" — bar clipped (real {real:.0f}px)" if real > w else ""
        half = max(w / 2, self.op_w / 2) + 20
        lx = self.cx - half - self.lbl_w if (self.labels_left if left is None else left) else self.cx + half
        cv.node(cv.uid("note-lbl"), lx, self.y - 10, self.lbl_w, 34, f"`{label}`{clip}")
        self._link(ids[0], ids[-1])
        self.y += lanes * 14 + GAP
        return ids[0], ids[-1]


# ======================= model.canvas =======================
M = Canvas(kt=0.08, clip=1400, lbl_w=440)
ROW_W, ROW_H, ROW_GAP, BAR_W, BAR_H = 300, 34, 6, 2600, 44
n_full = sum(t == "full_attention" for t in TYPES)
assert all((t == "full_attention") == (THETA[i] == 0) for i, t in enumerate(TYPES)), "NoPE ≠ full layers"

M.node("cap-title", -900, -1270, 1250, 180,
       "# Muse Glimmer 30B — model\n"
       "Read top → bottom. **Grey bars = tensors**, length ∝ channels per token (0.08 px/channel). "
       "**Boxes = operations**, all the same size; where the parameters live is the pair of bars above. "
       "T = sequence length, P = image patches. *Layer* tab = one decoder layer in detail.")
M.node("cap-legend", 450, -1270, 1250, 180,
       f"**Dense** decoder, {N} identical-size layers ({fmt(P['layer'])} each), pattern [sliding, sliding, sliding, full] ×{N // 4}.  \n"
       f"**Grey rows** — sliding window {WIN} tokens + RoPE θ {C['rope_parameters']['rope_theta']:,.0f} · "
       f"**Cyan rows** — full causal attention with **no positional encoding** (NoPE, layer_rope_theta = 0).  \n"
       f"All layers: GQA {HQ} Q / {HKV} KV heads × {HD}, sigmoid-gated attention output, SwiGLU {FF}, "
       f"4 RMSNorms per layer (sandwich). Context {C['max_position_embeddings']:,}.")

# ---- text input ----
cx = ROW_W / 2
f = Flow(M, cx, -960)
f.op("**raw text**  \"The cat sat…\"", ACT, key="raw")
f.op(f"**tokenizer** (BPE, vocab {VOCAB:,})", ACT, key="tok")
f.tensor(40 / M.kt, "token ids [T] — one integer per token", key="ids")
f.op("**embed** — look up row *id*\nthen RMSNorm (no weight)", EMB, key="embed", h=72)
x0, _ = f.tensor(DIM, f"x [T, {DIM}]  (+ image tokens spliced in)", key="x")
ROW0 = f.y + 40

# ---- layer rows ----
prev = f.last
for i in range(N):
    y = ROW0 + i * (ROW_H + ROW_GAP)
    full = TYPES[i] == "full_attention"
    txt = f"**L{i}** · " + ("full · NoPE" if full else f"sliding {WIN} · RoPE")
    M.node(f"L{i}", 0, y, ROW_W, ROW_H, txt, ROW_COLOR[TYPES[i]])
    M.edge(prev, f"L{i}")
    prev = f"L{i}"
yend = ROW0 + N * (ROW_H + ROW_GAP) - ROW_GAP
M.node("band", -16, ROW0 - 30, ROW_W + 32, yend - ROW0 + 46, f"{N} DECODER LAYERS · {fmt(P['layers'])}", "#38bdf8", group=True)

# ---- notes left of the layer stack (below the vision branch) ----
kv = 2 * HKV * HD
M.node("note-pattern", -560, ROW0 + 1000, 480, 230,
       f"**Layer pattern** — {N - n_full} sliding + {n_full} full layers.\n"
       f"Sliding layers attend to the last {WIN} tokens and rotate q/k with RoPE; full layers see the whole context and "
       "carry no position signal at all (they rely on causal masking and the sliding layers for order).\n"
       f"KV cache per token per layer: K+V = 2 × {HKV} × {HD} = {kv} values — a sliding layer keeps at most {WIN} tokens, "
       f"so long-context memory is dominated by the {n_full} full layers.")
M.node("note-layer", -560, ROW0 + 1270, 480, 150,
       f"**One layer** = {fmt(P['layer'])}: attention {fmt(P['attn'])} (q, k, v, o + a {HQ * HD}-wide output gate) "
       f"+ SwiGLU MLP {fmt(P['mlp'])} ({P['mlp'] / P['layer']:.0%}).\n"
       f"Logits: lm head × {C['output_multiplier']:.4f}, then softcap {C['final_logit_softcapping']:.0f}·tanh(z / "
       f"{C['final_logit_softcapping']:.0f}).")

# ---- vision branch (perception encoder), spliced into x ----
v = Flow(M, -480, -960, labels_left=True)
v.op("**image** pixels (video: frame pairs)", ACT, key="img")
v.tensor(PATCH_IN, f"patches [P, {PATCH_IN}]  ({PT}×{PS}×{PS}×3)", key="patch")
v.op(f"**patch embed** {PATCH_IN} → {VD}\n+ learned {V['pos_emb_height']}×{V['pos_emb_width']} pos table (bilinear)",
     VISION, P["patch"], key="pe", h=90)
v.tensor(VD, f"[P, {VD}]", key="vf0")
v.op(f"**ViT ×{VL}** — {V_WIN} window + {VL - V_WIN} full\n2D RoPE · LayerNorm · GELU MLP {V['intermediate_size']}",
     VISION, P["vit"] - P["patch"], key="vit", h=90)
v.tensor(VD, f"[P, {VD}]", key="vf1")
v.op(f"**pixel shuffle** {MS}×{MS} patches → 1 token", VISION, key="ps")
v.tensor(VD * MS * MS, f"[P/{MS * MS}, {VD * MS * MS}]", key="vf2")
v.op(f"**adapter** {VD * MS * MS} → {CFG['projector_hidden_size']} → {CFG['projector_hidden_size']} (GELU ×2)",
     VISION, P["adapter"], key="ada", h=72)
v.op(f"**projection** {CFG['projector_hidden_size']} → {DIM}\n+ RMSNorm (no weight)", VISION, P["vproj"], key="vpr", h=90)
v.tensor(DIM, f"image tokens [P/{MS * MS}, {DIM}]", key="vt")
M.edge(v.last, x0, ("right", "left"), "spliced into x")

# ---- output ----
o = Flow(M, cx, yend + 60, last=f"L{N - 1}")
o.tensor(DIM, f"h [T, {DIM}]  after L{N - 1}", key="ho")
o.op("**final RMSNorm**", NORM, key="fn")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm head** {DIM} → {VOCAB:,}", HEAD, P["head"], note="not tied to embed", key="head")
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op(f"**× {C['output_multiplier']:.3f} → softcap {C['final_logit_softcapping']:.0f}**", HEAD, key="cap")
o.op("**softmax → sample**", ACT, key="samp")
next_id = o.op("**next token id(s)**", ACT, key="next")

# ---- DFlash drafter (separate repo), fed by 5 tapped layers ----
B = AC["block_size"]
tap_x = ROW_W + 30
d = Flow(M, 2000, yend + 110)
cat, _ = d.tensor(DIM * len(TAPS), f"concat hidden of {len(TAPS)} layers  [T, {DIM * len(TAPS)}]", key="dcat")
for t in TAPS:
    tap = M.node(f"note-tap{t}", tap_x, ROW0 + t * (ROW_H + ROW_GAP), 230, ROW_H, f"↳ L{t} hidden → drafter")
    M.edge(tap, cat, ("right", "top"), color=DRAFT)
d.op(f"**fc** {DIM * len(TAPS)} → {DIM} + RMSNorm", DRAFT, P["draft_fc"], key="dfc")
d.op(f"**DFlash drafter** — {AC['num_hidden_layers']} layers\nsliding {AC['sliding_window']} · GQA "
     f"{AC['num_attention_heads']}/{AC['num_key_value_heads']} · SwiGLU {AC['intermediate_size']}",
     DRAFT, P["draft"] - P["draft_fc"], key="dblk", h=90)
d.op(f"**target's lm head** → {B} draft tokens\nin ONE forward (block diffusion)", DRAFT, key="dhead", h=72)
d.tensor(40 / M.kt, f"draft ids [{B}]", key="dids")
ver = d.op(f"**verify**: main model scores all {B}\ndrafts in one forward pass", ACT, key="dver", h=72)
M.edge(ver, next_id, ("left", "right"), "keeps the drafts it agrees with", DRAFT)
M.node("note-dflash", d.cx - OP_W / 2, d.y - 20, 640, 150,
       f"Speculative decoding, shipped as a separate repo (meta-models/Muse-Glimmer-30B-assistant, {fmt(P['draft'])}). "
       f"The drafter has no embedding or head of its own: it embeds the last accepted token + {B - 1} mask tokens "
       f"with the target's table, attends bidirectionally inside the block and causally to the tapped context "
       f"(target_layer_ids {TAPS}). Not counted in the budget bars above.")


# ---- budget bars ----
def budget():
    """(name, colour, stored, multiplied per text token, why-zero)."""
    vision = P["vit"] + P["adapter"] + P["vproj"]
    rows = [
        (f"{N} layers", FFN, P["layers"], P["layers"]),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, P["head"], P["head"]),
        ("vision tower + adapter", VISION, vision, 0, "runs per image"),
    ]
    other = P["all"] - sum(r[2] for r in rows)
    assert other < 1e6, other  # only the final norm is left
    return rows


def bar(key, y, title, rows, idx):
    total = sum(r[idx] for r in rows)
    M.node(f"cap-{key}t", -900, y - 46, BAR_W, 40, f"**{title}** — total {fmt(total)}")
    x, legend = -900, []
    # segments too thin to draw go last, so any blank stretch is at the bar's end, not mid-bar
    for k in sorted(range(len(rows)), key=lambda k: BAR_W * rows[k][idx] / total < MIN_SEG):
        r = rows[k]
        share = r[idx] / total
        w = BAR_W * share
        if w >= MIN_SEG:  # integer edges so neighbouring segments meet exactly
            M.node(f"{key}{k}", round(x), y, round(x + w) - round(x), BAR_H, f"**{r[0]}**" if w > 150 else "", r[1])
        x += w
        why = r[4] if len(r) > 4 and not r[idx] else ""
        legend.append(f"{r[0]} **{fmt(r[idx]) if r[idx] else '0'}**" + (f" ({why})" if why else f" ({share:.1%})"))
    M.node(f"cap-{key}l", -900, y + BAR_H + 6, BAR_W, 40, " · ".join(legend))


BUD = budget()
bar("bp", -1640, "Where the parameters are stored (main checkpoint, BF16)", BUD, 2)
bar("bc", -1460, "What one text token is multiplied by — dense, so every layer weight; only the lookup table and the per-image vision path drop out",
    BUD, 3)
M.save("model.canvas")

# ======================= block.canvas =======================
K = Canvas(kt=0.05, clip=400, lbl_w=240)
COL = 560  # spacing of the q / k / v / gate columns
K.node("cap-btitle", -700, -330, 1500, 150,
       f"# Muse Glimmer 30B — one decoder layer (L0–L{N - 1}, all the same size: {fmt(P['layer'])})\n"
       "Grey bars = tensors (0.05 px/channel), boxes = operations with their weights. Sandwich norms: each sub-block is "
       "wrapped by an RMSNorm before *and* after, then added to the residual. The four layer norms scale by (1 + w).")
K.node("cap-blegend", 900, -330, 900, 150,
       f"Sliding layers ({N - n_full}): RoPE on q/k, causal window {WIN}.  \n"
       f"Full layers ({n_full}, every 4th): **no RoPE** (NoPE), causal over the whole context.  \n"
       f"Everything else is identical, so both types share this diagram.")

m = Flow(K, 0, 0, lbl_w=440)
xin, _ = m.tensor(DIM, f"x [T, {DIM}]  residual stream in", key="xin")
m.op("**input norm** — RMSNorm", NORM, key="n1")
_, hb = m.tensor(DIM, f"h [T, {DIM}]", key="h")
yb = m.y + 280  # room for the fan-out to the four projections
ap = lambda name: params_of(LM + rf"layers\.0\.self_attn\.{name}\.weight")
q = Flow(K, 0, yb, last=hb)
q.op(f"**q_proj** {DIM} → {HQ * HD}", ATT, ap("q_proj"), key="q")
q.tensor(HQ * HD, f"q [T, {HQ}, {HD}]", key="qt")
q.op(f"**qk-norm** (RMS, no weight)\n× {C['qk_scale_factor']}", NORM, key="qn")
q.op(f"**RoPE** θ {C['rope_parameters']['rope_theta']:,.0f}\nsliding layers only", ATT, key="qr")
k = Flow(K, COL, yb, last=hb)
k.op(f"**k_proj** {DIM} → {HKV * HD}", ATT, ap("k_proj"), key="k")
k.tensor(HKV * HD, f"k [T, {HKV}, {HD}]", key="kt")
k.op("**qk-norm** (RMS, no weight)", NORM, key="kn")
k.op("**RoPE**\nsliding layers only", ATT, key="kr")
vv = Flow(K, 2 * COL, yb, last=hb)
vv.op(f"**v_proj** {DIM} → {HKV * HD}", ATT, ap("v_proj"), key="v")
vv.tensor(HKV * HD, f"v [T, {HKV}, {HD}]", key="vt")
g = Flow(K, 3 * COL, yb, last=hb)
g.op(f"**gate_proj** {DIM} → {HQ * HD}", ATT, ap("gate_proj"), key="g")
g.tensor(HQ * HD, f"g [T, {HQ * HD}]", key="gt")
g.op("**sigmoid**", ATT, key="gs")

ya = q.y
att = K.node("att", -OP_W / 2, ya, COL + OP_W, 90,
             f"**attention** — {HQ} query heads share {HKV} KV heads ({HQ // HKV}:1 GQA), head {HD}, "
             f"scale 1/√{HD}\nsliding: causal window {WIN} · full: causal over all tokens", ATT)
K.edge(q.last, att)
K.edge(k.last, att)
K.edge(vv.last, att, ("bottom", "right"))
m = Flow(K, 0, ya + 90 + GAP + 20, last=att, lbl_w=440)
m.tensor(HQ * HD, f"[T, {HQ * HD}]  heads concatenated", key="ao")
mul = m.op("**× gate** (elementwise)", ATT, key="mul")
K.edge(g.last, mul, ("bottom", "right"), "per-channel 0…1")
m.tensor(HQ * HD, f"[T, {HQ * HD}]", key="ag")
m.op(f"**o_proj** {HQ * HD} → {DIM}", ATT, ap("o_proj"), key="o")
m.tensor(DIM, f"[T, {DIM}]", key="ot")
m.op("**post-attention norm** — RMSNorm", NORM, key="n2")
add1 = m.op("**+ residual**", ACT, key="add1")
K.edge(xin, add1, ("left", "left"), "residual")
xm, _ = m.tensor(DIM, f"x [T, {DIM}]", key="xm")
m.op("**pre-FFN norm** — RMSNorm", NORM, key="n3")
m.tensor(DIM, f"h [T, {DIM}]", key="h2")
mp = lambda name: params_of(LM + rf"layers\.0\.mlp\.{name}\.weight")
m.op(f"**gate_proj ‖ up_proj** {DIM} → 2 × {FF}", FFN, mp("gate_proj") + mp("up_proj"), key="gu")
m.tensor(FF, f"gate, up [T, {FF}]", lanes=2, key="gut")
m.op("**SiLU(gate) · up**", FFN, key="swi")
m.tensor(FF, f"[T, {FF}]", key="ff")
m.op(f"**down_proj** {FF} → {DIM}", FFN, mp("down_proj"), key="down")
m.tensor(DIM, f"[T, {DIM}]", key="dt")
m.op("**post-FFN norm** — RMSNorm", NORM, key="n4")
add2 = m.op("**+ residual**", ACT, key="add2")
K.edge(xm, add2, ("left", "left"), "residual")
m.tensor(DIM, f"x [T, {DIM}]  → next layer", key="xout")
K.node("note-gate", 3 * COL + OP_W / 2 + 300, yb, 420, 150,
       f"**Gated attention**: a {fmt(ap('gate_proj'))} gate_proj reads the same normed input; sigmoid(g) scales every "
       "channel of the attention output before o_proj, so a head can switch itself off per token.")
K.node("note-qk", -1100, yb, 420, 150,
       f"**qk-norm** has no learned weight; q is then multiplied by qk_scale_factor {C['qk_scale_factor']} "
       f"before the usual 1/√{HD} scaling. KV cache per token: {kv} values (K+V, {HKV} heads × {HD}).")
K.save("block.canvas")

print("ok", [(r[0], fmt(r[2]), fmt(r[3])) for r in BUD], "total", fmt(P["all"]), "drafter", fmt(P["draft"]))
