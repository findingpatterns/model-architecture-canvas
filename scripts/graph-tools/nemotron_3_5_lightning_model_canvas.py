"""NVIDIA Nemotron 3.5 Lightning (30B-A3B): model tab + layer tab, both derived from the checkpoint.

Read top → bottom:
  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale everywhere)
  - boxes       = operations; parameter / compute shares are the two bars at the top of the model tab
  - layer rows  = one per layer, coloured by config.json `layers_block_type` (Mamba-2 / MoE / attention)
  - note-* ids  = annotations (dim, borderless); cap-* ids = borderless captions
Numbers come from shapes.json (safetensors headers of nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
and its config.json; the dataflow from transformers' models/nemotron_h/modeling_nemotron_h.py.
The MTP head (mtp.* tensors) is not implemented in transformers; its wiring is read off its tensor names.

usage: python3 nemotron_3_5_lightning_model_canvas.py <shapes.json> <config.json> <models/nemotron-3-5-lightning>
"""
import json
import math
import os
import re
import sys

S = json.load(open(sys.argv[1]))
C = json.load(open(sys.argv[2]))
OUT = sys.argv[3]

TYPES = C["layers_block_type"]
N, DIM, VOCAB = C["num_hidden_layers"], C["hidden_size"], C["vocab_size"]
MH, MHD, NG, ST, CONV = C["mamba_num_heads"], C["mamba_head_dim"], C["n_groups"], C["ssm_state_size"], C["conv_kernel"]
D_IN = MH * MHD  # Mamba inner width (the code uses heads × head_dim, not expand × hidden)
CONV_DIM = D_IN + 2 * NG * ST
PROJ = D_IN + CONV_DIM + MH
QH, KVH, HD = C["num_attention_heads"], C["num_key_value_heads"], C["head_dim"]
R, K, FF, SFF = C["n_routed_experts"], C["num_experts_per_tok"], C["moe_intermediate_size"], C["moe_shared_expert_intermediate_size"]

# colours (JSON Canvas presets "1"–"6" or hex)
EMB, NORM, HEAD, ACT = "6", "3", "2", "#64748b"
MAMBA, MOE, ATTN, MTP = "#38bdf8", "4", "1", "#f59e0b"
TYPE_COLOR = {"mamba": MAMBA, "moe": MOE, "attention": ATTN}
TYPE_NAME = {"mamba": "Mamba-2", "moe": "MoE", "attention": "Attention"}

# geometry
KT, CLIP_W = 0.08, 1200  # tensor bars: px per channel, longest drawn bar
ROW_W, ROW_H, ROW_GAP = 300, 34, 6
OP_W, OP_H, GAP = 300, 56, 40
BAR_X, BAR_W, BAR_H = -1000, 2600, 44
LBL_W = 440
MIN_SEG = 16  # the viewer draws no box narrower than ~14px


def params_of(pattern):
    rx = re.compile(pattern)
    return sum(math.prod(shp) for k, (dt, shp) in S.items() if rx.fullmatch(k))


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


LAYER_P = [params_of(rf"backbone\.layers\.{i}\..*") for i in range(N)]
EXPERT = params_of(rf"backbone\.layers\.{TYPES.index('moe')}\.mixer\.experts\.0\..*")
P = {
    "embed": params_of(r"backbone\.embeddings\.weight"),
    "head": params_of(r"lm_head\.weight"),
    "mtp": params_of(r"mtp\..*"),
    "mtp_expert": params_of(r"mtp\.layers\.\d+\.mixer\.experts\.0\..*"),
}


def type_p(t):
    return LAYER_P[TYPES.index(t)]


def budget():
    """[(name, colour, stored params, params one token is multiplied by, why-zero)] for the checkpoint.

    Active = only the K routed experts a token is sent to; embed rows are looked up, not multiplied;
    the MTP head only runs when it is used as a speculative drafter, so it is left out of per-token compute
    (that is how the card's "3B active" is counted).
    """
    rows = []
    for t in ("moe", "mamba", "attention"):
        n = TYPES.count(t)
        stored = sum(LAYER_P[i] for i in range(N) if TYPES[i] == t)
        active = stored - (n * (R - K) * EXPERT if t == "moe" else 0)
        rows.append((f"{TYPE_NAME[t]} ×{n}", TYPE_COLOR[t], stored, active))
    rows += [
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, P["head"], P["head"]),
        ("MTP head", MTP, P["mtp"], 0, "only while drafting"),
    ]
    other = params_of(r".*") - sum(r[2] for r in rows)
    if other >= 1e6:
        rows.append(("other", ACT, other, other))
    return rows


nodes, edges = [], []
_seq = [0]


def node(nid, x, y, w, h, text="", color=None, group=False):
    text = re.sub(r"(?<=[^\n])\n(?=[^\n-])", "  \n", text)  # markdown hard break; keeps paragraphs and lists
    d = dict(id=nid, type="group" if group else "text", x=round(x), y=round(y), width=round(w), height=round(h))
    d["label" if group else "text"] = text
    if color:
        d["color"] = color
    nodes.append(d)
    return nid


def edge(a, b, sides=("bottom", "top"), label=None, color=None):
    d = dict(id=f"e{len(edges)}", fromNode=a, toNode=b, fromSide=sides[0], toSide=sides[1])
    if label:
        d["label"] = label
    if color:
        d["color"] = color
    edges.append(d)


def save(name):
    json.dump(dict(nodes=nodes, edges=edges), open(os.path.join(OUT, name), "w"), indent=1, ensure_ascii=False)
    nodes.clear()
    edges.clear()


class Flow:
    """A vertical chain of ops and tensors centred on cx; each item is linked to the previous one."""

    def __init__(self, cx, y, last=None, labels_left=False):
        self.cx, self.y, self.last, self.labels_left = cx, y, last, labels_left

    def _id(self, key):
        _seq[0] += 1
        return f"{key}{_seq[0]}"

    def _link(self, top, bottom):
        if self.last:
            edge(self.last, top)
        self.last = bottom

    def op(self, text, color, params=0, note="", key="op", h=OP_H, w=OP_W):
        if params:
            text += f"\n{fmt(params)}" + (f" · {note}" if note else "")
        nid = node(self._id(key), self.cx - w / 2, self.y, w, h, text, color)
        self._link(nid, nid)
        self.y += h + GAP
        return nid

    def tensor(self, dim, label, lanes=1, key="t"):
        real = dim * KT
        w = min(real, CLIP_W)
        ids = [node(self._id(key), self.cx - w / 2, self.y + j * 14, w, 9, "", ACT) for j in range(lanes)]
        clip = f" — bar clipped (real {real:.0f}px)" if real > w else ""
        half = max(w / 2, OP_W / 2) + 20
        lx = self.cx - half - LBL_W if self.labels_left else self.cx + half
        node(self._id("note-lbl"), lx, self.y - 10, LBL_W, 34, f"`{label}`{clip}")
        self._link(ids[0], ids[-1])
        self.y += lanes * 14 + GAP
        return ids[0], ids[-1]


def bar(key, y, title, rows, idx):
    total = sum(r[idx] for r in rows)
    node(f"cap-{key}t", BAR_X, y - 46, BAR_W, 40, f"**{title}** — total {fmt(total)}")
    x = BAR_X
    legend = []
    # segments too thin to draw go last, so any blank stretch is at the bar's end, not mid-bar
    for k in sorted(range(len(rows)), key=lambda k: BAR_W * rows[k][idx] / total < MIN_SEG):
        r = rows[k]
        share = r[idx] / total
        w = BAR_W * share
        if w >= MIN_SEG:  # integer edges so neighbouring segments meet exactly
            node(f"{key}{k}", round(x), y, round(x + w) - round(x), BAR_H, f"**{r[0]}**" if w > 150 else "", r[1])
        x += w
        why = r[4] if len(r) > 4 and not r[idx] else ""
        legend.append(f"{r[0]} **{fmt(r[idx]) if r[idx] else '0'}**" + (f" ({why})" if why else f" ({share:.1%})"))
    node(f"cap-{key}l", BAR_X, y + BAR_H + 6, BAR_W, 40, " · ".join(legend))


# ======================================================================= model.canvas
B = budget()
bar("bp", -1640, "Where the parameters are stored", B, 2)
bar("bc", -1460, "What one token is multiplied by (active weights)", B, 3)

pattern = "".join({"mamba": "M", "moe": "E", "attention": "A"}[t] for t in TYPES)
node("cap-title", BAR_X, -1270, 1250, 220,
     "# Nemotron 3.5 Lightning — model\n"
     f"Hybrid of **{TYPES.count('mamba')} Mamba-2**, **{TYPES.count('moe')} MoE** and **{TYPES.count('attention')} attention** "
     f"layers ({N} in all). **Grey bars = tensors**, length ∝ channels per token ({KT} px/channel). "
     "**Boxes = operations**; parameter and compute shares are the two bars above. T = sequence length. "
     "*Layer* tab = one layer of each type in detail.")
node("cap-legend", BAR_X + 1350, -1270, 1250, 220,
     "Every layer = RMSNorm → **one** mixer → add residual. Layer rows are coloured by `layers_block_type`:\n"
     f"**Mamba-2** (SSM, {MH} heads, state {ST}) {fmt(type_p('mamba'))} · "
     f"**MoE** ({K} of {R} experts + 1 shared) {fmt(type_p('moe'))}, of which "
     f"{fmt(type_p('moe') - (R - K) * EXPERT)} run per token · "
     f"**Attention** (GQA {QH} q / {KVH} kv heads, no RoPE) {fmt(type_p('attention'))}.\n"
     f"Pattern: `{pattern}`")

cx = ROW_W / 2
f = Flow(cx, -960)
f.op("**raw text**  \"The cat sat…\"", ACT, key="raw")
f.op(f"**tokenizer** (BPE, vocab {VOCAB})", ACT, key="tok")
f.tensor(40 / KT, "token ids [T] — one integer per token", key="ids")
f.op("**embed** — look up row *id*", EMB, P["embed"], note=f"table {VOCAB} × {DIM}", key="embed")
f.tensor(DIM, f"x [T, {DIM}]", key="x")
ROW0 = f.y + 30

prev = f.last
for i, t in enumerate(TYPES):
    y = ROW0 + i * (ROW_H + ROW_GAP)
    nid = node(f"L{i}", 0, y, ROW_W, ROW_H, f"**L{i}** · {TYPE_NAME[t]}", TYPE_COLOR[t])
    edge(prev, nid)
    prev = nid
ylast = ROW0 + (N - 1) * (ROW_H + ROW_GAP)
node("band", -16, ROW0 - 30, ROW_W + 32, ylast - ROW0 + ROW_H + 46, f"{N} LAYERS · residual stream x [T, {DIM}]", "#94a3b8", group=True)

# beside the stack: what each layer type keeps between tokens, the part that sets long-context cost
kv = 2 * KVH * HD
node("note-cache", ROW_W + 60, ROW0 + 6 * (ROW_H + ROW_GAP), 520, 230,
     "**What a layer keeps while generating**\n"
     f"- Mamba-2: fixed state {MH} heads × {MHD} × {ST} + conv window {CONV - 1} × {CONV_DIM} — same size at any context length\n"
     f"- Attention: K and V of every past token, {KVH} heads × {HD} × 2 = {kv} values per token per layer\n"
     f"- MoE: nothing (per-token MLP)\n\n"
     f"Only {TYPES.count('attention')} of {N} layers grow with context; the card lists up to 1M tokens.")
node("note-pos", ROW_W + 60, ROW0 + 14 * (ROW_H + ROW_GAP), 520, 120,
     "**No positional encoding.** The attention layers apply no RoPE (`apply_rotary_pos_emb` is never called in "
     "`NemotronHAttention.forward`); token order reaches the model through the causal Mamba-2 layers.")
node("note-moe", ROW_W + 60, ROW0 + 20 * (ROW_H + ROW_GAP), 520, 120,
     f"**MoE router**: sigmoid scores over {R} experts, a learned bias (`e_score_correction_bias`) only for choosing "
     f"the top {K}; weights renormalised × {C['routed_scaling_factor']}. Experts are 2-matrix ReLU² MLPs (no gate matrix).")

o = Flow(cx, ylast + ROW_H + 60, last=f"L{N - 1}", labels_left=True)
h_out, _ = o.tensor(DIM, f"x [T, {DIM}]  after L{N - 1}", key="ho")
o.op("**final RMSNorm** (norm_f)", NORM, key="normf")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm head** — {DIM} → {VOCAB}", HEAD, P["head"], key="head")
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op("**softmax → sample**", ACT, key="samp")
next_id = o.op("**next token id**", ACT, key="next")

# ---------- MTP head: one extra attention + MoE pair that drafts token t+2 ----------
m = Flow(1800, ylast + ROW_H - 280)  # verify box lands level with "next token id"
m.op("**embed** (shared) of token t+1", EMB, key="memb")
m.tensor(DIM, f"e [T, {DIM}]", key="me")
cat = m.op("**enorm(e) ‖ hnorm(h)** — concat", MTP, key="mcat")
edge(h_out, cat, ("right", "left"), "h from the trunk", MTP)
m.tensor(2 * DIM, f"[T, {2 * DIM}]", key="mc")
m.op(f"**eh_proj** {2 * DIM} → {DIM}", MTP, params_of(r"mtp\.layers\.0\.eh_proj\.weight"), key="meh")
m.tensor(DIM, f"[T, {DIM}]", key="m1")
m.op("**attention layer** (as in trunk)", ATTN, params_of(r"mtp\.layers\.0\.(?!eh_proj|enorm|hnorm).*"), key="matt")
m.op(f"**MoE layer** ({K} of {R} + shared)", MOE, params_of(r"mtp\.layers\.1\.(?!final_layernorm).*"), key="mmoe")
m.op("**final_layernorm → lm head** (shared)", HEAD, key="mhead")
m.tensor(VOCAB, f"draft logits for token t+2 [T, {VOCAB}]", key="ml")
ver = m.op("**verify**: main model checks the draft\nin its next forward pass", ACT, key="mver")
edge(ver, next_id, ("left", "right"), "accepted drafts", MTP)
node("note-mtp", m.cx - OP_W / 2, m.y - 20, 620, 130,
     f"**Multi-token prediction** ({C['num_nextn_predict_layers']} MTP module, {fmt(P['mtp'])} stored, "
     f"{fmt(P['mtp'] - (R - K) * P['mtp_expert'])} active). Trained to predict one token further; at serving time it is "
     "one of the card's speculative-decoding options (besides external DSpark / DFlash drafters). "
     "Wiring read from `mtp.*` tensor names — transformers ignores these weights.")

save("model.canvas")

# ======================================================================= block.canvas
node("cap-btitle", -500, -420, 2900, 110,
     "# Nemotron 3.5 Lightning — one layer of each type\n"
     f"Each layer: `x + mixer(RMSNorm(x))`, with exactly one mixer. Shapes per token; T = sequence length. "
     f"Parameter counts from the checkpoint (L{TYPES.index('mamba')}, L{TYPES.index('attention')}, L{TYPES.index('moe')}).")


def layer_head(fl, title, color, params):
    fl.op(f"**{title}**", color, params, key="hd")
    fl.tensor(DIM, f"x [T, {DIM}]  layer input", key="in")
    fl.op("**RMSNorm**", NORM, key="nrm")
    fl.tensor(DIM, f"x̂ [T, {DIM}]", key="xh")


# ---- Mamba-2 column ----
mb = Flow(0, -240)
layer_head(mb, f"Mamba-2 layer (L{TYPES.index('mamba')})", MAMBA, type_p("mamba"))
mb.op(f"**in_proj** {DIM} → {PROJ}", MAMBA, params_of(r"backbone\.layers\.0\.mixer\.in_proj\.weight"), key="inp")
_, proj = mb.tensor(PROJ, f"[T, {PROJ}] = z {D_IN} ‖ xBC {CONV_DIM} ‖ dt {MH}", key="pj")
mb.op(f"**causal conv1d** on xBC — depthwise, kernel {CONV},\nthen SiLU", MAMBA, key="conv", h=70)
mb.tensor(CONV_DIM, f"xBC [T, {CONV_DIM}]", key="xbc")
mb.op(f"**split** x {D_IN} · B {NG * ST} · C {NG * ST}", MAMBA, key="spl")
mb.tensor(D_IN, f"x [T, {MH} heads, {MHD}] · B, C [T, {NG} groups, {ST}]", key="xs")
scan = mb.op(f"**SSD scan** (chunks of {C['chunk_size']})\nh ← exp(Δ·A)·h + Δ·B·x\ny = C·h + D·x", MAMBA, key="scan", h=96)
mb.tensor(D_IN, f"y [T, {D_IN}]", key="y")
gn = mb.op(f"**gated RMSNorm**: y · SiLU(z),\nnorm per {NG} groups of {D_IN // NG}", MAMBA, key="gn", h=70)
mb.tensor(D_IN, f"[T, {D_IN}]", key="g")
mb.op(f"**out_proj** {D_IN} → {DIM}", MAMBA, params_of(r"backbone\.layers\.0\.mixer\.out_proj\.weight"), key="out")
mb.tensor(DIM, f"[T, {DIM}]", key="mo")
mb.op("**+ residual** (layer input)", ACT, key="res")
mb.tensor(DIM, f"x [T, {DIM}]  to next layer", key="mx")

# side inputs of the scan and the gate, split off the in_proj output
side_x = -700
sd = Flow(side_x, next(n["y"] for n in nodes if n["id"] == proj) + 140, labels_left=True)
sd.tensor(MH, f"dt [T, {MH}]", key="dt")
edge(proj, sd.last, ("left", "top"))
dto = sd.op(f"**Δ = softplus(dt + dt_bias)**\n≥ {C['time_step_min']} · A = −exp(A_log)", MAMBA, key="dtop", h=70)
edge(dto, scan, ("bottom", "left"), "Δ, A (per head)")
sz = Flow(side_x, next(n["y"] for n in nodes if n["id"] == scan) + 150, labels_left=True)
sz.tensor(D_IN, f"z [T, {D_IN}]  gate", key="z")
edge(dto, sz.last, ("bottom", "top"))
edge(sz.last, gn, ("bottom", "left"))
node("note-state", -1100, next(n["y"] for n in nodes if n["id"] == scan) - 10, 360, 120,
     f"State per sequence: {MH} heads × {MHD} × {ST} (fp32 cache per config). "
     f"B, C are shared by the {MH // NG} heads of a group.")

# ---- attention column ----
ax = 1500
at = Flow(ax, -240)
li = TYPES.index("attention")
layer_head(at, f"Attention layer (L{li})", ATTN, type_p("attention"))
at.op(f"**q / k / v proj** {DIM} → {QH * HD} / {KVH * HD} / {KVH * HD}", ATTN,
      params_of(rf"backbone\.layers\.{li}\.mixer\.[qkv]_proj\.weight"), key="qkv", h=70)
at.tensor(QH * HD, f"q [T, {QH}, {HD}] · k, v [T, {KVH}, {HD}]", key="qkvt")
at.op(f"**causal attention**, softmax(q·kᵀ/√{HD})·v\n{QH // KVH} q heads share each kv head\nno RoPE", ATTN, key="sdpa", h=96)
at.tensor(QH * HD, f"[T, {QH * HD}]", key="ao")
at.op(f"**o_proj** {QH * HD} → {DIM}", ATTN, params_of(rf"backbone\.layers\.{li}\.mixer\.o_proj\.weight"), key="oproj")
at.tensor(DIM, f"[T, {DIM}]", key="aout")
at.op("**+ residual** (layer input)", ACT, key="ares")
at.tensor(DIM, f"x [T, {DIM}]  to next layer", key="ax")

# ---- MoE column ----
ex = 3000
mo = Flow(ex, -240, labels_left=True)
lm = TYPES.index("moe")
layer_head(mo, f"MoE layer (L{lm})", MOE, type_p("moe"))
xh = mo.last
mo.op(f"**router** {DIM} → {R}, sigmoid", MOE, params_of(rf"backbone\.layers\.{lm}\.mixer\.gate\.weight"), key="rt")
mo.tensor(R, f"scores [T, {R}]", key="sc")
mo.op(f"**top-{K}** on score + bias;\nweights renormalised × {C['routed_scaling_factor']}", MOE, key="tk", h=70)
mo.op(f"**{K} routed experts** (of {R})\nup {DIM} → {FF} · ReLU² · down → {DIM}", MOE, EXPERT, note="each", key="exp", h=90)
mo.tensor(DIM, f"Σ weightᵢ · expertᵢ(x̂)  [T, {DIM}]", key="rs")
add = mo.op("**+ shared expert**", MOE, key="add")
mo.tensor(DIM, f"[T, {DIM}]", key="mout")
mo.op("**+ residual** (layer input)", ACT, key="mres")
mo.tensor(DIM, f"x [T, {DIM}]  to next layer", key="mx2")
# the shared expert runs on every token, beside the router
sh = Flow(ex + 700, next(n["y"] for n in nodes if n["id"] == xh) + 160)
s1 = sh.op(f"**shared expert** (every token)\nup {DIM} → {SFF} · ReLU² · down → {DIM}", MOE,
           params_of(rf"backbone\.layers\.{lm}\.mixer\.shared_experts\..*"), key="shx", h=90)
edge(xh, s1, ("right", "top"))
sh.tensor(DIM, f"[T, {DIM}]", key="sht")
edge(sh.last, add, ("bottom", "right"))

save("block.canvas")
print("ok", [(r[0], fmt(r[2]), fmt(r[3])) for r in B], "pattern", pattern)
