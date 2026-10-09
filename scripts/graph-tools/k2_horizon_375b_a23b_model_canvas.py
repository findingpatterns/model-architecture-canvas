"""K2-Horizon-375B-A23B (IFM / MBZUAI): model.canvas (all 61 layers) + block.canvas (one MoE layer).

Read top → bottom:
  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale everywhere)
  - boxes       = operations, all one size; parameter / compute shares are the two bars at the top
  - layer rows  = one per layer, coloured by FFN type (dense SwiGLU for config.mlp_only_layers, MoE otherwise)
  - note-* ids  = annotations (dim, borderless); cap-* ids = borderless captions
Every number comes from config.json, the safetensors headers (shapes.json) or modeling_k2_horizon.py.

usage: python3 k2_horizon_375b_a23b_model_canvas.py <shapes.json> <config.json> <models/k2-horizon-375b-a23b>
"""
import json
import math
import os
import re
import sys

S = json.load(open(sys.argv[1]))
C = json.load(open(sys.argv[2]))
OUT = sys.argv[3]

N = C["num_hidden_layers"]
DIM, VOCAB = C["hidden_size"], C["vocab_size"]
NH, NKV, HD, RD = C["num_attention_heads"], C["num_key_value_heads"], C["head_dim"], C["rope_head_dim"]
E, K, NSH = C["num_experts"], C["num_experts_per_tok"], C["num_shared_experts"]
MI, DI = C["moe_intermediate_size"], C["intermediate_size"]
DENSE = set(C["mlp_only_layers"])
CTX = C["max_position_embeddings"]
THETA = C["rope_parameters"]["rope_theta"]
assert C["decoder_sparse_step"] == 1 and C["mova_num_experts"] == 0 and not C["query_key_norm"]
assert C["attention_gate_func"] is None and C["sliding_window"] is None and not C["tie_word_embeddings"]

# colours (JSON Canvas presets "1"–"6" or hex)
EMB, NORM, ATTN, MOE, HEAD, ACT = "6", "3", "5", "4", "2", "#64748b"
DENSE_C, SHARED_C, ROUTER_C = "#f472b6", "#14b8a6", "#f59e0b"

# geometry
KT, CLIP_W = 0.08, 1800  # tensor bars: px per channel, longest drawn bar
ROW_W, ROW_H, ROW_GAP = 300, 34, 6
L_X, R_X = 0, 900  # left edge of the two layer columns
OP_W, OP_H, GAP = 300, 56, 40
BAR_W, BAR_H = 2600, 44
LBL_W = 440
MIN_SEG = 16  # the viewer draws no box narrower than ~14px


def params_of(pattern):
    rx = re.compile(pattern)
    return sum(math.prod(shp) for k, (dt, shp) in S.items() if rx.fullmatch(k))


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


TOTAL = params_of(r".*")
P = dict(
    embed=params_of(r"model\.embed_tokens\.weight"),
    head=params_of(r"lm_head\.weight"),
    attn=params_of(r"model\.layers\.\d+\.self_attn\..*"),
    dense=params_of(r"model\.layers\.\d+\.mlp\.(gate|up|down)_proj\.weight"),
    routed=params_of(r"model\.layers\.\d+\.mlp\.experts\..*"),
    shared=params_of(r"model\.layers\.\d+\.mlp\.shared_experts\..*"),
    router=params_of(r"model\.layers\.\d+\.mlp\.gate\.(weight|bias)"),
)
P["norms"] = params_of(r".*norm\.weight")
MOE_LAYERS = N - len(DENSE)
EXPERT = params_of(r"model\.layers\.3\.mlp\.experts\.0\..*")
ATTN_L = params_of(r"model\.layers\.0\.self_attn\..*")
DENSE_L = params_of(r"model\.layers\.0\.mlp\.(gate|up|down)_proj\.weight")
assert P["routed"] == MOE_LAYERS * E * EXPERT
assert sum(P.values()) == TOTAL, "every tensor must belong to one budget row"
ACTIVE_ROUTED = MOE_LAYERS * K * EXPERT
ACTIVE = TOTAL - P["embed"] - (P["routed"] - ACTIVE_ROUTED)  # multiplied per token (embed = row lookup)
ACTIVE_NONEMB = ACTIVE - P["head"]
KV_PER_TOKEN = 2 * NKV * HD * N * 2  # K and V, BF16 bytes, all layers


def budget():
    """[(name, colour, stored params, params multiplied per token, why-zero)]"""
    return [
        (f"routed experts {MOE_LAYERS}×{E}", MOE, P["routed"], ACTIVE_ROUTED),
        (f"attention ×{N}", ATTN, P["attn"], P["attn"]),
        (f"shared expert ×{MOE_LAYERS}", SHARED_C, P["shared"], P["shared"]),
        (f"dense FFN ×{len(DENSE)}", DENSE_C, P["dense"], P["dense"]),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, P["head"], P["head"]),
        ("routers + norms", ACT, P["router"] + P["norms"], P["router"] + P["norms"]),
    ]


nodes, edges = [], []


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


_seq = [0]


class Flow:
    """A vertical chain of ops and tensors centred on cx; each item is linked to the previous one."""

    def __init__(self, cx, y, last=None, labels_left=False):
        self.cx, self.y, self.last, self.labels_left = cx, y, last, labels_left

    def _id(self, key):
        _seq[0] += 1
        return f"{key}{_seq[0]}"

    def _link(self, top, bottom, label=None):
        if self.last:
            edge(self.last, top, label=label)
        self.last = bottom

    def op(self, text, color, params=0, note="", key="op", h=OP_H, w=OP_W):
        if params:
            text += f"\n{fmt(params)}" + (f" · {note}" if note else "")
        nid = node(self._id(key), self.cx - w / 2, self.y, w, h, text, color)
        self._link(nid, nid)
        self.y += h + GAP
        return nid

    def tensor(self, dim, label, key="t"):
        real = dim * KT
        w = min(real, CLIP_W)
        nid = node(self._id(key), self.cx - w / 2, self.y, w, 9, "", ACT)
        clip = f" — bar clipped (real {real:.0f}px)" if real > w else ""
        half = max(w / 2, OP_W / 2) + 20
        lx = self.cx - half - LBL_W if self.labels_left else self.cx + half
        node(self._id("note-lbl"), lx, self.y - 10, LBL_W, 34, f"`{label}`{clip}")
        self._link(nid, nid)
        self.y += 9 + GAP
        return nid


def bar(key, x0, y, title, rows, idx, total):
    node(f"cap-{key}t", x0, y - 46, BAR_W, 40, f"**{title}** — total {fmt(total)}")
    x = x0
    legend = []
    # segments too thin to draw go last, so any blank stretch is at the bar's end, not mid-bar
    for k in sorted(range(len(rows)), key=lambda k: BAR_W * rows[k][idx] / total < MIN_SEG):
        r = rows[k]
        share = r[idx] / total
        w = BAR_W * share
        if w >= MIN_SEG:  # integer edges so neighbouring segments meet exactly
            node(f"{key}{k}", round(x), y, round(x + w) - round(x), BAR_H, f"**{r[0]}**" if w > 220 else "", r[1])
        x += w
        why = r[4] if len(r) > 4 and not r[idx] else ""
        legend.append(f"{r[0]} **{fmt(r[idx]) if r[idx] else '0'}**" + (f" ({why})" if why else f" ({share:.1%})"))
    node(f"cap-{key}l", x0, y + BAR_H + 6, BAR_W, 64, " · ".join(legend))


def save(name):
    json.dump(dict(nodes=nodes, edges=edges), open(os.path.join(OUT, name), "w"), indent=1, ensure_ascii=False)


# ====================================================================================== model.canvas
B = budget()
bar("bp", -900, -1660, "Where the parameters are stored (from safetensors shapes)", B, 2, TOTAL)
bar("bc", -900, -1470, "What one token is multiplied by (active weights)", B, 3, ACTIVE)

node("cap-title", -900, -1300, 1250, 200,
     "# K2-Horizon-375B-A23B — model\n"
     f"Read top → bottom, left column (L0–{(N + 1) // 2 - 1}) then right column (L{(N + 1) // 2}–{N - 1}). "
     f"**Grey bars = tensors**, length ∝ channels per token ({KT} px/channel). "
     "**Boxes = operations**, all the same size; where parameters and compute live is the pair of bars above. "
     "T = sequence length. *Layer* tab = one MoE layer in detail.")
node("cap-legend", 450, -1300, 1250, 200,
     f"Layer rows — **dense**: SwiGLU FFN {DIM}→{DI}→{DIM} (L{min(DENSE)}–{max(DENSE)}, `mlp_only_layers`) · "
     f"**MoE**: sigmoid router picks {K} of {E} experts ({DIM}→{MI}→{DIM}) + {NSH} always-on shared expert.  \n"
     f"Every layer has the same attention: plain GQA, {NH} query / {NKV} KV heads × {HD}, RoPE on {RD} of {HD} "
     f"dims, full causal over {CTX:,} tokens (no sliding window, no MLA, no linear attention).  \n"
     f"Stored **{fmt(TOTAL)}** ({fmt(TOTAL - P['embed'] - P['head'])} without embed + head ≈ the name's 375B); "
     f"active **{fmt(ACTIVE)}** per token ({fmt(ACTIVE_NONEMB)} without lm head ≈ the name's A23B).")

# input flow (above the left column)
f = Flow(L_X + ROW_W / 2, -980)
f.op("**raw text**  \"The cat sat…\"", ACT, key="raw")
f.op(f"**tokenizer** (vocab {VOCAB:,})", ACT, key="tok")
f.tensor(40 / KT, "token ids [T] — one integer per token", key="ids")
f.op("**embed_tokens** — look up row *id*", EMB, P["embed"], note=f"{VOCAB} × {DIM}", key="embed")
f.tensor(DIM, f"x [T, {DIM}]", key="x")
ROW0 = f.y + 40
HALF = (N + 1) // 2


def row_xy(i):
    col, r = divmod(i, HALF)
    return (L_X if col == 0 else R_X), ROW0 + r * (ROW_H + ROW_GAP)


prev = f.last
for i in range(N):
    x, y = row_xy(i)
    dense = i in DENSE
    txt = f"**L{i}** · " + (f"dense FFN {DI}" if dense else f"MoE {K}/{E} + shared")
    nid = node(f"L{i}", x, y, ROW_W, ROW_H, txt, DENSE_C if dense else MOE)
    if i == HALF:
        _, yl = row_xy(HALF - 1)
        hand = Flow(L_X + ROW_W / 2, yl + ROW_H + 60, last=prev, labels_left=True)
        prev = hand.tensor(DIM, f"x [T, {DIM}]  after L{HALF - 1} → L{HALF}", key="hm")
        edge(prev, nid, ("right", "left"))
    else:
        edge(prev, nid)
    prev = nid

for col, (lo, hi) in enumerate(((0, HALF), (HALF, N))):
    x, y0 = row_xy(lo)
    _, y1 = row_xy(hi - 1)
    node(f"band{col}", x - 16, y0 - 30, ROW_W + 32, y1 - y0 + ROW_H + 46, f"DECODER L{lo}–{hi - 1}",
         "#38bdf8" if col == 0 else "#a3e635", group=True)

# per-layer anatomy notes, outside the columns (the gap between columns carries the hand-off edge)
_, y0 = row_xy(0)
node("note-layer", L_X - 560, y0 - 30, 500, 250,
     "**Inside every layer** (pre-norm residual):\n"
     "x += o_proj(GQA(RMSNorm(x)))\n"
     "x += FFN(RMSNorm(x))\n\n"
     f"Attention ≈ **{fmt(ATTN_L)}** / layer (q, o {DIM}×{DIM}; k, v {NKV * HD}×{DIM}).\n"
     f"Dense FFN ≈ **{fmt(DENSE_L)}** / layer.\n"
     f"MoE ≈ **{fmt(E * EXPERT + EXPERT * NSH)}** / layer stored, {fmt(K * EXPERT + EXPERT * NSH)} active "
     f"({E} experts × {fmt(EXPERT)}, + shared).")
_, y3 = row_xy(len(DENSE) + 3)
node("note-router", L_X - 560, y3, 500, 200,
     "**Router** (`mlp.gate`, has a bias):\n"
     f"score = sigmoid(W·x) over {E} experts;\n"
     f"top-{K} picked by score **+ bias** (bias steers selection only);\n"
     f"weights = picked scores, normalised to sum 1, × {C['router_scaling_factor']}.\n"
     f"Shared expert ({NSH} × {MI}) runs on every token.")
_, yr = row_xy(HALF)
node("note-kv", R_X + ROW_W + 40, yr - 30, 480, 210,
     f"**KV cache**: K and V for {NKV} heads × {HD} per layer = {2 * NKV * HD:,} values / token / layer.\n"
     f"All {N} layers, BF16: **{KV_PER_TOKEN / 1024:.0f} KiB per token** → "
     f"{KV_PER_TOKEN * CTX / 1e9:.0f} GB for one {CTX // 1024}K-token context.\n"
     f"RoPE θ = {THETA:.0e}, only on {RD} of {HD} head dims; the other {HD - RD} carry no position.")

# output flow (below the right column)
_, ylast = row_xy(N - 1)
o = Flow(R_X + ROW_W / 2, ylast + ROW_H + 60, last=f"L{N - 1}")
o.tensor(DIM, f"x [T, {DIM}]  after L{N - 1}", key="xo")
o.op("**final RMSNorm** (`model.norm`)", NORM, key="fn")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm_head** — {DIM} → {VOCAB}", HEAD, P["head"], note="untied", key="head")
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op("**softmax → sample**", ACT, key="samp")
o.op("**next token id**", ACT, key="next")
node("note-src", R_X + ROW_W / 2 + 280, o.y - 70 - 96, 640, 150,
     "**Sources**: config.json + modeling_k2_horizon.py + safetensors headers of "
     f"IFM/K2-Horizon-375B-A23B ({len(S):,} tensors, all BF16). No MTP / draft head and no vision tower in the "
     "checkpoint. Parameter counts are computed from tensor shapes, not taken from the model card.")

save("model.canvas")
model_counts = (len(nodes), len(edges))

# ====================================================================================== block.canvas
nodes.clear()
edges.clear()
_seq[0] = 0
QX, KX, VX = -900, 0, 1000  # q / k / v columns (also router / experts / shared)
BRANCH = 150  # extra drop before a fan-out so the curves clear the middle column

node("cap-title", -1300, -560, 1250, 170,
     f"# One MoE layer (L{len(DENSE)}–{N - 1}) — dataflow with shapes\n"
     f"Grey bars = tensors (length ∝ channels, {KT} px/channel); boxes = operations with their weight count. "
     f"T = sequence length. L{min(DENSE)}–{max(DENSE)} are identical except the FFN (see bottom note).")
node("cap-legend", 0, -560, 1150, 170,
     f"Attention = plain **GQA**: {NH} query heads share {NKV} KV heads ({NH // NKV} queries per KV head), "
     f"head_dim {HD}. **Partial RoPE**: the first {RD} dims of each q/k head (interleaved layout) are rotated, "
     f"the last {HD - RD} pass through unchanged. No QK-norm, no output gate, no sliding window (config).")

m = Flow(KX, -300)
xin = m.tensor(DIM, f"x [T, {DIM}]  from previous layer", key="xin")
m.op(f"**input_layernorm** (RMSNorm)\n{DIM} weights", NORM, key="n1")
xn = m.tensor(DIM, f"x̂ [T, {DIM}]", key="xn")
ybr = m.y + BRANCH

q = Flow(QX, ybr, last=xn, labels_left=True)
q.op(f"**q_proj** {DIM} → {NH}×{HD}", ATTN, NH * HD * DIM, key="q")
q.tensor(NH * HD, f"q [T, {NH}, {HD}]", key="qt")
q.op(f"**RoPE** on dims 0–{RD - 1}\n(θ = {THETA:.0e}); {RD}–{HD - 1} untouched", ATTN, key="qr")
qr = q.tensor(NH * HD, f"q [T, {NH}, {HD}]", key="qrt")

k = Flow(KX, ybr, last=xn)
k.op(f"**k_proj** {DIM} → {NKV}×{HD}", ATTN, NKV * HD * DIM, key="k")
k.tensor(NKV * HD, f"k [T, {NKV}, {HD}]", key="kt")
k.op(f"**RoPE** on dims 0–{RD - 1}", ATTN, key="kr")
kr = k.tensor(NKV * HD, f"k [T, {NKV}, {HD}] → KV cache", key="krt")

v = Flow(VX, ybr, last=xn)
v.op(f"**v_proj** {DIM} → {NKV}×{HD}", ATTN, NKV * HD * DIM, key="v")
vt = v.tensor(NKV * HD, f"v [T, {NKV}, {HD}] → KV cache", key="vt")

m.y = k.y
m.last = None
att = m.op(f"**causal GQA attention**\nsoftmax(q·kᵀ / √{HD}) · v, full context", ATTN, key="att", h=70, w=380)
edge(qr, att)
edge(kr, att)
edge(vt, att, ("bottom", "right"))
m.tensor(NH * HD, f"attn [T, {NH}×{HD} = {NH * HD}]", key="ao")
m.op(f"**o_proj** {NH * HD} → {DIM}", ATTN, DIM * NH * HD, key="o")
m.tensor(DIM, f"[T, {DIM}]", key="ot")
add1 = m.op("**+ residual**", ACT, key="add1", w=560)  # wider than the bars so the skip edge lands clear of them
edge(xin, add1, ("left", "left"), "residual", ACT)
m.labels_left = True  # the second skip edge leaves h1 on the right
h1 = m.tensor(DIM, f"h [T, {DIM}]", key="h1")
m.op(f"**post_attention_layernorm**\n{DIM} weights", NORM, key="n2")
hn = m.tensor(DIM, f"ĥ [T, {DIM}]", key="hn")
m.labels_left = False
ymoe = m.y + BRANCH

r = Flow(QX, ymoe, last=hn, labels_left=True)
r.op(f"**router** {DIM} → {E}, sigmoid", ROUTER_C, E * DIM + E, key="r")
r.tensor(E, f"scores [T, {E}]", key="rs")
r.op(f"**top-{K}** of score + bias\nnormalise, × {C['router_scaling_factor']}", ROUTER_C, key="rk")
rsel = r.tensor(40 / KT, f"{K} expert ids + weights per token", key="rw")

sh = Flow(VX, ymoe, last=hn)
sh.op(f"**shared expert** {DIM}→{MI * NSH}→{DIM}", SHARED_C, EXPERT * NSH, key="sh")
sht = sh.tensor(DIM, f"[T, {DIM}]", key="sht")

ex = Flow(KX, ymoe, last=hn, labels_left=True)  # right side is the shared-expert column
ex.op(f"**gate, up** of the {K} picked\nexperts: {DIM} → {MI} each", MOE, key="eg")
ex.tensor(MI, f"[T, {K}, {MI}]  SiLU(gate) ⊙ up", key="et")
ex.op(f"**down_proj** {MI} → {DIM}", MOE, key="ed")
ex.tensor(DIM, f"[T, {K}, {DIM}]", key="eo")
wsum = ex.op(f"**weighted sum** over the {K}", MOE, key="ws")
edge(rsel, wsum, ("bottom", "left"), color=ROUTER_C)
ws_t = ex.tensor(DIM, f"routed [T, {DIM}]", key="wst")
node("note-experts", VX + 180, sh.y + 40, 520, 130,
     f"**{E} routed experts** per layer, each SwiGLU {DIM}→{MI}→{DIM} = {fmt(EXPERT)}; "
     f"{fmt(E * EXPERT)} stored per layer, only {K} run per token ({fmt(K * EXPERT)}).")

m.y = max(ex.y, sh.y)
m.last = None
s = m.op("**routed + shared**", MOE, key="sum")
edge(ws_t, s)
edge(sht, s, ("bottom", "right"))
add2 = m.op("**+ residual**", ACT, key="add2", w=560)
edge(h1, add2, ("right", "right"), "residual", ACT)
m.tensor(DIM, f"x [T, {DIM}]  to next layer", key="xout")
node("note-dense", KX - 640, m.y + 10, 1280, 110,
     f"**Dense layers L{min(DENSE)}–{max(DENSE)}**: same attention; the router / experts / shared expert above are replaced by "
     f"one SwiGLU MLP {DIM} → {DI} → {DIM} ({fmt(DENSE_L)}). Layer total: attention {fmt(ATTN_L)} + "
     f"MoE {fmt(E * EXPERT + EXPERT * NSH + E * DIM + E)} stored / "
     f"{fmt(K * EXPERT + EXPERT * NSH + E * DIM + E)} active.")

save("block.canvas")
print("model.canvas", model_counts, "block.canvas", (len(nodes), len(edges)))
print("total", fmt(TOTAL), "active", fmt(ACTIVE), "active non-emb", fmt(ACTIVE_NONEMB),
      "stored non-emb", fmt(TOTAL - P["embed"] - P["head"]), "kv/token KiB", KV_PER_TOKEN / 1024)
print([(r[0], fmt(r[2]), fmt(r[3])) for r in B])
