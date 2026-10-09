"""Tencent Hy4-preview (HYV4): model-level canvas (all 78 layers + MTP) and a one-layer detail canvas.

Read top → bottom:
  - grey bars   = tensors between ops; bar length ∝ channels per token
  - boxes       = operations, all one size; parameter / compute shares are the two bars at the top
  - layer rows  = one per layer, coloured by FFN type and DSA indexer type (config.json)
  - note-* ids  = annotations (dim, borderless); cap-* ids = borderless captions
Numbers come from the checkpoint: shapes.json (safetensors headers of tencent/Hy4-preview) + config.json.
Dataflow follows transformers' models/hy_v4/modular_hy_v4.py; the MTP layer is not implemented there
(its weights are skipped on load), so the MTP branch is read off its tensor names.

usage: python3 hunyuan_hy4_preview_model_canvas.py <shapes.json> <config.json> <models/hunyuan-hy4-preview>
"""
import json
import math
import os
import re
import sys

S = json.load(open(sys.argv[1]))
C = json.load(open(sys.argv[2]))
OUT = sys.argv[3]

N, HALF = C["num_hidden_layers"], C["num_hidden_layers"] // 2
DIM, VOCAB, HC = C["hidden_size"], C["vocab_size"], C["hc_mult"]
NH, QR, KR = C["num_attention_heads"], C["q_lora_rank"], C["kv_lora_rank"]
NOPE, ROPE, VD = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
IH, ID, TOPK = C["index_n_heads"], C["index_head_dim"], C["index_topk"]
R, A, MI = C["n_routed_experts"], C["num_experts_per_tok"], C["moe_intermediate_size"]
MLP_T, IDX_T = C["mlp_layer_types"], C["indexer_types"]

# colours (JSON Canvas presets "1"–"6" or hex)
EMB, HCC, NORM, FFN, HEAD, ACT, ATT = "6", "1", "3", "4", "2", "#64748b", "5"
MTP, IDX, GATE = "#e879f9", "#f59e0b", "#38bdf8"
ROW_COLOR = {"dense": "#f97316", "full": "5", "shared": "#94a3b8"}

OP_W, OP_H, GAP = 300, 56, 40
LBL_W = 440
MIN_SEG = 16  # the viewer draws no box narrower than ~14px


def params_of(pattern):
    rx = re.compile(pattern)
    return sum(math.prod(shp) for k, (dt, shp) in S.items() if rx.fullmatch(k))


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else (f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K")


EXPERT = params_of(r"model\.layers\.1\.mlp\.experts\..*") // R  # one routed expert (gate_up + down)
LAYER_P = [params_of(rf"model\.layers\.{i}\..*") for i in range(N)]
MTP_P = params_of(r"model\.mtp_layers\..*")
N_MTP = C["num_nextn_predict_layers"]
P = {
    "embed": params_of(r"model\.embed_tokens\.weight"),
    "head": params_of(r"lm_head\.weight"),
    "layers": sum(LAYER_P),
    "ihc": params_of(r"model\.layers\.\d+\.hc_.*") + params_of(r"model\.hc_head\..*"),
    "indexer": params_of(r"model\.layers\.0\.self_attn\.indexer\..*"),
    "attn": params_of(r"model\.layers\.1\.self_attn\.(?!indexer\.).*"),
    "dense": params_of(r"model\.layers\.0\.mlp\..*"),
    "moe": params_of(r"model\.layers\.1\.mlp\..*"),
    "shared": params_of(r"model\.layers\.1\.mlp\.shared_experts\..*"),
    "router": params_of(r"model\.layers\.1\.mlp\.gate\..*"),
}
N_MOE = sum(t == "sparse" for t in MLP_T)


def budget():
    """[(name, colour, stored params, params multiplied per token, why-zero)] for the whole checkpoint.

    Per-token compute counts only the weights one token is multiplied by: the top-k routed experts
    (not all of them) and nothing for the embedding row lookup.
    """
    layers_active = P["layers"] - N_MOE * (R - A) * EXPERT
    mtp_active = MTP_P - N_MTP * (R - A) * EXPERT
    rows = [
        (f"{N} layers", FFN, P["layers"], layers_active),
        # speculative-decoding head: shown, but not summed into "active per text token"
        (f"MTP ×{N_MTP}", MTP, MTP_P, 0, f"draft layer, {fmt(mtp_active)} per draft step, not counted"),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, P["head"], P["head"]),
    ]
    other = params_of(r".*") - sum(r[2] for r in rows)
    if other >= 1e6:
        rows.append(("other", ACT, other, other))
    return rows


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

    def __init__(self, cx, y, last=None, labels_left=False, kt=0.08, clip=1800, lbl_w=LBL_W):
        self.cx, self.y, self.last, self.labels_left = cx, y, last, labels_left
        self.kt, self.clip, self.lbl_w = kt, clip, lbl_w

    def _id(self, key):
        _seq[0] += 1
        return f"{key}{_seq[0]}"

    def _link(self, top, bottom, label=None):
        if self.last:
            edge(self.last, top, label=label)
        self.last = bottom

    def op(self, text, color, params=0, note="", key="op", h=OP_H):
        if params:
            text += f"\n{fmt(params)}" + (f" · {note}" if note else "")
        nid = node(self._id(key), self.cx - OP_W / 2, self.y, OP_W, h, text, color)
        self._link(nid, nid)
        self.y += h + GAP
        return nid

    def tensor(self, dim, label, lanes=1, key="t"):
        real = dim * self.kt
        w = min(real, self.clip)
        ids = [node(self._id(key), self.cx - w / 2, self.y + j * 14, w, 9, "", ACT) for j in range(lanes)]
        clip = f" — clipped, real {real:.0f}px" if real > w else ""
        half = max(w / 2, OP_W / 2) + 20
        lx = self.cx - half - self.lbl_w if self.labels_left else self.cx + half
        node(self._id("note-lbl"), lx, self.y - 10, self.lbl_w, 34, f"`{label}`{clip}")
        self._link(ids[0], ids[-1])
        self.y += lanes * 14 + GAP
        return ids[0], ids[-1]

    def skip(self, dy):
        self.y += dy


def last_full(i):
    return max(k for k in range(i + 1) if IDX_T[k] == "full")


def row_kind(i):
    return "dense" if MLP_T[i] == "dense" else IDX_T[i]


def save(name):
    json.dump(dict(nodes=nodes, edges=edges), open(os.path.join(OUT, name), "w"), indent=1, ensure_ascii=False)
    nodes.clear()
    edges.clear()


# =====================================================================================  model.canvas
KT = 0.08
ROW_W, ROW_H, ROW_GAP = 300, 34, 6
ENC_X, DEC_X = 0, 900
BAR_W, BAR_H = 2600, 44

n_full = IDX_T.count("full")
node("cap-title", -900, -1270, 1250, 200,
     "# Tencent Hy4-preview — model\n"
     f"Read top → bottom, left column (L0–{HALF - 1}) then right column (L{HALF}–{N - 1}). "
     f"**Grey bars = tensors**, length ∝ channels per token ({KT} px/channel). "
     "**Boxes = operations**, all one size; where parameters and compute live is the pair of bars above. "
     f"T = sequence length (up to {C['max_position_embeddings']:,} = 1M). *Layer* tab = one layer in detail.  \n"
     "Sources: config.json + safetensors headers of tencent/Hy4-preview; dataflow from transformers `modular_hy_v4.py`.")

layer_kinds = {k: [LAYER_P[i] for i in range(N) if row_kind(i) == k] for k in ROW_COLOR}
node("cap-legend", 450, -1270, 1250, 200,
     f"Layer rows — **orange**: dense SwiGLU FFN ({C['intermediate_size']}) · **blue**: MoE + its own DSA indexer "
     f"(scores all past tokens, keeps top {TOPK}) · **grey**: MoE, **reuses** the top-{TOPK} picks of the last indexer "
     f"layer (IndexCache). {n_full} layers run an indexer, {N - n_full} reuse one.  \n"
     f"Every layer: Gated DSA attention (MLA, {NH} heads, q latent {QR}, kv latent {KR}) ≈ **{fmt(P['attn'])}** "
     f"(+{fmt(P['indexer'])} indexer) · MoE {R} experts top-{A} + 1 shared ≈ **{fmt(P['moe'])}** "
     f"({R * EXPERT / P['moe']:.1%} in routed experts). All {N} layers = **{fmt(P['layers'])}**.")

# ---------- input flow ----------
cx_enc = ENC_X + ROW_W / 2
f = Flow(cx_enc, -1000)
f.op("**raw text**  \"The cat sat…\"", ACT, key="raw")
f.op(f"**tokenizer** (BPE, vocab {VOCAB})", ACT, key="tok")
f.tensor(40 / KT, "token ids [T] — one integer per token", key="ids")
emb_op = f.op("**embed** — look up row *id*", EMB, P["embed"], note=f"table {VOCAB} × {DIM}", key="embed")
f.tensor(DIM, f"x [T, {DIM}]", key="x")
f.op(f"**expand ×{HC}** — copy into {HC} residual streams (iHC)", HCC, key="hcx")
f.tensor(DIM, f"h [T, {HC}, {DIM}]", lanes=HC, key="h")
ROW0 = f.y + 30


def row_xy(i):
    col, r = divmod(i, HALF)
    return (ENC_X if col == 0 else DEC_X), ROW0 + r * (ROW_H + ROW_GAP)


# ---------- layer rows ----------
prev = f.last
for i in range(N):
    x, y = row_xy(i)
    k = row_kind(i)
    ffn = "dense FFN" if MLP_T[i] == "dense" else "MoE"
    idx = "own indexer" if IDX_T[i] == "full" else f"reuses L{last_full(i)} top-k"
    nid = node(f"L{i}", x, y, ROW_W, ROW_H, f"**L{i}** · {ffn} · {idx}", ROW_COLOR[k])
    if i == HALF:
        _, yl = row_xy(HALF - 1)
        hand = Flow(ENC_X + ROW_W / 2, yl + ROW_H + 60, last=prev, labels_left=True)
        _, prev = hand.tensor(DIM, f"h [T, {HC}, {DIM}]  after L{HALF - 1} → L{HALF}", lanes=HC, key="hm")
        edge(prev, nid, ("right", "left"))
    else:
        edge(prev, nid)
    prev = nid

for col, (lo, hi) in enumerate(((0, HALF), (HALF, N))):
    x, y0 = row_xy(lo)
    _, y1 = row_xy(hi - 1)
    node(f"band{col}", x - 16, y0 - 30, ROW_W + 32, y1 - y0 + ROW_H + 46, f"L{lo}–{hi - 1}", "#a3e635", group=True)

_, ymid = row_xy(4)
node("note-rows", ENC_X - 500, ymid, 440, 210,
     f"Each layer = 2 sub-blocks (attention, FFN). Each sub-block reads the {HC} residual streams through iHC: "
     f"a learned sigmoid mix of the {HC} streams → one {DIM} vector → RMSNorm → sub-block → added back to every "
     f"stream with its own sigmoid weight (×{C['hc_magnitude']:g}). Identity residual; no stream-mixing matrix.")
_, ymid = row_xy(HALF + 4)
node("note-rows2", DEC_X + ROW_W + 40, ymid, 440, 160,
     f"Indexer pattern (config `indexer_types`): L0 and every 4th layer from L1 (L1, L5, … L{N - 1}) compute fresh "
     f"top-{TOPK} key positions; the 3 layers after each reuse them — no indexer weights in those layers.")

# ---------- output flow ----------
_, ylast = row_xy(N - 1)
o = Flow(DEC_X + ROW_W / 2, ylast + ROW_H + 60, last=f"L{N - 1}")
o.tensor(DIM, f"h [T, {HC}, {DIM}]  after L{N - 1}", lanes=HC, key="ho")
hch = o.op(f"**hc_head** — sigmoid-weighted sum of {HC} streams", HCC, key="hch")
o.op("**final RMSNorm**", NORM, key="norm")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm head** (fp32) — {DIM} → {VOCAB}", HEAD, P["head"], key="head")
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op("**softmax → sample**", ACT, key="samp")
next_id = o.op("**next token id**", ACT, key="next")

# ---------- MTP branch (speculative decoding) ----------
m = Flow(DEC_X + 2700, ylast + ROW_H + 60)
m.op("**embed** next token t+1\n(shared table) → enorm", EMB, key="memb")
mcat, _ = m.tensor(2 * DIM, f"concat [hnorm h_t, enorm e_t+1] [T, {2 * DIM}]", key="mcat")
edge(hch, mcat, ("right", "left"), "h of token t", MTP)
m.op(f"**eh_proj** {2 * DIM} → {DIM}", MTP, key="meh")
m.tensor(DIM, f"x [T, {DIM}]", key="mx")
m.op(f"**MTP decoder layer**\nGated DSA · MoE top-{A} of {R}", MTP, MTP_P, h=84, key="mblk")
m.op("**final_layernorm → lm head** (shared)", HEAD, key="mhead")
m.tensor(40 / KT, "draft token t+2", key="mdraft")
ver = m.op("**verify**: main model checks drafts\nin one forward pass", ACT, key="mver")
edge(ver, next_id, ("left", "right"), "accepted drafts", MTP)
node("note-mtp", m.cx - OP_W / 2, m.y - 10, 640, 120,
     f"{N_MTP} native MTP layer ({fmt(MTP_P)} stored, mostly its {R} experts). It has no embed / lm-head tensors of its "
     "own, so it uses the main ones. transformers skips these weights on load; vLLM / SGLang use them for "
     "speculative decoding (README: `num_speculative_tokens: 3`, method `mtp`). Wiring (hnorm / enorm / eh_proj) "
     "is read off the tensor names; the exact tap point of h is not in public code.")


def bar(key, y, title, rows, idx):
    total = sum(r[idx] for r in rows)
    node(f"cap-{key}t", -900, y - 46, BAR_W, 40, f"**{title}** — total {fmt(total)}")
    x = -900
    legend = []
    # segments too thin to draw go last, so any blank stretch is at the bar's end, not mid-bar
    for k in sorted(range(len(rows)), key=lambda k: BAR_W * rows[k][idx] / total < MIN_SEG):
        r = rows[k]
        share = r[idx] / total
        w = BAR_W * share
        if w >= MIN_SEG:
            node(f"{key}{k}", round(x), y, round(x + w) - round(x), BAR_H, f"**{r[0]}**" if w > 150 else "", r[1])
        x += w
        why = r[4] if len(r) > 4 and not r[idx] else ""
        legend.append(f"{r[0]} **{fmt(r[idx]) if r[idx] else '0'}**" + (f" ({why})" if why else f" ({share:.1%})"))
    node(f"cap-{key}l", -900, y + BAR_H + 6, BAR_W, 40, " · ".join(legend))


B = budget()
bar("bp", -1680, "Where the parameters are stored (safetensors, BF16)", B, 2)
bb = B[0][3] + B[3][3]  # backbone weights one token is multiplied by
bar("bc", -1500, f"What one token is multiplied by (top-{A} of {R} experts; backbone {fmt(bb)}, "
    f"{fmt(bb + P['embed'])} counting the embed row as README's 49B does)", B, 3)
save("model.canvas")

# =====================================================================================  block.canvas
KB = 0.04  # px per channel in the layer view
bk = dict(kt=KB, clip=900, lbl_w=400)
L = 1  # a MoE layer that owns an indexer
node("cap-title", -1900, -460, 1500, 150,
     f"# One layer in detail — L{L} (MoE, own indexer)\n"
     f"Grey bars = tensors (length ∝ channels, {KB} px/channel), boxes = ops with their weight shapes from the checkpoint. "
     f"T = tokens, K = cached past tokens. Layers marked *reuses top-k* skip the orange indexer column and take its "
     f"top-{TOPK} picks from the last indexer layer; L0 swaps the MoE for a dense SwiGLU FFN.")

top = Flow(0, -260, **bk)
top.tensor(DIM, f"h [T, {HC}, {DIM}] · {HC} streams", lanes=HC, key="bh")
top.op(f"**iHC pre (attn)** — fn {HC * 2}×{HC * DIM}\npre[{HC}] mixes streams · post[{HC}] kept", HCC, h=70, key="bhc")
top.tensor(DIM, f"x [T, {DIM}]", key="bx")
top.op("**input RMSNorm**", NORM, key="bn")
_, xn = top.tensor(DIM, f"x̂ [T, {DIM}] → 4 branches", key="bxn")

COL = {"idx": -2000, "q": -1200, "kv": -100, "g": 950}
y0 = top.y + 200
q = Flow(COL["q"], y0, last=xn, **bk)
q.op(f"**q_a_proj** {DIM} → {QR}", ATT, key="qa")
q.op("**q_a_layernorm**", NORM, key="qn")
_, qlat = q.tensor(QR, f"q latent [T, {QR}]", key="ql")
q.op(f"**q_b_proj** {QR} → {NH}×{NOPE + ROPE}", ATT, key="qb")
q.tensor(NH * (NOPE + ROPE), f"q [T, {NH}, {NOPE}+{ROPE} rope]", key="qq")
q.op(f"**RoPE** on the {ROPE}-dim slice", ATT, key="qr")
q_end = q.last

kv = Flow(COL["kv"], y0, last=xn, **bk)
kv.op(f"**kv_a_proj_with_mqa** {DIM} → {KR + ROPE}", ATT, key="kva")
kv.tensor(KR + ROPE, f"[T, {KR + ROPE}] = {KR} latent + {ROPE} rope", key="kvl")
kv.op(f"**kv_a_layernorm** ({KR}) · RoPE ({ROPE})\n→ KV cache: {KR + ROPE} / token", NORM, h=70, key="kvn")
kv.op(f"**kv_b_proj** {KR} → {NH}×({NOPE}+{VD})", ATT, key="kvb")
kv.tensor(NH * (NOPE + ROPE), f"k [K, {NH}, {NOPE}+{ROPE}] · v [K, {NH}, {VD}]", key="kvkv")
kv_end = kv.last

ix = Flow(COL["idx"], y0, last=xn, **bk)
ix.op(f"**wk** {DIM} → {ID} + LayerNorm\n**weights_proj** {DIM} → {IH}", IDX, h=70, key="ik")
ix.skip(60)
iq = ix.op(f"**wq_b** {QR} → {IH}×{ID}\n(from the q latent)", IDX, h=70, key="iq")
edge(qlat, iq, ("left", "right"), color=IDX)
ix.tensor(IH * ID, f"index q [T, {IH}, {ID}] · k [K, {ID}]", key="iqk")
ix.op(f"**score** = Σ_h w_h · ReLU(q_h·k)\nRoPE on last {ROPE} of {ID}", IDX, h=70, key="isc")
ix.op(f"**top-{TOPK}** key positions / query", IDX, key="itop")
ix_end = ix.last

g = Flow(COL["g"], y0, last=xn, **bk)
g.op(f"**gate_proj** {DIM} → {NH}×{VD}", GATE, key="gp")
g.tensor(NH * VD, f"gate [T, {NH}, {VD}]", key="gg")

ymerge = max(q.y, kv.y, ix.y) + 120
g.skip(ymerge - g.y)
g_end = g.op("**sigmoid** → (0, 1) per channel", GATE, key="gs")
a = Flow(COL["kv"] - 400, ymerge, labels_left=True, **bk)
att = a.op(f"**sparse MLA attention**\nsoftmax over top-{TOPK} keys + 1 learned sink logit / head", ATT, h=84, key="att")
edge(q_end, att, label=f"q [T, {NH}, {NOPE + ROPE}]")
edge(kv_end, att)
edge(ix_end, att, ("bottom", "left"), f"top-{TOPK} ids", IDX)
a.tensor(NH * VD, f"o [T, {NH}, {VD}]", key="ao")
gm = a.op("**o × gate** — elementwise output gate", GATE, key="agm")
edge(g_end, gm, ("bottom", "right"), color=GATE)
a.op(f"**o_proj** {NH * VD} → {DIM}", ATT, key="aop")
a.tensor(DIM, f"y [T, {DIM}]", key="ay")
a.op(f"**iHC post** — h_i += post_i · y\n(post ∈ (0, {C['hc_magnitude']:g}), identity residual)", HCC, h=70, key="apost")
a.tensor(DIM, f"h [T, {HC}, {DIM}]", lanes=HC, key="ah")
a.op(f"**iHC pre (FFN)** — fn {HC * 2}×{HC * DIM}", HCC, key="fhc")
a.op("**post_attention RMSNorm**", NORM, key="fn")
_, fx = a.tensor(DIM, f"x̂ [T, {DIM}] → 3 branches", key="fx")

y1 = a.y + 200
rt = Flow(a.cx - 900, y1, last=fx, labels_left=True, **bk)
rt.op(f"**router** {DIM} → {R}, sigmoid\n+ bias → top-{A}", FFN, h=70, key="rg")
rt.tensor(A * 40, f"{A} ids + weights (renorm × {C['routed_scaling_factor']})", key="rw")
rt_end = rt.last

ex = Flow(a.cx, y1, last=fx, **bk)
xg = ex.op(f"**{A} routed experts** of {R}\ngate_up {DIM} → 2×{MI}", FFN, h=70, key="xg")
ex.tensor(2 * MI, f"[T, {A}, {2 * MI}]", key="xgu")
ex.op(f"**SwiGLU**, clamp ±{C['swiglu_limit']:g}", FFN, key="xs")
ex.op(f"**down** {MI} → {DIM}, × weight, Σ {A}", FFN, key="xd")
ex_end = ex.last
edge(rt_end, xg, ("right", "left"), color=FFN)

sh = Flow(a.cx + 900, y1, last=fx, **bk)
sh.op(f"**shared expert** (always on)\nSwiGLU {DIM} → {MI} → {DIM}", FFN, h=70, key="se")
sh_end = sh.last

s = Flow(a.cx, ex.y + 80, **bk)
sm = s.op("**sum** routed + shared", FFN, key="sum")
edge(ex_end, sm)
edge(sh_end, sm, ("bottom", "right"))
s.tensor(DIM, f"y [T, {DIM}]", key="sy")
s.op("**iHC post (FFN)** — h_i += post_i · y", HCC, key="spost")
s.tensor(DIM, f"h [T, {HC}, {DIM}] → next layer", lanes=HC, key="sh")

node("note-params", a.cx + 700, ymerge, 560, 150,
     f"Params of L{L}: attention {fmt(P['attn'])} + indexer {fmt(P['indexer'])} · MoE {fmt(P['moe'])} "
     f"({R} × {fmt(EXPERT)} experts, shared {fmt(P['shared'])}, router {fmt(P['router'])}) · iHC "
     f"{fmt(params_of(rf'model\.layers\.{L}\.hc_.*'))}. Per token only {A} experts run.")
save("block.canvas")
print("ok", [(r[0], fmt(r[2]), fmt(r[3])) for r in B], "expert", fmt(EXPERT))
