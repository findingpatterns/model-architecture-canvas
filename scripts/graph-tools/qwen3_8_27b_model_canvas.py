"""Qwen3.8-27B: model-level canvas (all 64 layers + vision + MTP) and layer-level canvas (both token mixers).

Read top → bottom:
  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale everywhere)
  - boxes       = operations, all one size; parameter shares are the two bars at the top
  - layer rows  = one per layer, coloured by token mixer (config.json text_config.layer_types)
  - note-* ids  = annotations (styled dim, borderless); cap-* ids = borderless captions
Numbers come from config.json + shapes.json (safetensors headers of Qwen/Qwen3.8-27B); the dataflow
follows transformers' modeling_qwen3_5.py, the MTP head follows vLLM's qwen3_5_mtp.py.

usage: python3 qwen3_8_27b_model_canvas.py <shapes.json> <config.json> <models/qwen3-8-27b>
"""
import json
import math
import os
import re
import sys

S = json.load(open(sys.argv[1]))
CFG = json.load(open(sys.argv[2]))
OUT = sys.argv[3]
C, V = CFG["text_config"], CFG["vision_config"]

N = C["num_hidden_layers"]
TYPES = C["layer_types"]
DIM, VOCAB, FF = C["hidden_size"], C["vocab_size"], C["intermediate_size"]
HQ, HKV, HD = C["num_attention_heads"], C["num_key_value_heads"], C["head_dim"]
LK, LV, LKD, LVD = C["linear_num_key_heads"], C["linear_num_value_heads"], C["linear_key_head_dim"], C["linear_value_head_dim"]
ROPE = C["rope_parameters"]
ROT = int(HD * ROPE["partial_rotary_factor"])
KEY_DIM, VAL_DIM = LK * LKD, LV * LVD
LM = "model.language_model."

# colours (JSON Canvas presets "1"–"6" or hex)
EMB, NORM, FFN, HEAD, ACT = "6", "3", "4", "2", "#64748b"
DN, ATT, MTP, VISION = "#22d3ee", "#f97316", "#f59e0b", "#a78bfa"
TYPE_COLOR = {"linear_attention": DN, "full_attention": ATT}
TYPE_NAME = {"linear_attention": "Gated DeltaNet", "full_attention": "Gated Attention"}

# geometry
KT, CLIP_W = 0.08, 1800  # tensor bars: px per channel, longest drawn bar
ROW_W = 300
ENC_X = 0  # one column: a decoder-only stack; repeated layers are drawn once with ×N (see segments())
OP_W, OP_H, GAP = 300, 56, 40
BAR_W, BAR_H = 2600, 44
LBL_W = 440
MIN_SEG = 16  # thinner budget segments would be drawn wider than their share and overlap


def params_of(pattern):
    rx = re.compile(pattern)
    return sum(math.prod(shp) for k, (dt, shp) in S.items() if rx.fullmatch(k))


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


P = {
    "embed": params_of(LM + r"embed_tokens\.weight"),
    "head": params_of(r"lm_head\.weight"),
    "dn": params_of(LM + r"layers\.\d+\.linear_attn\..*"),
    "att": params_of(LM + r"layers\.\d+\.self_attn\..*"),
    "ffn": params_of(LM + r"layers\.\d+\.mlp\..*"),
    "vit": params_of(r"model\.visual\.(?!merger\.).*"),
    "merger": params_of(r"model\.visual\.merger\..*"),
    "mtp": params_of(r"mtp\..*"),
}
LAYER_P = [params_of(LM + rf"layers\.{i}\..*") for i in range(N)]
N_DN, N_ATT = TYPES.count("linear_attention"), TYPES.count("full_attention")
I_DN, I_ATT = TYPES.index("linear_attention"), TYPES.index("full_attention")


def budget():
    """[(name, colour, stored params, params multiplied per text token[, why zero])]."""
    rows = [
        (f"{N_DN} DeltaNet mixers", DN, P["dn"], P["dn"]),
        (f"{N_ATT} attention mixers", ATT, P["att"], P["att"]),
        (f"{N} SwiGLU FFNs", FFN, P["ffn"], P["ffn"]),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, P["head"], P["head"]),
        ("MTP", MTP, P["mtp"], 0, "only when drafting"),
        ("ViT + merger", VISION, P["vit"] + P["merger"], 0, "runs per image"),
    ]
    other = params_of(r".*") - sum(r[2] for r in rows)
    rows.append(("norms", NORM, other, other))  # RMSNorm weights: every param is accounted for
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

    def __init__(self, cx, y, last=None, labels_left=False):
        self.cx, self.y, self.last, self.labels_left = cx, y, last, labels_left

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

    def tensor(self, dim, label, key="t"):
        real = dim * KT
        w = min(real, CLIP_W)
        nid = node(self._id(key), self.cx - w / 2, self.y, w, 9, "", ACT)
        clip = f" — bar clipped (real {real:.0f}px)" if real > w else ""
        half = max(w / 2, OP_W / 2) + 20
        lx = self.cx - half - LBL_W if self.labels_left else self.cx + half
        node(self._id("note-lbl"), lx, self.y - 10, LBL_W, 34, f"`{label}`{clip}")
        self._link(nid, nid)
        self.y += 14 + GAP
        return nid


def bar(key, x0, y, title, rows, idx):
    total = sum(r[idx] for r in rows)
    node(f"cap-{key}t", x0, y - 46, BAR_W, 40, f"**{title}** — total {fmt(total)}")
    x, legend = x0, []
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
    node(f"cap-{key}l", x0, y + BAR_H + 6, BAR_W, 40, " · ".join(legend))


def save(name):
    json.dump(dict(nodes=nodes, edges=edges), open(os.path.join(OUT, name), "w"), indent=1, ensure_ascii=False)
    nodes.clear()
    edges.clear()


# KV cache vs recurrent state, from config (bf16 KV, fp32 DeltaNet state per mamba_ssm_dtype)
CTX = C["max_position_embeddings"]
KV_TOK = 2 * HKV * HD  # values per token per attention layer
STATE = LV * LKD * LVD  # values per DeltaNet layer, independent of T
kv_gb = N_ATT * KV_TOK * CTX * 2 / 1e9
st_mb = N_DN * STATE * 4 / 1e6

# =================================== model.canvas ===================================
node("cap-title", -900, -1270, 1250, 180,
     "# Qwen3.8-27B — model\n"
     f"One continuous decoder-only stack L0–{N - 1}, read top → bottom; layers that repeat are drawn once with **×N**. "
     f"**Grey bars = tensors**, length ∝ channels per token ({KT} px/channel). "
     "**Boxes = operations**, all the same size; where the parameters live is the pair of bars above. "
     "T = sequence length. Zoom in: *Layer* tab = both token mixers + FFN.")
node("cap-legend", 450, -1270, 1250, 180,
     f"Layer rows — **Gated DeltaNet** (cyan): linear attention with a fixed {LV}×{LKD}×{LVD} state, no KV cache · "
     f"**Gated Attention** (orange): every {C['full_attention_interval']}th layer, GQA {HQ}q/{HKV}kv heads × {HD}, "
     f"KV cache grows with T. Hidden layout = {N // C['full_attention_interval']} × (3 DeltaNet + 1 attention), "
     f"every layer followed by the same dense SwiGLU FFN ({DIM} → {FF} → {DIM}).  \n"
     f"DeltaNet layer **{fmt(LAYER_P[I_DN])}**, attention layer **{fmt(LAYER_P[I_ATT])}**; "
     f"all {N} = **{fmt(sum(LAYER_P))}**. Dense: every token uses every layer weight. "
     "The barcode on the right lists every layer in order.")

cx_enc = ENC_X + ROW_W / 2
f = Flow(cx_enc, -960)
f.op("**raw text**  \"The cat sat…\"", ACT, key="raw")
f.op(f"**tokenizer** (BPE, vocab {VOCAB})", ACT, key="tok")
f.tensor(40 / KT, "token ids [T] — one integer per token", key="ids")
f.op("**embed** — look up row *id*", EMB, P["embed"], note=f"table {VOCAB} × {DIM}", key="embed")
x0 = f.tensor(DIM, f"x [T, {DIM}]  (+ image/video tokens spliced in)", key="x")
ROW0 = f.y + 30


def kind(i):
    return TYPES[i]


def runs(start, stop):
    """Consecutive layers of the same kind → [("run", first, count)]."""
    out = []
    for i in range(start, stop):
        if out and kind(i) == kind(out[-1][1]):
            out[-1] = ("run", out[-1][1], out[-1][2] + 1)
        else:
            out.append(("run", i, 1))
    return out


def segments():
    """Run-length prefix + the longest periodic tail (period ≥ 2, repeated ≥ 2×), so a repeating stack is drawn once."""
    for s0 in range(N):
        for p in range(2, 9):
            if (N - s0) % p == 0 and (N - s0) // p >= 2 and all(kind(i) == kind(s0 + (i - s0) % p) for i in range(s0, N)):
                return runs(0, s0) + [("period", s0, p, (N - s0) // p)]
    return runs(0, N)


SEG_W, SEG_H, SEG_GAP = 460, 56, 26
CX = cx_enc
GX = CX - SEG_W / 2 - 60  # left edge of the ×N group frame
GW = SEG_W + 120
y = ROW0
prev = f.last
y_group = ROW0
for seg in segments():
    if seg[0] == "run":
        _, a, n = seg
        span = f"**L{a}–{a + n - 1}** ×{n}" if n > 1 else f"**L{a}**"
        nid = node(f"seg{a}", CX - SEG_W / 2, y, SEG_W, SEG_H, f"{span}\n{TYPE_NAME[kind(a)]} + FFN", TYPE_COLOR[kind(a)])
        edge(prev, nid)
        prev = nid
        y += SEG_H + SEG_GAP
        continue
    _, s0, p, reps = seg
    y += 40  # room for the group's label, which the viewer draws above the frame
    y_group = y
    ry = y + 50
    first = None
    for j in range(p):
        ls = [s0 + j + r * p for r in range(reps)]
        nid = node(f"per{j}", CX - SEG_W / 2, ry, SEG_W, SEG_H,
                   f"{TYPE_NAME[kind(s0 + j)]} + FFN\nL{ls[0]}, {ls[1]}, …, {ls[-1]}", TYPE_COLOR[kind(s0 + j)])
        edge(prev, nid)
        first = first or nid
        prev = nid
        ry += SEG_H + SEG_GAP
    edge(prev, first, ("left", "left"))  # loop back: the period runs again
    node("cap-repeat", GX - 170, (y + 50 + ry - SEG_GAP) / 2 - 30, 150, 60, f"## ↺ ×{reps}")
    gh = ry - y + 10
    node("grp-period", GX, y, GW, gh, f"×{reps} — L{s0}–{N - 1}, a period of {p} layers", group=True)
    y += gh + SEG_GAP
LAST = prev

# barcode: every layer in order, one thin cell each, coloured like the rows above
CELL, CELL_GAP = 16, 2
BX, BY = GX + GW + 120, ROW0 + 40
node("cap-barcode", BX, BY - 50, N * (CELL + CELL_GAP), 40, f"**Every layer, L0 → L{N - 1}** (1 cell = 1 layer)")
for i in range(N):
    node(f"lyr{i}", BX + i * (CELL + CELL_GAP), BY, CELL, 44, "", TYPE_COLOR[kind(i)])
for i in sorted({*range(0, N, 10), N - 1} - {i for i in range(N - 4, N - 1) if i % 10 == 0}):  # no tick crowding the last one
    node(f"note-tick{i}", BX + i * (CELL + CELL_GAP) - 4, BY + 50, 60, 30, f"L{i}")

node("note-mixers", GX - 170 - 40 - 440, y_group, 440, 300,
     f"**Why mix the two?** A DeltaNet layer keeps one {LKD}×{LVD} matrix per head ({LV} heads) "
     f"that it updates token by token — memory and compute per token stay constant however long the context is.  \n"
     f"An attention layer stores K and V of every past token ({KV_TOK} values/token/layer).  \n"
     f"At {CTX:,} tokens: {N_ATT} attention layers ≈ **{kv_gb:.1f} GB** KV (bf16) vs "
     f"{N_DN} DeltaNet layers ≈ **{st_mb:.0f} MB** state (fp32).")
node("note-rope", BX, BY + 120, 440, 250,
     f"**Position**: only attention layers use RoPE — partial ({ROT} of {HD} dims), "
     f"multimodal (interleaved t/h/w sections {'/'.join(map(str, ROPE['mrope_section']))}), θ = {ROPE['rope_theta']:.0e}. "
     f"DeltaNet layers get order from their causal conv ({C['linear_conv_kernel_dim']} taps) and recurrence.  \n"
     f"Context {CTX:,} native; the model card gives YaRN ×4 to reach ~1M.")

# ---------- vision branch, spliced into x ----------
v = Flow(ENC_X - 480, -960, labels_left=True)
v.op("**image / video** frames", ACT, key="img")
pd = V["in_channels"] * V["temporal_patch_size"] * V["patch_size"] ** 2
v.tensor(pd, f"patches [P, {pd}] = 3 × {V['temporal_patch_size']} frames × {V['patch_size']}²", key="patch")
v.op(f"**ViT ×{V['depth']}** (+ learned pos-embed)", VISION, P["vit"], note=f"width {V['hidden_size']}", key="vit")
v.tensor(V["hidden_size"], f"[P, {V['hidden_size']}]", key="vf")
r = V["spatial_merge_size"]
v.op(f"**merger MLP** — merge {r}×{r} patches", VISION, P["merger"], key="mrg")
v.tensor(V["out_hidden_size"], f"image tokens [P/{r * r}, {V['out_hidden_size']}]", key="vt")
edge(v.last, x0, ("right", "left"), "spliced into x")

# ---------- output flow (below the right column) ----------
o = Flow(CX, y + 40, last=LAST, labels_left=True)
o.tensor(DIM, f"h [T, {DIM}]  after L{N - 1}", key="ho")
fnorm = o.op("**final RMSNorm**", NORM, key="fn")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm head** — {DIM} → {VOCAB}", HEAD, P["head"], note="not tied", key="head")
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op("**softmax → sample**", ACT, key="samp")
o.op("**next token id**", ACT, key="next")

# ---------- MTP: one extra full-attention layer predicting one more token ----------
m = Flow(CX + 1300, y + 140)
cat = m.tensor(2 * DIM, f"[T, {2 * DIM}] = norm(embed(token t+1)) ‖ norm(x)", key="mcat")
edge(fnorm, cat, ("right", "top"), "normed x", MTP)
m.op(f"**fc** {2 * DIM} → {DIM}", MTP, params_of(r"mtp\.fc\..*"), key="mfc")
m.tensor(DIM, f"[T, {DIM}]", key="mh")
m.op(f"**1 Gated Attention layer** + FFN", MTP, params_of(r"mtp\.layers\..*"), key="mly")
m.op("**RMSNorm → shared lm head**", MTP, key="mhd")
m.op("**draft token t+2**", ACT, key="mdr")
node("note-mtp", m.cx - OP_W / 2, m.y - 20, 560, 130,
     f"Multi-token prediction head ({fmt(P['mtp'])}, {C['mtp_num_hidden_layers']} layer). It reuses the main embed table "
     "and lm head (no own copies in the checkpoint). Serving engines run it as a draft model for speculative "
     "decoding; the main model then verifies the draft. Wiring per vLLM qwen3_5_mtp.py.")

B = budget()
bar("bp", -900, -1640, "Where the parameters are stored", B, 2)
bar("bc", -900, -1460, "What one text token is multiplied by (active weights)", B, 3)
save("model.canvas")

# =================================== block.canvas ===================================
_seq[0] = 0
LX, RX = -1100, 1100  # DeltaNet column / attention column centres
node("cap-btitle", -1400, -560, 1300, 150,
     "# Qwen3.8-27B — one layer\n"
     f"Every layer: RMSNorm → **token mixer** → +residual → RMSNorm → **SwiGLU FFN** → +residual. "
     f"Left mixer is used by {N_DN} layers (3 of every 4), right mixer by {N_ATT}. "
     "Grey bars = tensors, length ∝ channels per token.")
node("cap-bnote", 100, -560, 1300, 150,
     "RMSNorms here scale by (1 + w) with w stored zero-centred. "
     f"Parameter counts in boxes are per layer (from safetensors shapes). DeltaNet heads: {LK} for q/k, {LV} for v, "
     f"dim {LKD}; attention heads: {HQ} q, {HKV} kv, dim {HD}.")

b = Flow(0, -360)
b.tensor(DIM, f"x [T, {DIM}]  residual stream in", key="bx")
b.op("**input RMSNorm**", NORM, key="bn1")
xh = b.tensor(DIM, f"x̂ [T, {DIM}]", key="bxh")

dn_pre = f"{LM}layers.{I_DN}.linear_attn."
at_pre = f"{LM}layers.{I_ATT}.self_attn."
YCOL = b.y + 180

# ---- Gated DeltaNet column ----
d = Flow(LX, YCOL, last=xh, labels_left=True)
d.op(f"**in_proj_qkv** {DIM} → {2 * KEY_DIM + VAL_DIM}", DN, params_of(re.escape(dn_pre) + r"in_proj_qkv\..*"), key="dqkv")
d.tensor(2 * KEY_DIM + VAL_DIM, f"qkv [T, {2 * KEY_DIM + VAL_DIM}] = q {KEY_DIM} · k {KEY_DIM} · v {VAL_DIM}", key="dt1")
d.op(f"**causal conv1d** ({C['linear_conv_kernel_dim']} taps, per channel) + SiLU", DN,
     params_of(re.escape(dn_pre) + r"conv1d\..*"), key="dconv")
d.op(f"**split heads** · L2-norm q, k\nrepeat q, k {LK} → {LV} heads", DN, key="dsplit")
d.tensor(3 * VAL_DIM, f"q, k, v [T, {LV}, {LKD}] each", key="dt2")
rule = d.op("**gated delta rule** (per head)\nS ← α·S ;  S ← S + k ⊗ β(v − Sᵀk) ;  o = Sᵀq", DN, key="drule", h=80)
d.tensor(VAL_DIM, f"o [T, {LV}, {LVD}]   state S [{LV}, {LKD}, {LVD}]", key="dt3")
gnorm = d.op("**gated RMSNorm** — norm(o) · SiLU(z)", DN, key="dgn")
d.tensor(VAL_DIM, f"[T, {VAL_DIM}]", key="dt4")
d.op(f"**out_proj** {VAL_DIM} → {DIM}", DN, params_of(re.escape(dn_pre) + r"out_proj\..*"), key="dout")
dend = d.tensor(DIM, f"[T, {DIM}]", key="dt5")

SX = -200  # side boxes between the two columns
_ry = next(n for n in nodes if n["id"] == rule)["y"]
_gy = next(n for n in nodes if n["id"] == gnorm)["y"]
ab = node("dab", SX - OP_W / 2, _ry - 120, OP_W, 100,
          f"**in_proj_a, in_proj_b** x̂ → {LV} each\nβ = σ(b) · α = exp(−e^A_log · softplus(a + dt_bias))\n"
          f"{fmt(params_of(re.escape(dn_pre) + r'(in_proj_[ab]\..*|A_log|dt_bias)'))}", DN)
edge(ab, rule, ("bottom", "right"), "α, β")
z = node("dz", SX - OP_W / 2, _gy - 40, OP_W, 76,
         f"**in_proj_z** x̂ → {VAL_DIM}\n{fmt(params_of(re.escape(dn_pre) + r'in_proj_z\..*'))}", DN)
edge(z, gnorm, ("left", "right"), "z")

# ---- Gated Attention column ----
a = Flow(RX, YCOL, last=xh)
qkv_p = params_of(re.escape(at_pre) + r"[qkv]_proj\..*")
a.op(f"**q_proj** {DIM} → {2 * HQ * HD} · **k/v_proj** → {HKV * HD}", ATT, qkv_p, key="aqkv")
a.tensor(2 * HQ * HD + 2 * HKV * HD, f"q, gate [T, {HQ}, {HD}] · k, v [T, {HKV}, {HD}]", key="at1")
a.op(f"**q_norm, k_norm** (RMSNorm per head)\n**partial mRoPE** on {ROT} of {HD} dims", ATT,
     key="arope", h=80)
attn = a.op(f"**causal softmax attention**\nGQA {HQ} q / {HKV} kv heads ({HQ // HKV}:1)", ATT, key="aatt", h=80)
a.tensor(HQ * HD, f"[T, {HQ * HD}]", key="at2")
a.op("**× sigmoid(gate)** — per-channel output gate", ATT, key="agate")
a.tensor(HQ * HD, f"[T, {HQ * HD}]", key="at3")
a.op(f"**o_proj** {HQ * HD} → {DIM}", ATT, params_of(re.escape(at_pre) + r"o_proj\..*"), key="aout")
aend = a.tensor(DIM, f"[T, {DIM}]", key="at4")
_ay = next(n for n in nodes if n["id"] == attn)["y"]
kv = node("akv", RX - OP_W / 2 - 180 - 240, _ay, 240, 80,
          f"**KV cache**\nk, v [T, {HKV}, {HD}] per layer", ATT)
edge(kv, attn, ("right", "left"), "past k, v")

# ---- merge + FFN ----
YM = max(d.y, a.y) + 120
r1 = node("badd1", -OP_W / 2, YM, OP_W, OP_H, "**+ residual** (x)", ACT)
edge(dend, r1, ("bottom", "left"), "3 of 4 layers")
edge(aend, r1, ("bottom", "right"), "1 of 4 layers")
g = Flow(0, YM + OP_H + GAP, last=r1)
g.tensor(DIM, f"h [T, {DIM}]", key="bh")
g.op("**post-attention RMSNorm**", NORM, key="bn2")
g.op(f"**gate_proj, up_proj** {DIM} → {FF} each", FFN,
     params_of(re.escape(f"{LM}layers.0.mlp.") + r"(gate|up)_proj\..*"), key="bgu")
g.tensor(FF, f"SiLU(gate) · up  [T, {FF}]", key="bff")
g.op(f"**down_proj** {FF} → {DIM}", FFN, params_of(re.escape(f"{LM}layers.0.mlp.") + r"down_proj\..*"), key="bdn")
g.tensor(DIM, f"[T, {DIM}]", key="bfo")
g.op("**+ residual** (h)", ACT, key="badd2")
g.tensor(DIM, f"x [T, {DIM}]  → next layer", key="bout")
save("block.canvas")

print("ok", [(r[0], fmt(r[2]), fmt(r[3])) for r in B], f"KV {kv_gb:.1f}GB state {st_mb:.0f}MB")
