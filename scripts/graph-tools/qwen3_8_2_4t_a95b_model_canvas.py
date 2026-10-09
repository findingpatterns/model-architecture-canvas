"""Qwen3.8-2.4T-A95B (Qwen) — model and layer canvases, derived from the checkpoint.

The repo ships `Qwen3_5MoeForCausalLM` (model_type qwen3_5_moe_text): the Qwen3.5-397B-A17B text
architecture scaled up (92 layers × 8192, same 512-expert top-10 MoE), text only, with one MTP layer.
The dataflow follows transformers' models/qwen3_5_moe/modeling_qwen3_5_moe.py; the MTP head follows
vLLM's qwen3_5_mtp.py; `output_gate_type` (not read by transformers) is the activation of the DeltaNet
gated RMSNorm in SGLang's qwen3_5.py. Every number comes from shapes.json (safetensors headers) or
config.json; nothing is typed in by hand.

  model.canvas — input → embed → 92 layer rows (coloured Gated DeltaNet / gated attention) → head,
                 MTP draft branch, stored vs active parameter bars on top
  block.canvas — one decoder layer: both token mixers side by side, then the MoE

Conventions: grey bars = tensors (length ∝ channels per token), boxes = operations, note-* ids are
annotations, cap-* ids are borderless captions.

usage: python3 qwen3_8_2_4t_a95b_model_canvas.py <shapes.json> <config.json> <models/qwen3-8-2-4t-a95b>
"""
import json
import math
import os
import re
import sys

S = json.load(open(sys.argv[1]))
C = json.load(open(sys.argv[2]))
OUT = sys.argv[3]

N, DIM, VOCAB = C["num_hidden_layers"], C["hidden_size"], C["vocab_size"]
TYPES = C["layer_types"]
E, K = C["num_experts"], C["num_experts_per_tok"]
HD, NQ, NKV = C["head_dim"], C["num_attention_heads"], C["num_key_value_heads"]
LKH, LVH, LKD, LVD = C["linear_num_key_heads"], C["linear_num_value_heads"], C["linear_key_head_dim"], C["linear_value_head_dim"]
ROPE = C["rope_parameters"]
ROT = int(HD * ROPE["partial_rotary_factor"])
LIN, FULL = "linear_attention", "full_attention"
N_LIN, N_FULL = TYPES.count(LIN), TYPES.count(FULL)
FIRST_LIN, FIRST_FULL = TYPES.index(LIN), TYPES.index(FULL)
LP = r"model\.layers\."
GATE_ACT = C.get("output_gate_type") or C["hidden_act"]

# colours (JSON Canvas presets "1"–"6" or hex)
EMB, NORM, FFN, HEAD, ACT = "6", "3", "4", "2", "#64748b"
GDN, ATT, SHARED, MTP = "#22d3ee", "#f97316", "#a3e635", "#f59e0b"


def params_of(pattern):
    """Weight elements matching pattern (the checkpoint is plain BF16: no scales, no packing)."""
    rx = re.compile(pattern)
    return sum(math.prod(shp) for k, (dt, shp) in S.items() if rx.fullmatch(k))


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


P = {
    "embed": params_of(r"model\.embed_tokens\.weight"),
    "head": params_of(r"lm_head\.weight"),
    "experts": params_of(LP + r"\d+\.mlp\.experts\..*"),
    # experts are stored fused per layer ([E, 2·I, D] + [E, D, I]); one expert = layer-0 tensors / E
    "expert1": params_of(LP + r"0\.mlp\.experts\..*") // E,
    "shared": params_of(LP + r"\d+\.mlp\.(shared_expert\..*|shared_expert_gate\.weight|gate\.weight)"),
    "shared1": params_of(LP + r"0\.mlp\.(shared_expert\..*|shared_expert_gate\.weight)"),
    "router1": params_of(LP + r"0\.mlp\.gate\.weight"),
    "gdn": params_of(LP + r"\d+\.linear_attn\..*"),
    "att": params_of(LP + r"\d+\.self_attn\..*"),
    "mtp": params_of(r"mtp\..*"),
    "all": params_of(r".*"),
}
LAYER_P = {t: params_of(LP + rf"{TYPES.index(t)}\..*") for t in (LIN, FULL)}
DTYPES = sorted({dt for dt, _ in S.values()})


def tp(n):
    return n if isinstance(n, str) else f"{n:,}".replace(",", " ")


def budget():
    """[(name, colour, stored params, params one text token is multiplied by, why-zero)]."""
    routed_active = P["experts"] * K // E
    rows = [
        (f"routed experts ({E}/layer)", FFN, P["experts"], routed_active),
        ("shared expert + router", SHARED, P["shared"], P["shared"]),
        (f"Gated DeltaNet ×{N_LIN}", GDN, P["gdn"], P["gdn"]),
        (f"gated attention ×{N_FULL}", ATT, P["att"], P["att"]),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, P["head"], P["head"]),
        ("MTP", MTP, P["mtp"], 0, "only when drafting"),
    ]
    other = P["all"] - sum(r[2] for r in rows)
    rows.append(("norms", NORM, other, other))  # layer + final RMSNorm weights: every param is accounted for
    return rows


# ---------- canvas primitives ----------
class Canvas:
    def __init__(self, kt, clip, op_w=300, op_h=56, gap=40, lbl_w=440):
        self.nodes, self.edges = [], []
        self.kt, self.clip, self.op_w, self.op_h, self.gap, self.lbl_w = kt, clip, op_w, op_h, gap, lbl_w
        self.seq = 0

    def node(self, nid, x, y, w, h, text="", color=None, group=False):
        text = re.sub(r"(?<=[^\n])\n(?=[^\n-])", "  \n", text)  # markdown hard break; keeps lists
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

    def __init__(self, cv, cx, y, last=None, labels_left=False):
        self.cv, self.cx, self.y, self.last, self.labels_left = cv, cx, y, last, labels_left

    def _link(self, top, bottom):
        if self.last:
            self.cv.edge(self.last, top)
        self.last = bottom

    def op(self, text, color, params=0, note="", key="op", h=None):
        cv = self.cv
        if params:
            text += f"\n{fmt(params)}" + (f" · {note}" if note else "")
        h = h or cv.op_h
        nid = cv.node(cv.uid(key), self.cx - cv.op_w / 2, self.y, cv.op_w, h, text, color)
        self._link(nid, nid)
        self.y += h + cv.gap
        return nid

    def tensor(self, dim, label, key="t"):
        cv = self.cv
        real = dim * cv.kt
        w = min(real, cv.clip)
        nid = cv.node(cv.uid(key), self.cx - w / 2, self.y, w, 9, "", ACT)
        clip = f" — bar clipped (real {real:.0f}px)" if real > w else ""
        half = max(w / 2, cv.op_w / 2) + 20
        lx = self.cx - half - cv.lbl_w if self.labels_left else self.cx + half
        cv.node(cv.uid("note-lbl"), lx, self.y - 10, cv.lbl_w, 34, f"`{label}`{clip}")
        self._link(nid, nid)
        self.y += 14 + cv.gap
        return nid


# =====================================================================================
# model.canvas
# =====================================================================================
KT = 0.08
M = Canvas(KT, 1800)
ROW_W, ROW_H, ROW_GAP = 300, 34, 6
COL_X = (0,)  # one column: repeated layers are drawn once with ×N (see segments())
BAR_W, BAR_H, MIN_SEG = 2600, 44, 16
TYPE_NAME = {LIN: "Gated DeltaNet", FULL: "gated attention"}
TYPE_COLOR = {LIN: GDN, FULL: ATT}

M.node("cap-title", -900, -1270, 1250, 200,
       "# Qwen3.8-2.4T-A95B — model\n"
       f"`{C['architectures'][0]}` (`{C['model_type']}`): the Qwen3.5 hybrid design scaled to **{fmt(P['all'])}** "
       f"stored parameters, text only. "
       f"One continuous stack L0–{N - 1}, read top → bottom; layers that repeat are drawn once with **×N**. "
       f"**Grey bars = tensors**, length ∝ channels per token ({KT} px/channel); **boxes = operations**. "
       "T = sequence length. *Layer* tab = one layer in detail.")
M.node("cap-legend", 450, -1270, 1250, 200,
       f"Layer rows — **Gated DeltaNet** ({N_LIN} layers, linear attention, fixed-size state) · "
       f"**gated attention** ({N_FULL} layers, every {C['full_attention_interval']}th, softmax + KV cache). "
       f"Every layer then runs the same MoE: router picks **{K} of {E}** experts (SwiGLU, {C['moe_intermediate_size']} wide) "
       f"plus 1 always-on **shared expert**.  \n"
       f"Layer ≈ **{fmt(LAYER_P[LIN])}** (DeltaNet) / **{fmt(LAYER_P[FULL])}** (attention), "
       f"{P['expert1'] * E / LAYER_P[LIN]:.0%} of it routed experts. All weights {'/'.join(DTYPES)}. "
       f"Context {tp(C['max_position_embeddings'])} tokens.")

# ---------- input flow ----------
cx0 = COL_X[0] + ROW_W / 2
f = Flow(M, cx0, -960)
f.op("**raw text**  \"The cat sat…\"", ACT, key="raw")
f.op(f"**tokenizer** (vocab {tp(VOCAB)})", ACT, key="tok")
f.tensor(40 / KT, "token ids [T] — one integer per token", key="ids")
emb = f.op("**embed** — look up row *id*", EMB, P["embed"], note=f"{tp(VOCAB)} × {DIM}", key="embed")
x0 = f.tensor(DIM, f"x [T, {DIM}]", key="x")
ROW0 = f.y + 40


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
    """Run-length prefix + the longest periodic tail (period ≥ 2 of mixed kinds, repeated ≥ 2×)."""
    for s0 in range(N):
        for p in range(2, 9):
            if ((N - s0) % p == 0 and (N - s0) // p >= 2 and len({kind(s0 + j) for j in range(p)}) > 1
                    and all(kind(i) == kind(s0 + (i - s0) % p) for i in range(s0, N))):
                return runs(0, s0) + [("period", s0, p, (N - s0) // p)]
    return runs(0, N)


# ---------- layer stack ----------
SEG_W, SEG_H, SEG_GAP = 460, 56, 26
CX = ROW_W / 2
y = ROW0
prev = f.last
for seg in segments():
    if seg[0] == "run":
        _, a, n = seg
        span = f"**L{a}–{a + n - 1}** ×{n}" if n > 1 else f"**L{a}**"
        nid = M.node(f"seg{a}", CX - SEG_W / 2, y, SEG_W, SEG_H, f"{span}\n{TYPE_NAME[kind(a)]} + MoE", TYPE_COLOR[kind(a)])
        M.edge(prev, nid)
        prev = nid
        y += SEG_H + SEG_GAP
        continue
    _, s0, p, reps = seg
    y += 40  # room for the group's label, which the viewer draws above the frame
    gx = CX - SEG_W / 2 - 60
    ry = y + 50
    first = None
    for j in range(p):
        ls = [s0 + j + r * p for r in range(reps)]
        nid = M.node(f"per{j}", CX - SEG_W / 2, ry, SEG_W, SEG_H,
                     f"{TYPE_NAME[kind(s0 + j)]} + MoE\nL{ls[0]}, {ls[1]}, …, {ls[-1]}", TYPE_COLOR[kind(s0 + j)])
        M.edge(prev, nid)
        first = first or nid
        prev = nid
        ry += SEG_H + SEG_GAP
    M.edge(prev, first, ("left", "left"))  # loop back: the period runs again
    M.node("cap-repeat", gx - 170, (y + 50 + ry - SEG_GAP) / 2 - 30, 150, 60, f"## ↺ ×{reps}")
    gh = ry - y + 10
    M.node("grp-period", gx, y, SEG_W + 120, gh, f"×{reps} — L{s0}–{N - 1}, a period of {p} layers", group=True)
    y += gh + SEG_GAP
LAST, STACK_END = prev, y

# barcode: every layer in order, one thin cell each, coloured like the rows above
CELL, CELL_GAP = 16, 2
BX, BY = CX + SEG_W / 2 + 160, ROW0 + 40
M.node("cap-barcode", BX, BY - 50, N * (CELL + CELL_GAP), 40,
       f"**Every layer, L0 → L{N - 1}** (1 cell = 1 layer; cyan Gated DeltaNet, orange gated attention)")
for i in range(N):
    M.node(f"lyr{i}", BX + i * (CELL + CELL_GAP), BY, CELL, 44, "", TYPE_COLOR[kind(i)])
for i in sorted(set(range(0, N, 10)) | ({N - 1} if (N - 1) % 10 >= 4 else set())):
    M.node(f"note-tick{i}", BX + i * (CELL + CELL_GAP) - 4, BY + 50, 60, 30, f"L{i}")

# annotations: left of the stack and under the barcode
ya = ROW0
gdn_state = LVH * LVD * LKD
kv_tok = 2 * NKV * HD
M.node("note-gdn", -1400, ya, 460, 250,
       f"**Gated DeltaNet rows** ({N_LIN}): linear attention. Each layer keeps one recurrent state "
       f"[{LVH} heads × {LKD} × {LVD}] = {tp(gdn_state)} values, the same size at token 10 or token "
       f"{tp(C['max_position_embeddings'])} — no KV cache that grows with T. "
       f"Short causal conv (kernel {C['linear_conv_kernel_dim']}) on q/k/v, per-head decay and write strength (β).")
M.node("note-full", -900, ya, 460, 220,
       f"**Gated attention rows** ({N_FULL}, L{FIRST_FULL}, L{FIRST_FULL + 4}, …): softmax attention, "
       f"GQA {NQ} query : {NKV} KV heads × {HD}, a sigmoid output gate per channel, "
       f"RoPE on {ROT} of {HD} dims. Only these layers cache K/V: {tp(kv_tok)} values per token each.")
yb = BY + 110
M.node("note-moe", BX, yb, 460, 220,
       f"**MoE in every layer**: softmax router over {E} experts → top-{K}, weights renormalised. "
       f"Expert = SwiGLU {DIM}→{C['moe_intermediate_size']}→{DIM} ({fmt(P['expert1'])}). "
       f"Shared expert (same width) is added through a sigmoid gate. "
       f"Per token: {K} + 1 experts of {E} → ~{(K + 1) * P['expert1'] / (E * P['expert1'] + P['shared1']):.1%} of FFN weights.")
M.node("note-norm", BX + 500, yb, 460, 130,
       "Pre-norm residual blocks; RMSNorm is zero-centred (scale = 1 + w). "
       "q/k get their own per-head RMSNorm in the attention layers.")
M.node("note-scale", BX + 1000, yb, 460, 190,
       f"**Same recipe as Qwen3.5-397B-A17B**, scaled: {N} layers × {DIM} wide "
       f"(vs 60 × 4096), {NQ} q / {NKV} kv heads, {LVH} DeltaNet value heads; the MoE shape "
       f"({E} experts, top-{K}, {C['moe_intermediate_size']} wide) is unchanged. No vision tower ships in this repo.")

# ---------- output flow ----------
cx1 = CX
o = Flow(M, cx1, STACK_END + 40, last=LAST, labels_left=True)
o.tensor(DIM, f"h [T, {DIM}]  after L{N - 1}", key="ho")
fnorm = o.op("**final RMSNorm**", NORM, key="fn")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm head** — {DIM} → {tp(VOCAB)}", HEAD, P["head"], note="untied", key="head")
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op("**softmax → sample**", ACT, key="samp")
o.op("**next token id**", ACT, key="next")

# ---------- MTP: one extra gated-attention + MoE layer that drafts the following token ----------
mt = Flow(M, cx1 + 1900, STACK_END + 200)
cat = mt.tensor(2 * DIM, f"[T, {2 * DIM}] = norm(embed(token t+1)) ‖ norm(x)", key="mcat")
M.edge(fnorm, cat, ("right", "top"), "normed x", MTP)
mt.op(f"**fc** {2 * DIM} → {DIM}", MTP, params_of(r"mtp\.fc\..*"), key="mfc")
mt.tensor(DIM, f"[T, {DIM}]", key="mh")
mt.op("**1 gated-attention layer** + MoE", MTP, params_of(r"mtp\.layers\..*"),
      note=f"{E} experts", key="mly")
mt.op("**RMSNorm → shared lm head**", MTP, key="mhd")
mt.op("**draft token t+2**", ACT, key="mdr")
M.node("note-mtp", mt.cx - M.op_w / 2, mt.y - 10, 600, 170,
       f"Multi-token prediction head ({fmt(P['mtp'])}, {C['mtp_num_hidden_layers']} layer, its own full "
       f"{E}-expert MoE). `mtp_use_dedicated_embeddings: {str(C['mtp_use_dedicated_embeddings']).lower()}` — it reuses the main "
       "embed table and lm head (no copies in the checkpoint). Serving engines run it as a draft for speculative "
       "decoding (the card: MTP trained with multiple steps); the main model verifies. Wiring per vLLM qwen3_5_mtp.py.")


# ---------- budget bars ----------
def bar(cv, key, y, title, rows, idx):
    total = sum(r[idx] for r in rows)
    cv.node(f"cap-{key}t", -900, y - 46, BAR_W, 40, f"**{title}** — total {fmt(total)}")
    x, legend = -900, []
    # segments too thin to draw go last, so any blank stretch is at the bar's end, not mid-bar
    for k in sorted(range(len(rows)), key=lambda k: BAR_W * rows[k][idx] / total < MIN_SEG):
        r = rows[k]
        share = r[idx] / total
        w = BAR_W * share
        if w >= MIN_SEG:  # integer edges so neighbouring segments meet exactly, never overlap
            cv.node(f"{key}{k}", round(x), y, round(x + w) - round(x), BAR_H, f"**{r[0]}**" if w > 200 else "", r[1])
        x += w
        why = r[4] if len(r) > 4 and not r[idx] else ""
        legend.append(f"{r[0]} **{fmt(r[idx]) if r[idx] else '0'}**" + (f" ({why})" if why else f" ({share:.1%})"))
    cv.node(f"cap-{key}l", -900, y + BAR_H + 6, BAR_W, 40, " · ".join(legend))


B = budget()
bar(M, "bp", -1640, "Where the parameters are stored", B, 2)
bar(M, "bc", -1460, "What one text token is multiplied by (active weights)", B, 3)
M.save("model.canvas")

# =====================================================================================
# block.canvas — one decoder layer
# =====================================================================================
KB = 0.06
L = Canvas(KB, 720, op_w=320, op_h=64, lbl_w=560)
XG, XA = -720, 720  # the two token-mixer columns
SIDE_W = 270
L.node("cap-btitle", -1400, -560, 1300, 190,
       "# Qwen3.8-2.4T-A95B — one decoder layer\n"
       f"Pre-norm residual block. The token mixer is **either** Gated DeltaNet (left, {N_LIN} layers) "
       f"**or** gated softmax attention (right, {N_FULL} layers, every {C['full_attention_interval']}th); "
       f"the MoE below is the same in all {N}. Bars ∝ channels ({KB} px/channel). Shapes per token.")
L.node("cap-bsrc", 100, -560, 1300, 190,
       "Traced from transformers `modeling_qwen3_5_moe.py` (Qwen3_5MoeGatedDeltaNet, Qwen3_5MoeAttention, "
       "Qwen3_5MoeSparseMoeBlock) and checked against the checkpoint's tensor names/shapes. "
       f"Parameter counts are per layer, all {'/'.join(DTYPES)}. Routed experts are stored fused: "
       f"gate_up_proj [{E}, {2 * C['moe_intermediate_size']}, {DIM}] + down_proj [{E}, {DIM}, {C['moe_intermediate_size']}].")

m = Flow(L, 0, -320)
m.tensor(DIM, f"h [T, {DIM}]  residual stream in")
h_in = m.last
nrm = m.op("**RMSNorm** (input_layernorm)", NORM)
xin = m.tensor(DIM, f"x [T, {DIM}]")
ysplit = m.y + 30

# --- Gated DeltaNet column ---
qk, vd = LKH * LKD, LVH * LVD
g = Flow(L, XG, ysplit, last=xin, labels_left=True)
g.op(f"**in_proj_qkv** {DIM} → {qk * 2 + vd}", GDN, params_of(LP + rf"{FIRST_LIN}\.linear_attn\.in_proj_qkv\.weight"))
g.tensor(qk * 2 + vd, f"[T, {qk * 2 + vd}] = q {qk} · k {qk} · v {vd}")
g.op(f"**causal conv1d** (k={C['linear_conv_kernel_dim']}, per channel) + SiLU", GDN)
g.tensor(qk * 2 + vd, f"q,k [T, {LKH}, {LKD}] · v [T, {LVH}, {LVD}]")
g.op(f"**L2-norm q, k**; repeat {LKH} → {LVH} heads", GDN)
g.tensor(LVH * LKD * 2 + vd, f"q,k,v [T, {LVH}, {LKD}]")
rule = g.op(f"**gated delta rule**\nS ← α·S(I − β k kᵀ) + β v kᵀ,  o = S q\nstate S [{LVH}, {LKD}, {LVD}] per layer", GDN, h=96)
g.tensor(vd, f"o [T, {LVH}, {LVD}]")
gn = g.op(f"**gated RMSNorm** per head\nnorm(o) · {GATE_ACT}(z)", GDN)
g.tensor(vd, f"[T, {vd}]")
g.op(f"**out_proj** {vd} → {DIM}", GDN, params_of(LP + rf"{FIRST_LIN}\.linear_attn\.out_proj\.weight"))
g_out = g.tensor(DIM, f"[T, {DIM}]  mixer out")

sx = -SIDE_W - 60  # side inputs of the DeltaNet sit between the two columns
ab = L.node("gdn-ab", sx, L.nodes[[n["id"] for n in L.nodes].index(rule)]["y"], SIDE_W, 110,
            f"**in_proj_a, in_proj_b** (from x)\nα = exp(−e^A_log · softplus(a + dt_bias))\n"
            f"β = sigmoid(b)  · [T, {LVH}] each", GDN)
L.edge(ab, rule, ("left", "right"), "decay α, write β")
zbox = L.node("gdn-z", sx, L.nodes[[n["id"] for n in L.nodes].index(gn)]["y"], SIDE_W, 64,
              f"**in_proj_z** (from x) → z [T, {vd}]", GDN)
L.edge(zbox, gn, ("left", "right"))

# --- gated attention column ---
a = Flow(L, XA, ysplit, last=xin)
qp = params_of(LP + rf"{FIRST_FULL}\.self_attn\.q_proj\.weight")
a.op(f"**q_proj** {DIM} → {NQ * HD * 2}", ATT, qp)
a.tensor(NQ * HD * 2, f"[T, {NQ}, {HD}+{HD}] = query | gate")
a.op(f"**split**; RMSNorm on q (per head)", ATT)
a.tensor(NQ * HD, f"q [T, {NQ}, {HD}]")
a.op(f"**RoPE** on {ROT} of {HD} dims\n(θ = {tp(ROPE['rope_theta'])})", ATT, h=80)
a.tensor(NQ * HD, f"q [T, {NQ}, {HD}]")
sdpa = a.op(f"**causal softmax attention**\nGQA: {NQ // NKV} query heads share each K/V head", ATT, h=80)
a.tensor(NQ * HD, f"[T, {NQ * HD}]")
gate = a.op("**× sigmoid(gate)** per channel", ATT)
a.tensor(NQ * HD, f"[T, {NQ * HD}]")
a.op(f"**o_proj** {NQ * HD} → {DIM}", ATT, params_of(LP + rf"{FIRST_FULL}\.self_attn\.o_proj\.weight"))
a_out = a.tensor(DIM, f"[T, {DIM}]  mixer out")
kvp = params_of(LP + rf"{FIRST_FULL}\.self_attn\.(k|v)_proj\.weight")
kv = L.node("att-kv", 60, L.nodes[[n["id"] for n in L.nodes].index(sdpa)]["y"] - 20, SIDE_W, 130,
            f"**k_proj, v_proj** (from x) {DIM} → {NKV}×{HD} each · {fmt(kvp)}\nRMSNorm + RoPE on k\n"
            f"KV cache: {tp(kv_tok)} values / token", ATT)
L.edge(kv, sdpa, ("right", "left"), "K, V")

# --- residual add, then MoE ---
ymerge = max(g.y, a.y) + 40
r = Flow(L, 0, ymerge)
add1 = r.op("**residual add**\nh (layer input) + mixer out", ACT)
L.edge(g_out, add1, ("bottom", "left"))
L.edge(a_out, add1, ("bottom", "right"))
r.tensor(DIM, f"h [T, {DIM}]")
r.labels_left = True
r.op("**RMSNorm** (post_attention_layernorm)", NORM)
xm = r.tensor(DIM, f"x [T, {DIM}]")
r.op(f"**router** {DIM} → {E}, softmax → top-{K}\nweights renormalised to sum 1", FFN, P["router1"], h=80)
r.tensor(40 / KB, f"expert ids + weights [T, {K}]")
r.op(f"**{K} routed experts** of {E}\nSwiGLU {DIM}→{C['moe_intermediate_size']}→{DIM}", FFN,
     K * P["expert1"], note=f"active of {fmt(E * P['expert1'])}", h=80)
r.tensor(DIM, f"[T, {DIM}]  weighted sum")
msum = r.op("**sum** routed + gated shared", FFN)
sh = L.node("moe-shared", 420, L.nodes[[n["id"] for n in L.nodes].index(msum)]["y"] - 150, 340, 110,
            f"**shared expert** (every token)\nSwiGLU {DIM}→{C['shared_expert_intermediate_size']}→{DIM} × sigmoid(gate)\n"
            f"{fmt(P['shared1'])}", SHARED)
L.edge(xm, sh, ("right", "top"))
L.edge(sh, msum, ("bottom", "right"))
r.tensor(DIM, f"[T, {DIM}]  MoE out")
r.op("**residual add** → h [T, " + str(DIM) + "] out", ACT)
L.save("block.canvas")

print("ok", [(b[0], fmt(b[2]), fmt(b[3])) for b in B], "total", P["all"],
      "active", sum(b[3] for b in B))
