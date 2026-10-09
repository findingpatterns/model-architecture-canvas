"""Atria Dawn Preview (internlm/Atria-Dawn-Preview): model tab + one-layer tab.

The checkpoint is GlmMoeDsaForCausalLM (model_type glm_moe_dsa, the GLM-5.2 architecture): MLA attention
with DeepSeek Sparse Attention (DSA) whose lightning-indexer top-k is computed on "full" layers and
reused by the following "shared" layers, a 256-expert MoE, and one MTP layer.

Read top → bottom:
  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale everywhere)
  - boxes       = operations, all one size; parameter / compute shares are the two bars at the top
  - layer rows  = one per layer, coloured by FFN type and by whether the layer runs its own indexer
  - note-* ids  = annotations (dim, borderless); cap-* ids = borderless captions
Everything is derived from shapes.json (safetensors headers) + config.json of the checkpoint.

usage: python3 atria_dawn_preview_model_canvas.py <shapes.json> <config.json> <models/atria-dawn-preview>
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
IDX_T, MLP_T = C["indexer_types"], C["mlp_layer_types"]
R, A = C["n_routed_experts"], C["num_experts_per_tok"]
NH, QR, KVR = C["num_attention_heads"], C["q_lora_rank"], C["kv_lora_rank"]
NOPE, ROPE, VD = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
IH, ID, TOPK = C["index_n_heads"], C["index_head_dim"], C["index_topk"]
MOE_I, DENSE_I = C["moe_intermediate_size"], C["intermediate_size"]
MTP = N  # the MTP layer is stored as layers.<N>

# colours (JSON Canvas presets "1"–"6" or hex)
EMB, NORM, FFN, HEAD, ACT, ATT = "6", "3", "4", "2", "#64748b", "5"
IDX, MTPC, SHARED, DENSE = "#f59e0b", "#e879f9", "#a3e635", "#38bdf8"

# geometry
KT, CLIP_W = 0.08, 1800  # tensor bars: px per channel, longest drawn bar
ROW_W, ROW_H, ROW_GAP = 300, 34, 6
OP_W, OP_H, GAP = 300, 56, 40
BAR_W, BAR_H = 2600, 44
LBL_W = 440
MIN_SEG = 16  # the viewer draws no box narrower than ~14px

L = r"model\.layers\."


def params_of(pattern):
    rx = re.compile(pattern)
    return sum(math.prod(shp) for k, (dt, shp) in S.items() if rx.fullmatch(k))


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


MAIN = rf"{L}(?:[0-9]|[1-6][0-9]|7[0-7])\."  # layers 0 … N-1 (N = 78), not the MTP layer
assert N == 78, "MAIN regex assumes 78 layers"
P = {
    "embed": params_of(r"model\.embed_tokens\.weight"),
    "head": params_of(r"lm_head\.weight"),
    "attn": params_of(MAIN + r"self_attn\.(?!indexer\.).*"),
    "idx": params_of(MAIN + r"self_attn\.indexer\..*"),
    "dense": params_of(MAIN + r"mlp\.(gate|up|down)_proj\.weight"),
    "routed": params_of(MAIN + r"mlp\.experts\..*"),
    "shared": params_of(MAIN + r"mlp\.(shared_experts|gate)\..*"),
    "mtp": params_of(rf"{L}{MTP}\..*"),
    "expert": params_of(rf"{L}3\.mlp\.experts\.0\..*"),
}
TOTAL = params_of(r".*")
N_FULL = sum(t == "full" for t in IDX_T)
N_DENSE = sum(t == "dense" for t in MLP_T)


def budget():
    """[(name, colour, stored, multiplied per token, why-zero)] for the whole checkpoint."""
    rows = [
        (f"MLA attention ×{N}", ATT, P["attn"], P["attn"]),
        (f"DSA indexer ×{N_FULL}", IDX, P["idx"], P["idx"]),
        (f"dense FFN ×{N_DENSE}", DENSE, P["dense"], P["dense"]),
        (f"routed experts ({R}/layer)", FFN, P["routed"], P["routed"] * A // R),
        ("shared expert + router", SHARED, P["shared"], P["shared"]),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, P["head"], P["head"]),
        ("MTP layer", MTPC, P["mtp"], 0, "draft only"),
    ]
    other = TOTAL - sum(r[2] for r in rows)  # RMSNorm weights
    rows.append(("norms", ACT, other, other))
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

    def __init__(self, cx, y, last=None, labels_left=False, op_w=OP_W, lbl_w=LBL_W, kt=KT):
        self.cx, self.y, self.last, self.labels_left = cx, y, last, labels_left
        self.op_w, self.lbl_w, self.kt = op_w, lbl_w, kt

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
        nid = node(self._id(key), self.cx - self.op_w / 2, self.y, self.op_w, h, text, color)
        self._link(nid, nid)
        self.y += h + GAP
        return nid

    def tensor(self, dim, label, key="t"):
        real = dim * self.kt
        w = min(real, CLIP_W)
        nid = node(self._id(key), self.cx - w / 2, self.y, w, 9, "", ACT)
        clip = f" — bar clipped (real {real:.0f}px)" if real > w else ""
        half = max(w / 2, self.op_w / 2) + 20
        lx = self.cx - half - self.lbl_w if self.labels_left else self.cx + half
        node(self._id("note-lbl"), lx, self.y - 12, self.lbl_w, 34, f"`{label}`{clip}")
        self._link(nid, nid)
        self.y += 9 + GAP
        return nid


def save(name):
    json.dump(dict(nodes=nodes, edges=edges), open(os.path.join(OUT, name), "w"), indent=1, ensure_ascii=False)
    nodes.clear()
    edges.clear()


def bar(key, y, title, rows, idx, x0=-900):
    total = sum(r[idx] for r in rows)
    node(f"cap-{key}t", x0, y - 46, BAR_W, 40, f"**{title}** — total {fmt(total)}")
    x = x0
    legend = []
    # segments too thin to draw go last, so any blank stretch is at the bar's end, not mid-bar
    for k in sorted(range(len(rows)), key=lambda k: BAR_W * rows[k][idx] / total < MIN_SEG):
        r = rows[k]
        share = r[idx] / total
        w = BAR_W * share
        if w >= MIN_SEG:  # integer edges so neighbouring segments meet exactly
            node(f"{key}{k}", round(x), y, round(x + w) - round(x), BAR_H, f"**{r[0]}**" if w > 170 else "", r[1])
        x += w
        why = r[4] if len(r) > 4 and not r[idx] else ""
        legend.append(f"{r[0]} **{fmt(r[idx]) if r[idx] else '0'}**" + (f" ({why})" if why else f" ({share:.1%})"))
    node(f"cap-{key}l", x0, y + BAR_H + 6, BAR_W, 64, " · ".join(legend))


# =====================================================================================
# model.canvas
# =====================================================================================
GROUPS = []  # [first full layer, …its shared followers]
for i, t in enumerate(IDX_T):
    if t == "full":
        GROUPS.append([i])
    else:
        GROUPS[-1].append(i)
# one column: a decoder-only stack; repeated layers are drawn once with ×N (see segments())
COL_X = (0,)

node("cap-title", -900, -1250, 1250, 200,
     "# Atria Dawn Preview — model\n"
     f"Shanghai AI Lab, built on the **GLM-5.2** architecture (`{C['architectures'][0]}`, model_type "
     f"`{C['model_type']}`). One continuous decoder-only stack L0–{N - 1}, read top → bottom; layers that repeat are drawn once with **×N**. "
     f"**Grey bars = tensors**, length ∝ channels per token ({KT} px/channel). **Boxes = operations**, all one size. "
     "T = sequence length. *Layer* tab = one decoder layer in detail.")
node("cap-legend", 450, -1250, 1250, 200,
     f"Layer rows — **blue**: dense SwiGLU FFN ({DENSE_I}) · **cyan**: MoE + its own DSA indexer · **plain**: MoE, "
     f"reuses the indexer's top-{TOPK} picks of the layer above (`indexer_types` = shared). "
     f"Orange = whose top-k picks a layer uses. The barcode on the right lists every layer in order.  \n"
     f"All numbers from `config.json` + safetensors headers ({len(S)} tensors, {fmt(TOTAL)} params). "
     f"Model card: \"744B MoE\", 256K context; `max_position_embeddings` = {C['max_position_embeddings']}.")

# ---------- input flow (above the left column) ----------
f = Flow(COL_X[0] + ROW_W / 2, -960)
f.op("**raw text**  \"The cat sat…\"", ACT, key="raw")
f.op(f"**tokenizer** (vocab {VOCAB})", ACT, key="tok")
ids = f.tensor(40 / KT, "token ids [T] — one integer per token", key="ids")
f.op("**embed_tokens** — look up row *id*", EMB, P["embed"], note=f"{VOCAB} × {DIM}", key="embed")
f.tensor(DIM, f"x [T, {DIM}]", key="x")
ROW0 = f.y + 30


def kind(i):
    return MLP_T[i], IDX_T[i]


def kind_label(k):
    ffn = "dense FFN" if k[0] == "dense" else f"MoE {A}/{R}"
    return f"{ffn} · " + ("own DSA indexer" if k[1] == "full" else "reuse top-k")


def kind_color(k):
    if k[0] == "dense":
        return DENSE
    return ATT if k[1] == "full" else None


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


SEG_W, SEG_H, SEG_GAP, IDX_W = 460, 56, 26, 200
CX = ROW_W / 2
SEGS = segments()
y = ROW0
prev = f.last
for seg in SEGS:
    if seg[0] == "run":
        _, a, n = seg
        k = kind(a)
        span = f"**L{a}–{a + n - 1}** ×{n}" if n > 1 else f"**L{a}**"
        nid = node(f"seg{a}", CX - SEG_W / 2, y, SEG_W, SEG_H, f"{span}\n{kind_label(k)}", kind_color(k))
        edge(prev, nid)
        if k[1] == "shared":  # these layers use the picks of the last indexer layer before them
            node(f"idx{a}", CX + SEG_W / 2 + 30, y, IDX_W, SEG_H, f"top-k of **L{a - 1}**", IDX)
        prev = nid
        y += SEG_H + SEG_GAP
        continue
    _, s0, p, reps = seg
    y += 40  # room for the group's label, which the viewer draws above the frame
    gx, gy = CX - SEG_W / 2 - 60, y
    ry = y + 50
    first = None
    for j in range(p):
        k = kind(s0 + j)
        ls = [s0 + j + r * p for r in range(reps)]
        nid = node(f"per{j}", CX - SEG_W / 2, ry, SEG_W, SEG_H,
                   f"{kind_label(k)}\nL{ls[0]}, {ls[1]}, …, {ls[-1]}", kind_color(k))
        edge(prev, nid)
        first = first or nid
        prev = nid
        ry += SEG_H + SEG_GAP
    if kind(s0)[1] == "full" and all(kind(s0 + j)[1] == "shared" for j in range(1, p)):
        node("idx-per", CX + SEG_W / 2 + 30, y + 50, IDX_W, p * SEG_H + (p - 1) * SEG_GAP,
             f"**top-{TOPK} picks** of row 1\nreused by rows 2–{p}", IDX)
    edge(prev, first, ("left", "left"))  # loop back: the period runs again
    node("cap-repeat", gx - 170, (y + 50 + ry - SEG_GAP) / 2 - 30, 150, 60, f"## ↺ ×{reps}")
    gh = ry - y + 10
    node("grp-period", gx, gy, SEG_W + 120 + IDX_W + 30, gh,
         f"×{reps} — L{s0}–{N - 1}, a period of {p} layers", group=True)
    y += gh + SEG_GAP
LAST = prev

# barcode: every layer in order, one thin cell each, coloured like the rows above
CELL, CELL_GAP = 16, 2
BX, BY = CX + SEG_W / 2 + IDX_W + 160, ROW0 + 40
node("cap-barcode", BX, BY - 50, N * (CELL + CELL_GAP), 40,
     f"**Every layer, L0 → L{N - 1}** (1 cell = 1 layer; plain = MoE that reuses top-k)")
for i in range(N):
    node(f"lyr{i}", BX + i * (CELL + CELL_GAP), BY, CELL, 44, "", kind_color(kind(i)) or ACT)
for i in list(range(0, N, 10)) + [N - 1]:
    node(f"note-tick{i}", BX + i * (CELL + CELL_GAP) - 4, BY + 50, 60, 30, f"L{i}")


# ---------- output flow (below the stack) ----------
o = Flow(COL_X[0] + ROW_W / 2, y + 40, last=LAST, labels_left=True)
h_out = o.tensor(DIM, f"h [T, {DIM}] after L{N - 1}", key="ho")
o.op("**final RMSNorm**", NORM, key="fnorm")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm_head** — {DIM} → {VOCAB}", HEAD, P["head"], note="untied", key="head")
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op("**softmax → sample**", ACT, key="samp")
next_id = o.op("**next token id**", ACT, key="next")

# ---------- MTP: one extra decoder layer that drafts token t+2 ----------
yh = next(n["y"] for n in nodes if n["id"] == h_out)
MX = COL_X[0] + 1700
hn = node("mtp-hnorm", MX - 330, yh - 23, OP_W, OP_H, "**hnorm** (RMSNorm)\nof main-model hidden h", MTPC)
en = node("mtp-enorm", MX + 30, yh - 23, OP_W, OP_H, "**enorm** (RMSNorm)\nof embed(token t+1), shared table", MTPC)
edge(h_out, hn, ("right", "left"))
m = Flow(MX - 150 + 15, yh + OP_H + GAP + 20)
cat = m.tensor(2 * DIM, f"concat [T, {2 * DIM}]", key="mcat")
edge(hn, cat)
edge(en, cat)
m.op(f"**eh_proj** {2 * DIM} → {DIM}", MTPC, key="meh")
m.tensor(DIM, f"x [T, {DIM}]", key="mx")
m.op(f"**MTP decoder layer**\nMLA + DSA indexer + MoE {A}/{R}", MTPC, P["mtp"], note="layer 78", key="mlayer", h=80)
m.tensor(DIM, f"h' [T, {DIM}]", key="mh")
m.op("**shared_head.norm → lm_head**\n(lm_head shared with main model)", MTPC, key="mhead")
m.tensor(VOCAB, f"draft logits [T, {VOCAB}]", key="mlog")
m.op(f"**draft token t+2**\n(num_nextn_predict_layers = {C['num_nextn_predict_layers']})", MTPC, key="mdraft")
node("note-mtp", MX + 620, yh + OP_H + GAP + 20 + 120, 520, 150,
     "Multi-token prediction (DeepSeek-V3 style, tensors `layers.78.{eh_proj,enorm,hnorm,shared_head}`). "
     "Serving engines use it as a speculative draft head: the main model verifies the drafted token in its "
     "next forward pass. Not used for plain next-token sampling, so it counts as 0 in the active-weight bar.")

# ---------- budget ----------
B = budget()
bar("bp", -1640, "Where the parameters are stored", B, 2)
bar("bc", -1440, "What one token is multiplied by (active weights, main model)", B, 3)
save("model.canvas")
print("model ok", [(r[0], fmt(r[2]), fmt(r[3])) for r in B], "total", fmt(TOTAL),
      "active", fmt(sum(r[3] for r in B)), "without MTP", fmt(TOTAL - P["mtp"]))


# =====================================================================================
# block.canvas — one MoE layer that runs its own indexer
# =====================================================================================
LI = next(i for i in range(N) if MLP_T[i] == "sparse" and IDX_T[i] == "full")
FOLLOW = next(g for g in GROUPS if g[0] == LI)
KB = 0.03  # px per channel in this tab (wider tensors than the model tab)


def lp(rest):
    return params_of(rf"{L}{LI}\.{rest}")


def w(name):
    """`out×in` of a weight in layer LI, straight from the safetensors header."""
    return "×".join(map(str, S[f"model.layers.{LI}.{name}.weight"][1]))


XQ, XK, XI = 0, -1150, 1350  # column centres: query path, latent-KV path, indexer
QKD = NOPE + ROPE

node("cap-btitle", -1700, -560, 1300, 190,
     f"# One decoder layer — L{LI} (MoE, runs its own DSA indexer)\n"
     f"MLA attention ({NH} heads, q rank {QR}, KV rank {KVR}) restricted by DeepSeek Sparse Attention to the "
     f"top-{TOPK} tokens per query, then a {R}-expert MoE (top-{A} + 1 shared). Tensor shapes per token; "
     f"bars ∝ channels ({KB} px/channel). Weight shapes `out×in` from the safetensors headers.")
node("cap-blegend", -350, -560, 1300, 190,
     f"**Layer variants** (config `indexer_types`, `mlp_layer_types`): {N_FULL} of {N} layers run the indexer "
     f"(orange) — L0–2, then every {C['index_topk_freq']}th; the other {N - N_FULL} skip it and reuse the "
     f"previous indexer layer's top-{TOPK} token ids. L0–{N_DENSE - 1} replace the MoE by a dense SwiGLU FFN "
     f"({DENSE_I} hidden, `{'×'.join(map(str, S['model.layers.0.mlp.gate_proj.weight'][1]))}` gate). "
     f"RoPE (θ = {C['rope_parameters']['rope_theta']:.0f}, interleaved) touches only the {ROPE}-dim rope slices.")

f = Flow(XQ, -300, kt=KB, lbl_w=380)
f.tensor(DIM, f"x [T, {DIM}]  (residual stream)", key="bx")
norm_in = f.op("**input_layernorm** (RMSNorm)", NORM, key="bn1")
xh = f.tensor(DIM, f"x̂ [T, {DIM}]", key="bxh")
f.op(f"**q_a_proj** `{w('self_attn.q_a_proj')}`\nquery down-projection", ATT, key="bqa")
qnorm = f.op(f"**q_a_layernorm** (RMSNorm {QR})", NORM, key="bqan")
f.tensor(QR, f"q_resid [T, {QR}]", key="bqr")
f.op(f"**q_b_proj** `{w('self_attn.q_b_proj')}`\n→ {NH} heads × {QKD}", ATT, key="bqb")
f.tensor(NH * QKD, f"q [T, {NH}×{QKD}]", key="bq")
f.op(f"**split** q_nope {NOPE} | q_rope {ROPE}\nRoPE on q_rope", ATT, key="bqs")
q_last = f.tensor(NH * QKD, f"q [T, {NH}, {QKD}]", key="bq2")

# latent KV path (MLA): one small vector per token is all that is cached
k = Flow(XK, 0, kt=KB, labels_left=True, lbl_w=380)
k.y = next(n["y"] for n in nodes if n["id"] == xh) + 9 + GAP + OP_H + GAP  # one row below q_a_proj
k.op(f"**kv_a_proj_with_mqa** `{w('self_attn.kv_a_proj_with_mqa')}`\n= {KVR} latent + {ROPE} rope", ATT, key="bkva")
edge(norm_in, k.last, ("left", "top"))
k.tensor(KVR + ROPE, f"[T, {KVR + ROPE}]", key="bkv0")
k.op(f"**kv_a_layernorm** on the {KVR} latent\nRoPE on k_rope ({ROPE}, 1 head)", NORM, key="bkvn")
k.tensor(KVR + ROPE, f"KV cache: c_kv [T, {KVR}] + k_rope [T, {ROPE}]", key="bkvc")
k.op(f"**kv_b_proj** `{w('self_attn.kv_b_proj')}`\n→ {NH} heads × ({NOPE} k_nope + {VD} v)", ATT, key="bkvb")
k.tensor(NH * (NOPE + VD), f"[T, {NH}×{NOPE + VD}]", key="bkv1")
k.op(f"**k** = k_nope {NOPE} | k_rope {ROPE} (shared)\n**v** = {VD} per head", ATT, key="bkvs")
kv_last = k.tensor(NH * VD, f"k, v [T, {NH}, {VD}] each", key="bkv2")

# DSA lightning indexer: scores every past token cheaply, keeps the top-k
yq = next(n["y"] for n in nodes if n["id"] == qnorm)
iq = node("bidxq", XI - 340 - OP_W / 2, yq, OP_W, OP_H + 10,
          f"**indexer.wq_b** `{w('self_attn.indexer.wq_b')}`\n{IH} heads × {ID} (from q_resid)", IDX)
ik = node("bidxk", XI - OP_W / 2, yq, OP_W, OP_H + 10,
          f"**indexer.wk** `{w('self_attn.indexer.wk')}`\nwith LayerNorm · 1 key head × {ID}", IDX)
iw = node("bidxw", XI + 340 - OP_W / 2, yq, OP_W, OP_H + 10,
          f"**weights_proj** `{w('self_attn.indexer.weights_proj')}`\nper-head weight per query", IDX)
edge(qnorm, iq, ("right", "left"))
edge(norm_in, ik, ("right", "top"))
edge(norm_in, iw, ("right", "top"))
ix = Flow(XI, yq + OP_H + 10 + GAP + 30, kt=KB, lbl_w=380)
sc = ix.op(f"**index score** = Σ_heads w · ReLU(q·k)\nover all earlier tokens (RoPE on {ROPE} dims)", IDX, key="bisc", h=66)
for s_ in (iq, ik, iw):
    edge(s_, sc)
ix.tensor(64 / KB, "score [T, T_ctx]", key="bis")
ix.op(f"**top-{TOPK}** token ids per query\n{lp('self_attn.indexer..*') / 1e6:.1f}M params", IDX, key="bitk")
idx_last = ix.tensor(TOPK, f"top-k ids [T, {TOPK}] → reused by L{FOLLOW[1]}–{FOLLOW[-1]}", key="bit")

# sparse attention over the selected tokens
ya = max(f.y, k.y, ix.y) + 40
a = Flow(XQ, ya, kt=KB, lbl_w=380)
att = a.op(f"**sparse MLA attention** · {NH} heads\neach query attends only to its {TOPK} picked tokens", ATT,
           key="batt", h=66)
edge(q_last, att)
edge(kv_last, att, ("bottom", "left"))
edge(idx_last, att, ("bottom", "right"))
a.tensor(NH * VD, f"o [T, {NH}×{VD}]", key="bo")
a.op(f"**o_proj** `{w('self_attn.o_proj')}`\nthen add residual x", ATT, lp(r"self_attn\.(?!indexer\.).*"), note="attention", key="bop", h=66)
a.tensor(DIM, f"h [T, {DIM}]", key="bh")
pn = a.op("**post_attention_layernorm** (RMSNorm)", NORM, key="bn2")
a.tensor(DIM, f"ĥ [T, {DIM}]", key="bhh")
a.op(f"**router** `{w('mlp.gate')}` · sigmoid\nscore bias (aux-loss-free) · top-{A}, renorm ×{C['routed_scaling_factor']}",
     FFN, key="brt", h=66)
a.tensor(40 / KB, f"{A} expert ids + weights per token", key="brw")
ex = a.op(f"**{A} of {R} routed experts** · SwiGLU\n{DIM} → {MOE_I} → {DIM} each", FFN,
          P["expert"] * A, note=f"active of {fmt(lp(r'mlp\.experts\..*'))}", key="bex", h=66)
# shared expert runs for every token, beside the routed ones
yex = next(n["y"] for n in nodes if n["id"] == ex)
sh = node("bshared", XI - OP_W / 2, yex, OP_W, 66,
          f"**shared expert** · SwiGLU\n{DIM} → {MOE_I * C['n_shared_experts']} → {DIM} · "
          f"{fmt(lp(r'mlp\.shared_experts\..*'))}", SHARED)
edge(pn, sh, ("right", "top"))
a.tensor(DIM, f"Σ weighted expert outputs [T, {DIM}]", key="bsum")
add = a.op("**add** shared + routed\nplus residual h", ACT, key="badd")
edge(sh, add, ("bottom", "right"))
a.tensor(DIM, f"x [T, {DIM}] → next layer", key="bout")
node("note-shared-layer", -1700, ya, 520, 150,
     f"On the {N - N_FULL} shared layers the orange indexer is absent (no `indexer.*` tensors); the layer "
     "takes the previous indexer layer's top-k ids as-is. The KV cache still holds a fresh latent per layer.")
save("block.canvas")
print("block ok, layer", LI)
