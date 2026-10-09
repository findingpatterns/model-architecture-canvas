"""GLM-5.3 (zai-org/GLM-5.3): model-level and layer-level canvases, derived from the checkpoint.

Inputs are the repo's config.json and shapes.json (safetensors headers, see fetch_safetensors_shapes.py).
Dataflow follows transformers' modeling_glm_moe_dsa.py (GlmMoeDsaForCausalLM); the MTP block
(model.layers.78) is not in transformers and is wired from its tensor names (enorm / hnorm / eh_proj /
shared_head), the DeepSeek-V3 next-token-prediction layout.

  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale per canvas)
  - boxes       = operations, all one size; parameter / compute shares are the two bars at the top
  - note-* ids  = annotations (dim, borderless); cap-* ids = borderless captions

usage: python3 glm_5_3_model_canvas.py <shapes.json> <config.json> <models/glm-5-3>
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
H, NOPE, ROPE, VD = C["num_attention_heads"], C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
QL, KVL = C["q_lora_rank"], C["kv_lora_rank"]
IH, ID, TOPK = C["index_n_heads"], C["index_head_dim"], C["index_topk"]
R, A, MI = C["n_routed_experts"], C["num_experts_per_tok"], C["moe_intermediate_size"]
IDX_T, MLP_T = C["indexer_types"], C["mlp_layer_types"]
MTP_L = N  # the next-token-prediction block is stored as model.layers.<N>
THETA = C["rope_parameters"]["rope_theta"]

EMB, NORM, FFN, HEAD, ACT, ATT, IDX, MTP = "6", "3", "4", "2", "#64748b", "1", "5", "#f59e0b"
ROW_COLOR = {("dense", "full"): "#fbbf24", ("sparse", "full"): "5", ("sparse", "shared"): "#94a3b8"}


def params_of(pattern):
    rx = re.compile(pattern)
    return sum(math.prod(shp) for k, (dt, shp) in S.items() if rx.fullmatch(k) and not k.endswith("_scale_inv"))


MAIN = r"model\.layers\.(?:[0-9]|[1-6][0-9]|7[0-7])"  # L0–77; layer 78 is the MTP block
assert N == 78, "MAIN regex assumes 78 main layers"
P = dict(
    embed=params_of(r"model\.embed_tokens\.weight"),
    head=params_of(r"lm_head\.weight"),
    attn=params_of(MAIN + r"\.self_attn\.(?!indexer\.).*"),
    idx=params_of(MAIN + r"\.self_attn\.indexer\..*"),
    dense=params_of(MAIN + r"\.mlp\.(?:gate|up|down)_proj\..*"),
    routed=params_of(MAIN + r"\.mlp\.experts\..*"),
    shared=params_of(MAIN + r"\.mlp\.(?:shared_experts|gate)\..*"),
    mtp=params_of(rf"model\.layers\.{MTP_L}\..*"),
    expert=params_of(r"model\.layers\.3\.mlp\.experts\.0\..*"),
    attn1=params_of(r"model\.layers\.0\.self_attn\.(?!indexer\.).*"),
    idx1=params_of(r"model\.layers\.0\.self_attn\.indexer\..*"),
    shexp=params_of(r"model\.layers\.3\.mlp\.shared_experts\..*"),
    router=params_of(r"model\.layers\.3\.mlp\.gate\..*"),
    densel=params_of(r"model\.layers\.0\.mlp\..*"),
    ehproj=params_of(rf"model\.layers\.{MTP_L}\.eh_proj\..*"),
)
N_MOE = MLP_T.count("sparse")
N_FULL = IDX_T.count("full")
# every indexer the checkpoint stores must belong to a "full" layer (or the MTP block)
stored_idx = sorted({int(re.match(r"model\.layers\.(\d+)", k)[1]) for k in S if ".indexer." in k})
assert stored_idx == [i for i in range(N) if IDX_T[i] == "full"] + [MTP_L], stored_idx


def budget():
    """[(name, colour, stored params, params multiplied per token in the main forward, why-zero)]"""
    rows = [
        (f"routed experts {R}×{N_MOE}", FFN, P["routed"], N_MOE * A * P["expert"]),
        (f"MLA attention ×{N}", ATT, P["attn"], P["attn"]),
        (f"DSA indexers ×{N_FULL}", IDX, P["idx"], P["idx"]),
        (f"shared expert + router ×{N_MOE}", "#22c55e", P["shared"], P["shared"]),
        (f"dense FFN ×{MLP_T.count('dense')}", "#fbbf24", P["dense"], P["dense"]),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, P["head"], P["head"]),
        ("MTP block", MTP, P["mtp"], 0, "speculative drafting only"),
    ]
    other = params_of(r".*") - sum(r[2] for r in rows)
    if other >= 1e6:
        rows.append(("other", ACT, other, other))
    return rows


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


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


def save(name):
    json.dump(dict(nodes=nodes, edges=edges), open(os.path.join(OUT, name), "w"), indent=1, ensure_ascii=False)
    print("wrote", name, len(nodes), "nodes", len(edges), "edges")
    nodes.clear()
    edges.clear()


_seq = [0]


class Flow:
    """A vertical chain of ops and tensors centred on cx; each item is linked to the previous one."""

    def __init__(self, cx, y, last=None, labels_left=False, kt=0.08, clip=1800, lbl=440, op_w=300):
        self.cx, self.y, self.last, self.labels_left = cx, y, last, labels_left
        self.kt, self.clip, self.lbl, self.op_w = kt, clip, lbl, op_w

    def _id(self, key):
        _seq[0] += 1
        return f"{key}{_seq[0]}"

    def _link(self, top, bottom):
        if self.last:
            edge(self.last, top)
        self.last = bottom

    def op(self, text, color, params=0, note="", key="op", h=56):
        if params:
            text += f"\n{fmt(params)}" + (f" · {note}" if note else "")
        nid = node(self._id(key), self.cx - self.op_w / 2, self.y, self.op_w, h, text, color)
        self._link(nid, nid)
        self.y += h + 40
        return nid

    def tensor(self, dim, label, key="t", left=None):
        real = dim * self.kt
        w = max(16, min(real, self.clip))
        nid = node(self._id(key), self.cx - w / 2, self.y, w, 9, "", ACT)
        clip = f" — bar clipped (real {real:.0f}px)" if real > w else ""
        half = max(w / 2, self.op_w / 2) + 20
        lx = self.cx - half - self.lbl if (self.labels_left if left is None else left) else self.cx + half
        node(self._id("note-lbl"), lx, self.y - 12, self.lbl, 34, f"`{label}`{clip}")
        self._link(nid, nid)
        self.y += 9 + 40
        return nid


# ======================================================================= model.canvas
ROW_W = 300
COL_X = (0,)  # one column: a decoder-only stack; repeated layers are drawn once with ×N (see segments())
BAR_W, BAR_H, MIN_SEG = 2600, 44, 16


def bar(key, y, title, rows, idx):
    total = sum(r[idx] for r in rows)
    node(f"cap-{key}t", -900, y - 46, BAR_W, 40, f"**{title}** — total {fmt(total)}")
    x, legend = -900, []
    # segments too thin to draw go last, so any blank stretch is at the bar's end, not mid-bar
    for k in sorted(range(len(rows)), key=lambda k: BAR_W * rows[k][idx] / total < MIN_SEG):
        r = rows[k]
        share = r[idx] / total
        w = BAR_W * share
        if w >= MIN_SEG:  # integer edges so neighbouring segments meet exactly
            node(f"{key}{k}", round(x), y, round(x + w) - round(x), BAR_H, f"**{r[0]}**" if w > 220 else "", r[1])
        x += w
        why = r[4] if len(r) > 4 and not r[idx] else ""
        legend.append(f"{r[0]} **{fmt(r[idx]) if r[idx] else '0'}**" + (f" ({why})" if why else f" ({share:.1%})" if share >= 0.001 else " (<0.1%)"))
    node(f"cap-{key}l", -900, y + BAR_H + 6, BAR_W, 64, " · ".join(legend))


B = budget()
bar("bp", -1700, "Where the parameters are stored", B, 2)
bar("bc", -1500, "What one token is multiplied by in the main forward (active weights)", B, 3)

node("cap-title", -900, -1300, 1250, 200,
     "# GLM-5.3 — model\n"
     f"One continuous decoder-only stack L0–{N - 1}, read top → bottom; layers that repeat are drawn once with **×N**. "
     "**Grey bars = tensors**, length ∝ channels per token (0.08 px/channel). **Boxes = operations**, all one size; "
     "where parameters and compute live is the pair of bars above. T = sequence length. "
     "*Layer* tab = one decoder layer in detail.")
node("cap-legend", 450, -1300, 1250, 200,
     f"Every layer = **MLA attention** ({H} heads, KV squeezed to a {KVL}+{ROPE} latent) made sparse by **DSA**: "
     f"each query attends only the top {TOPK} past tokens chosen by a small indexer.  \n"
     f"Row colours — **amber**: dense SwiGLU FFN + own indexer (L0–2) · **cyan**: MoE + own indexer · "
     f"**grey**: MoE, *reuses* the top-{TOPK} picks of the cyan layer above (no indexer weights). "
     f"MoE = {A} of {R} routed experts + 1 shared, sigmoid router. The barcode on the right lists every layer in order.")

# ---------- input flow ----------
cx1 = COL_X[0] + ROW_W / 2
f = Flow(cx1, -980)
f.op("**raw text**  \"The cat sat…\"", ACT, key="raw")
f.op(f"**tokenizer** (vocab {VOCAB})", ACT, key="tok")
f.tensor(40 / 0.08, "token ids [T] — one integer per token", key="ids")
f.op("**embed** — look up row *id*", EMB, P["embed"], note=f"table {VOCAB} × {DIM}", key="embed")
x0 = f.tensor(DIM, f"x [T, {DIM}]", key="x")
ROW0 = f.y + 30


def kind(i):
    return MLP_T[i], IDX_T[i]


def kind_label(k):
    ffn = "dense FFN" if k[0] == "dense" else f"MoE {A}/{R}"
    return f"{ffn} · " + ("own indexer" if k[1] == "full" else "reuses top-k")


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
CX = cx1
y = ROW0
prev = x0
for seg in segments():
    if seg[0] == "run":
        _, a, n = seg
        k = kind(a)
        span = f"**L{a}–{a + n - 1}** ×{n}" if n > 1 else f"**L{a}**"
        nid = node(f"seg{a}", CX - SEG_W / 2, y, SEG_W, SEG_H, f"{span}\n{kind_label(k)}", ROW_COLOR[k])
        edge(prev, nid)
        if k[1] == "shared":  # these layers use the picks of the last indexer layer before them
            node(f"idx{a}", CX + SEG_W / 2 + 30, y, IDX_W, SEG_H, f"top-{TOPK} of **L{a - 1}**", IDX)
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
                   f"{kind_label(k)}\nL{ls[0]}, {ls[1]}, …, {ls[-1]}", ROW_COLOR[k])
        edge(prev, nid)
        first = first or nid
        prev = nid
        ry += SEG_H + SEG_GAP
    if kind(s0)[1] == "full" and all(kind(s0 + j)[1] == "shared" for j in range(1, p)):
        node("idx-per", CX + SEG_W / 2 + 30, y + 50, IDX_W, p * SEG_H + (p - 1) * SEG_GAP,
             f"**indexer** of row 1 → top-{TOPK}\nreused by rows 2–{p}", IDX)
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
node("cap-barcode", BX, BY - 50, N * (CELL + CELL_GAP), 40, f"**Every layer, L0 → L{N - 1}** (1 cell = 1 layer)")
for i in range(N):
    node(f"lyr{i}", BX + i * (CELL + CELL_GAP), BY, CELL, 44, "", ROW_COLOR[kind(i)])
for i in sorted({*range(0, N, 10), N - 1} - {i for i in range(N - 4, N - 1) if i % 10 == 0}):  # no tick crowding the last one
    node(f"note-tick{i}", BX + i * (CELL + CELL_GAP) - 4, BY + 50, 60, 30, f"L{i}")

# ---------- output flow (below the right column) ----------
o = Flow(cx1, y + 40, last=LAST, labels_left=True)
h_out = o.tensor(DIM, f"x [T, {DIM}]  after L{N - 1}", key="xo")
o.op("**final RMSNorm**", NORM, key="fn")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm head** — {DIM} → {VOCAB}", HEAD, P["head"], note="not tied to embed", key="head")
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op("**softmax → sample**", ACT, key="samp")
next_id = o.op("**next token id**", ACT, key="next")

# ---------- MTP block (model.layers.78): drafts one extra token ----------
ylast = next(n["y"] for n in nodes if n["id"] == h_out)
tap = node("note-tap", cx1 + 600, ylast - 12, 260, 34, "↳ main hidden state → MTP")
edge(h_out, tap, ("right", "left"), color=MTP)
m = Flow(2000, ylast + 60)
m.op("**embed** next token — same table as the main embed", EMB, key="membed")
m.tensor(DIM, f"e [T, {DIM}]", key="me")
cat = m.op("**enorm**(e) ‖ **hnorm**(h) — RMSNorm each, concat", MTP, key="mcat")
edge(tap, cat, ("right", "left"), color=MTP)
m.tensor(2 * DIM, f"[T, {2 * DIM}]", key="mc")
m.op(f"**eh_proj** {2 * DIM} → {DIM}", MTP, P["ehproj"], key="meh")
m.tensor(DIM, f"[T, {DIM}]", key="mh")
m.op(f"**1 decoder layer** — MLA + own indexer + MoE {A}/{R}", MTP, P["mtp"] - P["ehproj"], note="incl. norms", key="mblk")
m.op("**shared_head.norm → lm head** (main's)", HEAD, key="mhead")
m.tensor(40 / 0.08, "draft token t+2", key="mdraft")
ver = m.op("**verify**: main model checks the draft in its next forward", ACT, key="mver")
edge(ver, next_id, ("left", "right"), "accepted drafts = extra tokens", MTP)
node("note-mtp", m.cx - 150, m.y - 10, 600, 120,
     f"MTP = multi-token prediction ({C['num_nextn_predict_layers']} block, {fmt(P['mtp'])} stored). Trained to predict one "
     "token further; at inference serving stacks use it as a speculative-decoding draft. Not part of transformers' "
     "GlmMoeDsaForCausalLM — wiring here is read from its tensor names (no embed / head tensors of its own).")

# ---------- sources and the sibling Flash model ----------
ybot = o.y
node("note-src", -900, ybot + 200, 1300, 150,
     f"**Sources.** config.json + safetensors headers of zai-org/GLM-5.3 ({len(S)} tensors; FP8 E4M3 weights in "
     "128×128 blocks, scales not counted). Main model (L0–77, embed, head) = "
     f"**{fmt(sum(r[2] for r in B) - P['mtp'])}**, +{fmt(P['mtp'])} MTP. Dataflow from transformers "
     f"modeling_glm_moe_dsa.py. Context {C['max_position_embeddings']:,} positions; RoPE θ = {THETA:,} on {ROPE} dims only.")
node("note-flash", -900, ybot + 380, 1300, 150,
     "**GLM-5.3-Flash is a different design** (glm5_next, ~321B in safetensors): 45 layers, hidden 4096, hybrid of "
     "34 KDA linear-attention + 11 DSA layers, mHC ×4 residual lanes, 288 routed experts, NoPE MLA, and a 24-layer ViT "
     "for images/video. Not drawn here.")

save("model.canvas")

# ======================================================================= block.canvas
KT, CLIP, LBL = 0.05, 480, 300
IK, IQ, QC, KC = -1700, -1100, -350, 550  # indexer-key, indexer-query, query and key/value columns


def flow(cx, y, last=None, left=False):
    return Flow(cx, y, last=last, labels_left=left, kt=KT, clip=CLIP, lbl=LBL)


node("cap-btitle", -1900, -560, 1500, 170,
     "# GLM-5.3 — one decoder layer\n"
     f"A MoE layer that runs its own DSA indexer (L6, L10, … L74). Grey bars = tensors "
     f"({KT} px/channel, long ones clipped), boxes = operations with their weights per layer. Read top → bottom; "
     "the four middle columns run side by side.")
node("cap-bnote", -300, -560, 1500, 170,
     f"**Layer variants.** {N - N_FULL} of {N} layers are *shared*: they have no indexer weights and reuse the top-{TOPK} "
     f"indices of the nearest full layer above (left two columns skipped). L0–2 use a dense SwiGLU FFN "
     f"{DIM}→{C['intermediate_size']}→{DIM} ({fmt(P['densel'])}) instead of the MoE. Attention, norms and residuals are identical everywhere.")

mf = flow(0, -330)
mf.tensor(DIM, f"x [T, {DIM}] from previous layer", key="bx")
mf.op("**RMSNorm** (input_layernorm)", NORM, key="bn1")
xn = mf.tensor(DIM, f"x̂ [T, {DIM}]", key="bxn")
YB = mf.y + 40

# query path: low-rank query, then per-head split into no-position and rotary parts
q = flow(QC, YB, last=xn)
q.op(f"**q_a_proj** {DIM}→{QL} + RMSNorm", ATT, params_of(r"model\.layers\.0\.self_attn\.q_a_(?:proj|layernorm)\..*"), key="qa")
cq = q.tensor(QL, f"c_q [T, {QL}]  query latent", key="cq")
q.op(f"**q_b_proj** {QL} → {H}×{NOPE + ROPE}", ATT, params_of(r"model\.layers\.0\.self_attn\.q_b_proj\..*"), key="qb")
q.tensor(H * (NOPE + ROPE), f"q [T, {H}, {NOPE + ROPE}]", key="q")
q.op(f"split {NOPE} ‖ {ROPE} · **RoPE** on the {ROPE}", ATT, key="qr")
q_out = q.tensor(H * (NOPE + ROPE), f"q [T, {H}, {NOPE}+{ROPE}]", key="qo")

# key/value path: one shared latent per token is all the KV cache keeps
kv = flow(KC, YB, last=xn)
kv.op(f"**kv_a_proj_with_mqa** {DIM}→{KVL + ROPE}", ATT, params_of(r"model\.layers\.0\.self_attn\.kv_a_proj_with_mqa\..*"), key="kva")
kv.tensor(KVL + ROPE, f"[T, {KVL + ROPE}] = {KVL} latent ‖ {ROPE} rope", key="kvl")
kv.op(f"{KVL}: **RMSNorm** · {ROPE}: **RoPE** (1 head)", ATT, key="kvn")
kv.tensor(KVL + ROPE, f"KV cache: {KVL + ROPE} values / token / layer", key="kvc")
kv.op(f"**kv_b_proj** {KVL} → {H}×({NOPE}+{VD})", ATT, params_of(r"model\.layers\.0\.self_attn\.kv_b_proj\..*"), key="kvb")
kv.tensor(H * (NOPE + VD), f"k_nope [T,{H},{NOPE}] ‖ v [T,{H},{VD}]", key="kvx")
kv.op(f"k = [k_nope ‖ shared rope key]", ATT, key="kcat")
kv_out = kv.tensor(H * (NOPE + ROPE), f"k [T,{H},{NOPE + ROPE}] · v [T,{H},{VD}]", key="kvo")

# DSA indexer: cheap multi-head scores over all past tokens → top-k token ids
ik = flow(IK, YB, last=xn, left=True)
ik.op(f"**wk** {DIM}→{ID} + LayerNorm\n**weights_proj** {DIM}→{IH}", IDX,
      params_of(r"model\.layers\.0\.self_attn\.indexer\.(?:wk|k_norm|weights_proj)\..*"), key="iwk", h=80)
edges[-1].update(fromSide="left")  # leave x̂ sideways so the curve stays above the query column
k_i = ik.tensor(ID, f"k_I [T, {ID}] (cached) · w [T, {IH}]", key="ik")
iq = flow(IQ, YB + 56 + 40 + 9 + 40, last=cq)  # level with q_b_proj, fed sideways by c_q
iwq = iq.op(f"**wq_b** {QL} → {IH}×{ID}", IDX, params_of(r"model\.layers\.0\.self_attn\.indexer\.wq_b\..*"), key="iwq")
edges[-1].update(fromSide="left", toSide="right")  # c_q feeds the indexer query sideways
iq.tensor(IH * ID, f"q_I [T, {IH}, {ID}]", key="iqt")
isc = iq.op(f"**RoPE** first {ROPE} dims · score = Σ_h w_h·ReLU(q_I,h·k_I)", IDX, key="isc", h=80)
edge(k_i, isc, ("bottom", "left"))
iq.tensor(80 / KT, "index scores [T, T]  (causal)", key="isco")
iq.op(f"**top-{TOPK}** per query", IDX, key="itop")
idx_out = iq.tensor(160 / KT, f"top-k ids [T, {TOPK}]", key="iids")

# sparse attention core
YA = max(q.y, kv.y, iq.y) + 60
core = flow(0, YA)
att = core.op(f"**sparse MLA attention** — {H} heads; each query sees only its {TOPK} picked tokens", ATT, key="att", h=80)
edge(q_out, att)
edge(kv_out, att)
edge(idx_out, att, ("bottom", "left"))
core.tensor(H * VD, f"o [T, {H}×{VD}]", key="ao")
core.op(f"**o_proj** {H * VD} → {DIM}", ATT, params_of(r"model\.layers\.0\.self_attn\.o_proj\..*"), key="op")
core.tensor(DIM, f"[T, {DIM}]", key="ao2")
core.op("**+ residual** (x from the top)", ACT, key="r1")
core.tensor(DIM, f"h [T, {DIM}]", key="h1")
core.op("**RMSNorm** (post_attention_layernorm)", NORM, key="bn2")
hn = core.tensor(DIM, f"ĥ [T, {DIM}]", key="hn", left=True)
rt = core.op(f"**router** {DIM}→{R}, sigmoid · top-{A} by score+bias · weights normalised ×{C['routed_scaling_factor']}",
             FFN, P["router"], key="rt", h=80)
sh = node("shexp", 350, core.y - 80 - 40, 300, 80,
          f"**shared expert** (always on)  \nSwiGLU {DIM}→{MI * C['n_shared_experts']}→{DIM}  \n{fmt(P['shexp'])}", "#22c55e")
edge(hn, sh, ("right", "top"))
core.tensor(40 / KT, f"{A} expert ids + weights / token", key="rid")
ex = core.op(f"**{A} of {R} routed experts**, each SwiGLU {DIM}→{MI}→{DIM}", FFN, P["expert"],
             note=f"each · {fmt(R * P['expert'])} all", key="rex", h=80)
sm = core.op(f"**Σ** weighted experts + shared expert, **+ residual**", ACT, key="sum")
edge(sh, sm, ("bottom", "right"))
core.tensor(DIM, f"x [T, {DIM}] → next layer", key="bout")
node("note-kv", KC + CLIP / 2 + 20 + LBL + 40, YB - 20, 520, 130,
     f"Why MLA: the cache keeps one {KVL}+{ROPE} latent per token instead of {H} heads × ({NOPE + ROPE}+{VD}) keys and "
     f"values ({H * (NOPE + ROPE + VD):,} numbers); kv_b_proj re-expands it. Full layers also cache a {ID}-dim indexer key.")

save("block.canvas")
print("budget", [(r[0], fmt(r[2]), fmt(r[3])) for r in B])
print("active total", fmt(sum(r[3] for r in B)), "stored total", fmt(sum(r[2] for r in B)))
