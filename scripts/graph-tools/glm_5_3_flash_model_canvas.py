"""GLM-5.3-Flash (zai-org/GLM-5.3-Flash): model-level and layer-level canvases, derived from the checkpoint.

Inputs are the repo's config.json and shapes.json (safetensors headers, see fetch_safetensors_shapes.py).
Dataflow follows transformers' modeling_glm5_next.py (Glm5NextForConditionalGeneration): a hybrid of
Kimi-style KDA linear attention and DeepSeek-Sparse-Attention MLA layers, wrapped in mHC (manifold-constrained
hyper-connections, 4 residual lanes), plus a 24-block ViT. The MTP block (language_model.layers.45) is not in
transformers and is wired from its tensor names (enorm / hnorm / eh_proj / shared_head), the DeepSeek-V3 layout.

  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale per canvas)
  - boxes       = operations, all one size; parameter / compute shares are the two bars at the top
  - note-* ids  = annotations (dim, borderless); cap-* ids = borderless captions

usage: python3 glm_5_3_flash_model_canvas.py <shapes.json> <config.json> <models/glm-5-3-flash>
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
LA = C["linear_attn_config"]

N, DIM, VOCAB = C["num_hidden_layers"], C["hidden_size"], C["vocab_size"]
H, NOPE, ROPE, VD = C["num_attention_heads"], C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
QL, KVL = C["q_lora_rank"], C["kv_lora_rank"]
IH, ID, TOPK, POOL = C["index_n_heads"], C["index_head_dim"], C["index_topk"], C["index_kpool"]
R, A, MI = C["n_routed_experts"], C["num_experts_per_tok"], C["moe_intermediate_size"]
LH, LD, CONV, LB = LA["num_heads"], LA["head_dim"], LA["short_conv_kernel_size"], LA["gate_lower_bound"]
TYPES, MLP_T, IDX_T = C["layer_types"], C["mlp_layer_types"], C["indexer_types"]
HC = C["hc_mult"]
CTX = C["max_position_embeddings"]
MTP_L = N  # the next-token-prediction block is stored as language_model.layers.<N>
LM = r"model\.language_model\."
KDA_T, DSA_T = "linear_attention", "deepseek_sparse_attention"

EMB, NORM, FFN, HEAD, ACT, IDX, MTP = "6", "3", "4", "2", "#64748b", "5", "#f59e0b"
KDA, DSA, VISION, HCC = "#22d3ee", "#f97316", "#a78bfa", "#e879f9"
ROW_COLOR = {(KDA_T, "dense"): "#fbbf24", (KDA_T, "sparse"): KDA, (DSA_T, "sparse"): DSA}


def params_of(pattern):
    rx = re.compile(pattern)
    return sum(math.prod(shp) for k, (dt, shp) in S.items() if rx.fullmatch(k) and not k.endswith("_scale_inv"))


MAIN = LM + r"layers\.(?:[0-9]|[1-3][0-9]|4[0-4])"  # L0–44; layer 45 is the MTP block
assert N == 45, "MAIN regex assumes 45 main layers"
I_KDA, I_DSA, I_MOE = TYPES.index(KDA_T), TYPES.index(DSA_T), MLP_T.index("sparse")
L_ = lambda i: LM + rf"layers\.{i}\."  # noqa: E731
P = dict(
    embed=params_of(LM + r"embed_tokens\.weight"),
    head=params_of(r"lm_head\.weight"),
    kda=sum(params_of(L_(i) + r"self_attn\..*") for i in range(N) if TYPES[i] == KDA_T),
    mla=sum(params_of(L_(i) + r"self_attn\.(?!indexer\.).*") for i in range(N) if TYPES[i] == DSA_T),
    idx=params_of(MAIN + r"\.self_attn\.indexer\..*"),
    dense=params_of(MAIN + r"\.mlp\.(?:gate|up|down)_proj\..*"),
    routed=params_of(MAIN + r"\.mlp\.experts\..*"),
    shared=params_of(MAIN + r"\.mlp\.(?:shared_experts|gate)\..*"),
    hc=params_of(MAIN + r"\.hc_.*"),
    mtp=params_of(L_(MTP_L) + r".*"),
    vit=params_of(r"model\.visual\.(?!merger\.|downsample\.).*"),
    merger=params_of(r"model\.visual\.(?:merger|downsample)\..*"),
    expert=params_of(L_(I_MOE) + r"mlp\.experts\.0\..*"),
    shexp=params_of(L_(I_MOE) + r"mlp\.shared_experts\..*"),
    router=params_of(L_(I_MOE) + r"mlp\.gate\..*"),
    densel=params_of(L_(0) + r"mlp\..*"),
    ehproj=params_of(L_(MTP_L) + r"eh_proj\..*"),
    hc1=params_of(L_(0) + r"hc_attn_.*"),
    kda1=params_of(L_(I_KDA) + r"self_attn\..*"),
    dsa1=params_of(L_(I_DSA) + r"self_attn\..*"),
)
N_KDA, N_DSA, N_MOE = TYPES.count(KDA_T), TYPES.count(DSA_T), MLP_T.count("sparse")
# every DSA layer (and the MTP block) stores its own indexer: no cross-layer top-k sharing in this model
stored_idx = sorted({int(re.match(LM + r"layers\.(\d+)", k)[1]) for k in S if ".indexer." in k})
assert stored_idx == [i for i in range(N) if TYPES[i] == DSA_T] + [MTP_L], stored_idx
assert set(IDX_T) == {"full"}
assert not any(k.startswith(f"model.language_model.layers.{MTP_L}.hc_") for k in S)  # MTP block has no mHC


def budget():
    """[(name, colour, stored params, params multiplied per text token in the main forward, why-zero)]"""
    rows = [
        (f"routed experts {R}×{N_MOE}", FFN, P["routed"], N_MOE * A * P["expert"]),
        (f"KDA linear attention ×{N_KDA}", KDA, P["kda"], P["kda"]),
        (f"MLA attention ×{N_DSA}", DSA, P["mla"], P["mla"]),
        (f"DSA indexers ×{N_DSA}", IDX, P["idx"], P["idx"]),
        (f"shared expert + router ×{N_MOE}", "#22c55e", P["shared"], P["shared"]),
        (f"dense FFN ×{MLP_T.count('dense')}", "#fbbf24", P["dense"], P["dense"]),
        (f"mHC mixers ×{2 * N}", HCC, P["hc"], P["hc"]),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, P["head"], P["head"]),
        ("MTP block", MTP, P["mtp"], 0, "only when drafting"),
        ("ViT + merger", VISION, P["vit"] + P["merger"], 0, "runs per image"),
    ]
    other = params_of(r".*") - sum(r[2] for r in rows)
    rows.append(("norms", NORM, other, other))  # RMSNorm weights: every stored param is accounted for
    return rows


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


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


def box_y(nid):
    return next(n for n in nodes if n["id"] == nid)["y"]


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
ROW_W, ROW_H, ROW_GAP = 300, 34, 6
COL_X = (0,)
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
bar("bc", -1500, "What one text token is multiplied by in the main forward (active weights)", B, 3)

# memory that grows with context vs memory that does not (values per token from the shapes; bf16 cache, fp32 KDA state)
IDX_CACHE = 2 * ID + 1  # indexer caches k_I, the k-pool gate scores and a valid flag per token
kv_gb = N_DSA * (KVL + ROPE + IDX_CACHE) * CTX * 2 / 1e9
st_mb = N_KDA * LH * LD * LD * 4 / 1e6

node("cap-title", -900, -1300, 1250, 200,
     "# GLM-5.3-Flash — model\n"
     f"One continuous decoder stack L0–{N - 1}, read top → bottom; layers that repeat are drawn once with **×N**. "
     "**Grey bars = tensors**, length ∝ channels per token (0.08 px/channel). **Boxes = operations**, all one size; "
     "where parameters and compute live is the pair of bars above. T = sequence length. "
     "*Layer* tab = one decoder layer in detail (both token mixers).")
node("cap-legend", 450, -1300, 1250, 200,
     f"Row colours — **cyan**: KDA linear attention ({LH} heads, fixed {LD}×{LD} state per head) + MoE · "
     f"**amber**: KDA + dense SwiGLU FFN (L0–2) · **orange**: MLA made sparse by **DSA** — each query attends only "
     f"the top {TOPK} past tokens picked by its own indexer — + MoE. Layout = {N_DSA} × (3 KDA + 1 DSA) + 1 KDA.  \n"
     f"MoE = {A} of {R} routed experts + 1 shared, sigmoid router. The residual stream is **{HC} lanes** wide (mHC); "
     "each sub-layer reads a weighted mix of the lanes and writes back into all of them.")

# ---------- input flow ----------
cx1 = COL_X[0] + ROW_W / 2
f = Flow(cx1, -980)
f.op("**raw text**  \"The cat sat…\"", ACT, key="raw")
f.op(f"**tokenizer** (vocab {VOCAB})", ACT, key="tok")
f.tensor(40 / 0.08, "token ids [T] — one integer per token", key="ids")
f.op("**embed** — look up row *id*", EMB, P["embed"], note=f"table {VOCAB} × {DIM}", key="embed")
x0 = f.tensor(DIM, f"x [T, {DIM}]  (+ image/video tokens spliced in)", key="x")
f.op(f"**copy into {HC} residual lanes** (mHC)", HCC, key="hcin")
f.tensor(HC * DIM, f"X [T, {HC}, {DIM}]", key="xl")
ROW0 = f.y + 30


def kind(i):
    return TYPES[i], MLP_T[i]


def kind_label(k):
    return ("KDA" if k[0] == KDA_T else "MLA + DSA") + " · " + ("dense FFN" if k[1] == "dense" else f"MoE {A}/{R}")


def kind_color(k):
    return ROW_COLOR[k]


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
        for p in range(2, 13):
            if (N - s0) % p == 0 and (N - s0) // p >= 2 and all(kind(i) == kind(s0 + (i - s0) % p) for i in range(s0, N)):
                return runs(0, s0) + [("period", s0, p, (N - s0) // p)]
    return runs(0, N)


# one column: a decoder-only stack; repeated layers are drawn once with ×N
SEG_W, SEG_H, SEG_GAP = 460, 56, 26
CX = ROW_W / 2
y = ROW0
prev = f.last
for seg in segments():
    if seg[0] == "run":
        _, a, n = seg
        span = f"**L{a}–{a + n - 1}** ×{n}" if n > 1 else f"**L{a}**"
        nid = node(f"seg{a}", CX - SEG_W / 2, y, SEG_W, SEG_H, f"{span}\n{kind_label(kind(a))}", kind_color(kind(a)))
        edge(prev, nid)
        prev = nid
        y += SEG_H + SEG_GAP
        continue
    _, s0, p, reps = seg
    y += 40  # room for the group's label, which the viewer draws above the frame
    gx, gy = CX - SEG_W / 2 - 60, y
    ry = y + 50
    first = None
    for j in range(p):
        ls = [s0 + j + r * p for r in range(reps)]
        nid = node(f"per{j}", CX - SEG_W / 2, ry, SEG_W, SEG_H,
                   f"{kind_label(kind(s0 + j))}\nL{ls[0]}, {ls[1]}, …, {ls[-1]}", kind_color(kind(s0 + j)))
        edge(prev, nid)
        first = first or nid
        prev = nid
        ry += SEG_H + SEG_GAP
    edge(prev, first, ("left", "left"))  # loop back: the period runs again
    node("cap-repeat", gx - 170, (y + 50 + ry - SEG_GAP) / 2 - 30, 150, 60, f"## ↺ ×{reps}")
    gh = ry - y + 10
    node("grp-period", gx, gy, SEG_W + 120, gh, f"×{reps} — L{s0}–{N - 1}, a period of {p} layers", group=True)
    y += gh + SEG_GAP
LAST = prev

# barcode: every layer in order, one thin cell each, coloured like the rows
CELL, CELL_GAP = 16, 2
BX, BY = CX + SEG_W / 2 + 160, ROW0 + 40
node("cap-barcode", BX, BY - 50, max(N * (CELL + CELL_GAP), 600), 40,
     f"**Every layer, L0 → L{N - 1}** (1 cell = 1 layer, colours as in the rows)")
for i in range(N):
    node(f"lyr{i}", BX + i * (CELL + CELL_GAP), BY, CELL, 44, "", kind_color(kind(i)))
for i in sorted(set(range(0, N - 4, 10)) | {N - 1}):
    node(f"note-tick{i}", BX + i * (CELL + CELL_GAP) - 4, BY + 50, 60, 30, f"L{i}")

ymid = ROW0
node("note-mix", COL_X[0] - 620, ymid, 460, 330,
     f"**Why mix the two?** A KDA layer keeps one {LD}×{LD} matrix per head ({LH} heads) and updates it token by "
     "token, so its memory and compute per token stay flat however long the context.  \n"
     f"A DSA layer caches a {KVL}-value MLA latent + a {IDX_CACHE}-value indexer entry per past token.  \n"
     f"At {CTX:,} tokens: {N_DSA} DSA layers ≈ **{kv_gb:.1f} GB** cache (bf16) vs "
     f"{N_KDA} KDA layers ≈ **{st_mb:.0f} MB** state (fp32).")
node("note-pos", BX, BY + 120, 600, 200,
     f"**No positional encoding in the LLM.** qk_rope_head_dim = {ROPE}: MLA keys and queries are all "
     f"\"nope\" ({NOPE} dims/head) and the forward passes no RoPE. Token order comes from KDA's causal "
     f"conv ({CONV} taps) and recurrence. Only the ViT uses (2-D) RoPE. Context {CTX:,} positions.")

# ---------- vision branch, spliced into x ----------
v = Flow(-1100, -980, labels_left=True)
v.op("**image / video** frames", ACT, key="img")
pd = V["in_channels"] * V["temporal_patch_size"] * V["patch_size"] ** 2
v.tensor(pd, f"patches [P, {pd}] = 3 × {V['temporal_patch_size']} frames × {V['patch_size']}²", key="patch")
v.op(f"**ViT ×{V['depth']}** — Conv3d patch embed, 2-D RoPE", VISION, P["vit"], note=f"width {V['hidden_size']}", key="vit")
v.tensor(V["hidden_size"], f"[P, {V['hidden_size']}]", key="vf")
r = V["spatial_merge_size"]
v.op(f"**downsample** conv {r}×{r} + **merger** SwiGLU MLP", VISION, P["merger"], key="mrg")
v.tensor(V["out_hidden_size"], f"image tokens [P/{r * r}, {V['out_hidden_size']}]", key="vt")
edge(v.last, x0, ("right", "left"), "spliced into x")

# ---------- output flow (below the stack) ----------
o = Flow(cx1, y + 40, last=LAST, labels_left=True)
ylast = o.y
o.tensor(HC * DIM, f"X [T, {HC}, {DIM}]  after L{N - 1}", key="xo")
o.op(f"**hc_head** — plain mean of the {HC} lanes", HCC, key="hch")
o.tensor(DIM, f"x [T, {DIM}]", key="xm")
o.op("**final RMSNorm**", NORM, key="fn")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm head** — {DIM} → {VOCAB}", HEAD, P["head"], note="not tied to embed", key="head")
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op("**softmax → sample**", ACT, key="samp")
next_id = o.op("**next token id**", ACT, key="next")

# ---------- MTP block (language_model.layers.45): drafts one extra token ----------
tap = node("note-tap", cx1 + 720, ylast - 12, 260, 34, "↳ main hidden state → MTP")
m = Flow(2900, ylast + ROW_H + 60)
m.op("**embed** next token — same table as the main embed", EMB, key="membed")
m.tensor(DIM, f"e [T, {DIM}]", key="me")
cat = m.op("**enorm**(e) ‖ **hnorm**(h) — RMSNorm each, concat", MTP, key="mcat")
edge(tap, cat, ("right", "left"), color=MTP)
m.tensor(2 * DIM, f"[T, {2 * DIM}]", key="mc")
m.op(f"**eh_proj** {2 * DIM} → {DIM}", MTP, P["ehproj"], key="meh")
m.tensor(DIM, f"[T, {DIM}]", key="mh")
m.op(f"**1 decoder layer** — MLA + own indexer + MoE {A}/{R}, no mHC", MTP, P["mtp"] - P["ehproj"], note="incl. norms", key="mblk")
m.op("**shared_head.norm → lm head** (main's)", HEAD, key="mhead")
m.tensor(40 / 0.08, "draft token t+2", key="mdraft")
ver = m.op("**verify**: main model checks the draft in its next forward", ACT, key="mver")
edge(ver, next_id, ("left", "right"), "accepted drafts = extra tokens", MTP)
node("note-mtp", m.cx - 150, m.y - 10, 600, 140,
     f"MTP = multi-token prediction ({C['num_nextn_predict_layers']} block, {fmt(P['mtp'])} stored). Trained to predict one "
     "token further; serving stacks use it as a speculative-decoding draft. Not part of transformers' "
     "Glm5NextForConditionalGeneration — wiring read from its tensor names (no embed / head of its own, and no hc_* "
     "tensors, so it runs on a single 4096-wide stream).")

# ---------- sources and the big sibling ----------
ybot = max(o.y, m.y + 160) + 40 - ROW_H - 200
n_fp8 = sum(math.prod(s) for k, (d, s) in S.items() if d == "F8_E4M3")
node("note-src", -900, ybot + ROW_H + 200, 1300, 150,
     f"**Sources.** config.json + safetensors headers of zai-org/GLM-5.3-Flash ({len(S)} tensors). FP8 E4M3 in "
     f"128×128 blocks for experts, dense/shared FFNs and MLA projections ({fmt(n_fp8)} params; scales not counted); "
     "KDA, indexers, mHC, embed, head and ViT stay BF16. Main model (L0–44, embed, head) = "
     f"**{fmt(sum(r[2] for r in B) - P['mtp'] - P['vit'] - P['merger'])}**, +{fmt(P['mtp'])} MTP, "
     f"+{fmt(P['vit'] + P['merger'])} vision. Dataflow from transformers modeling_glm5_next.py.")
node("note-big", -900, ybot + ROW_H + 380, 1300, 170,
     "**vs GLM-5.3** (the big sibling, 753B): 78 → 45 layers, hidden 6144 → 4096, 256 → 288 routed experts "
     "(still top-8 + 1 shared). GLM-5.3 runs MLA+DSA in every layer with RoPE on 64 dims and lets 57 layers reuse "
     f"another layer's top-k; Flash makes 34 of 45 layers KDA linear attention, drops RoPE, gives each DSA layer its own "
     f"indexer scoring {POOL}-token pools, widens the residual into {HC} mHC lanes and adds a ViT for images and video.")

save("model.canvas")

# ======================================================================= block.canvas
_seq[0] = 0
KT, CLIP, LBL = 0.05, 480, 300
IK, IQ, QC, KC, LX, SX = -1700, -1100, -350, 550, 1700, 2560  # DSA columns, KDA column, KDA side boxes


def flow(cx, y, last=None, left=False):
    return Flow(cx, y, last=last, labels_left=left, kt=KT, clip=CLIP, lbl=LBL)


node("cap-btitle", -1900, -600, 1500, 200,
     "# GLM-5.3-Flash — one decoder layer\n"
     f"mHC reads the {HC} residual lanes → RMSNorm → **token mixer** → mHC writes back → mHC reads → RMSNorm → "
     f"**MoE** → mHC writes back. Left four columns = the **MLA + DSA** mixer ({N_DSA} layers: L3, L7, … L43); "
     f"right column = the **KDA** mixer ({N_KDA} layers). Grey bars = tensors ({KT} px/channel, long ones clipped); "
     "boxes = operations with their weights per layer.")
node("cap-bnote", -300, -600, 1500, 200,
     f"**Layer variants.** L0–2 use a dense SwiGLU FFN {DIM}→{C['intermediate_size']}→{DIM} ({fmt(P['densel'])}) "
     f"instead of the MoE. Mixer weights per layer: KDA **{fmt(P['kda1'])}**, MLA + indexer **{fmt(P['dsa1'])}**. "
     f"The two mHC mixers are {fmt(P['hc1'])} each. Every DSA layer has its own indexer (indexer_types all \"full\"). "
     "Checkpoint stores q/k/v short convs and each expert as separate tensors; transformers fuses them on load.")

mf = flow(0, -330)
mf.tensor(HC * DIM, f"X [T, {HC}, {DIM}] lanes from previous layer", key="bx")
mf.op(f"**mHC read** (attn_hc): RMSNorm(flatten {HC * DIM}) → {(2 + HC) * HC} weights; "
      f"x = Σ pre·X", HCC, P["hc1"], key="bhc1", h=80)
mf.tensor(DIM, f"x [T, {DIM}]", key="bxc")
mf.op("**RMSNorm** (input_layernorm)", NORM, key="bn1")
xn = mf.tensor(DIM, f"x̂ [T, {DIM}]", key="bxn")
YB = mf.y + 40

# ---- MLA + DSA: query path (no RoPE: every head dim is "nope") ----
q = flow(QC, YB, last=xn)
q.op(f"**q_a_proj** {DIM}→{QL} + RMSNorm", DSA, params_of(L_(I_DSA) + r"self_attn\.q_a_(?:proj|layernorm)\..*"), key="qa")
cq = q.tensor(QL, f"c_q [T, {QL}]  query latent", key="cq")
q.op(f"**q_b_proj** {QL} → {H}×{NOPE + ROPE}", DSA, params_of(L_(I_DSA) + r"self_attn\.q_b_proj\..*"), key="qb")
q_out = q.tensor(H * (NOPE + ROPE), f"q [T, {H}, {NOPE + ROPE}]  no RoPE", key="qo")

# ---- key/value path: one shared latent per token is all the cache keeps ----
kv = flow(KC, YB, last=xn)
kv.op(f"**kv_a_proj_with_mqa** {DIM}→{KVL + ROPE}", DSA, params_of(L_(I_DSA) + r"self_attn\.kv_a_proj_with_mqa\..*"), key="kva")
kv.tensor(KVL + ROPE, f"[T, {KVL + ROPE}] latent (0 rope dims)", key="kvl")
kv.op(f"**RMSNorm** (kv_a_layernorm)", DSA, key="kvn")
kv.tensor(KVL + ROPE, f"cache: {KVL} values / token / layer", key="kvc")
kv.op(f"**kv_b_proj** {KVL} → {H}×({NOPE}+{VD})", DSA, params_of(L_(I_DSA) + r"self_attn\.kv_b_proj\..*"), key="kvb")
kv_out = kv.tensor(H * (NOPE + VD), f"k [T,{H},{NOPE}] ‖ v [T,{H},{VD}]", key="kvo")

# ---- DSA indexer: scores 4-token pools of past keys → top-k token ids ----
ik = flow(IK, YB, last=xn, left=True)
ik.op(f"**wk** {DIM}→{ID} + LayerNorm · **weights_proj** {DIM}→{IH} · **pool gate** {DIM}→{ID}", IDX,
      params_of(L_(I_DSA) + r"self_attn\.indexer\.(?:wk|k_norm|weights_proj|index_kpool_compress_.*)(?:\..*)?"),
      key="iwk", h=80)
edges[-1].update(fromSide="left")  # leave x̂ sideways so the curve stays above the query column
ik.tensor(2 * ID + 1, f"k_I, gate [T, {ID}] each (cached) · w [T, {IH}]", key="ik")
ik.op(f"**k-pool**: softmax(gate + ape) mean of every {POOL} keys", IDX, key="ipool")
k_pool = ik.tensor(ID, f"pool keys [T/{POOL}, {ID}]", key="ipk")
iq = flow(IQ, YB + 56 + 40 + 9 + 40, last=cq)  # level with q_b_proj, fed sideways by c_q
iq.op(f"**wq_b** {QL} → {IH}×{ID}", IDX, params_of(L_(I_DSA) + r"self_attn\.indexer\.wq_b\..*"), key="iwq")
edges[-1].update(fromSide="left", toSide="right")  # c_q feeds the indexer query sideways
iq.tensor(IH * ID, f"q_I [T, {IH}, {ID}]", key="iqt")
isc = iq.op("score = Σ_h w_h·ReLU(q_I,h · pool key)", IDX, key="isc", h=80)
edge(k_pool, isc, ("bottom", "left"))
iq.tensor(80 / KT, f"pool scores [T, T/{POOL}]  (causal)", key="isco")
iq.op(f"**top-{TOPK // POOL} pools** ×{POOL} tokens + open tail", IDX, key="itop")
idx_out = iq.tensor(160 / KT, f"ids [T, ≤{TOPK}+{POOL - 1}]", key="iids")

# ---- sparse attention core ----
YA = max(q.y, kv.y, iq.y, ik.y) + 60
core = flow(0, YA)
att = core.op(f"**sparse MLA attention** — {H} heads × {NOPE}; each query sees only its picked tokens", DSA, key="att", h=80)
edge(q_out, att)
edge(kv_out, att)
edge(idx_out, att, ("bottom", "left"))
core.tensor(H * VD, f"o [T, {H}×{VD}]", key="ao")
core.op(f"**o_proj** {H * VD} → {DIM}", DSA, params_of(L_(I_DSA) + r"self_attn\.o_proj\..*"), key="aop")
dsa_end = core.tensor(DIM, f"[T, {DIM}]", key="ao2")

# ---- KDA column (Kimi Delta Attention) ----
kp = L_(I_KDA) + r"self_attn\."
k = flow(LX, YB, last=xn)
k.op(f"**q_proj, k_proj, v_proj** {DIM} → {LH * LD} each", KDA, params_of(kp + r"[qkv]_proj\..*"), key="kqkv")
edges[-1].update(toSide="left")  # enter sideways so the curve clears the MLA key/value column
k.tensor(3 * LH * LD, f"q, k, v [T, {3 * LH * LD}]", key="kt1")
k.op(f"**short conv** ({CONV} taps, per channel) + SiLU", KDA, params_of(kp + r"[qkv]_conv1d\..*"), key="kconv")
k.tensor(3 * LH * LD, f"q, k, v [T, {LH}, {LD}] each · L2-norm q, k", key="kt2")
rule = k.op(f"**delta rule**, per-channel decay\nS ← (I − β k kᵀ)·Diag(α)·S + β k vᵀ ;  o = Sᵀq", KDA, key="krule", h=80)
k.tensor(LH * LD, f"o [T, {LH}, {LD}]   state S [{LH}, {LD}, {LD}]", key="kt3")
gnorm = k.op("**gated RMSNorm** — norm(o) · σ(g)", KDA, params_of(kp + r"o_norm\..*"), key="kgn")
k.tensor(LH * LD, f"[T, {LH * LD}]", key="kt4")
k.op(f"**o_proj** {LH * LD} → {DIM}", KDA, params_of(kp + r"o_proj\..*"), key="kout")
kda_end = k.tensor(DIM, f"[T, {DIM}]", key="kt5")
ab = node("kab", SX - 150, box_y(rule) - 10, 300, 100,
          f"**f_a→f_b** {DIM}→{LD}→{LH * LD} + dt_bias, A_log · **b_proj** → {LH}\n"
          f"log α = {LB:g}·σ(e^A·(f + dt_bias)) · β = σ(b)\n"
          f"{fmt(params_of(kp + r'(?:[fb]_.*|A_log|dt_bias)'))}", KDA)
edge(ab, rule, ("left", "right"), "α, β")
gz = node("kg", SX - 150, box_y(gnorm) - 12, 300, 80,
          f"**g_a→g_b** {DIM}→{LD}→{LH * LD}\n{fmt(params_of(kp + r'g_[ab]_proj\..*'))}", KDA)
edge(gz, gnorm, ("left", "right"), "g")

# ---- mHC write-back, then the MoE ----
YM = max(core.y, k.y) + 80
w1 = node("bw1", -150, YM, 300, 80, "**mHC write**: X ← post ⊗ out + combᵀ·X  (X = lanes from the top)", HCC)
edge(dsa_end, w1, label=f"{N_DSA} of {N} layers")
edge(kda_end, w1, ("bottom", "right"), f"{N_KDA} of {N} layers")
g = flow(0, YM + 80 + 40, last=w1)
g.tensor(HC * DIM, f"X [T, {HC}, {DIM}]", key="bh1")
g.op("**mHC read** (ffn_hc) — same form, own weights", HCC, P["hc1"], key="bhc2")
g.tensor(DIM, f"h [T, {DIM}]", key="bh2")
g.op("**RMSNorm** (post_attention_layernorm)", NORM, key="bn2")
hn = g.tensor(DIM, f"ĥ [T, {DIM}]", key="hn", left=True)
rt = g.op(f"**router** {DIM}→{R}, sigmoid · top-{A} by score+bias · weights normalised ×{C['routed_scaling_factor']}",
          FFN, P["router"], key="rt", h=80)
sh = node("shexp", 350, g.y - 80 - 40, 300, 80,
          f"**shared expert** (always on)  \nSwiGLU {DIM}→{MI * C['n_shared_experts']}→{DIM}  \n{fmt(P['shexp'])}", "#22c55e")
edge(hn, sh, ("right", "top"))
g.tensor(40 / KT, f"{A} expert ids + weights / token", key="rid")
g.op(f"**{A} of {R} routed experts**, each SwiGLU {DIM}→{MI}→{DIM}", FFN, P["expert"],
     note=f"each · {fmt(R * P['expert'])} all", key="rex", h=80)
sm = g.op("**Σ** weighted experts + shared expert", ACT, key="sum")
edge(sh, sm, ("bottom", "right"))
g.tensor(DIM, f"[T, {DIM}]", key="bmo")
g.op("**mHC write**: X ← post ⊗ out + combᵀ·X", HCC, key="bw2")
g.tensor(HC * DIM, f"X [T, {HC}, {DIM}] → next layer", key="bout")
node("note-kv", 1300, -600, 420, 200,
     f"Why MLA: the cache keeps one {KVL}-value latent per token instead of {H} heads × ({NOPE}+{VD}) keys and "
     f"values ({H * (NOPE + VD):,} numbers); kv_b_proj re-expands it. The indexer also caches {IDX_CACHE} values / token. "
     f"KDA instead keeps a fixed {LH}×{LD}×{LD} state.")
node("note-hc", -1900 - 0, YM, 900, 130,
     f"**mHC** ({HC} lanes): per token, a {HC * DIM}→{(2 + HC) * HC} projection gives pre [{HC}] (how to read the lanes), "
     f"post [{HC}] (how to write the output back) and comb [{HC}×{HC}] (how lanes mix), comb made doubly stochastic by "
     f"{C['hc_sinkhorn_iters']} Sinkhorn steps so the stream norm cannot blow up.")

save("block.canvas")
print("budget", [(r[0], fmt(r[2]), fmt(r[3])) for r in B])
print("active total", fmt(sum(r[3] for r in B)), sum(r[3] for r in B), "stored total", fmt(sum(r[2] for r in B)), sum(r[2] for r in B))
print(f"cache {kv_gb:.1f}GB state {st_mb:.0f}MB")
