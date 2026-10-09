"""MiniMax-M3 (MiniMaxAI/MiniMax-M3): model-level and layer-level canvases, derived from the checkpoint.

Inputs are the repo's config.json and shapes.json (safetensors headers, see fetch_safetensors_shapes.py).
Dataflow follows transformers' modular_minimax_m3_vl.py (MiniMaxM3SparseForConditionalGeneration):
CLIP-style vision tower with 3D RoPE, a two-stage GELU projector, and a 60-layer GQA + MoE text model
where 57 layers run MiniMax Sparse Attention (a lightning indexer picks key blocks). The config declares
MTP modules but the checkpoint ships no MTP tensors, so none are drawn.

  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale per canvas)
  - boxes       = operations, all one size; parameter / compute shares are the two bars at the top
  - note-* ids  = annotations (dim, borderless); cap-* ids = borderless captions

usage: python3 minimax_m3_model_canvas.py <shapes.json> <config.json> <models/minimax-m3>
"""
import json
import math
import os
import re
import sys

S = json.load(open(sys.argv[1]))
CFG = json.load(open(sys.argv[2]))
OUT = sys.argv[3]
C, V, SA = CFG["text_config"], CFG["vision_config"], CFG["text_config"]["sparse_attention_config"]

N, DIM, VOCAB = C["num_hidden_layers"], C["hidden_size"], C["vocab_size"]
H, KVH, HD, ROT = C["num_attention_heads"], C["num_key_value_heads"], C["head_dim"], C["rotary_dim"]
R, A, MI, SI, DI = C["num_local_experts"], C["num_experts_per_tok"], C["intermediate_size"], C["shared_intermediate_size"], C["dense_intermediate_size"]
IH, ID, BS, TOPB, LOCB = SA["sparse_num_index_heads"], SA["sparse_index_dim"], SA["sparse_block_size"], SA["sparse_topk_blocks"], SA["sparse_local_block"]
MOE_F, SPARSE_F = C["moe_layer_freq"], SA["sparse_attention_freq"]
THETA, CTX = C["rope_theta"], C["max_position_embeddings"]
VD, VL, VH, VI, PS = V["hidden_size"], V["num_hidden_layers"], V["num_attention_heads"], V["intermediate_size"], V["patch_size"]
MS, TPS = CFG["img_token_compression_config"]["spatial_merge_size"], CFG["img_token_compression_config"]["temporal_patch_size"]
PH = CFG["projector_hidden_size"]

EMB, NORM, FFN, HEAD, ACT, ATT, IDX, VIS = "6", "3", "4", "2", "#64748b", "1", "5", "#a78bfa"
ROW_COLOR = {(0, 0): "#fbbf24", (1, 1): "5"}  # (moe?, sparse?) → colour

T = "language_model.model.layers"


def params_of(pattern):
    rx = re.compile(pattern)
    return sum(math.prod(shp) for k, (dt, shp) in S.items() if rx.fullmatch(k))


# layer kinds read from config must match what the checkpoint stores
moe_l = sorted({int(re.match(r"language_model\.model\.layers\.(\d+)", k)[1]) for k in S if ".block_sparse_moe." in k})
idx_l = sorted({int(re.match(r"language_model\.model\.layers\.(\d+)", k)[1]) for k in S if ".index_q_proj." in k})
assert moe_l == [i for i in range(N) if MOE_F[i]], moe_l
assert idx_l == [i for i in range(N) if SPARSE_F[i]], idx_l
assert not any(k.startswith("mtp") or ".mtp." in k for k in S), "checkpoint unexpectedly has MTP tensors"
assert all(dt in ("BF16", "F32") for dt, _ in S.values()), "unpacked BF16/F32 checkpoint expected"
assert set(ROW_COLOR) >= {(MOE_F[i], SPARSE_F[i]) for i in range(N)}
N_MOE, N_SP = len(moe_l), len(idx_l)
N_DENSE = N - N_MOE
L_MOE = moe_l[0]

P = dict(
    embed=params_of(r"language_model\.model\.embed_tokens\.weight"),
    head=params_of(r"language_model\.lm_head\.weight"),
    attn=params_of(T + r"\.\d+\.self_attn\.(?!index_).*"),
    idx=params_of(T + r"\.\d+\.self_attn\.index_.*"),
    dense=params_of(T + r"\.\d+\.mlp\..*"),
    routed=params_of(T + r"\.\d+\.block_sparse_moe\.experts\..*"),
    shared=params_of(T + r"\.\d+\.block_sparse_moe\.(?:shared_experts|gate|e_score_correction_bias)\b.*"),
    norms=params_of(T + r"\.\d+\.(?:input|post_attention)_layernorm\.weight") + params_of(r"language_model\.model\.norm\.weight"),
    patch=params_of(r"vision_tower\.vision_model\.embeddings\..*"),
    vit=params_of(r"vision_tower\.vision_model\.(?:encoder|pre_layrnorm)\..*"),
    proj=params_of(r"multi_modal_projector\..*"),
    merge=params_of(r"patch_merge_mlp\..*"),
    total=params_of(r".*"),
    expert=params_of(rf"{T}\.{L_MOE}\.block_sparse_moe\.experts\.0\..*"),
    attn1=params_of(rf"{T}\.{L_MOE}\.self_attn\.(?!index_).*"),
    idx1=params_of(rf"{T}\.{L_MOE}\.self_attn\.index_.*"),
    shexp=params_of(rf"{T}\.{L_MOE}\.block_sparse_moe\.shared_experts\..*"),
    router=params_of(rf"{T}\.{L_MOE}\.block_sparse_moe\.(?:gate\.weight|e_score_correction_bias)"),
    densel=params_of(rf"{T}\.0\.mlp\..*"),
)


def w(name, layer=L_MOE):
    """params of one named tensor group in one layer"""
    return params_of(rf"{T}\.{layer}\.{name}")


def budget():
    """[(name, colour, stored params, params multiplied per text token in the main forward, why-zero)]"""
    rows = [
        (f"routed experts {R}×{N_MOE}", FFN, P["routed"], N_MOE * A * P["expert"]),
        (f"GQA attention ×{N}", ATT, P["attn"], P["attn"]),
        (f"MSA indexers ×{N_SP}", IDX, P["idx"], P["idx"]),
        (f"shared expert + router ×{N_MOE}", "#22c55e", P["shared"], P["shared"]),
        (f"dense FFN ×{N_DENSE}", "#fbbf24", P["dense"], P["dense"]),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, P["head"], P["head"]),
        ("vision tower + projectors", VIS, P["patch"] + P["vit"] + P["proj"] + P["merge"], 0, "runs per image"),
        ("RMSNorms", NORM, P["norms"], P["norms"]),
    ]
    assert sum(r[2] for r in rows) == P["total"], (sum(r[2] for r in rows), P["total"])
    return rows


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M" if n >= 1e5 else f"{n / 1e3:.1f}K"


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
ROW_W, ROW_H, ROW_GAP = 300, 34, 6
COL_X = (0,)  # one column: repeated layers are drawn once with ×N (see segments())
BAR_W, BAR_H, MIN_SEG = 2600, 44, 16
OP_H, GAP = 56, 40


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
STORED, ACTIVE = sum(r[2] for r in B), sum(r[3] for r in B)
bar("bp", -1700, "Where the parameters are stored", B, 2)
bar("bc", -1500, "What one text token is multiplied by in the main forward (active weights)", B, 3)

node("cap-title", -900, -1300, 1250, 200,
     "# MiniMax-M3 — model\n"
     f"One continuous stack L0–{N - 1}, read top → bottom; layers that repeat are drawn once with **×N**; images enter from the far left. "
     "**Grey bars = tensors**, length ∝ channels per token (0.08 px/channel). **Boxes = operations**, all one size; "
     "where parameters and compute live is the pair of bars above. T = sequence length, P = vision patches. "
     "*Layer* tab = one decoder layer in detail.")
node("cap-legend", 450, -1300, 1250, 200,
     f"Every layer = **GQA attention** ({H} query / {KVH} KV heads × {HD}, per-head QK-norm, RoPE on {ROT} of {HD} dims).  \n"
     f"Row colours — **amber**: dense SwiGLU FFN {DI} + *full* causal attention (L0–{N_DENSE - 1}) · "
     f"**cyan**: MoE + **MiniMax Sparse Attention** — a {IH}-head lightning indexer scores {BS}-token key blocks and each "
     f"query attends only its top {TOPB} blocks (+ {LOCB} local), ≤ {TOPB * BS:,} keys. "
     f"MoE = {A} of {R} routed experts + 1 shared, sigmoid router, routed sum ×{C['routed_scaling_factor']}.")

# ---------- vision branch (far left); its last tensor feeds the splice op ----------
VCX = -1300
v = Flow(VCX, -980, labels_left=True)
v.op("**image / video** frames (dynamic resolution)", ACT, key="img")
v.tensor(3 * TPS * PS * PS, f"patches [P, {3 * TPS * PS * PS}]  (3 ch × {TPS} frames × {PS}×{PS})", key="patch")
v.op(f"**patch embed** Conv3d {TPS}×{PS}×{PS} → {VD}, no bias", VIS, P["patch"], key="pemb")
v.tensor(VD, f"[P, {VD}]", key="vf0")
v.op(f"**pre-LayerNorm + CLIP encoder ×{VL}** · {VH} heads · 3D RoPE · MLP {VI} GELU", VIS, P["vit"], key="vit", h=80)
v.tensor(VD, f"[P, {VD}]", key="vf1")
v.op(f"**projector** {VD} → {PH} → {DIM}, GELU (per patch)", VIS, P["proj"], key="proj")
v.tensor(DIM, f"[P, {DIM}]", key="vf2")
v.op(f"**group {MS}×{MS}** neighbouring patches into channels", VIS, key="grp")
v.tensor(DIM * MS * MS, f"[P/{MS * MS}, {DIM * MS * MS}]", key="vf3")
v.op(f"**patch_merge_mlp** {DIM * MS * MS} → {PH} → {DIM}, GELU", VIS, P["merge"], key="merge")
vt = v.tensor(DIM, f"image tokens [P/{MS * MS}, {DIM}]", key="vt")
vt_y = next(n["y"] for n in nodes if n["id"] == vt)

# ---------- text input flow, placed so the splice op lines up with the image tokens ----------
cx1 = COL_X[0] + ROW_W / 2
pre = 3 * (OP_H + GAP) + 2 * (9 + GAP)  # raw, tok, embed ops + ids, x tensors
f = Flow(cx1, vt_y + 4 - OP_H / 2 - pre)
f.op("**raw text** + image / video placeholders", ACT, key="raw")
f.op(f"**tokenizer** (BPE, vocab {VOCAB:,})", ACT, key="tok")
f.tensor(40 / 0.08, "token ids [T] — one integer per token", key="ids")
f.op("**embed** — look up row *id*", EMB, P["embed"], note=f"table {VOCAB} × {DIM}", key="embed")
f.tensor(DIM, f"x [T, {DIM}]", key="xt")
splice = f.op(f"**splice**: image tokens replace\nimage / video placeholder ids ({CFG['image_token_index']}, {CFG['video_token_index']})", ACT, key="splice")
edge(vt, splice, ("right", "left"), color=VIS)
x0 = f.tensor(DIM, f"x [T, {DIM}]", key="x")
ROW0 = f.y + 30

node("note-vision", VCX - 700, vt_y + 60, 720, 170,
     f"Vision tower: CLIP-style ViT ({VL} pre-LN blocks, width {VD}, q/k/v/out with bias) whose queries and keys get a "
     f"3D (t, h, w) RoPE. Each patch is first projected to {DIM} by `multi_modal_projector`, then {MS}×{MS} neighbours are "
     f"concatenated and fused to one text-width token by `patch_merge_mlp`, so the LM sees P/{MS * MS} image tokens. "
     "Video frames take the same path.")


def kind(i):
    return MOE_F[i], SPARSE_F[i]


def kind_label(k):
    return ("MoE" if k[0] else "dense FFN") + " · " + ("MSA sparse" if k[1] else "full attn")


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


SEG_W, SEG_H, SEG_GAP = 460, 56, 26
CX = ROW_W / 2
y = ROW0
prev = x0
for seg in segments():
    if seg[0] == "run":
        _, a, n = seg
        span = f"**L{a}–{a + n - 1}** ×{n}" if n > 1 else f"**L{a}**"
        nid = node(f"seg{a}", CX - SEG_W / 2, y, SEG_W, SEG_H, f"{span}\n{kind_label(kind(a))}", ROW_COLOR[kind(a)])
        edge(prev, nid)
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
        nid = node(f"per{j}", CX - SEG_W / 2, ry, SEG_W, SEG_H,
                   f"{kind_label(kind(s0 + j))}\nL{ls[0]}, {ls[1]}, …, {ls[-1]}", ROW_COLOR[kind(s0 + j)])
        edge(prev, nid)
        first = first or nid
        prev = nid
        ry += SEG_H + SEG_GAP
    edge(prev, first, ("left", "left"))  # loop back: the period runs again
    node("cap-repeat", gx - 170, (y + 50 + ry - SEG_GAP) / 2 - 30, 150, 60, f"## ↺ ×{reps}")
    gh = ry - y + 10
    node("grp-period", gx, y, SEG_W + 120, gh, f"×{reps} — L{s0}–{N - 1}, a period of {p} layers", group=True)
    y += gh + SEG_GAP
LAST, STACK_END = prev, y

# barcode: every layer in order, one thin cell each, coloured like the rows above
CELL, CELL_GAP = 16, 2
BX, BY = CX + SEG_W / 2 + 160, ROW0 + 40
node("cap-barcode", BX, BY - 50, N * (CELL + CELL_GAP), 40,
     f"**Every layer, L0 → L{N - 1}** (1 cell = 1 layer; amber dense + full attn, cyan MoE + MSA)")
for i in range(N):
    node(f"lyr{i}", BX + i * (CELL + CELL_GAP), BY, CELL, 44, "", ROW_COLOR[kind(i)])
for i in sorted(set(range(0, N, 10)) | ({N - 1} if (N - 1) % 10 >= 4 else set())):
    node(f"note-tick{i}", BX + i * (CELL + CELL_GAP) - 4, BY + 50, 60, 30, f"L{i}")

# side notes under the barcode
node("note-dense", BX, BY + 110, 560, 90,
     f"L0–{N_DENSE - 1}: no indexer weights — plain causal attention over every past token; dense SwiGLU {DIM}→{DI}→{DIM}.")
node("note-sparse", BX, BY + 230, 560, 150,
     f"L{N_DENSE}–{N - 1}: each layer runs its *own* indexer ({fmt(P['idx1'])}); it adds a {ID}-dim key per token to the "
     f"cache next to the {2 * KVH * HD} K+V values. Every one of the {KVH} KV groups gets its own block pick, so at 1M "
     f"context a query reads {TOPB * BS:,} keys instead of up to {CTX:,}.")

# ---------- output flow (below the right column) ----------
o = Flow(CX, STACK_END + 40, last=LAST, labels_left=True)
o.tensor(DIM, f"x [T, {DIM}]  after L{N - 1}", key="xo")
o.op("**final RMSNorm** (Gemma-style, × (1+w))", NORM, key="fn")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm head** — {DIM} → {VOCAB}", HEAD, P["head"], note="not tied to embed", key="head")
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op("**softmax → sample**", ACT, key="samp")
o.op("**next token id**", ACT, key="next")
node("note-mtp", CX + 300, o.y - 140, 700, 150,
     f"**No drafting head drawn.** config.json declares `num_mtp_modules` = {C['num_mtp_modules']} and "
     f"`num_nextn_predict_layers` = {C['num_nextn_predict_layers']}, but none of the {len(S):,} checkpoint tensors is an "
     "MTP weight, and transformers drops `mtp.*` keys on load.")

# ---------- sources ----------
node("note-src", -900, o.y + 40, 1300, 170,
     f"**Sources.** config.json + safetensors headers of MiniMaxAI/MiniMax-M3 ({len(S):,} tensors, BF16; router weights "
     f"and the patch-embed conv in F32). Text model (L0–{N - 1}, embed, head) = "
     f"**{fmt(STORED - B[7][2])}**, + {fmt(B[7][2])} vision. Dataflow from transformers modular_minimax_m3_vl.py. "
     f"Context {CTX:,} positions; RoPE θ = {THETA:,} on the first {ROT} of {HD} head dims, no scaling.")

save("model.canvas")

# ======================================================================= block.canvas
KT, CLIP, LBL = 0.05, 480, 300
IK, IQ, QC, KC = -1700, -1100, -350, 550  # indexer-key, indexer-query, query and key/value columns


def flow(cx, y, last=None, left=False):
    return Flow(cx, y, last=last, labels_left=left, kt=KT, clip=CLIP, lbl=LBL)


node("cap-btitle", -1900, -560, 1500, 170,
     "# MiniMax-M3 — one decoder layer\n"
     f"A MoE layer with MiniMax Sparse Attention (L{N_DENSE}–{N - 1}). Grey bars = tensors "
     f"({KT} px/channel, long ones clipped), boxes = operations with their weights per layer. Read top → bottom; "
     "the four middle columns run side by side.")
node("cap-bnote", -300, -560, 1500, 170,
     f"**Layer variants.** L0–{N_DENSE - 1} have no indexer (left two columns skipped): every query attends all past "
     f"tokens. They also swap the MoE for a dense SwiGLU-OAI FFN {DIM}→{DI}→{DIM} ({fmt(P['densel'])}). "
     "Attention projections, QK-norm, partial RoPE, norms and residuals are identical in all 60 layers.")

mf = flow(0, -330)
mf.tensor(DIM, f"x [T, {DIM}] from previous layer", key="bx")
mf.op("**RMSNorm** (input_layernorm, × (1+w))", NORM, key="bn1")
xn = mf.tensor(DIM, f"x̂ [T, {DIM}]", key="bxn")
YB = mf.y + 40

# query path
q = flow(QC, YB, last=xn)
q.op(f"**q_proj** {DIM} → {H}×{HD}", ATT, w(r"self_attn\.q_proj\..*"), key="qp")
q.tensor(H * HD, f"q [T, {H}, {HD}]", key="q")
q.op(f"**q_norm** per head ({HD}) · **RoPE** on first {ROT} dims", ATT, key="qn")
q_out = q.tensor(H * HD, f"q [T, {H}, {HD}]", key="qo")

# key/value path
kv = flow(KC, YB, last=xn)
kv.op(f"**k_proj, v_proj** {DIM} → {KVH}×{HD} each", ATT, w(r"self_attn\.[kv]_proj\..*"), key="kvp")
kv.tensor(2 * KVH * HD, f"k, v [T, {KVH}, {HD}]", key="kv")
kv.op(f"k: **k_norm** per head · **RoPE** first {ROT} dims", ATT, key="kn")
kv_out = kv.tensor(2 * KVH * HD, f"KV cache: {2 * KVH * HD} values / token / layer", key="kvc")

# lightning indexer: cheap per-group scores over all past keys → top key blocks
ik = flow(IK, YB, last=xn, left=True)
ik.op(f"**index_k_proj** {DIM}→{ID}\n+ RMSNorm · RoPE first {ROT}", IDX, w(r"self_attn\.index_k_(?:proj|norm)\..*"), key="ikp", h=80)
edges[-1].update(fromSide="left")  # leave x̂ sideways so the curve stays above the query column
k_i = ik.tensor(ID, f"k_I [T, {ID}] (cached, 1 head)", key="ik")
iq = flow(IQ, YB, last=xn)
iq.op(f"**index_q_proj** {DIM}→{IH}×{ID}\n+ RMSNorm · RoPE first {ROT}", IDX, w(r"self_attn\.index_q_(?:proj|norm)\..*"), key="iqp", h=80)
edges[-1].update(fromSide="left", toSide="right")
iq.tensor(IH * ID, f"q_I [T, {IH}, {ID}]", key="iqt")
isc = iq.op(f"score = q_I,h · k_I (causal) · **max-pool** over {BS}-key blocks", IDX, key="isc", h=80)
edge(k_i, isc, ("bottom", "left"))
iq.tensor(80 / KT, f"block scores [T, {IH}, T/{BS}]", key="isco")
iq.op(f"local {LOCB} block forced in · **top-{TOPB}** blocks per head", IDX, key="itop")
idx_out = iq.tensor(160 / KT, f"block ids [T, {IH}, {TOPB}]", key="iids")

# sparse attention core
YA = max(q.y, kv.y, iq.y) + 60
core = flow(0, YA)
att = core.op(f"**block-sparse GQA attention** — {H} heads in {KVH} groups; each group reads its {TOPB}×{BS} picked keys",
              ATT, key="att", h=80)
edge(q_out, att)
edge(kv_out, att)
edge(idx_out, att, ("bottom", "left"))
core.tensor(H * HD, f"o [T, {H}×{HD}]", key="ao")
core.op(f"**o_proj** {H * HD} → {DIM}", ATT, w(r"self_attn\.o_proj\..*"), key="op")
core.tensor(DIM, f"[T, {DIM}]", key="ao2")
core.op("**+ residual** (x from the top)", ACT, key="r1")
core.tensor(DIM, f"h [T, {DIM}]", key="h1")
core.op("**RMSNorm** (post_attention_layernorm)", NORM, key="bn2")
hn = core.tensor(DIM, f"ĥ [T, {DIM}]", key="hn", left=True)
rt = core.op(f"**router** {DIM}→{R} (F32), sigmoid · top-{A} by score+bias · weights normalised",
             FFN, P["router"], key="rt", h=80)
sh = node("shexp", 350, core.y - 80 - 40, 300, 80,
          f"**shared expert** (always on)  \nSwiGLU-OAI {DIM}→{SI}→{DIM}  \n{fmt(P['shexp'])}", "#22c55e")
edge(hn, sh, ("right", "top"))
core.tensor(40 / KT, f"{A} expert ids + weights / token", key="rid")
ex = core.op(f"**{A} of {R} routed experts**, each SwiGLU-OAI {DIM}→{MI}→{DIM}", FFN, P["expert"],
             note=f"each · {fmt(R * P['expert'])} all", key="rex", h=80)
sm = core.op(f"**Σ** weighted experts **×{C['routed_scaling_factor']}** + shared expert, **+ residual**", ACT, key="sum", h=80)
edge(sh, sm, ("bottom", "right"))
core.tensor(DIM, f"x [T, {DIM}] → next layer", key="bout")
node("note-kv", KC + CLIP / 2 + 20 + LBL + 40, YB - 20, 520, 150,
     f"GQA: {H // KVH} query heads share each K/V head, so the cache holds {2 * KVH * HD} numbers per token per layer "
     f"(+ {ID} for the indexer key). The indexer has one head per KV group ({IH} = {KVH}) and no value path; it only "
     "decides which key blocks the main attention may read.")
node("note-act", 700, YA + 300, 560, 170,
     f"SwiGLU-OAI (experts, shared expert, dense FFN): gate clamped ≤ {C['swiglu_limit']}, up clamped to "
     f"±{C['swiglu_limit']}, out = (up + 1) · gate · σ({C['swiglu_alpha']}·gate). Experts store w1 = gate, w3 = up, "
     f"w2 = down, each {MI}×{DIM}.")

save("block.canvas")
print("budget", [(r[0], fmt(r[2]), fmt(r[3])) for r in B])
print("active total", ACTIVE, fmt(ACTIVE), "stored total", STORED, fmt(STORED))
