"""Kimi K3 (moonshotai/Kimi-K3): model-level and layer-level canvases, derived from the checkpoint.

Inputs are the repo's config.json and shapes.json (safetensors headers, see fetch_safetensors_shapes.py).
Dataflow follows the repo's own modeling_kimi_linear.py (KimiLinearForCausalLM: KDA + gated MLA,
latent MoE, Attention Residuals) and modeling_kimi_k3.py (MoonViT vision tower + patch-merger projector).

  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale per canvas)
  - boxes       = operations, all one size; parameter / compute shares are the two bars at the top
  - layer rows  = one per layer, coloured by token mixer (linear_attn_config.kda_layers / full_attn_layers)
  - note-* ids  = annotations (dim, borderless); cap-* ids = borderless captions

Routed-expert weights are MXFP4: two 4-bit values per U8 byte in *.weight_packed, so those tensors
count 2 parameters per stored element; the U8 *.weight_scale tensors (one shared exponent per 32
weights) are quantisation metadata and are not counted.

usage: python3 kimi_k3_model_canvas.py <shapes.json> <config.json> <models/kimi-k3>
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
LAC = C["linear_attn_config"]

N, DIM, VOCAB = C["num_hidden_layers"], C["hidden_size"], C["vocab_size"]
H, NOPE, ROPE, VD = C["num_attention_heads"], C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
QL, KVL = C["q_lora_rank"], C["kv_lora_rank"]
KH, KD, CONV = LAC["num_heads"], LAC["head_dim"], LAC["short_conv_kernel_size"]
R, A, MI, LAT = C["num_experts"], C["num_experts_per_token"], C["moe_intermediate_size"], C["routed_expert_hidden_size"]
NSH, FF, BLK = C["num_shared_experts"], C["intermediate_size"], C["attn_res_block_size"]
CTX = C["max_position_embeddings"]
# config lists layers 1-based (modeling: is_kda_layer(i) == (i + 1) in kda_layers)
KDA_SET = {i - 1 for i in LAC["kda_layers"]}
MLA_SET = {i - 1 for i in LAC["full_attn_layers"]}
assert KDA_SET | MLA_SET == set(range(N)) and not KDA_SET & MLA_SET
TYPES = ["kda" if i in KDA_SET else "mla" for i in range(N)]
DENSE = [i < C["first_k_dense_replace"] for i in range(N)]
LM = "language_model.model."

EMB, NORM, FFN, HEAD, ACT = "6", "3", "4", "2", "#64748b"
KDA_C, MLA_C, VISION, RES, SHARED, LATC = "#22d3ee", "#f97316", "#a78bfa", "#e879f9", "#22c55e", "#10b981"
TYPE_COLOR = {"kda": KDA_C, "mla": MLA_C}
TYPE_NAME = {"kda": "KDA", "mla": "gated MLA"}


def count(k, dt, shp):
    if k.endswith(".weight_scale"):
        return 0  # MXFP4 shared exponents (metadata)
    n = math.prod(shp)
    return 2 * n if k.endswith(".weight_packed") else n  # 2 × FP4 per U8 byte


def params_of(pattern):
    rx = re.compile(pattern)
    return sum(count(k, dt, shp) for k, (dt, shp) in S.items() if rx.fullmatch(k))


L = re.escape(LM) + r"layers\.\d+\."
I_KDA, I_MLA = TYPES.index("kda", 1), TYPES.index("mla")  # L1 = first KDA layer with a MoE
P = dict(
    embed=params_of(re.escape(LM) + r"embed_tokens\.weight"),
    head=params_of(r"language_model\.lm_head\.weight"),
    kda=sum(params_of(re.escape(LM) + rf"layers\.{i}\.self_attn\..*") for i in sorted(KDA_SET)),
    mla=sum(params_of(re.escape(LM) + rf"layers\.{i}\.self_attn\..*") for i in sorted(MLA_SET)),
    dense=params_of(L + r"mlp\..*"),
    routed=params_of(L + r"block_sparse_moe\.experts\..*"),
    latent=params_of(L + r"block_sparse_moe\.routed_expert_(?:down_proj|up_proj|norm)\..*"),
    shared=params_of(L + r"block_sparse_moe\.(?:shared_experts|gate)\..*"),
    vit=params_of(r"vision_tower\..*"),
    proj=params_of(r"mm_projector\..*"),
    expert=params_of(re.escape(LM) + r"layers\.1\.block_sparse_moe\.experts\.0\..*"),
    kda1=params_of(re.escape(LM) + rf"layers\.{I_KDA}\.self_attn\..*"),
    mla1=params_of(re.escape(LM) + rf"layers\.{I_MLA}\.self_attn\..*"),
    shexp=params_of(re.escape(LM) + r"layers\.1\.block_sparse_moe\.shared_experts\..*"),
    router=params_of(re.escape(LM) + r"layers\.1\.block_sparse_moe\.gate\..*"),
    lat1=params_of(re.escape(LM) + r"layers\.1\.block_sparse_moe\.routed_expert_(?:down_proj|up_proj|norm)\..*"),
    densel=params_of(re.escape(LM) + r"layers\.0\.mlp\..*"),
    res1=params_of(re.escape(LM) + r"layers\.1\.(?:self_attention|mlp)_res_(?:norm|proj)\..*"),
)
N_MOE = N - sum(DENSE)
N_KDA, N_MLA = len(KDA_SET), len(MLA_SET)
stored_moe = sorted({int(re.match(re.escape(LM) + r"layers\.(\d+)", k)[1]) for k in S if ".block_sparse_moe." in k})
assert stored_moe == [i for i in range(N) if not DENSE[i]], "MoE layers must match first_k_dense_replace"
assert P["expert"] == 3 * LAT * MI, "routed expert = w1, w3 (LAT→MI) + w2 (MI→LAT)"
SNAPS = [i for i in range(N) if i % BLK == 0]  # layers that push a block snapshot (L0 pushes the embedding)


def budget():
    """[(name, colour, stored params, params multiplied per text token in the main forward, why-zero)]"""
    rows = [
        (f"routed experts {R}×{N_MOE}", FFN, P["routed"], N_MOE * A * P["expert"]),
        (f"KDA mixers ×{N_KDA}", KDA_C, P["kda"], P["kda"]),
        (f"gated MLA ×{N_MLA}", MLA_C, P["mla"], P["mla"]),
        (f"shared experts + router ×{N_MOE}", SHARED, P["shared"], P["shared"]),
        (f"latent down/up proj ×{N_MOE}", LATC, P["latent"], P["latent"]),
        (f"dense FFN ×{sum(DENSE)}", "#fbbf24", P["dense"], P["dense"]),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, P["head"], P["head"]),
        ("vision tower + projector", VISION, P["vit"] + P["proj"], 0, "runs per image"),
    ]
    other = params_of(r".*") - sum(r[2] for r in rows)
    rows.append(("norms + AttnRes queries", NORM, other, other))  # every remaining param is a norm / AttnRes vector
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

LAST2 = [i for i in range(N - 2, N) if TYPES[i] == "mla"]
node("cap-title", -900, -1300, 1250, 200,
     "# Kimi K3 — model\n"
     f"One continuous decoder stack L0–{N - 1}, read top → bottom; layers that repeat are drawn once with **×N**. "
     "**Grey bars = tensors**, length ∝ channels per token (0.08 px/channel). **Boxes = operations**, all one size; "
     "where parameters and compute live is the pair of bars above. T = sequence length. "
     "*Layer* tab = one decoder layer in detail.")
node("cap-legend", 450, -1300, 1250, 200,
     f"Row colours — **cyan: KDA** (Kimi Delta Attention, linear: fixed {KH}×{KD}×{KD} state, no KV cache) ×{N_KDA} · "
     f"**orange: gated MLA** (full softmax attention, {H} heads, {KVL}+{ROPE} latent KV cache, no RoPE) ×{N_MLA}. "
     f"Pattern 3 KDA + 1 MLA, and the last two layers (L{N - 2}, L{N - 1}) are both MLA.  \n"
     f"Every layer but L0 is MoE: **{A} of {R}** routed experts in a {LAT}-dim latent space + {NSH} shared experts, sigmoid router. "
     f"L0 has a dense FFN. Side boxes = **Attention-Residual blocks** of {BLK} layers.")

# ---------- input flow ----------
cx1 = COL_X[0] + ROW_W / 2
f = Flow(cx1, -980)
f.op("**raw text**  \"The cat sat…\"", ACT, key="raw")
f.op(f"**tokenizer** (tiktoken BPE, vocab {VOCAB})", ACT, key="tok")
f.tensor(40 / 0.08, "token ids [T] — one integer per token", key="ids")
f.op("**embed** — look up row *id*", EMB, P["embed"], note=f"table {VOCAB} × {DIM}", key="embed")
x0 = f.tensor(DIM, f"x [T, {DIM}]  (+ image tokens spliced in)", key="x")

# ---------- vision branch (MoonViT), spliced into x ----------
VW, MK = V["vt_hidden_size"], V["merge_kernel_size"]
v = Flow(-760, -1080, labels_left=True)
v.op("**image / video** frames", ACT, key="img")
pd = 3 * V["patch_size"] ** 2
v.tensor(pd, f"patches [P, {pd}] = 3 × {V['patch_size']}²", key="patch")
v.op(f"**patch embed** conv {V['patch_size']}×{V['patch_size']} → {VW} + learned {V['init_pos_emb_height']}×{V['init_pos_emb_width']} pos-embed",
     VISION, params_of(r"vision_tower\.patch_embed\..*"), key="pe", h=80)
v.tensor(VW, f"[P, {VW}]", key="vp")
v.op(f"**ViT ×{V['vt_num_hidden_layers']}** — {V['vt_num_attention_heads']} heads, 2D RoPE, GELU MLP {VW}→{V['vt_intermediate_size']}",
     VISION, params_of(r"vision_tower\.encoder\..*"), key="vit", h=80)
v.tensor(VW, f"[P, {VW}]", key="vf")
mw = VW * MK[0] * MK[1]
v.op(f"**merge {MK[0]}×{MK[1]}** patches (+ average over frames)", VISION, key="mrg")
v.tensor(mw, f"[P/{MK[0] * MK[1]}, {mw}]", key="vm")
v.op(f"**projector** {mw}→{mw}→{DIM} GELU + RMSNorm", VISION, P["proj"], key="vproj")
v.tensor(DIM, f"image tokens [P/{MK[0] * MK[1]}, {DIM}]", key="vt")
edge(v.last, x0, ("right", "left"), "spliced into x", VISION)
ROW0 = max(f.y, v.y) + 60


def kind(i):
    return TYPES[i], DENSE[i]


def kind_label(k):
    return f"{TYPE_NAME[k[0]]} · " + ("dense FFN" if k[1] else f"latent MoE {A}/{R}")


def kind_color(k):
    return TYPE_COLOR[k[0]]


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
    """Run-length prefix + the longest periodic stretch (period ≥ 2, repeated ≥ 2×) + run-length suffix.

    Kimi K3 ends on two MLA layers (L91–92) that break the 3 KDA : 1 MLA rhythm, so the period need not
    reach the last layer."""
    best = None  # (layers covered, start, period, repeats)
    for s0 in range(N):
        for p in range(2, 13):
            reps = 1
            while s0 + (reps + 1) * p <= N and all(kind(s0 + reps * p + j) == kind(s0 + j) for j in range(p)):
                reps += 1
            if reps >= 2 and (best is None or reps * p > best[0]):
                best = (reps * p, s0, p, reps)
    if best is None:
        return runs(0, N)
    _, s0, p, reps = best
    return runs(0, s0) + [("period", s0, p, reps)] + runs(s0 + p * reps, N)


# one column: a decoder-only stack; repeated layers are drawn once with ×N
SEG_W, SEG_H, SEG_GAP = 460, 56, 26
CX = ROW_W / 2
y = ROW0
prev = x0
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
    node("grp-period", gx, gy, SEG_W + 120, gh, f"×{reps} — L{s0}–{s0 + p * reps - 1}, a period of {p} layers", group=True)
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

# Attention-Residual blocks: the first layer of each block stores a snapshot; marked under the barcode
for s0 in SNAPS:
    node(f"snap{s0}", BX + s0 * (CELL + CELL_GAP), BY + 84, CELL, 16, "", RES)
node("note-snap", BX, BY + 110, max(N * (CELL + CELL_GAP), 600), 60,
     f"**pink** = AttnRes block start: L{SNAPS[0]}, L{SNAPS[1]}, …, L{SNAPS[-1]} (every {BLK}th layer) store a snapshot "
     "of the running sum before adding their own output.")

# ---------- output flow (below the stack) ----------
o = Flow(cx1, max(y, ROW0 + 840) + 40, last=LAST, labels_left=True)  # below the side notes
o.tensor(DIM, f"s [T, {DIM}]  partial sum of L{SNAPS[-1]}–{N - 1}", key="xo")
o.op(f"**AttnRes output mix** — softmax over b0…b{len(SNAPS) - 1} + s", RES,
     params_of(re.escape(LM) + r"output_attn_res_(?:norm|proj)\..*"), key="omix", h=80)
o.tensor(DIM, f"x [T, {DIM}]", key="xm")
o.op("**final RMSNorm**", NORM, key="fn")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm head** — {DIM} → {VOCAB}", HEAD, P["head"], note="not tied to embed", key="head")
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op("**softmax → sample**", ACT, key="samp")
o.op("**next token id**", ACT, key="next")

# ---------- notes ----------
KV_TOK = KVL + ROPE  # cached values per token per MLA layer (latent + shared key part)
STATE = KH * KD * KD  # values per KDA layer, independent of T
ynote = ROW0
node("note-attnres", -1440, ynote, 980, 330,
     f"**Attention Residuals (AttnRes).** There is no plain residual stream. Layers are grouped in blocks of {BLK}; "
     f"each block's outputs are summed into a partial sum *s*, and the first layer of each block freezes the previous "
     f"sum as a snapshot b (b0 = the embedding). Before its attention and again before its FFN, every layer feeds on a "
     f"**softmax-weighted mix of all snapshots so far + s** (L0's attention reads the embedding directly) — weights from one learned {DIM}-vector (proj × RMSNorm "
     f"weight) scored against each RMS-normalised candidate, per token. A layer *returns* s + its outputs, not x + outputs. "
     f"{len(SNAPS)} snapshots at the end; the output mix sees all {len(SNAPS)} + s.")
ynote2 = ROW0 + 360
node("note-mixers", -1440, ynote2, 980, 250,
     f"**Why mix KDA and MLA?** A KDA layer keeps one {KD}×{KD} matrix per head ({KH} heads = {STATE / 1e6:.2f}M values) "
     f"that it decays per channel and updates token by token — cost per token is constant however long the context.  \n"
     f"An MLA layer caches a {KVL}+{ROPE} = {KV_TOK}-value latent for every past token. At {CTX:,} tokens: "
     f"{N_MLA} MLA layers ≈ **{N_MLA * KV_TOK * CTX / 1e9:.1f}G** cached values vs "
     f"{N_KDA} KDA layers ≈ **{N_KDA * STATE / 1e6:.0f}M** state values (constant).")
ynote3 = ROW0 + 640
node("note-nope", -1440, ynote3, 980, 200,
     f"**No positional encoding anywhere in the text model** (mla_use_nope, no rotary weights or calls). Order comes "
     f"from the KDA layers: a causal {CONV}-tap conv on q/k/v and the recurrence itself. MLA keeps DeepSeek's split "
     f"({NOPE} + {ROPE} dims per head, the {ROPE} shared across heads) but never rotates it. Context {CTX:,} positions.")
node("note-moe", BX, BY + 220, 760, 300,
     f"**Latent MoE.** Routed experts don't see the {DIM}-wide hidden state: one down-proj squeezes it to {LAT}, the "
     f"{A} chosen experts (each SiTU-GLU {LAT}→{MI}→{LAT}, {fmt(P['expert'])}) work there, then RMSNorm + up-proj back "
     f"to {DIM}. This halves every expert. The {NSH} shared experts (one SiTU-GLU {DIM}→{NSH * MI}→{DIM}) and the "
     f"router stay full width. Expert weights are MXFP4 (4-bit, one shared exponent per 32 weights); everything else BF16.")
node("note-situ", BX, BY + 550, 760, 180,
     f"**SiTU-GLU** (every FFN): out = β·tanh(g/β)·σ(g) · β'·tanh(u/β') with g = gate, u = up, "
     f"β = {C['activation_situ_beta']:g}, β' = {C['activation_situ_linear_beta']:g} — a SiLU-like gate with both "
     "branches soft-clipped.")
ybot = o.y + 40 - ROW_H - 200
main = sum(r[2] for r in B) - P["vit"] - P["proj"]
node("note-src", -900, ybot + ROW_H + 200, 1300, 170,
     f"**Sources.** config.json + safetensors headers of moonshotai/Kimi-K3 ({len(S):,} tensors). Text model "
     f"(L0–{N - 1}, embed, head) = **{fmt(main)}**, + {fmt(P['vit'] + P['proj'])} vision. Routed experts are "
     "MXFP4 packed 2 per byte (counted ×2; U8 scales not counted). Dataflow from the repo's modeling_kimi_linear.py "
     "and modeling_kimi_k3.py. No MTP / draft head in the checkpoint.")

save("model.canvas")

# ======================================================================= block.canvas
KT, CLIP, LBL = 0.05, 480, 300
LX, SX, QC, KC, MC = -1250, -520, 700, 1650, 1175  # KDA, KDA side boxes, MLA query, MLA key/value, MLA core


def flow(cx, y, last=None, left=False):
    return Flow(cx, y, last=last, labels_left=left, kt=KT, clip=CLIP, lbl=LBL)


def lp(i, pat):
    return params_of(re.escape(f"{LM}layers.{i}.") + pat)


node("cap-btitle", -1900, -760, 1500, 170,
     "# Kimi K3 — one decoder layer\n"
     f"A MoE layer (all but L0). Grey bars = tensors ({KT} px/channel, long ones clipped), boxes = operations with their "
     f"weights per layer. The token mixer is **either** the KDA column (left, {N_KDA} layers) **or** the gated-MLA "
     f"columns (right, {N_MLA} layers).")
node("cap-bnote", -300, -760, 1500, 170,
     f"**Layer variants.** KDA layer mixer {fmt(P['kda1'])}, MLA layer mixer {fmt(P['mla1'])}. L0 uses a dense "
     f"SiTU-GLU FFN {DIM}→{FF}→{DIM} ({fmt(P['densel'])}) instead of the MoE. AttnRes, norms and MoE are identical "
     f"everywhere ({fmt(P['res1'])} of AttnRes vectors per layer).")

mf = flow(0, -520)
s_in = mf.tensor(DIM, f"s [T, {DIM}]  partial sum of this block", key="bs")
mix1 = mf.op("**AttnRes mix** (self_attention_res) — softmax over snapshots b + s, weighted sum", RES,
             lp(1, r"self_attention_res_(?:norm|proj)\..*"), key="mix1", h=80)
snap = node("bsnap", -820, -420, 320, 80,
            f"**block snapshots** b0…b(k−1)\n[T, k, {DIM}], k ≤ {len(SNAPS)}", RES)
edge(snap, mix1, ("right", "left"), "b")
mf.tensor(DIM, f"x [T, {DIM}]", key="bx")
mf.op("**RMSNorm** (input_layernorm)", NORM, key="bn1")
xn = mf.tensor(DIM, f"x̂ [T, {DIM}]", key="bxn")
node("note-push", 560, -560, 520, 110,
     f"At L0, L{BLK}, L{2 * BLK} … L{SNAPS[-1]} the mix runs, then s is frozen as the next snapshot and reset to 0, "
     "so this layer's outputs start a new block sum.")
YB = mf.y + 60

# ---- KDA column (linear attention, recurrent state) ----
d = flow(LX, YB, last=xn, left=True)
d.op(f"**q_proj, k_proj, v_proj** {DIM} → {KH * KD} each", KDA_C, lp(I_KDA, r"self_attn\.[qkv]_proj\..*"), key="dqkv")
d.tensor(KH * KD, f"q, k, v [T, {KH * KD}] each", key="dt1")
d.op(f"**short conv1d** ({CONV} taps, per channel) + SiLU — on q, k, v", KDA_C,
     lp(I_KDA, r"self_attn\.[qkv]_conv1d\..*"), key="dconv", h=80)
d.op(f"**split {KH} heads** × {KD} · L2-norm q, k", KDA_C, key="dsplit")
d.tensor(3 * KH * KD, f"q, k, v [T, {KH}, {KD}] each", key="dt2")
rule = d.op("**KDA delta rule** (per head)\nS ← Diag(α)·S ;  S ← S + β·k(vᵀ − kᵀS) ;  o = Sᵀq", KDA_C, key="drule", h=80)
d.tensor(KH * KD, f"o [T, {KH}, {KD}]   state S [{KH}, {KD}, {KD}]", key="dt3")
gnorm = d.op("**gated RMSNorm** (o_norm) — norm(o) · σ(g)", KDA_C, lp(I_KDA, r"self_attn\.o_norm\..*"), key="dgn")
d.tensor(KH * KD, f"[T, {KH * KD}]", key="dt4")
d.op(f"**o_proj** {KH * KD} → {DIM}", KDA_C, lp(I_KDA, r"self_attn\.o_proj\..*"), key="dout")
dend = d.tensor(DIM, f"[T, {DIM}]", key="dt5")

_ry = next(n for n in nodes if n["id"] == rule)["y"]
_gy = next(n for n in nodes if n["id"] == gnorm)["y"]
alog = S[f"{LM}layers.{I_KDA}.self_attn.A_log"][1][0]
ab = node("dab", SX - 160, _ry - 150, 320, 150,
          f"**f_a_proj** {DIM}→{KD}, **f_b_proj** {KD}→{KH * KD} + dt_bias, A_log → per-channel decay α "
          f"(gate_lower_bound {LAC['gate_lower_bound']:g})\n**b_proj** {DIM}→{KH} → β = σ(b)\n"
          f"{fmt(lp(I_KDA, r'self_attn\.(?:f_a_proj|f_b_proj|b_proj)\..*|' + re.escape(f'{LM}layers.{I_KDA}.') + r'self_attn\.(?:A_log|dt_bias)'))}",
          KDA_C)
edge(ab, rule, ("left", "right"), "α, β")
gz = node("dg", SX - 160, _gy - 10, 320, 76,
          f"**g_proj** {DIM} → {KH * KD} (full-rank output gate)\n{fmt(lp(I_KDA, r'self_attn\.g_proj\..*'))}", KDA_C)
edge(gz, gnorm, ("left", "right"), "g")

# ---- gated MLA (full attention, latent KV cache, no RoPE) ----
q = flow(QC, YB, last=xn, left=True)
q.op(f"**q_a_proj** {DIM}→{QL} + RMSNorm", MLA_C, lp(I_MLA, r"self_attn\.q_a_(?:proj|layernorm)\..*"), key="qa")
q.tensor(QL, f"c_q [T, {QL}]  query latent", key="cq")
q.op(f"**q_b_proj** {QL} → {H}×{NOPE + ROPE}", MLA_C, lp(I_MLA, r"self_attn\.q_b_proj\..*"), key="qb")
q_out = q.tensor(H * (NOPE + ROPE), f"q [T, {H}, {NOPE}+{ROPE}]  (no RoPE)", key="qo")

kv = flow(KC, YB, last=xn)
kv.op(f"**kv_a_proj_with_mqa** {DIM}→{KVL + ROPE}", MLA_C, lp(I_MLA, r"self_attn\.kv_a_proj_with_mqa\..*"), key="kva")
kv.tensor(KVL + ROPE, f"[T, {KVL + ROPE}] = {KVL} latent ‖ {ROPE} shared key", key="kvl")
kv.op(f"{KVL}: **RMSNorm** · {ROPE}: kept as-is", MLA_C, lp(I_MLA, r"self_attn\.kv_a_layernorm\..*"), key="kvn")
kv.tensor(KVL + ROPE, f"KV cache: {KVL + ROPE} values / token / layer", key="kvc")
kv.op(f"**kv_b_proj** {KVL} → {H}×({NOPE}+{VD})", MLA_C, lp(I_MLA, r"self_attn\.kv_b_proj\..*"), key="kvb")
kv.tensor(H * (NOPE + VD), f"k_nope [T,{H},{NOPE}] ‖ v [T,{H},{VD}]", key="kvx")
kv.op(f"k = [k_nope ‖ shared {ROPE}-dim key]", MLA_C, key="kcat")
kv_out = kv.tensor(H * (NOPE + ROPE), f"k [T,{H},{NOPE + ROPE}] · v [T,{H},{VD}]", key="kvo")

core = flow(MC, max(q.y, kv.y) + 40)
att = core.op(f"**causal softmax attention**\n{H} heads, qk {NOPE + ROPE} / v {VD}", MLA_C, key="att", h=80)
edge(q_out, att)
edge(kv_out, att)
core.tensor(H * VD, f"o [T, {H}×{VD}]", key="ao")
mg = core.op(f"**× σ(g_proj(x̂))** — output gate {DIM}→{H * VD}", MLA_C, lp(I_MLA, r"self_attn\.g_proj\..*"), key="ag")
core.tensor(H * VD, f"[T, {H * VD}]", key="ag2")
core.op(f"**o_proj** {H * VD} → {DIM}", MLA_C, lp(I_MLA, r"self_attn\.o_proj\..*"), key="aop")
mend = core.tensor(DIM, f"[T, {DIM}]", key="ao3")
node("note-kv", KC + CLIP / 2 + 20 + LBL + 40, YB - 20, 480, 150,
     f"Why MLA: the cache keeps one {KVL}+{ROPE} latent per token instead of {H} heads × ({NOPE + ROPE}+{VD}) keys "
     f"and values ({H * (NOPE + ROPE + VD):,} numbers); kv_b_proj re-expands it. Kimi K3 adds a sigmoid output gate "
     "and drops RoPE.")

# ---- merge + AttnRes + latent MoE ----
YM = max(d.y, core.y) + 100
r1 = node("badd1", -150, YM, 300, 56, "**s ← s + mixer output**", ACT)
edge(dend, r1, ("bottom", "left"), f"{N_KDA} KDA layers")
edge(mend, r1, ("bottom", "right"), f"{N_MLA} MLA layers")
g = flow(0, YM + 56 + 40, last=r1)
g.tensor(DIM, f"s [T, {DIM}]", key="bs2")
mix2 = g.op("**AttnRes mix** (mlp_res) — softmax over the same b + updated s", RES,
            lp(1, r"mlp_res_(?:norm|proj)\..*"), key="mix2", h=80)
snap2 = node("bsnap2", -820, g.y - 80 - 40, 320, 80, f"same snapshots b0…b(k−1)", RES)
edge(snap2, mix2, ("right", "left"), "b")
g.tensor(DIM, f"h [T, {DIM}]", key="bh")
g.op("**RMSNorm** (post_attention_layernorm)", NORM, key="bn2")
hn = g.tensor(DIM, f"ĥ [T, {DIM}]", key="hn", left=True)
g.op(f"**router** {DIM}→{R}, sigmoid · top-{A} by score+bias · weights renormalised ×{C['routed_scaling_factor']:g}",
     FFN, P["router"], key="rt", h=80)
sh = node("shexp", 350, g.y - 80 - 40, 300, 80,
          f"**{NSH} shared experts** (always on)  \nSiTU-GLU {DIM}→{NSH * MI}→{DIM}  \n{fmt(P['shexp'])}", SHARED)
edge(hn, sh, ("right", "top"))
g.tensor(40 / KT, f"{A} expert ids + weights / token", key="rid", left=True)
g.op(f"**routed_expert_down_proj** {DIM} → {LAT}", LATC, lp(1, r"block_sparse_moe\.routed_expert_down_proj\..*"), key="ldn")
g.tensor(LAT, f"z [T, {LAT}]  latent", key="lz", left=True)
g.op(f"**{A} of {R} routed experts**, each SiTU-GLU {LAT}→{MI}→{LAT} (MXFP4)", FFN, P["expert"],
     note=f"each · {fmt(R * P['expert'])} all", key="rex", h=80)
g.tensor(LAT, f"Σ weighted [T, {LAT}]", key="lsum", left=True)
g.op(f"**RMSNorm → routed_expert_up_proj** {LAT} → {DIM}", LATC,
     lp(1, r"block_sparse_moe\.routed_expert_(?:norm|up_proj)\..*"), key="lup")
g.tensor(DIM, f"[T, {DIM}]", key="lo", left=True)
sm = g.op("**+ shared experts**, then **s ← s + FFN output**", ACT, key="sum")
edge(sh, sm, ("bottom", "right"))
g.tensor(DIM, f"s [T, {DIM}] → next layer", key="bout")

save("block.canvas")
print("budget", [(r[0], fmt(r[2]), fmt(r[3])) for r in B])
print("active total", sum(r[3] for r in B), fmt(sum(r[3] for r in B)), "stored total", sum(r[2] for r in B), fmt(sum(r[2] for r in B)))
