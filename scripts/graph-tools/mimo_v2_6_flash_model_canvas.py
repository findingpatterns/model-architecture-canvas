"""Model + Layer tabs of Xiaomi MiMo-V2.6-Flash (HF: XiaomiMiMo/MiMo-V2.6-Flash-RL).

Read top → bottom:
  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale per canvas)
  - boxes       = operations, all one size; parameter / compute shares are the two bars at the top
  - layer rows  = one per layer, coloured by attention type from config.hybrid_layer_pattern
  - note-* ids  = annotations (dim, borderless); cap-* ids = borderless captions
Numbers come from the checkpoint: shapes.json (safetensors headers of the main index), extra.json
(headers of dflash/dflash_draft_model.safetensors and audio_tokenizer/model.safetensors),
config.json and dflash/config.json. Routed-expert weights are stored as U8 = two packed MXFP4
values, so they count twice; `*.weight_scale*` tensors are quantisation scales, not parameters.

usage: python3 mimo_v2_6_flash_model_canvas.py <shapes.json> <extra.json> <config.json> <dflash_config.json> <models/mimo-v2-6-flash>
"""
import json
import math
import os
import re
import sys

S = json.load(open(sys.argv[1]))
X = json.load(open(sys.argv[2]))
C = json.load(open(sys.argv[3]))
DF = json.load(open(sys.argv[4]))
OUT = sys.argv[5]
DFS, ATS = X["dflash/dflash_draft_model.safetensors"], X["audio_tokenizer/model.safetensors"]

N, DIM, VOCAB = C["num_hidden_layers"], C["hidden_size"], C["vocab_size"]
PATTERN, MOE_FREQ = C["hybrid_layer_pattern"], C["moe_layer_freq"]  # pattern: 1 = SWA, 0 = global attention
R, A = C["n_routed_experts"], C["num_experts_per_tok"]
VC, AC = C["vision_config"], C["audio_config"]
DC = DF["dflash_config"]
TAPS = DC["target_layer_ids"]
MTP_N = C["num_nextn_predict_layers"]

# colours (JSON Canvas presets "1"–"6" or hex)
EMB, NORM, FFN, HEAD, ACT, ATTN = "6", "3", "4", "2", "#64748b", "5"
SWA_C, GA_C, DENSE_C = "#94a3b8", "5", "#f59e0b"
VISION, AUDIO, MTP_C, DFL = "#a78bfa", "#f472b6", "#22d3ee", "#a3e635"

OP_W, OP_H, GAP = 300, 56, 40
LBL_W = 440
MIN_SEG = 16  # the viewer draws no box narrower than ~14px, so thinner budget segments are left out


def params_of(pattern, src=S):
    rx = re.compile(pattern)
    return sum(math.prod(shp) * (2 if dt == "U8" else 1)
               for k, (dt, shp) in src.items() if rx.fullmatch(k) and "weight_scale" not in k)


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


def mode(i):
    if i == 0 or not MOE_FREQ[i]:
        return "GA dense"
    return "SWA" if PATTERN[i] == 1 else "GA"


ROW_COLOR = {"SWA": SWA_C, "GA": GA_C, "GA dense": DENSE_C}
N_SWA = sum(PATTERN)
N_GA = N - N_SWA

P = {
    "embed": params_of(r"model\.embed_tokens\.weight"),
    "head": params_of(r"lm_head\.weight"),
    "vit": params_of(r"visual\..*"),
    "vit_merger": params_of(r"visual\.merger\..*"),
    "aud_local": params_of(r"audio_encoder\.input_local_transformer\..*"),
    "aud_proj": params_of(r"audio_encoder\.projection\..*"),
    "speech_emb": params_of(r"speech_embeddings\..*"),
    "mtp": params_of(r"model\.mtp\..*"),
    "layers": params_of(r"model\.layers\..*"),
    "experts": params_of(r"model\.layers\.\d+\.mlp\.experts\..*"),
    "expert": params_of(r"model\.layers\.1\.mlp\.experts\.0\..*"),
    "router": params_of(r"model\.layers\.1\.mlp\.gate\..*"),
    "attn_swa": params_of(r"model\.layers\.1\.self_attn\..*"),
    "attn_ga": params_of(r"model\.layers\.5\.self_attn\..*"),
    "dense_ffn": params_of(r"model\.layers\.0\.mlp\..*"),
    "dflash": params_of(r".*", DFS),
    "tok_enc": params_of(r"encoder\.(?!quantizer\.).*", ATS),
    "total": params_of(r".*"),
}
LAYER_P = [params_of(rf"model\.layers\.{i}\..*") for i in range(N)]
N_MOE = sum(1 for i in range(N) if MOE_FREQ[i])


def budget():
    """[(name, colour, stored params, params multiplied per text token[, why zero])] for the main checkpoint."""
    backbone = P["layers"] - P["experts"]  # attention, routers, norms, dense FFN of L0
    audio = P["aud_local"] + P["aud_proj"] + P["speech_emb"]
    rows = [
        (f"routed experts ({N_MOE}×{R})", FFN, P["experts"], N_MOE * A * P["expert"]),
        ("attention + router + L0 FFN", ATTN, backbone, backbone),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, P["head"], P["head"]),
        (f"MTP ×{MTP_N}", MTP_C, P["mtp"], 0, "only when drafting"),
        ("ViT + merger", VISION, P["vit"], 0, "runs per image / frame"),
        ("audio encoder", AUDIO, audio, 0, "runs per audio clip"),
    ]
    other = P["total"] - sum(r[2] for r in rows)
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

    def __init__(self, cx, y, kt, clip, last=None, labels_left=False):
        self.cx, self.y, self.kt, self.clip, self.last, self.labels_left = cx, y, kt, clip, last, labels_left

    def _id(self, key):
        _seq[0] += 1
        return f"{key}{_seq[0]}"

    def _link(self, top, bottom, label=None):
        if self.last:
            edge(self.last, top, label=label)
        self.last = bottom

    def op(self, text, color, params=0, note="", key="op", h=OP_H):
        """Operation box; sizes are told by the budget bars, so every box has the same width."""
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
        clip = f" — bar clipped (real {real:.0f}px)" if real > w else ""
        half = max(w / 2, OP_W / 2) + 20
        lx = self.cx - half - LBL_W if self.labels_left else self.cx + half
        node(self._id("note-lbl"), lx, self.y - 10, LBL_W, 34, f"`{label}`{clip}")
        self._link(ids[0], ids[-1])
        self.y += lanes * 14 + GAP
        return ids[0], ids[-1]


def save(name):
    json.dump(dict(nodes=nodes, edges=edges), open(os.path.join(OUT, name), "w"), indent=1, ensure_ascii=False)
    nodes.clear()
    edges.clear()


def attn_shape(swa):
    pre = "swa_" if swa else ""
    hq = C[f"{pre}num_attention_heads"]
    hkv = C[f"{pre}num_key_value_heads"]
    hd = C[f"{pre}head_dim"]
    vd = C[f"{pre}v_head_dim"]
    return hq, hkv, hd, vd


ROPE_DIM = int(C["head_dim"] * C["partial_rotary_factor"])

# =====================================================================================
# model.canvas
# =====================================================================================
KT, CLIP = 0.06, 1200
ROW_W, ROW_H, ROW_GAP = 300, 34, 6
CX = ROW_W / 2
BAR_X, BAR_W, BAR_H = -1300, 2900, 44


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
            node(f"{key}{k}", round(x), y, round(x + w) - round(x), BAR_H, f"**{r[0]}**" if w > 220 else "", r[1])
        x += w
        why = r[4] if len(r) > 4 and not r[idx] else ""
        legend.append(f"{r[0]} **{fmt(r[idx]) if r[idx] else '0'}**" + (f" ({why})" if why else f" ({share:.1%})"))
    node(f"cap-{key}l", BAR_X, y + BAR_H + 6, BAR_W, 64, " · ".join(legend))


B = budget()
bar("bp", -1760, "Where the parameters are stored (main checkpoint)", B, 2)
bar("bc", -1560, "What one text token is multiplied by (active weights)", B, 3)

hq_s, hkv_s, hd_s, vd_s = attn_shape(True)
hq_g, hkv_g, hd_g, vd_g = attn_shape(False)
node("cap-title", BAR_X, -1420, 1400, 200,
     "# MiMo-V2.6-Flash — model\n"
     f"Read top → bottom. **Grey bars = tensors**, length ∝ channels per token ({KT} px/channel). "
     "**Boxes = operations**, all the same size; where the parameters and the compute live is the pair of bars above. "
     "T = sequence length. Zoom in: *Layer* tab = one decoder layer.  \n"
     "Source: HF `XiaomiMiMo/MiMo-V2.6-Flash-RL` config.json, modeling_mimo_v2.py, dflash/ and safetensors headers.")
node("cap-legend", BAR_X + 1460, -1420, 1440, 200,
     f"Layer rows — **SWA** ({N_SWA}): sliding window {C['sliding_window']} tokens, {hq_s} Q / {hkv_s} KV heads, "
     f"learned per-head sink logit, RoPE θ {C['swa_rope_theta']:.0e} · **GA** ({N_GA}): global causal attention, "
     f"{hq_g} Q / {hkv_g} KV heads, RoPE θ {C['rope_theta']:.0e} (1M context) · **L0** is GA with a dense FFN.  \n"
     f"Every other layer: MoE, {R} routed experts, top-{A}, sigmoid router, no shared expert. "
     f"MoE layer ≈ **{fmt(LAYER_P[1])}**, {P['experts'] / P['layers']:.0%} of all layer weights are routed experts. "
     f"Pattern: L0 GA · (4 SWA, GA) · then (5 SWA, GA) ×{(N - 6) // 6}.")

# ---------- input flow ----------
f = Flow(CX, -1140, KT, CLIP)
f.op("**raw text**  \"The cat sat…\"", ACT, key="raw")
f.op(f"**tokenizer** (BPE, vocab {VOCAB})", ACT, key="tok")
f.tensor(40 / KT, "token ids [T] — one integer per token", key="ids")
f.op("**embed_tokens** — look up row *id*", EMB, P["embed"], note=f"{VOCAB} × {DIM}", key="embed")
x_top = f.y
x0, _ = f.tensor(DIM, f"x [T, {DIM}]  (image / video / audio tokens spliced in)", key="x")
ROW0 = f.y + 30


def row_y(i):
    return ROW0 + i * (ROW_H + ROW_GAP)


# ---------- vision branch (left column, ends level with x) ----------
VX = -560
pt = VC["temporal_patch_size"] * VC["patch_size"] ** 2 * VC["in_chans"]
m = VC["spatial_merge_size"]
v = Flow(VX, x_top - 3 * (OP_H + GAP) - 2 * (9 + GAP), KT, CLIP, labels_left=True)
v.op("**image / video frames**", ACT, key="img")
v.tensor(pt, f"patches [P, {pt}] = {VC['temporal_patch_size']}×{VC['patch_size']}×{VC['patch_size']}×3", key="patch")
v.op(f"**MiMo ViT ×{VC['depth']}** ({len(VC['fullatt_block_indexes'])} full + "
     f"{VC['depth'] - len(VC['fullatt_block_indexes'])} window attn)", VISION, P["vit"] - P["vit_merger"], key="vit")
v.tensor(VC["hidden_size"], f"[P, {VC['hidden_size']}]", key="vf")
v.op(f"**merger** — {m}×{m} patches → MLP", VISION, P["vit_merger"], key="mrg")
v.tensor(DIM, f"vision tokens [P/{m * m}, {DIM}]", key="vt")
edge(v.last, x0, ("right", "left"), "spliced into x")

# ---------- audio branch (left column, below vision) ----------
g, ch = AC["group_size"], AC["audio_channels"]
a = Flow(VX, x_top + 120, KT, CLIP, labels_left=True)
a.op(f"**audio** 24 kHz → {C['processor_config']['audio_n_mels']}-bin mel", ACT, key="wav")
a.op(f"**AudioTokenizer encoder** (separate file)\n24 layers → {ch}-codebook RVQ", AUDIO, P["tok_enc"], key="atok")
a.tensor(40 / KT, f"codes [F, {ch}] at 25 Hz", key="codes")
a.op(f"**speech embeddings** ×{ch}, summed", AUDIO, P["speech_emb"], key="semb")
a.tensor(AC["input_local_dim"], f"[F/{g}, {g}, {AC['input_local_dim']}]", lanes=g, key="ae")
a.op(f"**audio patch encoder** ×{AC['input_local_layers']}\nbidirectional, within {g} frames", AUDIO,
     P["aud_local"], key="aloc")
a.tensor(AC["input_local_dim"] * g, f"[F/{g}, {AC['input_local_dim'] * g}]  ({g} frames → 1 token, 6.25 Hz)", key="ag")
a.op(f"**projection MLP** {AC['input_local_dim'] * g} → {AC['input_local_dim'] * g * 4} → {DIM}", AUDIO,
     P["aud_proj"], key="aproj")
a.tensor(DIM, f"audio tokens [F/{g}, {DIM}]", key="at")
edge(a.last, x0, ("right", "left"), "spliced into x")

# ---------- layer rows ----------
prev = f.last
for i in range(N):
    mo = mode(i)
    tag = "  **▸ DFlash**" if i in TAPS else ""
    nid = node(f"L{i}", 0, row_y(i), ROW_W, ROW_H, f"**L{i}** · {mo}{tag}", ROW_COLOR[mo])
    edge(prev, nid)
    prev = nid

# ---------- output flow (below the rows) ----------
ybot = row_y(N - 1) + ROW_H
o = Flow(CX, ybot + 60, KT, CLIP, last=f"L{N - 1}")
hout, _ = o.tensor(DIM, f"h [T, {DIM}]  after L{N - 1}", key="ho")
o.op("**final RMSNorm**", NORM, key="fn")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm_head** {DIM} → {VOCAB}\n(not tied to embed)", HEAD, P["head"], key="head", h=76)
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op("**softmax → sample**", ACT, key="samp")
next_id = o.op("**next token id(s)**", ACT, key="next")

# ---------- MTP: built-in multi-token-prediction modules (left, below the rows) ----------
mt = Flow(VX - 100, ybot + 160, KT, CLIP, labels_left=True)
mcat, _ = mt.tensor(2 * DIM, f"concat[norm(embed(t+1)), norm(h)]  [T, {2 * DIM}]", key="mcat")
edge(hout, mcat, ("left", "top"), color=MTP_C)
mt.op(f"**eh_proj** {2 * DIM} → {DIM}", MTP_C, key="meh")
mt.op(f"**MTP block** — SWA attn ({hq_s}/{hkv_s} heads)\n+ dense FFN {C['intermediate_size']}", MTP_C,
      P["mtp"] / MTP_N, note="each", key="mblk", h=76)
mt.op("**final norm → lm_head** (shared)", MTP_C, key="mhead")
mt.tensor(40 / KT, f"draft token t+2 … ×{MTP_N} chained modules", key="mids")
node("note-mtp", mt.cx - OP_W / 2 - 160, mt.y - 10, 620, 130,
     f"**{MTP_N} MTP modules** (`model.mtp.layers.0–{MTP_N - 1}`, {fmt(P['mtp'])}) ship in the main checkpoint; "
     "the HF modeling code skips them (`_keys_to_ignore_on_load_unexpected`) — serving engines use them as "
     "the EAGLE-style draft head (the README's SGLang recipe: `--enable-multi-layer-eagle`, 3 steps).")

# ---------- DFlash: separate 5-layer block-parallel drafter fed by 5 target layers ----------
DX = 1900
nb = DC["block_size"]
dcat_y = ybot + 110
d = Flow(DX, dcat_y, KT, CLIP)
dcat, _ = d.tensor(len(TAPS) * DIM, f"concat outputs of L{', L'.join(map(str, TAPS))}  [T, {len(TAPS) * DIM}]", key="dcat")
for t in TAPS:
    tap = node(f"note-tap{t}", ROW_W + 30, row_y(t), 230, ROW_H, f"↳ output of L{t} → DFlash")
    edge(tap, dcat, ("right", "top"), color=DFL)
d.op(f"**fc** {len(TAPS) * DIM} → {DIM} + RMSNorm\n= context K/V for the drafter", DFL, key="dfc", h=76)
d.op(f"**block of {nb}**: last token + {nb - 1} [MASK]\n→ embed_tokens (shared)", DFL, key="dblk", h=76)
d.op(f"**DFlash drafter** ×{DF['num_hidden_layers']} layers\nSWA {DF['sliding_window']}, "
     f"non-causal in block", DFL, P["dflash"], note="separate file", key="ddr", h=96)
d.op("**lm_head** (shared)", DFL, key="dhead")
d.tensor(40 / KT, f"{nb - 1} draft tokens in one pass", key="dids")
ver = d.op(f"**verify**: target runs all {nb - 1} drafts\nin ONE forward pass", ACT, key="dver", h=76)
edge(ver, next_id, ("left", "right"), "keeps the drafts it agrees with", DFL)
node("note-dflash", DX - OP_W / 2, d.y, 640, 110,
     f"`dflash/` ({fmt(P['dflash'])}, Qwen3-style layers, {DF['num_attention_heads']} Q / {DF['num_key_value_heads']} KV "
     f"heads of {DF['head_dim']}). Each draft layer attends to [context K/V ; block K/V]; the README calls it a "
     f"\"5-layer SWA MTP drafter (DFlash-style)\" that predicts {nb - 1} subsequent tokens per forward pass.")

save("model.canvas")

# =====================================================================================
# block.canvas — one MoE decoder layer (SWA and GA variants side by side)
# =====================================================================================
KT, CLIP = 0.04, 1200
COL = 520  # column offset of the two attention variants / the experts vs router


def qkv(swa):
    hq, hkv, hd, vd = attn_shape(swa)
    return hq * hd + hkv * hd + hkv * vd


node("cap-btitle", -1300, -420, 2600, 150,
     "# MiMo-V2.6-Flash — one decoder layer\n"
     f"Pre-norm residual block: RMSNorm → attention → add → RMSNorm → MoE → add. Left column = **SWA** layer "
     f"({N_SWA} of {N}), right column = **GA** layer ({N_GA} of {N}); the MoE half is identical in both. "
     f"Grey bars = tensors ({KT} px/channel). Weights: attention qkv FP8, o_proj BF16, routed experts MXFP4.")

c = Flow(0, -220, KT, CLIP)
c.tensor(DIM, f"h [T, {DIM}]  from the previous layer", key="bh")
c.op("**input RMSNorm**", NORM, key="bn1")
xa, _ = c.tensor(DIM, f"x [T, {DIM}]", key="bx")
ysplit = c.y + 20
ends = []
for swa, sx in ((True, -COL), (False, COL)):
    hq, hkv, hd, vd = attn_shape(swa)
    col = Flow(sx, ysplit, KT, CLIP, last=xa, labels_left=swa)
    name = "SWA" if swa else "GA"
    col.op(f"**{name}: fused qkv_proj**\n{DIM} → {qkv(swa)}", SWA_C if swa else GA_C,
           params_of(rf"model\.layers\.{1 if swa else 5}\.self_attn\.qkv_proj\.weight"), key="qkv", h=76)
    col.tensor(qkv(swa), f"q | k | v  [T, {hq * hd} | {hkv * hd} | {hkv * vd}]", key="qkvt")
    col.op(f"**split heads**\nq {hq}×{hd} · k {hkv}×{hd} · v {hkv}×{vd}", SWA_C if swa else GA_C, key="split", h=76)
    theta = C["swa_rope_theta"] if swa else C["rope_theta"]
    col.op(f"**partial RoPE** on {ROPE_DIM} of {hd} q/k dims\nθ = {theta:.0e} · v × {C['attention_value_scale']}",
           SWA_C if swa else GA_C, key="rope", h=76)
    if swa:
        col.op(f"**sliding-window attention**\nwindow {C['sliding_window']} · GQA {hq // hkv}:1\n"
               f"+ learned sink logit per head ({hq})", SWA_C, key="att", h=96)
    else:
        col.op(f"**global causal attention**\nfull context (1M) · GQA {hq // hkv}:1\nno sink logit",
               GA_C, key="att", h=96)
    _, end = col.tensor(hq * vd, f"o [T, {hq}×{vd} = {hq * vd}]", key="ot")
    ends.append((end, col.y))

c.y = max(y for _, y in ends) + 20
c.last = None
oproj = c.op(f"**o_proj** {hq_s * vd_s} → {DIM} (BF16)", ATTN, params_of(r"model\.layers\.1\.self_attn\.o_proj\.weight"),
             key="oproj")
for end, _ in ends:
    edge(end, oproj)
c.op("**add residual** h + attn(x)", ACT, key="add1")
c.tensor(DIM, f"h' [T, {DIM}]", key="bh2")
c.op("**post-attention RMSNorm**", NORM, key="bn2")
xm, _ = c.tensor(DIM, f"x [T, {DIM}]", key="bxm")
ymoe = c.y + 20

# router (right column)
rt = Flow(COL, ymoe, KT, CLIP, last=xm)
rt.op(f"**router** gate {DIM} → {R} (BF16)", FFN, P["router"], key="gate")
rt.tensor(R, f"scores [T, {R}] = sigmoid(logits)", key="sc")
topk = rt.op(f"**top-{A}** on scores + e_score_correction_bias\n(bias only picks; weights use raw scores)",
             FFN, key="topk", h=76)
rt.op(f"**renormalise** the {A} weights to sum 1", FFN, key="norm")
_, wts = rt.tensor(A * 20, f"weights [T, {A}]", key="wt")

# experts (left column), aligned with the top-k box
ex = Flow(-COL, ymoe + (OP_H + GAP) + (9 + GAP), KT, CLIP, last=xm, labels_left=True)
eop = ex.op(f"**{A} of {R} routed experts** (SwiGLU)\ngate/up {DIM} → {C['moe_intermediate_size']}, "
            f"down → {DIM}", FFN, P["expert"], note="each, MXFP4", key="exp", h=76)
edge(topk, eop, ("left", "right"), f"top-{A} ids", FFN)
_, eend = ex.tensor(DIM, f"expert outputs [T, {A}, {DIM}]", lanes=A, key="eo")

c.y = max(ex.y, rt.y) + 20
c.last = None
comb = c.op(f"**weighted sum** Σ wᵢ · expertᵢ(x)", FFN, key="comb")
edge(eend, comb)
edge(wts, comb)
c.op("**add residual** h' + moe(x)", ACT, key="add2")
c.tensor(DIM, f"h [T, {DIM}]  → next layer", key="bout")

node("note-layerp", -1300, c.y + 10, 1250, 110,
     f"Per layer: SWA attention **{fmt(P['attn_swa'])}**, GA attention **{fmt(P['attn_ga'])}** (fewer KV heads), "
     f"router {fmt(P['router'])}, {R} experts × {fmt(P['expert'])} = **{fmt(R * P['expert'])}** stored, "
     f"{A} × {fmt(P['expert'])} = {fmt(A * P['expert'])} used per token. No shared expert (`n_shared_experts: null`).")
node("note-l0", 50, c.y + 10, 1250, 110,
     f"**L0** differs: global attention and a dense SwiGLU FFN {DIM} → {C['intermediate_size']} → {DIM} "
     f"({fmt(P['dense_ffn'])}) instead of the MoE. The {MTP_N} MTP blocks use SWA attention with the same dense FFN shape.")

save("block.canvas")
print("ok", [(r[0], fmt(r[2]), fmt(r[3])) for r in budget()], "total", fmt(P["total"]))
