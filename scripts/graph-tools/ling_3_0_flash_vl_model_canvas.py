"""Ling-3.0-flash-VL (inclusionAI, BailingMoeV3VL): model + layer canvases from the checkpoint.

Read top → bottom:
  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale everywhere)
  - boxes       = operations, all one size; parameter / compute shares are the two bars at the top
  - layer rows  = one per layer, coloured by attention type (KDA linear attention vs Gated MLA)
  - note-* ids  = annotations (styled dim, borderless); cap-* ids = borderless captions
Numbers come from shapes.json (safetensors headers of inclusionAI/Ling-3.0-flash-VL) + config.json;
the dataflow follows inclusionAI/vllm-ling-v3 (bailing_moe_v3.py, bailing_moe_v3_vl.py).

usage: python3 ling_3_0_flash_vl_model_canvas.py <shapes.json> <config.json> <models/ling-3-0-flash-vl>
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

N, DIM, VOCAB = C["num_hidden_layers"], C["hidden_size"], C["vocab_size"]
G = C["layer_group_size"]
H, HD = C["num_attention_heads"], C["head_dim"]
NOPE, ROPE, VHD, KVR = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"], C["kv_lora_rank"]
E, K, MI = C["num_experts"], C["num_experts_per_tok"], C["moe_intermediate_size"]
DENSE_K = C["first_k_dense_replace"]
VD, PS, TPS, MS = V["hidden_size"], V["patch_size"], V["temporal_patch_size"], V["spatial_merge_size"]


def is_kda(i):  # vllm-ling-v3 _is_kda_layer: every G-th layer (and any tail past the last full group) is MLA
    return not ((i + 1) % G == 0 or i >= N // G * G)


KDA_L = [i for i in range(N) if is_kda(i)]
MLA_L = [i for i in range(N) if not is_kda(i)]
# the checkpoint must agree with the rule: MLA layers carry kv_b_proj, KDA layers carry f_proj
assert all(f"model.layers.{i}.attention.kv_b_proj.weight" in S for i in MLA_L)
assert all(f"model.layers.{i}.attention.f_proj.weight" in S for i in KDA_L)

# colours (JSON Canvas presets "1"–"6" or hex)
KDA_C, MLA_C, EMB, NORM, FFN, HEAD, ACT, VIS = "5", "6", "#f472b6", "3", "4", "2", "#64748b", "#f59e0b"

# geometry
KT, CLIP_W = 0.08, 1800  # tensor bars: px per channel, longest drawn bar
ROW_W, ROW_H, ROW_GAP = 300, 34, 6
OP_W, OP_H, GAP = 300, 56, 40
BAR_W, BAR_H = 2600, 44
LBL_W = 440
MIN_SEG = 16  # the viewer draws no box narrower than ~14px


def params_of(pattern):
    rx = re.compile(pattern)
    return sum(math.prod(shp) for k, (dt, shp) in S.items() if rx.fullmatch(k))


def layers_re(idx):
    return "(" + "|".join(str(i) for i in idx) + ")"


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M" if n >= 1e5 else f"{n / 1e3:.1f}K"


P = {
    "embed": params_of(r"model\.word_embeddings\.weight"),
    "head": params_of(r"lm_head\.weight"),
    "kda": params_of(rf"model\.layers\.{layers_re(KDA_L)}\.attention\..*"),
    "mla": params_of(rf"model\.layers\.{layers_re(MLA_L)}\.attention\..*"),
    "dense": params_of(r"model\.layers\.\d+\.mlp\.(gate|up|down)_proj\.weight"),
    "routed": params_of(r"model\.layers\.\d+\.mlp\.experts\..*"),
    "shared": params_of(r"model\.layers\.\d+\.mlp\.shared_experts\..*"),
    "router": params_of(r"model\.layers\.\d+\.mlp\.gate\..*"),
    "patch": params_of(r"model\.visual\.(patch_embed|pos_embed)\..*"),
    "vit": params_of(r"model\.visual\.blocks\..*"),
    "merger": params_of(r"model\.visual\.merger\..*"),
    "proj": params_of(r"linear_proj\..*"),
    "total": params_of(r".*"),
}
EXPERT = params_of(r"model\.layers\.2\.mlp\.experts\.0\..*")
N_MOE = N - DENSE_K


def budget():
    """[(name, colour, stored params, params one text token is multiplied by[, why zero])]."""
    vision = P["patch"] + P["vit"] + P["merger"] + P["proj"]
    rows = [
        (f"{len(KDA_L)} KDA attn", KDA_C, P["kda"], P["kda"]),
        (f"{len(MLA_L)} MLA attn", MLA_C, P["mla"], P["mla"]),
        (f"dense FFN ×{DENSE_K}", FFN, P["dense"], P["dense"]),
        (f"routed experts {N_MOE}×{E}", "#86efac", P["routed"], N_MOE * K * EXPERT),
        ("shared experts + routers", FFN, P["shared"] + P["router"], P["shared"] + P["router"]),
        ("lm head", HEAD, P["head"], P["head"]),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("ViT + projector", VIS, vision, 0, "runs per image"),
    ]
    other = P["total"] - sum(r[2] for r in rows)
    rows.append(("norms", ACT, other, other))  # layer norms, o_norm, final norm
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

    def _link(self, top, bottom):
        if self.last:
            edge(self.last, top)
        self.last = bottom

    def op(self, text, color, params=0, note="", key="op"):
        if params:
            text += f"\n{fmt(params)}" + (f" · {note}" if note else "")
        nid = node(self._id(key), self.cx - OP_W / 2, self.y, OP_W, OP_H, text, color)
        self._link(nid, nid)
        self.y += OP_H + GAP
        return nid

    def tensor(self, dim, label, lanes=1, key="t"):
        real = dim * KT
        w = min(real, CLIP_W)
        ids = [node(self._id(key), self.cx - w / 2, self.y + j * 14, w, 9, "", ACT) for j in range(lanes)]
        clip = f" — bar clipped (real {real:.0f}px)" if real > w else ""
        half = max(w / 2, OP_W / 2) + 20
        lx = self.cx - half - LBL_W if self.labels_left else self.cx + half
        node(self._id("note-lbl"), lx, self.y - 10, LBL_W, 34, f"`{label}`{clip}")
        self._link(ids[0], ids[-1])
        self.y += lanes * 14 + GAP
        return ids[0], ids[-1]


def bar(key, x0, y, title, rows, idx):
    total = sum(r[idx] for r in rows)
    node(f"cap-{key}t", x0, y - 46, BAR_W, 40, f"**{title}** — total {fmt(total)}")
    x = x0
    legend = []
    # segments too thin to draw go last, so any blank stretch is at the bar's end, not mid-bar
    for k in sorted(range(len(rows)), key=lambda k: BAR_W * rows[k][idx] / total < MIN_SEG):
        r = rows[k]
        share = r[idx] / total
        w = BAR_W * share
        if w >= MIN_SEG:  # integer edges, and a 2px gap so neighbouring segments never touch
            node(f"{key}{k}", round(x), y, round(x + w) - round(x) - 2, BAR_H, f"**{r[0]}**" if w > 150 else "", r[1])
        x += w
        why = r[4] if len(r) > 4 and not r[idx] else ""
        legend.append(f"{r[0]} **{fmt(r[idx]) if r[idx] else '0'}**" + (f" ({why})" if why else f" ({share:.1%})"))
    node(f"cap-{key}l", x0, y + BAR_H + 6, BAR_W, 64, " · ".join(legend))


def save(name):
    json.dump(dict(nodes=nodes, edges=edges), open(os.path.join(OUT, name), "w"), indent=1, ensure_ascii=False)
    nodes.clear()
    edges.clear()


SRC = ("Sources: tensor shapes from the safetensors headers of inclusionAI/Ling-3.0-flash-VL, hyper-parameters "
       "from its config.json, dataflow from inclusionAI/vllm-ling-v3 (bailing_moe_v3.py, bailing_moe_v3_vl.py); "
       "total / active counts from the model card (124B / 5.5B).")

# =====================================================================================
# model.canvas
# =====================================================================================
B = budget()
bar("bp", -900, -1700, "Where the parameters are stored", B, 2)
bar("bc", -900, -1500, "What one text token is multiplied by (active weights; adding the "
    f"{fmt(P['embed'])} embedding row gives {fmt(sum(r[3] for r in B) + P['embed'])}, the card's 5.5B)", B, 3)

node("cap-title", -900, -1300, 1250, 200,
     "# Ling-3.0-flash-VL — model\n"
     f"Read top → bottom. **Grey bars = tensors**, length ∝ channels per token ({KT} px/channel). "
     "**Boxes = operations**, all the same size; where parameters and compute live is the pair of bars above. "
     "T = sequence length, P = image patches. *Layer* tab = one decoder layer in detail.")
node("cap-legend", 450, -1300, 1250, 200,
     f"Layer rows — **KDA** (Kimi Delta Attention, linear): fixed-size state [{H}, {HD}, {HD}] per sequence, no KV cache, "
     f"no position encoding · **Gated MLA**: softmax attention over a compressed KV cache ({KVR} + {ROPE} values per token), "
     f"M-RoPE, per-head sigmoid output gate. Pattern repeats every {G} layers: {G - 1} KDA → 1 MLA "
     f"({len(KDA_L)} : {len(MLA_L)}).  \n"
     f"FFN — L0–{DENSE_K - 1} dense SwiGLU {C['intermediate_size']}; L{DENSE_K}–{N - 1} MoE: {E} experts × {MI}, "
     f"top-{K} (sigmoid scores, {C['n_group']} groups → best {C['topk_group']}) + 1 shared expert. No MTP head in this checkpoint.")
node("note-src", -900, -1080, 2600, 40, SRC)

# ---------- vision branch (left), its last tensor feeds the splice op ----------
v = Flow(-480, -960, labels_left=True)
v.op("**image / video** frames", ACT, key="img")
v.tensor(3 * TPS * PS * PS, f"patches [P, {3 * TPS * PS * PS}]  (3 ch × {TPS} frames × {PS}×{PS})", key="patch")
v.op(f"**patch embed** Conv3d + learned pos ({V['num_position_embeddings']})", VIS, P["patch"], key="pemb")
v.tensor(VD, f"[P, {VD}]", key="vf0")
v.op(f"**ViT ×{V['depth']}** · {V['num_heads']} heads · MLP {V['intermediate_size']}", VIS, P["vit"], key="vit")
v.tensor(VD, f"[P, {VD}]", key="vf1")
v.op(f"**merger**: LayerNorm, concat {MS}×{MS} patches", VIS, P["merger"], key="merge")
v.tensor(VD * MS * MS, f"[P/{MS * MS}, {VD * MS * MS}]", key="vf2")
v.op(f"**projector** MLP {VD * MS * MS} → {DIM} → {DIM}, GELU", VIS, P["proj"], key="proj")
vt, _ = v.tensor(DIM, f"image tokens [P/{MS * MS}, {DIM}]", key="vt")
vt_y = next(n["y"] for n in nodes if n["id"] == vt)

# ---------- text input flow (centre column), placed so the splice op lines up with the image tokens ----------
cx = ROW_W / 2
pre = 3 * (OP_H + GAP) + 2 * (14 + GAP)  # raw, tok, embed ops + ids, x tensors
f = Flow(cx, vt_y + 4 - OP_H / 2 - pre)
f.op("**raw text** + image placeholders", ACT, key="raw")
f.op(f"**tokenizer** (BPE, vocab {VOCAB})", ACT, key="tok")
f.tensor(40 / KT, "token ids [T] — one integer per token", key="ids")
f.op("**embed** — look up row *id*", EMB, P["embed"], note=f"table {VOCAB} × {DIM}", key="embed")
f.tensor(DIM, f"x [T, {DIM}]", key="xt")
splice = f.op("**splice**: image tokens replace\n<|image_pad|> positions", ACT, key="splice")
edge(v.last, splice, ("right", "left"), color=VIS)
f.tensor(DIM, f"x [T, {DIM}]  + positions [3, T] (t, h, w)", key="x")
ROW0 = f.y + 10

node("note-vision", -900, vt_y + 60, 700, 150,
     f"Vision tower = Qwen3-VL-style ViT ({V['depth']} pre-LN blocks, width {VD}). The merger has no MLP of its own "
     f"(config `disable_merger_proj`; its `out_hidden_size` {V['out_hidden_size']} is unused) — the "
     f"{sum(1 for k_ in S if re.fullmatch(r'linear_proj\.\d+\.weight', k_))}-layer projector does the alignment. No DeepStack taps: the checkpoint has no "
     "deepstack_merger weights and vLLM disables them.")
node("note-mrope", -900, vt_y + 240, 700, 120,
     f"M-RoPE: text tokens get (t, h, w) = (n, n, n); image tokens get their grid cell. The {ROPE // 2} rotary "
     f"frequencies are split {'/'.join(map(str, CFG['mrope_section']))} over t/h/w (θ = {C['rope_theta']:.0e}). "
     f"Only the {len(MLA_L)} MLA layers use positions; KDA layers are position-free.")


# ---------- layer rows ----------
def row_y(i):
    return ROW0 + i * (ROW_H + ROW_GAP)


prev = f.last
for i in range(N):
    kind = "KDA" if is_kda(i) else "Gated MLA"
    ffn = "dense FFN" if i < DENSE_K else "MoE"
    nid = node(f"L{i}", 0, row_y(i), ROW_W, ROW_H, f"**L{i}** · {kind} · {ffn}", KDA_C if is_kda(i) else MLA_C)
    edge(prev, nid)
    prev = nid

for g in range((N + G - 1) // G):
    lo, hi = g * G, min(N, g * G + G) - 1
    node(f"note-grp{g}", ROW_W + 30, row_y(lo), 250, row_y(hi) - row_y(lo) + ROW_H,
         f"group {g + 1} · L{lo}–{hi}\n{sum(is_kda(i) for i in range(lo, hi + 1))} KDA → 1 MLA")

clamp_e = [i for i, v_ in enumerate(C["expert_swiglu_limit_list"]) if v_]
clamp_s = [i for i, v_ in enumerate(C["share_expert_swiglu_limit_list"]) if v_]
node("note-layer", ROW_W + 320, row_y(0), 620, 330,
     f"Every layer: RMSNorm → attention → add → RMSNorm → FFN → add (pre-norm, fused residual).  \n"
     f"KDA layer ≈ **{fmt(P['kda'] / len(KDA_L))}** attention; MLA layer ≈ **{fmt(P['mla'] / len(MLA_L))}**. "
     f"MoE layer ≈ **{fmt((P['routed'] + P['shared'] + P['router']) / N_MOE)}**, of which one token multiplies "
     f"≈ {fmt((K + 1) * EXPERT + P['router'] / N_MOE)} ({K} routed + 1 shared expert + router).  \n"
     f"SwiGLU clamp (config): routed experts limit {C['expert_swiglu_limit_list'][clamp_e[0]]} in L{clamp_e[0]}–{clamp_e[-1]}; "
     f"shared expert limit {C['share_expert_swiglu_limit_list'][clamp_s[0]]}/{C['share_expert_swiglu_limit_list'][-1]} "
     f"in L{clamp_s[0]}–{clamp_s[-1]}.  \n"
     f"Context: {C['max_position_embeddings'] // 1024}K native, 256K with YaRN ×2 (model card serving recipe).")

# ---------- output flow ----------
o = Flow(cx, row_y(N - 1) + ROW_H + 60, last=f"L{N - 1}")
o.tensor(DIM, f"h [T, {DIM}]  after L{N - 1}", key="ho")
o.op("**final RMSNorm**", NORM, params_of(r"model\.norm\.weight"), key="fnorm")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm head** — {DIM} → {VOCAB} (untied)", HEAD, P["head"], key="head")
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op("**softmax → sample**", ACT, key="samp")
o.op("**next token id**", ACT, key="next")

save("model.canvas")
print("model ok", [(r[0], fmt(r[2]), fmt(r[3])) for r in B], "total", fmt(P["total"]),
      "active", fmt(sum(r[3] for r in B)), "+embed", fmt(sum(r[3] for r in B) + P["embed"]))

# =====================================================================================
# block.canvas — one decoder layer, both attention variants side by side
# =====================================================================================
_seq[0] = 0
OP_H = 76  # some layer ops carry two lines of text plus a parameter line


def p1(name, layer):
    return params_of(rf"model\.layers\.{layer}\.{name}")


kl, ml, moe_l = KDA_L[-1], MLA_L[-1], N - 1  # representative layers (shapes are identical within a kind)

node("cap-title", -1100, -420, 1100, 150,
     "# Ling-3.0-flash-VL — one decoder layer\n"
     f"Left: **KDA** ({len(KDA_L)} layers). Right: **Gated MLA** ({len(MLA_L)} layers: "
     f"L{', L'.join(map(str, MLA_L))}). Both feed the same MoE FFN (L{DENSE_K}–{N - 1}). Shapes per token; "
     "parameter counts per layer from the checkpoint.")
node("note-bsrc", 100, -420, 1000, 150, SRC)

top = Flow(0, -220)
top.tensor(DIM, f"h [T, {DIM}]  residual stream in", key="hin")
nrm = top.op("**input RMSNorm**", NORM, p1(r"input_layernorm\.weight", kl), key="n1")
y0 = top.y + 30

node("cap-kda", -1040, y0 + 20, 420, 40, f"**KDA** — linear attention ({len(KDA_L)} layers)")
node("cap-mla", 620, y0 + 20, 420, 40, f"**Gated MLA** — softmax attention ({len(MLA_L)} layers)")

# ----- KDA column -----
k = Flow(-450, y0 + 80, last=nrm, labels_left=True)
a = r"attention\."
k.op(f"**q / k / v proj** {DIM} → 3 × {H * HD}", KDA_C, p1(a + r"[qkv]_proj\.weight", kl), key="kqkv")
k.tensor(H * HD, f"q, k, v [T, {H * HD}]", lanes=3, key="kqkvt")
k.op(f"**short conv** (causal, k={C['short_conv_kernel_size']}) + SiLU", KDA_C, p1(a + r"[qkv]_conv1d\.weight", kl), key="kconv")
k.tensor(H * HD, f"q, k, v [T, {H}, {HD}]  (q, k L2-normed)", lanes=3, key="kheads")
k.op(f"**gates** β = σ(b_proj) [T, {H}] ·\ndecay α = f_proj → A_log, dt_bias, ≥ {C['kda_lower_bound']}", KDA_C,
     p1(a + r"(b_proj\.weight|f_proj\.weight|A_log|dt_bias)", kl), key="kgate")
k.op(f"**delta rule** S ← (I − βkkᵀ)·diag(α)·S + βkvᵀ\nstate S [{H}, {HD}, {HD}], out o = Sᵀq", KDA_C, key="kdelta")
k.tensor(H * HD, f"o [T, {H}, {HD}]", key="ko")
k.op("**gated RMSNorm** norm(o) · σ(g_proj h)", KDA_C, p1(a + r"(g_proj\.weight|o_norm\.weight)", kl), key="kon")
k.op(f"**o_proj** {H * HD} → {DIM}", KDA_C, p1(a + r"o_proj\.weight", kl), key="kop")
kout, _ = k.tensor(DIM, f"attn out [T, {DIM}]", key="kout")

# ----- MLA column -----
m = Flow(450, y0 + 80, last=nrm)
m.op(f"**q_proj** {DIM} → {H} × {NOPE + ROPE}", MLA_C, p1(a + r"q_proj\.weight", ml), key="mq")
m.tensor(H * (NOPE + ROPE), f"q [T, {H}, {NOPE}+{ROPE}]  (nope + rope)", key="mqt")
m.op(f"**kv_a** {DIM} → {KVR} + {ROPE}, RMSNorm on {KVR}", MLA_C, p1(a + r"kv_a_(proj_with_mqa|layernorm)\.weight", ml), key="mkva")
m.tensor(KVR + ROPE, f"c_kv [T, {KVR}] + k_rope [T, {ROPE}]  — the KV cache", key="mc")
m.op(f"**M-RoPE** on the {ROPE} rope dims\n(t/h/w {'/'.join(map(str, CFG['mrope_section']))})", MLA_C, key="mrope")
m.op(f"**kv_b** {KVR} → {H} × ({NOPE} k + {VHD} v)", MLA_C, p1(a + r"kv_b_proj\.weight", ml), key="mkvb")
m.op(f"**softmax attention**, {H} heads, causal", MLA_C, key="mattn")
m.tensor(H * VHD, f"o [T, {H}, {VHD}]", key="mo")
m.op(f"**head gate** o · σ(g_proj h) [T, {H}]", MLA_C, p1(a + r"g_proj\.weight", ml), key="mg")
m.op(f"**dense** {H * VHD} → {DIM}", MLA_C, p1(a + r"dense\.weight", ml), key="md")
mout, _ = m.tensor(DIM, f"attn out [T, {DIM}]", key="mout")

# ----- residual + MoE -----
r = Flow(0, max(k.y, m.y) + 20)
add1 = r.op("**add residual**", ACT, key="add1")
edge(kout, add1, ("bottom", "left"))
edge(mout, add1, ("bottom", "right"))
r.op("**post-attention RMSNorm**", NORM, p1(r"post_attention_layernorm\.weight", kl), key="n2")
_, hm = r.tensor(DIM, f"h [T, {DIM}]", key="hm")
router = r.op(f"**router** {DIM} → {E}, sigmoid + bias\n{C['n_group']} groups → best {C['topk_group']} → top {K}",
              FFN, p1(r"mlp\.gate\..*", moe_l), key="router")
ys = r.y + 20
rout = node("routed", -450 - OP_W / 2, ys, OP_W, OP_H,
            f"**{K} routed experts** of {E}\nSwiGLU {DIM}→{MI}→{DIM} · {fmt(EXPERT)} each", "#86efac")
shr = node("shared", 450 - OP_W / 2, ys, OP_W, OP_H,
           f"**1 shared expert** (always on)\nSwiGLU {DIM}→{C['moe_shared_expert_intermediate_size']}→{DIM} · "
           f"{fmt(p1(r'mlp\.shared_experts\..*', moe_l))}", FFN)
edge(router, rout, ("bottom", "top"), "top-8 ids + weights")
edge(router, shr, ("bottom", "top"))
r.y = ys + OP_H + 80
mix = r.op(f"**Σ weights·experts × {C['routed_scaling_factor']}**\n+ shared expert", FFN, key="mix")
edge(rout, mix, ("bottom", "left"))
edge(shr, mix, ("bottom", "right"))
r.op("**add residual**", ACT, key="add2")
r.tensor(DIM, f"h [T, {DIM}]  → next layer", key="hout")

node("note-dense", 450 + OP_W / 2 + 40, ys - 150, 420, 110,
     f"L0–{DENSE_K - 1} replace the whole MoE block by one dense SwiGLU {DIM}→{C['intermediate_size']}→{DIM} "
     f"({fmt(params_of(r'model\.layers\.0\.mlp\..*'))} per layer).")
node("note-route", -450 - OP_W / 2 - 480, ys - 150, 440, 150,
     f"Expert choice uses sigmoid scores + a learned per-expert bias (load balancing, aux-loss-free); the "
     f"{K} chosen weights are renormalised to sum 1 and scaled by {C['routed_scaling_factor']}. One expert is "
     f"{fmt(EXPERT)}; all {E} are {fmt(E * EXPERT)} per layer.")
node("note-cache", 1200, y0 + 80, 420, 150,
     f"Decode memory per sequence: KDA keeps a {H}×{HD}×{HD} state + conv tails, constant in T. MLA caches "
     f"{KVR + ROPE} values per token per layer — only {len(MLA_L)} of {N} layers grow with context.")

save("block.canvas")
print("block ok")
