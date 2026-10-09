"""DiffusionGemma 26B-A4B-it (google, block-diffusion MoE LM): model.canvas + block.canvas.

Read top → bottom:
  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale per canvas)
  - boxes       = operations; where parameters / per-token compute live is the pair of bars at the top
  - note-* ids  = annotations (dim, borderless); cap-* ids = borderless titles / legends
Every number comes from the checkpoint: shapes.json (safetensors headers, fetch_safetensors_shapes.py),
config.json, generation_config.json, and transformers' diffusion_gemma code (modular_diffusion_gemma.py for
the encoder / decoder / self-conditioning, generation_diffusion_gemma.py for the denoising loop,
EntropyBoundSampler and StableAndConfidentStoppingCriteria) plus gemma4 for the router, experts and vision tower.

usage: python3 diffusiongemma_26b_a4b_it_model_canvas.py <shapes.json> <config.json> <generation_config.json>
       <models/diffusiongemma-26b-a4b-it>
"""
import json
import math
import os
import re
import sys

S = json.load(open(sys.argv[1]))
CFG = json.load(open(sys.argv[2]))
G = json.load(open(sys.argv[3]))
OUT = sys.argv[4]
C, V = CFG["text_config"], CFG["vision_config"]

N, DIM, VOCAB = C["num_hidden_layers"], C["hidden_size"], C["vocab_size"]
NH, NKV, HD = C["num_attention_heads"], C["num_key_value_heads"], C["head_dim"]
GKV, GHD = C["num_global_key_value_heads"], C["global_head_dim"]
E, K, MI, FI = C["num_experts"], C["top_k_experts"], C["moe_intermediate_size"], C["intermediate_size"]
WIN, TYPES, CAP_SOFT = C["sliding_window"], C["layer_types"], C["final_logit_softcapping"]
RP = C["rope_parameters"]
GROT = int(GHD * RP["full_attention"]["partial_rotary_factor"])
CANVAS = CFG["canvas_length"]
STEPS, TMAX, TMIN = G["max_denoising_steps"], G["t_max"], G["t_min"]
EB, STAB, CONF = G["sampler_config"]["entropy_bound"], G["stability_threshold"], G["confidence_threshold"]
N_SL, N_GL = TYPES.count("sliding_attention"), TYPES.count("full_attention")
SOFT = CFG["vision_soft_tokens_per_image"]

# colours (JSON Canvas presets "1"–"6" or hex)
EMB, ATT, NORM, FFN, HEAD, ACT = "6", "5", "3", "4", "2", "#64748b"
NOISE, DIFF, SC, MLPC, VIS = "#f59e0b", "#e879f9", "#22c55e", "#38bdf8", "#a78bfa"
ROW_COLOR = {"sliding_attention": "#94a3b8", "full_attention": "5"}

OP_W, OP_H, GAP = 300, 56, 40


def params_of(pattern):
    rx = re.compile(pattern)
    return sum(math.prod(shp) for k, (dt, shp) in S.items() if rx.fullmatch(k))


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M" if n >= 1e5 else f"{n}"


L = r"model\.decoder\.layers\.\d+"
SL0, GL0 = TYPES.index("sliding_attention"), TYPES.index("full_attention")
EXPERT = params_of(r"model\.decoder\.layers\.0\.experts\..*") // E
P = {
    "embed": params_of(r"model\.decoder\.embed_tokens\.weight"),
    "attn_s": sum(params_of(rf"model\.decoder\.layers\.{i}\.self_attn\..*") for i in range(N) if TYPES[i] == "sliding_attention"),
    "attn_g": sum(params_of(rf"model\.decoder\.layers\.{i}\.self_attn\..*") for i in range(N) if TYPES[i] == "full_attention"),
    "mlp": params_of(L + r"\.mlp\..*"),
    "routed": params_of(L + r"\.experts\..*"),
    "router": params_of(L + r"\.router\..*"),
    "norms": params_of(L + r"\.(?:\w*layernorm\w*\.weight|layer_scalar)") + params_of(r"model\.decoder\.norm\.weight"),
    "sc": params_of(r"model\.decoder\.self_conditioning\..*"),
    "enc": params_of(r"model\.encoder\.language_model\..*"),
    "vis": params_of(r"model\.encoder\.(?:vision_tower|embed_vision)\..*"),
    "all": params_of(r".*"),
}
assert P["all"] == sum(v for k, v in P.items() if k != "all"), "budget rows must cover every tensor"
assert P["enc"] == N, "the text encoder stores only its per-layer scalars; everything else is tied to the decoder"
assert "lm_head.weight" not in S and CFG["tie_word_embeddings"], "lm head is the embedding table"
A1 = {t: params_of(rf"model\.decoder\.layers\.{i}\.self_attn\..*") for t, i in (("s", SL0), ("g", GL0))}


def budget():
    """[(name, colour, stored, multiplied per canvas token, why-zero)]; K of E routed experts run per token."""
    return [
        (f"routed experts {E}×{N}", FFN, P["routed"], N * K * EXPERT),
        (f"sliding attention ×{N_SL}", ATT, P["attn_s"], P["attn_s"]),
        (f"global attention ×{N_GL}", "#0e7490", P["attn_g"], P["attn_g"]),
        (f"dense MLP ×{N} (always on)", MLPC, P["mlp"], P["mlp"]),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, 0, P["embed"], "tied: same table as embed"),
        ("vision tower + projector", VIS, P["vis"], 0, "runs per image"),
        ("self-conditioning MLP", SC, P["sc"], P["sc"]),
        (f"router ×{N}", "#fbbf24", P["router"], P["router"]),
        ("norms + layer scalars", NORM, P["norms"], P["norms"]),
        ("encoder layer scalars", ACT, P["enc"], 0, "prompt pass only"),
    ]


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


def ny(nid):
    return next(n for n in nodes if n["id"] == nid)["y"]


_seq = [0]


class Flow:
    """A vertical chain of ops and tensors centred on cx; each item is linked to the previous one."""

    def __init__(self, cx, y, last=None, labels_left=False, kt=0.08, clip=1100, lbl=440):
        self.cx, self.y, self.last, self.labels_left = cx, y, last, labels_left
        self.kt, self.clip, self.lbl = kt, clip, lbl

    def _id(self, key):
        _seq[0] += 1
        return f"{key}{_seq[0]}"

    def _link(self, top, bottom):
        if self.last:
            edge(self.last, top)
        self.last = bottom

    def op(self, text, color, params=0, note="", key="op", h=OP_H, w=OP_W):
        if params:
            text += f"\n{fmt(params)}" + (f" · {note}" if note else "")
        nid = node(self._id(key), self.cx - w / 2, self.y, w, h, text, color)
        self._link(nid, nid)
        self.y += h + GAP
        return nid

    def tensor(self, dim, label, key="t", lbl_dy=-12, left=None):
        real = dim * self.kt
        w = max(16, min(real, self.clip))
        nid = node(self._id(key), self.cx - w / 2, self.y, w, 9, "", ACT)
        clip = f" — bar clipped ({real:.0f}px)" if real > w else ""
        half = max(w / 2, OP_W / 2) + 20
        left = self.labels_left if left is None else left
        lx = self.cx - half - self.lbl if left else self.cx + half
        node(self._id("note-lbl"), lx, self.y + lbl_dy, self.lbl, 34, f"`{label}`{clip}")
        self._link(nid, nid)
        self.y += 9 + GAP
        return nid


MIN_SEG = 16  # the viewer draws no box narrower than ~14px


def bar(key, x0, y, width, title, rows, idx):
    total = sum(r[idx] for r in rows)
    node(f"cap-{key}t", x0, y - 46, width, 40, f"**{title}** — total {fmt(total)}")
    x, legend = x0, []
    # segments too thin to draw go last, so any blank stretch is at the bar's end, not mid-bar
    for k in sorted(range(len(rows)), key=lambda k: width * rows[k][idx] / total < MIN_SEG):
        r = rows[k]
        share = r[idx] / total
        w = width * share
        if w >= MIN_SEG:  # integer edges so neighbouring segments meet exactly
            node(f"{key}{k}", round(x), y, round(x + w) - round(x), 44, f"**{r[0]}**" if w > 220 else "", r[1])
        x += w
        why = r[4] if len(r) > 4 and not r[idx] else ""
        legend.append(f"{r[0]} **{fmt(r[idx]) if r[idx] else '0'}**"
                      + (f" ({why})" if why else f" ({share:.1%})" if share >= 0.001 else " (<0.1%)"))
    node(f"cap-{key}l", x0, y + 50, width, 64, " · ".join(legend))


def save(name):
    json.dump(dict(nodes=nodes, edges=edges), open(os.path.join(OUT, name), "w"), indent=1, ensure_ascii=False)
    nodes.clear()
    edges.clear()


# ---- layer stack: one column, repeated layers drawn once with ×N, plus a barcode of every layer ----
SEG_W, SEG_H, SEG_GAP = 460, 56, 26
CELL, CELL_GAP = 16, 2


def stack_segments(kind, n):
    """Longest periodic tail (period 2–8, repeated ≥ 2×, not constant), then the same for what is above it;
    whatever is left becomes runs of identical layers. → [("run", first, count) | ("period", first, period, reps)]."""
    def runs(a, b):
        out = []
        for i in range(a, b):
            if out and kind(i) == kind(out[-1][1]):
                out[-1] = ("run", out[-1][1], out[-1][2] + 1)
            else:
                out.append(("run", i, 1))
        return out

    def seg(stop):
        for s0 in range(stop):
            if all(kind(i) == kind(s0) for i in range(s0, stop)):
                break  # a constant tail is a plain run
            for p in range(2, 9):
                if (stop - s0) % p == 0 and (stop - s0) // p >= 2 and \
                        all(kind(i) == kind(s0 + (i - s0) % p) for i in range(s0, stop)):
                    return seg(s0) + [("period", s0, p, (stop - s0) // p)]
        return runs(0, stop)
    return seg(n)


def draw_stack(node, edge, n, kind, label, color, cx, y, prev, bx, by, legend, rep_x=None):
    """Draws the stack centred on cx from y down, linked after `prev`; barcode at (bx, by).
    Returns (last node id, y below the stack, {layer: row node id})."""
    row, ng = {}, 0
    for seg in stack_segments(kind, n):
        if seg[0] == "run":
            _, a, k = seg
            span = f"**L{a}–{a + k - 1}** ×{k}" if k > 1 else f"**L{a}**"
            nid = node(f"seg{a}", cx - SEG_W / 2, y, SEG_W, SEG_H, f"{span} · {label(a)}", color(a))
            edge(prev, nid)
            prev = nid
            row.update({i: nid for i in range(a, a + k)})
            y += SEG_H + SEG_GAP
            continue
        _, s0, p, reps = seg
        sfx = "" if ng == 0 else f"-{ng + 1}"  # first group keeps the plain ids
        ng += 1
        y += 40  # room for the group's label, which the viewer draws above the frame
        gy, ry, first = y, y + 50, None
        for j in range(p):
            ls = [s0 + j + r * p for r in range(reps)]
            nid = node(f"per{j}{sfx}", cx - SEG_W / 2, ry, SEG_W, SEG_H,
                       f"{label(s0 + j)}\nL{ls[0]}, {ls[1]}, …, {ls[-1]}" if reps > 2 else
                       f"{label(s0 + j)}\nL{ls[0]}, {ls[1]}", color(s0 + j))
            edge(prev, nid)
            first = first or nid
            prev = nid
            row.update({i: nid for i in ls})
            ry += SEG_H + SEG_GAP
        edge(prev, first, ("left", "left"))  # loop back: the period runs again
        node(f"cap-repeat{sfx}", cx - SEG_W / 2 - 230 if rep_x is None else rep_x,
             (gy + 50 + ry - SEG_GAP) / 2 - 30, 150, 60, f"## ↺ ×{reps}")
        gh = ry - gy + 10
        node(f"grp-period{sfx}", cx - SEG_W / 2 - 60, gy, SEG_W + 120, gh,
             f"×{reps} — L{s0}–{s0 + p * reps - 1}, a period of {p} layers", group=True)
        y += gh + SEG_GAP
    # barcode: every layer in order, one thin cell each, coloured like the rows
    node("cap-barcode", bx, by - 50, max(n * (CELL + CELL_GAP), 560), 40, f"**Every layer, L0 → L{n - 1}** ({legend})")
    for i in range(n):
        node(f"lyr{i}", bx + i * (CELL + CELL_GAP), by, CELL, 44, "", color(i) or ACT)
    for i in sorted(set(list(range(0, n, 10)) + [n - 1])):
        if i == n - 1 or n - 1 - i >= 4:  # keep ticks from colliding
            node(f"note-tick{i}", bx + i * (CELL + CELL_GAP) - 4, by + 50, 60, 30, f"L{i}")
    return prev, y - SEG_GAP, row


# =====================================================================================================
# model.canvas — prompt encoder, the 30 shared layers, and the denoising loop over one canvas
# =====================================================================================================
B = budget()
active = sum(r[3] for r in B)
BX, BW = -1300, 2600
bar("bp", BX, -2420, BW, "Where the parameters are stored", B, 2)
bar("bc", BX, -2220, BW, "What one canvas token is multiplied by per denoising step (active weights)", B, 3)

node("cap-title", BX, -2060, 1260, 230,
     "# DiffusionGemma 26B-A4B — model\n"
     f"A **block-diffusion** LM built on Gemma 4 26B-A4B. Text comes out in **canvases of {CANVAS} tokens**: a canvas "
     f"starts as random token ids and is denoised by up to {STEPS} full forward passes, each predicting all "
     f"{CANVAS} positions at once. Grey bars = tensors (0.08 px/channel), boxes = operations. "
     "*Layer* tab = one layer in detail.")
node("cap-legend", BX + 1340, -2060, 1260, 230,
     f"**One set of {N} layers, two passes.** The *encoder* pass runs them **causally** over the prompt (and over "
     f"each finished canvas) once, only to fill the KV cache. The *decoder* pass runs the same weights "
     f"**bidirectionally** over the canvas, reading that cache.  \n"
     f"Row colours — **grey**: sliding window {WIN}, {NH} q / {NKV} kv heads × {HD} · **cyan**: global, "
     f"{NH} q / {GKV} kv heads × {GHD}, K = V. Every layer: dense MLP + MoE {K} of {E} in parallel.")

# ---- centre column: the canvas being denoised ----
CY = -1760
node("canvas", -230, CY, 460, 90,
     f"**canvas** — {CANVAS} positions after the cached prefix  \n"
     "step 1: uniform random ids from the whole vocab (no mask token)", NOISE)
f = Flow(0, CY + 90 + GAP, last="canvas")
f.tensor(40 / 0.08, f"canvas ids [{CANVAS}]", key="ids")
f.op(f"**embed** — row lookup × √{DIM}", EMB, P["embed"], note=f"{VOCAB:,} × {DIM}", key="embed")
f.tensor(DIM, f"x [{CANVAS}, {DIM}]", key="x")
selfc = f.op("**self-conditioning** — soft embedding of the previous step's logits (softmax · table × √d; "
             f"zeros at step 1) → RMSNorm → gated MLP {DIM}→{FI}→{DIM}, **+ x** → RMSNorm", SC, P["sc"],
             key="sc", h=116, w=460)
f.tensor(DIM, f"x [{CANVAS}, {DIM}]", key="x2")

ROW0 = f.y + 40


def lab(i):
    if TYPES[i] == "sliding_attention":
        return f"sliding {WIN} · {NH}q/{NKV}kv × {HD} · MoE {K}/{E} + MLP"
    return f"global · {NH}q/{GKV}kv × {GHD}, K=V · MoE {K}/{E} + MLP"


LAST, yend, ROW = draw_stack(node, edge, N, lambda i: TYPES[i], lab, lambda i: ROW_COLOR[TYPES[i]], 0, ROW0, f.last,
                             -900, ROW0 + 460, "grey = sliding, cyan = global", rep_x=SEG_W / 2 + 60)
assert ROW[0] == "per0" and len(set(ROW.values())) == 6, "expected one period of 6 covering every layer"

o = Flow(0, yend + 60, last=LAST)
o.tensor(DIM, f"h [{CANVAS}, {DIM}]  after L{N - 1}", key="h")
o.op("**final RMSNorm**", NORM, key="norm")
o.op(f"**lm head** {DIM} → {VOCAB:,}", HEAD, note="tied to embed", key="head")
o.tensor(VOCAB, f"logits [{CANVAS}, {VOCAB:,}]", key="logits", left=True)
o.op(f"**soft-cap** {CAP_SOFT:.0f}·tanh(z/{CAP_SOFT:.0f})", HEAD, key="cap")
temp = o.op(f"**temperature** — linear {TMAX} at the first step → {TMIN} at the last", DIFF, key="temp", h=66, w=460)
o.op("**sample** x_D ~ softmax per position  \nand keep **argmax** x̂", DIFF, key="samp", h=66, w=460)
o.tensor(40 / 0.08, f"x_D, x̂, entropy H [{CANVAS}]", key="xd")
acc = o.op(f"**accept (entropy bound)** — sort positions by H, take the k lowest while "
           f"ΣH − max H ≤ {EB}; they get their sample x_D", DIFF, key="acc", h=96, w=460)
o.op("**renoise** — every other position ← a fresh uniform random id", NOISE, key="ren", h=66, w=460)
o.tensor(40 / 0.08, f"next canvas [{CANVAS}]", key="nc")
dec = o.op(f"**done?** x̂ unchanged for {STAB} step and mean H < {CONF},  \nor {STEPS} steps used", ACT,
           key="done", h=76, w=460)
commit = o.op(f"**yes → commit x̂** (the argmax canvas, not the sample) · ids after an EOS → pad · "
              f"stop on EOS, else next canvas", HEAD, key="commit", h=96, w=460)
edge(dec, commit, label="yes")  # replace the plain link with a labelled one
edges.pop(-2)

# loop lanes on the right: another denoising step, and the self-conditioning feedback
LX = 1660
lr1 = node("loop-r1", LX, ny(dec) + 8, 260, 60, "**no → next step**  \nwhole decoder pass again", NOISE)
edge(dec, lr1, ("right", "left"))
lr2 = node("loop-r2", LX, CY + 15, 260, 60, "↻ **denoising step**  \nnew canvas in", NOISE)
edge(lr1, lr2, ("top", "bottom"))
edge(lr2, "canvas", ("left", "right"), color=NOISE)
SX = 1310
ls1 = node("loop-s1", SX, ny(temp) + 3, 240, 60, "processed logits → next step", SC)
edge(temp, ls1, ("right", "left"))
ls2 = node("loop-s2", SX, ny(selfc) + 28, 240, 60, "self-conditioning input", SC)
edge(ls1, ls2, ("top", "bottom"))
edge(ls2, selfc, ("left", "right"), color=SC)

# ---- attention masks of the two passes ----
MX, MY, CELL = 630, ROW0 + 10, 110
node("cap-mask", MX, MY - 120, 620, 100,
     "**Who attends to whom** — rows = queries, columns = keys. Prefix = prompt + finished canvases (in the KV cache).")
names = ["prefix", "canvas"]
for i in range(2):
    node(f"cap-mrow{i}", MX, MY + i * (CELL + 8) + 36, 110, 34, ["**encoder**", "**decoder**"][i])
    node(f"cap-mcol{i}", MX + 120 + i * (CELL + 8), MY + 2 * (CELL + 8), CELL, 34, names[i])
cells = [["causal ◣", ""], ["all ✓", "all ↔"]]
for i in range(2):
    for j in range(2):
        node(f"mk{i}{j}", MX + 120 + j * (CELL + 8), MY + i * (CELL + 8), CELL, CELL, cells[i][j],
             "4" if cells[i][j] else None)
node("note-mask", MX, MY + 2 * (CELL + 8) + 50, 620, 220,
     f"Encoder: causal; image tokens also see their whole image (`use_bidirectional_attention = \"vision\"`, "
     f"sliding layers). Decoder: `is_causal = False` — every canvas position sees the cached prefix and the whole "
     f"canvas, incl. positions to its right. Sliding layers keep only the last {WIN} prefix tokens in the cache; "
     f"global layers see all of it. The decoder never writes the cache.")
node("note-layers", MX, MY + 2 * (CELL + 8) + 300, 620, 190,
     f"Per layer: attention {fmt(A1['s'])} (sliding) / {fmt(A1['g'])} (global) · dense MLP "
     f"{fmt(P['mlp'] / N)} · router {fmt(P['router'] / N)} · {E} experts {fmt(E * EXPERT)} ({fmt(EXPERT)} each, "
     f"SwiGLU-style {DIM}→{MI}→{DIM}, GELU-tanh).  \nPattern: [5 sliding, 1 global] × {N // 6}. "
     f"Context {C['max_position_embeddings']:,} positions.")

# ---- left column: the prompt / encoder pass that fills the KV cache ----
PX = -1250
p = Flow(PX, CY, labels_left=True)
p.op("**prompt** — chat template text, `<image>` placeholders", ACT, key="raw", w=420)
p.op(f"**tokenizer** (vocab {VOCAB:,})", ACT, key="tok")
p.tensor(40 / 0.08, "prompt ids [P]", key="pids")
p.op("**embed** (same table) × √d", EMB, key="pemb")
p.tensor(DIM, f"[P, {DIM}]", key="pe", left=False)
splice = p.op(f"**splice**: each image's placeholders ← its {SOFT} soft tokens", VIS, key="splice", w=420)
p.tensor(DIM, f"x [P, {DIM}]", key="px")
enc = p.op(f"**encoder pass** — the same {N} layers, run **once**, causal; no self-conditioning, no lm head",
           ACT, key="enc", h=96, w=420)
kv = p.op(f"**KV cache**, per layer  \nsliding: K, V {NKV} × {HD} for the last {WIN} tokens  \n"
          f"global: K (= V) {GKV} × {GHD} for every token", ATT, key="kv", h=116, w=420)
edge(kv, "grp-period", ("right", "left"), label="read-only", color=ATT)
ren = node("reenc", PX - 210, ny(commit), 420, 96,
           f"**encoder pass over the {CANVAS} committed ids** (causal, same layers) appends their K, V", ACT)
edge(commit, ren, ("left", "right"), color=HEAD)
edge(ren, kv, ("top", "bottom"), color=HEAD)

# ---- far left: vision tower, feeding the splice ----
VPX = -2300
vy = ny(splice) - 6 * (OP_H + GAP) - 2 * (9 + GAP)
v = Flow(VPX, vy, labels_left=True)
v.op(f"**image** → {V['patch_size']}×{V['patch_size']} patches", VIS, key="img")
v.op(f"**patch embed** {3 * V['patch_size'] ** 2} → {V['hidden_size']} + 2-D position table",
     VIS, params_of(r"model\.encoder\.vision_tower\.patch_embedder\..*"), key="patch", h=76, w=360)
v.tensor(V["hidden_size"], f"[patches, {V['hidden_size']}]", key="vp")
v.op(f"**ViT × {V['num_hidden_layers']}** · {V['num_attention_heads']} heads × {V['head_dim']} · MLP "
     f"{V['intermediate_size']}", VIS, params_of(r"model\.encoder\.vision_tower\.encoder\..*"), key="vit", h=76, w=360)
v.op(f"**pool {V['pooling_kernel_size']}×{V['pooling_kernel_size']}** · standardize", VIS, key="pool")
v.op(f"**embed_vision** RMSNorm → {V['hidden_size']} → {DIM}", VIS,
     params_of(r"model\.encoder\.embed_vision\..*"), key="evis", h=66)
vt = v.tensor(DIM, f"soft tokens [{SOFT}, {DIM}] per image", key="vt")
edge(vt, splice, ("right", "left"), color=VIS)
node("note-vis", VPX - 300, v.y + 20, 600, 150,
     f"Gemma 4 vision tower ({fmt(P['vis'])}). {SOFT} soft tokens per image by default; the card lists budgets of "
     "70 / 140 / 280 / 560 / 1120. Runs once per image, inside the encoder pass only.")

# ---- sampler notes (left of the sampler ops) ----
sy = ny(acc) - 120
node("note-eb", -1060, sy, 700, 230,
     f"**Nothing is frozen inside a canvas.** The accept set is recomputed every step from fresh logits; a "
     f"position not accepted this step is re-randomised even if it was accepted before. The canvas settles because "
     f"the model grows confident, and only x̂ of the last step is kept. Generation config: up to {STEPS} steps, "
     f"`max_new_tokens` {G['max_new_tokens']} (= 1 canvas), EOS ids {G['eos_token_id']}.")
node("note-card", -1060, sy + 260, 700, 130,
     "Model card: 15–20 tokens per forward pass on average, i.e. a 256-token canvas typically takes ~13–17 "
     "decoder passes instead of 256 autoregressive ones. The decoder also multiplies the tied table twice per "
     "position (lm head and soft embedding).")

# ---- sources + comparison with LLaDA2.2-mini ----
yb = ny(commit) + 180
node("note-llada", -1300, yb, 1400, 300,
     "**vs. LLaDA2.2-mini** (also a diffusion MoE, see that model)  \n"
     "- Noise: LLaDA starts a block as `<|mask|>` tokens and fills masks (masked diffusion); DiffusionGemma "
     "starts from **uniform random ids** and re-randomises rejected positions (uniform-noise diffusion).\n"
     f"- Block: LLaDA 32 tokens, one stack with a block-causal mask and no KV cache; here {CANVAS} tokens, an "
     "**encoder pass** caches the prefix once and a bidirectional **decoder pass** reuses the same weights.\n"
     "- Committing: LLaDA writes a mask when p > 0.5 and keeps it (rewrites / deletes via T2T edits); here an "
     f"**entropy-bound** set is accepted per step, nothing is frozen, and the canvas ends on stability + "
     f"confidence ({STEPS} steps max).\n"
     "- Extra: self-conditioning on the previous step's logits; LLaDA has none. LLaDA routes experts per 32-token "
     f"block; here plain per-token top-{K}.")
node("note-src", 200, yb, 1300, 300,
     f"**Sources.** config.json, generation_config.json and the safetensors headers of "
     f"google/diffusiongemma-26B-A4B-it ({len(S)} tensors, all BF16) — stored total **{fmt(P['all'])}** "
     f"({fmt(P['all'] - P['vis'])} without the vision tower; card: 25.2B), active **{fmt(active)}** (card: 3.8B). "
     "Dataflow from transformers' diffusion_gemma (DiffusionGemmaForBlockDiffusion, EntropyBoundSampler, "
     "StableAndConfidentStoppingCriteria). The text encoder stores only 30 `layer_scalar` values; all its other "
     "weights are tied to the decoder (`_tied_weights_keys`). The repo also ships a diffusers "
     "BlockRefinementScheduler config (block 32, 32 steps, threshold 0.95); transformers' `generate()` does not use it.")
save("model.canvas")

# =====================================================================================================
# block.canvas — one layer in detail (sliding shapes, global variant in brackets)
# =====================================================================================================
_seq[0] = 0
KT2, CLIP2, LBL2 = 0.05, 600, 400


def fl(cx, y, last=None, left=False):
    return Flow(cx, y, last=last, labels_left=left, kt=KT2, clip=CLIP2, lbl=LBL2)


QS, KS = NH * HD, NKV * HD
QG, KG = NH * GHD, GKV * GHD
node("cap-title", -1700, -480, 1600, 200,
     "# DiffusionGemma 26B-A4B — one layer\n"
     f"Shapes for a **sliding** layer (L0–4, 6–10, …); a **global** layer (L{GL0}, L{GL0 + 6}, … L{N - 1}) differs "
     "only in attention — its values are in [brackets]. T = tokens of this pass: the {CANVAS} canvas positions "
     "(decoder) or the new prompt / canvas ids (encoder). Grey bars = tensors "
     f"({KT2} px/channel), boxes = operations with their weights per layer.".replace("{CANVAS}", str(CANVAS)))
node("cap-legend", 0, -480, 1600, 200,
     "Gemma 4 layer: sandwich RMSNorms around attention and FFN, and the FFN is a **dense MLP and a MoE side by "
     "side**, summed. From `DiffusionGemmaDecoderTextLayer` / `DecoderTextAttention`, `Gemma4TextRouter`, "
     "`Gemma4TextExperts`. The encoder layer is the same code with a causal mask and a cache write; it shares "
     "every weight except `layer_scalar`.")

f = fl(0, -220)
xin = f.tensor(DIM, f"x [T, {DIM}]  (residual stream)", key="xin")
f.op("**RMSNorm** (input_layernorm)", NORM, key="n1")
xn = f.tensor(DIM, f"x̂ [T, {DIM}]", key="xn", lbl_dy=-30)
f.y += 40

QX, VX = -900, 900
yb0 = f.y
q = fl(QX, yb0, last=xn, left=True)
q.op(f"**q_proj** {DIM} → {QS} [{QG}]", ATT, params_of(rf"model\.decoder\.layers\.{SL0}\.self_attn\.q_proj\..*"),
     note=f"[{fmt(params_of(rf'model.decoder.layers.{GL0}.self_attn.q_proj.weight'))}]", key="qp")
q.tensor(QS, f"Q [T, {NH}, {HD}]  [{NH}, {GHD}]", key="qt")
q.op("**q_norm** — RMSNorm per head", ATT, key="qn")
q.op(f"**RoPE** θ {RP['sliding_attention']['rope_theta']:,.0f} on all {HD} dims  \n"
     f"[θ {RP['full_attention']['rope_theta']:,.0f} on {GROT} of {GHD} dims]", ATT, key="qr", h=76, w=360)
qo = q.tensor(QS, f"Q [T, {NH}, {HD}]", key="qo")

k = fl(0, yb0, last=xn)
k.op(f"**k_proj** {DIM} → {KS} [{KG}]", ATT, params_of(rf"model\.decoder\.layers\.{SL0}\.self_attn\.k_proj\..*"),
     note=f"[{fmt(params_of(rf'model.decoder.layers.{GL0}.self_attn.k_proj.weight'))}]", key="kp")
kt = k.tensor(KS, f"K [T, {NKV}, {HD}]  [{GKV}, {GHD}]", key="kt")
k.op("**k_norm** — RMSNorm per head", ATT, key="kn")
k.op("**RoPE** (same as Q)", ATT, key="kr")
ko = k.tensor(KS, f"K [T, {NKV}, {HD}]", key="ko")

v = fl(VX, yb0, last=xn)
v.op(f"**v_proj** {DIM} → {KS}  \n[none: V = raw k_proj output]", ATT,
     params_of(rf"model\.decoder\.layers\.{SL0}\.self_attn\.v_proj\..*"), key="vp", h=76, w=360)
v.tensor(KS, f"V [T, {NKV}, {HD}]  [{GKV}, {GHD}]", key="vt")
v.op("**v_norm** — RMSNorm, no weight", ATT, key="vn")
vo = v.tensor(KS, f"V [T, {NKV}, {HD}]", key="vo")

ay = max(q.y, k.y, v.y) + 20
att = node("attn", -300, ay, 600, 136,
           f"**attention**, scale 1 (Q, K already RMS-normed)  \nGQA: {NKV} [{GKV}] KV heads serve {NH} query heads  \n"
           f"keys = cached prefix ‖ this pass · decoder: no mask inside the canvas · encoder: causal, "
           f"sliding window {WIN} [none]", ATT)
edge(qo, att, ("bottom", "left"))
edge(ko, att)
edge(vo, att, ("bottom", "right"))
a = fl(0, ay + 136 + GAP, last=att)
a.tensor(QS, f"[T, {QS}]  [{QG}]", key="ao")
a.op(f"**o_proj** {QS} [{QG}] → {DIM}", ATT, params_of(rf"model\.decoder\.layers\.{SL0}\.self_attn\.o_proj\..*"),
     note=f"[{fmt(params_of(rf'model.decoder.layers.{GL0}.self_attn.o_proj.weight'))}]", key="o")
a.op("**RMSNorm** (post_attention_layernorm)", NORM, key="n2")
add1 = a.op("**⊕ residual add**", ACT, key="add1")
RES_X = -1700
res1 = node("res1", RES_X, ny(xin) - 15, 130, 40, "residual", ACT)
edge(xin, res1, ("left", "right"), color=ACT)
res1b = node("res1b", RES_X, ny(add1) + 8, 130, 40, "residual", ACT)
edge(res1, res1b, ("bottom", "top"), color=ACT)
edge(res1b, add1, ("right", "left"), color=ACT)
h = a.tensor(DIM, f"h [T, {DIM}]", key="h", lbl_dy=-30)
a.y += 40

# three branches from h: dense MLP (left), router (centre, reads raw h), routed experts (right)
by = a.y
m = fl(QX, by, last=h, left=True)
m.op("**RMSNorm** (pre_feedforward_layernorm)", NORM, key="pf1", w=360)
m.op(f"**dense MLP** — GELU-tanh(gate) · up  \n{DIM} → {FI} → {DIM}", MLPC, params_of(r"model\.decoder\.layers\.0\.mlp\..*"),
     key="mlp", h=96, w=360)
m.op("**RMSNorm** (post_feedforward_layernorm_1)", NORM, key="pf1b", w=360)
mo = m.tensor(DIM, f"h₁ [T, {DIM}]", key="mo")

r = fl(0, by, last=h, left=True)
r.op(f"**router** — RMSNorm (no weight) × scale × {DIM}^-½  \n→ proj {DIM} → {E}", "#fbbf24",
     params_of(r"model\.decoder\.layers\.0\.router\..*"), key="rt", h=96, w=360)
r.tensor(E, f"scores [T, {E}]", key="sc")
r.op(f"**softmax → top-{K}**, renormalise to 1, × per_expert_scale", "#fbbf24", key="tk", h=66, w=360)
ro = r.tensor(40 / KT2, f"expert ids, weights [T, {K}]", key="ids")

e = fl(VX, by, last=h)
e.op("**RMSNorm** (pre_feedforward_layernorm_2)", NORM, key="pf2", w=360)
ex_op = e.op(f"**{K} of {E} experts**  \neach GELU-tanh gated {DIM} → {MI} → {DIM}  \nweighted sum", FFN,
             K * EXPERT, note=f"active of {fmt(E * EXPERT)}", key="ex", h=116, w=360)
edge(ro, ex_op, ("right", "left"), label="which & how much")
e.op("**RMSNorm** (post_feedforward_layernorm_2)", NORM, key="pf2b", w=360)
eo = e.tensor(DIM, f"h₂ [T, {DIM}]", key="eo")

jy = max(m.y, r.y, e.y) + 20
j = fl(0, jy)
join = j.op("**h₁ + h₂**", FFN, key="sum")
edge(mo, join, ("bottom", "left"))
edge(eo, join, ("bottom", "right"))
j.op("**RMSNorm** (post_feedforward_layernorm)", NORM, key="pf")
add2 = j.op("**⊕ residual add** (h)", ACT, key="add2")
j.op("**× layer_scalar**", ACT, key="ls")
j.tensor(DIM, f"x [T, {DIM}]  → next layer", key="xout")
res2 = node("res2", RES_X, ny(h) - 15, 130, 40, "residual", ACT)
edge(h, res2, ("left", "right"), color=ACT)
res2b = node("res2b", RES_X, ny(add2) + 8, 130, 40, "residual", ACT)
edge(res2, res2b, ("bottom", "top"), color=ACT)
edge(res2b, add2, ("right", "left"), color=ACT)
node("note-moe", VX - 200, jy + 150, 700, 170,
     f"**Two FFNs per layer.** The dense MLP ({fmt(P['mlp'] / N)}) runs on every token — the card's \"1 shared\" "
     f"expert. The router reads h *before* any norm; the experts read pre_feedforward_layernorm_2(h). "
     f"Experts are stored fused: gate_up [{E}, {2 * MI}, {DIM}], down [{E}, {DIM}, {MI}].")
save("block.canvas")
print("budget", [(b[0], fmt(b[2]), fmt(b[3])) for b in B])
print("stored", fmt(P["all"]), P["all"], "active", fmt(active), active)
