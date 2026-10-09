"""Inkling (thinkingmachines/Inkling): model-level and layer-level canvases, derived from the checkpoint.

Inputs are the repo's config.json and shapes.json (safetensors headers, see fetch_safetensors_shapes.py).
Dataflow follows transformers' modeling_inkling.py (InklingForConditionalGeneration: InklingTextModel +
InklingVisionModel "hmlp" + InklingAudioModel "dmel"). The checkpoint keeps its own tensor names
(model.llm.* / model.visual.* / model.audio.* / model.mtp.*); boxes show both where they differ.
The 8 MTP layers (model.mtp.*) are skipped on load by transformers (_keys_to_ignore_on_load_unexpected),
so their wiring is read from tensor names + config.json's mtp_config.

  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale per canvas)
  - boxes       = operations, all one size; parameter / compute shares are the two bars at the top
  - note-* ids  = annotations (dim, borderless); cap-* ids = borderless captions

usage: python3 inkling_model_canvas.py <shapes.json> <config.json> <models/inkling>
"""
import json
import math
import os
import re
import sys

S = json.load(open(sys.argv[1]))
CFG = json.load(open(sys.argv[2]))
OUT = sys.argv[3]
C, VC, AC, MC = CFG["text_config"], CFG["vision_config"], CFG["audio_config"], CFG["mtp_config"]

N, DIM, VOCAB, UVOCAB = C["num_hidden_layers"], C["hidden_size"], C["vocab_size"], C["unpadded_vocab_size"]
HQ, HKV, HD = C["num_attention_heads"], C["num_key_value_heads"], C["head_dim"]
SHQ, SHKV, SHD, WIN = C["swa_num_attention_heads"], C["swa_num_key_value_heads"], C["swa_head_dim"], C["sliding_window_size"]
DREL, REXT = C["d_rel"], C["rel_extent"]
R, A, SH, MI, DFF = C["n_routed_experts"], C["num_experts_per_tok"], C["n_shared_experts"], C["intermediate_size"], C["dense_intermediate_size"]
KS, MUP, CTX = C["sconv_kernel_size"], C["logits_mup_width_multiplier"], C["model_max_length"]
LOCAL = set(C["local_layer_ids"])
N_DENSE = C["dense_mlp_idx"]  # layers [0, dense_mlp_idx) use a dense SwiGLU FFN
NMTP, MTP_LOCAL = MC["num_nextn_predict_layers"], set(MC["local_layer_ids"])
PS, PT, NCH, VL = VC["patch_size"], VC["temporal_patch_size"], VC["n_channels"], VC["n_layers"]
MELS, MELV = AC["n_mel_bins"], AC["mel_vocab_size"]
FULL = [i for i in range(N) if i not in LOCAL]

EMB, NORM, FFN, HEAD, ACT, ATT, MTP = "6", "3", "4", "2", "#64748b", "5", "#f59e0b"
VISION, AUDIO, SHARED, REL = "#a78bfa", "#f472b6", "#22c55e", "1"
ROW_COLOR = {("dense", "sliding"): "#fbbf24", ("sparse", "sliding"): "#94a3b8", ("sparse", "full"): "5"}


def params_of(pattern):
    rx = re.compile(pattern)
    return sum(math.prod(shp) for k, (dt, shp) in S.items() if rx.fullmatch(k))


LL = r"model\.llm\.layers\.\d+\."
P = dict(
    all=params_of(r".*"),
    embed=params_of(r"model\.llm\.embed\.weight"),
    head=params_of(r"model\.llm\.unembed\.weight"),
    attn=params_of(LL + r"(?:attn\..*|attn_sconv\.weight)"),
    dense=params_of(LL + r"mlp\.(?:w13_dn|w2_md)\.weight|" + LL + r"mlp\.global_scale"),
    routed=params_of(LL + r"mlp\.experts\..*"),
    shared=params_of(LL + r"mlp\.(?:shared_experts|gate)\..*"),
    rest=params_of(LL + r"(?:attn_norm|mlp_norm|mlp_sconv)\.weight|model\.llm\.(?:embed_)?norm\.weight"),
    mtp=params_of(r"model\.mtp\..*"),
    vision=params_of(r"model\.visual\..*"),
    audio=params_of(r"model\.audio\..*"),
    # one sliding MoE layer (L2) and one full MoE layer (L5), op by op
    expert=params_of(r"model\.llm\.layers\.2\.mlp\.experts\..*") // R,
    shexp=params_of(r"model\.llm\.layers\.2\.mlp\.shared_experts\..*"),
    router=params_of(r"model\.llm\.layers\.2\.mlp\.gate\..*"),
    densel=params_of(r"model\.llm\.layers\.0\.mlp\..*"),
    attn_s=params_of(r"model\.llm\.layers\.2\.(?:attn\..*|attn_sconv\.weight)"),
    attn_f=params_of(r"model\.llm\.layers\.5\.(?:attn\..*|attn_sconv\.weight)"),
    mtpin=params_of(r"model\.mtp\.layers\.\d+\.(?:input_proj|embed_norm|hidden_norm)\.weight"),
)
N_MOE = N - N_DENSE
assert sorted({int(re.match(r"model\.llm\.layers\.(\d+)", k)[1]) for k in S if ".mlp.experts." in k}) == list(range(N_DENSE, N))
# full-attention layers are exactly the ones whose k projection has num_key_value_heads (not swa_*) heads
assert FULL == [i for i in range(N) if S[f"model.llm.layers.{i}.attn.wk_dv.weight"][1][0] == HKV * HD]
assert all(S[f"model.mtp.layers.{i}.transformer_block.attn.wk_dv.weight"][1][0] == (SHKV * SHD if i in MTP_LOCAL else HKV * HD)
           for i in range(NMTP))
assert S["model.llm.layers.2.mlp.gate.weight"][1] == [R + SH, DIM]
assert S["model.audio.encoder.weight"][1] == [MELS * MELV, DIM]


def tw(name, layer=2):
    """params of one tensor in main layer <layer>"""
    return params_of(rf"model\.llm\.layers\.{layer}\.{re.escape(name)}")


def budget():
    """[(name, colour, stored params, params multiplied per text token in the main forward, why-zero)]"""
    rows = [
        (f"routed experts {R}×{N_MOE}", FFN, P["routed"], N_MOE * A * P["expert"]),
        (f"attention ×{N}", ATT, P["attn"], P["attn"]),
        (f"shared experts ×{SH} + router ×{N_MOE}", SHARED, P["shared"], P["shared"]),
        (f"dense FFN ×{N_DENSE}", "#fbbf24", P["dense"], P["dense"]),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, P["head"], P["head"]),
        (f"MTP ×{NMTP}", MTP, P["mtp"], 0, "only when drafting"),
        ("vision encoder", VISION, P["vision"], 0, "runs per image"),
        ("audio encoder", AUDIO, P["audio"], 0, "runs per audio frame"),
        ("norms + MLP short-convs", NORM, P["rest"], P["rest"]),
    ]
    assert sum(r[2] for r in rows) == P["all"], (sum(r[2] for r in rows), P["all"])
    return rows


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.1f}K"


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


# ======================================================================= model.canvas
ROW_W, ROW_H, ROW_GAP = 300, 34, 6
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
bar("bp", -1700, "Where the parameters are stored (BF16 checkpoint)", B, 2)
bar("bc", -1500, "What one text token is multiplied by in the main forward (active weights)", B, 3)

node("cap-title", -900, -1300, 1250, 200,
     "# Inkling — model\n"
     "Read top → bottom. **Grey bars = tensors**, length ∝ channels per token (0.08 px/channel). "
     "**Boxes = operations**, all one size; where parameters and compute live is the pair of bars above. "
     "T = sequence length, P = 40×40 image patches, F = 50 ms audio frames. *Layer* tab = one decoder layer in detail.")
node("cap-legend", 450, -1300, 1250, 200,
     f"Every layer = **GQA attention with no RoPE**: position comes from a learned, input-dependent relative bias over "
     f"the last {WIN} (sliding) or {REXT} (global) distances, plus depthwise causal short-convs (kernel {KS}) on k, v and both "
     f"sub-block outputs.  \nRow colours — **amber**: sliding {WIN} + dense SwiGLU FFN (L0–{N_DENSE - 1}) · **grey**: sliding "
     f"{WIN}, {SHKV} KV heads + MoE · **cyan**: global, {HKV} KV heads + MoE (every 6th layer). "
     f"MoE = {A} of {R} routed experts + {SH} shared, one sigmoid router scores all {R + SH}.")

# ---------- three input branches, merged into one embedding sequence ----------
cx = ROW_W / 2
f = Flow(cx, -980)
f.op("**raw text**  \"Describe this…\"", ACT, key="raw")
f.op(f"**tokenizer** (tiktoken, vocab {VOCAB:,})", ACT, key="tok")
f.tensor(40 / 0.08, "token ids [T] — image / audio placeholders included", key="ids")
f.op("**embed** — look up row *id*\nthen **embed_norm** (RMSNorm)", EMB, P["embed"], note=f"table {VOCAB:,} × {DIM}", key="embed", h=90)
emb = f.tensor(DIM, f"e [T, {DIM}]", key="e")

# vision: hierarchical MLP ("hmlp") — fold space/time into channels in 4 steps, one token per patch
VIS = [(f"model.visual.layers.linear_{i}.weight", S[f"model.visual.layers.linear_{i}.weight"][1]) for i in range(VL)]
FOLD = ["5×5 pixels", "2×2", "4×4", f"{PT} frames"]  # factors of patch 40 = 5·2·4 and temporal 2 (plan_out_scales)
assert VIS[0][1][1] == 5 * 5 * NCH and VIS[2][1][0] == PS * PS * NCH and VIS[-1][1] == [DIM, PT * PS * PS * NCH]
v = Flow(-560, -980, labels_left=True)
v.op("**image** → 40×40 patches\n(still image repeated as 2 frames)", ACT, key="img", h=72)
v.tensor(PT * PS * PS * NCH, f"patches [P, {PT}×{PS}×{PS}×{NCH} = {PT * PS * PS * NCH}]", key="vp")
for i, (name, (o_, i_)) in enumerate(VIS):
    last = i == VL - 1
    v.op(f"**fold {FOLD[i]} → linear_{i}** {i_} → {o_}" + ("" if last else "\nRMSNorm · GELU"), VISION,
         params_of(re.escape(name) + "|" + re.escape(f"model.visual.layers.norm_{i}.weight")), key=f"hm{i}", h=56 if last else 72)
    v.tensor(o_, f"[P, {o_}]" + ("" if last else f"  per {['5×5', '10×10', '40×40'][i]}-pixel cell"), key=f"vh{i}")
v.op("**final_norm** (RMSNorm)", VISION, params_of(r"model\.visual\.final_norm\.weight"), key="vfn")
vt = v.tensor(DIM, f"image tokens [P, {DIM}] — 1 per patch", key="vt")

# audio: discretised mel ("dMel") — each of 80 bins is a 16-level token; one embedding row per (bin, level), summed
a = Flow(1200, -980)
a.op("**audio** 16 kHz WAV", ACT, key="wav")
a.op(f"**dMel**: {MELS}-bin log-mel every 50 ms\n(hop 800), each bin → {MELV} levels in [{AC['dmel_min_value']:g}, {AC['dmel_max_value']:g}]",
     AUDIO, key="dmel", h=72)
a.tensor(MELS, f"levels [F, {MELS}] — integers 0…{MELV - 1}", key="al")
a.op(f"**embed** row bin·{MELV} + level\n(table {MELS * MELV} × {DIM}), **sum** over {MELS} bins", AUDIO,
     params_of(r"model\.audio\.encoder\.weight"), key="aemb", h=90)
a.tensor(DIM, f"[F, {DIM}]", key="ae")
a.op("**final_norm** (RMSNorm)", AUDIO, params_of(r"model\.audio\.final_norm\.weight"), key="afn")
at = a.tensor(DIM, f"audio tokens [F, {DIM}] — 1 per 50 ms", key="at")

f.y = max(f.y, v.y, a.y) + 20
mrg = f.op("**splice**: image / audio tokens\nreplace their placeholder rows", ACT, key="merge", h=72)
edge(vt, mrg, ("bottom", "left"), color=VISION)
edge(at, mrg, ("bottom", "right"), color=AUDIO)
x0 = f.tensor(DIM, f"x [T, {DIM}]", key="x")
ROW0 = f.y + 30

kind = lambda i: ("dense" if i < N_DENSE else "sparse", "sliding" if i in LOCAL else "full")
lab = lambda i: f"{'sliding ' + str(WIN) if i in LOCAL else 'global'} · {'dense FFN' if i < N_DENSE else 'MoE'}"
LAST, yend, ROW = draw_stack(node, edge, N, kind, lab, lambda i: ROW_COLOR[kind(i)], cx, ROW0, x0,
                             cx + 700, ROW0 + 40, "amber = dense FFN, grey = sliding MoE, cyan = global MoE",
                             rep_x=cx + SEG_W / 2 + 330)
ylast = max(yend, ROW0 + 710) - ROW_H  # below the notes beside the stack

kvs, kvf = 2 * SHKV * SHD, 2 * HKV * HD
node("note-pattern", -760, ROW0, 600, 260,
     f"**Layer pattern** — {N - len(FULL)} sliding + {len(FULL)} global layers ([sliding ×5, global] ×{N // 6}).\n"
     f"Sliding: {SHQ} query / {SHKV} KV heads, causal window {WIN}, relative bias over all {WIN} distances. "
     f"Global: {HQ} query / {HKV} KV heads over the whole context; relative bias only for the last {REXT} tokens, "
     "beyond that no position signal at all, and past position "
     f"{C['log_scaling_n_floor']:,} queries are scaled by 1 + {C['log_scaling_alpha']}·ln(pos / {C['log_scaling_n_floor']:,}).")
node("note-kv", -760, ROW0 + 300, 600, 170,
     f"**KV cache per token per layer**: sliding K+V = 2×{SHKV}×{SHD} = {kvs} values but only the last {WIN} tokens; "
     f"global 2×{HKV}×{HD} = {kvf} values for every token — so at long context the {len(FULL)} global layers hold "
     f"nearly all of it. Each layer also keeps {KS - 1} past steps for each of its 4 short-convs.")
node("note-layer", -760, ROW0 + 510, 600, 200,
     f"**One MoE layer** ≈ attention {fmt(P['attn_s'])} (sliding) / {fmt(P['attn_f'])} (global) + router {fmt(P['router'])} "
     f"+ {SH} shared experts {fmt(P['shexp'])} + {R} routed experts × {fmt(P['expert'])}, of which {A} run per token. "
     f"Expert SwiGLU width {MI}; dense layers L0–{N_DENSE - 1} use one SwiGLU {DFF} ({fmt(P['densel'])}).")

# ---------- output ----------
o = Flow(cx, ylast + ROW_H + 60, last=LAST, labels_left=True)
o.tensor(DIM, f"h [T, {DIM}]  after L{N - 1}", key="ho")
o.op("**final RMSNorm** (llm.norm)", NORM, key="fn")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**÷ {MUP:g}** (muP width multiplier)", ACT, key="mup")
o.op(f"**lm head** (unembed) {DIM} → {VOCAB:,}", HEAD, P["head"], note="not tied to embed", key="head")
o.tensor(VOCAB, f"logits [T, {VOCAB:,}]", key="logits")
o.op(f"**keep first {UVOCAB:,}** (rest is padding)", HEAD, key="slice")
o.op("**softmax → sample**", ACT, key="samp")
next_id = o.op("**next token id**", ACT, key="next")

# ---------- MTP: 8 chained next-token-prediction layers (model.mtp.layers.0–7) ----------
tap = node("note-tap", cx + SEG_W / 2 + 30, next(n["y"] for n in nodes if n["id"] == ROW[N - 1]) + 11, 260, ROW_H,
           f"↳ L{N - 1} hidden state → MTP")
m = Flow(1700, ylast + ROW_H + 60)
m.op("**embed** token t+k — main table\n→ MTP **embed_norm**", EMB, key="membed", h=72)
m.tensor(DIM, f"e [T, {DIM}]", key="me")
cat = m.op("**hidden_norm**(h) ‖ **embed_norm**(e)\nRMSNorm each, hidden first", MTP, key="mcat", h=72)
edge(tap, cat, ("right", "left"), color=MTP)
m.tensor(2 * DIM, f"[T, {2 * DIM}]", key="mc")
m.op(f"**input_proj** {2 * DIM} → {DIM}", MTP, P["mtpin"] // NMTP, note="incl. both norms", key="min")
m.tensor(DIM, f"h_k [T, {DIM}]", key="mh")
blk = m.op(f"**1 decoder block** — attention as listed →\n+ **dense** SwiGLU {DFF} (no MoE)", MTP,
           (P["mtp"] - P["mtpin"]) // NMTP, note="avg per layer", key="mblk", h=72)
m.op("**→ lm head** (main's; no head\ntensors of its own)", HEAD, key="mhead", h=72)
m.tensor(40 / 0.08, "draft token t+k+1", key="mdraft")
ver = m.op("**verify**: main model checks the\ndrafts in its next forward", ACT, key="mver", h=72)
edge(ver, next_id, ("left", "right"), "accepted drafts = extra tokens", MTP)

MX = 2750
yb = node("note-mrows", MX, ylast + ROW_H + 60 - 50, ROW_W, 40, f"**{NMTP} MTP layers**, each feeds the next:")
prevm = None
for k in range(NMTP):
    y = ylast + ROW_H + 60 + k * (ROW_H + ROW_GAP)
    att = f"sliding {WIN}" if k in MTP_LOCAL else "global"
    nid = node(f"M{k}", MX, y, ROW_W, ROW_H, f"**MTP{k}** · {att} → t+{k + 2}", MTP)
    if prevm:
        edge(prevm, nid)
    prevm = nid
node("note-mtp", MX, ylast + ROW_H + 60 + NMTP * (ROW_H + ROW_GAP) + 20, 640, 230,
     f"MTP = multi-token prediction ({NMTP} layers, {fmt(P['mtp'])} stored). Layer k mixes the hidden state of "
     "layer k−1 (the main model's for k = 0) with the embedding of the next known token and predicts one token further, "
     "so the chain can draft several tokens per step for speculative decoding. transformers skips model.mtp.* on load; "
     "this wiring is read from its tensor names and config.json's mtp_config (chain_hidden_post_norm = "
     f"{str(MC['chain_hidden_post_norm']).lower()}).")

node("note-src", -900, o.y + 120, 1300, 170,
     f"**Sources.** config.json + safetensors headers of thinkingmachines/Inkling ({len(S)} tensors, BF16 + F32 router "
     f"biases; experts stored as stacked [{R}, …] tensors). All tensors sum to **{fmt(P['all'])}** (the README says 975B "
     f"total, 41B active). Dataflow from transformers modeling_inkling.py. Context {CTX:,} tokens (model_max_length); "
     "no rotary embedding anywhere.")

save("model.canvas")

# ======================================================================= block.canvas
KT, CLIP, LBL = 0.05, 480, 300
RC, QC, KC, VC_ = -1750, -1000, -200, 600  # relative-bias, query, key and value columns


def flow(cx, y, last=None, left=False):
    return Flow(cx, y, last=last, labels_left=left, kt=KT, clip=CLIP, lbl=LBL)


node("cap-btitle", -2100, -560, 1500, 170,
     "# Inkling — one decoder layer\n"
     f"A sliding-window MoE layer (L2–4, L6–10, …; {N - len(FULL) - N_DENSE} of {N}). Grey bars = tensors "
     f"({KT} px/channel, long ones clipped), boxes = operations with their weights per layer (checkpoint name, transformers name). "
     "Read top → bottom; the four middle columns run side by side.")
node("cap-bnote", -500, -560, 1500, 170,
     f"**Layer variants.** {len(FULL)} *global* layers (L{FULL[0]}, L{FULL[1]}, … L{FULL[-1]}): {HKV} KV heads (wk/wv {DIM}→{HKV * HD}), "
     f"no window, relative bias over {REXT} distances, log-scaled queries ({fmt(P['attn_f'])} attention). "
     f"L0–{N_DENSE - 1}: dense SwiGLU {DIM}→{DFF}→{DIM} × global_scale ({fmt(P['densel'])}) instead of the MoE. "
     "Norms, short-convs and residuals are identical everywhere.")

mf = flow(0, -330)
mf.tensor(DIM, f"x [T, {DIM}] from previous layer", key="bx")
mf.op("**RMSNorm** (attn_norm)", NORM, key="bn1")
xn = mf.tensor(DIM, f"x̂ [T, {DIM}]", key="bxn")
YB = mf.y + 60

conv = f"**short-conv**: causal depthwise, kernel {KS}\n+ its own input (fp32)"
r = flow(RC, YB, last=xn, left=True)
r.op(f"**wr_du** (r_proj) {DIM} → {SHQ}×{DREL}", REL, tw("attn.wr_du.weight"), key="rp")
edges[-1].update(fromSide="left")
r.tensor(SHQ * DREL, f"r [T, {SHQ}, {DREL}]", key="rt")
r.op(f"**rel_logits_proj**: r_h · P ({DREL}×{WIN})\none bias per head per distance", REL, tw("attn.rel_logits_proj.proj"), key="rl", h=80)
r.tensor(SHQ * WIN, f"bias [T, {SHQ}, {WIN} distances]", key="rb")
r_out = r.op(f"**gather by distance** q−k\n0 beyond {WIN}", REL, key="rg", h=72)

q = flow(QC, YB, last=xn)
q.op(f"**wq_du** (q_proj) {DIM} → {SHQ}×{SHD}", ATT, tw("attn.wq_du.weight"), key="qp")
q.tensor(SHQ * SHD, f"q [T, {SHQ}, {SHD}]", key="qt")
q.op("**q_norm** — RMSNorm per head", NORM, tw("attn.q_norm.weight"), key="qn")
q_out = q.tensor(SHQ * SHD, f"q [T, {SHQ}, {SHD}]", key="qo")

k = flow(KC, YB, last=xn)
k.op(f"**wk_dv** (k_proj) {DIM} → {SHKV}×{SHD}", ATT, tw("attn.wk_dv.weight"), key="kp")
k.tensor(SHKV * SHD, f"k [T, {SHKV * SHD}]", key="kt")
k.op(conv, ATT, tw("attn.k_sconv.weight"), key="kc", h=80)
k.op("**k_norm** — RMSNorm per head", NORM, tw("attn.k_norm.weight"), key="kn")
k_out = k.tensor(SHKV * SHD, f"k [T, {SHKV}, {SHD}]  (cached)", key="ko")

vv = flow(VC_, YB, last=xn)
vv.op(f"**wv_dv** (v_proj) {DIM} → {SHKV}×{SHD}", ATT, tw("attn.wv_dv.weight"), key="vp")
vv.tensor(SHKV * SHD, f"v [T, {SHKV * SHD}]", key="vt")
vv.op(conv, ATT, tw("attn.v_sconv.weight"), key="vc", h=80)
v_out = vv.tensor(SHKV * SHD, f"v [T, {SHKV}, {SHD}]  (cached)", key="vo")

YA = max(r.y, q.y, k.y, vv.y) + 60
core = flow(0, YA)
core.op_w = 700
att = core.op(f"**attention** — {SHQ} query heads share {SHKV} KV heads ({SHQ // SHKV}:1)\n"
              f"score = q·k / {SHD} + relative bias · causal window {WIN}", ATT, key="att", h=80)
core.op_w = 300
edge(q_out, att, ("bottom", "left"))
edge(k_out, att, ("bottom", "top"))
edge(v_out, att, ("bottom", "right"))
edge(r_out, att, ("bottom", "left"), color=REL)
core.tensor(SHQ * SHD, f"o [T, {SHQ}×{SHD}]", key="ao")
core.op(f"**wo_ud** (o_proj) {SHQ * SHD} → {DIM}", ATT, tw("attn.wo_ud.weight"), key="op")
core.tensor(DIM, f"[T, {DIM}]", key="ao2")
core.op(conv.replace("**short-conv**", "**attn_sconv**"), ATT, tw("attn_sconv.weight"), key="asc", h=80)
core.op("**+ residual** (x from the top)", ACT, key="r1")
core.tensor(DIM, f"h [T, {DIM}]", key="h1")
core.op("**RMSNorm** (mlp_norm)", NORM, key="bn2")
hn = core.tensor(DIM, f"ĥ [T, {DIM}]", key="hn", left=True)
rt = core.op(f"**router** {DIM}→{R}+{SH}, sigmoid · top-{A} of {R} by score+bias", FFN, P["router"], key="rt", h=80)
sh = node("shexp", 420, core.y - 80 - 40, 300, 80,
          f"**{SH} shared experts** (always on)  \nSwiGLU {DIM}→{MI}→{DIM} each  \n{fmt(P['shexp'])}", SHARED)
edge(hn, sh, ("right", "top"))
core.tensor(40 / KT, f"{A} expert ids + {A}+{SH} weights / token", key="rid")
core.op(f"**weights**: softmax of log σ over the {A} picked\n+ {SH} shared logits, × {C['route_scale']:g} × global_scale",
        FFN, key="rw", h=80)
core.op(f"**{A} of {R} routed experts**, each SwiGLU {DIM}→{MI}→{DIM}", FFN, P["expert"],
        note=f"each · {fmt(R * P['expert'])} all", key="rex", h=80)
sm = core.op("**Σ** weighted routed + shared experts", ACT, key="sum")
edge(sh, sm, ("bottom", "right"))
core.tensor(DIM, f"[T, {DIM}]", key="mo")
core.op(conv.replace("**short-conv**", "**mlp_sconv**"), FFN, tw("mlp_sconv.weight"), key="msc", h=80)
core.op("**+ residual**", ACT, key="r2")
core.tensor(DIM, f"x [T, {DIM}] → next layer", key="bout")

node("note-pos", RC - 150, YA + 140, 560, 210,
     "**No RoPE.** Each token's r vector mixes a learned bank of bias-vs-distance profiles (P), so every head adds its own "
     f"content-dependent bias to the last {WIN} keys ({REXT} in global layers). q and k are RMS-normalised per head, "
     f"hence the 1/{SHD} scale instead of 1/√{SHD}.")
node("note-sink", 820, YA + 520, 520, 190,
     f"**Shared-expert sink.** The router has {R + SH} rows: the {SH} shared experts are scored with the routed ones, "
     f"and the {A}+{SH} picked weights are normalised together, so a token can lean on the shared experts or on its "
     "routed picks. e_score_correction_bias (mlp.gate.bias) only steers which experts get picked.")

save("block.canvas")
print("budget", [(r_[0], fmt(r_[2]), fmt(r_[3])) for r_ in B])
print("active total", sum(r_[3] for r_ in B), fmt(sum(r_[3] for r_ in B)), "stored total", sum(r_[2] for r_ in B), fmt(sum(r_[2] for r_ in B)))
