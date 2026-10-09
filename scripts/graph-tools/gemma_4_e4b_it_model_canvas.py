"""Gemma 4 E4B-it (google/gemma-4-E4B-it): whole-model canvas + one-decoder-layer canvas.

Read top → bottom:
  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale per canvas)
  - boxes       = operations, all one size; parameter / compute shares are the two bars at the top of model.canvas
  - layer rows  = repeating layers drawn once with ×N (barcode = every layer), coloured by attention type × whether the layer computes its own K/V
  - note-* ids  = annotations (dim, borderless); cap-* ids = borderless captions
Every number comes from config.json + the safetensors header of the HF repo (single file model.safetensors,
no index — shapes.json is that header) and the transformers `gemma4` modeling code (modeling_gemma4.py):
Gemma4ForConditionalGeneration → Gemma4TextModel (per-layer embeddings, KV sharing), Gemma4VisionModel,
Gemma4AudioModel, Gemma4MultimodalEmbedder.

usage: python3 gemma_4_e4b_it_model_canvas.py <shapes.json> <config.json> <models/gemma-4-e4b-it>
"""
import json
import math
import os
import re
import sys

S = json.load(open(sys.argv[1]))
CFG = json.load(open(sys.argv[2]))
OUT = sys.argv[3]
C, V, A = CFG["text_config"], CFG["vision_config"], CFG["audio_config"]

N, DIM, VOCAB, FF = C["num_hidden_layers"], C["hidden_size"], C["vocab_size"], C["intermediate_size"]
HQ, HKV, HD, GHD, WIN = C["num_attention_heads"], C["num_key_value_heads"], C["head_dim"], C["global_head_dim"], C["sliding_window"]
PLD, PLV = C["hidden_size_per_layer_input"], C["vocab_size_per_layer_input"]
TYPES, NSH = C["layer_types"], C["num_kv_shared_layers"]
RP = C["rope_parameters"]
FIRST_SH = N - NSH  # first layer that reuses K/V instead of computing it
SOFTCAP = C["final_logit_softcapping"]
VD, VL, PS, POOL = V["hidden_size"], V["num_hidden_layers"], V["patch_size"], V["pooling_kernel_size"]
AD, AL, AOUT = A["hidden_size"], A["num_hidden_layers"], A["output_proj_dims"]
SUB = A["subsampling_conv_channels"]
MEL = 128  # processor_config.json feature_extractor.feature_size (16 kHz, 10 ms hop)
assert not C["attention_k_eq_v"] and not C["use_double_wide_mlp"] and not C["enable_moe_block"]
assert C["tie_word_embeddings"] and "lm_head.weight" not in S


def last_before(kind):
    """The last layer before the sharing point of this type — its K/V are reused by every later layer of the type."""
    return max(i for i in range(FIRST_SH) if TYPES[i] == kind)


SRC = {k: last_before(k) for k in ("sliding_attention", "full_attention")}
FULL = [i for i, t in enumerate(TYPES) if t == "full_attention"]
N_FULL = len(FULL)

# colours (JSON Canvas presets "1"–"6" or hex)
EMB, NORM, FFN, HEAD, ACT, ATT = "6", "3", "4", "2", "#64748b", "5"
VISION, AUDIO, PLE, KVSH = "#a78bfa", "#f472b6", "#a3e635", "#f59e0b"
ROW_COLOR = {("sliding_attention", False): "#94a3b8", ("full_attention", False): "5",
             ("sliding_attention", True): "#cbd5e1", ("full_attention", True): "#67e8f9"}

OP_W, OP_H, GAP = 300, 56, 40
MIN_SEG = 16  # the viewer draws no box narrower than ~14px


def params_of(pattern):
    rx = re.compile(pattern)
    return sum(math.prod(s) for k, (_, s) in S.items() if rx.fullmatch(k))


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M" if n >= 1e5 else f"{n / 1e3:.1f}K"


LM = r"model\.language_model\."
SH = LM + r"layers\.(?:" + "|".join(str(i) for i in range(FIRST_SH, N)) + r")\."
P = {
    "all": params_of(r".*"),
    "embed": params_of(LM + r"embed_tokens\.weight"),
    "ple": params_of(LM + r"embed_tokens_per_layer\.weight"),
    "pleproj": params_of(LM + r"per_layer_model_projection\.weight"),
    "plenorm": params_of(LM + r"per_layer_projection_norm\.weight"),
    "mlp": params_of(LM + r"layers\.\d+\.mlp\..*"),
    "attn": params_of(LM + r"layers\.\d+\.self_attn\..*"),
    "dead": params_of(SH + r"self_attn\.(?:k_proj|v_proj|k_norm)\..*"),
    "plel": params_of(LM + r"layers\.\d+\.(?:per_layer_input_gate|per_layer_projection|post_per_layer_input_norm)\..*"),
    "norms": params_of(LM + r"(?:layers\.\d+\.(?:input_layernorm|post_attention_layernorm|pre_feedforward_layernorm|"
                       r"post_feedforward_layernorm)\.weight|layers\.\d+\.layer_scalar|norm\.weight)"),
    "vit": params_of(r"model\.vision_tower\..*"),
    "vpatch": params_of(r"model\.vision_tower\.patch_embedder\..*"),
    "vemb": params_of(r"model\.embed_vision\..*"),
    "aud": params_of(r"model\.audio_tower\..*"),
    "asub": params_of(r"model\.audio_tower\.subsample_conv_projection\..*"),
    "aout": params_of(r"model\.audio_tower\.output_proj\..*"),
    "aemb": params_of(r"model\.embed_audio\..*"),
}
lp = lambda i, name: params_of(LM + rf"layers\.{i}\.{name}")
SL0, FL0 = 0, FULL[0]  # representative sliding / full layer
P["mlp1"] = lp(SL0, r"mlp\..*")
P["plel1"] = lp(SL0, r"(?:per_layer_input_gate|per_layer_projection|post_per_layer_input_norm)\..*")
# the checkpoint stores K/V weights for every layer; transformers drops them for the KV-shared ones on load
assert all(lp(i, r"self_attn\.k_proj\.weight") for i in range(N))
assert P["dead"] == sum(lp(i, r"self_attn\.(?:k_proj|v_proj|k_norm)\..*") for i in range(FIRST_SH, N))
assert PLV * N * PLD == P["ple"] and VOCAB * DIM == P["embed"]
assert not any("altup" in k or "laurel" in k for k in S), "Gemma-3n-style AltUp/LAuReL tensors present"


class Canvas:
    def __init__(self, kt, clip, lbl_w):
        self.nodes, self.edges, self.kt, self.clip, self.lbl_w, self.seq = [], [], kt, clip, lbl_w, 0

    def node(self, nid, x, y, w, h, text="", color=None, group=False):
        text = re.sub(r"(?<=[^\n])\n(?=[^\n-])", "  \n", text)  # markdown hard break; keeps paragraphs and lists
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

    def __init__(self, cv, cx, y, last=None, labels_left=False, op_w=OP_W, lbl_w=None):
        self.cv, self.cx, self.y, self.last, self.labels_left, self.op_w = cv, cx, y, last, labels_left, op_w
        self.lbl_w = lbl_w or cv.lbl_w

    def _link(self, top, bottom):
        if self.last:
            self.cv.edge(self.last, top)
        self.last = bottom

    def op(self, text, color, params=0, note="", key="op", h=OP_H):
        if params:
            text += f"\n{fmt(params)}" + (f" · {note}" if note else "")
        nid = self.cv.node(self.cv.uid(key), self.cx - self.op_w / 2, self.y, self.op_w, h, text, color)
        self._link(nid, nid)
        self.y += h + GAP
        return nid

    def tensor(self, dim, label, lanes=1, key="t", left=None):
        cv = self.cv
        real = dim * cv.kt
        w = max(16, min(real, cv.clip))
        ids = [cv.node(cv.uid(key), self.cx - w / 2, self.y + j * 14, w, 9, "", ACT) for j in range(lanes)]
        clip = f" — bar clipped (real {real:.0f}px)" if real > w else ""
        half = max(w / 2, self.op_w / 2) + 20
        lx = self.cx - half - self.lbl_w if (self.labels_left if left is None else left) else self.cx + half
        cv.node(cv.uid("note-lbl"), lx, self.y - 10, self.lbl_w, 34, f"`{label}`{clip}")
        self._link(ids[0], ids[-1])
        self.y += lanes * 14 + GAP
        return ids[0], ids[-1]


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


# ======================= model.canvas =======================
M = Canvas(kt=0.08, clip=1400, lbl_w=440)
ROW_W, ROW_H, ROW_GAP, BAR_W, BAR_H = 300, 34, 6, 2600, 44
n_slide = N - N_FULL
kv_s, kv_f = 2 * HKV * HD, 2 * HKV * GHD  # K+V values per token per layer

M.node("cap-title", -900, -1270, 1250, 180,
       "# Gemma 4 E4B-it — model\n"
       "Read top → bottom. **Grey bars = tensors**, length ∝ channels per token (0.08 px/channel). "
       "**Boxes = operations**, all the same size; where parameters and compute live is the pair of bars above. "
       "T = sequence length. *Layer* tab = one decoder layer in detail.")
M.node("cap-legend", 450, -1270, 1250, 180,
       f"**Dense** decoder, {N} layers: {n_slide} sliding-window ({WIN}) + {N_FULL} full, pattern [5 sliding, 1 full] ×{N // 6}.  \n"
       f"Rows — **grey**: sliding, own K/V · **cyan**: full, own K/V · **pale grey / pale cyan**: L{FIRST_SH}–{N - 1} compute "
       f"**no K/V** and reuse the cache of L{SRC['sliding_attention']} / L{SRC['full_attention']}.  \n"
       f"Every layer also gets its own {PLD}-wide **per-layer embedding** slice (green). "
       f"Images, video and audio enter through their own encoders (left).")

# ---- text input (main column) ----
cx = ROW_W / 2
f = Flow(M, cx, -960)
f.op("**raw text**  \"The cat sat…\"", ACT, key="raw")
tok = f.op(f"**tokenizer** (vocab {VOCAB:,})", ACT, key="tok")
f.tensor(40 / M.kt, "token ids [T] — one integer per token", key="ids", left=True)
f.op(f"**embed** — look up row *id*, × √{DIM}", EMB, P["embed"], note=f"table {VOCAB:,} × {DIM}", key="embed", h=72)
f.tensor(DIM, f"x [T, {DIM}]", key="xe")
splice = f.op("**splice** image / audio soft tokens\ninto their placeholder positions", ACT, key="splice", h=72)
x0, _ = f.tensor(DIM, f"x [T, {DIM}]", key="x")
ROW0 = f.y + 40

# ---- per-layer embeddings (PLE) column ----
PX = 1150
pf = Flow(M, PX, M.nodes[[n["id"] for n in M.nodes].index(tok)]["y"])
p1 = pf.op(f"**per-layer embed** — look up row *id*, × √{PLD}\ntable {PLV:,} × {N * PLD:,} = {N} × {PLD}",
           PLE, P["ple"], note="lookup only", key="ple", h=90)
M.edge(tok, p1, ("right", "left"), "token ids")
pf.tensor(N * PLD, f"token part [T, {N}, {PLD}]", key="plet")
pf.y = M.nodes[[n["id"] for n in M.nodes].index(splice)]["y"] + 140
comb = pf.op(f"**(context + token) × 1/√2**", PLE, key="plec")
p2 = M.node("pleproj", 520, M.nodes[[n["id"] for n in M.nodes].index(splice)]["y"], OP_W, 90,
            f"**per_layer_model_projection** {DIM} → {N}×{PLD}, × 1/√{DIM} · RMSNorm({PLD})  \n"
            f"{fmt(P['pleproj'] + P['plenorm'])}", PLE)
M.edge(splice, p2, ("right", "left"))
M.edge(p2, comb, ("right", "left"), "context part")
pl_t, _ = pf.tensor(N * PLD, f"per-layer inputs [T, {N}, {PLD}]", key="plo")

# ---- layer stack ----
kind = lambda i: (TYPES[i], i >= FIRST_SH)


def lab(i):
    att = f"full · head {GHD}" if TYPES[i] == "full_attention" else f"sliding {WIN}"
    return f"{att} · " + (f"K/V from L{SRC[TYPES[i]]}" if i >= FIRST_SH else "own K/V")


LAST, yend, ROW = draw_stack(M.node, M.edge, N, kind, lab, lambda i: ROW_COLOR[kind(i)], cx, ROW0, x0,
                             1100, next(n["y"] for n in M.nodes if n["id"] == pl_t) + 160, "grey / cyan = own K/V, pale = shares K/V", rep_x=cx + SEG_W / 2 + 60)
node_at = lambda nid: next(n for n in M.nodes if n["id"] == nid)
g1, g2 = node_at("grp-period"), node_at("grp-period-2")
assert ROW[0] == "per0" and ROW[FIRST_SH] == "per0-2", "expected two periods: own K/V, then shared K/V"

# PLE slices feed every layer (box alongside the stack)
ply = node_at(pl_t)["y"] + 60
ple_box = M.node("plebox", 700, ply, 260, max(yend - ply, 420),
                 f"**per-layer input** slice *i* [T, {PLD}] → **layer i**  \n\n"
                 f"Inside each layer, after the FFN: gate {DIM}→{PLD}, GELU, × slice, project {PLD}→{DIM}, "
                 f"RMSNorm, + residual ({fmt(P['plel1'])} per layer).  \n\n"
                 f"So every layer sees its own learned {PLD}-number code for the current token — "
                 f"extra capacity that costs a table lookup, not a matmul.", PLE)
M.edge(pl_t, ple_box, ("bottom", "top"))

# KV-sharing groups, left of the stack, each beside its period
KX = cx - SEG_W / 2 - 60 - 30 - 270
own = M.node("kvown", KX, g1["y"], 270, g1["height"],
             f"**L0–{FIRST_SH - 1} compute K/V**  \nKV cache per token: sliding {kv_s:,} values (2 × {HKV} × {HD}), "
             f"full {kv_f:,} (2 × {HKV} × {GHD}).  \n\nL{SRC['sliding_attention']} (last sliding) and "
             f"L{SRC['full_attention']} (last full) keep their K/V for the shared layers below.", "#94a3b8")
shared = M.node("kvsh", KX, g2["y"], 270, g2["height"],
                f"**L{FIRST_SH}–{N - 1}: KV sharing** — {NSH} layers have no k_proj / v_proj in use and no cache of "
                f"their own. Sliding layers attend over L{SRC['sliding_attention']}'s K/V, full layers "
                f"({', '.join(f'L{i}' for i in FULL if i >= FIRST_SH)}) over L{SRC['full_attention']}'s.  \n\n"
                f"Cache shrinks from {N} to {FIRST_SH} layers' worth. The checkpoint still stores their K/V "
                f"weights ({fmt(P['dead'])}); transformers drops them on load.", KVSH)
M.edge(own, shared, ("bottom", "top"), f"K/V of L{SRC['sliding_attention']}, L{SRC['full_attention']}", KVSH)

# ---- vision branch ----
VX = -900
v = Flow(M, VX, -960, labels_left=True)
v.op("**image / video frame** pixels", ACT, key="img")
v.tensor(3 * PS * PS, f"patches [P, {3 * PS * PS}]  ({PS}×{PS}×3)", key="vp")
v.op(f"**patch embed** {3 * PS * PS} → {VD}\n+ learned x/y position tables", VISION, P["vpatch"], key="vpe", h=72)
v.tensor(VD, f"[P, {VD}]", key="v0")
v.op(f"**ViT ×{VL}** — bidirectional, 2D RoPE\n{V['num_attention_heads']} heads × {V['head_dim']} · GELU MLP {V['intermediate_size']}",
     VISION, P["vit"] - P["vpatch"], key="vit", h=72)
v.tensor(VD, f"[P, {VD}]", key="v1")
v.op(f"**avg-pool** {POOL}×{POOL} patches → 1 token\n× √{VD}", VISION, key="vpool", h=72)
v.tensor(VD, f"[P/{POOL * POOL}, {VD}]  ({CFG['vision_soft_tokens_per_image']} per image by default)", key="v2")
v.op(f"**embed_vision**: RMSNorm (no weight)\n+ linear {VD} → {DIM}", VISION, P["vemb"], key="vemb", h=72)
v.tensor(DIM, f"image tokens [P/{POOL * POOL}, {DIM}]", key="vt")
M.edge(v.last, splice, ("right", "left"), color=VISION)

# ---- audio branch ----
AX = -1700
a = Flow(M, AX, -560, labels_left=True)  # ends below the vision column, so its edge to splice clears it
a.op("**audio** 16 kHz → log-mel", ACT, key="wav")
a.tensor(MEL, f"mel [F, {MEL}]  (10 ms frames)", key="mel")
a.op(f"**2 × conv 3×3 stride 2** ({SUB[0]}, {SUB[1]} ch)\n+ linear {SUB[0] // 4 * SUB[1]} → {AD}",
     AUDIO, P["asub"], key="asub", h=72)
a.tensor(AD, f"[F/4, {AD}]  (40 ms per token)", key="a0")
a.op(f"**Conformer ×{AL}** — FFN · local attn · conv · FFN\n{A['num_attention_heads']} heads, chunk "
     f"{A['attention_chunk_size']}, {A['attention_context_left'] - 1} chunks left",
     AUDIO, P["aud"] - P["asub"] - P["aout"], key="aconf", h=90)
a.tensor(AD, f"[F/4, {AD}]", key="a1")
a.op(f"**output_proj** {AD} → {AOUT}", AUDIO, P["aout"], key="aout")
a.tensor(AOUT, f"[F/4, {AOUT}]", key="a2")
a.op(f"**embed_audio**: RMSNorm (no weight)\n+ linear {AOUT} → {DIM}", AUDIO, P["aemb"], key="aemb", h=72)
a.tensor(DIM, f"audio tokens [F/4, {DIM}]", key="at")
M.edge(a.last, splice, ("right", "left"), color=AUDIO)

# ---- output ----
o = Flow(M, cx, max(yend, a.y + 20 + 850) + 60, last=LAST)
o.tensor(DIM, f"h [T, {DIM}]  after L{N - 1}", key="ho")
o.op("**final RMSNorm**", NORM, key="fn")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm head** {DIM} → {VOCAB:,}\n= embed table, transposed (tied)", HEAD, key="head", h=72)
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op(f"**softcap** {SOFTCAP:.0f}·tanh(z / {SOFTCAP:.0f})", HEAD, key="cap")
o.op("**softmax → sample**", ACT, key="samp")
o.op("**next token id**", ACT, key="next")

# ---- notes ----
NX, NY = -1700, a.y + 20
M.node("note-why", NX, NY, 1150, 330,
       f"**Why 8B stored but \"E4B\".** {fmt(P['all'])} parameters are stored, but {fmt(P['ple'] + P['embed'])} of them are "
       f"*lookup tables*: the per-layer-embedding table ({fmt(P['ple'])}) and the token embedding ({fmt(P['embed'])}). "
       f"A token reads one row of each ({N * PLD:,} + {DIM:,} numbers) — no multiply. "
       f"Google's \"4.5B effective (8B with embeddings)\" = everything minus those two tables "
       f"({fmt(P['all'] - P['ple'] - P['embed'])}).  \n\n"
       f"Per text token the matmuls touch {fmt(P['mlp'] + P['attn'] - P['dead'] + P['plel'] + P['pleproj'] + P['plenorm'] + P['norms'] + P['embed'])}: "
       f"the {N} layers (minus the unused K/V of the shared ones), the PLE projection, and the tied lm head "
       f"(the same {fmt(P['embed'])} matrix, used as a matmul this time). The vision ({fmt(P['vit'] + P['vemb'])}) and "
       f"audio ({fmt(P['aud'] + P['aemb'])}) encoders run once per image / clip.")
M.node("note-tricks", NX, NY + 360, 1150, 300,
       "**On-device tricks visible in the weights**  \n"
       f"- **Per-layer embeddings**: a second, {N * PLD:,}-wide embedding split into {N} slices of {PLD}; "
       "it can sit in slow memory because a token only reads one row.\n"
       f"- **KV sharing**: the last {NSH} layers reuse the K/V of L{SRC['sliding_attention']} / L{SRC['full_attention']} — "
       f"no cache writes and no k/v matmuls there.\n"
       f"- **Sliding window {WIN}** on {n_slide} of {N} layers: their cache never exceeds {WIN} tokens.\n"
       f"- **Full layers** use {GHD}-dim heads with p-RoPE (θ {RP['full_attention']['rope_theta']:,.0f}, only "
       f"{RP['full_attention']['partial_rotary_factor']:.0%} of dims rotate); sliding use {HD}-dim heads, RoPE θ "
       f"{RP['sliding_attention']['rope_theta']:,.0f}.\n"
       "- Unlike Gemma 3n there are **no AltUp / LAuReL tensors**, and every layer has the same "
       f"{FF:,}-wide MLP (use_double_wide_mlp = false).")
M.node("note-src", NX, NY + 690, 1150, 150,
       f"**Sources.** config.json + the safetensors header of google/gemma-4-E4B-it ({len(S):,} tensors, all BF16, one "
       f"file — no index). Dataflow from transformers modeling_gemma4.py. Context {C['max_position_embeddings']:,} "
       f"positions. Logits softcapped at {SOFTCAP:.0f}. Attention scale 1 (q and k are RMS-normed; v is RMS-normed without weight).")


# ---- budget bars ----
def budget():
    """(name, colour, stored, multiplied per text token, why-zero)."""
    rows = [
        (f"FFN ×{N}", FFN, P["mlp"], P["mlp"]),
        (f"attention ×{N}", ATT, P["attn"] - P["dead"], P["attn"] - P["dead"]),
        (f"unused K/V of the {NSH} shared layers", KVSH, P["dead"], 0, "dropped on load — K/V come from earlier layers"),
        ("per-layer embed table", PLE, P["ple"], 0, f"row lookup, {N}×{PLD} numbers"),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, 0, P["embed"], "tied — the embed table"),
        (f"PLE gates + projections ×{N}", "#65a30d", P["plel"], P["plel"]),
        ("PLE context projection", "#4d7c0f", P["pleproj"] + P["plenorm"], P["pleproj"] + P["plenorm"]),
        ("vision encoder", VISION, P["vit"] + P["vemb"], 0, "runs per image"),
        ("audio encoder", AUDIO, P["aud"] + P["aemb"], 0, "runs per audio clip"),
        ("norms + layer scalars", NORM, P["norms"], P["norms"]),
    ]
    assert sum(r[2] for r in rows) == P["all"], (sum(r[2] for r in rows), P["all"])
    return rows


def bar(key, y, title, rows, idx):
    total = sum(r[idx] for r in rows)
    M.node(f"cap-{key}t", -900, y - 46, BAR_W, 40, f"**{title}** — total {fmt(total)}")
    x, legend = -900, []
    # segments too thin to draw go last, so any blank stretch is at the bar's end, not mid-bar
    for k in sorted(range(len(rows)), key=lambda k: BAR_W * rows[k][idx] / total < MIN_SEG):
        r = rows[k]
        share = r[idx] / total
        w = BAR_W * share
        if w >= MIN_SEG:  # integer edges so neighbouring segments meet exactly
            M.node(f"{key}{k}", round(x), y, round(x + w) - round(x), BAR_H, f"**{r[0]}**" if w > 220 else "", r[1])
        x += w
        why = r[4] if len(r) > 4 and not r[idx] else ""
        legend.append(f"{r[0]} **{fmt(r[idx]) if r[idx] else '0'}**" + (f" ({why})" if why else f" ({share:.1%})" if share >= 0.001 else " (<0.1%)"))
    M.node(f"cap-{key}l", -900, y + BAR_H + 6, BAR_W, 64, " · ".join(legend))


BUD = budget()
bar("bp", -1680, "Where the parameters are stored (BF16)", BUD, 2)
bar("bc", -1480, "What one text token is multiplied by — lookup tables, unused K/V and the per-image / per-clip encoders drop out",
    BUD, 3)
M.save("model.canvas")

# ======================= block.canvas =======================
K = Canvas(kt=0.05, clip=520, lbl_w=300)
COL = 520  # spacing of the q / k / v columns
ap = lambda i, name: lp(i, rf"self_attn\.{name}\.weight")
K.node("cap-btitle", -700, -360, 1500, 180,
       f"# Gemma 4 E4B — one decoder layer\n"
       f"Drawn for a sliding layer (L{SL0}); full layers differ only in head size and RoPE (right). Grey bars = tensors "
       f"(0.05 px/channel, long ones clipped), boxes = operations with their weights. Sandwich norms around attention "
       f"and FFN, then the per-layer-embedding block, then × a learned layer scalar.")
K.node("cap-blegend", 900, -360, 1000, 180,
       f"**Sliding** ({n_slide} layers): {HQ} q / {HKV} kv heads × {HD}, window {WIN}, RoPE θ "
       f"{RP['sliding_attention']['rope_theta']:,.0f}.  \n"
       f"**Full** ({N_FULL}: {', '.join(f'L{i}' for i in FULL)}): {HQ} q / {HKV} kv heads × {GHD}, causal over everything, "
       f"p-RoPE θ {RP['full_attention']['rope_theta']:,.0f} on {RP['full_attention']['partial_rotary_factor']:.0%} of dims.  \n"
       f"**KV-shared** (L{FIRST_SH}–{N - 1}): the k / v columns are skipped (orange).")

def residual(src, dst):
    """Skip connection routed through a small label in a lane left of the column, clear of the ops."""
    y = next(n["y"] for n in K.nodes if n["id"] == src)
    a = K.node(K.uid("note-res"), -620, y - 12, 120, 34, "residual")
    K.edge(src, a, ("left", "right"))
    K.edge(a, dst, ("bottom", "left"))


m = Flow(K, 0, -100, lbl_w=440)
xin, _ = m.tensor(DIM, f"x [T, {DIM}]  residual stream in", key="xin")
m.op("**input norm** — RMSNorm", NORM, key="n1")
_, hb = m.tensor(DIM, f"h [T, {DIM}]", key="h")
yb = m.y + 200
q = Flow(K, 0, yb, last=hb, labels_left=True)
q.op(f"**q_proj** {DIM} → {HQ}×{HD}", ATT, ap(SL0, "q_proj"), note=f"full: {HQ}×{GHD}, {fmt(ap(FL0, 'q_proj'))}", key="q", h=72)
q.tensor(HQ * HD, f"q [T, {HQ}, {HD}]", key="qt")
q.op("**q_norm** — RMSNorm per head", NORM, ap(SL0, "q_norm"), key="qn", h=72)
q.op("**RoPE**", ATT, key="qr")
k = Flow(K, COL, yb, last=hb, lbl_w=230)
k.op(f"**k_proj** {DIM} → {HKV}×{HD}", ATT, ap(SL0, "k_proj"), note=f"full: {fmt(ap(FL0, 'k_proj'))}", key="k", h=72)
k.tensor(HKV * HD, f"k [T, {HKV}, {HD}]", key="kt")
k.op("**k_norm** — RMSNorm per head", NORM, ap(SL0, "k_norm"), key="kn", h=72)
k.op("**RoPE** → KV cache", ATT, key="kr")
vv = Flow(K, 2 * COL, yb, last=hb, lbl_w=230)
vv.op(f"**v_proj** {DIM} → {HKV}×{HD}", ATT, ap(SL0, "v_proj"), note=f"full: {fmt(ap(FL0, 'v_proj'))}", key="v", h=72)
vv.tensor(HKV * HD, f"v [T, {HKV}, {HD}]", key="vt")
vv.op("**v_norm** — RMSNorm, no weight", NORM, key="vn", h=72)
vv.op("→ KV cache", ATT, key="vc")
shr = K.node("kvshared", 3 * COL - 60, yb, 340, 3 * OP_H + 2 * 72 + 2 * GAP + 9,
             f"**KV-shared layers (L{FIRST_SH}–{N - 1})** skip the k and v columns entirely: attention reads the "
             f"K/V that L{SRC['sliding_attention']} (sliding) or L{SRC['full_attention']} (full) computed for the same tokens.  \n\n"
             f"Their k/v weights ({fmt(P['dead'])} in total) are in the checkpoint but never loaded.", KVSH)

ya = q.y + 10
att = K.node("att", -OP_W / 2, ya, 2 * COL + OP_W, 90,
             f"**attention** — {HQ} query heads share {HKV} KV heads ({HQ // HKV}:1 GQA), scale 1 (q, k already normed)  \n"
             f"sliding: causal window {WIN} · full: causal over all tokens", ATT)
K.edge(q.last, att)
K.edge(k.last, att)
K.edge(vv.last, att, ("bottom", "right"))
K.edge(shr, att, ("bottom", "right"), "shared K/V", KVSH)
m = Flow(K, 0, ya + 90 + GAP + 20, last=att, lbl_w=440)
m.tensor(HQ * HD, f"[T, {HQ * HD}]  heads concatenated (full: {HQ * GHD})", key="ao")
m.op(f"**o_proj** {HQ * HD} → {DIM}", ATT, ap(SL0, "o_proj"), note=f"full: {fmt(ap(FL0, 'o_proj'))}", key="o", h=72)
m.tensor(DIM, f"[T, {DIM}]", key="ot")
m.op("**post-attention norm** — RMSNorm", NORM, key="n2")
add1 = m.op("**+ residual**", ACT, key="add1")
residual(xin, add1)
xm, _ = m.tensor(DIM, f"x [T, {DIM}]", key="xm")
m.op("**pre-FFN norm** — RMSNorm", NORM, key="n3")
m.tensor(DIM, f"h [T, {DIM}]", key="h2")
mp = lambda name: lp(SL0, rf"mlp\.{name}\.weight")
m.op(f"**gate_proj ‖ up_proj** {DIM} → 2 × {FF}", FFN, mp("gate_proj") + mp("up_proj"), key="gu", h=72)
m.tensor(FF, f"gate, up [T, {FF}]", lanes=2, key="gut")
m.op("**GELU-tanh(gate) · up**", FFN, key="act")
m.tensor(FF, f"[T, {FF}]", key="ff")
m.op(f"**down_proj** {FF} → {DIM}", FFN, mp("down_proj"), key="down", h=72)
m.tensor(DIM, f"[T, {DIM}]", key="dt")
m.op("**post-FFN norm** — RMSNorm", NORM, key="n4")
add2 = m.op("**+ residual**", ACT, key="add2")
residual(xm, add2)
xp, _ = m.tensor(DIM, f"x [T, {DIM}]", key="xp")
pp = lambda name: lp(SL0, rf"{name}\.weight")
m.op(f"**per_layer_input_gate** {DIM} → {PLD}", PLE, pp("per_layer_input_gate"), key="pg", h=72)
m.tensor(PLD, f"[T, {PLD}]", key="pgt")
m.op("**GELU-tanh**", PLE, key="pga")
mul = m.op(f"**× per-layer input** slice *i*", PLE, key="pmul")
sl = K.node("plslice", COL + 160, K.nodes[[n["id"] for n in K.nodes].index(mul)]["y"] - 160, 340, 110,
            f"**per-layer input** [T, {PLD}] for this layer: (token-row lookup + projection of the input embedding) "
            f"× 1/√2 — built once before L0 (Model tab)", PLE)
K.edge(sl, mul, ("bottom", "right"), color=PLE)
m.tensor(PLD, f"[T, {PLD}]", key="pmt")
m.op(f"**per_layer_projection** {PLD} → {DIM}", PLE, pp("per_layer_projection"), key="pp", h=72)
m.tensor(DIM, f"[T, {DIM}]", key="ppt")
m.op("**post_per_layer_input_norm** — RMSNorm", NORM, key="n5")
add3 = m.op("**+ residual**", ACT, key="add3")
residual(xp, add3)
m.op("**× layer_scalar** (1 learned number)", ACT, key="ls")
m.tensor(DIM, f"x [T, {DIM}]  → next layer", key="xout")
K.node("note-full", 2 * COL + OP_W / 2 + 20 + 230 + 60, yb - 220, 520, 190,
       f"**Full layers** (every 6th): head {GHD} instead of {HD}, so q / o are 2× wider and K/V cache "
       f"{kv_f:,} values per token vs {kv_s:,}. **p-RoPE**: only the first "
       f"{int(RP['full_attention']['partial_rotary_factor'] * GHD)} of {GHD} dims rotate (θ "
       f"{RP['full_attention']['rope_theta']:,.0f}); the rest carry no position. The last layer is always full.")
K.node("note-ffn", -1150, ya + 400, 480, 150,
       f"**One sliding layer** = {fmt(sum(lp(SL0, r'.*') for _ in [0]))}: FFN {fmt(P['mlp1'])} "
       f"({P['mlp1'] / lp(SL0, r'.*'):.0%}), attention {fmt(lp(SL0, r'self_attn\..*'))}, per-layer-input block "
       f"{fmt(P['plel1'])}. A full layer = {fmt(lp(FL0, r'.*'))}.")
K.save("block.canvas")

print("ok", [(r[0], fmt(r[2]), fmt(r[3])) for r in BUD])
print("stored", P["all"], fmt(P["all"]), "active", sum(r[3] for r in BUD), fmt(sum(r[3] for r in BUD)))
