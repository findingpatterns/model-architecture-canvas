"""Gemma 4 31B-it (google/gemma-4-31B-it): model-level and layer-level canvases, derived from the checkpoint.

Inputs are the repo's config.json and shapes.json (safetensors headers, see fetch_safetensors_shapes.py), plus
the same two files for the separately shipped drafter google/gemma-4-31B-it-assistant (a single
model.safetensors, header read the same way). Dataflow follows transformers' models/gemma4/modeling_gemma4.py
(Gemma4ForConditionalGeneration), models/gemma4_assistant/modeling_gemma4_assistant.py and
generation/candidate_generator.py (SinglePositionMultiTokenCandidateGenerator), GitHub main.

  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale per canvas)
  - boxes       = operations, all one size; parameter / compute shares are the two bars at the top
  - note-* ids  = annotations (dim, borderless); cap-* ids = borderless captions

usage: python3 gemma_4_31b_it_model_canvas.py <shapes.json> <config.json> <assistant_shapes.json>
       <assistant_config.json> <models/gemma-4-31b-it>
"""
import json
import math
import os
import re
import sys

S = json.load(open(sys.argv[1]))
CFG = json.load(open(sys.argv[2]))
AS = json.load(open(sys.argv[3]))
ACFG = json.load(open(sys.argv[4]))
OUT = sys.argv[5]
C, V, AC = CFG["text_config"], CFG["vision_config"], ACFG["text_config"]

N, DIM, VOCAB, FF = C["num_hidden_layers"], C["hidden_size"], C["vocab_size"], C["intermediate_size"]
HQ, HKV, HD, WIN = C["num_attention_heads"], C["num_key_value_heads"], C["head_dim"], C["sliding_window"]
GKV, GHD = C["num_global_key_value_heads"], C["global_head_dim"]
TYPES, ROPE = C["layer_types"], C["rope_parameters"]
SOFTCAP = C["final_logit_softcapping"]
ROT = int(ROPE["full_attention"]["partial_rotary_factor"] * GHD // 2) * 2  # rotated dims in a full layer (proportional RoPE)
VD, VL, PS, VH, VHD = V["hidden_size"], V["num_hidden_layers"], V["patch_size"], V["num_attention_heads"], V["head_dim"]
POOL, SOFT = V["pooling_kernel_size"], CFG["vision_soft_tokens_per_image"]
PATCH_IN = 3 * PS * PS
AD, AN, AFF, ATYPES = AC["hidden_size"], AC["num_hidden_layers"], AC["intermediate_size"], AC["layer_types"]
assert C["attention_k_eq_v"] and not C["enable_moe_block"] and not C["hidden_size_per_layer_input"]
assert C["num_kv_shared_layers"] == 0 and AC["num_kv_shared_layers"] == AN, "drafter must share all its KV"
assert CFG["audio_config"] is None and C["tie_word_embeddings"]
FULL = [i for i, t in enumerate(TYPES) if t == "full_attention"]
SLID = [i for i, t in enumerate(TYPES) if t == "sliding_attention"]
LAST_S, LAST_F = SLID[-1], FULL[-1]  # these two layers hand their K/V to the drafter (store_full_length_kv)

# colours (JSON Canvas presets "1"–"6" or hex)
EMB, NORM, FFN, HEAD, ACT, ATT = "6", "3", "4", "2", "#64748b", "5"
VISION, DRAFT = "#a78bfa", "#f59e0b"
ROW_COLOR = {"sliding_attention": "#94a3b8", "full_attention": "5"}

OP_W, OP_H, GAP = 300, 56, 40
MIN_SEG = 16  # the viewer draws no box narrower than ~14px


def params_of(pattern, shapes=S):
    rx = re.compile(pattern)
    return sum(math.prod(s) for k, (_, s) in shapes.items() if rx.fullmatch(k))


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.1f}K"


LM = r"model\.language_model\."
L = lambda i: LM + rf"layers\.{i}\."
P = {
    "embed": params_of(LM + r"embed_tokens\.weight"),
    "attn_s": sum(params_of(L(i) + r"self_attn\..*") for i in SLID),
    "attn_f": sum(params_of(L(i) + r"self_attn\..*") for i in FULL),
    "mlp": params_of(LM + r"layers\.\d+\.mlp\..*"),
    "lnorm": params_of(LM + r"layers\.\d+\.(?:\w+_layernorm\.weight|layer_scalar)") + params_of(LM + r"norm\.weight"),
    "layers": params_of(LM + r"layers\..*"),
    "layer_s": params_of(L(SLID[0]) + r".*"),
    "layer_f": params_of(L(FULL[0]) + r".*"),
    "vit": params_of(r"model\.vision_tower\..*"),
    "patch": params_of(r"model\.vision_tower\.patch_embedder\..*"),
    "vstd": params_of(r"model\.vision_tower\.std_(?:bias|scale)"),
    "vemb": params_of(r"model\.embed_vision\..*"),
    "all": params_of(r".*"),
    "draft": params_of(r".*", AS),
    "d_pre": params_of(r"pre_projection\.weight", AS),
    "d_post": params_of(r"post_projection\.weight", AS),
    "d_emb": params_of(r"model\.embed_tokens\.weight", AS),
    "d_layers": params_of(r"model\.layers\..*", AS),
}
assert not any(k.startswith("lm_head") for k in S), "lm head must be tied (no own tensor)"
assert all(params_of(L(i) + r".*") == P["layer_s"] for i in SLID) and all(params_of(L(i) + r".*") == P["layer_f"] for i in FULL)
assert not any(re.fullmatch(L(i) + r"self_attn\.v_proj\..*", k) for i in FULL for k in S), "full layers have no v_proj"
assert not any(".k_proj." in k or ".v_proj." in k for k in AS), "drafter has no K/V projections"


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
        print("wrote", name, len(self.nodes), "nodes", len(self.edges), "edges")


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
period = FULL[1] - FULL[0]
assert all(TYPES[i] == ("full_attention" if i % period == period - 1 else "sliding_attention") for i in range(N))

M.node("cap-title", -900, -1270, 1250, 180,
       "# Gemma 4 31B-it — model\n"
       "Read top → bottom. **Grey bars = tensors**, length ∝ channels per token (0.08 px/channel). "
       "**Boxes = operations**, all the same size; where the parameters live is the pair of bars above. "
       "T = sequence length, P = image patches. *Layer* tab = one decoder layer in detail.")
M.node("cap-legend", 450, -1270, 1250, 180,
       f"**Dense** decoder, {N} layers, pattern [{period - 1} × sliding, full] ×{N // period}.  \n"
       f"**Grey rows** — sliding window {WIN} · GQA {HQ} Q / {HKV} KV × {HD} · RoPE θ {ROPE['sliding_attention']['rope_theta']:,.0f}.  \n"
       f"**Cyan rows** — full causal attention · {HQ} Q / {GKV} KV × {GHD} · **K = V** (no v_proj) · "
       f"proportional RoPE θ {ROPE['full_attention']['rope_theta']:,.0f} on {ROT} of {GHD} dims.  \n"
       f"All layers: q/k RMSNorm, attention scale 1, GeGLU {FF}, 4 sandwich RMSNorms, learned layer scalar.")

# ---- text input ----
cx = ROW_W / 2
f = Flow(M, cx, -960)
f.op("**raw text**  \"The cat sat…\"", ACT, key="raw")
f.op(f"**tokenizer** (vocab {VOCAB:,})", ACT, key="tok")
f.tensor(40 / M.kt, "token ids [T] — one integer per token", key="ids")
f.op(f"**embed** — look up row *id*\n× √{DIM} ≈ {DIM ** 0.5:.1f}", EMB, P["embed"], note="tied to lm head", key="embed", h=90)
x0, _ = f.tensor(DIM, f"x [T, {DIM}]  (+ image tokens spliced in)", key="x")
ROW0 = f.y + 40

# ---- layer stack ----
kind = lambda i: TYPES[i]
lab = lambda i: (f"full · {GKV} KV × {GHD} · K=V" if TYPES[i] == "full_attention" else f"sliding {WIN} · {HKV} KV × {HD}")
LAST, yend, ROW = draw_stack(M.node, M.edge, N, kind, lab, lambda i: ROW_COLOR[TYPES[i]], cx, ROW0, f.last,
                             cx + 900, ROW0 + 40, "grey = sliding, cyan = full", rep_x=cx + SEG_W / 2 + 390)
yend = max(yend, ROW0 + 1400)  # leave room for the notes beside the stack

# ---- notes left of the layer stack (below the vision branch) ----
kv_s, kv_f = 2 * HKV * HD, 2 * GKV * GHD
M.node("note-pattern", -560, ROW0 + 760, 480, 290,
       f"**Layer pattern** — {len(SLID)} sliding + {len(FULL)} full layers (L{FULL[0]}, L{FULL[1]}, … L{FULL[-1]}; "
       "the last layer is always full).\n"
       f"Sliding layers see the last {WIN} tokens; full layers see the whole context with fewer, wider KV heads.\n"
       f"KV cache per token per layer: sliding 2 × {HKV} × {HD} = {kv_s:,} values (kept for {WIN} tokens at most); "
       f"full 2 × {GKV} × {GHD} = {kv_f:,} — K and V come from the same k_proj, V just skips k_norm's weight and RoPE.")
M.node("note-layer", -560, ROW0 + 1090, 480, 200,
       f"**Layer sizes** — sliding {fmt(P['layer_s'])} (attention {fmt(P['attn_s'] // len(SLID))}), "
       f"full {fmt(P['layer_f'])} (attention {fmt(P['attn_f'] // len(FULL))}); GeGLU MLP {fmt(P['mlp'] // N)} in every layer.\n"
       f"Logits: tied lm head, then softcap {SOFTCAP:.0f}·tanh(z / {SOFTCAP:.0f}).\n"
       "Image tokens attend to each other bidirectionally (use_bidirectional_attention = \"vision\"); text stays causal.")

# ---- vision branch (SigLIP-style ViT), spliced into x ----
NP = SOFT * POOL * POOL
VIDEO_SOFT = 70  # processor_config.json video_processor.max_soft_tokens
v = Flow(M, -480, -960, labels_left=True)
v.op("**image** pixels (video: sampled frames)", ACT, key="img")
v.tensor(PATCH_IN, f"patches [P, {PATCH_IN}]  ({PS}×{PS}×3), P = {NP:,} by default", key="patch")
v.op(f"**patch embed** {PATCH_IN} → {VD}\n+ learned x / y position tables 2 × {V['position_embedding_size']:,}",
     VISION, P["patch"], key="pe", h=90)
v.tensor(VD, f"[P, {VD}]", key="vf0")
v.op(f"**ViT ×{VL}** — bidirectional · {VH} heads × {VHD}\n2D RoPE θ {V['rope_parameters']['rope_theta']:.0f} · "
     f"q/k norm · GeGLU {V['intermediate_size']}", VISION, P["vit"] - P["patch"] - P["vstd"], key="vit", h=90)
v.tensor(VD, f"[P, {VD}]", key="vf1")
v.op(f"**pool** {POOL}×{POOL} patches → 1 token (average)\n× √{VD}, standardise (std_bias / std_scale)",
     VISION, P["vstd"], key="pool", h=90)
v.tensor(VD, f"[P/{POOL * POOL}, {VD}]  ({SOFT} per image, {VIDEO_SOFT} per video frame by default)", key="vf2")
v.op(f"**embed_vision** — RMSNorm (no weight)\n+ linear {VD} → {DIM}", VISION, P["vemb"], key="vpr", h=90)
v.tensor(DIM, f"image tokens [P/{POOL * POOL}, {DIM}]", key="vt")
M.edge(v.last, x0, ("right", "left"), "spliced into x")

# ---- output ----
o = Flow(M, cx, yend + 60, last=LAST)
o.tensor(DIM, f"h [T, {DIM}]  after L{N - 1}", key="ho")
o.op("**final RMSNorm**", NORM, key="fn")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm head** {DIM} → {VOCAB:,}\n= the embed table, transposed (tied)", HEAD, key="head", h=72)
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op(f"**softcap** {SOFTCAP:.0f}·tanh(z / {SOFTCAP:.0f})", HEAD, key="cap")
o.op("**softmax → sample**", ACT, key="samp")
next_id = o.op("**next token id(s)**", ACT, key="next")

# ---- MTP drafter (separate repo google/gemma-4-31B-it-assistant): reads the 31B's cache + last hidden ----
NDRAFT = 6  # num_assistant_tokens in the assistant's generation_config.json
tap_x = cx + SEG_W / 2 + 30
row_y = lambda i: next(n["y"] for n in M.nodes if n["id"] == ROW[i])
tap_s = M.node("note-taps", tap_x, row_y(LAST_S), 330, ROW_H, f"↳ L{LAST_S} K, V (last sliding) → drafter")
tap_f = M.node("note-tapf", tap_x, row_y(LAST_F), 330, ROW_H, f"↳ L{LAST_F} K, V (last full) → drafter")
d = Flow(M, 2000, yend - 260)
cat, _ = d.tensor(2 * DIM, f"[1, {2 * DIM}] = embed(last token) ‖ last hidden h of the 31B", key="dcat")
d.op(f"**pre_projection** {2 * DIM} → {AD}", DRAFT, P["d_pre"], key="dpre")
d.tensor(AD, f"[1, {AD}]", key="dx")
blk = d.op(f"**{AN} drafter layers** [{', '.join('S' if t.startswith('sliding') else 'F' for t in ATYPES)}]\n"
           f"q-only attention onto the 31B's K, V · GeGLU {AFF}", DRAFT, P["d_layers"], key="dblk", h=90)
M.edge(tap_s, blk, ("right", "left"), color=DRAFT)
M.edge(tap_f, blk, ("right", "left"), color=DRAFT)
d.op("**final RMSNorm**", DRAFT, key="dnorm")
dh_y = d.y
dh, _ = d.tensor(AD, f"h' [1, {AD}]", key="dh")
d.op(f"**own lm head** {AD} → {VOCAB:,}\n(tied to its own embed table)", DRAFT, P["d_emb"], key="dhead", h=90)
d.tensor(40 / M.kt, f"draft ids — {NDRAFT} per round, one per step", key="dids")
ver = d.op(f"**verify**: main model scores all {NDRAFT}\ndrafts in one forward pass", ACT, key="dver", h=72)
M.edge(ver, next_id, ("left", "right"), "keeps the drafts it agrees with", DRAFT)
M.node("dpost", d.cx - OP_W / 2 - 60 - OP_W, dh_y - 40, OP_W, 90,
       f"**post_projection** {AD} → {DIM}\nh' becomes the next step's “last hidden”\n{fmt(P['d_post'])}", DRAFT)
M.edge(dh, "dpost", ("left", "right"), color=DRAFT)
hid = M.node("note-taph", cx + DIM * M.kt / 2 + 20 + 440 + 40, yend + 60 - 10, 360, 34, "↳ last hidden h + embed(last token) → drafter")
M.edge(hid, cat, ("right", "left"), color=DRAFT)
M.node("note-draft", d.cx - OP_W / 2, d.y - 10, 700, 200,
       f"**MTP drafter, shipped separately** (google/gemma-4-31B-it-assistant, {fmt(P['draft'])}, {len(AS)} tensors; "
       "**not counted** in the bars above). It has no K/V projections: every drafter layer attends with its own queries to "
       f"the 31B's cached keys/values (sliding layers → L{LAST_S}, full layer → L{LAST_F}) at one fixed position, so it "
       f"needs no prefill. Each step feeds the token it just drafted (embedded by the 31B's table) plus its projected "
       f"hidden state back in. use_ordered_embeddings = false, so the centroid-masked head is off and no centroid "
       "tensors ship.")

# ---- sources ----
M.node("note-src", -1100, ROW0 + 1330, 620, 260,
       f"**Sources.** config.json + safetensors headers of google/gemma-4-31B-it ({len(S)} tensors, all BF16) and of the "
       f"assistant repo. Dataflow from transformers models/gemma4 and models/gemma4_assistant (GitHub main). "
       f"No audio tower in this size (audio_config = null). Context {C['max_position_embeddings']:,} positions. "
       "RMSNorms here scale by w (not 1 + w). The per-layer scalar is a stored 1-element tensor; its value is not shown.")


# ---- budget bars ----
def budget():
    """(name, colour, stored, multiplied per text token, why-zero)."""
    vision = P["vit"] + P["vemb"]
    rows = [
        (f"GeGLU MLP ×{N}", FFN, P["mlp"], P["mlp"]),
        (f"attention, {len(SLID)} sliding", "#94a3b8", P["attn_s"], P["attn_s"]),
        (f"attention, {len(FULL)} full", ATT, P["attn_f"], P["attn_f"]),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, 0, P["embed"], "tied: same tensor as embed"),
        ("vision tower + embed_vision", VISION, vision, 0, "runs per image"),
        ("norms + layer scalars", NORM, P["lnorm"], P["lnorm"]),
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
bar("bp", -1700, "Where the parameters are stored (main checkpoint, BF16; the separate drafter is not included)", BUD, 2)
bar("bc", -1500, "What one text token is multiplied by — dense, so every layer weight plus the tied lm head; "
    "the lookup and the per-image vision path drop out", BUD, 3)
M.save("model.canvas")

# ======================= block.canvas =======================
K = Canvas(kt=0.05, clip=400, lbl_w=240)
COL = 620  # spacing of the q / k / v columns
FX = 2400  # left edge of the full-attention variant panel
K.node("cap-btitle", -700, -330, 1500, 150,
       f"# Gemma 4 31B-it — one decoder layer (sliding, {len(SLID)} of {N}: {fmt(P['layer_s'])})\n"
       "Grey bars = tensors (0.05 px/channel), boxes = operations with their weights. Sandwich norms: each sub-block is "
       "wrapped by an RMSNorm before *and* after, then added to the residual; the layer output is scaled by a learned scalar.")
K.node("cap-blegend", 900, -330, 1300, 150,
       f"Sliding layers: causal window {WIN}, GQA {HQ}/{HKV} × {HD}, RoPE θ {ROPE['sliding_attention']['rope_theta']:,.0f} on all {HD} dims.  \n"
       f"The {len(FULL)} full layers (every {period}th, {fmt(P['layer_f'])}) differ only in attention — see the panel on the right. "
       "Norms, residuals and the MLP are identical.")

m = Flow(K, 0, 0, lbl_w=440)
xin, _ = m.tensor(DIM, f"x [T, {DIM}]  residual stream in", key="xin")
m.op("**input norm** — RMSNorm", NORM, key="n1")
_, hb = m.tensor(DIM, f"h [T, {DIM}]", key="h")
yb = m.y + 240  # room for the fan-out to the three projections
ap = lambda name, i=SLID[0]: params_of(L(i) + rf"self_attn\.{name}\.weight")
q = Flow(K, 0, yb, last=hb)
q.op(f"**q_proj** {DIM} → {HQ * HD}", ATT, ap("q_proj"), key="q")
q.tensor(HQ * HD, f"q [T, {HQ}, {HD}]", key="qt")
q.op("**q_norm** — RMSNorm per head", NORM, ap("q_norm"), key="qn")
q.op(f"**RoPE** θ {ROPE['sliding_attention']['rope_theta']:,.0f}", ATT, key="qr")
k = Flow(K, COL, yb, last=hb)
k.op(f"**k_proj** {DIM} → {HKV * HD}", ATT, ap("k_proj"), key="k")
k.tensor(HKV * HD, f"k [T, {HKV}, {HD}]", key="kt")
k.op("**k_norm** — RMSNorm per head", NORM, ap("k_norm"), key="kn")
k.op(f"**RoPE** θ {ROPE['sliding_attention']['rope_theta']:,.0f}", ATT, key="kr")
vv = Flow(K, 2 * COL, yb, last=hb)
vv.op(f"**v_proj** {DIM} → {HKV * HD}", ATT, ap("v_proj"), key="v")
vv.tensor(HKV * HD, f"v [T, {HKV}, {HD}]", key="vt")
vv.op("**v_norm** — RMSNorm, no weight", NORM, key="vn")

ya = q.y
att = K.node("att", -OP_W / 2, ya, COL + OP_W, 90,
             f"**attention** — {HQ} query heads share {HKV} KV heads ({HQ // HKV}:1 GQA), head {HD}, "
             f"scale 1 (q/k norms set the range)\ncausal window: last {WIN} tokens", ATT)
K.edge(q.last, att)
K.edge(k.last, att)
K.edge(vv.last, att, ("bottom", "right"))
m = Flow(K, 0, ya + 90 + GAP + 20, last=att, lbl_w=440)
m.tensor(HQ * HD, f"[T, {HQ * HD}]  heads concatenated", key="ao")
m.op(f"**o_proj** {HQ * HD} → {DIM}", ATT, ap("o_proj"), key="o")
m.tensor(DIM, f"[T, {DIM}]", key="ot")
m.op("**post-attention norm** — RMSNorm", NORM, key="n2")
add1 = m.op("**+ residual**", ACT, key="add1")
K.edge(xin, add1, ("left", "left"), "residual")
xm, _ = m.tensor(DIM, f"x [T, {DIM}]", key="xm")
m.op("**pre-FFN norm** — RMSNorm", NORM, key="n3")
m.tensor(DIM, f"h [T, {DIM}]", key="h2")
mp = lambda name: params_of(L(0) + rf"mlp\.{name}\.weight")
m.op(f"**gate_proj ‖ up_proj** {DIM} → 2 × {FF}", FFN, mp("gate_proj") + mp("up_proj"), key="gu")
m.tensor(FF, f"gate, up [T, {FF}]", lanes=2, key="gut")
m.op("**GELU-tanh(gate) · up**", FFN, key="geglu")
m.tensor(FF, f"[T, {FF}]", key="ff")
m.op(f"**down_proj** {FF} → {DIM}", FFN, mp("down_proj"), key="down")
m.tensor(DIM, f"[T, {DIM}]", key="dt")
m.op("**post-FFN norm** — RMSNorm", NORM, key="n4")
add2 = m.op("**+ residual**", ACT, key="add2")
K.edge(xm, add2, ("left", "left"), "residual")
m.tensor(DIM, f"x [T, {DIM}]", key="xs")
m.op("**× layer_scalar** (1 learned number)", ACT, key="ls")
m.tensor(DIM, f"x [T, {DIM}]  → next layer", key="xout")
K.node("note-qk", -1100, yb, 420, 190,
       f"**q/k norm** — RMSNorm with a learned {HD}-wide weight on every head; v gets a weightless RMSNorm. "
       f"Because q and k are normalised, the attention scale is 1 instead of 1/√{HD}. "
       f"KV cache per token: {kv_s:,} values (K+V, {HKV} heads × {HD}), kept for the last {WIN} tokens.")
K.node("note-norm", -1100, ya + 300, 420, 120,
       "All four layer norms are RMSNorm × w (plain weight, not 1 + w). "
       f"MLP is GeGLU: gelu_pytorch_tanh(gate) × up, {fmt(P['mlp'] // N)} per layer.")

# ---- full-attention variant (L5, L11, … L59) ----
fa = lambda name: ap(name, FULL[0])
K.node("cap-full", FX - 200, yb - 260, 1300, 110,
       f"**Full-attention layers** (L{', L'.join(str(i) for i in FULL[:3])} … L{FULL[-1]}) — attention only; "
       f"the rest of the layer is the same. {GKV} wide KV heads, and **K = V**: there is no v_proj.")
fq = Flow(K, FX, yb, labels_left=True)
fq.op(f"**q_proj** {DIM} → {HQ * GHD}", ATT, fa("q_proj"), key="fq")
fq.tensor(HQ * GHD, f"q [T, {HQ}, {GHD}]", key="fqt")
fq.op("**q_norm** — RMSNorm per head", NORM, fa("q_norm"), key="fqn")
fq.op(f"**RoPE** on {ROT} of {GHD} dims\nθ {ROPE['full_attention']['rope_theta']:,.0f}", ATT, key="fqr", h=72)
fk = Flow(K, FX + COL, yb, labels_left=True)
fk.op(f"**k_proj** {DIM} → {GKV * GHD}", ATT, fa("k_proj"), key="fk")
_, kraw = fk.tensor(GKV * GHD, f"k [T, {GKV}, {GHD}]", key="fkt")
fk.op("**k_norm** — RMSNorm per head", NORM, fa("k_norm"), key="fkn")
fk.op(f"**RoPE** on {ROT} of {GHD} dims", ATT, key="fkr", h=72)
fv = Flow(K, FX + 2 * COL, yb + OP_H + GAP + 9 + GAP)
fvn = fv.op("**v = same k_proj output**\nv_norm (no weight), no RoPE", NORM, key="fvn", h=72)
K.edge(kraw, fvn, ("right", "left"))
fya = max(fq.y, fk.y)
fatt = K.node("fatt", FX - OP_W / 2, fya, COL + OP_W, 90,
              f"**attention** — {HQ} query heads share {GKV} KV heads ({HQ // GKV}:1 GQA), head {GHD}, scale 1\n"
              "causal over the whole context", ATT)
K.edge(fq.last, fatt)
K.edge(fk.last, fatt)
K.edge(fv.last, fatt, ("bottom", "right"))
fo = Flow(K, FX, fya + 90 + GAP + 20, last=fatt, lbl_w=440)
fo.tensor(HQ * GHD, f"[T, {HQ * GHD}]  heads concatenated", key="fao")
fo.op(f"**o_proj** {HQ * GHD} → {DIM}", ATT, fa("o_proj"), key="fo")
fo.tensor(DIM, f"[T, {DIM}]  → post-attention norm, as on the left", key="fot")
K.node("note-full", FX - OP_W / 2, fo.y + 10, 900, 180,
       f"**Proportional RoPE**: frequencies are laid out as if all {GHD} dims rotated, but only the first {ROT // 2} "
       f"frequency pairs are non-zero — {ROT} dims carry position, the other {GHD - ROT} are position-free.  \n"
       f"KV cache per token: 2 × {GKV} × {GHD} = {kv_f:,} values, for every past token. "
       f"Attention weights: {fmt(P['attn_f'] // len(FULL))} vs {fmt(P['attn_s'] // len(SLID))} in a sliding layer.")
K.save("block.canvas")

act = sum(r[3] for r in BUD)
print("budget", [(r[0], fmt(r[2]), fmt(r[3])) for r in BUD])
print("stored", P["all"], fmt(P["all"]), "active", act, fmt(act), "drafter", P["draft"], fmt(P["draft"]))
