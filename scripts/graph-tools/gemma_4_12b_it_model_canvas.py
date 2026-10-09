"""Gemma 4 12B Unified (google/gemma-4-12B-it): whole-model canvas + one-decoder-layer canvas.

"Unified" = encoder-free multimodal: there is no vision tower and no audio tower. Raw 48×48 pixel patches
and raw 40 ms waveform frames are each pushed through a couple of norms + ONE linear layer straight into
the decoder's embedding space, then spliced into the token sequence like word embeddings.

Read top → bottom:
  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale per canvas)
  - boxes       = operations, all the same size; parameter / compute shares are the two bars at the top
  - layer rows  = repeating layers drawn once with ×N (barcode = every layer), coloured by layer type from config.json (sliding vs full attention)
  - note-* ids  = annotations (dim, borderless); cap-* ids = borderless captions
Every number comes from config.json + the safetensors header (shapes.json) of the HF repo, the MTP drafter's
config/shapes (google/gemma-4-12B-it-assistant) and the transformers `gemma4_unified`, `gemma4`,
`gemma4_unified_assistant` modeling code (+ generation/candidate_generator.py for the drafter loop).

usage: python3 gemma_4_12b_it_model_canvas.py <shapes.json> <config.json> <asst_shapes.json> <asst_config.json> <models/gemma-4-12b-it>
"""
import json
import math
import os
import re
import sys

S = json.load(open(sys.argv[1]))
CFG = json.load(open(sys.argv[2]))
AS = json.load(open(sys.argv[3]))
AC = json.load(open(sys.argv[4]))
OUT = sys.argv[5]
C, V, AU = CFG["text_config"], CFG["vision_config"], CFG["audio_config"]
ACT_T = AC["text_config"]

N, DIM, VOCAB, FF = C["num_hidden_layers"], C["hidden_size"], C["vocab_size"], C["intermediate_size"]
HQ, HKV, HD = C["num_attention_heads"], C["num_key_value_heads"], C["head_dim"]
GKV, GHD, WIN = C["num_global_key_value_heads"], C["global_head_dim"], C["sliding_window"]
TYPES = C["layer_types"]
RS, RF = C["rope_parameters"]["sliding_attention"], C["rope_parameters"]["full_attention"]
ROT = int(GHD * RF["partial_rotary_factor"])
SOFTCAP = C["final_logit_softcapping"]
PS, POOL, MPS = V["patch_size"], V["pooling_kernel_size"], V["model_patch_size"]
PATCH_IN = MPS * MPS * 3
MM, POSN = V["mm_embed_dim"], V["mm_posemb_size"]
AUD = AU["audio_embed_dim"]  # raw samples per audio token
FULL = [i for i, t in enumerate(TYPES) if t == "full_attention"]
SLID = [i for i, t in enumerate(TYPES) if t == "sliding_attention"]
LAST_S, LAST_F = SLID[-1], FULL[-1]
assert C["attention_k_eq_v"] and C["tie_word_embeddings"] and not C["enable_moe_block"]
assert PATCH_IN == S["model.vision_embedder.patch_dense.weight"][1][1], "patch width ≠ 48·48·3"
assert AUD == S["model.embed_audio.embedding_projection.weight"][1][1]

# colours (JSON Canvas presets "1"–"6" or hex)
EMB, NORM, FFN, HEAD, ACT, ATT = "6", "3", "4", "2", "#64748b", "5"
VISION, AUDIO, DRAFT = "#a78bfa", "#f472b6", "#f59e0b"
ROW_COLOR = {"sliding_attention": "#94a3b8", "full_attention": "5"}

OP_W, OP_H, GAP = 300, 56, 40
MIN_SEG = 16  # the viewer draws no box narrower than ~14px


def params_of(pattern, shapes=S):
    rx = re.compile(pattern)
    return sum(math.prod(s) for k, (_, s) in shapes.items() if rx.fullmatch(k))


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


def layers_re(idx):
    return r"(?:" + "|".join(str(i) for i in idx) + r")"


LM = r"model\.language_model\."
NORMS = r"(?:input_layernorm|post_attention_layernorm|pre_feedforward_layernorm|post_feedforward_layernorm|layer_scalar|self_attn\.[qk]_norm)"
P = {
    "embed": params_of(LM + r"embed_tokens\.weight"),
    "attn_s": params_of(LM + rf"layers\.{layers_re(SLID)}\.self_attn\.[qkvo]_proj\.weight"),
    "attn_f": params_of(LM + rf"layers\.{layers_re(FULL)}\.self_attn\.[qkvo]_proj\.weight"),
    "mlp": params_of(LM + r"layers\.\d+\.mlp\..*"),
    "norms": params_of(LM + rf"layers\.\d+\.{NORMS}(?:\.weight)?") + params_of(LM + r"norm\.weight"),
    "vis": params_of(r"model\.vision_embedder\..*") + params_of(r"model\.embed_vision\..*"),
    "aud": params_of(r"model\.embed_audio\..*"),
    "all": params_of(r".*"),
    "layer_s": params_of(LM + rf"layers\.{SLID[0]}\..*"),
    "layer_f": params_of(LM + rf"layers\.{FULL[0]}\..*"),
    "mlp1": params_of(LM + r"layers\.0\.mlp\..*"),
    "patch": params_of(r"model\.vision_embedder\.patch_.*"),
    "pos": params_of(r"model\.vision_embedder\.pos_.*"),
    "vproj": params_of(r"model\.embed_vision\.embedding_projection\.weight"),
    "draft": params_of(r".*", AS),
    "d_emb": params_of(r"model\.embed_tokens\.weight", AS),
    "d_pre": params_of(r"pre_projection\.weight", AS),
    "d_post": params_of(r"post_projection\.weight", AS),
}
P["layers"] = params_of(LM + r"layers\..*")
assert all(params_of(LM + rf"layers\.{i}\..*") == P["layer_s"] for i in SLID), "sliding layers differ"
assert all(params_of(LM + rf"layers\.{i}\..*") == P["layer_f"] for i in FULL), "full layers differ"
# full layers store no v_proj: K and V come from the same k_proj
assert not any(f"layers.{i}.self_attn.v_proj.weight" in k for i in FULL for k in S)
assert S[f"model.language_model.layers.{FULL[0]}.self_attn.k_proj.weight"][1] == [GKV * GHD, DIM]
P["d_layers"] = P["draft"] - P["d_emb"] - P["d_pre"] - P["d_post"]


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
pattern = TYPES[: FULL[0] + 1]
assert TYPES == pattern * (N // len(pattern)), "layer pattern is not a clean repeat"

M.node("cap-title", -900, -1300, 1250, 200,
       "# Gemma 4 12B Unified — model\n"
       "Read top → bottom. **Grey bars = tensors**, length ∝ channels per token (0.08 px/channel). "
       "**Boxes = operations**, all one size; where parameters and compute live is the pair of bars above. "
       "T = sequence length, P = image patches, A = audio frames. *Layer* tab = one decoder layer in detail.")
M.node("cap-legend", 450, -1300, 1250, 200,
       f"**Dense** decoder, {N} layers, pattern [{len(pattern) - 1} sliding, 1 full] ×{N // len(pattern)}.  \n"
       f"**Grey rows** — sliding window {WIN}, GQA {HQ} q / {HKV} kv × {HD}, RoPE θ {RS['rope_theta']:,.0f} · "
       f"**Cyan rows** — full causal attention, {HQ} q / **{GKV} kv** × {GHD} with **K = V** (one projection), "
       f"p-RoPE θ {RF['rope_theta']:,.0f} on {ROT} of {GHD} dims.  \n"
       f"**Encoder-free**: image patches (violet) and audio frames (pink) enter through one linear layer each — no ViT, no conformer.")

# ---- text input ----
cx = ROW_W / 2
f = Flow(M, cx, -1040)
f.op("**raw text**  \"The cat sat…\"\nwith image / audio placeholders", ACT, key="raw", h=72)
f.op(f"**tokenizer** (vocab {VOCAB:,})", ACT, key="tok")
f.tensor(40 / M.kt, "token ids [T] — one integer per token", key="ids")
f.op(f"**embed** — look up row *id*, × √{DIM}", EMB, P["embed"], note=f"table {VOCAB:,} × {DIM}", key="embed", h=72)
f.tensor(DIM, f"x [T, {DIM}]  (placeholder rows = pad)", key="x0")
splice = f.op("**splice** — overwrite every image / video / audio placeholder row with a soft token",
              ACT, key="splice", h=90)
f.tensor(DIM, f"x [T, {DIM}]", key="x")
ROW0 = f.y + 40

# ---- vision branch: encoder-free patch embedder (images and video frames) ----
v = Flow(M, -480, -1040, labels_left=True)
v.op(f"**image** (or video frame) — resized so it\nfits ≤ {CFG.get('image_seq_length', 280)} tokens (70 per video frame)", ACT, key="img", h=72)
v.op(f"**patchify** {PS}px, then **merge** {POOL}×{POOL}\n→ one {MPS}×{MPS} RGB patch per token", VISION, key="pm", h=72)
v.tensor(PATCH_IN, f"raw pixels [P, {PATCH_IN}]  ({MPS}×{MPS}×3)", key="vp")
v.op(f"**LayerNorm → Linear** {PATCH_IN} → {MM}\n**→ LayerNorm**", VISION, P["patch"], key="pd", h=90)
v.tensor(MM, f"[P, {MM}]", key="vf0")
v.op(f"**+ 2D pos**: row x + row y of a\n{POSN}×2×{MM} table → LayerNorm", VISION, P["pos"], key="pos", h=90)
v.tensor(MM, f"[P, {MM}]", key="vf1")
v.op(f"**RMSNorm** (no weight) **→ Linear**\n{MM} → {DIM}", VISION, P["vproj"], key="vpr", h=90)
v.tensor(DIM, f"image tokens [P, {DIM}]", key="vt")
M.edge(v.last, splice, ("right", "left"), color=VISION)

# ---- audio branch: raw waveform frames, one linear layer ----
a = Flow(M, 1250, -1040)
a.op(f"**audio** waveform, {CFG.get('audio_ms_per_token', 40)} ms per token\n(16 kHz, ≤ 30 s)", ACT, key="aud", h=72)
a.op(f"**chunk** into frames of {AUD} samples\nno mel spectrogram, no conv", AUDIO, key="ach", h=72)
a.tensor(AUD, f"raw samples [A, {AUD}]", key="af")
a.op(f"**RMSNorm** (no weight) **→ Linear**\n{AUD} → {DIM}", AUDIO, P["aud"], key="apr", h=90)
a.tensor(DIM, f"audio tokens [A, {DIM}]", key="at")
M.edge(a.last, splice, ("left", "right"), color=AUDIO)

# ---- layer stack ----
lab = lambda i: (f"full · MQA {GHD} · K=V" if TYPES[i] == "full_attention" else f"sliding {WIN} · GQA {HD}")
LAST, yend, ROW = draw_stack(M.node, M.edge, N, lambda i: TYPES[i], lab, lambda i: ROW_COLOR[TYPES[i]], cx, ROW0, f.last,
                             cx + 900, ROW0 + 40, "grey = sliding, cyan = full", rep_x=cx + SEG_W / 2 + 390)
yend = max(yend, v.y + 20 + 1080)  # leave room for the notes beside the stack
row_y = lambda i: next(n["y"] for n in M.nodes if n["id"] == ROW[i])

# ---- notes left of the layer stack ----
NX, NY = -720, v.y + 20  # below the vision branch, clear of the period's loop-back edge
M.node("note-unified", NX, NY, 540, 300,
       "**What \"unified\" means.** Other Gemma 4 models run images through a ViT (~550M on 31B) and audio "
       "through a conformer (~300M on E2B/E4B) before the LLM. This one has **no encoder at all**: a 48×48 patch "
       f"or a 40 ms frame is normalised and hit with one matrix ({fmt(P['vis'])} vision, {fmt(P['aud'])} audio in "
       "total), so all perception happens inside the same 48 decoder layers that read text — and one fine-tune "
       "trains everything.")
M.node("note-mask", NX, NY + 330, 540, 200,
       "**Mask.** Text is causal. Tokens of one image / video frame see each other **bidirectionally**, but only in "
       f"the {len(SLID)} sliding layers; the {len(FULL)} full layers stay causal everywhere "
       "(use_bidirectional_attention = \"vision\"). Audio tokens are causal.")
kv_s, kv_f = 2 * HKV * HD, 2 * GKV * GHD
M.node("note-pattern", NX, NY + 560, 540, 260,
       f"**Layer pattern** — {len(SLID)} sliding + {len(FULL)} full (every {len(pattern)}th, and the last).\n"
       f"KV cache per token: sliding {kv_s:,} values but only for the last {WIN} tokens; full {kv_f:,} values "
       f"(one {GHD}-wide head, K and V from the same projection) for the whole context — so a full layer costs "
       f"{kv_s // kv_f}× less cache per token than a sliding one would.")
M.node("note-layer", NX, NY + 850, 540, 200,
       f"**One layer** — sliding {fmt(P['layer_s'])}, full {fmt(P['layer_f'])}; the GELU-tanh MLP "
       f"{DIM}→{FF}→{DIM} is {fmt(P['mlp1'])} of it. Sandwich RMSNorms (before *and* after each sub-block), "
       "qk-norm instead of 1/√d scaling, and a learned per-layer output scalar.")

# ---- output ----
o = Flow(M, cx, yend + 60, last=LAST)
o.tensor(DIM, f"h [T, {DIM}]  after L{N - 1}", key="ho")
o.op("**final RMSNorm**", NORM, key="fn")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm head** {DIM} → {VOCAB:,}\n= embed table, transposed (tied)", HEAD, key="head", h=72)
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op(f"**softcap** {SOFTCAP:.0f}·tanh(z / {SOFTCAP:.0f})", HEAD, key="cap")
o.op("**softmax → sample**", ACT, key="samp")
next_id = o.op("**next token id**", ACT, key="next")

# ---- MTP drafter (separate repo): Q-only layers that read the target's KV cache ----
DL = ACT_T["num_hidden_layers"]
d_types = ACT_T["layer_types"]
assert ACT_T["num_kv_shared_layers"] == DL and AC["backbone_hidden_size"] == DIM
assert not any("k_proj" in k or "v_proj" in k for k in AS), "drafter should hold no K/V projections"
DX = 1550
d = Flow(M, DX, o.y - OP_H - GAP - 620)  # so `verify` lands level with `next token id`
cat, _ = d.tensor(2 * DIM, f"embed(last token) ‖ h  [1, {2 * DIM}]", key="dcat")
for li, txt in ((LAST_S, f"↳ L{LAST_S} K,V (last sliding) → drafter"), (LAST_F, f"↳ L{LAST_F} K,V + last hidden h → drafter")):
    tap = M.node(f"note-tap{li}", cx + SEG_W / 2 + 30, row_y(li) + 11, 330, ROW_H, txt)
    M.edge(tap, cat, ("right", "top"), color=DRAFT)
d.op(f"**pre_projection** {2 * DIM} → {ACT_T['hidden_size']}", DRAFT, P["d_pre"], key="dpre")
d.tensor(ACT_T["hidden_size"], f"[1, {ACT_T['hidden_size']}]", key="dh")
d.op(f"**{DL} drafter layers** ({d_types.count('sliding_attention')} sliding + {d_types.count('full_attention')} full)\n"
     f"**Q-only** attention onto the target's K,V · GELU MLP {ACT_T['intermediate_size']}",
     DRAFT, P["d_layers"], note="incl. norms", key="dblk", h=90)
d.op(f"**own lm head** {ACT_T['hidden_size']} → {VOCAB:,} (tied)", DRAFT, P["d_emb"], key="dhead", h=72)
d.tensor(40 / M.kt, "draft token id", key="dids")
d.op(f"**post_projection** {ACT_T['hidden_size']} → {DIM}\n= next h; loop for the next draft", DRAFT, P["d_post"], key="dpost", h=90)
ver = d.op("**verify**: main model checks all\ndrafts in one forward pass", ACT, key="dver", h=72)
M.edge(ver, next_id, ("left", "right"), "accepted drafts = extra tokens", DRAFT)
M.node("note-mtp", DX - OP_W / 2, d.y - 10, 640, 170,
       f"MTP drafter, shipped as a separate repo (google/gemma-4-12B-it-assistant, {fmt(P['draft'])}). It has **no K/V "
       f"weights**: all {DL} layers attend to the target's cached K,V of L{LAST_S} / L{LAST_F}. Input = target embed of "
       "the last token ‖ target hidden h; each step drafts one token and feeds post_projection(out) back as h. "
       "Not counted in the budget bars.")


# ---- budget bars ----
def budget():
    """(name, colour, stored, multiplied per text token, why-zero)."""
    rows = [
        (f"GELU MLP ×{N}", FFN, P["mlp"], P["mlp"]),
        (f"sliding attention ×{len(SLID)}", "#94a3b8", P["attn_s"], P["attn_s"]),
        (f"full attention ×{len(FULL)}", ATT, P["attn_f"], P["attn_f"]),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, 0, P["embed"], "tied: the embed table, stored once"),
        ("vision embedder", VISION, P["vis"], 0, "runs per image"),
        ("audio embedder", AUDIO, P["aud"], 0, "runs per audio clip"),
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
bar("bp", -1700, "Where the parameters are stored (one BF16 safetensors file)", BUD, 2)
bar("bc", -1500, "What one text token is multiplied by — dense, so every layer weight; the lookup and the per-image / per-clip embedders drop out",
    BUD, 3)
M.save("model.canvas")

# ======================= block.canvas =======================
K = Canvas(kt=0.05, clip=300, lbl_w=240)
COL = 600
SQ, SK, SV = -2100, -2100 + COL, -2100 + 2 * COL  # sliding q / k / v columns
FQ, FK, FV = 800, 800 + COL, 800 + 2 * COL  # full q / k / v columns
K.node("cap-btitle", -2200, -420, 1700, 170,
       f"# Gemma 4 12B Unified — one decoder layer\n"
       f"Left: the attention of the {len(SLID)} **sliding** layers ({fmt(P['layer_s'])} per layer). Right: the "
       f"{len(FULL)} **full** layers ({fmt(P['layer_f'])}). A layer takes one of the two; norms, MLP, residuals are "
       f"shared. Grey bars = tensors ({K.kt} px/channel, long ones clipped), boxes = operations with their weights.")
K.node("cap-blegend", 800, -420, 1500, 170,
       f"**Full layers save memory three ways**: one KV head ({GKV} × {GHD}) for all {HQ} query heads, **K = V** "
       f"(no v_proj — keys and values are the same projection, normalised differently), and **p-RoPE** that "
       f"rotates only {ROT} of {GHD} dims (θ {RF['rope_theta']:,.0f}), leaving the rest position-free for long range.")

ap = lambda i, name: params_of(LM + rf"layers\.{i}\.self_attn\.{name}\.weight")
m = Flow(K, 0, -180, lbl_w=300)
xin, _ = m.tensor(DIM, f"x [T, {DIM}]  residual stream in", key="xin")
m.op("**input norm** — RMSNorm", NORM, key="n1")
_, hb = m.tensor(DIM, f"x̂ [T, {DIM}]", key="h")
yb = m.y + 300  # room for the fan-out to the projections

s0, f0 = SLID[0], FULL[0]
# sliding: GQA, head 256, RoPE on every dim
q = Flow(K, SQ, yb, last=hb)
q.op(f"**q_proj** {DIM} → {HQ}×{HD}", ATT, ap(s0, "q_proj"), key="sq")
q.tensor(HQ * HD, f"q [T, {HQ}, {HD}]", key="sqt")
q.op(f"**q_norm** RMSNorm per head ({HD})", NORM, key="sqn")
q.op(f"**RoPE** θ {RS['rope_theta']:,.0f}, all {HD} dims", ATT, key="sqr")
k = Flow(K, SK, yb, last=hb)
k.op(f"**k_proj** {DIM} → {HKV}×{HD}", ATT, ap(s0, "k_proj"), key="sk")
k.tensor(HKV * HD, f"k [T, {HKV}, {HD}]", key="skt")
k.op(f"**k_norm** RMSNorm per head ({HD})", NORM, key="skn")
k.op(f"**RoPE** θ {RS['rope_theta']:,.0f}", ATT, key="skr")
vv = Flow(K, SV, yb, last=hb)
vv.op(f"**v_proj** {DIM} → {HKV}×{HD}", ATT, ap(s0, "v_proj"), key="sv")
vv.tensor(HKV * HD, f"v [T, {HKV}, {HD}]", key="svt")
vv.op("**v_norm** RMSNorm (no weight)", NORM, key="svn")
ya = q.y
satt = K.node("satt", SQ - OP_W / 2, ya, 2 * COL + OP_W, 90,
              f"**sliding-window attention** — {HQ} q heads share {HKV} KV heads ({HQ // HKV}:1 GQA), head {HD}, "
              f"scale 1 (qk-norm does the scaling)\ncausal, last {WIN} tokens · image tokens also see their own image", ATT)
K.edge(q.last, satt)
K.edge(k.last, satt)
K.edge(vv.last, satt, ("bottom", "right"))
sc = Flow(K, SK, ya + 90 + GAP, last=satt)
sc.tensor(HQ * HD, f"[T, {HQ * HD}]  heads concatenated", key="sao")
sc.op(f"**o_proj** {HQ * HD} → {DIM}", ATT, ap(s0, "o_proj"), key="so")
s_out, _ = sc.tensor(DIM, f"[T, {DIM}]", key="sot")

# full: MQA, head 512, K = V, proportional RoPE on a quarter of the dims
q = Flow(K, FQ, yb, last=hb, labels_left=True)
q.op(f"**q_proj** {DIM} → {HQ}×{GHD}", ATT, ap(f0, "q_proj"), key="fq")
q.tensor(HQ * GHD, f"q [T, {HQ}, {GHD}]", key="fqt")
q.op(f"**q_norm** RMSNorm per head ({GHD})", NORM, key="fqn")
q.op(f"**p-RoPE** θ {RF['rope_theta']:,.0f}\nrotates {ROT} of {GHD} dims", ATT, key="fqr", h=72)
k = Flow(K, FK, yb, last=hb, labels_left=True)
k.op(f"**k_proj** {DIM} → {GKV}×{GHD}", ATT, ap(f0, "k_proj"), key="fk")
_, kraw = k.tensor(GKV * GHD, f"k = v [T, {GKV}, {GHD}]", key="fkt")
k.op(f"**k_norm** RMSNorm ({GHD})", NORM, key="fkn")
k.op(f"**p-RoPE** θ {RF['rope_theta']:,.0f}\nrotates {ROT} of {GHD} dims", ATT, key="fkr", h=72)
vv = Flow(K, FV, k.y - 72 - GAP, last=kraw)
vv.op("**v_norm** RMSNorm (no weight)\non the *same* k_proj output", NORM, key="fvn", h=72)
K.edges[-1].update(fromSide="right")
ya = max(q.y, k.y, vv.y)
fatt = K.node("fatt", FQ - OP_W / 2, ya, 2 * COL + OP_W, 90,
              f"**full causal attention** — all {HQ} q heads share **{GKV}** KV head (MQA), head {GHD}, scale 1\n"
              "whole context, always causal (no bidirectional image block here)", ATT)
K.edge(q.last, fatt)
K.edge(k.last, fatt)
K.edge(vv.last, fatt)
fc = Flow(K, FK, ya + 90 + GAP, last=fatt, labels_left=True)
fc.tensor(HQ * GHD, f"[T, {HQ * GHD}]  heads concatenated", key="fao")
fc.op(f"**o_proj** {HQ * GHD} → {DIM}", ATT, ap(f0, "o_proj"), key="fo")
f_out, _ = fc.tensor(DIM, f"[T, {DIM}]", key="fot")

# shared tail: post-attn norm, residual, MLP, residual, layer scalar
m = Flow(K, 0, max(sc.y, fc.y) + 160, lbl_w=300)
n2 = m.op("**post-attention norm** — RMSNorm", NORM, key="n2")
K.edge(s_out, n2, ("bottom", "left"), f"{len(SLID)} sliding layers")
K.edge(f_out, n2, ("bottom", "right"), f"{len(FULL)} full layers")
add1 = m.op("**+ residual**", ACT, key="add1")
K.edge(xin, add1, ("left", "left"), "residual")
xm, _ = m.tensor(DIM, f"x [T, {DIM}]", key="xm")
m.op("**pre-FFN norm** — RMSNorm", NORM, key="n3")
m.tensor(DIM, f"h [T, {DIM}]", key="h2")
mp = lambda name: params_of(LM + rf"layers\.0\.mlp\.{name}\.weight")
m.op(f"**gate_proj ‖ up_proj** {DIM} → 2 × {FF}", FFN, mp("gate_proj") + mp("up_proj"), key="gu")
m.tensor(FF, f"gate, up [T, {FF}]", lanes=2, key="gut")
m.op("**GELU-tanh(gate) · up**", FFN, key="gel")
m.tensor(FF, f"[T, {FF}]", key="ff")
m.op(f"**down_proj** {FF} → {DIM}", FFN, mp("down_proj"), key="down")
m.tensor(DIM, f"[T, {DIM}]", key="dt")
m.op("**post-FFN norm** — RMSNorm", NORM, key="n4")
add2 = m.op("**+ residual**", ACT, key="add2")
K.edge(xm, add2, ("left", "left"), "residual")
m.tensor(DIM, f"x [T, {DIM}]", key="xs")
m.op("**× layer_scalar** (1 learned number)", ACT, key="ls")
m.tensor(DIM, f"x [T, {DIM}]  → next layer", key="xout")
K.node("note-kv", FV + OP_W / 2 + 40, yb, 440, 200,
       f"**KV cache per token** — sliding: K+V = 2 × {HKV} × {HD} = {kv_s:,} values, kept only for the last {WIN} "
       f"tokens. Full: K and V = 2 × {GKV} × {GHD} = {kv_f:,} values for every token. K = V shares the weight, "
       "not the cached tensor: K gets k_norm + p-RoPE, V only a weightless RMSNorm.")
K.node("note-scale", SQ - OP_W / 2 - 480, yb, 440, 170,
       "**qk-norm**: q and k are RMS-normalised per head with a learned weight, and the attention scale is 1 "
       "(no 1/√d). v is RMS-normalised without a weight. Same in both layer types.")
K.save("block.canvas")

act = sum(r[3] for r in BUD)
print("budget", [(r[0], fmt(r[2]), fmt(r[3])) for r in BUD])
print(f"stored {P['all']:,} ({fmt(P['all'])})  active {act:,} ({fmt(act)})  drafter {P['draft']:,}")
