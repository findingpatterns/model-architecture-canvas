"""Qwen3.8-Flash-Next (Qwen/Qwen3.8-Flash-Next) — model and layer canvases, derived from the checkpoint.

`Qwen4ExpForConditionalGeneration` (model_type qwen4_exp): the Qwen3.5 hybrid (Gated DeltaNet 3 : 1 gated
attention, 512-expert MoE, ViT) rebuilt with four new parts, all traced from transformers'
models/qwen4_exp/modular_qwen4_exp.py:
  - Gated Residual: the residual stream is 4 lanes wide (hc_count) and every block reads / writes it
    through learned gates (Qwen4ExpTextGatedResidual) instead of pre-norm + add
  - QSA (Qwen Sparse Attention): the attention layers attend only to top-scoring 4-token blocks
    picked by a small MQA indexer (Qwen4ExpTextQSAIndexer)
  - PLE n-gram embedding: one layer adds hashed bigram / trigram embeddings (Qwen4ExpTextPLELayer)
  - the MTP head (not in transformers) is wired from its tensor names
Every number comes from shapes.json (safetensors headers) or config.json.

  model.canvas — input → embed → 4 lanes → 48 layer rows → lane mixer → head, plus vision, n-gram and MTP
  block.canvas — one decoder layer: gated residual read/write around both token mixers and the MoE

Conventions: grey bars = tensors (length ∝ channels per token), boxes = operations, note-* ids are
annotations, cap-* ids are borderless captions.

usage: python3 qwen3_8_flash_next_model_canvas.py <shapes.json> <config.json> <models/qwen3-8-flash-next>
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
TYPES = C["layer_types"]
E, K = C["num_experts"], C["num_experts_per_tok"]
HD, NQ, NKV = C["head_dim"], C["num_attention_heads"], C["num_key_value_heads"]
LKH, LVH, LKD, LVD = C["linear_num_key_heads"], C["linear_num_value_heads"], C["linear_key_head_dim"], C["linear_value_head_dim"]
ROT = int(HD * C["rope_parameters"]["partial_rotary_factor"])
HC, HCR = C["hc_count"], C["hc_lowrank"]
HCD = HC * DIM  # width of the 4-lane residual stream
IXH, IXD, IXB, IXC = C["indexer_n_heads"], C["indexer_head_dim"], C["indexer_budget"], C["indexer_compress_ratio"]
NG, HPN, PLE_D = C["ngram_size"], C["heads_per_ngram"], C["ple_embed_dim"]
NGH = (NG - 1) * HPN  # hashed n-gram heads (bigram + trigram)
PLE_L = [i - 1 for i in C["ple_layer_ids"]]  # config ids are one-indexed
LIN, FULL = "linear_attention", "full_attention"
N_LIN, N_FULL = TYPES.count(LIN), TYPES.count(FULL)
FIRST_LIN, FIRST_FULL = TYPES.index(LIN), TYPES.index(FULL)
LP = r"model\.language_model\.layers\."
LM = r"model\.language_model\."

# colours (JSON Canvas presets "1"–"6" or hex)
EMB, FFN, HEAD, ACT = "6", "4", "2", "#64748b"
GDN, ATT, SHARED, VISION = "#22d3ee", "#f97316", "#a3e635", "#a78bfa"
IDX, GRES, NGRAM, MTP = "#f43f5e", "#eab308", "#d946ef", "#f59e0b"


def is_weight(k, dt):
    """Integer tensors are the n-gram hash buffers (multipliers, head sizes, offsets), not weights."""
    return not dt.startswith("I") and not k.endswith("_scale")


def params_of(pattern):
    rx = re.compile(pattern)
    return sum(math.prod(shp) for k, (dt, shp) in S.items() if rx.fullmatch(k) and is_weight(k, dt))


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


def tp(n):
    return n if isinstance(n, str) else f"{n:,}".replace(",", " ")


_gu = S[f"model.language_model.layers.0.mlp.experts.gate_up_proj"][1]
_dn = S[f"model.language_model.layers.0.mlp.experts.down_proj"][1]
assert _gu[0] == _dn[0] == E
P = {
    "embed": params_of(LM + r"embed_tokens\.weight"),
    "head": params_of(r"lm_head\.weight"),
    "vit": params_of(r"model\.visual\.(?!merger\.).*"),
    "merger": params_of(r"model\.visual\.merger\..*"),
    "experts": params_of(LP + r"\d+\.mlp\.experts\..*"),
    "expert1": _gu[1] * _gu[2] + _dn[1] * _dn[2],  # one expert's slice of the fused [E, ...] tensors
    "shared": params_of(LP + r"\d+\.mlp\.(shared_expert\..*|shared_expert_gate\.weight|gate\.weight)"),
    "shared1": params_of(LP + r"0\.mlp\.(shared_expert\..*|shared_expert_gate\.weight)"),
    "router1": params_of(LP + r"0\.mlp\.gate\.weight"),
    "gdn": params_of(LP + r"\d+\.linear_attn\..*"),
    "att": params_of(LP + r"\d+\.self_attn\.(?!indexer\.).*"),
    "idx": params_of(LP + r"\d+\.self_attn\.indexer\..*"),
    "idx1": params_of(LP + rf"{FIRST_FULL}\.self_attn\.indexer\..*"),
    "gres": params_of(LP + r"\d+\.(attn|mlp)_hyper_connection\..*") + params_of(LM + r"hyper_connection_mixer\..*"),
    "gres1": params_of(LP + r"0\.attn_hyper_connection\..*"),
    "mixer": params_of(LM + r"hyper_connection_mixer\..*"),
    "ngram": params_of(LP + r"\d+\.ple\.ple_embedding\..*"),
    "ple": params_of(LP + r"\d+\.ple\.(?!ple_embedding\.).*"),
    "mtp": params_of(r"mtp\..*"),
    "all": params_of(r".*"),
}
LAYER_P = {t: params_of(LP + rf"{TYPES.index(t)}\..*") for t in (LIN, FULL)}
NG_ROWS = sum(S[k][1][0] for k in S if ".ngram_embedding." in k)
NG_PARTS = sum(1 for k in S if ".ngram_embedding." in k)
NG_HD = PLE_D // NGH
assert [int(re.search(r"layers\.(\d+)\.ple", k)[1]) for k in S if k.endswith("ple.value_proj.weight")] == PLE_L
assert sorted({int(re.search(r"layers\.(\d+)\.self_attn", k)[1]) for k in S if k.startswith("model") and ".self_attn." in k}) \
    == [i for i in range(N) if TYPES[i] == FULL]
DTYPES = sorted({dt for k, (dt, _) in S.items() if is_weight(k, dt)})


def budget():
    """[(name, colour, stored params, params one text token is multiplied by, why-zero)]."""
    routed_active = N * K * P["expert1"]
    rows = [
        (f"routed experts ({E}/layer)", FFN, P["experts"], routed_active),
        ("n-gram embedding (L%d)" % PLE_L[0], NGRAM, P["ngram"], 0, "row lookup"),
        ("shared expert + router", SHARED, P["shared"], P["shared"]),
        (f"Gated DeltaNet ×{N_LIN}", GDN, P["gdn"], P["gdn"]),
        (f"QSA attention ×{N_FULL}", ATT, P["att"], P["att"]),
        (f"QSA indexers ×{N_FULL}", IDX, P["idx"], P["idx"]),
        (f"gated residuals ×{2 * N + 1}", GRES, P["gres"], P["gres"]),
        ("PLE projections + conv", NGRAM, P["ple"], P["ple"]),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, P["head"], P["head"]),
        ("MTP", MTP, P["mtp"], 0, "only when drafting"),
        ("ViT + merger", VISION, P["vit"] + P["merger"], 0, "runs per image"),
    ]
    other = P["all"] - sum(r[2] for r in rows)
    assert other == 0, other  # every weight tensor is in exactly one row
    return rows


# ---------- canvas primitives ----------
class Canvas:
    def __init__(self, kt, clip, op_w=300, op_h=56, gap=40, lbl_w=440):
        self.nodes, self.edges = [], []
        self.kt, self.clip, self.op_w, self.op_h, self.gap, self.lbl_w = kt, clip, op_w, op_h, gap, lbl_w
        self.seq = 0

    def node(self, nid, x, y, w, h, text="", color=None, group=False):
        text = re.sub(r"(?<=[^\n])\n(?=[^\n-])", "  \n", text)  # markdown hard break; keeps lists
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

    def y_of(self, nid):
        return next(n["y"] for n in self.nodes if n["id"] == nid)

    def save(self, name):
        json.dump(dict(nodes=self.nodes, edges=self.edges), open(os.path.join(OUT, name), "w"), indent=1, ensure_ascii=False)


class Flow:
    """A vertical chain of ops and tensors centred on cx; each item is linked to the previous one."""

    def __init__(self, cv, cx, y, last=None, labels_left=False):
        self.cv, self.cx, self.y, self.last, self.labels_left = cv, cx, y, last, labels_left

    def _link(self, top, bottom):
        if self.last:
            self.cv.edge(self.last, top)
        self.last = bottom

    def op(self, text, color, params=0, note="", key="op", h=None):
        cv = self.cv
        if params:
            text += f"\n{fmt(params)}" + (f" · {note}" if note else "")
        h = h or cv.op_h
        nid = cv.node(cv.uid(key), self.cx - cv.op_w / 2, self.y, cv.op_w, h, text, color)
        self._link(nid, nid)
        self.y += h + cv.gap
        return nid

    def tensor(self, dim, label, key="t", left=None):
        cv = self.cv
        real = dim * cv.kt
        w = min(real, cv.clip)
        nid = cv.node(cv.uid(key), self.cx - w / 2, self.y, w, 9, "", ACT)
        clip = f" — bar clipped (real {real:.0f}px)" if real > w else ""
        half = max(w / 2, cv.op_w / 2) + 20
        lx = self.cx - half - cv.lbl_w if (self.labels_left if left is None else left) else self.cx + half
        cv.node(cv.uid("note-lbl"), lx, self.y - 10, cv.lbl_w, 34, f"`{label}`{clip}")
        self._link(nid, nid)
        self.y += 14 + cv.gap
        return nid


# =====================================================================================
# model.canvas
# =====================================================================================
KT = 0.08
M = Canvas(KT, 1800)
ROW_W, ROW_H, ROW_GAP = 300, 34, 6
COL_X = (0,)
BAR_W, BAR_H, MIN_SEG = 2600, 44, 16
TYPE_NAME = {LIN: "Gated DeltaNet", FULL: "QSA attention"}
TYPE_COLOR = {LIN: GDN, FULL: ATT}

M.node("cap-title", -900, -1300, 1250, 230,
       "# Qwen3.8-Flash-Next — model\n"
       f"`{CFG['architectures'][0]}` (model_type `{CFG['model_type']}`), Qwen's experimental preview of the Qwen4 "
       f"architecture. One continuous decoder stack L0–{N - 1}, read top → bottom; layers that repeat are drawn once with **×N**. "
       f"**Grey bars = tensors**, length ∝ channels per token ({KT} px/channel); **boxes = operations**. "
       "T = sequence length. *Layer* tab = one layer in detail.")
M.node("cap-legend", 450, -1300, 1250, 230,
       f"Layer rows — **Gated DeltaNet** ({N_LIN} layers, linear attention, fixed-size state) · "
       f"**QSA attention** ({N_FULL}, every {C['full_attention_interval']}th: softmax over the top {IXB} tokens "
       f"an indexer picks in {IXC}-token blocks) · **pink** = the one DeltaNet layer that also adds hashed n-gram "
       f"embeddings (PLE). Every layer: MoE **{K} of {E}** experts + 1 shared.  \n"
       f"The residual stream is **{HC} lanes × {DIM} = {HCD}** wide; blocks read it and write back through "
       f"gated residuals. Layer ≈ **{fmt(LAYER_P[LIN])}** (DeltaNet) / **{fmt(LAYER_P[FULL])}** (QSA). "
       f"All weights {', '.join(DTYPES)}.")

# ---------- input flow ----------
cx0 = COL_X[0] + ROW_W / 2
f = Flow(M, cx0, -960)
f.op("**raw text**  \"The cat sat…\"", ACT, key="raw")
tok = f.op(f"**tokenizer** (vocab {tp(VOCAB)})", ACT, key="tok")
f.tensor(40 / KT, "token ids [T] — one integer per token", key="ids")
f.op("**embed** — look up row *id*", EMB, P["embed"], note=f"{tp(VOCAB)} × {DIM}", key="embed")
x0 = f.tensor(DIM, f"x [T, {DIM}]  (+ image/video tokens spliced in)", key="x")
f.op(f"**copy into {HC} lanes**", GRES, key="rep")
f.tensor(HCD, f"H [T, {HC}×{DIM}] residual lanes", key="H", left=True)
ROW0 = f.y + 40


def kind(i):
    return TYPES[i], i in PLE_L


def kind_label(k):
    return TYPE_NAME[k[0]] + (" + n-gram PLE" if k[1] else "") + f" + MoE {K}/{E}"


def kind_color(k):
    return NGRAM if k[1] else TYPE_COLOR[k[0]]


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
ROW_OF = {}  # layer → the row node that draws it
for seg in segments():
    if seg[0] == "run":
        _, a, n = seg
        span = f"**L{a}–{a + n - 1}** ×{n}" if n > 1 else f"**L{a}**"
        nid = M.node(f"seg{a}", CX - SEG_W / 2, y, SEG_W, SEG_H, f"{span}\n{kind_label(kind(a))}", kind_color(kind(a)))
        M.edge(prev, nid)
        ROW_OF.update({i: nid for i in range(a, a + n)})
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
        nid = M.node(f"per{j}", CX - SEG_W / 2, ry, SEG_W, SEG_H,
                   f"{kind_label(kind(s0 + j))}\nL{ls[0]}, {ls[1]}, …, {ls[-1]}", kind_color(kind(s0 + j)))
        M.edge(prev, nid)
        first = first or nid
        ROW_OF.update({l_: nid for l_ in ls})
        prev = nid
        ry += SEG_H + SEG_GAP
    M.edge(prev, first, ("left", "left"))  # loop back: the period runs again
    M.node("cap-repeat", gx - 170, (y + 50 + ry - SEG_GAP) / 2 - 30, 150, 60, f"## ↺ ×{reps}")
    gh = ry - y + 10
    M.node("grp-period", gx, gy, SEG_W + 120, gh, f"×{reps} — L{s0}–{N - 1}, a period of {p} layers", group=True)
    y += gh + SEG_GAP
LAST = prev

# barcode: every layer in order, one thin cell each, coloured like the rows
CELL, CELL_GAP = 16, 2
BX, BY = 1600, ROW0 + 40  # right of the n-gram branch, so its edge into L1 stays clear
M.node("cap-barcode", BX, BY - 50, max(N * (CELL + CELL_GAP), 600), 40,
     f"**Every layer, L0 → L{N - 1}** (1 cell = 1 layer, colours as in the rows)")
for i in range(N):
    M.node(f"lyr{i}", BX + i * (CELL + CELL_GAP), BY, CELL, 44, "", kind_color(kind(i)))
for i in sorted(set(range(0, N - 4, 10)) | {N - 1}):
    M.node(f"note-tick{i}", BX + i * (CELL + CELL_GAP) - 4, BY + 50, 60, 30, f"L{i}")

# ---------- n-gram embedding branch (PLE), fed by the token ids ----------
cxn = 1150
n = Flow(M, cxn, M.y_of(tok))
hsh = n.op(f"**hash {', '.join(f'{g}-grams' for g in range(2, NG + 1))}**\n{HPN} hashes each → {NGH} ids / token", NGRAM, key="hash", h=64)
M.edge(tok, hsh, ("right", "left"), "token ids")
n.tensor(NGH * 40, f"n-gram ids [T, {NGH}]", key="ngid")
n.op(f"**n-gram table** lookup — {NGH} rows × {NG_HD}", NGRAM, P["ngram"], note=f"{tp(NG_ROWS)} rows", key="ngt", h=64)
ng_out = n.tensor(PLE_D, f"e_ngram [T, {NGH}×{NG_HD} = {PLE_D}]", key="nge")
for li in PLE_L:
    M.edge(ng_out, ROW_OF[li], ("bottom", "right"), f"into L{li}", NGRAM)

# annotations: left of the stack and under the barcode
ya = ROW0
gdn_state = LVH * LVD * LKD
kv_tok = 2 * NKV * HD
NOTE_X = COL_X[0] - 900
M.node("note-gdn", NOTE_X, ya + 120, 480, 230,
       f"**Gated DeltaNet rows** ({N_LIN}): linear attention as in Qwen3.5. Each layer keeps one recurrent state "
       f"[{LVH} heads × {LKD} × {LVD}] = {tp(gdn_state)} values, the same size at any T — no growing KV cache. "
       f"New here: the output gate is **sigmoid** (`output_gate_type`), Qwen3.5 used SiLU.")
M.node("note-full", NOTE_X, ya + 380, 480, 270,
       f"**QSA rows** ({N_FULL}, L{FIRST_FULL}, L{FIRST_FULL + 4}, …): Qwen3.5's gated GQA attention "
       f"({NQ} q : {NKV} KV heads × {HD}, RoPE on {ROT} dims) made sparse. A {IXH}-head MQA indexer "
       f"({fmt(P['idx1'])}) mean-pools keys over {IXC}-token blocks, scores them and keeps the top "
       f"{IXB // IXC} blocks = {IXB} tokens per query. Cache per token: {tp(kv_tok)} K/V values + a {IXD}-dim indexer key.")
M.node("note-ple", NOTE_X, ya + 680, 480, 290,
       f"**n-gram PLE** (L{PLE_L[0]} only, `ple_layer_ids` {C['ple_layer_ids']} is one-indexed): the previous "
       f"{NG - 1} token ids are hashed (XOR of multiplied ids, mod {NGH} distinct primes ≥ "
       f"{tp(C['ngram_vocab_size_base'])}) into one table of **{fmt(P['ngram'])}** params — stored as "
       f"{NG_PARTS} shards, {tp(NG_ROWS)} rows × {NG_HD}. Per token it is {NGH} row lookups, so it costs memory, "
       "not compute. History resets at EOS.")
M.node("note-moe", BX, BY + 120, 460, 230,
       f"**MoE in every layer**: softmax router over {E} experts → top-{K}, weights renormalised. "
       f"Expert = SwiGLU {DIM}→{C['moe_intermediate_size']}→{DIM} ({fmt(P['expert1'])}). "
       f"Shared expert (same width) added through a sigmoid gate. "
       f"Per token: {K} + 1 of {E} experts → ~{(K + 1) * P['expert1'] / (E * P['expert1'] + P['shared1']):.1%} of FFN weights.")
M.node("note-gres", BX + 500, BY + 120, 460, 300,
       f"**Gated residual** replaces pre-norm + add. Before each mixer and each MoE: RMSNorm per lane, "
       f"read gate σ(up(SiLU(down(H)))) with a rank-{HCR} bottleneck, average the {HC} gated lanes → one "
       f"{DIM}-wide input. The block output is added back to every lane × its own scalar 2σ(·). "
       f"{fmt(P['gres1'])} per gate, {2 * N} gates + a final mixer = {fmt(P['gres'])}. "
       "There is no separate input / post-attention / final RMSNorm.")

# ---------- vision tower, spliced into x ----------
ps, tps, ms = V["patch_size"], V["temporal_patch_size"], V["spatial_merge_size"]
pin = V["in_channels"] * tps * ps * ps
v = Flow(M, COL_X[0] - 480, -960, labels_left=True)
v.op("**image / video** frames", ACT, key="img")
v.tensor(pin, f"patches [P, {pin}]  ({tps}×{ps}×{ps}×{V['in_channels']})", key="patch")
v.op(f"**ViT ×{V['depth']}** (+ learned pos-embed)", VISION, P["vit"], note=f"width {V['hidden_size']}", key="vit")
v.tensor(V["hidden_size"], f"[P, {V['hidden_size']}]", key="vf")
v.op(f"**merger MLP** — {ms}×{ms} patches → 1", VISION, P["merger"], key="mer")
vt = v.tensor(DIM, f"visual tokens [P/{ms * ms}, {DIM}]", key="vt")
M.edge(vt, x0, ("right", "left"), "spliced into x")

# ---------- output flow ----------
o = Flow(M, cx0, max(y, ya + 680 + 290) + 40, last=LAST, labels_left=True)  # below the side notes
ylast = o.y - ROW_H - 60
h_out = o.tensor(HCD, f"H [T, {HCD}]  after L{N - 1}", key="ho")
o.op(f"**lane mixer** — gated read, {HC} lanes → 1\n(hyper_connection_mixer; acts as final norm)", GRES, P["mixer"], key="mix", h=80)
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm head** — {DIM} → {tp(VOCAB)}", HEAD, P["head"], note="untied", key="head")
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op("**softmax → sample**", ACT, key="samp")
o.op("**next token id**", ACT, key="next")

# ---------- MTP: one extra QSA + MoE layer drafting one more token ----------
mx = 2750
mt = Flow(M, mx, ylast + ROW_H + 60)
mh = mt.op(f"**pre_fc_norm_hidden** (per lane) → **fc_hidden** {DIM}→{DIM}", MTP, params_of(r"mtp\.(pre_fc_norm_hidden|fc_hidden)\..*"), key="mfh", h=64)
M.edge(h_out, mh, ("right", "left"), f"H after L{N - 1}", MTP)
me = M.node("mtp-emb", mx + 400, mt.y_of if False else M.y_of(mh), 340, 64,
            f"embed(token t+1) → **pre_fc_norm_embedding → fc_embedding** {DIM}→{DIM} · "
            f"{fmt(params_of(r'mtp\.(pre_fc_norm_embedding|fc_embedding)\..*'))}", MTP)
mt.tensor(HCD, f"[T, {HCD}]  lanes for the MTP layer", key="mlane", left=True)
mly = mt.op(f"**1 layer**: gated residuals + QSA attention + MoE {K}/{E}", MTP,
            params_of(r"mtp\.layers\..*"), key="mly", h=64)
M.edge(me, mly, ("bottom", "right"))
mt.op("**lane mixer → shared lm head**", MTP, params_of(r"mtp\.hyper_connection_mixer\..*"), key="mhd")
mt.op("**draft token t+2** → verified by the main model", ACT, key="mdr")
M.node("note-mtp", mx - 150, mt.y, 600, 190,
       f"Multi-token prediction head ({fmt(P['mtp'])}, {C['mtp_num_hidden_layers']} layer, config `mtp.hybrid: "
       f"{str(C['mtp']['hybrid']).lower()}`). Not in transformers (its loader drops `mtp.*`): wiring is read off the tensor "
       f"names. pre_fc_norm_hidden is {HCD} wide, so it takes the {HC}-lane state; how the two fc outputs are combined "
       "is not public. No own embed / head — it reuses the main ones.")

# ---------- what is new ----------
M.node("note-new", -900, max(o.y, mt.y + 190) + 120, 1300, 260,
       "**New vs Qwen3.5 / Qwen3.8** (same 3 : 1 DeltaNet / attention rhythm, same 512-expert MoE and ViT):\n"
       f"- **gated residual**: {HC} residual lanes ({HCD} wide) read / written through learned gates\n"
       f"- **QSA**: attention layers see only top-{IXB} tokens chosen in {IXC}-token blocks by an indexer\n"
       f"- **n-gram PLE**: a {fmt(P['ngram'])} hashed bigram/trigram table feeding L{PLE_L[0]}\n"
       "- DeltaNet output gate sigmoid instead of SiLU; no standalone RMSNorms")

# ---------- budget bars ----------
def bar(cv, key, y, title, rows, idx):
    total = sum(r[idx] for r in rows)
    cv.node(f"cap-{key}t", -900, y - 46, BAR_W, 40, f"**{title}** — total {fmt(total)}")
    x, legend = -900, []
    # segments too thin to draw go last, so any blank stretch is at the bar's end, not mid-bar
    for k in sorted(range(len(rows)), key=lambda k: BAR_W * rows[k][idx] / total < MIN_SEG):
        r = rows[k]
        share = r[idx] / total
        w = BAR_W * share
        if w >= MIN_SEG:  # integer edges so neighbouring segments meet exactly, never overlap
            cv.node(f"{key}{k}", round(x), y, round(x + w) - round(x), BAR_H, f"**{r[0]}**" if w > 200 else "", r[1])
        x += w
        why = r[4] if len(r) > 4 and not r[idx] else ""
        legend.append(f"{r[0]} **{fmt(r[idx]) if r[idx] else '0'}**" + (f" ({why})" if why else f" ({share:.1%})" if share >= 0.001 else " (<0.1%)"))
    cv.node(f"cap-{key}l", -900, y + BAR_H + 6, BAR_W, 64, " · ".join(legend))


B = budget()
bar(M, "bp", -1720, "Where the parameters are stored", B, 2)
bar(M, "bc", -1520, "What one text token is multiplied by (active weights)", B, 3)
M.save("model.canvas")

# =====================================================================================
# block.canvas — one decoder layer
# =====================================================================================
KB = 0.06
L = Canvas(KB, 720, op_w=320, op_h=64, lbl_w=560)
XG, XA, XI = -1060, 760, 2400  # DeltaNet, attention and indexer columns
SIDE_W = 270
L.node("cap-btitle", -1400, -760, 1300, 210,
       "# Qwen3.8-Flash-Next — one decoder layer\n"
       f"The residual stream is **{HC} lanes** ([T, {HC}×{DIM}]). A gated residual **reads** one {DIM}-wide input "
       f"from it, the token mixer runs — **either** Gated DeltaNet (left, {N_LIN} layers) **or** QSA attention "
       f"(right, {N_FULL} layers) — and its output is **written** back to every lane. The same again for the MoE. "
       f"Bars ∝ channels ({KB} px/channel). Shapes per token.")
L.node("cap-bsrc", 100, -760, 1300, 210,
       "Traced from transformers `modular_qwen4_exp.py` (Qwen4ExpTextDecoderLayer, Qwen4ExpTextGatedResidual, "
       "Qwen4ExpTextQSAIndexer, Qwen4ExpTextPLELayer; GatedDeltaNet / attention / MoE inherited from Qwen3.5 and "
       "Qwen3-Next) and checked against the checkpoint's tensor names/shapes. Parameter counts are per layer. "
       f"All weights {', '.join(DTYPES)}.")

m = Flow(L, 0, -480)
m.tensor(HCD, f"H [T, {HC}×{DIM}]  residual lanes in")
ple_add = m.op(f"**+ PLE** n-gram injection\n(L{PLE_L[0]} only; identity elsewhere)", NGRAM, h=72)
m.tensor(HCD, f"H [T, {HCD}]")


def gated_read(flow, which, left=False):
    """The read half of Qwen4ExpTextGatedResidual; returns (read op, inject op, x tensor)."""
    rd = flow.op(f"**{which}_hyper_connection** read\nRMSNorm per lane → gate = σ(up(SiLU(down Ĥ / {HC})))\n"
                 f"down {HCD}→{HCR}, up {HCR}→{HCD} · x = mean over lanes of gate ⊙ Ĥ",
                 GRES, P["gres1"], note="incl. inject", h=130)
    inj = L.node(L.uid("inj"), -SIDE_W - 420, L.y_of(rd) + 20, SIDE_W + 30, 90,
                 f"**block_inject_weight** Ĥ {HCD}→{HC}\nw = 2·σ(· / {HC})  · [T, {HC}]", GRES)
    L.edge(rd, inj, ("left", "right"))
    return rd, inj, flow.tensor(DIM, f"x [T, {DIM}]  block input", left=left)


_, inj1, xin = gated_read(m, "attn")
ysplit = m.y + 30

# --- Gated DeltaNet column ---
qk, vd = LKH * LKD, LVH * LVD
g = Flow(L, XG, ysplit, last=xin, labels_left=True)
g.op(f"**in_proj_qkv** {DIM} → {qk * 2 + vd}", GDN, params_of(LP + rf"{FIRST_LIN}\.linear_attn\.in_proj_qkv\.weight"))
g.tensor(qk * 2 + vd, f"[T, {qk * 2 + vd}] = q {qk} · k {qk} · v {vd}")
g.op(f"**causal conv1d** (k={C['linear_conv_kernel_dim']}, per channel) + SiLU", GDN)
g.tensor(qk * 2 + vd, f"q,k [T, {LKH}, {LKD}] · v [T, {LVH}, {LVD}]")
g.op(f"**L2-norm q, k**; repeat {LKH} → {LVH} heads", GDN)
g.tensor(LVH * LKD * 2 + vd, f"q,k,v [T, {LVH}, {LKD}]")
rule = g.op(f"**gated delta rule**\nS ← α·S(I − β k kᵀ) + β v kᵀ,  o = S q\nstate S [{LVH}, {LKD}, {LVD}] per layer", GDN, h=96)
g.tensor(vd, f"o [T, {LVH}, {LVD}]")
gn = g.op(f"**gated RMSNorm** per head\nnorm(o) · **sigmoid**(z)", GDN)
g.tensor(vd, f"[T, {vd}]")
g.op(f"**out_proj** {vd} → {DIM}", GDN, params_of(LP + rf"{FIRST_LIN}\.linear_attn\.out_proj\.weight"))
g_out = g.tensor(DIM, f"[T, {DIM}]  mixer out")

sx = -SIDE_W - 60  # side inputs of the DeltaNet sit between the two columns
ab = L.node("gdn-ab", sx, L.y_of(rule), SIDE_W, 110,
            f"**in_proj_a, in_proj_b** (from x)\nα = exp(−e^A_log · softplus(a + dt_bias))\n"
            f"β = sigmoid(b)  · [T, {LVH}] each", GDN)
L.edge(ab, rule, ("left", "right"), "decay α, write β")
zbox = L.node("gdn-z", sx, L.y_of(gn), SIDE_W, 64, f"**in_proj_z** (from x) → z [T, {vd}]", GDN)
L.edge(zbox, gn, ("left", "right"))

# --- QSA attention column ---
a = Flow(L, XA, ysplit, last=xin)
qp = params_of(LP + rf"{FIRST_FULL}\.self_attn\.q_proj\.weight")
a.op(f"**q_proj** {DIM} → {NQ * HD * 2}", ATT, qp)
a.tensor(NQ * HD * 2, f"[T, {NQ}, {HD}+{HD}] = query | gate", left=True)
a.op(f"**split**; RMSNorm on q (per head)", ATT)
a.tensor(NQ * HD, f"q [T, {NQ}, {HD}]")
a.op(f"**RoPE** on {ROT} of {HD} dims\n(M-RoPE t/h/w {'/'.join(map(str, C['rope_parameters']['mrope_section']))} pairs, θ={tp(C['rope_parameters']['rope_theta'])})", ATT, h=80)
a.tensor(NQ * HD, f"q [T, {NQ}, {HD}]")
sdpa = a.op(f"**sparse softmax attention**\ncausal ∧ indexer-selected tokens only\nGQA: {NQ // NKV} query heads per K/V head", ATT, h=96)
# labels below the attention go left, so the indexer's mask edge reaches its right side unobstructed
a.tensor(NQ * HD, f"[T, {NQ * HD}]", left=True)
a.op("**× sigmoid(gate)** per channel", ATT)
a.tensor(NQ * HD, f"[T, {NQ * HD}]", left=True)
a.op(f"**o_proj** {NQ * HD} → {DIM}", ATT, params_of(LP + rf"{FIRST_FULL}\.self_attn\.o_proj\.weight"))
a_out = a.tensor(DIM, f"[T, {DIM}]  mixer out")
kvp = params_of(LP + rf"{FIRST_FULL}\.self_attn\.(k|v)_proj\.weight")
kv = L.node("att-kv", 60, L.y_of(sdpa) - 20, SIDE_W, 140,
            f"**k_proj, v_proj** (from x) {DIM} → {NKV}×{HD} each · {fmt(kvp)}\nRMSNorm + RoPE on k\n"
            f"KV cache: {tp(kv_tok)} values / token", ATT)
L.edge(kv, sdpa, ("right", "left"), "K, V")

# --- QSA indexer column ---
ix = Flow(L, XI, ysplit + 260)
ixp = ix.op(f"**index_qk_proj** {DIM} → ({IXH}+1)×{IXD}", IDX,
            params_of(LP + rf"{FIRST_FULL}\.self_attn\.indexer\.index_qk_proj\.weight"))
L.edge(xin, ixp)  # leaves x from below, clear of its label
ix.tensor((IXH + 1) * IXD, f"q_I [T, {IXH}, {IXD}] · k_I [T, {IXD}] (cached)")
ix.op(f"**keys**: mean of each {IXC}-token block → RMSNorm → RoPE at block start\n**queries**: RMSNorm → RoPE", IDX, h=96)
ix.tensor(IXH * IXD + IXD, f"q_I [T, {IXH}, {IXD}] · block keys [T/{IXC}, {IXD}]")
ix.op(f"**score** = Σ_h ReLU(q_I,h · k_blk) / √{IXD}", IDX)
ix.tensor(80 / KB, f"block scores [T, T/{IXC}]  (causal)")
ix.op(f"**top-{IXB // IXC} blocks** = {IXB} tokens\n+ the unfinished last block", IDX, h=72)
sel = ix.tensor(160 / KB, f"selected tokens [T, ≤{IXB + IXC - 1}] → mask")
L.edge(sel, sdpa, ("bottom", "right"), "token mask")

# --- write back, then the MoE half ---
ymerge = max(g.y, a.y, ix.y) + 40
r = Flow(L, 0, ymerge)
w1 = r.op(f"**write**: H + w ⊗ mixer out\n(mixer out added to each lane × its own w)", GRES, h=80)
L.edge(g_out, w1, ("bottom", "left"))
L.edge(a_out, w1, ("bottom", "right"))
L.edge(inj1, w1, ("bottom", "left"), "w")
r.tensor(HCD, f"H [T, {HCD}]")
_, inj2, xm = gated_read(r, "mlp", left=True)
r.op(f"**router** {DIM} → {E}, softmax → top-{K}\nweights renormalised to sum 1", FFN, P["router1"], h=80)
r.tensor(40 / KB, f"expert ids + weights [T, {K}]")
r.op(f"**{K} routed experts** of {E}\nSwiGLU {DIM}→{C['moe_intermediate_size']}→{DIM}", FFN,
     K * P["expert1"], note=f"active of {fmt(E * P['expert1'])}", h=80)
r.tensor(DIM, f"[T, {DIM}]  weighted sum")
msum = r.op("**sum** routed + gated shared", FFN)
sh = L.node("moe-shared", 820, L.y_of(msum) - 150, 340, 110,
            f"**shared expert** (every token)\nSwiGLU {DIM}→{C['shared_expert_intermediate_size']}→{DIM} × sigmoid(gate)\n"
            f"{fmt(P['shared1'])}", SHARED)
L.edge(xm, sh, ("right", "top"))
L.edge(sh, msum, ("bottom", "right"))
r.tensor(DIM, f"[T, {DIM}]  MoE out")
w2 = r.op(f"**write**: H + w ⊗ MoE out → H [T, {HCD}] out", GRES, h=72)
L.edge(inj2, w2, ("left", "left"), "w")  # round the left, clear of the x label

# --- PLE detail (L1 only), left of the main path ---
PX = -2700
pl = Flow(L, PX, -1180, labels_left=True)  # ends level with the PLE injection it feeds
pl.op(f"**n-gram ids** of token t ({NGH} hashes of the last 2 / 3 tokens)", NGRAM, h=64)
pl.op(f"**n-gram table** lookup → {NGH} × {NG_HD}", NGRAM, P["ngram"], note="row lookup", h=64)
pl.tensor(PLE_D, f"e [T, {PLE_D}]")
pl.op(f"**key_proj** {PLE_D}→{HCD} + RMSNorm per lane\n**value_proj** {PLE_D}→{DIM}", NGRAM,
      params_of(LP + rf"{PLE_L[0]}\.ple\.(key_proj|value_proj|norm_key)\..*"), h=96)
pl.tensor(HCD + DIM, f"k [T, {HC}, {DIM}] · v [T, {DIM}]")
gq = pl.op(f"**per-lane gate** s = ⟨k_l, RMSNorm(H)_l⟩ / √{DIM}\ng_l = σ(sign(s)·√|s|) · v", NGRAM, h=80)
pl.tensor(HCD, f"g [T, {HC}, {DIM}]")
pl.op(f"**+ SiLU(dilated depthwise conv)** of RMSNorm(g)\nkernel {C['ple_conv_kernel_size']}, dilation {NG} (looks back {(C['ple_conv_kernel_size'] - 1) * NG} tokens)",
      NGRAM, h=80)
ple_out = pl.tensor(HCD, f"PLE out [T, {HCD}] → added to H")
L.edge(ple_out, ple_add, ("right", "left"), f"L{PLE_L[0]} only", NGRAM)
L.save("block.canvas")

print("ok", [(b[0], fmt(b[2]), fmt(b[3])) for b in B])
print("total", P["all"], fmt(P["all"]), "active", sum(b[3] for b in B), fmt(sum(b[3] for b in B)))
