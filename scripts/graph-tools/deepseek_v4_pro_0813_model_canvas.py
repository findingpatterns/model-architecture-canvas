"""DeepSeek-V4-Pro-0813 (deepseek-ai/DeepSeek-V4-Pro-0813): model-level and layer-level canvases.

Inputs are the repo's config.json, inference/config.json (the reference code's ModelArgs) and shapes.json
(safetensors headers, see fetch_safetensors_shapes.py). Dataflow follows the repo's inference/model.py
(Transformer / Block / Attention / Compressor / Indexer / MoE / DSparkBlock); every op is cross-checked
against the checkpoint tensor names.

  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale per canvas)
  - boxes       = operations, all one size; parameter / compute shares are the two bars at the top
  - note-* ids  = annotations (dim, borderless); cap-* ids = borderless captions

Packed FP4: routed-expert weights are stored as I8 tensors holding two FP4 (e2m1) values per byte
([out, in/2]); they are counted as 2 parameters per stored element. Block scales (*.scale) are not counted.

usage: python3 deepseek_v4_pro_0813_model_canvas.py <shapes.json> <config.json> <inference_config.json>
       <models/deepseek-v4-pro-0813>
"""
import json
import math
import os
import re
import sys

S = json.load(open(sys.argv[1]))
C = json.load(open(sys.argv[2]))
IC = json.load(open(sys.argv[3]))  # inference/config.json: n_mtp_layers lives only here
OUT = sys.argv[4]

N, DIM, VOCAB = C["num_hidden_layers"], C["hidden_size"], C["vocab_size"]
H, HD, ROPE = C["num_attention_heads"], C["head_dim"], C["qk_rope_head_dim"]
QL, OG, OL = C["q_lora_rank"], C["o_groups"], C["o_lora_rank"]
IH, ID, TOPK = C["index_n_heads"], C["index_head_dim"], C["index_topk"]
R, A, MI = C["n_routed_experts"], C["num_experts_per_tok"], C["moe_intermediate_size"]
WIN, RATIO, HC = C["sliding_window"], C["compress_ratios"], C["hc_mult"]
NHASH, TIDS = C["num_hash_layers"], C["dspark_target_layer_ids"]
NMTP, NB = IC["n_mtp_layers"], C["dspark_block_size"]
assert len(RATIO) == N + NMTP and all(r == 0 for r in RATIO[N:]), "DSpark stages are window-only"
assert IC["n_layers"] == N and IC["dim"] == DIM

EMB, NORM, FFN, HEAD, ACT, ATT, IDX, HCC = "6", "3", "4", "2", "#64748b", "1", "5", "#e879f9"
DSPARK, CMP = "#f59e0b", "#38bdf8"
ROW_COLOR = {4: "5", 128: "#94a3b8"}


def params_of(pattern):
    """Logical parameters: I8 tensors pack two FP4 values per byte; *.scale block scales are skipped."""
    rx = re.compile(pattern)
    return sum(math.prod(shp) * (2 if dt == "I8" else 1)
               for k, (dt, shp) in S.items() if rx.fullmatch(k) and not k.endswith(".scale"))


L = r"layers\.\d+\."
P = dict(
    embed=params_of(r"embed\.weight"),
    head=params_of(r"head\.weight"),
    routed=params_of(L + r"ffn\.experts\..*"),
    attn=params_of(L + r"attn\.(?!compressor\.|indexer\.).*"),
    comp=params_of(L + r"attn\.compressor\..*"),
    idx=params_of(L + r"attn\.indexer\..*"),
    shared=params_of(L + r"ffn\.(?:shared_experts\..*|gate\.weight|gate\.bias)"),
    tid=params_of(L + r"ffn\.gate\.tid2eid"),
    hc=params_of(L + r"hc_.*"),
    mtp=params_of(r"mtp\..*"),
    expert=params_of(r"layers\.4\.ffn\.experts\.0\..*"),
    shexp=params_of(r"layers\.4\.ffn\.shared_experts\..*"),
    router=params_of(r"layers\.4\.ffn\.gate\..*"),
    hchead=params_of(r"hc_head_.*"),
    mproj=params_of(r"mtp\.0\.main_(?:proj|norm)\..*"),
    markov=params_of(r"mtp\.\d+\.markov_head\..*"),
    conf=params_of(r"mtp\.\d+\.confidence_head\..*"),
    mtphead=params_of(r"mtp\.\d+\.(?:hc_head_.*|norm\.weight)"),
)
P["mblocks"] = P["mtp"] - P["mproj"] - P["markov"] - P["conf"] - P["mtphead"]
N4 = sum(1 for r in RATIO[:N] if r == 4)
N128 = sum(1 for r in RATIO[:N] if r == 128)
assert N4 + N128 == N
# every indexer the checkpoint stores belongs to a ÷4 layer; hash tables only in the first NHASH layers
stored_idx = sorted({int(re.match(r"layers\.(\d+)", k)[1]) for k in S if ".indexer." in k})
assert stored_idx == [i for i in range(N) if RATIO[i] == 4], stored_idx
assert sorted({int(re.match(r"layers\.(\d+)", k)[1]) for k in S if k.endswith("tid2eid")}) == list(range(NHASH))
assert all(len({k.split(".")[4] for k in S if k.startswith(f"layers.{i}.ffn.experts.")}) == R for i in (0, N - 1))


def budget():
    """[(name, colour, stored params, params multiplied per text token in the main forward, why-zero)]"""
    rows = [
        (f"routed experts {R}×{N} (FP4)", FFN, P["routed"], N * A * P["expert"]),
        (f"attention ×{N}", ATT, P["attn"], P["attn"]),
        (f"shared expert + router ×{N}", "#22c55e", P["shared"], P["shared"]),
        (f"KV compressors ×{N}", CMP, P["comp"], P["comp"]),
        (f"indexers ×{N4}", IDX, P["idx"], P["idx"]),
        (f"mHC mixers ×{N}", HCC, P["hc"], P["hc"]),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, P["head"], P["head"]),
        (f"DSpark ×{NMTP}", DSPARK, P["mtp"], 0, "only when drafting"),
        (f"hash tables ×{NHASH}", "#fbbf24", P["tid"], 0, "row lookup"),
    ]
    other = params_of(r".*") - sum(r[2] for r in rows)  # RMSNorm weights + final mHC head
    rows.append(("norms + mHC head", ACT, other, other))
    return rows


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


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

    def tensor(self, dim, label, key="t", left=None, lanes=1):
        """Grey bar (one per residual lane when lanes > 1); returns the last bar's id."""
        real = dim * self.kt
        w = max(16, min(real, self.clip))
        ids = [node(self._id(key), self.cx - w / 2, self.y + j * 14, w, 9, "", ACT) for j in range(lanes)]
        clip = f" — bar clipped (real {real:.0f}px)" if real > w else ""
        half = max(w / 2, self.op_w / 2) + 20
        lx = self.cx - half - self.lbl if (self.labels_left if left is None else left) else self.cx + half
        node(self._id("note-lbl"), lx, self.y - 12, self.lbl, 34, f"`{label}`{clip}")
        self._link(ids[0], ids[-1])
        self.y += 9 + (lanes - 1) * 14 + 40
        return ids[-1]


# ======================================================================= model.canvas
ROW_W, ROW_H, ROW_GAP = 300, 34, 6
COL_X = (0,)  # one column: repeated layers are drawn once with ×N (see segments())
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

node("cap-title", -900, -1300, 1250, 200,
     "# DeepSeek-V4-Pro-0813 — model\n"
     f"One continuous stack L0–{N - 1}, read top → bottom; layers that repeat are drawn once with **×N**. "
     "**Grey bars = tensors**, length ∝ channels per token (0.08 px/channel); stacked bars = the "
     f"{HC} mHC residual lanes. **Boxes = operations**, all one size; where parameters and compute live is the pair "
     "of bars above. T = sequence length. *Layer* tab = one decoder layer in detail.")
node("cap-legend", 450, -1300, 1250, 200,
     f"Every layer: {H} query heads over **one shared {HD}-dim KV head**; each query sees the last **{WIN} raw tokens** "
     "(sliding window) plus **compressed KV** blocks.  \n"
     f"Row colours — **cyan**: ÷4 compression ({N4} layers), a small indexer picks the top {TOPK} blocks · "
     f"**grey**: ÷128 compression ({N128} layers), all blocks are read. "
     f"MoE everywhere = {A} of {R} routed experts + 1 shared; L0–{NHASH - 1} route by a fixed token-id table (hash). "
     "The barcode on the right lists every layer in order.")

# ---------- input flow ----------
cx1 = COL_X[0] + ROW_W / 2
f = Flow(cx1, -980)
f.op("**raw text**  \"The cat sat…\"", ACT, key="raw")
f.op(f"**tokenizer** (vocab {VOCAB:,})", ACT, key="tok")
f.tensor(40 / 0.08, "token ids [T] — one integer per token", key="ids")
f.op("**embed** — look up row *id*", EMB, P["embed"], note=f"table {VOCAB} × {DIM}", key="embed")
f.tensor(DIM, f"x [T, {DIM}]", key="x")
f.op(f"**expand ×{HC}** — copy into {HC} residual lanes (mHC)", HCC, key="hcx")
f.tensor(DIM, f"h [T, {HC}, {DIM}]", key="h", lanes=HC)
ROW0 = f.y + 30


def kind(i):
    return RATIO[i], i < NHASH


def kind_label(k):
    att = f"KV ÷4 · top-{TOPK}" if k[0] == 4 else "KV ÷128 · all blocks"
    return att + (" · **hash** routing" if k[1] else "")


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
prev = f.last
for seg in segments():
    if seg[0] == "run":
        _, a, n = seg
        span = f"**L{a}–{a + n - 1}** ×{n}" if n > 1 else f"**L{a}**"
        nid = node(f"seg{a}", CX - SEG_W / 2, y, SEG_W, SEG_H, f"{span}\n{kind_label(kind(a))}", ROW_COLOR[RATIO[a]])
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
                   f"{kind_label(kind(s0 + j))}\nL{ls[0]}, {ls[1]}, …, {ls[-1]}", ROW_COLOR[RATIO[s0 + j]])
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
     f"**Every layer, L0 → L{N - 1}** (1 cell = 1 layer; cyan ÷4, grey ÷128; amber line = hash routing)")
for i in range(N):
    node(f"lyr{i}", BX + i * (CELL + CELL_GAP), BY, CELL, 44, "", ROW_COLOR[RATIO[i]])
for i in sorted(set(range(0, N, 10)) | ({N - 1} if (N - 1) % 10 >= 4 else set())):
    node(f"note-tick{i}", BX + i * (CELL + CELL_GAP) - 4, BY + 50, 60, 30, f"L{i}")
node("grp-hash", BX - 4, BY - 6, NHASH * (CELL + CELL_GAP) + 6, 56, "", "#fbbf24", group=True)
node("note-pattern", BX, BY + 100, 700, 150,
     f"Compression pattern (config compress_ratios): L0–1 ÷128, then ÷4 / ÷128 alternate (even layers ÷4). "
     f"÷4 layers pool every 4 tokens (overlapping windows) into one {HD}-dim latent and index them; ÷128 layers "
     f"pool 128 tokens into one — at 1M tokens that is {C['max_position_embeddings'] // 128:,} blocks, all attended. "
     f"L0–{NHASH - 1} (amber frame) route experts by a fixed tid2eid table {VOCAB}×{A}.")

# ---------- output flow (below the right column) ----------
o = Flow(CX, STACK_END + 40, last=LAST, labels_left=True)
o.tensor(DIM, f"h [T, {HC}, {DIM}]  after L{N - 1}", key="xo", lanes=HC)
o.op(f"**mHC head** — merge {HC} lanes (sigmoid weights)", HCC, P["hchead"], key="hch")
o.tensor(DIM, f"x [T, {DIM}]", key="xm")
o.op("**final RMSNorm**", NORM, key="fn")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm head** — {DIM} → {VOCAB}", HEAD, P["head"], note="not tied to embed", key="head")
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op("**softmax → sample**", ACT, key="samp")
next_id = o.op("**next token id**", ACT, key="next")

# ---------- DSpark: a block drafter stored under mtp.* and fed by the last three layers ----------
d = Flow(3000, STACK_END + 160)
cat = d.tensor(DIM * len(TIDS), f"lane-mean of L{TIDS[0]}–{TIDS[-1]}, concat [T, {DIM * len(TIDS)}]", key="dcat")
for j, t in enumerate(TIDS):
    tap = node(f"note-tap{t}", BX + N * (CELL + CELL_GAP) + 40, BY + 300 + j * 50, 260, ROW_H, f"↳ output of L{t} → DSpark")
    edge(tap, cat, ("right", "top"), color=DSPARK)
d.op(f"**main_proj** {DIM * len(TIDS)} → {DIM} + RMSNorm", DSPARK, P["mproj"], key="dproj")
d.tensor(DIM, f"main_x [T, {DIM}] → keys/values of the draft window", key="dmx")
blk = d.op(f"**{NMTP} DSpark blocks** — mHC, window attention ({WIN}) over main_x + the {NB} draft slots, "
           f"MoE {A}/{R}", DSPARK, P["mblocks"], key="dblk", h=80)
emb = node("dembed", d.cx + 230, nodes[-1]["y"], 300, 80,
           f"**embed** (main's table): [next token, {NB - 1}× noise id {C['dspark_noise_token_id']}] → ×{HC} lanes", EMB)
edge(emb, blk, ("left", "right"))
d.op(f"**mHC head → norm → lm head** (main's) + Markov bias (rank {C['dspark_markov_rank']})", DSPARK,
     P["mtphead"] + P["markov"], key="dhead", h=80)
d.op("**confidence head** — one score per draft", DSPARK, P["conf"], key="dconf")
d.tensor(40 / 0.08, f"draft ids [{NB}] + confidences", key="dids")
ver = d.op(f"**verify**: main model checks all {NB} drafts in one forward", ACT, key="dver")
edge(ver, next_id, ("left", "right"), "accepted drafts = extra tokens", DSPARK)
node("note-dspark", d.cx - 150, d.y - 10, 640, 150,
     f"DSpark = speculative block drafter ({NMTP} stages, {fmt(P['mtp'])} stored, checkpoint namespace mtp.0–{NMTP - 1}). "
     f"It reads the main model's hidden states and drafts {NB} tokens per round; the main model keeps the drafts it "
     f"agrees with. config.json says num_nextn_predict_layers = {C['num_nextn_predict_layers']}, but the checkpoint "
     f"and inference/config.json (n_mtp_layers = {NMTP}) have {NMTP} stages. Verify/accept is done by the serving "
     "stack; the repo's code ships the draft forward only.")

# ---------- sources ----------
main = sum(r[2] for r in B) - P["mtp"]
node("note-src", -1400, ROW0 + 40, 1100, 230,
     f"**Sources.** config.json, inference/config.json and safetensors headers of deepseek-ai/DeepSeek-V4-Pro-0813 "
     f"({len(S):,} tensors). Attention / shared experts FP8 E4M3 (128×128 block scales, not counted); routed experts "
     f"FP4 e2m1 packed 2 per byte in I8 tensors, **counted as 2 params per byte** (1/32 scales not counted). "
     f"Main model = **{fmt(main)}**, +{fmt(P['mtp'])} DSpark. Dataflow from the repo's inference/model.py. "
     f"Context {C['max_position_embeddings']:,} = {C['rope_scaling']['original_max_position_embeddings']:,} × YaRN "
     f"{C['rope_scaling']['factor']}.")

save("model.canvas")

# ======================================================================= block.canvas
KT, CLIP, LBL = 0.05, 480, 300
IK, IQ, QC, KC, CC = -2700, -1850, -950, 0, 950  # indexer-key, indexer-query, query, window KV, compressed KV
BL = 4  # drawn layer: a ÷4 layer with learned (score-based) routing
assert RATIO[BL] == 4 and BL >= NHASH


def lp(pat):
    return params_of(rf"layers\.{BL}\.{pat}")


def flow(cx, y, last=None, left=False):
    return Flow(cx, y, last=last, labels_left=left, kt=KT, clip=CLIP, lbl=LBL)


node("cap-btitle", -2900, -760, 1700, 190,
     "# DeepSeek-V4-Pro-0813 — one decoder layer\n"
     f"L{BL}, a ÷4 layer with its own indexer and learned routing. Grey bars = tensors ({KT} px/channel, long ones "
     f"clipped; stacked = {HC} mHC lanes), boxes = operations with their weights per layer. Read top → bottom; the five "
     "middle columns run side by side.")
node("cap-bnote", -1100, -760, 1700, 190,
     f"**Layer variants.** {N128} of {N} layers compress ÷128 instead: no indexer (left two columns skipped), "
     f"compressor {DIM}→{HD} without overlap ({fmt(params_of(r'layers\.3\.attn\.compressor\..*'))}), and every block "
     f"is attended. L0–{NHASH - 1} replace the learned top-{A} pick by a fixed token-id → expert table (tid2eid). "
     f"Attention, MoE and mHC are otherwise identical everywhere.")

mf = flow(0, -520)
mf.tensor(DIM, f"h [T, {HC}, {DIM}] from previous layer", key="bh", lanes=HC)
mf.op(f"**mHC pre** — hc_attn_fn {HC * DIM}→{(2 + HC) * HC}, Sinkhorn ×{C['hc_sinkhorn_iters']} → "
      f"pre / post / comb; x = Σ pre·lanes", HCC, lp(r"hc_attn_.*"), key="hpa", h=80)
mf.tensor(DIM, f"x [T, {DIM}]", key="bx")
mf.op("**RMSNorm** (attn_norm)", NORM, key="bn1")
xn = mf.tensor(DIM, f"x̂ [T, {DIM}]", key="bxn")
YB = mf.y + 40

# query path: low-rank query, per-head RMS scaling, RoPE on the last 64 dims
q = flow(QC, YB, last=xn)
q.op(f"**wq_a** {DIM}→{QL} + q_norm", ATT, lp(r"attn\.(?:wq_a|q_norm)\..*"), key="qa")
cq = q.tensor(QL, f"qr [T, {QL}]  query latent", key="cq")
q.op(f"**wq_b** {QL} → {H}×{HD}", ATT, lp(r"attn\.wq_b\..*"), key="qb")
q.tensor(H * HD, f"q [T, {H}, {HD}]", key="q")
q.op(f"per-head RMS scale · **RoPE** on last {ROPE}", ATT, key="qr")
q_out = q.tensor(H * HD, f"q [T, {H}, {HD}]", key="qo")

# window KV: one shared head per token, used as both key and value
kv = flow(KC, YB, last=xn)
kv.op(f"**wkv** {DIM}→{HD} + kv_norm", ATT, lp(r"attn\.(?:wkv|kv_norm)\..*"), key="kva")
kv.tensor(HD, f"kv [T, {HD}]  (1 head, K = V)", key="kvl")
kv.op(f"**RoPE** last {ROPE} · FP8-round other {HD - ROPE}", ATT, key="kvn")
kv_out = kv.tensor(HD, f"window cache: last {WIN} tokens × {HD}", key="kvc")

# compressed KV: gated pooling over 4 tokens, overlapping windows → one latent per 4 tokens
cc = flow(CC, YB, last=xn)
cc.op(f"**compressor** wkv, wgate {DIM}→{2 * HD} + ape", CMP, lp(r"attn\.compressor\..*"), key="cwk")
cc.tensor(2 * HD, f"kv, score [T, 2×{HD}]  (overlap halves)", key="cks")
cc.op("**softmax-gated pool** of 8 slots (2 windows of 4) → RMSNorm · RoPE", CMP, key="cpool", h=80)
c_out = cc.tensor(HD, f"compressed KV [T/4, {HD}]", key="cko")

# indexer: scores the compressed blocks with its own small compressor → top-k block ids
ik = flow(IK, YB, last=xn, left=True)
ik.op(f"**indexer compressor** {DIM}→{2 * ID} ×2 + ape · Hadamard", IDX, lp(r"attn\.indexer\.compressor\..*"), key="iwk", h=80)
edges[-1].update(fromSide="left")  # leave x̂ sideways so the curve stays above the query column
k_i = ik.tensor(ID, f"k_I [T/4, {ID}] (cached, FP4-rounded)", key="ik")
ik.op(f"**weights_proj** {DIM}→{IH}", IDX, lp(r"attn\.indexer\.weights_proj\..*"), key="iwp")
w_i = ik.tensor(IH, f"w [T, {IH}]  per-head weights", key="iw")
iq = flow(IQ, YB + 56 + 40 + 9 + 40, last=cq)  # level with wq_b, fed sideways by qr
iq.op(f"**indexer wq_b** {QL} → {IH}×{ID}", IDX, lp(r"attn\.indexer\.wq_b\..*"), key="iwq")
edges[-1].update(fromSide="left", toSide="right")  # qr feeds the indexer query sideways
iq.tensor(IH * ID, f"q_I [T, {IH}, {ID}]", key="iqt")
isc = iq.op(f"**RoPE · Hadamard** · score = Σ_h w_h·ReLU(q_I,h·k_I)", IDX, key="isc", h=80)
edge(k_i, isc, ("right", "left"))
edge(w_i, isc, ("right", "left"))
iq.tensor(80 / KT, "block scores [T, T/4]  (causal)", key="isco")
iq.op(f"**top-{TOPK}** blocks per query", IDX, key="itop")
idx_out = iq.tensor(160 / KT, f"block ids [T, {TOPK}]", key="iids")

# sparse attention core
YA = max(q.y, kv.y, iq.y, cc.y, ik.y) + 60
core = flow(0, YA)
att = core.op(f"**sparse attention** — {H} heads, 1 KV head; keys = {WIN} window + {TOPK} picked blocks; "
              "learned per-head sink", ATT, lp(r"attn\.attn_sink"), key="att", h=80)
edge(q_out, att, ("bottom", "left"))
edge(kv_out, att)
edge(c_out, att, ("bottom", "right"))
edge(idx_out, att, ("bottom", "left"))
core.tensor(H * HD, f"o [T, {H}, {HD}]", key="ao")
core.op(f"inverse **RoPE** last {ROPE} · split into {OG} groups of {H // OG} heads", ATT, key="og")
core.tensor(H * HD, f"o [T, {OG}, {H * HD // OG}]", key="ao1")
core.op(f"**wo_a** per group {H * HD // OG} → {OL}", ATT, lp(r"attn\.wo_a\..*"), key="woa")
core.tensor(OG * OL, f"[T, {OG}×{OL}]", key="ao2")
core.op(f"**wo_b** {OG * OL} → {DIM}", ATT, lp(r"attn\.wo_b\..*"), key="wob")
core.tensor(DIM, f"[T, {DIM}]", key="ao3")
core.op("**mHC post** — lanes = post·out + comb·lanes", HCC, key="hpo1")
core.tensor(DIM, f"h [T, {HC}, {DIM}]", key="h1", lanes=HC)
core.op(f"**mHC pre** (hc_ffn_fn, own weights) → x", HCC, lp(r"hc_ffn_.*"), key="hpf")
core.tensor(DIM, f"x [T, {DIM}]", key="h2")
core.op("**RMSNorm** (ffn_norm)", NORM, key="bn2")
hn = core.tensor(DIM, f"x̂ [T, {DIM}]", key="hn", left=True)
rt = core.op(f"**router** {DIM}→{R}, √softplus · top-{A} by score+bias · weights normalised ×{C['routed_scaling_factor']}",
             FFN, P["router"], key="rt", h=80)
sh = node("shexp", 350, core.y - 80 - 40, 300, 80,
          f"**shared expert** (always on)  \nSwiGLU {DIM}→{MI}→{DIM}, FP8  \n{fmt(P['shexp'])}", "#22c55e")
edge(hn, sh, ("right", "top"))
core.tensor(40 / KT, f"{A} expert ids + weights / token", key="rid")
ex = core.op(f"**{A} of {R} routed experts**, each SwiGLU {DIM}→{MI}→{DIM}, FP4, clamp ±{C['swiglu_limit']:g}",
             FFN, P["expert"], note=f"each · {fmt(R * P['expert'])} all", key="rex", h=80)
sm = core.op("**Σ** weighted experts + shared expert", ACT, key="sum")
edge(sh, sm, ("bottom", "right"))
core.tensor(DIM, f"[T, {DIM}]", key="fo")
core.op("**mHC post** — lanes = post·out + comb·lanes", HCC, key="hpo2")
core.tensor(DIM, f"h [T, {HC}, {DIM}] → next layer", key="bout", lanes=HC)
node("note-kv", CC + 260, YA - 10, 520, 150,
     f"Why so cheap at 1M tokens: the cache keeps one {HD}-dim vector per token for the last {WIN} tokens, one per "
     f"4 tokens (÷4 layers) or per 128 tokens (÷128 layers) beyond that, plus a {ID}-dim indexer key per block. "
     f"Each query reads at most {WIN} + {TOPK} vectors in ÷4 layers.")

save("block.canvas")
print("budget", [(r[0], fmt(r[2]), fmt(r[3])) for r in B])
print("active total", sum(r[3] for r in B), fmt(sum(r[3] for r in B)), "stored total", sum(r[2] for r in B),
      fmt(sum(r[2] for r in B)), "all tensors", params_of(r".*"))
