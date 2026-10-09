"""Model-level tab of DeepSeek-V4.1-Flash: dataflow, all 40 layers, and parameter sizes in one canvas.

Read top → bottom, encoder column then decoder column:
  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale everywhere)
  - boxes       = operations, all one size; parameter / compute shares are the two bars at the top
  - layer rows  = one per layer, coloured by where its attention KV comes from (config.json)
  - note-* ids  = annotations (styled dim, borderless by web/styles.css); cap-* ids = borderless captions
Everything is derived from the checkpoint: shapes.json (safetensors headers) + config.json.

usage: python3 deepseek_v4_1_flash_model_canvas.py <shapes.json> <config.json> <models/deepseek-v4-1-flash>
"""
import json
import math
import os
import re
import sys

S = json.load(open(sys.argv[1]))
C = json.load(open(sys.argv[2]))
OUT = sys.argv[3]

N, HALF = C["n_layers"], C["n_layers"] // 2
RATIO = C["compress_ratios"]
KV_SRC, IDX_SRC = set(C["kv_source_layers"]), set(C["index_source_layers"])
ENGRAM = C["engram_layer_ids"]
DIM, VOCAB = C["dim"], C["vocab_size"]

# colours (JSON Canvas presets "1"–"6" or hex)
EMB, HC, NORM, FFN, HEAD, ACT, ENG = "6", "1", "3", "4", "2", "#64748b", "#e879f9"
DSPARK, VISION = "#f59e0b", "#a78bfa"
BAND_COLOR = {"encoder": "#38bdf8", "decoder": "#a3e635"}
MODE_COLOR = {"SWA": "#94a3b8", "Full": "5", "Reindex": "6", "Reuse": None}

# geometry
KT, CLIP_W = 0.08, 1800  # tensor bars: px per channel, longest drawn bar
ROW_W, ROW_H, ROW_GAP = 300, 34, 6
ENC_X, DEC_X = 0, 900  # left edge of the two layer columns
OP_W, OP_H, GAP = 300, 56, 40
BAR_W, BAR_H = 2600, 44  # parameter / compute budget bars (each is 100% of its total)
LBL_W = 440  # tensor label width
MIN_SEG = 16  # the viewer draws no box narrower than ~14px, so thinner budget segments would overlap


def params_of(pattern):
    rx = re.compile(pattern)
    return sum(math.prod(shp) * (2 if dt == "I8" else 1)  # I8 packs two FP4 values
               for k, (dt, shp) in S.items() if rx.fullmatch(k) and not k.endswith(".scale"))


P = {
    "vit": params_of(r"vision\..*"),
    "aligner": params_of(r"aligner\..*"),
    "embed": params_of(r"embed\.weight"),
    "head": params_of(r"head\.weight"),
    "dspark": params_of(r"mtp\.0\..*"),
}


LAYER_P = [params_of(rf"layers\.{i}\.(?!engram\.).*") for i in range(N)]
ENGRAM_P = {li: params_of(rf"layers\.{li}\.engram\..*") for li in ENGRAM}


def budget():
    """[(name, colour, stored params, params multiplied per text token)] for the whole checkpoint.

    Per-token compute counts weights a token is actually multiplied by: only the activated routed
    experts, nothing for table lookups (embed rows, Engram rows) or for the vision tower (runs per image).
    """
    R, A = C["n_routed_experts"], C["n_activated_experts"]
    layers = params_of(r"layers\.\d+\.(?!engram\.).*")
    expert = params_of(r"layers\.0\.ffn\.experts\.0\..*")
    engram = params_of(r"layers\.\d+\.engram\..*")
    engram_mm = params_of(r"layers\.\d+\.engram\.(?!embed\.).*")  # projections + gate; table rows are read, not multiplied
    mtp = params_of(r"mtp\..*")
    mtp_expert = params_of(r"mtp\.0\.ffn\.experts\.0\..*")
    mtp_skip = C["n_mtp_layers"] * (C["dspark_n_routed_experts"] - C["dspark_n_activated_experts"]) * mtp_expert
    vision = P["vit"] + P["aligner"]
    rows = [
        (f"{N} layers", FFN, layers, layers - N * (R - A) * expert),
        ("Engram ×2", ENG, engram, engram_mm),
        # drafting runs beside the main forward, so it is shown but never summed into "active per text token"
        ("DSpark ×3", DSPARK, mtp, 0, f"draft model, {fmt(mtp - mtp_skip)} per draft step, not counted"),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, P["head"], P["head"]),
        ("ViT + aligner", VISION, vision, 0, "runs per image"),
    ]
    other = params_of(r".*") - sum(r[2] for r in rows)
    if other >= 1e6:  # top-level norms / mHC heads, when they are not negligible
        rows.append(("other", ACT, other, other))
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


_seq = [0]


class Flow:
    """A vertical chain of ops and tensors centred on cx; each item is linked to the previous one."""

    def __init__(self, cx, y, last=None, labels_left=False):
        self.cx, self.y, self.last, self.labels_left = cx, y, last, labels_left

    def _id(self, key):
        _seq[0] += 1
        return f"{key}{_seq[0]}"

    def _link(self, top, bottom, label=None):
        if self.last:
            edge(self.last, top, label=label)
        self.last = bottom

    def op(self, text, color, params=0, note="", key="op"):
        """Operation box; sizes are told by the budget bars, so every box has the same size."""
        if params:
            text += f"\n{fmt(params)}" + (f" · {note}" if note else "")
        nid = node(self._id(key), self.cx - OP_W / 2, self.y, OP_W, OP_H, text, color)
        self._link(nid, nid)
        self.y += OP_H + GAP
        return nid

    def tensor(self, dim, label, lanes=1, key="t"):
        w = dim * KT
        real = w
        w = min(w, CLIP_W)
        ids = [node(self._id(key), self.cx - w / 2, self.y + j * 14, w, 9, "", ACT) for j in range(lanes)]
        clip = f" — bar clipped (real {real:.0f}px)" if real > w else ""
        half = max(w / 2, OP_W / 2) + 20
        lx = self.cx - half - LBL_W if self.labels_left else self.cx + half
        node(self._id("note-lbl"), lx, self.y - 10, LBL_W, 34, f"`{label}`{clip}")
        self._link(ids[0], ids[-1])
        self.y += lanes * 14 + GAP
        return ids[0], ids[-1]


def mode(i):
    if RATIO[i] == 0:
        return "SWA"
    if i in KV_SRC:
        return "Full"
    if i in IDX_SRC:
        return "Reindex"
    return "Reuse"


# ---------- header ----------
node("cap-title", -900, -1270, 1250, 180,
     "# DeepSeek-V4.1-Flash — model\n"
     f"Read top → bottom, left column (L0–{HALF - 1}) then right column (L{HALF}–{N - 1}). "
     f"**Grey bars = tensors**, length ∝ channels per token ({KT} px/channel). "
     "**Boxes = operations**, all the same size; where the parameters and the compute live is the pair of bars above. "
     "T = sequence length. Zoom in: *Layer* tab = one layer, *Attention* tab = its attention.")
def extra(m):
    """Extra parameters a layer of this mode carries over a Reuse layer."""
    ls = [LAYER_P[i] for i in range(N) if mode(i) == m]
    return fmt(max(ls) - min(LAYER_P)) if ls else "0"


node("cap-legend", 450, -1270, 1250, 180,
     f"Layer rows — **SWA**: sliding window ({C['window_size']} tokens) only · **Full**: computes its own compressed KV + "
     f"indexer, shared by the next layers · **Reindex**: reuses that KV, runs a fresh indexer (top {C['index_topk']}) · "
     "**Reuse**: reuses KV and index picks · **+E**: reads an Engram table · **▸ DSpark**: output also feeds the draft model.  \n"
     f"Each layer ≈ **{fmt(min(LAYER_P))}** (Full +{extra('Full')} for its KV compressor + indexer, Reindex +{extra('Reindex')}); "
     f"all {N} = **{fmt(sum(LAYER_P))}**, ~{params_of(r'layers\.0\.ffn\.experts\..*') / LAYER_P[0]:.0%} of it in the "
     f"{C['n_routed_experts']} routed experts, of which {C['n_activated_experts']} run per token.")

# ---------- input flow (above the encoder column) ----------
cx_enc = ENC_X + ROW_W / 2
f = Flow(cx_enc, -960)
f.op("**raw text**  \"The cat sat…\"", ACT, key="raw")
f.op(f"**tokenizer** (BPE, vocab {VOCAB})", ACT, key="tok")
f.tensor(40 / KT, "token ids [T] — one integer per token", key="ids")  # drawn as a short stub
f.op("**embed** — look up row *id*", EMB, P["embed"], note=f"table {VOCAB} × {DIM}", key="embed")
x0, _ = f.tensor(DIM, f"x [T, {DIM}]  (+ image tokens spliced in)", key="x")
f.op(f"**expand ×{C['hc_mult']}** — copy into {C['hc_mult']} residual lanes (mHC)", HC, key="hcx")
f.tensor(DIM, f"h [T, {C['hc_mult']}, {DIM}]", lanes=C["hc_mult"], key="h")
ROW0 = f.y + 30


def row_xy(i):
    col, r = divmod(i, HALF)
    return (ENC_X if col == 0 else DEC_X), ROW0 + r * (ROW_H + ROW_GAP)


# ---------- layer rows ----------
prev = f.last
for i in range(N):
    x, y = row_xy(i)
    m = mode(i)
    tag = ("  **+E**" if i in ENGRAM else "") + ("  **▸ DSpark**" if i in C["dspark_target_layer_ids"] else "")
    nid = node(f"L{i}", x, y, ROW_W, ROW_H, f"**L{i}** · {m}{tag}", MODE_COLOR[m])
    if i == HALF:
        # hand-off tensor under the encoder column: its edge to L20 starts below the KV boxes
        # instead of leaving L19 sideways behind them
        _, yl = row_xy(HALF - 1)
        hand = Flow(ENC_X + ROW_W / 2, yl + ROW_H + 60, last=prev, labels_left=True)
        _, prev = hand.tensor(DIM, f"h [T, {C['hc_mult']}, {DIM}]  after L{HALF - 1} → L{HALF}", lanes=C["hc_mult"], key="hm")
        edge(prev, nid, ("right", "left"))
    else:
        edge(prev, nid)
    prev = nid

for col, (lo, hi, name) in enumerate(((0, HALF, "encoder"), (HALF, N, "decoder"))):
    x, y0 = row_xy(lo)
    _, y1 = row_xy(hi - 1)
    # coloured band + short label; the compression ratios are a borderless note beside the band
    node(f"band{col}", x - 16, y0 - 30, ROW_W + 32, y1 - y0 + ROW_H + 46, f"{name.upper()} L{lo}–{hi - 1}",
         BAND_COLOR[name], group=True)
    parts = []
    for r in sorted({RATIO[i] for i in range(lo, hi)}, reverse=True):
        ls = [i for i in range(lo, hi) if RATIO[i] == r]
        span = f"L{ls[0]}–{ls[-1]}" if len(ls) > 1 else f"L{ls[0]}"
        parts.append(f"{span}: " + (f"compressed KV, {r} token{'s' if r > 1 else ''} → 1 latent" if r else "sliding window only"))
    # encoder: beside the window-only rows L0–1 (free space); decoder: above its KV box
    info_y = y0 - 20 if RATIO[lo] == 0 else y0 - 110
    # an annotation of the band (its label already names it), so no heading of its own
    node(f"note-bandinfo{col}", x + ROW_W + 30, info_y + 30, 420, 66, "\n".join(f"- {p}" for p in parts))

for s in sorted(KV_SRC):  # which layers share whose compressed KV
    members = [i for i in range(N) if RATIO[i] == RATIO[s] and max(k for k in KV_SRC if k <= i and RATIO[k] == RATIO[i]) == s]
    x, y0 = row_xy(members[0])
    _, y1 = row_xy(members[-1])
    node(f"kv{s}", x + ROW_W + 30, y0, 150, y1 - y0 + ROW_H, f"**KV from L{s}**\nratio {RATIO[s]}\nL{members[0]}–{members[-1]}", "5")

cs = C.get("candidate_source_layer", -1)
if cs >= 0:
    x, y = row_xy(cs)
    node("cand", x + ROW_W + 200, y, 300, 70,
         f"**candidate pre-filter** (L{cs})\nkeeps top {C['candidate_topk_blocks']} blocks of {C['candidate_block_size']}", "5")

# ---------- Engram tables, wired to their layer ----------
for li, rows in zip(ENGRAM, C["engram_num_embeddings"]):
    _, y = row_xy(li)
    node(f"eng{li}", ENC_X - 640, y - 20, 340, 76,
         f"**Engram table (L{li})** · {fmt(ENGRAM_P[li])}\n{rows / 1e6:.0f}M rows × {C['engram_head_dim']} · "
         f"n-grams ≤ {C['engram_max_ngram_size']} · {C['engram_n_heads']} hash heads", ENG)
    edge(f"eng{li}", f"L{li}", ("right", "left"), "reads a few rows / token", ENG)

# ---------- vision branch, spliced into x ----------
# sits right beside the text path, labels on its outer side so they stay clear of the main column
v = Flow(ENC_X - 480, -960, labels_left=True)
v.op("**image** pixels", ACT, key="img")
v.tensor(C["vision_patch_size"] ** 2 * 3, f"patches [P, {C['vision_patch_size'] ** 2 * 3}]", key="patch")
v.op(f"**ViT ×{C['vision_n_layers']}**", VISION, P["vit"], key="vit")
v.tensor(C["vision_dim"], f"[P, {C['vision_dim']}]", key="vf")
r = C["vision_downsample_ratio"]
v.op(f"**aligner MLP** — merge {r}×{r} patches", VISION, P["aligner"], key="ali")
v.tensor(DIM, f"image tokens [P/{r * r}, {DIM}]", key="vt")
edge(v.last, x0, ("right", "left"), "spliced into x")

# ---------- output flow (below the decoder column) ----------
_, ylast = row_xy(N - 1)
o = Flow(DEC_X + ROW_W / 2, ylast + ROW_H + 60, last=f"L{N - 1}")
_, h40 = o.tensor(DIM, f"h [T, {C['hc_mult']}, {DIM}]  after L{N - 1}", lanes=C["hc_mult"], key="ho")
o.op(f"**merge {C['hc_mult']} lanes → final norm**", NORM, key="hcf")
o.tensor(DIM, f"x [T, {DIM}]", key="xf")
o.op(f"**lm head** — {DIM} → {VOCAB}", HEAD, P["head"], key="head")
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op("**softmax → sample**", ACT, key="samp")
next_id = o.op("**next token id(s)**", ACT, key="next")

# ---------- DSpark: a small draft model fed by the last three layers ----------
# The decoder's KV box sits right of L20–39, so a tap marker is placed beside each tapped row (past
# the KV box) and the draft model's flow starts below the rows — no edge runs behind another box.
tids = C["dspark_target_layer_ids"]
nb, R, A = C["dspark_block_size"], C["dspark_n_routed_experts"], C["dspark_n_activated_experts"]
_, ybot = row_xy(N - 1)
tap_x = DEC_X + ROW_W + 30 + 150 + 16
d = Flow(DEC_X + 1720, ybot + ROW_H + 110)  # right of the logits bar and its label
cat, _ = d.tensor(DIM * len(tids), f"concat hidden of L{tids[0]}–{tids[-1]}  [T, {DIM * len(tids)}]", key="dcat")
for t in tids:
    _, ty = row_xy(t)
    # an annotation, not a node of its own: it marks where L{t}'s output is read off
    tap = node(f"note-tap{t}", tap_x, ty, 230, ROW_H, f"↳ output of L{t} → DSpark")
    edge(tap, cat, ("right", "top"), color=DSPARK)
d.op(f"**main_proj** {DIM * len(tids)} → {DIM} + norm", DSPARK, key="dproj")
d.op(f"**DSpark draft model** — {C['n_mtp_layers']} blocks\n"
     f"MoE top-{A} of {R} · {fmt(P['dspark'])} per block", DSPARK, key="dblk")
d.op(f"**head + Markov bias** → {nb} draft tokens\nand a confidence score", DSPARK, key="dhead")
d.tensor(40 / KT, f"draft ids [{nb}]", key="dids")
ver = d.op(f"**verify**: big model runs all {nb} drafts\nin ONE forward pass", ACT, key="dver")
# the loop closes on the main output: accepted drafts become emitted tokens
edge(ver, next_id, ("left", "right"), "keeps the drafts it agrees with", DSPARK)
node("note-dspark", d.cx - OP_W / 2, d.y - 20, 600, 110,
     f"Speculative decoding. Without DSpark: 1 big forward per token. With it: the small model guesses "
     f"{nb} tokens, the big model checks them together; every draft it agrees with is a token gained without "
     "its own big forward. The verify/accept loop is the standard scheme — this repo's code ships only "
     "the draft model's forward pass.")

# ---------- budget: where the parameters live vs. what one token actually multiplies ----------
def bar(key, y, title, rows, idx):
    total = sum(r[idx] for r in rows)
    node(f"cap-{key}t", -900, y - 46, BAR_W, 40, f"**{title}** — total {fmt(total)}")
    x = -900
    legend = []
    # segments too thin to draw go last, so any blank stretch is at the bar's end, not mid-bar
    for k in sorted(range(len(rows)), key=lambda k: BAR_W * rows[k][idx] / total < MIN_SEG):
        r = rows[k]
        share = r[idx] / total
        w = BAR_W * share
        if w >= MIN_SEG:  # integer edges so neighbouring segments meet exactly
            node(f"{key}{k}", round(x), y, round(x + w) - round(x), BAR_H, f"**{r[0]}**" if w > 150 else "", r[1])
        x += w
        why = r[4] if len(r) > 4 and not r[idx] else ""
        legend.append(f"{r[0]} **{fmt(r[idx]) if r[idx] else '0'}**" + (f" ({why})" if why else f" ({share:.1%})"))
    node(f"cap-{key}l", -900, y + BAR_H + 6, BAR_W, 40, " · ".join(legend))


B = budget()
bar("bp", -1640, "Where the parameters are stored", B, 2)
bar("bc", -1460, "What one text token is multiplied by (active weights)", B, 3)

json.dump(dict(nodes=nodes, edges=edges), open(os.path.join(OUT, "model.canvas"), "w"), indent=1, ensure_ascii=False)
print("ok", [(r[0], fmt(r[2]), fmt(r[3])) for r in budget()])
