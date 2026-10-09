"""LLaDA2.2-mini (inclusionAI, masked-diffusion MoE LM): model.canvas + block.canvas.

Read top → bottom:
  - grey bars   = tensors between ops; bar length ∝ channels per token (one scale everywhere)
  - boxes       = operations; where parameters / per-token compute live is the pair of bars at the top
  - note-* ids  = annotations (dim, borderless); cap-* ids = borderless titles / legends
Every number comes from the checkpoint: shapes.json (safetensors headers, fetch_safetensors_shapes.py),
config.json, and the repo's modeling_llada2_moe.py (generate / _joint_decode_block / LLaDA2MoeGate).

usage: python3 llada_2_2_mini_model_canvas.py <shapes.json> <config.json> <models/llada-2-2-mini>
"""
import json
import math
import os
import re
import sys

S = json.load(open(sys.argv[1]))
C = json.load(open(sys.argv[2]))
OUT = sys.argv[3]

N, DIM, VOCAB = C["num_hidden_layers"], C["hidden_size"], C["vocab_size"]
NH, NKV, HD = C["num_attention_heads"], C["num_key_value_heads"], C["head_dim"]
E, K, CAP, BLK = C["num_experts"], C["num_experts_per_tok"], C["expert_capacity"], C["block_size"]
MI, FI, DENSE = C["moe_intermediate_size"], C["intermediate_size"], C["first_k_dense_replace"]
ROPE = int(HD * C["partial_rotary_factor"])
# defaults of LLaDA2MoeModelLM.generate() in modeling_llada2_moe.py (not in config.json)
GEN = dict(mask_id=156895, delete=156930, split=156931, eos=156892, threshold=0.5, editing_threshold=0.0,
           steps=32, max_post_steps=16)

# colours (JSON Canvas presets "1"–"6" or hex)
EMB, ATT, NORM, FFN, HEAD, ACT = "6", "5", "3", "4", "2", "#64748b"
MASK, DIFF, SHARED, DENSEC = "#f59e0b", "#e879f9", "#38bdf8", "#a3e635"

KT, CLIP_W = 0.08, 1100  # tensor bars: px per channel, longest drawn bar
OP_W, OP_H, GAP = 300, 56, 40
LBL_W = 440


def params_of(pattern):
    rx = re.compile(pattern)
    return sum(math.prod(shp) for k, (dt, shp) in S.items() if rx.fullmatch(k))


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


EXPERT = params_of(r"model\.layers\.1\.mlp\.experts\.0\..*")
P = {
    "embed": params_of(r"model\.word_embeddings\.weight"),
    "head": params_of(r"lm_head\.weight"),
    "attn": params_of(r"model\.layers\.\d+\.(attention\..*|input_layernorm\.weight|post_attention_layernorm\.weight)"),
    "dense": params_of(rf"model\.layers\.({'|'.join(str(i) for i in range(DENSE))})\.mlp\..*"),
    "routed": params_of(r"model\.layers\.\d+\.mlp\.experts\..*"),
    "shared": params_of(r"model\.layers\.\d+\.mlp\.(shared_experts\..*|gate\..*)"),
    "norm": params_of(r"model\.norm\.weight"),
    "all": params_of(r".*"),
}
N_MOE = N - DENSE
assert P["all"] == sum(v for k, v in P.items() if k != "all"), "budget rows must cover every tensor"


def budget():
    """[(name, colour, stored, multiplied per token, why-zero)]; only K of E routed experts run per token."""
    return [
        ("attention ×20", ATT, P["attn"], P["attn"]),
        (f"routed experts {E}×{N_MOE}", FFN, P["routed"], N_MOE * K * EXPERT),
        ("shared expert + router", SHARED, P["shared"], P["shared"]),
        (f"dense FFN (L0)", DENSEC, P["dense"], P["dense"]),
        ("embed", EMB, P["embed"], 0, "row lookup"),
        ("lm head", HEAD, P["head"], P["head"]),
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

    def op(self, text, color, params=0, note="", key="op", h=OP_H, w=OP_W):
        if params:
            text += f"\n{fmt(params)}" + (f" · {note}" if note else "")
        nid = node(self._id(key), self.cx - w / 2, self.y, w, h, text, color)
        self._link(nid, nid)
        self.y += h + GAP
        return nid

    def tensor(self, dim, label, key="t", lbl_dy=-10, left=None):
        real = dim * KT
        w = min(real, CLIP_W)
        nid = node(self._id(key), self.cx - w / 2, self.y, w, 9, "", ACT)
        clip = f" — bar clipped ({real:.0f}px)" if real > w else ""
        half = max(w / 2, OP_W / 2) + 20
        left = self.labels_left if left is None else left
        lx = self.cx - half - LBL_W if left else self.cx + half
        # lbl_dy < -10 lifts the label above the bar's bottom, clear of edges that fan out below it
        node(self._id("note-lbl"), lx, self.y + lbl_dy, LBL_W, 34, f"`{label}`{clip}")
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
            node(f"{key}{k}", round(x), y, round(x + w) - round(x), 44, f"**{r[0]}**" if w > 170 else "", r[1])
        x += w
        why = r[4] if len(r) > 4 and not r[idx] else ""
        legend.append(f"{r[0]} **{fmt(r[idx]) if r[idx] else '0'}**" + (f" ({why})" if why else f" ({share:.1%})"))
    node(f"cap-{key}l", x0, y + 50, width, 40, " · ".join(legend))


def save(name):
    json.dump(dict(nodes=nodes, edges=edges), open(os.path.join(OUT, name), "w"), indent=1, ensure_ascii=False)
    nodes.clear()
    edges.clear()


# =====================================================================================================
# model.canvas — whole model + the denoising loop
# =====================================================================================================
B = budget()
BX, BW = -1300, 2600
bar("bp", BX, -2160, BW, "Where the parameters are stored", B, 2)
bar("bc", BX, -1980, BW, "What one token is multiplied by (active weights)", B, 3)
active = sum(r[3] for r in B)

node("cap-title", BX, -1840, 1260, 200,
     "# LLaDA2.2-mini — model\n"
     f"A **masked-diffusion** LM: it does not emit one token per forward pass. Text is produced in blocks of "
     f"{BLK} positions that start as `<|mask|>` and are filled in over several **denoising steps**, each one a full "
     f"forward pass. Grey bars = tensors (length ∝ channels, {KT} px/channel), boxes = operations. "
     "T = prompt + finished blocks + the active block. *Layer* tab = one layer in detail.")
node("cap-legend", BX + 1340, -1840, 1260, 200,
     f"**Numbers**: shapes from the safetensors headers ({len(S)} tensors, BF16), sizes from config.json, decoding "
     f"rules and their defaults from `generate()` / `_joint_decode_block()` in the repo's modeling_llada2_moe.py.  \n"
     f"Active per token ≈ **{fmt(active)}** without the embedding table, {fmt(active + P['embed'])} with it — the "
     f"model card's \"1.4B activated\" matches the second. Stored total **{fmt(P['all'])}** (card: 16B).")

# ---- the working sequence x: prompt | finished blocks | active block | (future) ----
SY = -1320
node("strip", -820, SY - 40, 1640, 150, "x — the working sequence, one token id per position", ACT, group=True)
node("seq-prompt", -800, SY, 300, 90, "**prompt**  \nchat template, already tokens", ACT)
node("seq-done", -480, SY, 300, 90, f"**finished blocks**  \n{BLK} tokens each, frozen", ACT)
node("seq-active", -150, SY, 300, 90, f"**active block**  \n{BLK} × `<|mask|>` at first", MASK)
node("seq-future", 180, SY, 620, 90, f"**later blocks** — still all masks and *not fed* to the model: each step "
     "runs the forward on `x[:, :block_end]` only", None)

f = Flow(0, SY + 150, last="seq-active")
f.tensor(40 / KT, f"token ids [T]  (T = block_end)", key="ids")
f.op("**embed** — look up row *id*", EMB, P["embed"], note=f"{VOCAB} × {DIM}", key="embed")
f.tensor(DIM, f"x [T, {DIM}]", key="x")

ROW_W, ROW_H, ROW_GAP = 380, 34, 6
ROW0 = f.y + 40
prev = f.last
for i in range(N):
    y = ROW0 + i * (ROW_H + ROW_GAP)
    txt = (f"**L{i}** · attention + dense SwiGLU {FI}" if i < DENSE
           else f"**L{i}** · attention + MoE {K}/{E} + 1 shared")
    nid = node(f"L{i}", -ROW_W / 2, y, ROW_W, ROW_H, txt, DENSEC if i < DENSE else FFN)
    edge(prev, nid)
    prev = nid
yend = ROW0 + N * (ROW_H + ROW_GAP) - ROW_GAP
node("band", -ROW_W / 2 - 16, ROW0 - 34, ROW_W + 32, yend - ROW0 + 50, f"{N} layers, identical except L0", "#94a3b8",
     group=True)
lay = params_of(r"model\.layers\.1\..*")
node("note-layers", ROW_W / 2 + 40, ROW0, 440, 190,
     f"Each MoE layer = {fmt(lay)} stored, of which {fmt(E * EXPERT)} in {E} routed experts "
     f"({fmt(EXPERT)} each); a token runs {K} of them plus the shared one.  \n"
     f"Attention: GQA {NH} query / {NKV} KV heads × {HD}, **not causal** (`is_causal = False`).")

# ---- the attention mask actually used by generate(): block-causal ----
MX, MYC, CELL = -1010, ROW0 + 120, 64
node("cap-mask", MX - 10, ROW0 - 60, 520, 130,
     "**Attention mask** (same in all layers) — rows = queries, columns = keys, by block. "
     "Inside a block everything sees everything; a block also sees all *earlier* blocks.")
names = ["P", "B1", "B2", "act"]
for i in range(4):
    node(f"cap-mrow{i}", MX - 10, MYC + i * (CELL + 6) + 16, 60, 34, names[i])
    node(f"cap-mcol{i}", MX + 60 + i * (CELL + 6), MYC + 4 * (CELL + 6), CELL, 34, names[i])
    for j in range(4):
        node(f"mk{i}{j}", MX + 60 + j * (CELL + 6), MYC + i * (CELL + 6), CELL, CELL,
             "↔" if i == j else "", "4" if j <= i else None)
node("note-mask", MX - 10, MYC + 4 * (CELL + 6) + 50, 520, 150,
     f"Built in `generate()`: `tril(ones(blocks, blocks))` expanded to {BLK}×{BLK} tiles. An autoregressive LM "
     "would mask per *token* (strict lower triangle); here a masked position can use tokens to its **right** "
     "inside its block.")

o = Flow(0, yend + 60, last=f"L{N - 1}")
o.tensor(DIM, f"h [T, {DIM}]  after L{N - 1}", key="h")
o.op("**final RMSNorm**", NORM, key="norm")
o.op(f"**lm head** — {DIM} → {VOCAB}", HEAD, P["head"], note="untied", key="head")
o.tensor(VOCAB, f"logits [T, {VOCAB}]", key="logits")
o.op(f"**keep the active block's {BLK} rows**", ACT, key="slice")
o.tensor(VOCAB, f"logits [{BLK}, {VOCAB}]", key="blog")
o.op("**softmax per position** → best token x̂  \nand its probability p (confidence)", DIFF, key="soft", h=76)
o.tensor(40 / KT, f"x̂, p [{BLK}]", key="xp")
m2t = o.op(f"**M2T** · mask → token  \nfill `<|mask|>` slots with p > {GEN['threshold']}; at least the "
           f"schedule's share ({BLK} masks over {GEN['steps']} steps)", DIFF, key="m2t", h=96, w=420)
o.op(f"**T2T** · token → token  \nrewrite an already-filled slot when x̂ differs and p > {GEN['editing_threshold']}",
     DIFF, key="t2t", h=76, w=420)
o.op("**Levenshtein edits**  \n`DELETE` drops its slot · `INSERT` (split) puts a fresh `<|mask|>` before "
     f"the token · block re-padded / cut to {BLK}", DIFF, key="edit", h=96, w=420)
o.tensor(40 / KT, f"updated active block [{BLK}]", key="upd")
dec = o.op("**block done?**  \nno masks left and unchanged, or the post-fill budget is spent", ACT, key="done",
           h=76, w=420)
commit = o.op(f"**yes → commit** the block, append the next {BLK} masks  \n(stop at `<|endoftext|>`)", HEAD,
              key="commit", h=76, w=420)
edge(dec, commit, label="yes")  # replace the plain link with a labelled one
edges.pop(-2)

# loop lanes: right = another denoising step on the same block, left = block finished
LX = 1250
lr1 = node("loop-r1", LX, nodes[[n["id"] for n in nodes].index(dec)]["y"] + 8, 260, 60,
           "**no → next step**  \nwhole forward again", MASK)
edge(dec, lr1, ("right", "left"))
lr2 = node("loop-r2", LX, SY - 230, 260, 60, "↻ **denoising step**  \nwrites into the active block", MASK)
edge(lr1, lr2, ("top", "bottom"))
edge(lr2, "seq-active", ("left", "top"), color=MASK)

cy = nodes[[n["id"] for n in nodes].index(commit)]["y"]
ll1 = node("loop-l1", -1520, cy + 8, 260, 60, "**next block**  \nactive → finished", HEAD)
edge(commit, ll1, ("left", "right"))
ll2 = node("loop-l2", -1520, SY - 230, 260, 60, "**block committed**", HEAD)
edge(ll1, ll2, ("top", "bottom"))
edge(ll2, "seq-done", ("right", "top"), color=HEAD)

# AR vs this model, beside the decoding ops
ny = nodes[[n["id"] for n in nodes].index(m2t)]["y"]
node("note-ar", -1180, ny, 640, 230,
     "**vs. autoregressive decoding**  \n"
     "- AR: causal mask, 1 forward → 1 new token at the right end, never revised.\n"
     f"- Here: 1 forward predicts all {BLK} slots of the block at once; only confident ones are written, so a "
     f"block of {BLK} needs between 1 and ~{GEN['steps']} forwards (+ up to {GEN['max_post_steps']} refinement "
     "steps), and written tokens can still be rewritten (T2T), deleted or split.\n"
     "- This HF code keeps no KV cache: every step re-runs the full prefix.")
node("note-sample", 560, ny + 130, 600, 120,
     f"Defaults (`generate()`): temperature 0 → x̂ = argmax; with temperature > 0, M2T samples (top-k / top-p) "
     f"but p is still read from the unfiltered softmax. `<|mask|>` = id {GEN['mask_id']}, DELETE / INSERT = "
     f"reserved ids {GEN['delete']} / {GEN['split']}. Anti-loop resampling if a block state repeats.")
save("model.canvas")

# =====================================================================================================
# block.canvas — one MoE layer (L1–L19) in detail
# =====================================================================================================
_seq[0] = 0
QKV = (NH + 2 * NKV) * HD
node("cap-title", -1500, -420, 1500, 170,
     f"# LLaDA2.2-mini — one layer (L{DENSE}–L{N - 1})\n"
     "Pre-norm block: **bidirectional GQA attention** then a **block-routed MoE**. Shapes per forward pass, "
     f"T = tokens fed (multiple of {BLK}). L0 is identical except its FFN is one dense SwiGLU "
     f"{DIM} → {FI} → {DIM} ({fmt(P['dense'])}).")
node("cap-legend", 60, -420, 1400, 170,
     f"Layer parameters: attention {fmt(P['attn'] / N)} · router {fmt(params_of(r'model\.layers\.1\.mlp\.gate\..*'))} · "
     f"shared expert {fmt(params_of(r'model\.layers\.1\.mlp\.shared_experts\..*'))} · {E} routed experts "
     f"{fmt(E * EXPERT)}.  \nFrom `LLaDA2MoeAttention`, `LLaDA2MoeGate.block_routing` and "
     "`LLaDA2MoeSparseMoeBlock` in modeling_llada2_moe.py; shapes from the checkpoint.")

f = Flow(0, -180)
xin = f.tensor(DIM, f"x [T, {DIM}]  (residual stream)", key="xin")
f.op("**RMSNorm** (input_layernorm)", NORM, key="n1")
f.op(f"**query_key_value** {DIM} → {QKV}", ATT, params_of(r"model\.layers\.1\.attention\.query_key_value\..*"),
     note="no bias", key="qkv")
qkv = f.tensor(QKV, f"qkv [T, {NH}+{NKV}+{NKV} heads, {HD}]", key="qkvt", lbl_dy=-30)
f.y += 60

QX, VX = -900, 900
q = Flow(QX, f.y + 20, last=qkv, labels_left=True)
q.op(f"**Q** {NH} heads · RMSNorm per head", ATT, key="qn")
q.op(f"**RoPE** on {ROPE} of {HD} dims  \nθ = {C['rope_theta']:,}", ATT, key="qr", h=66)
qo = q.tensor(NH * HD, f"Q [T, {NH}, {HD}]", key="qt")
k = Flow(0, f.y + 20, last=qkv)
k.op(f"**K** {NKV} heads · RMSNorm per head", ATT, key="kn")
k.op(f"**RoPE** on {ROPE} of {HD} dims", ATT, key="kr")
ko = k.tensor(NKV * HD, f"K [T, {NKV}, {HD}]", key="kt")
v = Flow(VX, f.y + 20, last=qkv)
vo = v.tensor(NKV * HD, f"V [T, {NKV}, {HD}]", key="vt")

ay = max(q.y, k.y) + 20
att = node("attn", -260, ay, 520, 116,
           f"**attention** (SDPA, scale 1/√{HD})  \nGQA: each of {NKV} KV heads serves {NH // NKV} query heads  \n"
           f"**not causal** — mask = block-causal: full inside a {BLK}-token block, earlier blocks visible",
           ATT)
edge(qo, att, ("bottom", "left"))
edge(ko, att)
edge(vo, att, ("bottom", "right"))
a = Flow(0, ay + 116 + GAP, last=att)
a.tensor(NH * HD, f"[T, {NH}·{HD}]", key="ao")
a.op(f"**dense** (o-proj) {NH * HD} → {DIM}", ATT, params_of(r"model\.layers\.1\.attention\.dense\..*"), key="o")
add1 = a.op("**⊕ residual add**", ACT, key="add1")
RES_X = -640  # residual lane between the left branch column and the main column
yin = nodes[[n["id"] for n in nodes].index(xin)]["y"]
res1 = node("res1", RES_X, yin - 15, 130, 40, "residual", ACT)
edge(xin, res1, ("left", "right"), color=ACT)
edge(res1, add1, ("bottom", "left"), color=ACT)
x2 = a.tensor(DIM, f"x [T, {DIM}]", key="x2")
a.op("**RMSNorm** (post_attention_layernorm)", NORM, key="n2")
hn = a.tensor(DIM, f"h [T, {DIM}]", key="hn", lbl_dy=-30, left=True)

# three branches from h: shared expert (left), router (centre), routed experts (right)
by = a.y + 20
s = Flow(QX, by, last=hn, labels_left=True)
s.op(f"**shared expert**  \nSwiGLU {DIM} → {MI} → {DIM}", SHARED,
     params_of(r"model\.layers\.1\.mlp\.shared_experts\..*"), key="sh", h=96)
so = s.tensor(DIM, f"[T, {DIM}]", key="sht")

r = Flow(0, by, last=hn, labels_left=True)
r.op(f"**router** {DIM} → {E} (fp32)  \nsigmoid scores", SHARED, key="gate", h=66)
r.tensor(E, f"scores [T, {E}]", key="sc")
r.op("**+ expert_bias**  \n(used for picking only)", SHARED, key="bias", h=66)
r.op(f"**block routing** — per {BLK}-token block:  \nmax score over its tokens → keep top **{CAP}** of {E}",
     MASK, key="blk", h=76, w=420)
r.tensor(CAP, f"allowed experts [T/{BLK}, {CAP}]", key="al")
r.op(f"**per token top-{K}** among the {CAP} allowed  \nweights = scores / sum × {C['routed_scaling_factor']}",
     SHARED, key="tk", h=76, w=420)
ro = r.tensor(40 / KT, f"expert ids, weights [T, {K}]", key="ids")

yh = nodes[[n["id"] for n in nodes].index(hn)]["y"]
relay = node("hrelay", VX - 110, yh - 15, 220, 40, "h → routed experts", FFN)
edge(hn, relay, ("right", "left"))
ex = Flow(VX, r.y - 49 + 30, last=relay)  # below the router's output, so its edge runs right / down
ex_op = ex.op(f"**{K} routed experts**  \neach SwiGLU {DIM} → {MI} → {DIM}  \nweighted sum", FFN,
              K * EXPERT, note=f"active of {fmt(E * EXPERT)}", key="ex", h=116)
edge(ro, ex_op, ("right", "left"), label="which & how much")
exo = ex.tensor(DIM, f"[T, {DIM}]", key="ext")

jy = max(s.y, r.y, ex.y) + 20
j = Flow(0, jy)
join = j.op("**sum** routed + shared", FFN, key="sum")
edge(so, join, ("bottom", "left"))
edge(exo, join, ("bottom", "right"))
add2 = j.op("**⊕ residual add**", ACT, key="add2")
yx2 = nodes[[n["id"] for n in nodes].index(x2)]["y"]
res2 = node("res2", -1640, yx2 - 15, 130, 40, "residual", ACT)
edge(x2, res2, ("left", "right"), color=ACT)
ya2 = nodes[[n["id"] for n in nodes].index(add2)]["y"]
res3 = node("res3", -1640, ya2 + 8, 130, 40, "residual", ACT)  # lane runs down outside the shared-expert labels
edge(res2, res3, ("bottom", "top"), color=ACT)
edge(res3, add2, ("right", "left"), color=ACT)
j.tensor(DIM, f"x [T, {DIM}]  → next layer", key="xout")
node("note-route", VX - 220, jy - 10 + 2 * (OP_H + GAP), 700, 150,
     f"**Block routing** (new in LLaDA2.2): all {BLK} tokens of a block may only use the same {CAP} experts, so a "
     f"block's MoE touches ≤ {CAP} of {E} expert weight sets. config.json also lists `n_group` / `topk_group`, "
     "but this modeling code does not use them.")
save("block.canvas")
print("ok", [(b[0], fmt(b[2]), fmt(b[3])) for b in B], "active", fmt(active))
