"""Draw DeepSeek-V4.1-Flash as true-scale rectangles in JSON Canvas.

Every weight matrix W[out, in] is a rectangle: width = in, height = out, ONE linear scale for the
whole model, so area = parameter count. Repeated tensors (heads, experts, layers) are stacks of
offset rectangles. Activations (one token's vector) are thin bars whose width = channels.
The model-level tab is drawn by deepseek_v4_1_flash_model_canvas.py.

usage: python3 deepseek_v4_1_flash_matrix_canvas.py <shapes.json> <models/deepseek-v4-1-flash>
"""
import json
import math
import os
import re
import sys

S = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
K = 0.04  # px per channel, everywhere
CAP_W = 360
MAX_DRAWN = 8  # stacks show this many offset copies, labelled ×N
# semantic colors (same legend as the other tabs)
EMB, NORM, ATT, FFN, HC, OUTC, ACT = "6", "3", "5", "4", "1", "2", "#64748b"


def shape(name):
    """[out, in] of a checkpoint tensor; I8 tensors pack two FP4 values along `in`."""
    dt, shp = S[name]
    out, inn = (shp + [1])[:2] if len(shp) == 1 else shp[:2]
    return out, inn * (2 if dt == "I8" else 1), dt


def params_of(pattern):
    rx = re.compile(pattern)
    total = 0
    for k, (dt, shp) in S.items():
        if rx.fullmatch(k) and not k.endswith(".scale"):
            total += math.prod(shp) * (2 if dt == "I8" else 1)
    return total


def fmt(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


class Canvas:
    def __init__(self):
        self.nodes, self.edges, self.n = [], [], 0

    def node(self, x, y, w, h, text="", color=None, id=None):
        self.n += 1
        nid = id or f"n{self.n}"
        d = dict(id=nid, type="text", x=round(x), y=round(y), width=max(2, round(w)), height=max(2, round(h)), text=text)
        if color:
            d["color"] = color
        self.nodes.append(d)
        return nid

    def edge(self, a, b, label=None, sides=("bottom", "top")):
        d = dict(id=f"e{len(self.edges)}", fromNode=a, toNode=b, fromSide=sides[0], toSide=sides[1])
        if label:
            d["label"] = label
        self.edges.append(d)

    def save(self, name):
        json.dump(dict(nodes=self.nodes, edges=self.edges), open(os.path.join(OUT, name), "w"), indent=1, ensure_ascii=False)


class Column:
    """Stacks drawings top→bottom around center x, chaining each new item to the previous one."""

    def __init__(self, c, cx, y, gap=40):
        self.c, self.cx, self.y, self.gap, self.last = c, cx, y, gap, None
        self.entry_sides = ("bottom", "top")  # sides of the first edge into this column

    def _link(self, nid, link):
        if link and self.last:
            self.c.edge(self.last, nid, sides=self.entry_sides)
            self.entry_sides = ("bottom", "top")
        self.last = nid

    def matrix(self, name, out, inn, color, caption, stack=1, drawn=None, link=True):
        """Rectangle (or stack) of a W[out, in] matrix at true scale."""
        w, h = inn * K, out * K
        drawn = min(stack, drawn or MAX_DRAWN)
        step = 6 if drawn > 1 else 0
        x0 = self.cx - w / 2
        back = None
        for j in reversed(range(1, drawn)):  # back copies first so the front one paints on top
            nid = self.c.node(x0 + j * step, self.y + j * step, w, h, "", color)
            back = back or nid  # deepest copy: outgoing edges leave from it so they clear the stack
        inside = f"**{name}**" if w >= 70 and h >= 26 else ""
        front = self.c.node(x0, self.y, w, h, inside, color)
        tag = f" · ×{stack}" + (f" ({drawn} drawn)" if drawn < stack else "") if stack > 1 else ""
        self.c.node(x0 + w + (drawn - 1) * step + 14, self.y, CAP_W, 64,
                    f"**{name}** `{out}×{inn}`{tag}\n{caption}")
        self._link(front, link)
        if back:
            self.last = back
        # advance past the taller of drawing and caption so captions of thin matrices never collide
        self.y += max(h + (drawn - 1) * step, 64) + self.gap
        return front

    def act(self, dim, note="", caption="right"):
        """One token's activation vector: a bar `dim` channels wide.
        caption = right | left | above; moving it frees that side of the bar for fan-out edges."""
        w = dim * K
        nid = self.c.node(self.cx - w / 2, self.y, w, 10, "", ACT)
        cap_x, cap_y = {"right": (self.cx + w / 2 + 14, self.y - 12), "left": (self.cx - w / 2 - 274, self.y - 12),
                        "above": (self.cx - 130, self.y - 46)}[caption]
        self.c.node(cap_x, cap_y, 260, 34, f"`{dim}`{(' ' + note) if note else ''}")
        self._link(nid, True)
        self.y += 10 + self.gap
        return nid


def legend(c, x, y):
    c.node(x, y, 520, 230,
           "# How to read\n"
           f"- **Rectangle = weight matrix W[out×in]**: width = in, height = out, **{K} px per channel** "
           "in this tab → **area = parameters**.\n"
           "- **Stacked offset copies** = the same matrix repeated (heads, experts, layers).\n"
           "- **Thin grey bar** = one token's activation; width = channels.\n"
           "- Zoom out to compare; nothing is resized for readability.")


L = "layers.2."
SIDE_DY, SA_GAP, QR_SIDES = 150, 600, ("right", "top")  # attention tab: side-column drop, gap above sparse_attn, qr→indexer sides

# ================= Block: every matrix of one layer =================
c = Canvas()
c.node(-1400, -520, 900, 160,
       "# One layer (L2, Full mode) — true-scale matrices\nAttention ≈ 0.13B · shared expert + gate ≈ 0.04B · "
       f"routed experts ≈ {fmt(params_of(L + r'ffn\.experts\..*'))} (384 × 3 matrices). Same scale for every rectangle.")
legend(c, -400, -520)
col = Column(c, 0, -250)
col.act(20480, "residual stream = 4 copies × 5120 (mHC)")
o, i, _ = shape(L + "hc_attn_fn")
col.matrix("hc_attn_fn", o, i, HC, "mixing weights → pre / post / comb (Sinkhorn)")
col.act(5120, "hc_pre → one 5120 vector")
col.matrix("attn_norm", 1, 5120, NORM, "RMSNorm weight (a single row)")
o, i, _ = shape(L + "attn.wq_a.weight")
col.matrix("wq_a", o, i, ATT, "Q down-projection (FP8)")
col.act(1280, "qr")
o, i, _ = shape(L + "attn.wq_b.weight")
col.matrix("wq_b", o, i, ATT, "Q up-projection → 64 heads × 512 (FP8)")
col.act(32768, "q: 64 heads × 512")
col.y += 260  # room for the KV/indexer edge to arrive from the left below the 32768-wide q bar
sa = col.c.node(col.cx - 60, col.y, 120, 40, "**sparse_attn**", ATT)
col._link(sa, True)
col.y += 80
o, i, _ = shape(L + "attn.wo_a.weight")
col.matrix("wo_a", o // 8, i, ATT, "block-diagonal: 8 groups, each 4096 → 1024 (FP8)", stack=8)
col.act(8192, "8 × 1024")
o, i, _ = shape(L + "attn.wo_b.weight")
col.matrix("wo_b", o, i, ATT, "back to model width (FP8)")
col.act(5120, "→ hc_post → 4 × 5120")
o, i, _ = shape(L + "hc_ffn_fn")
col.matrix("hc_ffn_fn", o, i, HC, "FFN-side mixing weights")
col.act(5120, "hc_pre")
col.matrix("ffn_norm", 1, 5120, NORM, "RMSNorm weight")
o, i, _ = shape(L + "ffn.gate.weight")
gate = col.matrix("gate", o, i, FFN, "router: 384 expert scores, top-6 (BF16)")
moe_top = col.y + 40

# KV + compressor + indexer side column (feeds sparse_attn)
kv = Column(c, -1500, 120)
kv.last = None
o, i, _ = shape(L + "attn.wkv.weight")
kv.matrix("wkv", o, i, ATT, "single KV head (FP8) → sliding window 128")
kv.act(512, "kv")
o, i, _ = shape(L + "attn.compressor.wkv.weight")
kv.matrix("compressor.wkv", o, i, FFN, "KV compressor (Full layers only, BF16)", link=False)
o, i, _ = shape(L + "attn.compressor.wgate.weight")
kv.matrix("compressor.wgate", o, i, FFN, "pooling gate (ratio 2 layers)")
o, i, _ = shape(L + "attn.indexer.wq_b.weight")
kv.matrix("indexer.wq_b", o, i, FFN, "indexer queries 32 heads × 128 (from qr)", link=False)
o, i, _ = shape(L + "attn.indexer.weights_proj.weight")
kv.matrix("indexer.weights_proj", o, i, FFN, "per-head weights")
c.edge(kv.last, sa, "top-512 + window", sides=("bottom", "left"))

# MoE: shared expert and routed experts side by side, true scale
sh = Column(c, -900, moe_top)
sh.last = gate
o, i, _ = shape(L + "ffn.shared_experts.w1.weight")
sh.matrix("shared w1 / w3", o, i, ATT, "gate + up projections (FP8), always active", stack=2)
sh.act(2304, "SwiGLU hidden")
o, i, _ = shape(L + "ffn.shared_experts.w2.weight")
sh.matrix("shared w2", o, i, ATT, "down projection")
rt = Column(c, 700, moe_top)
rt.last = gate
o, i, _ = shape(L + "ffn.experts.0.w1.weight")
rt.matrix("expert w1 / w3", o, i, FFN, "routed experts, FP4 — every one of 384 has this pair; 6 run per token",
          stack=384)
rt.act(2304, "SwiGLU hidden (6 experts in parallel)")
o, i, _ = shape(L + "ffn.experts.0.w2.weight")
rt.matrix("expert w2", o, i, FFN, "down projection, ×384", stack=384)
c.save("block.canvas")


# ================= Attention: CSA2 internals, same absolute scale =================
c = Canvas()
c.node(-1900, -560, 1000, 170,
       "# CSA2 attention (L2, Full mode) — true-scale matrices\nQ is an hourglass (wq_a flat → wq_b tall); KV is one thin 512-wide head. "
       "Compressor + indexer exist only in some layers: **Full** = own compressor + indexer (2, 8, 14, 20), "
       "**Reindex** = own indexer, shared KV (24, 28, 32, 36), **Reuse** = neither.")
legend(c, -800, -560)
x0 = Column(c, 0, -300)
xin = x0.act(5120, "x after attn_norm", caption="above")
q = Column(c, 0, x0.y)
q.last = xin
o, i, _ = shape(L + "attn.wq_a.weight")
q.matrix("wq_a", o, i, ATT, "Q down-projection (FP8)")
q.matrix("q_norm", 1, 1280, NORM, "RMSNorm")
qr = q.act(1280, "qr (also feeds the indexer)", caption="left")
o, i, _ = shape(L + "attn.wq_b.weight")
q.matrix("wq_b", o, i, ATT, "→ 64 heads × 512, RoPE on last 64 dims")
q.act(32768, "q")
kv = Column(c, -1300, x0.y + SIDE_DY)
kv.last = xin
kv.entry_sides = ("left", "top")
o, i, _ = shape(L + "attn.wkv.weight")
kv.matrix("wkv", o, i, ATT, "single KV head (FP8)")
kv.matrix("kv_norm", 1, 512, NORM, "RMSNorm + RoPE → FP8 window cache (128 tokens)")
kv.act(512, "window KV")
cm = Column(c, 1500, x0.y + SIDE_DY)
cm.last = xin
cm.entry_sides = ("right", "top")
o, i, _ = shape(L + "attn.compressor.wkv.weight")
cm.matrix("compressor.wkv", o, i, FFN, "KV latent per pooled group (BF16)")
o, i, _ = shape(L + "attn.compressor.wgate.weight")
cm.matrix("compressor.wgate", o, i, FFN, "softmax pooling gate (ratio-2 layers only)")
cm.act(512, "compressed KV → FP4 cache, shared with later layers")
ix = Column(c, 1500, cm.y + 40)
ix.last = qr
ix.entry_sides = QR_SIDES
o, i, _ = shape(L + "attn.indexer.wq_b.weight")
ix.matrix("indexer.wq_b", o, i, FFN, "indexer queries 32 heads × 128 (FP8)")
o, i, _ = shape(L + "attn.indexer.wk.weight")
ix.matrix("indexer.wk", o, i, FFN, "index keys from the compressed latent")
o, i, _ = shape(L + "attn.indexer.weights_proj.weight")
ix.matrix("indexer.weights_proj", o, i, FFN, "per-head weights from x")
ix.act(512, "top-512 compressed positions")
m = Column(c, 0, max(q.y, kv.y, ix.y) + SA_GAP)
sa = c.node(-90, m.y, 180, 44, "**sparse_attn**\nwindow 128 + top-512", ATT)
for col, into in ((q, "top"), (kv, "left"), (ix, "right")):
    c.edge(col.last, sa, sides=("bottom", into))
m.last = sa
m.y += 90
m.act(32768, "64 heads × 512")
o, i, _ = shape(L + "attn.wo_a.weight")
m.matrix("wo_a", o // 8, i, ATT, "block-diagonal: 8 groups × (4096 → 1024)", stack=8)
m.act(8192)
o, i, _ = shape(L + "attn.wo_b.weight")
m.matrix("wo_b", o, i, ATT, "back to 5120 → hc_post")
m.act(5120)
c.save("attention.canvas")
