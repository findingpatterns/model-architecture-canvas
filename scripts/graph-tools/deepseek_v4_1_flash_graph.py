"""Emit models/deepseek-v4-1-flash/graph.json: semantic nodes with dims; params computed from real
checkpoint shapes. usage: python3 deepseek_v4_1_flash_graph.py <shapes.json> <graph.json>"""
import json, math, os, re, sys

HERE = os.path.dirname(__file__)
S = json.load(open(sys.argv[1]))


def numel(k):
    dt, shp = S[k]
    return math.prod(shp) * (2 if dt == "I8" else 1)  # I8 packs two FP4 values


def params(*patterns):
    """Sum params of tensors matching any regex (scales excluded)."""
    rx = [re.compile(p) for p in patterns]
    return sum(numel(k) for k in S if not k.endswith(".scale") and any(r.fullmatch(k) for r in rx))


def dtype(pattern):
    for k in sorted(S):
        if re.fullmatch(pattern, k) and not k.endswith(".scale"):
            return S[k][0]
    return None


L = r"layers\.2\."  # representative Full-mode layer (has compressor + indexer)
nodes, edges = [], []


def n(id, label, kind, out=None, inp=None, parent=None, tensors=None, note=None, repeat=None, active=None, **kw):
    d = dict(id=id, label=label, kind=kind)
    if parent: d["parent"] = parent
    if inp: d["dimIn"] = inp
    if out: d["dimOut"] = out
    if repeat: d["repeat"] = repeat
    if tensors:
        d["tensors"] = tensors
        d["params"] = params(*tensors)
        if kind != "group":
            d["dtype"] = dtype(tensors[0])
    if active is not None: d["activeParams"] = active
    if note: d["note"] = note
    d.update(kw)
    nodes.append(d)


def e(a, b, label=None):
    edges.append([a, b] + ([label] if label else []))


# ---- top level ----
n("tokens", "text tokens", "io", out=1, note="Token ids [S]; width shown as 1 channel.")
n("image", "image", "io", out=588, note="14×14 RGB patches = 588 values each.")
n("vit", "DeepSeek-ViT", "embed", inp=588, out=1024, repeat=32, tensors=[r"vision\..*"],
  note="32-layer ViT trained from scratch, 2D-RoPE, d=1024, 16 heads.")
n("aligner", "Aligner MLP", "embed", inp=9216, out=5120, tensors=[r"aligner\..*"],
  note="3×3 pixel-unshuffle (9×1024 = 9216) → 2-layer MLP → 5120.")
n("embed", "embed", "embed", inp=1, out=5120, tensors=[r"embed\.weight", r"image_(start|end|newline)"],
  note="129280 × 5120 table + learned image delimiters.")
n("merge", "merge image spans", "op", inp=5120, out=5120, note="Image token slots overwritten by aligner rows.")
n("hash", "Engram n-gram hash", "op", out=1, note="Up-to-4-gram hashes of text tokens feed Engram at L1 and L14.")
n("hcx", "expand ×4 (mHC)", "hc", inp=5120, out=20480, note="Residual stream becomes hc_mult=4 parallel copies.")
n("stack", "Transformer layers", "group", inp=20480, out=20480, repeat=40,
  tensors=[r"layers\..*"],
  note="Layers 0–19 = causal encoder, 20–39 = decoder (global KV projected from L20). Params = all 40 layers incl. Engram.")
n("hcf", "hc_pre collapse", "hc", inp=20480, out=5120)
n("norm", "final RMSNorm", "norm", inp=5120, out=5120, tensors=[r"norm\.weight"])
n("head", "lm head", "out", inp=5120, out=129280, tensors=[r"head\.weight"], note="5120 → 129280 vocab logits.")
n("logits", "next token", "io", inp=129280, out=1)
n("dspark", "DSpark MTP", "group", inp=15360, out=129280, repeat=3, tensors=[r"mtp\..*"],
  note="Speculative decoding: taps attention input of L37–39 (3×5120), drafts 5 tokens per step.")
for a, b in [("tokens", "embed"), ("image", "vit"), ("vit", "aligner"), ("aligner", "merge"), ("embed", "merge"),
             ("merge", "hcx"), ("hcx", "stack"), ("stack", "hcf"), ("hcf", "norm"), ("norm", "head"),
             ("head", "logits"), ("tokens", "hash")]:
    e(a, b)
e("hash", "eng", "hash ids")
e("stack", "dspark", "L37–39 hidden")

# ---- one layer (inside stack) ----
P = "stack"
n("eng", "Engram (L1, L14)", "embed", parent=P, inp=20480, out=20480, tensors=[r"layers\.1\.engram\..*"],
  active=0, note="Conditional memory: hashed n-gram rows → key per hc copy + value, gated into the stream. "
                 "~98B params each, but only a few rows are read per token.")
n("hpa", "hc_pre", "hc", parent=P, inp=20480, out=5120, tensors=[L + r"hc_attn_.*"],
  note="Collapse 4 copies with pre_mix from previous FFN; computes this sublayer's post/comb (Sinkhorn ×20).")
n("an", "attn_norm", "norm", parent=P, inp=5120, out=5120, tensors=[L + r"attn_norm\.weight"])
n("attn", "CSA2 attention", "group", parent=P, inp=5120, out=5120,
  tensors=[L + r"attn\..*"], note="Shown for a Full-mode layer (L2). Reuse layers have no compressor/indexer.")
n("hqa", "hc_post", "hc", parent=P, inp=5120, out=20480, note="post·out + comb·residual → 4 copies.")
n("hpf", "hc_pre", "hc", parent=P, inp=20480, out=5120, tensors=[L + r"hc_ffn_.*"])
n("fn", "ffn_norm", "norm", parent=P, inp=5120, out=5120, tensors=[L + r"ffn_norm\.weight"])
n("moe", "MoE", "group", parent=P, inp=5120, out=5120, tensors=[L + r"ffn\..*"],
  active=params(L + r"ffn\.(gate|shared_experts)\..*") + params(L + r"ffn\.experts\..*") * 6 // 384)
n("hqf", "hc_post", "hc", parent=P, inp=5120, out=20480)
for a, b in [("eng", "hpa"), ("hpa", "an"), ("an", "attn"), ("attn", "hqa"), ("hqa", "hpf"), ("hpf", "fn"),
             ("fn", "moe"), ("moe", "hqf")]:
    e(a, b)

# ---- attention internals ----
A = "attn"
n("wqa", "wq_a", "linear", parent=A, inp=5120, out=1280, tensors=[L + r"attn\.wq_a\.weight"])
n("qn", "q_norm", "norm", parent=A, inp=1280, out=1280, tensors=[L + r"attn\.q_norm\.weight"])
n("wqb", "wq_b", "linear", parent=A, inp=1280, out=32768, tensors=[L + r"attn\.wq_b\.weight"],
  note="64 heads × 512; RoPE on last 64 dims.")
n("wkv", "wkv", "linear", parent=A, inp=5120, out=512, tensors=[L + r"attn\.wkv\.weight"], note="Single KV head.")
n("kvn", "kv_norm + RoPE", "norm", parent=A, inp=512, out=512, tensors=[L + r"attn\.kv_norm\.weight"])
n("win", "sliding window 128", "attn", parent=A, inp=512, out=512, note="FP8 ring buffer of the last 128 tokens.")
n("cmp", "Compressor", "attn", parent=A, inp=5120, out=512, tensors=[L + r"attn\.compressor\..*"],
  note="Only KV-source layers (2, 8, 14, 20). Ratio 2 in encoder (gated pooling), 1 in decoder. Cache stored FP4.")
n("idx", "Indexer", "attn", parent=A, inp=1280, out=512, tensors=[L + r"attn\.indexer\..*"],
  note="32 heads × 128 scores compressed positions → top-512. Index sources: 2,8,14,20,24,28,32,36; "
       "L24–36 search only inside L20's candidate pool.")
n("sa", "sparse_attn", "attn", parent=A, inp=32768, out=32768, tensors=[L + r"attn\.attn_sink"],
  note="Attends over window 128 + top-512 compressed positions, with a learned sink per head.")
n("woa", "wo_a (8 groups)", "linear", parent=A, inp=32768, out=8192, tensors=[L + r"attn\.wo_a\.weight"],
  note="Block-diagonal: 8 × (4096 → 1024).")
n("wob", "wo_b", "linear", parent=A, inp=8192, out=5120, tensors=[L + r"attn\.wo_b\.weight"])
for a, b in [("wqa", "qn"), ("qn", "wqb"), ("wqb", "sa"), ("wkv", "kvn"), ("kvn", "win"), ("win", "sa"),
             ("cmp", "idx"), ("idx", "sa"), ("sa", "woa"), ("woa", "wob")]:
    e(a, b)
e("qn", "idx", "qr")

# ---- MoE internals ----
M = "moe"
n("gate", "gate (top-6)", "moe", parent=M, inp=5120, out=384, tensors=[L + r"ffn\.gate\..*"],
  note="sqrtsoftplus scores, noaux_tc bias, separate bias_vl for image tokens, ×1.5 routed scale.")
n("routed", "routed experts", "moe", parent=M, inp=5120, out=5120, repeat=384, tensors=[L + r"ffn\.experts\..*"],
  active=params(L + r"ffn\.experts\..*") * 6 // 384, note="SwiGLU 5120 → 2304 → 5120, FP4. 6 of 384 active per token.")
n("shared", "shared expert", "moe", parent=M, inp=5120, out=5120, tensors=[L + r"ffn\.shared_experts\..*"],
  note="SwiGLU 5120 → 2304 → 5120, FP8, always active.")
n("sum", "Σ weighted", "op", parent=M, inp=5120, out=5120)
for a, b in [("gate", "routed"), ("routed", "sum"), ("shared", "sum")]:
    e(a, b)

# ---- per-layer table ----
layers = []
for i in range(40):
    p = rf"layers\.{i}\."
    exp = params(p + r"ffn\.experts\..*")
    row = dict(id=f"L{i}", experts=exp, shared=params(p + r"ffn\.(gate|shared_experts)\..*"),
               attention=params(p + r"attn\..*"), engram=params(p + r"engram\..*"),
               other=params(p + r"(hc_.*|attn_norm.*|ffn_norm.*)"))
    row["active"] = row["attention"] + row["shared"] + row["other"] + exp * 6 // 384
    mode = "SWA" if i < 2 else ("Full" if i in (2, 8, 14, 20) else ("Reindex" if i in (24, 28, 32, 36) else "Reuse"))
    row["mode"] = mode
    layers.append(row)

g = dict(model="deepseek-ai/DeepSeek-V4.1-Flash", name="DeepSeek-V4.1-Flash",
         source="inference/model.py + safetensors headers (all 96,085 tensors)",
         totalParams=sum(numel(k) for k in S if not k.endswith(".scale")),
         nodes=nodes, edges=edges, layers=layers)
json.dump(g, open(sys.argv[2], "w"), indent=1)
print(len(nodes), "nodes", round(g["totalParams"] / 1e9, 1), "B")
