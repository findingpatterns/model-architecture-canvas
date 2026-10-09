"""Lint a JSON Canvas the way json-canvas-viewer@4.3.2 draws it.

Reports: duplicate ids, dangling edges, overlapping text nodes, edges passing behind a node that is
not one of their endpoints, and edge labels covering a node. Curves use the viewer's own
getControlPoints (offset = clamp((min(|dx|,|dy|) + 0.3*max) / 2, 60, 300)).
"""
import collections
import json
import sys


def side_pt(n, s):
    x, y, w, h = n["x"], n["y"], n["width"], n["height"]
    return {"top": (x + w / 2, y), "bottom": (x + w / 2, y + h), "left": (x, y + h / 2), "right": (x + w, y + h / 2)}[s]


def ctrl(p0, p3, fs, ts):
    dx, dy = p3[0] - p0[0], p3[1] - p0[1]
    c = max(60, min(300, (min(abs(dx), abs(dy)) + 0.3 * max(abs(dx), abs(dy))) * 0.5))
    off = {"top": (0, -c), "bottom": (0, c), "left": (-c, 0), "right": (c, 0)}
    return (p0[0] + off[fs][0], p0[1] + off[fs][1]), (p3[0] + off[ts][0], p3[1] + off[ts][1])


def bez(p0, p1, p2, p3, t):
    u = 1 - t
    return tuple(u ** 3 * p0[i] + 3 * u * u * t * p1[i] + 3 * u * t * t * p2[i] + t ** 3 * p3[i] for i in (0, 1))


def inside(pt, n, pad=0):
    return n["x"] - pad < pt[0] < n["x"] + n["width"] + pad and n["y"] - pad < pt[1] < n["y"] + n["height"] + pad


def overlap(a, b):
    return a["x"] < b["x"] + b["width"] and b["x"] < a["x"] + a["width"] and a["y"] < b["y"] + b["height"] and b["y"] < a["y"] + a["height"]


def lint(path):
    c = json.load(open(path))
    nodes, edges = c["nodes"], c["edges"]
    by = {n["id"]: n for n in nodes}
    out = []
    dup = [k for k, v in collections.Counter(n["id"] for n in nodes).items() if v > 1]
    if dup:
        out.append(f"duplicate node ids: {dup}")
    out += [f"dangling edge {e['id']}" for e in edges if e["fromNode"] not in by or e["toNode"] not in by]
    text = [n for n in nodes if n["type"] != "group"]
    for i, a in enumerate(text):
        for b in text[i + 1:]:
            if overlap(a, b):
                out.append(f"overlap {a['id']} / {b['id']}")
    for e in edges:
        a, b = by[e["fromNode"]], by[e["toNode"]]
        p0, p3 = side_pt(a, e.get("fromSide", "right")), side_pt(b, e.get("toSide", "left"))
        p1, p2 = ctrl(p0, p3, e.get("fromSide", "right"), e.get("toSide", "left"))
        hit = set()
        for k in range(1, 100):
            pt = bez(p0, p1, p2, p3, k / 100)
            for n in text:
                if n["id"] not in (a["id"], b["id"]) and inside(pt, n):
                    hit.add(n["id"])
        if hit:
            out.append(f"edge {a['id']}→{b['id']} passes behind {sorted(hit)}")
        if e.get("label"):
            mx, my = bez(p0, p1, p2, p3, 0.5)
            w = len(e["label"]) * 9.5 + 16  # 18px font
            lab = dict(x=mx - w / 2, y=my - 14, width=w, height=28)
            cov = [n["id"] for n in text if overlap(lab, n)]
            if cov:
                out.append(f"edge label '{e['label']}' covers {cov}")
    return out


for p in sys.argv[1:]:
    res = lint(p)
    print(f"== {p}: {len(res)} issue(s)")
    for r in res:
        print("  ", r)
