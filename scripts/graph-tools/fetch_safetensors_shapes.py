"""Fetch dtype + shape of every tensor in a Hugging Face checkpoint by reading only the
safetensors headers (HTTP Range requests), never the weights.

usage: python3 fetch_safetensors_shapes.py <hf-model-id> <out.json>
"""
import concurrent.futures as cf
import json
import struct
import subprocess
import sys

model_id, out_path = sys.argv[1], sys.argv[2]
base = f"https://huggingface.co/{model_id}/resolve/main/"


def get(path, start, end):
    # curl follows the CDN redirect reliably; retry a few times on flaky ranges
    for _ in range(4):
        r = subprocess.run(["curl", "-sfL", "-r", f"{start}-{end}", base + path], capture_output=True)
        if r.returncode == 0 and len(r.stdout) == end - start + 1:
            return r.stdout
    raise RuntimeError(f"range fetch failed: {path} {start}-{end}")


def header(shard):
    n = struct.unpack("<Q", get(shard, 0, 7))[0]
    h = json.loads(get(shard, 8, 7 + n))
    h.pop("__metadata__", None)
    return {k: [v["dtype"], v["shape"]] for k, v in h.items()}


index = json.loads(subprocess.run(["curl", "-sfL", base + "model.safetensors.index.json"],
                                  capture_output=True, check=True).stdout)
shards = sorted(set(index["weight_map"].values()))
shapes = {}
with cf.ThreadPoolExecutor(6) as ex:
    for part in ex.map(header, shards):
        shapes.update(part)
json.dump(shapes, open(out_path, "w"))
print(f"{len(shapes)} tensors from {len(shards)} shards -> {out_path}")
