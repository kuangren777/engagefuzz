"""Generate RQ1 and RQ3 figures as PDFs for the paper. Data only, no hand numbers.
python3 make_figs.py   -> out/figs/rq1_levelrate.pdf, rq3_density.pdf
"""
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "out", "figs")
sys.path.insert(0, os.path.join(HERE, "..", "engine"))

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
DISP = {"qwen3-8b-yf": "Qwen3-8B", "qwen3-32b-local": "Qwen3-32B",
        "llama31-8b-local": "Llama-3.1-8B", "granite31-8b-local": "Granite-3.1-8B"}


def rq1():
    d = json.load(open(os.path.join(HERE, "..", "runs", "rq1", "rq1_stats.json")))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for m in ["qwen3-8b-yf", "qwen3-32b-local", "llama31-8b-local", "granite31-8b-local"]:
        lr = d[m]["level_to_rate"]
        xs = sorted(int(k) for k in lr)
        ys = [lr[str(k)][1] for k in xs]
        ax.plot(xs, ys, marker="o", ms=4, label=DISP[m])
    ax.set_xlabel("proximity level"); ax.set_ylabel("mean trigger rate of the unit")
    ax.set_xticks([0, 1, 2, 3, 4]); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "rq1_levelrate.pdf"))


def rq3():
    import sys as _sys
    _sys.path.insert(0, os.path.join(HERE, "..", "engine")); import engagefuzz as ef
    import json as _json
    R = [_json.loads(l) for l in open(os.path.join(HERE, "..", "testbed", "data", "frontier.jsonl"))]
    R = [r for r in R if r.get("err") is None and r.get("trace") is not None]
    for r in R:
        r["_l21"] = ef.prox_v2.feats(r)["level21"]
    nv = [r for r in R if not r["security"]]
    asr = 100 * sum(bool(r["security"]) for r in R) / max(1, len(R))
    groups = {f"six frontier models, pooled (ASR {asr:.1f}%)": nv}
    import collections
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    colors = ["#d9d9d9", "#bdbdbd", "#fdae6b", "#e6550d", "#a50f15"]
    bottoms = [0.0] * len(groups)
    labels = list(groups)
    for lvl in range(4):   # binary failures cannot be at level four
        fr = [sum(r["_l21"] == lvl for r in groups[g]) / max(1, len(groups[g])) for g in labels]
        ax.bar(labels, fr, bottom=bottoms, color=colors[lvl], label=f"level {lvl}")
        bottoms = [b + f for b, f in zip(bottoms, fr)]
    ax.set_ylabel("share of binary-failure runs")
    ax.legend(fontsize=7, frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "rq3_density.pdf"))


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    rq1(); rq3()
    print("wrote", os.listdir(FIGS))
