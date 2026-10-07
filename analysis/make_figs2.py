"""Expansion figures for the S&S draft, all from committed run data.
python3 make_figs2.py -> out/figs/{rq2_box,rq2_fresh,rq4_injeccurve,rq4_frontierbars}.pdf
"""
import collections
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "out", "figs")
RUNS = os.path.join(HERE, "..", "runs")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
ARMS = ["prox-v21", "random", "binary", "coverage"]
ARMLBL = {"prox-v21": "proximity", "random": "random", "binary": "binary", "coverage": "coverage"}
PAT = re.compile(r".+?_(None|both)_(banking|slack|travel)_(user_task_\d+)_(injection_task_\d+)_([a-z0-9-]+)_(\d+)\.jsonl$")


def load_campaigns(model):
    """per-campaign: arm -> list of (violations, fresh_violations)."""
    by = collections.defaultdict(list)
    for f in glob.glob(os.path.join(RUNS, "rq2", f"{model}_*_[0-9].jsonl")):
        m = PAT.match(os.path.basename(f))
        if not m:
            continue
        R = [json.loads(x) for x in open(f)]
        if len(R) < 40:
            continue
        arm = m.group(5)
        v = sum(bool(r["security"]) for r in R)
        fv = sum(bool(r["security"]) and not r.get("memo_hit") for r in R)
        by[arm].append((v, fv))
    return by


def rq2_box():
    models = [("qwen3-8b-local", "Qwen3-8B"), ("llama31-8b-local", "Llama-3.1-8B")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=False)
    for ax, (mk, disp) in zip(axes, models):
        by = load_campaigns(mk)
        data = [[v for v, _ in by[a]] for a in ARMS]
        ax.boxplot(data, showmeans=True, widths=0.6, tick_labels=[ARMLBL[a] for a in ARMS])
        ax.set_title(disp, fontsize=9)
        ax.set_ylabel("violations per campaign")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "rq2_box.pdf"))


def rq2_fresh():
    models = [("qwen3-8b-local", "Qwen3-8B"), ("llama31-8b-local", "Llama-3.1-8B")]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    x = range(len(ARMS)); w = 0.35
    for mi, (mk, disp) in enumerate(models):
        by = load_campaigns(mk)
        allm = [sum(v for v, _ in by[a]) / max(1, len(by[a])) for a in ARMS]
        frm = [sum(fv for _, fv in by[a]) / max(1, len(by[a])) for a in ARMS]
        off = -w / 2 if mi == 0 else w / 2
        ax.bar([i + off for i in x], allm, w, label=f"{disp} all", alpha=0.9)
        ax.bar([i + off for i in x], frm, w, label=f"{disp} fresh", alpha=0.5, hatch="//")
    ax.set_xticks(list(x)); ax.set_xticklabels([ARMLBL[a] for a in ARMS], rotation=20)
    ax.set_ylabel("mean violations per campaign")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "rq2_fresh.pdf"))


def rq4_curve():
    R = [json.loads(l) for l in open(os.path.join(RUNS, "rq4", "injecagent_llama.jsonl")) if json.loads(l)["err"] is None]
    cells = collections.defaultdict(list)
    for r in R:
        cells[r["idx"]].append(r)
    import statistics as st
    rate = {k: st.mean(bool(x["security"]) for x in v) for k, v in cells.items()}
    lv = collections.defaultdict(list)
    for r in R:
        lv[r["level_ia"]].append(rate[r["idx"]])
    xs = sorted(lv); ys = [st.mean(lv[k]) for k in xs]
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.plot(xs, ys, marker="o", ms=5, color="#e6550d")
    ax.set_xlabel("proximity level (InjecAgent)"); ax.set_ylabel("mean per-case trigger rate")
    ax.set_xticks(xs)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "rq4_injeccurve.pdf"))


def rq4_frontier():
    files = sorted(glob.glob(os.path.join(RUNS, "rq4", "injecagent_frontier_*.jsonl")))
    models = [os.path.basename(f).split("injecagent_frontier_")[1].replace(".jsonl", "") for f in files]
    colors = ["#dddddd", "#bdbdbd", "#fdae6b", "#e6550d", "#a50f15"]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    bottoms = [0.0] * len(files)
    for lvl in range(1, 5):
        fr = []
        for f in files:
            R = [json.loads(l) for l in open(f) if json.loads(l)["err"] is None]
            fr.append(sum(r["level_ia"] == lvl for r in R) / max(1, len(R)))
        ax.bar(models, fr, bottom=bottoms, color=colors[lvl], label=f"level {lvl}")
        bottoms = [b + f for b, f in zip(bottoms, fr)]
    ax.set_ylabel("share of runs"); ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=6, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "rq4_frontierbars.pdf"))


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    rq2_box(); rq2_fresh(); rq4_curve(); rq4_frontier()
    print("wrote rq2_box, rq2_fresh, rq4_injeccurve, rq4_frontierbars")
