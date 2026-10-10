"""Data-driven tables for the expansion: RQ1 per-model level->rate with bootstrap CI, and the
RQ2 memo-hit-rate table. python3 make_tables2.py -> out/{rq1_levelrate_table,memo_table}.tex
"""
import collections
import glob
import json
import os
import random
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
RUNS = os.path.join(HERE, "..", "runs")
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import engagefuzz as ef  # noqa: E402

MODELS = [("qwen3-8b-yf", "Qwen3-8B"), ("qwen3-32b-local", "Qwen3-32B"),
          ("llama31-8b-local", "Llama-3.1-8B"), ("granite31-8b-local", "Granite-3.1-8B")]


def ci(xs, B=2000, seed=0):
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed); out = []
    for _ in range(B):
        out.append(st.mean(rng.choice(xs) for _ in xs))
    out.sort()
    return out[int(0.025 * B)], out[int(0.975 * B) - 1]


def rq1_table():
    rows = []
    for mk, disp in MODELS:
        files = [f for f in glob.glob(os.path.join(RUNS, "rq1", f"census_{mk}*.jsonl")) if "ABORTED" not in f]
        R = [json.loads(l) for f in files for l in open(f)]
        R = [r for r in R if r.get("err") is None and r.get("trace") is not None]
        if not R:
            continue
        cells = collections.defaultdict(list)
        for r in R:
            cells[(r["suite"], r["ut"], r["it"], r["defense"])].append(r)
        rate = {k: st.mean(bool(x["security"]) for x in v) for k, v in cells.items()}
        lv = collections.defaultdict(list)
        for r in R:
            lv[ef.prox_v2.feats(r)["level21"]].append(rate[(r["suite"], r["ut"], r["it"], r["defense"])])
        cellsline = []
        for L in range(5):
            if lv[L]:
                lo, hi = ci(lv[L])
                cellsline.append(f"{st.mean(lv[L]):.2f} [{lo:.2f}, {hi:.2f}]")
            else:
                cellsline.append("--")
        rows.append(f"{disp} & " + " & ".join(cellsline) + r" \\")
    tab = [r"\begin{table}[t]\centering",
           r"\caption{RQ1 mean trigger rate of a unit by proximity level, per model, with 95\% "
           r"bootstrap intervals over units. A dash marks a level with no units.}", r"\label{tab:rq1levelrate}",
           r"\begin{tabular}{lccccc}", r"\toprule",
           r"Model & L0 & L1 & L2 & L3 & L4 \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "rq1_levelrate_table.tex"), "w").write("\n".join(tab) + "\n")


def memo_table():
    rows = []
    for tag, disp in [("qwen", "Qwen3-8B"), ("llama", "Llama-3.1-8B")]:
        d = json.load(open(os.path.join(RUNS, "rq2", f"rq2_paired_{tag}.json")))
        for arm in ["prox-v21", "random", "binary", "coverage"]:
            a = d["arms"][arm]
            rows.append((disp, {"prox-v21": "proximity"}.get(arm, arm), a["memo_hit"], a["replay_viol"]))
        if tag == "qwen":
            t = json.load(open(os.path.join(RUNS, "rq2", "rq2_tmap_qwen.json")))["arms"]["tmap-ladder"]
            rows.append((disp, "T-MAP ladder", t["memo_hit"], t["replay_viol"]))
    body = []
    for disp, arm, mh, rv in rows:
        body.append(f"{disp} & {arm} & {mh:.2f} & {rv} " + r"\\")
    tab = [r"\begin{table}[t]\centering",
           r"\caption{Execution-memo hit rate and replay violations per arm. A replay violation is a "
           r"violation served from the memo rather than a fresh execution.}", r"\label{tab:memo}",
           r"\begin{tabular}{llcc}", r"\toprule",
           r"Model & Reward & memo-hit rate & replay violations \\", r"\midrule"] + body + \
          [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "memo_table.tex"), "w").write("\n".join(tab) + "\n")


if __name__ == "__main__":
    rq1_table(); memo_table()
    print("wrote rq1_levelrate_table.tex, memo_table.tex")
