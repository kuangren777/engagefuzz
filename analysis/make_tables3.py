"""Expansion tables 3: per-suite RQ2 breakdown (exploratory) and RQ1 power analysis (exploratory).
python3 make_tables3.py -> out/{rq2_suite_table,rq1_power_table}.tex
"""
import collections
import glob
import json
import math
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
RUNS = os.path.join(HERE, "..", "runs")
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import engagefuzz as ef  # noqa: E402

PAT = re.compile(r".+?_(None|both)_(banking|slack|travel)_(user_task_\d+)_(injection_task_\d+)_([a-z0-9-]+)_(\d+)\.jsonl$")


def a12(x, y):
    return sum((a > b) + 0.5 * (a == b) for a, b in zip(x, y)) / len(x)


def rq2_suite():
    rows = []
    for mk, disp in [("qwen3-8b-local", "Qwen3-8B"), ("llama31-8b-local", "Llama-3.1-8B")]:
        camp = collections.defaultdict(dict)  # (suite,seed,target) -> arm -> viol
        for f in glob.glob(os.path.join(RUNS, "rq2", f"{mk}_*_[0-9].jsonl")):
            m = PAT.match(os.path.basename(f))
            if not m:
                continue
            R = [json.loads(x) for x in open(f)]
            if len(R) < 40:
                continue
            dfn, s, ut, it, arm, seed = m.groups()
            camp[(s, seed, (ut, it, dfn))][arm] = sum(bool(r["security"]) for r in R)
        for suite in ("banking", "slack", "travel"):
            units = [(k, v) for k, v in camp.items() if k[0] == suite and "prox-v21" in v and "binary" in v]
            if not units:
                continue
            px = [v["prox-v21"] for _, v in units]; bn = [v["binary"] for _, v in units]
            rows.append(f"{disp} & {suite} & {len(units)} & {st.mean(px):.2f} & {st.mean(bn):.2f} & {a12(px, bn):.3f} " + r"\\")
    tab = [r"\begin{table}[t]\centering",
           r"\caption{Per-suite view of RQ2 (exploratory, added after the pre-registered analysis). "
           r"Mean violations per campaign for proximity and binary success. The paired $\hat A_{12}$ "
           r"of proximity against binary within each suite.}",
           r"\label{tab:rq2suite}", r"\begin{tabular}{llcccc}", r"\toprule",
           r"Model & Suite & units & proximity $\uparrow$ & binary $\uparrow$ & $\hat A_{12}$ $\uparrow$ \\", r"\midrule"] + rows + \
          [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "rq2_suite_table.tex"), "w").write("\n".join(tab) + "\n")


def min_detectable_rho(n, power=0.8, alpha=0.05):
    """Two-sided Spearman via Fisher z: smallest rho detectable at given n, power."""
    if n < 5:
        return float("nan")
    from statistics import NormalDist
    z = (NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)) / math.sqrt(n - 3)
    return math.tanh(z)


def rq1_power():
    rows = []
    for mk, disp in [("qwen3-8b-yf", "Qwen3-8B"), ("qwen3-32b-local", "Qwen3-32B"),
                     ("llama31-8b-local", "Llama-3.1-8B"), ("granite31-8b-local", "Granite-3.1-8B")]:
        files = [f for f in glob.glob(os.path.join(RUNS, "rq1", f"census_{mk}*.jsonl")) if "ABORTED" not in f]
        R = [json.loads(l) for f in files for l in open(f)]
        R = [r for r in R if r.get("err") is None and r.get("trace") is not None]
        cells = collections.defaultdict(list)
        for r in R:
            cells[(r["suite"], r["ut"], r["it"], r["defense"])].append(r)
        rate = {k: st.mean(bool(x["security"]) for x in v) for k, v in cells.items()}
        mixed = sum(0 < v < 1 for v in rate.values())
        asr = st.mean(bool(r["security"]) for r in R)
        # fail-only units with variation (the ones a correlation can use)
        usable = sum(1 for k, v in cells.items() if any(not x["security"] for x in v))
        mdr = min_detectable_rho(usable)
        rows.append(f"{disp} & {asr:.3f} & {len(cells)} & {mixed} & {usable} & {"--" if mdr != mdr else f"{mdr:.2f}"} " + r"\\")
    tab = [r"\begin{table}[t]\centering",
           r"\caption{Power of the per-model RQ1 test (exploratory). A correlation can only use units "
           r"whose trigger rate varies, so on models with an attack-success rate near zero few units "
           r"are informative and only a large $\rho$ is detectable. ASR is the attack-success rate, "
           r"mixed is the count of units with a trigger rate strictly between zero and one and the "
           r"detectable $\rho$ is computed at the usable-unit count with 80\% power.}",
           r"\label{tab:rq1power}", r"\begin{tabular}{lccccc}", r"\toprule",
           r"Model & ASR & units & mixed & usable & min detectable $\rho$ \\", r"\midrule"] + rows + \
          [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "rq1_power_table.tex"), "w").write("\n".join(tab) + "\n")


if __name__ == "__main__":
    rq2_suite(); rq1_power()
    print("wrote rq2_suite_table.tex, rq1_power_table.tex")
