"""RQ2 per-target table (exploratory): mean violations per campaign for each reward arm
per selected target, plus paired A12 of prox-v21 vs binary over the 5 seeds.
python3 make_rq2_targets_table.py -> out/rq2_targets_table.tex
"""
import collections
import glob
import json
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out", "rq2_targets_table.tex")
RUNS = os.path.join(HERE, "..", "runs")
PAT = re.compile(r".+?_(None|both)_(banking|slack|travel)_(user_task_\d+)_(injection_task_\d+)_([a-z0-9-]+)_(\d+)\.jsonl$")
DISP = {"qwen3-8b-local": "Qwen3-8B", "llama31-8b-local": "Llama-3.1-8B"}
ARMS = ["prox-v21", "random", "binary", "coverage", "tmap-ladder"]


def a12(x, y):
    return sum((a > b) + 0.5 * (a == b) for a, b in zip(x, y)) / len(x)


def main():
    rows = []
    for mk in ("qwen3-8b-local", "llama31-8b-local"):
        camp = collections.defaultdict(dict)  # (suite,ut,it,dfn) -> (seed) -> arm -> viol
        for f in glob.glob(os.path.join(RUNS, "rq2", f"{mk}_*_[0-9].jsonl")):
            m = PAT.match(os.path.basename(f))
            if not m:
                continue
            R = [json.loads(x) for x in open(f)]
            if len(R) < 40:
                continue
            dfn, s, ut, it, arm, seed = m.groups()
            camp[(s, ut, it, dfn)][(arm, int(seed))] = sum(bool(r["security"]) for r in R)
        for (s, ut, it, dfn), v in sorted(camp.items()):
            cells = []
            for arm in ARMS:
                xs = [vi for (a, _), vi in v.items() if a == arm]
                cells.append(f"{st.mean(xs):.2f}" if xs else "--")
            px = [vi for (a, _), vi in sorted(v.items()) if a == "prox-v21"]
            bn = [vi for (a, sd), vi in sorted(v.items()) if a == "binary" and ("prox-v21", sd) in v]
            A = f"{a12(px, bn):.3f}" if px and bn and len(px) == len(bn) else "--"
            rows.append(f"{DISP[mk]} & {s}:{ut.split('_')[-1]}/{it.split('_')[-1]} & " + " & ".join(cells) + f" & {A} " + r"\\")
    tab = [r"\begin{table}[t]\centering",
           r"\caption{Per-target view of RQ2 (exploratory). Mean violations per campaign for each "
           r"reward arm on every selected target, with the paired $\hat A_{12}$ of proximity against "
           r"binary success over the five seeds. The judge-graded arm runs on Qwen3-8B only. A dash "
           r"marks an arm that was not run on that model.}",
           r"\label{tab:rq2targets}", r"\begin{tabular}{llccccc c}", r"\toprule",
           r"Model & Target & prox & rand & binary & cov & T-MAP & $\hat A_{12}$ p/b \\", r"\midrule"] + \
          rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(OUT, "w").write("\n".join(tab) + "\n")
    print(f"rq2_targets_table: {len(rows)} target rows")


if __name__ == "__main__":
    main()
