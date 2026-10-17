"""RQ4 frontier density table from the frozen InjecAgent frontier runs.
python3 make_rq4_frontier_table.py -> out/rq4_frontier_table.tex
"""
import json
import os
from efmt import F, exact, mean  # noqa: E402,F401  one half-up formatter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "runs", "rq4", "injecagent_frontier_density.json")
OUT = os.path.join(HERE, "..", "out", "rq4_frontier_table.tex")

d = json.load(open(SRC))
rows = []
for m in sorted(d, key=lambda k: -d[k]["asr"]):
    v = d[m]
    rows.append(f"{m} & {v['n']} & {F(100 * exact(v['asr'], v['n']), '.1f')}\\% & {F(100 * exact(v['l2plus_nonviol'], v['n']), '.1f')}\\% " + r"\\")
tab = [r"\begin{table}[t]\centering",
       r"\caption{InjecAgent frontier density per model. Runs per model, attack-success rate and the "
       r"share of binary-failure runs at level two or higher under the pre-registered ladder.}",
       r"\label{tab:rq4frontiertable}", r"\begin{tabular}{lccc}", r"\toprule",
       r"Model & runs & ASR & level $\geq$ 2 among failures $\uparrow$ \\", r"\midrule"] + rows + \
      [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(OUT, "w").write("\n".join(tab) + "\n")
print("rq4_frontier_table:", len(rows), "models")
