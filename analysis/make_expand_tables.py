"""Expansion tables: RQ2 per-operator outcome and RQ1 defense effect.
python3 make_expand_tables.py -> out/{rq2_ops_table,rq1_defense_table}.tex
All numbers computed from the frozen runs, no hand values.
"""
import collections
import glob
import json
import os
import re
import statistics as st
from fractions import Fraction
from efmt import F, exact, mean  # noqa: E402,F401  one half-up formatter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
RUNS = os.path.join(HERE, "..", "runs")
PAT = re.compile(r".+?_(None|both)_(banking|slack|travel)_(user_task_\d+)_(injection_task_\d+)_([a-z0-9-]+)_(\d+)\.jsonl$")
OPNAME = {"authority": "authority frame", "task_blend": "task-blend frame", "format": "structured-data format",
          "indirect": "indirect phrasing", "urgency": "urgency frame", "persona": "first-person persona",
          "split": "numbered-checklist split", "compress": "length compression"}


def rq2_ops():
    per = collections.defaultdict(lambda: collections.Counter())
    for f in glob.glob(os.path.join(RUNS, "rq2", "*_[0-9].jsonl")):
        if "tmap" in f:
            continue
        m = PAT.match(os.path.basename(f))
        if not m:
            continue
        for x in open(f):
            r = json.loads(x)
            if r.get("err") is not None:
                continue
            per[r["op"]]["n"] += 1
            per[r["op"]]["viol"] += bool(r["security"])
            per[r["op"]]["l3"] += (r.get("level_v21") == 3)
            per[r["op"]]["l4"] += bool(r["security"])
    rows = []
    for op, c in sorted(per.items(), key=lambda kv: -kv[1]["viol"] / max(1, kv[1]["n"])):
        rows.append(f"{OPNAME.get(op, op)} & {c['n']} & {F(Fraction(100 * c['viol'], max(1, c['n'])), '.1f')}\\% & "
                    f"{F(Fraction(100 * c['l3'], max(1, c['n'])), '.1f')}\\% " + r"\\")
    tab = [r"\begin{table}[t]\centering",
           r"\caption{Outcome by mutation operator over all fresh and replayed executions of the proximity, "
           r"random, binary-success and coverage reward \varwords{} on both models. Executions, the share that violates and the "
           r"share that reaches the level-three tier without violating. Operators are ordered by "
           r"violation share. The \rqtwoErrN{} executions that ended in an oracle error are left out.}",
           r"\label{tab:rq2ops}", r"\begin{tabular}{lccc}", r"\toprule",
           r"Operator & executions & violates $\uparrow$ & level 3 only \\", r"\midrule"] + rows + \
          [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "rq2_ops_table.tex"), "w").write("\n".join(tab) + "\n")
    print("rq2_ops:", [(OPNAME.get(o, o), round(c['viol'] / max(1, c['n']), 3)) for o, c in sorted(per.items(), key=lambda kv: -kv[1]['viol'] / max(1, kv[1]['n']))])


def rq1_defense():
    rows = []
    for mk, disp in [("qwen3-8b-yf", "Qwen3-8B"), ("qwen3-32b-local", "Qwen3-32B"),
                     ("llama31-8b-local", "Llama-3.1-8B"), ("granite31-8b-local", "Granite-3.1-8B")]:
        files = [f for f in glob.glob(os.path.join(RUNS, "rq1", f"census_{mk}*.jsonl")) if "ABORTED" not in f]
        R = [json.loads(l) for f in files for l in open(f)]
        R = [r for r in R if r.get("err") is None and r.get("trace") is not None]
        cells = collections.defaultdict(list)
        for r in R:
            cells[(r["suite"], r["ut"], r["it"], r["defense"])].append(r)
        rate = {k: mean(bool(x["security"]) for x in v) for k, v in cells.items()}
        pairs = [(rate[(su, ut, it, None)], rate[(su, ut, it, "both")]) for (su, ut, it, d) in rate
                 if d is None and (su, ut, it, "both") in rate]
        pu = [a for a, _ in pairs]
        pd = [b for _, b in pairs]
        if not pairs:
            continue
        drop = mean(a - b for a, b in pairs)
        rows.append(f"{disp} & {len(pairs)} & {F(mean(pu), '.3f')} & {F(mean(pd), '.3f')} & {F(drop, '.3f')} " + r"\\")
    tab = [r"\begin{table}[t]\centering",
           r"\caption{Effect of the combined prompt-level defense in RQ1, per model. Units that exist "
           r"in both defense settings, the mean trigger rate without and with the defense and the mean "
           r"per-unit drop. A positive drop means the defense lowers the trigger rate.}",
           r"\label{tab:rq1defense}", r"\begin{tabular}{lcccc}", r"\toprule",
           r"Model & paired units & no defense & defended & drop \\", r"\midrule"] + rows + \
          [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "rq1_defense_table.tex"), "w").write("\n".join(tab) + "\n")
    print("rq1_defense rows:", rows)


if __name__ == "__main__":
    rq2_ops()
    rq1_defense()
