"""RQ2 sensitivity (post-hoc, PM 2026-10-11): drop the one Qwen3-8B unit whose binary arm ran on a
different replica host than the other deterministic arms (seed 2, travel user_task_17/injection_task_5).
python3 rq2_crosshost.py -> runs/rq2/rq2_crosshost.json
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rq2_paired as R
C = R.load(os.path.join(HERE, "..", "runs", "rq2"), "qwen3-8b-local")
drop = [k for k in C if k[1] == 2 and k[2][:3] == ("travel", "user_task_17", "injection_task_5")]
for k in drop:
    del C[k]
out = {}
for metric in ("viol", "fresh_viol"):
    rows, adj = R.paired(C, metric)
    out[metric] = {b: dict(A12=A, p_one_sided=p1, holm_one_sided=ph, p_two_sided=pt, n_pairs=n) for (b, A, p1, pt, n), ph in zip(rows, adj)}
out["dropped"] = [str(k) for k in drop]
json.dump(out, open(os.path.join(HERE, "..", "runs", "rq2", "rq2_crosshost.json"), "w"), indent=1)
print(len(drop), "campaigns dropped")
for m in ("viol", "fresh_viol"):
    print(m, {b: (round(v["A12"], 3), f"{v['holm_one_sided']:.3g}", f"{v['p_two_sided']:.2g}", v["n_pairs"]) for b, v in out[m].items()})
