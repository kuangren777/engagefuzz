"""RQ2 T-MAP-style graded-reward arm (pre-registered, plan.md §9, PM #238).

Reuses rq2_paired's estimator unchanged (a12 + Wilcoxon signed-rank + Holm, paired by CRN on
the unit = (seed, target)). Compares tmap-ladder vs binary and vs prox-v21. Primary metric is
violations per campaign; A12 > 0.5 favours tmap-ladder, > 0.60 one-sided Holm p < 0.05 would
mean it beats that baseline. Pre-registered directional expectation: tmap-ladder does NOT beat
binary.

python3 rq2_tmap.py <runs_dir> <model> [--json out.json]
Run from analysis/ (imports rq2_paired) or with analysis/ on PYTHONPATH.
"""
import json
import sys

import rq2_paired as R

R.MAIN = "tmap-ladder"
R.BASE = ["binary", "prox-v21"]


def main():
    runs, model = sys.argv[1], sys.argv[2]
    C = R.load(runs, model)
    budget = max(v["n"] for v in C.values())
    arms = {a: [v for k, v in C.items() if k[0] == a] for a in [R.MAIN] + R.BASE}
    res = {"model": model, "budget": budget, "n_units": len({k[1:] for k in C}), "arms": {}}
    print(f"== {model} tmap-ladder: units={res['n_units']} budget={budget} "
          f"err={sum(v['err'] for v in C.values())}")
    for a, L in arms.items():
        if not L:
            print(f"  {a}: NO DATA")
            continue
        memo = sum(v["hits"] for v in L) / max(1, sum(v["n"] for v in L))
        res["arms"][a] = dict(n=len(L), viol_mean=sum(v["viol"] for v in L) / max(1, len(L)),
                              fresh_viol_mean=sum(v["fresh_viol"] for v in L) / max(1, len(L)),
                              memo_hit=memo, replay_viol=sum(v["replay_viol"] for v in L))
        print(f"  {a:12s} n={len(L)} viol={res['arms'][a]['viol_mean']:.2f} "
              f"fresh={res['arms'][a]['fresh_viol_mean']:.2f} memo={memo:.2f} "
              f"replay_viol={res['arms'][a]['replay_viol']}")
    res["tests"] = {}
    for label, metric in [("primary_viol", "viol"), ("secondary_fresh", "fresh_viol")]:
        rows, adj = R.paired(C, metric)
        res["tests"][label] = {}
        for (b, A, p1, p2, n), ph in zip(rows, adj):
            res["tests"][label][b] = dict(A12=A, p_one_sided=p1, holm_p=ph, p_two_sided=p2, n_pairs=n)
            print(f"  [{label}] tmap-ladder vs {b:9s}: A12={A:.3f} 1-sided p={p1:.3g} "
                  f"Holm={ph:.3g} 2-sided p={p2:.3g} (n={n})")
    pv = res["tests"].get("primary_viol", {})
    beats_binary = pv.get("binary", {}).get("A12", 0) >= 0.60 and pv.get("binary", {}).get("holm_p", 1) < 0.05
    res["tmap_beats_binary"] = bool(beats_binary)
    print(f"== tmap-ladder beats binary (primary): {res['tmap_beats_binary']}")
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
