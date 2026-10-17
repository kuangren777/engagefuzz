"""RQ2 unique-violation view (post-hoc, PM cold read 2026-10-11 M8). The headline counts every confirmed
execution, including re-triggers of a known payload. Here each campaign is scored by (a) the number of
distinct violating payload texts and (b) whether it found any violation at all. Same estimator as
rq2_paired.py (paired by seed and target, A12, one-sided Wilcoxon with Holm, plus two-sided Holm).
python3 rq2_unique.py -> runs/rq2/rq2_unique.json
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rq2_paired as R

RUNS = os.path.join(HERE, "..", "runs", "rq2")


def main():
    out = {}
    for model in ("qwen3-8b-local", "llama31-8b-local"):
        C = R.load(RUNS, model)
        for k, v in C.items():
            v["any_viol"] = int(v["viol"] > 0)
        res = {"arms": {}}
        for a in [R.MAIN] + R.BASE:
            L = [v for k, v in C.items() if k[0] == a]
            res["arms"][a] = dict(n=len(L), distinct_mean=sum(v["distinct"] for v in L) / max(1, len(L)),
                                  success_share=sum(v["any_viol"] for v in L) / max(1, len(L)))
        for metric in ("distinct", "any_viol"):
            rows, adj = R.paired(C, metric)
            p2 = sorted((r[3], r[0]) for r in rows)
            adj2, run = {}, 0.0
            for k, (pv, b) in enumerate(p2):
                run = max(run, min(1.0, (len(p2) - k) * pv)); adj2[b] = run
            res[metric] = {b: dict(A12=A, p_one_sided=p1, holm_one_sided=ph, p_two_sided=pt, holm_two_sided=adj2[b], n_pairs=n)
                           for (b, A, p1, pt, n), ph in zip(rows, adj)}
        out[model] = res
        print(model, {a: (round(v["distinct_mean"], 2), round(v["success_share"], 3)) for a, v in res["arms"].items()})
        for metric in ("distinct", "any_viol"):
            print("  ", metric, {b: (round(v["A12"], 3), f"{v['holm_one_sided']:.3g}", f"{v['holm_two_sided']:.3g}") for b, v in res[metric].items()})
    json.dump(out, open(os.path.join(RUNS, "rq2_unique.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
