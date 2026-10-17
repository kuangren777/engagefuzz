"""Plan §11 exploratory definition ablation (cloud A100, five arms, separate memo per arm).
Same estimator as E-PX3 (px3_fresh.py): distinct violating payload texts per campaign, paired by
(seed, target), A12 + two-sided Wilcoxon, Holm over the three ablation arms; first-success log-rank.
Comparisons: each of prox-nowf, prox-l3, prox-perm against prox-v21 (Holm over 3) and against binary
(Holm over 3); binary against prox-v21 as the cross-host replication of E-PX3.
python3 ablate_r3.py -> runs/px/ablate_r3.json
"""
import glob, json, os
from scipy.stats import wilcoxon
import px3_fresh as X

HERE = os.path.dirname(os.path.abspath(__file__))
X.D = os.path.join(HERE, "..", "runs", "rq2_ablate")
X.ARMS = ["binary", "prox-v21", "prox-nowf", "prox-l3", "prox-perm"]
ABL = ["prox-nowf", "prox-l3", "prox-perm"]


def pair_test(C, units, a, b):
    x = [C[(a,) + u]["distinct"] for u in units]; y = [C[(b,) + u]["distinct"] for u in units]
    p = wilcoxon(x, y, zero_method="wilcox").pvalue if any(i != j for i, j in zip(x, y)) else 1.0
    return dict(A12=X.a12(x, y), p_two_sided=p)


def main():
    C = X.load()
    units = sorted({k[1:] for k in C if all((a,) + k[1:] in C for a in X.ARMS)})
    out = {"units": len(units), "arms": {}, "vs_prox": {}, "vs_binary": {}, "logrank_vs_prox": {}}
    for a in X.ARMS:
        L = [C[(a,) + u] for u in units]
        tt = [v["first"] or X.BUDGET for v in L]; ee = [v["first"] is not None for v in L]
        out["arms"][a] = dict(executions=sum(v["n"] for v in L), errors=sum(v["err"] for v in L),
                              memo_hits=sum(v["hits"] for v in L),
                              distinct_mean=sum(v["distinct"] for v in L) / len(L),
                              success_share=sum(ee) / len(L), km_median_first=X.km_median(tt, ee))
    for fam, ref in (("vs_prox", "prox-v21"), ("vs_binary", "binary")):
        res = {a: pair_test(C, units, a, ref) for a in ABL}
        for a, h in zip(ABL, X.holm([res[a]["p_two_sided"] for a in ABL])):
            res[a]["holm"] = h
        out[fam] = res
    out["binary_vs_prox_replication"] = pair_test(C, units, "prox-v21", "binary")
    for a in ABL:
        ta = [C[(a,) + u]["first"] or X.BUDGET for u in units]; ea = [C[(a,) + u]["first"] is not None for u in units]
        tb = [C[("prox-v21",) + u]["first"] or X.BUDGET for u in units]; eb = [C[("prox-v21",) + u]["first"] is not None for u in units]
        st, p = X.logrank(ta, ea, tb, eb)
        out["logrank_vs_prox"][a] = dict(chi2=st, p=p)
    json.dump(out, open(os.path.join(HERE, "..", "runs", "px", "ablate_r3.json"), "w"), indent=1)
    print("units", len(units))
    for a, v in out["arms"].items():
        print(f"  {a:10} exec={v['executions']} err={v['errors']} memo={v['memo_hits']} distinct={v['distinct_mean']:.2f} "
              f"success={v['success_share']:.3f} KM={v['km_median_first']}")
    for fam in ("vs_prox", "vs_binary"):
        for a, v in out[fam].items():
            print(f"  {fam:9} {a:10} A12={v['A12']:.3f} p2={v['p_two_sided']:.3g} Holm={v['holm']:.3g}")
    r = out["binary_vs_prox_replication"]
    print(f"  replication prox-v21 vs binary A12={r['A12']:.3f} p2={r['p_two_sided']:.3g}")
    for a, v in out["logrank_vs_prox"].items():
        print(f"  logrank {a:10} vs prox p={v['p']:.3g}")


if __name__ == "__main__":
    main()
