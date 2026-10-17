"""E-PX3 (plan §10.3): equal-budget search with a separate memo per arm, so every execution an arm counts
was produced by that arm. Arms binary, prox-v21, tmap-ladder on the 8 Qwen3-8B RQ2 targets x seeds 0-4,
budget 40.
Primary: distinct violating payload texts per campaign. Paired by (seed, target): A12 and two-sided
Wilcoxon signed-rank for the three pairs, Holm over the three.
Secondary: executions to first success, censored at 40; Kaplan-Meier median per arm, pairwise log-rank.
python3 px3_fresh.py -> runs/px/px3.json
"""
import glob, json, math, os, re
from scipy.stats import wilcoxon, chi2

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "runs", "rq2_fresh")
PAT = re.compile(r"qwen3-8b-local_(None|both)_(\w+?)_(user_task_\d+)_(injection_task_\d+)_([a-z0-9-]+)_(\d+)\.jsonl$")
ARMS = ["binary", "prox-v21", "tmap-ladder"]
PAIRS = [("prox-v21", "binary"), ("tmap-ladder", "binary"), ("tmap-ladder", "prox-v21")]
BUDGET = 40


def load():
    C = {}
    for a in ARMS:
        for f in glob.glob(os.path.join(D, a, "qwen3-8b-local_*_[0-9].jsonl")):
            m = PAT.search(f)
            dfn, s, ut, it, arm, seed = m.groups()
            R = [json.loads(l) for l in open(f)]
            first = next((i + 1 for i, r in enumerate(R) if r.get("security")), None)
            C[(a, int(seed), (s, ut, it, dfn))] = dict(
                n=len(R), viol=sum(bool(r.get("security")) for r in R),
                distinct=len({r.get("text") for r in R if r.get("security")}),
                first=first, err=sum(r.get("err") is not None for r in R),
                hits=sum(bool(r.get("memo_hit")) for r in R))
    return C


def a12(x, y):
    return sum((a > b) + 0.5 * (a == b) for a, b in zip(x, y)) / len(x)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def km_median(times, events):
    s, at = 1.0, len(times)
    for t in sorted(set(t for t, e in zip(times, events) if e)):
        d = sum(1 for tt, e in zip(times, events) if tt == t and e)
        at = sum(1 for tt in times if tt >= t)
        s *= 1 - d / at
        if s <= 0.5:
            return t
    return None   # median not reached within the budget


def logrank(t1, e1, t2, e2):
    times = sorted(set([t for t, e in zip(t1, e1) if e] + [t for t, e in zip(t2, e2) if e]))
    O1 = E1 = V = 0.0
    for t in times:
        n1 = sum(1 for x in t1 if x >= t); n2 = sum(1 for x in t2 if x >= t)
        d1 = sum(1 for x, e in zip(t1, e1) if x == t and e); d2 = sum(1 for x, e in zip(t2, e2) if x == t and e)
        n, d = n1 + n2, d1 + d2
        if n < 2:
            continue
        O1 += d1; E1 += d * n1 / n
        V += d * (n1 / n) * (n2 / n) * (n - d) / (n - 1)
    stat = (O1 - E1) ** 2 / V if V > 0 else 0.0
    return stat, 1 - chi2.cdf(stat, 1)


def main():
    C = load()
    units = sorted({k[1:] for k in C if all((a,) + k[1:] in C for a in ARMS)})
    out = {"units": len(units), "arms": {}, "primary": {}, "secondary": {}}
    for a in ARMS:
        L = [C[(a,) + u] for u in units]
        tt = [v["first"] or BUDGET for v in L]; ee = [v["first"] is not None for v in L]
        out["arms"][a] = dict(executions=sum(v["n"] for v in L), errors=sum(v["err"] for v in L), memo_hits=sum(v["hits"] for v in L),
                              distinct_mean=sum(v["distinct"] for v in L) / len(L), viol_mean=sum(v["viol"] for v in L) / len(L),
                              success_share=sum(ee) / len(L), km_median_first=km_median(tt, ee))
    ps = []
    for a, b in PAIRS:
        x = [C[(a,) + u]["distinct"] for u in units]; y = [C[(b,) + u]["distinct"] for u in units]
        p = wilcoxon(x, y, zero_method="wilcox").pvalue if any(i != j for i, j in zip(x, y)) else 1.0
        out["primary"][f"{a}_vs_{b}"] = dict(A12=a12(x, y), p_two_sided=p); ps.append(p)
    for (a, b), h in zip(PAIRS, holm(ps)):
        out["primary"][f"{a}_vs_{b}"]["holm"] = h
    for a, b in PAIRS:
        ta = [C[(a,) + u]["first"] or BUDGET for u in units]; ea = [C[(a,) + u]["first"] is not None for u in units]
        tb = [C[(b,) + u]["first"] or BUDGET for u in units]; eb = [C[(b,) + u]["first"] is not None for u in units]
        st, p = logrank(ta, ea, tb, eb)
        out["secondary"][f"{a}_vs_{b}"] = dict(logrank_chi2=st, p=p)
    json.dump(out, open(os.path.join(HERE, "..", "runs", "px", "px3.json"), "w"), indent=1)
    print("units", len(units))
    for a, v in out["arms"].items():
        print(f"  {a:12} exec={v['executions']} err={v['errors']} memo_hits={v['memo_hits']} distinct={v['distinct_mean']:.2f} "
              f"viol={v['viol_mean']:.2f} success={v['success_share']:.3f} KMmedian={v['km_median_first']}")
    for k, v in out["primary"].items():
        print(f"  primary {k:26} A12={v['A12']:.3f} p2={v['p_two_sided']:.3g} Holm={v['holm']:.3g}")
    for k, v in out["secondary"].items():
        print(f"  logrank {k:26} chi2={v['logrank_chi2']:.2f} p={v['p']:.3g}")


if __name__ == "__main__":
    main()
