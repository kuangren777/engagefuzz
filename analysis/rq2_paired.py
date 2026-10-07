"""RQ2 paired analysis (corrected design, pre-registered before the rerun).

Seeds are independent (memo now seed-scoped); arms are paired within a (seed, target) via common
random numbers. Unit = (seed, target). For prox-v21 vs each baseline:
  primary   : violations per campaign, paired A12 + Wilcoxon signed-rank (two-sided reported,
              one-sided greater for the C1 decision), Holm over the 3 baselines.
  secondary : (a) distinct violating payloads per campaign; (b) fresh-only violations (executions
              with memo_hit False); (c) KM median executions-to-first-violation + log-rank.
C1 PASS iff paired A12 >= 0.60 and Holm one-sided p < 0.05 vs every baseline.
Also reports per arm: memo-hit rate, replay violations, per-target and per-suite violations.

python3 rq2_paired.py <runs_dir> <model> [--json out.json]
"""
import collections
import glob
import json
import os
import re
import sys

from scipy.stats import wilcoxon

MAIN = "prox-v21"
BASE = ["random", "binary", "coverage"]
PAT = re.compile(r".+?_(None|both)_(banking|slack|travel)_(user_task_\d+)_(injection_task_\d+)_([a-z0-9-]+)_(\d+)\.jsonl$")


def load(runs, model):
    C = {}
    for f in glob.glob(os.path.join(runs, f"{model}_*_[0-9].jsonl")):
        m = PAT.match(os.path.basename(f))
        if not m:
            continue
        dfn, s, ut, it, arm, seed = m.groups()
        R = [json.loads(x) for x in open(f)]
        first = next((i + 1 for i, r in enumerate(R) if r["security"]), None)
        C[(arm, int(seed), (s, ut, it, dfn))] = dict(
            rows=R, n=len(R), viol=sum(bool(r["security"]) for r in R),
            distinct=len({r.get("text") for r in R if r["security"]}),
            fresh_viol=sum(bool(r["security"]) and not r.get("memo_hit") for r in R),
            first=first, hits=sum(bool(r.get("memo_hit")) for r in R),
            replay_viol=sum(bool(r["security"]) and r.get("memo_hit") for r in R),
            err=sum(r.get("err") is not None for r in R), suite=s)
    return C


def a12(x, y):
    gt = sum((a > b) + 0.5 * (a == b) for a, b in zip(x, y))
    return gt / len(x)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def km_median(times, budget):
    ev = sorted((t if t is not None else budget + 0.5, t is not None) for t in times)
    s, n = 1.0, len(ev)
    for i, (t, e) in enumerate(ev):
        if e:
            s *= 1 - 1 / (n - i)
            if s <= 0.5:
                return t
    return None


def paired(C, metric, budget=None):
    """Return per-baseline (A12, one-sided p, two-sided p) on matched (seed,target) units."""
    units = sorted({k[1:] for k in C})
    out, ps1 = [], []
    for b in BASE:
        xs, ys = [], []
        for seed, tgt in units:
            if (MAIN, seed, tgt) in C and (b, seed, tgt) in C:
                xs.append(C[(MAIN, seed, tgt)][metric]); ys.append(C[(b, seed, tgt)][metric])
        if not xs:
            continue
        A = a12(xs, ys)
        d = [a - b_ for a, b_ in zip(xs, ys)]
        if any(d):
            p2 = wilcoxon(xs, ys, zero_method="wilcox").pvalue
            p1 = wilcoxon(xs, ys, alternative="greater", zero_method="wilcox").pvalue
        else:
            p2 = p1 = 1.0
        out.append((b, A, p1, p2, len(xs))); ps1.append(p1)
    return out, holm(ps1)


def main():
    runs, model = sys.argv[1], sys.argv[2]
    C = load(runs, model)
    budget = max(v["n"] for v in C.values())
    arms = {a: [v for k, v in C.items() if k[0] == a] for a in [MAIN] + BASE}
    res = {"model": model, "budget": budget, "n_units": len({k[1:] for k in C}), "arms": {}}
    print(f"== {model}: units={res['n_units']} budget={budget} err={sum(v['err'] for v in C.values())}")
    for a, L in arms.items():
        memo = sum(v["hits"] for v in L) / max(1, sum(v["n"] for v in L))
        bysuite = collections.Counter()
        for v in L:
            bysuite[v["suite"]] += v["viol"]
        res["arms"][a] = dict(
            n=len(L), viol_mean=sum(v["viol"] for v in L) / max(1, len(L)),
            distinct_mean=sum(v["distinct"] for v in L) / max(1, len(L)),
            fresh_viol_mean=sum(v["fresh_viol"] for v in L) / max(1, len(L)),
            memo_hit=memo, replay_viol=sum(v["replay_viol"] for v in L),
            km_median_first=km_median([v["first"] for v in L], budget),
            per_suite=dict(bysuite))
        print(f"  {a:9s} viol={res['arms'][a]['viol_mean']:.2f} distinct={res['arms'][a]['distinct_mean']:.2f} "
              f"fresh={res['arms'][a]['fresh_viol_mean']:.2f} memo={memo:.2f} replay_viol={res['arms'][a]['replay_viol']} "
              f"suite={dict(bysuite)}")
    res["tests"] = {}
    for label, metric in [("primary_viol", "viol"), ("secondary_distinct", "distinct"),
                          ("secondary_fresh", "fresh_viol")]:
        rows, adj = paired(C, metric)
        res["tests"][label] = {}
        for (b, A, p1, p2, n), ph in zip(rows, adj):
            res["tests"][label][b] = dict(A12=A, p_one_sided=p1, holm_p=ph, p_two_sided=p2, n_pairs=n)
            print(f"  [{label}] {MAIN} vs {b:8s}: A12={A:.3f} 1-sided p={p1:.3g} Holm={ph:.3g} 2-sided p={p2:.3g} (n={n})")
    pv = res["tests"]["primary_viol"]
    passed = all(pv[b]["A12"] >= 0.60 and pv[b]["holm_p"] < 0.05 for b in BASE if b in pv)
    res["C1_verdict"] = "PASS" if (passed and len(pv) == 3) else "FAIL"
    print(f"== C1 (paired, primary=violations): {res['C1_verdict']}")
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
