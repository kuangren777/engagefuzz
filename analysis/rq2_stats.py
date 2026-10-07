"""RQ2 statistics, pre-registered in plan.md §4 (frozen bedfedd3).

python3 rq2_stats.py <runs_dir> [--gate]

--gate : early stop gate (PM 2026-10-06). Looks only at row counts, errors and the prox-v21 level
         distribution of seed 0. Prints no between-arm comparison. STOP if level>=2 share < 5%.
default: per model, Vargha-Delaney A12 + one-sided Mann-Whitney (prox-v21 > baseline) on violations
         per campaign, Holm over 3 baselines; Kaplan-Meier median + log-rank on executions to first
         violation (censored at budget); targets with >=1 violation; memo hit / error rates.
"""
import collections
import glob
import json
import os
import re
import sys

from scipy.stats import mannwhitneyu

MAIN = "prox-v21"
BASE = ["random", "binary", "coverage"]
PAT = re.compile(r"(?P<model>.+?)_(?P<defense>None|both|spotlighting|repeat_user_prompt)_"
                 r"(?P<s>banking|slack|travel)_(?P<ut>user_task_\d+)_(?P<it>injection_task_\d+)_"
                 r"(?P<arm>[a-z0-9-]+)_(?P<seed>\d+)\.jsonl$")


def load(d):
    C = []
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        m = PAT.match(os.path.basename(f))
        if not m:
            continue
        R = [json.loads(x) for x in open(f)]
        first = next((i + 1 for i, r in enumerate(R) if r["security"]), None)
        C.append(dict(m.groupdict(), seed=int(m["seed"]), rows=R, n=len(R),
                      viol=sum(bool(r["security"]) for r in R), first=first,
                      distinct=len({r.get("text") for r in R if r["security"]}),
                      err=sum(r.get("err") is not None for r in R),
                      hits=sum(bool(r.get("memo_hit")) for r in R)))
    return C


def a12(x, y):
    gt = sum((a > b) + 0.5 * (a == b) for a in x for b in y)
    return gt / (len(x) * len(y))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def km_median(times, budget):
    """times: first-violation index or None (censored at budget). Returns KM median or None."""
    ev = sorted((t if t is not None else budget + 0.5, t is not None) for t in times)
    s, n = 1.0, len(ev)
    for i, (t, e) in enumerate(ev):
        if e:
            s *= 1 - 1 / (n - i)
            if s <= 0.5:
                return t
    return None


def logrank(t1, t2, budget):
    from scipy.stats import chi2
    data = [(t if t is not None else budget + 0.5, t is not None, g) for g, T in ((0, t1), (1, t2)) for t in T]
    times = sorted({t for t, e, _ in data if e})
    O = E = V = 0.0
    for t in times:
        at = [(tt, e, g) for tt, e, g in data if tt >= t]
        n, n1 = len(at), sum(1 for x in at if x[2] == 0)
        d = sum(1 for tt, e, g in at if tt == t and e)
        d1 = sum(1 for tt, e, g in at if tt == t and e and g == 0)
        O += d1; E += d * n1 / n
        V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1) if n > 1 else 0
    return float(chi2.sf((O - E) ** 2 / V, 1)) if V > 0 else float("nan")


def gate(C):
    print("EARLY STOP GATE (seed 0 only; no between-arm comparison)")
    for model in sorted({c["model"] for c in C}):
        S = [c for c in C if c["model"] == model and c["seed"] == 0]
        rows = sum(c["n"] for c in S); err = sum(c["err"] for c in S)
        P = [r for c in S if c["arm"] == MAIN for r in c["rows"] if r.get("level_v21") is not None]
        dist = collections.Counter(r["level_v21"] for r in P)
        share = sum(v for k, v in dist.items() if k >= 2) / max(1, len(P))
        print(f"  {model}: campaigns={len(S)} rows={rows} err={err} {MAIN} levels={dict(sorted(dist.items()))} "
              f"level>=2 share={share:.3f} -> {'STOP, report PM' if share < 0.05 else 'continue'}")


def main():
    _main()
    if "--json" in sys.argv:
        json.dump(OUT, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


def _main():
    d = sys.argv[1]
    C = load(d)
    if "--gate" in sys.argv:
        return gate(C)
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    quiet = "--quiet" in sys.argv
    if quiet:
        import io, contextlib
        buf = contextlib.redirect_stdout(io.StringIO()); buf.__enter__()
    for model in sorted({c["model"] for c in C}):
        M = [c for c in C if c["model"] == model]
        budget = max(c["n"] for c in M)
        print(f"== {model}: campaigns={len(M)} budget={budget} err={sum(c['err'] for c in M)} "
              f"memo_hit={sum(c['hits'] for c in M) / max(1, sum(c['n'] for c in M)):.3f}")
        by = {a: [c for c in M if c["arm"] == a] for a in [MAIN] + BASE}
        for a, L in by.items():
            tg = {(c["s"], c["ut"], c["it"]) for c in L if c["viol"]}
            print(f"  {a:9s} n={len(L)} viol/campaign mean={sum(c['viol'] for c in L) / max(1, len(L)):.2f} "
                  f"distinct mean={sum(c['distinct'] for c in L) / max(1, len(L)):.2f} "
                  f"targets_hit={len(tg)} KM_median_first={km_median([c['first'] for c in L], budget)}")
        # secondary (pre-registered a541fe02): distinct violating payloads per campaign
        xd = [c["distinct"] for c in by[MAIN]]
        psd, rowsd = [], []
        for b in BASE:
            yd = [c["distinct"] for c in by[b]]
            if xd and yd:
                p = mannwhitneyu(xd, yd, alternative="greater").pvalue
                psd.append(p); rowsd.append((b, a12(xd, yd), p))
        for (b, A, p), ph in zip(rowsd, holm(psd)):
            print(f"  [secondary distinct] {MAIN} vs {b:8s}: A12={A:.3f} MW p={p:.3g} Holm p={ph:.3g}")
        x = [c["viol"] for c in by[MAIN]]
        ps, rows = [], []
        for b in BASE:
            y = [c["viol"] for c in by[b]]
            if not x or not y:
                continue
            p = mannwhitneyu(x, y, alternative="greater").pvalue
            ps.append(p)
            rows.append((b, a12(x, y), p, logrank([c["first"] for c in by[MAIN]], [c["first"] for c in by[b]], budget)))
        for (b, A, p, lr), ph in zip(rows, holm(ps)):
            print(f"  {MAIN} vs {b:8s}: A12={A:.3f} MW p={p:.3g} Holm p={ph:.3g} logrank p={lr:.3g}")
        OUT[model] = dict(
            budget=budget, campaigns=len(M), err=sum(c["err"] for c in M),
            arms={a: dict(n=len(L), viol_mean=sum(c["viol"] for c in L) / max(1, len(L)),
                          hit_frac=sum(c["viol"] > 0 for c in L) / max(1, len(L)),
                          distinct_mean=sum(c["distinct"] for c in L) / max(1, len(L)),
                          targets_hit=len({(c["s"], c["ut"], c["it"]) for c in L if c["viol"]}),
                          km_median_first=km_median([c["first"] for c in L], budget)) for a, L in by.items()},
            tests={b: dict(A12=A, mw_p=p, holm_p=ph, logrank_p=lr) for (b, A, p, lr), ph in zip(rows, holm(ps))},
            secondary_distinct={b: dict(A12=A, mw_p=p, holm_p=ph) for (b, A, p), ph in zip(rowsd, holm(psd))})


OUT = {}

if __name__ == "__main__":
    main()
