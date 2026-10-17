"""RQ1 predictive validity, one-shot analysis (pre-registered in plan.md §6 before any census was read).

python3 rq1_stats.py ../runs/rq1 [--json out.json]

Unit = (cell, defense) on the 42 RQ1 cells (10 reps undefended, 5 under `both`).
Rows with err (connection, ctx overflow) are missing. ABORTED files are never read.
Primary per model: fail-only Spearman rho between the unit's mean v2.1 level over its failed runs
and the unit's trigger rate; minfail = 1 (primary) and 3 (reported); 95% CI by unit bootstrap (2000).
Also per model: Kendall tau_b between run-level v2.1 level and its unit's trigger rate (monotonicity,
tie-aware); the same fail-only rho for v2, v1 and read-only coverage (distinct read-only tools).
Pooled: the same rho on model-centred values (unit mean level and rate minus the model mean).
"""
import collections
import glob
import json
import os
import random
import re
import statistics as st
import sys

from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import engagefuzz as ef  # noqa: E402  (prox_v2, reduce_prox, SIDE_PREFIX via harness path)

MODELS = ["qwen3-8b-yf", "llama31-8b-local", "qwen3-32b-local", "granite31-8b-local"]
PASS_K = 3   # C2: Holm-adjusted one-sided p < 0.05 on >= PASS_K models (4 of 5 once gpt-oss joins)


def load(d):
    R = collections.defaultdict(list)
    for f in glob.glob(os.path.join(d, "census_*.jsonl")):
        if "ABORTED" in f:
            continue
        m = re.match(r"census_(.+?)(?:_p\d+)?\.jsonl$", os.path.basename(f))
        for line in open(f):
            r = json.loads(line)
            if r.get("err") is None and r.get("trace") is not None:
                R[m.group(1)].append(r)
    return R


def feats(r):
    p2 = ef.prox_v2.feats(r)
    return dict(v21=p2["level21"], v2=p2["level2"], v1=ef.reduce_prox.feats(r)["level"], cov=p2["n_ro"],
                sec=int(bool(r["security"])), unit=(r["suite"], r["ut"], r["it"], r["defense"]))


def units(F, key, minfail):
    U = collections.defaultdict(list)
    for f in F:
        U[f["unit"]].append(f)
    X, Y, K = [], [], []
    for u, v in U.items():
        fail = [f for f in v if not f["sec"]]
        if len(fail) >= minfail:
            X.append(st.mean(f[key] for f in fail)); Y.append(st.mean(f["sec"] for f in v)); K.append(u)
    return X, Y, K


def rho_ci(X, Y, B=2000, seed=0):
    rng = random.Random(seed)
    n, out = len(X), []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        xs, ys = [X[i] for i in idx], [Y[i] for i in idx]
        if len(set(xs)) > 1 and len(set(ys)) > 1:
            out.append(spearmanr(xs, ys)[0])
    out.sort()
    return (round(float(out[int(0.025 * len(out))]), 3), round(float(out[int(0.975 * len(out)) - 1]), 3)) if out else (None, None)


def main(d, out_json=None):
    R = load(d)
    res, pooled = {}, {"X": [], "Y": []}
    for m in MODELS:
        if m not in R:
            print(f"== {m}: no census rows"); continue
        F = [feats(r) for r in R[m]]
        U = collections.Counter(f["unit"] for f in F)
        rate = {u: st.mean(f["sec"] for f in F if f["unit"] == u) for u in U}
        tau = kendalltau([f["v21"] for f in F], [rate[f["unit"]] for f in F])
        lv = collections.defaultdict(list)
        for f in F:
            lv[f["v21"]].append(rate[f["unit"]])
        r_m = dict(rows=len(F), units=len(U), asr=st.mean(f["sec"] for f in F),
                   mixed_units=sum(0 < v < 1 for v in rate.values()),
                   level_to_rate={k: (len(v), round(st.mean(v), 3)) for k, v in sorted(lv.items())},
                   kendall_tau_b=tau[0], kendall_p=tau[1], rho={})
        for key in ("v21", "v2", "v1", "cov"):
            for mf in (1, 3):
                X, Y, _ = units(F, key, mf)
                ok = len(X) > 2 and len(set(X)) > 1 and len(set(Y)) > 1
                rr = spearmanr(X, Y, alternative="greater") if ok else (float("nan"), float("nan"))
                ci = rho_ci(X, Y) if (ok and key == "v21") else (None, None)
                r_m["rho"][f"{key}_minfail{mf}"] = dict(n=len(X), rho=rr[0], p=rr[1], ci95=ci)
        X, Y, _ = units(F, "v21", 1)
        mx, my = (st.mean(X), st.mean(Y)) if X else (0, 0)
        pooled["X"] += [x - mx for x in X]; pooled["Y"] += [y - my for y in Y]
        res[m] = r_m
        p = r_m["rho"]["v21_minfail1"]
        print(f"== {m}: rows={len(F)} units={len(U)} ASR={r_m['asr']:.3f} mixed={r_m['mixed_units']} "
              f"tau_b={tau[0]:.3f} (p={tau[1]:.3g})")
        print(f"   level->(n, mean unit rate) {r_m['level_to_rate']}")
        print(f"   fail-only rho v2.1 minfail1 n={p['n']} rho={p['rho']:.3f} p={p['p']:.3g} CI95={p['ci95']}")
        for k in ("v21_minfail3", "v2_minfail1", "v1_minfail1", "cov_minfail1"):
            q = r_m["rho"][k]
            print(f"   {k:14s} n={q['n']} rho={q['rho']:.3f} p={q['p']:.3g}")
    # C2 decision (PM 2026-10-06): one-sided rho > 0 per model, Holm over the models present
    ms = [m for m in MODELS if m in res]
    ps = [res[m]["rho"]["v21_minfail1"]["p"] for m in ms]
    ps = [1.0 if p != p else p for p in ps]                     # nan -> 1
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    passed = [m for m, a in zip(ms, adj) if a < 0.05 and res[m]["rho"]["v21_minfail1"]["rho"] > 0]
    for m, a in zip(ms, adj):
        res[m]["holm_p"] = a
        print(f"   C2 {m:20s} rho={res[m]['rho']['v21_minfail1']['rho']:.3f} one-sided p={res[m]['rho']['v21_minfail1']['p']:.3g} Holm p={a:.3g}")
    res["C2"] = dict(models=ms, passed=passed, need=PASS_K, verdict="PASS" if len(passed) >= PASS_K else "FAIL")
    print(f"== C2 verdict: {res['C2']['verdict']} ({len(passed)}/{len(ms)} models pass, need {PASS_K})")
    # secondary, report only: DerSimonian-Laird random-effects pooled rho over models (Fisher z, var 1.06/(n-3))
    import math
    zs = [(math.atanh(max(-0.999, min(0.999, res[m]["rho"]["v21_minfail1"]["rho"]))),
           1.06 / (res[m]["rho"]["v21_minfail1"]["n"] - 3)) for m in ms if res[m]["rho"]["v21_minfail1"]["n"] > 3
          and res[m]["rho"]["v21_minfail1"]["rho"] == res[m]["rho"]["v21_minfail1"]["rho"]]
    if len(zs) >= 2:
        w = [1 / v for _, v in zs]
        zf = sum(wi * z for wi, (z, _) in zip(w, zs)) / sum(w)
        Q = sum(wi * (z - zf) ** 2 for wi, (z, _) in zip(w, zs))
        t2 = max(0.0, (Q - (len(zs) - 1)) / (sum(w) - sum(wi * wi for wi in w) / sum(w)))
        wr = [1 / (v + t2) for _, v in zs]
        zr = sum(wi * z for wi, (z, _) in zip(wr, zs)) / sum(wr)
        se = (1 / sum(wr)) ** 0.5
        res["random_effects"] = dict(k=len(zs), rho=math.tanh(zr), ci95=(math.tanh(zr - 1.96 * se), math.tanh(zr + 1.96 * se)),
                                     tau2=t2, Q=Q)
        print(f"== random-effects pooled rho (report only): {math.tanh(zr):.3f} "
              f"CI95=({math.tanh(zr - 1.96 * se):.3f}, {math.tanh(zr + 1.96 * se):.3f}) k={len(zs)} tau2={t2:.3f}")
    if len(pooled["X"]) > 2:
        rr = spearmanr(pooled["X"], pooled["Y"])
        res["pooled_model_centred"] = dict(n=len(pooled["X"]), rho=rr[0], p=rr[1])
        print(f"== pooled (model-centred, v2.1, minfail1): n={len(pooled['X'])} rho={rr[0]:.3f} p={rr[1]:.3g}")
    if out_json:
        json.dump(res, open(out_json, "w"), indent=1, default=str)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[a.index("--json") + 1] if "--json" in a else None)
