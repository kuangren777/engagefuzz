"""RQ1 with the target as cluster (post-hoc, codex round 1 R1/S3). A unit is a target under one defense
setting, so each target gives two units. This resamples targets, keeping both units of a target together,
for (a) each model's same-repetition fail-only Spearman rho, (b) the pooled model-centred same-repetition
rho and (c) the pooled held-out rho (even reps -> level, odd reps -> rate, as rq1_heldout.py).
Reports the 95% cluster-bootstrap interval and the share of bootstrap draws with rho <= 0 (one-sided p).
python3 rq1_cluster.py -> runs/px/rq1_cluster.json
"""
import collections, glob, json, os, random, statistics as st, sys
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import engagefuzz as ef  # noqa: E402
from rq1_heldout import MODELS  # noqa: E402

B, SEED = 4000, 20261011


def units(mk):
    files = [f for f in glob.glob(os.path.join(HERE, "..", "runs", "rq1", f"census_{mk}*.jsonl")) if "ABORTED" not in f]
    R = [json.loads(l) for f in files for l in open(f)]
    R = [r for r in R if r.get("err") is None and r.get("trace") is not None]
    cells = collections.defaultdict(list)
    for r in R:
        r["_lv"] = ef.prox_v2.feats(r)["level21"]
        cells[(r["suite"], r["ut"], r["it"], r["defense"])].append(r)
    same, held = [], []   # (target, x, y)
    for (s, ut, it, d), v in cells.items():
        fa = [x for x in v if not x["security"]]
        if fa:
            same.append(((s, ut, it), st.mean(x["_lv"] for x in fa), st.mean(bool(x["security"]) for x in v)))
        A = [x for x in v if x["rep"] % 2 == 0]; Bv = [x for x in v if x["rep"] % 2 == 1]
        fA = [x for x in A if not x["security"]]
        if A and Bv and fA:
            held.append(((s, ut, it), st.mean(x["_lv"] for x in fA), st.mean(bool(x["security"]) for x in Bv)))
    return same, held


def rho(rows):
    X = [r[1] for r in rows]; Y = [r[2] for r in rows]
    if len(X) < 3 or len(set(X)) < 2 or len(set(Y)) < 2:
        return float("nan")
    return spearmanr(X, Y)[0]


def centred(per_model):
    out = []
    for rows in per_model:
        if rows:
            mx = st.mean(r[1] for r in rows); my = st.mean(r[2] for r in rows)
            out += [(r[0], r[1] - mx, r[2] - my) for r in rows]
    return out


def boot(per_model, rng, pooled):
    """per_model: list of row lists; clusters = targets within each model."""
    groups = []
    for rows in per_model:
        g = collections.defaultdict(list)
        for r in rows:
            g[r[0]].append(r)
        groups.append(list(g.values()))
    vals = []
    for _ in range(B):
        draw = [[r for _ in gl for r in gl[rng.randrange(len(gl))]] if gl else [] for gl in groups]
        v = rho(centred(draw)) if pooled else rho(draw[0])
        if v == v:
            vals.append(v)
    vals.sort()
    return dict(ci95=[vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]],
                p_one_sided=sum(v <= 0 for v in vals) / len(vals), draws=len(vals))


def main():
    rng = random.Random(SEED)
    S, H, out = [], [], {"per_model": {}}
    for mk, tag in MODELS:
        same, held = units(mk)
        S.append(same); H.append(held)
        out["per_model"][mk] = dict(rho=rho(same), n=len(same), **boot([same], rng, False))
    out["pooled_same"] = dict(rho=rho(centred(S)), **boot(S, rng, True))
    out["pooled_heldout"] = dict(rho=rho(centred(H)), **boot(H, rng, True))
    json.dump(out, open(os.path.join(HERE, "..", "runs", "px", "rq1_cluster.json"), "w"), indent=1)
    for k, v in out["per_model"].items():
        print(f"{k:20} rho={v['rho']:.3f} n={v['n']} CI[{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}] p1={v['p_one_sided']:.4f}")
    for k in ("pooled_same", "pooled_heldout"):
        v = out[k]; print(f"{k}: rho={v['rho']:.3f} CI[{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}] p1={v['p_one_sided']:.4f}")


if __name__ == "__main__":
    main()
