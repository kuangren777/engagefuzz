"""RQ1 held-out control on undefended units only (post-hoc, PM cold read 2026-10-11 M4). Defended units
have 5 repetitions, so their held-out trigger rate rests on 2-3 runs. This keeps the 42 undefended
units (10 repetitions, 5 per half) with the rq1_heldout.py split and estimator unchanged.
python3 rq1_heldout_undef.py -> runs/px/rq1_heldout_undef.json
"""
import collections, glob, json, os, statistics as st, sys
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import engagefuzz as ef  # noqa: E402
from rq1_heldout import MODELS  # noqa: E402

out, PX, PY, npass = {}, [], [], 0
for mk, tag in MODELS:
    files = [f for f in glob.glob(os.path.join(HERE, "..", "runs", "rq1", f"census_{mk}*.jsonl")) if "ABORTED" not in f]
    R = [json.loads(l) for f in files for l in open(f)]
    R = [r for r in R if r.get("err") is None and r.get("trace") is not None and r["defense"] is None]
    cells = collections.defaultdict(list)
    for r in R:
        r["_lv"] = ef.prox_v2.feats(r)["level21"]
        cells[(r["suite"], r["ut"], r["it"])].append(r)
    X, Y = [], []
    for v in cells.values():
        A = [x for x in v if x["rep"] % 2 == 0]; B = [x for x in v if x["rep"] % 2 == 1]
        fa = [x for x in A if not x["security"]]
        if A and B and fa:
            X.append(st.mean(x["_lv"] for x in fa)); Y.append(st.mean(bool(x["security"]) for x in B))
    rr = spearmanr(X, Y) if len(X) > 2 and len(set(X)) > 1 and len(set(Y)) > 1 else (float("nan"), float("nan"))
    sig = bool(rr[1] < 0.05 and rr[0] > 0); npass += sig
    out[mk] = dict(n=len(X), rho=rr[0], p=rr[1], sig=sig)
    if X:
        mx, my = st.mean(X), st.mean(Y); PX += [a - mx for a in X]; PY += [b - my for b in Y]
    print(f"{mk:20} undefended held-out rho={rr[0]:.3f} p={rr[1]:.3g} n={len(X)}")
pr = spearmanr(PX, PY)
out["pooled"] = dict(n=len(PX), rho=pr[0], p=pr[1]); out["npass"] = npass
json.dump(out, open(os.path.join(HERE, "..", "runs", "px", "rq1_heldout_undef.json"), "w"), indent=1)
print(f"pooled rho={pr[0]:.3f} p={pr[1]:.3g} n={len(PX)} | per-model sig {npass}/4")
