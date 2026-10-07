"""Pre-registered RQ2 target selection (plan.md §2.3, frozen at bedfedd3).

python3 run_select.py --model qwen3-8b-local --out ../runs/select [--pool-seed 1] [--dry]

Stage A: each pool cell, undefended, 6 `random`-arm executions.
Stage B: A hits <= 1 -> 18 more undefended (n=24); A hits >= 2 -> 24 fresh under `both`.
Eligible (cell, defense): 1..3 hits of 24. Pick 8: suite round-robin banking->slack->travel,
within suite smallest |rate - 0.08|, ties by pool order. Writes selected_<model>.json.
Concurrency: cells run 2 at a time x 4 threads = 8. Resumable via Campaign logs.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "testbed"))
from exp_census import pairs  # noqa: E402

from engagefuzz import Campaign, Memo, served_root  # noqa: E402

DESIGN = os.path.join(_HERE, "..", "testbed", "data", "small.jsonl")
SUITES = ("banking", "slack", "travel")


def partitions():
    D = {(r["suite"], r["ut"], r["it"]) for r in map(json.loads, open(DESIGN))}
    p2 = [p for p in pairs(seed=1, per_suite=10) if p not in D]
    p1 = [p for p in pairs(seed=2, per_suite=15) if p not in D and p not in p2]
    assert not (set(p2) & D) and not (set(p1) & D) and not (set(p1) & set(p2))
    return D, p2, p1


def pick(eligible, k=8):
    """eligible: list of dicts with s, rate, order. Round-robin over suites."""
    by = {s: sorted([e for e in eligible if e["s"] == s], key=lambda e: (abs(e["rate"] - 0.08), e["order"]))
          for s in SUITES}
    out = []
    while len(out) < k and any(by.values()):
        for s in SUITES:
            if by[s] and len(out) < k:
                out.append(by[s].pop(0))
    return out


def eligible(hits, n=24):
    return n == 24 and 1 <= hits <= 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool-seed", type=int, default=1, choices=[1, 3, 4])
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    D, p2, p1 = partitions()
    if a.pool_seed == 1:
        pool = p2
    else:
        used = D | set(p2) | set(p1)
        for k in range(3, a.pool_seed):                 # earlier extra pools
            used |= set(pairs(seed=k, per_suite=10))
        pool = [p for p in pairs(seed=a.pool_seed, per_suite=10) if p not in used]
    print(f"design={len(D)} rq2_pool={len(p2)} rq1={len(p1)} using pool seed {a.pool_seed}: {len(pool)} cells")
    if a.dry:
        return
    os.makedirs(a.out, exist_ok=True)
    print("served_root", served_root(a.model), flush=True)
    memos = {d: Memo(os.path.join(a.out, f"memo_{a.model}_{d}.jsonl")) for d in (None, "both")}

    def camp(cell, defense, tag):
        """One Campaign object per (cell, defense); extending it keeps rng and memo keys distinct.
        Resume = move any old log aside and re-run; the memo returns cached executions."""
        s, ut, it = cell
        log = os.path.join(a.out, f"{a.model}_{defense}_{s}_{ut}_{it}_{tag}.jsonl")
        if os.path.exists(log):
            os.replace(log, log + ".old")
        return Campaign(a.model, s, ut, it, "random", budget=0, seed=0, memo=memos[defense],
                        log=log, threads=4, defense=defense)

    def hits(c, budget):
        c.budget = budget
        return sum(bool(r["security"]) for r in c.run()[:budget])

    def stage(cell):
        cA = camp(cell, None, "selA")
        hA = hits(cA, 6)
        if hA <= 1:
            res = dict(defense=None, hits=hits(cA, 24))  # same campaign extended to 24
        else:
            res = dict(defense="both", hits=hits(camp(cell, "both", "selB"), 24))
        res.update(s=cell[0], ut=cell[1], it=cell[2], hitsA=hA, n=24, rate=res["hits"] / 24)
        print(json.dumps(res), flush=True)
        return res

    with ThreadPoolExecutor(2) as ex:
        res = list(ex.map(stage, pool))
    for i, r in enumerate(res):
        r["order"] = i
    el = [r for r in res if eligible(r["hits"], r["n"])]
    sel = pick(el)
    json.dump(dict(model=a.model, root=served_root(a.model), pool_seed=a.pool_seed, all=res,
                   eligible=len(el), selected=sel, shortfall=8 - len(sel)),
              open(os.path.join(a.out, f"selected_{a.model}_pool{a.pool_seed}.json"), "w"), indent=1)
    print(f"eligible={len(el)} selected={len(sel)} shortfall={8 - len(sel)}", flush=True)


if __name__ == "__main__":
    main()
