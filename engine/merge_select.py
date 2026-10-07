"""Merge per-pool selection results into the RQ2 target list (plan.md §2.3 + §6 clarification).

python3 merge_select.py --model llama31-8b-local --dir ../runs/select

Union of eligible cells over pools 1, 3, 4 (in that order); same round-robin pick as run_select.pick,
earlier pools win ties. Prints the decision: OK (8), NEXT_POOL (<8 and a pool is left),
RUN_REDUCED (6-7 after pool 4), STOP (<6 after pool 4). Writes targets_<model>.json on OK/RUN_REDUCED.
"""
import argparse
import json
import os

from run_select import eligible, pick

POOLS = [1, 3, 4]


def decide(results_by_pool):
    """results_by_pool: {pool_seed: [cell result dicts]} for the pools that have finished."""
    el = []
    for rank, ps in enumerate(POOLS):
        for r in results_by_pool.get(ps, []):
            if eligible(r["hits"], r["n"]):
                el.append(dict(r, pool=ps, order=rank * 1000 + r["order"]))
    sel = pick(el)
    done = [p for p in POOLS if p in results_by_pool]
    left = [p for p in POOLS if p not in results_by_pool]
    if len(sel) == 8:
        verdict = "OK"
    elif left:
        verdict = f"NEXT_POOL {left[0]}"
    elif len(sel) >= 6:
        verdict = "RUN_REDUCED"
    else:
        verdict = "STOP"
    return dict(pools_done=done, eligible=len(el), selected=sel, verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    res = {}
    for ps in POOLS:
        f = os.path.join(a.dir, f"selected_{a.model}_pool{ps}.json")
        if os.path.exists(f):
            res[ps] = json.load(open(f))["all"]
    d = decide(res)
    print(json.dumps({k: v for k, v in d.items() if k != "selected"}))
    for s in d["selected"]:
        print(f"  pool{s['pool']} {s['s']}:{s['ut']}:{s['it']} defense={s['defense']} hits={s['hits']}/24")
    if d["verdict"] in ("OK", "RUN_REDUCED"):
        json.dump(dict(model=a.model, **d), open(os.path.join(a.dir, f"targets_{a.model}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
