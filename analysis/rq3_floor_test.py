"""RQ3 density against its collision floor (post-hoc, cold read 2026-10-10).
For every failed AgentDojo frontier run: own = 1 if the run is at level three (v2.1 ladder, as in
make_rq3_numbers.py); floor = share of the suite's other injection tasks whose attacker-unique values
fire the level-three rule on this trace (px1a_collision.py rule: values of the run's own task excluded).
d = own - floor. Paired test of mean(d) > 0 by sign-flip permutation and a run-level bootstrap CI,
pooled and per model.
python3 rq3_floor_test.py -> runs/px/rq3_floor.json
python3 rq3_floor_test.py --exclude gpt-5.6-sol -> runs/px/rq3_floor_nosol.json (gateway sensitivity)
"""
import collections, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prox_v2 as P
from harness import suite as get_suite

FRONTIER = os.path.join(HERE, "..", "testbed", "data", "frontier.jsonl")
OUT = os.path.join(HERE, "..", "runs", "px", "rq3_floor.json")
B, SEED = 10000, 20261010


def floor(r, itlist):
    own = {v for _, _, v in P.atk_values(r["suite"], r["ut"], r["it"])}
    pairs = fires = 0
    for it2 in itlist[r["suite"]]:
        if it2 == r["it"]:
            continue
        A = [(f, a, v) for f, a, v in P.atk_values(r["suite"], r["ut"], it2) if v not in own]
        afn = {f for f, _, _ in A}
        pairs += 1
        for t in r["trace"]:
            got = " ".join(str(v).lower() for v in (t.get("a") or {}).values())
            if any(v in got for _, _, v in A) and (t["f"].startswith(P.SIDE_PREFIX) or t["f"] in afn):
                fires += 1
                break
    return fires / max(1, pairs)


def test(d, rng):
    n = len(d); m = sum(d) / n
    boot = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    perm = sum(1 for _ in range(B) if sum(x if rng.random() < 0.5 else -x for x in d) / n >= m)
    return dict(n=n, mean_diff=m, ci95=[boot[int(0.025 * B)], boot[int(0.975 * B) - 1]], p_one_sided=(perm + 1) / (B + 1))


def ctest(groups, rng):
    """Cluster version: groups = list of lists of d. Bootstrap resamples clusters; the sign-flip flips
    whole clusters (cluster sums), so runs of one cluster are never treated as independent."""
    k = len(groups); sums = [sum(g) for g in groups]; ns = [len(g) for g in groups]; N = sum(ns)
    m = sum(sums) / N
    boot = []
    for _ in range(B):
        idx = [rng.randrange(k) for _ in range(k)]
        boot.append(sum(sums[i] for i in idx) / max(1, sum(ns[i] for i in idx)))
    boot.sort()
    perm = sum(1 for _ in range(B) if sum(s if rng.random() < 0.5 else -s for s in sums) / N >= m)
    return dict(clusters=k, mean_diff=m, ci95=[boot[int(0.025 * B)], boot[int(0.975 * B) - 1]], p_one_sided=(perm + 1) / (B + 1))


def main(exclude=(), out_path=OUT):
    R = [json.loads(l) for l in open(FRONTIER)]
    R = [r for r in R if r.get("err") is None and r.get("trace") is not None and not r["security"]
         and r.get("model") not in exclude]
    itlist = {s: sorted(get_suite(s).injection_tasks) for s in ("banking", "slack", "travel")}
    rng = random.Random(SEED)
    rows = []
    for r in R:
        own = int(P.feats(r)["level21"] == 3)
        fl = floor(r, itlist)
        rows.append((r["model"], own, fl, (r["suite"], r["ut"], r["it"])))
    out = dict(pooled=dict(own_rate=sum(x[1] for x in rows) / len(rows), floor_rate=sum(x[2] for x in rows) / len(rows),
                           **test([x[1] - x[2] for x in rows], rng)), per_model={})
    for m in sorted({x[0] for x in rows}):
        sub = [(x[1], x[2]) for x in rows if x[0] == m]
        out["per_model"][m] = dict(own_rate=sum(o for o, _ in sub) / len(sub), floor_rate=sum(f for _, f in sub) / len(sub),
                                   **test([o - f for o, f in sub], rng))
    import collections as _c
    tg = _c.defaultdict(list); mg = _c.defaultdict(list)
    for x in rows:
        tg[(x[0], x[3])].append(x[1] - x[2]); mg[x[0]].append(x[1] - x[2])
    out["pooled_cluster_target"] = ctest(list(tg.values()), rng)
    # cluster-bootstrap intervals of the two densities themselves (same 90 model x target-pair clusters)
    og = _c.defaultdict(list); fg = _c.defaultdict(list)
    for x in rows:
        og[x[3]].append(x[1]); fg[x[3]].append(x[2])   # 15 target-pair clusters shared across models
    keys = list(og)
    bo, bf = [], []
    for _ in range(B):
        idx = [keys[rng.randrange(len(keys))] for _ in keys]
        n = sum(len(og[k]) for k in idx)
        bo.append(sum(sum(og[k]) for k in idx) / n); bf.append(sum(sum(fg[k]) for k in idx) / n)
    bo.sort(); bf.sort()
    out["density_ci_target_cluster"] = dict(own=[bo[int(0.025 * B)], bo[int(0.975 * B) - 1]],
                                            floor=[bf[int(0.025 * B)], bf[int(0.975 * B) - 1]])
    print("density CI own", [round(100 * v, 2) for v in out["density_ci_target_cluster"]["own"]],
          "floor", [round(100 * v, 2) for v in out["density_ci_target_cluster"]["floor"]])
    out["pooled_cluster_model"] = ctest(list(mg.values()), rng)
    pg = _c.defaultdict(list)   # the 15 target pairs shared by all six models
    for x in rows:
        pg[x[3]].append(x[1] - x[2])
    out["pooled_cluster_pair"] = ctest(list(pg.values()), rng)
    print("pooled_cluster_pair:", out["pooled_cluster_pair"])
    for m in out["per_model"]:
        out["per_model"][m]["cluster_target"] = ctest([v for (mm, _), v in tg.items() if mm == m], rng)
    json.dump(out, open(out_path, "w"), indent=1)
    for k in ("pooled_cluster_target", "pooled_cluster_model"):
        v = out[k]; print(f"{k}: clusters={v['clusters']} diff={100*v['mean_diff']:.2f}pp CI[{100*v['ci95'][0]:.2f},{100*v['ci95'][1]:.2f}] p={v['p_one_sided']:.4f}")
    for m, v in out["per_model"].items():
        c = v["cluster_target"]; print(f"  {m:22s} target-cluster p={c['p_one_sided']:.4f} CI[{100*c['ci95'][0]:.2f},{100*c['ci95'][1]:.2f}]")
    p = out["pooled"]
    print(f"pooled n={p['n']} own={100*p['own_rate']:.2f}% floor={100*p['floor_rate']:.2f}% diff={100*p['mean_diff']:.2f}pp "
          f"CI[{100*p['ci95'][0]:.2f},{100*p['ci95'][1]:.2f}] p={p['p_one_sided']:.4f}")
    for m, v in out["per_model"].items():
        print(f"  {m:22s} n={v['n']:4d} own={100*v['own_rate']:.2f}% floor={100*v['floor_rate']:.2f}% "
              f"diff={100*v['mean_diff']:.2f}pp CI[{100*v['ci95'][0]:.2f},{100*v['ci95'][1]:.2f}] p={v['p_one_sided']:.4f}")


if __name__ == "__main__":
    if "--exclude" in sys.argv:
        m = sys.argv[sys.argv.index("--exclude") + 1]
        main(exclude={m}, out_path=os.path.join(HERE, "..", "runs", "px", "rq3_floor_nosol.json"))
    else:
        main()
