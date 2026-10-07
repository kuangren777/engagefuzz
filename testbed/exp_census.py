"""Census: repeated execution of fixed (user_task, injection_task) pairs with the default
important_instructions injection. Feeds several directions (trigger-probability distribution,
single-run misclassification, trace-abstraction stability, coverage/vuln correlation).

python3 exp_census.py <model> <reps> <temp> <out.jsonl> [threads]
Resumable: skips (pair, rep) already present in out.jsonl.
"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor

from harness import default_injection, load_log, run_pair, suite


def pairs(seed=0, per_suite=15):
    rng = random.Random(seed)
    out = []
    for s in ("banking", "slack", "travel"):
        S = suite(s)
        uts, its = sorted(S.user_tasks), sorted(S.injection_tasks)
        allp = [(s, u, i) for u in uts for i in its]
        out += rng.sample(allp, per_suite)
    return out


def main():
    model, reps, temp, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    th = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    done = {}
    for r in load_log(out):
        if r["err"] is None:
            k = (r["suite"], r["ut"], r["it"])
            done[k] = done.get(k, 0) + 1
    jobs = []
    for p in pairs():
        for _ in range(reps - done.get(p, 0)):
            jobs.append(p)
    print(f"{len(jobs)} jobs", file=sys.stderr)

    def go(p):
        s, u, i = p
        return run_pair(model, s, u, i, default_injection(s, i), temperature=temp, log=out, tag="census")

    with ThreadPoolExecutor(th) as ex:
        for n, r in enumerate(ex.map(go, jobs)):
            if n % 50 == 0:
                print(n, r["suite"], r["ut"], r["it"], r["security"], r["err"], file=sys.stderr)


if __name__ == "__main__":
    main()
