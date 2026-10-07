"""RQ1 census on the pre-registered RQ1 cells (plan.md §2.2): default payload, T=1.0,
10 reps undefended + 5 reps under `both`.

python3 run_census.py --model qwen3-32b-local --port 8031 --out ../runs/rq1 [--cap 8|16] [--single-call]

--port registers the model in this process only (shared harness file untouched); the name must be
the vLLM served id. --single-call routes through harness _SingleCallClient (Llama/Mistral).
Resumable: (cell, defense, rep) keys already in the output are skipped. Rows carry served_root.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from engagefuzz import prox_v2, run_flags, served_root  # sets sys.path for testbed
import harness  # noqa: E402
from run_select import partitions  # noqa: E402

REPS = {None: 10, "both": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes jobs with md5(key) %% n == i")
    ap.add_argument("--metrics", default=None, help="vLLM /metrics URL; required when --cap > 8")
    ap.add_argument("--single-call", action="store_true")
    ap.add_argument("--sniff", type=int, default=20, help="probe runs before the full census (0 = off)")
    ap.add_argument("--backend", default="gpu-a", help="free-form label recorded in each row (e.g. gpu-a, gpu-b)")
    a = ap.parse_args()
    assert a.cap <= 8 or (a.cap <= 16 and a.metrics), "cap > 8 needs --metrics (falls back to 8 on waiting>0)"
    from run_e2 import Limiter, guard
    lim = Limiter(a.cap)
    if a.cap > 8:
        threading.Thread(target=guard, args=(lim, a.metrics, 8, sys.stdout), daemon=True).start()
    run_pair = lim.wrap(harness.run_pair)
    if a.port:
        harness.LOCAL_PORTS[a.model] = a.port
        harness.LOCAL_MODELS.add(a.model)
        if a.single_call:
            harness.SINGLE_CALL_MODELS.add(a.model)
    root = served_root(a.model)
    try:
        import urllib.request
        vllm = json.load(urllib.request.urlopen(f"http://127.0.0.1:{harness.LOCAL_PORTS.get(a.model, 8021)}/version",
                                                timeout=10)).get("version")
    except Exception as e:
        vllm = f"ERR {e!r}"
    print("served_root", root, "vllm", vllm, "backend", a.backend, flush=True)
    if not root or root.startswith("ERR") or f"{a.model}=" not in root:
        sys.exit(f"served id mismatch: {root}")
    _, _, cells = partitions()
    os.makedirs(a.out, exist_ok=True)
    si, sn = map(int, a.shard.split("/"))
    # one file per port (replicas audited separately); resume reads every non-aborted file of this model
    out = os.path.join(a.out, f"census_{a.model}_p{a.port}.jsonl" if sn > 1 else f"census_{a.model}.jsonl")
    done = set()
    for f in glob.glob(os.path.join(a.out, f"census_{a.model}*.jsonl")):
        if "ABORTED" in f:
            continue
        for line in open(f):
            r = json.loads(line)
            if r.get("err") is None:                 # errored runs are retried on resume
                done.add((r["suite"], r["ut"], r["it"], r["defense"], r["rep"]))
    jobs = [(c, d, k) for c in cells for d, n in REPS.items() for k in range(n)
            if (*c, d, k) not in done
            and int(hashlib.md5(repr((*c, d, k)).encode()).hexdigest(), 16) % sn == si]
    print(f"{len(cells)} cells, {len(jobs)} runs to do", flush=True)
    lk = threading.Lock()

    def go(j):
        (s, ut, it), d, k = j
        r = run_pair(a.model, s, ut, it, harness.default_injection(s, it), defense=d)
        r.update(**run_flags(a.model, r), port=a.port)
        r.update(rep=k, served_root=root, backend=a.backend, vllm=vllm,
                 # model-format failure: tool call written into text, nothing parsed (PM: report separately)
                 fmt_fail=bool(r.get("err") is None and not r.get("trace") and "<tool_call>" in (r.get("final") or "")))
        if r.get("err") is None and r.get("trace") is not None:
            f = prox_v2.feats(r)
            r.update(level_v21=f["level21"], level_v2=f["level2"])
        with lk, open(out, "a") as fh:
            fh.write(json.dumps(r, default=str) + "\n")

    if a.sniff and jobs:
        # tool-use sniff: a model that never calls tools gives level 0 everywhere and no RQ1 signal
        rng = __import__("random").Random(0)
        probe = rng.sample(jobs, min(a.sniff, len(jobs)))
        with ThreadPoolExecutor(a.cap) as ex:
            list(ex.map(go, probe))
        keys = {(*c, d, k) for c, d, k in probe}
        rows = [r for r in map(json.loads, open(out))
                if (r["suite"], r["ut"], r["it"], r["defense"], r["rep"]) in keys]
        ok_keys = {(r["suite"], r["ut"], r["it"], r["defense"], r["rep"]) for r in rows if r.get("err") is None}
        rows = [r for r in rows if r.get("err") is None]          # denominator: connected runs only
        rate = sum(bool(r.get("trace")) for r in rows) / max(1, len(rows))
        ff = sum(bool(r.get("fmt_fail")) for r in rows) / max(1, len(rows))
        print(f"sniff: {len(rows)}/{len(probe)} connected, tool-call rate {rate:.2f}, fmt_fail rate {ff:.2f}", flush=True)
        if rate < 0.5:
            sys.exit(f"ABORT: tool-call rate {rate:.2f} < 0.5 on sniff; rename the output before any rerun")
        jobs = [j for j in jobs if (*j[0], j[1], j[2]) not in ok_keys]   # errored probes are retried
    with ThreadPoolExecutor(a.cap) as ex:
        list(ex.map(go, jobs))
    print("done", flush=True)


if __name__ == "__main__":
    main()
