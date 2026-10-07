"""Closed-loop reward comparison driver (RQ2 / RQ4 / M0 smoke).

python3 run_e2.py --model qwen3-8b-local --targets banking:user_task_2:injection_task_5 \
    --arms random,binary,coverage,prox-v2 --seeds 0 --budget 20 --out ../runs/m0 [--defense both] \
    [--par 2 --threads 4]

Concurrency against the victim = par x threads (keep <= 8 per port). Resumable: a campaign whose
log already has `budget` rows is skipped; a partial log is moved aside and the campaign re-run
(memo hits make the re-run cheap). One memo file per (model, defense) in --out.
"""
import argparse
import hashlib
import sys
import itertools
import json
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from engagefuzz import Campaign, Memo, run_pair, served_root
import harness  # noqa: E402

HIGH_CAP_PORTS = {8031, 8032, 8034}  # dedicated backend ports allowed cap <= 16


class Limiter:
    """Global cap on in-flight victim executions; capacity can drop at runtime."""

    def __init__(self, cap):
        self.cap, self.used, self.cv = cap, 0, threading.Condition()

    def set(self, cap):
        with self.cv:
            self.cap = cap
            self.cv.notify_all()

    def wrap(self, fn):
        def run(*a, **kw):
            with self.cv:
                self.cv.wait_for(lambda: self.used < self.cap)
                self.used += 1
            try:
                return fn(*a, **kw)
            finally:
                with self.cv:
                    self.used -= 1
                    self.cv.notify_all()
        return run


def parse_kv(text):
    """KV-cache usage; vLLM <= 0.8 names it gpu_cache_usage_perc, newer versions kv_cache_usage_perc."""
    m = re.search(r"^vllm:(?:gpu|kv)_cache_usage_perc\{[^}]*\} (\S+)", text, re.M)
    if not m:
        raise ValueError("no KV-cache metric")
    return float(m.group(1))


def guard(lim, metrics_url, fallback, log):
    """PM rule 2026-10-06: cap 16 on 8021 only while waiting == 0 and KV cache < 0.7; else 8 for good."""
    while lim.cap > fallback:
        try:
            t = urllib.request.urlopen(metrics_url, timeout=10).read().decode()
            w = float(re.search(r"^vllm:num_requests_waiting\{[^}]*\} (\S+)", t, re.M).group(1))
            kv = parse_kv(t)
            if w > 0 or kv >= 0.7:
                lim.set(fallback)
                print(json.dumps(dict(event="concurrency_fallback", to=fallback, waiting=w, kv=kv,
                                      ts=time.time())), file=log, flush=True)
        except Exception as e:                       # fail safe: unreadable metrics -> fallback cap
            lim.set(fallback)
            print(f"guard metrics error {e!r}; cap -> {fallback}", file=log, flush=True)
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", help="comma list of suite:ut:it (all use --defense)")
    ap.add_argument("--targets-json", help="targets_<model>.json from merge_select: per-target defense")
    ap.add_argument("--arms", default="random,binary,coverage,prox-v21")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--defense", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--par", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cap", type=int, default=8, help="global in-flight cap (16 allowed on 8021 only)")
    ap.add_argument("--metrics", default=None, help="vLLM /metrics URL; required when --cap > 8 on 8021")
    ap.add_argument("--port", type=int, help="register --model on this local port (this process only)")
    ap.add_argument("--backend", default="gpu-a", help="free-form label recorded in each row (e.g. gpu-a, gpu-b)")
    ap.add_argument("--shard", default="0/1", help="i/n: keep a (seed,target) on this shard iff md5 %% n == i (all its arms together, CRN preserved)")
    ap.add_argument("--shard-salt", default="", help="salt prepended to the shard hash so incomplete cells can be re-split deterministically")
    a = ap.parse_args()
    # cap > 8 requires the metrics guard active (it falls back to 8 on waiting>0/high KV on any port)
    yf = a.backend == "gpu-b" and a.port in HIGH_CAP_PORTS
    assert a.cap <= 8 or (a.cap <= 16 and (yf or a.metrics)), \
        "cap>8 requires --metrics (guard) or a dedicated high-cap port"
    if a.port:
        harness.LOCAL_PORTS[a.model] = a.port
        harness.LOCAL_MODELS.add(a.model)
    root = served_root(a.model)
    if not root or root.startswith("ERR") or f"{a.model}=" not in root:
        sys.exit(f"served id mismatch: {root}")
    lim = Limiter(a.cap)
    print(json.dumps(dict(event="concurrency", cap=a.cap, ts=time.time())), flush=True)
    if a.cap > 8:
        threading.Thread(target=guard, args=(lim, a.metrics, 8, sys.stdout), daemon=True).start()
    os.makedirs(a.out, exist_ok=True)
    print("served_root", root, "backend", a.backend, flush=True)
    port = a.port or (8021 if a.backend == "gpu-a" else None)
    memo = Memo(os.path.join(a.out, f"memo_{a.model}.jsonl"))   # key already carries the defense
    if a.targets_json:
        T = json.load(open(a.targets_json))
        assert T["model"] == a.model, (T["model"], a.model)
        targets = [((t["s"], t["ut"], t["it"]), t["defense"]) for t in T["selected"]]
    else:
        targets = [(tuple(t.split(":")), a.defense) for t in a.targets.split(",")]
    jobs = []
    # seed-major order: every target's seed 0 finishes first (PM early-stop gate reads seed 0)
    for seed, ((s, ut, it), dfn), arm in itertools.product([int(x) for x in a.seeds.split(",")], targets,
                                                           a.arms.split(",")):
        log = os.path.join(a.out, f"{a.model}_{dfn}_{s}_{ut}_{it}_{arm}_{seed}.jsonl")
        n = sum(1 for _ in open(log)) if os.path.exists(log) else 0
        if n >= a.budget:
            continue
        si, sn = map(int, a.shard.split("/"))
        if sn > 1 and int(hashlib.md5((a.shard_salt + repr((seed, s, ut, it))).encode()).hexdigest(), 16) % sn != si:
            continue                                  # (seed,target) belongs to another shard
        if n:
            os.replace(log, log + f".partial{int(time.time())}")
        jobs.append(dict(s=s, ut=ut, it=it, seed=seed, arm=arm, log=log, defense=dfn))
    print(f"{len(jobs)} campaigns to run", flush=True)

    def go(j):
        t0 = time.time()
        c = Campaign(a.model, j["s"], j["ut"], j["it"], j["arm"], budget=a.budget, seed=j["seed"],
                     memo=memo, log=j["log"], threads=a.threads, defense=j["defense"],
                     run_fn=lim.wrap(run_pair), backend=a.backend)
        c.port = port
        R = c.run()
        hits = sum(r["memo_hit"] for r in R)
        err = sum(r.get("err") is not None for r in R)
        print(json.dumps(dict(arm=j["arm"], seed=j["seed"], target=f"{j['s']}:{j['ut']}:{j['it']}",
                              n=len(R), viol=sum(bool(r["security"]) for r in R), memo_hits=hits, err=err,
                              levels=[r["level_v21"] for r in R], wall_s=round(time.time() - t0))), flush=True)

    with ThreadPoolExecutor(a.par) as ex:
        list(ex.map(go, jobs))


if __name__ == "__main__":
    main()
