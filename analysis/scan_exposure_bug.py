"""Reproducible scan behind the SEVERE-1 scope claim (plan §7): the exposure misdetection only
affects mutated payloads, so RQ1 censuses and the design cells need no rerun.

A misdetection here = a row the harness flagged exposed=False that is nonetheless a security
violation (security=True). A violating run proves the injection reached the agent, so exposed
must be True; exposed=False on such a row is a false negative of the harness exposure check.

python3 scan_exposure_bug.py        # prints per-dataset rows / violations / misdetections
Expected (reported to PM 2026-10-06, for af1 re-check):
  RQ1 censuses        0 misdetections / 592 violations   (default payload)
  design cells        0 / 75                               (default payload)
  RQ2 (superseded)    25 Llama + 34 Qwen                   (mutated payloads)
  selection           30                                   (mutated; selection ignores exposure)
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")


def scan(files, label):
    n = viol = bad = 0
    for f in files:
        if "ABORTED" in f or "superseded" in f and "rq2_superseded" not in f:
            continue
        for line in open(f):
            r = json.loads(line)
            if r.get("err") is not None or r.get("trace") is None:
                continue
            n += 1
            if r["security"]:
                viol += 1
                bad += not r.get("exposed")
    print(f"{label:28s} rows={n:5d} violations={viol:4d} misdetections(exposed=False&viol)={bad}")
    return bad


def main():
    scan(glob.glob(os.path.join(RUNS, "rq1", "census_*.jsonl")), "RQ1 censuses (default)")
    scan(glob.glob(os.path.join(HERE, "..", "testbed", "data", "small.jsonl")), "design cells (default)")
    scan(glob.glob(os.path.join(RUNS, "rq2_superseded", "*_[0-9].jsonl")), "RQ2 superseded (mutated)")
    scan(glob.glob(os.path.join(RUNS, "select", "*_sel[AB].jsonl")), "selection (mutated)")


if __name__ == "__main__":
    main()
