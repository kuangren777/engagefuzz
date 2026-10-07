"""RQ3 signal density from target-selection runs (random arm, plan.md §6 item 6).

python3 select_exposure.py ../runs/select
Per model x suite x defense: rows, exposure rate, mean tool calls, v2.1 level distribution, ASR.
"""
import collections
import glob
import json
import os
import re
import sys

PAT = re.compile(r"(?P<model>.+?)_(?P<defense>None|both)_(?P<s>banking|slack|travel)_.*_sel[AB]\.jsonl$")


def main(d):
    G = collections.defaultdict(list)
    for f in glob.glob(os.path.join(d, "*_sel[AB].jsonl")):
        m = PAT.match(os.path.basename(f))
        if m:
            for line in open(f):
                G[(m["model"], m["s"], m["defense"])].append(json.loads(line))
    print("model,suite,defense,rows,exposed,mean_calls,asr,level_dist_v21")
    for k in sorted(G):
        R = [r for r in G[k] if r.get("err") is None]
        n = len(R)
        lv = collections.Counter(r.get("level_v21") for r in R)
        print(f"{k[0]},{k[1]},{k[2]},{n},{sum(bool(r['exposed']) for r in R) / n:.3f},"
              f"{sum(len(r['trace'] or []) for r in R) / n:.2f},{sum(bool(r['security']) for r in R) / n:.3f},"
              f"\"{dict(sorted(lv.items()))}\"")


if __name__ == "__main__":
    main(sys.argv[1])
