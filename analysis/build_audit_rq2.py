"""Build a result-audit pack for one model's RQ2 run (raw records -> audit.py input).

python3 build_audit_rq2.py ../runs/rq2 llama31-8b-local ../runs/rq2/audit_llama31-8b-local

summary.json   : rq2_stats.py --json output for this model (the numbers that will be reported)
samples/*.json : campaign records (kind=campaign, per-arm fields viol__<arm>, hit__<arm>) and
                 execution records (kind=exec) for polarity anchors
provenance.json: live executions (memo misses without error), served roots, driver log path
audit_config.json: recompute per-arm viol_mean / hit_frac / n; polarity anchors on the oracle
"""
import json
import os
import subprocess
import sys

from rq2_stats import BASE, MAIN, load

ARMS = [MAIN] + BASE


def main(runs, model, out):
    os.makedirs(os.path.join(out, "samples"), exist_ok=True)
    C = [c for c in load(runs) if c["model"] == model]
    camp, exe = [], []
    for c in C:
        r = dict(kind="campaign", arm=c["arm"], seed=c["seed"], target=f"{c['s']}:{c['ut']}:{c['it']}",
                 n=c["n"], err=c["err"], viol=c["viol"])
        r[f"viol__{c['arm']}"] = c["viol"]
        r[f"hit__{c['arm']}"] = c["viol"] > 0
        r[f"n__{c['arm']}"] = 1
        camp.append(r)
        for x in c["rows"]:
            exe.append(dict(kind="exec", arm=c["arm"], security=bool(x["security"]), exposed=bool(x.get("exposed")),
                            level_v21=x.get("level_v21"), err=x.get("err") is not None,
                            memo_hit=bool(x.get("memo_hit")), served_root=x.get("served_root")))
    json.dump(camp + exe, open(os.path.join(out, "samples", "records.json"), "w"))

    tmp = os.path.join(out, "_stats.json")
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "rq2_stats.py"), runs,
                    "--json", tmp, "--quiet"], check=True)
    S = json.load(open(tmp))[model]
    json.dump(S, open(os.path.join(out, "summary.json"), "w"), indent=1)

    metrics = {}
    for a in ARMS:
        metrics[f"viol_mean__{a}"] = dict(field=f"viol__{a}", agg="mean", claimed=f"arms.{a}.viol_mean", tol=1e-9)
        metrics[f"hit_frac__{a}"] = dict(field=f"hit__{a}", agg="frac_true", claimed=f"arms.{a}.hit_frac", tol=1e-9)
        metrics[f"n__{a}"] = dict(field=f"n__{a}", agg="sum", claimed=f"arms.{a}.n")
    cfg = dict(
        metrics=metrics,
        polarity=[
            dict(name="oracle_violation_is_level4", sample_filter={"kind": "exec", "security": True, "err": False},
                 field="level_v21", expect=4, min_fraction=1.0),
            dict(name="not_exposed_nonviolating_is_level0",
                 sample_filter={"kind": "exec", "security": False, "exposed": False, "err": False},
                 field="level_v21", expect=0, min_fraction=1.0),
        ],
        provenance=dict(request_count_field="request_count", served_id_field="served_ids",
                        relay_log="driver.log", min_requests=1000))
    json.dump(cfg, open(os.path.join(out, "audit_config.json"), "w"), indent=1)

    live = [e for e in exe if not e["memo_hit"] and not e["err"]]
    json.dump(dict(request_count=len(live), served_ids=sorted({e["served_root"] for e in exe if e["served_root"]}),
                   campaigns=len(C), exec_rows=len(exe), memo_hits=sum(e["memo_hit"] for e in exe)),
              open(os.path.join(out, "provenance.json"), "w"), indent=1)
    src = os.path.join(runs, f"driver_{model}.log")
    dst = os.path.join(out, "driver.log")
    if not os.path.exists(dst):
        os.symlink(os.path.relpath(src, out), dst)
    print(f"audit pack: {out}  campaigns={len(C)} exec={len(exe)} live={len(live)}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
