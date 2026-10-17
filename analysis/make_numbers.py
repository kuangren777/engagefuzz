"""RQ1 and RQ4 headline macros from the frozen statistics files.

python3 make_numbers.py -> out/{numbers,rq4_numbers}.tex

These two macro files were first written by hand from
runs/rq1/rq1_stats.json (rq1_stats.py output) and runs/rq4/*.json. This script reproduces them
byte for byte from the same files, so every number in them traces to shipped data.
"""
import json
import os
from efmt import F, exact, mean  # noqa: E402,F401  one half-up formatter

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")
OUT = os.path.join(HERE, "..", "out")

# file label -> (macro stem, display name), paper order
RQ1_MODELS = [("qwen3-8b-yf", "QwenEightB", "Qwen3-8B"),
              ("llama31-8b-local", "Llama", "Llama-3.1-8B"),
              ("qwen3-32b-local", "QwenThirtyTwoB", "Qwen3-32B"),
              ("granite31-8b-local", "Granite", "Granite-3.1-8B")]
PRIMARY = "v21_minfail1"  # pre-registered fail-only rho, proximity ladder v2.1
# InjecAgent data-stealing pool after excluding cases without an attacker-unique value
# (ds_base + ds_enhanced = 1088, 26 excluded), the pool injecagent/injecagent_run.py samples from.
# Recounted from the InjecAgent data when present; 1062 is that count (rechecked 2026-10-14).
INJEC_POOL = 1062
INJEC_SRC = os.environ.get("INJECAGENT_SRC", "")


def injec_pool():
    if not os.path.isdir(os.path.join(INJEC_SRC, "data")):
        return INJEC_POOL
    import sys
    sys.path.insert(0, os.path.join(HERE, "..", "injecagent"))
    import injecagent_prox as prox
    cases = [c for fn in ("test_cases_ds_base.json", "test_cases_ds_enhanced.json")
             for c in json.load(open(os.path.join(INJEC_SRC, "data", fn)))]
    n = sum(1 for c in cases if prox.attacker_values(c))
    assert n == INJEC_POOL, f"InjecAgent pool recount {n} != {INJEC_POOL}"
    return n


def m(name, val):
    return "\\newcommand{\\%s}{%s}" % (name, val)


def rq1():
    d = json.load(open(os.path.join(RUNS, "rq1", "rq1_stats.json")))
    L = ["% auto-generated from runs/rq1/rq1_stats.json; do not hand-edit"]
    for key, stem, disp in RQ1_MODELS:
        s = d[key]
        r = s["rho"][PRIMARY]
        L += [m(f"rqone{stem}Rho", f"{F(r['rho'], '.3f')}"),
              m(f"rqone{stem}Holm", f"{F(s['holm_p'], '.3g')}"),
              m(f"rqone{stem}Tau", f"{F(s['kendall_tau_b'], '.3f')}"),
              m(f"rqone{stem}ASR", f"{F(exact(s['asr'], s['rows']), '.3f')}"),
              m(f"rqone{stem}N", str(r["n"])),
              m(f"rqone{stem}Disp", disp)]
    re_, pc, c2 = d["random_effects"], d["pooled_model_centred"], d["C2"]
    L += [m("rqonePooledRho", f"{F(re_['rho'], '.3f')}"),
          m("rqonePooledLo", f"{F(re_['ci95'][0], '.3f')}"),
          m("rqonePooledHi", f"{F(re_['ci95'][1], '.3f')}"),
          m("rqoneCentredRho", f"{F(pc['rho'], '.3f')}"),
          m("rqoneCentredP", f"{F(pc['p'], '.3g')}"),
          m("rqonePass", str(len(c2["passed"]))),
          "",
          "% RQ3 frontier signal density, from analysis/v2_frontier.csv (level 3 identical in v2/v2.1)"]
    open(os.path.join(OUT, "numbers.tex"), "w").write("\n".join(L) + "\n")


def rq4():
    v = json.load(open(os.path.join(RUNS, "rq4", "injecagent_validity.json")))["llama31-8b-local"]
    fr = json.load(open(os.path.join(RUNS, "rq4", "injecagent_frontier_density.json")))
    r = v["rho"][PRIMARY]
    L = ["% RQ4 macros",
         m("rqfourLlamaRho", f"{F(r['rho'], '.3f')}"),
         m("rqfourLlamaP", f"{F(r['p'], '.2g')}"),
         m("rqfourLlamaN", str(r["n"])),
         m("rqfourLlamaTau", f"{F(v['kendall_tau_b'], '.3f')}"),
         m("rqfourLlamaASR", f"{F(exact(v['asr'], 10 ** 4), '.3f')}"),
         m("rqfourLlamaRhoSide", f"{F(v['rho_sideeffect'], '.3f')}"),
         m("rqfourLlamaRhoDev", f"{F(v['rho_anytool'], '.3f')}"),
         m("rqfourFrontierModels", str(len(fr))),
         m("rqfourFrontierASRmax", f"{F(100 * max(exact(x['asr'], x['n']) for x in fr.values()), '.1f')}"),
         m("rqfourFrontierLtwoMax", f"{F(100 * max(exact(x['l2plus_nonviol'], x['n']) for x in fr.values()), '.1f')}"),
         m("rqfourInjecPool", str(injec_pool()))]
    open(os.path.join(OUT, "rq4_numbers.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    rq1()
    rq4()
    print("wrote numbers.tex, rq4_numbers.tex")
