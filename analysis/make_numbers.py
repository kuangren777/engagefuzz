"""RQ1 and RQ4 headline macros from the frozen statistics files.

python3 make_numbers.py -> out/numbers.tex, out/rq4_numbers.tex

Release-only regenerator. In the working project these two macro files were written by hand from
runs/rq1/rq1_stats.json (rq1_stats.py output) and runs/rq4/*.json. This script reproduces them
byte for byte from the same files, so every number in them traces to shipped data.
"""
import json
import os

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
# (ds_base + ds_enhanced = 1088, 26 excluded). Needs the InjecAgent data to recount, see NOTES_errata.md.
INJEC_POOL = 1062


def m(name, val):
    return "\\newcommand{\\%s}{%s}" % (name, val)


def rq1():
    d = json.load(open(os.path.join(RUNS, "rq1", "rq1_stats.json")))
    L = ["% auto-generated from runs/rq1/rq1_stats.json; do not hand-edit"]
    for key, stem, disp in RQ1_MODELS:
        s = d[key]
        r = s["rho"][PRIMARY]
        L += [m(f"rqone{stem}Rho", f"{r['rho']:.3f}"),
              m(f"rqone{stem}P", f"{r['p']:.3g}"),
              m(f"rqone{stem}Holm", f"{s['holm_p']:.3g}"),
              m(f"rqone{stem}Tau", f"{s['kendall_tau_b']:.3f}"),
              m(f"rqone{stem}ASR", f"{s['asr']:.3f}"),
              m(f"rqone{stem}N", str(r["n"])),
              m(f"rqone{stem}Disp", disp)]
    re_, pc, c2 = d["random_effects"], d["pooled_model_centred"], d["C2"]
    L += [m("rqonePooledRho", f"{re_['rho']:.3f}"),
          m("rqonePooledLo", f"{re_['ci95'][0]:.3f}"),
          m("rqonePooledHi", f"{re_['ci95'][1]:.3f}"),
          m("rqoneCentredRho", f"{pc['rho']:.3f}"),
          m("rqoneCentredP", f"{pc['p']:.3g}"),
          m("rqonePass", str(len(c2["passed"]))),
          m("rqoneNeed", str(c2["need"])),
          "",
          "% RQ3 frontier signal density, from analysis/v2_frontier.csv (level 3 identical in v2/v2.1)"]
    open(os.path.join(OUT, "numbers.tex"), "w").write("\n".join(L) + "\n")


def rq4():
    v = json.load(open(os.path.join(RUNS, "rq4", "injecagent_validity.json")))["llama31-8b-local"]
    fr = json.load(open(os.path.join(RUNS, "rq4", "injecagent_frontier_density.json")))
    r = v["rho"][PRIMARY]
    L = ["% RQ4 macros",
         m("rqfourLlamaRho", f"{r['rho']:.3f}"),
         m("rqfourLlamaP", f"{r['p']:.2g}"),
         m("rqfourLlamaN", str(r["n"])),
         m("rqfourLlamaTau", f"{v['kendall_tau_b']:.3f}"),
         m("rqfourLlamaASR", f"{v['asr']:.3f}"),
         m("rqfourLlamaRhoSide", f"{v['rho_sideeffect']:.3f}"),
         m("rqfourLlamaRhoDev", f"{v['rho_anytool']:.3f}"),
         m("rqfourFrontierModels", str(len(fr))),
         m("rqfourFrontierASRmax", f"{max(x['asr'] for x in fr.values()):.3f}"),
         m("rqfourFrontierLtwoMax", f"{max(x['l2plus_nonviol'] for x in fr.values()):.3f}"),
         m("rqfourInjecPool", str(INJEC_POOL))]
    open(os.path.join(OUT, "rq4_numbers.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    rq1()
    rq4()
    print("wrote numbers.tex, rq4_numbers.tex")
