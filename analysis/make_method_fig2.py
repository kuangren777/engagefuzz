"""Method figure v3 (redraw). Topology: measure path on top, pipeline mid,
ladder lower-right, reward path down the right margin, return along the left margin.
Canvas 8.5cm wide, all fonts >= 7pt at final size. No text-arrow crossings.
python3 make_method_fig2.py [out.pdf]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

W, H = 3.35, 3.05            # 8.5cm x 7.7cm
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, 100)
ax.set_ylim(0, 97)
ax.axis("off")
FS = 7.0
EC, LWC = "#222222", "#444444"
BLUE, RED = "#1a3f6f", "#7a1f1f"


def box(x, y, w, h, text, fs=FS, fc="#f2f2f2", dashed=False, color=EC):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.4",
                                lw=0.8, ec=EC, fc=fc, linestyle="--" if dashed else "-",
                                mutation_aspect=0.55))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            linespacing=1.3, color=color)
    return dict(cx=x + w / 2, cy=y + h / 2, x=x, y=y, w=w, h=h,
                l=(x, y + h / 2), r=(x + w, y + h / 2), t=(x + w / 2, y + h), b=(x + w / 2, y))


def arrow(p1, p2, label=None, fs=7.0, dx=0, dy=0, conn="arc3,rad=0", dashed=False,
          style="-|>", lcolor="#222222"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=8, lw=0.8,
                                 color=LWC, connectionstyle=conn,
                                 linestyle=(0, (3, 2)) if dashed else "-"))
    if label:
        ax.text((p1[0] + p2[0]) / 2 + dx, (p1[1] + p2[1]) / 2 + dy, label, ha="center",
                va="center", fontsize=fs, color=lcolor)


# ---- pipeline (middle band) ------------------------------------------------
pool = box(3, 66, 11, 11, "pool\n$\\mathcal{P}$")
par = box(20, 66, 11, 11, "par.\n$p$")
exe = box(36, 66, 15, 11, "execute\nAgentDojo")
tra = box(56, 66, 6, 11, "$\\tau$", fc="white")
sco = box(67, 66, 15, 11, "$s(\\tau)$", fc="white")
arrow(pool["r"], par["l"], "UCB1", dy=3.0)
arrow(par["r"], exe["l"], "mutate\n$op{\\in}O$\n$|O|{=}8$", dy=6.0)
arrow(exe["r"], tra["l"])
arrow(tra["r"], sco["l"])

# memo cache under execute
memo = box(36, 50, 20, 10, "memo\n$K{=}(tgt,p,op,k,seed)$", dashed=True)
arrow((exe["cx"] - 4, exe["y"] + 0.5), (memo["cx"] - 4, memo["t"][1] - 0.5),
      dashed=True, style="<|-|>")

# ---- measure path (top) ----------------------------------------------------
mea = box(20, 86, 20, 9, "per-unit $\\bar{s}_u$", fc="#e8eef7", color=BLUE)
rho = box(46, 85, 34, 10, "$\\rho(\\bar{s}_u,\\ \\mathrm{trig})$\nRQ1 / RQ4",
          fc="#e8eef7", color=BLUE)
arrow(mea["r"], rho["l"])
arrow((sco["cx"], sco["t"][1] + 0.5), (rho["cx"] + 8, rho["b"][1] - 0.5),
      conn="arc3,rad=-0.15")
ax.text(sco["cx"] + 2.5, 82.5, "measure", fontsize=7.0, ha="left", color=BLUE)

# ---- ladder (lower right, fed from s) --------------------------------------
lad = box(52, 32, 44, 26, "", fc="white")
ax.text(lad["cx"], lad["t"][1] - 2.6, "first match, top $\\rightarrow$ bottom",
        fontsize=7.0, ha="center")
rows = ["L4  oracle fires",
        "L3  side-effect $\\wedge\\ \\mathrm{uniq}(v)$",
        "L2  side-effect $\\wedge\\ f{\\notin}F_{user}$",
        "L1  seen, no L2/L3 call",
        "L0  unseen"]
for i, r in enumerate(rows):
    ax.text(lad["x"] + 2.5, lad["t"][1] - 6.6 - i * 3.5, r, fontsize=7.0, ha="left",
            va="center")
arrow((sco["cx"] + 3, sco["b"][1] - 0.5), (lad["cx"] + 10, lad["t"][1] + 0.5),
      conn="arc3,rad=0.12")

# ---- reward path: right margin down, bottom band, left margin back ---------
rew = box(76, 18, 12, 9, "$r{=}s/4$", fc="#f7e8e8", color=RED)
arrow((sco["r"][0] + 0.5, sco["cy"] - 2), (99, 60), conn="arc3,rad=0.0", style="-")
arrow((99, 60), (rew["cx"] + 2, rew["t"][1] + 0.5), conn="arc3,rad=0.0")
ax.text(95.0, 61.2, "reward", fontsize=7.0, ha="right", color=RED)
ax.text(95.3, 34.0, "RQ2", fontsize=7.0, ha="center", color=RED)
adm = box(40, 18, 28, 9, "admit if $r{>}0\\ \\wedge\\ r{\\geq}r_p$",
          fc="#f7e8e8", color=RED)
arrow(rew["l"], adm["r"])
arrow((adm["l"][0] - 0.5, adm["cy"]), (4, 40), conn="arc3,rad=0.0", style="-")
arrow((4, 40), (pool["l"][0] - 0.5, pool["cy"] - 2), conn="arc3,rad=0.0")

fig.tight_layout(pad=0.12)
out = sys.argv[1] if len(sys.argv) > 1 else "method2.pdf"
fig.savefig(out)
fig.savefig(out.rsplit(".", 1)[0] + ".png", dpi=150)
print("wrote", out)
