#! /home/jtlaune/.pythonvenvs/science/bin/python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os

sys.path.append("/home/jtlaune/mmr/")
mpl.rcParams.update(
    {"font.size": 24, "figure.facecolor": "white", "figure.figsize": (8, 6)}
)

i = 0
mup = 1e-3
T = 2e5
Tm = -1e6
tol = 1e-6

if not os.path.isdir("images/"):
    os.mkdir("images/")

galigns = np.loadtxt("behaviors.txt", skiprows=1, delimiter=",")
ep = galigns[:, 0]
eeq = galigns[:, 1]
aligns = galigns[:, 3]
print(ep, eeq, aligns)
mask_circ = aligns > 0
mask_align = aligns < 1

ep_circ = np.ma.masked_array(ep, mask=mask_circ)
ep_align = np.ma.masked_array(ep, mask=mask_align)

eeq_circ = np.ma.masked_array(eeq, mask=mask_circ)
eeq_align = np.ma.masked_array(eeq, mask=mask_align)

fig, ax = plt.subplots()
ax.scatter(
    ep_circ,
    eeq_circ,
    marker="o",
    s=120,
    facecolors="none",
    edgecolors="k",
    label=r"$\gamma$-circulating",
)
ax.scatter(ep_align, eeq_align, marker="o", c="k", s=120, label=r"$\gamma$-aligned")
ax.set_xscale("symlog", linthreshx=0.01, linscalex=0.2, subsx=[2, 3, 4, 5, 6, 7, 8, 9])
ax.set_yscale("log", subsy=[2, 3, 4, 5, 6, 7, 8, 9])
ax.set_xlabel(r"$e_{\rm p}$", fontsize=32)
ax.set_ylabel(r"$e_{\rm d}$", fontsize=32)
ax.tick_params(length=8, which="major")
ax.tick_params(length=4, which="minor")
ax.legend(bbox_to_anchor=(1.01, 1.03))
ax.set_title(r"External MMR $\gamma$-alignment")
fig.savefig("images/galigns.png", bbox_inches="tight")
