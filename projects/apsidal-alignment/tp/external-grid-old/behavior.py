import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 20})

sims = np.loadtxt("behaviors.txt", delimiter=",")

ep = sims[:,0]
eq = sims[:,1]

capalign = np.ones((sims.shape[0], 2))*-1
capcirc  = np.ones((sims.shape[0], 2))*-1

for i in range(len(sims[:,0])):
    if sims[i,2] == 0:
        capalign[i,:] = sims[i,0:2]
    if sims[i,2] == 1:
        capcirc[i,:] = sims[i,0:2]

figname = "behaviors.png"
fig, ax = plt.subplots()

ax.scatter(3.16e-2,5.62e-2,marker="*",s=400,c="k")
ax.scatter(5.62e-2,1.78e-2,marker="*",s=400,c="k")

ax.scatter(capalign[:,0], capalign[:,1], marker="*", s=200, label="capture, aligned")
ax.scatter(capcirc[:,0], capcirc[:,1], marker="*", s=200, label="capture, circulating")


ax.legend(bbox_to_anchor=(1.05, 1))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$e_p$")
ax.set_ylabel(r"$e_{eq}$")
ax.set_xlim((ep[0]*0.8, ep[-1]*1.2))
ax.set_ylim((eq[0]*0.8, eq[-1]*1.2))
fig.savefig(figname, bbox_inches="tight")
