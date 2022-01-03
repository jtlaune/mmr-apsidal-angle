import numpy as np
import scipy as sp
from scipy import optimize
from IPython.display import display, clear_output
import matplotlib as mpl
import matplotlib.pyplot as plt
import rebound
import sys
import os
import importlib
sys.path.append("/home/jtlaune/mmr/")
import mmr
import run
import runparams
from runparams import *

mpl.rcParams.update({'font.size': 24})

def plotsim(fig, ax, teval, theta, a1, ap, e1, eeq, ebar, g, barg, suptitle, figname):
    ax[0,0].plot(teval/1e5, theta)
    ax[0,1].plot(teval/1e5, (a1/ap)**1.5)
    ax[1,0].plot(teval/1e5, e1)
    ax[1,1].plot(teval/1e5, ebar)
    ax[2,0].plot(teval/1e5, g)
    ax[2,1].plot(teval/1e5, barg)
    
    ax[0,0].set_title(r"$\theta$")
    ax[0,1].set_title(r"Period Ratio")
    ax[1,0].set_title(r"$e_1$")
    ax[1,1].set_title(r"$\overline{e}$")
    ax[2,0].set_title(r"$\gamma$")
    ax[2,1].set_title(r"$\overline{\gamma}$")
    ax[0,0].set_xlabel(r"[$10^5$ orbits]")
    ax[0,1].set_xlabel(r"[$10^5$ orbits]")
    ax[1,0].set_xlabel(r"[$10^5$ orbits]")
    ax[1,1].set_xlabel(r"[$10^5$ orbits]")

    eylim = (0, np.max([np.max(e1)*1.1, np.max(ebar)*1.1]))
    
    ax[0,0].set_ylim((-np.pi,np.pi))
    ax[1,0].axhline(y=eeq, c="k", ls="--")
    ax[1,0].set_ylim(eylim)
    ax[1,1].set_ylim(eylim)
    ax[2,0].set_ylim((-np.pi,np.pi))
    ax[2,1].set_ylim((-np.pi,np.pi))
    fig.suptitle(suptitle, fontsize=16)
    fig.tight_layout()
    fig.savefig(figname)

j = 2
e0 = 0.0
ap = 1.0
theta0 = 0
g0 = 0.0
eta0 = 1.0
a0 = 1.4
lambda0 = 0.0

eps = np.logspace(np.log10(1e-2), np.log10(0.1), 5)
eeqs = np.logspace(np.log10(1e-2), np.log10(0.1), 5)
ratios = eeqs**2*2*j
i = 0
for ep in eps:
    for ratio in ratios:
        i+=1
        Tm = -1e6
        Te = -ratio*Tm
        T = 5e5
        mup = 1e-3
        
        sim = run.tp_intH(j, mup, ep, e0, ap, g0, a0, lambda0)
        teval, thetap, newresin, newresout, R, eta, a1, e1, newein, k, kc, alpha0, alpha, g, L, G, ebar, barg = sim.int_Hsec(0, T,
                                                                                                                        1e-6,
                                                                                                                        Tm, Te)
        label = "{:0.2e}-{:0.2e}".format(ep, sim.e_eq)
        fig,ax=plt.subplots(3,2,figsize=(10,15))
        suptitle = "$\\mu_{{p}} = {:0.2e}$; $e_p={:0.2e}$; $e_{{\\rm eq}} = {:0.2e}$".format(sim.mup, ep, sim.e_eq)
        figname = "external-grid-"+label+".png"
        plotsim(fig, ax, teval, newresout, a1, ap, e1, sim.e_eq, ebar, g, barg, suptitle, figname)
        np.savez(label+".npz", thetap=thetap,L=L,g=g,G=G)
        
