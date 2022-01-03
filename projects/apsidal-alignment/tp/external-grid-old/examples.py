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

mpl.rcParams.update({'font.size': 20})

def plotsim(fig, ax, teval, theta, a1, ap, e1, eeq, ebar, g, barg, suptitle, figname):
    ax[0,0].plot(teval/1e5, theta)
    ax[0,1].plot(teval/1e5, (a1/ap)**1.5)
    ax[1,0].plot(teval/1e5, e1)
    ax[1,1].plot(teval/1e5, ebar)
    ax[2,0].plot(teval/1e5, g)
    ax[2,1].plot(teval/1e5, barg)
    
    ax[0,0].set_title(r"$\overline{\theta}$")
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
    
    ax[0,0].set_ylim((0,2*np.pi))
    ax[0,0].axhline(y=np.pi, c="k", ls="--")
    ax[1,0].axhline(y=eeq, c="k", ls="--")
    ax[1,0].set_ylim(eylim)
    ax[0,1].set_ylim((1.48, 1.4**(3./2.)))
    ax[1,1].set_ylim(eylim)
    ax[2,0].set_ylim((-np.pi,np.pi))
    ax[2,1].set_ylim((-np.pi,np.pi))
    fig.suptitle(suptitle)
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

examples = zip([eps[2], eps[3]], [ratios[3], ratios[1]])
for ep, ratio in examples:
        i+=1
        Tm = -1e6
        Te = -ratio*Tm
        T = 5e5
        mup = 1e-3
        e_eq = np.sqrt(ratio/2./(j))

        
        sim = run.tp_intH(j, mup, ep, e0, ap, g0, a0, lambda0)
        label = "{:0.2e}-{:0.2e}".format(ep, e_eq)
        data = np.load(label+".npz")
        thetap = data["thetap"]
        L = data["L"]
        g = data["g"]
        G = data["G"]

        
        e1 = np.sqrt(1-(1-G/L)**2)
        a1 = L**2*sim.ap
        if Tm > 0:
            alpha = a1/sim.ap
            barg = np.arctan2(e1*np.sin(g), 
                              ((e1*np.cos(g)
                                +sim.B(alpha)/sim.A(alpha)*sim.ep))) 
            ebar = sim.ebar(sim.ep, e1, sim.A(alpha), sim.B(alpha), g)
        else:
            alpha = sim.ap/a1
            barg = np.arctan2(e1*np.sin(g), 
                              ((e1*np.cos(g)
                                +sim.A(alpha)/sim.B(alpha)*sim.ep))) 
            ebar = sim.ebar(sim.ep, e1, sim.B(alpha), sim.A(alpha), g)
        #newresin = ((sim.j+1)*sim.n_p*teval/sim.tau - sim.j*l +barg)%(2*np.pi)
        #newresout = ((sim.j+1)*l - sim.j*sim.n_p*teval/sim.tau +barg)%(2*np.pi)
        newresin = (thetap + barg)%(2*np.pi)
        newresout = (thetap + barg)%(2*np.pi)
        for i in range(len(newresin)):
            if newresin[i]>np.pi:
                newresin[i] = newresin[i] - 2*np.pi
            if barg[i]>np.pi:
                barg[i] = barg[i] - 2*np.pi
        alpha0 = alpha*(1+sim.j*ebar**2)
        k =((sim.j+1)*alpha0**1.5-sim.j)
        kc = (3**(1./3.)*(sim.mup*alpha0 \
                               *sim.j*np.abs(sim.A(alpha0)))**(2./3.))
        eta = k/kc
        R = sim.R(ebar, sim.j, sim.A(alpha0), sim.mup, alpha0)
        #H = (eta*R - R**2 + np.sqrt(R)*np.cos(newresin))
        # turn teval into planet orbits
        #teval = teval/sim.tau*(2*np.pi/sim.n_p)
        teval = np.linspace(0,T,len(L))

        fig,ax=plt.subplots(3,2,figsize=(10,15), constrained_layout=True)
        suptitle = "$\\mu_{{p}} = {:0.2e}$; $e_p={:0.2e}$; $e_{{\\rm eq}} = {:0.2e}$".format(sim.mup, ep,e_eq)
        figname = "example-"+label+".png"
        plotsim(fig, ax, teval, newresout, a1, ap, e1, e_eq, ebar, g, barg, suptitle, figname)
        
