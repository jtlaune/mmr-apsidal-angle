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
    ax[0,0].axhline(y=0., c="k", ls="--")
    ax[1,0].axhline(y=eeq, c="k", ls="--")
    ax[1,0].set_ylim(eylim)
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
a0 = 0.7
lambda0 = 0.0

eps = np.logspace(np.log10(1e-2), np.log10(0.1), 5)
eeqs = np.logspace(np.log10(1e-2), np.log10(0.1), 5)
ratios = eeqs**2*2*(j+1)
i = 0
for ep in eps:
    for ratio in ratios:
        i+=1
        Tm = 1e6
        Te = ratio*Tm
        T = 5e5
        stable= j/(np.sqrt(3)*(j+1)**1.5*0.8*j)*(Te/(Tm/1.5))**1.5
        unstable= j**2/(8*np.sqrt(3)*(j+1)**1.5*0.8*j)*(Te/(Tm/1.5))**1.5
        capture =(2.5/(j**(5./3.)*(2*np.pi*Tm/1.5)*(j+1)/j))**(0.75)
        mup = 1.5*stable
        #mup = 5e-3
        
        sim = run.tp_intH(j, mup, ep, e0, ap, g0, a0, lambda0)
        teval, thetap, newresin, newresout, R, eta, a1, e1, k, kc, alpha0, alpha, g, L, G, ebar, barg = sim.int_Hsec(0, T,
                                                                                                                        1e-6,
                                                                                                                        Tm, Te)
        label = "{:0.2e}-{:0.2e}".format(ep, sim.e_eq)
        fig,ax=plt.subplots(3,2,figsize=(10,15), constrained_layout=True)
        suptitle = "$\\mu_{{p}} = {:0.2e}$; $e_p={:0.2e}$; $e_{{\\rm eq}} = {:0.2e}$".format(sim.mup, ep, sim.e_eq)
        figname = "internal-grid-"+label+".png"
        plotsim(fig, ax, teval, newresin, a1, ap, e1, sim.e_eq, ebar, g, barg, suptitle, figname)
        #fig, ax = plt.subplots(3)
        #ax[0].scatter(R*np.cos(newresin), R*np.sin(newresin), c=teval)
        #ax[1].scatter(teval, eta)
        #it = np.where(teval > 4e5)[0][0]
        #etaavg = np.average(eta[it:])
        #H = lambda x,y : etaavg*x - x*x + np.sqrt(x)*np.cos(y)
        #R = np.linspace(0,4,1000)
        #T = np.linspace(0,2*np.pi,1000)
        #RR, TT = np.meshgrid(R,T)
        #ax[2].contour(RR*np.cos(TT),RR*np.sin(TT), H(RR,TT))
        #ax[2].set_title(etaavg)
        #fig.savefig("phasediagweirdres.png")
        print(i)
        
