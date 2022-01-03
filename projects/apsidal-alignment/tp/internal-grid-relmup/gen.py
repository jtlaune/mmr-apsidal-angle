#! /home/jtlaune/.pythonvenvs/science/bin/python
import numpy as np
import scipy as sp
from scipy import optimize
from IPython.display import display, clear_output
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
import importlib

sys.path.append("/home/jtlaune/mmr/")
mpl.rcParams.update({'font.size': 20})
import plotting
from plotting import *
import mmr
import run

j = 2
e0 = 0.0
ap = 1.0
theta0 = 0
g0 = 0.0
eta0 = 1.0
a0 = 0.72
lambda0 = 0.0

N=11
eccs = np.zeros(N)
eccs[1:] = np.logspace(-2,-1,10)
eps = eccs
eeqs = eccs[1:]
ratios = eeqs**2*2*j
i = 0
T = 7.5e4
Tm = 1e6
tol = 1e-6

# ep = 0
for ir, ratio in enumerate(ratios):
    ep = 0
    i+=1
    Te = ratio*Tm

    stable= j/(np.sqrt(3)*(j+1)**1.5*0.8*j)*(Te/(Tm/1.5))**1.5
    mup = 1.5*stable
    label = "{:0.2e}-{:0.2e}-{:0.2e}".format(ep, eeqs[ir], mup)

    print("{:0.1f}%: {}"
          .format(100*i/(N*(N-1)), label),
          end="\r")

    sim = run.tp_intH(j, mup, ep, e0, ap, g0, a0, lambda0)
    (teval, thetap, newresin,
     newresout, eta, a1, e1, k, kc,
     alpha0, alpha, g, L, G,
     ebar, barg, x,y) = sim.int_Hsec(0, T, tol, Tm, Te)

    resangle = (thetap+barg)%(2*np.pi)
    ifrac50 = int(0.5*len(resangle))
    resdiff = resangle[ifrac50:] 
    resdiff = resdiff - 2*np.pi*(resdiff>np.pi)
    if np.any(np.abs(resdiff) > 0.3*np.pi):
        raise Warning("broke res > T/2")
    

    np.savez(label+".npz", teval=teval, thetap=thetap,
             L=L, g=g, G=G, x=x, y=y)
# ep>0
for ie, ep in enumerate(eps[1:]):
    for ir, ratio in enumerate(ratios[ie:]):
        i+=1
        Te = ratio*Tm

        stable= j/(np.sqrt(3)*(j+1)**1.5*0.8*j)*(Te/(Tm/1.5))**1.5
        mup = 1.5*stable

        label = "{:0.2e}-{:0.2e}-{:0.2e}".format(ep, eeqs[ir], mup)

        print("{:0.1f}%: {}"
              .format(100*i/(N*(N-1)), label),
              end="\r")

        sim = run.tp_intH(j, mup, ep, e0, ap, g0, a0, lambda0)
        (teval, thetap, newresin,
         newresout, eta, a1, e1, k, kc,
         alpha0, alpha, g, L, G,
         ebar, barg, x,y) = sim.int_Hsec(0, T, tol, Tm, Te)

        resangle = (thetap+barg)%(2*np.pi)
        ifrac50 = int(0.5*len(resangle))
        resdiff = resangle[ifrac50:]
        resdiff = resdiff - 2*np.pi*(resdiff>np.pi)
        if not np.any(np.abs(resdiff) > 0.3*np.pi):
            np.savez(label+".npz", teval=teval, thetap=thetap,
                     L=L, g=g, G=G, x=x, y=y)
        else:
            with open("brokeres.txt", "a") as f:
                f.write("{}\n".format(label))
        

        
