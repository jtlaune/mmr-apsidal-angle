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

# params 
from params import *

i = 0
mup = 1e-3
T = 7.5e4
Tm = -1e6
tol = 1e-6

for ep in eps:
    for ir, ratio in enumerate(ratios):
        i+=1
        Te = -ratio*Tm
        label = "{:0.2e}-{:0.2e}".format(ep, eeqs[ir])

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
        resdiff = np.abs(resangle[ifrac50:] - np.pi)
        if np.any(resdiff > 0.2*np.pi):
            raise Warning("broke res > T/2")
        

        np.savez(label+".npz", teval=teval, thetap=thetap,
                 L=L, g=g, G=G, x=x, y=y)
        
