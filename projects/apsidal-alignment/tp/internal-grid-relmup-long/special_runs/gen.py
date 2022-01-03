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
i = 0
T = 4e5
Tm = 1e6
tol = 1e-6

ep = eccs[-1]
eeq0 = eccs[-1]
ratio = eeq0**2*2*j
Te = ratio*Tm

stable= j/(np.sqrt(3)*(j+1)**1.5*0.8*j)*(Te/(Tm/1.5))**1.5
mup = 1.5*stable

label = "{:0.2e}-{:0.2e}-{:0.2e}".format(ep, eeq0, mup)

print("{:0.1f}%: {}"
      .format(100*i/(N*(N-1)), label),
      end="\r")

sim = run.tp_intH(j, mup, ep, e0, ap, g0, a0, lambda0)
(teval, thetap, newresin,
 newresout, eta, a1, e1, k, kc,
 alpha0, alpha, g, L, G,
 ebar, barg, x,y) = sim.int_Hsec(0, T, tol, Tm, Te)

np.savez(label+".npz", teval=teval, thetap=thetap,
         L=L, g=g, G=G, x=x, y=y)

