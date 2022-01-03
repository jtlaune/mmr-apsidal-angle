#! /home/jtlaune/.pythonvenvs/science/bin/python
from multiprocessing import Pool, TimeoutError
import subprocess
import numpy as np
from numpy import sqrt, pi, cos, sin, abs
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
import importlib

sys.path.append("/home/jtlaune/multi-planet-architecture/")
import run
import plotting
from plotting import plotsim, loadsim
from helper import *

#################
# CONFIGURATION #
#################
# derived parameters
eeq2 = 0.01

# const params
j = 2
mu1 = 1e-3
q = 1e3
a0 = 1.0
Tm1 = np.inf
Te1 = np.inf
Tm2 = 1e5
Te2 = eeq2**2*Tm2*2*(j+1)
e1d = 0.0
e2d = 0.0
#T = 20.*Te2
#print(T)
T = Tm2
e1_0 = 0.1
e2_0 = 0.01
alpha2_0 = ((j+1)/j)**(2./3)*(1 + eeq2**2)

# varying params
#tols = 10.**np.linspace(-12,-5, 8)
tols = [1e-9]
print(tols)

for tol in tols:
    sim = run.comp_mass_intH(j, mu1, q, a0, Tm1, Tm2, Te1, Te2, e1d=e1d, e2d=e2d)
    (teval, theta, a1, a2, e1, e2,
     g1, g2, L1, L2, x1, y1, x2, y2) = sim.int_Hsec(T, tol, alpha2_0,
                                                    e1_0, e2_0,
                                                    verbose=True,
                                                    secular=True)
    
    dirname = "long"
    filename = f"tol-{tol:0.2e}.npz"
    np.savez(
        os.path.join(dirname, filename),
        teval=teval,
        thetap=theta,
        a1=a1,
        a2=a2,
        e1=e1,
        e2=e2,
        g1=g1,
        g2=g2,
        L1=L1,
        L2=L2,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )
