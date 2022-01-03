#! /home/jtlaune/miniconda3/envs/science/bin/python
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

mpl.rcParams.update({"font.size": 20, "figure.facecolor": "white"})


#################
# CONFIGURATION #
#################
j = 2
a0 = 1.0
alpha_0 = (j/(j+1))**(2./3.)
Nqs = 10
qs = np.logspace(np.log10(0.5),np.log10(0.85),int(Nqs/2))
qs = np.append(qs, np.logspace(np.log10(1.15),np.log10(2.0),int(Nqs/2)))
print(qs)
Nqs = len(qs)
overwrite = True
totmass = 1e-3
e0 = 0.001

######################
# Varying parameters #
######################
Tw0 = 1000.
TeRatios = np.sqrt(qs)

####################
# THREADING ARRAYS #
####################
TE_FUNCS = np.zeros(Nqs)
G1_0 = np.array([np.random.uniform(0, 2*np.pi) for i in range(Nqs)])
G2_0 = np.array([np.random.uniform(0, 2*np.pi) for i in range(Nqs)])
HS = np.ones(Nqs)*0.03
JS = np.ones(Nqs)*j
A0S = np.ones(Nqs)*a0
QS = np.array(qs)
MU2 = totmass/(1+QS)
MU1 = totmass - MU2
ALPHA_0 = alpha_0*np.ones(Nqs)
TE1 = Tw0/TeRatios
TE2 = Tw0*TeRatios
#TM1 = np.infty*np.ones(Nqs)
TM1 = TE1/3.46/HS**2*(-1*(qs<1) + 1*(qs>=1))
TM2 = TE2/3.46/HS**2*(-1*(qs<1) + 1*(qs>=1))
TS = 150.*np.maximum(TE1, TE2)
print(TS)
#E1_0 = np.minimum(0.1/sqrt(QS), 0.1*np.ones(Nqs))
#E2_0 = np.minimum(0.1*sqrt(QS), 0.1*np.ones(Nqs))
E1_0 = np.ones(Nqs)*e0
E2_0 = np.ones(Nqs)*e0
print(E1_0,E2_0)
E1DS = np.zeros(Nqs)
E2DS = np.zeros(Nqs)
CUTOFFS = TS
#ALPHA2_0 = (3/2.)**(2./3)*(1+E2_0**2+E1_0**2)
ALPHA2_0 = (1.6)**(2./3)*np.ones(Nqs)

NAMES = np.array([f"q{QS[i]:0.2f}" for i in range(Nqs)])

DIRNAMES = np.array([f"standard-h-{HS[i]:0.2f}-Tw0-{int(Tw0)}"
                     for i in range(len(QS))])
DIRNAMES_NOSEC = np.array([DIRNAMES[i]+"-nosec" for i in range(Nqs)])

################
# WITH SECULAR #
################
RUN_PARAMS = np.column_stack((HS, JS, A0S, QS, MU1, TS, TE1, TE2, TM1,
                              TM2, E1_0, E2_0, E1DS, E2DS, ALPHA2_0,
                              NAMES, DIRNAMES, CUTOFFS, TE_FUNCS,
                              G1_0, G2_0))
print(f"Running {RUN_PARAMS.shape[0]} simulations...")
integrate = run.run_compmass_set(verbose=True, overwrite=overwrite,
                                 secular=True)

Nproc = 16
N_sims = len(QS)
with Pool(processes=min(Nproc, N_sims)) as pool:
    pool.map(integrate, RUN_PARAMS)

###################
# WITHOUT SECULAR #
###################
RUN_PARAMS = np.column_stack((HS, JS, A0S, QS, MU1, TS, TE1, TE2, TM1,
                              TM2, E1_0, E2_0, E1DS, E2DS, ALPHA2_0,
                              NAMES, DIRNAMES_NOSEC, CUTOFFS, TE_FUNCS,
                              G1_0, G2_0))

print(f"Running {RUN_PARAMS.shape[0]} simulations...")
integrate = run.run_compmass_set(verbose=True, overwrite=overwrite,
                                 secular=False)

N_sims = len(QS)
with Pool(processes=min(8, N_sims)) as pool:
    pool.map(integrate, RUN_PARAMS)
