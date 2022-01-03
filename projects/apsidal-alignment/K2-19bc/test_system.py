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

mpl.rcParams.update({"font.size": 20, "figure.facecolor": "white"})

#################
# CONFIGURATION #
#################
j = 2
a0 = 1.0
h = 0.025
alpha_0 = (j/(j+1))**(2./3.)
Nqs = 8
qs = np.ones(Nqs)*3.00
overwrite = True
totmass = 1.11e-4 + 3.69e-5
Tw0 = 1e3
e2d = 0.0
e1d = 0.2
T_frac = 20.
cutoff_frac = 0.5

####################
# THREADING ARRAYS #
####################
HS = np.ones(Nqs)*h
JS = np.ones(Nqs)*j
A0S = np.ones(Nqs)*a0
QS = qs
MU2 = totmass/(1+QS)
MU1 = totmass - MU2
print(MU1, MU2)
TM1 = -np.ones(Nqs)*1e6
TM2 = TM1/10.
TE1 = np.ones(Nqs)*Tw0
TE2 = np.logspace(np.log10(Tw0)-2, np.log10(Tw0)+2, Nqs)
#TE1 = np.logspace(np.log10(Tw0)-2, np.log10(Tw0)+2, Nqs)
#TE2 = np.ones(Nqs)*Tw0
TS = T_frac*TE2
CUTOFFS = cutoff_frac*TS
E1_0 = np.ones(Nqs)*e1d
E2_0 = np.ones(Nqs)*0.03
E1DS = np.ones(Nqs)*e1d
E2DS = np.ones(Nqs)*e2d
ALPHA2_0 = (3/2.)**(2./3)*(1+e1d**2+e2d**2)*np.ones(len(QS))
NAMES = np.array([f"q{qit:0.3f}-Te1-{TE1[i]:0.3f}-Te2-{TE2[i]:0.3f}"
                  for i, qit in enumerate(QS)])

################
# WITH SECULAR #
################
DIRNAMES = np.array([f"./e1d{e1d:0.3f}-e2d{e2d:0.3f}-Tw0{Tw0}-cutoff{cutoff_frac}" for i
                        in range(Nqs)])
RUN_PARAMS = np.column_stack((HS, JS, A0S, QS, MU1, TS, TE1, TE2, TM1,
                              TM2, E1_0, E2_0, E1DS, E2DS, ALPHA2_0,
                              NAMES, DIRNAMES, CUTOFFS))
print(f"Running {RUN_PARAMS.shape[0]} simulations...")
integrate = run.run_compmass_set(verbose=True, overwrite=overwrite,
                                 secular=True)
N_sims = len(QS)
with Pool(processes=min(8, N_sims)) as pool:
    pool.map(integrate, RUN_PARAMS)
print(TS, CUTOFFS)

###################
# WITHOUT SECULAR #
###################
DIRNAMES = np.array([f"./e1d{e1d:0.3f}-e2d{e2d:0.3f}-Tw0{Tw0}-cutoff{cutoff_frac}-nosec" for i
                        in range(Nqs)])
RUN_PARAMS = np.column_stack((HS, JS, A0S, QS, MU1, TS, TE1, TE2, TM1,
                              TM2, E1_0, E2_0, E1DS, E2DS, ALPHA2_0,
                              NAMES, DIRNAMES, CUTOFFS))
print(f"Running {RUN_PARAMS.shape[0]} simulations...")
integrate = run.run_compmass_set(verbose=True, overwrite=overwrite,
                                 secular=False)
N_sims = len(QS)
with Pool(processes=min(8, N_sims)) as pool:
    pool.map(integrate, RUN_PARAMS)

print(TS, CUTOFFS)
