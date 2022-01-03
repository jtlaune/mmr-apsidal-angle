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
h = 0.1
Nqs = 1
qs = np.ones(Nqs)*2
overwrite = True
totmass = 1e-4
#e2d = 0.0
#e1d = 0.2
TeRatios = sqrt(qs)

######################
# Varying parameters #
######################
E1_0 = np.ones(Nqs)*0.001
E2_0 = np.ones(Nqs)*0.001
E1DS = np.ones(Nqs)*0.0
E2DS = np.ones(Nqs)*0.0


#eccs = np.array([0.1])
#E1_0, E2_0 = np.meshgrid(eccs, eccs)
#E1_0 = np.flip(E1_0.flatten())
#E2_0 = np.flip(E2_0.flatten())

####################
# THREADING ARRAYS #
####################
HS = np.ones(Nqs)*h
JS = np.ones(Nqs)*j
A0S = np.ones(Nqs)*a0
QS = qs
MU2 = totmass/(1+QS)
MU1 = totmass - MU2
TE1 = np.infty
TE2 = np.infty
TM1 = np.infty
TM2 = -1e5
TS = 7e4
#############################################################
# BUG: SETTING CUTOFF TO T RESULTS IN DIFFERENCES BETWEEN T #
# VALUES. LIKELY A FACTOR OF 2PI THING.                     #
#############################################################
cutoff_frac = 0.8
CUTOFFS = TS*cutoff_frac
ALPHA2_0 = (3/2.)**(2./3)*1.3
NAMES = np.array([f"e10-{E1_0[i]:0.3f}-e20-{E2_0[i]:0.3f}"
                  for i, qit in enumerate(QS)])

DIRNAMES = np.array([f"./onlyDrag-h-{h:0.2f}-cut-{cutoff_frac:0.2f}-mutot-{totmass:0.1e}" for i
                        in range(Nqs)])
DIRNAMES_NOSEC = np.array([DIRNAMES[i]+"_NOSEC" for i in range(Nqs)])

################
# WITH SECULAR #
################
RUN_PARAMS = np.column_stack((HS, JS, A0S, QS, MU1, TS, TE1, TE2, TM1,
                              TM2, E1_0, E2_0, E1DS, E2DS, ALPHA2_0,
                              NAMES, DIRNAMES, CUTOFFS))
print(RUN_PARAMS)
print(f"Running {RUN_PARAMS.shape[0]} simulations...")
integrate = run.run_compmass_set(verbose=True, overwrite=overwrite,
                                 secular=True, method="RK45")
N_sims = len(QS)
with Pool(processes=min(8, N_sims)) as pool:
    pool.map(integrate, RUN_PARAMS)

###################
# WITHOUT SECULAR #
###################
#RUN_PARAMS = np.column_stack((HS, JS, A0S, QS, MU1, TS, TE1, TE2, TM1,
#                              TM2, E1_0, E2_0, E1DS, E2DS, ALPHA2_0,
#                              NAMES, DIRNAMES_NOSEC, CUTOFFS))
#print(f"Running {RUN_PARAMS.shape[0]} simulations...")
#integrate = run.run_compmass_set(verbose=True, overwrite=overwrite,
#                                 secular=False)
#N_sims = len(QS)
#with Pool(processes=min(8, N_sims)) as pool:
#    pool.map(integrate, RUN_PARAMS)
