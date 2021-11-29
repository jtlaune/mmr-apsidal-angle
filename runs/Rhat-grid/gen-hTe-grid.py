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
h = sqrt(0.001)
j = 2
a0 = 0.1
qRun = 16
#Nqs = 16*2
Nqs = 16
qs = np.ones(Nqs)
qs[:qRun] = qs[:qRun]*2
#qs[qRun:] = qs[qRun:]*0.5
overwrite = False
totmass = 1.0e-4
Tw0 = 1000
TeRatios = sqrt(qs)

######################
# Varying parameters #
######################
E1_0 = np.ones(Nqs)*0.001
E2_0 = np.ones(Nqs)*0.001

e1ds = np.linspace(0,0.3,4)
e2ds = np.linspace(0,0.3,4)

E1DS_single, E2DS_single = np.meshgrid(e1ds, e2ds)
E1DS_single = E1DS_single.flatten()
E2DS_single = E2DS_single.flatten()

E1DS = np.array([])
E2DS = np.array([])
for i in range(2):
    E1DS = np.append(E1DS, E1DS_single)
    E2DS = np.append(E2DS, E2DS_single)
print(len(E1DS))

G1_0 = np.array([np.random.uniform(0, 2*np.pi) for i in range(Nqs)])
G2_0 = np.array([np.random.uniform(0, 2*np.pi) for i in range(Nqs)])

####################
# THREADING ARRAYS #
####################
HS = np.ones(Nqs)*h
JS = np.ones(Nqs)*j
A0S = np.ones(Nqs)*a0
QS = qs
MU2 = totmass/(1+QS)
MU1 = totmass - MU2

TE_FUNCS = np.zeros(Nqs)
TE1 = Tw0*TeRatios
TE2 = Tw0/TeRatios
TM1 = TE1/3.46/HS**2*(1*(QS<1) - 1*(QS>=1))
TM2 = TE2/3.46/HS**2*(1*(QS<1) - 1*(QS>=1))
TS = 30.*np.maximum(TE1, TE2)
#############################################################
# BUG: SETTING CUTOFF TO T RESULTS IN DIFFERENCES BETWEEN T #
# VALUES. LIKELY A FACTOR OF 2PI THING.                     #
#############################################################
cutoff_frac = 1.0
CUTOFFS = TS*cutoff_frac
ALPHA2_0 = (3/2.)**(2./3)*np.ones(Nqs) #*(0.95*(QS>=1) + 1.05*(QS<1))
NAMES = np.array([f"e1d-{E1DS[i]:0.1f}-e2d-{E2DS[i]:0.1f}"
                  for i, qit in enumerate(QS)])

DIRNAMES = np.array([f"./driveTe-h-{h:0.2f}-mutot-{totmass:0.1e}-Tw0-{Tw0}-q{QS[i]:0.1f}" for i
                        in range(Nqs)])
DIRNAMES_NOSEC = np.array([DIRNAMES[i]+"_NOSEC" for i in range(Nqs)])

################
# WITH SECULAR #
################
RUN_PARAMS = np.column_stack((HS, JS, A0S, QS, MU1, TS, TE1, TE2, TM1,
                              TM2, E1_0, E2_0, E1DS, E2DS, ALPHA2_0,
                              NAMES, DIRNAMES, CUTOFFS, TE_FUNCS,
                              G1_0, G2_0))
#RUN_PARAMS = RUN_PARAMS[-1:,:]
print(RUN_PARAMS.shape)
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
#                              NAMES, DIRNAMES_NOSEC, CUTOFFS, TE_FUNCS,
#                              G1_0, G2_0))
#print(f"Running {RUN_PARAMS.shape[0]} simulations...")
#integrate = run.run_compmass_set(verbose=True, overwrite=overwrite,
#                                 secular=False)
#N_sims = len(QS)
#with Pool(processes=min(8, N_sims)) as pool:
#    pool.map(integrate, RUN_PARAMS)
