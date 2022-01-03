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

q=2.0
e1d=0.2
e2d=0.2
Tscaled=150
prefix="closeres"

###########################################################################
#################
# CONFIGURATION #
#################
h = 0.03
j = 2
a0 = 1.0
Nqs = 1
qs = np.ones(Nqs)*q
overwrite = False
totmass = 1.0e-4
Tw0 = 1000
TeRatios = sqrt(qs)

######################
# Varying parameters #
######################

#e1ds = np.linspace(0,0.1,4)
#e2ds = np.linspace(0,0.1,4)
#e1ds = np.linspace(0,0.3,4)
#e2ds = np.linspace(0,0.3,4)

E1DS = np.ones(Nqs)*e1d
E2DS = np.ones(Nqs)*e2d

E1_0 = np.ones(Nqs)*0.001
E2_0 = np.ones(Nqs)*0.001
#E1_0 = np.copy(E1DS)
#E2_0 = np.copy(E2DS)
#E1_0[E1_0==0] = 0.001
#E2_0[E2_0==0] = 0.001
print(E1_0)

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
TE1 = Tw0/TeRatios
TE2 = Tw0*TeRatios
#TM1 = np.infty*np.ones(Nqs)
TM1 = TE1/3.46/HS**2*(-1*(qs<1) + 1*(qs>=1))
TM2 = TE2/3.46/HS**2*(-1*(qs<1) + 1*(qs>=1))
TS = Tscaled*np.maximum(TE1, TE2)
#############################################################
# BUG: SETTING CUTOFF TO T RESULTS IN DIFFERENCES BETWEEN T #
# VALUES. LIKELY A FACTOR OF 2PI THING.                     #
#############################################################
cutoff_frac = 1.0
CUTOFFS = TS*cutoff_frac
#ALPHA2_0 = (startfrac2*3/2.)**(2./3)*np.ones(Nqs) #*(0.95*(QS>=1) + 1.05*(QS<1))
NAMES = np.array([f"e1d-{E1DS[i]:0.3f}-e2d-{E2DS[i]:0.3f}"
                  for i, qit in enumerate(QS)])

if prefix == "inres":
    ################
    # inres        #
    ################
    ALPHA2_0 = (1.5)**(2./3)*np.ones(Nqs)
    DIRNAMES = np.array([f"./longruns-{prefix}-driveTe-h-{h:0.2f}-mutot-{totmass:0.1e}-Tw0-{Tw0}-q{QS[i]:0.1f}" for i
                            in range(Nqs)])
    RUN_PARAMS = np.column_stack((HS, JS, A0S, QS, MU1, TS, TE1, TE2, TM1,
                                  TM2, E1_0, E2_0, E1DS, E2DS, ALPHA2_0,
                                  NAMES, DIRNAMES, CUTOFFS, TE_FUNCS,
                                  G1_0, G2_0))
    
    #i = 3
    #RUN_PARAMS = RUN_PARAMS[i:i+1,:]
    print(f"Running {RUN_PARAMS.shape[0]} simulations...")
    integrate = run.run_compmass_set(verbose=True, overwrite=overwrite,
                                     secular=True, method="RK45")
    Nproc = 16
    N_sims = len(QS)
    with Pool(processes=min(Nproc, N_sims)) as pool:
        pool.map(integrate, RUN_PARAMS)
if prefix == "closeres":
    ################
    # closeres #
    ################
    ALPHA2_0 = (1.55)**(2./3)*np.ones(Nqs)
    DIRNAMES = np.array([f"./longruns-{prefix}-driveTe-h-{h:0.2f}-mutot-{totmass:0.1e}-Tw0-{Tw0}-q{QS[i]:0.1f}" for i
                            in range(Nqs)])
    RUN_PARAMS = np.column_stack((HS, JS, A0S, QS, MU1, TS, TE1, TE2, TM1,
                                  TM2, E1_0, E2_0, E1DS, E2DS, ALPHA2_0,
                                  NAMES, DIRNAMES, CUTOFFS, TE_FUNCS,
                                  G1_0, G2_0))
    
    #i = 3
    #RUN_PARAMS = RUN_PARAMS[i:i+1,:]
    print(f"Running {RUN_PARAMS.shape[0]} simulations...")
    integrate = run.run_compmass_set(verbose=True, overwrite=overwrite,
                                     secular=True, method="RK45")
    Nproc = 16
    N_sims = len(QS)
    with Pool(processes=min(Nproc, N_sims)) as pool:
        pool.map(integrate, RUN_PARAMS)
