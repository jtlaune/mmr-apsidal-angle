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
h = 0.03
alpha_0 = (j/(j+1))**(2./3.)
Nqs = 8
qs = np.ones(Nqs)*2
overwrite = True
totmass = 1e-4
Tw0 = 1000
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
TE_FUNCS = np.zeros(Nqs)
G1_0 = np.array([np.random.uniform(0, 2*np.pi) for i in range(Nqs)])
G2_0 = np.array([np.random.uniform(0, 2*np.pi) for i in range(Nqs)])
HS = np.ones(Nqs)*h
JS = np.ones(Nqs)*j
A0S = np.ones(Nqs)*a0
QS = qs
MU2 = totmass/(1+QS)
MU1 = totmass - MU2
TE1 = Tw0/TeRatios
TE2 = Tw0*TeRatios
#TM1 = np.infty*np.ones(Nqs)
TM1 = TE1/3.46/HS**2*(-1*(qs<1) + 1*(qs>=1))
TM2 = TE2/3.46/HS**2*(-1*(qs<1) + 1*(qs>=1))
TS = 30.*np.maximum(TE1, TE2)
ALPHA_0 = alpha_0*np.ones(Nqs)
#############################################################
# BUG: SETTING CUTOFF TO T RESULTS IN DIFFERENCES BETWEEN T #
# VALUES. LIKELY A FACTOR OF 2PI THING.                     #
#############################################################
cutoff_frac = 1.0
CUTOFFS = TS*cutoff_frac
ALPHA2_0 = (1.6)**(2./3)*np.ones(Nqs)

#def muext(omeff, aext):
def omeffs(a0, j, muext, aext):
    alpha = (j/(j+1))**(2./3)
    a1 = alpha*a0
    a2 = a0
    om1 = om1ext_np(muext, a1, a2, aext)
    om2 = ompext_np(muext, a1, a2, aext)
    return(om2 - om1)

AEXTS = np.linspace(1.5,10,Nqs,endpoint=True)
MUEXTS = np.ones(Nqs)*1e-2
OMEFFS = omeffs(A0S, j, MUEXTS, AEXTS)

NAMES = np.array([f"omeff-{OMEFFS[i]:0.1e}-e1d-{E1DS[i]:0.3f}-e2d-{E2DS[i]:0.3f}"
                  for i, qit in enumerate(QS)])

DIRNAMES = np.array([f"./h-{h:0.2f}-Tw0-{Tw0}-mutot-{totmass:0.1e}" for i
                        in range(Nqs)])
DIRNAMES_NOSEC = np.array([DIRNAMES[i]+"_NOSEC" for i in range(Nqs)])

################
# WITH SECULAR #
################
RUN_PARAMS = np.column_stack((HS, JS, A0S, QS, MU1, TS, TE1, TE2, TM1,
                              TM2, E1_0, E2_0, E1DS, E2DS, ALPHA2_0,
                              NAMES, DIRNAMES, CUTOFFS, TE_FUNCS,
                              G1_0, G2_0,MUEXTS, AEXTS))
print(RUN_PARAMS)
print(f"Running {RUN_PARAMS.shape[0]} simulations...")
integrate = run.run_compmass_set_omeff(verbose=True, overwrite=overwrite,
                                 secular=True, method="RK45")
Nproc=16
N_sims = len(QS)
with Pool(processes=min(Nproc, N_sims)) as pool:
    pool.map(integrate, RUN_PARAMS)

####################
## WITHOUT SECULAR #
####################
#RUN_PARAMS = np.column_stack((HS, JS, A0S, QS, MU1, TS, TE1, TE2, TM1,
#                              TM2, E1_0, E2_0, E1DS, E2DS, ALPHA2_0,
#                              NAMES, DIRNAMES_NOSEC, CUTOFFS, TE_FUNCS,
#                              G1_0, G2_0,MUEXTS, AEXTS))
#print(f"Running {RUN_PARAMS.shape[0]} simulations...")
#integrate = run.run_compmass_set(verbose=True, overwrite=overwrite,
#                                 secular=False)
#N_sims = len(QS)
#with Pool(processes=min(8, N_sims)) as pool:
#    pool.map(integrate, RUN_PARAMS)
