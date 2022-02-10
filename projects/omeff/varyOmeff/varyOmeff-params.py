import subprocess
import mpa.fndefs as fns
import numpy as np
from numpy import sqrt, pi, cos, sin, abs
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os

_, RUNSNAME = os.path.split(os.getcwd())
import importlib

sys.path.append("/home/jtlaune/multi-planet-architecture/mpa/")
import mpa
import mpa.fndefs as fns


#################
# CONFIGURATION #
#################
j = 2
a0 = 1.0
h = 0.03
alpha_0 = (j / (j + 1)) ** (2.0 / 3.0)
Nqs = 33
qs = np.ones(Nqs) * 0.5
totmass = 1e-3
Tw0 = 1000
TeRatios = qs

######################
# Varying parameters #
######################
E1_0 = np.ones(Nqs) * 0.01
E2_0 = np.ones(Nqs) * 0.01
E1DS = np.ones(Nqs) * 0.0
E2DS = np.ones(Nqs) * 0.0


# eccs = np.array([0.1])
# E1_0, E2_0 = np.meshgrid(eccs, eccs)
# E1_0 = np.flip(E1_0.flatten())
# E2_0 = np.flip(E2_0.flatten())

####################
# THREADING ARRAYS #
####################
G1_0 = np.array([np.random.uniform(0, 2 * np.pi) for i in range(Nqs)])
G2_0 = np.array([np.random.uniform(0, 2 * np.pi) for i in range(Nqs)])
HS = np.ones(Nqs) * h
JS = np.ones(Nqs) * j
A0S = np.ones(Nqs) * a0
QS = qs
MU2 = totmass / (1 + QS)
MU1 = totmass - MU2
TE1 = Tw0 / sqrt(TeRatios)
TE2 = Tw0 * sqrt(TeRatios)
TM1 = TE1 / 3.46 / HS**2 * (-1 * (qs < 1) + 1 * (qs >= 1))
# TM1 = TE1/3.46/HS**2*(-1*(qs<1) + 1*(qs>=1))
TM2 = TE2 / 3.46 / HS**2 * (-1 * (qs < 1) + 1 * (qs >= 1))
TS = np.ones(Nqs) * 1e5
ALPHA_0 = alpha_0 * np.ones(Nqs)
#############################################################
# BUG: SETTING CUTOFF TO T RESULTS IN DIFFERENCES BETWEEN T #
# VALUES. LIKELY A FACTOR OF 2PI THING.                     #
#############################################################
cutoff_frac = 1.0
CUTOFFS = TS * cutoff_frac
ALPHA2_0 = (1.65) ** (2.0 / 3) * np.ones(Nqs)

##########
# OMEFFS #
##########
OMEFFS = np.zeros(Nqs)
OMEFFS[1:17] = np.logspace(-9, -5, 16)
OMEFFS[17:] = np.logspace(-5.5, -2, Nqs-17)
# 
OMEFFS = -OMEFFS

NAMES = np.array([f"{i:03d}-omeff0-{OMEFFS[i]:0.3e}" for i, qit in enumerate(QS)])

DIRNAMES = np.array(
    [f"q{QS[i]:0.2f}/h-{h:0.2f}" f"-Tw0-{Tw0}-mutot-{totmass:0.1e}" for i in range(Nqs)]
)

DIRNAMES_NOSEC = np.array([DIRNAMES[i] + "_NOSEC" for i in range(Nqs)])

################
# WITH SECULAR #
################
RUN_PARAMS = np.column_stack(
    (
        HS,
        JS,
        A0S,
        QS,
        MU1,
        TS,
        TE1,
        TE2,
        TM1,
        TM2,
        E1_0,
        E2_0,
        E1DS,
        E2DS,
        ALPHA2_0,
        NAMES,
        DIRNAMES,
        CUTOFFS,
        G1_0,
        G2_0,
        OMEFFS,
    )
)
