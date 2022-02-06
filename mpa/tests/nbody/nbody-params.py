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

mpl.rcParams.update({"font.size": 20, "figure.facecolor": "white"})

#################
# CONFIGURATION #
#################
j = 2
a0 = 1.0
h = 0.03
alpha_0 = (j / (j + 1)) ** (2.0 / 3.0)
Nqs = 16
qs = np.ones(Nqs) * 1. # external
totmass = 1e-4
Tw0 = 1000
TeRatios = qs
ntps = 100.

######################
# Varying parameters #
######################
E1_0 = np.ones(Nqs) * 0.01
E2_0 = np.ones(Nqs) * 0.01
G1_0 = np.array([np.random.uniform(0, 2 * np.pi) for i in range(Nqs)])
G2_0 = np.array([np.random.uniform(0, 2 * np.pi) for i in range(Nqs)])
HS = np.ones(Nqs) * h
JS = np.ones(Nqs) * j
A0S = np.ones(Nqs) * a0
NTPS = np.ones(Nqs) * ntps
QS = qs
MUTOT = totmass*np.ones(Nqs)
TE = Tw0*np.ones(Nqs)
TM = TE / 3.46 / HS**2 * (-1 * (qs < 1) + 1 * (qs >= 1))
# TM1 = TE1/3.46/HS**2*(-1*(qs<1) + 1*(qs>=1))
TS = 10.*np.ones(Nqs) # 0.01 * np.maximum(TE1, TE2)
#############################################################
# BUG: SETTING CUTOFF TO T RESULTS IN DIFFERENCES BETWEEN T #
# VALUES. LIKELY A FACTOR OF 2PI THING.                     #
#############################################################
cutoff_frac = 1.0
CUTOFFS = TS * cutoff_frac
ALPHA2_0 = (1.85) ** (2.0 / 3) * np.ones(Nqs)
NAMES = np.array(
    [
        f"alpha2_0-{ALPHA2_0[i]}"
        for i, qit in enumerate(QS)
    ]
)

DIRNAMES = np.array(
    [f"q{QS[i]:0.1f}/h-{h:0.2f}" f"-Tw0-{Tw0}-mutot-{totmass:0.1e}" for i in range(Nqs)]
)

DIRNAMES_NOSEC = np.array([DIRNAMES[i] + "_NOSEC" for i in range(Nqs)])

################
# WITH SECULAR #
################
RUN_PARAMS = np.column_stack(
    (
        HS,
        QS,
        MUTOT,
        A0S,
        ALPHA2_0,
        TS,
        TE,
        TM,
        E1_0,
        E2_0,
        NAMES,
        DIRNAMES,
        CUTOFFS,
        G1_0,
        G2_0,
        NTPS,
    )
)
