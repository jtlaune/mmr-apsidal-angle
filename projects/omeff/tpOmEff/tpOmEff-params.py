import subprocess
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
chunk = 16
Nqs = 64
eps = [0.0, 0.001, 0.03, 0.1]
qs = np.ones(Nqs) *  1.1 # test particle outside
dirn = "lastrun"
totmass = 1e-4
Tw0 = 1000

######################
# Varying parameters #
######################
E1DS = np.ones(Nqs) * 0.0
E2DS = np.ones(Nqs) * 0.0

E2_0 = np.ones(Nqs) * 0.001
E1_0 = np.ones(Nqs)
for jit in range(4):
    E1_0[jit*chunk:(jit+1)*chunk] = eps[jit]*np.ones(chunk)

DIRNAMES = np.array(
    [f"{dirn}/Tw0{Tw0}/ep{E1_0[i]:0.3f}/" for i in range(Nqs)]
)

# eccs = np.array([0.1])
# E1_0, E2_0 = np.meshgrid(eccs, eccs)
# E1_0 = np.flip(E1_0.flatten())
# E2_0 = np.flip(E2_0.flatten())

####################
# THREADING ARRAYS #
####################
G1_0 = np.array([np.random.uniform(0, 2 * np.pi) for i in range(Nqs)])
G2_0 = np.array([np.random.uniform(0, 2 * np.pi) for i in range(Nqs)])
HS = np.ones(Nqs)*h
JS = np.ones(Nqs) * j
A0S = np.ones(Nqs) * a0
QS = qs
MUP = totmass*np.ones(Nqs)

##########################################################################
# Dissipative timescales                                                 #
##########################################################################
TE1 = np.ones(Nqs)
TE2 = np.ones(Nqs)
TM1 = np.ones(Nqs)
TM2 = np.ones(Nqs)

# Tms are opposite direction of the comparable mass case
# internal
# tp
TE1[qs < 1] = Tw0*TE1[qs < 1]
TM1[qs < 1] = TE1[qs < 1] / HS[qs < 1]**2
# mup, code should not take these
TE2[qs < 1] = TE2[qs < 1]*0.
TM2[qs < 1] = TM2[qs < 1]*0.
# external
# tp
TE2[qs > 1] = Tw0*TE2[qs > 1]
TM2[qs > 1] = -TE2[qs > 1] / HS[qs > 1]**2
# mup, code should not take these
TE1[qs > 1] = TE1[qs > 1]*0.
TM1[qs > 1] = TM1[qs > 1]*0.

#############################################################
# BUG: SETTING CUTOFF TO T RESULTS IN DIFFERENCES BETWEEN T #
# VALUES. LIKELY A FACTOR OF 2PI THING.                     #
#############################################################
cutoff_frac = 1.0
TS = 3e5 * np.ones(Nqs)
ALPHA_0 = alpha_0 * np.ones(Nqs)
CUTOFFS = TS * cutoff_frac
ALPHA2_0 = (1.65) ** (2.0 / 3) * np.ones(Nqs)


##########
# OMEFFS #
##########
OMEFFS2 = np.ones(Nqs)
for jit in range(int(Nqs/chunk)):
    OMEFFS2[jit*chunk:(jit+1)*chunk] = -np.logspace(-4, -2, chunk)

OMEFFS1 = np.zeros(Nqs)

NAMES = np.array(
    [
        f"{str(i).zfill(4)}-mup{MUP[i]:0.2e}-omeff{OMEFFS1[i]:0.2e}"
        for i, qit in enumerate(QS)
    ]
)
print(NAMES)

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
        MUP,
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
        OMEFFS1,
        OMEFFS2,
    )
)
