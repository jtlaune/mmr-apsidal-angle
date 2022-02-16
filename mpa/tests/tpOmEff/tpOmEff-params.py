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

mpl.rcParams.update({"font.size": 20, "figure.facecolor": "white"})

#################
# CONFIGURATION #
#################
j = 2
a0 = 1.0
#h = 0.03
alpha_0 = (j / (j + 1)) ** (2.0 / 3.0)
Nqs = 8
qs = np.ones(Nqs) * 0.001  # test particle outside
totmass = 1e-4
Tw0 = 1000

######################
# Varying parameters #
######################
E1_0 = np.ones(Nqs) * 0.0
E2_0 = np.ones(Nqs) * 0.001
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
HS = np.linspace(0.01, 0.09, Nqs)
JS = np.ones(Nqs) * j
A0S = np.ones(Nqs) * a0
QS = qs
MU2 = totmass / (1 + QS)
MU1 = totmass - MU2

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
TE1[qs < 1] = Tw0 * TE1[qs < 1]
TM1[qs < 1] = TE1[qs < 1] / 3.46 / HS[qs < 1]**2 
# mup, code should not take these
TE2[qs < 1] = TE2[qs < 1]*0.
TM2[qs < 1] = TM2[qs < 1]*0.
# external
# tp
TE2[qs > 1] = Tw0 * TE2[qs > 1]
TM2[qs > 1] = -TE2[qs > 1] / 3.46 / HS[qs > 1]**2
# mup, code should not take these
TE1[qs > 1] = TE1[qs > 1]*0.
TM1[qs > 1] = TM1[qs > 1]*0.

#############################################################
# BUG: SETTING CUTOFF TO T RESULTS IN DIFFERENCES BETWEEN T #
# VALUES. LIKELY A FACTOR OF 2PI THING.                     #
#############################################################
cutoff_frac = 1.0
TS = 1e4 * np.ones(Nqs)  # 0.01 * np.maximum(TE1, TE2)
ALPHA_0 = alpha_0 * np.ones(Nqs)
CUTOFFS = TS * cutoff_frac
ALPHA2_0 = (1.55) ** (2.0 / 3) * np.ones(Nqs)

# def muext(omeff, aext):
def omeffs(q, a0, j, muext, aext):
    alpha = (j / (j + 1)) ** (2.0 / 3)
    a1 = alpha * a0
    a2 = a0
    om1 = q * fns.om1ext_n2(muext, a1, a2, aext)
    om2 = fns.om2ext_n2(muext, a2, aext)
    return om1 - om2


##########
# OMEFFS #
##########
OMEFFS1 = np.linspace(1e-2, 1e-1, Nqs)
OMEFFS2 = np.zeros(Nqs)

NAMES = np.array(
    [
        f"omeff-{OMEFFS1[i]:0.3e}" f"-e1d-{E1DS[i]:0.3f}-e2d-{E2DS[i]:0.3f}"
        for i, qit in enumerate(QS)
    ]
)

DIRNAMES = np.array(
    [f"q{QS[i]:0.1f}/"+f"Tw0-{Tw0}-mup-{totmass:0.1e}" for i in range(Nqs)]
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
        OMEFFS1,
        OMEFFS2,
    )
)
