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

qs = [0.5, 1., 2.]
h = 0.03

for q in qs:
    #################
    # CONFIGURATION #
    #################
    j = 2
    a0 = 1.0
    alpha_0 = (j/(j+1))**(2./3.)
    overwrite = True
    totmass = 1e-3
    
    ######################
    # Varying parameters #
    ######################
    Tw0 = 1000.
    rats = np.array([0.25, 0.5, 0.75, 1.5, 2.5, 5, 10])
    TeRatios = np.sqrt(rats)
    Nqs = len(TeRatios)
    qs = np.ones(Nqs)*q
    
    ####################
    # THREADING ARRAYS #
    ####################
    TE_FUNCS = np.zeros(Nqs)
    G1_0 = np.array([np.random.uniform(0, 2*np.pi) for i in range(Nqs)])
    G2_0 = np.array([np.random.uniform(0, 2*np.pi) for i in range(Nqs)])
    HS = np.ones(Nqs)*h
    JS = np.ones(Nqs)*j
    A0S = np.ones(Nqs)*a0
    QS = np.array(qs)
    MU2 = totmass/(1+QS)
    MU1 = totmass - MU2
    ALPHA_0 = alpha_0*np.ones(Nqs)
    TE1 = Tw0*TeRatios
    TE2 = Tw0/TeRatios
    TM1 = TE1/3.46/HS**2*(1*(TeRatios<1) - 1*(TeRatios>=1))
    TM2 = TE2/3.46/HS**2*(1*(TeRatios<1) - 1*(TeRatios>=1))

    TS = 65.*np.maximum(TE1, TE2)
    #TS = 8.*np.maximum(TE1, TE2)
    #TS[0] = 5*TS[0]

    E1_0 = np.ones(Nqs)*0.1/sqrt(QS)
    E2_0 = np.ones(Nqs)*0.1*sqrt(QS)
    E1DS = np.zeros(Nqs)
    E2DS = np.zeros(Nqs)
    CUTOFFS = TS
    ALPHA2_0 = (1.6)**(2./3)*(1+E2_0**2+E1_0**2)
    NAMES = np.array([f"ratio-{rats[i]}" for i in range(len(QS))])
    
    DIRNAMES = np.array([f"./varyTe-q{QS[i]}-h-{h}-Tw0-{int(Tw0)}" for i in range(Nqs)])
    print(DIRNAMES)

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
    
    N_sims = RUN_PARAMS.shape[0]
    with Pool(processes=min(8, N_sims)) as pool:
        pool.map(integrate, RUN_PARAMS)
