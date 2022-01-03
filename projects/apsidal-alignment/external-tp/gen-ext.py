from multiprocessing import Pool, TimeoutError
import numpy as np
import scipy as sp
from scipy import optimize
from IPython.display import display, clear_output
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
import importlib

sys.path.append("/home/jtlaune/multi-planet-architecture/")
mpl.rcParams.update({'font.size': 20})
import run

#official runs: nruns = 17
nruns = 9

# official runs: H = 0.03
H   = np.ones(nruns)*0.075
J   = np.ones(nruns)*2.
MUP = np.ones(nruns)*1e-4
AP  = np.ones(nruns)
# official runs: A0  = np.ones(nruns)*(1.6)**(2./3)
A0  = np.ones(nruns)*(1.6)**(2./3)
EP = np.zeros(nruns)
# official runs: EP[1:]  = np.logspace(-3,-1,nruns-1)
EP[1:]  = np.logspace(-1,np.log10(0.2),nruns-1)
E0  = np.ones(nruns)*0.001
G0  = np.random.randn(nruns)*np.pi*2
TE  = np.ones(nruns)*1000.
TM  = TE/3.46/H**2*(-1*(A0>AP)+1*(AP>A0))
#final runs: T   = 150.*TE
T = 15.*TE
tploc = np.array(["ext" for i in range(nruns)])
tploc[A0<AP] = "int"
print(tploc)
DIRNAME    = np.array([f"h-{H[i]:0.3f}-{tploc[i]}"
                       for i in range(nruns)])
FILENAME   = np.array([f"ep-{EP[i]:0.4f}.npz" for i in range(nruns)])
FIGNAME    = np.array([f"ep-{EP[i]:0.4f}.png" for i in range(nruns)])
PARAMSNAME = np.array([f"parms-ep-{EP[i]:0.4f}.txt"
                       for i in range(nruns)])
TSCALE     = np.ones(nruns)*1e3
TOL        = np.ones(nruns)*1e-9
OVERWRITE  = np.ones(nruns)>0

#run.run_tp(h, j, mup, ap, a0, ep, e0, g0, Tm, Te, T, suptitle, dirname,
#           filename, figname, paramsname, tscale, overwrite=overwrite)
RUN_PARAMS = np.column_stack((H, J, MUP, AP, A0, EP, E0, G0, TM, TE,
                              T, DIRNAME, FILENAME, FIGNAME,
                              PARAMSNAME, TSCALE, TOL, OVERWRITE))
print(f"Running {RUN_PARAMS.shape[0]} simulations...")

Nproc = 8

with Pool(processes=min(Nproc, nruns)) as pool:
    pool.map(run.run_tp_set, RUN_PARAMS)
