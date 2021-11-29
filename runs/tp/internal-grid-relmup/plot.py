#! /home/jtlaune/.pythonvenvs/science/bin/python
import numpy as np
import scipy as sp
from scipy import optimize
from IPython.display import display, clear_output
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
import importlib

sys.path.append("/home/jtlaune/mmr/")
mpl.rcParams.update({'font.size': 20})
import plotting
from plotting import *
import mmr
import run

# params 
from params import *

i = 0
mup = 1e-3
T = 7.5e4
Tm = 1e6
tol = 1e-6

if not os.path.isdir("images/"):
    os.mkdir("images/")

for ep in eps:
    for ir, ratio in enumerate(ratios):
        i+=1
        Te = ratio*Tm
        stable= j/(np.sqrt(3)*(j+1)**1.5*0.8*j)*(Te/(Tm/1.5))**1.5
        mup = 1.5*stable
        label = "{:0.2e}-{:0.2e}-{:0.2e}".format(ep, eeqs[ir], mup)

        print("{:0.1f}%: {}"
              .format(100*i/(N*(N-1)), label),
              end="\r")

        if os.path.exists(label+".npz"):
            sim = loadsim(label+".npz", ap, j, ep)
            fig, ax = plt.subplots(3, figsize=(6,18))
            plotsim(fig, ax, sim["teval"],
                    "ep={:0.2f} ed={:0.2f}".format(ep, eeqs[ir]), 1e4, 24,
                    (r"$\overline{\theta}$", sim["bartheta"]),
                    (r"$e_1$", sim["e1"]),
                    (r"$\gamma$", sim["g"]))
            fig.savefig("images/"+label+".png", bbox_inches="tight")
            plt.close(fig)
