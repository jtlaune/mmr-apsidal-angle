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
T = 2e5
Tm = -1e6
tol = 1e-6

if not os.path.isdir("images/"):
    os.mkdir("images/")

filename = "behaviors.txt"
with open(filename, "w") as f:
    f.write("ep, eeq, mup, aligned\n")
    pass

for ep in eps:
    for ir, ratio in enumerate(ratios):
        i+=1
        Te = -ratio*Tm
        label = "{:0.2e}-{:0.2e}".format(ep, eeqs[ir])

        print("{:0.1f}%: {}"
              .format(100*i/(N*(N-1)), label),
              end="\r")

        sim = loadsim(label+".npz", ap, j, ep)
        
        ifrac = int(0.75*len(sim["teval"]))
        bartheta = sim["bartheta"][ifrac:]
        g = sim["g"][ifrac:]

        inres = 1*np.all(np.abs(bartheta-np.pi) < 0.3*np.pi)
        aligned = 1*np.all(np.abs(g)<0.75*np.pi)

        with open(filename, "a") as f:
            datastr = ", ".join([str(pt) for pt in [ep, eeqs[ir], mup, aligned, inres]])
            f.write(datastr+"\n")
