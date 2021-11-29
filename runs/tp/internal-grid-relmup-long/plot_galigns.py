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
Tm = 1e6
tol = 1e-6
filename = "behaviors.txt"

if not os.path.isdir("images/"):
    os.mkdir("images/")


galigns = np.loadtxt("behaviors.txt", skiprows=1, delimiter=",")
ep = galigns[:,0]
eeq = galigns[:,1]
aligns = galigns[:,3]
inres = galigns[:,4]
fig, ax = plt.subplots()
checkres = -1*(inres < 1)+inres
print(ep, eeq, aligns,checkres)
ax.scatter(ep*checkres, eeq*checkres, c=aligns)
#aligns[checkres.astype(int)] = 2
ax.scatter(ep, eeq, c=aligns)
ax.set_xlim((0,0.105))
ax.set_ylim((0,0.105))
fig.savefig("images/galigns.png", bbox_inches="tight")
