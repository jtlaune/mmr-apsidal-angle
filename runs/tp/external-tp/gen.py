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

h=0.03
j=2
mup=1e-3
a0=0.7
ap = 1.
tploc = "int"
ep=0.1
e0=0.001
g0=np.random.randn()*2*np.pi
Te=1000
Tm =Te/3.46/h**2*(-1*(tploc=="ext")+1*(tploc=="int"))
T=1*Te
suptitle="test"
dirname="./"
filename="test.npz"
figname="test.png"
paramsname="params-test.txt"
tscale=1e3
overwrite=False

run.run_tp(h, j, mup, ap, a0, ep, e0, g0, Tm, Te, T, suptitle, dirname,
           filename, figname, paramsname, tscale, overwrite=overwrite)
        
