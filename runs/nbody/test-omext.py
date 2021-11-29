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
import rebound
sys.path.append("/home/jtlaune/mmr/")
import run
import plotting
from plotting import plotsim, loadsim
from helper import *
import nbody_run

mpl.rcParams.update({'font.size': 20})

class test_omext:
    def __init__(self, omext):
        self.omext = omext
    def force(self, reb_sim):
        ps = reb_sim.contents.particles
        star = ps[0]
        part = ps[1]

        r2 = part.x*part.x + part.y*part.y + part.z*part.z

        preom = self.omext*(16*np.pi*np.pi*np.pi*part.a*part.a)/(part.n*r2*r2*np.sqrt(r2))
        part.ax += part.x*preom
        part.ay += part.y*preom




# (ap, mup, ep, T, Te, Tm, om_eff, dirname, label):
# Test particle outside of massive planet migrating inwards.  This
# should work up to some inconsistencies with T_m and T_e
# definitions in literature.
######################
# Initial Conditions #
######################
T = 1000
j = 2
e0 = 0.0
ap = 1.0
theta0 = 0
g0 = 0.0
eta0 = 1.0
lambda0 = 0.0
mup = 1e-4
om_ext = 1e-3


dirname = "testsuite/test-omext/mup{:0.2e}/om{:0.2e}".format(mup,om_ext) 
label = "e{:0.2e}".format(e0)
filename = os.path.join(dirname, label+".bin")
figname = os.path.join(dirname, label+".png")
if not os.path.isdir(dirname):
    os.makedirs(dirname)

sim = rebound.Simulation()
sim.units = ('yr', 'AU', 'Msun')
Mstar = 1.

sim.add(m=Mstar)
sim.add(m=mup, e=e0, a=ap, inc=0., f=0., Omega=0., omega=0)
sim.move_to_com()

numpts = 1000

orbits = sim.calculate_orbits()
part = orbits[0]
migforce = test_omext(om_ext)
print("om_ext = {:0.2e}".format(migforce.omext))
sim.additional_forces = migforce.force
#sim.force_is_velocity_dependent = 1 # need rebound to update velocities

#######
# RUN #
#######
rerun = False
if (not os.path.exists(filename)) or rerun:
    sim.automateSimulationArchive(filename, interval=T/numpts, deletefile=True)
    print("Saving to directory... {}".format(dirname))
    numstatus = 100
    for i in range(numstatus):
        sim.integrate(sim.t+T/numstatus)
        print("{:0.0f}%".format(i/numstatus*100), end="\r")

########
# LOAD #
########
sa = rebound.SimulationArchive(filename)
teval = np.zeros(len(sa))
e1 = np.zeros(len(sa))
a1 = np.zeros(len(sa))
p1 = np.zeros(len(sa))
l1 = np.zeros(len(sa))
w1 = np.zeros(len(sa))
pw1 = np.zeros(len(sa))
g1 = np.zeros(len(sa))
f1 = np.zeros(len(sa))
for it, sim in enumerate(sa):
    part = sim.particles[1].calculate_orbit(primary=sim.particles[0])
    teval[it] = sim.t
    e1[it] = part.e
    a1[it] = part.a
    p1[it] = part.P
    l1[it] = part.l
    w1[it] = part.omega
    pw1[it] = part.pomega
    g1[it] = -part.pomega
    f1[it] = part.f 
del sa

########
# PLOT #
########

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(teval, g1%(2*np.pi), s=2, label="nbody")
testom = (teval*om_ext*4*np.pi)%(2*np.pi)
ax.plot(teval, testom, "k--", label="predicted")
ax.set_xlabel("orbits")
ax.set_ylabel(r"$\gamma$")
ax.legend(loc="lower right")
fig.savefig(figname, bbox_inches="tight")

