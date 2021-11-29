#! /home/jtlaune/.pythonvenvs/science/bin/python
import numpy as np
import sys

sys.path.append("/home/jtlaune/mmr/")
import run

# params
from tp_runs.params import *

i = 0
mup = 1e-3
Tm = -1e6
tol = 1e-6

# (ep, ed)
points = ((eccs[-1], eccs[1]), (eccs[1], eccs[-1]))
T = 1e6

for pt in points:
    i += 1
    ep = pt[0]
    eeq = pt[1]
    ratio = eeq ** 2 * 2 * j
    Te = -ratio * Tm
    label = "{:0.2e}-{:0.2e}".format(ep, eeq)
    print("running {}".format(label))
    sim = run.tp_intH(j, mup, ep, e0, ap, g0, a0, lambda0)
    (
        teval,
        thetap,
        newresin,
        newresout,
        eta,
        a1,
        e1,
        k,
        kc,
        alpha0,
        alpha,
        g,
        L,
        G,
        ebar,
        barg,
        x,
        y,
    ) = sim.int_Hsec(0, T, tol, Tm, Te)
    np.savez(label + ".npz", teval=teval, thetap=thetap, L=L, g=g, G=G, x=x, y=y)
