import numpy as np
import scipy as sp
from scipy import optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
import importlib
import run

sys.path.append("/home/jtlaune/mmr/")

def loadsim(filename, ap, j, ep):
    sim = np.load(filename)
    
    thetap = sim["thetap"]
    L = sim["L"]
    g = sim["g"]
    G = sim["G"]
    x = sim["x"]
    y = sim["y"]
    teval = sim["teval"]

    p0 = ((j+1)/j)
    a1 = L*L*ap
    e1 = np.sqrt(2*G/L)

    n1 = a1**(2./3.)
    n_p = ap**(2./3.)
    Pratio = n1/n_p 

    if a1[0] > ap:
        alpha = 1./(L*L)
    elif a1[0] < ap:
        alpha = (L*L)

    A = alpha*run.B(alpha,j)
    B = alpha*run.A(alpha,j)
    C = alpha*run.C(alpha)
    D = alpha*run.D(alpha)

    barg = np.arctan2(e1*np.sin(g), 
                                 ((e1*np.cos(g)
                                   +B/A*ep))) 
    barg = (barg) % (2*np.pi)
    ebar = np.sqrt(e1**2 + 2*B/A*ep*e1*np.cos(g) + B**2/A**2*ep**2)
    theta = (thetap + g) % (2*np.pi)
    bartheta = (thetap + barg) % (2*np.pi)

    datadict = {"teval":    teval,
                "thetap":   thetap,
                "theta":    theta,
                "bartheta": bartheta,
                "a1":       a1,
                "Pratio":   Pratio,
                "e1":       e1,
                "ebar":     ebar,
                "g":        g,
                "barg":     barg,
                "L":        L,
                "G":        G,
                "x":        x,
                "y":        y,
                "A":        A,
                "B":        B,
                "C":        C,
                "D":        D}        

    return(datadict)

def plotsim(fig, axes, teval, suptitle, tscale, fontsize, *argv, xlabel=True, yfigupper=0.9):
    for i, value in enumerate(argv):
        ax = axes.flatten()[i]

        if len(value) == 4:
            ylim0 = value[2]
            ylim1 = value[3]
            ax.set_ylim((ylim0,ylim1))

        varname = value[0]
        data = value[1]

        ax.scatter(teval/tscale, data, s=2, c="k", alpha=0.15)
        if xlabel:
            ax.set_xlabel("t [{:0.1e} orbits]".format(tscale), fontsize=fontsize)
        ax.set_xlim((teval[0]/tscale, teval[-1]/tscale))
        ax.tick_params(labelsize=fontsize)

        ax.set_title(varname, fontsize=fontsize)

    fig.suptitle(suptitle, fontsize=fontsize)
    fig.tight_layout(rect=[0, 0.03, 1, yfigupper])


def plotphase(fig, axes, teval, suptitle, tscale, fontsize, *argv):
    N = len(argv)
    for i, value in enumerate(argv):
        ax = axes.flatten()[i]

        xlabel = value[0]
        ylabel = value[1]
        q = value[2]
        p = value[3]

        cs = ax.scatter(p*np.cos(q),p*np.sin(q), c=teval/tscale, s=2)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        if i == N-1:
            box = ax.get_position()
            ax.set_position([box.x0*1.05, box.y0, box.width, box.height])
            # create color bar
            axColor = plt.axes([box.x0*1.05 + box.width * 1.05, box.y0, 0.01, box.height])
            plt.colorbar(cs, cax = axColor, orientation="vertical", label="t [{:0.1e} orbits]".format(tscale), fontsize=fontsize)

    fig.suptitle(suptitle, fontsize=fontsize)
    fig.tight_layout(rect=[0, 0.03, 1, 0.9])

