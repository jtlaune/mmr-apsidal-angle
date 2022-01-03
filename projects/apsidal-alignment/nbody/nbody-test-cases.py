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

def gamma_components(eqs, eps, om_effs, Tm, T, mup, case):
    ######################
    # Initial Conditions #
    ######################
    j = 2
    e0 = 0.0
    ap = 1.0
    theta0 = 0
    g0 = 0.0
    eta0 = 1.0
    lambda0 = 0.0
    aext = 3.0
    
    i=0
    for eeq0 in eqs:
        if Tm < 0:
            ratio0 = eeq0**2*2*j
            loc = "external"
            a0 = 1.35
        else:
            ratio0 = eeq0**2*2*(j+1)
            loc = "internal"
            a0 = 0.73
        for ep in eps:
            for om_eff in om_effs:
                Te = np.abs(ratio0*Tm)
                
                label = "omeff-{:0.2e}".format(om_eff)
                dirname = "testsuite/{}/{:0.2e}/eq{:0.2e}/ep{:0.2e}/".format(loc,mup,eeq0,ep) 
                filename = os.path.join(dirname, label+".bin")

                collect = "testsuite/collect/" \
                    + case + "-gammacomps-eq{:0.2e}-ep{:0.2e}-om{:0.2e}.png" \
                    .format(eeq0, ep, om_eff)
                collectdir = os.path.dirname(collect)
                if not os.path.isdir(collectdir):
                    os.makedirs(collectdir, exist_ok=True)

                ########
                # LOAD #
                ########
                sa = rebound.SimulationArchive(filename)
                teval = np.zeros(len(sa))
                e1 = np.zeros(len(sa))
                e2 = np.zeros(len(sa))
                a1 = np.zeros(len(sa))
                a2 = np.zeros(len(sa))
                p1 = np.zeros(len(sa))
                p2 = np.zeros(len(sa))
                l1 = np.zeros(len(sa))
                l2 = np.zeros(len(sa))
                w1 = np.zeros(len(sa))
                w2 = np.zeros(len(sa))
                pw1 = np.zeros(len(sa))
                pw2 = np.zeros(len(sa))
                g1 = np.zeros(len(sa))
                g2 = np.zeros(len(sa))
                n2 = np.zeros(len(sa))
                f1 = np.zeros(len(sa))
                f2 = np.zeros(len(sa))
                for it, sim in enumerate(sa):
                    tp = sim.particles[1].calculate_orbit(primary=sim.particles[0])
                    mp = sim.particles[2].calculate_orbit(primary=sim.particles[0])

                    teval[it] = sim.t
                    e1[it] = tp.e
                    e2[it] = mp.e
                    a1[it] = tp.a
                    a2[it] = mp.a
                    p1[it] = tp.P
                    p2[it] = mp.P
                    l1[it] = tp.l
                    l2[it] = mp.l
                    w1[it] = tp.omega
                    w2[it] = mp.omega
                    pw1[it] = tp.pomega
                    pw2[it] = mp.pomega
                    g1[it] = -tp.pomega
                    g2[it] = -mp.pomega
                    n2[it] = mp.n
                    f1[it] = tp.f 
                    f2[it] = mp.f 
                del sa

                #########################
                # Variable calculations #
                #########################
                if Tm > 0:
                    alpha = a1/a2
                    thetap = ((j+1)*l2 - j*l1)%(2*np.pi)
                    Avals = A(alpha,j)
                    Bvals = B(alpha,j) 
                    Cvals = C(alpha)
                    Dvals = D(alpha) 
                    barg = np.arctan2(e1*np.sin(g1), 
                                      ((e1*np.cos(g1)
                                        +Bvals/Avals*e2))) 
                    ebar = ebarfunc(e2, e1, Avals, Bvals, g1)

                if Tm < 0:
                    alpha = a2/a1
                    thetap = (j+1)*l1 - j*l2
                    Bvals = alpha*A(alpha,j) 
                    Avals = alpha*B(alpha,j)
                    Cvals = alpha*C(alpha)
                    Dvals = alpha*D(alpha) 
                    barg = np.arctan2(e1*np.sin(g1), 
                                      ((e1*np.cos(g1)
                                        +Avals/Bvals*e2))) 
                    ebar = ebarfunc(e2, e1, Bvals, Avals, g1)

                theta = (thetap + g1)%(2*np.pi)
                newres = (thetap + barg)%(2*np.pi)
                L = np.sqrt(alpha)
                G = 0.5*L*e1*e1

                x = np.sqrt(G)*np.cos(g1)
                y = np.sqrt(G)*np.sin(g1)
                xdot = np.gradient(x, teval)
                ydot = np.gradient(y, teval)
                gammadot = (x*ydot - y*xdot)/(x**2 + y**2)
                
                ########
                # PLOT #
                ########
                fig,ax=plt.subplots(4,figsize=(6,24))
                fontsize = 24
                
                suptitle = "$\\mu_{{p}} = {:0.2e}$; $e_p={:0.2e}$;\n" \
                    "$e_{{\\rm eq}} = {:0.2e}$; $\\omega_{{\\rm eff}} = " \
                    "{:0.2e}$" \
                    .format(mup, ep, eeq0, om_eff)
                tscale = 1e5
                
                print("{}: plotting eeq{:0.2e} ep{:0.2e} omeff{:0.2e}..."
                      .format(i,eeq0,ep,om_eff))
                
                gammadot = (x*ydot - y*xdot)/(x**2 + y**2)
                first = -Avals*np.cos(theta)/np.sqrt(2*G*L)
                second = -2*Cvals/L
                third = -Dvals*ep*np.cos(g1)/np.sqrt(2*G*L)

                ax[0].scatter(teval/tscale, first, label="first", s=2)
                ax[0].scatter(teval/tscale, second, label="second", s=2)
                ax[0].scatter(teval/tscale, third, label="third", s=2)
                ax[0].scatter(teval/tscale, gammadot, label="total", s=2)
                ax[0].set_yscale("symlog", linthreshy=1e-4)
                ax[0].set_ylim((-1e3,1e3))
                ax[0].set_xlabel(r"t [10^5 yrs]")
                ax[0].legend()

                ax[1].scatter(teval, g1%(2*np.pi),s=2)
                ax[1].set_ylim((0,2*np.pi))

                ax[2].scatter(teval, theta%(2*np.pi),s=2)
                ax[2].set_ylim((0,2*np.pi))

                ax[3].scatter(teval, newres%(2*np.pi), s=2)
                ax[3].set_ylim((0,2*np.pi))

                fig.suptitle(suptitle)
                #fig.tight_layout(rect=[0, 0.03, 1, 0.9])

                fig.savefig(collect,bbox_inches="tight")
                plt.close(fig)
                i += 1


def run_tests(eqs, eps, om_effs, Tm, T, mup, case, rerun=False):
    ######################
    # Initial Conditions #
    ######################
    j = 2
    e0 = 0.0
    ap = 1.0
    theta0 = 0
    g0 = 0.0
    eta0 = 1.0
    lambda0 = 0.0
    aext = 3.0
    
    i=0
    for eeq0 in eqs:
        if Tm < 0:
            ratio0 = eeq0**2*2*j
            loc = "external"
            a0 = 1.32
        else:
            ratio0 = eeq0**2*2*(j+1)
            loc = "internal"
            a0 = 0.73
        for ep in eps:
            for om_eff in om_effs:
                Te = np.abs(ratio0*Tm)
                
                label = "omeff-{:0.2e}".format(om_eff)
                dirname = "testsuite/{}/{:0.2e}/eq{:0.2e}/ep{:0.2e}/".format(loc,mup,eeq0,ep) 
                filename = os.path.join(dirname, label+".bin")

                collect = "testsuite/collect/" \
                    + case + "-eq{:0.2e}-ep{:0.2e}-om{:0.2e}.png" \
                    .format(eeq0, ep, om_eff)
                collectdir = os.path.dirname(collect)
                if not os.path.isdir(collectdir):
                    os.makedirs(collectdir, exist_ok=True)
                
                #######
                # RUN #
                #######
                if (not os.path.exists(filename)) or rerun:
                    print("{}: running eeq{:0.2e} ep{:0.2e} omeff{:0.2e}..."
                          .format(i,eeq0,ep,om_eff))
                    if not os.path.isdir(dirname):
                        os.makedirs(dirname, exist_ok=True)
                    nbody_run.run_tp_ext_simple(ap,mup, ep, T, Te, Tm, om_eff, dirname, label)

                ########
                # LOAD #
                ########
                sa = rebound.SimulationArchive(filename)
                teval = np.zeros(len(sa))
                e1 = np.zeros(len(sa))
                e2 = np.zeros(len(sa))
                a1 = np.zeros(len(sa))
                a2 = np.zeros(len(sa))
                p1 = np.zeros(len(sa))
                p2 = np.zeros(len(sa))
                l1 = np.zeros(len(sa))
                l2 = np.zeros(len(sa))
                w1 = np.zeros(len(sa))
                w2 = np.zeros(len(sa))
                pw1 = np.zeros(len(sa))
                pw2 = np.zeros(len(sa))
                g1 = np.zeros(len(sa))
                g2 = np.zeros(len(sa))
                n2 = np.zeros(len(sa))
                f1 = np.zeros(len(sa))
                f2 = np.zeros(len(sa))
                for it, sim in enumerate(sa):
                    tp = sim.particles[1].calculate_orbit(primary=sim.particles[0])
                    mp = sim.particles[2].calculate_orbit(primary=sim.particles[0])

                    teval[it] = sim.t
                    e1[it] = tp.e
                    e2[it] = mp.e
                    a1[it] = tp.a
                    a2[it] = mp.a
                    p1[it] = tp.P
                    p2[it] = mp.P
                    l1[it] = tp.l
                    l2[it] = mp.l
                    w1[it] = tp.omega
                    w2[it] = mp.omega
                    pw1[it] = tp.pomega
                    pw2[it] = mp.pomega
                    g1[it] = -tp.pomega
                    g2[it] = -mp.pomega
                    n2[it] = mp.n
                    f1[it] = tp.f 
                    f2[it] = mp.f 
                del sa

                #########################
                # Variable calculations #
                #########################
                if Tm > 0:
                    alpha = a1/a2
                    thetap = ((j+1)*l2 - j*l1)%(2*np.pi)
                    Avals = A(alpha,j)
                    Bvals = B(alpha,j) 
                    barg = np.arctan2(e1*np.sin(g1), 
                                      ((e1*np.cos(g1)
                                        +Bvals/Avals*e2))) 
                    ebar = ebarfunc(e2, e1, Avals, Bvals, g1)

                if Tm < 0:
                    alpha = a2/a1
                    thetap = (j+1)*l1 - j*l2
                    Bvals = alpha*A(alpha,j) 
                    Avals = alpha*B(alpha,j)
                    barg = np.arctan2(e1*np.sin(g1), 
                                      ((e1*np.cos(g1)
                                        +Avals/Bvals*e2))) 
                    ebar = ebarfunc(e2, e1, Bvals, Avals, g1)

                theta = (thetap + g1)%(2*np.pi)
                newres = (thetap + barg)%(2*np.pi)
                
                ########
                # PLOT #
                ########
                fig,ax=plt.subplots(3,2,figsize=(12,18))
                fontsize = 24
                
                suptitle = "$\\mu_{{p}} = {:0.2e}$; $e_p={:0.2e}$;\n" \
                    "$e_{{\\rm eq}} = {:0.2e}$; $\\omega_{{\\rm eff}} = " \
                    "{:0.2e}$" \
                    .format(mup, ep, eeq0, om_eff)
                tscale = 1e5
                
                print("{}: plotting eeq{:0.2e} ep{:0.2e} omeff{:0.2e}..."
                      .format(i,eeq0,ep,om_eff))
                plotsim(fig, ax, teval, suptitle, tscale, fontsize,
                        (r"$\overline{\theta}$", newres, 0, 2*np.pi),
                        (r"$a_1$", a1),
                        (r"$e_1$", e1),
                        (r"$\overline{e}$", ebar),
                        (r"$\gamma-\gamma_p$", g1-g2),
                        (r"$\overline{\gamma}$", barg))

                eylim = (0, np.max(np.array([np.max(e1),np.max(ebar)]))*1.1)
                ax[1,0].set_ylim(eylim)
                ax[1,0].axhline(y=eeq0,ls="--",c="k")
                ax[1,1].set_ylim(eylim)
                ax[1,1].axhline(y=eeq0,ls="--",c="k")
                fig.savefig(collect)
                plt.close(fig)
                i += 1


def testcases():                
    eccs= np.zeros(6)
    eccs[1:] = np.logspace(np.log10(1e-2), np.log10(0.1), 5)
    eps = eccs
    eqs = eccs[1:]

        
    # may need
    # eylim = (0, np.max(np.array([np.max(e1),np.max(ebar)]))*1.1)
    ###################
    # TEST CONDITIONS #
    ###################

    #############################
    ## EQUILIBRIUM ECCENTRICITY #
    #############################
    #om_ext = [0.] # no precession
    ##eqs = [eqs[0],eqs[2],eqs[4]]
    #eqs = [eqs[2]]
    #
    ######################
    ## Circular external #
    ######################
    #Tm = -1e6
    #T = 1e5
    #mup = 1e-4
    #
    #eps = [0.]
    #run_tests(eqs, eps, om_ext, Tm, T, mup, "eq-ecc/circ-ext")

    #######################
    # EXTERNAL PRECESSION #
    #######################
    mup = 1e-4
    n = 10
    om_effs = np.zeros(n+1)
    om_effs[1:] = np.logspace(-1,1,n)*mup**(2./3.)
    #######################
    # Eccentric external  #
    #######################
    eps = [eps[-1]]
    eqs = [eqs[0]]
    #om_ext = [om_effs[1], om_effs[5],om_effs[6], 0.8,1.3]
    #om_ext = [-om_effs[6]]
    om_ext = [1e-3]
    Tm = -1e6
    T = 2.0e5
    mup = 1e-4
    
    run_tests(eqs, eps, om_ext, Tm, T, mup, "precess", rerun=False)
    gamma_components(eqs, eps, om_ext, Tm, T, mup, "precess")
    

if __name__ == "__main__":
    testcases()
