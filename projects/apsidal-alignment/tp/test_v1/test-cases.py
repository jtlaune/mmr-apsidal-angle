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
import run
sys.path.append("/home/jtlaune/mmr/")
import plotting
from plotting import plotsim, loadsim
from helper import *

mpl.rcParams.update({'font.size': 20})

def run_tests(eqs, eps, om_effs, Tm, T, mup, case):
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
                figname = dirname + label+".png"
                filename = dirname + label+".npz"
                collect = "testsuite/collect/" + case + "-eq{:0.2e}-ep{:0.2e}-om{:0.2e}.png".format(eeq0, ep, om_eff)
                collectdir = os.path.dirname(collect)
                if not os.path.isdir(collectdir):
                    os.makedirs(collectdir, exist_ok=True)
                
                if os.path.exists(filename):
                    #data = loadsim(filename, ap, j, ep)
                    #teval     = data["teval"]
                    #newresout = data["bartheta"]
                    #a1        = data["a1"]
                    #e1        = data["e1"]
                    #ebar      = data["ebar"]
                    #g         = data["g"] %(2*np.pi)
                    #barg      = data["barg"]
                    pass
                    
                else:
                    print("{}: running eeq{:0.2e} ep{:0.2e} omeff{:0.2e}..."
                          .format(i,eeq0,ep,om_eff))
                    sim = run.tp_intH(j, mup, ep, e0, ap, g0, a0, lambda0)
                    (teval, thetap, newresin, newresout, eta, a1, e1, k, kc,
                     alpha0, alpha, g, L, G, ebar, barg, x,y) = sim.int_Hsec(0, T, 1e-6, Tm,
                                                                             Te, om_eff, aext)
                    if not os.path.isdir(dirname):
                        os.makedirs(dirname, exist_ok=True)

                    np.savez(filename,  teval=teval, thetap=thetap, L=L, g=g, G=G, x=x,y=y)
                        
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
                            (r"$\overline{\theta}$", newresout, 0, 2*np.pi),
                            (r"$a_1$", a1),
                            (r"$e_1$", e1),
                            (r"$\overline{e}$", ebar),
                            (r"$\gamma$", g),
                            (r"$\overline{\gamma}$", barg))

                    eylim = (0, np.max(np.array([np.max(e1),np.max(ebar)]))*1.1)
                    ax[1,0].set_ylim(eylim)
                    ax[1,0].axhline(y=eeq0,ls="--",c="k")
                    ax[1,1].set_ylim(eylim)
                    ax[1,1].axhline(y=eeq0,ls="--",c="k")
                    fig.savefig(figname)
                    fig.savefig(collect)
                    plt.close(fig)
                i += 1


def testcases():                
    eccs= np.zeros(6)
    eccs[1:] = np.logspace(np.log10(1e-2), np.log10(0.1), 5)
    eps = eccs
    eqs = eccs[1:]
    
    n = 10
    om_effs = np.zeros(n+1)
    
    # may need
    # eylim = (0, np.max(np.array([np.max(e1),np.max(ebar)]))*1.1)
    # om_effs[1:] = np.logspace(-1,1,n)*mup**(2./3.)
    ###################
    # TEST CONDITIONS #
    ###################
    om_effs = [0.] # no precession

    ############################
    # EQUILIBRIUM ECCENTRICITY #
    ############################
    eqs = [eqs[0],eqs[2],eqs[4]]
    
    #####################
    # Circular internal #
    #####################
    Tm = 1e6
    T = 2e5
    mup = 1e-4
    
    eps = [0.]
    run_tests(eqs, eps, om_effs, Tm, T, mup, "eq-ecc/circ-int")

    #####################
    # Circular external #
    #####################
    Tm = -1e6
    T = 2e5
    mup = 1e-4
    
    eps = [0.]
    run_tests(eqs, eps, om_effs, Tm, T, mup, "eq-ecc/circ-ext")

    #####################
    # Small ep internal #
    #####################
    Tm = 1e6
    T = 2e5
    mup = 1e-4
    
    eps = [0.01]
    run_tests(eqs, eps, om_effs, Tm, T, mup, "eq-ecc/smallecc-int")

    #####################
    # Small ep external #
    #####################
    Tm = -1e6
    T = 2e5
    mup = 1e-4
    
    eps = [0.01]
    run_tests(eqs, eps, om_effs, Tm, T, mup, "eq-ecc/smallecc-ext")

    #################
    # STABILITY MUP #
    #################
    eq0 = 0.05
    j = 2

    Tm = 5e5
    T = 5e5

    # internal and external capture condition: mu_cap^(4/3) >> 1/Tm
    mu_cap = (1/Tm)**(3./4.)

    ######################
    ## Circular internal #
    ######################

    # internal capture conditions (approximates):
    # particle overdamps and escapes mup < mu_esc
    # particle finite librates mu_esc < mup < mu_lib
    # particle reaches an equilibrium w/ no lib mup > mu_lib
    alpha = (j/(j+1))**(2./3.)
    A = run.A(alpha,j)
    mu_esc = np.abs(eq0**3*(3*j**2/(8*alpha*A)))
    mu_lib = np.abs(eq0**3*(3*j/(alpha*A)))
    print(A,alpha)
    print(mu_cap,mu_esc, mu_lib)

    # no capture
    mup = 0.01*mu_cap
    eps = [0.0]
    eqs = [eq0]
    run_tests(eqs, eps, om_effs, Tm, T, mup, "stability/circ-int/nocap")

    # escape
    mup = log_mean(mu_esc,mu_cap)
    print("escape case", mup)
    eps = [0.0]
    eqs = [eq0]
    run_tests(eqs, eps, om_effs, Tm, T, mup, "stability/circ-int/escape")

    # finite libration
    mup = log_mean(mu_esc,mu_lib)
    print("lib case", mup)
    eps = [0.0]
    eqs = [eq0]
    run_tests(eqs, eps, om_effs, Tm, T, mup, "stability/circ-int/finlib")

    # no libration equilibrium
    mup = 2*mu_lib
    print("eq case", mup)
    eps = [0.0]
    eqs = [eq0]
    run_tests(eqs, eps, om_effs, Tm, T, mup, "stability/circ-int/stable")
                     
    #####################
    # Circular external #
    #####################
    Tm = -Tm

    # external capture conditions (approximates):
    # mup << mu_cap no capture
    # mup >> mu_cap capture, always stable
    print(mu_cap)

    # no capture
    mup = 0.01*mu_cap
    eps = [0.0]
    eqs = [eq0]
    run_tests(eqs, eps, om_effs, Tm, T, mup, "stability/circ-ext/nocap")

    # capture
    mup = 100*mu_cap
    eps = [0.0]
    eqs = [eq0]
    run_tests(eqs, eps, om_effs, Tm, T, mup, "stability/circ-ext/cap")

    ######################
    ## Smallecc internal #
    ######################
    Tm = np.abs(Tm)

    # internal capture conditions (approximates):
    # particle overdamps and escapes mup < mu_esc
    # particle finite librates mu_esc < mup < mu_lib
    # particle reaches an equilibrium w/ no lib mup > mu_lib
    alpha = (j/(j+1))**(2./3.)
    A = run.A(alpha,j)
    mu_esc = np.abs(eq0**3*(3*j**2/(8*alpha*A)))
    mu_lib = np.abs(eq0**3*(3*j/(alpha*A)))
    print(A,alpha)
    print(mu_cap,mu_esc, mu_lib)

    # no capture
    mup = 0.01*mu_cap
    eps = [0.01]
    eqs = [eq0]
    run_tests(eqs, eps, om_effs, Tm, T, mup, "stability/smallecc-int/nocap")

    # escape
    mup = log_mean(mu_esc,mu_cap)
    print("escape case", mup)
    eps = [0.01]
    eqs = [eq0]
    run_tests(eqs, eps, om_effs, Tm, T, mup, "stability/smallecc-int/escape")

    # finite libration
    mup = log_mean(mu_esc,mu_lib)
    print("lib case", mup)
    eps = [0.01]
    eqs = [eq0]
    run_tests(eqs, eps, om_effs, Tm, T, mup, "stability/smallecc-int/finlib")

    # no libration equilibrium
    mup = 2*mu_lib
    print("eq case", mup)
    eps = [0.01]
    eqs = [eq0]
    run_tests(eqs, eps, om_effs, Tm, T, mup, "stability/smallecc-int/stable")

    #####################
    # Smallecc external #
    #####################
    Tm = -Tm

    # external capture conditions (approximates):
    # mup << mu_cap no capture
    # mup >> mu_cap capture, always stable
    print(mu_cap)

    # no capture
    mup = 0.01*mu_cap
    eps = [0.01]
    eqs = [eq0]
    run_tests(eqs, eps, om_effs, Tm, T, mup, "stability/smallecc-ext/nocap")

    # capture
    mup = 100*mu_cap
    eps = [0.01]
    eqs = [eq0]
    run_tests(eqs, eps, om_effs, Tm, T, mup, "stability/smallecc-ext/cap")

if __name__ == "__main__":
    testcases()
