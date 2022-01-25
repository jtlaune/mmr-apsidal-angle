from . import *
from .resonance import FOCompMassOmeff
from .plotting import plotsim
from .fndefs import *
from multiprocessing import Pool

def run_tp(h, j, mup, ap, a0, ep, e0, g0, Tm, Te, T, suptitle,
           dirname, filename, figname, paramsname, tscale=1e3,
           tol=1e-9, overwrite=False):
    lambda0 = np.random.randn()*2*np.pi
    t0 = 0.
    t1 = T
    # this is a sloppy way but preserves the code for test particles
    if Tm > 0:
        tploc = "int"
    if Tm < 0:
        tploc = "ext"

    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)
    if os.path.exists(os.path.join(dirname, filename)):
        if overwrite:
            print("overwriting...")
            sim = tp_intH(j, mup, ep, e0, ap, g0, a0, lambda0)

            (teval, thetap, newresin, newresout,
             eta, a1, e1, k, kc,
             alpha0, alpha, g, L, G,
             ebar, barg, x, y, ) = sim.int_Hsec(t0, t1, tol, Tm=Tm,
                                                Te=Te, om_eff=None,
                                                aext=None)

            if tploc == "int":
                teval=teval
                theta=thetap
                a1=a1
                a2=np.ones(len(teval))*ap
                e1=e1
                e2=np.ones(len(teval))*ep
                g1=g
                g2=np.zeros(len(teval))
                L1=L
                L2=np.ones(len(teval))
                x1=x
                y1=y
                x2=e2
                y2=g2
                
                np.savez( os.path.join(dirname, filename),
                          teval=teval, thetap=theta, a1=a1, a2=a2,
                          e1=e1, e2=e2, g1=g1, g2=g2, L1=L1, L2=L2,
                          x1=x1, y1=y1, x2=x2, y2=y2, )

            elif tploc == "ext":
                teval=teval
                theta=thetap
                a2=a1
                a1=np.ones(len(teval))*ap
                e2=e1
                e1=np.ones(len(teval))*ep
                g2=g
                g1=np.zeros(len(teval))
                L1=np.ones(len(teval))
                L2=L

                x1=e1
                y1=np.zeros(len(teval))
                x2=x
                y2=y
                
                np.savez( os.path.join(dirname, filename),
                          teval=teval, thetap=theta, a1=a1, a2=a2,
                          e1=e1, e2=e2, g1=g1, g2=g2, L1=L1, L2=L2,
                          x1=x1, y1=y1, x2=x2, y2=y2, )

        else:
            data = np.load(os.path.join(dirname, filename))
            teval  = data["teval"]
            theta = data["thetap"]
            a1 = data["a1"]
            a2 = data["a2"]
            e1 = data["e1"]
            e2 = data["e2"]
            g1 = data["g1"]
            g2 = data["g2"]
            L1 = data["L1"]
            L2 = data["L2"]
            x1 = data["x1"]
            y1 = data["y1"]
            x2 = data["x2"]
            y2 = data["y2"]

    else:
        sim = tp_intH(j, mup, ep, e0, ap, g0, a0, lambda0)

        (teval, thetap, newresin, newresout,
         eta, a1, e1, k, kc,
         alpha0, alpha, g, L, G,
         ebar, barg, x, y, ) = sim.int_Hsec(t0, t1, tol, Tm=Tm,
                                            Te=Te, om_eff=None,
                                            aext=None)

        if tploc == "int":
            teval=teval
            theta=thetap
            a1=a1
            a2=np.ones(len(teval))*ap
            e1=e1
            e2=np.ones(len(teval))*ep
            g1=g
            g2=np.zeros(len(teval))
            L1=L
            L2=np.ones(len(teval))
            x1=x
            y1=y
            x2=e2
            y2=g2
            
            np.savez( os.path.join(dirname, filename),
                      teval=teval, thetap=theta, a1=a1, a2=a2,
                      e1=e1, e2=e2, g1=g1, g2=g2, L1=L1, L2=L2,
                      x1=x1, y1=y1, x2=x2, y2=y2, )

        elif tploc == "ext":
            teval=teval
            theta=thetap
            a2=a1
            a1=np.ones(len(teval))*ap
            e2=e1
            e1=np.ones(len(teval))*ep
            g2=g
            g1=np.zeros(len(teval))
            L1=np.ones(len(teval))
            L2=L

            x1=e1
            y1=np.zeros(len(teval))
            x2=x
            y2=y
            
            np.savez( os.path.join(dirname, filename),
                      teval=teval, thetap=theta, a1=a1, a2=a2,
                      e1=e1, e2=e2, g1=g1, g2=g2, L1=L1, L2=L2,
                      x1=x1, y1=y1, x2=x2, y2=y2, )

    theta1 = (theta + g1) % (2 * np.pi)
    theta2 = (theta + g2) % (2 * np.pi)
    alpha = a1 / a2
    period_ratio = (alpha) ** (1.5)
    pnom = j / (j + 1)
    pdiff = period_ratio - pnom

    f1 = A(alpha, j)
    f2 = B(alpha, j)
    barg1 = np.arctan2(e2*np.sin(g2), e2*np.cos(g2) + f2*e1/f1)
    barg2 = np.arctan2(e1*np.sin(g1), e1*np.cos(g1) + f1*e2/f2)

    bartheta1 = (theta + barg1) % (2*np.pi)
    bartheta2 = (theta + barg2) % (2*np.pi)
    # from the reducing rotation (Henrard et al 1986)
    hattheta1 = np.arctan2(e1*sin(theta1) + f2/f1*e2*sin(theta2),
                          e1*cos(theta1) + f2/f1*e2*cos(theta2))
    hattheta2 = np.arctan2(e2*sin(theta2) + f1/f2*e1*sin(theta1),
                           e2*cos(theta2) + f1/f2*e1*cos(theta1))
    fig, ax = plt.subplots(5, 2, figsize=(10, 22))
    fontsize = 24

    # make a quick diagnostic plot
    plotsim(
        fig,
        ax,
        teval,
        suptitle,
        tscale,
        fontsize,
        (r"$a_1$", a1),
        (r"$\varpi_1-\varpi_2$", g1-g2),
        (r"$e_1$", e1),
        (r"$e_2$", e2),
        (r"$\theta_1$", theta1),
        (r"$\theta_2$", theta2),
        (r"$\overline{\theta}_1$", bartheta1),
        (r"$\overline{\theta}_2$", bartheta2),
        (r"$\hat{\theta}_1$", hattheta1),
        (r"$\hat{\theta}_2$", hattheta2),
        yfigupper=0.99,
    )

    ax[0, 0].scatter(teval / tscale, a2, s=2)
    axp = ax[0, 0].twinx()
    axp.scatter(teval/tscale, period_ratio, s=2, c="r")
    axp.set_ylabel(r"$P_1/P_2$", fontsize=fontsize)
    axp.tick_params(labelsize=fontsize)
    fig.subplots_adjust(wspace=0.6)

    fig.savefig(os.path.join(dirname, figname))

    commit = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    variables = [h, j, mup, ap, a0, ep, e0, g0, Tm, Te, T]
    variable_names = ["h", " j", "mup", "ap", "a0", "ep", "e0", "g0",
                       "Tm", "Te", "T",]


    with open(os.path.join(dirname, paramsname), "w+") as f:
        f.write("".join(["{} = {}\n".format(name, variable) for 
                         name, variable in zip(variable_names, variables)]))
        f.write("\ncommit {}".format(commit))

    return fig


def run_tp_set(params):
    h = float(params[0])
    j = float(params[1])
    mup = float(params[2])
    ap = float(params[3])
    a0 = float(params[4])
    if a0 > ap: tploc="ext"
    else: tploc="int"
    ep = float(params[5])
    e0 = float(params[6])
    g0 = float(params[7])
    Tm = float(params[8])
    Te = float(params[9])
    T = float(params[10])

    dirname = params[11]
    filename = params[12]
    figname = params[13]
    paramsname = params[14]

    tscale = float(params[15])
    tol = float(params[16])

    overwrite = (params[17]=="True")
    print(overwrite)

    if tploc == "int":
        eeq = np.sqrt(np.abs(Te/2/(j+1)/Tm))
    elif tploc == "ext":
        eeq = np.sqrt(np.abs(Te/2/j/Tm))

    suptitle = (f"{filename}\n" + f"T={T:0.1e} q={tploc} tp\n" \
                r"$\mu_{p}=$ " + f"{mup:0.2e}\n" \
                f"Tm={Tm:0.1e} Te={Te:0.1e}\n" \
                r"$h$ = " + f"{h:0.3f}\n" \
                r"$e_{tp,eq}$ = " + f"{eeq:0.3f}\n" \
                r"$e_{p}$ = " + f"{ep:0.3f}")

    run_tp(h, j, mup, ap, a0, ep, e0, g0, Tm, Te, T, suptitle,
           dirname, filename, figname, paramsname, tscale=1e3,
           tol=1e-9, overwrite=overwrite)


def run_compmass(h, j, mu1, q, a0, alpha2_0, e1_0, e2_0, g1_0, g2_0,
                 Tm1, Tm2, Te1, Te2, T, suptitle, dirname, filename,
                 figname, paramsname, verbose=False, tscale=1e3,
                 secular=True, e1d=None, e2d=None, overwrite=False,
                 cutoff=np.infty, method="RK45", Te_func=False):
    print(method)
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)
    if os.path.exists(os.path.join(dirname, filename)):
        if overwrite:
            sim = comp_mass_intH(j, mu1, q, a0, Tm1, Tm2, Te1, Te2,
                                 e1d=e1d, e2d=e2d, cutoff=cutoff, Te_func=Te_func)
            (teval, theta, a1, a2, e1, e2,
            g1, g2, L1, L2, x1, y1, x2, y2) = sim.int_Hsec(T, 1e-9,
                                                           alpha2_0, e1_0,
                                                           e2_0,g1_0, g2_0,
                                                           verbose=verbose,
                                                           secular=secular,
                                                           method=method)
            print("FILENAME="+os.path.join(dirname, filename))
            np.savez(
                os.path.join(dirname, filename),
                teval=teval,
                thetap=theta,
                a1=a1,
                a2=a2,
                e1=e1,
                e2=e2,
                g1=g1,
                g2=g2,
                L1=L1,
                L2=L2,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        else:
            print("FILENAME="+os.path.join(dirname, filename))
            data = np.load(os.path.join(dirname, filename))
            teval  = data["teval"]
            theta = data["thetap"]
            a1     = data["a1"]
            a2     = data["a2"]
            e1     = data["e1"]
            e2     = data["e2"]
            g1     = data["g1"]
            g2     = data["g2"]
            L1     = data["L1"]
            L2     = data["L2"]
            x1     = data["x1"]
            y1     = data["y1"]
            x2     = data["x2"]
            y2     = data["y2"]
            
    else:
        sim = comp_mass_intH(j, mu1, q, a0, Tm1, Tm2, Te1, Te2,
                             e1d=e1d, e2d=e2d, cutoff=cutoff, Te_func=Te_func)
        (teval, theta, a1, a2, e1, e2,
        g1, g2, L1, L2, x1, y1, x2, y2) = sim.int_Hsec(T, 1e-9,
                                                       alpha2_0, e1_0,
                                                       e2_0,g1_0, g2_0,
                                                       verbose=verbose,
                                                       secular=secular,
                                                       method=method)
        np.savez(
            os.path.join(dirname, filename),
            teval=teval,
            thetap=theta,
            a1=a1,
            a2=a2,
            e1=e1,
            e2=e2,
            g1=g1,
            g2=g2,
            L1=L1,
            L2=L2,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )


    theta1 = (theta + g1) % (2 * np.pi)
    theta2 = (theta + g2) % (2 * np.pi)
    alpha = a1 / a2
    period_ratio = (alpha) ** (1.5)
    pnom = j / (j + 1)
    pdiff = period_ratio - pnom

    f1 = A(alpha, j)
    f2 = B(alpha, j)
    barg1 = np.arctan2(e2*np.sin(g2), e2*np.cos(g2) + f2*e1/f1)
    barg2 = np.arctan2(e1*np.sin(g1), e1*np.cos(g1) + f1*e2/f2)

    bartheta1 = (theta + barg1) % (2*np.pi)
    bartheta2 = (theta + barg2) % (2*np.pi)
    # from the reducing rotation (Henrard et al 1986)
    hattheta1 = np.arctan2(e1*sin(theta1) + f2/f1*e2*sin(theta2),
                          e1*cos(theta1) + f2/f1*e2*cos(theta2))
    hattheta2 = np.arctan2(e2*sin(theta2) + f1/f2*e1*sin(theta1),
                           e2*cos(theta2) + f1/f2*e1*cos(theta1))
    fig, ax = plt.subplots(5, 2, figsize=(10, 22))
    fontsize = 24

    # make a quick diagnostic plot
    plotsim(
        fig,
        ax,
        teval,
        suptitle,
        tscale,
        fontsize,
        (r"$a_1$", a1),
        (r"$\varpi_1-\varpi_2$", g1-g2),
        (r"$e_1$", e1),
        (r"$e_2$", e2),
        (r"$\theta_1$", theta1),
        (r"$\theta_2$", theta2),
        (r"$\overline{\theta}_1$", bartheta1),
        (r"$\overline{\theta}_2$", bartheta2),
        (r"$\hat{\theta}_1$", hattheta1),
        (r"$\hat{\theta}_2$", hattheta2),
        yfigupper=0.99,
    )

    ax[0, 0].scatter(teval / tscale, a2, s=2)
    axp = ax[0, 0].twinx()
    axp.scatter(teval/tscale, period_ratio, s=2, c="r")
    axp.set_ylabel(r"$P_1/P_2$", fontsize=fontsize)
    axp.tick_params(labelsize=fontsize)
    fig.subplots_adjust(wspace=0.6)

    fig.savefig(os.path.join(dirname, figname))

    commit = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    variables = [j,q,mu1,a0,alpha2_0,Tm1,Tm2,Te1,Te2,T]
    variable_names = ["j","q","mu1","a0",
                      "alpha2_0","Tm1","Tm2","Te1","Te2","T"]

    with open(os.path.join(dirname, paramsname), "w+") as f:
        f.write("".join(["{} = {}\n".format(name, variable) for 
                         name, variable in zip(variable_names, variables)]))
        if not secular:
            f.write("\nSECULAR TERMS OFF\n")
        f.write("\ncommit {}".format(commit))

    return fig


class run_compmass_set:
    def __init__(self, verbose=False, overwrite=False, secular=True, method="RK45"):
        self.verbose   = verbose
        self.overwrite = overwrite
        self.secular   = secular
        self.method    = method
    def __call__(self, params):
        h = np.float64(params[0])
        j = np.float64(params[1])
        a0 = np.float64(params[2])
        q = np.float64(params[3])
        mu1 = np.float64(params[4])
        T = np.float64(params[5])

        Te_func = int(float(params[18]))
        if Te_func:
            Te1 = params[6]
            Te2 = params[7]
            Tm1 = params[8]
            Tm2 = params[9]
        else:
            Te1 = np.float64(params[6])
            Te2 = np.float64(params[7])
            Tm1 = np.float64(params[8])
            Tm2 = np.float64(params[9])

        e1_0 = np.float64(params[10])
        e2_0 = np.float64(params[11])
        e1d = np.float64(params[12])
        e2d = np.float64(params[13])
        alpha2_0 = np.float64(params[14])
        name = params[15]
        dirname = params[16]
        cutoff = np.float64(params[17])
        g1_0 = np.float64(params[19])
        g2_0 = np.float64(params[20])
        filename   = f"{name}.npz"
        figname    = f"{name}.png"
        paramsname = f"params-{name}.txt"
        if Te_func:
            suptitle = (f"{filename}\n" \
                        f"T={T:0.1e} q={q} " + r"$\mu_{1}=$ " + f"{mu1:0.2e}\n" \
                        r"$e_{1,d}$ = " + f"{e1d:0.3f} " \
                        r"$e_{2,d}$ = " + f"{e2d:0.3f}")
        else:
            suptitle = (f"{filename}\n" \
                        f"T={T:0.1e} q={q} " + r"$\mu_{1}=$ " + f"{mu1:0.2e}\n" \
                        f"Tm1={Tm1:0.1e} Te1={Te1:0.1e}\n" \
                        f"Tm2={Tm2:0.1e} Te2={Te2:0.1e}\n" \
                        r"$e_{1,d}$ = " + f"{e1d:0.3f} " \
                        r"$e_{2,d}$ = " + f"{e2d:0.3f}")
        run_compmass(h, j, mu1, q, a0, alpha2_0, e1_0, e2_0,g1_0,
                     g2_0, Tm1, Tm2, Te1, Te2, T, suptitle, dirname,
                     filename, figname, paramsname,
                     verbose=self.verbose, secular=self.secular,
                     e1d=e1d, e2d=e2d, overwrite=self.overwrite,
                     cutoff=cutoff, method=self.method,
                     Te_func=Te_func)


def run_compmass_omeff(h, j, mu1, q, a0, alpha2_0, e1_0, e2_0, g1_0,
                       g2_0, Tm1, Tm2, Te1, Te2, T, suptitle, dirname,
                       filename, figname, paramsname, verbose, tscale,
                       secular, e1d, e2d, overwrite, cutoff, method,
                       Te_func, muext, aext):
    print(method)
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)
    if os.path.exists(os.path.join(dirname, filename)):
        if overwrite:
            sim = FOCompMassOmeff(j, mu1, q, a0, Tm1, Tm2, Te1, Te2,
                                  e1d, e2d, cutoff, Te_func, aext, muext)
            (teval, theta, a1, a2, e1, e2,
            g1, g2, L1, L2, x1, y1, x2, y2) = sim.int_Hsec(T, 1e-9,
                                                           alpha2_0, e1_0,
                                                           e2_0,g1_0, g2_0,
                                                           verbose=verbose,
                                                           secular=secular,
                                                           method=method)
            print("DATAFILEPATH="+os.path.join(dirname, filename))
            np.savez(
                os.path.join(dirname, filename),
                teval=teval,
                thetap=theta,
                a1=a1,
                a2=a2,
                e1=e1,
                e2=e2,
                g1=g1,
                g2=g2,
                L1=L1,
                L2=L2,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        else:
            data = np.load(os.path.join(dirname, filename))
            teval  = data["teval"]
            theta = data["thetap"]
            a1     = data["a1"]
            a2     = data["a2"]
            e1     = data["e1"]
            e2     = data["e2"]
            g1     = data["g1"]
            g2     = data["g2"]
            L1     = data["L1"]
            L2     = data["L2"]
            x1     = data["x1"]
            y1     = data["y1"]
            x2     = data["x2"]
            y2     = data["y2"]
            
    else:
        sim = FOCompMassOmeff(j, mu1, q, a0, Tm1, Tm2, Te1, Te2,
                              e1d, e2d, cutoff, Te_func, aext, muext)
        (teval, theta, a1, a2, e1, e2,
        g1, g2, L1, L2, x1, y1, x2, y2) = sim.int_Hsec(T, 1e-9,
                                                       alpha2_0, e1_0,
                                                       e2_0,g1_0, g2_0,
                                                       verbose=verbose,
                                                       secular=secular,
                                                       method=method)
        np.savez(
            os.path.join(dirname, filename),
            teval=teval,
            thetap=theta,
            a1=a1,
            a2=a2,
            e1=e1,
            e2=e2,
            g1=g1,
            g2=g2,
            L1=L1,
            L2=L2,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )


    theta1 = (theta + g1) % (2 * np.pi)
    theta2 = (theta + g2) % (2 * np.pi)
    alpha = a1 / a2
    period_ratio = (alpha) ** (1.5)
    pnom = j / (j + 1)
    pdiff = period_ratio - pnom

    f1 = f27lc(alpha, j)
    f2 = f31lc(alpha, j)
    barg1 = np.arctan2(e2*np.sin(g2), e2*np.cos(g2) + f2*e1/f1)
    barg2 = np.arctan2(e1*np.sin(g1), e1*np.cos(g1) + f1*e2/f2)

    bartheta1 = (theta + barg1) % (2*np.pi)
    bartheta2 = (theta + barg2) % (2*np.pi)
    # from the reducing rotation (Henrard et al 1986)
    hattheta1 = np.arctan2(e1*sin(theta1) + f2/f1*e2*sin(theta2),
                          e1*cos(theta1) + f2/f1*e2*cos(theta2))
    hattheta2 = np.arctan2(e2*sin(theta2) + f1/f2*e1*sin(theta1),
                           e2*cos(theta2) + f1/f2*e1*cos(theta1))
    fig, ax = plt.subplots(5, 2, figsize=(10, 22))
    fontsize = 24

    # make a quick diagnostic plot
    plotsim(
        fig,
        ax,
        teval,
        suptitle,
        tscale,
        fontsize,
        (r"$a_1$", a1),
        (r"$\varpi_1-\varpi_2$", g1-g2),
        (r"$e_1$", e1),
        (r"$e_2$", e2),
        (r"$\theta_1$", theta1),
        (r"$\theta_2$", theta2),
        (r"$\overline{\theta}_1$", bartheta1),
        (r"$\overline{\theta}_2$", bartheta2),
        (r"$\hat{\theta}_1$", hattheta1),
        (r"$\hat{\theta}_2$", hattheta2),
        yfigupper=0.99,
    )

    ax[0, 0].scatter(teval / tscale, a2, s=2)
    axp = ax[0, 0].twinx()
    axp.scatter(teval/tscale, period_ratio, s=2, c="r")
    axp.set_ylabel(r"$P_1/P_2$", fontsize=fontsize)
    axp.tick_params(labelsize=fontsize)
    fig.subplots_adjust(wspace=0.6)

    fig.savefig(os.path.join(dirname, figname))

    commit = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    variables = [j,q,mu1,a0,alpha2_0,Tm1,Tm2,Te1,Te2,T]
    variable_names = ["j","q","mu1","a0",
                      "alpha2_0","Tm1","Tm2","Te1","Te2","T"]

    with open(os.path.join(dirname, paramsname), "w+") as f:
        f.write("".join(["{} = {}\n".format(name, variable) for 
                         name, variable in zip(variable_names, variables)]))
        if not secular:
            f.write("\nSECULAR TERMS OFF\n")
        f.write("\ncommit {}".format(commit))

    return fig
    

class run_compmass_set_omeff(run_compmass_set):
    def __init__(self, verbose=False, overwrite=False, secular=True, method="RK45"):
        self.verbose   = verbose
        self.overwrite = overwrite
        self.secular   = secular
        self.method    = method
        self.tscale = 1e3

    def __call__(self, params):
        h = np.float64(params[0])
        j = np.float64(params[1])
        a0 = np.float64(params[2])
        q = np.float64(params[3])
        mu1 = np.float64(params[4])
        T = np.float64(params[5])

        Te_func = int(float(params[18]))
        if Te_func:
            Te1 = params[6]
            Te2 = params[7]
            Tm1 = params[8]
            Tm2 = params[9]
        else:
            Te1 = np.float64(params[6])
            Te2 = np.float64(params[7])
            Tm1 = np.float64(params[8])
            Tm2 = np.float64(params[9])

        e1_0 = np.float64(params[10])
        e2_0 = np.float64(params[11])
        e1d = np.float64(params[12])
        e2d = np.float64(params[13])
        alpha2_0 = np.float64(params[14])
        name = params[15]
        dirname = params[16]
        cutoff = np.float64(params[17])
        g1_0 = np.float64(params[19])
        g2_0 = np.float64(params[20])
        muext = np.float64(params[21])
        aext = np.float64(params[22])
        filename   = f"{name}.npz"
        figname    = f"{name}.png"
        paramsname = f"params-{name}.txt"
        if Te_func:
            suptitle = (f"{filename}\n" \
                        f"T={T:0.1e} q={q} " + r"$\mu_{1}=$ " + f"{mu1:0.2e}\n" \
                        r"$a_{\rm ext}$ = " + f"{aext:0.3f} " \
                        r"$\mu_{\rm ext}$ = " + f"{muext:0.3f}")
        else:
            suptitle = (f"{filename}\n" \
                        f"T={T:0.1e} q={q} " + r"$\mu_{1}=$ " + f"{mu1:0.2e}\n" \
                        f"Tm1={Tm1:0.1e} Te1={Te1:0.1e}\n" \
                        f"Tm2={Tm2:0.1e} Te2={Te2:0.1e}\n" \
                        r"$a_{\rm ext}$ = " + f"{aext:0.3f} " \
                        r"$\mu_{\rm ext}$ = " + f"{muext:0.3f}")
        run_compmass_omeff(h, j, mu1, q, a0, alpha2_0, e1_0,
                           e2_0,g1_0, g2_0, Tm1, Tm2, Te1, Te2, T,
                           suptitle, dirname, filename, figname,
                           paramsname, self.verbose, self.tscale,
                           self.secular, e1d, e2d, self.overwrite,
                           cutoff, self.method, Te_func,muext, aext)


class SimSeries(object):
    """
    - file management
    - setting up files, reading RUN_PARAMS from file
    - loading data from npz files
    """
    def __init__(self, series, projectdir, load=True):
        #self.RUN_PARAMS = load_params(paramsname)
        self.seriesname = series
        self.pdir = projectdir
        self.sdir = os.path.join(self.pdir, self.seriesname)
        self.paramsfpath = os.path.join(self.sdir, self.seriesname+"-params.py")
        self.load = load
        self.data = {}
        self.initialize()

    def initialize(self):
        self.RUN_PARAMS = self.load_params(self.paramsfpath)

    def load_run(self, ind):
        params = self.RUN_PARAMS
        Nqs = len(params[:,0])
        name = params[ind,15]
        dirname = params[ind,16]
        dirname = os.path.join(self.seriesname, dirname)
        filename   = f"{name}.npz"
        try:
            data = np.load(os.path.join(dirname, filename))
            self.data[ind] = data
        except FileNotFoundError as err:
            print(f"Cannot find file {filename}... have you run it?")
            raise err

    def load_all_runs(self):
        params = self.RUN_PARAMS
        Nqs = len(params[:,0])
        for ind in range(Nqs):
            self.load_run(ind)

    def load_params(self, filepath):
        spec = importlib.util.spec_from_file_location("_", filepath)
        _ = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_)
        print(f"Loading run file {filepath} in directory {os.getcwd()}")
        return(np.array(_.RUN_PARAMS))
    

class SeriesFOCompmass(SimSeries):
    """
    - Class to run first order comparable mass simulations
    - define which physics to include for compmass, i.e. omeff, dissipative, etc
      - could possibly be defined in a separate file like fargo?
    """
#    def __init__(
    def __call__(self, Nproc=8):
        # change to series directory
        if not os.path.exists(self.seriesname):
            os.mkdir(self.seriesname)
        os.chdir(self.seriesname)
        print(os.getcwd())

        N_sims = self.RUN_PARAMS.shape[0]

        overwrite = not self.load
        integrate = run_compmass_set_omeff(verbose=True,
                                           overwrite=overwrite,
                                           secular=True, method="RK45")
        np.savez("RUN_PARAMS", self.RUN_PARAMS)
        print(self.RUN_PARAMS)
        print(f"Running {N_sims} simulations...")
        
        with Pool(processes=min(Nproc, N_sims)) as pool:
            pool.map(integrate, self.RUN_PARAMS)
        os.chdir(self.pdir)
