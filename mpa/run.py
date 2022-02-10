import numpy as np
import subprocess
import matplotlib.pyplot as plt
from numpy import sin, cos
import os
from .resonance import FOCompMassOmeff, FOCompMass
from .plotting import plotsim
from . import fndefs as fns
import matplotlib as mpl
from .mpl_styles import analytic

##########################################################################
# Decorators                                                             #
##########################################################################
def series_dir(f):
    # first check if series directory exists then change into
    # directory, then change out of it only to be used with
    # .initialize() and .__call__() methods
    def wrapper1(*args):
        s = args[0]
        pwd = os.path.abspath(os.getcwd())
        if not os.path.exists(s.sdir):
            os.mkdir(s.sdir)
        os.chdir(s.sdir)
        # do stuff before
        try:
            f(*args)
        except TypeError as err:
            raise err
        # do stuff after
        os.chdir(pwd)

    return wrapper1


def params_load(f):
    # alternative to doing stuff like this in SimSet objects
    # will probably define in run.py and fix code there next
    # h = params[0]
    # ...
    # ...
    # etc
    def wrapper1(*args):
        names = args[0].params_spec
        # do stuff before
        vals = args[1]

        for val, name in zip(vals, names):
            try:
                args[0].params[name] = float(val)
            except ValueError:
                args[0].params[name] = str(val)
        try:
            f(*args)
        except TypeError as err:
            print(f)
            raise err
        # do stuff after

    return wrapper1


##########################################################################
# Run functions
##########################################################################
def run_tp(
    h,
    j,
    mup,
    ap,
    a0,
    ep,
    e0,
    g0,
    Tm,
    Te,
    T,
    suptitle,
    dirname,
    filename,
    figname,
    paramsname,
    tscale=1e3,
    tol=1e-9,
    overwrite=False,
):
    lambda0 = np.random.randn() * 2 * np.pi
    t0 = 0.0
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
            sim = fns.tp_intH(j, mup, ep, e0, ap, g0, a0, lambda0)

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
            ) = sim.int_Hsec(t0, t1, tol, Tm=Tm, Te=Te, om_eff=None, aext=None)

            if tploc == "int":
                teval = teval
                theta = thetap
                a1 = a1
                a2 = np.ones(len(teval)) * ap
                e1 = e1
                e2 = np.ones(len(teval)) * ep
                g1 = g
                g2 = np.zeros(len(teval))
                L1 = L
                L2 = np.ones(len(teval))
                x1 = x
                y1 = y
                x2 = e2
                y2 = g2

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

            elif tploc == "ext":
                teval = teval
                theta = thetap
                a2 = a1
                a1 = np.ones(len(teval)) * ap
                e2 = e1
                e1 = np.ones(len(teval)) * ep
                g2 = g
                g1 = np.zeros(len(teval))
                L1 = np.ones(len(teval))
                L2 = L

                x1 = e1
                y1 = np.zeros(len(teval))
                x2 = x
                y2 = y

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
            teval = data["teval"]
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
        sim = fns.tp_intH(j, mup, ep, e0, ap, g0, a0, lambda0)

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
        ) = sim.int_Hsec(t0, t1, tol, Tm=Tm, Te=Te, om_eff=None, aext=None)

        if tploc == "int":
            teval = teval
            theta = thetap
            a1 = a1
            a2 = np.ones(len(teval)) * ap
            e1 = e1
            e2 = np.ones(len(teval)) * ep
            g1 = g
            g2 = np.zeros(len(teval))
            L1 = L
            L2 = np.ones(len(teval))
            x1 = x
            y1 = y
            x2 = e2
            y2 = g2

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

        elif tploc == "ext":
            teval = teval
            theta = thetap
            a2 = a1
            a1 = np.ones(len(teval)) * ap
            e2 = e1
            e1 = np.ones(len(teval)) * ep
            g2 = g
            g1 = np.zeros(len(teval))
            L1 = np.ones(len(teval))
            L2 = L

            x1 = e1
            y1 = np.zeros(len(teval))
            x2 = x
            y2 = y

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

    f1 = fns.f1(alpha, j)
    f2 = fns.B(alpha, j)
    barg1 = np.arctan2(e2 * np.sin(g2), e2 * np.cos(g2) + f2 * e1 / f1)
    barg2 = np.arctan2(e1 * np.sin(g1), e1 * np.cos(g1) + f1 * e2 / f2)

    bartheta1 = (theta + barg1) % (2 * np.pi)
    bartheta2 = (theta + barg2) % (2 * np.pi)
    # from the reducing rotation (Henrard et al 1986)
    hattheta1 = np.arctan2(
        e1 * sin(theta1) + f2 / f1 * e2 * sin(theta2),
        e1 * cos(theta1) + f2 / f1 * e2 * cos(theta2),
    )
    hattheta2 = np.arctan2(
        e2 * sin(theta2) + f1 / f2 * e1 * sin(theta1),
        e2 * cos(theta2) + f1 / f2 * e1 * cos(theta1),
    )
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
        (r"$\varpi_1-\varpi_2$", g1 - g2),
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
    axp.scatter(teval / tscale, period_ratio, s=2, c="r")
    axp.set_ylabel(r"$P_1/P_2$", fontsize=fontsize)
    axp.tick_params(labelsize=fontsize)
    fig.subplots_adjust(wspace=0.6)

    fig.savefig(os.path.join(dirname, figname))

    commit = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode()
        .strip()
    )

    variables = [h, j, mup, ap, a0, ep, e0, g0, Tm, Te, T]
    variable_names = ["h", " j", "mup", "ap", "a0", "ep", "e0", "g0", "Tm", "Te", "T"]

    with open(os.path.join(dirname, paramsname), "w+") as f:
        f.write(
            "".join(
                [
                    "{} = {}\n".format(name, variable)
                    for name, variable in zip(variable_names, variables)
                ]
            )
        )
        f.write("\ncommit {}".format(commit))

    return fig


@mpl.rc_context(analytic)
def run_compmass(
    verbose,
    tscale,
    secular,
    overwrite,
    method,
    suptitle,
    filename,
    figname,
    paramsname,  # end of positional params
    h,
    j,
    a0,
    q,
    mu1,
    T,
    Te1,
    Te2,
    Tm1,
    Tm2,
    e1_0,
    e2_0,
    e1d,
    e2d,
    alpha2_0,
    name,
    dirname,
    cutoff,
    g1_0,
    g2_0,
):
    Te_func = 0.0  # I don't think i use this. can delete this param in future
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)
    if os.path.exists(os.path.join(dirname, filename)):
        sim = FOCompMass(
            j,
            mu1,
            q,
            a0,
            Tm1,
            Tm2,
            Te1,
            Te2,
            e1d,
            e2d,
            cutoff,
            Te_func,
        )
        (teval, theta, a1, a2, e1, e2, g1, g2, L1, L2, x1, y1, x2, y2,) = sim.int_Hsec(
            T,
            1e-9,
            alpha2_0,
            e1_0,
            e2_0,
            g1_0,
            g2_0,
            verbose=verbose,
            secular=secular,
            method=method,
            # need to phase this out and add overwrite into class defs
        )
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
        sim = FOCompMassOmeff(
            j, mu1, q, a0, Tm1, Tm2, Te1, Te2, e1d, e2d, cutoff, Te_func
        )
        (teval, theta, a1, a2, e1, e2, g1, g2, L1, L2, x1, y1, x2, y2) = sim.int_Hsec(
            T,
            1e-9,
            alpha2_0,
            e1_0,
            e2_0,
            g1_0,
            g2_0,
            verbose=verbose,
            secular=secular,
            method=method,
        )
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

    f1 = fns.f27lc(alpha, j)
    f2 = fns.f31lc(alpha, j)
    barg1 = np.arctan2(e2 * np.sin(g2), e2 * np.cos(g2) + f2 * e1 / f1)
    barg2 = np.arctan2(e1 * np.sin(g1), e1 * np.cos(g1) + f1 * e2 / f2)

    bartheta1 = (theta + barg1) % (2 * np.pi)
    bartheta2 = (theta + barg2) % (2 * np.pi)
    # from the reducing rotation (Henrard et al 1986)
    hattheta1 = np.arctan2(
        e1 * sin(theta1) + f2 / f1 * e2 * sin(theta2),
        e1 * cos(theta1) + f2 / f1 * e2 * cos(theta2),
    )
    hattheta2 = np.arctan2(
        e2 * sin(theta2) + f1 / f2 * e1 * sin(theta1),
        e2 * cos(theta2) + f1 / f2 * e1 * cos(theta1),
    )
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
        (r"$\varpi_1-\varpi_2$", g1 - g2),
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
    axp.scatter(teval / tscale, period_ratio, s=2, c="r")
    axp.set_ylabel(r"$P_1/P_2$", fontsize=fontsize)
    axp.tick_params(labelsize=fontsize)
    fig.subplots_adjust(wspace=0.6)

    fig.savefig(os.path.join(dirname, figname))

    commit = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode()
        .strip()
    )

    variables = [j, q, mu1, a0, alpha2_0, Tm1, Tm2, Te1, Te2, T]
    variable_names = [
        "j",
        "q",
        "mu1",
        "a0",
        "alpha2_0",
        "Tm1",
        "Tm2",
        "Te1",
        "Te2",
        "T",
    ]

    with open(os.path.join(dirname, paramsname), "w+") as f:
        f.write(
            "".join(
                [
                    "{} = {}\n".format(name, variable)
                    for name, variable in zip(variable_names, variables)
                ]
            )
        )
        if not secular:
            f.write("\nSECULAR TERMS OFF\n")
        f.write("\ncommit {}".format(commit))

    return fig


@mpl.rc_context(analytic)
def run_compmass_omeff(
    verbose,
    tscale,
    secular,
    overwrite,
    method,
    suptitle,
    filename,
    figname,
    paramsname,  # end of positional params
    h,
    j,
    a0,
    q,
    mu1,
    T,
    Te1,
    Te2,
    Tm1,
    Tm2,
    e1_0,
    e2_0,
    e1d,
    e2d,
    alpha2_0,
    name,
    dirname,
    cutoff,
    g1_0,
    g2_0,
    omeff,
):
    Te_func = 0.0  # I don't think i use this. can delete this param in future
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)
    if os.path.exists(os.path.join(dirname, filename)):
        if overwrite:
            print("this code running")
            sim = FOCompMassOmeff(
                j,
                mu1,
                q,
                a0,
                Tm1,
                Tm2,
                Te1,
                Te2,
                e1d,
                e2d,
                cutoff,
                Te_func,
                omeff,
            )
            (
                teval,
                theta,
                a1,
                a2,
                e1,
                e2,
                g1,
                g2,
                L1,
                L2,
                x1,
                y1,
                x2,
                y2,
            ) = sim.int_Hsec(
                T,
                1e-9,
                alpha2_0,
                e1_0,
                e2_0,
                g1_0,
                g2_0,
                verbose=verbose,
                secular=secular,
                method=method,
                # need to phase this out and add overwrite into class defs
            )
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

        else: return(None)
    else:
        sim = FOCompMassOmeff(
            j,
            mu1,
            q,
            a0,
            Tm1,
            Tm2,
            Te1,
            Te2,
            e1d,
            e2d,
            cutoff,
            Te_func,
            omeff,
        )
        (teval, theta, a1, a2, e1, e2, g1, g2, L1, L2, x1, y1, x2, y2) = sim.int_Hsec(
            T,
            1e-9,
            alpha2_0,
            e1_0,
            e2_0,
            g1_0,
            g2_0,
            verbose=verbose,
            secular=secular,
            method=method,
        )
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

    f1 = fns.f27lc(alpha, j)
    f2 = fns.f31lc(alpha, j)
    barg1 = np.arctan2(e2 * np.sin(g2), e2 * np.cos(g2) + f2 * e1 / f1)
    barg2 = np.arctan2(e1 * np.sin(g1), e1 * np.cos(g1) + f1 * e2 / f2)

    bartheta1 = (theta + barg1) % (2 * np.pi)
    bartheta2 = (theta + barg2) % (2 * np.pi)
    # from the reducing rotation (Henrard et al 1986)
    hattheta1 = np.arctan2(
        e1 * sin(theta1) + f2 / f1 * e2 * sin(theta2),
        e1 * cos(theta1) + f2 / f1 * e2 * cos(theta2),
    )
    hattheta2 = np.arctan2(
        e2 * sin(theta2) + f1 / f2 * e1 * sin(theta1),
        e2 * cos(theta2) + f1 / f2 * e1 * cos(theta1),
    )
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
        (r"$\varpi_1-\varpi_2$", g1 - g2),
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
    axp.scatter(teval / tscale, period_ratio, s=2, c="r")
    axp.set_ylabel(r"$P_1/P_2$", fontsize=fontsize)
    axp.tick_params(labelsize=fontsize)
    fig.subplots_adjust(wspace=0.6)

    fig.savefig(os.path.join(dirname, figname))

    commit = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode()
        .strip()
    )

    variables = [j, q, mu1, a0, alpha2_0, Tm1, Tm2, Te1, Te2, T]
    variable_names = [
        "j",
        "q",
        "mu1",
        "a0",
        "alpha2_0",
        "Tm1",
        "Tm2",
        "Te1",
        "Te2",
        "T",
    ]

    with open(os.path.join(dirname, paramsname), "w+") as f:
        f.write(
            "".join(
                [
                    "{} = {}\n".format(name, variable)
                    for name, variable in zip(variable_names, variables)
                ]
            )
        )
        if not secular:
            f.write("\nSECULAR TERMS OFF\n")
        f.write("\ncommit {}".format(commit))

    return fig


##########################################################################
# Simulation sets, one+ parallel executions
##########################################################################
class SimSet(object):
    params = {}

    def __init__(
        self, verbose=False, overwrite=False, secular=True, tscale=1000.0, method="RK45"
    ):
        self.verbose = verbose
        self.overwrite = overwrite
        self.secular = secular
        self.method = method
        self.tscale = tscale


class TpSet(SimSet):
    def __call__(params):
        h = float(params[0])
        j = float(params[1])
        mup = float(params[2])
        ap = float(params[3])
        a0 = float(params[4])
        if a0 > ap:
            tploc = "ext"
        else:
            tploc = "int"
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

        overwrite = params[17] == "True"

        if tploc == "int":
            eeq = np.sqrt(np.abs(Te / 2 / (j + 1) / Tm))
        elif tploc == "ext":
            eeq = np.sqrt(np.abs(Te / 2 / j / Tm))

        suptitle = (
            f"{filename}\n" + f"T={T:0.1e} q={tploc} tp\n"
            r"$\mu_{p}=$ " + f"{mup:0.2e}\n"
            f"Tm={Tm:0.1e} Te={Te:0.1e}\n"
            r"$h$ = " + f"{h:0.3f}\n"
            r"$e_{tp,eq}$ = " + f"{eeq:0.3f}\n"
            r"$e_{p}$ = " + f"{ep:0.3f}"
        )

        run_tp(
            h,
            j,
            mup,
            ap,
            a0,
            ep,
            e0,
            g0,
            Tm,
            Te,
            T,
            suptitle,
            dirname,
            filename,
            figname,
            paramsname,
            tscale=tscale,
            tol=tol,
            overwrite=overwrite,
        )


class CompmassSet(SimSet):
    params_spec = [
        "h",
        "j",
        "a0",
        "q",
        "mu1",
        "T",
        "Te1",
        "Te2",
        "Tm1",
        "Tm2",
        "e1_0",
        "e2_0",
        "e1d",
        "e2d",
        "alpha2_0",
        "name",
        "dirname",
        "cutoff",
        "g1_0",
        "g2_0",
    ]

    @params_load
    def __call__(self, params):
        name = self.params["name"]
        T = self.params["T"]
        Te1 = self.params["Te1"]
        Te2 = self.params["Te2"]
        e1d = self.params["e1d"]
        e2d = self.params["e2d"]
        Tm1 = self.params["Tm1"]
        Tm2 = self.params["Tm2"]
        q = self.params["q"]
        mu1 = self.params["mu1"]

        filename = f"{name}.npz"
        figname = f"{name}.png"
        paramsname = f"params-{name}.txt"
        suptitle = (
            f"{filename}\n"
            f"T={T:0.1e} q={q} " + r"$\mu_{1}=$ " + f"{mu1:0.2e}\n"
            f"Tm1={Tm1:0.1e} Te1={Te1:0.1e}\n"
            f"Tm2={Tm2:0.1e} Te2={Te2:0.1e}\n"
            r"$e_{1,d}$ = " + f"{e1d:0.3f} "
            r"$e_{2,d}$ = " + f"{e2d:0.3f}"
        )

        run_compmass(
            self.verbose,
            self.tscale,
            self.secular,
            self.overwrite,
            self.method,
            suptitle,
            filename,
            figname,
            paramsname,  # end of positional params
            **self.params,
        )


class CompmassSetOmeff(SimSet):
    params_spec = [
        "h",
        "j",
        "a0",
        "q",
        "mu1",
        "T",
        "Te1",
        "Te2",
        "Tm1",
        "Tm2",
        "e1_0",
        "e2_0",
        "e1d",
        "e2d",
        "alpha2_0",
        "name",
        "dirname",
        "cutoff",
        "g1_0",
        "g2_0",
        "omeff",
    ]

    @params_load
    def __call__(self, params):  # params NEEDS to be here for
        # decorator to work.
        # TODO put params in argument of params_load
        name = self.params["name"]
        T = self.params["T"]
        Te1 = self.params["Te1"]
        Te2 = self.params["Te2"]
        Tm1 = self.params["Tm1"]
        Tm2 = self.params["Tm2"]
        q = self.params["q"]
        mu1 = self.params["mu1"]
        omeff = self.params["omeff"]

        filename = f"{name}.npz"
        figname = f"{name}.png"
        paramsname = f"params-{name}.txt"
        suptitle = (
            f"{filename}\n"
            f"T={T:0.1e} q={q} " + r"$\mu_{1}=$ " + f"{mu1:0.2e}\n"
            f"Tm1={Tm1:0.1e} Te1={Te1:0.1e}\n"
            f"Tm2={Tm2:0.1e} Te2={Te2:0.1e}\n"
            r"$\omega_{\rm eff}$ = " + f"{omeff:0.3e}"
        )

        run_compmass_omeff(
            self.verbose,
            self.tscale,
            self.secular,
            self.overwrite,
            self.method,
            suptitle,
            filename,
            figname,
            paramsname,
            **self.params,
        )
