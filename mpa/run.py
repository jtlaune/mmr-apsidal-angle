import numpy as np
import subprocess
import matplotlib.pyplot as plt
from numpy import sin, cos
import os
from .resonance import FOCompMassOmeff, FOCompMass, FOTestPartOmeff
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
    omeff1,
    omeff2,
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
                omeff1,
                omeff2,
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

        else:
            return None
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
            omeff1,
            omeff2,
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
def run_tp_omeff(
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
    mup,
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
    om_ext,
    om_pext,
):
    print(om_ext)
    print(om_pext)
    lambda0 = np.random.randn() * 2 * np.pi
    t0 = 0.0
    t1 = T

    if q < 1:
        mu1 = 0.0
        mu2 = mup
    else:
        mu2 = 0.0
        mu1 = mup

    a1_0 = a0
    a2_0 = a0 * alpha2_0
    tol = 1e-9

    # this is a sloppy way but preserves the code for test particles
    # TODO should do type checking and have q input be like 0 and
    # infty for clarity
    if q < 1:
        tploc = "int"
        mup = mu2
        ep = e2_0
        ap = a2_0
        e0 = e1_0
        a0 = a1_0
        g0 = g1_0
        Tm = Tm1
        Te = Te1
    elif q > 1:
        tploc = "ext"
        mup = mu1
        ep = e1_0
        ap = a1_0
        e0 = e2_0
        a0 = a2_0
        g0 = g2_0
        Tm = Tm2
        Te = Te2
    else:  # something is messed up
        raise Warning("q is neither > or < 1")

    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)

    sim = FOTestPartOmeff(j, mup, ep, e0, ap, g0, a0, lambda0, cutoff)

    (teval, theta0, a, L, e, x, y, g, G, pomp) = sim.int_Hsec(
        t0, t1, tol, Tm=Tm, Te=Te, om_pext=om_pext, om_ext=om_ext, secular=secular
    )

    if tploc == "int":
        teval = teval
        a1 = a
        e1 = e
        g1 = g
        g2 = -pomp
        L1 = L
        x1 = x
        y1 = y

        a2 = np.ones(len(teval)) * ap
        e2 = np.ones(len(teval)) * ep
        L2 = np.ones(len(teval))
        x2 = ep
        y2 = np.zeros(len(teval))

    elif tploc == "ext":
        teval = teval
        a2 = a
        e2 = e
        g1 = -pomp
        g2 = g
        L2 = L
        x2 = x
        y2 = y

        a1 = np.ones(len(teval)) * ap
        e1 = np.ones(len(teval)) * ep
        L1 = np.ones(len(teval))
        x1 = ep
        y1 = np.zeros(len(teval))

    np.savez(
        os.path.join(dirname, filename),
        teval=teval,
        theta0=theta0,
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

    theta1 = (theta0 + g1) % (2 * np.pi)
    theta2 = (theta0 + g2) % (2 * np.pi)
    alpha = a1 / a2
    period_ratio = (alpha) ** (1.5)

    f1 = -fns.f27lc(alpha, j)
    f2 = -fns.f31lc(alpha, j)
    barg1 = np.arctan2(e2 * np.sin(g2), e2 * np.cos(g2) + f2 * e1 / f1)
    barg2 = np.arctan2(e1 * np.sin(g1), e1 * np.cos(g1) + f1 * e2 / f2)

    bartheta1 = (theta0 + barg1) % (2 * np.pi)
    bartheta2 = (theta0 + barg2) % (2 * np.pi)
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
        (r"$e_1$", e1),
        (r"$e_2$", e2),
        (r"$\theta_1$", theta1),
        (r"$\theta_2$", theta2),
        (r"$\varpi_1$", -g1),
        (r"$\varpi_2$", -g2),
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
