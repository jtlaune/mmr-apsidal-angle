import numpy as np
from numpy import sin, cos, sqrt, pi
import scipy as sp
import scipy.integrate
import LaplaceCoefficients as LC
import helper
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from plotting import *
from helper import *
import subprocess

class check_ratio_cm:
    def __init__(self, ratio0, q):
        self.ratio0 = ratio0
        self.terminal = True
        self.q = q

    def __call__(self, t, Y):
        L1 = Y[1]
        L2 = Y[2]
        alpha1 = L1 ** 2 / self.q ** 2
        alpha2 = L2 ** 2 
        alpha = alpha1 / alpha2
        return(alpha - self.ratio0)
    
class check_ratio_tp:
    def __init__(self, ratio0):
        self.ratio0 = ratio0
        self.terminal = True

    def __call__(self, t, Y):
        L = Y[1]
        ratio = L * L
        return ratio - self.ratio0


class resonance:
    # Parent class for j:j+1 resonance with useful functions
    # that sets constants and initial values.
    def ebar(self, ep, e, A, B, g):
        return helper.ebarfunc(ep, e, A, B, g)

    def eta(self, ep, e, A, B, g):
        return helper.etafunc(ep, e, A, B, g)

    def alpha0(self, a, ap, j, ebar):
        return helper.alpha0func(a, ap, j, ebar)

    def A(self, alpha):
        return helper.A(alpha, self.j)

    def B(self, alpha):
        return helper.B(alpha, self.j)

    def C(self, alpha):
        return helper.C(alpha)

    def D(self, alpha):
        return helper.D(alpha)

    def omega(self, a, ap, mu):
        if a <= ap:
            return (
                (np.pi * mu)
                * (a / ap) ** 2
                / (2 * a * sqrt(a))
                * LC.b(1.5, 1.0, (a / ap))
            )
        if a > ap:
            return (
                (np.pi * mu) * (a / ap) / (2 * a * sqrt(a)) * LC.b(1.5, 1.0, (a / ap))
            )

    def nu(self, a, ap, mu):
        if a <= ap:
            return (
                (np.pi * mu)
                * (a / ap) ** 2
                / (2 * a * sqrt(a))
                * LC.b(1.5, 2.0, (a / ap))
            )
        if a > ap:
            return (
                (np.pi * mu) * (a / ap) / (2 * a * sqrt(a)) * LC.b(1.5, 2.0, (a / ap))
            )


class tp_intH(resonance):
    # This class integrates the Hamiltonian for an inner test
    # particle, with and without migration. Without migration we may
    # put the particle directly into resonance. In this case we still
    # have all the other initial conditions set to exact resonance,
    # but this doesn't matter after significant migration anyways.
    def __init__(self, j, mup, ep, e0, ap, g0, a0, lambda0):
        # set other params and dirname in child classes
        self.j = j
        self.mup = mup
        self.ep = ep
        self.e0 = e0
        self.ap = ap
        self.g0 = g0
        self.a0 = a0
        self.lambda0 = lambda0

    def H2dofsec(self, t, Y):
        thetap = Y[0]
        L = Y[1]
        x = Y[2]
        y = Y[3]

        g = np.arctan2(y, x)
        G = x * x + y * y

        mup = self.mup
        ep = self.ep
        j = self.j

        if self.Tm > 0:
            alpha = L * L
            theta = thetap + g
            dtheta_dl = -j
            A = self.A(alpha)
            B = self.B(alpha)
            C = self.C(alpha)
            D = self.D(alpha)
        else:
            alpha = 1.0 / (L * L)
            theta = thetap + g
            dtheta_dl = j + 1
            B = alpha * self.A(alpha)
            A = alpha * self.B(alpha)
            C = alpha * self.C(alpha)
            D = alpha * self.D(alpha)

        e = np.sqrt(2 * G / L)

        ldot = 1 / (L * L * L) + mup * (
            0.5 * A * e / L * np.cos(theta)
            + C * e * e / L
            + D * ep * e / (2 * L) * np.cos(g)
        )

        # including because we need to examine small changes in the resonant location
        ldot = ldot - mup * L * LC.Db(0.5, 0.0, alpha)

        Ldot = -mup * dtheta_dl * (A * e * np.sin(theta) + B * ep * np.sin(thetap))

        sqrtGgdot = (
            A * np.sqrt(1 / 2 / L) * np.cos(theta)
            + 2 * C * np.sqrt(G) / L
            + D * ep * np.sqrt(1 / 2 / L) * np.cos(g)
        )

        GdotoversqrtG = A * np.sqrt(2 / L) * np.sin(theta) + D * ep * np.sqrt(
            2 / L
        ) * np.sin(g)

        xdot = mup * (-0.5 * np.cos(g) * GdotoversqrtG + np.sin(g) * sqrtGgdot)

        ydot = mup * (-0.5 * np.sin(g) * GdotoversqrtG - np.cos(g) * sqrtGgdot)

        if self.migrate:
            # Here we're using time = tau*t
            # Add in the dissipative terms for migration
            Tm = self.Tm * self.tau
            Te = self.Te * self.tau
            Ldot = Ldot + (L / 2) * (1 / Tm - 4 * G / L / Te)
            xdot = xdot + cos(g) * sqrt(G) * (
                -1.0 / Te + 0.25 * (1.0 / Tm - 4 * G / Te / L)
            )
            ydot = ydot + sin(g) * sqrt(G) * (
                -1.0 / Te + 0.25 * (1.0 / Tm - 4 * G / Te / L)
            )

        if self.Tm > 0:
            thetapdot = (j + 1) * self.n_p / self.tau - j * ldot
        else:
            thetapdot = (j + 1) * ldot - j * self.n_p / self.tau

        if self.perturb:
            a = L * L * self.ap
            om1ext = om1ext_np(self.muext, a, self.ap, self.aext)
            ompext = ompext_np(self.muext, a, self.ap, self.aext)

            om_eff = om1ext - ompext

            xdot = xdot + self.om_eff * sqrt(G) * sin(g)
            ydot = ydot - self.om_eff * sqrt(G) * cos(g)

            # see above for the ldot secular force from mup
            # these are just ldot secular forcing from ext
            ldotext = (
                -self.muext
                * L
                * (self.ap * self.ap / self.aext / self.aext)
                * LC.Db(0.5, 0, a / self.aext)
            )
            lpdotext = (
                -(self.ap * self.ap / self.aext / self.aext)
                * self.muext
                * LC.Db(0.5, 0, self.ap / self.aext)
            )

            if self.Tm > 0:
                thetapdot = thetapdot + (j + 1) * lpdotext - j * ldotext
            if self.Tm < 0:
                thetapdot = thetapdot - j * lpdotext + (j + 1) * ldotext

        return np.array([thetapdot, Ldot, xdot, ydot])

    def int_Hsec(self, t0, t1, tol, Tm=None, Te=None, om_eff=None, aext=None):
        # Here we're using time = tau*t
        self.migrate = False
        if Tm is not None and Te is not None:
            self.migrate = True
            self.Tm = Tm
            self.Te = Te

        int_cond = None
        if Tm > 0:
            self.e_eq = np.sqrt(self.Te / (2 * (self.j + 1) * self.Tm))
            int_cond = check_ratio_tp(0.8)
            int_cond.terminal = True
        else:
            self.e_eq = np.sqrt(self.Te / (2 * self.j * np.abs(self.Tm)))
            int_cond = check_ratio_tp(1.25)
            int_cond.terminal = True

        self.perturb = False
        # if muext is not None and aext is not None:
        if om_eff is not None and aext is not None:
            # will be applied in the rotating frame so that varpi_p =
            # const = 0. assume that e_ext = 0.
            self.perturb = True
            self.aext = aext
            self.om_eff = om_eff
            if self.Tm > 0:
                ares = (self.j / (self.j + 1)) ** (2.0 / 3.0)
            if self.Tm < 0:
                ares = ((self.j + 1) / self.j) ** (2.0 / 3.0)
            self.muext = self.om_eff / (
                om1ext_np(1.0, ares, self.ap, self.aext)
                - ompext_np(1.0, ares, self.ap, self.aext)
            )

        self.theta_eq = np.arcsin(
            self.e_eq
            / (0.8 * self.j * self.mup * Te * (2 * np.pi * (self.j + 1) / self.j))
        )

        self.n_p = 2 * np.pi / sqrt(self.ap)
        # have to use tau = n_p, since anything else changes the
        # scaling of the Hamiltonian and the variables. messes results
        # up. can fix by adjusting EoM accordingly
        self.tau = self.n_p

        Lambda0 = np.sqrt(self.a0 / self.ap)
        Gamma0 = Lambda0 * (1 - np.sqrt(1 - self.e0 ** 2))

        RHS = self.H2dofsec
        IV = (
            self.lambda0,
            Lambda0,
            sqrt(Gamma0) * cos(self.g0),
            sqrt(Gamma0) * sin(self.g0),
        )
        # scaled H by GM/ap => tau -> sqrt(GM/ap) t; Lambda -> Lambda/sqrt(GM/ap)
        teval = np.linspace(t0, t1, 300000) * self.tau
        span = (teval[0], teval[-1])
        sol = sp.integrate.solve_ivp(
            RHS,
            span,
            IV,
            method="RK45",
            events=int_cond,
            t_eval=teval,
            rtol=tol,
            atol=tol,
            dense_output=True,
        )

        thetap = sol.y[0, :]
        L = sol.y[1, :]
        x = sol.y[2, :]
        y = sol.y[3, :]
        teval = teval[0 : len(thetap)]
        g = np.arctan2(y, x)
        G = x ** 2 + y ** 2

        e1 = np.sqrt(1 - (1 - G / L) ** 2)
        a1 = L ** 2 * self.ap
        if self.Tm > 0:
            alpha = a1 / self.ap
            barg = np.arctan2(
                e1 * np.sin(g),
                ((e1 * np.cos(g) + self.B(alpha) / self.A(alpha) * self.ep)),
            )
            ebar = self.ebar(self.ep, e1, self.A(alpha), self.B(alpha), g)
        else:
            alpha = self.ap / a1
            barg = np.arctan2(
                e1 * np.sin(g),
                ((e1 * np.cos(g) + self.A(alpha) / self.B(alpha) * self.ep)),
            )
            ebar = self.ebar(self.ep, e1, self.B(alpha), self.A(alpha), g)
        newresin = (thetap + barg) % (2 * np.pi)
        newresout = (thetap + barg) % (2 * np.pi)
        for i in range(len(newresin)):
            if newresin[i] > np.pi:
                newresin[i] = newresin[i] - 2 * np.pi
            if barg[i] > np.pi:
                barg[i] = barg[i] - 2 * np.pi
        alpha0 = alpha * (1 + self.j * ebar ** 2)
        k = (self.j + 1) * alpha0 ** 1.5 - self.j
        kc = 3 ** (1.0 / 3.0) * (
            self.mup * alpha0 * self.j * np.abs(self.A(alpha0))
        ) ** (2.0 / 3.0)
        eta = k / kc
        teval = teval / self.tau

        return (
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
        )

    def evecsec(self, t, Y):
        x = Y[0]
        y = Y[1]
        a = Y[2]
        ap = self.ap
        e = sqrt(x ** 2 + y ** 2)

        xdot = -self.omega(a, ap, self.mup) * y
        ydot = self.omega(a, ap, self.mup) * x - self.nu(a, ap, self.mup) * self.ep
        adot = 0.0

        if self.migrate:
            xdot = xdot - x / self.Te
            ydot = ydot - y / self.Te
            adot = adot + a * (1 / self.Tm - 2 * e * e / self.Te)

        return (xdot, ydot, adot)

    def int_evecsec(self, e0, t0, t1, tol, Tm=None, Te=None):
        self.migrate = False
        if Tm is not None and Te is not None:
            self.migrate = True
            self.Tm = Tm
            self.Te = Te

        if Tm > 0:
            self.e_eq = np.sqrt(self.Te / (2 * (self.j + 1) * self.Tm))
        else:
            self.e_eq = np.sqrt(self.Te / (2 * self.j * np.abs(self.Tm)))

        RHS = self.evecsec
        IV = (e0 * cos(-self.g0), e0 * sin(-self.g0), self.a0)
        # scaled H by GM/ap => tau -> sqrt(GM/ap) t; Lambda -> Lambda/sqrt(GM/ap)
        teval = np.linspace(t0, t1, 1000)
        span = (teval[0], teval[-1])
        sol = sp.integrate.solve_ivp(
            RHS,
            span,
            IV,
            method="RK45",
            t_eval=teval,
            rtol=tol,
            atol=tol,
            dense_output=True,
        )
        x = sol.y[0, :]
        y = sol.y[1, :]
        a = sol.y[2, :]
        gamma = -np.arctan2(y, x)
        e = sqrt(x ** 2 + y ** 2)

        return (teval, e, gamma, a)


class comp_mass_intH(resonance):
    # This class will integrate two planets with mass ratio q=m1/m2.
    # We will have T_m1,2 and T_e1,2 as parameters.
    def __init__(self, j, mu1, q, a0, Tm1, Tm2, Te1, Te2, e1d=None, e2d=None, cutoff=np.infty, Te_func=False):
        self.j = j
        self.mu1 = mu1
        self.q = q
        self.a0 = a0

        self.T0 = 2 * np.pi
        # This seems super sloppy. should probably do some type
        # checking or at least make all of them functions rather than
        # constants, but this would screw up a lot of other code.
        # Also, Te_func will toggle Tm functions as well but too lazy
        # to rename.
        self.Tm1 = Tm1
        self.Tm2 = Tm2
        self.Te_func = Te_func
        self.Te1 = Te1
        self.Te2 = Te2
        self.e1d = e1d
        self.e2d = e2d
        self.cutoff = cutoff * self.T0

    def H4dofsec(self, t, Y):

        # 7 variables
        theta = Y[0]
        L1 = Y[1]
        L2 = Y[2]
        x1 = Y[3]
        y1 = Y[4]
        x2 = Y[5]
        y2 = Y[6]

        g1 = np.arctan2(y1, x1)
        G1 = sqrt(x1 * x1 + y1 * y1)
        g2 = np.arctan2(y2, x2)
        G2 = sqrt(x2 * x2 + y2 * y2)

        j = self.j
        mu2 = self.mu1 / self.q

        e1 = np.sqrt(1 - (1 - G1 / L1) ** 2)
        e2 = np.sqrt(1 - (1 - G2 / L2) ** 2)

        # the Hamiltonian is -(constants) as the internal TP
        # Hamiltonian, i.e. it matches MD
        alpha1 = L1 * L1 / self.q / self.q
        alpha2 = L2 * L2
        alpha = alpha1 / alpha2
        theta1 = theta + g1
        theta2 = theta + g2
        f1 = -self.A(alpha)
        f2 = -self.B(alpha)
        C = self.C(alpha)
        D = self.D(alpha)

        ###################
        # Resonant forces #
        ###################
        l1dot = ((1 / alpha1 / sqrt(alpha1))
            + mu2 * f1 * e1 * cos(theta1) / (
            2 * alpha2 * sqrt(alpha1)
        ))
        l2dot = ((1 / alpha2 / sqrt(alpha2))
                 + self.q * mu2 / (alpha2 * sqrt(alpha2)) * (
            2 * f1 * e1 * cos(theta1) + 2.5 * f2 * e2 * cos(theta2)
        ))

        Ldot_prefactor = (
            self.q * mu2 / alpha2 * (f1 * e1 * sin(theta1) + f2 * e2 * sin(theta2))
        )
        L1dot = j * Ldot_prefactor
        L2dot = -(j + 1) * Ldot_prefactor

        e1g1dot = -mu2 * f1 * cos(theta1) / sqrt(alpha1) / alpha2
        e2g2dot = -self.q * mu2 * f2 * cos(theta2) / sqrt(alpha2) / alpha2

        G1dot = -self.q * mu2 * f1 * e1 * sin(theta1) / alpha2
        G2dot = -self.q * mu2 * f2 * e2 * sin(theta2) / alpha2

        x1dot = G1dot * cos(g1) - 0.5 * L1 * e1 * sin(g1) * e1g1dot
        y1dot = G1dot * sin(g1) + 0.5 * L1 * e1 * cos(g1) * e1g1dot

        x2dot = G2dot * cos(g2) - 0.5 * L2 * e2 * sin(g2) * e2g2dot
        y2dot = G2dot * sin(g2) + 0.5 * L2 * e2 * cos(g2) * e2g2dot

        thetadot = (j + 1) * l2dot - j * l1dot

        ##################
        # Secular forces #
        ##################
        if self.secular:
            l1dot_sec = (mu2/alpha2/sqrt(alpha1)
                         * (2*C*e1*e1+D*e1*e2/2*cos(g1-g2)))
            l2dot_sec = (self.q*mu2/alpha2/sqrt(alpha2)
                         * ((2*C*e1*e1 +3*C*e2*e2+ 2.5*D*e1*e2/2*cos(g1-g2))))
            G1dot_sec = -self.q*mu2*D*e1*e2/alpha2*sin(g1-g2)
            G2dot_sec = self.q*mu2*D*e1*e2/alpha2*sin(g1-g2)

            e1g1dot_sec = (-mu2/alpha2/sqrt(alpha1)*(2*C*e1+D*e2))
            e2g2dot_sec = (-self.q*mu2/alpha2/sqrt(alpha2)*(2*C*e2+D*e1))

            x1dot_sec = G1dot_sec * cos(g1) - 0.5 * L1 * e1 * sin(g1) * e1g1dot_sec
            y1dot_sec = G1dot_sec * sin(g1) + 0.5 * L1 * e1 * cos(g1) * e1g1dot_sec

            x2dot_sec = G2dot_sec * cos(g2) - 0.5 * L2 * e2 * sin(g2) * e2g2dot_sec
            y2dot_sec = G2dot_sec * sin(g2) + 0.5 * L2 * e2 * cos(g2) * e2g2dot_sec

        else:
            l1dot_sec = 0.
            l2dot_sec = 0.

            G1dot_sec = 0.
            G2dot_sec = 0.

            e1g1dot_sec = 0.
            e2g2dot_sec = 0.

            x1dot_sec = 0.
            y1dot_sec = 0.

            x2dot_sec = 0.
            y2dot_sec = 0.

        l1dot = l1dot + l1dot_sec
        x1dot = x1dot + x1dot_sec
        y1dot = y1dot + y1dot_sec

        l2dot = l2dot + l2dot_sec
        x2dot = x2dot + x2dot_sec
        y2dot = y2dot + y2dot_sec


        #############
        # MIGRATION #
        #############

        # Add in the dissipative terms for migration
        # convert time units
        if t < self.cutoff:
            T0 = self.T0

            if self.Te_func:
                Tm1 = self.Tm1(e1, t) * T0
                Tm2 = self.Tm2(e2, t) * T0
                Te1 = self.Te1(e1, t) * T0
                Te2 = self.Te2(e2, t) * T0
            # this is legacy code. should change e1d and e2d into
            # same format as Te_func
            else:
                Tm1 = self.Tm1 * T0
                Tm2 = self.Tm2 * T0
                Te1 = self.Te1 * T0
                Te2 = self.Te2 * T0
                if self.e1d:
                    Te1 = self.Te1/(e1-self.e1d)
                if self.e2d:
                    Te2 = self.Te2/(e2-self.e2d)

            L1dot_dis = (L1 / 2) * (1 / Tm1 - 2 * e1*e1 / Te1)
            L2dot_dis = (L2 / 2) * (1 / Tm2 - 2 * e2*e2 / Te2)

            L1dot = L1dot + L1dot_dis
            L2dot = L2dot + L2dot_dis

            G1dot_dis = (L1dot_dis * G1 / L1) - 2 * G1 / Te1
            G2dot_dis = (L2dot_dis * G2 / L2) - 2 * G2 / Te2

            x1dot = x1dot + cos(g1) * G1dot_dis
            y1dot = y1dot + sin(g1) * G1dot_dis

            x2dot = x2dot + cos(g2) * G2dot_dis
            y2dot = y2dot + sin(g2) * G2dot_dis

        if self.verbose:
            print(("alpha1: {:0.2f}    " \
                  "alpha2: {:0.2f}    " \
                  "alpha: {:0.2f}    " \
                  "theta1: {:0.2f}    " \
                  "theta2: {:0.2f}    " \
                  "done%: {:0.2f}" \
                  .format((L1/self.q)**2,
                          L2**2,
                          (L1/L2/self.q)**2,
                          (theta1 % (2*pi)),
                          (theta2 % (2*pi)),
                          100.*t/self.T,
                          )), end="\r")

        return(np.array([thetadot, L1dot, L2dot, x1dot,
                        y1dot, x2dot, y2dot]))

    def int_Hsec(self, t1, tol, alpha2_0, e1_0, e2_0, g1_0, g2_0,
                 verbose=False, secular=True, method="RK45"):
        self.secular = secular
        self.verbose = verbose
        self.T = self.T0*t1
        int_cond_min = check_ratio_cm(0.5, self.q)
        int_cond_max = check_ratio_cm(0.9, self.q)

        Lambda1_0 = self.q * 1
        Lambda2_0 = sqrt(alpha2_0)

        # set initial eccentricities
        G1_0 = 0.5 * Lambda1_0 * e1_0 ** 2
        G2_0 = 0.5 * Lambda2_0 * e2_0 ** 2

        # initial pomegas
        #g10 = -pi/4
        #g20 = 3*pi/4
        g10 = g1_0
        g20 = g2_0

        x1_0 = G1_0*cos(g10)
        y1_0 = G1_0*sin(g10)
        x2_0 = G2_0*cos(g20)
        y2_0 = G2_0*sin(g20)
        IV = (0, Lambda1_0, Lambda2_0, x1_0, y1_0, x2_0, y2_0)

        g1_0 = np.arctan2(y1_0, x1_0)
        g2_0 = np.arctan2(y2_0, x2_0)

        RHS = self.H4dofsec

        teval = np.linspace(0., t1, 300000) * self.T0
        span = (teval[0], teval[-1])
        sol = sp.integrate.solve_ivp(
            RHS,
            span,
            IV,
            method=method,
            events=[int_cond_min, int_cond_max],
            t_eval=teval,
            rtol=tol,
            atol=tol,
            dense_output=True,
        )

        theta = sol.y[0, :]
        L1 = sol.y[1, :]
        L2 = sol.y[2, :]
        x1 = sol.y[3, :]
        y1 = sol.y[4, :]
        x2 = sol.y[5, :]
        y2 = sol.y[6, :]

        teval = teval[0 : len(theta)]

        g1 = np.arctan2(y1, x1)
        G1 = sqrt(x1 ** 2 + y1 ** 2)

        g2 = np.arctan2(y2, x2)
        G2 = sqrt(x2 ** 2 + y2 ** 2)

        e1 = np.sqrt(1 - (1 - G1 / L1) ** 2)
        e2 = np.sqrt(1 - (1 - G2 / L2) ** 2)
        a1 = L1 ** 2 * self.a0 / self.q ** 2
        a2 = L2 ** 2 * self.a0
        alpha = a1 / a2

        # convert back to time units
        teval = teval / self.T0

        return (
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
        )


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
