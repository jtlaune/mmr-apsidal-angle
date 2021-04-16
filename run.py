import numpy as np
from numpy import sin, cos, sqrt
import scipy as sp
import LaplaceCoefficients as LC
import helper
from helper import *


class check_ratio:
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
        super().__init__(j, mup, ep, e0, ap, g0, a0, lambda0)

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
            int_cond = check_ratio(0.8)
            int_cond.terminal = True
        else:
            self.e_eq = np.sqrt(self.Te / (2 * self.j * np.abs(self.Tm)))
            int_cond = check_ratio(1.25)
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
    # This class will take two comparable mass planets and push the
    # inner one towards the outer one. We will have T_m,1 and T_e,1
    # and T_e,2.  Unlike the test particle classes I am not bothering
    # with setting exact initial resonance parameters with the
    # comparable mass case. There will only be migration in this case.
    #
    # Left to fill in are calculating variables for the
    # Hamiltonian. This is just a shell class right now.
    def __init__(self, j, mu1, q, a0, Tm, Te1, Te2):
        self.j = j
        self.mu1 = mu1
        self.q = q
        self.a0 = a0

        self.Tm = Tm
        self.Te1 = Te1
        self.Te2 = Te2

    def calc_alphanom(self):
        # nominal resonance location
        self.alphanom = (self.j / (self.j + 1)) ** (2.0 / 3.0)

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
        G1 = x1 * x1 + y1 * y1
        g2 = np.arctan2(y2, x2)
        G2 = x2 * x2 + y2 * y2

        j = self.j
        mu1 = self.mu1
        mu2 = self.mu1 / self.q

        e1 = np.sqrt(2 * G1 / L1)
        e2 = np.sqrt(2 * G2 / L2)

        # the Hamiltonian is -(constants) as the internal TP
        # Hamiltonian
        alpha1 = L1 * L1 / self.q / self.q
        alpha2 = L2 * L2
        theta1 = theta + g1
        theta2 = theta + g2
        f1 = -self.A(alpha)
        f2 = -self.B(alpha)

        l1dot = (1 / alpha1 / sqrt(alpha1)) + mu2 * f1 * e1 * cos(theta1) / (
            2 * alpha2 * sqrt(alpha1)
        )
        l2dot = (1 / alpha2 / sqrt(alpha2)) + self.q * mu2 / (alpha2 * sqrt(alpha2)) * (
            2 * f1 * e1 * cos(theta1) + 2.5 * f2 * e2 * cos(theta2)
        )

        Ldot_prefactor = (
            self.q * mu2 / alpha2 * (f1 * e1 * sin(theta1) + f2 * e2 * sin(theta2))
        )
        L1dot = j * Ldot_prefactor
        L2dot = -(j + 1) * Ldot_prefactor

        e1g1dot = -self.q * mu2 * f1 * cos(theta1) / alpha1 / alpha2
        e2g2dot = -self.q * mu2 * f2 * cos(theta2) / alpha2 / alpha2

        G1dot = self.q * mu2 * f1 * e1 * sin(theta1) / alpha2
        G2dot = -self.q * mu2 * f2 * e2 * sin(theta2) / alpha2

        x1dot = G1dot * cos(g1) - 0.5 * L1 * e1 * sin(g1) * e1g1dot
        y1dot = G1dot * sin(g1) + 0.5 * L1 * e1 * cos(g1) * e1g1dot

        x2dot = G2dot * cos(g2) - 0.5 * L2 * e2 * sin(g2) * e2g2dot
        y2dot = G2dot * sin(g2) + 0.5 * L2 * e2 * cos(g2) * e2g2dot

        thetadot = (j + 1) * l2dot - j * l1dot
        if self.migrate:
            # Here we're using time = tau*t
            # Add in the dissipative terms for migration

            # convert time units
            T0 = 2 * np.pi
            Tm1 = self.Tm * T0
            Te1 = self.Te1 * T0
            # Tm2 = infinity
            Te2 = self.Te2 * T0

            L1dot = L1dot + (L1 / 2) * (1 / Tm1 - 4 * G1 / L1 / Te1)
            L2dot = L2dot + (L2 / 2) * (-4 * G2 / L2 / Te2)

            x1dot = x1dot + cos(g1) * sqrt(G1) * (
                -1.0 / Te1 + 0.25 * (1.0 / Tm1 - 4 * G1 / Te1 / L1)
            )
            y1dot = y1dot + sin(g1) * sqrt(G1) * (
                -1.0 / Te1 + 0.25 * (1.0 / Tm1 - 4 * G1 / Te1 / L1)
            )

            x2dot = x2dot + cos(g2) * sqrt(G2) * (
                -1.0 / Te2 + 0.25 * (-4 * G2 / Te2 / L2)
            )
            y2dot = y2dot + sin(g2) * sqrt(G2) * (
                -1.0 / Te2 + 0.25 * (-4 * G2 / Te2 / L2)
            )

        return np.array([thetadot, L1dot, L2dot, x1dot, y1dot, x2dot, y2dot])

    def int_Hsec(self, t0, t1, tol):
        self.e_eq = np.sqrt(self.Te1 / (2 * (self.j + 1) * self.Tm))
        print("e1 eq =", self.e_eq)
        print("mu^1/3=", self.mu2 ** (1.0 / 3.0))

        G = 4 * np.pi ** 2
        self.scale_factor = G ** 2 * self.mu1 ** 1.5 * self.mu2 ** 1.5
        print(self.scale_factor)
        print(sqrt(self.scale_factor))
        t0 = t0
        t1 = t1

        # Star is 1 solar mass
        Lambda10 = self.mu1 * np.sqrt(G * self.a1) / sqrt(self.scale_factor)
        Lambda20 = self.mu2 * np.sqrt(G * self.a2) / sqrt(self.scale_factor)
        Gamma10 = Lambda10 * (0.5 * 0.0 ** 2)
        Gamma20 = Lambda20 * (0.5 * 0.0 ** 2)

        l10 = 0.0
        l20 = 0.0
        g10 = 0.0
        g20 = 0.0

        x10 = sqrt(Gamma10) * cos(g10)
        y10 = sqrt(Gamma10) * sin(g10)
        x20 = sqrt(Gamma20) * cos(g20)
        y20 = sqrt(Gamma20) * sin(g20)

        RHS = self.H4dofsec
        IV = (l10, Lambda10, l20, Lambda20, x10, y10, x20, y20)
        print("IV=", IV)
        teval = np.linspace(t0, t1, 1000) * sqrt(self.scale_factor)
        span = (teval[0], teval[-1])
        sol = sp.integrate.solve_ivp(
            RHS, span, IV, t_eval=teval, method="RK45", rtol=tol, atol=tol
        )

        l1 = sol.y[0, :]
        L1 = sol.y[1, :] * sqrt(self.scale_factor)
        l2 = sol.y[2, :]
        L2 = sol.y[3, :] * sqrt(self.scale_factor)
        x1 = sol.y[4, :]
        y1 = sol.y[5, :]
        x2 = sol.y[6, :]
        y2 = sol.y[7, :]

        g1 = np.arctan2(y1, x1)
        G1 = (x1 ** 2 + y1 ** 2) * sqrt(self.scale_factor)
        g2 = np.arctan2(y2, x2)
        G2 = (x2 ** 2 + y2 ** 2) * sqrt(self.scale_factor)

        theta1 = (self.j + 1) * l2 - self.j * l1 + g1
        theta2 = (self.j + 1) * l2 - self.j * l1 + g2

        # e1 = np.sqrt(2*G/L)
        e1 = np.sqrt(1 - (1 - G1 / L1) ** 2)
        e2 = np.sqrt(1 - (1 - G2 / L2) ** 2)
        a1 = (L1 / self.mu1) ** 2 / G
        a2 = (L2 / self.mu2) ** 2 / G

        teval = teval / sqrt(self.scale_factor)

        return (teval, a1, e1, G1, theta1, a2, e2, G2, theta2)
