from . import fndefs as fns
import numpy as np
from numpy import sqrt, cos, sin, pi
import scipy as sp
from . import LaplaceCoefficients as LC


class resonance(object):
    # Parent class for j:j+k resonance with useful functions
    # that sets secular/external effects
    def __init__(self, j, datafn=None):
        self.j = 2
        self.loaded = False

    def f3(self, alpha):
        return fns.sqr_ei_lc(alpha)

    def f4(self, alpha):
        return fns.eiej_lc(alpha)


class FirstOrder(resonance):
    """
    - No indirect terms implemented (MD p330, only applicable for 2:1)
    - j+1:j resonance, for j>1
    - MD p228: ' is outer secondary, unprimed is inner secondary

    - notes on f1 and f2:
        sign so Hamiltonian takes form:
        H = -H_kep -H_res -Hsec

    - also serves as the
    """

    def is_loaded(self, func):
        def wrapper():
            if self.loaded:
                func()
            else:
                raise Warning("No data in simulation")

        return wrapper

    def f1(self, alpha):
        return -fns.f27lc(alpha, self.j)

    def f2(self, alpha):
        return -fns.f31lc(alpha, self.j)


class FOCompMass(FirstOrder):
    # This class will integrate two planets with mass ratio q=m1/m2.
    # We will have T_m1,2 and T_e1,2 as parameters.
    def __init__(
        self,
        j,
        mu1,
        q,
        a0,
        Tm1,
        Tm2,
        Te1,
        Te2,
        e1d=None,
        e2d=None,
        cutoff=np.infty,
        Te_func=False,
    ):
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
        if np.any(np.isnan(Y)):
            raise Warning("nans detected")

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
        f1 = self.f1(alpha)
        f2 = self.f2(alpha)
        C = self.f3(alpha)
        D = self.f4(alpha)

        ###################
        # Resonant forces #
        ###################
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
            l1dot_sec = (
                mu2
                / alpha2
                / sqrt(alpha1)
                * (2 * C * e1 * e1 + D * e1 * e2 / 2 * cos(g1 - g2))
            )
            l2dot_sec = (
                self.q
                * mu2
                / alpha2
                / sqrt(alpha2)
                * (
                    (
                        2 * C * e1 * e1
                        + 3 * C * e2 * e2
                        + 2.5 * D * e1 * e2 / 2 * cos(g1 - g2)
                    )
                )
            )
            G1dot_sec = -self.q * mu2 * D * e1 * e2 / alpha2 * sin(g1 - g2)
            G2dot_sec = self.q * mu2 * D * e1 * e2 / alpha2 * sin(g1 - g2)

            e1g1dot_sec = -mu2 / alpha2 / sqrt(alpha1) * (2 * C * e1 + D * e2)
            e2g2dot_sec = -self.q * mu2 / alpha2 / sqrt(alpha2) * (2 * C * e2 + D * e1)

            x1dot_sec = G1dot_sec * cos(g1) - 0.5 * L1 * e1 * sin(g1) * e1g1dot_sec
            y1dot_sec = G1dot_sec * sin(g1) + 0.5 * L1 * e1 * cos(g1) * e1g1dot_sec

            x2dot_sec = G2dot_sec * cos(g2) - 0.5 * L2 * e2 * sin(g2) * e2g2dot_sec
            y2dot_sec = G2dot_sec * sin(g2) + 0.5 * L2 * e2 * cos(g2) * e2g2dot_sec

        else:
            l1dot_sec = 0.0
            l2dot_sec = 0.0

            G1dot_sec = 0.0
            G2dot_sec = 0.0

            e1g1dot_sec = 0.0
            e2g2dot_sec = 0.0

            x1dot_sec = 0.0
            y1dot_sec = 0.0

            x2dot_sec = 0.0
            y2dot_sec = 0.0

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
                    Te1 = self.Te1 / (e1 - self.e1d)
                if self.e2d:
                    Te2 = self.Te2 / (e2 - self.e2d)

            L1dot_dis = (L1 / 2) * (1 / Tm1 - 2 * e1 * e1 / Te1)
            L2dot_dis = (L2 / 2) * (1 / Tm2 - 2 * e2 * e2 / Te2)

            L1dot = L1dot + L1dot_dis
            L2dot = L2dot + L2dot_dis

            G1dot_dis = (L1dot_dis * G1 / L1) - 2 * G1 / Te1
            G2dot_dis = (L2dot_dis * G2 / L2) - 2 * G2 / Te2

            x1dot = x1dot + cos(g1) * G1dot_dis
            y1dot = y1dot + sin(g1) * G1dot_dis

            x2dot = x2dot + cos(g2) * G2dot_dis
            y2dot = y2dot + sin(g2) * G2dot_dis

        if self.verbose:
            print(
                (
                    "a1: {:0.2f} "
                    "a2: {:0.2f} "
                    "alpha: {:0.2f} "
                    "th1: {:0.2f} "
                    "th2: {:0.2f} "
                    "%: {:0.2f} ".format(
                        (L1 / self.q) ** 2,
                        L2**2,
                        (L1 / L2 / self.q) ** 2,
                        (theta1 % (2 * pi)),
                        (theta2 % (2 * pi)),
                        100.0 * t / self.T,
                    )
                ),
                end="\r",
            )

        return np.array([thetadot, L1dot, L2dot, x1dot, y1dot, x2dot, y2dot])

    def int_Hsec(
        self,
        t1,
        tol,
        alpha2_0,
        e1_0,
        e2_0,
        g1_0,
        g2_0,
        verbose=False,
        secular=True,
        method="RK45",
    ):
        self.secular = secular
        print(self.secular)
        self.verbose = verbose
        self.T = self.T0 * t1
        int_cond_min = fns.check_ratio_cm(0.6, self.q)
        int_cond_max = fns.check_ratio_cm(0.9, self.q)

        Lambda1_0 = self.q * 1
        Lambda2_0 = sqrt(alpha2_0)

        # set initial eccentricities
        G1_0 = 0.5 * Lambda1_0 * e1_0**2
        G2_0 = 0.5 * Lambda2_0 * e2_0**2

        # initial pomegas
        # g10 = -pi/4
        # g20 = 3*pi/4
        g10 = g1_0
        g20 = g2_0

        x1_0 = G1_0 * cos(g10)
        y1_0 = G1_0 * sin(g10)
        x2_0 = G2_0 * cos(g20)
        y2_0 = G2_0 * sin(g20)
        IV = (0, Lambda1_0, Lambda2_0, x1_0, y1_0, x2_0, y2_0)

        g1_0 = np.arctan2(y1_0, x1_0)
        g2_0 = np.arctan2(y2_0, x2_0)

        RHS = self.H4dofsec

        teval = np.linspace(0.0, t1, 300000) * self.T0
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
        G1 = sqrt(x1**2 + y1**2)

        g2 = np.arctan2(y2, x2)
        G2 = sqrt(x2**2 + y2**2)

        e1 = np.sqrt(1 - (1 - G1 / L1) ** 2)
        e2 = np.sqrt(1 - (1 - G2 / L2) ** 2)
        a1 = L1**2 * self.a0 / self.q**2
        a2 = L2**2 * self.a0
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


class FOCompMassOmeff(FOCompMass):
    def __init__(
        self,
        j,
        mu1,
        q,
        a0,
        Tm1,
        Tm2,
        Te1,
        Te2,
        e1d=None,
        e2d=None,
        cutoff=np.infty,
        Te_func=False,
        omeff1=0.0,
        omeff2=0.0,
    ):
        super().__init__(
            j,
            mu1,
            q,
            a0,
            Tm1,
            Tm2,
            Te1,
            Te2,
            e1d=None,
            e2d=None,
            cutoff=np.infty,
            Te_func=False,
        )
        self.omeff1 = omeff1 / self.T0
        self.omeff2 = omeff2 / self.T0
        self.perturb = False
        if np.abs(self.omeff1) > 0.0:
            self.perturb = True
        elif np.abs(self.omeff2) > 0.0:
            self.perturb = True

    def H4dofsec(self, t, Y):
        (thetadot, L1dot, L2dot, x1dot, y1dot, x2dot, y2dot) = super().H4dofsec(t, Y)

        if self.perturb:
            ################### old stuff
            # 7 variables
            L1 = Y[1]
            L2 = Y[2]
            x1 = Y[3]
            y1 = Y[4]
            x2 = Y[5]
            y2 = Y[6]

            alpha = (L1 / L2) ** 2
            a2 = L2**2 * self.a0
            a1 = alpha * a2

            g1 = np.arctan2(y1, x1)
            G1 = sqrt(x1 * x1 + y1 * y1)
            g2 = np.arctan2(y2, x2)
            G2 = sqrt(x2 * x2 + y2 * y2)

            e1 = np.sqrt(1 - (1 - G1 / L1) ** 2)
            e2 = np.sqrt(1 - (1 - G2 / L2) ** 2)

            j = self.j

            # the Hamiltonian is -(constants) as the internal TP
            # Hamiltonian, i.e. it matches MD
            alpha1 = L1 * L1 / self.q / self.q
            alpha2 = L2 * L2
            alpha = alpha1 / alpha2

            ################### new stuff

            # doing this in the frame of the outer planet
            g1dotext = -self.omeff1
            g2dotext = -self.omeff2

            x1dot = x1dot - g1dotext * y1
            y1dot = y1dot + g1dotext * x1
            x2dot = x2dot - g2dotext * y2
            y2dot = y2dot + g2dotext * x2

            # see parent for the ldot secular force from mup these are
            # just ldot secular forcing from ext. important for fine
            # detail but not implemented
            # thetadot = thetadot + (j + 1) * l2dotext - j * l1dotext

        ###################

        return np.array([thetadot, L1dot, L2dot, x1dot, y1dot, x2dot, y2dot])


class FOTestPartOmeff(FirstOrder):
    # This class integrates the Hamiltonian for a test
    # particle with migration.
    def __init__(self, j, mup, ep, e0, ap, g0, a0, lambda0, cutoff_frac):
        # set other params and dirname in child classes
        self.j = j
        self.mup = mup
        self.ep = ep
        self.e0 = e0
        self.ap = ap
        self.g0 = g0
        self.a0 = a0
        self.lambda0 = lambda0
        self.i_step = 0
        self.cutoff_frac = cutoff_frac

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
        theta = thetap + g 

        # tploc=int
        if self.Tm > 0:
            alpha = L * L
            dtheta_dl = -j
            A = self.f1(alpha)
            B = self.f2(alpha)
        # tploc=ext
        else:
            alpha = 1.0 / (L * L)
            dtheta_dl = j + 1
            A = alpha * self.f2(alpha)
            B = alpha * self.f1(alpha)

        # secular components
        C = self.f3(alpha)
        D = self.f4(alpha)

        if self.i_step < 1:
            print(Y, alpha)
            self.i_step = self.i_step + 1

        e = np.sqrt(1 - (1 - G / L) ** 2)

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

        if self.migrate and (t < self.cutoff_time):
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
            xdot = xdot + self.omEff * sqrt(G) * sin(g)
            ydot = ydot - self.omEff * sqrt(G) * cos(g)

        # TODO: NEED TO INCLUDE THESE in final study
        # see above for the ldot secular force from mup
        # these are just ldot secular forcing from ext
        # if self.ldotExtSecForcing:
        # ldotext = (
        #    -self.muext
        #    * L
        #    * (self.ap * self.ap / self.aext / self.aext)
        #    * LC.Db(0.5, 0, a / self.aext)
        # )
        # lpdotext = (
        #    -(self.ap * self.ap / self.aext / self.aext)
        #    * self.muext
        #    * LC.Db(0.5, 0, self.ap / self.aext)
        # )
        # if self.Tm > 0:
        #    thetapdot = thetapdot + (j + 1) * lpdotext - j * ldotext
        # if self.Tm < 0:
        #    thetapdot = thetapdot - j * lpdotext + (j + 1) * ldotext
        print(
            (
                "alpha: {:0.2f}    "
                "theta: {:0.2f}    "
                "done%: {:0.2f}".format(
                    alpha,
                    (theta % (2 * pi)),
                    100.0 * t / self.T,
                )
            ),
            end="\r",
        )

        return np.array([thetapdot, Ldot, xdot, ydot])

    def int_Hsec(self, t0, t1, tol, Tm=None, Te=None, om_eff=None):
        # TEMP: testing out various values of thetap0. need to change this back and not commit it
        #thetap0 = np.random.rand() * 2 * np.pi
        #
        thetap0 = np.pi
        # Here we're using time = tau*t
        self.migrate = False
        if Tm is not None and Te is not None:
            self.migrate = True
            self.Tm = Tm
            self.Te = Te

        self.perturb = False
        # if muext is not None and aext is not None:
        self.omEff = om_eff
        if self.omEff is not None:
            self.perturb = True

        self.n_p = 2 * np.pi / sqrt(self.ap)
        # have to use tau = n_p, since anything else changes the
        # scaling of the Hamiltonian and the variables. messes results
        # up. can fix by adjusting EoM accordingly
        self.tau = self.n_p

        self.T = t1 * self.tau
        self.cutoff_time = self.cutoff_frac*self.T

        # integration conditions
        # internal
        int_cond_min1 = fns.check_ratio_tp(0.5)
        int_cond_max1 = fns.check_ratio_tp(0.9)

        # external
        int_cond_min2 = fns.check_ratio_tp(1 / 0.5)
        int_cond_max2 = fns.check_ratio_tp(1 / 0.9)

        Lambda0 = np.sqrt(self.a0 / self.ap)
        Gamma0 = Lambda0 * (1 - np.sqrt(1 - self.e0**2))

        RHS = self.H2dofsec
        IV = (
            thetap0,
            Lambda0,
            sqrt(Gamma0) * cos(self.g0),
            sqrt(Gamma0) * sin(self.g0),
        )
        # scaled H by GM/ap => tau -> sqrt(GM/ap) t; Lambda -> Lambda/sqrt(GM/ap)
        teval = np.linspace(t0, t1, 300000) * self.tau
        span = (teval[0], teval[-1])
        print(self.Tm, self.Te)
        sol = sp.integrate.solve_ivp(
            RHS,
            span,
            IV,
            method="RK45",
            t_eval=teval,
            rtol=tol,
            atol=tol,
            dense_output=True,
            events=[int_cond_min1, int_cond_max1, int_cond_min2, int_cond_max2],
        )

        thetap = sol.y[0, :]
        lenSolns = len(thetap)  # for simulations which exited early
        teval = teval[:lenSolns] / self.tau
        L = sol.y[1, :]
        x = sol.y[2, :]
        y = sol.y[3, :]
        g = np.arctan2(y, x)
        G = x**2 + y**2
        e = np.sqrt(1 - (1 - G / L) ** 2)
        a = L**2 * self.ap

        return (
            teval,
            thetap,
            a,
            L,
            e,
            x,
            y,
            g,
            G,
        )

    def evecsec(self, t, Y):
        x = Y[0]
        y = Y[1]
        a = Y[2]
        ap = self.ap
        e = sqrt(x**2 + y**2)

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
        teval = np.linspace(t0, t1, 300000)
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
        e = sqrt(x**2 + y**2)

        return (teval, e, gamma, a)
