from . import *
import scipy as sp
import numpy as np
import unittest
import os.path
from .series import FOCompmassSeries
from .series import FOomEffSeries
from . import LaplaceCoefficients as LC
from . import nbody
from .fndefs import radNormNegpi, radNormZero
from .mpl_styles import analytic

class ResonanceTestCase(unittest.TestCase):
    origindir, filename = os.path.split(os.path.abspath(__file__))
    projectdir = os.path.join(origindir, "tests/")

    def test_module_import(self):
        # intentionally redundant. testing the import as i transition to a
        # module.
        self.assertEqual(0.0, 0.0)

    def test_FO_LaplaceCoefficients(self):
        j = 2.0
        alpha = (j / (j + 1)) ** (2.0 / 3)
        FO = FirstOrder(j)
        # self.assertAlmostEqual(FO.f1(alpha), -2.0)
        # self.assertAlmostEqual(FO.f2(alpha), 2.5)
        self.assertAlmostEqual(FO.f1(alpha), 2.0252226899385954)
        self.assertAlmostEqual(FO.f2(alpha), -2.4840051833039407)
        self.assertAlmostEqual(FO.f3(alpha), 1.1527998000076145)
        self.assertAlmostEqual(FO.f4(alpha), -2.000522975124446)

    def test_SimSeries_init(self):
        # get run dir information
        # seriesname is simulation "series"
        try:
            print(self.origindir)
            # os.chdir("tests/")
            seriesname = "init"
            seriesdir = os.path.join(self.projectdir, seriesname)
            series = FOCompmassSeries(seriesname, seriesdir, load=False)
            self.assertTrue(isinstance(series, FOCompmassSeries))
        except FileNotFoundError as err:
            raise err
        finally:
            os.chdir(self.origindir)

    def test_Compmass_disTscales(self):
        seriesname = "disTscales"
        seriesdir = os.path.join(self.projectdir, seriesname)
        series = FOCompmassSeries(seriesname, seriesdir, load=True)
        params = series.RUN_PARAMS
        #####################################
        Te1 = np.float64(params[0, 6])
        Te2 = np.float64(params[0, 7])
        Tm1 = np.float64(params[0, 8])
        Tm2 = np.float64(params[0, 9])
        alpha2_0 = np.float64(params[0, 14])
        #####################################
        rundata = series.data[0]
        print(rundata)

        teval = rundata["teval"]
        it = int(0.1 * len(teval))
        teval = teval[0:it]
        a1 = rundata["a1"][0:it]
        a2 = rundata["a2"][0:it]
        a1dot = np.gradient(a1, teval)
        a2dot = np.gradient(a2, teval)
        avg_a1_a1dot = np.average(a1 / a1dot)
        avg_a2_a2dot = np.average(a2 / a2dot)
        print(avg_a2_a2dot)
        print(Tm2)
        print(Tm2 / avg_a2_a2dot)
        self.assertTrue((avg_a2_a2dot - Tm2) < 0.001 * np.abs(Tm2))

        os.chdir(self.origindir)

    def test_Compmass_secular(self):
        seriesname = "secular"
        seriesdir = os.path.join(self.projectdir, seriesname)
        series = FOCompmassSeries(seriesname, seriesdir, load=True)
        params = series.RUN_PARAMS

        #######################################################################
        # params
        #######################################################################
        mu1 = np.float64(params[:, 4])
        q = np.float64(params[:, 3])
        mu2 = mu1/q
        a0 = np.float64(params[:, 2])
        j = np.float64(params[:, 1])
        #######################################################################

        fig, ax = plt.subplots()

        rundata = series.data[0]

        teval = rundata["teval"]
        it = int(0.3*len(teval))
        teval = teval[0:it]
        print(a0)
        a1 = rundata["a1"][0:it]
        a2 = rundata["a2"][0:it]
        g1 = radNormNegpi(rundata["g1"][0:it])
        g2 = radNormNegpi(rundata["g2"][0:it])
        dg = radNormNegpi(g1-g2)

        dotg1 = np.gradient(g1, teval)
        avgdotg1 = np.average(dotg1)
        dotg2 = np.gradient(g2, teval)
        avgdotg2 = np.average(dotg2)

        ax.set_title(r"$q=$" + f"{q}")

        ddg = np.gradient(dg, teval)
        ax.plot(teval, ddg, ls="--", c="k", label="data")
        #ax.plot(teval, oms, label="analytic")
        ax.legend()

        figfp = os.path.join(seriesdir, "test-secular.png")
        fig.savefig(figfp, bbox_inches="tight")

        os.chdir(self.origindir)

    def test_Compmass_eqecc(self):
        pass

    @mpl.rc_context(analytic)
    def test_Compmass_OmEff(self):
        seriesname = "omEff"
        seriesdir = os.path.join(self.projectdir, seriesname)
        series = FOomEffSeries(seriesname, seriesdir, load=True)
        params = series.RUN_PARAMS

        fig, ax = plt.subplots()
        q = params[0,3] # constant
        n = len(series.data)
        for i in range(n):
            prescomEff = float(params[i,-2]) # prescribed omext1
            print(i)
            sim = series.data[i]
            teval = sim["teval"]
            diffom = sim["g2"] - sim["g1"] # actual
            calcomEff = np.average(np.gradient(diffom, teval)) # average
            print(prescomEff)
            print(calcomEff)
                                                               # calculated
            ax.scatter(prescomEff, (calcomEff), c="k", s=10)
            # ax.scatter(teval, diffom, c="k", s=10)

        ax.legend()
        ax.set_title(r"$q=$" + f"{q}")
        ax.set_xlabel(r"prescribed $\omega_{\rm eff}$")
        ax.set_ylabel(r"simulation $\omega_{\rm eff}$")
        figfp = os.path.join(seriesdir, "test-omEff.png")
        fig.savefig(figfp, bbox_inches="tight")

        os.chdir(self.origindir)


class NbodyTestCase(unittest.TestCase):
    origindir, filename = os.path.split(os.path.abspath(__file__))
    projectdir = os.path.join(origindir, "tests/")

    def test_nbody_run(self):
        seriesname = "nbody"
        seriesdir = os.path.join(self.projectdir, seriesname)
        sim = nbody.NbodyMigTrapSeries(seriesname, seriesdir, load=False)
        sim(16)
