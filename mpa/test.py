from . import *
import numpy as np
import unittest
import os.path
from .series import FOCompmassSeries
from .series import FOomEffSeries
from . import LaplaceCoefficients as LC


class ResonanceTestCase(unittest.TestCase):
    origindir, filename = os.path.split(os.path.abspath(__file__))
    projectdir = os.path.join(origindir, "tests/")

    def test_module_import(self):
        # intentionally redundant. testing the import as i transition to a
        # module.
        self.assertEqual(0.,0.)


    def test_FO_LaplaceCoefficients(self):
        j = 2.
        alpha = (j/(j+1))**(2./3)
        FO = FirstOrder(j)
        #self.assertAlmostEqual(FO.f1(alpha), -2.0)
        #self.assertAlmostEqual(FO.f2(alpha), 2.5)
        self.assertAlmostEqual(FO.f1(alpha), 2.0252226899385954)
        self.assertAlmostEqual(FO.f2(alpha), -2.4840051833039407)
        self.assertAlmostEqual(FO.f3(alpha), 1.1527998000076145)
        self.assertAlmostEqual(FO.f4(alpha), -2.000522975124446)


    def test_SimSeries_init(self):
        # get run dir information
        # seriesname is simulation "series"
        try:
            print(self.origindir)
            #os.chdir("tests/")
            seriesname  = "init"
            seriesdir = os.path.join(self.projectdir, seriesname)
            series = FOCompmassSeries(seriesname, seriesdir, load=False)
            self.assertTrue(isinstance(series, FOCompmassSeries))
        except FileNotFoundError as err:
            raise err
        finally:
            os.chdir(self.origindir)


    def test_Compmass_disTscales(self):
        seriesname  = "disTscales"
        seriesdir = os.path.join(self.projectdir, seriesname)
        series = FOCompmassSeries(seriesname, seriesdir, load=True)
        params = series.RUN_PARAMS
#####################################
        Te1 = np.float64(params[0,6])
        Te2 = np.float64(params[0,7])
        Tm1 = np.float64(params[0,8])
        Tm2 = np.float64(params[0,9])
        alpha2_0 = np.float64(params[0,14])
#####################################
        rundata = series.data[0]
        print(rundata)

        teval = rundata["teval"]
        it = int(0.1*len(teval))
        teval = teval[0:it]
        a1 = rundata["a1"][0:it]
        a2 = rundata["a2"][0:it]
        a1dot = np.gradient(a1, teval)
        a2dot = np.gradient(a2, teval)
        avg_a1_a1dot = np.average(a1/a1dot)
        avg_a2_a2dot = np.average(a2/a2dot)
        print(avg_a2_a2dot)
        print(Tm2)
        print(Tm2/avg_a2_a2dot)
        self.assertTrue((avg_a2_a2dot-Tm2) < 0.001*np.abs(Tm2))

        os.chdir(self.origindir)


    def test_Compmass_secular(self):
        pass


    def test_Compmass_eqEccs(self):
        pass


    def test_Compmass_OmEff(self):
        seriesname  = "omEff"
        seriesdir = os.path.join(self.projectdir, seriesname)
        series = FOomEffSeries(seriesname, seriesdir, load=True)
        params = series.RUN_PARAMS

#####################################
        Te1 = np.float64(params[0, 6])
        Te2 = np.float64(params[0, 7])
        Tm1 = np.float64(params[0, 8])
        Tm2 = np.float64(params[0, 9])
        alpha2_0 = np.float64(params[0, 14])
#####################################

        Nruns = len(series.data)
        halfN = int(Nruns/2)

        QS =    np.float64(params[:,3])
        AEXTS =    np.float64(params[:,-1])
        MUEXTS =   np.float64(params[:,-2])
        NAMES =    params[:,-7]
        ALPHA2_0 = np.float64(params[:, -8])
        G1_0 = np.float64(params[:, -4]) % (2*np.pi)
        G2_0 = np.float64(params[:, -3]) % (2*np.pi)
        len_pre = len('omeff-')
        len_num = 10
        OMEFFS = np.array([float(name[len_pre:len_pre+len_num]) for name in NAMES])

        fig, ax = plt.subplots()
        # varying a_ext np.linspace(2., 8, halfN)
        for jit in range(halfN):
            rundata = series.data[jit]
            teval = rundata["teval"]
            it = int(len(teval))
            teval = teval[0:it]
            a1 = rundata["a1"][0:it]
            a2 = rundata["a2"][0:it]
            g1 = rundata["g1"][0:it] % (2*np.pi)
            g2 = rundata["g2"][0:it] % (2*np.pi)

            g1 = g1 - 2*np.pi*(g1>(2*np.pi))
            g2 = g2 - 2*np.pi*(g2>(2*np.pi))

            dotg1 = np.gradient(g1, teval)
            avgdotg1 = np.average(dotg1)
            dotg2 = np.gradient(g2, teval)
            avgdotg2 = np.average(dotg2)

            a0 = ALPHA2_0[jit]
            g20 = G2_0[jit]
            print(a0)
            print(a2)
            alphaext = AEXTS[jit]/a0
            alpha2 = a2/a0
            L2 = np.sqrt(alpha2)

            dotg2_ext = -np.average((0.25 * (1 / L2) * MUEXTS[jit] * (a2 / alphaext) *
                         LC.b(1.5, 1, a2 / alphaext)))

            ax.scatter(teval, g2, s=1)
            ax.scatter(teval, g20+dotg2_ext*teval, s=0.1, c="k", ls="--")
            #ax.scatter(MUEXTS[jit], dotg2_ext/avgdotg2, c="r")

        # varying mu_ext np.logspace(-3, -2, halfN)

        ax.set_title(r"$q=$"+f"{QS[jit]}")
        ax.set_ylabel(r"$\gamma_2$")
        # ax.set_yscale("log")
        figfp = os.path.join(seriesdir, "test-omEff.png")
        fig.savefig(figfp, bbox_inches="tight")

        os.chdir(self.origindir)

