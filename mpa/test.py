from . import *
import unittest
import os.path


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
        self.assertAlmostEqual(FO.f1(alpha),2.0252226899385954)
        self.assertAlmostEqual(FO.f2(alpha),-2.4840051833039407)
        self.assertAlmostEqual(FO.f3(alpha),1.1527998000076145)
        self.assertAlmostEqual(FO.f4(alpha),-2.000522975124446)


    def test_SimSeries_init(self):
        # get run dir information
        # seriesname is simulation "series"
        try:
            print(self.origindir)
            #os.chdir("tests/")
            seriesname  = "init"
            series = FOCompmassSeries(seriesname, self.projectdir, load=False)
            self.assertTrue(isinstance(series, FOCompmassSeries))
        except FileNotFoundError as err:
            raise err
        finally:
            os.chdir(self.origindir)


    def test_compmass_disTscales(self):
        seriesname  = "disTscales"
        series = FOCompmassSeries(seriesname, self.projectdir, load=True)
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
