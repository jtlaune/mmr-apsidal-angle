from . import *


class NbodyMigTrapSeries(SimSeries):
    def __call__(self, Nproc=8):
        # change to series directory
        if not os.path.exists(self.seriesname):
            os.mkdir(self.seriesname)
        os.chdir(self.seriesname)
        print(os.getcwd())

        N_sims = self.RUN_PARAMS.shape[0]

        overwrite = not self.load
        integrate = CompmassSetOmeff(verbose=True,
                                       overwrite=overwrite,
                                       secular=True,
                                       method="RK45")
        np.savez("RUN_PARAMS", self.RUN_PARAMS)
        print(self.RUN_PARAMS)
        print(f"Running {N_sims} simulations...")
        
        with Pool(processes=min(Nproc, N_sims)) as pool:
            pool.map(integrate, self.RUN_PARAMS)
        os.chdir(self.pdir)

class NbodySet(SimSet):
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
