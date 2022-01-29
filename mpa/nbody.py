from . import *

def run_Nbody(#.... params):
    #take params and handoff to rebound

class NbodySet(SimSet):
    
    def __call__(self, params):
        # h = params[0]
        # ...
        # ...
        # etc
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
        run_Nbody(verbose=self.verbose, secular=self.secular,
                  overwrite=self.overwrite, method=self.method)


class NbodyMigTrapSeries(SimSeries):
    @series_dir
    def __call__(self, Nproc=8):
        # change to series directory
        N_sims = self.RUN_PARAMS.shape[0]
        overwrite = not self.load
        integrate = NbodySet(verbose=True,
                             overwrite=overwrite,
                             secular=True,
                             method="RK45")
        np.savez("RUN_PARAMS", self.RUN_PARAMS)
        
        with Pool(processes=min(Nproc, N_sims)) as pool:
            pool.map(integrate, self.RUN_PARAMS)

