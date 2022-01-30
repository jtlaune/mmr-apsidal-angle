from . import *
from .run import *


class NbodyTPSet(SimSet):
    self.params_spec = [
        "h",
        "q",
        "mutot",
        "delta_a",
        "a01",
        "a02",
        "T",
        "Te1",
        "Te2",
        "Tm1",
        "Tm2",
        "name",
    ]

    @params_load
    def __call__(self):
        filename = f"{name}.npz"
        figname = f"{name}.png"
        paramsname = f"params-{name}.txt"
        suptitle = ( f"{filename}\n" f"T={T:0.1e} q={q} " +
                     r"$\mu_{\rm tot}=$ " + f"{mutot:0.2e}\n" )

        # def run_Nbody().... params):
        #    take params and handoff to rebound
        # run_Nbody(
        #     params...
        #     verbose=self.verbose,
        #     secular=self.secular,
        #     overwrite=self.overwrite,
        #     method=self.method,
        # )

class NbodyMigTrapSeries(SimSeries):
    @series_dir
    def __call__(self, Nproc=8):
        # change to series directory
        N_sims = self.RUN_PARAMS.shape[0]
        overwrite = not self.load
        integrate = NbodySet(
            verbose=True, overwrite=overwrite, secular=True, method="RK45"
        )
        np.savez("RUN_PARAMS", self.RUN_PARAMS)

        with Pool(processes=min(Nproc, N_sims)) as pool:
            pool.map(integrate, self.RUN_PARAMS)
