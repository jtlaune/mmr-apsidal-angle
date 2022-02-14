from pathlib import Path
from .run import *
from .series import SimSeries
import numpy as np
import reboundx
import rebound
from multiprocessing import Pool, TimeoutError


def run_Nbody(
    verbose,
    tscale,
    secular,
    overwrite,
    method,
    suptitle,
    filename,
    figname,
    paramsname,
    h,
    q,
    mutot,
    a0,
    alpha2_0,
    T,
    Te,
    Tm,
    e10,
    e20,
    name,
    dirname,
    cutoff,
    g1_0,
    g2_0,
    N_tps,
):
    # Adapted from:
    # [[https://rebound.readthedocs.io/en/doctest/ipython/Megno.html][rebound
    # docs]]

    sim = rebound.Simulation()
    sim.add(m=1.0)  # Star
    sim.integrator = "whfast"

    # sim.ri_whfast.safe_mode = 0
    # sim.dt = 5.0

    if q not in [0.0, 1.0]:
        raise Warning("Nbody only implemented for q=[0.,1.0], i.e [internal, external]")
    if q == 0.0:
        # internal
        sim.add(m=mutot, a=alpha2_0, l=0.0, pomega=g2_0, e=e20)
        # add TPs at random l10
        for l10 in np.random.uniform(0.0, 2 * np.pi, N_tps):
            sim.add(m=0.0, a=a0, l=l10, pomega=g1_0, e=e10)
    if q == 1.0:
        # external
        sim.add(m=mutot, a=a0, l=0.0, pomega=g1_0, e=e10)
        for l20 in np.random.uniform(0.0, 2 * np.pi, int(N_tps)):
            sim.add(m=0.0, a=alpha2_0, l=l20, pomega=g2_0, e=e20)

    # from
    # https://rebound.readthedocs.io/en/latest/ipython_examples/Testparticles/
    sim.N_active = 2
    #

    sim.move_to_com()

    # sim.init_megno()
    sim.exit_max_distance = 20.0

    fpath = os.path.join(dirname, filename)
    fdir, fname = os.path.split(fpath)
    Path(fdir).mkdir(parents=True, exist_ok=True )
    print(fpath)
    sim.automateSimulationArchive(fpath, step=100)

    try:
        # Integrate for 2pi*T from params (i.e. T is in in units of [P0=(a0)^1.5]
        sim.integrate(T * 2.0 * np.pi, exact_finish_time=0)
        # integrate for 500 years, integrating to the nearest
        # timestep for each output to keep the timestep constant
        # and preserve WHFast's symplectic nature

        # megno = sim.calculate_megno()
        # return megno
    except rebound.Escape as err:
        # return 10.0  # At least one particle got ejected, returning large MEGNO.
        print("A particle escaped...")
        raise err


class NbodyTPSet(SimSet):
    params_spec = [
        "h",
        "q",
        "mutot",
        "a0",
        "alpha2_0",
        "T",
        "Te",
        "Tm",
        "e10",
        "e20",
        "name",
        "dirname",
        "cutoff",
        "g1_0",
        "g2_0",
        "N_tps",
    ]

    @params_load
    def __call__(self, params):
        name = self.params["name"]
        mutot = self.params["mutot"]
        q = self.params["q"]
        T = self.params["T"]
        filename = f"{name}.bin"
        figname = f"{name}.png"
        paramsname = f"params-{name}.txt"
        suptitle = (
            f"{filename}\n"
            f"T={T:0.1e} q={q} " + r"$\mu_{\rm tot}=$ " + f"{mutot:0.2e}\n"
        )

        run_Nbody(  # loaded from decorator
            self.verbose,
            self.tscale,
            self.secular,
            self.overwrite,
            self.method,
            suptitle,
            filename,
            figname,
            paramsname,
            **self.params,
        )

    # def run_Nbody().... params):
    #    take params and handoff to rebound


class NbodyMigTrapSeries(SimSeries):
    @series_dir
    def __call__(self, Nproc=8):
        print("__call__ is running")
        if self.load:
            self.load_all_runs()
        else:
            # change to series directory
            N_sims = self.RUN_PARAMS.shape[0]
            overwrite = not self.load
            integrate = NbodyTPSet(
                verbose=self.verbose, overwrite=self.overwrite, secular=True, method="RK45"
            )
            np.savez("RUN_PARAMS", self.RUN_PARAMS)

            with Pool(processes=min(Nproc, N_sims)) as pool:
                pool.map(integrate, self.RUN_PARAMS)
