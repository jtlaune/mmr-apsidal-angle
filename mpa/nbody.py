from .run import *
import reboundx
import rebound


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
        suptitle = (
            f"{filename}\n"
            f"T={T:0.1e} q={q} " + r"$\mu_{\rm tot}=$ " + f"{mutot:0.2e}\n"
        )
        self.run_Nbody(self.params)  # loaded from decorator

    # def run_Nbody().... params):
    #    take params and handoff to rebound
    # run_Nbody(
    #     params...
    #     verbose=self.verbose,
    #     secular=self.secular,
    #     overwrite=self.overwrite,
    #     method=self.method,
    # )

    # in past, have had run_* functions as standalone definitions.
    # now it seems like this is the more logical place
    def run_Nbody(self, params):
        # Adapted from:
        # [[https://rebound.readthedocs.io/en/doctest/ipython/Megno.html][rebound
        # docs]]
        # unpack parameters
        h = params["h"]
        j = params["j"]
        a0 = params["a0"]
        q = params["q"]
        mu1 = params["mu1"]
        T = params["t"]
        Te1 = params["te1"]
        Te2 = params["te2"]
        Tm1 = params["tm1"]
        Tm2 = params["tm2"]
        e1_0 = params["e1_0"]
        e2_0 = params["e2_0"]
        e1d = params["e1d"]
        e2d = params["e2d"]
        alpha2_0 = params["alpha2_0"]
        name = params["name"]
        dirname = params["dirname"]
        cutoff = params["cutoff"]
        g1_0 = params["g1_0"]
        g2_0 = params["g2_0"]

        sim = rebound.Simulation()
        sim.integrator = "whfast"
        sim.ri_whfast.safe_mode = 0
        sim.dt = 5.0
        sim.add(m=1.0)  # Star
        sim.add(m=0.000954, a=5.204, M=0.600, omega=0.257, e=0.048)
        sim.add(m=0.0, a=a, M=0.871, omega=1.616, e=e)
        sim.move_to_com()

        sim.init_megno()
        sim.exit_max_distance = 20.0
        try:
            sim.integrate(
                5e2 * 2.0 * np.pi, exact_finish_time=0
            )  # integrate for 500 years, integrating to the nearest
            # timestep for each output to keep the timestep constant and preserve WHFast's symplectic nature
            megno = sim.calculate_megno()
            return megno
        except rebound.Escape:
            return 10.0  # At least one particle got ejected, returning large MEGNO.


class NbodyMigTrapSeries(SimSeries):
    @series_dir
    def __call__(self, Nproc=8):
        # change to series directory
        N_sims = self.RUN_PARAMS.shape[0]
        overwrite = not self.load
        integrate = NbodyTPSet(
            verbose=True, overwrite=overwrite, secular=True, method="RK45"
        )
        np.savez("RUN_PARAMS", self.RUN_PARAMS)

        with Pool(processes=min(Nproc, N_sims)) as pool:
            pool.map(integrate, self.RUN_PARAMS)
