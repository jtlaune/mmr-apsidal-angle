from pathlib import Path
from .run import *
from .series import SimSeries
import numpy as np
import reboundx
import rebound
from multiprocessing import Pool, TimeoutError

from rebound import InterruptiblePool


def run_NbodyMigTrap(
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
    g1_0,
    g2_0,
    N_tps,
    Nout,
    da_tps,
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
            if da_tps > 0: a_tp0 = np.random.uniform(a0-da_tps, a0+da_tps)
            else: a_tp0 = a0
            sim.add(m=0.0, a=a_tp0, l=l10, pomega=g1_0, e=e10)
    if q == 1.0:
        # external
        sim.add(m=mutot, a=a0, l=0.0, pomega=g1_0, e=e10)
        for l20 in np.random.uniform(0.0, 2 * np.pi, int(N_tps)):
            if da_tps > 0: a_tp0 = np.random.uniform(alpha2_0-da_tps, alpha2_0+da_tps)
            else: a_tp0 = alpha2_0
            sim.add(m=0.0, a=a_tp0, l=l20, pomega=g2_0, e=e20)

    # from
    # https://rebound.readthedocs.io/en/latest/ipython_examples/Testparticles/
    sim.N_active = 2
    sim.move_to_com()

    # sim.init_megno()
    sim.exit_max_distance = 3.0

    fpath = os.path.join(dirname, filename)
    fdir, fname = os.path.split(fpath)
    Path(fdir).mkdir(parents=True, exist_ok=True)
    print(fpath)

    # sim.automateSimulationArchive(fpath, step=100, deletefile=overwrite)

    ######################################################################
    # Initiate reboundx                                                  #
    ######################################################################
    rebx = reboundx.Extras(sim)
    # Add the migration force
    rebx = reboundx.Extras(sim)
    mof = rebx.load_force("modify_orbits_forces")
    rebx.add_force(mof)

    # add exponential mig to each test particle
    if q == 0.0:
        # internal
        for particle in sim.particles[2:]:
            particle.params["tau_a"] = Tm
            particle.params["tau_e"] = -Te
    elif q == 1.0:
        # external
        for particle in sim.particles[2:]:
            particle.params["tau_a"] = -Tm
            particle.params["tau_e"] = -Te

    try:
        # Integrate for 2pi*T from params (i.e. T is in in units of
        # [P0=(a0)^1.5]
        Nout, N_tps = int(Nout), int(N_tps)
        times = np.linspace(0.0, T, Nout)
        smas = np.zeros((Nout, N_tps))
        eccs = np.zeros((Nout, N_tps))
        lams = np.zeros((Nout, N_tps))
        poms = np.zeros((Nout, N_tps))
        for i, time in enumerate(times):
            print(f"{100*i/Nout:0.2f}%", end="\r")
            sim.integrate(time * 2.0 * np.pi)#, exact_finish_time=0)
            smas[i, :] = np.array([particle.a for particle in sim.particles[2:]])
            eccs[i, :] = np.array([particle.e for particle in sim.particles[2:]])
            lams[i, :] = np.array([particle.l for particle in sim.particles[2:]])
            poms[i, :] = np.array([particle.l for particle in sim.particles[2:]])
        # integrate for T years, integrating to the nearest
        # timestep for each output to keep the timestep constant
        # and preserve WHFast's symplectic nature

        # megno = sim.calculate_megno()
        # return megno
    except KeyboardInterrupt as err:
        print("Keyboard interrupted...")
    except rebound.Escape as err:
        # return 10.0  # At least one particle got ejected, returning large MEGNO.
        print("A particle escaped...")
    finally:
        # exit gracefully
        np.savez(fpath, teval=times, smas=smas, eccs=eccs, lams=lams, poms=poms)


class NbodyMigTrapSet(SimSet):
    # uses randomized li's for TPs but sets initial gi's to gi_0
    params_spec = [
        "h",
        "q",
        "mutot",
        "a0",  # location of inner body or bodies
        "alpha2_0",  # ratio of outer body/bodies to a0
        "T",
        "Te",
        "Tm",
        "e10",
        "e20",
        "name",
        "dirname",
        "g1_0",  # initial pericenter inner body/bodies
        "g2_0",  # initial pericenter outer body/bodies
        "N_tps",  # number of TPs to initialize
        "Nout",  # number of outputs to save
        "da_tps", # 1/2 full width
    ]

    @params_load
    def __call__(self, params):
        name = self.params["name"]
        mutot = self.params["mutot"]
        q = self.params["q"]
        T = self.params["T"]
        filename = f"{name}.npz"
        figname = f"{name}.png"
        paramsname = f"params-{name}.txt"
        suptitle = (
            f"{filename}\n"
            f"T={T:0.1e} q={q} " + r"$\mu_{\rm tot}=$ " + f"{mutot:0.2e}\n"
        )

        run_NbodyMigTrap(  # loaded from decorator
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
    def load_run(self, ind):
        params = self.RUN_PARAMS
        name = params[ind, 10]
        dirname = params[ind, 11]
        dirname = os.path.join(self.sdir, dirname)
        filename = f"{name}.npz"
        try:
            #data = rebound.SimulationArchive(os.path.join(dirname, filename))
            data = np.load(os.path.join(dirname, filename))
            self.data[ind] = data
        except FileNotFoundError as err:
            print(f"Cannot find file {filename}... have you run it?")
            raise err

    @series_dir
    def __call__(self, Nproc=8):
        print("__call__ is running")
        if self.load:
            self.load_all_runs()
        else:
            # change to series directory
            N_sims = self.RUN_PARAMS.shape[0]
            overwrite = not self.load
            integrate = NbodyMigTrapSet(
                verbose=self.verbose,
                overwrite=self.overwrite,
                secular=True,
                method="RK45",
            )
            np.savez("RUN_PARAMS", self.RUN_PARAMS)

            # Rebound's "pool" works as a drop in replacement
            # which responds better to C-c kbd interruptions
            with InterruptiblePool(processes=min(Nproc, N_sims)) as pool:
                pool.map(integrate, self.RUN_PARAMS)
