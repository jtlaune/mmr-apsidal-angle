import importlib
import os
from .run import series_dir

# from .run import CompmassSet, TPSetOmeff
import numpy as np

# from .run import CompmassSetOmeff
from multiprocessing import Pool, TimeoutError
from .run import params_load

from .run import run_compmass
from .run import run_compmass_omeff
from .run import run_tp_omeff

##########################################################################
# Simulation sets, one+ parallel executions
##########################################################################


class SimSet(object):
    params = {}

    def __init__(
        self,
        verbose=False,
        overwrite=False,
        secular=True,
        tscale=1000.0,
        method="RK45",
    ):
        self.verbose = verbose
        self.overwrite = overwrite
        self.secular = secular
        self.method = method
        self.tscale = tscale


class CompmassSet(SimSet):
    params_seec = [
        "h",
        "j",
        "a0",
        "q",
        "mu1",
        "T",
        "Te1",
        "Te2",
        "Tm1",
        "Tm2",
        "e1_0",
        "e2_0",
        "e1d",
        "e2d",
        "alpha2_0",
        "name",
        "dirname",
        "cutoff",
        "g1_0",
        "g2_0",
    ]

    @params_load
    def __call__(self, params):
        name = self.params["name"]
        T = self.params["T"]
        Te1 = self.params["Te1"]
        Te2 = self.params["Te2"]
        e1d = self.params["e1d"]
        e2d = self.params["e2d"]
        Tm1 = self.params["Tm1"]
        Tm2 = self.params["Tm2"]
        q = self.params["q"]
        mu1 = self.params["mu1"]

        filename = f"{name}.npz"
        figname = f"{name}.png"
        paramsname = f"params-{name}.txt"
        suptitle = (
            f"{filename}\n"
            f"T={T:0.1e} q={q} " + r"$\mu_{1}=$ " + f"{mu1:0.2e}\n"
            f"Tm1={Tm1:0.1e} Te1={Te1:0.1e}\n"
            f"Tm2={Tm2:0.1e} Te2={Te2:0.1e}\n"
            r"$e_{1,d}$ = " + f"{e1d:0.3f} "
            r"$e_{2,d}$ = " + f"{e2d:0.3f}"
        )

        run_compmass(
            self.verbose,
            self.tscale,
            self.secular,
            self.overwrite,
            self.method,
            suptitle,
            filename,
            figname,
            paramsname,  # end of positional params
            **self.params,
        )


class CompmassSetOmeff(SimSet):
    params_spec = [
        "h",
        "j",
        "a0",
        "q",
        "mu1",
        "T",
        "Te1",
        "Te2",
        "Tm1",
        "Tm2",
        "e1_0",
        "e2_0",
        "e1d",
        "e2d",
        "alpha2_0",
        "name",
        "dirname",
        "cutoff",
        "g1_0",
        "g2_0",
        "omeff1",
        "omeff2",
    ]

    @params_load
    def __call__(self, params):  # params NEEDS to be here for
        # decorator to work.
        # TODO put params in argument of params_load
        name = self.params["name"]
        T = self.params["T"]
        Te1 = self.params["Te1"]
        Te2 = self.params["Te2"]
        Tm1 = self.params["Tm1"]
        Tm2 = self.params["Tm2"]
        q = self.params["q"]
        mu1 = self.params["mu1"]
        omeff1 = self.params["omeff1"]
        omeff2 = self.params["omeff2"]

        filename = f"{name}.npz"
        figname = f"{name}.png"
        paramsname = f"params-{name}.txt"
        suptitle = (
            f"{filename}\n"
            f"T={T:0.1e} q={q} " + r"$\mu_{1}=$ " + f"{mu1:0.2e}\n"
            f"Tm1={Tm1:0.1e} Te1={Te1:0.1e}\n"
            f"Tm2={Tm2:0.1e} Te2={Te2:0.1e}\n"
            r"$\omega_{\rm 1,ext}$ = " + f"{omeff1:0.3e}"
            r"$\omega_{\rm 2,ext}$ = " + f"{omeff2:0.3e}"
        )

        run_compmass_omeff(
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


class TPSetOmeff(SimSet):
    params_spec = [
        "h",
        "j",
        "a0",
        "q",
        "mup",
        "T",
        "Te1",
        "Te2",
        "Tm1",
        "Tm2",
        "e1_0",
        "e2_0",
        "e1d",
        "e2d",
        "alpha2_0",
        "name",
        "dirname",
        "cutoff",
        "g1_0",
        "g2_0",
        "om_ext",
        "om_pext",
    ]

    def __init__(
        self,
        verbose=False,
        overwrite=False,
        secular=True,
        tscale=1000.0,
        method="RK45",
        cresswell_Te=False,
    ):
        super().__init__(
            verbose=verbose,
            overwrite=overwrite,
            secular=secular,
            tscale=tscale,
            method=method,
        )
        self.cresswell_Te = cresswell_Te

    @params_load
    def __call__(self, params):  # params NEEDS to be here for
        # decorator to work.
        # TODO put params in argument of params_load
        name = self.params["name"]
        T = self.params["T"]
        Te1 = self.params["Te1"]
        Te2 = self.params["Te2"]
        Tm1 = self.params["Tm1"]
        Tm2 = self.params["Tm2"]
        q = self.params["q"]
        mup = self.params["mup"]
        om_ext = self.params["om_ext"]
        om_pext = self.params["om_pext"]

        filename = f"{name}.npz"
        figname = f"{name}.png"
        paramsname = f"params-{name}.txt"
        suptitle = (
            f"{filename}\n"
            f"T={T:0.1e} q={q} " + r"$\mu_{p}=$ " + f"{mup:0.2e}\n"
            f"Tm1={Tm1:0.1e} Te1={Te1:0.1e}\n"
            f"Tm2={Tm2:0.1e} Te2={Te2:0.1e}\n"
            # r"$\omega_{\rm 1,ext}$ = " + f"{omeff1:0.3e}"
            # r"$\omega_{\rm 2,ext}$ = " + f"{omeff2:0.3e}"
        )

        run_tp_omeff(
            self.verbose,
            self.tscale,
            self.secular,
            self.overwrite,
            self.method,
            self.cresswell_Te,
            suptitle,
            filename,
            figname,
            paramsname,
            **self.params,
        )


##########################################################################
# Simulation series, projects, "sections in a paper"
##########################################################################
class SimSeries(object):
    """
    - file management
    - setting up files, reading RUN_PARAMS from file
    - loading data from npz files
    - params=RUN_PARAMS is useful if working interactively
    """

    def __init__(
        self,
        name,
        seriesdir,
        load=False,
        secular=True,
        verbose=True,
        overwrite=True,
        loadall=True,
        params=None,  # if you want to directly supply RUN_PARAMS
    ):
        # self.RUN_PARAMS = load_params(paramsname)
        self.seriesname = name
        self.sdir = seriesdir
        self.paramsfpath = os.path.join(self.sdir, f"{self.seriesname}-params.py")
        self.data = {}
        self.load = load
        self.verbose = verbose
        self.overwrite = overwrite
        self.loadall = loadall
        self.secular = secular
        if params is None:
            self.initialize()
        else:
            self.RUN_PARAMS = params

    @series_dir
    def initialize(self):
        self.RUN_PARAMS = self.load_params(self.paramsfpath)
        if self.load:
            self.load_all_runs()

    def load_params(self, filepath):
        spec = importlib.util.spec_from_file_location("_", filepath)
        _ = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_)
        return np.array(_.RUN_PARAMS)

    def load_run(self, ind):
        params = self.RUN_PARAMS
        Nqs = len(params[:, 0])
        name = params[ind, 15]
        dirname = params[ind, 16]
        dirname = os.path.join(self.sdir, dirname)
        filename = f"{name}.npz"
        try:
            data = np.load(os.path.join(dirname, filename))
            self.data[ind] = data
        except FileNotFoundError as err:
            print(
                f"In directory {dirname} \n"
                f"cannot find file {filename}... have you run it?"
            )
            if self.loadall:
                raise err
            else:
                self.data[ind] = None

    def load_all_runs(self):
        params = self.RUN_PARAMS
        Nqs = len(params[:, 0])
        for ind in range(Nqs):
            self.load_run(ind)


class FOCompmassSeries(SimSeries):
    """
    - Class to run first order comparable mass simulations
    - define which physics to include for compmass, i.e. omeff, dissipative, etc
      - could possibly be defined in a separate file like fargo?
    """

    # This decorator takes care of the project/series/runsim file
    # management. intended to be executed from runsim (symbolic link)
    @series_dir
    def __call__(self, Nproc=8):
        if self.load:
            self.load_all_runs()
        else:
            N_sims = self.RUN_PARAMS.shape[0]
            integrate = CompmassSet(
                verbose=self.verbose,
                overwrite=self.overwrite,
                secular=self.secular,
                method="RK45",
            )

            with Pool(processes=min(Nproc, N_sims)) as pool:
                pool.map(integrate, self.RUN_PARAMS)


class FOomEffSeries(SimSeries):
    """
    - Class to run first order comparable mass simulations with omeff
    """

    # This decorator takes care of the project/series/runsim file
    # management. intended to be executed from runsim (symbolic link)
    @series_dir
    def __call__(self, Nproc=8):
        if self.load:
            self.load_all_runs()
        else:
            N_sims = self.RUN_PARAMS.shape[0]
            integrate = CompmassSetOmeff(
                verbose=self.verbose,
                overwrite=self.overwrite,
                secular=self.secular,
                method="RK45",
            )

            with Pool(processes=min(Nproc, N_sims)) as pool:
                pool.map(integrate, self.RUN_PARAMS)


class FOomEffTPSeries(SimSeries):
    """
    - Class to run first order TP simulations with omeff
    - if cresswell_Te==True: use Cresswell & Nelson 2008 e/h-dependent Te
    """

    def __init__(
        self,
        name,
        seriesdir,
        load=False,
        secular=True,
        verbose=True,
        overwrite=True,
        loadall=True,
        params=None,  # if you want to directly supply RUN_PARAMS
        cresswell_Te=False,
    ):
        super().__init__(
            name,
            seriesdir,
            load=load,
            secular=secular,
            verbose=verbose,
            overwrite=overwrite,
            loadall=loadall,
            params=params,
        )
        self.cresswell_Te = cresswell_Te

    # This decorator takes care of the project/series/runsim file
    # management. intended to be executed from runsim (symbolic link)
    @series_dir
    def __call__(self, Nproc=8):
        if self.load:
            self.load_all_runs()
        else:
            N_sims = self.RUN_PARAMS.shape[0]
            integrate = TPSetOmeff(
                verbose=self.verbose,
                overwrite=self.overwrite,
                secular=self.secular,
                cresswell_Te=self.cresswell_Te,
                method="RK45",
            )

            with Pool(processes=min(Nproc, N_sims)) as pool:
                pool.map(integrate, self.RUN_PARAMS)
