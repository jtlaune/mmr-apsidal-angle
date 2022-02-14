import importlib
import os
from .run import series_dir
from .run import CompmassSet, TPSetOmeff
import numpy as np
from .run import CompmassSetOmeff
from multiprocessing import Pool, TimeoutError


##########################################################################
# Simulation series, projects, "sections in a paper"
##########################################################################
class SimSeries(object):
    """
    - file management
    - setting up files, reading RUN_PARAMS from file
    - loading data from npz files
    """

    def __init__(self, name, seriesdir, load=False, verbose=True, overwrite=True):
        # self.RUN_PARAMS = load_params(paramsname)
        self.seriesname = name
        self.sdir = seriesdir
        self.paramsfpath = os.path.join(self.sdir, f"{self.seriesname}-params.py")
        self.data = {}
        self.load = load
        self.verbose = verbose
        self.overwrite = overwrite
        self.initialize()

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
            print(f"Cannot find file {filename}... have you run it?")
            raise err

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
                secular=True,
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
                secular=True,
                method="RK45",
            )

            with Pool(processes=min(Nproc, N_sims)) as pool:
                pool.map(integrate, self.RUN_PARAMS)


class FOomEffTPSeries(SimSeries):
    """
    - Class to run first order TP simulations with omeff
    """

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
                secular=True,
                method="RK45",
            )

            with Pool(processes=min(Nproc, N_sims)) as pool:
                pool.map(integrate, self.RUN_PARAMS)
