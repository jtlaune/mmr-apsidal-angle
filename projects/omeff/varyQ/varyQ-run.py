#!/home/jtlaune/miniconda3/envs/science/bin/python
from multiprocessing import Pool, TimeoutError
import sys
import importlib
import importlib.util
import os

sys.path.append("/home/jtlaune/multi-planet-architecture/")
import run
import plotting
from plotting import plotsim, loadsim
from helper import *

# get run dir information
# seriesname is simulation "series"
abspath, filename = os.path.split(os.readlink(__file__))
seriesname = os.path.basename(filename).split("-")[0]
paramsname = seriesname+"-params.py"
runpath = os.path.join(abspath, seriesname)
print(abspath, filename)

def load_params(filepath):
    spec = importlib.util.spec_from_file_location("_", filepath)
    _ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_)
    print(f"Loading run file {filepath} in directory {os.getcwd()}")
    return(np.array(_.RUN_PARAMS))

if not os.path.exists(seriesname):
    os.mkdir(seriesname)

# change to run directory
os.chdir(seriesname)

#################
# Configuration #
#################
overwrite = True


###########
# Execute #
###########
Nproc=8
RUN_PARAMS = load_params(paramsname)
N_sims = RUN_PARAMS.shape[0]
integrate = run.run_compmass_set_omeff(verbose=True, overwrite=overwrite,
                                 secular=True, method="RK45")
np.savez("RUN_PARAMS", RUN_PARAMS)
print(RUN_PARAMS)
print(f"Running {N_sims} simulations...")

with Pool(processes=min(Nproc, N_sims)) as pool:
    pool.map(integrate, RUN_PARAMS)
