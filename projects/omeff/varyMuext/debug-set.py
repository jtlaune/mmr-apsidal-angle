import sys
import numpy as np
sys.path.append("/home/jtlaune/multi-planet-architecture/")
import mpa
from mpa.run import run_compmass_omeff
RUN_PARAMS = np.load("RUN_PARAMS.npz")
RUN_PARAMS = RUN_PARAMS.f.arr_0
params = []
for par in RUN_PARAMS[0]:
    try:
        params = params + [float(par)]
    except ValueError:
        params = params + [str(par)]
print(params)

verbose = True
tscale = 1000.
secular = True
overwrite = True
method = "DOP853"
suptitle = "debugging run"
filename = "debug.npz"
figname = "debug.png"
paramsname = "params-debug.txt"

run_compmass_omeff(
    verbose,
    tscale,
    secular,
    overwrite,
    method,
    suptitle,
    filename,
    figname,
    paramsname,
    *params
)
