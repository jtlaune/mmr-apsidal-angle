#!/home/jtlaune/miniconda3/envs/science/bin/python
import time
import sys
import os
sys.path.append("/home/jtlaune/multi-planet-architecture/")
import mpa
from mpa.series import FOomEffTPSeries

# get run dir information
# seriesname is simulation "series"

cwdpath = os.path.abspath(os.getcwd())
#projpath, _ = os.path.split(cwdpath)
projpath, _ = os.path.split(cwdpath)

seriesname = "cutoff"
seriesdir = os.path.join(projpath, "tpOmEff")
paramsname = seriesname+"-params.py"
runpath = seriesdir
series = FOomEffTPSeries(seriesname, runpath, load=False,
                         verbose=True, overwrite=True)
#import pdb; pdb.set_trace()
start = time.time()
series(16)
end = time.time()
print("done in " + f"{end - start:0.2f}" + "s")
