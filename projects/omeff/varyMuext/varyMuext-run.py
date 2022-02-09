#!/home/jtlaune/miniconda3/envs/science/bin/python
import sys
import os
sys.path.append("/home/jtlaune/multi-planet-architecture/")
import mpa
from mpa.series import FOomEffSeries

# get run dir information
# seriesname is simulation "series"

cwdpath = os.path.abspath(os.getcwd())
#projpath, _ = os.path.split(cwdpath)
projpath = cwdpath

seriesname = "varyMuext"
seriesdir = os.path.join(projpath, seriesname)
paramsname = seriesname+"-params.py"
runpath = seriesdir

series = FOomEffSeries(seriesname, runpath, load=False)
#import pdb; pdb.set_trace()
series(16)
