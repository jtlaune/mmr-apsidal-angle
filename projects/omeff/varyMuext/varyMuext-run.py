#!/home/jtlaune/miniconda3/envs/science/bin/python
import sys
import os
sys.path.append("/home/jtlaune/multi-planet-architecture/")
import mpa
from mpa.series import FOomEffSeries

# get run dir information
# seriesname is simulation "series"

# for symlink execution
#abspath, filename = os.path.split(os.path.abspath(os.readlink(__file__)))
#seriesname = os.path.basename(filename).split("-")[0]

cwdpath = os.path.abspath(os.getcwd())
projpath, _ = os.path.split(cwdpath)
print(_)
seriesname = "varyMuext"
seriesdir = os.path.join(projpath, seriesname)
paramsname = seriesname+"-params.py"
runpath = seriesdir

series = FOomEffSeries(seriesname, runpath, load=False)
#import pdb; pdb.set_trace()
series(16)
