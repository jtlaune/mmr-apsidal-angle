#!/home/jtlaune/miniconda3/envs/science/bin/python
import sys
import os
sys.path.append("/home/jtlaune/multi-planet-architecture/")
import mpa
from mpa.series import FOomEffSeries

# get run dir information
# seriesname is simulation "series"
abspath, filename = os.path.split(os.path.abspath(os.readlink(__file__)))
seriesname = os.path.basename(filename).split("-")[0]
paramsname = seriesname+"-params.py"
runpath = abspath

series = FOomEffSeries(seriesname, runpath, load=False)
series(16)
