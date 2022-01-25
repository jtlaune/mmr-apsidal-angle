import sys
import os
#!/home/jtlaune/miniconda3/envs/science/bin/python
sys.path.append("/home/jtlaune/multi-planet-architecture/")
import mpa

# get run dir information
# seriesname is simulation "series"
abspath, filename = os.path.split(os.path.abspath(os.readlink(__file__)))
seriesname = os.path.basename(filename).split("-")[0]
paramsname = seriesname+"-params.py"
runpath = abspath

series = mpa.SetFOCompmassOmeff(seriesname, runpath, load=False)
series(16)
