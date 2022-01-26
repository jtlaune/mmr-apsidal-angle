from mpa import *

origindir, filename = os.path.split(__file__)
projectdir = os.path.join(origindir, "mpa/tests/")
os.chdir("mpa/tests/")
seriesname  = "disTscales"
series = SeriesFOCompmass(seriesname, projectdir, load=False)
series(1)
