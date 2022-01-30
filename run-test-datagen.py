import mpa
import mpa.test
import os

origindir = mpa.test.ResonanceTestCase.projectdir
filename = mpa.test.ResonanceTestCase.filename
projectdir = mpa.test.ResonanceTestCase.projectdir

os.chdir("mpa/tests/")
seriesname = "disTscales"
series = mpa.FOCompmassSeries(seriesname, projectdir, load=False)
print(series.seriesname)
print(dir(series))
series()
