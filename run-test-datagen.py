import mpa
import mpa.series
import mpa.test
import os

origindir = mpa.test.ResonanceTestCase.projectdir
filename = mpa.test.ResonanceTestCase.filename
testdir = mpa.test.ResonanceTestCase.projectdir

# run these in the test directory
print(testdir)
os.chdir(testdir)

#seriesname = "disTscales"
#seriesdir = os.path.join(testdir, seriesname)
#series = mpa.series.FOCompmassSeries(seriesname, seriesdir, load=False)
#series()
#
#seriesname = "omEff"
#seriesdir = os.path.join(testdir, seriesname)
#series = mpa.series.FOCompmassSeries(seriesname, seriesdir, load=False)
#series()

seriesname = "secular"
seriesdir = os.path.join(testdir, seriesname)
series = mpa.series.FOCompmassSeries(seriesname, seriesdir, load=False)
series()
