import mpa
import mpa.series
import mpa.test
import os

origindir = mpa.test.ResonanceTestCase.projectdir
filename = mpa.test.ResonanceTestCase.filename
testdir = mpa.test.ResonanceTestCase.projectdir

# run these in the test directory
os.chdir(testdir)

#seriesname = "disTscales"
#seriesdir = os.path.join(testdir, seriesname)
#series = mpa.series.FOCompmassSeries(seriesname, seriesdir, load=False)
#series()

#seriesname = "omEff"
#seriesdir = os.path.join(testdir, seriesname)
#series = mpa.series.FOomEffSeries(seriesname, seriesdir, load=False)
#series()

#seriesname = "secular"
#seriesdir = os.path.join(testdir, seriesname)
#series = mpa.series.FOCompmassSeries(seriesname, seriesdir, load=False)
#series()

#seriesname = "nosecular"
#seriesdir = os.path.join(testdir, "secular")
#series = mpa.series.FOCompmassSeries(seriesname, seriesdir, load=False, secular=False)
#series()

seriesname = "TPsecular"
seriesdir = os.path.join(testdir, "secular")
series = mpa.series.FOomEffTPSeries(seriesname, seriesdir, load=False)
series()

#seriesname = "tpDisTscales"
#seriesdir = os.path.join(testdir, seriesname)
#series = mpa.series.FOomEffTPSeries(seriesname, seriesdir, load=False)
#series()

#seriesname = "tpOmEff"
#seriesdir = os.path.join(testdir, seriesname)
#series = mpa.series.FOomEffTPSeries(seriesname, seriesdir, load=False)
#series()
