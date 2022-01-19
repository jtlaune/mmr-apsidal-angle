import numpy as np
import rebound
import sys
import os
import math

sys.path.append("/home/jtlaune/mmr/")
import LaplaceCoefficients as LC
import mmr_rebound
import run

def run_tp_ext_simple(ap, mup, ep, T, Te, Tm, om_eff, dirname, label):
    # Test particle outside of massive planet migrating inwards.  This
    # should work up to some inconsistencies with T_m and T_e
    # definitions in literature.

    figname = os.path.join(dirname, label+".png")
    filename = os.path.join(dirname, label+".bin")

    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    Mstar = 1.

    sim.add(m=Mstar)
    sim.add(m=0.0, e=0.0, a=1.35, inc=0., f=0., Omega=0., omega=0)
    sim.add(m=mup, e=ep, a=ap, inc=0., f=0., Omega=0., omega=0)
    sim.move_to_com()

    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    numpts = 1000

    sim.automateSimulationArchive(filename, interval=T/numpts,deletefile=True)
    print("Saving to directory... {}".format(dirname))

    orbits = sim.calculate_orbits()
    testpart = orbits[0]
    migforce = mmr_rebound.test_inwards_simple(np.abs(Tm), Te, omext=om_eff)
    print("T_e = {:0.2e}\n" \
          "T_m = {:0.2e}\n" \
          "om_ext = {:0.2e}".format(migforce.Te, migforce.Tm, migforce.omext))
    sim.additional_forces = migforce.force
    sim.force_is_velocity_dependent = 1 # need rebound to update velocities
    
    numstatus = 100
    for i in range(numstatus):
        sim.integrate(sim.t+T/numstatus)
        print("{:0.0f}%".format(i/numstatus*100), end="\r")
