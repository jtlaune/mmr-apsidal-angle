import numpy as np
from astropy import constants as const
Msun_g = const.M_sun.cgs.value
au_cm = const.au.cgs.value

class test_inwards_simple:
    # Simplest possible case. Massive outer planet (p=primed) on fixed
    # arbitrary orbit. Initial inner planet at a0. Migration timescale
    # Tm. No eccentricity damping timescale. Units in msun, yr, au.
    # Must have test particle first in simulation.
    def __init__(self, Tm, Te):
        self.Tm = Tm
        self.Te = Te
    def force(self, reb_sim):
        ps = reb_sim.contents.particles
        star = ps[0]
        tpart = ps[1]

        if tpart.m > 0:
            raise Warning("Test particle must come first in simulation.")

        # T_m
        tpart.ax += -tpart.vx*(1/self.Tm/2)
        tpart.ay += -tpart.vy*(1/self.Tm/2)
        tpart.az += -tpart.vz*(1/self.Tm/2)

        r2 = tpart.x*tpart.x + tpart.y*tpart.y + tpart.z*tpart.z
        vdotr = tpart.x*tpart.vx + tpart.y*tpart.vy + tpart.z*tpart.vz

        # T_e
        tpart.ax += -2*tpart.x*vdotr*(1/self.Te/r2)
        tpart.ay += -2*tpart.y*vdotr*(1/self.Te/r2)
        tpart.az += -2*tpart.z*vdotr*(1/self.Te/r2)

class test_outwards_simple:
    def __init__(self, Tm, Te):
        self.Tm = Tm
        self.Te = Te
    def force(self, reb_sim):
        ps = reb_sim.contents.particles
        star = ps[0]
        tpart = ps[1]

        if tpart.m > 0:
            raise Warning("Test particle must come first in simulation.")

        # T_m
        tpart.ax += tpart.vx*(1/self.Tm/2)
        tpart.ay += tpart.vy*(1/self.Tm/2)
        tpart.az += tpart.vz*(1/self.Tm/2)

        r2 = tpart.x*tpart.x + tpart.y*tpart.y + tpart.z*tpart.z
        vdotr = tpart.x*tpart.vx + tpart.y*tpart.vy + tpart.z*tpart.vz

        # T_e
        tpart.ax += -2*tpart.x*vdotr*(1/self.Te/r2)
        tpart.ay += -2*tpart.y*vdotr*(1/self.Te/r2)
        tpart.az += -2*tpart.z*vdotr*(1/self.Te/r2)




class test_onlyTe:
    def __init__(self, Te):
        self.Te = Te
    def force(self, reb_sim):
        ps = reb_sim.contents.particles
        star = ps[0]
        tpart = ps[1]

        if tpart.m > 0:
            raise Warning("Test particle must come first in simulation.")

        r2 = tpart.x*tpart.x + tpart.y*tpart.y + tpart.z*tpart.z
        vdotr = tpart.x*tpart.vx + tpart.y*tpart.vy + tpart.z*tpart.vz

        # T_e
        tpart.ax += -2*tpart.x*vdotr*(1/self.Te/r2)
        tpart.ay += -2*tpart.y*vdotr*(1/self.Te/r2)
        tpart.az += -2*tpart.z*vdotr*(1/self.Te/r2)




class test_outwards_realistic:
    def __init__(self, Mstar, mmotion, ap, mup, Tmp, e_eq0, j, Hr, Tresratio=10):
        # calculates realistic (eccentricity dependent) forces for
        # massive/massless particle near a j:j+1 MMR according to
        # Xu+2018 setup. assumes beta=0. must have test particle first
        # in simulation. This calculates timescales wrt these inputs
        # in the same way for Xu+2018 Fig. 3-7
        
        # Mstar: mass star in [Msun]
        # ap: semi-major axis of outer planet
        # mup: mass ratio outer planet
        # Sigma: disk surface density g/cm^2
        # e_eq0: equilibrium eccentricity
        # j: j:j+1 mmr
        # Hr: scale height of disk (assumed constant)
        # Tresratio: Te0/Tres
        
        self.mup = mup
        self.Tmp = Tmp
        self.e_eq0 = e_eq0
        self.Hr = Hr
        self.j = j
        # Calculate initial time constants

        # Check if we're getting realistic Sigma values
        Sigma = Hr*Hr*Mstar/Tmp/ap**0.5/mup/2/np.pi # Msun/au^2 i think
        Sigma = Sigma*Msun_g/au_cm**2 # convert msun/au^2
        # we're keeping Tmp constant for simplicity
        #self.Tmp = ((Mstar**Hr**2)/(Sigma*ap**2*mup))*ap**(3/2)/2/np.pi/2.7

        Tres0 = 0.8*self.j**(-4/3)*self.mup**(-2/3)*(2*np.pi/mmotion)
        self.Te0 = Tresratio*Tres0
        self.Tm0 = abs(1/(1/self.Tmp - 2*(self.j+1)*self.e_eq0**2/self.Te0))
        Tmeff = (1/self.Tmp - 1/self.Te0)**(-1)
        
        print("Sigma        = {} g/cm^2\n" \
              "a20          = {} au\n" \
              "j            = {} au\n" \
              "mu           = {}\n" \
              "e_eq,0       = {}\n" \
              "e_eq,0/h     = {}\n" \
              "mu'/e_eq,0^3 = {}\n" \
              "Tres  = {}\n" \
              "Te0   = {}\n" \
              "Tm0   = {}\n" \
              "Tm'   = {}\n" \
              "Tm_eff= {}".format(Sigma, ap, j, mup, e_eq0, e_eq0/Hr,
                                  mup/e_eq0**3, Tres0, self.Te0,
                                  self.Tm0, self.Tmp, Tmeff))

    def force(self, reb_sim):
        ps = reb_sim.contents.particles
        star = ps[0]
        tpart = ps[1]
        mplan = ps[2]

        # Xu+2018 eq. 7
        # Nelson & Cresswell 2008 corrections

        # Eccentricity damping force
        e = tpart.e
        Te = self.Te0*(1-0.14*(e/self.Hr)**2+0.06*(e/self.Hr)**3)
        vdotr = tpart.vx*tpart.x + tpart.vy*tpart.y + tpart.vz*tpart.z
        r2 = tpart.x*tpart.x + tpart.y*tpart.y + tpart.z*tpart.z
        preTe = 2*vdotr/r2/Te
        
        # Modify Tm for test particle eccentricity
        Tm = self.Tm0 \
             *(1+(e/2.25/self.Hr)**1.2+(e/2.84/self.Hr)**6) \
             /(1-(e/2.02/self.Hr)**4)
        #Tmeffinv = (2*(self.j+1)*self.e_eq0*self.e_eq0/Te)

        if len(ps) > 3:
            raise Warning("This routine only deals with one " \
                          "test particle and one massive particle.")

        tpart.ax += tpart.vx*(-1/Tm - 2*e*e/Te + 1/self.Tmp) - preTe*tpart.x
        tpart.ay += tpart.vy*(-1/Tm - 2*e*e/Te + 1/self.Tmp) - preTe*tpart.y
        tpart.az += tpart.vz*(-1/Tm - 2*e*e/Te + 1/self.Tmp) - preTe*tpart.z

class test_particle_simple:
    def __init__(self, planets, Tmprime, e_eq0, j, Hr):
        # calculates simple (constant) forces for massive/massless
        # particle near a j:j+1 MMR according to Xu+2018
        # setup. assumes beta=0. must have test particle first in
        # simulation. Since m=0, this calculates Tm0 as proportional
        # to Te0
        
        # planets: planets from rebound simulation
        # Tmprime: migration timescale of massive outer particle
        # e_eq0: equilibrium eccentricity
        # j: j:j+1 mmr
        # Hr: scale height of disk (assumed constant)

        # Calculate initial Nelson Cresswell timescales
        #self.Tm = np.zeros(len(planets))
        #self.Te = np.zeros(len(planets))
        self.Te = 2*e_eq0*e_eq0*(j+1)*Tmprime
        self.Tm = 1.56*self.Te/2.7/Hr/Hr
        self.Tmp = Tmprime
        #print(Te0, Tm0, Tmprime)

        print("Simple migration test particle case")
        print("Tm' is {:10.2f}x Te0".format(Tmprime/self.Te))
        print("Tm' is {:10.2f}x Tm0".format(Tmprime/self.Tm))

    def force(self, reb_sim):
        ps = reb_sim.contents.particles
        star = ps[0]
        tpart = ps[1]
        mplan = ps[2]

        # Xu+2018 eq. 7
        # Nelson & Cresswell 2008 corrections

        # Eccentricity damping force
        e = tpart.e
        #Te = self.Te0*(1-0.14*(e/self.Hr)**2+0.06*(e/self.Hr)**3)
        vdotr = tpart.vx*tpart.x + tpart.vy*tpart.y + tpart.vz*tpart.z
        r2 = tpart.x*tpart.x + tpart.y*tpart.y + tpart.z*tpart.z
        preTe = 2*vdotr/r2/self.Te
        
        # Modify Tm for test particle eccentricity
        #Tm = self.Tm0 \
        #     *(1+(e/2.25/self.Hr)**1.2+(e/2.84/self.Hr)**6) \
        #     /(1-(e/2.02/self.Hr)**4)
        #Tmeffinv = (2*(self.j+1)*self.e_eq0*self.e_eq0/Te)

        if len(ps) > 3:
            raise Warning("This routine only deals with one " \
                          "test particle and one massive particle.")

        tpart.ax += tpart.vx*(-1/self.Tm - 2*e*e/self.Te + 1/self.Tmp) - preTe*tpart.x
        tpart.ay += tpart.vy*(-1/self.Tm - 2*e*e/self.Te + 1/self.Tmp) - preTe*tpart.y
        tpart.az += tpart.vz*(-1/self.Tm - 2*e*e/self.Te + 1/self.Tmp) - preTe*tpart.z
