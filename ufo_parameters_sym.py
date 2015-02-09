# Requires python >= 2.7 because of OrderedDict

# The sympy version. Totally not finished, barely begun.

from sy.__future__ import division
import physconst as pc
import numpy as np
import norm
import eos
from collections import OrderedDict
import sympy as sy


class CompositionUfo(eos.CompositionBase):
    """
    Currently identical class to Eos/CompositionBase
    """

    def __init__(self, mu=0.6165):
        eos.CompositionBase.__init__(self, mu)


class CompositionISM(eos.CompositionBase):
    """
    Currently identical class to Eos/CompositionBase
    """

    def __init__(self, mu=0.6165):
        eos.CompositionBase.__init__(self, mu)


class UfoParams():
    """
    This class contains functions to calculate parameters of a relativistic ufo
    and corresponding non-relativistic ufo. Everything is in cgs. If other units
    are required, PhysNorm classes should be used.
    """

    def __init__(self,
                 power=1.e44,
                 angle=30,
                 speed=0.1,
                 mdot=0.01,
                 rufo=0.01,
                 wufo=0.002,
                 dens_ambient=1.0,
                 temp_ambient=1.e7,
                 gamma=1.6666666666,
                 pmode=0,
                 dmode=0,
                 norm=norm.PhysNorm(x=pc.kpc, t=pc.kyr, dens=0.6165 * pc.amu,
                                    temp=(pc.kpc / pc.kyr) ** 2 * pc.amu / pc.kboltz, curr=1)
    ):
        """
        Parameters

          power                Ufo power (erg s^-1)
          angle                Angle with disc (in degrees)
          speed                Ufo speed (in c)
          mdot                 Ufo mass outflow rate (Msun/yr)
          rufo                 Ufo radius (kpc)
          wufo                Ufo outflow annulus thickness (kpc)
          temp_ambient         Reference temperature of background ISM (K).
          dens_ambient         Reference density of background ISM (amu*mu, mean
                               mass per particle).
          gamma                Adiabiatic index non-relativistic
          pmode                Mode of calculation w.r.t pressure:
                               0: pressure is calculated from parameters
                                  assuming given power denotes the total power
                                  P = 1/2 mdot v^2 + gm/(gm-1) A p v
                               1: pressure is set to pressure equilibrium
                                  with surrounding, power given is assumed to be
                                  P = 1/2 mdot v^2 + gm/(gm-1) A p v
                                  The outflow speed v is adjusted
                               2: pressure is set to pressure equilibrium
                                  with surrounding, power given is assumed to be
                                  P = 1/2 mdot v^2 + gm/(gm-1) A p v
                                  The mass outflow rate mdot is adjusted
                               3: pressure is set to pressure equilibrium
                                  with surrounding, power given is assumed to be
                                  P = 1/2 mdot v^2. The outflow speed v is
                                  adjusted. The total power is calculated.
                               4: pressure is set to pressure equilibrium
                                  with surrounding, power given is assumed to be
                                  P = 1/2 mdot v^2. The mass outflow rate mdot is
                                  adjusted. The total power is calculated.
          dmode                Mode of calculation w.r.t density:
                               0: density is calculated from mdot = rho v A
                               1: density is set equal to that of surrounding
                                  medium. The outflow speed v is adjusted
                               2: density is set equal to that of surrounding
                                  medium. The mass outflow rate mdot is adjusted


          norm                 Normalization object for internal calculations

        Internally, everything is then converted into and handled in units of

        unit density = mean mass per particle, mu*amu
        unit length = kpc
        unit time = kyr
        unit temperature = (kpc/kyr)**2*amu/kboltz

        To recover cgs units or other units, use norm.PhysNorm class and physconst module.

        """

        # All attributes in class are stored in dictionary
        # self.__dict__ whose update function can be used to 
        # append it with a list of local variables.
        self.__dict__.update(locals())
        del self.__dict__['self']

        # Dictionary of variable names that were given
        args = self.__dict__.copy()

        # Dictionary of variable names, their type of units
        self.defs = OrderedDict([
            ('power', ('epwr', 'P', 'Ufo power')),
            ('angle', ('none', 'phi', 'Angle of UFO with disc')),
            ('speed', ('v', 'v', 'Ufo speed')),
            ('mdot', ('mdot', 'm_t', 'Ufo mass outflow rate')),
            ('rufo', ('x', 'r', 'Ufo radius')),
            ('wufo', ('x', 'w', 'Width of launching region')),
            ('temp_ambient', ('temp', 'T_a', 'Reference temperature of background ISM.')),
            ('dens_ambient', ('dens', 'rho_a', 'Reference density of background ISM.')),
            ('gamma', ('none', 'gamma', 'Adiabiatic index')),
            ('pres_ambient', ('pres', 'p_a', 'Reference pressure of background ISM.')),
            ('eint_ambient', ('eint', 'epsilon_a', 'Ambient internal energy')),
            ('vsnd_ambient', ('v', 'a_a', 'Ambient sound speed')),
            ('area', ('area', 'A', 'Outflow area')),
            ('pres', ('pres', 'p', 'Ufo pressure')),
            ('dens', ('dens', 'rho', 'Ufo density')),
            ('temp', ('temp', 'T', 'Ufo temperature')),
            ('mach', ('none', 'M', 'Ufo mach number')),
            ('eflx', ('eflx', 'F', 'Ufo energy flux.')),
            ('pratio', ('none', 'xi', 'Pressure ratio (ufo/ISM).')),
            ('dratio', ('none', 'zeta', 'Density ratio (ufo/ISM).')),
            ('eint', ('eint', 'epsilon', 'Ufo Specific internal energy')),
            ('enth', ('eint', 'h', 'Ufo Specific enthalpy')),
            ('pdot', ('pdot', 'f_t', 'Ufo Momentum injection rate.')),
            ('pflx', ('pres', 'f_A', 'Ufo Momentum flux.')),
            ('r1', ('x', 'r_1', 'Inner wind launching radius.')),
            ('r2', ('x', 'r_2', 'Outer wind launching radius.')),
            ('delta', ('x', 'Delta', 'w projected onto disc.'))
        ])

        # Capture normalization object
        self.norm = norm

        # Create and keep a composition object for ufo
        self.uc = CompositionUfo()
        self.muu = self.uc.mu

        # Create and keep a composition object for ambient gas
        self.ic = CompositionISM()
        self.mua = self.ic.mu

        # Create an EOS object for ufo
        self.eosu = eos.EOSIdeal(comp=self.uc)

        # Create an EOS object for ambient gas
        self.eosa = eos.EOSIdeal(comp=self.ic)

        # Correct all input parameters into normalization units here first.

        # Ufo power needs to be converted to code units first
        args['power'] = args['power'] / getattr(self.norm, self.defs['power'][0])
        args['speed'] = args['speed'] * pc.c / getattr(self.norm, self.defs['speed'][0])
        args['mdot'] = args['mdot'] * pc.msun / pc.yr / getattr(self.norm, self.defs['mdot'][0])
        args['temp_ambient'] = args['temp_ambient'] / getattr(self.norm, self.defs['temp_ambient'][0])
        self.__dict__.update(args)
        self.args = args

        # Create sympy symbols for all vars
        self.syv = self.defs.fromkeys()
        for key, val in self.defs.items():
            self.syv[key] = sy.Symbol(val[1])
            setattr(self, val[1], sy.Symbol(val[1]))

        # The equations relevant to this class (These expressions equal zero)
        eq_power = self.P - (0.5 * self.mdot * self.v ** 2 +
                             self.gamma / (self.gamma - 1) *
                             self.A * self.p * self.v)

        eq_mdot = self.mdot - self.rho * self.v * self.A

        eq_ekin = self.K - 0.5 * self.mdot * self.v ** 2






