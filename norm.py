# Normalizations
# Normalizing constants to convert between
# code units and cgs units. Conversion
# routines that convert in either direction
# given a variable are provided.
# Requires python >= 2.7 because of OrderedDict

from collections import OrderedDict

import numpy as np
import numpy.linalg as la


class PhysNorm():
    """
    Class that stores units (dimensions) and scaling factors for a selection of physical
    quantities which have dimensions are derivable from the set of five SI base quantities,
    length, mass, time, current, and temperature.

    Note, this class does not make any assumption of an equation of state, from
    which one could obtain the scaling factor of one unknown (e.g. temperature). 
    The class also doesn't assume between which two systems the scaling obtains.

    """

    def __init__(self, **kwargs):
        """
        kwargs          Varname - scaling value pairs. Currently supported 
                        varnames are those in the keys of self.defs, namely:
                        x, m, t, curr, temp, v, dens, pres, pmom, pdot, pflx, 
                        ener, epwr, eflx, eint, edot, cool, mdot, area, volume, 
                        newton
        """

        # Independent SI base dimensions
        self.dimdefs = ['length', 'mass', 'time', 'current', 'temperature']
        self.ndims = len(self.dimdefs)

        # The tuple of dimensions (fundamental, or base quantities) is
        # length, mass, time, current, and temperature
        # (L, M, T, A, K)
        # The number in the tuple determines the power of the dimension.
        #
        #           Dimensions [0]        Description [2]
        self.defs = OrderedDict([
            ('x', ((1, 0, 0, 0, 0), 'position or displacement')),
            ('m', ((0, 1, 0, 0, 0), 'mass')),
            ('t', ((0, 0, 1, 0, 0), 'time')),
            ('curr', ((0, 0, 0, 1, 0), 'electric current')),
            ('temp', ((0, 0, 0, 0, 1), 'temperature')),
            ('v', ((1, 0, -1, 0, 0), 'speed')),
            ('dens', ((-3, 1, 0, 0, 0), 'mass density')),
            ('pres', ((-1, 1, -2, 0, 0), 'pressure, or energy density')),
            ('pmom', ((1, 1, -1, 0, 0), 'linear momentum')),
            ('pdot', ((1, 1, -2, 0, 0), 'rate of change of linear momentum')),
            ('pflx', ((-1, 1, -2, 0, 0), 'linear momentum flux (pressure)')),
            ('ener', ((2, 1, -2, 0, 0), 'energy')),
            ('epwr', ((2, 1, -3, 0, 0), 'power or luminosity (energy per unit time)')),
            ('eflx', ((0, 1, -3, 0, 0), 'energy flux')),
            ('eint', ((2, 0, -2, 0, 0), 'specific (internal) energy, energy per unit mass')),
            ('edot', ((2, 0, -3, 0, 0), 'rate of change of specific internal energy density')),
            ('cool', ((5, 1, -3, 0, 0), 'rate of change of internal energy per unit density squared')),
            ('mdot', ((0, 1, -1, 0, 0), 'mass outflow/accretion/loading/etc rate')),
            ('area', ((2, 0, 0, 0, 0), 'area')),
            ('volume', ((3, 0, 0, 0, 0), 'volume')),
            ('newton', ((3, -1, -2, 0, 0), 'Newtons gravitational constant')),
            ('none', ((0, 0, 0, 0, 0), 'dimensionless quantity'))
        ])

        # Test if all keys are known.
        for k in kwargs:
            if k not in self.defs:
                raise ValueError('Error, Unknown key ' + k + '.')

        # Create ordered dictionary of kwargs
        kwargs_od = OrderedDict.fromkeys(self.defs)
        for k, v in kwargs_od.items():
            if k in kwargs:
                kwargs_od[k] = kwargs[k]
            else:
                kwargs_od.pop(k)
        self.kwargs = kwargs_od
        dims = [self.defs[k][0] for k, v in kwargs_od.items() if v is not None]

        # Test if one of the dimension powers is zero.
        # If so, issue error message and exit
        cs = map(lambda l: np.sum(map(abs, l)), zip(*dims))
        for i, s in enumerate(cs):
            if s == 0:
                raise ValueError('This set of scalings is incomplete. No finite dimension for ' + self.dimdefs[i] + '.')

        # Coefficient matrix
        cm = np.array(dims)

        # Calculate powers to construct all scalings
        scalings = OrderedDict()
        powers = OrderedDict.fromkeys(self.defs)
        for ip in powers:
            powers[ip] = la.solve(cm.T, np.array(self.defs[ip][0]))
            scalings[ip] = np.prod(np.array(kwargs_od.values()) ** powers[ip])
        self.powers = powers
        self.scalings = scalings

        # Create attribute for this class of each variable
        #self.__dict__.update(scalings)
        for k, v in scalings.items(): setattr(self, k, v)

        return

    def print_scalings(self):
        """
        Output a two column table of var name and scaling factor for all
        variables in this class.
        """
        for k, v in self.scalings.items(): print(format(k, '16s') + format(v, '>16.8e'))

