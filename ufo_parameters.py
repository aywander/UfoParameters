## Requires python >= 2.7 because of OrderedDict
import physconst as pc
import numpy as np
import norm
import eos
from collections import OrderedDict


class CompositionUfo(eos.CompositionBase):
    '''
    Currently identical class to Eos/CompositionBase
    '''
    def __init__(self, mu=0.6165):
        eos.CompositionBase.__init__(self, mu)


class CompositionISM(eos.CompositionBase):
    '''
    Currently identical class to Eos/CompositionBase
    '''
    def __init__(self, mu=0.6165):
        eos.CompositionBase.__init__(self, mu)

 
class UfoParams():
    ''' 
    This class contains functions to calculate parameters of a relativistic ufo
    and corresponding non-relativistic ufo. Everything is in cgs. If other units
    are required, PhysNorm classes should be used.
    '''
    def __init__(self,
                 power=4.e43,
                 angle=30,
                 speed=0.03,
                 mdot=0.1,
                 rufo=0.000001,
                 wufo=0.0000002,
                 dens_ambient=1.0,
                 temp_ambient=1.e7,
                 gamma=1.6666666666,
                 norm=norm.PhysNorm(x=pc.kpc, t=pc.kyr, dens=0.6165*pc.amu,
                                    temp=(pc.kpc/pc.kyr)**2*pc.amu/pc.kboltz, curr=1)
                ):
        '''
        Parameters

          power                Ufo power (erg s^-1)
          angle                Polar angle (in degrees)
          speed                Ufo speed (in c)
          mdot                 Ufo mass outflow rate (Msun/yr)
          rufo                 Ufo radius (kpc)
          wufo                 Ufo outflow annulus thickness (kpc)
          temp_ambient         Reference temperature of background ISM (K).
          dens_ambient         Reference density of background ISM (amu*mu, mean 
                               mass per particle). 
          gamma                Adiabiatic index non-relativistic
          norm                 Normalization object for output. 
                               Internally all units are brought to this base too.

        '''

        ## All attributes in class are stored in dictionary
        ## self.__dict__ whose update function can be used to 
        ## append it with a list of local variables.
        self.__dict__.update(locals()); del self.__dict__['self']

        ## Dictionary of variable names that were given
        args = self.__dict__.copy()

        ## Dictionary of variable names, their type of units
        self.defs = OrderedDict([
        ('power'          , ('epwr', 'Ufo power'                                  )),
        ('angle'          , ('none', 'Polar angle of UFO'                         )),
        ('speed'          , ('v'   , 'Ufo speed'                                  )),
        ('mdot'           , ('mdot', 'Ufo mass outflow rate'                      )),
        ('rufo'           , ('x'   , 'Ufo radius'                                 )),
        ('wufo'           , ('x'   , 'Width of launching region'                  )),
        ('temp_ambient'   , ('temp', 'Reference temperature of background ISM.'   )),
        ('dens_ambient'   , ('dens', 'Reference density of background ISM.'       )),
        ('gamma'          , ('none', 'Adiabiatic index'                           )),
        ('pres_ambient'   , ('pres', 'Reference pressure of background ISM.'      )),
        ('eint_ambient'   , ('eint', 'Ambient internal energy'                    )),
        ('vsnd_ambient'   , ('v'   , 'Ambient sound speed'                        )),
        ('area'           , ('area', 'Outflow area'                               )),
        ('pres'           , ('pres', 'Ufo pressure'                               )),
        ('dens'           , ('dens', 'Ufo density'                                )),
        ('temp'           , ('temp', 'Ufo temperature'                            )),
        ('mach'           , ('none', 'Ufo mach number'                            )),
        ('eflx'           , ('eflx', 'Ufo energy flux.'                           )),
        ('pratio'         , ('none', 'Pressure ratio (ufo/ISM).'                  )),
        ('dratio'         , ('none', 'Density ratio (ufo/ISM).'                   )),
        ('eint'           , ('eint', 'Ufo Specific internal energy'               )),
        ('enth'           , ('eint', 'Ufo Specific enthalpy'                      )),
        ('pdot'           , ('pdot', 'Ufo Momentum injection rate.'               )),
        ('pflx'           , ('pres', 'Ufo Momentum flux.'                         )),
        ('r1'             , ('x'   , 'Inner wind launching radius.'               )),
        ('r2'             , ('x'   , 'Outer wind launching radius.'               )),
        ('delta'          , ('x'   , 'w projected onto disc.'                     ))
        ]) 


        ## Capture normalization object
        self.norm = norm

        ## Create and keep a composition object for ufo
        self.uc = CompositionUfo()
        self.muu = self.uc.mu

        ## Create and keep a composition object for ambient gas
        self.ic = CompositionISM()
        self.mua = self.ic.mu

        ## Create an EOS object for ufo
        self.eosu = eos.EOSIdeal(comp=self.uc, inorm=norm, onorm=norm)

        ## Create an EOS object for ambient gas
        self.eosa = eos.EOSIdeal(comp=self.ic, inorm=norm, onorm=norm)

        ## Correct all input parameters into normalization units here first.
        args['power'] = args['power']/getattr(self.norm, self.defs['power'][0])
        args['speed'] = args['speed']*pc.c/getattr(self.norm, self.defs['speed'][0])
        args['mdot'] = args['mdot']*pc.msun/pc.yr/getattr(self.norm, self.defs['mdot'][0])
        args['rufo'] = args['rufo']*pc.kpc/getattr(self.norm, self.defs['rufo'][0])
        args['wufo'] = args['wufo']*pc.kpc/getattr(self.norm, self.defs['wufo'][0])
        args['dens_ambient'] = args['dens_ambient']*self.mua*pc.amu/\
                getattr(self.norm, self.defs['dens_ambient'][0])
        args['temp_ambient'] = args['temp_ambient']/\
                getattr(self.norm, self.defs['temp_ambient'][0])
        self.__dict__.update(args)
        self.args = args


        ## Create update functions for all variables, and update all 
        for var in self.defs:
            if not args.has_key(var): 
                setattr(self, 'upd_'+var, self.create_upd_fn(var))
                getattr(self, 'upd_'+var)()

        ## Create dictionaries
        self.update_all_dictionaries()

    def update_all(self):
        """ 
        Updates all attributes
        excpet those that have changed.
        This logic doesn't work yet, because
        I need to exclude those vars that have
        been changed. Perhaps some logic as to how to
        treat the cases where more vars than equations
        have been changed needs to be included.
        Need to at least create a dictionary of
        changed flags. 
        
        When changing attributes directly, these 
        changes need to be registered to the 
        changed-dict with a function that compares
        new and old dict. An old dict would need to
        be maintained. Setter functions could also
        be used/generated.

        You must also make sure that the decision of 
        retrieval of values from the dictionary or from 
        attributes is consistent throughtout the code.

        Remember that the use of sympy needs to be
        trialed at some point.
        """
        for var in self.defs:
            if not self.args.has_key(var): 
                getattr(self, 'upd_'+var)()
        self.update_all_dictionaries()

    def update_all_dictionaries(self):
        self.update_dictionary_code()
        self.update_dictionary_cgs()

    def update_dictionary_code(self):

        ## Create/update dictionary of all variables in code units
        self.vars_code = OrderedDict()
        for k in self.defs: self.vars_code[k] =  getattr(self, k)

    def update_dictionary_cgs(self):

        ## Create/update dictionary of all vars in cgs
        self.vars_cgs = OrderedDict()
        for k in self.defs: self.vars_cgs[k] =  getattr(self, k)*self.norm.scalings[self.defs[k][0]]

    def print_scalings(self):
        self.norm.print_scalings()

    def print_all_code(self):

        self.update_dictionary_code()
        for var, val in self.vars_code.items():
            print(format(var,'16s')+format(val,'>16.8e'))

    def print_all_cgs(self):

        self.update_dictionary_cgs()
        for var, val in self.vars_cgs.items():
            print(format(var,'16s')+format(val,'>16.8e'))

    def print_all(self):

        self.update_all_dictionaries()
        for var, val_code, val_cgs in zip(self.vars_code.keys(),
                                          self.vars_code.values(), 
                                          self.vars_cgs.values()):
            print(format(var,'16s')+format(val_code,'>16.8e')+3*' '+format(val_cgs,'>16.8e'))

    def create_upd_fn(self, var):
        eqn_fn = getattr(self, 'eqn_'+var)

        def upd_fn(**kwargs):
            """ 
            This function is auto-generated. It calculates the specific variable 
            in its name and updates the respective attribute with the value. The
            function also updates the variable dictionaries.

            See equivalent eqn_ function for list of arguments
            """
            setattr(self, var, eqn_fn(**kwargs))
            return getattr(self, var)

        return upd_fn

    def _default_attributes(self, vardict):
        '''
        vardict         Dictionary of varables to be set to self.<var> if <var> == None
                        Usually vardict is generated in calling funciton with locals()

        Note: A dictionary is mutable so this should work.
        '''
        for (k, v) in vardict.items(): 
            a = 1
            if v == None: vardict[k] = getattr(self, k)

    def eqn_pres_ambient(self, dens_ambient=None, temp_ambient=None, mua=None):
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if mua == None: mua = self.mua

        return self.eosa.pres_from_dens_temp(dens_ambient, temp_ambient, mua)

    def eqn_eint_ambient(self, dens_ambient=None, temp_ambient=None, gamma=None):
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if gamma == None: gamma = self.gamma
        pres_ambient =  self.eqn_pres_ambient(dens_ambient, temp_ambient)
        return pres_ambient/(dens_ambient*(gamma - 1.))

    def eqn_vsnd_ambient(self, dens_ambient=None, temp_ambient=None, gamma=None):
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if gamma == None: gamma = self.gamma
        pres_ambient =  self.eqn_pres_ambient(dens_ambient, temp_ambient)
        return np.sqrt(gamma*pres_ambient/dens_ambient)

    def eqn_area(self, rufo=None, wufo=None):
        """
        Area does not depend on angle!
        """
        if rufo == None: rufo = self.rufo
        if wufo == None: wufo = self.wufo
        return 2*np.pi*rufo*wufo

    def eqn_pres(self, power=None, speed=None, mdot=None, rufo=None, wufo=None, gamma=None):
        if power == None: power = self.power
        if speed == None: speed = self.speed
        if mdot == None: mdot = self.mdot
        if rufo == None: rufo = self.rufo
        if wufo == None: wufo = self.wufo
        if gamma == None: gamma = self.gamma

        area = self.eqn_area(rufo, wufo)
        return (power - 0.5*mdot*speed**2)*(gamma-1)/(gamma*area*speed)

    def eqn_eflx(self, power=None, rufo=None, wufo=None):
        if power == None: power = self.power
        if wufo == None: wufo = self.wufo
        area = self.eqn_area(rufo, wufo)
        return  power/area

    def eqn_dens(self, speed=None, mdot=None, rufo=None, wufo=None):
        if speed == None: speed = self.speed
        if mdot == None: mdot = self.mdot
        if rufo == None: rufo = self.rufo
        if wufo == None: wufo = self.wufo

        area = self.eqn_area(rufo, wufo)
        return mdot/(area*speed)

    def eqn_temp(self, power=None, speed=None, mdot=None, rufo=None, wufo=None,
                 gamma=None, muu=None):
        if power == None: power = self.power
        if speed == None: speed = self.speed
        if mdot == None: mdot = self.mdot
        if rufo == None: rufo = self.rufo
        if wufo == None: wufo = self.wufo
        if gamma == None: gamma = self.gamma
        if muu == None: muu = self.muu

        dens = self.eqn_pres(power, speed, mdot, rufo, wufo, gamma)
        pres = self.eqn_dens(speed, mdot, rufo, wufo)

        return self.eosu.temp_from_dens_pres(dens, pres, muu)

    def eqn_vsnd(self, power=None, speed=None, mdot=None, rufo=None, wufo=None, gamma=None):
        if power == None: power = self.power
        if speed == None: speed = self.speed
        if mdot == None: mdot = self.mdot
        if rufo == None: rufo = self.rufo
        if wufo == None: wufo = self.wufo
        if gamma == None: gamma = self.gamma

        dens = self.eqn_pres(power, speed, mdot, rufo, wufo, gamma)
        pres = self.eqn_dens(speed, mdot, rufo, wufo)
        return np.sqrt(gamma*pres/dens)

    def eqn_mach_internal(self, power=None, speed=None, mdot=None, rufo=None, wufo=None, gamma=None):
        '''
        internal mach number in ufo wind
        '''
        if power == None: power = self.power
        if speed == None: speed = self.speed
        if mdot == None: mdot = self.mdot
        if rufo == None: rufo = self.rufo
        if wufo == None: wufo = self.wufo
        if gamma == None: gamma = self.gamma

        pres = self.eqn_pres(power, speed, mdot, rufo, wufo, gamma)
        dens = self.eqn_dens(speed, mdot, rufo, wufo)
        vsnd = self.eqn_vsnd(pres, dens, gamma)
        return speed/vsnd

    def eqn_mach(self, speed=None, dens_ambient=None, temp_ambient=None, gamma=None):
        '''
        mach number of ufo wind w.r.t to ambient speed of sound
        '''
        if speed == None: speed = self.speed
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if dens_ambient == None: dens_ambient = self.dens_ambient
        if gamma == None: gamma = self.gamma

        vsnd = self.eqn_vsnd_ambient(dens_ambient, temp_ambient, gamma)
        return speed/vsnd

    def eqn_pratio(self, power=None, speed=None, mdot=None, rufo=None, wufo=None, gamma=None, 
                   dens_ambient=None, temp_ambient=None):
        if power == None: power = self.power
        if speed == None: speed = self.speed
        if mdot == None: mdot = self.mdot
        if rufo == None: rufo = self.rufo
        if wufo == None: wufo = self.wufo
        if gamma == None: gamma = self.gamma
        if temp_ambient == None: temp_ambient = self.temp_ambient
        if dens_ambient == None: dens_ambient = self.dens_ambient


        pres = self.eqn_pres(power, speed, mdot, rufo, wufo, gamma)
        pres_ambient = self.eqn_pres_ambient(dens_ambient, temp_ambient)

        return pres/pres_ambient

    def eqn_dratio(self, speed=None, mdot=None, rufo=None, wufo=None, dens_ambient=None):
        if speed == None: speed = self.speed
        if mdot == None: mdot = self.mdot
        if rufo == None: rufo = self.rufo
        if wufo == None: wufo = self.wufo
        if dens_ambient == None: dens_ambient = self.dens_ambient

        dens = self.eqn_dens(speed, mdot, rufo, wufo)

        return dens/dens_ambient


    def eqn_eint(self, power=None, speed=None, mdot=None, rufo=None, wufo=None,
                 gamma=None):

        if power == None: power = self.power
        if speed == None: speed = self.speed
        if mdot == None: mdot = self.mdot
        if rufo == None: rufo = self.rufo
        if wufo == None: wufo = self.wufo
        if gamma == None: gamma = self.gamma

        pres = self.eqn_pres(power, speed, mdot, rufo, wufo, gamma)
        dens = self.eqn_dens(speed, mdot, rufo, wufo)

        return 1./(gamma - 1.)*pres/dens
        

    def eqn_enth(self, power=None, speed=None, mdot=None, rufo=None, wufo=None,
                 gamma=None):

        if power == None: power = self.power
        if speed == None: speed = self.speed
        if mdot == None: mdot = self.mdot
        if rufo == None: rufo = self.rufo
        if wufo == None: wufo = self.wufo
        if gamma == None: gamma = self.gamma

        pres = self.eqn_pres(power, speed, mdot, rufo, wufo, gamma)
        dens = self.eqn_dens(speed, mdot, rufo, wufo)
        eint = self.eqn_eint(power, speed, mdot, rufo, wufo, gamma)

        return  eint + pres/dens
 

    def eqn_pdot(self, speed=None, mdot=None):

        if speed == None: speed = self.speed
        if mdot == None: mdot = self.mdot

        return mdot*speed


    def eqn_pflx(self, speed=None, mdot=None, rufo=None, wufo=None):

        if speed == None: speed = self.speed
        if mdot == None: mdot = self.mdot
        if rufo == None: rufo = self.rufo
        if wufo == None: wufo = self.wufo

        area = self.eqn_area(rufo, wufo)
        pdot = self.eqn_pdot(speed, mdot)

        return pdot/area

    def eqn_r1(self, angle=None, rufo=None, wufo=None):

        if angle == None: angle = self.angle
        if rufo == None: rufo = self.rufo
        if wufo == None: wufo = self.wufo

        return rufo - 0.5*wufo/np.cos(np.pi/180*angle)

    def eqn_r2(self, angle=None, rufo=None, wufo=None):

        if angle == None: angle = self.angle
        if rufo == None: rufo = self.rufo
        if wufo == None: wufo = self.wufo

        return rufo + 0.5*wufo/np.cos(np.pi/180*angle)


    def eqn_delta(self, angle=None, rufo=None, wufo=None):

        if angle == None: angle = self.angle
        if wufo == None: wufo = self.wufo

        return wufo/np.cos(np.pi/180*angle)


