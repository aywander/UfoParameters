import physconst as pc
import norm

class CompositionBase():
    """
    Calculation of mean mass per particle mu as a constant.
    Derive classes from this and modify the function eqn_mu() to 
    create a class for substances with different microphysics, e.g.,
    ones that depend on thermodynamic quantities.
    """
    def __init__(self, mu=0.6165):
        """
        mu       mean mass per particle in units of atomic mass unit. 
        """

        ## All attributes in class are stored in dictionary
        ## self.__dict__ whose update function can be used to 
        ## append it with a list of local variables.
        self.__dict__.update(locals())
        del self.__dict__['self']

    def eqn_mu(self):
        """
        Calculation of mu as a constant.
        """
        return self.mu




class AtomicISM(CompositionBase):
    def __init__(self, mu=1.21):
        """
        mu       mean mass per particle in units of atomic mass unit. 
        """
        CompositionBase.__init__(self, mu)


class IonizedISM(CompositionBase):
    def __init__(self, mu=0.6165):
        """
        mu       mean mass per particle in units of atomic mass unit. 
        """
        CompositionBase.__init__(self, mu)


class EOSBase():

    def __init__(self, dens=None, pres=None, temp=None, comp=IonizedISM(),
                 inorm=norm.PhysNorm(x=1., m=1., t=1., curr=1., temp=1.),
                 onorm=norm.PhysNorm(x=1., m=1., t=1., curr=1., temp=1.)):
        """
        inorm      Normalization of input. 
        onorm      Normalization of output.
        comp       Composition object (also defined in eos.py)
                   This is mainly to get the value of mu.

        Variables are immediately converted to cgs internally with inorm. 
        Output is normalized with onorm.
        """

        mu = comp.mu

        state_vars = ['dens', 'pres', 'temp', 'eint', 'enth', 'entr']

        if dens != None: dens = dens*inorm.dens
        if pres != None: pres = pres*inorm.pres
        if temp != None: temp = temp*inorm.temp

        self.__dict__.update(locals()); del self.__dict__['self']



class EOSIdeal(EOSBase):

    def __init__(self, dens=None, pres=None, temp=None, comp=IonizedISM(),
                 inorm=norm.PhysNorm(x=1., m=1., t=1., curr=1., temp=1.),
                 onorm=norm.PhysNorm(x=1., m=1., t=1., curr=1., temp=1.)):
        """
        inorm      Normalization of input. 
        onorm      Normalization of output.

        Variables are immediately converted to cgs internally with inorm. 
        Output is normalized with onorm.

        All calculations of pressure, density, and temperature go via the
        functions {pres|dens|temp}_from_{...}. The other functions just
        make sure the correct one is called.
        """

        self.eos_modes = ['dens_pres', 'dens_temp', 'pres_temp']

        EOSBase.__init__(self, dens, pres, temp, comp, inorm, onorm)

    def eos(self, v1, v2, mode, mu=None):
        """
        Ideal equation of state. Calculate third variable.

        v1, v2      The order of variables in the function 
                    is given by self.state_vars
        mode        A string giving the missing variable.
                    Any of self.eos_modes, "dens_pres", "pres_temp", 
                    "dens_temp", ...
        """

        if mu == None: mu = self.mu

        if mode == 'dens_pres':
            v3 = self.temp_from_dens_pres(v1, v2, mu)
        elif mode == 'dens_temp':
            v3 = self.pres_from_dens_temp(v1, v2, mu)
        elif mode == 'pres_temp':
            v3 = self.dens_from_pres_temp(v1, v2, mu)
        else:
            print('Undefined mode: '+mode)
            print('Currently only dens_pres, dens_temp, and pres_temp accepted.')
            exit()

        return v3

    def auto_eos(self):
        """
        Auto complete the last variable not. 
        """
        if self.temp == None and self.dens and self.pres:
            self.temp = self.temp_from_dens_pres()
            self.pres = self.pres/self.onorm.pres
            self.dens = self.dens/self.onorm.dens
            return self.temp

        elif self.pres == None and self.dens and self.temp:
            self.pres = self.pres_from_dens_temp()
            self.temp = self.temp/self.onorm.temp
            self.dens = self.dens/self.onorm.dens
            return self.pres

        elif self.dens == None and self.pres and self.temp:
            self.dens = self.dens_from_pres_temp()
            self.pres = self.pres/self.onorm.pres
            self.temp = self.temp/self.onorm.temp
            return self.dens

        else:
            print('Combination of values stored in this instance does not allow auto completion:')
            print('self.dens = ' + str(self.dens))
            print('self.pres = ' + str(self.pres))
            print('self.temp = ' + str(self.temp))
            raise(ValueError)

    def pres_from_dens_temp(self, dens=None, temp=None, mu=None):
        """
        Every calculation of pres uses this function. 
        mu is considered a variable of the EOS.
        """

        dens = self.dens if dens == None else dens*self.inorm.dens
        temp = self.temp if temp == None else temp*self.inorm.temp
        if mu == None: mu = self.mu

        return dens*temp*pc.kboltz/(mu*pc.amu)/self.onorm.pres

    def dens_from_pres_temp(self, pres=None, temp=None, mu=None):
        """
        Every calculation of dens uses this function. 
        mu is considered a variable of the EOS.
        """
        pres = self.pres if pres == None else pres*self.inorm.pres
        temp = self.temp if temp == None else temp*self.inorm.temp
        if mu == None: mu = self.mu

        return mu*pc.amu*pres/(temp*pc.kboltz)/self.onorm.dens

    def temp_from_dens_pres(self, dens=None, pres=None, mu=None):
        """
        Every calculation of temp uses this function.
        mu is considered a variable of the EOS.
        """
        dens = self.dens if dens == None else dens*self.inorm.dens
        pres = self.pres if pres == None else pres*self.inorm.pres
        if mu == None: mu = self.mu

        return mu*pc.amu*pres/(dens*pc.kboltz)/self.onorm.temp



