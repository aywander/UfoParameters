
import ufo_parameters as up

p = up.UfoParams(
    power=1.e44,
    angle=30,
    speed=0.01,
    mdot=0.1,
    rufo=1.e-2,
    wufo=1.e-2,
    dens_ambient=1.0,
    temp_ambient=1.e7,
    gamma=1.6666666666,
)


p.print_all()
#p.print_scalings() 
