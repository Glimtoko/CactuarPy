import time

import numpy

import set_mesh 
import core

# Problem specification
problem = "sod20"
ncells = 250
invert = False
dtinit = 0.1
output = "result_{}.dat".format(problem)

# EoS
gamma = 1.4

# Problem domain
length = 1.0
interface_position = 0.5

# Solver settings
flux_function = core.get_flux_Godunov
riemann_solver = core.riemann_iterative


rho, momentum, E, P, u, cx, tend, dx = set_mesh.set_mesh(
    problem,
    gamma=gamma,
    L=length,
    x0=interface_position,
    ncells=ncells,
    invert=invert
)

time1 = time.perf_counter()
core.main_loop(
    rho,
    P,
    u,
    E,
    momentum,
    tend,
    dtinit,
    dx,
    flux_function,
    gamma=gamma,
    model=riemann_solver
)
time2 = time.perf_counter()

print("\nTime in main loop = {}s".format(time2 - time1))

ein = numpy.zeros(len(rho))
for i in range(len(rho)):
    ein[i] = E[i]/rho[i] - 0.5*u[i]*u[i]
    

with open(output, "w") as ofile:
    for i in range(len(rho)):
        ofile.write("{} {} {} {} {} {}\n".format(i, cx[i], rho[i], P[i], u[i], ein[i]))
