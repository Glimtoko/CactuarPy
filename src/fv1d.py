import numpy

import set_mesh 
import core
import riemann_solvers as rs

import matplotlib.pyplot as plot

# Problem specification
problem = "sod"
ncells = 1000
invert = False
dtinit = 0.1

# EoS
gamma = 1.4

# Problem domain
length = 1.0
interface_position = 0.5

# Solver settings
flux_function = core.get_flux_HLLC
riemann_solver = None


rho, momentum, E, P, u, cx, tend, dx = set_mesh.set_mesh(
    problem,
    gamma=gamma,
    L=length,
    x0=interface_position,
    ncells=ncells,
    invert=invert
)

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

ein = numpy.zeros(len(rho))
for i in range(len(rho)):
    ein[i] = E[i]/rho[i] - 0.5*u[i]*u[i]
    
    
#plot.plot(x, exact.rho, "k", label="Exact (iterative Riemann solve)")
plot.scatter(cx, rho, s=2.0)
plot.title(r"$\rho$")
plot.xlabel("x (cm)")

plot.savefig("test.png")
    

# results[flux] = {
#     "X": copy.deepcopy(cx),
#     "RHO": copy.deepcopy(rho),
#     "P": copy.deepcopy(P),
#     "E": copy.deepcopy(E),
#     "EIN": copy.deepcopy(ein),
#     "U": copy.deepcopy(u),
#     }

# # Random Choice method uses bespoke main loop
# rho, momentum, E, P, u, cx, tend, dx = set_mesh.set_mesh(
#     "sod",
#     gamma=1.4,
#     L=1.0,
#     x0=0.5,
#     ncells=ncells,
#     invert=False
# )
# main_loop.main_loop_rcm(
#     rho,
#     P,
#     u,
#     E,
#     tend,
#     0.1,
#     dx,
#     gamma=1.4
# )

# results["Random Choice Method"] = {
#     "X": copy.deepcopy(cx),
#     "RHO": copy.deepcopy(rho),
#     "P": copy.deepcopy(P),
#     "E": copy.deepcopy(E),
#     "EIN": copy.deepcopy(ein),
#     "U": copy.deepcopy(u),
#     }

# rhoL, uL, PL, rhoR, uR, PR, name = set_mesh.get_initial("sod")

# x, exact = riemann.exact(uL, uR, rhoL, rhoR, PL, PR, t=tend, diaphram=0.5)

# s = 2

# fig = plot.figure(figsize=(image_size, image_size))
# plot.subplots_adjust(hspace=0.7)

# axes = fig.add_subplot(221)
# plot.plot(x, exact.rho, "k", label="Exact (iterative Riemann solve)")
# for flux in results.keys():
#     plot.scatter(results[flux]["X"], results[flux]["RHO"], s=s, label=flux)
# plot.title(r"$\rho$")
# plot.xlabel("x (cm)")
    
# axes = fig.add_subplot(224)
# plot.plot(x, exact.E, "k")
# for flux in results.keys():
#     plot.scatter(results[flux]["X"], results[flux]["EIN"], s=s)
# plot.title(r"Internal Energy")
# plot.xlabel("x (cm)")

# axes = fig.add_subplot(222)
# plot.plot(x, exact.u, "k")
# for flux in results.keys():
#     plot.scatter(results[flux]["X"], results[flux]["U"], s=s)
# plot.title(r"Velocity")
# plot.xlabel("x (cm)")

# axes = fig.add_subplot(223)
# plot.plot(x, exact.P, "k")
# for flux in results.keys():
#     plot.scatter(results[flux]["X"], results[flux]["P"], s=s)
# plot.title(r"Pressure")
# plot.xlabel("x (cm)")

    
# fig.legend(loc=0)

# plot.savefig("sod.png")