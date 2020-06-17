import math
import sys
import riemann_solvers as rs

def main_loop(rho, P, u, E, momentum, tend, dtmax, dx, flux_function, gamma=1.4, model=rs.riemann_iterative):
    ncells = len(P) - 2
    
    FL = [None]*(ncells+1)
    FR = [None]*(ncells+1)
    
    # Main loop
    t = 0.0
    step = 1
    while True:
        S = 0
        for i in range(1, ncells+1):
            a = math.sqrt((gamma*P[i])/rho[i])
            S = max(S, a + u[i])
            
        dt = min(dtmax, 0.6*dx/S)
        
        print("step: {0:4d}, t = {1:6.4f}, dt = {2:8.6e}".format(step, t, dt))
        
        for i in range(1, ncells+1):    
            FL[i] = flux_function(u[i-1], rho[i-1], P[i-1], u[i], rho[i], P[i], gamma, model)
            FR[i] = flux_function(u[i], rho[i], P[i], u[i+1], rho[i+1], P[i+1], gamma, model)
     
        f = dt/dx
        
        for i in range(1, ncells+1):
            momL = FL[i].mom
            momR = FR[i].mom
            EL = FL[i].E
            ER = FR[i].E
                  
            # Conservative update
            rho[i] += f*(FL[i].rho - FR[i].rho)
            momentum[i] += f*(momL - momR)
            E[i] += f*(EL - ER)
            
            # Primative update
            P[i] = (gamma - 1.0)*(E[i] - 0.5*rho[i]*u[i]*u[i])
            u[i] = momentum[i]/rho[i]
    
        
            if P[i] < 0.0:
                print("\nNegative pressure in cell {}".format(i))
                print("Pressure =",P[i])
                print("Energy =",E[i])
                print("Rho =",rho[i])
                print("u =",u[i])
                sys.exit()
    
            if rho[i] < 0.0:
                print("\nNegative density in cell {}".format(i))
                print("Pressure =",P[i])
                print("Energy =",E[i])
                print("Rho =",rho[i])
                print("u =",u[i])
                sys.exit()         
        
        if t > tend:
            break
        t += dt
        step += 1
        
    return step


def main_loop_rcm(rho, P, u, E, tend, dtmax, dx, gamma=1.4):
    import random
    
    import riemann_solvers as riemann
    
    ncells = len(P) - 2
    
    W = [None]*(ncells+1)
    
    # Main loop
    t = 0.0
    step = 1
    while True:
        S = 0
        for i in range(1, ncells+1):
            a = math.sqrt((gamma*P[i])/rho[i])
            S = max(S, a + u[i])
            
        dt = min(dtmax, 0.4*dx/S)
        
        print("step: {0:4d}, t = {1:6.4f}, dt = {2:8.6e}".format(step, t, dt))
        
        for i in range(1, ncells+1):    
            theta = random.random()
            if theta <= 0.5:
                W[i] = riemann.solve(u[i-1], rho[i-1], P[i-1], u[i], rho[i], P[i], dx*theta, dt, gamma)
            else:
                theta1 = theta - 1.0
                W[i] = riemann.solve(u[i], rho[i], P[i], u[i+1], rho[i+1], P[i+1], dx*theta1, dt, gamma)
                    
        for i in range(1, ncells+1):             
            # Update
            rho[i] = W[i].rho
            u[i] = W[i].u
            P[i] = W[i].P
            
            # Calculate energy
            e = P[i]/((gamma - 1.0)*rho[i])
            E[i] = rho[i] * (0.5*u[i]*u[i] + e)
    
        
            if P[i] < 0.0:
                print("\nNegative pressure in cell {}".format(i))
                print("Pressure =",P[i])
                print("Energy =",E[i])
                print("Rho =",rho[i])
                print("u =",u[i])
                sys.exit()
    
            if rho[i] < 0.0:
                print("\nNegative density in cell {}".format(i))
                print("Pressure =",P[i])
                print("Energy =",E[i])
                print("Rho =",rho[i])
                print("u =",u[i])
                sys.exit()         
        
        if t > tend:
            break
        t += dt
        step += 1
        
    return step
