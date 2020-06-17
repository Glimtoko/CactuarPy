cimport cython
from libc.math cimport sqrt


@cython.cdivision(True)
cdef double rhostar_shock(double P, double Pk, double rhok, double gamma):
    f = (gamma - 1.0)/(gamma + 1.0)
    
    return rhok * (P/Pk + f) / (f*P/Pk + 1.0)


@cython.cdivision(True)
cdef double SL_shock(double P, double Pk, double rhok, double uk, double gamma):
    ak = sqrt((gamma*Pk)/rhok)
    
    return uk - ak*((gamma + 1.0)/(2.0*gamma)*(P/Pk) + (gamma - 1.0)/(2.0*gamma))**0.5


@cython.cdivision(True)
cdef double SR_shock(double P, double Pk, double rhok, double uk, double gamma):
    ak = sqrt((gamma*Pk)/rhok)
    
    return uk + ak*((gamma + 1.0)/(2.0*gamma)*(P/Pk) + (gamma - 1.0)/(2.0*gamma))**0.5


@cython.cdivision(True)
cdef double rhostar_rarefaction(double P, double Pk, double rhok, double gamma):
    return rhok * (P/Pk)**(1.0/gamma)


@cython.cdivision(True)
cdef (double, double, double) rhoLfan_rarefaction(
    double P,
    double PL,
    double rhoL,
    double uL,
    double gamma,
    double S
):
    aL = sqrt((gamma*PL)/rhoL)
    f = (2*gamma)/(gamma - 1.0)
    
    rho = rhoL*(2.0/(gamma + 1.0) + (gamma - 1.0)/((gamma + 1.0)*aL)*(uL - S))**(2/(gamma - 1.0))
    u = 2.0/(gamma+1.0)*(aL + (gamma - 1.0)/2.0 * uL + S)
    P = PL * (2.0/(gamma + 1.0) + (gamma - 1.0)/((gamma + 1.0)*aL) * (uL - S))**f
    
    return rho, P, u


@cython.cdivision(True)
cdef (double, double, double) rhoRfan_rarefaction(
    double P,
    double PR,
    double rhoR,
    double uR,
    double gamma,
    double S
):
    aR = sqrt((gamma*PR)/rhoR)
    f = 2*gamma/(gamma - 1.0)
    
    rho = rhoR*(2.0/(gamma + 1.0) - (gamma - 1.0)/((gamma + 1.0)*aR)*(uR - S))**(2/(gamma - 1.0))
    u = 2.0/(gamma+1.0)*(-aR + (gamma - 1.0)/2.0 * uR + S)
    P = PR * (2.0/(gamma + 1.0) - (gamma - 1.0)/((gamma + 1.0)*aR) * (uR - S))**f
    
    return rho, P, u


@cython.cdivision(True)
cdef (double, double, double) rhox_L_shock(
    double P,
    double PL,
    double rhoL,
    double uL,
    double gamma,
    double ustar,
    double S
):
    SL = SL_shock(P, PL, rhoL, uL, gamma)
    if S <= SL:
        return rhoL, PL, uL
    else:
        return rhostar_shock(P, PL, rhoL, gamma), P, ustar


@cython.cdivision(True)
cdef (double, double, double) rhox_R_shock(
    double P,
    double PR,
    double rhoR,
    double uR,
    double gamma,
    double ustar,
    double S
):
    SR = SR_shock(P, PR, rhoR, uR, gamma)
    if S >= SR:
        return rhoR, PR, uR
    else:
        return rhostar_shock(P, PR, rhoR, gamma), P, ustar


@cython.cdivision(True)
cdef (double, double, double) rhox_L_rarefaction(
    double P,
    double PL,
    double rhoL,
    double uL,
    double gamma,
    double ustar,
    double S
):
    aL = sqrt((gamma*PL)/rhoL)
    astar = aL*(P/PL)**((gamma - 1.0)/(2.0*gamma))
    
    SHL = uL - aL
    STL = ustar - astar
    
    if S <= SHL:
        return (rhoL, PL, uL)
    elif S <= STL:
        return rhoLfan_rarefaction(P, PL, rhoL, uL, gamma, S)
    else:
        return rhostar_rarefaction(P, PL, rhoL, gamma), P, ustar


@cython.cdivision(True)
cdef (double, double, double) rhox_R_rarefaction(
    double P,
    double PR,
    double rhoR,
    double uR,
    double gamma,
    double ustar,
    double S
):
    aR = sqrt((gamma*PR)/rhoR)
    astar = aR*(P/PR)**((gamma - 1.0)/(2.0*gamma))
    
    SHR = uR + aR
    STR = ustar + astar
    
    if S >= SHR:
        return rhoR, PR, uR
    elif S >= STR:
        return rhoRfan_rarefaction(P, PR, rhoR, uR, gamma, S)
    else:
        return rhostar_rarefaction(P, PR, rhoR, gamma), P, ustar


@cython.cdivision(True)
cdef (double, double, double) rhox_L(
    double P,
    double PL,
    double rhoL,
    double uL,
    double gamma,
    double ustar,
    double S
):
    if P > PL:
        return rhox_L_shock(P, PL, rhoL, uL, gamma, ustar, S)
    else:
        return rhox_L_rarefaction(P, PL, rhoL, uL, gamma, ustar, S)


@cython.cdivision(True)
cdef (double, double, double) rhox_R(
    double P,
    double PR,
    double rhoR,
    double uR,
    double gamma,
    double ustar,
    double S
):
    if P > PR:
        return rhox_R_shock(P, PR, rhoR, uR, gamma, ustar, S)
    else:
        return rhox_R_rarefaction(P, PR, rhoR, uR, gamma, ustar, S)


@cython.cdivision(True)
cdef (double, double, double) sample_exact(
    double Pstar,
    double ustar,
    double x,
    double t,
    double uL,
    double rhoL,
    double PL,
    double uR,
    double rhoR,
    double PR,
    double gamma=1.4
):
    S = x/t
    if S < ustar:
        rhox, Px, ux = rhox_L(Pstar, PL, rhoL, uL, gamma, ustar, S)
    else:
        rhox, Px, ux = rhox_R(Pstar, PR, rhoR, uR, gamma, ustar, S)
    
    return rhox, Px, ux


@cython.cdivision(True)
cdef (double, double, double) sample_approx(
    double Pstar,
    double ustar,
    double rhoLstar,
    double rhoRstar,
    double uL,
    double rhoL,
    double PL,
    double uR,
    double rhoR,
    double PR,
    double gamma=1.4
):
    S = 0
    P = Pstar
    if S < ustar:
        # Left side
        if P > PL:
            # Shock
            SL = SL_shock(P, PL, rhoL, uL, gamma)
            if S <= SL:
                return rhoL, PL, uL
            else:
                return rhoLstar, P, ustar
        else:
            # Rarefaction
            aL = sqrt((gamma*PL)/rhoL)
            astar = aL*(P/PL)**((gamma - 1.0)/(2.0*gamma))
            
            SHL = uL - aL
            STL = ustar - astar
            
            if S <= SHL:
                return (rhoL, PL, uL)
            elif S <= STL:
                return rhoLfan_rarefaction(P, PL, rhoL, uL, gamma, 0)
            else:
                return rhoLstar, P, ustar
        
    else:
        # Right side
        if P > PR:
            # Shock
            SR = SR_shock(P, PR, rhoR, uR, gamma)
            if S >= SR:
                return rhoR, PR, uR
            else:
                return rhoRstar, P, ustar
            
        else:
            # Rarefaction
            aR = sqrt((gamma*PR)/rhoR)
            astar = aR*(P/PR)**((gamma - 1.0)/(2.0*gamma))
            
            SHR = uR + aR
            STR = ustar + astar
            
            if S >= SHR:
                return rhoR, PR, uR
            elif S >= STR:
                return rhoRfan_rarefaction(P, PR, rhoR, uR, gamma, S)
            else:
                return rhoRstar, P, ustar