import math

class solution():
    def __init__(self, rho, P, u, E):
        self.rho = rho
        self.P = P
        self.u = u
        self.E = E


def rhostar_shock(P, Pk, rhok, gamma):
    f = (gamma - 1.0)/(gamma + 1.0)
    
    return rhok * (P/Pk + f) / (f*P/Pk + 1.0)


def SL_shock(P, Pk, rhok, uk, gamma):
    ak = math.sqrt((gamma*Pk)/rhok)
    
    return uk - ak*((gamma + 1.0)/(2.0*gamma)*(P/Pk) + (gamma - 1.0)/(2.0*gamma))**0.5


def SR_shock(P, Pk, rhok, uk, gamma):
    ak = math.sqrt((gamma*Pk)/rhok)
    
    return uk + ak*((gamma + 1.0)/(2.0*gamma)*(P/Pk) + (gamma - 1.0)/(2.0*gamma))**0.5


def rhostar_rarefaction(P, Pk, rhok, gamma):
    return rhok * (P/Pk)**(1.0/gamma)


def rhoLfan_rarefaction(P, PL, rhoL, uL, gamma, S):
    aL = math.sqrt((gamma*PL)/rhoL)
    f = (2*gamma)/(gamma - 1.0)
    
    rho = rhoL*(2.0/(gamma + 1.0) + (gamma - 1.0)/((gamma + 1.0)*aL)*(uL - S))**(2/(gamma - 1.0))
    u = 2.0/(gamma+1.0)*(aL + (gamma - 1.0)/2.0 * uL + S)
    P = PL * (2.0/(gamma + 1.0) + (gamma - 1.0)/((gamma + 1.0)*aL) * (uL - S))**f
    
    return rho, P, u


def rhoRfan_rarefaction(P, PR, rhoR, uR, gamma, S):
    aR = math.sqrt((gamma*PR)/rhoR)
    f = 2*gamma/(gamma - 1.0)
    
    rho = rhoR*(2.0/(gamma + 1.0) - (gamma - 1.0)/((gamma + 1.0)*aR)*(uR - S))**(2/(gamma - 1.0))
    u = 2.0/(gamma+1.0)*(-aR + (gamma - 1.0)/2.0 * uR + S)
    P = PR * (2.0/(gamma + 1.0) - (gamma - 1.0)/((gamma + 1.0)*aR) * (uR - S))**f
    
    return rho, P, u


def rhox_L_shock(P, PL, rhoL, uL, gamma, ustar, S):
    SL = SL_shock(P, PL, rhoL, uL, gamma)
    if S <= SL:
        return rhoL, PL, uL
    else:
        return rhostar_shock(P, PL, rhoL, gamma), P, ustar


def rhox_R_shock(P, PR, rhoR, uR, gamma, ustar, S):
    SR = SR_shock(P, PR, rhoR, uR, gamma)
    if S >= SR:
        return rhoR, PR, uR
    else:
        return rhostar_shock(P, PR, rhoR, gamma), P, ustar


def rhox_L_rarefaction(P, PL, rhoL, uL, gamma, ustar, S):
    aL = math.sqrt((gamma*PL)/rhoL)
    astar = aL*(P/PL)**((gamma - 1.0)/(2.0*gamma))
    
    SHL = uL - aL
    STL = ustar - astar
    
    if S <= SHL:
        return (rhoL, PL, uL)
    elif S <= STL:
        return rhoLfan_rarefaction(P, PL, rhoL, uL, gamma, S)
    else:
        return rhostar_rarefaction(P, PL, rhoL, gamma), P, ustar


def rhox_R_rarefaction(P, PR, rhoR, uR, gamma, ustar, S):
    aR = math.sqrt((gamma*PR)/rhoR)
    astar = aR*(P/PR)**((gamma - 1.0)/(2.0*gamma))
    
    SHR = uR + aR
    STR = ustar + astar
    
    if S >= SHR:
        return rhoR, PR, uR
    elif S >= STR:
        return rhoRfan_rarefaction(P, PR, rhoR, uR, gamma, S)
    else:
        return rhostar_rarefaction(P, PR, rhoR, gamma), P, ustar


def rhox_L(P, PL, rhoL, uL, gamma, ustar, S):
    if P > PL:
        return rhox_L_shock(P, PL, rhoL, uL, gamma, ustar, S)
    else:
        return rhox_L_rarefaction(P, PL, rhoL, uL, gamma, ustar, S)


def rhox_R(P, PR, rhoR, uR, gamma, ustar, S):
    if P > PR:
        return rhox_R_shock(P, PR, rhoR, uR, gamma, ustar, S)
    else:
        return rhox_R_rarefaction(P, PR, rhoR, uR, gamma, ustar, S)


def sample_exact(Pstar, ustar, x, t, uL, rhoL, PL, uR, rhoR, PR, gamma=1.4):
    S = x/t
    if S < ustar:
        rhox, Px, ux = rhox_L(Pstar, PL, rhoL, uL, gamma, ustar, S)
    else:
        rhox, Px, ux = rhox_R(Pstar, PR, rhoR, uR, gamma, ustar, S)
    
    return rhox, Px, ux


def sample_approx(Pstar, ustar, rhoLstar, rhoRstar, uL, rhoL, PL, uR, rhoR, PR, gamma=1.4):
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
            aL = math.sqrt((gamma*PL)/rhoL)
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
            aR = math.sqrt((gamma*PR)/rhoR)
            astar = aR*(P/PR)**((gamma - 1.0)/(2.0*gamma))
            
            SHR = uR + aR
            STR = ustar + astar
            
            if S >= SHR:
                return rhoR, PR, uR
            elif S >= STR:
                return rhoRfan_rarefaction(P, PR, rhoR, uR, gamma, S)
            else:
                return rhoRstar, P, ustar