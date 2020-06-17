import math

def fk_shock(P, Pk, rhok, gamma):
    Ak = 2.0/((gamma + 1.0)*rhok)
    Bk = ((gamma - 1.0)/(gamma + 1.0)) * Pk
    
    return (P - Pk) * math.sqrt(Ak/(P + Bk))


def fk_rarefaction(P, Pk, rhok, gamma):
    f = (gamma - 1.0)/(2.0*gamma)
    ak = math.sqrt((gamma*Pk)/rhok)
    
    return (2.0*ak)/(gamma - 1.0) * ((P/Pk)**f - 1.0)


def fk(P, Pk, rhok, gamma):
    if P >= Pk:
        return fk_shock(P, Pk, rhok, gamma)
    else:
        return fk_rarefaction(P, Pk, rhok, gamma)
    
    
def fdashk_shock(P, Pk, rhok, gamma):
    Ak = 2.0/((gamma + 1.0)*rhok)
    Bk = ((gamma - 1.0)/(gamma + 1.0)) * Pk
    
    return math.sqrt(Ak/(P + Bk)) * (1.0 - (P - Pk)/(2.0*(Bk + P)))


def fdashk_rarefaction(P, Pk, rhok, gamma):
    f = -(gamma + 1.0)/(2.0*gamma)
    ak = math.sqrt((gamma*Pk)/rhok)
    
    return 1.0/(rhok*ak) * (P/Pk)**f


def fdashk(P, Pk, rhok, gamma):
    if P >= Pk:
        return fdashk_shock(P, Pk, rhok, gamma)
    else:
        return fdashk_rarefaction(P, Pk, rhok, gamma)
    
    
def get_P(P0, PL, rhoL, PR, rhoR, uL, uR, gamma):
    du = uR - uL
    f = fk(P0, PL, rhoL, gamma) + fk(P0, PR, rhoR, gamma) + du
    fdash = fdashk(P0, PL, rhoL, gamma) + fdashk(P0, PR, rhoR, gamma)
    
    return P0 - (f/fdash)


def rPc(Pnew, Pold):
    return abs(Pnew - Pold)/(0.5*(Pnew + Pold))