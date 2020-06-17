cimport cython
from libc.math cimport sqrt, abs


@cython.cdivision(True)
cdef double fk_shock(
    double P,
    double Pk,
    double rhok,
    double gamma
):
    Ak = 2.0/((gamma + 1.0)*rhok)
    Bk = ((gamma - 1.0)/(gamma + 1.0)) * Pk
    
    return (P - Pk) * sqrt(Ak/(P + Bk))


@cython.cdivision(True)
cdef double fk_rarefaction(
    double P,
    double Pk,
    double rhok,
    double gamma
):
    f = (gamma - 1.0)/(2.0*gamma)
    ak = sqrt((gamma*Pk)/rhok)
    
    return (2.0*ak)/(gamma - 1.0) * ((P/Pk)**f - 1.0)


@cython.cdivision(True)
cdef double fk(
    double P,
    double Pk,
    double rhok,
    double gamma
):
    if P >= Pk:
        return fk_shock(P, Pk, rhok, gamma)
    else:
        return fk_rarefaction(P, Pk, rhok, gamma)
    
    
@cython.cdivision(True)
cdef double fdashk_shock(
    double P,
    double Pk,
    double rhok,
    double gamma
):
    Ak = 2.0/((gamma + 1.0)*rhok)
    Bk = ((gamma - 1.0)/(gamma + 1.0)) * Pk
    
    return sqrt(Ak/(P + Bk)) * (1.0 - (P - Pk)/(2.0*(Bk + P)))


@cython.cdivision(True)
cdef double fdashk_rarefaction(
    double P,
    double Pk,
    double rhok,
    double gamma
):
    f = -(gamma + 1.0)/(2.0*gamma)
    ak = sqrt((gamma*Pk)/rhok)
    
    return 1.0/(rhok*ak) * (P/Pk)**f


@cython.cdivision(True)
cdef double fdashk(
    double P,
    double Pk,
    double rhok,
    double gamma
):
    if P >= Pk:
        return fdashk_shock(P, Pk, rhok, gamma)
    else:
        return fdashk_rarefaction(P, Pk, rhok, gamma)
    
    
@cython.cdivision(True)
cdef double get_P(
    double P0,
    double PL,
    double rhoL,
    double PR,
    double rhoR,
    double uL,
    double uR,
    double gamma
):
    cdef double du = uR - uL
    cdef double fL = fk(P0, PL, rhoL, gamma)
    cdef double f = fk(P0, PL, rhoL, gamma) + fk(P0, PR, rhoR, gamma) + du
    cdef double fdash = fdashk(P0, PL, rhoL, gamma) + fdashk(P0, PR, rhoR, gamma)
    
    return P0 - (f/fdash)


@cython.cdivision(True)
cdef double rPc(
    double Pnew,
    double Pold
):
    return abs(Pnew - Pold)/(0.5*(Pnew + Pold))