import math
#import riemann_solvers as riemann

cimport cython
from libc.math cimport sqrt

@cython.cdivision(True)
def get_flux_fvs_SW(
        double uL,
        double rhoL,
        double PL,
        double uR,
        double rhoR,
        double PR,
        double gamma=1.4,
        model=None
        
):
    cdef double aL = sqrt((gamma*PL)/rhoL)
    cdef double HL = 0.5*(uL*uL) + (aL*aL)/(gamma - 1.0)

    cdef double aR = sqrt((gamma*PR)/rhoR)
    cdef double HR = 0.5*(uR*uR) + (aR*aR)/(gamma - 1.0)
    
    cdef double l1L = uL - aL
    cdef double l2L = uL
    cdef double l3L = uL + aL

    cdef double l1R = uR - aR
    cdef double l2R = uR
    cdef double l3R = uR + aR

    cdef double l1p = 0.5*(l1L + abs(l1L))
    cdef double l1m = 0.5*(l1R - abs(l1R))
    cdef double l2p = 0.5*(l2L + abs(l2L))
    cdef double l2m = 0.5*(l2R - abs(l2R))
    cdef double l3p = 0.5*(l3L + abs(l3L))
    cdef double l3m = 0.5*(l3R - abs(l3R))
    
    cdef double fL = rhoL/(2.0*gamma)
    cdef double fR = rhoR/(2.0*gamma)
    
    cdef double rho_fluxp = fL*(l1p + 2.0*(gamma - 1.0)*l2p + l3p)
    cdef double rho_fluxm = fR*(l1m + 2.0*(gamma - 1.0)*l2m + l3m)
    
    cdef double mom_fluxp = fL*((uL - aL)*l1p + 2.0*(gamma - 1.0)*uL*l2p + (uL + aL)*l3p)
    cdef double mom_fluxm = fR*((uR - aR)*l1m + 2.0*(gamma - 1.0)*uR*l2m + (uR + aR)*l3m)
    
    cdef double E_fluxp = fL*((HL - uL*aL)*l1p + (gamma - 1.0)*uL*uL*l2p + (HL + uL*aL)*l3p)
    cdef double E_fluxm = fR*((HR - uR*aR)*l1m + (gamma - 1.0)*uR*uR*l2m + (HR + uR*aR)*l3m)
    
    cdef (double, double, double) flux
    flux = (rho_fluxp + rho_fluxm, mom_fluxp + mom_fluxm, E_fluxp + E_fluxm)
    
    return flux


@cython.cdivision(True)
def get_flux_fvs_VL(
        double uL,
        double rhoL,
        double PL,
        double uR,
        double rhoR,
        double PR,
        double gamma=1.4,
        model=None
        
):
    cdef double aL = sqrt((gamma*PL)/rhoL)
    cdef double ML = uL/aL

    cdef double aR = sqrt((gamma*PR)/rhoR)
    cdef double MR = uR/aR
    
    cdef double fp = 0.25*rhoL*aL*(1.0 + ML)**2
    cdef double fm = -0.25*rhoR*aR*(1.0 - MR)**2
    cdef double fg = ((gamma - 1.0)/2.0)
    
    cdef double rho_fluxp = fp
    cdef double rho_fluxm = fm
    
    cdef double mom_fluxp = fp*( ((2.0*aL)/gamma) * (fg*ML + 1.0) )
    cdef double mom_fluxm = fm*( ((2.0*aR)/gamma) * (fg*MR - 1.0) )
    
    cdef double E_fluxp = fp*( (2.0*aL*aL)/(gamma*gamma - 1.0) * (fg*ML + 1.0)**2)
    cdef double E_fluxm = fm*( (2.0*aR*aR)/(gamma*gamma - 1.0) * (fg*MR - 1.0)**2)

    cdef (double, double, double) flux
    flux = (rho_fluxp + rho_fluxm, mom_fluxp + mom_fluxm, E_fluxp + E_fluxm)
    
    return flux


@cython.cdivision(True)
def get_flux_Godunov(
        double uL,
        double rhoL,
        double PL,
        double uR,
        double rhoR,
        double PR,
        double gamma=1.4,
        model=riemann_iterative
        
):
    cdef double rho, P, u, E
    rho, P, u, E = solve(uL, rhoL, PL, uR, rhoR, PR, 0.0, 1.0, gamma, model)
            
    cdef double f_rho = rho*u
    cdef double f_mom = rho*u*u + P
    cdef double f_E = u*(E + P)
    
    cdef (double, double, double) flux = (f_rho, f_mom, f_E)
    
    return flux


@cython.cdivision(True)
def get_flux_HLLC(
        double uL,
        double rhoL,
        double PL,
        double uR,
        double rhoR,
        double PR,
        double gamma=1.4,
        model=None
        
):
    # Pressure estimate from PVRS solver
    cdef double aL = sqrt((gamma*PL)/rhoL)
    cdef double aR = sqrt((gamma*PR)/rhoR)
    
    cdef double rho_bar = 0.5*(rhoL + rhoR)
    cdef double a_bar = 0.5*(aL + aR)
    
    cdef double Pguess = 0.5*(PL + PR) - 0.5*(uR - uL)*(rho_bar*a_bar)
    
    cdef double qL
    if Pguess <= PL:
        qL = 1.0
    else:
        qL = sqrt(1.0 + (gamma + 1.0)/(2.0*gamma)*(Pguess/PL - 1.0))
        
    if Pguess <= PR:
        qR = 1.0
    else:
        qR = sqrt(1.0 + (gamma + 1.0)/(2.0*gamma)*(Pguess/PR - 1.0))
    
    # Estimate wave speeds
    cdef double SL = uL - aL*qL
    cdef double SR = uR + aR*qR
    
    cdef double Sstar = PR - PL + rhoL*uL*(SL - uL) - rhoR*uR*(SR - uR)
    Sstar /= (rhoL*(SL - uL) - rhoR*(SR - uR))

    # Get energy and momenta on boundaries
    cdef double eL = PL/((gamma - 1.0)*rhoL)
    cdef double EL = rhoL*(0.5*uL*uL + eL)
    cdef double momL = uL*rhoL

    cdef double eR = PR/((gamma - 1.0)*rhoR)
    cdef double ER = rhoR*(0.5*uR*uR + eR)
    cdef double momR = uR*rhoR
    
    cdef double f_rho, f_mom, f_E
    
    cdef double PLR, d, fL_rho, fL_mom, fL_E, fR_rho, fR_mom, fR_E
    
    # Get flux   
    if 0 <= SL:
        # Flux = F(UL)
        f_rho = rhoL*uL
        f_mom = rhoL*uL*uL + PL
        f_E = uL*(EL + PL)
        
    elif SL < 0 <= Sstar:
        # Flux = F(*L)
        PLR = 0.5*(PL + PR + rhoL*(SL - uL)*(Sstar - uL) + rhoR*(SR - uR)*(Sstar - uR))
        d = SL - Sstar

        # Set F(UL)
        fL_rho = rhoL*uL
        fL_mom = rhoL*uL*uL + PL
        fL_E = uL*(EL + PL)
        
        # Find F(*L)
        f_rho = (Sstar*(SL*rhoL - fL_rho))/d
        f_mom = (Sstar*(SL*momL - fL_mom) + SL*PLR)/d
        f_E = (Sstar*(SL*EL - fL_E) + SL*PLR*Sstar)/d
                
    elif Sstar < 0 <= SR:
        # Flux = F(*R)
        PLR = 0.5*(PL + PR + rhoL*(SL - uL)*(Sstar - uL) + rhoR*(SR - uR)*(Sstar - uR))
        d = SR - Sstar
      
        # Set F(UR)
        fR_rho = rhoR*uR
        fR_mom = rhoR*uR*uR + PR
        fR_E = uR*(ER + PR)
        
        # Find F(*R)
        f_rho = (Sstar*(SR*rhoR - fR_rho))/d
        f_mom = (Sstar*(SR*momR - fR_mom) + SR*PLR)/d
        f_E = (Sstar*(SR*ER - fR_E) + SR*PLR*Sstar)/d
        
    else:
        # Flux = F(UR)
        f_rho = rhoR*uR
        f_mom = rhoR*uR*uR + PR
        f_E = uR*(ER + PR)
        
    cdef (double, double, double) flux = (f_rho, f_mom, f_E)
    
    return flux