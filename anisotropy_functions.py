# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 09:34:00 2016

@author: Agus
"""

# Define functions for anisotropy and crossed intensities calculated from parameters or data

def Anisotropy_FromInt(Ipar, Iper):
    return (Ipar - Iper)/(Ipar + 2*Iper)

def Anisotropy_FromFit(m, Am, Ad, b):
    return (Ad*b+(Am-b*Ad)*m)/(b+(1-b)*m)

def Fluos_FromInt(I_par, I_per):
    return I_par + 2 * I_per

def Fluos_FromFit(m, b, c_b_1):
    return 3000*c_b_1*(m+(1-m)*b)

def Int_par(M, A_1, A_2, b, c_b_1):
    return 3000*(2/3)*(((A_1+0.5)-(A_2+0.5)*b) *M + (A_2+0.5)*b)*c_b_1

def Int_per(M, A_1, A_2, b, c_b_1):
    return 3000*(1/3)*(((1-A_1) - (1-A_2)*b) *M + (1-A_2)*b)*c_b_1

def Int_par_n(M, A_1, A_2, b):
    return (2)*(((A_1+0.5)-(A_2+0.5)*b) *M + (A_2+0.5)*b)

def Int_per_n(M, A_1, A_2, b):
    return (((1-A_1) - (1-A_2)*b) *M + (1-A_2)*b)

def Int_par_r(M, A_1, A_2, b):
    return Int_par_n(M, A_1, A_2, b)/(Int_par_n(M, A_1, A_2, b)+2*Int_per_n(M, A_1, A_2, b))

def Int_per_r(M, A_1, A_2, b):
    return Int_per_n(M, A_1, A_2, b)/(Int_par_n(M, A_1, A_2, b)+2*Int_per_n(M, A_1, A_2, b))

def I_par_FromAnisotropy(Aniso, Fluo):
    return (2/3)*Fluo*(Aniso+0.5)

def I_per_FromAnisotropy(Aniso, Fluo):
    return (1/3)*Fluo*(1-Aniso)