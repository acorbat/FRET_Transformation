# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:01:38 2017

@author: Agus
"""
import numpy as np
import pandas as pd
import lmfit as lm

from scipy.integrate import ode

#%% Define initial concentrations Parameters from Sorger

params = lm.Parameters()
# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
params.add_many(
        ('Lsat', 6E4, False, None, None, None, None), 
        ('L50', 3000, False, None, None, None, None),
        ('RnosiRNA', 200, False, None, None, None, None),
        ('RsiRNA', 100000, False, None, None, None, None),
        ('flip', 1E2, False, None, None, None, None),
        ('pC8', 2E4, False, None, None, None, None),
        ('Bar', 1E3, False, None, None, None, None),
        ('pC3', 1E4, False, None, None, None, None),
        ('pC6', 1E4, False, None, None, None, None),
        ('XIAP', 1E5, False, None, None, None, None),
        ('PARP', 1E6, False, None, None, None, None),
        ('Bid', 4E4, False, None, None, None, None),
        ('Bcl2c', 2E4, False, None, None, None, None),
        ('Bax', 1E5, False, None, None, None, None),
        ('Bcl2', 2E4, False, None, None, None, None),
        ('M', 5E5, False, None, None, None, None),
        ('CytoC', 5E5, False, None, None, None, None),
        ('Smac', 1E5, False, None, None, None, None),
        ('pC9', 1E5, False, None, None, None, None),
        ('Apaf', 1E5, False, None, None, None, None),
        ('S3', 1E3, False, None, None, None, None),
        ('S8', 1E3, False, None, None, None, None),
        ('S9', 1E3, False, None, None, None, None),
        ('transloc', .01, False, None, None, None, None),
        ('v', .07, False, None, None, None, None)
        )
params.add('Lfactor', value = params['L50']/50, vary=False)
# Lsat: saturating level of ligand (corresponding to ~1000 ng/ml SuperKiller TRAIL)
# L50: baseline level of ligand for most experiments (corresponding to 50 ng/ml SuperKiller TRAIL)
# RnosiRNA: TRAIL receptor (for experiments not involving siRNA)
# RsiRNA: TRAIL receptor for experiments involving siRNA; this is set higher than in non-siRNA experiments to reflect the experimentally observed sensitization caused by both targeting and non-targeting siRNA transfection, which is believed to occur at least partly through the upregulation of death receptors by an interferon pathway.
# flip: Flip
# pc8: procaspase-8 (pro-C8)
# Bar: Bifunctional apoptosis regulator
# pC3: procaspase-3 (pro-C3)
# pC6: procaspase-6 (pro-C6)  
# XIAP: X-linked inhibitor of apoptosis protein  
# PARP: C3* substrate
# Bid: Bid
# Bcl2c: cytosolic Bcl-2
# Bax: Bax
# Bcl2: mitochondrial Bcl-2  
# M: mitochondrial binding sites for activated Bax
# CytoC: cytochrome c
# Smac: Smac
# pc9: procaspase-9 (pro-C9) 1E4/2# 
# Apaf: Apaf-1
## our Sensors
# S3: Sensor for Caspase 3
# S8: Sensor for Caspase 8
# S9: Sensor for Caspase 9

# Lfactor: relationship of ligand concentration in the model (in # molecules/cell) to actual TRAIL concentration (in ng/ml)
# transloc: rate of translocation between the cytosolic and mitochondrial compartments
# v: mitochondria compartment volume/cell volume


# Initialize the full vector of initial conditions (IC) 
def _init_cc_from_params(params):
    """Cast Parameters into the list of initial conditions"""
    IC = np.zeros(68)
    IC[1]  = params['L50']
    IC[2]  = params['RnosiRNA']
    IC[5]  = params['flip']
    IC[7]  = params['pC8']
    IC[10] = params['Bar']
    IC[12] = params['pC3']
    IC[15] = params['pC6']
    IC[19] = params['XIAP']
    IC[21] = params['PARP']
    IC[24] = params['Bid']
    IC[27] = params['Bcl2c']
    IC[29] = params['Bax']
    IC[33] = params['Bcl2']
    IC[39] = params['M']
    IC[42] = params['CytoC']
    IC[45] = params['Smac']
    IC[49] = params['Apaf']
    IC[52] = params['pC9']
    IC[59] = params['S3']
    IC[62] = params['S8']
    IC[65] = params['S9']
    
    return IC


#%% Define rate reactions from Sorger

params.add_many(
        ('L_ku', 4E-7, False, None, None, None, None), 
        ('L_kd', 1E-3, False, None, None, None, None), 
        ('L_kc', 1E-5, False, None, None, None, None), 
        ('flip_ku', 1E-6, False, None, None, None, None),
        ('flip_kd', 1E-3, False, None, None, None, None),
        ('DISC_ku', 1E-6, False, None, None, None, None), 
        ('DISC_kd', 1E-3, False, None, None, None, None), 
        ('DISC_kc', 1, False, None, None, None, None), 
        ('Bar_ku', 1E-6, False, None, None, None, None),
        ('Bar_kd', 1E-3, False, None, None, None, None),
        ('C8_ku', 1E-7, False, None, None, None, None), 
        ('C8_kd', 1E-3, False, None, None, None, None), 
        ('C8_kc', 1, False, None, None, None, None), 
        ('C3_ku', 1E-6, False, None, None, None, None), 
        ('C3_kd', 1E-3, False, None, None, None, None), 
        ('C3_kc', 1, False, None, None, None, None), 
        ('C6_ku', 3E-8, False, None, None, None, None), 
        ('C6_kd', 1E-3, False, None, None, None, None), 
        ('C6_kc', 1, False, None, None, None, None),
        ('XIAP_ku', 2E-6, False, None, None, None, None), 
        ('XIAP_kd', 1E-3, False, None, None, None, None), 
        ('XIAP_kc', .1, False, None, None, None, None),
        ('PARP_ku', 1E-6, False, None, None, None, None), 
        ('PARP_kd', 1E-2, False, None, None, None, None), 
        ('PARP_kc', 1, False, None, None, None, None),
        ('Bcl2c_ku', 1E-6, False, None, None, None, None),
        ('Bcl2c_kd', 1E-3, False, None, None, None, None),
        ('tBid_ku', 1E-7, False, None, None, None, None), 
        ('tBid_kd', 1E-3, False, None, None, None, None), 
        ('tBid_kc', 1, False, None, None, None, None),
        ('Bcl2_ku', 1E-6, False, None, None, None, None),
        ('Bcl2_kd', 1E-3, False, None, None, None, None),
        ('MBax_ku', 1E-6, False, None, None, None, None),
        ('MBax_kd', 1E-3, False, None, None, None, None),
        ('Bax4_ku', 1E-6, False, None, None, None, None), 
        ('Bax4_kd', 1E-3, False, None, None, None, None), 
        ('Bax4_kc', 1, False, None, None, None, None),
        ('AMito_ku', 2E-6, False, None, None, None, None), 
        ('AMito_kd', 1E-3, False, None, None, None, None), 
        ('AMito_kc', 10, False, None, None, None, None),
        ('CytoC_ku', 5E-7, False, None, None, None, None), 
        ('CytoC_kd', 1E-3, False, None, None, None, None), 
        ('CytoC_kc', 1, False, None, None, None, None),
        ('Apaf_ku', 5E-8, False, None, None, None, None),
        ('Apaf_kd', 1E-3, False, None, None, None, None),
        ('C9_ku', 5E-9, False, None, None, None, None), 
        ('C9_kd', 1E-3, False, None, None, None, None), 
        ('C9_kc', 1, False, None, None, None, None), 
        ('SMAC_ku', 7E-6, False, None, None, None, None), 
        ('SMAC_kd', 1E-3, False, None, None, None, None), 
        )

def _k_from_params(params):
    """Cast Parameters into the list of k, k_ and kc"""
    k = np.zeros(32)
    k_ = np.zeros(32)
    kc = np.zeros(29)
    
    # L + pR <--> L:pR --> R*
    k[1]  = params['L_ku']
    k_[1] = params['L_kd']
    kc[1] = params['L_kc']
    
    # flip + DISC <-->  flip:DISC  
    k[2]  = params['flip_ku']
    k_[2] = params['flip_kd']
    
    # pC8 + DISC <--> DISC:pC8 --> C8 + DISC
    k[3]  = params['DISC_ku']
    k_[3] = params['DISC_kd']
    kc[3] = params['DISC_kc']
    
    # C8 + BAR <--> BAR:C8 
    k[4]  = params['Bar_ku']
    k_[4] = params['Bar_kd']
    
    # pC3 + C8 <--> pC3:C8 --> C3 + C8
    k[5]  = params['C8_ku']
    k_[5] = params['C8_kd']
    kc[5] = params['C8_kc']
    
    # pC6 + C3 <--> pC6:C3 --> C6 + C3
    k[6]  = params['C3_ku']
    k_[6] = params['C3_kd']
    kc[6] = params['C3_kc']
    
    # pC8 + C6 <--> pC8:C6 --> C8 + C6
    k[7]  = params['C6_ku']
    k_[7] = params['C6_kd']
    kc[7] = params['C6_kc'] 
    
    # XIAP + C3 <--> XIAP:C3 --> XIAP + C3_U
    k[8]  = params['XIAP_ku']
    k_[8] = params['XIAP_kd']
    kc[8] = params['XIAP_kc']
    
    # PARP + C3 <--> PARP:C3 --> CPARP + C3; NOT repetead with C3
    k[9]  = params['PARP_ku']
    k_[9] = params['PARP_kd']
    kc[9] = params['PARP_kc']
    
    # Bid + C8 <--> Bid:C8 --> tBid + C8; Repeated with C8
    k[10]  = params['C8_ku']
    k_[10] = params['C8_kd']
    kc[10] = params['C8_kc']
    
    # tBid + Bcl2c <-->  tBid:Bcl2c  
    k[11]  = params['Bcl2c_ku']
    k_[11] = params['Bcl2c_kd']
    
    # Bax + tBid <--> Bax:tBid --> aBax + tBid 
    k[12]  = params['tBid_ku']
    k_[12] = params['tBid_kd']
    kc[12] = params['tBid_kc']
    
    # aBax <-->  MBax 
    k[13]  = params['transloc']
    k_[13] = params['transloc']
    
    # MBax + Bcl2 <-->  MBax:Bcl2  
    k[14]  = params['Bcl2_ku']
    k_[14] = params['Bcl2_kd']
    
    # MBax + MBax <-->  MBax:MBax == Bax2
    k[15]  = params['MBax_ku']
    k_[15] = params['MBax_kd']
    
    # Bax2 + Bcl2 <-->  MBax2:Bcl2; Repetaed with Bcl2  
    k[16]  = params['Bcl2_ku']
    k_[16] = params['Bcl2_kd']
    
    # Bax2 + Bax2 <-->  Bax2:Bax2 == Bax4; Repetead with MBax
    k[17]  = params['MBax_ku']
    k_[17] = params['MBax_kd']
    
    # Bax4 + Bcl2 <-->  MBax4:Bcl2; Repetaed with Bcl2  
    k[18]  = params['Bcl2_ku']
    k_[18] = params['Bcl2_kd']
    
    # Bax4 + Mit0 <-->  Bax4:Mito -->  AMito  
    k[19]  = params['Bax4_ku']
    k_[19] = params['Bax4_kd']
    kc[19] = params['Bax4_kc']
    
    # AMit0 + mCtoC <-->  AMito:mCytoC --> AMito + ACytoC  
    k[20]  = params['AMito_ku']
    k_[20] = params['AMito_kd']
    kc[20] = params['AMito_kc']
    
    # AMit0 + mSMac <-->  AMito:mSmac --> AMito + ASMAC; Repetead with AMito
    k[21]  = params['AMito_ku']
    k_[21] = params['AMito_kd']
    kc[21] = params['AMito_kc']
    
    # ACytoC <-->  cCytoC
    k[22]  = params['transloc']
    k_[22] = params['transloc']
    
    # Apaf + cCytoC <-->  Apaf:cCytoC  
    k[23]  = params['CytoC_ku']
    k_[23] = params['CytoC_kd']
    kc[23] = params['CytoC_kc']
    
    # Apaf:cCytoC + Procasp9 <-->  Apoptosome  
    k[24]  = params['Apaf_ku']
    k_[24] = params['Apaf_kd']
    
    # Apop + pCasp3 <-->  Apop:cCasp3 --> Apop + Casp3  
    k[25]  = params['C9_ku']
    k_[25] = params['C9_kd']
    kc[25] = params['C9_kc']
    
    # ASmac <-->  cSmac
    k[26]  = params['transloc']
    k_[26] = params['transloc']
    
    # Apop + XIAP <-->  Apop:XIAP; Repeated XIAP  
    k[27]  = params['XIAP_ku']
    k_[27] = params['XIAP_kd']
    
    # cSmac + XIAP <-->  cSmac:XIAP  
    k[28]  = params['SMAC_ku']
    k_[28] = params['SMAC_kd']
    
    ### Our Sensor Reactions
    # S3 + C3 <--> S3:C8 --> SC3 + C3; Repeated from PARP (should it be C3?)
    k[29]  = params['PARP_ku']
    k_[29] = params['PARP_kd']
    kc[26] = params['PARP_kc']
    
    # S8 + C8 <--> S8:C8 --> SC8 + C8
    k[30]  = params['C8_ku']
    k_[30] = params['C8_kd']
    kc[27] = params['C8_kc']
    
    # S9 + Apop <--> Apop:C9 --> SC9 + Apop
    k[31]  = params['C9_ku']
    k_[31] = params['C9_kd']
    kc[28] = params['C9_kc']
    
    return k, k_, kc


#%% Define ODE System


def rhs(t, x, k, k_, kc, v):
    """Ordinary Differential Equations defined from the k, k_, kc and v"""
    # ODE with dimensions 
    xp = np.array(x)
    
    xp[1] = -k[1]*x[1]*x[2] +k_[1]*x[3]  # Ligand 
    
    xp[2] = -k[1]*x[1]*x[2] +k_[1]*x[3]  # R
    
    xp[3] =  k[1]*x[1]*x[2] -k_[1]*x[3] -kc[1]*x[3]  # L:R complex
    
    xp[4] = kc[1]*x[3] \
    -k[2]*x[4]*x[5] +k_[2]*x[6] \
    -k[3]*x[4]*x[7] +k_[3]*x[8] +kc[3]*x[8] # R*
    
    xp[5]= -k[2]*x[4]*x[5] +k_[2]*x[6]  # flip
    
    xp[6]=  k[2]*x[4]*x[5] -k_[2]*x[6]  # flip:R*
    
    xp[7] = -k[3]*x[4]*x[7] +k_[3]*x[8] \
    -k[7]*x[7]*x[17] +k_[7]*x[18]  # pC8 
    
    xp[8] =  k[3]*x[4]*x[7] -k_[3]*x[8] -kc[3]*x[8]  # R*:pC8
    
    xp[9] =  kc[3]*x[8] \
    -k[4]*x[9]*x[10] +k_[4]*x[11] \
    -k[5]*x[9]*x[12] +k_[5]*x[13] +kc[5]*x[13] \
    +kc[7]*x[18] \
    -k[10]*x[9]*x[24] +k_[10]*x[25] +kc[10]*x[25] \
    -k[30]*x[9]*x[62] +k_[30]*x[63] +kc[27]*x[63] # C8
    
    xp[10] = -k[4]*x[9]*x[10] +k_[4]*x[11]  # Bar
    
    xp[11] =  k[4]*x[9]*x[10] -k_[4]*x[11]  # Bar:C8
    
    xp[12]= -k[5]*x[9]*x[12] +k_[5]*x[13] \
    -k[25]*x[12]*x[53] +k_[25]*x[54]  # pC3
    
    xp[13]=  k[5]*x[9]*x[12] -k_[5]*x[13] -kc[5]*x[13]  # C8:pC3
    
    xp[14]=  kc[5]*x[13] \
    -k[6]*x[14]*x[15] +k_[6]*x[16] +kc[6]*x[16] \
    -k[8]*x[14]*x[19] +k_[8]*x[20] \
    -k[9]*x[14]*x[21] +k_[9]*x[22] +kc[9]*x[22] \
    +kc[25]*x[54] \
    -k[29]*x[14]*x[59] + k_[29]*x[60] + kc[26]*x[60] # C3 
    
    xp[15]= -k[6]*x[14]*x[15] +k_[6]*x[16]  # pC6
    
    xp[16]=  k[6]*x[14]*x[15] -k_[6]*x[16] -kc[6]*x[16]  # C3:pC6
    
    xp[17]=  kc[6]*x[16] \
    -k[7]*x[7]*x[17] +k_[7]*x[18] +kc[7]*x[18]  # C6
    
    xp[18]=  k[7]*x[7]*x[17] -k_[7]*x[18] -kc[7]*x[18]  # C6:pC8
    
    xp[19]= -k[8]*x[14]*x[19] +k_[8]*x[20] +kc[8]*x[20] \
    -k[27]*x[19]*x[53] +k_[27]*x[56] \
    -k[28]*x[19]*x[55] +k_[28]*x[57]  # XIAP
    
    xp[20]=  k[8]*x[14]*x[19] -k_[8]*x[20] -kc[8]*x[20]  # XIAP:C3
    
    xp[21]= -k[9]*x[14]*x[21] +k_[9]*x[22]  # PARP
    
    xp[22]=  k[9]*x[14]*x[21] -k_[9]*x[22] -kc[9]*x[22]  # C3:PARP
    
    xp[23]= kc[9]*x[22]  # CPARP
    
    xp[24]= -k[10]*x[9]*x[24] +k_[10]*x[25]  # Bid
    
    xp[25]=  k[10]*x[9]*x[24] -k_[10]*x[25] -kc[10]*x[25]  # C8:Bid
    
    xp[26]=  kc[10]*x[25] \
    -k[11]*x[26]*x[27] +k_[11]*x[28] \
    -k[12]*x[26]*x[29] +k_[12]*x[30] + kc[12]*x[30] # tBid
    
    xp[27]= -k[11]*x[26]*x[27] +k_[11]*x[28]  # Bcl2c
    
    xp[28]= +k[11]*x[26]*x[27] -k_[11]*x[28]  # Bcl2c:tBid
    
    xp[29]= -k[12]*x[26]*x[29] +k_[12]*x[30]  # Bax
    
    xp[30]=  k[12]*x[26]*x[29] -k_[12]*x[30] - kc[12]*x[30]  # tBid:Bax
    
    xp[31]=  kc[12]*x[30] \
    -k[13]*x[31] + k_[13]*x[32]  # Bax*
    
    xp[32]=  k[13]*x[31] - k_[13]*x[32] \
    -1/v*k[14]*x[32]*x[33] +k_[14]*x[34] \
    -1/v*2*k[15]*x[32]**2 +2*k_[15]*x[35]  # Baxm
    
    xp[33]= -1/v*k[14]*x[32]*x[33] +k_[14]*x[34] \
    -1/v*k[16]*x[33]*x[35] +k_[16]*x[36] \
    -1/v*k[18]*x[33]*x[37] +k_[18]*x[38]  # Bcl2 
    
    xp[34]=  1/v*k[14]*x[32]*x[33] -k_[14]*x[34]  # Baxm:Bcl2
    
    xp[35]=  1/v*k[15]*x[32]**2 -k_[15]*x[35] \
    -1/v*k[16]*x[33]*x[35] +k_[16]*x[36] \
    -2/v*k[17]*x[35]**2 +2*k_[17]*x[37]  # Bax2
    
    xp[36]=  1/v*k[16]*x[33]*x[35] -k_[16]*x[36]  # Bax2:Bcl2
    
    xp[37]= 1/v*k[17]*x[35]**2 -k_[17]*x[37] \
    -1/v*k[18]*x[33]*x[37] +k_[18]*x[38] \
    -1/v*k[19]*x[39]*x[37] +k_[19]*x[40]  # Bax4 
    
    xp[38]= 1/v*k[18]*x[33]*x[37] -k_[18]*x[38]  # Bax4:Bcl2
    
    xp[39]= -1/v*k[19]*x[39]*x[37] +k_[19]*x[40] # M
    
    xp[40]=  1/v*k[19]*x[39]*x[37] -k_[19]*x[40] -kc[19]*x[40]  # Bax4:M
    
    xp[41]=  kc[19]*x[40] \
    -1/v*k[20]*x[41]*x[42] +k_[20]*x[43] +kc[20]*x[43] \
    -1/v*k[21]*x[41]*x[45] +k_[21]*x[46] +kc[21]*x[46]  # M*
    
    xp[42]= -1/v*k[20]*x[41]*x[42] +k_[20]*x[43]  # CytoCm
    
    xp[43]=  1/v*k[20]*x[41]*x[42] -k_[20]*x[43] -kc[20]*x[43]  # M*:CytoCm
    
    xp[44]=  kc[20]*x[43] \
    -k[22]*x[44] +k_[22]*x[48]  # CytoCr
    
    xp[45]= -1/v*k[21]*x[41]*x[45] +k_[21]*x[46]  # Smacm
    
    xp[46]=  1/v*k[21]*x[41]*x[45] -k_[21]*x[46] -kc[21]*x[46]  # M*:Smacm
    
    xp[47]=  kc[21]*x[46] \
    -k[26]*x[47] +k_[26]*x[55]  # Smacr
    
    xp[48]=  k[22]*x[44] -k_[22]*x[48] \
    -k[23]*x[48]*x[49] +k_[23]*x[50] +kc[23]*x[50]  # CytoC
    
    xp[49]= -k[23]*x[48]*x[49] +k_[23]*x[50]  # Apaf
    
    xp[50]=  k[23]*x[48]*x[49] -k_[23]*x[50] -kc[23]*x[50]  # Apaf:CytoC
    
    xp[51]= kc[23]*x[50] \
    -k[24]*x[51]*x[52] +k_[24]*x[53] # Apaf*
    
    xp[52]= -k[24]*x[51]*x[52] +k_[24]*x[53]  # pC9
    
    xp[53]=  k[24]*x[51]*x[52] -k_[24]*x[53] \
    -k[25]*x[12]*x[53] +k_[25]*x[54] +kc[25]*x[54] \
    -k[27]*x[19]*x[53] +k_[27]*x[56] \
    -k[31]*x[53]*x[65] +k_[31]*x[66] + kc[28]*x[66] # Apop
    
    xp[54]=  k[25]*x[12]*x[53] -k_[25]*x[54] -kc[25]*x[54]  # Apop:pC3
    
    xp[55]=  k[26]*x[47] -k_[26]*x[55] \
    -k[28]*x[19]*x[55] +k_[28]*x[57]  # Smac
    
    xp[56]=  k[27]*x[19]*x[53] -k_[27]*x[56]  # Apop:XIAP 
    
    xp[57]=  k[28]*x[19]*x[55] -k_[28]*x[57]  # Smac:XIap
    
    xp[58]=  kc[8]*x[20] # C3_Ub
    
    xp[59]= -k[29]*x[14]*x[59] +k_[29]*x[60] # S3
    
    xp[60]=  k[29]*x[14]*x[59] -k_[29]*x[60] -kc[26]*x[60] # S3:C3
    
    xp[61]=  2*kc[26]*x[60] #SC3
    
    xp[62]= -k[30]*x[9]*x[62] +k_[30]*x[63] # S8
    
    xp[63]=  k[30]*x[9]*x[62] -k_[30]*x[63] -kc[27]*x[63] # S8:C8
    
    xp[64]=  2*kc[27]*x[63] #SC8
    
    xp[65]= -k[31]*x[53]*x[65] +k_[31]*x[66] # S9
    
    xp[66]=  k[31]*x[53]*x[65] -k_[31]*x[66] -kc[28]*x[66] # S9:C9
    
    xp[67]=  2*kc[28]*x[66] #SC9
    
    return xp


#%% Define simulator function

def simulate(t, params):
    """It simulates Sorger Model for the t times using the parameters in 
    params."""
    simulation = ode(rhs).set_integrator('lsoda')
    
    x = np.zeros((68, len(t)))
    x[:, 0] = _init_cc_from_params(params)
    k, k_, kc = _k_from_params(params)
    v = params['v']
    
    simulation.set_initial_value(x[:, 0], t[0]).set_f_params(k, k_, kc, v)
    
    for i, this_t in enumerate(t[1:]):
        i +=1
        if simulation.successful():
            simulation.integrate(this_t)
            x[:, i] = simulation.y
        else:
            print('simulation aborted')
            break
    
    # List of variables used
    variables = ['Ligand', 'R', 'L:R', 'R*', 'flip', 'flip:R*', 'pC8', 
                 'R*:pC8', 'C8', 'Bar', 'Bar:C8', 'pC3', 'C8:pC3', 'C3', 'pC6',
                 'C3:pC6', 'C6', 'C6:pC8', 'XIAP', 'XIAP:C3', 'PARP', 
                 'C3:PARP', 'CPARP', 'Bid', 'C8:Bid', 'tBid', 'Bcl2c', 
                 'Bcl2c:tBid', 'Bax', 'tBid:Bax', 'Bax*','Baxm', 'Bcl2', 
                 'Baxm:Bcl2', 'Bax2', 'Bax2:Bcl2', 'Bax4', 'Bax4:Bcl2', 'M', 
                 'Bax4:M', 'M*', 'CytoCm', 'M*:CytoCm', 'CytoCr', 'Smacm', 
                 'M*:Smacm', 'Smacr', 'CytoC', 'Apaf', 'Apaf:CytoC', 'Apaf*',
                 'pC9', 'Apop', 'Apop:pC3', 'Smac', 'Apop:XIAP', 'Smac:XIAP',
                 'C3_Ub', 'S3', 'S3:C3', 'SC3', 'S8', 'S8:C8', 'SC8', 'S9', 
                 'S9:C9', 'SC9']
    
    this_dict = {var: x[i+1, :] for i, var in enumerate(variables)}
    res = pd.DataFrame.from_dict(this_dict)
    res['t'] = t/60
    
    return res