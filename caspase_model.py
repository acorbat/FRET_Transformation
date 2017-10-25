# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:01:38 2017

@author: Agus
"""
import numpy as np

from scipy.integrate import ode

# Define initial concentrations
Lsat    = 6E4 # saturating level of ligand (corresponding to ~1000 ng/ml SuperKiller TRAIL)
L50     = 3000 # baseline level of ligand for most experiments (corresponding to 50 ng/ml SuperKiller TRAIL)
RnosiRNA= 200 # TRAIL receptor (for experiments not involving siRNA)
RsiRNA  = 100000 # TRAIL receptor for experiments involving siRNA; this is set higher than in non-siRNA experiments to reflect the experimentally observed sensitization caused by both targeting and non-targeting siRNA transfection, which is believed to occur at least partly through the upregulation of death receptors by an interferon pathway.
flip  = 1E2  # Flip
pC8   = 2E4  # procaspase-8 (pro-C8)
Bar   = 1E3  # Bifunctional apoptosis regulator
pC3   = 1E4  # procaspase-3 (pro-C3)
pC6   = 1E4  # procaspase-6 (pro-C6)  
XIAP  = 1E5  # X-linked inhibitor of apoptosis protein  
PARP  = 1E6  # C3* substrate
Bid   = 4E4  # Bid
Bcl2c = 2E4  # cytosolic Bcl-2
Bax   = 1E5  # Bax
Bcl2  = 2E4  # mitochondrial Bcl-2  
M     = 5E5  # mitochondrial binding sites for activated Bax
CytoC = 5E5  # cytochrome c
Smac  = 1E5  # Smac    
pC9   = 1E5  # procaspase-9 (pro-C9) 1E4/2# 
Apaf  = 1E5  # Apaf-1

#our Sensors
S3 = 1E3     # Sensor for Caspase 3
S8 = 1E3     # Sensor for Caspase 8
S9 = 1E3     # Sensor for Caspase 9

Lfactor=L50/50 #relationship of ligand concentration in the model (in # molecules/cell) to actual TRAIL concentration (in ng/ml)

transloc=.01# rate of translocation between the cytosolic and mitochondrial compartments

v=.07 # mitochondria compartment volume/cell volume

# Initialize the full vector of initial conditions (IC) 
IC = np.zeros(68)
IC[1] = L50
IC[2] = RnosiRNA
IC[5] = flip
IC[7] = pC8
IC[10] = Bar
IC[12] = pC3
IC[15] = pC6
IC[19] = XIAP
IC[21] = PARP
IC[24] = Bid
IC[27] = Bcl2c
IC[29] = Bax
IC[33] = Bcl2
IC[39] = M
IC[42] = CytoC
IC[45] = Smac
IC[49] = Apaf
IC[52] = pC9
IC[59] = S3
IC[62] = S8
IC[65] = S9 #our Sensors


# Define rate reactions
k = np.zeros(32)
k_ = np.zeros(32)
kc = np.zeros(29)

# L + pR <--> L:pR --> R*
k[1]=4E-7
k_[1]=1E-3
kc[1]=1E-5

# flip + DISC <-->  flip:DISC  
k[2]=1E-6
k_[2]=1E-3

# pC8 + DISC <--> DISC:pC8 --> C8 + DISC
k[3]=1E-6
k_[3]=1E-3
kc[3]=1

# C8 + BAR <--> BAR:C8 
k[4]=1E-6
k_[4]=1E-3

# pC3 + C8 <--> pC3:C8 --> C3 + C8
k[5]=1E-7
k_[5]=1E-3
kc[5]=1

# pC6 + C3 <--> pC6:C3 --> C6 + C3
k[6]=1E-6
k_[6]=1E-3
kc[6]=1

# pC8 + C6 <--> pC8:C6 --> C8 + C6
k[7]=3E-8
k_[7]=1E-3
kc[7]=1 

# XIAP + C3 <--> XIAP:C3 --> XIAP + C3_U
#if C3_degrad == 1:
if True:
    k[8]=2E-6
    k_[8]=1E-3
    kc[8]=.1
else:
    k[8]=0
    k_[8]=0
    kc[8]=0

# PARP + C3 <--> PARP:C3 --> CPARP + C3
k[9]=1E-6
k_[9]=1E-2
kc[9]=1

# Bid + C8 <--> Bid:C8 --> tBid + C8
k[10]=1E-7
k_[10]=1E-3
kc[10]=1

# tBid + Bcl2c <-->  tBid:Bcl2c  
k[11]=1E-6
k_[11]=1E-3 

# Bax + tBid <--> Bax:tBid --> aBax + tBid 
#if Slower_Bax_Activation==0:
if True:
    k[12]=1E-7
    k_[12]=1E-3
    kc[12]=1
else:
    k[12]= 7*1E-8
    k_[12]=1E-3
    kc[12]=1

# aBax <-->  MBax 
k[13]=transloc
k_[13]=transloc

# MBax + Bcl2 <-->  MBax:Bcl2  
k[14]=1E-6
k_[14]=1E-3 

# MBax + MBax <-->  MBax:MBax == Bax2
k[15]=1E-6
k_[15]=1E-3

# Bax2 + Bcl2 <-->  MBax2:Bcl2  
k[16]=1E-6
k_[16]=1E-3 

# Bax2 + Bax2 <-->  Bax2:Bax2 == Bax4
k[17]=1E-6
k_[17]=1E-3

# Bax4 + Bcl2 <-->  MBax4:Bcl2  
k[18]=1E-6
k_[18]=1E-3 

# Bax4 + Mit0 <-->  Bax4:Mito -->  AMito  
k[19]=1E-6
k_[19]=1E-3
kc[19]=1

# AMit0 + mCtoC <-->  AMito:mCytoC --> AMito + ACytoC  
k[20]=2E-6
k_[20]=1E-3
kc[20]=10 

# AMit0 + mSMac <-->  AMito:mSmac --> AMito + ASMAC  
k[21]=2E-6
k_[21]=1E-3
kc[21]=10 

# ACytoC <-->  cCytoC
k[22]=transloc
k_[22]=transloc

# Apaf + cCytoC <-->  Apaf:cCytoC  
k[23]=5E-7
k_[23]=1E-3
kc[23]=1 

# Apaf:cCytoC + Procasp9 <-->  Apoptosome  
k[24]=5E-8
k_[24]=1E-3

# Apop + pCasp3 <-->  Apop:cCasp3 --> Apop + Casp3  
k[25]=5E-9
k_[25]=1E-3
kc[25]=1

# ASmac <-->  cSmac
k[26]=transloc
k_[26]=transloc

# Apop + XIAP <-->  Apop:XIAP  
k[27]=2E-6
k_[27]=1E-3

# cSmac + XIAP <-->  cSmac:XIAP  
k[28]=7E-6
k_[28]=1E-3

### Our Sensor Reactions
# S3 + C3 <--> S3:C8 --> SC3 + C3
# k[29]=1E-7 k_[29]=1E-3 kc[26]=1 # This is seen for C8
k[29]=1E-6
k_[29]=1E-2
kc[26]=1


# S8 + C8 <--> S8:C8 --> SC8 + C8
k[30]=1E-7
k_[30]=1E-3
kc[27]=1

# S9 + Apop <--> Apop:C9 --> SC9 + Apop
# k[31]=1E-7 k_[31]=1E-3 kc[28]=1 # This is seen for C8
k[31]=5E-9
k_[31]=1E-3
kc[28]=1


def rhs(t, x, k, k_, kc, v):
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


# Define simulator function
def simulate(t, IC):
    simulation = ode(rhs).set_integrator('lsoda', nsteps=100000)
    
    x = np.zeros((len(IC), len(t)))
    x[:, 0] = IC
    
    simulation.set_initial_value(x[:, 0], t[0]).set_f_params(k, k_, kc, v)
    
    for i, this_t in enumerate(t[1:]):
        i +=1
        if simulation.successful():
            simulation.integrate(this_t)
            x[:, i] = simulation.y
        else:
            print('simulation aborted')
            break
    
    return x