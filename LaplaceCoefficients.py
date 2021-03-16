###
# Computation of Laplace coefficients from Murray & Dermott 1999
# Original Author: Laetitia Rodet 02-2020
###
import numpy as np
from scipy.special import hyp2f1

def Pochhammer(a, k):
    if k == 0:
        return(1.)
    else:
        return( (a+k-1)*Pochhammer(a, k-1) )

def b(s, j, alpha):
    if j >= 0: # Eq. 7.87
        return(2*Pochhammer(s, j)/Pochhammer(1, j)*(alpha**j)*hyp2f1(s, s+j, j+1, alpha*alpha))
    else: # Eq. 6.69
        return(b(s, -j, alpha))

def Db(s, j, alpha): # Eq. 6.70
    aux = b(s+1, j-1, alpha) - 2*alpha*b(s+1, j, alpha) + b(s+1, j+1, alpha)
    return(s*aux)

def D2b(s, j, alpha): # Eq. 6.71
    aux = Db(s+1, j-1, alpha) - 2*alpha*Db(s+1, j, alpha) - 2*b(s+1, j, alpha) + Db(s+1, j+1, alpha)
    return(s*aux)

# p+q:p MMR, Table 8.1
#p = 2; q = 1
#j = p+q
#if q == 1:
#    def fd(alpha, j):
#        aux = -2*j*b(0.5, j, alpha) - alpha*Db(0.5, j, alpha)
#        aux /= 2.
#        return(aux)
#elif q == 2:
#    def fd(alpha, j):
#        aux = (-5*j + 4*j*j)*b(0.5, j, alpha) + (-2+4*j)*alpha*Db(0.5, j, alpha) + alpha*alpha*D2b(0.5, j, alpha)
#        aux /= 8.
#        return(aux)
#
#alphares = (float(j-q)/j)**(2./3.)
#afd = alphares*fd(alphares, j)
#
#print(p+q, ":", p, "MMR")
#print("alphares", alphares)
#print("alpha fd", afd)
