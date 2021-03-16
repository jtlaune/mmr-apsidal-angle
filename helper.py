import numpy as np
import math
import LaplaceCoefficients as LC

# helper functions
def log_mean(x1, x2):
    return(np.exp(0.5*(np.log(x1)+np.log(x2))))
def om1ext_np(muext, a, ap, aext):
    return(muext*(a/ap)*np.sqrt(a/ap)*a*a/4/aext/aext*LC.b(1.5,1,a/aext))
def ompext_np(muext, a, ap, aext):
    return(muext*ap*ap/4/aext/aext*LC.b(1.5,1,ap/aext))
def A(alpha,j):
    # fd1 in MD
    return(0.5*(-2*(j+1)*LC.b(0.5,j+1,alpha)
                  -alpha*LC.Db(0.5,j+1,alpha)))
def B(alpha,j):
    # fd2 in MD
    return(0.5*((-1+2*(j+1))*LC.b(0.5,j,alpha)
                 +alpha*LC.Db(0.5,j,alpha)))
def C(alpha):
    # These are the values given in Murray Dermott.
    # signs on C and D are to be consistent with Laetitia's notes
    # fs1
    return((0.25*alpha*LC.Db(0.5,0,alpha) +
             alpha**2/8*0.5*(LC.Db(1.5,1,alpha)
                             -2*alpha*LC.Db(1.5,0,alpha) +LC.Db(1.5,1,alpha)
                             -2*LC.b(1.5,0,alpha))))
def D(alpha):
    # These are the values given in Murray Dermott.
    # signs on C and D are to be consistent with Laetitia's notes
    # fs1
    return((0.5*LC.b(0.5,1,alpha) - 0.5*alpha*LC.Db(0.5,1.,alpha)
             -0.25*alpha**2*0.5*(LC.Db(1.5,0,alpha)
                                 -2*alpha*LC.Db(1.5,1,alpha) +LC.Db(1.5,2,alpha)
                                 -2*LC.b(1.5,1,alpha))))

def check_Lgtr1(t, Y):
       L = Y[1]
       return (L-1)
def check_Lless1(t, Y):
        L = Y[1]
        return (1-L)
def smooth(y, box_pts):
    # convolve over #box_pts
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def round_sig(x, sig=2):
  return round(x, sig-int(math.floor(math.log10(abs(x))))-1)
def ebarfunc(ep, e, A, B, g):
    return(np.sqrt(e**2 + 2*B/A*ep*e*np.cos(g) + B**2/A**2*ep**2))
def etafunc(ep, e, A, B, g):
    return(sqrt(e**2 + 2*B/A*ep*e*cos(g) + B**2/A**2*ep**2))
def alpha0func(a, ap, j, ebar):
    return((a/ap)*(1+j*ebar**2))
