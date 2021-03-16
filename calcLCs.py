import LaplaceCoefficients as LC
# MD notation
def fd1(j, alpha):
    return(0.5*(-2*(j+1)*LC.b(0.5,j+1,alpha)
                  -alpha*LC.Db(0.5,j+1,alpha)))
def fd2(j, alpha):
    return(0.5*((-1+2*(j+1))*LC.b(0.5,j,alpha)
                 +alpha*LC.Db(0.5,j,alpha)))
def fs1(alpha):
    return((0.25*alpha*LC.Db(0.5,0,alpha) +
             alpha**2/8*0.5*(LC.Db(1.5,1,alpha)
                             -2*alpha*LC.Db(1.5,0,alpha) +LC.Db(1.5,1,alpha)
                             -2*LC.b(1.5,0,alpha))))
def fs2(alpha):
    return((0.5*LC.b(0.5,1,alpha) - 0.5*alpha*LC.Db(0.5,1.,alpha)
             -0.25*alpha**2*0.5*(LC.Db(1.5,0,alpha)
                                 -2*alpha*LC.Db(1.5,1,alpha) +LC.Db(1.5,2,alpha)
                                 -2*LC.b(1.5,1,alpha))))

print("Internal:")
j = 2.
alpha = (j/(j+1.))**(2./3.)
print("fd1 = {:0.2f}".format(fd1(j, alpha)))
print("fd2 = {:0.2f}".format(fd2(j, alpha)))
print("fs1 = {:0.2f}".format(fs1(alpha)))
print("fs2 = {:0.2f}".format(fs2(alpha)))

print("External:")
print("alpha*fd1 = {:0.2f}".format(alpha*fd1(j, alpha)))
print("alpha*fd2 = {:0.2f}".format(alpha*fd2(j, alpha)))
print("alpha*fs1 = {:0.2f}".format(alpha*fs1(alpha)))
print("alpha*fs2 = {:0.2f}".format(alpha*fs2(alpha)))

