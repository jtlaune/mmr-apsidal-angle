from . import *
# helper functions
def log_mean(x1, x2):
    return np.exp(0.5 * (np.log(x1) + np.log(x2)))


def om1ext_np(muext, a, ap, aext):
    return (
        muext
        * (a / ap)
        * np.sqrt(a / ap)
        * a
        * a
        / 4
        / aext
        / aext
        * LC.b(1.5, 1, a / aext)
    )


def ompext_np(muext, a, ap, aext):
    return muext * ap * ap / 4 / aext / aext * LC.b(1.5, 1, ap / aext)


def f27lc(alpha, j):
    """
    f27 in MD p543
    (1/2)[−2 j − αD] b^(j)_{1/2}(α) x [e1cos(theta1)]
    """
    return(0.5 * ( -2 * (j + 1) * LC.b(0.5, j + 1, alpha) - alpha *
                   LC.Db(0.5, j + 1, alpha) ))


def f31lc(alpha, j):
    """
    f31 in MD p543
    (1/2)[−1 + 2 j + αD] b^(j-1)_{1/2}(α) x [e2cos(theta2)]
    """
    return(0.5 * ( (-1 + 2 * (j + 1)) * LC.b(0.5, j, alpha) + alpha *
                   LC.Db(0.5, j, alpha) ))


def sqr_ei_lc(alpha):
    """
    MD p275
    f3lc x [e1^2 + e2^2]
    (1/8)[2α_12 D + α_12^2 D^2]b_{1/2}^(0)
    """
    return(0.25 * alpha * LC.Db(0.5, 0, alpha) + alpha ** 2 / 8 * 0.5
           * ( LC.Db(1.5, 1, alpha) - 2 * alpha * LC.Db(1.5, 0, alpha) +
               LC.Db(1.5, 1, alpha) - 2 * LC.b(1.5, 0, alpha)))


def eiej_lc(alpha):
    """
    MD p275
    f4lc x [e1e2]
    (1/4)[2 − 2α_12 D − α_12^2 D^2]b_{1/2}(1)
    """
    # These are the values given in Murray Dermott.
    # signs on C and D are to be consistent with Laetitia's notes
    # fs1
    return(
        0.5 * LC.b(0.5, 1, alpha)
        - 0.5 * alpha * LC.Db(0.5, 1.0, alpha)
        - 0.25
        * alpha ** 2
        * 0.5
        * (
            LC.Db(1.5, 0, alpha)
            - 2 * alpha * LC.Db(1.5, 1, alpha)
            + LC.Db(1.5, 2, alpha)
            - 2 * LC.b(1.5, 1, alpha)
        )
    )


def smooth(y, box_pts):
    # convolve over #box_pts
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def round_sig(x, sig=2):
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def ebarfunc(ep, e, A, B, g):
    return np.sqrt(e ** 2 + 2 * B / A * ep * e * np.cos(g) + B ** 2 / A ** 2 * ep ** 2)


def etafunc(ep, e, A, B, g):
    return sqrt(e ** 2 + 2 * B / A * ep * e * cos(g) + B ** 2 / A ** 2 * ep ** 2)


def alpha0func(a, ap, j, ebar):
    return (a / ap) * (1 + j * ebar ** 2)


def format_e(x, pos):
    if x != 0:
        a = "{:0.1e}".format(x)
    else:
        a = "0"
    return a


def check_Lgtr1(t, Y):
    L = Y[1]
    return L - 1


def check_Lless1(t, Y):
    L = Y[1]
    return 1 - L


class check_ratio_cm:
    def __init__(self, ratio0, q):
        self.ratio0 = ratio0
        self.terminal = True
        self.q = q

    def __call__(self, t, Y):
        theta = Y[0]
        L1 = Y[1]
        L2 = Y[2]
        # super sloppy lol
        if np.isnan(theta):
            return(np.random.randn(1)[0])
        alpha1 = L1 ** 2 / self.q ** 2
        alpha2 = L2 ** 2 
        alpha = alpha1 / alpha2
        return(alpha - self.ratio0)
    

class check_ratio_tp:
    def __init__(self, ratio0):
        self.ratio0 = ratio0
        self.terminal = True

    def __call__(self, t, Y):
        L = Y[1]
        ratio = L * L
        return ratio - self.ratio0


def omega(self, a, ap, mu):
    if a <= ap:
        return (
            (np.pi * mu)
            * (a / ap) ** 2
            / (2 * a * sqrt(a))
            * LC.b(1.5, 1.0, (a / ap))
        )
    if a > ap:
        return (
            (np.pi * mu) * (a / ap) / (2 * a * sqrt(a)) * LC.b(1.5, 1.0, (a / ap))
        )

def nu(self, a, ap, mu):
    if a <= ap:
        return (
            (np.pi * mu)
            * (a / ap) ** 2
            / (2 * a * sqrt(a))
            * LC.b(1.5, 2.0, (a / ap))
        )
    if a > ap:
        return (
            (np.pi * mu) * (a / ap) / (2 * a * sqrt(a)) * LC.b(1.5, 2.0, (a / ap))
        )
