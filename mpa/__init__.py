import os
import math
import subprocess

import numpy as np
from numpy import sin, cos, sqrt, pi, abs

import scipy as sp
import scipy.integrate
from scipy.special import hyp2f1

import matplotlib as mpl
import matplotlib.pyplot as plt

# from Laetitia
from . import LaplaceCoefficients as LC

# paralleliation 
from .run import *
from .plotting import *
from .helper import *
from .fndefs import *
# definitions of f1, f2, omega, nu
# replaces helper
from .plotter import *
# unified plotting parameters & size/axis
from .resonance import *
