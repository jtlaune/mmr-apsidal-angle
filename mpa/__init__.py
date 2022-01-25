import os
import math
import subprocess
import sys

import importlib
import importlib.util

import numpy as np
from numpy import sin, cos, sqrt, pi, abs

import scipy as sp
import scipy.integrate

import matplotlib as mpl
import matplotlib.pyplot as plt

# from Laetitia
from . import LaplaceCoefficients as LC

# paralleliation 
from .run import *
from .plotting import *
from .fndefs import *
# definitions of f1, f2, omega, nu
# replaces helper
from .plotting import *
# unified plotting parameters & size/axis
from .resonance import *
