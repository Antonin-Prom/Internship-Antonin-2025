import numpy as np 
import pymd
from pymd.builder import *

phi = 0.4
L = 50
a = 1.0
random_init(phi, L, rcut=a, outfile='init.json')
