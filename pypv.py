#!/usr/bin/env python3

import ctypes 
import numpy as np 
from numpy.ctypeslib import ndpointer 
import sys

try:
	_eval = ctypes.CDLL('./polyval.so')
except:
	print('polyval.so not found') 
	print('build with gcc -fPIC -shared -o polyval.so polyval.c')
	print('or use np.polyval instead (slower since written in python)')
	sys.exit()

def PolyVal(B, x):
	# return np.polyval(B, x)
	N = B.shape[1] 
	p = B.shape[0] 
	val = np.zeros(N) 
	_eval.PolyVal(N, p, B.ctypes, ctypes.c_double(x), val.ctypes) 
	return val 