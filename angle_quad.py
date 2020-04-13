#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class LegendreQuad:
	def __init__(self, N):
		self.N = N 
		self.mu, self.w = np.polynomial.legendre.leggauss(N)

class DoubleLegendreQuad:
	def __init__(self, N):
		mu, w = np.polynomial.legendre.leggauss(N)
		mul = .5*mu - .5 
		mur = .5*mu + .5 
		self.mu = np.concatenate((mul, mur))
		self.w = np.concatenate((.5*w, .5*w))
		self.N = len(self.w)