#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from .basis import * 
from .pypv import PolyVal

class Element:
	def __init__(self, basis, line, elno=-1):
		self.basis = basis 
		self.Nn = basis.p+1
		self.line = line 
		self.h = line[1] - line[0] 
		self.ElNo = elno 
		self.J = self.h/2 # jacobian 
		self.nodes = self.Transform(basis.ip) 

	def CalcShape(self, xi):
		return PolyVal(self.basis.B, xi)

	def CalcGradShape(self, xi):
		return PolyVal(self.basis.dB, xi)

	def CalcPhysGradShape(self, xi):
		return self.CalcGradShape(xi)/self.J 

	def Jacobian(self, xi):
		return self.J 

	def Transform(self, xi):
		return (self.line[0] + self.line[1])/2 + self.h/2*xi 

	def Interpolate(self, xi, u):
		shape = self.CalcShape(xi)
		return np.dot(u, shape) 

	def InterpolateGrad(self, xi, u):
		pgshape = self.CalcPhysGradShape(xi)
		return np.dot(u, pgshape) 

	def InverseMap(self, x, tol=1e-12, niter=20):
		xi = 0 
		for n in range(niter):
			xi_old = xi 
			xi = xi_old + 1/self.Jacobian(xi_old)*(x - self.Transform(xi_old))

			diff = abs(self.Transform(xi) - x)
			if (diff < tol):
				break 

		if (diff > tol):
			print('inverse map not converged')

		return xi 
