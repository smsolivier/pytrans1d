#!/usr/bin/env python3

import numpy as np
import warnings
import time 

from .sweep import * 
from .. import utils 

class TVector:
	def __init__(self, space, quad):
		self.N = quad.N
		self.space = space 
		self.quad = quad 
		self.gf = [] 
		for n in range(self.N):
			self.gf.append(GridFunction(self.space))

	def GetAngle(self, angle):
		return self.gf[angle]

	def SetAngle(self, angle, gf):
		assert(isinstance(gf, np.ndarray))
		self.gf[angle].data = gf

	def Project(self, f):
		for a in range(self.N):
			spat = lambda x: f(x, self.quad.mu[a])
			self.gf[a].Project(spat) 

class Sn:
	def __init__(self, sweeper):
		self.sweeper = sweeper 
		self.space = sweeper.space 
		self.mu = sweeper.mu 
		self.w = sweeper.w 
		self.N = sweeper.N 
		self.LOUD = sweeper.LOUD 
		self.p = self.space.basis.p

	def ComputeScalarFlux(self, psi):
		phi = GridFunction(self.space) 
		for a in range(self.N):
			phi.data += self.w[a] * psi.GetAngle(a).data 

		return phi 

	def ComputeCurrent(self, psi):
		J = GridFunction(self.space)
		for a in range(self.N):
			J.data += self.w[a] * self.mu[a] * psi.GetAngle(a).data 

		return J 

	def SourceIteration(self, psi, niter=50, tol=1e-6):
		phi_old = GridFunction(self.space) 
		phi = self.ComputeScalarFlux(psi) 
		for n in range(niter):
			start = time.time() 
			phi_old.data = phi.data.copy()
			self.sweeper.Sweep(psi, phi) 
			phi = self.ComputeScalarFlux(psi) 
			norm = phi.L2Diff(phi_old, 2*self.p+1) 
			if (self.LOUD):
				el = time.time() - start 
				print('i={:3}, norm={:.3e}, {:.2f} s/iter'.format(n+1, norm, el))

			if (norm < tol):
				break 

		if (norm > tol):
			warnings.warn('source iteration not converged. final tol={:.3e}'.format(norm), 
				utils.ToleranceWarning, stacklevel=2)

		return phi 
