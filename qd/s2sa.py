#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from .sn import * 

class S2SA(Sn):
	def __init__(self, sweeper):
		Sn.__init__(self, sweeper) 

		qorder = 2*self.p+1 
		mu, w = np.polynomial.legendre.leggauss(2) 
		L = []
		Mt = Assemble(self.space, MassIntegrator, self.sweeper.sigma_t, qorder)
		Ms = Assemble(self.space, MassIntegrator, self.sweeper.sigma_s, qorder)/2 
		G = Assemble(self.space, WeakConvectionIntegrator, 1, qorder)
		for n in range(len(w)):
			F = FaceAssembleAll(self.space, UpwindIntegrator, mu[n])
			A = mu[n]*G + F + Mt - Ms 
			L.append(A)  

		A = sp.bmat([[L[0], -Ms], [-Ms, L[1]]])
		self.lu = spla.splu(A.tocsc())
		self.rhs = np.zeros(2*self.space.Nu) 

	def FormRHS(self, phi, phi_old):
		diff = GridFunction(self.space)
		diff.data = phi.data - phi_old.data 
		s = self.sweeper.FormScattering(diff)
		return s.data

	def SourceIteration(self, psi, niter=50, tol=1e-6):
		phi_old = GridFunction(self.space)
		phi = self.ComputeScalarFlux(psi) 
		for n in range(niter):
			start = time.time() 
			phi_old.data = phi.data.copy() 
			self.sweeper.Sweep(psi, phi) 
			phi = self.ComputeScalarFlux(psi) 
			Q = self.FormRHS(phi, phi_old) 
			self.rhs[:self.space.Nu] = Q
			self.rhs[self.space.Nu:] = Q 
			s2psi = self.lu.solve(self.rhs) 
			dphi = s2psi[:self.space.Nu] + s2psi[self.space.Nu:]
			phi.data += dphi
			norm = phi.L2Diff(phi_old, 2*self.p+1)
			if (self.LOUD):
				el = time.time() - start 
				print('i={:3}, norm={:.3e}, {:.3f} s/iter'.format(n+1, norm, el))

			if (norm < tol):
				break 

		if (norm > tol):
			print(colored('WARNING not converged! Final tol = {:.3e}'.format(norm), 'red'))

		return phi 
