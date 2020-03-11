#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from sweep import * 
import time 
from termcolor import colored

class TVector:
	def __init__(self, space, N):
		self.N = N
		self.mu, self.w = quadrature.Get(N)
		self.space = space 
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
			spat = lambda x: f(x, self.mu[a])
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
			print(colored('WARNING not converged! Final tol = {:.3e}'.format(norm), 'red'))

		return phi 

if __name__=='__main__':
	Ne = 10
	p = 4
	N = 8
	basis = LegendreBasis(p)
	xe = np.linspace(0,1,Ne+1)
	space = L2Space(xe, basis) 
	psi = TVector(space, N) 
	sigma_t = lambda x: 1 
	sigma_s = lambda x: .9
	Q = lambda x, mu: (mu*np.pi*np.cos(np.pi*x) + (sigma_t(x)-sigma_s(x))*np.sin(np.pi*x))/2
	psi_in = lambda x, mu: 0 
	sweep = DirectSweeper(space, N, sigma_t, sigma_s, Q, psi_in)
	sn = Sn(sweep) 
	phi = sn.SourceIteration(psi)
	phi_ex = lambda x: np.sin(np.pi*x) 
	print('err = {:.3e}'.format(phi.L2Error(phi_ex, 2*p+1)))

	plt.plot(space.x, phi.data, '-o')
	xex = np.linspace(0,1,100)
	plt.plot(xex, phi_ex(xex), '--')
	plt.show()