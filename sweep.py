#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from quadrature import quadrature
from integrators import * 

class AbstractSweeper:
	def __init__(self, space, N, sigma_t, sigma_s, Q, psi_in, LOUD=True, scatter_space=None):
		self.space = space 
		self.N = N 
		self.mu, self.w = quadrature.Get(self.N) 
		self.sigma_t = sigma_t 
		self.sigma_s = sigma_s 
		self.Q = Q 
		self.psi_in = psi_in 
		self.LOUD = LOUD 
		self.scatter_space = scatter_space
		if (scatter_space==None):
			self.scatter_space = self.space 

	def FormScattering(self, phi):
		scat = GridFunction(self.space)
		scat.data = self.Ms * phi.data * .5 
		return scat 

class DirectSweeper(AbstractSweeper):
	def __init__(self, space, N, sigma_t, sigma_s, Q, psi_in, LOUD=True, scatter_space=None):
		AbstractSweeper.__init__(self, space, N, sigma_t, sigma_s, Q, psi_in, LOUD, scatter_space)

		p = self.space.basis.p
		self.Mt = Assemble(self.space, MassIntegrator, self.sigma_t, 2*p+1)
		self.Ms = MixAssemble(self.space, self.scatter_space, MixMassIntegrator, self.sigma_s, 2*p+1)
		G = Assemble(self.space, WeakConvectionIntegrator, 1, 2*p+1)
		self.LHS = [] 
		self.lu = [] 
		self.RHS = [] 
		for n in range(self.N):
			mu = self.mu[n] 
			F = FaceAssembleAll(self.space, UpwindIntegrator, mu) 
			I = BdrFaceAssembleRHS(self.space, InflowIntegrator, [mu, lambda x: psi_in(x,mu)])
			b = AssembleRHS(self.space, DomainIntegrator, lambda x: self.Q(x,mu), 2*p+1) 
			A = mu*G + F + self.Mt 
			b += I 

			self.LHS.append(A) 
			self.RHS.append(b)
			self.lu.append(spla.splu(A))

	def Sweep(self, psi, phi):
		scat = self.FormScattering(phi) 
		for a in range(self.N):
			angle = self.lu[a].solve(self.RHS[a] + scat.data) 
			psi.SetAngle(a, angle) 
			