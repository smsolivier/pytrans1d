#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from qdf import * 
from sn import * 

class AbstractVEF(Sn):
	def __init__(self, phi_space, J_space, sweeper):
		Sn.__init__(self, sweeper) 
		self.phi_space = phi_space 
		self.J_space = J_space 

		p = J_space.basis.p
		self.p = p 
		sigma_a = lambda x: sweeper.sigma_t(x) - sweeper.sigma_s(x)
		self.Mt = Assemble(self.J_space, MassIntegrator, sweeper.sigma_t, 2*p+1)
		self.Ma = Assemble(self.phi_space, MassIntegrator, sigma_a, 2*p+1)
		self.D = MixAssemble(self.phi_space, self.J_space, MixDivIntegrator, 1, 2*p+1) 

		self.Q0 = np.zeros(phi_space.Nu)
		self.Q1 = np.zeros(J_space.Nu)
		for a in range(self.N):
			mu = self.mu[a]
			self.Q0 += AssembleRHS(phi_space, DomainIntegrator, lambda x: sweeper.Q(x,mu), 2*p+1)*self.w[a]
			self.Q1 += AssembleRHS(J_space, DomainIntegrator, lambda x: sweeper.Q(x,mu), 2*p+1)*mu*self.w[a]

		self.qdf = QDFactors(self.space, self.N, self.sweeper.psi_in) 

	def SourceIteration(self, psi, niter=50, tol=1e-6):
		phi = GridFunction(self.phi_space)
		phi_old = GridFunction(self.phi_space)
		for n in range(niter):
			start = time.time() 
			phi_old.data = phi.data.copy() 
			self.sweeper.Sweep(psi, phi) 
			phi = self.Mult(psi)
			norm = phi.L2Diff(phi_old, 2*self.p+1)

			if (self.LOUD):
				el = time.time() - start 
				print('i={:3}, norm={:.3e}, {:.2f} s/iter'.format(n+1, norm, el))

			if (norm < tol):
				break 

		if (norm > tol):
			print(colored('WARNING not converged! Final tol = {:.3e}'.format(norm), 'red'))

		return phi 

class VEF(AbstractVEF): 
	def __init__(self, phi_space, J_space, sweeper):
		AbstractVEF.__init__(self, phi_space, J_space, sweeper)

	def Mult(self, psi):
		self.qdf.Compute(psi) 
		B = BdrFaceAssemble(self.J_space, MLBdrIntegrator, self.qdf)
		G = MixAssemble(self.J_space, self.phi_space, MixWeakEddDivIntegrator, self.qdf, 2*self.p+1) 
		qin = BdrFaceAssembleRHS(self.J_space, VEFInflowIntegrator, self.qdf) 

		Mt = self.Mt + B 
		Q1 = self.Q1 + qin 

		A = sp.bmat([[Mt, G], [self.D, self.Ma]]).tocsc()
		rhs = np.concatenate((Q1, self.Q0))

		x = spla.spsolve(A, rhs) 
		phi = GridFunction(self.phi_space)
		phi.data = x[self.J_space.Nu:]

		return phi 

class VEFH(AbstractVEF):
	def __init__(self, phi_space, J_space, sweeper):
		AbstractVEF.__init__(self, phi_space, J_space, sweeper) 
		basis = LagrangeBasis(1)
		self.m_space = H1Space(self.phi_space.xe, basis) 

if __name__=='__main__':
	Ne = 10 
	N = 8
	p = 1 
	xe = np.linspace(0,1,Ne+1)
	leg = LegendreBasis(p)
	lob = LobattoBasis(p+1) 	
	phi_space = L2Space(xe, leg)
	J_space = H1Space(xe, lob) 
	sigma_t = lambda x: 1 
	sigma_s = lambda x: .1 
	Q = lambda x, mu: 1 
	psi_in = lambda x, mu: 0 
	sweep = DirectSweeper(phi_space, N, sigma_t, sigma_s, Q, psi_in)
	vef = VEF(phi_space, J_space, sweep)
	psi = TVector(phi_space, N)
	psi.Project(lambda x, mu: 1)
	phi = vef.SourceIteration(psi)
	# phi = vef.Mult(psi) 

	plt.plot(phi_space.x, phi.data, '-o')
	plt.show()