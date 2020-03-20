#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from sn import * 

class P1SA(Sn):
	def __init__(self, sweeper):
		Sn.__init__(self, sweeper) 

		qorder = 2*self.p+1 
		sigma_a = lambda x: sweeper.sigma_t(x) - sweeper.sigma_s(x)
		Ma = Assemble(self.space, MassIntegrator, sigma_a, qorder)
		D = MixAssemble(self.space, self.space, WeakMixDivIntegrator, 1, qorder) 
		GT = MixAssemble(self.space, self.space, MixDivIntegrator, 1, qorder) 
		G = -GT.transpose()
		Mt = 3*Assemble(self.space, MassIntegrator, sweeper.sigma_t, qorder) 

		D += MixFaceAssembleAll(self.space, self.space, MixJumpAvgIntegrator, 1)
		Ma += FaceAssembleAll(self.space, JumpJumpIntegrator, .25)
		G += MixFaceAssembleAll(self.space, self.space, MixJumpAvgIntegrator, 1)
		Mt += FaceAssembleAll(self.space, JumpJumpIntegrator, 1)

		A = sp.bmat([[Ma, D], [G, Mt]])
		self.lu = spla.splu(A.tocsc()) 
		self.rhs = np.zeros(2*self.space.Nu)

	def FormRHS(self, phi, phi_old):
		diff = GridFunction(self.space)
		diff.data = phi.data - phi_old.data 
		s = self.sweeper.FormScattering(diff) 
		return s.data * 2

	def SourceIteration(self, psi, niter=50, tol=1e-6):
		phi_old = GridFunction(self.space) 
		phi = self.ComputeScalarFlux(psi) 
		for n in range(niter):
			start = time.time() 
			phi_old.data = phi.data.copy()
			self.sweeper.Sweep(psi, phi) 
			phi = self.ComputeScalarFlux(psi) 
			self.rhs[:self.space.Nu] = self.FormRHS(phi, phi_old) 
			x = self.lu.solve(self.rhs) 
			phi.data += x[:self.space.Nu]
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
	p = 1 
	N = 8
	xe = np.linspace(0,1,Ne+1)
	leg = LegendreBasis(p) 
	space = L2Space(xe, leg) 

	eps = 1e-4
	sigma_t = lambda x: 1/eps 
	sigma_s = lambda x: 1/eps - eps 
	Q = lambda x, mu: eps 
	psi_in = lambda x, mu: 0 

	sweep = DirectSweeper(space, N, sigma_t, sigma_s, Q, psi_in)
	p1sa = P1SA(sweep) 
	psi = TVector(space, N) 
	phi = p1sa.SourceIteration(psi)

	plt.plot(space.x, phi.data, '-o')
	plt.show()