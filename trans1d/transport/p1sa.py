#!/usr/bin/env python3

import numpy as np
import warnings

from .sn import * 
from .. import utils 

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
			warnings.warn('source iteration not converged. final tol={:.3e}'.format(norm), 
				utils.ToleranceWarning, stacklevel=2)

		return phi 
