#!/usr/bin/env python3

import numpy as np
import warnings

from .qdf import * 
from .sn import * 
from trans1d.fem.linsolver import * 
from .. import utils 

def VEFSIPIntegrator(face_t, c):
	sigma_t = c[0]
	qdf = c[1]
	kappa = c[2] 
	xi1 = face_t.IPTrans(0)
	xi2 = face_t.IPTrans(1)
	X1 = face_t.el1.Transform(xi1)
	X2 = face_t.el2.Transform(xi2)
	sigma1 = sigma_t(X1)
	sigma2 = sigma_t(X2) 
	s1 = face_t.el1.CalcShape(xi1)
	s2 = face_t.el2.CalcShape(xi2)
	jump = np.concatenate((s1, -s2))

	gs1 = face_t.el1.CalcPhysGradShape(xi1)*qdf.EvalFactor(face_t.el1, xi1)
	gs2 = face_t.el2.CalcPhysGradShape(xi2)*qdf.EvalFactor(face_t.el2, xi2)
	dE1 = s1*qdf.EvalFactorDeriv(face_t.el1, xi1)
	dE2 = s2*qdf.EvalFactorDeriv(face_t.el2, xi2)
	avg = np.concatenate(((gs1+dE1)/sigma1, (gs2+dE2)/sigma2)) * (1 if face_t.boundary else .5) * face_t.nor

	elmat = np.outer(jump, avg)
	h = face_t.el1.h
	pen = np.outer(jump, jump)*kappa/h
	return -elmat - elmat.transpose() + pen  

def SourceSIP(face_t, c):
	xi1 = face_t.IPTrans(0)
	xi2 = face_t.IPTrans(1)
	X = face_t.el1.Transform(xi1)
	s1 = face_t.el1.CalcShape(xi1)
	s2 = face_t.el2.CalcShape(xi2)
	if (face_t.boundary):
		jump = s1 
	else:
		jump = np.concatenate((s1, -s2))
	return -jump * c(X) * face_t.nor 

def SIPInflow(face_t, qdf):
	assert(face_t.boundary)
	xi1 = face_t.IPTrans(0)
	s = face_t.el1.CalcShape(xi1)
	Jin = qdf.EvalJinBdr(face_t)*2 
	return -Jin*s

def SIPBC(face_t, qdf):
	assert(face_t.boundary)
	xi1 = face_t.IPTrans(0)
	s = face_t.el1.CalcShape(xi1)
	Eb = qdf.EvalG(face_t)
	return linalg.Outer(Eb, s,s)

class SIPVEF(Sn):
	def __init__(self, fes, sweeper):
		Sn.__init__(self, sweeper)
		self.fes = fes 
		p = self.fes.basis.p 
		self.sigma_a = lambda x: self.sweeper.sigma_t(x) - self.sweeper.sigma_s(x) 
		self.Ma = Assemble(self.fes, MassIntegrator, self.sigma_a, 2*p+1)

		Q0 = np.zeros(fes.Nu)
		Q1 = np.zeros(fes.Nu)
		for a in range(self.N):
			mu = self.mu[a]
			Q0 += AssembleRHS(self.fes, DomainIntegrator, lambda x: sweeper.Q(x,mu), 2*p+1)*self.w[a]
			Q1 += AssembleRHS(self.fes, GradDomainIntegrator, 
				lambda x: sweeper.Q(x,mu)/self.sweeper.sigma_t(x), 2*p+1)*mu*self.w[a]
			Q1 += FaceAssembleRHS(self.fes, SourceSIP, 
				lambda x: sweeper.Q(x,mu)/sweeper.sigma_t(x))*mu*self.w[a]

		self.qdf = QDFactors(self.space, self.sweeper.quad, self.sweeper.psi_in) 
		self.Q = Q0 + Q1 + BdrFaceAssembleRHS(self.fes, SIPInflow, self.qdf)

	def SourceIteration(self, psi, niter=50, tol=1e-6):
		# phi = self.ComputeScalarFlux(psi)
		phi = GridFunction(self.fes)
		phi_old = GridFunction(self.fes)
		for n in range(niter):
			start = time.time() 
			phi_old.data = phi.data.copy() 
			self.sweeper.Sweep(psi, phi) 
			phi = self.Mult(psi)
			norm = phi.L2Diff(phi_old, 2*self.p+1)
			if (self.LOUD):
				el = time.time() - start 
				print('i={:3}, norm={:.3e}, {:.2f} s/iter'.format(
					n+1, norm, el))

			if (norm < tol):
				break 

		if (norm > tol):
			warnings.warn('source iteration not converged. final tol={:.3e}'.format(norm), 
				utils.ToleranceWarning, stacklevel=2)

		return phi 

	def Mult(self, psi):
		self.qdf.Compute(psi)
		p = self.fes.basis.p 
		K = Assemble(self.fes, VEFPoissonIntegrator, [self.qdf, self.sweeper.sigma_t], 2*p+1)
		F = FaceAssemble(self.fes, VEFSIPIntegrator, [self.sweeper.sigma_t, self.qdf, (p+1)**2]) \
			+ BdrFaceAssemble(self.fes, SIPBC, self.qdf)

		M = K + self.Ma + F 
		phi = GridFunction(self.fes)
		phi.data = spla.spsolve(M, self.Q)
		return phi 