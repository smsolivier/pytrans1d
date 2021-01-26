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

	Eu = qdf.EvalFactorBdr(face_t)
	E1 = qdf.EvalFactor(face_t.el1, xi1)
	E2 = qdf.EvalFactor(face_t.el2, xi2)
	Es = E1 if face_t.el1.ElNo < face_t.el2.ElNo else E2 
	dEu = qdf.EvalFactorDerivBdr(face_t)
	dE1 = qdf.EvalFactorDeriv(face_t.el1, xi1)
	dE2 = qdf.EvalFactorDeriv(face_t.el2, xi2)
	dEs = dE1 if face_t.el1.ElNo < face_t.el2.ElNo else dE2 
	gs1 = face_t.el1.CalcPhysGradShape(xi1)*Eu
	gs2 = face_t.el2.CalcPhysGradShape(xi2)*Eu
	sd1 = s1*dEu
	sd2 = s2*dEu
	avg = np.concatenate(((gs1+sd1)/sigma1, (gs2+sd2)/sigma2)) * (1 if face_t.boundary else .5) * face_t.nor

	elmat = np.outer(jump, avg)
	h = face_t.el1.h
	pen = np.outer(jump, jump)*kappa/h
	return -elmat - elmat.T + pen  

def VEFSIPIntegratorNew(face_t, c):
	sigma_t = c[0]
	qdf = c[1]
	kappa = c[2] 
	xi1 = face_t.IPTrans(0)
	xi2 = face_t.IPTrans(1)
	bfac = 1 if face_t.boundary else .5 
	X1 = face_t.el1.Transform(xi1)
	X2 = face_t.el2.Transform(xi2)
	sigma1 = sigma_t(X1)
	sigma2 = sigma_t(X2) 
	s1 = face_t.el1.CalcShape(xi1)
	s2 = face_t.el2.CalcShape(xi2)
	gs1 = face_t.el1.CalcPhysGradShape(xi1)
	gs2 = face_t.el2.CalcPhysGradShape(xi2)
	sj = np.concatenate((s1, -s2))
	gsa = np.concatenate((gs1/sigma1, gs2/sigma2)) * bfac 

	E1 = qdf.EvalFactor(face_t.el1, xi1)
	E2 = qdf.EvalFactor(face_t.el2, xi2)
	dE1 = qdf.EvalFactorDeriv(face_t.el1, xi1)
	dE2 = qdf.EvalFactorDeriv(face_t.el2, xi2)
	Eavg = np.concatenate(( (E1*gs1 + dE1*s1)/sigma1, (E2*gs2 + dE2*s2)/sigma2 )) * bfac * face_t.nor 
	Ejump = np.concatenate((E1*s1, -E2*s2)) * face_t.nor 

	return -np.outer(sj, Eavg) - np.outer(gsa, Ejump) + kappa*np.outer(sj, sj)/face_t.el1.h

def VEFBR2Integrator(face_t, c):
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

	Eu = qdf.EvalFactorBdr(face_t)
	E1 = qdf.EvalFactor(face_t.el1, xi1)
	E2 = qdf.EvalFactor(face_t.el2, xi2)
	Es = E1 if face_t.el1.ElNo < face_t.el2.ElNo else E2 
	dE1 = qdf.EvalFactorDeriv(face_t.el1, xi1)
	dE2 = qdf.EvalFactorDeriv(face_t.el2, xi2)
	dEs = dE1 if face_t.el1.ElNo < face_t.el2.ElNo else dE2 
	gs1 = face_t.el1.CalcPhysGradShape(xi1)*Es
	gs2 = face_t.el2.CalcPhysGradShape(xi2)*Es
	sd1 = s1*dEs
	sd2 = s2*dEs
	avg = np.concatenate(((gs1+sd1)/sigma1, (gs2+sd2)/sigma2)) * (1 if face_t.boundary else .5) * face_t.nor
	jga = np.outer(jump, avg)

	a = np.concatenate((s1, s2)) * (1 if face_t.boundary else .5) 
	B = -np.outer(a,jump)

	m = MassIntegrator(face_t.el1, lambda x: 1, 2*face_t.el1.basis.p+1)
	minv = np.linalg.inv(m) 
	Minv = np.block([[minv,0*minv], [0*minv,minv]])
	br = kappa*np.linalg.multi_dot([B.T, Minv, B])

	return br - jga - jga.T

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

class SIPVEF:
	def __init__(self, fes, qdf, sigma_t, sigma_s, source):
		self.fes = fes 
		self.qdf = qdf 
		self.sigma_t = sigma_t 
		self.sigma_s = sigma_s 
		self.sigma_a = lambda x: sigma_t(x) - sigma_s(x) 
		p = self.fes.basis.p 
		self.Ma = Assemble(self.fes, MassIntegrator, self.sigma_a, 2*p+1)

		Q0 = np.zeros(fes.Nu)
		Q1 = np.zeros(fes.Nu)
		quad = qdf.quad
		for a in range(quad.N):
			mu = quad.mu[a]
			Q0 += AssembleRHS(self.fes, DomainIntegrator, lambda x: source(x,mu), 2*p+1)*quad.w[a]
			Q1 += AssembleRHS(self.fes, GradDomainIntegrator, 
				lambda x: source(x,mu)/sigma_t(x), 2*p+1)*mu*quad.w[a]
			Q1 += FaceAssembleRHS(self.fes, SourceSIP, 
				lambda x: source(x,mu)/sigma_t(x))*mu*quad.w[a]

		self.Q = Q0 + Q1 + BdrFaceAssembleRHS(self.fes, SIPInflow, self.qdf)

	def Mult(self, psi):
		self.qdf.Compute(psi)
		p = self.fes.basis.p 
		K = Assemble(self.fes, VEFPoissonIntegrator, [self.qdf, self.sigma_t], 2*p+1)
		F = FaceAssemble(self.fes, VEFSIPIntegratorNew, [self.sigma_t, self.qdf, (p+1)**2]) \
			+ BdrFaceAssemble(self.fes, SIPBC, self.qdf)

		M = K + self.Ma + F 
		phi = GridFunction(self.fes)
		phi.data = spla.spsolve(M, self.Q)
		return phi 

class BR2VEF(SIPVEF):
	def __init__(self, fes, qdf, sigma_t, sigma_s, source):
		SIPVEF.__init__(self, fes, qdf, sigma_t, sigma_s, source)

	def Mult(self, psi):
		self.qdf.Compute(psi)
		p = self.fes.basis.p 
		K = Assemble(self.fes, VEFPoissonIntegrator, [self.qdf, self.sigma_t], 2*p+1)
		F = FaceAssemble(self.fes, VEFBR2Integrator, [self.sigma_t, self.qdf, 5]) \
			+ BdrFaceAssemble(self.fes, SIPBC, self.qdf)

		A = K + self.Ma + F 
		phi = GridFunction(self.fes)
		phi.data = spla.spsolve(A, self.Q)
		return phi