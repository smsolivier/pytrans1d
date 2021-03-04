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
	scale = c[3]
	xi1 = face_t.IPTrans(0)
	xi2 = face_t.IPTrans(1)
	bfac = 1 if face_t.boundary else .5 
	X1 = face_t.el1.Transform(0)
	X2 = face_t.el2.Transform(0) 
	sigma1 = sigma_t(X1)
	sigma2 = sigma_t(X2) 
	sigma_h = (1/sigma1 + 1/sigma2)/2
	w1 = sigma2 / (sigma1 + sigma2)
	w2 = sigma1 / (sigma1 + sigma2)
	s1 = face_t.el1.CalcShape(xi1)
	s2 = face_t.el2.CalcShape(xi2)
	gs1 = face_t.el1.CalcPhysGradShape(xi1)
	gs2 = face_t.el2.CalcPhysGradShape(xi2)
	sj = np.concatenate((s1, -s2))
	gsa = np.concatenate((gs1/sigma1, gs2/sigma2)) * bfac 
	gsa2 = np.concatenate((gs1, gs2)) * bfac 

	Eu = qdf.EvalFactorBdr(face_t)
	dEu = qdf.EvalFactorDerivBdr(face_t)
	E1 = qdf.EvalFactor(face_t.el1, xi1)
	E2 = qdf.EvalFactor(face_t.el2, xi2)
	Eh = E1 + E2 
	dE1 = qdf.EvalFactorDeriv(face_t.el1, xi1)
	dE2 = qdf.EvalFactorDeriv(face_t.el2, xi2)
	Eavg = np.concatenate(( (E1*gs1 + dE1*s1)/sigma1, (E2*gs2 + dE2*s2)/sigma2 )) * bfac * face_t.nor 
	Ejump = np.concatenate((E1*s1, -E2*s2)) * face_t.nor 
	Ejump2 = np.concatenate((E1*s1/sigma1, -E2*s2/sigma2))

	if (scale):
		f = (E1/sigma1/face_t.el1.h + E2/sigma2/face_t.el2.h)/2 
	else:
		f = (E1/sigma1 + E2/sigma2)/2 
	return -np.outer(sj, Eavg) - np.outer(gsa, Ejump) + kappa*np.outer(sj, sj)*f 

def VEFEllIntegrator(face, c):
	qdf = c[0] 
	sigma_t = c[1] 
	xi1 = face.IPTrans(0) 
	xi2 = face.IPTrans(1)
	bfac = 0 if face.boundary else .5
	gs1 = face.el1.CalcPhysGradShape(xi1)
	gs2 = face.el2.CalcPhysGradShape(xi2)
	gj = np.concatenate((gs1, -gs2))
	Ehat = qdf.EvalFactorBdr(face)
	E1 = qdf.EvalFactor(face.el1, xi1)
	E2 = qdf.EvalFactor(face.el2, xi2)
	X1 = face.el1.Transform(0)
	X2 = face.el2.Transform(0)
	sigma1 = sigma_t(X1)
	sigma2 = sigma_t(X2)
	s1 = face.el1.CalcShape(xi1)
	s2 = face.el2.CalcShape(xi2)
	sa = np.concatenate(((Ehat - E1)/sigma1*s1, (Ehat - E2)/sigma2*s2)) * face.nor * bfac 
	return np.outer(gj, sa)

def VEFBR2Integrator(face_t, c):
	sigma_t = c[0]
	qdf = c[1]
	kappa = c[2] 
	xi1 = face_t.IPTrans(0)
	xi2 = face_t.IPTrans(1)
	bfac = 1 if face_t.boundary else .5 
	X1 = face_t.el1.Transform(0)
	X2 = face_t.el2.Transform(0)
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

	a = np.concatenate((s1, s2)) * (1 if face_t.boundary else .5) 
	B = -np.outer(a,sj)

	m = MassIntegrator(face_t.el1, lambda x: 1, 2*face_t.el1.basis.p+1)
	minv = np.linalg.inv(m) 
	Minv = np.block([[minv,0*minv], [0*minv,minv]])
	br = kappa*np.linalg.multi_dot([B.T, Minv, B])

	sigma_h = 1/sigma1 + 1/sigma2
	return br*sigma_h - np.outer(sj, Eavg) - np.outer(gsa, Ejump)

def SourceSIP(face_t, c):
	Q1 = c[0]
	sigma_t = c[1]
	xi1 = face_t.IPTrans(0)
	xi2 = face_t.IPTrans(1)
	X1 = face_t.el1.Transform(xi1)
	X2 = face_t.el2.Transform(xi2)
	sigma1 = sigma_t(face_t.el1.Transform(0))
	sigma2 = sigma_t(face_t.el2.Transform(0))
	s1 = face_t.el1.CalcShape(xi1)
	s2 = face_t.el2.CalcShape(xi2)
	if (face_t.boundary):
		jump = s1 
	else:
		jump = np.concatenate((s1, -s2))
	eps = face_t.nor * 1e-14
	return -jump * .5*(Q1(X1-eps)/sigma1 + Q1(X2+eps)/sigma2) * face_t.nor 

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

def LDGLiftIntegrator(face_t, c):
	sigma_t = c[0]
	qdf = c[1]
	xi1 = face_t.IPTrans(0)
	xi2 = face_t.IPTrans(1)
	X1 = face_t.el1.Transform(0)
	X2 = face_t.el2.Transform(0)
	sigma1 = sigma_t(X1)
	sigma2 = sigma_t(X2) 
	s1 = face_t.el1.CalcShape(xi1)
	s2 = face_t.el2.CalcShape(xi2)
	jump = np.concatenate((s1, -s2))
	avg = np.concatenate((s1, s2)) * (1 if face_t.boundary else .5) * face_t.nor 

	Eu = qdf.EvalFactorBdr(face_t)
	E1 = qdf.EvalFactor(face_t.el1, xi1)
	E2 = qdf.EvalFactor(face_t.el2, xi2)
	# sigmau = (sigma1 + sigma2)/2
	Ejump = np.concatenate((s1*E1, -s2*E1)) 

	m1 = MassIntegrator(face_t.el1, lambda x: 1, 2*face_t.el1.basis.p+1)
	m2 = MassIntegrator(face_t.el2, lambda x: 1, 2*face_t.el2.basis.p+1)
	mit1 = MassIntegrator(face_t.el1, lambda x: 1/sigma_t(x), 2*face_t.el1.basis.p+1)
	mit2 = MassIntegrator(face_t.el2, lambda x: 1/sigma_t(x), 2*face_t.el2.basis.p+1)
	minv1 = np.linalg.inv(m1) 
	minv2 = np.linalg.inv(m2) 
	Minv = np.block([[minv1,np.zeros((m1.shape[0],m2.shape[1]))], [np.zeros((m2.shape[0], m1.shape[1])),minv2]])
	M = np.block([[mit1,np.zeros((m1.shape[0],m2.shape[1]))], [np.zeros((m2.shape[0], m1.shape[1])),mit2]])
	ldg = np.linalg.multi_dot([np.outer(jump, avg), Minv.T, M, Minv, np.outer(avg, Ejump)])
	return ldg 

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
				[lambda x: source(x,mu), sigma_t])*mu*quad.w[a]

		self.Q = Q0 + Q1 + BdrFaceAssembleRHS(self.fes, SIPInflow, self.qdf)

	def Mult(self, psi):
		self.qdf.Compute(psi)
		p = self.fes.basis.p 
		K = Assemble(self.fes, VEFPoissonIntegrator, [self.qdf, self.sigma_t], 2*p+1)
		F = FaceAssemble(self.fes, VEFSIPIntegratorNew, [self.sigma_t, self.qdf, (p+1)**2, True]) \
			+ BdrFaceAssemble(self.fes, SIPBC, self.qdf) 
		# F += FaceAssemble(self.fes, VEFEllIntegrator, [self.qdf, self.sigma_t])

		self.K = K + self.Ma + F 
		phi = GridFunction(self.fes)
		phi.data = spla.spsolve(self.K, self.Q)
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

def MixVEFAvgJump(face1, face2, c):
	qdf = c[0]
	sigma_t = c[1] 
	xi1 = face1.IPTrans(0)
	xi2 = face1.IPTrans(1)
	s11 = face1.el1.CalcShape(xi1)
	s12 = face1.el2.CalcShape(xi2)
	avg = np.concatenate((s11, s12)) * (1 if face1.boundary else .5)

	sigma1 = sigma_t(face1.el1.Transform(0))
	sigma2 = sigma_t(face1.el2.Transform(0))
	E1 = qdf.EvalFactor(face1.el1, xi1)
	E2 = qdf.EvalFactor(face1.el2, xi2)
	Eu = qdf.EvalFactorBdr(face1)
	Es = E1 if face1.el1.ElNo > face1.el2.ElNo else E2
	Ea = .5*(E1 + E2)
	s21 = face2.el1.CalcShape(xi1)
	s22 = face2.el2.CalcShape(xi2)
	jump = np.concatenate((s21*E1/sigma1, -s22*E2/sigma2))

	return linalg.Outer(face1.nor, avg, jump)

class LiftedLDGVEF(SIPVEF):
	def __init__(self, fes, qdf, sigma_t, sigma_s, source, scale=True):
		SIPVEF.__init__(self, fes, qdf, sigma_t, sigma_s, source)
		self.scale = scale 

	def Mult(self, psi):
		self.qdf.Compute(psi)
		p = self.fes.basis.p 
		K = Assemble(self.fes, VEFPoissonIntegrator, [self.qdf, self.sigma_t], 2*p+1)
		F = FaceAssemble(self.fes, VEFSIPIntegratorNew, [self.sigma_t, self.qdf, (p+1)**2, self.scale]) \
			+ BdrFaceAssemble(self.fes, SIPBC, self.qdf)
		B1 = MixFaceAssemble(self.fes, self.fes, MixJumpAvgIntegrator, 1)
		B2 = MixFaceAssemble(self.fes, self.fes, MixVEFAvgJump, [self.qdf, lambda x: 1])
		Minv = Assemble(self.fes, InverseMassIntegrator, self.sigma_t, 2*self.fes.basis.p+1)
		R = B1 * Minv * B2
		Rl = FaceAssemble(self.fes, LDGLiftIntegrator, [self.sigma_t, self.qdf])

		self.K = K + self.Ma + F + Rl
		phi = GridFunction(self.fes)
		phi.data = spla.spsolve(self.K, self.Q)
		return phi 