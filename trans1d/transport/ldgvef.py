#!/usr/bin/env python3

import numpy as np
import warnings
import pyamg 

from .qdf import * 
from .sn import * 
from trans1d.fem.linsolver import * 
from .. import utils 

def MixVEFJumpAvg(face1, face2, qdf):
	xi1 = face1.IPTrans(0)
	xi2 = face1.IPTrans(1)
	s11 = face1.el1.CalcShape(xi1)
	s12 = face1.el2.CalcShape(xi2)
	jump = np.concatenate((s11, -s12))

	E1 = qdf.EvalFactor(face1.el1, xi1)
	E2 = qdf.EvalFactor(face1.el2, xi2)
	Eu = qdf.EvalFactorBdr(face1)
	Es = E1 if face1.el1.ElNo > face1.el2.ElNo else E2
	Ea = .5*(E1 + E2)
	s21 = face2.el1.CalcShape(xi1)*E1
	s22 = face2.el2.CalcShape(xi2)*E2
	avg = (1 if face1.boundary else .5)*np.concatenate((s21, s22))

	return linalg.Outer(face1.nor, jump, avg)

def MLBdrTerm(face, qdf):
	xi1 = face.IPTrans(0)
	xi2 = face.IPTrans(1)
	s1 = face.el1.CalcShape(xi1)
	s2 = face.el2.CalcShape(xi2)
	jump = np.concatenate((s1, -s2))
	G = qdf.EvalG(face)
	return linalg.Outer(G, jump, jump)

def MLBdrInflowTerm(face, qdf):
	xi1 = face.IPTrans(0)
	s1 = face.el1.CalcShape(xi1)
	return -2*s1 * qdf.EvalJinBdr(face)

def AlphaIntegrator(face, qdf):
	xi1 = face.IPTrans(0)
	xi2 = face.IPTrans(1)
	s1 = face.el1.CalcShape(xi1)
	s2 = face.el2.CalcShape(xi2)
	j = np.concatenate((s1, -s2))
	alpha1 = qdf.EvalAlpha(face.el1, xi1)
	alpha2 = qdf.EvalAlpha(face.el2, xi2)

	return j * (alpha1 - alpha2) 

def ConsistentRHSIntegrator(face, c):
	xi = face.IPTrans(0)
	s = face.el1.CalcShape(xi)
	return s * c(face)

def MixBdrMassIntegrator(face1, face2, c):
	xi = face1.IPTrans(0)
	s1 = face1.el1.CalcShape(xi)
	s2 = face2.el1.CalcShape(xi)
	return np.outer(s1, s2) * c 

class NPI:
	def __init__(self, sweeper, vef, fes, psi):
		self.sweeper = sweeper 
		self.vef = vef 
		self.psi = psi 
		self.fes = fes 

	def Mult(self, phi):
		self.sweeper.Sweep(self.psi, phi)
		new = self.vef.Mult(self.psi)
		return new

	def F(self, phi):
		gf = GridFunction(self.fes)
		gf.data = phi
		self.sweeper.Sweep(self.psi, gf)
		new = self.vef.Mult(self.psi)
		return phi - new.data 

class LDGVEF:
	def __init__(self, sfes, vfes, qdf, sigma_t, sigma_s, source):
		self.sfes = sfes
		self.vfes = vfes 
		self.qdf = qdf 
		self.Q0 = np.zeros(sfes.Nu)
		self.Q1 = np.zeros(vfes.Nu)
		quad = qdf.quad 
		for a in range(quad.N):
			mu = quad.mu[a]
			self.Q0 += AssembleRHS(self.sfes, DomainIntegrator, lambda x: source(x,mu), 2*sfes.basis.p+1)*quad.w[a]
			self.Q1 += AssembleRHS(self.vfes, DomainIntegrator, lambda x: source(x,mu), 2*vfes.basis.p+1)*mu*quad.w[a]

		self.Q0 += BdrFaceAssembleRHS(self.sfes, MLBdrInflowTerm, self.qdf)

		self.Mtinv = Assemble(self.vfes, InverseMassIntegrator, sigma_t, 2*vfes.basis.p+1)
		self.Ma = Assemble(self.sfes, MassIntegrator, lambda x: sigma_t(x)-sigma_s(x), 2*sfes.basis.p+1) \
			+ FaceAssemble(self.sfes, JumpJumpIntegrator, sfes.Ne*(sfes.basis.p+1)**2)
		self.D = MixAssemble(self.sfes, self.vfes, WeakMixDivIntegrator, 1, 2*vfes.basis.p+1) \
			+ MixFaceAssemble(self.sfes, self.vfes, MixJumpAvgIntegrator, 1) 

	def Mult(self, psi):
		self.qdf.Compute(psi) 
		G = MixAssemble(self.vfes, self.sfes, MixWeakEddDivIntegrator, self.qdf, 2*self.vfes.basis.p+1) \
			+ MixFaceAssembleAll(self.vfes, self.sfes, MixVEFJumpAvg, self.qdf)
		B = BdrFaceAssemble(self.sfes, MLBdrTerm, self.qdf)
		Ma = self.Ma + B
		S = Ma - self.D*self.Mtinv*G 
		b = self.Q0 - self.D*self.Mtinv*self.Q1 
		phi = GridFunction(self.sfes)
		phi.data = spla.spsolve(S, b)
		r = np.linalg.norm(S*phi.data - b)
		if (r>1e-10):
			warnings.warn('LDGVEF residual = {:.3e}'.format(r))
		return phi 

class ConsistentLDGVEF:
	def __init__(self, sfes, vfes, qdf, sigma_t, sigma_s, source):
		self.sfes = sfes
		self.vfes = vfes 
		self.qdf = qdf 
		self.Q0 = np.zeros(sfes.Nu)
		self.Q1 = np.zeros(vfes.Nu)
		quad = qdf.quad 
		for a in range(quad.N):
			mu = quad.mu[a]
			self.Q0 += AssembleRHS(self.sfes, DomainIntegrator, lambda x: source(x,mu), 2*sfes.basis.p+1)*quad.w[a]
			self.Q1 += AssembleRHS(self.vfes, DomainIntegrator, lambda x: source(x,mu), 2*vfes.basis.p+1)*mu*quad.w[a]

		self.Mtinv = Assemble(self.vfes, InverseMassIntegrator, sigma_t, 2*vfes.basis.p+1)
		self.Ma = Assemble(self.sfes, MassIntegrator, lambda x: sigma_t(x)-sigma_s(x), 2*sfes.basis.p+1)
		self.D = MixAssemble(self.sfes, self.vfes, WeakMixDivIntegrator, 1, 2*vfes.basis.p+1) \
			+ MixFaceAssemble(self.sfes, self.vfes, MixJumpAvgIntegrator, 1) \
			+ MixFaceAssembleBdr(self.sfes, self.vfes, MixBdrMassIntegrator, .5)

	def Mult(self, psi):
		self.qdf.Compute(psi)
		G = MixAssemble(self.vfes, self.sfes, MixWeakEddDivIntegrator, self.qdf, 2*self.vfes.basis.p+1) \
			+ MixFaceAssemble(self.vfes, self.sfes, MixVEFJumpAvg, self.qdf)
		B = .5*FaceAssembleAll(self.sfes, MLBdrTerm, self.qdf)
		c0 = BdrFaceAssembleRHS(self.sfes, ConsistentRHSIntegrator, self.qdf.EvalJinBdr) 
		c1 = BdrFaceAssembleRHS(self.vfes, ConsistentRHSIntegrator, self.qdf.EvalPoutBdr) \
			+ BdrFaceAssembleRHS(self.vfes, ConsistentRHSIntegrator, self.qdf.EvalPinBdr)
		b = self.Q0-c0 - (self.D*self.Mtinv*(self.Q1 - c1 - FaceAssembleRHS(self.vfes, AlphaIntegrator, self.qdf)))
		S = self.Ma + B - self.D*self.Mtinv*G
		phi = GridFunction(self.sfes)
		phi.data = spla.spsolve(S, b) 
		return phi 