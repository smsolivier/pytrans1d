#!/usr/bin/env python3

import numpy as np
import warnings

from trans1d.fem.fespace import * 
from trans1d.fem.quadrature import quadrature 
from .. import utils 

class QDFactors:
	def __init__(self, tspace, quad, psi_in=None):
		self.space = tspace 
		self.N = quad.N
		self.quad = quad 
		self.mu = quad.mu
		self.w = quad.w 
		self.psi_in = psi_in 
		if (self.psi_in==None):
			self.psi_in = lambda x, mu: 0 

		self.P = GridFunction(self.space) 
		self.phi = GridFunction(self.space) 

	def Compute(self, psi):
		self.psi = psi 
		self.P.data *= 0 
		self.phi.data *= 0 

		neg = False
		for a in range(self.N):
			mu = self.mu[a] 
			self.P.data += self.mu[a]**2 * self.w[a] * psi.GetAngle(a).data 
			self.phi.data += self.w[a] * psi.GetAngle(a).data 
			if (psi.GetAngle(a).data<0).any():
				neg = True

		if (neg):
			warnings.warn('negative psi detected', utils.NegativityWarning, stacklevel=2)

	def EvalFactor(self, el, xi):
		return self.P.Interpolate(el.ElNo, xi) / self.phi.Interpolate(el.ElNo, xi) 

	def EvalFactorDeriv(self, el, xi):
		n = el.ElNo
		P = self.P.Interpolate(n, xi)
		dP = self.P.InterpolateGrad(n, xi) 
		phi = self.phi.Interpolate(n, xi)
		dphi = self.phi.InterpolateGrad(n, xi) 

		return (dP*phi - P*dphi)/phi**2 

	def EvalFactorDerivBdr(self, face):
		e = face.el1.ElNo
		ep = face.el2.ElNo 
		P = 0 
		dP = 0 
		phi = 0 
		dphi = 0 
		for a in range(self.N):
			mu = self.mu[a] 
			w = self.w[a] 
			if (mu*face.nor>0 or face.boundary):
				elno = e 
				xi = face.IPTrans(0)
			else:
				elno = ep 
				xi = face.IPTrans(1)
			psi = self.psi.GetAngle(a)
			psi_at_ip = psi.Interpolate(elno, xi)
			dpsi_at_ip = psi.InterpolateGrad(elno, xi)
			P += mu**2 * w * psi_at_ip
			dP += mu**2 * w * dpsi_at_ip 
			phi += w * psi_at_ip 
			dphi += w * dpsi_at_ip 

		return (dP*phi - P*dphi)/phi**2 

	def EvalFactorBdr(self, face):
		P = 0 
		phi = 0 
		for a in range(self.N):
			mu = self.mu[a] 
			if (mu*face.nor>0 or face.boundary):
				el = face.el1.ElNo
				xi = face.IPTrans(0)
			else:
				el = face.el2.ElNo
				xi = face.IPTrans(1)
			psi_at_ip = self.psi.GetAngle(a).Interpolate(el, xi)
			P += mu**2 * self.w[a] * psi_at_ip
			phi += self.w[a] * psi_at_ip

		return P/phi 

	def EvalG(self, face_t):
		t = 0 
		b = 0 
		for a in range(self.N):
			mu = self.mu[a] 
			if (face_t.nor*mu>0 or face_t.boundary):
				el = face_t.el1
				xi = face_t.IPTrans(0)
			else:
				el = face_t.el2 
				xi = face_t.IPTrans(1)
			psi_at_ip = self.psi.GetAngle(a).Interpolate(el.ElNo, xi) 
			t += abs(mu)*self.w[a] * psi_at_ip 
			b += psi_at_ip * self.w[a] 

		return t/b

	def EvalGInt(self, el, xi):
		t = 0 
		phi = self.phi.Interpolate(el.ElNo, xi)
		for a in range(self.N):
			mu = self.mu[a]
			w = self.w[a] 
			psi_at_ip = self.psi.GetAngle(a).Interpolate(el.ElNo, xi)
			t += abs(mu) * w * psi_at_ip 

		return t/phi 

	def EvalCp(self, face):
		if (face.nor>0 or face.boundary):
			el = face.el1 
			xi = face.IPTrans(0)
		else:
			el = face.el2 
			xi = face.IPTrans(1)

		t = 0 
		b = 0 
		for a in range(self.N):
			mu = self.mu[a] 
			w = self.w[a] 
			if (face.nor*mu>0):
				psi_at_ip = self.psi.GetAngle(a).Interpolate(el.ElNo, xi)
				t += mu*face.nor * w * psi_at_ip
				b += w * psi_at_ip

		return abs(t/b) 

	def EvalCm(self, face):
		if (face.nor<0 or face.boundary):
			el = face.el1 
			xi = face.IPTrans(0)
		else:
			el = face.el2 
			xi = face.IPTrans(1)

		t = 0 
		b = 0 
		for a in range(self.N):
			mu = self.mu[a] 
			w = self.w[a] 
			if (face.nor*mu<0):
				psi_at_ip = self.psi.GetAngle(a).Interpolate(el.ElNo, xi)
				t += mu*face.nor * w * psi_at_ip
				b += w * psi_at_ip

		return abs(t/b) 

	def EvalAlpha(self, el, xi):
		alpha = 0 
		for a in range(self.N):
			psi_at_ip = self.psi.GetAngle(a).Interpolate(el.ElNo, xi)
			alpha += self.w[a] * self.mu[a] * abs(self.mu[a]) * psi_at_ip

		return alpha / 2

	def EvalJinBdr(self, face_t):
		xi1 = face_t.IPTrans(0)
		x = face_t.el1.Transform(xi1)
		Jin = 0 
		for a in range(self.N):
			if (self.mu[a]*face_t.nor<0):
				Jin += face_t.nor*self.mu[a] * self.w[a] * self.psi_in(x,self.mu[a])

		return Jin

	def EvalJoutBdr(self, face):
		xi1 = face.IPTrans(0)
		Jout = 0 
		for a in range(self.N):
			if (self.mu[a]*face.nor<0):
				psi_at_ip = self.psi.GetAngle(a).Interpolate(face.el1.ElNo, xi1)
				Jout += face.nor * self.mu[a] * self.w[a] * psi_at_ip

		return Jout 

	def EvalPhiInBdr(self, face_t):
		xi1 = face_t.IPTrans(0)
		x = face_t.el1.Transform(xi1)
		phi_in = 0 
		for a in range(self.N):
			if (self.mu[a]*face_t.nor<0):
				phi_in += self.w[a] * self.psi_in(x,self.mu[a])

		return phi_in

	def EvalPinBdr(self, face):
		xi1 = face.IPTrans(0)
		x = face.el1.Transform(xi1)
		Pin = 0 
		for a in range(self.N):
			if (self.mu[a]*face.nor<0):
				Pin += face.nor * self.mu[a]**2 * self.psi_in(x, self.mu[a])

		return Pin 

	def EvalPoutBdr(self, face):
		xi1 = face.IPTrans(0)
		Pout = 0 
		for a in range(self.N):
			if (self.mu[a]*face.nor>0):
				psi_at_ip = self.psi.GetAngle(a).Interpolate(face.el1.ElNo, xi1)
				Pout += face.nor * self.mu[a]**2 * psi_at_ip

		return Pout

