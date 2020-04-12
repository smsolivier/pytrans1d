#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from fespace import * 
from quadrature import quadrature 

class QDFactors:
	def __init__(self, tspace, N, psi_in=None):
		self.space = tspace 
		self.N = N 
		self.mu, self.w = quadrature.Get(self.N) 
		self.psi_in = psi_in 
		if (self.psi_in==None):
			self.psi_in = lambda x, mu: 0 

		self.P = GridFunction(self.space) 
		self.phi = GridFunction(self.space) 

	def Compute(self, psi):
		self.psi = psi 
		self.P.data *= 0 
		self.phi.data *= 0 

		for a in range(self.N):
			mu = self.mu[a] 
			self.P.data += self.mu[a]**2 * self.w[a] * psi.GetAngle(a).data 
			self.phi.data += self.w[a] * psi.GetAngle(a).data 

	def EvalFactor(self, el, xi):
		return self.P.Interpolate(el.ElNo, xi) / self.phi.Interpolate(el.ElNo, xi) 

	def EvalFactorDeriv(self, el, xi):
		n = el.ElNo
		P = self.P.Interpolate(n, xi)
		dP = self.P.InterpolateGrad(n, xi) 
		phi = self.phi.Interpolate(n, xi)
		dphi = self.phi.InterpolateGrad(n, xi) 

		return (dP*phi - P*dphi)/phi**2 

	def EvalFactorBdr(self, face):
		P = 0 
		phi = 0 
		for a in range(self.N):
			mu = self.mu[a] 
			if (mu*face.nor>0):
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
		xi = face_t.IPTrans(0)
		t = 0 
		b = 0 
		for a in range(self.N):
			mu = self.mu[a] 
			psi_at_ip = self.psi.GetAngle(a).Interpolate(face_t.el1.ElNo, xi) 
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
		if (face.nor>0):
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
		if (face.nor<0):
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

	def EvalJinBdr(self, face_t):
		xi1 = face_t.IPTrans(0)
		x = face_t.el1.Transform(xi1)
		Jin = 0 
		for a in range(self.N):
			if (self.mu[a]*face_t.nor<0):
				Jin += face_t.nor*self.mu[a] * self.w[a] * self.psi_in(x,self.mu[a])

		return Jin

	def EvalPhiInBdr(self, face_t):
		xi1 = face_t.IPTrans(0)
		x = face_t.el1.Transform(xi1)
		phi_in = 0 
		for a in range(self.N):
			if (self.mu[a]*face_t.nor<0):
				phi_in += self.w[a] * self.psi_in(x,self.mu[a])

		return phi_in
