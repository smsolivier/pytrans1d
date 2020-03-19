#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from quadrature import quadrature 
from fespace import *
import scipy.sparse as sp 
import scipy.sparse.linalg as spla 

class COOBuilder:
	def __init__(self, m, n=None):
		self.m = m 
		self.n = n
		if (n==None):
			self.n = self.m
		self.row = [] 
		self.col = [] 
		self.data = []

	def __setitem__(self, key, item):
		for i in range(len(key[0])):
			for j in range(len(key[1])):
				if (abs(item[i,j])>1e-15):
					self.row.append(key[0][i])
					self.col.append(key[1][j])
					self.data.append(item[i,j])

	def Get(self):
		return sp.coo_matrix((self.data, (self.row, self.col)), (self.m, self.n)).tocsc()

def MassIntegrator(el, c, qorder):
	ip, w = quadrature.Get(qorder)
	elmat = np.zeros((el.Nn, el.Nn))

	for n in range(len(w)):
		s = el.CalcShape(ip[n]) 
		X = el.Transform(ip[n]) 
		coef = c(X) 
		elmat += np.outer(s, s) * coef * w[n] * el.Jacobian(ip[n]) 

	return elmat 

def MassIntegratorLumped(el, c, qorder):
	ip, w = quadrature.GetLumped(el)
	elmat = np.zeros((el.Nn, el.Nn))

	for n in range(len(w)):
		s = el.CalcShape(ip[n]) 
		X = el.Transform(ip[n])
		coef = c(X)
		elmat += np.outer(s,s) * coef * w[n] * el.Jacobian(ip[n])

	return elmat 

def MixMassIntegrator(el1, el2, c, qorder):
	ip, w = quadrature.Get(qorder)
	elmat = np.zeros((el1.Nn, el2.Nn))

	for n in range(len(w)):
		s1 = el1.CalcShape(ip[n]) 
		s2 = el2.CalcShape(ip[n]) 
		X = el1.Transform(ip[n]) 
		coef = c(X)
		elmat += np.outer(s1, s2) * coef * w[n] * el1.Jacobian(ip[n]) 

	return elmat 

def WeakPoissonIntegrator(el, c, qorder):
	ip, w = quadrature.Get(qorder)
	elmat = np.zeros((el.Nn, el.Nn))

	for n in range(len(w)):
		g = el.CalcPhysGradShape(ip[n]) 
		X = el.Transform(ip[n]) 
		coef = c(X) 
		elmat += np.outer(g, g) * coef * w[n] * el.Jacobian(ip[n]) 

	return elmat 

def VEFPoissonIntegrator(el, c, qorder):
	qdf = c[0] 
	sigma_t = c[1]
	ip, w = quadrature.Get(qorder)
	elmat = np.zeros((el.Nn, el.Nn))

	for n in range(len(w)):
		g = el.CalcPhysGradShape(ip[n]) 
		s = el.CalcShape(ip[n])
		E = qdf.EvalFactor(el, ip[n]) 
		dE = qdf.EvalFactorDeriv(el, ip[n]) 
		X = el.Transform(ip[n]) 
		sig_eval = sigma_t(X) 

		m1 = np.outer(g, s) / sig_eval * dE * w[n] * el.Jacobian(ip[n]) 
		m2 = np.outer(g, g) / sig_eval * E * w[n] * el.Jacobian(ip[n]) 

		elmat += m1 + m2 

	return elmat 

def WeakConvectionIntegrator(el, c, qorder):
	ip, w = quadrature.Get(qorder)
	elmat = np.zeros((el.Nn, el.Nn))

	for n in range(len(w)):
		g = el.CalcPhysGradShape(ip[n]) 
		s = el.CalcShape(ip[n]) 
		elmat -= np.outer(g, s) * c * w[n] * el.Jacobian(ip[n]) 

	return elmat 

def UpwindIntegrator(face_t, c):
	xi1 = face_t.IPTrans(0)
	xi2 = face_t.IPTrans(1) 
	s1 = face_t.el1.CalcShape(xi1)
	s2 = face_t.el2.CalcShape(xi2) 

	jump = np.concatenate((s1, -s2))
	avg = .5*np.concatenate((s1, s2))

	elmat = c*face_t.nor*np.outer(jump, avg) + .5*abs(c)*np.outer(jump, jump) 
	return elmat 

def InflowIntegrator(face_t, c):
	mu = c[0]
	psi_in = c[1] 

	xi1 = face_t.IPTrans(0) 
	if (mu*face_t.nor<0):
		s = face_t.el1.CalcShape(xi1) 
		X = face_t.el1.Transform(xi1) 
		return s*psi_in(X)*abs(mu) 
	else:
		return np.zeros(face_t.el1.Nn)

def MixDivIntegrator(el1, el2, c, qorder):
	ip, w = quadrature.Get(qorder)
	elmat = np.zeros((el1.Nn, el2.Nn))

	for n in range(len(w)):
		s = el1.CalcShape(ip[n]) 
		g = el2.CalcPhysGradShape(ip[n]) 
		elmat += np.outer(s, g) * w[n] * el1.Jacobian(ip[n]) 

	return elmat 

def MixWeakEddDivIntegrator(el1, el2, qdf, qorder):
	ip, w = quadrature.Get(qorder)
	elmat = np.zeros((el1.Nn, el2.Nn))

	for n in range(len(w)):
		g = el1.CalcPhysGradShape(ip[n]) 
		s = el2.CalcShape(ip[n]) 
		E = qdf.EvalFactor(el1, ip[n]) 
		elmat -= np.outer(g, s) * E * w[n] * el1.Jacobian(ip[n]) 

	return elmat 

def MLBdrIntegrator(face_t, qdf):
	xi1 = face_t.IPTrans(0)
	s = face_t.el1.CalcShape(xi1)
	E = qdf.EvalFactor(face_t.el1, xi1) 
	G = qdf.EvalG(face_t)
	elmat = np.outer(s, s) * E / G 
	return elmat 

def VEFInflowIntegrator(face_t, qdf):
	xi1 = face_t.IPTrans(0)
	s = face_t.el1.CalcShape(xi1)
	E = qdf.EvalFactor(face_t.el1, xi1)
	G = qdf.EvalG(face_t)
	elvec = 2*s * E / G * qdf.EvalJinBdr(face_t) * face_t.nor
	return elvec 

def ConstraintIntegrator(face1, face2, c):
	xi1 = face1.IPTrans(0)
	xi2 = face2.IPTrans(1) 
	s11 = face1.el1.CalcShape(xi1)
	s12 = face1.el2.CalcShape(xi2) 
	avg = .5*np.concatenate((s11, s12))
	s21 = face2.el1.CalcShape(xi1)
	s22 = face2.el2.CalcShape(xi2) 
	jump = np.concatenate((s21, -s22))
	return np.outer(avg, jump) * c * face1.nor

def UpwEddConstraintIntegrator(face1, face2, qdf):
	xi1 = face1.IPTrans(0)
	xi2 = face2.IPTrans(1) 
	s11 = face1.el1.CalcShape(xi1)
	s12 = face1.el2.CalcShape(xi2) 
	jump = np.concatenate((s11, -s12))
	s21 = face2.el1.CalcShape(xi1)
	s22 = face2.el2.CalcShape(xi2) 
	avg = .5*np.concatenate((s21, s22))
	E = qdf.EvalFactorBdr(face1)
	return np.outer(jump, avg) * E * face1.nor

def EddConstraintIntegrator(face1, face2, qdf):
	xi1 = face1.IPTrans(0)
	xi2 = face2.IPTrans(1) 
	s11 = face1.el1.CalcShape(xi1)
	s12 = face1.el2.CalcShape(xi2) 
	E1 = qdf.EvalFactor(face1.el1, xi1)
	E2 = qdf.EvalFactor(face1.el2, xi2)
	jump = np.concatenate((E1*s11, -E2*s12))
	s21 = face2.el1.CalcShape(xi1)
	s22 = face2.el2.CalcShape(xi2) 
	avg = .5*np.concatenate((s21, s22))
	return np.outer(jump, avg) * face1.nor

def DomainIntegrator(el, c, qorder):
	ip, w = quadrature.Get(qorder)
	elvec = np.zeros(el.Nn)

	for n in range(len(w)):
		s = el.CalcShape(ip[n]) 
		X = el.Transform(ip[n]) 
		elvec += s * c(X) * w[n] * el.Jacobian(ip[n]) 

	return elvec 

def GradDomainIntegrator(el, c, qorder):
	ip, w = quadrature.Get(qorder)
	elvec = np.zeros(el.Nn)

	for n in range(len(w)):
		g = el.CalcPhysGradShape(ip[n]) 
		X = el.Transform(ip[n]) 
		c_eval = c(X) 
		elvec += g * c_eval * w[n] * el.Jacobian(ip[n]) 

	return elvec 

def Assemble(space, integrator, c, qorder):
	coo = COOBuilder(space.Nu, space.Nu)
	for e in range(space.Ne):
		elmat = integrator(space.el[e], c, qorder)
		coo[space.dofs[e], space.dofs[e]] = elmat 

	return coo.Get()

def MixAssemble(space1, space2, integrator, c, qorder):
	coo = COOBuilder(space1.Nu, space2.Nu)
	for e in range(space1.Ne):
		elmat = integrator(space1.el[e], space2.el[e], c, qorder)
		coo[space1.dofs[e], space2.dofs[e]] = elmat 

	return coo.Get()

def AssembleRHS(space, integrator, c, qorder):
	b = np.zeros(space.Nu)
	for e in range(space.Ne):
		el = space.el[e]
		elvec = integrator(el, c, qorder)
		b[space.dofs[e]] += elvec 

	return b

def FaceAssemble(space, integrator, c):
	coo = COOBuilder(space.Nu, space.Nu)
	for face_t in space.iface:
		elmat = integrator(face_t, c) 
		dof1 = space.dofs[face_t.el1.ElNo]
		dof2 = space.dofs[face_t.el2.ElNo] 
		dof = np.concatenate((dof1, dof2)) 
		coo[dof, dof] = elmat 

	return coo.Get()

def BdrFaceAssemble(space, integrator, c):
	coo = COOBuilder(space.Nu, space.Nu)
	for face_t in space.bface:
		elmat = integrator(face_t, c) 
		dof1 = space.dofs[face_t.el1.ElNo]
		dof2 = space.dofs[face_t.el2.ElNo] 
		coo[dof1, dof2] = elmat 

	return coo.Get()

def MixFaceAssemble(space1, space2, integrator, c):
	coo = COOBuilder(space1.Nu, space2.Nu)
	for f in range(len(space1.iface)):
		face1 = space1.iface[f]
		face2 = space2.iface[f]
		elmat = integrator(face1, face2, c)
		dof1 = np.concatenate((space1.dofs[face1.el1.ElNo], space1.dofs[face1.el2.ElNo]))
		dof2 = np.concatenate((space2.dofs[face2.el1.ElNo], space2.dofs[face2.el2.ElNo]))
		coo[dof1, dof2] = elmat 

	return coo.Get()

def MixFaceAssembleBdr(space1, space2, integrator, c):
	coo = COOBuilder(space1.Nu, space2.Nu)
	for f in range(len(space1.bface)):
		face1 = space1.bface[f]
		face2 = space2.bface[f]
		elmat = integrator(face1, face2, c)
		dof1 = space1.dofs[face1.el1.ElNo] 
		dof2 = space2.dofs[face2.el1.ElNo]
		coo[dof1, dof2] = elmat 

	return coo.Get()

def MixFaceAssembleAll(space1, space2, integrator, c):
	return MixFaceAssemble(space1, space2, integrator, c) + MixFaceAssembleBdr(space1, space2, integrator, c)

def BdrFaceAssembleRHS(space, integrator, c):
	b = np.zeros(space.Nu)

	for face_t in space.bface:
		elvec = integrator(face_t, c)
		b[space.dofs[face_t.el1.ElNo]] += elvec 

	return b

def FaceAssembleAll(space, integrator, c):
	return FaceAssemble(space, integrator, c) + BdrFaceAssemble(space, integrator, c)

if __name__=='__main__':
	Ne = 6
	p = 2
	xe = np.linspace(0,1,Ne+1)
	basis = LegendreBasis(p)
	space = L2Space(xe, basis)
	mu = 1
	F = FaceAssembleAll(space, UpwindIntegrator, mu)
	G = Assemble(space, WeakConvectionIntegrator, mu, 2*p-1)
	Mt = Assemble(space, MassIntegrator, lambda x: 1, 2*p-1) 

	A = G + F + Mt

	u = GridFunction(space) 
	I = BdrFaceAssembleRHS(space, InflowIntegrator, [mu, lambda x: 1])
	u.data = spla.spsolve(A, I)
	err = u.L2Error(lambda x: np.exp(-x), 2*p+1)
	print('err = {:.3e}'.format(err))

	plt.plot(space.x, u.data, '-o')
	xex = np.linspace(0,1,100)
	plt.plot(xex, np.exp(-xex), '--')
	plt.show()