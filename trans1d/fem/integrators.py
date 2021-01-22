#!/usr/bin/env python3

import numpy as np

from .quadrature import quadrature 
from .fespace import *
import scipy.sparse as sp 
import scipy.sparse.linalg as spla 
from . import linalg 

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
		linalg.AddOuter(coef*w[n]*el.Jacobian(ip[n]), s, s, elmat) 

	return elmat 

def InverseMassIntegrator(el, c, qorder):
	M = MassIntegrator(el, c, qorder)
	return np.linalg.inv(M)

def MassIntegratorLumped(el, c, qorder):
	ip, w = quadrature.GetLumped(el)
	elmat = np.zeros((el.Nn, el.Nn))

	for n in range(len(w)):
		s = el.CalcShape(ip[n]) 
		X = el.Transform(ip[n])
		coef = c(X)
		linalg.AddOuter(coef*w[n]*el.Jacobian(ip[n]), s, s, elmat)

	return elmat 

def MassIntegratorRowSum(el, c, qorder):
	M = MassIntegrator(el, c, qorder)
	for i in range(M.shape[0]):
		tmp = 0 
		for j in range(M.shape[1]):
			tmp += M[i,j] 
			M[i,j] = 0 

		M[i,i] = tmp 

	return M 

def MixMassIntegrator(el1, el2, c, qorder):
	ip, w = quadrature.Get(qorder)
	elmat = np.zeros((el1.Nn, el2.Nn))

	for n in range(len(w)):
		s1 = el1.CalcShape(ip[n]) 
		s2 = el2.CalcShape(ip[n]) 
		X = el1.Transform(ip[n]) 
		coef = c(X)
		linalg.AddOuter(coef*w[n]*el1.Jacobian(ip[n]), s1, s2, elmat)

	return elmat 

def WeakPoissonIntegrator(el, c, qorder):
	ip, w = quadrature.Get(qorder)
	elmat = np.zeros((el.Nn, el.Nn))

	for n in range(len(w)):
		g = el.CalcPhysGradShape(ip[n]) 
		X = el.Transform(ip[n]) 
		coef = c(X) 
		linalg.AddOuter(coef*w[n]*el.Jacobian(ip[n]), g, g, elmat)

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

		jac = el.Jacobian(ip[n])
		linalg.AddOuter(dE/sig_eval*w[n]*jac, g, s, elmat)
		linalg.AddOuter(E/sig_eval*w[n]*jac, g, g, elmat)

	return elmat 

def ConvectionIntegrator(el, c, qorder):
	ip, w = quadrature.Get(qorder)
	elmat = np.zeros((el.Nn, el.Nn))
	for n in range(len(w)):
		s = el.CalcShape(ip[n])
		g = el.CalcPhysGradShape(ip[n])
		linalg.AddOuter(c*w[n]*el.Jacobian(ip[n]), s, g, elmat) 

	return elmat 

def WeakConvectionIntegrator(el, c, qorder):
	ip, w = quadrature.Get(qorder)
	elmat = np.zeros((el.Nn, el.Nn))

	for n in range(len(w)):
		g = el.CalcPhysGradShape(ip[n]) 
		s = el.CalcShape(ip[n]) 
		linalg.AddOuter(-c*w[n]*el.Jacobian(ip[n]), g, s, elmat)

	return elmat 

def UpwindIntegrator(face_t, c):
	xi1 = face_t.IPTrans(0)
	xi2 = face_t.IPTrans(1) 
	s1 = face_t.el1.CalcShape(xi1)
	s2 = face_t.el2.CalcShape(xi2) 

	jump = np.concatenate((s1, -s2))
	avg = .5*np.concatenate((s1, s2))

	elmat = linalg.Outer(c*face_t.nor, jump, avg) + linalg.Outer(.5*abs(c), jump, jump) 
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
		linalg.AddOuter(el1.Jacobian(ip[n])*c*w[n], s, g, elmat)

	return elmat 

def WeakMixDivIntegrator(el1, el2, c, qorder):
	ip, w = quadrature.Get(qorder)
	elmat = np.zeros((el1.Nn, el2.Nn))

	for n in range(len(w)):
		g = el1.CalcPhysGradShape(ip[n]) 
		s = el2.CalcShape(ip[n]) 
		linalg.AddOuter(-w[n]*el2.Jacobian(ip[n])*c, g, s, elmat)

	return elmat 

def MixJumpAvgIntegrator(face1, face2, c):
	xi1 = face1.IPTrans(0)
	xi2 = face1.IPTrans(1)
	s11 = face1.el1.CalcShape(xi1)
	s12 = face1.el2.CalcShape(xi2)
	jump = np.concatenate((s11, -s12))

	s21 = face2.el1.CalcShape(xi1)
	s22 = face2.el2.CalcShape(xi2)
	avg = (1 if face1.boundary else .5)*np.concatenate((s21, s22))

	return linalg.Outer(c*face1.nor, jump, avg)

def JumpJumpIntegrator(face, c):
	xi1 = face.IPTrans(0)
	xi2 = face.IPTrans(1)
	s1 = face.el1.CalcShape(xi1)
	s2 = face.el2.CalcShape(xi2)
	jump = np.concatenate((s1, -s2))
	return linalg.Outer(c, jump, jump)

def BR2Integrator(face_t, c):
	xi1 = face_t.IPTrans(0)
	xi2 = face_t.IPTrans(1)

	s1 = face_t.el1.CalcShape(xi1)
	s2 = face_t.el2.CalcShape(xi2)
	j = np.concatenate((s1, -s2))

	gs1 = face_t.el1.CalcPhysGradShape(xi1)
	gs2 = face_t.el2.CalcPhysGradShape(xi2)
	ga = np.concatenate((gs1, gs2)) * (1 if face_t.boundary else .5) * face_t.nor 
	jga = np.outer(j,ga)

	a = np.concatenate((s1, s2)) * (1 if face_t.boundary else .5) 
	B = -np.outer(a,j)

	m = MassIntegrator(face_t.el1, lambda x: 1, 2*face_t.el1.basis.p+1)
	minv = np.linalg.inv(m) 
	Minv = np.block([[minv,0*minv], [0*minv,minv]])
	br = c*np.linalg.multi_dot([B.T, Minv, B])
	return br - jga - jga.T 

def MixWeakEddDivIntegrator(el1, el2, qdf, qorder):
	ip, w = quadrature.Get(qorder)
	elmat = np.zeros((el1.Nn, el2.Nn))

	for n in range(len(w)):
		g = el1.CalcPhysGradShape(ip[n]) 
		s = el2.CalcShape(ip[n]) 
		E = qdf.EvalFactor(el1, ip[n]) 
		linalg.AddOuter(-E*w[n]*el1.Jacobian(ip[n]), g, s, elmat)

	return elmat 

def MLBdrIntegrator(face_t, qdf):
	xi1 = face_t.IPTrans(0)
	s = face_t.el1.CalcShape(xi1)
	E = qdf.EvalFactor(face_t.el1, xi1) 
	G = qdf.EvalG(face_t)
	elmat = linalg.Outer(E/G, s, s)
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
	return linalg.Outer(c*face1.nor, avg, jump)

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
	return linalg.Outer(E*face1.nor, jump, avg)

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
	return linalg.Outer(face1.nor, jump, avg)

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
		coo[dof1, dof1] = elmat 

	return coo.Get()

def FaceAssembleAll(space, integrator, c):
	return FaceAssemble(space, integrator, c) + BdrFaceAssemble(space, integrator, c)

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

def FaceAssembleRHS(space, integrator, c):
	b = np.zeros(space.Nu)
	for face_t in space.iface: 
		elvec = integrator(face_t, c)
		dofs = np.concatenate((space.dofs[face_t.el1.ElNo], space.dofs[face_t.el2.ElNo]))
		b[dofs] += elvec 

	return b

def FaceAssembleAllRHS(space, integrator, c):
	return FaceAssembleRHS(space, integrator, c) + BdrFaceAssembleRHS(space, integrator, c)