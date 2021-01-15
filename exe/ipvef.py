#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans1d import * 
pi = np.pi

def SIPIntegrator(face_t, c):
	xi1 = face_t.IPTrans(0)
	xi2 = face_t.IPTrans(1)
	s1 = face_t.el1.CalcShape(xi1)
	s2 = face_t.el2.CalcShape(xi2)
	jump = np.concatenate((s1, -s2))

	gs1 = face_t.el1.CalcPhysGradShape(xi1)
	gs2 = face_t.el2.CalcPhysGradShape(xi2)
	avg = np.concatenate((gs1, gs2)) * (1 if face_t.boundary else .5) * face_t.nor

	elmat = np.outer(jump, avg)
	pen = np.outer(jump, jump)*c
	return -elmat - elmat.transpose() + pen

# def VEFSIPIntegrator(face_t, c):
# 	sigma_t = c[0]
# 	E = c[1]
# 	dE = c[2]
# 	kappa = c[3] 
# 	xi1 = face_t.IPTrans(0)
# 	xi2 = face_t.IPTrans(1)
# 	X1 = face_t.el1.Transform(xi1)
# 	X2 = face_t.el2.Transform(xi2)
# 	sigma1 = sigma_t(X1)
# 	sigma2 = sigma_t(X2) 
# 	s1 = face_t.el1.CalcShape(xi1)
# 	s2 = face_t.el2.CalcShape(xi2)
# 	jump = np.concatenate((s1, -s2))

# 	gs1 = face_t.el1.CalcPhysGradShape(xi1)*E(X1)
# 	gs2 = face_t.el2.CalcPhysGradShape(xi2)*E(X2)
# 	dE1 = s1*dE(X1)
# 	dE2 = s2*dE(X2)
# 	avg = np.concatenate(((gs1+dE1)/sigma1, (gs2+dE2)/sigma2)) * (1 if face_t.boundary else .5) * face_t.nor

# 	elmat = np.outer(jump, avg)
# 	pen = np.outer(jump, jump)*kappa
# 	return -elmat - elmat.transpose() + pen  

def VEFDiffusionIntegrator(el, c, qorder):
	sigma_t = c[0]
	E = c[1]
	dE = c[2] 
	ip, w = quadrature.Get(qorder)
	elmat = np.zeros((el.Nn, el.Nn))
	for n in range(len(w)):
		g = el.CalcPhysGradShape(ip[n]) 
		s = el.CalcShape(ip[n])
		X = el.Transform(ip[n]) 

		jac = el.Jacobian(ip[n])
		sig = sigma_t(X)
		linalg.AddOuter(dE(X)/sig*w[n]*jac, g, s, elmat)
		linalg.AddOuter(E(X)/sig*w[n]*jac, g, g, elmat)

	return elmat 

def SourceSIP(face_t, c):
	sigma_t = c[0]
	Q1 = c[1] 
	xi1 = face_t.IPTrans(0)
	xi2 = face_t.IPTrans(1)
	X = face_t.el1.Transform(xi1)
	s1 = face_t.el1.CalcShape(xi1)
	s2 = face_t.el2.CalcShape(xi2)
	if (face_t.boundary):
		jump = s1 
	else:
		jump = np.concatenate((s1, -s2))
	return -jump * Q1(X)/sigma_t(X) * face_t.nor 

def Error(Ne, p):
	xe = np.linspace(0,1,Ne+1)
	fes = L2Space(xe, LobattoBasis(p))
	t = 1
	a = .1
	s = t-a 
	# E = lambda x: 1/3 + np.sin(2*pi*x)/10
	# dE = lambda x: pi/5*np.cos(2*pi*x)
	# E = lambda x: (-3*(x-1)*x + 5*np.sin(pi*x))/5/(x-x**2 + 3*np.sin(pi*x))
	# dE = lambda x: 4/5*(pi*(x-1)*x*np.cos(pi*x) + (1-2*x)*np.sin(pi*x))/(x-x**2+3*np.sin(pi*x))**2
	E = lambda x: (3*(1-x)*x + 5*np.sin(pi*x))/(5*(x-x**2 + 3*np.sin(pi*x)))
	dE = lambda x: 4/5*(pi*(x-1)*x*np.cos(pi*x) + (1-2*x)*np.sin(pi*x))/(x-x**2+3*np.sin(pi*x))**2
	Eb = (3+6*pi)/(4+12*pi)
	tfes = L2Space(xe, LegendreBasis(p+1))
	quad = LegendreQuad(6)
	psi = TVector(tfes, quad)
	psi_ex = lambda x, mu: .5*(np.sin(pi*x)+mu**2*x*(1-x))
	psi.Project(psi_ex)
	qdf = QDFactors(tfes, quad, psi_ex)
	qdf.Compute(psi) 

	# K = Assemble(fes, VEFDiffusionIntegrator, [lambda x: t, E, dE], 2*p+1)
	K = Assemble(fes, VEFPoissonIntegrator, [qdf, lambda x: t], 2*p+1)
	A = Assemble(fes, MassIntegrator, lambda x: a, 2*p+1)
	# F = FaceAssembleAll(fes, VEFSIPIntegrator, [lambda x: t, E, dE, Ne*(p+1)**2])
	F = FaceAssemble(fes, VEFSIPIntegrator, [lambda x: t, qdf, (p+1)**2]) \
		+ BdrFaceAssemble(fes, SIPBC, qdf)
	M = K + A + F

	# Q0 = lambda x: (pi**2/3 + a)*np.sin(pi*x)
	Q0 = lambda x: a*x/3 -a*x**2/3 + a*np.sin(pi*x)
	Q1 = lambda x: 1/5-2*x/5 + 1/3*pi*np.cos(pi*x)
	b = AssembleRHS(fes, DomainIntegrator, Q0, 2*p+1) \
		+ AssembleRHS(fes, GradDomainIntegrator, lambda x: Q1(x)/t, 2*p+1) \
		+ FaceAssembleRHS(fes, SourceSIP, [lambda x: t, Q1])
	# b = AssembleRHS(fes, DomainIntegrator, lambda x: 1, 2*p+1)

	phi = GridFunction(fes)
	phi.data = spla.spsolve(M, b) 
	# uex = lambda x: np.sin(pi*x)
	uex = lambda x: np.sin(pi*x) + 1/3*x*(1-x)
	r = np.linalg.norm(M*phi.data - b)
	if (r>1e-10):
		print('res = {:.3e}'.format(r))
	err = phi.L2Error(uex, 2*p+2)	
	x,subel = phi.EvalSubEl(20)
	plt.plot(x,subel)
	plt.plot(x,uex(x),'--')
	plt.show()
	return err 

p = 1
Ne = 20
E1 = Error(Ne, p)
E2 = Error(2*Ne, p)
print('E1 = {:.3e}, E2 = {:.3e}, ooa = {:.3f}'.format(E1, E2, np.log2(E1/E2)))