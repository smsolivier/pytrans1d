#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans1d import * 

class QDFactorsAnalytic:
	def Compute(self, psi):
		self.psi = psi 

	def EvalFactor(self, el, xi):
		X = el.Transform(xi)
		return E(X) 

	def EvalFactorDeriv(self, el, xi):
		X = el.Transform(xi)
		return dE(X) 

	def EvalFactorBdr(self, face):
		xi = face_t.IPTrans(0)
		X = face_t.el1.Transform(xi) 
		return E(X) 

	def EvalG(self, face_t):
		xi = face_t.IPTrans(0)
		X = face_t.el1.Transform(xi) 
		return Eb(X) 

	def EvalGInt(self, el, xi):
		X = el.Transform(xi)
		return Eb(X) 

	def EvalCp(self, face):
		raise NotImplementedError('consistent not implemented') 

	def EvalCm(self, face):
		raise NotImplementedError('consistent not implemented') 

	def EvalJinBdr(self, face_t):
		if (face_t.el1.ElNo==0):
			return -Jl(0)
		else:
			return Jr(1) 

	def EvalPhiInBdr(self, face_t):
		raise NotImplementedError('consistent not implemented') 


def SetupMMS(alpha, beta, gamma, delta, eta, sigt, sigs):
	L = 1 + 2*eta 
	f = lambda x: alpha*np.sin(np.pi*x) + delta 
	df = lambda x: alpha*np.pi*np.cos(np.pi*x) 
	g = lambda x: beta*np.sin(2*np.pi*x) 
	dg = lambda x: beta*2*np.pi*np.cos(2*np.pi*x) 
	h = lambda x: gamma*np.sin(3*np.pi*(x+eta)/L)
	dh = lambda x: gamma*3*np.pi/L*np.cos(3*np.pi*(x+eta)/L)
	psi_ex = lambda x, mu: .5*(f(x) + mu*g(x) + mu**2*h(x))
	dpsi = lambda x, mu: .5*(df(x) + mu*dg(x) + mu**2*dh(x))
	phi_ex = lambda x: f(x) + 1/3*h(x)
	J_ex = lambda x: 1/3*g(x)
	Jl = lambda x: f(x)/4 + 1/6*g(x) + 1/8*h(x)
	Jr = lambda x: -f(x)/4 + 1/6*g(x) - 1/8*h(x)
	E = lambda x: (1/3*f(x) + 1/5*h(x))/phi_ex(x) 
	G = lambda x: (.5*f(x) + .25*h(x))/phi_ex(x)
	def dE(x):
		t = (1/3*df(x) + 1/5*dh(x))*(f(x) + 1/3*h(x)) - (1/3*f(x) + 1/5*h(x))*(df(x) + 1/3*dh(x))
		b = (f(x) + 1/3*h(x))**2 
		return t/b 
	Q = lambda x, mu: mu*dpsi(x,mu) + sigt(x)*psi_ex(x,mu) - sigs(x)/2*phi_ex(x) 

	return psi_ex, phi_ex, J_ex, Jl, Jr, E, dE, G, Q

Ne = 10 
p = 3 
if (len(sys.argv)>1):
	Ne = int(sys.argv[1])
if (len(sys.argv)>2):
	p = int(sys.argv[2])
N = 8 
quad = LegendreQuad(N) 
sigma_t = lambda x: 1
sigma_s = lambda x: .9 
sigma_a = lambda x: sigma_t(x) - sigma_s(x) 
psi_ex, phi_ex, J_ex, Jl, Jr, E, dE, Eb, Q = SetupMMS(1, 1, 1, 1, .1, sigma_t, sigma_s) 

xe = np.linspace(0,1,Ne+1)
tspace = L2Space(xe, LegendreBasis(p
))
psi = TVector(tspace, quad) 
psi.Project(psi_ex) 
qdf = QDFactors(tspace, quad, psi_ex) 
# qdf = QDFactorsAnalytic()
qdf.Compute(psi) 

def Solve(Ne, p, qdf, pp):
	lob = LobattoBasis(p if pp else p+1)
	leg = LegendreBasis(p-1 if pp else p)
	Jspace = H1Space(xe, lob)
	phi_space = L2Space(xe, leg)

	qorder = 2*p+1 
	Mt = Assemble(Jspace, MassIntegrator, sigma_t, 2*p+1)
	B = BdrFaceAssemble(Jspace, MLBdrIntegrator, qdf)
	G = MixAssemble(Jspace, phi_space, MixWeakEddDivIntegrator, qdf, qorder) 
	D = MixAssemble(phi_space, Jspace, MixDivIntegrator, 1, qorder) 
	Ma = Assemble(phi_space, MassIntegrator, sigma_a, qorder)

	qin = BdrFaceAssembleRHS(Jspace, VEFInflowIntegrator, qdf) 

	Q0 = np.zeros(phi_space.Nu)
	Q1 = np.zeros(Jspace.Nu)
	for a in range(quad.N):
		mu = quad.mu[a]
		Q0 += AssembleRHS(phi_space, DomainIntegrator, lambda x: Q(x,mu), qorder)*quad.w[a]
		Q1 += AssembleRHS(Jspace, DomainIntegrator, lambda x: Q(x,mu), qorder)*mu*quad.w[a]

	A = sp.bmat([[Mt+B, G], [D, Ma]]).tocsc()
	rhs = np.concatenate((Q1+qin, Q0))

	x = spla.spsolve(A, rhs) 
	phi = GridFunction(phi_space)
	J = GridFunction(Jspace)
	J.data = x[:Jspace.Nu]
	phi.data = x[Jspace.Nu:]

	# k = p-1
	k = 0
	if (pp==1):
		star_space = L2Space(xe, type(phi_space.basis)(p))
		phi_star = GridFunction(star_space) 
		lam_space = L2Space(xe, LegendreBasis(k))
		for e in range(Ne):
			star_el = star_space.el[e] 
			lam_el = lam_space.el[e]
			phi_el = phi_space.el[e] 
			J_el = Jspace.el[e] 

			K = VEFPoissonIntegrator(star_el, [qdf, sigma_t], qorder)
			Ma = MassIntegrator(star_el, sigma_a, qorder) 

			Q0 = np.zeros(star_el.Nn)
			Q1 = np.zeros(star_el.Nn)
			for a in range(quad.N):
				w = quad.w[a]
				mu = quad.mu[a] 
				Q0 += w * DomainIntegrator(star_el, lambda x: Q(x,mu), qorder)
				Q1 += w * mu * GradDomainIntegrator(star_el, 
					lambda x: Q(x,mu)/sigma_t(x), qorder) 

			Jbdr = star_el.CalcShape(1)*J.Interpolate(e,1) - star_el.CalcShape(-1)*J.Interpolate(e,-1)
			f = Q0 + Q1 - Jbdr 

			M1 = MixMassIntegrator(star_el, lam_el, lambda x: 1, qorder)
			M2 = MixMassIntegrator(lam_el, phi_el, lambda x: 1, qorder) 
			g = np.dot(M2, phi.GetDof(e))

			A = np.bmat([[K+Ma, M1], [M1.transpose(), np.zeros((lam_el.Nn, lam_el.Nn))]])
			rhs = np.concatenate((f, g))

			x = np.linalg.solve(A, rhs) 
			phi_star.SetDof(e, x[:star_el.Nn])

		return phi_star 
	elif (pp==2):
		phi_star_space = L2Space(xe, type(leg)(p))
		J_star_space = H1Space(xe, type(lob)(p+1))
		lam_space = L2Space(xe, LegendreBasis(0))
		phi_star = GridFunction(phi_star_space)
		for e in range(Ne):
			phi_star_el = phi_star_space.el[e]
			lam_el = lam_space.el[e]
			Jstar_el = J_star_space.el[e]
			phi_el = phi_space.el[e]
			J_el = Jspace.el[e] 
			qorder = 2*(p+1)+1 

			Mt = MassIntegrator(Jstar_el, sigma_t, qorder)
			G = MixWeakEddDivIntegrator(Jstar_el, phi_star_el, qdf, qorder)
			D = MixDivIntegrator(phi_star_el, Jstar_el, 1, qorder)
			Ma = MassIntegrator(phi_star_el, sigma_a, qorder)

			Q0 = np.zeros(phi_star_el.Nn)
			Q1 = np.zeros(Jstar_el.Nn)
			for a in range(quad.N):
				mu = quad.mu[a]
				Q0 += DomainIntegrator(phi_star_el, lambda x: Q(x,mu), qorder)*quad.w[a]
				Q1 += DomainIntegrator(Jstar_el, lambda x: Q(x,mu), qorder)*mu*quad.w[a]

			Mt[0,:] = 0 
			G[0,:] = 0 
			Mt[0,0] = 1
			Mt[-1,:] = 0 
			G[-1,:] = 0
			Mt[-1,-1] = 1 
			Q1[0] = J.Interpolate(e,-1)
			Q1[-1] = J.Interpolate(e,1)

			M1 = MixMassIntegrator(phi_star_el, lam_el, lambda x: 1, qorder)
			M2 = MixMassIntegrator(lam_el, phi_el, lambda x: 1, qorder)

			A = np.block([
				[Mt, G, np.zeros((Jstar_el.Nn, lam_el.Nn))], 
				[D, Ma, M1], 
				[np.zeros((lam_el.Nn, Jstar_el.Nn)), M1.transpose(), 
				np.zeros((lam_el.Nn, lam_el.Nn))]])
			rhs = np.concatenate((Q1, Q0, np.dot(M2, phi.GetDof(e))))
			x = np.linalg.solve(A, rhs) 
			phi_star.SetDof(e, x[Jstar_el.Nn:Jstar_el.Nn+phi_star_el.Nn])

		return phi_star 
	else:
		return phi

phi = Solve(Ne, p, qdf, 0)
phi_pp = Solve(Ne, p, qdf, 1)
phi_pp2 = Solve(Ne, p, qdf, 2) 

err = phi.L2Error(phi_ex, 2*p+2)
err_pp = phi_pp.L2Error(phi_ex, 2*p+2) 
err_pp2 = phi_pp2.L2Error(phi_ex, 2*p+2)

print('err = {:.3e}'.format(err))
print('err_pp = {:.3e}'.format(err_pp))
print('err_pp2 = {:.3e}'.format(err_pp2))
print('ratio21 = {:.3f}'.format(err_pp/err))
print('ratio31 = {:.3f}'.format(err_pp2/err))

# plt.semilogy(phi.space.x, np.fabs(phi.data-phi_pp.data), '-o')
# plt.show()