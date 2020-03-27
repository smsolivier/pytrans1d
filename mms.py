#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from integrators import * 
from sn import * 
from qdf import * 
from vef import * 

def SolveH1Diffusion(Ne, p):
	xe = np.linspace(0, 1, Ne+1)
	basis = LagrangeBasis(p) 
	space = H1Space(xe, basis) 
	K = Assemble(space, WeakPoissonIntegrator, lambda x: 1, 2*p-1).tolil()
	b = AssembleRHS(space, DomainIntegrator, lambda x: np.pi**2*np.sin(np.pi*x), 2*p+1) 
	K[0,:] = 0 
	K[0,0] = 1 
	K[-1,:] = 0 
	K[-1,-1] = 1 
	b[0] = b[-1] = 0 
	phi = GridFunction(space) 
	phi.data = spla.spsolve(K.tocsc(), b) 
	ex = lambda x: np.sin(np.pi*x) 
	err = phi.L2Error(ex, 2*p+1) 

	return err 

def SolveSn(Ne, p):
	N = 8
	basis = LegendreBasis(p)
	xe = np.linspace(0,1,Ne+1)
	space = L2Space(xe, basis) 
	psi = TVector(space, N) 
	sigma_t = lambda x: 1 
	sigma_s = lambda x: .9
	Q = lambda x, mu: (mu*np.pi*np.cos(np.pi*x) + (sigma_t(x)-sigma_s(x))*np.sin(np.pi*x))/2
	psi_in = lambda x, mu: 0 
	sweep = DirectSweeper(space, N, sigma_t, sigma_s, Q, psi_in, False)
	sn = Sn(sweep) 
	phi = sn.SourceIteration(psi, tol=1e-12)
	phi_ex = lambda x: np.sin(np.pi*x) 
	return phi.L2Error(phi_ex, 2*p+1)

def SolveVEF(Ne, p):
	N = 8 
	leg = LegendreBasis(p-1)
	leg2 = LegendreBasis(p)
	lob = LobattoBasis(p) 
	xe = np.linspace(0,1,Ne+1)
	space = L2Space(xe, leg2)
	phi_space = L2Space(xe, leg)
	J_space = H1Space(xe, lob) 
	psi = TVector(space, N)
	alpha = 1 
	beta = .1
	gamma = 1 
	delta = 1
	eta = .1
	L = 1 + 2*eta
	psi_ex = lambda x, mu: .5*(alpha*np.sin(np.pi*(x+eta)/L) 
		+ beta*mu*np.sin(2*np.pi*x) + gamma*mu**2*x*(1-x) + delta)
	phi_ex = lambda x: alpha*np.sin(np.pi*(x+eta)/L) + gamma/3*x*(1-x) + delta 
	J_ex = lambda x: beta/3*np.sin(2*np.pi*x) 
	psi.Project(psi_ex)
	qdf = QDFactors(space, N, psi_ex) 
	qdf.Compute(psi) 
	sigma_t = lambda x: 1 
	sigma_s = lambda x: .1
	sigma_a = lambda x: sigma_t(x) - sigma_s(x) 
	Q = lambda x, mu: .5*(mu*alpha*np.pi/L*np.cos(np.pi*(x+eta)/L) + beta*mu**2*2*np.pi*np.cos(2*np.pi*x) 
		+ gamma*mu**3*(1-2*x)) + sigma_t(x)*psi_ex(x,mu) - sigma_s(x)/2*phi_ex(x)

	Q0 = np.zeros(phi_space.Nu)
	Q1 = np.zeros(J_space.Nu)
	mu, w = quadrature.Get(N)
	for a in range(N):
		Q0 += w[a] * AssembleRHS(phi_space, DomainIntegrator, lambda x: Q(x,mu[a]), 2*p+1)
		Q1 += mu[a] * w[a] * AssembleRHS(J_space, DomainIntegrator, lambda x: Q(x,mu[a]), 2*p+1) 

	qin = BdrFaceAssembleRHS(J_space, VEFInflowIntegrator, qdf) 
	Q1 += qin 

	Mt = Assemble(J_space, MassIntegrator, sigma_t, 2*p+1)
	B = BdrFaceAssemble(J_space, MLBdrIntegrator, qdf)
	Ma = Assemble(phi_space, MassIntegrator, sigma_a, 2*p+1)
	G = MixAssemble(J_space, phi_space, MixWeakEddDivIntegrator, qdf, 2*p+1)
	D = MixAssemble(phi_space, J_space, MixDivIntegrator, 1, 2*p+1) 

	A = sp.bmat([[Mt+B, G], [D, Ma]]).tocsc()
	rhs = np.concatenate((Q1, Q0))

	x = spla.spsolve(A, rhs) 
	phi = GridFunction(phi_space)
	phi.data = x[J_space.Nu:]
	J = GridFunction(J_space)
	J.data = x[:J_space.Nu]

	err = phi.L2Error(phi_ex, 2*p+1)
	jerr = J.L2Error(J_ex, 2*p+1) 
	return np.array([err, jerr])

def SolveHybDiffusion(Ne, p):
	xe = np.linspace(0,1,Ne+1)
	leg = LegendreBasis(p)
	lob = LobattoBasis(p+1)
	mcol = LagrangeBasis(1)
	l2 = L2Space(xe, leg)
	h1 = L2Space(xe, lob) 
	mspace = H1Space(xe, mcol) 

	Tex = lambda x: np.sin(3*np.pi*x)
	F = lambda x: 9*np.pi**2*np.sin(3*np.pi*x)
	qorder = max(2, 2*p+1)
	M = Assemble(h1, MassIntegrator, lambda x: -1, qorder)
	D = MixAssemble(l2, h1, MixDivIntegrator, 1, qorder)
	C = MixFaceAssemble(mspace, h1, ConstraintIntegrator, -1)
	f = AssembleRHS(l2, DomainIntegrator, F, qorder) 

	Minv = spla.inv(M)
	A = D*Minv*D.transpose()
	Ainv = spla.inv(A) 
	R = C*(Minv*D.transpose()*Ainv*D*Minv - Minv)*C.transpose()
	rhs = -C*Minv*D.transpose()*Ainv*f 
	R = R.tolil()
	R[0,0] = 1 
	R[-1,-1] = 1 
	rhs[0] = 0 
	rhs[-1] = 0 

	lam = GridFunction(mspace)
	# lam = spla.spsolve(R.tocsc(), rhs)
	lam.data, info = spla.cg(R.tocsc(), rhs, tol=1e-12) 
	res = np.linalg.norm(R*lam - rhs)
	if (res > 1e-10):
		print(colored('cg not converged. final tol = {:.3e}'.format(res), 'red'))

	T = GridFunction(l2)
	T.data = -Ainv*(f + D*Minv*C.transpose()*lam)

	err = T.L2ProjError(Tex, 2*p+1)
	merr = np.max(np.fabs(Tex(mspace.x) - lam.data))
	return err, merr 

def SolveHybVEF(Ne, p):
	N = 8 
	leg = LegendreBasis(p)
	leg2 = LegendreBasis(p+1)
	lob = LobattoBasis(p+1) 
	lag = LagrangeBasis(1)
	xe = np.linspace(0,1,Ne+1)
	space = L2Space(xe, leg2)
	phi_space = L2Space(xe, leg)
	J_space = L2Space(xe, lob) 
	m_space = H1Space(xe, lag) 
	psi = TVector(space, N)
	alpha = 1 
	beta = .1
	gamma = 1
	delta = 1
	eta = .1
	L = 1 + 2*eta
	psi_ex = lambda x, mu: .5*(alpha*np.sin(np.pi*(x+eta)/L) 
		+ beta*mu*x*(1-x) + gamma*mu**2*np.sin(2*np.pi*x) + delta)
	phi_ex = lambda x: alpha*np.sin(np.pi*(x+eta)/L) + gamma/3*np.sin(2*np.pi*x) + delta 
	psi.Project(psi_ex)
	qdf = QDFactors(space, N, psi_ex) 
	qdf.Compute(psi) 
	sigma_t = lambda x: 1 
	sigma_s = lambda x: .1
	sigma_a = lambda x: sigma_t(x) - sigma_s(x) 
	Q = lambda x, mu: .5*(mu*alpha*np.pi/L*np.cos(np.pi*(x+eta)/L) + beta*mu**2*(1-2*x) 
		+ gamma*mu**3*2*np.pi*np.cos(2*np.pi*x)) + sigma_t(x)*psi_ex(x,mu) - sigma_s(x)/2*phi_ex(x)

	Q0 = np.zeros(phi_space.Nu)
	Q1 = np.zeros(J_space.Nu)
	mu, w = quadrature.Get(N)
	qorder = max(2, 2*p+1)
	for a in range(N):
		Q0 += w[a] * AssembleRHS(phi_space, DomainIntegrator, lambda x: Q(x,mu[a]), qorder)
		Q1 += mu[a] * w[a] * AssembleRHS(J_space, DomainIntegrator, lambda x: Q(x,mu[a]), qorder) 

	qin = BdrFaceAssembleRHS(J_space, VEFInflowIntegrator, qdf) 
	Q1 += qin 

	Mt = Assemble(J_space, MassIntegrator, sigma_t, qorder)
	B = BdrFaceAssemble(J_space, MLBdrIntegrator, qdf)
	Mt += B
	Ma = Assemble(phi_space, MassIntegrator, sigma_a, qorder)
	G = MixAssemble(J_space, phi_space, MixWeakEddDivIntegrator, qdf, qorder)
	D = MixAssemble(phi_space, J_space, MixDivIntegrator, 1, qorder) 
	C1 = MixFaceAssemble(J_space, m_space, UpwEddConstraintIntegrator, qdf)
	C2 = MixFaceAssemble(m_space, J_space, ConstraintIntegrator, 1) 

	Mtinv = spla.inv(Mt)
	S = Ma - D*Mtinv*G 
	Sinv = spla.inv(S) 
	W = Mtinv + Mtinv*G*Sinv*D*Mtinv 
	X = -Mtinv*G*Sinv 
	Y = -Sinv*D*Mtinv 
	R = C2*W*C1 
	rhs = C2*W*Q1 + C2*X*Q0

	bdr = W*C1 
	bdr = bdr.tolil()
	bdr[0,:] *= -1 
	bdr[0,0] += qdf.EvalG(phi_space.bface[0])
	bdr[-1,-1] += qdf.EvalG(phi_space.bface[-1])
	bdr_rhs = W*Q1 + X*Q0 
	bdr_rhs[0] *= -1 
	bdr_rhs[0] -= 2*qdf.EvalJinBdr(phi_space.bface[0]) 
	bdr_rhs[-1] -= 2*qdf.EvalJinBdr(phi_space.bface[-1])

	R = R.tolil()
	R[0,:] = bdr[0,:]
	R[-1,:] = bdr[-1,:]
	rhs[0] = bdr_rhs[0]
	rhs[-1] = bdr_rhs[-1]

	lam = GridFunction(m_space)
	lam.data = spla.spsolve(R.tocsc(), rhs) 

	phi = GridFunction(phi_space)
	phi.data = Y*Q1 - Y*C1*lam + Sinv*Q0 

	err = phi.L2ProjError(phi_ex, 2*p+1)
	merr = np.max(np.fabs(phi_ex(m_space.x) - lam.data))
	return err, merr

def SolveVEFSn(Ne, p):
	N = 8 
	leg = LegendreBasis(p-1)
	lob = LobattoBasis(p)
	tleg = LegendreBasis(p) 
	xe = np.linspace(0,1,Ne+1)
	phi_space = L2Space(xe, leg)
	J_space = H1Space(xe, lob)
	tspace = L2Space(xe, tleg)

	alpha = 1 
	beta = .1
	gamma = 1
	delta = 1
	eta = .1
	L = 1 + 2*eta
	psi_ex = lambda x, mu: .5*(alpha*np.sin(np.pi*(x+eta)/L) 
		+ beta*mu*x*(1-x) + gamma*mu**2*np.sin(2*np.pi*x) + delta)
	phi_ex = lambda x: alpha*np.sin(np.pi*(x+eta)/L) + gamma/3*np.sin(2*np.pi*x) + delta
	sigma_t = lambda x: 1 
	sigma_s = lambda x: .1
	sigma_a = lambda x: sigma_t(x) - sigma_s(x) 
	Q = lambda x, mu: .5*(mu*alpha*np.pi/L*np.cos(np.pi*(x+eta)/L) + beta*mu**2*(1-2*x) 
		+ gamma*mu**3*2*np.pi*np.cos(2*np.pi*x)) + sigma_t(x)*psi_ex(x,mu) - sigma_s(x)/2*phi_ex(x)

	sweep = DirectSweeper(tspace, N, sigma_t, sigma_s, Q, psi_ex, False)
	block = BlockLDU(1e-12, 100, 1, False)
	vef = VEF(phi_space, J_space, sweep, block, True)
	psi = TVector(tspace, N)
	phi = vef.SourceIteration(psi)

	err = phi.L2Error(phi_ex, 2*p+1)
	return err 

def SolveVEFHSn(Ne, p):
	N = 8 
	leg = LegendreBasis(p-1)
	lob = LobattoBasis(p)
	tleg = LegendreBasis(p) 
	xe = np.linspace(0,1,Ne+1)
	phi_space = L2Space(xe, leg)
	J_space = H1Space(xe, lob)
	tspace = L2Space(xe, tleg)

	alpha = 1 
	beta = .1
	gamma = 1
	delta = 1
	eta = .1
	L = 1 + 2*eta
	psi_ex = lambda x, mu: .5*(alpha*np.sin(np.pi*(x+eta)/L) 
		+ beta*mu*x*(1-x) + gamma*mu**2*np.sin(2*np.pi*x) + delta)
	phi_ex = lambda x: alpha*np.sin(np.pi*(x+eta)/L) + gamma/3*np.sin(2*np.pi*x) + delta
	sigma_t = lambda x: 1 
	sigma_s = lambda x: .1
	sigma_a = lambda x: sigma_t(x) - sigma_s(x) 
	Q = lambda x, mu: .5*(mu*alpha*np.pi/L*np.cos(np.pi*(x+eta)/L) + beta*mu**2*(1-2*x) 
		+ gamma*mu**3*2*np.pi*np.cos(2*np.pi*x)) + sigma_t(x)*psi_ex(x,mu) - sigma_s(x)/2*phi_ex(x)

	sweep = DirectSweeper(tspace, N, sigma_t, sigma_s, Q, psi_ex, False)
	amg = AMGSolver(1e-12, 100, 1)
	vef = VEFH(phi_space, J_space, sweep, amg, True)
	psi = TVector(tspace, N)
	phi = vef.SourceIteration(psi)

	err = phi.L2Error(phi_ex, 2*p+1)
	return err 

Ne = 4
print('h1 diffusion:')
for p in range(1, 6):
	E1 = SolveH1Diffusion(Ne, p)
	E2 = SolveH1Diffusion(2*Ne, p)
	ooa = np.log(E1/E2)/np.log(2)
	color = 'green'
	if (abs(ooa-p-1) > .1):
		color = 'red'
	print(colored('   p={}, ooa={:.3f}'.format(p, ooa), color))

print('Sn:')
for p in range(1, 6):
	E1 = SolveSn(Ne, p)
	E2 = SolveSn(2*Ne, p)
	ooa = np.log(E1/E2)/np.log(2)
	color = 'green'
	if (abs(ooa-p-1) > .1):
		color = 'red'
	print(colored('   p={}, ooa={:.3f}'.format(p, ooa), color))

Ne = 10
print('VEF:')
for p in range(1, 6):
	E1 = SolveVEF(Ne, p)
	E2 = SolveVEF(2*Ne, p)
	ooa = np.log(E1/E2)/np.log(2)
	color = 'green'
	if (abs(ooa[0]-p) > .1 or abs(ooa[1]-p-1) > .1):
		color = 'red'
	print(colored('   p={}, ooa={:.3f} ({:.3e}, {:.3e}), jooa={:.3f} ({:.3e}, {:.3e})'.format(
		p, ooa[0], E1[0], E2[0], ooa[1], E1[1], E2[1]), color))

Ne = 10
print('Hyb Diffusion:')
for p in range(0, 6):
	E1, mE1 = SolveHybDiffusion(Ne, p)
	E2, mE2 = SolveHybDiffusion(2*Ne, p)
	ooa = np.log(E1/E2)/np.log(2)
	color = 'green'
	if (abs(ooa-p-2) > .1):
		color = 'red'
	print(colored('   p={}, ooa={:.3f}, m1={:.3e}, m2={:.3e}'.format(p, ooa, mE1, mE2), color))


Ne = 6
print('Hyb VEF:')
for p in range(0, 6):
	E1, mE1 = SolveHybVEF(Ne, p)
	E2, mE2 = SolveHybVEF(2*Ne, p)
	ooa = np.log(E1/E2)/np.log(2)
	color = 'green'
	if (abs(ooa-p-2) > .1):
		color = 'red'
	print(colored('   p={}, ooa={:.3f}, m1={:.3e}, m2={:.3e}'.format(p, ooa, mE1, mE2), color))

Ne = 5
print('Full VEF Alg')
for p in range(1,5):
	E1 = SolveVEFSn(Ne, p)
	E2 = SolveVEFSn(2*Ne, p)
	ooa = np.log(E1/E2)/np.log(2)
	color = 'green'
	if (abs(ooa-p-1)>.1):
		color = 'red'
	print(colored('   p={}, ooa={:.3f}'.format(p, ooa), color))

Ne = 5
print('Full VEFH Alg')
for p in range(1,5):
	E1 = SolveVEFHSn(Ne, p)
	E2 = SolveVEFHSn(2*Ne, p)
	ooa = np.log(E1/E2)/np.log(2)
	color = 'green'
	if (abs(ooa-p-1)>.1):
		color = 'red'
	print(colored('   p={}, ooa={:.3f}'.format(p, ooa), color))