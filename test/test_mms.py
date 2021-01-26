#!/usr/bin/env python3

import numpy as np
from scipy import optimize
from trans1d import * 
import pytest 

import warnings 

def H1Diffusion(Ne, p):
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

def BR2Diffusion(Ne, p):
	xe = np.linspace(0,1,Ne+1)
	space = L2Space(xe, LegendreBasis(p))
	K = Assemble(space, WeakPoissonIntegrator, lambda x: 1, 2*p+1) \
		+ FaceAssembleAll(space, BR2Integrator, 5)
	b = AssembleRHS(space, DomainIntegrator, lambda x: np.pi**2*np.sin(np.pi*x), 2*p+1)
	phi = GridFunction(space)
	phi.data = spla.spsolve(K, b)
	return phi.L2Error(lambda x: np.sin(np.pi*x), 2*p+2)

def Transport(Ne, p):
	N = 8
	quad = LegendreQuad(N)
	basis = LegendreBasis(p)
	xe = np.linspace(0,1,Ne+1)
	space = L2Space(xe, basis) 
	psi = TVector(space, quad) 
	sigma_t = lambda x: 1 
	sigma_s = lambda x: .9
	Q = lambda x, mu: (mu*np.pi*np.cos(np.pi*x) + (sigma_t(x)-sigma_s(x))*np.sin(np.pi*x))/2
	psi_in = lambda x, mu: 0 
	sweep = DirectSweeper(space, quad, sigma_t, sigma_s, Q, psi_in, False)
	sn = Sn(sweep) 
	phi = sn.SourceIteration(psi, tol=1e-12)
	phi_ex = lambda x: np.sin(np.pi*x) 
	return phi.L2Error(phi_ex, 2*p+2)

def FullVEF(Ne, p):
	N = 8 
	quad = LegendreQuad(N)
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

	sweep = DirectSweeper(tspace, quad, sigma_t, sigma_s, Q, psi_ex, False)
	block = BlockLDU(1e-12, 100, 1, False)
	vef = VEF(phi_space, J_space, sweep, None, True)
	psi = TVector(tspace, quad)
	phi = vef.SourceIteration(psi)

	err = phi.L2Error(phi_ex, 2*p+1)
	return err 

def FullVEFH(Ne, p):
	N = 8 
	quad = LegendreQuad(N)
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

	sweep = DirectSweeper(tspace, quad, sigma_t, sigma_s, Q, psi_ex, False)
	amg = AMGSolver(1e-12, 100, 1)
	vef = VEFH(phi_space, J_space, sweep, amg, True)
	psi = TVector(tspace, quad)
	phi = vef.SourceIteration(psi)

	err = phi.L2Error(phi_ex, 2*p+1)
	return err 

def FullVEFH2(Ne, p):
	N = 8 
	quad = LegendreQuad(N)
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

	sweep = DirectSweeper(tspace, quad, sigma_t, sigma_s, Q, psi_ex, False)
	vef = VEFH2(phi_space, J_space, sweep, None, True)
	psi = TVector(tspace, quad)
	phi = vef.SourceIteration(psi) 

	err = phi.L2Error(phi_ex, 2*p+1)
	return err 

def S2SATransport(Ne, p):
	N = 4
	quad = LegendreQuad(N)
	leg = LegendreBasis(p) 
	xe = np.linspace(0,1,Ne+1)
	space = L2Space(xe, leg)

	alpha = 1 
	beta = .1
	gamma = 1
	delta = 1
	eta = .1
	eps = 1e-3
	L = 1 + 2*eta
	psi_ex = lambda x, mu: .5*(alpha*np.sin(np.pi*(x+eta)/L) 
		+ beta*mu*x*(1-x) + gamma*mu**2*np.sin(2*np.pi*x) + delta)
	phi_ex = lambda x: alpha*np.sin(np.pi*(x+eta)/L) + gamma/3*np.sin(2*np.pi*x) + delta
	sigma_t = lambda x: 1/eps
	sigma_s = lambda x: 1/eps - eps
	sigma_a = lambda x: sigma_t(x) - sigma_s(x) 
	Q = lambda x, mu: .5*(mu*alpha*np.pi/L*np.cos(np.pi*(x+eta)/L) + beta*mu**2*(1-2*x) 
		+ gamma*mu**3*2*np.pi*np.cos(2*np.pi*x)) + sigma_t(x)*psi_ex(x,mu) - sigma_s(x)/2*phi_ex(x)

	sweep = DirectSweeper(space, quad, sigma_t, sigma_s, Q, psi_ex, False)
	sn = S2SA(sweep)
	psi = TVector(space, quad)
	phi = sn.SourceIteration(psi, tol=1e-10)

	return phi.L2Error(phi_ex, 2*p+1)

def P1SATransport(Ne, p):
	N = 4
	quad = LegendreQuad(N)
	leg = LegendreBasis(p) 
	xe = np.linspace(0,1,Ne+1)
	space = L2Space(xe, leg)

	alpha = 1 
	beta = .1
	gamma = 1
	delta = 1
	eta = .1
	eps = 1e-3
	L = 1 + 2*eta
	psi_ex = lambda x, mu: .5*(alpha*np.sin(np.pi*(x+eta)/L) 
		+ beta*mu*x*(1-x) + gamma*mu**2*np.sin(2*np.pi*x) + delta)
	phi_ex = lambda x: alpha*np.sin(np.pi*(x+eta)/L) + gamma/3*np.sin(2*np.pi*x) + delta
	sigma_t = lambda x: 1/eps
	sigma_s = lambda x: 1/eps - eps
	sigma_a = lambda x: sigma_t(x) - sigma_s(x) 
	Q = lambda x, mu: .5*(mu*alpha*np.pi/L*np.cos(np.pi*(x+eta)/L) + beta*mu**2*(1-2*x) 
		+ gamma*mu**3*2*np.pi*np.cos(2*np.pi*x)) + sigma_t(x)*psi_ex(x,mu) - sigma_s(x)/2*phi_ex(x)

	sweep = DirectSweeper(space, quad, sigma_t, sigma_s, Q, psi_ex, False)
	sn = P1SA(sweep)
	psi = TVector(space, quad)
	phi = sn.SourceIteration(psi, tol=1e-10)

	return phi.L2Error(phi_ex, 2*p+1)

def FullQD(Ne, p):
	N = 6
	quad = LegendreQuad(N)
	leg = LegendreBasis(p)
	xe = np.linspace(0,1, Ne+1)
	space = L2Space(xe, leg) 

	alpha = 1 
	beta = .1
	gamma = .1
	delta = 10
	eta = .1
	eps = 1e-1
	L = 1 + 2*eta
	psi_ex = lambda x, mu: .5*(alpha*np.sin(np.pi*(x+eta)/L) 
		+ beta*mu*x*(1-x) + gamma*mu**2*np.sin(2*np.pi*x) + delta)
	phi_ex = lambda x: alpha*np.sin(np.pi*(x+eta)/L) + gamma/3*np.sin(2*np.pi*x) + delta
	sigma_t = lambda x: 1/eps
	sigma_s = lambda x: 1/eps - eps
	sigma_a = lambda x: sigma_t(x) - sigma_s(x) 
	Q = lambda x, mu: .5*(mu*alpha*np.pi/L*np.cos(np.pi*(x+eta)/L) + beta*mu**2*(1-2*x) 
		+ gamma*mu**3*2*np.pi*np.cos(2*np.pi*x)) + sigma_t(x)*psi_ex(x,mu) - sigma_s(x)/2*phi_ex(x)

	sweep = DirectSweeper(space, quad, sigma_t, sigma_s, Q, psi_ex, False)
	qd = QD(space, space, sweep)
	psi = TVector(space, quad)
	psi.Project(lambda x, mu: 1)
	phi = qd.SourceIteration(psi, tol=1e-12)

	return phi.L2Error(phi_ex, 2*p+1)

def OnlySIPVEF(Ne, p):
	N = 6
	quad = LegendreQuad(N)
	xe = np.linspace(0,1,Ne+1)
	fes = L2Space(xe, LegendreBasis(p))
	tfes = L2Space(xe, LegendreBasis(p))

	alpha = 1 
	beta = .1
	gamma = .05
	delta = 10
	eta = .1
	eps = 1e-1
	L = 1 + 2*eta
	psi_ex = lambda x, mu: .5*(alpha*np.sin(np.pi*(x+eta)/L) 
		+ beta*mu*x*(1-x) + gamma*mu**2*np.sin(2*np.pi*x) + delta)
	phi_ex = lambda x: alpha*np.sin(np.pi*(x+eta)/L) + gamma/3*np.sin(2*np.pi*x) + delta
	sigma_t = lambda x: 1/eps
	sigma_s = lambda x: 1/eps - eps
	sigma_a = lambda x: sigma_t(x) - sigma_s(x) 
	Q = lambda x, mu: .5*(mu*alpha*np.pi/L*np.cos(np.pi*(x+eta)/L) + beta*mu**2*(1-2*x) 
		+ gamma*mu**3*2*np.pi*np.cos(2*np.pi*x)) + sigma_t(x)*psi_ex(x,mu) - sigma_s(x)/2*phi_ex(x)
	
	qdf = QDFactors(tfes, quad, psi_ex)
	sip = SIPVEF(fes, qdf, sigma_t, sigma_s, Q)
	psi = TVector(tfes, quad)
	psi.Project(psi_ex)
	phi = sip.Mult(psi)
	return phi.L2Error(phi_ex, 2*p+2)

def FullSIPVEF(Ne, p):
	N = 6
	quad = LegendreQuad(N)
	xe = np.linspace(0,1,Ne+1)
	fes = L2Space(xe, LegendreBasis(p))
	tfes = L2Space(xe, LegendreBasis(p))

	alpha = 1 
	beta = .1
	gamma = 1
	delta = 1/2
	eta = .1
	eps = 1e-1
	L = 1 + 2*eta
	psi_ex = lambda x, mu: .5*(alpha*np.sin(np.pi*(x+eta)/L) 
		+ beta*mu*x*(1-x) + gamma*mu**2*np.sin(2*np.pi*x) + delta)
	phi_ex = lambda x: alpha*np.sin(np.pi*(x+eta)/L) + gamma/3*np.sin(2*np.pi*x) + delta
	sigma_t = lambda x: 1
	sigma_s = lambda x: .1
	sigma_a = lambda x: sigma_t(x) - sigma_s(x) 
	Q = lambda x, mu: .5*(mu*alpha*np.pi/L*np.cos(np.pi*(x+eta)/L) + beta*mu**2*(1-2*x) 
		+ gamma*mu**3*2*np.pi*np.cos(2*np.pi*x)) + sigma_t(x)*psi_ex(x,mu) - sigma_s(x)/2*phi_ex(x)

	sweep = DirectSweeper(tfes, quad, sigma_t, sigma_s, Q, psi_ex, False)
	qdf = QDFactors(tfes, quad, psi_ex)
	vef = SIPVEF(fes, qdf, sigma_t, sigma_s, Q)
	psi = TVector(tfes, quad)
	psi.Project(lambda x, mu: 1)
	npi = NPI(sweep, vef, fes, psi)
	phi = GridFunction(fes)

	phi.data = optimize.newton_krylov(npi.F, np.ones(fes.Nu), f_tol=1e-10, maxiter=25)
	return phi.L2Error(phi_ex, 2*p+2)

def FullLDGVEF(Ne, p):
	xe = np.linspace(0,1,Ne+1)
	sfes = L2Space(xe, LegendreBasis(p))
	vfes = L2Space(xe, LegendreBasis(p))
	tfes = L2Space(xe, LegendreBasis(p))
	quad = LegendreQuad(6)

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
	source = lambda x, mu: .5*(mu*alpha*np.pi/L*np.cos(np.pi*(x+eta)/L) + beta*mu**2*(1-2*x) 
		+ gamma*mu**3*2*np.pi*np.cos(2*np.pi*x)) + sigma_t(x)*psi_ex(x,mu) - sigma_s(x)/2*phi_ex(x)

	qdf = QDFactors(tfes, quad, psi_ex)
	ldg = LDGVEF(sfes, vfes, qdf, sigma_t, sigma_s, source)
	sip = SIPVEF(sfes, qdf, sigma_t, sigma_s, source)
	sweeper = DirectSweeper(tfes, quad, sigma_t, sigma_s, source, psi_ex, False)

	psi = TVector(tfes, quad)
	psi.Project(lambda x, mu: 1)
	npi = NPI(sweeper, ldg, sfes, psi)
	phi = GridFunction(sfes)

	phi.data = optimize.anderson(npi.F, np.ones(sfes.Nu), maxiter=25, f_tol=1e-10)
	return phi.L2Error(phi_ex, 2*p+2)

def FullLiftedLDGVEF(Ne, p):
	xe = np.linspace(0,1,Ne+1)
	sfes = L2Space(xe, LegendreBasis(p))
	tfes = L2Space(xe, LegendreBasis(p))
	quad = LegendreQuad(6)

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
	source = lambda x, mu: .5*(mu*alpha*np.pi/L*np.cos(np.pi*(x+eta)/L) + beta*mu**2*(1-2*x) 
		+ gamma*mu**3*2*np.pi*np.cos(2*np.pi*x)) + sigma_t(x)*psi_ex(x,mu) - sigma_s(x)/2*phi_ex(x)

	qdf = QDFactors(tfes, quad, psi_ex)
	ldg = LiftedLDGVEF(sfes, qdf, sigma_t, sigma_s, source)
	sweeper = DirectSweeper(tfes, quad, sigma_t, sigma_s, source, psi_ex)

	psi = TVector(tfes, quad)
	psi.Project(lambda x, mu: 1)
	npi = NPI(sweeper, ldg, sfes, psi)
	phi = GridFunction(sfes)

	phi.data = optimize.anderson(npi.F, np.ones(sfes.Nu), maxiter=25, f_tol=1e-10)
	return phi.L2Error(phi_ex, 2*p+2)

def FullBR2VEF(Ne, p):
	xe = np.linspace(0,1,Ne+1)
	sfes = L2Space(xe, LegendreBasis(p))
	tfes = L2Space(xe, LegendreBasis(p))
	quad = LegendreQuad(6)

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
	source = lambda x, mu: .5*(mu*alpha*np.pi/L*np.cos(np.pi*(x+eta)/L) + beta*mu**2*(1-2*x) 
		+ gamma*mu**3*2*np.pi*np.cos(2*np.pi*x)) + sigma_t(x)*psi_ex(x,mu) - sigma_s(x)/2*phi_ex(x)

	qdf = QDFactors(tfes, quad, psi_ex)
	br2 = BR2VEF(sfes, qdf, sigma_t, sigma_s, source)
	sweeper = DirectSweeper(tfes, quad, sigma_t, sigma_s, source, psi_ex)

	psi = TVector(tfes, quad)
	psi.Project(lambda x, mu: 1)
	npi = NPI(sweeper, br2, sfes, psi)
	phi = GridFunction(sfes)

	phi.data = optimize.anderson(npi.F, np.ones(sfes.Nu), maxiter=25, f_tol=1e-10)
	return phi.L2Error(phi_ex, 2*p+2)

Ne = 10
@pytest.mark.parametrize('p', [1, 2, 3, 4])
@pytest.mark.parametrize('solver', [H1Diffusion, BR2Diffusion, Transport, 
	S2SATransport, P1SATransport, FullVEF, FullVEFH, 
	FullVEFH2, OnlySIPVEF, FullSIPVEF, FullLDGVEF, FullLiftedLDGVEF, FullBR2VEF])
def test_ooa(solver, p):
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
		warnings.filterwarnings('ignore', category=DeprecationWarning)

		E1 = solver(Ne, p)
		E2 = solver(2*Ne, p)
	ooa = np.log2(E1/E2)
	print('p={:.3f} ({:.3e}, {:.3e})'.format(ooa, E1, E2))
	assert(abs(p+1-ooa)<=.2)
