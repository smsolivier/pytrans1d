#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans1d import * 
import pytest

def test_iso():
	psi.Project(lambda x, mu: 1)
	qdf.Compute(psi)
	qdf.psi_in = lambda x, mu: 1 
	assert(qdf.EvalFactor(space.el[0],0)==pytest.approx(1/3))
	assert(qdf.EvalFactorBdr(space.iface[0])==pytest.approx(1/3))
	assert(qdf.EvalFactorDeriv(space.el[0], 0)==pytest.approx(0))
	assert(qdf.EvalGInt(space.el[0], 0)==pytest.approx(.5))
	assert(qdf.EvalG(space.iface[0])==pytest.approx(.5))
	assert(qdf.EvalCp(space.iface[0])==pytest.approx(.5))
	assert(qdf.EvalCm(space.iface[0])==pytest.approx(.5))
	assert(qdf.EvalJinBdr(space.bface[0])==pytest.approx(-.5))
	assert(qdf.EvalJinBdr(space.bface[-1])==pytest.approx(-.5))
	assert(qdf.EvalPhiInBdr(space.bface[0])==pytest.approx(1))
	assert(qdf.EvalPhiInBdr(space.bface[-1])==pytest.approx(1))

def test_lin():
	psi.Project(lambda x, mu: 1 + mu) 
	qdf.Compute(psi)
	assert(qdf.EvalFactor(space.el[0],0)==pytest.approx(1/3))
	assert(qdf.EvalFactorBdr(space.iface[0])==pytest.approx(1/3))
	assert(qdf.EvalFactorDeriv(space.el[0], 0)==pytest.approx(0))
	assert(qdf.EvalGInt(space.el[0], 0)==pytest.approx(.5))
	assert(qdf.EvalG(space.iface[0])==pytest.approx(.5))
	assert(qdf.EvalCp(space.iface[0])==pytest.approx(5/9))
	assert(qdf.EvalCm(space.iface[0])==pytest.approx(1/3))

def test_quad():
	psi.Project(lambda x, mu: 1 + mu**2) 
	qdf.Compute(psi)
	assert(qdf.EvalFactor(space.el[0],0)==pytest.approx(2/5))
	assert(qdf.EvalFactorBdr(space.iface[0])==pytest.approx(2/5))
	assert(qdf.EvalFactorDeriv(space.el[0], 0)==pytest.approx(0))
	assert(qdf.EvalGInt(space.el[0], 0)==pytest.approx(9/16))
	assert(qdf.EvalG(space.iface[0])==pytest.approx(9/16))
	assert(qdf.EvalCp(space.iface[0])==pytest.approx(9/16))
	assert(qdf.EvalCm(space.iface[0])==pytest.approx(9/16))

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
	Jl = lambda x: -f(x)/4 - 1/6*g(x) - 1/8*h(x)
	Jr = lambda x: -f(x)/4 + 1/6*g(x) - 1/8*h(x)
	phil = lambda x: f(x)/2 + g(x)/4 + h(x)/6 
	phir = lambda x: f(x)/2 - g(x)/4 + h(x)/6 
	E = lambda x: (1/3*f(x) + 1/5*h(x))/phi_ex(x) 
	G = lambda x: (.5*f(x) + .25*h(x))/phi_ex(x)
	def dE(x):
		t = (1/3*df(x) + 1/5*dh(x))*(f(x) + 1/3*h(x)) - (1/3*f(x) + 1/5*h(x))*(df(x) + 1/3*dh(x))
		b = (f(x) + 1/3*h(x))**2 
		return t/b 

	return psi_ex, phi_ex, Jl, Jr, phil, phir, E, dE, G

def test_quad_spat():
	x = .5
	e, xi = space.InverseMap(x) 
	el = space.el[e]

	face = space.iface[4]
	xif = face.IPTrans(0)
	xf = face.el1.Transform(xif) 

	psi_ex, phi_ex, Jl, Jr, phil, phir, E, dE, G = SetupMMS(
		1, .1, .1, 1, .1, lambda x: 1, lambda x: .9) 
	psi.Project(psi_ex)
	qdf.Compute(psi) 
	qdf.psi_in = psi_ex
	assert(qdf.EvalFactor(el, xi)==pytest.approx(E(x)))
	assert(qdf.EvalFactorBdr(face)==pytest.approx(E(xf)))
	assert(qdf.EvalFactorDeriv(el, xi)==pytest.approx(dE(x)))
	assert(qdf.EvalGInt(el, xi)==pytest.approx(G(x)))
	assert(qdf.EvalJinBdr(space.bface[0])==pytest.approx(Jl(0)))
	assert(qdf.EvalJinBdr(space.bface[-1])==pytest.approx(Jr(1)))
	assert(qdf.EvalPhiInBdr(space.bface[0])==pytest.approx(phil(0)))
	assert(qdf.EvalPhiInBdr(space.bface[-1])==pytest.approx(phir(1)))
	assert(qdf.EvalCp(face)==pytest.approx(abs(Jl(xf)/phil(xf))))
	assert(qdf.EvalCm(face)==pytest.approx(abs(Jr(xf)/phir(xf))))

N = 8
quad = DoubleLegendreQuad(N)
Ne = 25
p = 5
leg = LegendreBasis(p)
xe = np.linspace(0,1, Ne+1)
space = L2Space(xe, leg) 
psi = TVector(space, quad)
psi_in = lambda x, mu: 0 
qdf = QDFactors(space, quad, psi_in) 
psi.Project(lambda x, mu: 1)
qdf.Compute(psi) 