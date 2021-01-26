#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from trans1d import * 

def Error(Ne, p):
	xe = np.linspace(0,1,Ne+1)
	sfes = L2Space(xe, LegendreBasis(p))
	vfes = L2Space(xe, LegendreBasis(p))
	tfes = L2Space(xe, LobattoBasis(p))
	quad = LegendreQuad(6)

	alpha = 1 
	beta = .1
	gamma = 1
	delta = 1/2
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
	lldg = LiftedLDGVEF(sfes, qdf, sigma_t, sigma_s, source)
	sip = SIPVEF(sfes, qdf, sigma_t, sigma_s, source)
	br2 = BR2VEF(sfes, qdf, sigma_t, sigma_s, source)
	sweeper = DirectSweeper(tfes, quad, sigma_t, sigma_s, source, psi_ex, False)

	psi = TVector(tfes, quad)
	psi.Project(lambda x, mu: 1)
	npi = NPI(sweeper, ldg, sfes, psi)
	phi = GridFunction(sfes)
	global it
	global norm 
	it = 0
	norm = 0 
	def cb(x,f):
		global it 
		global norm 
		it += 1 
		norm = np.linalg.norm(f)

	# phi.data = optimize.anderson(npi.F, np.ones(sfes.Nu), callback=cb, f_tol=1e-10)
	fpi = FixedPointIteration(npi.F, 1e-10, 25, True)
	phi.data = fpi.Solve(np.ones(sfes.Nu))
	# phi.data = optimize.newton_krylov(npi.F, np.ones(sfes.Nu), callback=cb, f_tol=1e-10, maxiter=25)
	print('it = {}, norm = {:.3e}'.format(it, norm))
	# x,subel = phi.EvalSubEl(20)
	# plt.plot(x,subel)
	# plt.plot(x,phi_ex(x), '--')
	# plt.show()
	return phi.L2Error(phi_ex, 2*p+2)

Ne = 10
p = 2
E1 = Error(Ne, p)
E2 = Error(2*Ne, p)
print('p = {:.3f} ({:.3e}, {:.3e})'.format(np.log2(E1/E2), E1, E2))