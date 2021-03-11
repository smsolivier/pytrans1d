#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans1d import * 

p = 1
ooa = [] 
looa = [] 
sooa = [] 
Eps = np.logspace(0, -4, 15)
for eps in Eps:
	err = [] 
	lerr = [] 
	serr = [] 
	for Ne in [10,20]:
		xe = np.linspace(0,1,Ne+1)
		fes = L2Space(xe, LegendreBasis(p-1))
		vfes = H1Space(xe, LobattoBasis(p))
		tfes = L2Space(xe, LegendreBasis(p))
		quad = LegendreQuad(4)

		alpha = 1 
		beta = 0
		gamma = 0
		delta = 0
		eta = 0
		L = 1 + 2*eta
		psi_ex = lambda x, mu: .5*(alpha*np.sin(np.pi*(x+eta)/L) 
			+ beta*mu*x*(1-x) + gamma*mu**2*np.sin(2*np.pi*x) + delta)
		phi_ex = lambda x: alpha*np.sin(np.pi*(x+eta)/L) + gamma/3*np.sin(2*np.pi*x) + delta
		sigma_t = lambda x: 1/eps
		sigma_s = lambda x: 1/eps - eps 
		sigma_a = lambda x: sigma_t(x) - sigma_s(x) 
		source = lambda x, mu: .5*(mu*alpha*np.pi/L*np.cos(np.pi*(x+eta)/L) + beta*mu**2*(1-2*x) 
			+ gamma*mu**3*2*np.pi*np.cos(2*np.pi*x)) + sigma_t(x)*psi_ex(x,mu) - sigma_s(x)/2*phi_ex(x)

		sweeper = DirectSweeper(tfes, quad, sigma_t, sigma_s, source, psi_ex, False)
		sn = Sn(sweeper) 
		psi = TVector(tfes, quad)
		psi.Project(lambda x, mu: 1)

		qdf = QDFactors(tfes, quad, psi_ex) 
		h1xl2 = VEF(fes, vfes, sweeper, None, False)
		phi = h1xl2.SourceIteration(psi)
		sphi = sn.ComputeScalarFlux(psi)
		err.append(phi.L2Error(phi_ex, 2*p+2))
		lerr.append(phi.L2ProjError(phi_ex, 2*p+2))
		serr.append(sphi.L2Error(phi_ex, 2*p+2))

	ooa.append(np.log2(err[0]/err[1]))
	looa.append(np.log2(lerr[0]/lerr[1]))
	sooa.append(np.log2(serr[0]/serr[1]))

plt.plot(Eps, ooa, '-o', label=r'$\|\varphi - \phi_\mathrm{ex}\|$')
plt.plot(Eps, looa, '-o', label=r'$\|\varphi - \Pi \phi_\mathrm{ex}\|$')
plt.plot(Eps, sooa, '-o', label=r'$\|\phi - \phi_\mathrm{ex}\|$')
plt.xscale('log')
plt.xlabel(r'$\epsilon$')
plt.ylabel('Order of Convergence')
plt.legend()
plt.show()