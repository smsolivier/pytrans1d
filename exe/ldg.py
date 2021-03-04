#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from trans1d import * 

def Error(Ne, p):
	n = int(Ne/3)
	# xe = np.linspace(0,1,Ne+1)
	xe = np.unique(np.concatenate((np.linspace(0,1/3,n, endpoint=False), 
		np.linspace(1/3, 2/3,n, endpoint=False), np.linspace(2/3,1,n+1))))
	sfes = L2Space(xe, LobattoBasis(p))
	vfes = L2Space(xe, LobattoBasis(p))
	cfes = H1Space(xe, LobattoBasis(p+1))
	tfes = L2Space(xe, LobattoBasis(p))
	quad = LegendreQuad(4)

	alpha = 1 
	beta = 0
	gamma = 0
	delta = 1/2
	eta = .1
	L = 1 + 2*eta
	eps = 1e-2
	psi_ex = lambda x, mu: .5*(alpha*np.sin(np.pi*(x+eta)/L) 
		+ beta*mu*x*(1-x) + gamma*mu**2*np.sin(2*np.pi*x) + delta)
	# psi_ex = lambda x, mu: 0
	phi_ex = lambda x: alpha*np.sin(np.pi*(x+eta)/L) + gamma/3*np.sin(2*np.pi*x) + delta
	sigma_t = lambda x: eps if x>1/3 and x<2/3 else 1/eps*(1 + .1*np.sin(np.pi*x)) 
	sigma_s = lambda x: eps if x>1/3 and x<2/3 else eps
	# sigma_t = lambda x: 1/eps
	# sigma_s = lambda x: 1/eps - eps 
	sigma_a = lambda x: sigma_t(x) - sigma_s(x) 
	source = lambda x, mu: .5*(mu*alpha*np.pi/L*np.cos(np.pi*(x+eta)/L) + beta*mu**2*(1-2*x) 
		+ gamma*mu**3*2*np.pi*np.cos(2*np.pi*x)) + sigma_t(x)*psi_ex(x,mu) - sigma_s(x)/2*phi_ex(x)
	# source = lambda x, mu: 1

	qdf = QDFactors(tfes, quad, psi_ex)
	ldg = LDGVEF(sfes, vfes, qdf, sigma_t, sigma_s, source)
	lldg = LiftedLDGVEF(sfes, qdf, sigma_t, sigma_s, source, False)
	lldgs = LiftedLDGVEF(sfes, qdf, sigma_t, sigma_s, source, True)
	cldg = ConsistentLDGVEF(sfes, vfes, qdf, sigma_t, sigma_s, source)
	sip = SIPVEF(sfes, qdf, sigma_t, sigma_s, source)
	br2 = BR2VEF(sfes, qdf, sigma_t, sigma_s, source)
	sweeper = DirectSweeper(tfes, quad, sigma_t, sigma_s, source, psi_ex, False)
	h1xl2 = VEF(sfes, cfes, sweeper, None, False)
	sn = Sn(sweeper)

	# methods = [ldg, sip, br2, lldg]
	methods = [sip]
	names = ['SIP', 'CDG', 'LDG']
	psi = TVector(tfes, quad)
	# psi.Project(psi_ex)
	err = []
	integral = []
	energy = [] 
	bal = [] 
	for i in range(len(methods)):
		# phi = methods[i].Mult(psi)
		psi.Project(lambda x, mu: 1)
		npi = NPI(sweeper, methods[i], sfes, psi)
		phi = GridFunction(sfes)
		fpi = FixedPointIteration(npi.F, 1e-7, 100, True)
		phi.data = fpi.Solve(np.ones(sfes.Nu))
		err.append(phi.L2Error(phi_ex, 2*p+2))
		integral.append(phi.Integrate(2*p+1))
		bal.append(abs(np.ones(len(phi.data))@(methods[i].K*phi - methods[i].Q)))
		energy.append(np.dot(phi.data, methods[i].K*phi))

		# jump = 0 
		# for e in range(1,sfes.Ne-1):
		# 	jump += (phi.Interpolate(e-1,1)*qdf.EvalFactor(sfes.el[e-1],1) 
		# 		- phi.Interpolate(e,-1)*qdf.EvalFactor(sfes.el[e],-1))**2

		# print('jump = {:.3e}'.format(np.sqrt(jump)))
		# x,subel = phi.EvalSubEl()
		# plt.plot(x,subel)
		# x,subel = qdf.P.EvalSubEl()
		# plt.plot(x,subel, label='P')
		# x,subel = qdf.phi.EvalSubEl()
		# plt.plot(x,subel, label='phi')
		# xi = np.linspace(-1,1,2*p+2)
		# x = [] 
		# subel = [] 
		# for e in range(sfes.Ne):
		# 	for n in range(len(xi)):
		# 		subel.append(qdf.EvalFactor(sfes.el[e], xi[n]))
		# 		x.append(sfes.el[e].Transform(xi[n]))

		# plt.plot(x,subel, label='E')
	print(err)
	print(integral)
	print(energy) 
	print(bal)
	# plt.legend()
	# plt.show()

	# psi.Project(lambda x, mu: 1)
	# npi = NPI(sweeper, br2, sfes, psi)
	# phi = GridFunction(sfes)
	# fpi = FixedPointIteration(npi.F, 1e-10, 25, True)
	# phi.data = fpi.Solve(np.ones(sfes.Nu))
	# err = phi.L2Error(phi_ex, 2*p+2)
	# print('err = {:.3e}'.format(err))
	# x,subel = phi.EvalSubEl(20)
	# plt.plot(x,subel)
	# plt.plot(x,phi_ex(x), '--')
	# plt.show()
	return err

Ne = 9
p = 2
E1 = Error(Ne, p)
E2 = Error(2*Ne, p)
for e in range(len(E1)):
	print('p = {:.3f} ({:.3e}, {:.3e})'.format(np.log2(E1[e]/E2[e]), E1[e], E2[e]))

# p = 2
# Ne = 3**np.arange(1,6)
# err = np.zeros((1,len(Ne)))
# for n in range(len(Ne)):
# 	err[:,n] = Error(Ne[n], p)

# for i in range(err.shape[0]):
# 	plt.loglog(1/Ne, err[i,:], '-o')
# plt.legend()
# plt.show()

# Error(36, 2)

# plt.loglog(1/Ne, err[0,:], '-o', label='LDG')
# # plt.loglog(1/Ne, err[1,:], '-o', label='IP')
# # plt.loglog(1/Ne, err[2,:], '-o', label='BR2')
# plt.loglog(1/Ne, err[-1,:], '-o', label='CDG')
# plt.loglog(1/Ne, err[0,-1]*(Ne[-1]/Ne)**3, 'k--')
# plt.loglog(1/Ne, err[1,-1]*(Ne[-1]/Ne)**1.5, 'k--')
# plt.legend()
# plt.show()

