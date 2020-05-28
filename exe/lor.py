#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans1d import * 
from OutputCycler import OutputCycler 

oc = OutputCycler()

def KappaMix(Ne, p):
	xe = np.linspace(0, 1, Ne+1)
	lob = LobattoBasis(p)
	leg = LegendreBasis(p-1) if p>1 else LegendreBasis(0) 
	h1 = H1Space(xe, lob)
	l2 = L2Space(xe, leg) 

	qorder = 2*p+1
	Mt = Assemble(h1, MassIntegrator, lambda x: -1, qorder)
	Ma = Assemble(l2, MassIntegrator, lambda x: .1, qorder)
	D = MixAssemble(l2, h1, MixDivIntegrator, 1, qorder) 
	f = AssembleRHS(l2, DomainIntegrator, lambda x: np.pi**2*np.sin(np.pi*x), qorder)

	M = sp.bmat([[Mt, D.transpose()], [D, Ma]]).tocsc()
	rhs = np.concatenate((np.zeros(h1.Nu), f))

	lob_low = LobattoBasis(1)
	leg_low = LegendreBasis(0) 
	h1_low = H1Space(h1.x, lob_low)
	l2_low = L2Space(h1.x, leg_low) 

	Mt_low = Assemble(h1_low, MassIntegrator, lambda x: -1, qorder)
	Ma_low = Assemble(l2_low, MassIntegrator, lambda x: .1, qorder)
	D_low = MixAssemble(l2_low, h1_low, MixDivIntegrator, 1, qorder)
	Mlow = sp.bmat([[Mt_low, D_low.transpose()], [D_low, Ma_low]]).tocsc()

	prec = spla.inv(Mlow)*M
	return np.linalg.cond(prec.todense()), np.linalg.cond(M.todense())

def KappaSecond(Ne, p):
	xe = np.linspace(0, 1, Ne+1)
	lob = LobattoBasis(p)
	h1 = H1Space(xe, lob)
	h1l = H1Space(h1.x, LobattoBasis(1))

	K = Assemble(h1, WeakPoissonIntegrator, lambda x: 1, 2*p+1).tolil()
	Kl = Assemble(h1l, WeakPoissonIntegrator, lambda x: 1, 2*p+1).tolil()

	bn = [0,-1]
	K[bn,:] = 0 
	K[bn,bn] = 1 
	Kl[bn,:] = 0 
	Kl[bn,bn] = 1 
	K = K.tocsc()
	Kl = Kl.tocsc()

	prec = spla.inv(Kl)*K
	return np.linalg.cond(prec.todense()), np.linalg.cond(K.todense())

Ne = np.array([2,4,8,12,24])**2
p = np.array([2,3,4])
for i in range(len(p)):
	mprec = np.zeros(len(Ne))
	morig = np.zeros(len(Ne))
	sprec = np.zeros(len(Ne))
	sorig = np.zeros(len(Ne))
	for j in range(len(Ne)):
		mprec[j], morig[j] = KappaMix(Ne[j], p[i]) 
		sprec[j], sorig[j] = KappaSecond(Ne[j], p[i]) 

	plt.figure()
	plt.semilogy(Ne, mprec, '-o', label='Mixed LOR Prec.')
	plt.semilogy(Ne, morig, '-o', label='Mixed HO')
	plt.semilogy(Ne, sprec, '-o', label='H1 LOR Prec.')
	plt.semilogy(Ne, sorig, '-o', label='H1 HO')	
	plt.xlabel('Number of Elements')
	plt.ylabel('Condition Number')
	plt.legend(prop={'size':14})
	if (oc.Good()):
		plt.savefig(oc.Get())
if not(oc.Good()):
	plt.show()