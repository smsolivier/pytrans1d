#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans1d import * 

def Cond(Ne, p):
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

Ne = np.array([9, 36, 144])
p = 3 
prec = np.zeros(len(Ne))
orig = np.zeros(len(Ne))
for i in range(len(Ne)):
	prec[i], orig[i] = Cond(Ne[i], p) 

plt.semilogy(Ne, orig, '-o', label='Mixed')
plt.semilogy(Ne, prec, '-o', label='LOR Prec. Mixed')
plt.xlabel('Number of Elements')
plt.ylabel('Condition Number')
plt.legend()
plt.show()