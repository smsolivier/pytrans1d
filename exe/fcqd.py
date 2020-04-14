#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from qd import * 

import sys 
Ne = 10 
p = 1
if (len(sys.argv)>1):
	Ne = int(sys.argv[1])
if (len(sys.argv)>2):
	p = int(sys.argv[2])
N = 16
quad = LegendreQuad(N)
xe = np.linspace(0,1, Ne+1)
leg = LegendreBasis(p)
space = L2Space(xe, leg)

eps = 1e-2
sigma_t = lambda x: 1/eps 
sigma_s = lambda x: 1/eps - eps 

Q = lambda x, mu: eps
psi_in = lambda x, mu: 0
sweep = DirectSweeper(space, quad, sigma_t, sigma_s, Q, psi_in)
psi = TVector(space, quad)
qd = QD(space, space, sweep)
phi = qd.SourceIteration(psi, tol=1e-12) 
phi_sn = qd.ComputeScalarFlux(psi) 

# p1sa = P1SA(sweep)
# psi.Project(lambda x, mu: 0)
# phi_p1 = p1sa.SourceIteration(psi, tol=1e-12)

print('diff = {:.3e}'.format(phi.L2Diff(phi_sn, 2*p+1)))

# plt.figure()
# plt.plot(space.x, qd.qdf.P.data/qd.qdf.phi.data, '-o')

plt.figure()
plt.plot(phi.space.x, phi.data, '-o')
# plt.plot(phi_p1.space.x, phi_p1.data, '-o')
plt.show()
