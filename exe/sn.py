#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans1d import * 

Ne = 10
p = 4
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
sweep = DirectSweeper(space, quad, sigma_t, sigma_s, Q, psi_in)
sn = Sn(sweep) 
phi = sn.SourceIteration(psi)
phi_ex = lambda x: np.sin(np.pi*x) 
print('err = {:.3e}'.format(phi.L2Error(phi_ex, 2*p+1)))

plt.plot(space.x, phi.data, '-o')
xex = np.linspace(0,1,100)
plt.plot(xex, phi_ex(xex), '--')
plt.show()