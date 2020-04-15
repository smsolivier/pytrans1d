#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans1d import * 

Ne = 10 
p = 1 
N = 8
quad = LegendreQuad(N)
xe = np.linspace(0,1,Ne+1)
leg = LegendreBasis(p) 
space = L2Space(xe, leg) 

eps = 1e-4
sigma_t = lambda x: 1/eps 
sigma_s = lambda x: 1/eps - eps 
Q = lambda x, mu: eps 
psi_in = lambda x, mu: 0 

sweep = DirectSweeper(space, quad, sigma_t, sigma_s, Q, psi_in)
p1sa = P1SA(sweep) 
psi = TVector(space, quad) 
phi = p1sa.SourceIteration(psi)

plt.plot(space.x, phi.data, '-o')
plt.show()