#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from fem import * 

Ne = 10 
p = 1 
xe = np.linspace(0,1,Ne+1)
lob = LobattoBasis(p)
space = H1Space(xe, lob)
K = Assemble(space, WeakPoissonIntegrator, lambda x: 1, 2*p+1).tolil()
K[0,:] = 0 
K[0,0] = 1 
K[-1,:] = 0 
K[-1,-1] = 1 
b = AssembleRHS(space, DomainIntegrator, lambda x: 1, 2*p+1)
b[0] = 0 
b[-1] = 0 

gs = GaussSeidel(1e-10, 1000, True)
x = gs.Solve(K, b)

jac = Jacobi(1e-10, 1000, True)
x = jac.Solve(K,b) 

plt.plot(space.x, x, '-o')
plt.show()
