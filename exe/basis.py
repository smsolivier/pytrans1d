#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans1d.fem import basis, horner

b = basis.LagrangeBasis(2)
x = np.linspace(-1,1)
shape = np.zeros((len(x), b.N))

for i in range(len(x)):
	shape[i] = horner.PolyVal(b.B, x[i]) 

for i in range(b.N):
	plt.plot(x, shape[:,i])
plt.show()