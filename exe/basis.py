#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans1d.fem import basis, horner

b = basis.LagrangeBasis(2)
x = np.linspace(-1,1, 1000)
shape = horner.PolyVal2(b.B, x)

for i in range(b.N):
	plt.plot(x, shape[:,i])
plt.show()