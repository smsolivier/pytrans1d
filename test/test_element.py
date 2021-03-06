#!/usr/bin/env python3

import numpy as np

from trans1d.fem import * 

p = 2
basis = LegendreBasis(p)
el = Element(basis, [0,2])

def test_inverse():
	x = .75 
	xi = el.InverseMap(x)
	print(xi) 
	assert(xi==-.25)

