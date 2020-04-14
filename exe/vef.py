#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from qd import * 

Ne = 10
p = 2
if (len(sys.argv)>1):
	Ne = int(sys.argv[1])
if (len(sys.argv)>2):
	p = int(sys.argv[2])
N = 8
quad = LegendreQuad(N)
h = 1
L = Ne*h
xe = np.linspace(0,L,Ne+1)
leg = LegendreBasis(p)
lob = LobattoBasis(p+1) 	
phi_space = L2Space(xe, leg)
J_space = H1Space(xe, lob) 
eps = 1e-1
sigma_t = lambda x: 1/eps 
sigma_s = lambda x: 1/eps - eps 
Q = lambda x, mu: eps
psi_in = lambda x, mu: 0
sweep = DirectSweeper(phi_space, quad, sigma_t, sigma_s, Q, psi_in)
sn = Sn(sweep) 
ltol = 1e-8
inner = 1
maxiter = 50
pp = False
gs = GaussSeidel(ltol, 2, True)
block = BlockLDU(ltol, maxiter, inner, False)
# block = BlockLDURelax(ltol, maxiter, gs, inner, False)
# block = BlockTri(ltol, maxiter, inner, False)
# block = BlockDiag(ltol, maxiter, inner, False)
vef = VEF(phi_space, J_space, sweep, block, pp)
psi = TVector(phi_space, quad)
psi.Project(lambda x, mu: 1)
phi = vef.SourceIteration(psi)

# amg = AMGSolver(ltol, maxiter, inner, False)
# vefh = VEFH2(phi_space, J_space, sweep, None, False)
# psi.Project(lambda x, mu: 1)
# phi = vefh.SourceIteration(psi) 