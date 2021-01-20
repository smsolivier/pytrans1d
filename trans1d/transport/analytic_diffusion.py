#!/usr/bin/env python3

import numpy as np

def AnalyticDiffusion(sigt, siga, Q, gamma, bc):
	N = len(sigt)
	A = np.zeros((2*N, 2*N))
	rhs = np.zeros(2*N) 
	a = gamma[0] 
	b = gamma[-1]

	D = 1/3/sigt 
	L = np.sqrt(D/siga) 
	c = lambda x, i: np.sinh((x)/L[i]) 
	dc = lambda x, i: D[i]/L[i]*np.cosh((x)/L[i]) 
	d = lambda x, i: np.cosh((x)/L[i]) 
	dd = lambda x, i: D[i]/L[i]*np.sinh((x)/L[i]) 
	qs = lambda i: Q[i]/siga[i] 

	if (bc[0]['type']=='dirichlet'):
		A[0,0] = c(a,0)
		A[0,1] = d(a,0)
		rhs[0] = bc[0]['value'] - qs(0)
	elif (bc[0]['type']=='neumann'):
		A[0,0] = dc(a,0)
		A[0,1] = dd(a,0)
		rhs[0] = -bc[0]['value']
	else:
		raise RuntimeError('bc type not correct')

	for i in range(1,N):
		j = 2*i-1
		A[j,j-1] = c(gamma[i],i-1) 
		A[j,j] = d(gamma[i],i-1)
		A[j,j+1] = -c(gamma[i],i)
		A[j,j+2] = -d(gamma[i],i)
		rhs[j] = -qs(i-1) + qs(i)

		A[j+1,j-1] = dc(gamma[i],i-1)
		A[j+1,j] = dd(gamma[i],i-1)
		A[j+1,j+1] = -dc(gamma[i],i)
		A[j+1,j+2] = -dd(gamma[i],i)
		rhs[j+1] = 0 

	if (bc[1]['type']=='dirichlet'):
		A[-1,-2] = c(b, -1)
		A[-1,-1] = d(b, -1)
		rhs[-1] = bc[1]['value'] - qs(-1)
	elif (bc[1]['type']=='neumann'):
		A[-1,-2] = dc(b,-1)
		A[-1,-1] = dd(b,-1)
		rhs[-1] = bc[1]['value']
	else:
		raise RuntimeError('bc type not correct')

	print(A)
	coef = np.linalg.solve(A, rhs) 
	phis = [] 
	Js = [] 
	for i in range(N):
		phis.append(lambda x, i=i: coef[2*i]*c(x,i) + coef[2*i+1]*d(x,i) + qs(i))
		Js.append(lambda x, i=i: -coef[2*i]*dc(x,i) - coef[2*i+1]*dd(x,i))
	return lambda x: np.piecewise(x, [(x>=gamma[i])*(x<=gamma[i+1]) for i in range(N)], phis), \
		lambda x: np.piecewise(x, [(x>=gamma[i])*(x<=gamma[i+1]) for i in range(N)], Js)