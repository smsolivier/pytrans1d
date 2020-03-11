#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import quadpy 

def GenLobatto(p):
	N = p+1 
	rule = quadpy.line_segment.gauss_lobatto(N)
	ip = np.around(rule.points, 14) 
	B, dB = SolveVandermonde(ip)
	return ip, B, dB

def GenLagrange(p):
	N = p+1 
	ip = np.linspace(-1,1,N)
	B, dB = SolveVandermonde(ip)
	return ip, B, dB 

def GenLegendre(p):
	N = p+1 
	ip, w = np.polynomial.legendre.leggauss(N)
	B, dB = SolveVandermonde(ip)
	return ip, B, dB 

def SolveVandermonde(ip):
	N = len(ip) 
	coef = np.zeros((N,N))
	A = np.zeros((N,N))
	for i in range(N):
		for j in range(N):
			A[i,j] = ip[i]**j 

	for k in range(N):
		b = np.zeros(N)
		b[k] = 1 
		coef[k,:] = np.linalg.solve(A, b) 

	B = np.zeros((N,N))
	dB = np.zeros((N-1,N))

	for i in range(N):
		B[:,i] = coef[i,::-1]
		dB[:,i] = np.polyder(B[:,i])

	return B, dB 

class BasisCollection:
	def __init__(self, p, genfun):
		self.p = p 
		self.N = self.p+1 
		self.ip, self.B, self.dB = genfun(p) 

class LegendreBasis(BasisCollection):
	def __init__(self, p):
		BasisCollection.__init__(self, p, GenLegendre)

class LobattoBasis(BasisCollection):
	def __init__(self, p):
		BasisCollection.__init__(self, p, GenLobatto)

class LagrangeBasis(BasisCollection):
	def __init__(self, p):
		BasisCollection.__init__(self, p, GenLagrange)

if __name__=='__main__':
	p = 2
	# basis = LagrangeBasis(p)
	# basis = LobattoBasis(p)
	basis = LegendreBasis(p)

	x = np.linspace(-1,1,100)
	plt.axhline(1, color='k', alpha=.3)
	plt.axhline(0, color='k', alpha=.3)
	for i in range(basis.N):
		plt.axvline(basis.ip[i], color='k', alpha=.3)
		plt.plot(x, np.polyval(basis.B[:,i], x))
	plt.show()