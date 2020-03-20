#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import newton_cotes 
import sys 
from basis import * 

class Quadrature: 
	def __init__(self):
		self.leg_ip = []
		self.leg_w = [] 
		self.lob_ip = [] 
		self.lob_w = [] 
		self.nc_ip = [] 
		self.nc_w = [] 

		pmax = 50
		for p in range(1, pmax):
			ip, w = np.polynomial.legendre.leggauss(p)
			self.leg_ip.append(ip)
			self.leg_w.append(w) 

		for p in range(2, pmax):
			rule = quadpy.line_segment.gauss_lobatto(p) 
			self.lob_ip.append(np.around(rule.points, 14))
			self.lob_w.append(rule.weights) 

			ip = np.linspace(-1,1, p)
			w = newton_cotes(p-1, 1)[0]
			w *= 2/np.sum(w) 
			self.nc_ip.append(ip)
			self.nc_w.append(w) 

	def Get(self, p):
		return self.leg_ip[p-1], self.leg_w[p-1]

	def GetLumped(self, el):
		basis = el.basis
		if (isinstance(basis, LegendreBasis)):
			return self.leg_ip[basis.p], self.leg_w[basis.p]
		elif (isinstance(basis, LobattoBasis)):
			return self.lob_ip[basis.p-1], self.lob_w[basis.p-1]
		elif (isinstance(basis, LagrangeBasis)):
			return self.nc_ip[basis.p-1], self.nc_w[basis.p-1]
		else:
			print('basis not defined')
			sys.exit()

quadrature = Quadrature()

if __name__=='__main__':
	p = 1
	basis = LagrangeBasis(p)
	# basis = LobattoBasis(p)
	ip, w = quadrature.GetLumped(basis) 
	print(ip, w) 