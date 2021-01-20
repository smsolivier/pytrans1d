#!/usr/bin/env python3

import numpy as np
from scipy.integrate import newton_cotes 
import sys 
from .basis import * 
import math 

class Quadrature: 
	def __init__(self):
		self.leg_ip = []
		self.leg_w = [] 
		self.lob_ip = [] 
		self.lob_w = [] 
		self.nc_ip = [] 
		self.nc_w = [] 

		self.pmax = 50
		for p in range(1, self.pmax):
			ip, w = np.polynomial.legendre.leggauss(p)
			self.leg_ip.append(ip)
			self.leg_w.append(w) 

		for p in range(2, self.pmax):
			rule = quadpy.c1.gauss_lobatto(p) 
			# self.lob_ip.append(np.around(rule.points, 14))
			self.lob_ip.append(rule.points) 
			self.lob_w.append(rule.weights) 

			ip = np.linspace(-1,1, p)
			w = newton_cotes(p-1, 1)[0]
			w *= 2/np.sum(w) 
			self.nc_ip.append(ip)
			self.nc_w.append(w) 

	def Get(self, p):
		idx = math.ceil((p+1)/2)-1
		assert(idx<self.pmax)
		return self.leg_ip[idx], self.leg_w[idx]

	def GetLumped(self, el):
		basis = el.basis
		if (isinstance(basis, LegendreBasis)):
			return basis.ip, self.leg_w[basis.p]
		elif (isinstance(basis, LobattoBasis)):
			return basis.ip, self.lob_w[basis.p-1]
		elif (isinstance(basis, LagrangeBasis)):
			return basis.ip, self.nc_w[basis.p-1]
		else:
			print('basis not defined')
			sys.exit()

quadrature = Quadrature()
