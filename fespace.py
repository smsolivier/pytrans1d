#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys 

from element import * 

class FaceTrans:
	def __init__(self, els, edge):
		self.els = els 
		self.edge = edge 
		self.el1 = els[0] 
		self.boundary = False
		if (len(els)>1):
			self.el2 = els[1] 
		else:
			self.boundary = True 
			self.el2 = els[0]
		self.boundary = False 
		if (edge==1):
			self.ip = [1, -1] 
		elif (edge==-1):
			self.ip = [-1, 1]
		else:
			print('edge must be -1 or 1')
			sys.exit()
		self.nor = edge 

	def IPTrans(self, e):
		return self.ip[e] 

class FESpace:
	def __init__(self, xe, basis):
		self.xe = xe 
		self.Ne = len(xe)-1 
		self.el = [] 
		self.dofs = [] 
		self.basis = basis 

	def BuildFaces(self):
		self.iface = [] 
		self.bface = [] 
		for e in range(0, self.Ne-1):
			self.iface.append(FaceTrans([self.el[e], self.el[e+1]], 1))

		self.bface.append(FaceTrans([self.el[0]], -1))
		self.bface.append(FaceTrans([self.el[-1]], 1))

class L2Space(FESpace):
	def __init__(self, xe, basis):
		FESpace.__init__(self, xe, basis) 
		self.Nu = self.Ne * basis.N 
		self.x = np.zeros(self.Nu)

		count = 0 
		for e in range(self.Ne):
			self.el.append(Element(basis, [self.xe[e], self.xe[e+1]], e))
			dofs = np.arange(basis.N) + count 
			self.x[dofs] = self.el[e].nodes 
			count += basis.N 
			self.dofs.append(dofs) 

		self.BuildFaces()

class H1Space(FESpace):
	def __init__(self, xe, basis):
		FESpace.__init__(self, xe, basis) 
		assert(isinstance(basis, LagrangeBasis) or isinstance(basis, LobattoBasis))
		self.Nu = self.Ne*basis.p + 1
		self.x = np.zeros(self.Nu) 

		count = 0 
		for e in range(self.Ne):
			self.el.append(Element(basis, [self.xe[e], self.xe[e+1]], e))
			dofs = np.arange(count, count+basis.N)
			self.x[dofs] = self.el[e].nodes 
			self.dofs.append(dofs) 
			count += basis.N - 1 

		self.BuildFaces()

class GridFunction:
	def __init__(self, space):
		self.space = space 
		self.data = np.zeros(self.space.Nu) 

	def GetDof(self, e):
		return self.data[self.space.dofs[e]]

	def SetDof(self, e, vals):
		self.data[self.space.dofs[e]] = vals 

	def Interpolate(self, e, xi):
		return self.space.el[e].Interpolate(xi, self.GetDof(e))

	def InterpolateGrad(self, e, xi):
		return self.space.el[e].InterpolateGrad(xi, self.GetDof(e))
		
	def Project(self, func):
		for e in range(self.space.Ne):
			el = self.space.el[e]
			nodes = el.nodes 
			vals = func(nodes) 
			self.SetDof(e, vals) 

	def L2Error(self, ex, qorder):
		from quadrature import quadrature
		l2 = 0 
		ip, w = quadrature.Get(qorder)
		for e in range(self.space.Ne):
			el = self.space.el[e]
			for n in range(len(w)):
				X = el.Transform(ip[n]) 
				exact = ex(X) 
				fem = self.Interpolate(e, ip[n]) 
				diff = exact - fem 
				l2 += diff**2 * w[n] * el.Jacobian(ip[n]) 

		return np.sqrt(l2) 

	def L2ProjError(self, ex, qorder):
		gf = GridFunction(self.space)
		gf.Project(ex)
		return self.L2Diff(gf, qorder) 

	def L2Diff(self, gf, qorder):
		from quadrature import quadrature
		l2 = 0 
		ip, w = quadrature.Get(qorder)
		for e in range(self.space.Ne):
			el = self.space.el[e] 
			for n in range(len(w)):
				diff = self.Interpolate(e, ip[n]) - gf.Interpolate(e, ip[n]) 
				l2 += diff**2 * w[n] * el.Jacobian(ip[n]) 

		return np.sqrt(l2) 

	def __rmul__(self, A):
		return A*self.data 

if __name__=='__main__':
	Ne = 2
	p = 5	
	xe = np.linspace(-1, 1, Ne+1)
	# basis = LegendreBasis(p) 
	basis = LobattoBasis(p)
	# space = L2Space(xe, basis) 
	space = H1Space(xe, basis) 
	phi = GridFunction(space)
	phi.L2Error(lambda x: 1, 2)
	phi.Project(lambda x: x**2) 
	plt.plot(space.x, phi.data, '-o')
	plt.show()