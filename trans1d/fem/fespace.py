#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys 

from .element import * 

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

	def __repr__(self):
		return 'e = {}, e\' = {}, bdr = {}, nor = {}'.format(
			self.el1.ElNo, self.el2.ElNo, self.boundary, self.nor)

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

	def InverseMap(self, x):
		# determine element where x lies using bisection 
		e = int(self.Ne/2)
		it = 0
		while not(x>=self.xe[e] and x<=self.xe[e+1]):
			it += 1 
			if (x>self.xe[e+1]):
				e = e + int((self.Ne-e)/2)
			else:
				e = e - int(e/2)

		# use inverse map on element to find reference point 
		return e, self.el[e].InverseMap(x)

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

		self.edge_dof = []
		self.int_dof = []
		self.nint = 0 
		self.nedge = 1
		self.sci = [(basis.N-2)*self.Ne]

		count = 0 
		int_count = 0 
		for e in range(self.Ne):
			self.el.append(Element(basis, [self.xe[e], self.xe[e+1]], e))
			dofs = np.arange(count, count+basis.N)
			self.x[dofs] = self.el[e].nodes 
			self.dofs.append(dofs) 
			count += basis.N - 1 
			self.edge_dof.append([e, e+1])
			self.int_dof.append(np.arange(int_count, int_count+basis.N-2).tolist())
			self.sci += np.arange(int_count, int_count+basis.N-2).tolist() + [(basis.N-2)*self.Ne+e+1]
			int_count += basis.N-2
			self.nint += len(dofs[1:-1])
			self.nedge += 1 

		assert(self.nint + self.nedge == self.Nu)

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

	def Evaluate(self, x):
		e, xi = self.space.InverseMap(x)
		return self.Interpolate(e, xi)

	def InterpolateGrad(self, e, xi):
		return self.space.el[e].InterpolateGrad(xi, self.GetDof(e))
		
	def Project(self, func):
		for e in range(self.space.Ne):
			el = self.space.el[e]
			nodes = el.nodes 
			vals = func(nodes) 
			self.SetDof(e, vals) 

	def ProjectGF(self, gf):
		for e in range(self.space.Ne):
			el = self.space.el[e] 
			nodes = el.nodes 
			vals = np.zeros(len(nodes))
			for n in range(len(nodes)):
				vals[n] = gf.Interpolate(e, nodes[n]) 
			self.SetDof(e, vals) 

	def L2Error(self, ex, qorder):
		from .quadrature import quadrature
		l2 = 0 
		# ip, w = quadrature.Get(qorder)
		for e in range(self.space.Ne):
			el = self.space.el[e]
			ip, w = quadrature.GetLumped(el)
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

	def LinfEdgeError(self, ex):
		E = abs(ex(self.space.xe[0]) - self.Interpolate(0, -1))

		for e in range(self.space.Ne):
			err = abs(ex(self.space.xe[e+1]) - self.Interpolate(e, 1))
			if (err>E):
				E = err 

		return E 

	def L2EdgeError(self, ex):
		err = (ex(self.space.xe[0]) - self.Interpolate(0,-1))**2

		for e in range(self.space.Ne):
			err += (ex(self.space.xe[e+1]) - self.Interpolate(e, 1))**2

		return np.sqrt(err)

	def L2Diff(self, gf, qorder):
		from .quadrature import quadrature
		l2 = 0 
		ip, w = quadrature.Get(qorder)
		for e in range(self.space.Ne):
			el = self.space.el[e] 
			for n in range(len(w)):
				diff = self.Interpolate(e, ip[n]) - gf.Interpolate(e, ip[n]) 
				l2 += diff**2 * w[n] * el.Jacobian(ip[n]) 

		return np.sqrt(l2) 

	def L2Norm(self, qorder):
		from .quadrature import quadrature
		l2 = 0 
		ip, w = quadrature.Get(qorder)
		for e in range(self.space.Ne):
			el = self.space.el[e] 
			for n in range(len(w)):
				l2 += self.Interpolate(e,ip[n])**2 * w[n] * el.Jacobian(ip[n]) 

		return np.sqrt(l2) 

	def LinfError(self, ex, qorder):
		from .quadrature import quadrature
		linf = 0 
		ip, w = quadrature.Get(qorder)
		for e in range(self.space.Ne):
			el = self.space.el[e]
			for n in range(len(w)):
				X = el.Transform(ip[n]) 
				err = abs(ex(X) - self.Interpolate(e, ip[n]))
				if (err>linf):
					linf = err 

		return err 

	def DerivL2Norm(self, qorder):
		from .quadrature import quadrature
		l2 = 0 
		ip, w = quadrature.Get(qorder)
		for e in range(self.space.Ne):
			el = self.space.el[e] 
			for n in range(len(w)):
				l2 += self.InterpolateGrad(e, ip[n])**2 * w[n] * el.Jacobian(ip[n]) 

		return np.sqrt(l2) 		

	def __rmul__(self, A):
		return A*self.data 

	def EvalSubEl(self, a=2, b=1):
		sub = a*(self.space.basis.p+1) + b
		Ne = self.space.Ne
		x = np.zeros(Ne*sub) 
		u = np.zeros(Ne*sub)  
		for e in range(Ne):
			xi = np.linspace(-1, 1, sub)
			for i in range(sub):
				u[sub*e + i] = self.Interpolate(e, xi[i]) 
				x[sub*e + i] = self.space.el[e].Transform(xi[i]) 

		return x, u

	def EvalDerivSubEl(self, a=2, b=1):
		sub = a*(self.space.basis.p+1) + b 
		Ne = self.space.Ne 
		x = np.zeros(Ne*sub)
		u = np.zeros(Ne*sub)
		for e in range(Ne):
			xi = np.linspace(-1,1, sub)
			for i in range(len(xi)):
				u[sub*e + i] = self.InterpolateGrad(e, xi[i])
				x[sub*e + i] = self.space.el[e].Transform(xi[i])

		return x, u

	def Continuity(self):
		l2 = 0 
		for e in range(1, self.space.Ne-1):
			left = self.Interpolate(e, 1)
			right = self.Interpolate(e+1, -1) 
			l2 += (left - right)**2 

		return np.sqrt(l2)
		