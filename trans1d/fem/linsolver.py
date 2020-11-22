#!/usr/bin/env python3

import numpy as np
import warnings 
from .. import utils 

import scipy.sparse as sp
import scipy.sparse.linalg as spla 
import pyamg 
import time 
from termcolor import colored

class IterativeSolver:
	def __init__(self, itol, maxiter, LOUD=False):
		self.itol = itol
		self.maxiter = maxiter 
		self.LOUD = LOUD

		self.it = 0
		self.space = 3*' '
		self.norm = 0

	def Callback(self, r):
		self.it += 1 
		self.norm = np.linalg.norm(r)
		if (self.LOUD):
			print(self.space + 'i={:3}, norm={:.3e}'.format(self.it, self.norm))

	def Cleanup(self, info):
		if (info>0 or self.it==self.maxiter):
			warnings.warn('linear solver not converged. final tol={:.3e}, info={}'.format(self.norm, info), 
				utils.ToleranceWarning, stacklevel=2)
		if (info<0):
			raise RuntimeError('linear solver error. info={}'.format(info))

class BlockLDU(IterativeSolver):
	def __init__(self, itol, maxiter, inner=1, LOUD=False):
		IterativeSolver.__init__(self, itol, maxiter, LOUD)
		self.inner = inner 

	def Solve(self, A, Ainv, B, C, D, rhs):
		self.it = 0
		M = sp.bmat([[A,B], [C,D]]).tocsc()
		CAinv = C*Ainv 
		AinvB = Ainv*B 
		S = D - C*AinvB 
		amg = pyamg.ruge_stuben_solver(S.tocsr())

		def Prec(b):
			z1 = b[:A.shape[0]]
			z2 = b[A.shape[0]:] - CAinv*z1

			y1 = Ainv*z1
			y2 = amg.solve(z2, maxiter=self.inner)

			x2 = y2.copy()
			x1 = y1 - AinvB*x2 

			return np.concatenate((x1, x2))

		p2x2 = spla.LinearOperator(M.shape, Prec)
		x, info = spla.gmres(M, rhs, M=p2x2, atol=self.itol, tol=0,
			maxiter=self.maxiter, callback=self.Callback, callback_type='legacy', restart=None)
		self.Cleanup(info)

		return x 

class GaussSeidel(IterativeSolver):
	def __init__(self, itol, maxiter, LOUD=False):
		IterativeSolver.__init__(self, itol, maxiter, LOUD)

	def Solve(self, A, b):
		self.it = 0
		L = sp.tril(A,0).tocsr()
		U = sp.triu(A,1).tocsr()

		x = np.zeros(A.shape[0])
		for n in range(self.maxiter):
			x0 = x.copy()
			x = spla.spsolve_triangular(L, b - U*x0, lower=True)

			norm = np.linalg.norm(x - x0)
			if (norm < self.itol):
				break 

			self.Callback(norm)

		return x 

class SymGaussSeidel(IterativeSolver):
	def __init__(self, itol, maxiter, LOUD=False):
		IterativeSolver.__init__(self, itol, maxiter, LOUD)

	def Solve(self, A, b):
		self.it = 0 
		L1 = sp.tril(A,0).tocsr()
		U1 = sp.triu(A,1).tocsr()
		L2 = sp.tril(A,-1).tocsr()
		U2 = sp.triu(A,0).tocsr()

		x = np.zeros(A.shape[0])
		for n in range(self.maxiter):
			x0 = x.copy()
			x = spla.spsolve_triangular(L1, b - U1*x0, lower=True)
			x = spla.spsolve_triangular(U2, b - L2*x, lower=False)

			norm = np.linalg.norm(x - x0)
			if (norm<self.itol):
				break 

			self.Callback(norm)

		return x 

class Jacobi(IterativeSolver):
	def __init__(self, itol, maxiter, LOUD=False):
		IterativeSolver.__init__(self, itol, maxiter, LOUD)

	def Solve(self, A, b):
		self.it = 0
		D = A.diagonal()
		Aoff = A - sp.diags(D)

		x = np.zeros(A.shape[0])
		for n in range(self.maxiter):
			x0 = x.copy()
			x = (b - Aoff*x0)/D 

			norm = np.linalg.norm(x - x0)
			if (norm < self.itol):
				break 

			self.Callback(norm) 

		return x 

class BlockLDURelax(IterativeSolver):
	def __init__(self, itol, maxiter, relax, inner=1, LOUD=False):
		IterativeSolver.__init__(self, itol, maxiter, LOUD)
		self.inner = inner 
		self.relax = relax 
		self.relax.space = 2*self.space

	def Solve(self, A, Ainv, B, C, D, rhs):
		self.it = 0
		M = sp.bmat([[A,B], [C,D]]) 
		CAinv = C*Ainv 
		AinvB = Ainv*B 
		S = D - C*AinvB 
		amg = pyamg.ruge_stuben_solver(S.tocsr())

		def Prec(b):
			z1 = b[:A.shape[0]]
			z2 = b[A.shape[0]:] - CAinv*z1

			y1 = self.relax.Solve(A, z1)
			y2 = amg.solve(z2, maxiter=self.inner)

			x2 = y2.copy()
			x1 = y1 - AinvB*x2 

			return np.concatenate((x1, x2))

		p2x2 = spla.LinearOperator(M.shape, Prec)
		x, info = spla.gmres(M, rhs, M=p2x2, atol=self.itol, tol=0, maxiter=self.maxiter, callback=self.Callback)
		self.Cleanup(info)

		return x 

class BlockTri(IterativeSolver):
	def __init__(self, itol, maxiter, inner=1, LOUD=False):
		IterativeSolver.__init__(self, itol, maxiter, LOUD)
		self.inner = inner 

	def Solve(self, A, Ainv, B, C, D, rhs):
		self.it = 0 
		M = sp.bmat([[A,B], [C,D]])
		S = D - C*Ainv*B
		amg = pyamg.ruge_stuben_solver(S.tocsr())

		def Prec(b):
			x1 = Ainv*b[:A.shape[0]]
			x2 = amg.solve(b[A.shape[0]:] - C*x1, maxiter=self.inner)
			return np.concatenate((x1, x2))

		p = spla.LinearOperator(M.shape, Prec)
		x, info = spla.gmres(M, rhs, M=p, atol=self.itol, tol=0, maxiter=self.maxiter, callback=self.Callback)
		self.Cleanup(info)

		return x 

class BlockDiag(IterativeSolver):
	def __init__(self, itol, maxiter, inner=1, LOUD=False):
		IterativeSolver.__init__(self, itol, maxiter, LOUD)
		self.inner = inner 

	def Solve(self, A, Ainv, B, C, D, rhs):
		self.it = 0 
		M = sp.bmat([[A,B], [C,D]])
		S = D - C*Ainv*B
		amg = pyamg.ruge_stuben_solver(S.tocsr())

		def Prec(b):
			x1 = Ainv*b[:A.shape[0]]
			x2 = amg.solve(b[A.shape[0]:], maxiter=self.inner)
			return np.concatenate((x1, x2))

		p = spla.LinearOperator(M.shape, Prec)
		x, info = spla.gmres(M, rhs, M=p, atol=self.itol, tol=0, maxiter=self.maxiter, callback=self.Callback)
		self.Cleanup(info)

		return x 

class AMGSolver(IterativeSolver):
	def __init__(self, itol, maxiter, inner=1, smoother=None, LOUD=False):
		IterativeSolver.__init__(self, itol, maxiter, LOUD)
		self.inner = inner 
		self.smoother = smoother 		
		if (self.smoother!=None):
			self.smoother.space = 6*' '

	def Solve(self, A, Ahat, b):
		self.it = 0
		amg = pyamg.ruge_stuben_solver(Ahat.tocsr())
		# lu = spla.splu(Ahat.tocsc())
		def prec(x):
			y = amg.solve(x, maxiter=self.inner)
			# y = lu.solve(x) 
			if (self.smoother!=None):
				y += self.smoother.Solve(A, x)
				# y = self.smoother.Solve(A, y)

			return y 

		P = spla.LinearOperator(A.shape, prec)
		x, info = spla.gmres(A.tocsc(), b, M=P, callback=self.Callback, 
			callback_type='legacy', atol=self.itol, tol=0, maxiter=self.maxiter, restart=None)
		# x, info = pyamg.krylov.fgmres(A.tocsc(), b, M=P, tol=self.itol, maxiter=self.maxiter, restrt=A.shape[0], callback=self.Callback)
		# x, info = spla.lgmres(A.tocsc(), b, M=P, inner_m=10, callback=self.Callback, tol=self.itol, maxiter=self.maxiter)
		res = np.linalg.norm(A*x - b)
		if (res>self.itol):
			warnings.warn('residual={:.3e} is larger than tol={:.3e}'.format(res, self.itol), stacklevel=2)
		self.Cleanup(info)

		return x
