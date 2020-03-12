#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from qdf import * 
from sn import * 
from linsolver import * 
import pyamg 

class AbstractVEF(Sn):
	def __init__(self, phi_space, J_space, sweeper, lin_solver=None):
		Sn.__init__(self, sweeper) 
		self.phi_space = phi_space 
		self.J_space = J_space 
		self.lin_solver = lin_solver

		p = J_space.basis.p
		self.p = p 
		sigma_a = lambda x: sweeper.sigma_t(x) - sweeper.sigma_s(x)
		self.Mt = Assemble(self.J_space, MassIntegrator, sweeper.sigma_t, 2*p+1)
		self.Mtl = Assemble(self.J_space, MassIntegratorLumped, sweeper.sigma_t, 2*p+1)
		self.Ma = Assemble(self.phi_space, MassIntegrator, sigma_a, 2*p+1)
		self.D = MixAssemble(self.phi_space, self.J_space, MixDivIntegrator, 1, 2*p+1) 

		self.Q0 = np.zeros(phi_space.Nu)
		self.Q1 = np.zeros(J_space.Nu)
		for a in range(self.N):
			mu = self.mu[a]
			self.Q0 += AssembleRHS(phi_space, DomainIntegrator, lambda x: sweeper.Q(x,mu), 2*p+1)*self.w[a]
			self.Q1 += AssembleRHS(J_space, DomainIntegrator, lambda x: sweeper.Q(x,mu), 2*p+1)*mu*self.w[a]

		self.qdf = QDFactors(self.space, self.N, self.sweeper.psi_in) 

	def SourceIteration(self, psi, niter=50, tol=1e-6):
		phi = GridFunction(self.phi_space)
		phi_old = GridFunction(self.phi_space)
		linits = 0
		for n in range(niter):
			start = time.time() 
			phi_old.data = phi.data.copy() 
			self.sweeper.Sweep(psi, phi) 
			phi = self.Mult(psi)
			norm = phi.L2Diff(phi_old, 2*self.p+1)
			if (self.lin_solver!=None):
				linits += self.lin_solver.it 

			if (self.LOUD):
				el = time.time() - start 
				if (self.lin_solver!=None):
					print('i={:3}, norm={:.3e}, {:.2f} s/iter, {} linear iters'.format(
						n+1, norm, el, self.lin_solver.it))
				else:
					print('i={:3}, norm={:.3e}, {:.2f} s/iter'.format(
						n+1, norm, el))

			if (norm < tol):
				break 

		if (self.LOUD and self.lin_solver!=None):
			print('avg linear iters = {:.2f}'.format(linits/(n+1)))

		if (norm > tol):
			print(colored('WARNING not converged! Final tol = {:.3e}'.format(norm), 'red'))

		return phi 

class VEF(AbstractVEF): 
	def __init__(self, phi_space, J_space, sweeper, lin_solver):
		AbstractVEF.__init__(self, phi_space, J_space, sweeper, lin_solver)
		self.lin_solver = lin_solver 

	def Mult(self, psi):
		self.qdf.Compute(psi) 
		B = BdrFaceAssemble(self.J_space, MLBdrIntegrator, self.qdf)
		G = MixAssemble(self.J_space, self.phi_space, MixWeakEddDivIntegrator, self.qdf, 2*self.p+1) 
		qin = BdrFaceAssembleRHS(self.J_space, VEFInflowIntegrator, self.qdf) 

		Mt = self.Mt + B 
		Q1 = self.Q1 + qin 
		rhs = np.concatenate((Q1, self.Q0))

		if (self.lin_solver==None):
			A = sp.bmat([[Mt, G], [self.D, self.Ma]]).tocsc()
			x = spla.spsolve(A, rhs) 

		else:
			Ainv = spla.inv(self.Mtl + B)
			x = self.lin_solver.Solve(Mt, Ainv, G, self.D, self.Ma, rhs)

		phi = GridFunction(self.phi_space)
		phi.data = x[self.J_space.Nu:]

		return phi 

	def LDU(self, A, Ainv, G, rhs):
		M = sp.bmat([[A, G], [self.D, self.Ma]])

		DAinv = self.D*Ainv 
		AinvG = Ainv*G 
		S = self.Ma - self.D*Ainv*G
		amg = pyamg.ruge_stuben_solver(S.tocsr())
		# lu = spla.splu(S)
		def Prec(b):
			z1 = b[:A.shape[0]]
			z2 = b[A.shape[0]:] - DAinv*z1

			y1 = Ainv*z1
			y2 = amg.solve(z2, maxiter=1)
			# y2 = lu.solve(z2)

			x2 = y2.copy()
			x1 = y1 - AinvG*x2 

			return np.concatenate((x1, x2))

		p = spla.LinearOperator(M.shape, Prec)
		self.linear_its = 0
		def cb(r):
			self.linear_its += 1
			norm = np.linalg.norm(r)
			# print('   i={}, norm={:.3e}'.format(gmres, norm))

		x, info = spla.gmres(M, rhs, M=p, tol=self.lin_tol, maxiter=1000, callback=cb)

		phi = GridFunction(self.phi_space)
		phi.data = x[self.J_space.Nu:]

		return phi 

class VEFH(AbstractVEF):
	def __init__(self, phi_space, J_space, sweeper, lin_solver):
		if (isinstance(J_space, H1Space)):
			J_space = L2Space(J_space.xe, J_space.basis)
		AbstractVEF.__init__(self, phi_space, J_space, sweeper, lin_solver) 
		basis = LagrangeBasis(1)
		self.m_space = H1Space(self.phi_space.xe, basis) 
		self.C2 = MixFaceAssemble(self.m_space, self.J_space, ConstraintIntegrator, 1) 

	def Mult(self, psi):
		self.qdf.Compute(psi)
		B = BdrFaceAssemble(self.J_space, MLBdrIntegrator, self.qdf)
		G = MixAssemble(self.J_space, self.phi_space, MixWeakEddDivIntegrator, self.qdf, 2*self.p+1)
		qin = BdrFaceAssembleRHS(self.J_space, VEFInflowIntegrator, self.qdf)
		C1 = MixFaceAssemble(self.J_space, self.m_space, EddConstraintIntegrator, self.qdf)

		Mt = self.Mt + B
		Q1 = self.Q1 + qin 

		Mtinv = spla.inv(Mt)
		S = self.Ma - self.D*Mtinv*G 
		Sinv = spla.inv(S)

		W = Mtinv + Mtinv*G*Sinv*self.D*Mtinv 
		X = -Mtinv*G*Sinv 
		Y = -Sinv*self.D*Mtinv 

		R = self.C2*W*C1 
		rhs = self.C2*W*Q1 + self.C2*X*self.Q0

		bdr = W*C1 
		bdr = bdr.tolil()
		bdr[0,:] *= -1 
		bdr[0,0] += self.qdf.EvalG(self.phi_space.bface[0])
		bdr[-1,-1] += self.qdf.EvalG(self.phi_space.bface[-1])
		bdr_rhs = W*Q1 + X*self.Q0
		bdr_rhs[0] *= -1 
		bdr_rhs[0] -= 2*self.qdf.EvalJinBdr(self.phi_space.bface[0])
		bdr_rhs[-1] -= 2*self.qdf.EvalJinBdr(self.phi_space.bface[-1])

		R = R.tolil()
		R[0,:] = bdr[0,:]
		R[-1,:] = bdr[-1,:]
		rhs[0] = bdr_rhs[0]
		rhs[-1] = bdr_rhs[-1]

		lam = GridFunction(self.m_space)
		if (self.lin_solver==None):
			lam.data = spla.spsolve(R.tocsc(), rhs) 
		else:
			lam.data = self.lin_solver.Solve(R.tocsr(), rhs)

		phi = GridFunction(self.phi_space)
		phi.data = Y*Q1 - Y*C1*lam + Sinv*self.Q0

		return phi 

if __name__=='__main__':
	Ne = 10
	p = 2
	if (len(sys.argv)>1):
		Ne = int(sys.argv[1])
	if (len(sys.argv)>2):
		p = int(sys.argv[2])
	N = 8
	xe = np.linspace(0,1,Ne+1)
	leg = LegendreBasis(p)
	lob = LobattoBasis(p+1) 	
	phi_space = L2Space(xe, leg)
	J_space = H1Space(xe, lob) 
	eps = 1e-1
	sigma_t = lambda x: 1/eps 
	sigma_s = lambda x: 1/eps - eps 
	Q = lambda x, mu: eps
	psi_in = lambda x, mu: 0
	sweep = DirectSweeper(phi_space, N, sigma_t, sigma_s, Q, psi_in)
	sn = Sn(sweep) 
	ltol = 1e-8
	inner = 1
	maxiter = 50
	gs = GaussSeidel(ltol, 2, True)
	# block = BlockLDU(ltol, maxiter, inner, True)
	# block = BlockLDURelax(ltol, maxiter, gs, inner, False)
	block = BlockTri(ltol, maxiter, inner, False)
	# block = BlockDiag(ltol, maxiter, inner, False)
	vef = VEF(phi_space, J_space, sweep, block)
	amg = AMGSolver(ltol, maxiter, inner, False)
	vefh = VEFH(phi_space, J_space, sweep, amg)
	psi = TVector(phi_space, N)
	psi.Project(lambda x, mu: 1)
	phi = vef.SourceIteration(psi)
	psi.Project(lambda x, mu: 1)
	phi = vefh.SourceIteration(psi) 