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
		self.sigma_a = lambda x: sweeper.sigma_t(x) - sweeper.sigma_s(x)
		self.Mt = Assemble(self.J_space, MassIntegrator, sweeper.sigma_t, 2*p+1)
		self.Mtl = Assemble(self.J_space, MassIntegratorLumped, sweeper.sigma_t, 2*p+1)
		self.Ma = Assemble(self.phi_space, MassIntegrator, self.sigma_a, 2*p+1)
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
	def __init__(self, phi_space, J_space, sweeper, lin_solver=None, pp=True):
		if (pp):
			self.low_space = phi_space 
			AbstractVEF.__init__(self, self.low_space, J_space, sweeper, lin_solver)
			new_basis = type(phi_space.basis)(phi_space.basis.p+1)
			self.phi_space = type(phi_space)(phi_space.xe, new_basis) 
		else:
			AbstractVEF.__init__(self, phi_space, J_space, sweeper, lin_solver)
			self.low_space = phi_space 
		self.lin_solver = lin_solver 
		self.pp = pp 

	def Mult(self, psi):
		self.qdf.Compute(psi) 
		B = BdrFaceAssemble(self.J_space, MLBdrIntegrator, self.qdf)
		G = MixAssemble(self.J_space, self.low_space, MixWeakEddDivIntegrator, self.qdf, 2*self.p+1) 
		qin = BdrFaceAssembleRHS(self.J_space, VEFInflowIntegrator, self.qdf) 

		Mt = self.Mt + B 
		Q1 = self.Q1 + qin 
		rhs = np.concatenate((Q1, self.Q0))

		if (self.lin_solver==None):
			A = sp.bmat([[Mt, G], [self.D, self.Ma]]).tocsc()
			x = spla.spsolve(A, rhs) 

		else:
			Ainv = sp.diags(1/(self.Mtl + B).diagonal())
			x = self.lin_solver.Solve(Mt, Ainv, G, self.D, self.Ma, rhs)

		phi = GridFunction(self.low_space)
		phi.data = x[self.J_space.Nu:]
		J = GridFunction(self.J_space)
		J.data = x[:self.J_space.Nu] 

		if (self.pp):
			phi_star = GridFunction(self.phi_space) 
			p = self.phi_space.basis.p 
			qorder = max(2, 2*p+1) 
			for e in range(self.phi_space.Ne):
				el = self.phi_space.el[e]
				K = VEFPoissonIntegrator(el, [self.qdf, self.sweeper.sigma_t], qorder) 
				Ma = MassIntegrator(el, self.sigma_a, qorder)

				Q0 = np.zeros(el.Nn)
				Q1 = np.zeros(el.Nn)
				for a in range(self.N):
					w = self.w[a]
					mu = self.mu[a] 
					Q0 += w * DomainIntegrator(el, lambda x: self.sweeper.Q(x,mu), qorder)
					Q1 += w * mu * GradDomainIntegrator(el, 
						lambda x: self.sweeper.Q(x,mu)/self.sweeper.sigma_t(x), qorder) 

				f = Q0 + Q1 - el.CalcShape(1)*J.Interpolate(e, 1) \
					+ el.CalcShape(-1)*J.Interpolate(e, -1) 

				M1 = MixMassIntegrator(el, self.low_space.el[e], lambda x: 1, qorder) 
				M2 = MassIntegrator(self.low_space.el[e], lambda x: 1, qorder) 
				g = np.dot(M2, phi.GetDof(e))

				A = np.bmat([[K+Ma, M1], [M1.transpose(), np.zeros(M2.shape)]])
				rhs = np.concatenate((f, g))

				x = np.linalg.solve(A, rhs) 
				local = x[:el.Nn]
				phi_star.SetDof(e, local) 

			return phi_star 
		else:
			return phi 

class VEFH(AbstractVEF):
	def __init__(self, phi_space, J_space, sweeper, lin_solver=None, pp=True):
		if (isinstance(J_space, H1Space)):
			J_space = L2Space(J_space.xe, J_space.basis)
		if (pp):
			self.low_space = phi_space 
			AbstractVEF.__init__(self, self.low_space, J_space, sweeper, lin_solver)
			new_basis = type(phi_space.basis)(phi_space.basis.p+1)
			self.phi_space = type(phi_space)(phi_space.xe, new_basis)
		else:
			AbstractVEF.__init__(self, phi_space, J_space, sweeper, lin_solver) 
			self.low_space = phi_space 
		basis = LagrangeBasis(1)
		self.m_space = H1Space(self.phi_space.xe, basis) 
		self.C2 = MixFaceAssemble(self.m_space, self.J_space, ConstraintIntegrator, 1) 
		self.edd_constraint = UpwEddConstraintIntegrator
		self.pp = pp

	def Mult(self, psi):
		self.qdf.Compute(psi)
		qin = BdrFaceAssembleRHS(self.J_space, VEFInflowIntegrator, self.qdf)
		C1 = MixFaceAssemble(self.J_space, self.m_space, self.edd_constraint, self.qdf)
		Q1 = self.Q1 + qin 

		W, X, Y, Z = self.FormBlockInv()

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

		phi = GridFunction(self.low_space)
		phi.data = Y*Q1 - Y*C1*lam + Z*self.Q0

		if (self.pp):
			phi_star = GridFunction(self.phi_space)
			for e in range(self.phi_space.Ne):
				el = self.phi_space.el[e]
				s1 = el.CalcShape(-1)
				s2 = el.CalcShape(1)

				M1 = np.vstack((s1, s2))
				b1 = lam.GetDof(e)

				p = self.low_space.basis.p 
				if (p>0):
					basis = LegendreBasis(self.phi_space.basis.p-2)
					el2 = Element(basis, el.line) 
					M2 = MixMassIntegrator(el2, el, lambda x: 1, 2*p+1)
					mixmass = MixMassIntegrator(el2, self.low_space.el[e], lambda x: 1, 2*p+1)
					b2 = np.dot(mixmass, phi.GetDof(e))

					A = np.vstack((M1, M2))
					b = np.concatenate((b1, b2))

					local = np.linalg.solve(A, b) 
					phi_star.SetDof(e, local) 
				else:
					local = np.linalg.solve(M1, b1)
					phi_star.SetDof(e, local)

			return phi_star
		else:
			return phi 

	def FormBlockInv(self):
		W = COOBuilder(self.J_space.Nu)
		X = COOBuilder(self.J_space.Nu, self.low_space.Nu)
		Y = COOBuilder(self.low_space.Nu, self.J_space.Nu)
		Z = COOBuilder(self.low_space.Nu)

		for e in range(self.low_space.Ne):
			phi_el = self.low_space.el[e]
			J_el = self.J_space.el[e] 

			Ma = MassIntegrator(phi_el, self.sigma_a, 2*self.p+1)
			D = MixDivIntegrator(phi_el, J_el, 1, 2*self.p+1)
			Mt = MassIntegrator(J_el, self.sweeper.sigma_t, 2*self.p+1)
			if (phi_el.ElNo==0):
				B = MLBdrIntegrator(self.J_space.bface[0], self.qdf)
				Mt += B
			elif (phi_el.ElNo==self.low_space.Ne-1):
				B = MLBdrIntegrator(self.J_space.bface[-1], self.qdf)
				Mt += B
			Mtinv = np.linalg.inv(Mt)
			G = MixWeakEddDivIntegrator(J_el, phi_el, self.qdf, 2*self.p+1)

			S = Ma - np.linalg.multi_dot([D, Mtinv, G])
			Sinv = np.linalg.inv(S)
			w = Mtinv + np.linalg.multi_dot([Mtinv, G, Sinv, D, Mtinv])
			x = -np.linalg.multi_dot([Mtinv, G, Sinv])
			y = -np.linalg.multi_dot([Sinv, D, Mtinv])
			z = Sinv.copy() 

			W[self.J_space.dofs[e], self.J_space.dofs[e]] = w 
			X[self.J_space.dofs[e], self.low_space.dofs[e]] = x
			Y[self.low_space.dofs[e], self.J_space.dofs[e]] = y
			Z[self.low_space.dofs[e], self.low_space.dofs[e]] = z 

		return W.Get(), X.Get(), Y.Get(), Z.Get()

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
	block = BlockLDU(ltol, maxiter, inner, False)
	# block = BlockLDURelax(ltol, maxiter, gs, inner, False)
	# block = BlockTri(ltol, maxiter, inner, False)
	# block = BlockDiag(ltol, maxiter, inner, False)
	vef = VEF(phi_space, J_space, sweep, block)
	amg = AMGSolver(ltol, maxiter, inner, False)
	vefh = VEFH(phi_space, J_space, sweep, amg)
	psi = TVector(phi_space, N)
	psi.Project(lambda x, mu: 1)
	phi = vef.SourceIteration(psi)
	psi.Project(lambda x, mu: 1)
	phi = vefh.SourceIteration(psi) 