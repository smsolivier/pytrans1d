#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from .qdf import * 
from .sn import * 
from trans1d.fem.linsolver import * 
import pyamg 

class AbstractVEF(Sn):
	def __init__(self, phi_space, J_space, sweeper, lin_solver=None):
		Sn.__init__(self, sweeper) 
		self.phi_space = phi_space 
		self.J_space = J_space 
		self.lin_solver = lin_solver

		p = J_space.basis.p
		self.p = p 
		qorder = 2*p+1
		self.qorder = qorder 
		self.sigma_a = lambda x: sweeper.sigma_t(x) - sweeper.sigma_s(x)
		self.Mt = Assemble(self.J_space, MassIntegrator, sweeper.sigma_t, qorder)
		self.Mtl = Assemble(self.J_space, MassIntegratorLumped, sweeper.sigma_t, qorder)
		self.Ma = Assemble(self.phi_space, MassIntegrator, self.sigma_a, qorder)
		self.D = MixAssemble(self.phi_space, self.J_space, MixDivIntegrator, 1, qorder) 

		self.Q0 = np.zeros(phi_space.Nu)
		self.Q1 = np.zeros(J_space.Nu)
		for a in range(self.N):
			mu = self.mu[a]
			self.Q0 += AssembleRHS(phi_space, DomainIntegrator, lambda x: sweeper.Q(x,mu), qorder)*self.w[a]
			self.Q1 += AssembleRHS(J_space, DomainIntegrator, lambda x: sweeper.Q(x,mu), qorder)*mu*self.w[a]

		self.qdf = QDFactors(self.space, self.sweeper.quad, self.sweeper.psi_in) 
		self.k = 0
		self.linit = []

	def SourceIteration(self, psi, niter=50, tol=1e-6):
		phi = GridFunction(self.phi_space)
		phi_old = GridFunction(self.phi_space)
		self.linit = []
		for n in range(niter):
			start = time.time() 
			phi_old.data = phi.data.copy() 
			self.sweeper.Sweep(psi, phi) 
			phi, J = self.Mult(psi)
			norm = phi.L2Diff(phi_old, 2*self.p+1)
			if (self.lin_solver!=None):
				self.linit.append(self.lin_solver.it)

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
			self.avg_linit = np.mean(self.linit) 
			print('avg linear iters = {:.2f}'.format(self.avg_linit))

		if (norm > tol):
			print(colored('WARNING not converged! Final tol = {:.3e}'.format(norm), 'red'))

		return phi 

	def PostProcess(self, k, star_space, phi, J):
		phi_star = GridFunction(star_space)
		basis = LegendreBasis(k) 
		lam_space = L2Space(self.space.xe, basis) 
		for e in range(star_space.Ne):
			star_el = star_space.el[e] 
			lam_el = lam_space.el[e] 
			phi_el = phi.space.el[e]
			J_el = J.space.el[e] 

			K = VEFPoissonIntegrator(star_el, [self.qdf, self.sweeper.sigma_t], self.qorder)
			Ma = MassIntegrator(star_el, self.sigma_a, self.qorder)

			Q0 = np.zeros(star_el.Nn)
			Q1 = np.zeros(star_el.Nn)
			for a in range(self.N):
				w = self.w[a]
				mu = self.mu[a] 
				Q0 += w * DomainIntegrator(star_el, lambda x: self.sweeper.Q(x,mu), self.qorder)
				Q1 += w * mu * GradDomainIntegrator(star_el, 
					lambda x: self.sweeper.Q(x,mu)/self.sweeper.sigma_t(x), self.qorder) 

			Jbdr = star_el.CalcShape(1)*J.Interpolate(e,1) - star_el.CalcShape(-1)*J.Interpolate(e,-1)
			f = Q0 + Q1 - Jbdr 

			M1 = MixMassIntegrator(star_el, lam_el, lambda x: 1, self.qorder)
			M2 = MixMassIntegrator(lam_el, phi_el, lambda x: 1, self.qorder) 
			g = np.dot(M2, phi.GetDof(e))

			A = np.bmat([[K+Ma, M1], [M1.transpose(), np.zeros((lam_el.Nn, lam_el.Nn))]])
			rhs = np.concatenate((f, g))

			x = np.linalg.solve(A, rhs) 
			phi_star.SetDof(e, x[:star_el.Nn])
		return phi_star

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
		self.pp = pp 
		self.scl = False
		self.exl = False

	def Mult(self, psi):
		self.qdf.Compute(psi) 
		B = BdrFaceAssemble(self.J_space, MLBdrIntegrator, self.qdf)
		G = MixAssemble(self.J_space, self.low_space, MixWeakEddDivIntegrator, self.qdf, self.qorder) 
		qin = BdrFaceAssembleRHS(self.J_space, VEFInflowIntegrator, self.qdf) 

		Mt = self.Mt + B 
		Q1 = self.Q1 + qin 
		rhs = np.concatenate((Q1, self.Q0))

		if (self.lin_solver==None):
			A = sp.bmat([[Mt, G], [self.D, self.Ma]]).tocsc()
			x = spla.spsolve(A, rhs) 

		else:
			if (self.scl):
				Ainv = self.AssembleMtBInvSCLump()
			elif (self.exl):
				Ainv = spla.inv(Mt) 
			else:
				Ainv = sp.diags(1/(self.Mtl + B).diagonal())
			x = self.lin_solver.Solve(Mt, Ainv, G, self.D, self.Ma, rhs)

		phi = GridFunction(self.low_space)
		phi.data = x[self.J_space.Nu:]
		J = GridFunction(self.J_space)
		J.data = x[:self.J_space.Nu] 

		if (self.pp):
			phi_star = self.PostProcess(self.k, self.phi_space, phi, J) 
			return phi_star, J
		else:
			return phi, J

	def AssembleMtBInvSCLump(self):
		A = COOBuilder(self.J_space.nint)
		Ainv = COOBuilder(self.J_space.nint)
		B = COOBuilder(self.J_space.nint, self.J_space.nedge)
		C = COOBuilder(self.J_space.nedge, self.J_space.nint)
		D = COOBuilder(self.J_space.nedge, self.J_space.nedge)
		for e in range(self.J_space.Ne):
			elmat = MassIntegrator(self.J_space.el[e], self.sweeper.sigma_t, self.qorder)
			if (e==0):
				bdr = MLBdrIntegrator(self.J_space.bface[0], self.qdf)
				elmat += bdr
			elif (e==self.J_space.Ne-1):
				bdr = MLBdrIntegrator(self.J_space.bface[-1], self.qdf)
				elmat += bdr
			elmat_i_inv = np.linalg.inv(elmat[1:-1,1:-1])
			A[self.J_space.int_dof[e], self.J_space.int_dof[e]] = elmat[1:-1,1:-1]
			Ainv[self.J_space.int_dof[e], self.J_space.int_dof[e]] = elmat_i_inv 
			B[self.J_space.int_dof[e], self.J_space.edge_dof[e]] = elmat[1:-1,[0,-1]]
			C[self.J_space.edge_dof[e], self.J_space.int_dof[e]] = elmat[[0,-1],1:-1]
			d = np.zeros((2,2))
			d[0,0] = elmat[0,0] 
			d[0,-1] = elmat[0,-1] 
			d[-1,0] = elmat[-1,0]
			d[-1,-1] = elmat[-1,-1] 
			D[self.J_space.edge_dof[e], self.J_space.edge_dof[e]] = d 

		A = A.Get()
		Ainv = Ainv.Get()
		B = B.Get()
		C = C.Get()
		D = D.Get()

		S = D - C*Ainv*B 
		Sinv = sp.diags(1/np.squeeze(np.asarray(S.sum(axis=1))))

		W = Ainv + Ainv*B*Sinv*C*Ainv 
		X = -Ainv*B*Sinv
		Y = -Sinv*C*Ainv 
		Z = Sinv 

		M = sp.bmat([[A,B], [C,D]]).tocsc()
		Minv = sp.bmat([[W,X], [Y,Z]]).tocsc()

		return Minv[self.J_space.sci,:][:,self.J_space.sci]

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
		self.pp_type = 'lagrange'

	def Mult(self, psi, retlam=False):
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
		J = GridFunction(self.J_space)
		J.data = W*(Q1 - C1*lam) + X*self.Q0 

		if (self.pp and self.pp_type=='lagrange'):
			phi = self.PostProcessLagrange(self.phi_space, phi, J, lam)
		elif (self.pp and self.pp_type=='vef'):
			phi = self.PostProcess(self.k, self.phi_space, phi, J)

		if (retlam):
			return phi, J, lam 
		else:
			return phi, J

	def PostProcessLagrange(self, star_space, phi, J, lam):
		phi_star = GridFunction(star_space)
		for e in range(star_space.Ne):
			el = star_space.el[e]
			s1 = el.CalcShape(-1)
			s2 = el.CalcShape(1)

			M1 = np.vstack((s1, s2))
			b1 = lam.GetDof(e)

			p = phi.space.basis.p 
			if (p>0):
				basis = LegendreBasis(star_space.basis.p-2)
				el2 = Element(basis, el.line) 
				M2 = MixMassIntegrator(el2, el, lambda x: 1, self.qorder)
				mixmass = MixMassIntegrator(el2, phi.space.el[e], lambda x: 1, self.qorder)
				b2 = np.dot(mixmass, phi.GetDof(e))

				A = np.vstack((M1, M2))
				b = np.concatenate((b1, b2))

				local = np.linalg.solve(A, b) 
				phi_star.SetDof(e, local) 
			else:
				local = np.linalg.solve(M1, b1)
				phi_star.SetDof(e, local)
		return phi_star

	def FormBlockInv(self):
		W = COOBuilder(self.J_space.Nu)
		X = COOBuilder(self.J_space.Nu, self.low_space.Nu)
		Y = COOBuilder(self.low_space.Nu, self.J_space.Nu)
		Z = COOBuilder(self.low_space.Nu)

		for e in range(self.low_space.Ne):
			phi_el = self.low_space.el[e]
			J_el = self.J_space.el[e] 

			Ma = MassIntegrator(phi_el, self.sigma_a, self.qorder)
			D = MixDivIntegrator(phi_el, J_el, 1, self.qorder)
			Mt = MassIntegrator(J_el, self.sweeper.sigma_t, self.qorder)
			if (phi_el.ElNo==0):
				B = MLBdrIntegrator(self.J_space.bface[0], self.qdf)
				Mt += B
			elif (phi_el.ElNo==self.low_space.Ne-1):
				B = MLBdrIntegrator(self.J_space.bface[-1], self.qdf)
				Mt += B
			Mtinv = np.linalg.inv(Mt)
			G = MixWeakEddDivIntegrator(J_el, phi_el, self.qdf, self.qorder)

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

class VEFH2(VEFH):
	def __init__(self, phi_space, J_space, sweeper, lin_solver=None, pp=True):
		VEFH.__init__(self, phi_space, J_space, sweeper, lin_solver, pp)

	def Mult(self, psi, retlam=False):
		self.qdf.Compute(psi) 
		qin = BdrFaceAssembleRHS(self.J_space, VEFInflowIntegrator, self.qdf)
		C1 = MixFaceAssemble(self.J_space, self.m_space, self.edd_constraint, self.qdf)
		Q1 = self.Q1 + qin
		G = MixAssemble(self.J_space, self.low_space, MixWeakEddDivIntegrator, self.qdf, self.qorder) 

		Mtinv = self.FormMtInv()
		S = self.Ma - self.D*Mtinv*G 
		B = -self.D*Mtinv*C1 
		C = -self.C2*Mtinv*G
		D = -self.C2*Mtinv*C1 
		rhs1 = self.Q0.data - self.D*Mtinv*Q1
		rhs2 = -self.C2*Mtinv*Q1

		bdr_phi = (Mtinv*G).tolil()
		bdr_lam = (Mtinv*C1).tolil()
		bdr_rhs = Mtinv*Q1
		bdr_phi[0,:] *= -1 
		bdr_lam[0,:] *= -1
		bdr_lam[0,0] += self.qdf.EvalG(self.phi_space.bface[0])
		bdr_lam[-1,-1] += self.qdf.EvalG(self.phi_space.bface[-1])
		bdr_rhs[0] *= -1 
		bdr_rhs[0] -= 2*self.qdf.EvalJinBdr(self.phi_space.bface[0])
		bdr_rhs[-1] -= 2*self.qdf.EvalJinBdr(self.phi_space.bface[-1])

		C = C.tolil()
		D = D.tolil()
		C[0,:] = bdr_phi[0,:]
		C[-1,:] = bdr_phi[-1,:]
		D[0,:] = bdr_lam[0,:]
		D[-1,:] = bdr_lam[-1,:]
		rhs2[0] = bdr_rhs[0]
		rhs2[-1] = bdr_rhs[-1]

		M = sp.bmat([[S, B], [C, D]])
		rhs = np.concatenate((rhs1, rhs2))

		x = spla.spsolve(M.tocsc(), rhs) 
		phi = GridFunction(self.low_space)
		J = GridFunction(self.J_space)
		lam = GridFunction(self.m_space)
		phi.data = x[:self.low_space.Nu]
		lam.data = x[self.low_space.Nu:]

		J.data = Mtinv*(Q1 - G*phi.data - C1*lam.data)

		if (self.pp and self.pp_type=='lagrange'):
			phi = self.PostProcessLagrange(self.phi_space, phi, J, lam)
		elif (self.pp and self.pp_type=='vef'):
			phi = self.PostProcess(self.k, self.phi_space, phi, J)

		if (retlam):
			return phi, J, lam
		else:
			return phi, J 

	def FormMtInv(self):
		M = COOBuilder(self.J_space.Nu)

		for e in range(self.J_space.Ne):
			el = self.J_space.el[e]

			Mt = MassIntegrator(el, self.sweeper.sigma_t, self.qorder)
			if (el.ElNo==0):
				B = MLBdrIntegrator(self.J_space.bface[0], self.qdf)
				Mt += B 
			elif (el.ElNo==self.J_space.Nu-1):
				B = MLBdrIntegrator(self.J_space.bface[-1], self.qdf)
				Mt += B 
			Mtinv = np.linalg.inv(Mt)

			M[self.J_space.dofs[e], self.J_space.dofs[e]] = Mtinv 

		return M.Get()
