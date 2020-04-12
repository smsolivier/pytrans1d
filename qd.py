#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from qdf import * 
from sn import * 
from p1sa import * 

def JumpJumpQDFIntegrator(face, qdf):
	xi1 = face.IPTrans(0)
	xi2 = face.IPTrans(1)
	s1 = face.el1.CalcShape(xi1)
	s2 = face.el2.CalcShape(xi2)
	G1 = qdf.EvalGInt(face.el1, xi1)
	G2 = qdf.EvalGInt(face.el2, xi2)
	j1 = np.concatenate((s1, -s2))
	j2 = np.concatenate((G1*s1, -G2*s2))
	return .5*np.outer(j1, j2) 

def FM1Integrator(face, qdf):
	xi1 = face.IPTrans(0)
	xi2 = face.IPTrans(1)
	s1 = face.el1.CalcShape(xi1)
	s2 = face.el2.CalcShape(xi2)
	C1 = qdf.EvalCp(face)
	C2 = qdf.EvalCm(face)
	E1 = qdf.EvalFactor(face.el1, xi1)
	E2 = qdf.EvalFactor(face.el2, xi2)
	j1 = np.concatenate((E1*s1, -E2*s2))
	j2 = np.concatenate((s1/C1, -s2/C2))
	return .5*np.outer(j1, j2)

def FM2Integrator(face1, face2, qdf):
	xi1 = face1.IPTrans(0)
	xi2 = face1.IPTrans(1)
	s1 = face1.el1.CalcShape(xi1)
	s2 = face1.el2.CalcShape(xi2)
	E1 = qdf.EvalFactor(face1.el1, xi1)
	E2 = qdf.EvalFactor(face1.el2, xi2)
	j = np.concatenate((E1*s1, -E2*s2)) * face1.nor

	s1 = face2.el1.CalcShape(xi1)
	s2 = face2.el2.CalcShape(xi2)
	C1 = qdf.EvalCp(face2)
	C2 = qdf.EvalCm(face2)
	G1 = qdf.EvalGInt(face2.el1, xi1)
	G2 = qdf.EvalGInt(face2.el2, xi2)
	a = .5*np.concatenate((G1/C1*s1, G2/C2*s2))
	return np.outer(j, a)

def PhiInflowIntegrator(face, qdf):
	xi = face.IPTrans(0)
	s = face.el1.CalcShape(xi)
	E = qdf.EvalFactor(face.el1, xi)
	return s*E*face.nor*qdf.EvalPhiInBdr(face)

def JInflowIntegrator(face, qdf):
	xi = face.IPTrans(0)
	s = face.el1.CalcShape(xi)
	return s*qdf.EvalJinBdr(face)

class AbstractQD(Sn):
	def __init__(self, phi_space, J_space, sweeper, lin_solver=None):
		Sn.__init__(self, sweeper)
		self.phi_space = phi_space 
		self.J_space = J_space 
		self.lin_solver = lin_solver

		p = self.J_space.basis.p 
		self.qorder = qorder = 2*p+1
		self.sigma_a = lambda x: sweeper.sigma_t(x) - sweeper.sigma_s(x)
		self.Mt = Assemble(J_space, MassIntegrator, sweeper.sigma_t, qorder)
		self.Ma = Assemble(phi_space, MassIntegrator, self.sigma_a, qorder)
		self.D = MixAssemble(phi_space, J_space, WeakMixDivIntegrator, 1, qorder)
		self.D += MixFaceAssembleAll(phi_space, J_space, MixJumpAvgIntegrator, 1)

		self.Q0 = np.zeros(phi_space.Nu)
		self.Q1 = np.zeros(J_space.Nu)
		for a in range(self.N):
			mu = self.mu[a]
			self.Q0 += AssembleRHS(phi_space, DomainIntegrator, lambda x: sweeper.Q(x,mu), qorder)*self.w[a]
			self.Q1 += AssembleRHS(J_space, DomainIntegrator, lambda x: sweeper.Q(x,mu), qorder)*mu*self.w[a]

		self.qdf = QDFactors(self.space, self.N, self.sweeper.psi_in) 
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

class QD(AbstractQD):
	def __init__(self, phi_space, J_space, sweeper, lin_solver=None):
		AbstractQD.__init__(self, phi_space, J_space, sweeper, lin_solver)

	def Mult(self, psi):
		self.qdf.Compute(psi)
		G = MixAssemble(self.J_space, self.phi_space, 
			MixWeakEddDivIntegrator, self.qdf, self.qorder)
		Ma = self.Ma + FaceAssembleAll(self.phi_space, 
			JumpJumpQDFIntegrator, self.qdf)
		Mt = self.Mt + FaceAssembleAll(self.J_space, FM1Integrator, self.qdf)
		G += MixFaceAssembleAll(self.J_space, self.phi_space, FM2Integrator, self.qdf)

		Q0 = self.Q0 - BdrFaceAssembleRHS(self.phi_space, JInflowIntegrator, self.qdf)
		Q1 = self.Q1 - BdrFaceAssembleRHS(self.J_space, PhiInflowIntegrator, self.qdf)

		M = sp.bmat([[Mt, G], [self.D, Ma]])
		rhs = np.concatenate((Q1, Q0))

		x = spla.spsolve(M.tocsc(), rhs)
		phi = GridFunction(self.phi_space)
		J = GridFunction(self.J_space)

		phi.data = x[self.J_space.Nu:]
		J.data = x[:self.J_space.Nu]

		return phi, J 

# class QDLO(AbstractQD):
# 	def __init__(self, phi_space, J_space, sweeper, lin_solver=None):
# 		AbstractQD.__init__(self, phi_space, J_space, sweeper, lin_solver)


if __name__=='__main__':
	import sys 
	Ne = 10 
	p = 3
	if (len(sys.argv)>1):
		Ne = int(sys.argv[1])
	if (len(sys.argv)>2):
		p = int(sys.argv[2])
	N = 4
	xe = np.linspace(0,1, Ne+1)
	leg = LegendreBasis(p)
	space = L2Space(xe, leg)

	eps = .1
	sigma_t = lambda x: 1/eps 
	sigma_s = lambda x: 1/eps - eps 

	alpha = 1
	beta = 0 
	gamma = 0 
	delta = 2
	eta = 0 
	L = 1 + 2*eta 
	f = lambda x: alpha*np.sin(np.pi*x) + delta 
	df = lambda x: alpha*np.pi*np.cos(np.pi*x) 
	g = lambda x: beta*np.sin(2*np.pi*x) 
	dg = lambda x: beta*2*np.pi*np.cos(2*np.pi*x) 
	h = lambda x: gamma*np.sin(3*np.pi*(x+eta)/L)
	dh = lambda x: gamma*3*np.pi/L*np.cos(3*np.pi*(x+eta)/L)
	psi_ex = lambda x, mu: .5*(f(x) + mu*g(x) + mu**2*h(x))
	dpsi = lambda x, mu: .5*(df(x) + mu*dg(x) + mu**2*dh(x))
	phi_ex = lambda x: f(x) + 1/3*h(x)

	Q = lambda x, mu: mu*dpsi(x,mu) + sigma_t(x)*psi_ex(x,mu) - sigma_s(x)/2*phi_ex(x)
	psi_in = lambda x, mu: psi_ex(x,mu)
	sweep = DirectSweeper(space, N, sigma_t, sigma_s, Q, psi_in)

	qd = QD(space, space, sweep)
	psi = TVector(space, N)
	phi = qd.SourceIteration(psi, tol=1e-10) 
	phi_sn = qd.ComputeScalarFlux(psi) 

	print('diff = {:.3e}'.format(phi.L2Diff(phi_sn, 2*p+1)))
	print('err = {:.3e}'.format(phi.L2Error(phi_ex, 2*p+1)))

	# plt.figure()
	# plt.plot(space.x, qd.qdf.P.data/qd.qdf.phi.data, '-o')

	plt.figure()
	plt.plot(phi.space.x, phi.data, '-o')
	plt.plot(phi_sn.space.x, phi_sn.data, '-o')
	plt.show()
