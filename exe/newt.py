#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans1d import * 

tol = 1e-10

def FormDG(vfes, tfes, quad, qdf, phi_old, qorder):
	coo = COOBuilder(vfes.Nu, tfes.Nu*quad.N) 
	ip, w = quadrature.Get(qorder)
	for e in range(sfes.Ne):
		elmat = np.zeros((vfes.el[e].Nn, tfes.el[e].Nn*quad.N))
		for n in range(len(w)):
			gshape = vfes.el[e].CalcPhysGradShape(ip[n]) 
			phis = phi_old.Interpolate(e, ip[n]) / qdf.phi.Interpolate(e, ip[n]) 
			E = qdf.EvalFactor(tfes.el[e], ip[n]) 
			s = tfes.el[e].CalcShape(ip[n]) 
			M2 = np.zeros(tfes.el[e].Nn*quad.N) 
			M0 = np.zeros(tfes.el[e].Nn*quad.N) 
			for a in range(quad.N):
				M2[a*tfes.el[e].Nn:(a+1)*tfes.el[e].Nn] = quad.mu[a]**2 * quad.w[a] * s * phis 
				M0[a*tfes.el[e].Nn:(a+1)*tfes.el[e].Nn] = quad.w[a] * s * phis * E 

			B = M2 - M0 
			elmat += w[n] * vfes.el[e].Jacobian(ip[n]) * np.outer(gshape, B) 

		dofs = [] 
		for a in range(quad.N):
			dofs += (tfes.dofs[e]+a*tfes.Nu).tolist()
		coo[vfes.dofs[e], dofs] = elmat 

	return coo.Get()

def FullNewton(sigma_t, sigma_s, source, psi_in):
	Q0 = np.zeros(sfes.Nu)
	Q1 = np.zeros(vfes.Nu) 
	for a in range(quad.N):
		Q0 += quad.w[a] * AssembleRHS(sfes, DomainIntegrator, lambda x: source(x,quad.mu[a]), 2*p+1)
		Q1 += quad.mu[a] * quad.w[a] * AssembleRHS(vfes, DomainIntegrator, lambda x: source(x,quad.mu[a]), 2*p+1)

	sweep = DirectSweeper(tfes, quad, sigma_t, sigma_s, source, psi_in) 
	qdf = QDFactors(tfes, quad, psi_in) 
	Mt = Assemble(vfes, MassIntegrator, sigma_t, 2*p+1)
	sigma_a = lambda x: sigma_t(x) - sigma_s(x) 
	Ma = Assemble(sfes, MassIntegrator, sigma_a, 2*p+1)
	D = MixAssemble(sfes, vfes, MixDivIntegrator, 1, 2*p+1)

	Linv = sweep.FormInverse() 
	ms = MixAssemble(tfes, sfes, MixMassIntegrator, sigma_s, 2*p+1) / 2 
	Ms = sp.bmat([[ms] for a in range(quad.N)]) 
	LinvS = Linv*Ms 

	psi = TVector(tfes, quad) 
	psi.Project(lambda x, mu: 1)
	phi = GridFunction(sfes)
	J = GridFunction(vfes) 

	for k in range(25):
		qdf.Compute(psi) 
		Gk = MixAssemble(vfes, sfes, MixWeakEddDivIntegrator, qdf, 2*p+1)
		Vk = FormDG(vfes, tfes, quad, qdf, phi, 2*p+1) 
		VkLinvS = Vk*LinvS 
		A = sp.bmat([[Mt, Gk - VkLinvS], [D, Ma]]).tocsc()
		u = spla.spsolve(A, np.concatenate([Q1 - VkLinvS*phi, Q0]))
		norm = np.linalg.norm(u[vfes.Nu:] - phi.data)
		phi.data = u[vfes.Nu:] 
		sweep.Sweep(psi, phi) 

		# print('it = {}, norm = {:.3e}'.format(k+1, norm)) 
		if (norm < tol):
			break 

	return phi, k+1

def LaggedNewton(sigma_t, sigma_s, source, psi_in, ninner=1, itol=1e-12, omega=1):
	Q0 = np.zeros(sfes.Nu)
	Q1 = np.zeros(vfes.Nu) 
	for a in range(quad.N):
		Q0 += quad.w[a] * AssembleRHS(sfes, DomainIntegrator, lambda x: source(x,quad.mu[a]), 2*p+1)
		Q1 += quad.mu[a] * quad.w[a] * AssembleRHS(vfes, DomainIntegrator, lambda x: source(x,quad.mu[a]), 2*p+1)

	sweep = DirectSweeper(tfes, quad, sigma_t, sigma_s, source, psi_in) 
	qdf = QDFactors(tfes, quad, psi_in) 
	Mt = Assemble(vfes, MassIntegrator, sigma_t, 2*p+1)
	sigma_a = lambda x: sigma_t(x) - sigma_s(x) 
	Ma = Assemble(sfes, MassIntegrator, sigma_a, 2*p+1)
	D = MixAssemble(sfes, vfes, MixDivIntegrator, 1, 2*p+1)

	psi = TVector(tfes, quad) 
	psi.Project(lambda x, mu: 1)
	phi = GridFunction(sfes)
	J = GridFunction(vfes) 
	ns = 0 
	for k in range(25):
		qdf.Compute(psi) 
		Gk = MixAssemble(vfes, sfes, MixWeakEddDivIntegrator, qdf, 2*p+1)
		r1 = Mt*J.data + Gk*phi.data - Q1 
		r2 = D*J.data + Ma*phi.data - Q0 
		Vk = FormDG(vfes, tfes, quad, qdf, phi, 2*p+1) 
		A = sp.bmat([[Mt, Gk], [D, Ma]]).tocsc() 
		u = spla.spsolve(A, np.concatenate([Q1, Q0]))
		phi_new = GridFunction(sfes) 
		inorm = np.linalg.norm(phi.data - u[vfes.Nu:]) 
		phi_new.data = u[vfes.Nu:] 
		sweep.Sweep(psi, phi_new) 
		ns += 1 
		if (inorm>itol):
			for n in range(ninner):
				u = spla.spsolve(A, np.concatenate([Q1 + Vk*psi.data, Q0]))
				inorm = np.linalg.norm(phi_new.data - u[vfes.Nu:]) 
				phi_new.data = u[vfes.Nu:]
				sweep.Sweep(psi, phi_new)
				ns += 1 
				if (inorm<itol):
					break 

		norm = np.linalg.norm(phi.data - phi_new.data)
		phi.data = phi_new.data.copy() 
		J.data = u[:vfes.Nu]
		# print('it = {}, norm = {:.3e}'.format(k+1, norm)) 
		if (norm < tol):
			break 

	return phi, k+1, ns

def ApproxNewton(sigma_t, sigma_s, source, psi_in):
	Q0 = np.zeros(sfes.Nu)
	Q1 = np.zeros(vfes.Nu) 
	for a in range(quad.N):
		Q0 += quad.w[a] * AssembleRHS(sfes, DomainIntegrator, lambda x: source(x,quad.mu[a]), 2*p+1)
		Q1 += quad.mu[a] * quad.w[a] * AssembleRHS(vfes, DomainIntegrator, lambda x: source(x,quad.mu[a]), 2*p+1)

	sweep = DirectSweeper(tfes, quad, sigma_t, sigma_s, source, psi_in) 
	qdf = QDFactors(tfes, quad, psi_in) 
	Mt = Assemble(vfes, MassIntegrator, sigma_t, 2*p+1)
	sigma_a = lambda x: sigma_t(x) - sigma_s(x) 
	Ma = Assemble(sfes, MassIntegrator, sigma_a, 2*p+1)
	D = MixAssemble(sfes, vfes, MixDivIntegrator, 1, 2*p+1)

	Linv = sweep.FormInverse() 
	ms = MixAssemble(tfes, sfes, MixMassIntegrator, sigma_s, 2*p+1) / 2 
	Ms = sp.bmat([[ms] for a in range(quad.N)]) 
	LinvS = Linv*Ms 

	psi = TVector(tfes, quad) 
	psi.Project(lambda x, mu: 1)
	phi = GridFunction(sfes)
	J = GridFunction(vfes) 

	ns = 0
	for k in range(25):
		qdf.Compute(psi) 
		Gk = MixAssemble(vfes, sfes, MixWeakEddDivIntegrator, qdf, 2*p+1)
		Vk = FormDG(vfes, tfes, quad, qdf, phi, 2*p+1) 
		VkLinvS = Vk*LinvS 
		N = sp.bmat([[sp.eye(vfes.Nu)*0, -VkLinvS], [None, sp.eye(sfes.Nu)*0]])
		A = sp.bmat([[Mt, Gk], [D, Ma]]).tocsc()
		Ainv = spla.inv(A)
		ell = Ainv*N 
		B = (sp.eye(vfes.Nu+sfes.Nu) - ell)*Ainv 
		# u = spla.spsolve(A, np.concatenate([Q1 - VkLinvS*phi, Q0]))
		u = B*np.concatenate([Q1 - VkLinvS*phi, Q0])
		ns += 1
		norm = np.linalg.norm(u[vfes.Nu:] - phi.data)
		phi.data = u[vfes.Nu:] 
		sweep.Sweep(psi, phi) 
		ns += 1

		# print('it = {}, norm = {:.3e}'.format(k+1, norm)) 
		if (norm < tol):
			break 

	return phi, k+1, ns

def NoSweepNewton(sigma_t, sigma_s, source, psi_in):
	Q0 = np.zeros(sfes.Nu)
	Q1 = np.zeros(vfes.Nu) 
	for a in range(quad.N):
		Q0 += quad.w[a] * AssembleRHS(sfes, DomainIntegrator, lambda x: source(x,quad.mu[a]), 2*p+1)
		Q1 += quad.mu[a] * quad.w[a] * AssembleRHS(vfes, DomainIntegrator, lambda x: source(x,quad.mu[a]), 2*p+1)

	sweep = DirectSweeper(tfes, quad, sigma_t, sigma_s, source, psi_in) 
	qdf = QDFactors(tfes, quad, psi_in) 
	Mt = Assemble(vfes, MassIntegrator, sigma_t, 2*p+1)
	sigma_a = lambda x: sigma_t(x) - sigma_s(x) 
	Ma = Assemble(sfes, MassIntegrator, sigma_a, 2*p+1)
	D = MixAssemble(sfes, vfes, MixDivIntegrator, 1, 2*p+1)

	Linv = sweep.FormInverse() 
	ms = MixAssemble(tfes, sfes, MixMassIntegrator, sigma_s, 2*p+1) / 2 
	Ms = sp.bmat([[ms] for a in range(quad.N)]) 
	LinvS = Linv*Ms 

	psi = TVector(tfes, quad) 
	psi.Project(lambda x, mu: 1)
	qdf.Compute(psi)
	phi = GridFunction(sfes)
	phi.data += 1 
	J = GridFunction(vfes) 
	phip = GridFunction(sfes)

	ns = 0
	for k in range(25):
		Vk = FormDG(vfes, tfes, quad, qdf, phi, 2*p+1) 
		sweep.Sweep(psi, phi)
		ns += 1
		qdf.Compute(psi)
		Gk = MixAssemble(vfes, sfes, MixWeakEddDivIntegrator, qdf, 2*p+1)
		VkLinvS = Vk*LinvS 
		A = sp.bmat([[Mt, Gk], [D, Ma]]).tocsc()
		u = spla.spsolve(A, np.concatenate([Q1 + Vk*psi.data, Q0]))
		norm = np.linalg.norm(u[vfes.Nu:] - phi.data)
		phi.data = u[vfes.Nu:] 

		# print('it = {}, norm = {:.3e}'.format(k+1, norm)) 
		if (norm < tol):
			break 

	return phi, k+1, ns

Ne = 10 
p = 2
N = 8 
quad = LegendreQuad(N) 
xe = np.linspace(0,1,Ne+1)
tfes = L2Space(xe, LegendreBasis(p))
sfes = L2Space(xe, LegendreBasis(p-1))
vfes = H1Space(xe, LobattoBasis(p))

eps = np.geomspace(1e-4, .75, 8)
full = np.zeros(len(eps))
# inner = np.arange(0,3)
inner = [1,2]
itol = [1]
omega = [.5,5]
fp = np.zeros(len(eps))
fp_sweeps = np.zeros(len(eps))
lag = np.zeros((len(eps), len(inner)))
lag_sweeps = np.zeros((len(eps), len(inner)), dtype=int)
app = np.zeros(len(eps))
app_sweeps = np.zeros(len(eps))
for i,e in enumerate(eps): 
	sigma_t = lambda x: 1/e 
	sigma_s = lambda x: sigma_t(x) - e
	psi_in = lambda x, mu: 0 
	source = lambda x, mu: e 
	print('epsilon = {:.3e}'.format(e))

	phi, full[i] = FullNewton(sigma_t, sigma_s, source, psi_in) 
	phi, fp[i], fp_sweeps[i] = LaggedNewton(sigma_t, sigma_s, source, psi_in, 0)
	phi, app[i], app_sweeps[i] = NoSweepNewton(sigma_t, sigma_s, source, psi_in) 
	# for j in range(len(inner)):
		# phi, lag[i,j], lag_sweeps[i,j] = LaggedNewton(sigma_t, sigma_s, source, psi_in, inner[j], -1)

print(app_sweeps/fp_sweeps)
plt.figure()
plt.semilogx(eps, fp, '-o', label='FP')
plt.semilogx(eps, app, '-s', label='Approx')
# for i in range(len(inner)):
	# plt.semilogx(eps, lag[:,i], '-o', label='$k='+str(inner[i])+'$')
plt.semilogx(eps, full, '-o', label='Full')
plt.xlabel(r'$\epsilon$')
plt.ylabel('Number of Iterations') 
plt.legend()

plt.figure()
plt.semilogx(eps, fp_sweeps, '-o', label='Fixed Point') 
plt.semilogx(eps, app_sweeps, '-s', label='Approx')
# for i in range(len(omega)):
	# plt.semilogx(eps, lag_sweeps[:,i], '-o', label=r'$k=' + str(inner[i])+'$')

plt.xlabel(r'$\epsilon$')
plt.ylabel('Number of Sweeps') 
plt.legend()
plt.show() 

