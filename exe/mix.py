#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans1d import * 

Ne = 10
p = 2
if (len(sys.argv)>1):
	Ne = int(sys.argv[1])
if (len(sys.argv)>2):
	p = int(sys.argv[2])

def Error(Ne, p, pp):
	xe = np.linspace(0,1,Ne+1)
	leg = LobattoBasis(p-1 if pp else p)
	lob = LobattoBasis(p if pp else p+1) 	
	phi_space = L2Space(xe, leg)
	J_space = H1Space(xe, lob) 
	eps = 1e-1
	sigma_t = lambda x: 1/eps 
	sigma_s = lambda x: 1/eps - eps 
	sigma_a = lambda x: sigma_t(x) - sigma_s(x) 
	Q = lambda x: 1

	Mt = -3*Assemble(J_space, MassIntegrator, sigma_t, 2*p+1)
	D = MixAssemble(phi_space, J_space, MixDivIntegrator, 1, 2*p+1)
	Ma = Assemble(phi_space, MassIntegrator, sigma_a, 2*p+1)
	f = AssembleRHS(phi_space, DomainIntegrator, Q, 2*p+1)

	Mt[0,0] -= 2
	Mt[-1,-1] -= 2

	A = sp.bmat([[Mt, D.transpose()], [D, Ma]]).tocsc()
	rhs = np.concatenate((np.zeros(J_space.Nu), f))

	x = spla.spsolve(A, rhs) 
	phi = GridFunction(phi_space)
	J = GridFunction(J_space)
	phi.data = x[J_space.Nu:]
	J.data = x[:J_space.Nu]

	if (pp==1):
		star_space = L2Space(xe, type(leg)(p))
		phi_star = GridFunction(star_space)
		lam_space = L2Space(xe, LegendreBasis(0))
		for e in range(phi_space.Ne):
			star_el = star_space.el[e] 
			phi_el = phi_space.el[e] 
			J_el = J_space.el[e] 
			lam_el = lam_space.el[e] 

			K = WeakPoissonIntegrator(star_el, lambda x: 1/3/sigma_t(x), 2*p+1)
			Ma = MassIntegrator(star_el, sigma_a, 2*p+1)
			f = DomainIntegrator(star_el, Q, 2*p+1)
			f += star_el.CalcShape(-1)*J.Interpolate(e, -1) \
				- star_el.CalcShape(1)*J.Interpolate(e, 1) 

			M1 = MixMassIntegrator(star_el, lam_el, lambda x: 1, 2*p+1)
			M2 = MixMassIntegrator(lam_el, phi_el, lambda x: 1, 2*p+1)

			A = np.block([[K+Ma, M1], 
				[M1.transpose(), np.zeros((lam_el.Nn, lam_el.Nn))]])
			rhs = np.concatenate((f, np.dot(M2, phi.GetDof(e))))

			x = np.linalg.solve(A, rhs) 
			local = x[:star_el.Nn]
			phi_star.SetDof(e, local) 
		return phi_star
	elif (pp==2):
		phi_star_space = L2Space(xe, type(leg)(p))
		J_star_space = H1Space(xe, type(lob)(p+1))
		lam_space = L2Space(xe, LegendreBasis(0))
		phi_star = GridFunction(phi_star_space)
		for e in range(Ne):
			phi_star_el = phi_star_space.el[e]
			lam_el = lam_space.el[e]
			Jstar_el = J_star_space.el[e]
			phi_el = phi_space.el[e]
			J_el = J_space.el[e] 

			qorder = 2*(p+1) + 1
			Mt = -3*MassIntegrator(Jstar_el, sigma_t, qorder)
			D = MixDivIntegrator(phi_star_el, Jstar_el, 1, qorder)
			Ma = MassIntegrator(phi_star_el, sigma_a, qorder)
			f = DomainIntegrator(phi_star_el, Q, qorder)

			M1 = MixMassIntegrator(phi_star_el, lam_el, lambda x: 1, qorder)
			M2 = MixMassIntegrator(lam_el, phi_el, lambda x: 1, qorder)

			A = np.block([
				[Mt, D.transpose(), np.zeros((Jstar_el.Nn, lam_el.Nn))], 
				[D, Ma, M1],
				[np.zeros((lam_el.Nn, Jstar_el.Nn)), M1.transpose(), 
				np.zeros((lam_el.Nn, lam_el.Nn))]])
			rhs = np.concatenate((np.zeros(Jstar_el.Nn), f, np.dot(M2, phi.GetDof(e))))
			A[0,:] = 0 
			A[0,0] = 1 
			A[Jstar_el.Nn-1,:] = 0 
			A[Jstar_el.Nn-1,Jstar_el.Nn-1] = 1 
			rhs[0] = J.Interpolate(e,-1)
			rhs[Jstar_el.Nn-1] = J.Interpolate(e,1) 
			x = np.linalg.solve(A, rhs) 
			phi_star.SetDof(e, x[Jstar_el.Nn:Jstar_el.Nn+phi_star_el.Nn])

		return phi_star 

	else:
		return phi 

phi_ex = lambda x: np.sin(np.pi*x) 
phi1 = Error(Ne, p, 0)
phi2 = Error(Ne, p, 1) 
phi3 = Error(Ne, p, 2)

err1 = phi1.L2Error(phi_ex, 2*p+2)
err2 = phi2.L2Error(phi_ex, 2*p+2)
err3 = phi3.L2Error(phi_ex, 2*p+2)
print('err1 = {:.3e}\nerr2 = {:.3e}\nerr3 = {:.3e}'.format(err1, err2, err3))
print('diff21 = {:.3e}'.format(phi1.L2Diff(phi2, 2*p+1)))
print('diff31 = {:.3e}'.format(phi3.L2Diff(phi1, 2*p+1)))

# plt.semilogy(phi1.space.x, np.fabs(phi1.data - phi2.data))
plt.plot(phi1.space.x, phi1.data, '-o')
plt.plot(phi2.space.x, phi2.data, '-o')
plt.show()