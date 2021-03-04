#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans1d import * 

def SIPIntegrator(face_t, c):
	kappa = c[0] 
	scale = c[1] 
	xi1 = face_t.IPTrans(0)
	xi2 = face_t.IPTrans(1)

	s1 = face_t.el1.CalcShape(xi1)
	s2 = face_t.el2.CalcShape(xi2)
	j = np.concatenate((s1, -s2))

	gs1 = face_t.el1.CalcPhysGradShape(xi1)
	gs2 = face_t.el2.CalcPhysGradShape(xi2)
	ga = np.concatenate((gs1, gs2)) * (1 if face_t.boundary else .5) * face_t.nor 
	jga = np.outer(j,ga)
	return -jga - jga.T + kappa*np.outer(j,j)/(face_t.el1.h if scale else 1)

def CDG(face, c):
	sigma = c[0] 
	eta = c[1] 
	xi1 = face.IPTrans(0)
	xi2 = face.IPTrans(1)
	s1 = face.el1.CalcShape(xi1)
	s2 = face.el2.CalcShape(xi2)
	X1 = face.el1.Transform(0)
	X2 = face.el2.Transform(0)
	sigma1 = sigma(X1)
	sigma2 = sigma(X2) 
	bfac = 1 if face.boundary else .5 

	jump = np.concatenate((s1, -s2))
	avg = np.concatenate((s1, s2)) * bfac * face.nor 
	A = np.outer(avg, jump)

	minv1 = InverseMassIntegrator(face.el1, sigma, 2*face.el1.basis.p + 1)
	minv2 = InverseMassIntegrator(face.el2, sigma, 2*face.el2.basis.p + 1)
	z12 = np.zeros((minv1.shape[0], minv2.shape[1]))
	Minv = np.block([[minv1, z12], [z12.T, minv2]])
	return eta*np.linalg.multi_dot([A.T, Minv, A])

Ne = 10
p = 2

flux = []
lift = []
cdg = [] 
sip = []
for Ne in [10,20]:
	xe = np.linspace(0,1,Ne+1)
	sfes = L2Space(xe, LegendreBasis(p))
	vfes = L2Space(xe, LegendreBasis(p))
	kappa = (p+1)**2
	# kappa = 0
	eta = 1

	sigma = lambda x: 1

	Minv = Assemble(vfes, InverseMassIntegrator, sigma, 2*p+2)
	D = MixAssemble(sfes, vfes, WeakMixDivIntegrator, 1, 2*p+2) + MixFaceAssembleAll(sfes, vfes, MixJumpAvgIntegrator, 1)
	G = MixAssemble(vfes, sfes, WeakMixDivIntegrator, 1, 2*p+2) + MixFaceAssemble(vfes, sfes, MixJumpAvgIntegrator, 1)

	P = FaceAssembleAll(sfes, JumpJumpIntegrator, kappa)
	b = AssembleRHS(sfes, DomainIntegrator, lambda x: np.pi**2*np.sin(np.pi*x), 2*p+1)

	S = P - D*Minv*G
	B = MixFaceAssembleAll(sfes, vfes, MixJumpAvgIntegrator, 1)
	R = B*Minv*B.T
	K = Assemble(sfes, WeakPoissonIntegrator, sigma, 2*p+2) + FaceAssembleAll(sfes, SIPIntegrator, [kappa, False])
	Kh = Assemble(sfes, WeakPoissonIntegrator, sigma, 2*p+2) + FaceAssembleAll(sfes, SIPIntegrator, [kappa, True])

	Kc = K + FaceAssembleAll(sfes, CDG, [sigma, eta])
	Kl = K + R

	phi_ex = lambda x: np.sin(np.pi*x)
	phi_lift = GridFunction(sfes)
	phi_lift.data = spla.spsolve(Kl,b)
	lift.append(phi_lift.L2Error(phi_ex, 2*p+2))

	phi_cdg = GridFunction(sfes)
	phi_cdg.data = spla.spsolve(Kc,b)
	cdg.append(phi_cdg.L2Error(phi_ex, 2*p+2))

	phi_flux = GridFunction(sfes)
	phi_flux.data = spla.spsolve(S,b)
	flux.append(phi_flux.L2Error(phi_ex, 2*p+2))

	phi_sip = GridFunction(sfes)
	phi_sip.data = spla.spsolve(Kh,b)
	sip.append(phi_sip.L2Error(phi_ex, 2*p+2))

	x,subel = phi_lift.EvalSubEl()
	plt.plot(x,subel, label='Lift')
	x,subel = phi_flux.EvalSubEl()
	plt.plot(x,subel, label='Flux')
	x,subel = phi_cdg.EvalSubEl()
	plt.plot(x,subel, label='CDG')
	x,subel = phi_sip.EvalSubEl()
	plt.plot(x,subel, label='IP')
	plt.plot(x, phi_ex(x), '--')
	plt.legend()
	plt.show()

	# diff = phi.L2Diff(phi2, 2*p+2)
	# print('diff = {:.3e}'.format(diff))

print_err = lambda err, name: print(name + ' = {:.3f} ({:.3e}, {:.3e})'.format(np.log2(err[0]/err[1]), err[0], err[1]))
print_err(lift, 'lift')
print_err(flux, 'flux')
print_err(cdg, 'cdg')
print_err(sip, 'sip')