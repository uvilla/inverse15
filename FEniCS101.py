# FEniCS101 Tutorial
# 
# In this tutorial we consider the boundary value problem (BVP)
#
# - div (k grad u) = f     in Omega,
#                u = u0    on Gamma_D = Gamma_left U Gamma_right
#     k grad u . n = sigma on Gamma_N = Gamma_top  U Gamma_bottom,
# 
# where Omega = (0,1)^2, Gamma_D and and Gamma_N are the union of
# the left and right, and top and bottom boundaries of Omega, respectively.
#
# The diffusivity coefficient, forcing term and boundary conditions are chosen
# such that exact solution is
# $$ u_e(x,y) = \sin(2\pi x)\sin\left(\frac{\pi}{2}y\right). $$

# 1. Import modules

from dolfin import *

import math
import numpy as np
import logging

import matplotlib.pyplot as plt
import nb

logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
set_log_active(False)

# 2. Define the mesh and the finite element space

n = 16
degree = 1
mesh = RectangleMesh(0, 0, 1, 1, n, n)
nb.plot(mesh)

Vh  = FunctionSpace(mesh, 'Lagrange', degree)
print "dim(Vh) = ", Vh.dim()

# 3. Define boundary labels

class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 1) < DOLFIN_EPS
    
class BottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1]) < DOLFIN_EPS
    
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < DOLFIN_EPS
    
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 1) < DOLFIN_EPS
    
boundary_parts = FacetFunction("size_t", mesh)
boundary_parts.set_all(0)

Gamma_top = TopBoundary()
Gamma_top.mark(boundary_parts, 1)
Gamma_bottom = BottomBoundary()
Gamma_bottom.mark(boundary_parts, 2)
Gamma_left = LeftBoundary()
Gamma_left.mark(boundary_parts, 3)
Gamma_right = RightBoundary()
Gamma_right.mark(boundary_parts, 4)

# 4. Define the coefficients of the PDE and the boundary conditions


u_L = Constant(0.)
u_R = Constant(0.)

sigma_bottom = Expression('-(pi/2.0)*sin(2*pi*x[0])')
sigma_top    = Expression('0')

f = Expression('(4.0*pi*pi+pi*pi/4.0)*(sin(2*pi*x[0])*sin((pi/2.0)*x[1]))')

bcs = [DirichletBC(Vh, u_L, boundary_parts, 3),
       DirichletBC(Vh, u_R, boundary_parts, 4)]

ds = Measure("ds", subdomain_data=boundary_parts)



# 5. Define and solve the variational problem

u = TrialFunction(Vh)
v = TestFunction(Vh)
a = inner(nabla_grad(u), nabla_grad(v))*dx
L = f*v*dx + sigma_top*v*ds(1) + sigma_bottom*v*ds(2)

uh = Function(Vh)

#solve(a == L, uh, bcs=bcs)
A, b = assemble_system(a,L, bcs=bcs)
solve(A, uh.vector(), b, "cg")

nb.plot(uh)

# <markdowncell>

# 6. Compute the discretization error

u_e = Expression('sin(2*pi*x[0])*sin((pi/2.0)*x[1])')
grad_u_e = Expression( ('2*pi*cos(2*pi*x[0])*sin((pi/2.0)*x[1])', 'pi/2.0*sin(2*pi*x[0])*cos((pi/2.0)*x[1])'))

err_L2 = sqrt( assemble( (uh-u_e)**2*dx ) )
err_grad = sqrt( assemble( inner(nabla_grad(uh) - grad_u_e, nabla_grad(uh) - grad_u_e)*dx ) )
err_H1 = sqrt( err_L2**2 + err_grad**2)

print "|| u_h - u_e ||_L2 = ", err_L2
print "|| u_h - u_e ||_H1 = ", err_H1

# 7. Convergence of the finite element method

def compute(n, degree):
    mesh = RectangleMesh(0, 0, 1, 1, n, n)
    Vh  = FunctionSpace(mesh, 'Lagrange', degree)
    boundary_parts = FacetFunction("size_t", mesh)
    boundary_parts.set_all(0)
    
    Gamma_top = TopBoundary()
    Gamma_top.mark(boundary_parts, 1)
    Gamma_bottom = BottomBoundary()
    Gamma_bottom.mark(boundary_parts, 2)
    Gamma_left = LeftBoundary()
    Gamma_left.mark(boundary_parts, 3)
    Gamma_right = RightBoundary()
    Gamma_right.mark(boundary_parts, 4)
    
    bcs = [DirichletBC(Vh, u_L, boundary_parts, 3), DirichletBC(Vh, u_R, boundary_parts, 4)]
    ds = Measure("ds", subdomain_data=boundary_parts)
    
    u = TrialFunction(Vh)
    v = TestFunction(Vh)
    a = inner(nabla_grad(u), nabla_grad(v))*dx
    L = f*v*dx + sigma_top*v*ds(1) + sigma_bottom*v*ds(2)
    uh = Function(Vh)
    solve(a == L, uh, bcs=bcs)
    err_L2 = sqrt( assemble( (uh-u_e)**2*dx ) )
    err_grad = sqrt( assemble( inner(nabla_grad(uh) - grad_u_e, nabla_grad(uh) - grad_u_e)*dx ) )
    err_H1 = sqrt( err_L2**2 + err_grad**2)
    
    return err_L2, err_H1

nref = 5
n = 8*np.power(2,np.arange(0,nref))
h = 1./n

err_L2_P1 = np.zeros(nref)
err_H1_P1 = np.zeros(nref)
err_L2_P2 = np.zeros(nref)
err_H1_P2 = np.zeros(nref)

for i in range(nref):
    err_L2_P1[i], err_H1_P1[i] = compute(n[i], 1)
    err_L2_P2[i], err_H1_P2[i] = compute(n[i], 2)
    
plt.figure(figsize=(15,5))

plt.subplot(121)
plt.loglog(h, err_H1_P1, '-or')
plt.loglog(h, err_L2_P1, '-*b')
plt.loglog(h, h*.5*err_H1_P1[0]/h[0], '--g')
plt.loglog(h, np.power(h,2)*.5*np.power( err_L2_P1[0]/h[0], 2), '-.k')
plt.xlabel("Mesh size h")
plt.ylabel("Error")
plt.title("P1 Finite Element")
plt.legend(["H1 error", "L2 error", "First Order", "Second Order"], 'lower right')


plt.subplot(122)
plt.loglog(h, err_H1_P2, '-or')
plt.loglog(h, err_L2_P2, '-*b')
plt.loglog(h, np.power(h/h[0],2)*.5*err_H1_P2[0], '--g')
plt.loglog(h, np.power(h/h[0],3)*.5*err_L2_P2[0], '-.k')
plt.xlabel("Mesh size h")
plt.ylabel("Error")
plt.title("P2 Finite Element")
plt.legend(["H1 error", "L2 error", "Second Order", "Third Order"], 'lower right')

plt.show()

