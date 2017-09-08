# Coefficient field inversion in an elliptic partial differential equation
# 
# Consider the following problem:
#
# min_a J(a):=1/2 int_Omega (u-ud)^2 dx +gamma/2 int_Omega | grad a|^2 dx
# 
# where u is the solution of
#
# -div (a grad u) = f in Omega,
#               u = 0 on partial Omega.
# 
# Here a  the unknown coefficient field, ud denotes (possibly noisy) data, $f\in H^{-1}(Omega)$ a given force, and $ gamma >= 0$ the regularization parameter.

# 1. Import dependencies

from dolfin import *

import numpy as np
import time
import logging

import matplotlib.pyplot as plt
import nb

start = time.clock()

logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
set_log_active(False)

np.random.seed(seed=1)



# 2. Model set up:

# create mesh and define function spaces
nx = 32
ny = 32
mesh = UnitSquareMesh(nx, ny)
Va = FunctionSpace(mesh, 'Lagrange', 1)
Vu = FunctionSpace(mesh, 'Lagrange', 2)

# The true and inverted parameter
atrue = interpolate(Expression('8. - 4.*(pow(x[0] - 0.5,2) + pow(x[1] - 0.5,2) < pow(0.2,2))'), Va)
a = interpolate(Expression("4."),Va)

# define function for state and adjoint
u = Function(Vu)
p = Function(Vu)

# define Trial and Test Functions
u_trial, p_trial, a_trial = TrialFunction(Vu), TrialFunction(Vu), TrialFunction(Va)
u_test, p_test, a_test = TestFunction(Vu), TestFunction(Vu), TestFunction(Va)

# initialize input functions
f = Constant("1.0")
u0 = Constant("0.0")

# plot
plt.figure(figsize=(15,5))
nb.plot(mesh,subplot_loc=121, mytitle="Mesh", show_axis='on')
nb.plot(atrue,subplot_loc=122, mytitle="True parameter field")

# set up dirichlet boundary conditions
def boundary(x,on_boundary):
    return on_boundary

bc_state = DirichletBC(Vu, u0, boundary)
bc_adj = DirichletBC(Vu, Constant(0.), boundary)

# 3. The cost functional evaluation:

# Regularization parameter
gamma = 1e-10

# weak for for setting up the misfit and regularization compoment of the cost
W_equ   = inner(u_trial, u_test) * dx
R_equ   = gamma * inner(nabla_grad(a_trial), nabla_grad(a_test)) * dx

W = assemble(W_equ)
R = assemble(R_equ)

# Define cost function
def cost(u, ud, a, W, R):
    diff = u.vector() - ud.vector()
    reg = 0.5 * a.vector().inner(R*a.vector() ) 
    misfit = 0.5 * diff.inner(W * diff)
    return [reg + misfit, misfit, reg]

# 4. Set up synthetic observations:

# noise level
noise_level = 0.01

# weak form for setting up the synthetic observations
a_goal = inner( atrue * nabla_grad(u_trial), nabla_grad(u_test)) * dx
L_goal = f * u_test * dx

# solve the forward/state problem to generate synthetic observations
goal_A, goal_b = assemble_system(a_goal, L_goal, bc_state)

utrue = Function(Vu)
solve(goal_A, utrue.vector(), goal_b)

ud = Function(Vu)
ud.assign(utrue)

# perturb state solution and create synthetic measurements ud
# ud = u + ||u||/SNR * random.normal
MAX = ud.vector().norm("linf")
noise = Vector()
goal_A.init_vector(noise,1)
noise.set_local( noise_level * MAX * np.random.normal(0, 1, len(ud.vector().array())) )
bc_adj.apply(noise)

ud.vector().axpy(1., noise)

# plot
nb.multi1_plot([utrue, ud], ["State solution with atrue", "Synthetic observations"])

# 5. Setting up the state equations, right hand side for the adjoint and the necessary matrices:

# weak form for setting up the state equation
a_state = inner( a * nabla_grad(u_trial), nabla_grad(u_test)) * dx
L_state = f * u_test * dx

# weak form for setting up the adjoint equations
a_adj = inner( a * nabla_grad(p_trial), nabla_grad(p_test) ) * dx
L_adjoint = -inner(u - ud, u_test) * dx


# weak form for setting up matrices
CT_equ   = inner(a_test * nabla_grad(u), nabla_grad(p_trial)) * dx
M_equ   = inner(a_trial, a_test) * dx


# assemble matrices M
M = assemble(M_equ)

# <markdowncell>

# 6. Initial guess

# solve state equation
A, state_b = assemble_system (a_state, L_state, bc_state)
solve (A, u.vector(), state_b)

# evaluate cost
[cost_old, misfit_old, reg_old] = cost(u, ud, a, W, R)

# plot
plt.figure(figsize=(15,5))
nb.plot(a,subplot_loc=121, mytitle="a_ini", vmin=atrue.vector().min(), vmax=atrue.vector().max())
nb.plot(u,subplot_loc=122, mytitle="u(a_ini)")

# 7. The steepest descent with Armijo line search

# define parameters for the optimization
tol = 1e-4
maxiter = 1000
c_armijo = 1e-5

# initialize iter counters
iter = 1
converged = False

# initializations
g = Vector()
R.init_vector(g,0)

a_prev = Function(Va)

print "Nit  cost          misfit        reg         ||grad||       alpha  N backtrack"

while iter <  maxiter and not converged:

    # assemble matrix C
    CT =  assemble(CT_equ)

    # solve the adoint problem
    adj_A, adjoint_RHS = assemble_system(a_adj, L_adjoint, bc_adj)
    solve(adj_A, p.vector(), adjoint_RHS)

    # evaluate the  gradient
    MG = CT*p.vector() + R * a.vector()
    solve(M, g, MG)

    # calculate the norm of the gradient
    grad_norm2 = g.inner(MG)
    gradnorm = sqrt(grad_norm2)
    
    if iter == 1:
        gradnorm0 = gradnorm

    # linesearch
    it_backtrack = 0
    a_prev.assign(a)
    alpha = 8.e5
    backtrack_converged = False
    for it_backtrack in range(20):
        
        a.vector().axpy(-alpha, g )

        # solve the state/forward problem
        state_A, state_b = assemble_system(a_state, L_state, bc_state)
        solve(state_A, u.vector(), state_b)

        # evaluate cost
        [cost_new, misfit_new, reg_new] = cost(u, ud, a, W, R)

        # check if Armijo conditions are satisfied
        if cost_new < cost_old - alpha * c_armijo * grad_norm2:
            cost_old = cost_new
            backtrack_converged = True
            break
        else:
            alpha *= 0.5
            a.assign(a_prev)  # reset a
            
    if backtrack_converged == False:
        print "Backtracking failed. A sufficient descent direction was not found"
        converged = False
        break

    sp = ""
    print "%3d %1s %8.5e %1s %8.5e %1s %8.5e %1s %8.5e %1s %8.5e %1s %3d" % \
        (iter, sp, cost_new, sp, misfit_new, sp, reg_new, sp, \
        gradnorm, sp, alpha, sp, it_backtrack)

    # check for convergence
    if gradnorm < tol*gradnorm0 and iter > 1:
        converged = True
        print "Steepest descent converged in ",iter,"  iterations"
        
    iter += 1
    
if not converged:
    print "Steepest descent did not converge in ", maxiter, " iterations"

print "Time elapsed: ", time.clock()-start

nb.multi1_plot([atrue, a], ["atrue", "a"])
nb.multi1_plot([u,p], ["u","p"], same_colorbar=False)
plt.show()

