# Coefficient field inversion in an elliptic partial differential equation
# 
# Consider the following problem:
#
# min_a J(a):=1/2 int_Omega (u-ud)^2 dx +gamma/2 int_Omega | grad a|^2 dx
# 
# where u is the solution of
#
# -div (exp{a} grad u) = f in Omega,
#               u = 0 on partial Omega.
# 
# Here a  the unknown coefficient field, ud denotes (possibly noisy) data, $f\in H^{-1}(Omega)$ a given force, and $ gamma >= 0$ the regularization parameter.


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

# create mesh and define function spaces
nx = 64
ny = 64
mesh = UnitSquareMesh(nx, ny)
Va = FunctionSpace(mesh, 'Lagrange', 1)
Vu = FunctionSpace(mesh, 'Lagrange', 2)

# The true and inverted parameter
atrue = interpolate(Expression('log(2 + 7*(pow(pow(x[0] - 0.5,2) + pow(x[1] - 0.5,2),0.5) > 0.2))'),Va)
a = interpolate(Expression("log(2.0)"),Va)

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

# noise level
noise_level = 0.05

# weak form for setting up the synthetic observations
a_goal = inner(exp(atrue) * nabla_grad(u_trial), nabla_grad(u_test)) * dx
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


# Regularization parameter
gamma = 1e-8

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

# weak form for setting up the state equation
a_state = inner(exp(a) * nabla_grad(u_trial), nabla_grad(u_test)) * dx
L_state = f * u_test * dx

# weak form for setting up the adjoint equation
a_adj = inner(exp(a) * nabla_grad(p_trial), nabla_grad(p_test)) * dx
L_adj = -inner(u - ud, p_test) * dx

# weak form for setting up matrices
Wua_equ = inner(exp(a) * a_trial * nabla_grad(p_test), nabla_grad(p)) * dx
C_equ   = inner(exp(a) * a_trial * nabla_grad(u), nabla_grad(u_test)) * dx
Raa_equ = inner(exp(a) * a_trial * a_test *  nabla_grad(u),  nabla_grad(p)) * dx

M_equ   = inner(a_trial, a_test) * dx

# assemble matrix M
M = assemble(M_equ)

# solve state equation
state_A, state_b = assemble_system (a_state, L_state, bc_state)
solve (state_A, u.vector(), state_b)

# evaluate cost
[cost_old, misfit_old, reg_old] = cost(u, ud, a, W, R)

# plot
plt.figure(figsize=(15,5))
nb.plot(a,subplot_loc=121, mytitle="a_ini", vmin=atrue.vector().min(), vmax=atrue.vector().max())
nb.plot(u,subplot_loc=122, mytitle="u(a_ini)")


# define (Gauss-Newton) Hessian apply H * v
def Hess_GN (v, R, C, A, adj_A, W):
    rhs = -(C * v)
    bc_adj.apply(rhs)
    solve (A, du, rhs)
    rhs = - (W * du)
    bc_adj.apply(rhs)
    solve (adj_A, dp, rhs)
    CT_dp = Vector()
    C.init_vector(CT_dp, 1)
    C.transpmult(dp, CT_dp)
    H_V = R * v + CT_dp
    return H_V

# define (Newton) Hessian apply H * v
def Hess_Newton (v, R, Raa, C, A, adj_A, W, Wua):
    rhs = -(C * v)
    bc_adj.apply(rhs)
    solve (A, du, rhs)
    rhs = -(W * du) -  Wua * v
    bc_adj.apply(rhs)
    solve (adj_A, dp, rhs)
    CT_dp = Vector()
    C.init_vector(CT_dp, 1)
    C.transpmult(dp, CT_dp)
    Wua_du = Vector()
    Wua.init_vector(Wua_du, 1)
    Wua.transpmult(du, Wua_du)
    H_V = R*v + Raa*v + CT_dp + Wua_du
    return H_V

# Creat Class MyLinearOperator to perform Hessian function
class MyLinearOperator(LinearOperator):
    cgiter = 0
    def __init__(self, R, Raa, C, A, adj_A, W, Wua):
        LinearOperator.__init__(self, a_delta, a_delta)
        self.R = R
        self.Raa = Raa
        self.C = C
        self.A = A
        self.adj_A = adj_A
        self.W = W
        self.Wua = Wua

    # Hessian performed on x, output as generic vector y
    def mult(self, x, y):
        self.cgiter += 1
        y.zero()
        if iter <= 6:
            y.axpy(1., Hess_GN (x, self.R, self.C, self.A, self.adj_A, self.W) )
        else:
            y.axpy(1., Hess_Newton (x, self.R, self.Raa, self.C, self.A, self.adj_A, self.W, self.Wua) )

# define parameters for the optimization
tol = 1e-8
c = 1e-4
maxiter = 12
plot_on = False

# initialize iter counters
iter = 1
total_cg_iter = 0
converged = False

# initializations
g, a_delta = Vector(), Vector()
R.init_vector(a_delta,0)
R.init_vector(g,0)

du, dp = Vector(), Vector()
W.init_vector(du,1)
W.init_vector(dp,0)

a_prev, a_diff = Function(Va), Function(Va)

print "Nit   CGit   cost          misfit        reg           sqrt(-G*D)    ||grad||       alpha  tolcg"

while iter <  maxiter and not converged:

    # assemble matrix C
    C =  assemble(C_equ)

    # solve the adoint problem
    adjoint_A, adjoint_RHS = assemble_system(a_adj, L_adj, bc_adj)
    solve(adjoint_A, p.vector(), adjoint_RHS)

    # assemble W_ua and R
    Wua = assemble (Wua_equ)
    Raa = assemble (Raa_equ)

    # evaluate the  gradient
    CT_p = Vector()
    C.init_vector(CT_p,1)
    C.transpmult(p.vector(), CT_p)
    MG = CT_p + R * a.vector()
    solve(M, g, MG)

    # calculate the norm of the gradient
    grad2 = g.inner(MG)
    gradnorm = sqrt(grad2)

    # set the CG tolerance (use Eisenstat - Walker termination criterion)
    if iter == 1:
        gradnorm_ini = gradnorm
    tolcg = min(0.5, sqrt(gradnorm/gradnorm_ini))

    # define the Hessian apply operator (with preconditioner)
    Hess_Apply = MyLinearOperator(R, Raa, C, state_A, adjoint_A, W, Wua )
    P = R + gamma * M
    solver = PETScKrylovSolver("cg", "amg")
    solver.set_operators(Hess_Apply, P)
    solver.parameters["relative_tolerance"] = tolcg
    #solver.parameters["error_on_nonconvergence"] = False
    solver.parameters["nonzero_initial_guess"] = False

    # solve the Newton system H a_delta = - MG
    solver.solve(a_delta, -MG)
    total_cg_iter += Hess_Apply.cgiter
    
    # linesearch
    alpha = 1
    descent = 0
    no_backtrack = 0
    a_prev.assign(a)
    while descent == 0 and no_backtrack < 10:
        a.vector().axpy(alpha, a_delta )

        # solve the state/forward problem
        state_A, state_b = assemble_system(a_state, L_state, bc_state)
        solve(state_A, u.vector(), state_b)

        # evaluate cost
        [cost_new, misfit_new, reg_new] = cost(u, ud, a, W, R)

        # check if Armijo conditions are satisfied
        if cost_new < cost_old + alpha * c * MG.inner(a_delta):
            cost_old = cost_new
            descent = 1
        else:
            no_backtrack += 1
            alpha *= 0.5
            a.assign(a_prev)  # reset a

    # calculate sqrt(-G * D)
    graddir = sqrt(- MG.inner(a_delta) )

    sp = ""
    print "%2d %2s %2d %3s %8.5e %1s %8.5e %1s %8.5e %1s %8.5e %1s %8.5e %1s %5.2f %1s %5.3e" % \
        (iter, sp, Hess_Apply.cgiter, sp, cost_new, sp, misfit_new, sp, reg_new, sp, \
         graddir, sp, gradnorm, sp, alpha, sp, tolcg)

    if plot_on:
        nb.multi1_plot([a,u,p], ["a","u","p"], same_colorbar=False)
        
    
    # check for convergence
    if gradnorm < tol and iter > 1:
        converged = True
        print "Newton's method converged in ",iter,"  iterations"
        print "Total number of CG iterations: ", total_cg_iter
        
    iter += 1
    
if not converged:
    print "Newton's method did not converge in ", maxiter, " iterations"

print "Time elapsed: ", time.clock()-start

nb.multi1_plot([atrue, a], ["atrue", "a"])
nb.multi1_plot([u,p], ["u","p"], same_colorbar=False)

plt.show()

