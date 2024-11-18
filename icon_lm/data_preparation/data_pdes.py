import jax.numpy as jnp
from functools import partial
from einshape import jax_einshape as einshape
from collections import namedtuple
import jax
import data_utils as dutils 
jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=("N"))
def solve_poisson(L, N, u_left, u_right, c):
	'''
	du/dxx = c over domain [0,L]
	c: spatially varying function, size N-1,
	u_left, u_right: boundary conditions. 
	the output is the full solution, (N+1) grid point values.  
	'''
	dx = L / N
	# x = jnp.linspace(0, L, N+1)

	# finite difference matrix
	du = jnp.array([1.0] * (N-2) + [0.0])
	dl =  jnp.array([0.0] + [1.0] * (N-2))
	d = - 2.0 * jnp.ones((N-1,))

	b = c*dx*dx
	b = b.at[0].add(-u_left)
	b = b.at[-1].add(-u_right)

	out_u = dutils.tridiagonal_solve(dl, d, du, b)
	u = jnp.pad(out_u, (1, 1), mode='constant', constant_values=(u_left, u_right))
	return u

@partial(jax.jit, static_argnames=("N"))
def solve_linRD(L, N, u_left, u_right, a, k, c):
	'''
	- a u_xx + k(x) u = c, a > 0, k(x) > 0
	over domain [0,L]
	a, c : constants
	k(x) : spatially varying coefficient, size (N-1,), should be positive
	u_left, u_right:  u(0)=b0, u(1)=b1, boundary conditions. 
	the output is the full solution, (N+1) grid point values.  
	'''
	dx = L / N
	# x = jnp.linspace(0, L, N+1)

	# finite difference matrix
	du = - a * jnp.array([1.0] * (N-2) + [0.0])
	dl = - a * jnp.array([0.0] + [1.0] * (N-2))
	d =  (a * 2.0) * jnp.ones((N-1,)) + k * dx * dx

	b = c*dx*dx*jnp.ones((N-1,))
	b = b.at[0].add(a * u_left)
	b = b.at[-1].add(a * u_right)

	out_u = dutils.tridiagonal_solve(dl, d, du, b)
	u = jnp.pad(out_u, (1, 1), mode='constant', constant_values=(u_left, u_right))
	return u

@partial(jax.jit, static_argnames=("N"))
def solve_square(L, N, u, u_left, u_right, a, k):
	'''
	- a u_xx + k u^2 = c(x), a > 0, k > 0
	over domain [0,L]
	u_left, u_right, a, k : constant parameters
	c(x) : spatially varying coefficient, size (N+1,)
	u_left, u_right:  u(0)=b0, u(1)=b1, boundary conditions. 
	the output is the full solution and k, (N+1) grid point values.  
	u: a given profile (possibly a GP), need to be matched with u_left, u_right
		size [N+1,]
	'''
	dx = L / N
	new_u = u + jnp.linspace(u_left - u[0], u_right - u[-1], N+1)
	uxx = dutils.laplace_u(new_u, dx)
	c = -a * uxx+ k * new_u **2
	return new_u,c
		
@partial(jax.jit, static_argnames=("N"))
def solve_nonlinRD(L, N, u, u_left, u_right, a, k):
	'''
	- a u_xx + k u^3 = c(x), a > 0, k > 0
	over domain [0,L]
	u_left, u_right, a, k : constant parameters
	c(x) : spatially varying coefficient, size (N+1,)
	u_left, u_right:  u(0)=b0, u(1)=b1, boundary conditions. 
	the output is the full solution and k, (N+1) grid point values.  
	u: a given profile (possibly a GP), need to be matched with u_left, u_right
		size [N+1,]
	'''
	dx = L / N
	new_u = u + jnp.linspace(u_left - u[0], u_right - u[-1], N+1)
	uxx = dutils.laplace_u(new_u, dx)
	c = -a * uxx+ k * new_u **3
	return new_u,c

@partial(jax.jit, static_argnames=("N_t","N_x"))
def solve_lin2d(L_x, L_t, N_x, N_t, uxt_GP, coeffs):
	'''
	To be honest, this is not really 'solving' anything. 
	It is just applying the derivatives and the coefficients to the given u(x,t) to find g(x,t).
	That's why it returns g(x,t) instead of u(x,t).
	a*u_xx + b*u_xt + c*u_tt + d*u_x + e*u_t + f*u = g(x,t)
	over domain [0,L_t] x [0,L_x]
	coeffs: [a,b,c,d,e,f], constant parameters
	c(x,t): spatially & temporally varying function, size (N_t-1, N_x-1)
	'''

	# new_uxt_GP = dutils.apply_initial_conditions(uxt_GP, u_left, u_right) # (N_x+1, N_t+1)

	dx = L_x / N_x
	dt = L_t / N_t
	laplace_u_2d_x  = jax.vmap(dutils.laplace_u, in_axes=(0, None))
	laplace_u_2d_t = jax.vmap(dutils.laplace_u, in_axes=(1, None))
	gradient_u_2d_x = jax.vmap(dutils.gradient_u, in_axes=(0, None))
	gradient_u_2d_t = jax.vmap(dutils.gradient_u, in_axes=(1, None))
	u_xx = laplace_u_2d_x(uxt_GP, dx)
	u_tt = laplace_u_2d_t(uxt_GP.T, dt).T
	u_x = gradient_u_2d_x(uxt_GP, dx)
	u_t = gradient_u_2d_t(uxt_GP.T, dt).T
	u_xt = dutils.mixed_partial_derivative(uxt_GP, dx, dt)
	a, b, c, d, e, f = coeffs
	g = a * u_xx + b * u_xt + c * u_tt + d * u_x + e * u_t + f * uxt_GP # (N_x+1, N_t+1)
	return uxt_GP, g

@partial(jax.jit, static_argnames=("domlen", "trajlen"))
def solve_heat_homog(initial_condition, domlen, trajlen, k, ul, ur):
	'''Solve 1D homog. heat equation using explicit finite difference method
	@params:
		initial_condition: (num, trajlen), u(x,0)
		domlen: float, length of domain (for simplicity, insist that domlen of x = domlen of t)
		trajlen: int, number of spatial points (same with trajlen)
		dt: float, time step size
		k: float, thermal conductivity coefficient
		ul, ur: floats, boundary conditions
	returns:  
		u: array, (trajlen,trajlen)
	'''
	dx = dt = domlen / (trajlen)
	u = jnp.zeros((trajlen, trajlen)) #initialize temp array
	u = u.at[0,:].set(initial_condition)
	u = u.at[:,[0,-1]].set([ul, ur])

	for n in range(0, trajlen-1):
		for i in range(1, trajlen-1):
			u = u.at[n+1,i].set(u[n, i] + k * dt / (dx * dx) * (u[n, i+1] - 2 * u[n, i] + u[n, i-1]))
	return u

@partial(jax.jit, static_argnames=("domlen", "trajlen"))
def solve_heat_nonhomog(initial_condition, domlen, trajlen, k, a, ul, ur):
	'''Solve 1D nonhomog. heat equation using explicit finite difference method
	@params:
		initial_condition: (num, trajlen), u(x,0)
		domlen: float, length of domain (for simplicity, insist that domlen of x = domlen of t)
		trajlen: int, number of spatial points (same with trajlen)
		dt: float, time step size
		k: float, thermal conductivity coefficient
		a: float, heat transfer coefficient
		ul, ur: floats, boundary conditions
	returns:  
		u: array, (trajlen,trajlen)
	'''
	dx = dt = domlen/trajlen
	u = jnp.zeros((trajlen, trajlen)) #initialize temp array
	u = u.at[0,:].set(initial_condition)
	u = u.at[:,[0,-1]].set([ul,ur])

	for n in range(0, trajlen-1):
		for i in range(1, trajlen-1):
			diffusion = k * dt / (dx*dx) * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
			source = a * dt * u[n, i]
			u = u.at[n+1, i].set(u[n, i] + diffusion + source)
	return u

solve_poisson_batch = jax.jit(jax.vmap(solve_poisson, in_axes=(None, None, None, None, 0)), static_argnums=(1,))
solve_linRD_batch = jax.jit(jax.vmap(solve_linRD, in_axes=(None, None, None, None, None, 0, None)), static_argnums=(1,))
solve_square_batch = jax.jit(jax.vmap(solve_square, in_axes=(None, None, 0, None, None, None, None)), static_argnums=(1,))
solve_nonlinRD_batch = jax.jit(jax.vmap(solve_nonlinRD, in_axes=(None, None, 0, None, None, None, None)), static_argnums=(1,))
solve_lin2d_batch = jax.jit(jax.vmap(solve_lin2d, in_axes=(None, None, None, None, 0, None, None, None)), static_argnums=(0,1,2,3))
solve_heat_homog_batch = jax.jit(jax.vmap(solve_heat_homog, in_axes=(0, None, None, None, None, None)), static_argnums=(1,2))
solve_heat_nonhomog_batch = jax.jit(jax.vmap(solve_heat_nonhomog, in_axes=(0, None, None, None, None, None, None)), static_argnums=(1,2))

if __name__ == "__main__":
	import numpy as np
	out = solve_poisson(L = 1, N = 100, u_left = 1, u_right = 1, c = np.ones((99,)))
	print(out.shape, out)

	k_spatial = np.zeros((99,))
	out_porous = solve_linRD(L = 1, N = 100, u_left = 1, u_right = 1, a = -1.0, k = k_spatial, c = 1.0)
	assert np.allclose(out, out_porous)

	out_poc = solve_linRD(L = 1, N = 100, u_left = 1, u_right = 1, a = 0, k = np.ones((99,)), c = 1.0)
	assert np.allclose(out_poc, np.ones((101,), dtype=np.float64))

	for i in range(5):
		a = np.random.uniform(0.5, 1)
		k = np.random.uniform(0.5, 1, size=(10,99))
		c = np.random.uniform(-1, 1)
		out_batch = solve_linRD_batch(1, 100, 1, 1, a, k, c)
		for j in range(10):
			out = solve_linRD(L = 1, N = 100, u_left = 1, u_right = 1, a = a, k = k[j], c = c)
			assert np.allclose(out, out_batch[j])
			res = -a * (out[:-2] + out[2:] - 2*out[1:-1])/(0.01)**2 + k[j] * out[1:-1] - c
			assert np.allclose(res, np.zeros((99,), dtype=np.float64))


	x = np.linspace(0, 1, 101)
	u_input = np.cos(x)
	[new_u,out_c] = solve_square(L = 1, N = 100, u = u_input, u_left = 1, u_right = 0.1, a = -1.0, k = 1.0)
	print(out_c.shape, out_c)
	assert np.allclose(new_u[0], 1.0)
	assert np.allclose(new_u[-1], 0.1)

	x = np.linspace(0, 1, 101)
	u_input = np.cos(x)
	[new_u,out_c] = solve_nonlinRD(L = 1, N = 100, u = u_input, u_left = 1, u_right = 0.1, a = -1.0, k = 1.0)
	print(out_c.shape)
	assert np.allclose(new_u[0], 1.0)
	assert np.allclose(new_u[-1], 0.1)
