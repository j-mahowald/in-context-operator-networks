import numpy as np
import jax.numpy as jnp
import jax
from einshape import jax_einshape as einshape
import pickle
from functools import partial
import sys
sys.path.append('../')
import utils
from absl import app, flags, logging
import haiku as hk
import matplotlib.pyplot as plt

def rbf_kernel_3d(x1, x2, k_sigma, k_l):
    '''
    RBF kernel for 3D Gaussian process
    '''
    sqdist = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    return k_sigma * jnp.exp(-0.5 / k_l**2 * sqdist)

def generate_gaussian_process_3d(key, xs, ts, num, kernel, k_sigma, k_l):
    '''
    ts: 1D array (length,)
    xs: 1D array (length,)
    out: Gaussian process samples, 3D array (num, length, length)
    suggested k_sigma = 1.0, k_l = 0.2, kernel = rbf_kernel_3d

    '''
    X, T = jnp.meshgrid(xs, ts)
    points = jnp.vstack([X.ravel(), T.ravel()]).T
    cov = kernel(points, points, k_sigma=k_sigma, k_l=k_l)
    cov += 1e-6 * jnp.eye(cov.shape[0]) # Add a small jitter for numerical stability
    mean = jnp.zeros(cov.shape[0])
    cov = einshape('ii->nii', cov, n=num)
    u_sample = jax.random.multivariate_normal(key, mean=mean, cov=cov, shape=(num,), method='svd')
    out = u_sample.reshape((num, X.shape[0], X.shape[1])) # (num, N_x+1, N_t+1)
    return out

uxt_GP = generate_gaussian_process_3d(jax.random.PRNGKey(0), jnp.linspace(0, 1, 101), jnp.linspace(0, 1, 101), 1, rbf_kernel_3d, 1.0, 0.2)
# Reduce the dimension of the GP sample to 2D
uxt_GP = uxt_GP[0]

@jax.jit
def gradient_u(u, dx):
    ux = (u[2:] - u[:-2]) / (2 * dx)
    ux_left = (-3 * u[0] + 4 * u[1] - u[2]) / (2 * dx)
    ux_right = (3 * u[-1] - 4 * u[-2] + u[-3]) / (2 * dx)
    ux = jnp.pad(ux, (1, 1), mode='constant', constant_values=(ux_left, ux_right))
    return ux
gradient_u_batch = jax.jit(jax.vmap(gradient_u, in_axes=(0, None)))

@jax.jit
def laplace_u(u, dx):
  uxx = (u[:-2] + u[2:] - 2*u[1:-1])/dx**2 
  uxx_left = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3])/dx**2
  uxx_right = (2 * u[-1] - 5 * u[-2] + 4 * u[-3] - u[-4])/dx**2
  uxx = jnp.pad(uxx, (1, 1), mode='constant', constant_values = (uxx_left, uxx_right))
  return uxx
laplace_u_batch = jax.jit(jax.vmap(laplace_u, in_axes=(0, None)))

def mixed_partial_derivative(u, dx, dt):
    # Central difference for interior points
    u_xt = (u[2:,2:] - u[2:,:-2] - u[:-2,2:] + u[:-2,:-2]) / (4 * dx * dt)
    
    # Handle boundaries
    # Left and right edges (excluding corners)
    u_xt_left = (-u[1:,1:-1] + u[1:,:-2] + u[:-1,1:-1] - u[:-1,:-2]) / (2 * dx * dt)
    u_xt_right = (u[1:,2:] - u[1:,1:-1] - u[:-1,2:] + u[:-1,1:-1]) / (2 * dx * dt)
    
    # Top and bottom edges (excluding corners)
    u_xt_top = (-u[1:-1,1:] + u[1:-1,:-1] + u[:-2,1:] - u[:-2,:-1]) / (2 * dx * dt)
    u_xt_bottom = (u[2:,1:] - u[2:,:-1] - u[1:-1,1:] + u[1:-1,:-1]) / (2 * dx * dt)
    
    # Corners
    u_xt_tl = (-u[1,1] + u[1,0] + u[0,1] - u[0,0]) / (dx * dt)
    u_xt_tr = (u[1,-1] - u[1,-2] - u[0,-1] + u[0,-2]) / (dx * dt)
    u_xt_bl = (u[-1,1] - u[-1,0] - u[-2,1] + u[-2,0]) / (dx * dt)
    u_xt_br = (-u[-1,-1] + u[-1,-2] + u[-2,-1] - u[-2,-2]) / (dx * dt)
    
    # Pad the interior
    u_xt = jnp.pad(u_xt, ((1,1), (1,1)), mode='constant')
    
    # Fill in the edges and corners
    u_xt = u_xt.at[1:, 0].set(u_xt_left[:,0])
    u_xt = u_xt.at[1:, -1].set(u_xt_right[:,-1])
    u_xt = u_xt.at[0, 1:].set(u_xt_top[0,:])
    u_xt = u_xt.at[-1, 1:].set(u_xt_bottom[-1,:])
    u_xt = u_xt.at[0, 0].set(u_xt_tl)
    u_xt = u_xt.at[0, -1].set(u_xt_tr)
    u_xt = u_xt.at[-1, 0].set(u_xt_bl)
    u_xt = u_xt.at[-1, -1].set(u_xt_br)
    
    return u_xt

def solve_pde_linear_3d(L_x, L_t, N_x, N_t, uxt_GP, coeffs):
    '''
    To be honest, this is not really 'solving' anything. 
    It is just applying the derivatives and the coefficients to the given u(x,t) to find g(x,t).
    That's why it returns g(x,t) instead of u(x,t).
    a*u_xx + b*u_xt + c*u_tt + d*u_x + e*u_t + f*u = g(x,t)
    over domain [0,L_t] x [0,L_x]
    coeffs: [a,b,c,d,e,f], constant parameters
    c(x,t): spatially & temporally varying function, size (N_t-1, N_x-1)
    no ul or ur parameters, boundary conditions are not considered
    '''
    dx = L_x / N_x
    dt = L_t / N_t
    laplace_u_2d_x = jax.vmap(laplace_u, in_axes=(0, None))
    laplace_u_2d_t = jax.vmap(laplace_u, in_axes=(1, None))
    gradient_u_2d_x = jax.vmap(gradient_u, in_axes=(0, None))
    gradient_u_2d_t = jax.vmap(gradient_u, in_axes=(1, None))
    u_xx = laplace_u_2d_x(uxt_GP, dx)
    u_tt = laplace_u_2d_t(uxt_GP.T, dt).T
    u_x = gradient_u_2d_x(uxt_GP, dx)
    u_t = gradient_u_2d_t(uxt_GP.T, dt).T
    u_xt = mixed_partial_derivative(uxt_GP, dx, dt)
    a, b, c, d, e, f = coeffs
    g = a * u_xx + b * u_xt + c * u_tt + d * u_x + e * u_t + f * uxt_GP
    return g

rng = jax.random.PRNGKey(0)
keys = jax.random.split(rng, 6)
eqns = 1
coeffs = [jax.random.uniform(keys[i], (eqns,), minval=-1, maxval=1) for i in range(6)]
a, b, c, d, e, f = coeffs

g = solve_pde_linear_3d(1, 1, 50, 50, uxt_GP, coeffs)

# Plot uxt_GP and g
import matplotlib.pyplot as plt
plt.rcdefaults()
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
cax0 = ax[0].imshow(uxt_GP, cmap='viridis', aspect='auto', origin='lower')
fig.colorbar(cax0, ax=ax[0])
ax[0].set_title('u(x,t)')
cax1 = ax[1].imshow(g, cmap='viridis', aspect='auto', origin='lower')
fig.colorbar(cax1, ax=ax[1])
ax[1].set_title('g(x,t)')
plt.show()
