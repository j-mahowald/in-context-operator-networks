from scipy.spatial.distance import cdist
import jax
import jax.numpy as jnp
from einshape import jax_einshape as einshape
from functools import partial

# Define the covariance function
def rbf_kernel(x1, x2, sigma, l):
    """
    Radial basis function kernel
    """
    sq_norm = cdist(x1 / l, x2 / l, metric='sqeuclidean')
    return sigma**2 * jnp.exp(-0.5 * sq_norm)

# Define the covariance function
def rbf_kernel_jax(x1, x2, sigma, l):
    """
    Radial basis function kernel, only support 1D x1 and x2
    """
    xx1, xx2 = jnp.meshgrid(x1, x2, indexing='ij')
    sq_norm = (xx1-xx2)**2/(l**2)
    return sigma**2 * jnp.exp(-0.5 * sq_norm)

# Define the covariance function
def rbf_sin_kernel_jax(x1, x2, sigma, l):
    """
    suppose x1, x2 in [0,1],
    """
    xx1, xx2 = jnp.meshgrid(x1, x2, indexing='ij')
    sq_norm = (jnp.sin(jnp.pi*(xx1-xx2)))**2/(l**2)
    return sigma**2 * jnp.exp(-0.5 * sq_norm)

def rbf_circle_kernel_jax(x1, x2, sigma, l):
    """
    suppose x1, x2 in [0,1],
    """
    xx1, xx2 = jnp.meshgrid(x1, x2, indexing='ij')
    xx1_1 = jnp.sin(xx1 * 2 * jnp.pi)
    xx1_2 = jnp.cos(xx1 * 2 * jnp.pi)
    xx2_1 = jnp.sin(xx2 * 2 * jnp.pi)
    xx2_2 = jnp.cos(xx2 * 2 * jnp.pi)
    sq_norm = (xx1_1-xx2_1)**2/(l**2) + (xx1_2-xx2_2)**2/(l**2)
    return sigma**2 * jnp.exp(-0.5 * sq_norm)

def rbf_kernel_3d(x1, x2, k_sigma, k_l):
    '''
    RBF kernel for 3D Gaussian process
    '''
    sqdist = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    return k_sigma * jnp.exp(-0.5 / k_l**2 * sqdist)

@partial(jax.jit, static_argnames=('num','kernel'))
def generate_gaussian_process(key, ts, num, kernel, k_sigma, k_l):
  '''
  ts: 1D array (length,)
  out: Gaussian process samples, 2D array (num, length)
  '''
  length = len(ts)
  mean = jnp.zeros((num,length))
  cov = kernel(ts, ts, sigma=k_sigma, l=k_l)
  cov = einshape('ii->nii', cov, n = num)
  out = jax.random.multivariate_normal(key, mean=mean, cov=cov, shape=(num,), method='svd')
  return out

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
    u_sample = jax.random.multivariate_normal(key, mean=mean, cov=cov, shape=(num,), method='eigh')
    out = u_sample.reshape((num, X.shape[0], X.shape[1])) # (num, N_x+1, N_t+1)
    return out

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

@jax.jit
def mixed_partial_derivative(u, dx, dt):
    # u is a 2D array of shape (N_x+1, N_t+1)

    # Central difference for interior points
    u_xt = (u[2:, 2:] - u[:-2, 2:] - u[2:, :-2] + u[:-2, :-2]) / (4 * dx * dt)

    # Top & bottom edges (no corner), 1D array of length (N_t-1)
    u_xt_top = (u[1, 2:] - u[0, 2:] - u[1, :-2] + u[0, :-2]) / (2 * dx * dt)
    u_xt_bottom = (u[-1, 2:] - u[-2, 2:] - u[-1, :-2] + u[-2, :-2]) / (2 * dx * dt)

    # Left & right edges (no corner), 1D array of length (N_x-1)
    u_xt_left = (u[2:,1] - u[:-2, 1] - u[2:, 0] + u[:-2, 0]) / (2 * dx * dt)
    u_xt_right = (u[2:, -1] - u[:-2, -1] - u[2:, -2] + u[:-2, -2]) / (2 * dx * dt)

    # Corners
    u_xt_tl = (u[1,1] - u[0,1] - u[1,0] + u[0,0]) / (dx * dt)
    u_xt_tr = (u[1, -1] - u[0, -1] - u[1, -2] + u[0, -2]) / (dx * dt)
    u_xt_br = (u[-1, -1] - u[-1, -2] - u[-2, -1] + u[-2, -2]) / (dx * dt)
    u_xt_bl = (u[-1, 1] - u[-2, 1] - u[-1, 0] + u[-2, 0]) / (dx * dt)

    # Pad the interior
    u_xt = jnp.pad(u_xt, ((1, 1), (1, 1)), mode='constant')

    # Fill in the edges and corners
    u_xt = u_xt.at[0, 1:-1].set(u_xt_top)
    u_xt = u_xt.at[-1, 1:-1].set(u_xt_bottom)
    u_xt = u_xt.at[1:-1, 0].set(u_xt_left)
    u_xt = u_xt.at[1:-1, -1].set(u_xt_right)
    u_xt = u_xt.at[0, 0].set(u_xt_tl)
    u_xt = u_xt.at[0, -1].set(u_xt_tr)
    u_xt = u_xt.at[-1, -1].set(u_xt_br)
    u_xt = u_xt.at[-1, 0].set(u_xt_bl)

    return u_xt

# Fourth-order difference schemes
@jax.jit
def gradient_u_4th(u, dx):
    
    # Interior points
    ux = (u[:-4] - 8*u[1:-3] + 8*u[3:-1] - u[4:]) / (12*dx)

    # Immediate left and right points
    ux_left = (-3*u[0] - 10*u[1] + 18*u[2] - 6*u[3] + u[4]) / (12*dx)
    ux_right = (-1*u[-5] + 6*u[-4] - 18*u[-3] + 10*u[-2] + 3*u[-1]) / (12*dx)

    # Furthest left and right points
    ux_leftest = (-25*u[0] + 48*u[1] - 36*u[2] + 16*u[3] - 3*u[4]) / (12*dx)
    ux_rightest = (3*u[-5] - 16*u[-4] + 36*u[-3] - 48*u[-2] + 25*u[-1]) / (12*dx)

    # ux = jnp.pad(ux, ((2, 2),), mode='constant', constant_values=((ux_leftest, ux_left), (ux_right, ux_rightest)))
    ux = jnp.pad(ux, (1, 1), mode='constant', constant_values=(ux_left, ux_right))
    ux = jnp.pad(ux, (1, 1), mode='constant', constant_values=(ux_leftest, ux_rightest))
    return ux

@jax.jit
def laplace_u_4th(u, dx):

    # Interior points
    uxx = (-u[:-4] + 16*u[1:-3] -30*u[2:-2] + 16*u[3:-1] - u[4:]) / (12*(dx**2))

    # Immediate left and right points
    uxx_left = (11*u[0] - 20*u[1] + 6*u[2] + 4*u[3] - 1*u[4]) / (12*(dx**2))
    uxx_right = (-1*u[-1] + 4*u[-2] + 6*u[-3] - 20*u[-4] + 11*u[-5]) / (12*(dx**2))

    # Furthest left and right points
    uxx_leftest = (35*u[0] - 104*u[1] + 114*u[2] - 56*u[3] + 11*u[4]) / (12*(dx**2))
    uxx_rightest = (11*u[-1] - 56*u[-2] + 114*u[-3] - 104*u[-4] + 35*u[-5]) / (12*(dx**2))

    # uxx = jnp.pad(uxx, (2, 2), mode='constant', constant_values=((uxx_leftest, uxx_left), (uxx_right, uxx_rightest)))
    uxx = jnp.pad(uxx, (1, 1), mode='constant', constant_values=(uxx_left, uxx_right))
    uxx = jnp.pad(uxx, (1, 1), mode='constant', constant_values=(uxx_leftest, uxx_rightest))
    return uxx

def tridiagonal_solve(dl, d, du, b): 
  """Pure JAX implementation of `tridiagonal_solve`.""" 
  prepend_zero = lambda x: jnp.append(jnp.zeros([1], dtype=x.dtype), x[:-1]) 
  fwd1 = lambda tu_, x: x[1] / (x[0] - x[2] * tu_) 
  fwd2 = lambda b_, x: (x[0] - x[3] * b_) / (x[1] - x[3] * x[2]) 
  bwd1 = lambda x_, x: x[0] - x[1] * x_ 
  double = lambda f, args: (f(*args), f(*args)) 

  # Forward pass. 
  _, tu_ = jax.lax.scan(lambda tu_, x: double(fwd1, (tu_, x)), 
                    du[0] / d[0], 
                    (d, du, dl), 
                    unroll=32) 

  _, b_ = jax.lax.scan(lambda b_, x: double(fwd2, (b_, x)), 
                  b[0] / d[0], 
                  (b, d, prepend_zero(tu_), dl), 
                  unroll=32) 

  # Backsubstitution. 
  _, x_ = jax.lax.scan(lambda x_, x: double(bwd1, (x_, x)), 
                  b_[-1], 
                  (b_[::-1], tu_[::-1]), 
                  unroll=32) 

  return x_[::-1] 

def apply_initial_conditions(u, u_left, u_right):

    Nx, Nt = u.shape
    t = jnp.linspace(0, 1, Nt)

    diff_t0 = u_left - u[:, 0]
    diff_t1 = u_right - u[:, -1]

    correction = jnp.outer(diff_t0, (1 - t)) + jnp.outer(diff_t1, t)
    new_u = u + correction
    return new_u 
