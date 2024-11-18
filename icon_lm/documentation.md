`ode_auto_const`: $u'(t) = a_1c(t) + a_2, t \in [0,1]$ 
</br> 
$a_1 \in [0.5, 1.5], a_2 \in [-1, 1], u(0) \in [-1, 1]$ 
</br>
domain: $\Omega = L \cdot dt; L=50, dt=0.2 \implies \Omega = [0.0, 0.02, ..., 0.98], \Omega_{cond} = \Omega_{QoI} $, 

`ode_auto_linear1`: $u'(t) = a_1c(t)u(t) - a_2, t \in [0,1]$
</br> 
$a_1 \in [0.5, 1.5], a_2 \in [-1, 1], u(0) \in [-1, 1]$ 
</br>
domain: $\Omega = L \cdot dt; L=50, dt=0.2 \implies \Omega = [0.0, 0.02, ..., 0.98], \Omega_{cond} = \Omega_{QoI}$

`ode_auto_linear2`: $u'(t) = a_1u(t) + a_2c(t) - a_3, t \in [0,1]$
</br> 
$a_1 \in [-1, 1], a_2 \in [0.5, 1.5], a_3 \in [-1, 1], u(0) \in [-1, 1]$ 
</br>
domain: $\Omega = L \cdot dt; L=50, dt=0.2 \implies \Omega = [0.0, 0.02, ..., 0.98], \Omega_{cond} = \Omega_{QoI}$

`series_damped_oscillator`: $u(t) = Asin(\frac{2 \pi}{T}t + \eta) e^{kt}, t \in [0,1]$
</br>
$k \in [0,2], A \in [0.5, 1.5], T \in [0.1, 0.2], \eta \in [0, 2\pi]$
</br>
domain: $\Omega = 2L \cdot dt; L=50, dt=0.01, \Omega_{cond} = [0, 0.01, 0.02, ..., 0.49], \Omega_{QoI} = [0.50, 0.51, 0.52, ..., 0.99]$

`pde_poisson_spatial`: $u''(x) = c(x)$
</br>
$u(0), u(1) \in [0,1]$
</br>
domain: $\Omega = (L+1) \cdot dx; L=100, dt=0.01 \implies \Omega = [0, 0.01, 0.02, ..., 1.00], \Omega_{cond} = \Omega_{QoI}$

`pde_porous_spatial`: $-\lambda a u''(x) + k(x) u(x) = c, x \in [0, 1], \lambda = 0.05$
</br>
$u(0), u(1) \in [-1,1], c \in [-2, 2], a \in [0.5, 1.5]$
</br>
domain: $\Omega = (L+1) \cdot dx; L=100, dt=0.01 \implies \Omega = [0, 0.01, 0.02, ..., 1.00], \Omega_{cond} = \Omega_{QoI}$

`pde_cubic_spatial`: $-\lambda a u''(x) + ku^3  = c(x), x \in [0, 1], \lambda = 0.1$
</br>
$u(0), u(1) \in [-1,1], a \in [0.5, 1.5], k \in [0.5, 1.5]$
</br>
domain: $\Omega = (L+1) \cdot dx; L=100, dt=0.01 \implies \Omega = [0, 0.01, 0.02, ..., 1.00], \Omega_{cond} = \Omega_{QoI}$