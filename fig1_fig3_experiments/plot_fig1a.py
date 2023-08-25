import matplotlib.pyplot as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from scipy import special
import math

import numpy as np

# The following table of gradient norms is computed with the
# small feedforward network (described in the paper) and MNIST data set
# One can obtain the needed code simply by modifying the conv network in
# https://github.com/pytorch/opacus/blob/main/examples/mnist.py
# (then randomly select 1000 data elemenst and save their gradient norms along the training)

# this nr_iterations x nr_samples  numpy array contains the gradient norms of nr_samples data elements over nr_iterations iterations
grad_norms = np.load('./grad_table.npy')

print(grad_norms.shape)

nr_samples = grad_norms.shape[1]

print('nr of samples: ' + str(nr_samples))

nr_iterations = grad_norms.shape[0]

#this for ss=2.0
Clipping_constant=5.0
noise_sigma=2.0



# define sigma grid
lower_limit=2.0
upper_limit=30.0
n_sigmas=50
d_sigma=(upper_limit-lower_limit)/(n_sigmas-1)

sigma_array=np.linspace(lower_limit,upper_limit,n_sigmas)
L=20
nx=int(6E6)
q=300/60000
#q=0.01
dx = 2.0*L/nx # discretisation interval \Delta x
x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration
#  start of the integral domain
ii = int(np.floor(float(nx*(L+np.log(1-q))/(2*L))))


FFT_table=[]

for sigma in sigma_array:
    print(sigma)
    Linvx = (sigma**2)*np.log((np.exp(x[ii+1:])-(1-q))/q) + 0.5
    ALinvx = (1/np.sqrt(2*np.pi*sigma**2))*((1-q)*np.exp(-Linvx*Linvx/(2*sigma**2)) +
        q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2)));
    dLinvx = (sigma**2*np.exp(x[ii+1:]))/(np.exp(x[ii+1:])-(1-q));
    fx = np.zeros(nx)
    fx[ii+1:] =  np.real(ALinvx*dLinvx)
    #print(fx[510000:510100])
    half = int(nx/2)
    # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
    temp = np.copy(fx[half:])
    fx[half:] = np.copy(fx[:half])
    fx[:half] = temp
    # Compute the DFT
    FF1 = np.fft.fft(fx*dx)
    FFT_table.append(FF1)

print('precomputation of FFTs completed')


target_delta=1e-5

max_rounds=10000

epsilons_pld = []



# Take a random data element
ijk=19

print(ijk)

b = (Clipping_constant/grad_norms[:max_rounds,ijk])*noise_sigma

encoding=np.zeros(n_sigmas)
for ss in b:
    ii=min(int(np.floor((max(ss-lower_limit,0.0))/d_sigma)),n_sigmas-1)
    encoding[ii]+=1

cfx = np.ones(nx)
for ii in range(n_sigmas):
    if encoding[ii] > 0:
        cfx=cfx*((FFT_table[ii])**encoding[ii])

# Compute the inverse DFT
cfx = np.fft.ifft(cfx)
# Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
temp = np.copy(cfx[half:])
cfx[half:] = cfx[:half]
cfx[:half] = temp



nr_eps=30
eps_grid=np.linspace(0.01,0.8,nr_eps)
delta_grid=[]

sigma_grid=[]


# For a range of eps - values, find the mu - values for which the mu-GDP privacy curves cross at the eps,delta,
# and output the largest mu. There is small variation in these mu values since the (eps,delta)-privacy curve looks that of a Gaussian mechanism,
# due to large number of compositions (DP-SGD). This is explained by CLT.

# To find the mu values, apply bisection method

for ii in range(nr_eps):

    print('ii: ' + str(ii))

    eps_0 = eps_grid[ii]


    kk = int(np.floor(float(nx*(L+np.real(eps_0))/(2*L))))
    dexp_e = -np.exp(eps_0-x[kk+1:])
    exp_e = 1+dexp_e
    integrand = exp_e*cfx[kk+1:]
    delta_temp=np.real(np.sum(integrand))
    delta_grid.append(delta_temp)

    #binary iteration
    sigma_0=11.0
    ss_pld=1/sigma_0
    mu_pld=ss_pld**2/2
    delta_sigma=10.0
    delta_gdp = 0.5*(special.erfc((eps_0-mu_pld)/(np.sqrt(2)*ss_pld)) - np.exp(eps_0)*special.erfc((eps_0+mu_pld)/(np.sqrt(2)*ss_pld)))
    while abs(delta_temp-delta_gdp)/delta_temp>1e-5:

        if delta_temp < delta_gdp:
            sigma_0=sigma_0+delta_sigma
        if delta_temp > delta_gdp:
            sigma_0=sigma_0-delta_sigma
        delta_sigma=delta_sigma/2

        ss_pld=1/sigma_0
        mu_pld=ss_pld**2/2
        delta_gdp = 0.5*(special.erfc((eps_0-mu_pld)/(np.sqrt(2)*ss_pld)) - np.exp(eps_0)*special.erfc((eps_0+mu_pld)/(np.sqrt(2)*ss_pld)))
    sigma_grid.append(sigma_0)

print(sigma_grid)

sigma=min(sigma_grid)

delta_gdp=[]

for ii in range(nr_eps):

    eps_0 = eps_grid[ii]

    ss_pld=1/sigma
    mu_pld=ss_pld**2/2

    n=1
    delta_temp = 0.5*( special.erfc((eps_0-n*mu_pld)/(np.sqrt(2*n)*ss_pld)  ) - np.exp(eps_0)*special.erfc((eps_0+n*mu_pld)/(np.sqrt(2*n)*ss_pld)  ))
    delta_gdp.append(delta_temp)

epsilons_cdp=[]

for delta in delta_grid:

    rho=1/(2*sigma**2)

    epsilon = rho + np.sqrt(4*rho*np.log(np.sqrt(math.pi*rho)/delta))

    epsilons_cdp.append(epsilon)

pp = PdfPages('./plots/approx_gdp.pdf')
plot_ = plt.figure()
plt.rcParams.update({'font.size': 15.0})


legs=[]
legs.append(r'Tight DP-SGD $(\epsilon,\delta)$-DP')
legs.append(r'Approximative $\mu$-GDP Upper Bound')

plt.xlabel(r'$\varepsilon$')
plt.ylabel(r'$\delta$')


plt.semilogy(eps_grid,delta_grid)
plt.semilogy(eps_grid,delta_gdp,'--')
plt.legend(legs)
plt.tight_layout()

pp.savefig(plot_)
pp.close()

plt.show()
plt.close()
