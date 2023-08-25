import numpy as np

from scipy import special
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow_privacy.privacy.analysis import rdp_accountant

legs = []

epsilons_pld=[]

sigma0=100

sigma=sigma0

dds2 = []

ks=np.linspace(1,600,600)
delta_max=1e-5

def eps_delta(sigma,k,delta_max):

    ss_pld=1/sigma
    mu_pld=ss_pld**2/2

    n=k
    eps=10
    delta_eps=eps/2
    for ii in range(300):
        delta=0.5*( special.erfc((eps-n*mu_pld)/(np.sqrt(2*n)*ss_pld)  ) - np.exp(eps)*special.erfc((eps+n*mu_pld)/(np.sqrt(2*n)*ss_pld)  ))
        if delta<delta_max:
            eps=eps-delta_eps
            delta_eps=delta_eps/2
        else:
            eps=eps+delta_eps
            delta_eps=delta_eps/2

    return eps
# eps,delta via GDP
for k in ks:

    ss_pld=1/sigma
    mu_pld=ss_pld**2/2

    n=k

    eps=eps_delta(sigma,k,delta_max)
    epsilons_pld.append(eps)

epsilons_rdp=[]





for k in ks:

    sigma=sigma0/np.sqrt(k)

    max_order=32
    orders = np.array(range(2, max_order + 1))
    rdp = orders/(2*sigma**2)

    eps, delta, opt_order = rdp_accountant.get_privacy_spent(orders, rdp, target_delta=delta_max)
    epsilons_rdp.append(eps)



pp = PdfPages('./plots/Figure_gdp_steps.pdf')
plot_ = plt.figure()
plt.rcParams.update({'font.size': 15.0})

plt.plot(epsilons_pld,ks,'-',markersize=1)

plt.plot(epsilons_rdp,ks,'--',markersize=1)

plt.xlabel(r"$\epsilon$")#,fontsize=24)
plt.ylabel(r"max number of compositions $k$")#,fontsize=24)

legs = []
legs.append(r"$(\epsilon,\delta)$-DP via PLD Filtering")
legs.append(r"$(\epsilon,\delta)$-DP via RDP Filtering")


plt.legend(legs)
plt.tight_layout()

pp.savefig(plot_)
pp.close()

plt.show()
plt.close()
