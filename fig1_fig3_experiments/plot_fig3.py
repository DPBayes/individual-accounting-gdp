import numpy as np

from scipy import special
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow_privacy.privacy.analysis import rdp_accountant

legs = []

epsilons=np.linspace(0.1,.9,50)

sigma0=100
k=420
sigma=sigma0/np.sqrt(k)

dds2 = []

# eps,delta via GDP
for eps in epsilons:

    ss_pld=1/sigma
    mu_pld=ss_pld**2/2

    n=1
    dd2=0.5*( special.erfc((eps-n*mu_pld)/(np.sqrt(2*n)*ss_pld)  ) - np.exp(eps)*special.erfc((eps+n*mu_pld)/(np.sqrt(2*n)*ss_pld)  ))

    dds2.append(dd2)


dds4=[]

k=420
sigma=sigma0/np.sqrt(k)

max_order=32
orders = np.array(range(2, max_order + 1))
#orders = np.linspace(1.01,2000,max_order)
rdp = orders/(2*sigma**2)

for eps in epsilons:

    eps, delta, opt_order = rdp_accountant.get_privacy_spent(orders, rdp, target_eps=eps)
    dds4.append(delta)



pp = PdfPages('./plots/Figure_gdp.pdf')
plot_ = plt.figure()
plt.rcParams.update({'font.size': 15.0})

plt.semilogy(epsilons,dds4,'--',markersize=4)

plt.semilogy(epsilons,dds2,'-',markersize=4)

plt.xlabel(r"$\epsilon$")#,fontsize=24)
plt.ylabel(r"$\delta$")#,fontsize=24)

legs = []
legs.append(r"$(\epsilon,\delta)$-DP via RDP Filtering")

legs.append(r"$(\epsilon,\delta)$-DP via GDP Filtering")

plt.legend(legs)
plt.tight_layout()

pp.savefig(plot_)
pp.close()

plt.show()
plt.close()
