import numpy as np

from scipy import special
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow_privacy.privacy.analysis import rdp_accountant

legs = []

epsilons=np.linspace(0.1,1.1,50)

max_order=32


sigma0=100
k1=300
sigma=sigma0/np.sqrt(k1)

dds1 = []

# eps,delta via GDP
for eps in epsilons:

    ss_pld=1/sigma
    mu_pld=ss_pld**2/2

    n=1
    dd2=0.5*( special.erfc((eps-n*mu_pld)/(np.sqrt(2*n)*ss_pld)  ) - np.exp(eps)*special.erfc((eps+n*mu_pld)/(np.sqrt(2*n)*ss_pld)  ))

    dds1.append(dd2)


dds2=[]

sigma=sigma0/np.sqrt(k1)

orders = np.array(range(2, max_order + 1))

rdp = orders/(2*sigma**2)

for eps in epsilons:

    eps, delta, opt_order = rdp_accountant.get_privacy_spent(orders, rdp, target_eps=eps)
    dds2.append(delta)





k2=500
sigma=sigma0/np.sqrt(k2)

dds3 = []

# eps,delta via GDP
for eps in epsilons:

    ss_pld=1/sigma
    mu_pld=ss_pld**2/2

    n=1
    dd2=0.5*( special.erfc((eps-n*mu_pld)/(np.sqrt(2*n)*ss_pld)  ) - np.exp(eps)*special.erfc((eps+n*mu_pld)/(np.sqrt(2*n)*ss_pld)  ))

    dds3.append(dd2)


dds4=[]

sigma=sigma0/np.sqrt(k2)

orders = np.array(range(2, max_order + 1))
#orders = np.linspace(1.01,2000,max_order)
rdp = orders/(2*sigma**2)

for eps in epsilons:

    eps, delta, opt_order = rdp_accountant.get_privacy_spent(orders, rdp, target_eps=eps)
    dds4.append(delta)


k3=900
sigma=sigma0/np.sqrt(k3)

dds5 = []

# eps,delta via GDP
for eps in epsilons:

    ss_pld=1/sigma
    mu_pld=ss_pld**2/2

    n=1
    dd2=0.5*( special.erfc((eps-n*mu_pld)/(np.sqrt(2*n)*ss_pld)  ) - np.exp(eps)*special.erfc((eps+n*mu_pld)/(np.sqrt(2*n)*ss_pld)  ))

    dds5.append(dd2)


dds6=[]

sigma=sigma0/np.sqrt(k3)

orders = np.array(range(2, max_order + 1))

rdp = orders/(2*sigma**2)

for eps in epsilons:

    eps, delta, opt_order = rdp_accountant.get_privacy_spent(orders, rdp, target_eps=eps)
    dds6.append(delta)





pp = PdfPages('./plots/Figure_gdp_2.pdf')
plot_ = plt.figure()
plt.rcParams.update({'font.size': 12.0})





plt.semilogy(epsilons,dds2,'--',markersize=4)
plt.semilogy(epsilons,dds1,'-',markersize=4)

plt.semilogy(epsilons,dds4,'--',markersize=4)

plt.semilogy(epsilons,dds3,'-',markersize=4)

plt.semilogy(epsilons,dds6,'--',markersize=4)

plt.semilogy(epsilons,dds5,'-',markersize=4)

plt.xlabel(r"$\epsilon$")#,fontsize=24)
plt.ylabel(r"$\delta$")#,fontsize=24)

legs = []
legs.append(r"$(\epsilon,\delta)$-DP via RDP Filtering ($k=$" + str(k1) + ')')

legs.append(r"$(\epsilon,\delta)$-DP via GDP Filtering ($k=$" + str(k1) + ')')

legs.append(r"$(\epsilon,\delta)$-DP via RDP Filtering ($k=$" + str(k2) + ')')

legs.append(r"$(\epsilon,\delta)$-DP via GDP Filtering ($k=$" + str(k2) + ')')


legs.append(r"$(\epsilon,\delta)$-DP via RDP Filtering ($k=$" + str(k3) + ')')

legs.append(r"$(\epsilon,\delta)$-DP via GDP Filtering ($k=$" + str(k3) + ')')

plt.legend(legs)
plt.tight_layout()

pp.savefig(plot_)
pp.close()

plt.show()
plt.close()
