import matplotlib.pyplot as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np

subgroup_nr=0

epsilons_pld = pickle.load(open('./pickles/epsilons_pld_mnist.p', "rb"))
epsilons_tf = pickle.load(open('./pickles/epsilons_tf_mnist.p', "rb"))

pp = PdfPages('./plots/approx_pld_epsilons.pdf')
plot_ = plt.figure()
plt.rcParams.update({'font.size': 15.0})

bin_nr=40

legs=[]

plt.hist(epsilons_pld,bins=bin_nr, alpha=0.6)
legs.append(r'Individual $\epsilon$-values (via GDP)')

plt.hist(epsilons_tf,bins=bin_nr, alpha=0.6)
legs.append(r'Individual $\epsilon$-values (via RDP)')

plt.xlabel(r'$\varepsilon$')

plt.legend(legs)
plt.tight_layout()

pp.savefig(plot_)
pp.close()

plt.show()
plt.close()
