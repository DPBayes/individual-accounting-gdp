

import numpy as np

import time
import pickle


from tensorflow_privacy.privacy.analysis import rdp_accountant

# The following table of gradient norms is computed with the
# small feedforward network (described in the paper) and MNIST data set
# One can obtain the needed code simply by modifying the conv network in
# https://github.com/pytorch/opacus/blob/main/examples/mnist.py
# (then randomly select 1000 data elemenst and save their gradient norms along the training)

# this rounds x nr_samples  numpy array contains the gradient norms of nr_samples data elements
grad_norms = np.load('./grad_table.npy')


print(grad_norms.shape)
rounds = grad_norms.shape[0]
nr_samples = grad_norms.shape[1]

target_epsilon=None
target_delta=1e-5

epsilons_tf=[]
q=300/60000
sigma=2.0
C=5.0

max_order=32
orders = range(2, max_order + 1)

max_rounds=10000

nr_samples=100

for ii in range(nr_samples):
    print(ii)
    rdp = np.zeros_like(orders, dtype=float)
    for nc in range(max_rounds):
        dp_sigma=sigma*C/grad_norms[nc,ii]
        rdp += rdp_accountant.compute_rdp(q, dp_sigma, 1, orders)
    eps, delta, opt_order = rdp_accountant.get_privacy_spent(orders, rdp, target_delta=target_delta, target_eps=target_epsilon)
    epsilons_tf.append(eps)
    print(eps)

pickle.dump(epsilons_tf, open("./pickles/epsilons_tf_mnist_100.p", "wb"))
