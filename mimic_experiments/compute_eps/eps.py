import numpy as np
from scipy import special
from scipy.optimize import root_scalar
import warnings


def get_epsilon_using_norm(
    noise_multiplier: float,
    squared_clipping_norm: float,
    max_clipping_constant: float,
    delta=1e-5,
):
    def Phi(z):
        return 0.5 * (1 + special.erf(z / np.sqrt(2)))

    sigma_ep = noise_multiplier / np.sqrt(
        (squared_clipping_norm / max_clipping_constant ** 2)
    )

    def obtain_delta(epsilon, delta, sigma_ep):
        return Phi(1 / (2 * sigma_ep) - epsilon * sigma_ep) - np.exp(
            epsilon) * Phi(-1 / (2 * sigma_ep) - epsilon * sigma_ep) - delta

    result_object = root_scalar(obtain_delta, args=(delta, sigma_ep), bracket=[0, 200], method="bisect")

    if not result_object.converged:
        warnings.warn("The algorithm for retrieving epsilon has not converged.")

    return result_object.root
