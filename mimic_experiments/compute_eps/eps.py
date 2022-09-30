from scipy import special
import numpy as np


def get_epsilon_using_norm(
    noise_multiplier: float,
    squared_clipping_norm: float,
    max_clipping_constant: float,
    delta=1e-5,
):

    epsilon_grid = np.linspace(0.1, 40, num=int(1e5))[::-1]

    def Phi(z):
        return 0.5 * (1 + special.erf(z / np.sqrt(2)))

    sigma_ep = noise_multiplier / np.sqrt(
        (squared_clipping_norm / max_clipping_constant ** 2)
    )
    delta_grid = Phi(1 / (2 * sigma_ep) - epsilon_grid * sigma_ep) - np.exp(
        epsilon_grid
    ) * Phi(-1 / (2 * sigma_ep) - epsilon_grid * sigma_ep)
    best_delta = np.searchsorted(delta_grid, delta, side="right")

    return epsilon_grid[best_delta]