import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_cached_data(datapath: str):
    """
    Loads cached preprocessed mimic-iii data from folder.

    Parameters
    ----------
    datapath : str
        The path to the folder containing the cached datasets.

    Returns
    -------
    train_X : np.ndarry
        The cached training feature matrix.
    train_y : np.ndarry
        The cached training labels.
    test_X : np.ndarry
        The cached test feature matrix.
    test_y : np.ndarry
        The cached test labels.
    """
    train_X = np.load(os.path.join(datapath, "train_X.npy"))
    train_y = np.load(os.path.join(datapath, "train_y.npy"))
    test_X = np.load(os.path.join(datapath, "test_X.npy"))
    test_y = np.load(os.path.join(datapath, "test_y.npy"))

    return train_X, train_y, test_X, test_y


def get_loader(X, y, batch_size, device=DEVICE):
    """

    Parameters
    ----------
    X : np.ndarray
        The feature matrix.
    y : np.ndarray
        The labels.
    batch_size : int
        The size of each batch.
    device : str, optional
        The device on which the DataLoader will be saved, by default gpu if available.
    """
    return DataLoader(
        TensorDataset(
            torch.tensor(X).float().to(device), torch.tensor(y).float().to(device)
        ),
        batch_size=batch_size,
    )
