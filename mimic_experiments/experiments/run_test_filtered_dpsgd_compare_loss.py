import sys


from recording_utils.stats_recorder import write_stats
from recording_utils.gradient_recorder import (
    record_gradients_for_subgroups, save_gradients_to_npy)
from recording_utils.loss_recorder import record_loss_for_subgroups, save_loss_to_npy
from models.basic_mlp import MLPMultiLabel
from mimic_utils.mimic_dataset import MimicDataset
from mimic_utils.data_loaders import load_cached_data
from mimic_utils.auc_metrics import print_metrics_multilabel
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.data_loader import DPDataLoader
from opacus import PrivacyEngine
import torch
import numpy as np
import warnings
import json


def train(
    model,
    DEVICE,
    train_loader_0,
    optimizer,
    num_epochs,
    max_grad_norm,
    budget,
    criterion,
    N,
    train_X,
    train_y,
    test_X,
    test_y,
    batch_size,
    stats_path,
    gradient_save_path,
    loss_save_path,
    subgroup_config_path,
    T=None,
    filter=True,
    save_gradient_norms=False,
    privacy_engine=None,
    stop_epsilon=None,
):

    # Compute all the individual norms (actually the squared norms squares are saved here)
    grad_norms = torch.zeros(N).to(DEVICE)

    auc_dict = dict()
    auc_dict["ave_auc_macro"] = 0

    iteration = 0
    recorded_gradients = list()
    recorded_losses = list()
    recorded_losses_test = list()

    with open(subgroup_config_path, "r") as f:
        subgroup_dict = json.load(f)

    with open("picked_subgroups_test/config.json", "r") as f:
        subgroup_dict_test = json.load(f)

    for epoch in range(1, num_epochs + 1):
        print(epoch, flush=True)
        # print(privacy_engine.get_epsilon(1e-5))
        model.train()

        # Train loader returns also indices (vector idx)
        with BatchMemoryManager(data_loader=train_loader_0, max_physical_batch_size=5000, optimizer=optimizer) as train_loader_0_new:
            for _, (data, target, idx) in enumerate(train_loader_0_new):
                # print(len(idx))

                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                batch_grad_norms = torch.zeros(len(target)).cuda()

                # Clip each parameter's per-sample gradient
                for (ii, p) in enumerate(model.parameters()):

                    per_sample_grad = p.grad_sample

                    # dimension across which we compute the norms for this gradient part
                    # (here is difference e.g. between biases and weight matrices)
                    dims = list(range(1, len(per_sample_grad.shape)))

                    # compute the clipped norms. Gradients will be clipped in .backward()
                    per_sample_grad_norms = per_sample_grad.norm(dim=dims)

                    batch_grad_norms += per_sample_grad_norms ** 2

                # compute the clipped norms. Gradients will be then clipped in .backward()
                grad_norms[idx] += (
                    torch.sqrt(batch_grad_norms).clamp(max=max_grad_norm)
                ) ** 2

        del batch_grad_norms
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        # filter elements that are allowed to continue
        active_indices = [idx for idx in range(N) if grad_norms[idx] < budget]

        if len(active_indices) == 0:
            # print("all budgets exceeded, stopping at epoch: " + str(epoch))
            return

        train_loader = create_DP_dataloader_from_indicies(
            active_indices, train_X, train_y, batch_size
        )

        temp_grads = []
        for p in model.parameters():
            temp_grads.append(torch.zeros(p.data.shape).cuda())

        # Run full batch, sum up the gradients. Then we filter and replace the train_loader
        with BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=5000, optimizer=optimizer) as train_loader_new:
            for _, (data, target) in enumerate(train_loader_new):

                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()

                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        recorded_gradients.append(record_gradients_for_subgroups(
            subgroup_dict, model, optimizer, criterion, iteration=epoch, max_grad_norm=max_grad_norm
        ))

        recorded_losses.append(record_loss_for_subgroups(
            subgroup_dict, model, criterion, iteration=epoch
        ))

        recorded_losses_test.append(record_loss_for_subgroups(
            subgroup_dict_test, model, criterion, iteration=epoch
        ))

        rdp_epsilon = privacy_engine.get_epsilon(1e-5)

        if stop_epsilon and float(stop_epsilon) < rdp_epsilon:
            return auc_dict.get("ave_auc_macro")

        auc_dict = test(model, DEVICE, test_X, test_y)

        noise_multiplier = optimizer.noise_multiplier

        _save_stats(
            auc_dict,
            max(grad_norms),
            budget,
            epoch,
            active_indices,
            stats_path,
            max_grad_norm,
            T,
            rdp_epsilon,
            noise_multiplier,
        )

        del temp_grads
        optimizer.zero_grad()
        torch.cuda.empty_cache()

    save_gradients_to_npy(
        np.concatenate(recorded_gradients), path=gradient_save_path
    )
    save_loss_to_npy(
        np.concatenate(recorded_losses), path=loss_save_path
    )

    save_loss_to_npy(
        np.concatenate(recorded_losses_test), path="TESTLOSS_{}.npy".format(sys.argv[1])
    )
    return auc_dict.get("ave_auc_macro")


def create_DP_dataloader_from_indicies(
    indicies: list, X: torch.Tensor, y: torch.Tensor, batch_size: int
):
    dataset = torch.utils.data.TensorDataset(X[indicies], y[indicies])
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )
    dp_data_loader = DPDataLoader.from_data_loader(data_loader)
    return dp_data_loader


def test(model, DEVICE, test_X, test_y):
    model.eval()
    with torch.no_grad():
        test_activations = model(test_X).cpu()
        auc_dict = print_metrics_multilabel(test_y.cpu(), test_activations, verbose=0)
    return auc_dict


def _save_stats(
    auc_dict: dict,
    max_grad_norm_sum: float,
    budget: float,
    epoch: int,
    active_indices: list,
    stats_path: str,
    clipping_bound: float,
    T: int,
    rdp_epsilon: float,
    noise_multiplier: float,
):
    columns = [
        "epoch",
        "active_elements",
        "max_current_grad_norm_sum",
        "clipping_bound",
        "T",
        "budget",
        "rdp_epsilon",
        "noise_multiplier",
    ]
    stats_dict = dict()
    stats_dict["epoch"] = epoch
    stats_dict["max_current_grad_norm_sum"] = max_grad_norm_sum.item()
    stats_dict["clipping_bound"] = clipping_bound
    stats_dict["T"] = T
    stats_dict["budget"] = budget
    stats_dict["active_elements"] = len(active_indices)
    stats_dict["rdp_epsilon"] = rdp_epsilon
    stats_dict["noise_multiplier"] = noise_multiplier
    columns.append("ave_auc_macro")
    stats_dict["ave_auc_macro"] = auc_dict.get("ave_auc_macro")
    for i, v in enumerate(auc_dict.get("auc_scores")):
        title = "auc_{}".format(i)
        columns.append(title)
        stats_dict[title] = v

    write_stats(stats_dict, path=stats_path, columns=columns)


def train_with_params(
    max_per_sample_grad_norm: float,
    learning_rate: float,
    l2_reg: float,
    batch_size: int,
    num_epochs: int,
    sigma_tilde: float,
    T: int,
):
    torch.manual_seed(472368)
    warnings.filterwarnings("ignore", message=r".*Using a non-full backward hook.*")

    cached_data_path = "phenotyping/"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_X, train_y, test_X, test_y = load_cached_data(cached_data_path)
    train_X = torch.tensor(train_X).float().to(DEVICE)
    train_y = torch.tensor(train_y).float().to(DEVICE)
    test_X = torch.tensor(test_X).float().to(DEVICE)
    test_y = torch.tensor(test_y).float().to(DEVICE)

    N = len(train_X)

    sigma = sigma_tilde

    budget = (
        T * max_per_sample_grad_norm ** 2
    ) + 1e-3  # This will correspond to (eps,delta)-DP filtering with noise std sigma_tilde/sqrt(T)

    # This train loader is for the full batch and for checking all the individual gradient norms
    data_set = MimicDataset(train_X, train_y)
    train_loader_0 = torch.utils.data.DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # This train loader is the one that will be filtered. It will be replaced by a new one at each iteration
    train_loader = torch.utils.data.DataLoader(
        MimicDataset(train_X, train_y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = MLPMultiLabel(train_X.shape[1], train_y.shape[1]).to(DEVICE)

    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=l2_reg,
    )

    secure_rng = False
    privacy_engine = None
    privacy_engine = PrivacyEngine(secure_mode=secure_rng)

    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=max_per_sample_grad_norm,
        poisson_sampling=False,
    )

    return train(
        model,
        DEVICE,
        train_loader_0,
        optimizer,
        num_epochs,
        max_per_sample_grad_norm,
        budget,
        criterion,
        N,
        train_X,
        train_y,
        test_X,
        test_y,
        batch_size,
        stats_path="filtering_losses_testrun_{}.csv".format(
            sys.argv[1]),
        gradient_save_path="filtering_gradients_testrun_{}.npy".format(sys.argv[1]),
        loss_save_path="filtering_losses_testrun_{}.npy".format(sys.argv[1]),
        subgroup_config_path="picked_subgroups/config.json",
        filter=True,
        save_gradient_norms=False,
        privacy_engine=privacy_engine,
        stop_epsilon=None,
    )


if __name__ == "__main__":
    max_per_sample_grad_norm = 0.7872438989061009
    learning_rate = 0.009478653844790382
    l2_regularizer = 0.0009389723136946396
    sigma_tilde = 10.614652581338738

    train_with_params(
        max_per_sample_grad_norm=max_per_sample_grad_norm*float(format(sys.argv[1])),
        learning_rate=learning_rate,
        l2_reg=l2_regularizer,
        batch_size=29252,
        num_epochs=100,
        sigma_tilde=sigma_tilde,
        T=50,
    )
