import warnings
import torch
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def record_gradients_for_subgroups(
    subgroup_dict: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    iteration: int,
    max_grad_norm: float,
):

    all_gradients_list = list()
    for k, v in subgroup_dict.items():
        gradient_list = _create_list_of_gradients(
            model,
            optimizer,
            criterion,
            iteration,
            subgroup_id=v.get("subgroup_id"),
            sample_ids=np.load(v.get("sample_id_path")),
            subgroup_X=torch.Tensor(np.load(v.get("path_X"))).float().to(DEVICE),
            subgroup_y=torch.Tensor(np.load(v.get("path_y"))).float().to(DEVICE),
            max_grad_norm=max_grad_norm,
        )
        all_gradients_list.append(gradient_list)

    return np.concatenate(all_gradients_list)


def _create_list_of_gradients(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    iteration: int,
    subgroup_id: str,
    sample_ids: torch.Tensor,
    subgroup_X: torch.Tensor,
    subgroup_y: torch.Tensor,
    max_grad_norm: float
):
    gradients = list()

    # compute gradient for single data points
    # a naive, but simple solution
    for id, data, target in zip(sample_ids, subgroup_X, subgroup_y):

        # ignore pytorch warnings about hooks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # compute gradients for current sample
            optimizer.zero_grad()
            y_pred = model(torch.atleast_2d(data))
            loss = criterion(torch.atleast_2d(y_pred), torch.atleast_2d(target))
            loss.backward()

        # concatenate gradients to compute l2-norm per sample
        gradient_norms = list()
        total_gradient_norm = 0
        batch_grad_norms = 0
        for (ii, p) in enumerate(model.parameters()):

            per_sample_grad = p.grad_sample

            # dimension across which we compute the norms for this gradient part
            # (here is difference e.g. between biases and weight matrices)
            dims = list(range(1, len(per_sample_grad.shape)))

            # compute the clipped norms. Gradients will be clipped in .backward()
            per_sample_grad_norms = per_sample_grad.norm(dim=dims)

            batch_grad_norms += per_sample_grad_norms ** 2

            # compute the clipped norms. Gradients will be then clipped in .backward()
        total_gradient_norm += (
            torch.sqrt(batch_grad_norms).clamp(max=max_grad_norm)
        ) ** 2

        gradients.append([subgroup_id, iteration, id, total_gradient_norm.cpu().item()])

    optimizer.zero_grad()

    gradients = np.array(
        gradients
    )
    return gradients


def save_gradients_to_npy(gradient_array: np.ndarray, path: str):
    np.save(path, gradient_array, allow_pickle=False, fix_imports=False)
    print("Save gradients to", path)
