import warnings
import torch
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def record_loss_for_subgroups(
    subgroup_dict: dict,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    iteration: int,
):

    all_loss_list = list()
    for k, v in subgroup_dict.items():
        loss_list = _create_list_of_losses(
            model,
            criterion,
            iteration,
            subgroup_id=v.get("subgroup_id"),
            sample_ids=np.load(v.get("sample_id_path")),
            subgroup_X=torch.Tensor(np.load(v.get("path_X"))).float().to(DEVICE),
            subgroup_y=torch.Tensor(np.load(v.get("path_y"))).float().to(DEVICE),
        )
        all_loss_list.append(loss_list)

    return np.concatenate(all_loss_list)


def _create_list_of_losses(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    iteration: int,
    subgroup_id: str,
    sample_ids: torch.Tensor,
    subgroup_X: torch.Tensor,
    subgroup_y: torch.Tensor,
):
    losses = list()

    # compute gradient for single data points
    # a naive, but simple solution
    for id, data, target in zip(sample_ids, subgroup_X, subgroup_y):

        # ignore pytorch warnings about hooks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                y_pred = model(torch.atleast_2d(data))
                loss = criterion(torch.atleast_2d(y_pred), torch.atleast_2d(target))

        losses.append([subgroup_id, iteration, id, loss.cpu().item()])


    losses = np.array(
        losses
    )
    return losses


def save_loss_to_npy(loss_array: np.ndarray, path: str):
    np.save(path, loss_array, allow_pickle=False, fix_imports=False)
    print("Save loss to", path)
