import datetime
import os

import pandas as pd


def _get_current_timestamp():
    return "{:%Y-%b-%d %H:%M:%S}".format(datetime.datetime.now())


def write_stats(stats_dict: dict, path: str, columns: list):
    """
    Write stats from a dict to a csv.
    The header will only be written only for the first row of the file.

    Parameters
    ----------
    stats_dict : dict
        A dict containing the values to be recorded.
        The keys have to match the names of the columns.
    path : str
        The path to which the data should be written.
    columns : list
        The columns of the data that should be written to the file.
    """
    stats_list = list()

    for c in columns:
        if c == "timestamp":
            stats_list.append(_get_current_timestamp())
        else:
            stats_list.append(stats_dict.get(c))

    df = pd.DataFrame([stats_list])
    df.columns = columns
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
