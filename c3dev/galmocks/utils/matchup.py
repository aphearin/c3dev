"""
"""
import numpy as np
from scipy.spatial import cKDTree


def _get_data_block(*halo_properties):
    return np.vstack(halo_properties).T


def calculate_indx_correspondence(source_props, target_props):
    """For each target data object, find a closely matching source data object

    Parameters
    ----------
    source_props : list of n_props ndarrays
        Each ndarray should have shape (n_source, )

    target_props : list of n_props ndarrays
        Each ndarray should have shape (n_target, )

    Returns
    -------
    dd_match : ndarray of shape (n_target, )
        Euclidean distance between each target and its matching source object

    indx_match : ndarray of shape (n_target, )
        Index into the source object that is matched to each target

    """
    assert len(source_props) == len(target_props)
    X_source = _get_data_block(*source_props)
    X_target = _get_data_block(*target_props)
    source_tree = cKDTree(X_source)
    dd_match, indx_match = source_tree.query(X_target)
    return dd_match, indx_match
