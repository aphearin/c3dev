"""
"""
import numpy as np
from scipy.spatial import cKDTree
from halotools.utils import crossmatch
from warnings import warn


def _get_data_block(*halo_properties):
    return np.vstack(halo_properties).T


def calculate_indx_correspondence(source_props, target_props, n_threads=-1):
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
    dd_match, indx_match = source_tree.query(X_target, workers=n_threads)
    return dd_match, indx_match


def compute_hostid(upid, haloid):
    cenmsk = upid == -1
    hostid = np.copy(haloid)
    hostid[~cenmsk] = upid[~cenmsk]
    idxA, idxB = crossmatch(hostid, haloid)

    has_match = np.zeros(haloid.size).astype("bool")
    has_match[idxA] = True
    hostid[~has_match] = haloid[~has_match]
    return hostid, idxA, idxB, has_match


def compute_uber_host_indx(
    upid, haloid, max_order=20, fill_val=-99, return_internals=False
):
    hostid, idxA, idxB, has_match = compute_hostid(upid, haloid)
    cenmsk = hostid == haloid

    if len(idxA) != len(haloid):
        msg = "{0} values of upid have no match. Treating these objects as centrals"
        warn(msg.format(len(haloid) - len(idxA)))

    _integers = np.arange(haloid.size).astype(int)
    uber_host_indx = np.zeros_like(haloid) + fill_val
    uber_host_indx[cenmsk] = _integers[cenmsk]

    n_unmatched = np.count_nonzero(uber_host_indx == fill_val)
    counter = 0
    while (n_unmatched > 0) and (counter < max_order):
        uber_host_indx[idxA] = uber_host_indx[idxB]
        n_unmatched = np.count_nonzero(uber_host_indx == fill_val)
        counter += 1

    if return_internals:
        return uber_host_indx, idxA, idxB
    else:
        return uber_host_indx
