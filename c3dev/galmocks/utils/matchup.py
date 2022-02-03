"""
"""
import numpy as np


def _get_data_block(*halo_properties):
    return np.vstack(halo_properties).T
