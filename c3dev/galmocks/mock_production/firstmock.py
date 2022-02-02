"""
"""
import numpy as np
from copy import deepcopy
from ..galhalo_models.smhm import _get_cen_sat_percentile, mc_logsm
from ..galhalo_models.smhm import DEFAULT_SMHM_SCATTER, DEFAULT_SMHM_PARAMS

TNG_KEYS_TO_INHERIT = ("SubhaloSFR", "SubhaloMassType")


def make_mock(subhalos, ran_key):
    subhalos = get_value_added_subhalos(subhalos, ran_key)
    smhm_params = list(DEFAULT_SMHM_PARAMS.values())
    scatter = deepcopy(DEFAULT_SMHM_SCATTER)
    logmh = np.log10(subhalos["halo_mvir"])
    p = subhalos["p_conc"]
    subhalos["logsm"] = mc_logsm(smhm_params, logmh, p, scatter)
    return subhalos


def get_value_added_subhalos(subhalos, ran_key, nwin=201):

    cenmsk = subhalos["halo_hostid"] == subhalos["halo_id"]
    subhalos["p_conc"] = _get_cen_sat_percentile(
        subhalos["halo_mvir"], subhalos["halo_nfw_conc"], cenmsk, nwin, ran_key
    )
    return subhalos
