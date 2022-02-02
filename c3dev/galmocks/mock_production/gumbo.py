"""
"""
from copy import deepcopy
import numpy as np
from astropy.table import Table
from halotools.utils.conditional_percentile import rank_order_function
from halotools.utils import sliding_conditional_percentile
from ..galhalo_models.smhm import _get_cen_sat_percentile, mc_logsm
from ..galhalo_models.smhm import DEFAULT_SMHM_SCATTER, DEFAULT_SMHM_PARAMS
from ..data_loaders.load_unit_sims import read_unit_sim, UNIT_SIM_LBOX
from ..data_loaders.load_umachine import read_sfr_snapshot, SMDPL_LBOX
from ..utils.matchup import compute_uber_host_indx


TNG_KEYS_TO_INHERIT = ("SubhaloSFR", "SubhaloMassType")


def make_gumbo_v0p0(target_sim_fn, um_snap_fn):
    """"""
    # Read from disk
    unit_sim = Table(read_unit_sim(target_sim_fn))
    um = Table(read_sfr_snapshot(um_snap_fn))

    unit_sim_vbox = UNIT_SIM_LBOX ** 3
    smdpl_vbox = SMDPL_LBOX ** 3

    # Calculate the indexing array into the uber host halo
    unit_sim["uber_hostindx"] = compute_uber_host_indx(
        unit_sim["halo_upid"], unit_sim["halo_id"]
    )
    um["uber_hostindx"] = compute_uber_host_indx(um["upid"], um["id"])

    um["hostid"] = um["id"][um["uber_hostindx"]]
    unit_sim["hostid"] = unit_sim["halo_id"][unit_sim["uber_hostindx"]]

    um["host_mvir"] = um["m"][um["uber_hostindx"]]
    unit_sim["host_mvir"] = unit_sim["halo_mvir"][unit_sim["uber_hostindx"]]

    um["mu"] = um["m"] / um["host_mvir"]
    unit_sim["mu"] = unit_sim["halo_mvir"] / unit_sim["host_mvir"]

    um["host_vmax"] = um["v"][um["uber_hostindx"]]
    unit_sim["host_conc"] = unit_sim["halo_nfw_conc"][unit_sim["uber_hostindx"]]

    cenmsk_unit = unit_sim["hostid"] == unit_sim["halo_id"]
    unit_mvir_cens_rank = 1 + rank_order_function(unit_sim["host_mvir"][cenmsk_unit])
    unit_mvir_cens_cnd = unit_mvir_cens_rank / unit_sim_vbox

    unit_sim["host_mvir_lgcnd"] = np.zeros_like(um["host_mvir"])
    unit_sim["host_mvir_lgcnd"][cenmsk_unit] = unit_mvir_cens_cnd
    unit_sim["host_mvir_lgcnd"] = unit_sim["host_mvir_lgcnd"][unit_sim["uber_hostindx"]]

    cenmsk_um = um["hostid"] == um["id"]
    um_mvir_cens_rank = 1 + rank_order_function(um["host_mvir"][cenmsk_um])
    um_mvir_cens_cnd = um_mvir_cens_rank / smdpl_vbox

    um["host_mvir_lgcnd"] = np.zeros_like(um["host_mvir"])
    um["host_mvir_lgcnd"][cenmsk_um] = um_mvir_cens_cnd
    um["host_mvir_lgcnd"] = um["host_mvir_lgcnd"][um["uber_hostindx"]]

    um["host_vmax_perc"] = -1.0
    um["host_vmax_perc"][cenmsk_um] = sliding_conditional_percentile(
        um["m"][cenmsk_um], um["v"][cenmsk_um], 201
    )
    um["host_vmax_perc"] = um["host_vmax_perc"][um["uber_hostindx"]]

    unit_sim["host_conc_perc"] = -1.0
    unit_sim["host_conc_perc"][cenmsk_um] = sliding_conditional_percentile(
        unit_sim["halo_mvir"][cenmsk_um], unit_sim["halo_nfw_conc"][cenmsk_um], 201
    )
    unit_sim["host_conc_perc"] = unit_sim["host_conc_perc"][unit_sim["uber_hostindx"]]

    return unit_sim, um


def make_mock(target_sim_subhalos, ran_key):
    target_sim_subhalos = get_value_added_subhalos(target_sim_subhalos, ran_key)
    smhm_params = list(DEFAULT_SMHM_PARAMS.values())
    scatter = deepcopy(DEFAULT_SMHM_SCATTER)
    logmh = np.log10(target_sim_subhalos["halo_mvir"])
    p = target_sim_subhalos["p_conc"]
    target_sim_subhalos["logsm"] = mc_logsm(smhm_params, logmh, p, scatter)
    return target_sim_subhalos


def get_value_added_subhalos(subhalos, ran_key, nwin=201):

    cenmsk = subhalos["halo_hostid"] == subhalos["halo_id"]
    subhalos["p_conc"] = _get_cen_sat_percentile(
        subhalos["halo_mvir"], subhalos["halo_nfw_conc"], cenmsk, nwin, ran_key
    )
    return subhalos
