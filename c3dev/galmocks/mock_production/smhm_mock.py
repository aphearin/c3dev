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
from ..utils.matchup import compute_uber_host_indx, calculate_indx_correspondence


TNG_KEYS_TO_INHERIT = ("SubhaloSFR", "SubhaloMassType")


def make_v0p0(target_sim_fn, um_snap_fn):
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

    um["uber_hostid"] = um["id"][um["uber_hostindx"]]
    unit_sim["uber_hostid"] = unit_sim["halo_id"][unit_sim["uber_hostindx"]]

    um["host_mvir"] = um["m"][um["uber_hostindx"]]
    unit_sim["host_mvir"] = unit_sim["halo_mvir"][unit_sim["uber_hostindx"]]

    um["lgmu"] = np.log10(um["m"] / um["host_mvir"])
    unit_sim["lgmu"] = np.log10(unit_sim["halo_mvir"] / unit_sim["host_mvir"])

    um["host_vmax"] = um["v"][um["uber_hostindx"]]
    unit_sim["host_conc"] = unit_sim["halo_nfw_conc"][unit_sim["uber_hostindx"]]

    cenmsk_unit = unit_sim["uber_hostid"] == unit_sim["halo_id"]
    unit_mvir_cens_rank = 1 + rank_order_function(unit_sim["host_mvir"][cenmsk_unit])
    unit_mvir_cens_cnd = unit_mvir_cens_rank / unit_sim_vbox

    unit_sim["host_mvir_lgcnd"] = np.zeros_like(unit_sim["host_mvir"])
    unit_sim["host_mvir_lgcnd"][cenmsk_unit] = unit_mvir_cens_cnd
    unit_sim["host_mvir_lgcnd"] = unit_sim["host_mvir_lgcnd"][unit_sim["uber_hostindx"]]

    cenmsk_um = um["uber_hostid"] == um["id"]
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
    unit_sim["host_conc_perc"][cenmsk_unit] = sliding_conditional_percentile(
        unit_sim["halo_mvir"][cenmsk_unit], unit_sim["halo_nfw_conc"][cenmsk_unit], 201
    )
    unit_sim["host_conc_perc"] = unit_sim["host_conc_perc"][unit_sim["uber_hostindx"]]

    unit_sim.remove_columns(("halo_hostid", "halo_rs"))

    return unit_sim, um


def map_um_mstar_sfr_onto_unit(unit_sim, um, keys_to_inherit):
    cenmsk_unit = unit_sim["uber_hostid"] == unit_sim["halo_id"]
    cenmsk_um = um["uber_hostid"] == um["id"]

    source_keys = ["host_mvir_lgcnd", "host_vmax_perc"]
    source_props = [um[key][cenmsk_um] for key in source_keys]
    target_keys = ["host_mvir_lgcnd", "host_conc_perc"]
    target_props = [unit_sim[key][cenmsk_unit] for key in target_keys]
    _res = calculate_indx_correspondence(source_props, target_props)
    dd_match_cens, indx_match_cens = _res
    inherited_props_cens = [
        um[key][cenmsk_um][indx_match_cens] for key in keys_to_inherit
    ]

    source_keys = ["host_mvir_lgcnd", "lgmu", "host_vmax_perc"]
    source_props = [um[key][~cenmsk_um] for key in source_keys]
    target_keys = ["host_mvir_lgcnd", "lgmu", "host_conc_perc"]
    target_props = [unit_sim[key][~cenmsk_unit] for key in target_keys]
    _res = calculate_indx_correspondence(source_props, target_props)
    dd_match_sats, indx_match_sats = _res
    inherited_props_sats = [
        um[key][~cenmsk_um][indx_match_sats] for key in keys_to_inherit
    ]

    n_unit = len(unit_sim)

    for i, key in enumerate(keys_to_inherit):
        unit_sim["um_" + key] = np.zeros(n_unit)
        unit_sim["um_" + key][cenmsk_unit] = inherited_props_cens[i]
        unit_sim["um_" + key][~cenmsk_unit] = inherited_props_sats[i]

    return unit_sim


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
