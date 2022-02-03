"""
"""
import numpy as np
import argparse
from time import time
from astropy.table import Table
from c3dev.galmocks.data_loaders.load_umachine import DTYPE as UM_DTYPE
from c3dev.galmocks.utils import matchup, galmatch
import h5py


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("unit_sim_fn", help="path to unit sim subhalo catalog")
    parser.add_argument("um_fn", help="path to um sfr catalog")
    parser.add_argument("outname", help="Output fname")
    args = parser.parse_args()

    t0 = time()
    um = Table(np.fromfile(args.um_fn, dtype=UM_DTYPE))
    t1 = time()
    unit = Table.read(args.unit_sim_fn, path="data")
    t2 = time()
    print("{0:.1f} seconds to load UM".format(t1 - t0))
    print("{0:.1f} seconds to load UNIT".format(t2 - t1))

    um["uber_host_indx"] = matchup.compute_uber_host_indx(um["upid"], um["id"])
    t3 = time()
    unit["uber_host_indx"] = matchup.compute_uber_host_indx(
        unit["halo_upid"], unit["halo_id"]
    )
    t4 = time()
    print("{0:.1f} seconds to compute UM hostid".format(t3 - t2))
    print("{0:.1f} seconds to compute UNIT hostid".format(t4 - t3))

    um["uber_host_haloid"] = um["id"][um["uber_host_indx"]]
    um["mhost"] = um["m"][um["uber_host_indx"]]
    unit["uber_host_haloid"] = unit["halo_id"][unit["uber_host_indx"]]

    cenmsk_um = um["uber_host_haloid"] == um["id"]
    cenmsk_unit = unit["uber_host_haloid"] == unit["halo_id"]

    source_galaxies_host_halo_id = um["uber_host_haloid"]
    source_halo_ids = um["id"][cenmsk_um]
    target_halo_ids = unit["halo_id"][cenmsk_unit]
    source_halo_props = (np.log10(um["m"][cenmsk_um]),)
    target_halo_props = (np.log10(unit["halo_mvir"][cenmsk_unit]),)
    d = (
        source_galaxies_host_halo_id,
        source_halo_ids,
        target_halo_ids,
        source_halo_props,
        target_halo_props,
    )
    t5 = time()
    galsampler_res = galmatch.compute_source_galaxy_selection_indices(*d)
    t6 = time()
    print("{0:.1f} seconds to galsample".format(t6 - t5))

    keys_to_inherit = "m", "sm", "sfr", "uber_host_haloid", "id", "mhost"
    with h5py.File(args.outname, "w") as hdf:
        for key in keys_to_inherit:
            hdf[key] = um[key][galsampler_res.target_gals_selection_indx]
        hdf["target_gals_target_halo_ids"] = galsampler_res.target_gals_target_halo_ids
        hdf["target_gals_source_halo_ids"] = galsampler_res.target_gals_source_halo_ids

    t7 = time()
    print("{0:.1f} seconds to write mock to disk".format(t7 - t6))
    print("{0:.1f} seconds total runtime".format(t7 - t0))
