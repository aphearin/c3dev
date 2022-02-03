"""
"""
import argparse
from c3dev.galmocks.mock_production import gumbo
from time import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("unit_sim_fn", help="path to unit sim subhalo catalog")
    parser.add_argument("um_fn", help="path to um sfr catalog")
    parser.add_argument("outname", help="Output fname")
    args = parser.parse_args()

    t0 = time()
    unit_sim, um = gumbo.make_v0p0(args.unit_sim_fn, args.um_fn)
    t1 = time()
    print("{0:.1f} seconds to load value-added sims".format(t1 - t0))

    keys_to_inherit = "sm", "sfr"
    unit_sim = gumbo.map_um_mstar_sfr_onto_unit(unit_sim, um, keys_to_inherit)
    unit_sim.write(args.outname, path="data")
    t2 = time()
    print("{0:.1f} seconds total".format(t2 - t0))
