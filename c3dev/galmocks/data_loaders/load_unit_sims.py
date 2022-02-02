"""
"""
import os
from astropy.table import Table


TASSO = "/Users/aphearin/work/DATA/DESI/C3GMC/UNIT"
BEBOP = "/lcrc/project/halotools/C3GMC/UNIT"
UNIT_SIM_LBOX = 1000.0  # Mpc/h


def read_unit_sim(drn, bn="out_107p.list.hdf5"):
    subhalos = Table.read(os.path.join(drn, bn), path="data")
    return subhalos
