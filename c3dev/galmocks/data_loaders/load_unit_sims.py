"""
"""
import os
from astropy.table import Table


TASSO = "/Users/aphearin/work/DATA/DESI/C3GMC/UNIT"
BEBOP = "/lcrc/project/halotools/C3GMC/UNIT"


def read_unit_sim(drn, bn="out_107p.list.hdf5"):
    subhalos = Table.read(os.path.join(drn, bn), path="data")
    return subhalos
