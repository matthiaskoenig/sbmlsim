"""
Input and output functions for integration.
"""
# TODO: Decouple result file generation from database storage/interaction.

from __future__ import print_function, division
import os

import h5py
import numpy as np
from django.core.files import File
from simapp.models import Result

import copasi_tools
from multiscale.multiscale_settings import SIM_DIR


def create_simulation_directory(task):
    """ Create the folder to store odesim files.
        This has to be done on very computer ! not only on the database computer.
    """
    # TODO: handle the files correctly
    directory = os.path.join(SIM_DIR + "/" + str(task))
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('Task directory created: {}'.format(directory))


# ---------------------------------------------------------------------------------------------------------------------
#   CSV
# ---------------------------------------------------------------------------------------------------------------------
def csv_file(sim):
    return os.path.join(SIM_DIR, str(sim.task), "{}.csv".format(sim.pk))


def save_csv(filepath, data, header, keep_tmp=False):
    """ The storage as CSV and conversion to Rdata format is expensive.
        Better solution is the storage as b
        Probably better to store as HDF5 file. For single odesim?
    """
    np.savetxt(filepath, data, header=",".join(header), delimiter=",", fmt='%.12E')


# ---------------------------------------------------------------------------------------------------------------------
#   HDF5
# ---------------------------------------------------------------------------------------------------------------------
def hdf5_file(sim):
    return os.path.join(SIM_DIR, str(sim.task), "{}.h5".format(sim.pk))
    

def save_hdf5(filepath, data, header):
    """ Store numpy data as HDF5.
        Writing header and data.
        /data
        /header
    """
    f = h5py.File(filepath, 'w')
    f.create_dataset('data', data=data, compression="gzip", chunks=True)
    f.create_dataset('header', data=header, compression="gzip", dtype="S10", chunks=True)
    # f.create_dataset('time', data=data[:, 0], compression="gzip")
    f.close()


def load_hdf5(filepath):
    """ Read numpy data from HDF5.
        Read data and header
        /data
        /header
    """
    with h5py.File(filepath, 'r') as f:
        data = f['data'][()]
        header = f['header'][()]
    return data, header


def store_result_db(simulation, filepath, result_type):
    """ Store a result for the simulation in the database. """

    # TODO: add test
    f = open(filepath, 'r')
    myfile = File(f)
    result, _ = Result.objects.get_or_create(simulation=simulation, result_type=result_type)
    result.file = myfile
    result.save()


def store_config_file(sim, folder):
    """ Store the config file in the database. """
    # TODO: refactor, this is not working any more
    fname = copasi_tools.create_config_filename(sim, folder)
    config_file = copasi_tools.create_config_file_for_simulation(sim, fname)
    f = open(config_file, 'r')
    sim.file = File(f)
    sim.save()
    return config_file