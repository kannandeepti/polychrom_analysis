""" Script to analyze output of hybrid simulations 

Deepti Kannan, 2023"""

from tqdm import tqdm
import itertools
from itertools import combinations
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import sys
try:
    import polychrom
except:
    sys.path.append("/home/dkannan/git-remotes/polychrom/")
    import polychrom
from polychrom import contactmaps
from polychrom.hdf5_format import list_URIs, load_URI
from scipy.spatial.distance import pdist, squareform

import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from post_processing.analysis import *
from post_processing.msd import *

def process_spherical_well(simdir=Path('/net/levsha/share/deepti/simulations/spherical_well_test')):
    savepath = Path('/net/levsha/share/deepti/data') 
    conf_file = savepath / f"conformations/conformations_spherical_well_test.npy"
    confs = extract(simdir/'run0', start=5000, every_other=10)
    print(f"Extracted {len(confs)} conformations")
    np.save(conf_file, confs)
    mat = contactmaps.monomerResolutionContactMap(
        filenames=confs, cutoff=2.0
    )
    mat2 = mat / len(confs)
    # save cutoff radius = 2.0 contact map
    np.save(savepath/f"contact_maps/contact_map_spherical_well_cutoff2.0.npy", mat2)

def process_simulations(simdir=Path('/net/levsha/share/deepti/simulations/active16_inactive1_act19.0_center')):
    savepath = Path('/net/levsha/share/deepti/data') 
    basepaths = [d for d in simdir.glob('run*')]
    conf_file = savepath / f"conformations/conformations_active16_inactive1_act19_center.npy"
    conformations = []
    for i, basepath in enumerate(basepaths):
        print(basepath.name)
        confs = extract(basepath, start=20000, every_other=100)
        conformations += confs
        print(f"Extracted {len(confs)} conformations for simulation {str(basepath.name)}")
    np.save(conf_file, conformations)
    mat = contactmaps.monomerResolutionContactMap(
        filenames=conformations, cutoff=2.0
    )
    mat2 = mat / len(conformations)
    # save cutoff radius = 2.0 contact map
    np.save(savepath/f"contact_maps/contact_map_active16_inactive1_act19_center_cutoff2.0.npy", mat2)

def save_MSD_param_sweep(ids=None, ncores=25, start=1000, every_other=10, N=1156):
    """Compute the ensemble averaged MSD curves for active and inactive regions
    for all simulations in `basepath`.

    Parameters
    ----------
    basepath : str or Path
        path to simulation directory where each subdirectory is a replicate of an ensemble
    savefile : str or Path
        path to directory where MSD files will be written
    every_other : int
        skip ever_other conformations when computing MSDs
    ncores : int
        number of CPUs to parallelize computations over

    """
    simdir = Path('/net/levsha/share/deepti/simulations/chr2_Su2020')
    savepath = Path('/net/levsha/share/deepti/data') 
    basepaths = [d/'runs20000_1000_20copies' for d in simdir.iterdir()]
    simstrings = [str(d.name) for d in simdir.iterdir()]
    # 0 is cold (B) and 1 is hot (A)
    if ids is None:
        ids = np.load(savepath/'ABidentities_chr2_q_Su2020_2perlocus.npy')
    assert(np.all(np.logical_or(ids == 0, ids == 1)))
    single_param_msd = partial(compute_single_trajectory_msd, start=start,
            every_other=every_other, N=N)
    with mp.Pool(ncores) as p:
        msds = p.map(single_param_msd, basepaths)
    msds = np.array(msds)  # shape: (#directories, #timesteps, #monomers)
    for i in range(msds.shape[0]):
        # average over ensemble and over all monomers that have the same activity
        hot_msd_ave = msds[i, :, ids == 1].mean(axis=0)
        cold_msd_ave = msds[i, :, ids == 0].mean(axis=0)
        df = pd.DataFrame(columns=["Time", "active_MSD", "inactive_MSD"])
        # in units of blocks of time steps
        df["Time"] = np.arange(0, len(hot_msd_ave)) * every_other
        df["active_MSD"] = hot_msd_ave
        df["inactive_MSD"] = cold_msd_ave
        df.to_csv(savepath/f'msds_{simstrings[i]}_ens_ave.csv', index=False)
    return msds
