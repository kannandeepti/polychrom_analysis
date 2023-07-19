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

def process_param_sweep(simdir=Path('/net/levsha/share/deepti/simulations/chr2_Su2020')):
    savepath = Path('/net/levsha/share/deepti/data') 
    basepaths = [d for d in simdir.iterdir()]
    simstrings = [str(d.name) for d in simdir.iterdir()]
    print(simstrings)
    radius_of_gyration = []
    for i, basepath in enumerate(basepaths):
        conf_file = savepath / f"conformations/conformations_{simstrings[i]}.npy"
        if not conf_file.is_file() and (basepath/'runs40000_2000_20copies').is_dir():
            conformations = extract(basepath/'runs40000_2000_20copies', start=20000, every_other=200)
            print(f"Extracted {len(conformations)} conformations for simulation {simstrings[i]}")
            np.save(conf_file, conformations)
            Rg2 = mean_squared_separation(conformations, savepath/'distance_maps', simstrings[i], metric='euclidean', N=1156)
            radius_of_gyration.append(Rg2)
            mat = contactmaps.monomerResolutionContactMapSubchains(
                filenames=conformations, mapStarts=[i*1156 for i in range(20)], mapN=1156, cutoff=2.0
            )
            mat2 = mat / (len(conformations)*20)
            # save cutoff radius = 2.0 contact map
            np.save(savepath/f"contact_maps/contact_map_{simstrings[i]}_cutoff2.0.npy", mat2)
    return radius_of_gyration
    #save radius of gyration
    #df = pd.DataFrame()
    #df['sim'] = simstrings
    #df['Rg2'] = radius_of_gyration
    #df.to_csv(savepath/"distance_maps/radius_of_gyration_chr2_Su2020.csv", index=False)


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
    basepaths = []
    simstrings = []
    for d in simdir.iterdir():
        simstring = str(d.name)
        msdfile = savepath/f'msds_{simstring}_ens_ave.csv'
        if not msdfile.is_file() and (d/'runs40000_2000_20copies').is_dir():
            basepaths.append(d/'runs40000_2000_20copies')
            simstrings.append(simstring)

    # 0 is cold (B) and 1 is hot (A)
    if ids is None:
        ids = np.load(savepath/'ABidentities_chr2_q_Su2020_2perlocus.npy')
    assert(np.all(np.logical_or(ids == 0, ids == 1)))
    single_param_msd = partial(compute_single_trajectory_msd, start=start,
            every_other=every_other, N=N)
    with mp.Pool(ncores) as p:
        msds = p.map(single_param_msd, basepaths)
    #msds = np.array(msds)  # shape: (#directories, #timesteps, #monomers)
    #print(msds.shape)
    for i in range(len(msds)):
       # average over ensemble and over all monomers that have the same activity
       hot_msd_ave = msds[i][:, ids == 1].mean(axis=1)
       cold_msd_ave = msds[i][:, ids == 0].mean(axis=1)
       df = pd.DataFrame(columns=["Time", "active_MSD", "inactive_MSD"])
       # in units of blocks of time steps
       df["Time"] = np.arange(0, len(hot_msd_ave)) * every_other
       df["active_MSD"] = hot_msd_ave
       df["inactive_MSD"] = cold_msd_ave
       df.to_csv(savepath/f'msds_{simstrings[i]}_ens_ave.csv', index=False)
    return msds
