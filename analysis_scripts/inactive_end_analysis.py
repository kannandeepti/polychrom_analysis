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
    conf_file = savepath / f"conformations/conformations_spherical_well_test_repulsive_walls_width4_depth5.npy"
    confs = extract(simdir/'repulsive_walls_width4_depth5', start=5000, every_other=10)
    print(f"Extracted {len(confs)} conformations")
    np.save(conf_file, confs)
    mat = contactmaps.monomerResolutionContactMap(
        filenames=confs, cutoff=2.0
    )
    mat2 = mat / len(confs)
    # save cutoff radius = 2.0 contact map
    np.save(savepath/f"contact_maps/contact_map_spherical_well_repulsive_walls_width4_depth5_cutoff2.0.npy", mat2)

def process_simulations(simdir=Path('/net/levsha/share/deepti/simulations/active16_inactive1_act1.0_E00.5_center')):
    savepath = Path('/net/levsha/share/deepti/data') 
    basepaths = [d for d in simdir.glob('run*')]
    conf_file = savepath / f"conformations/conformations_active16_inactive1_act1_E00.5_center.npy"
    conformations = []
    for i, basepath in enumerate(basepaths):
        print(basepath.name)
        confs = extract(basepath, start=10000, every_other=50)
        conformations += confs
        print(f"Extracted {len(confs)} conformations for simulation {str(basepath.name)}")
    np.save(conf_file, conformations)
    mat = contactmaps.monomerResolutionContactMap(
        filenames=conformations, cutoff=2.0
    )
    mat2 = mat / len(conformations)
    # save cutoff radius = 2.0 contact map
    np.save(savepath/f"contact_maps/contact_map_active16_inactive1_act1_E00.5_center_cutoff2.0.npy", mat2)

def process(uri, blocks, bin_edges, chain_starts, chain_ends, cutoff=1.1):
    idx = int(uri.split('::')[-1])
    data = load_URI(uri)['pos']

    ser = {}
    chunk = np.searchsorted(blocks, idx, side='right')
    ser['chunk'] = [chunk]

    bins = None
    contacts = None
    for st, end in zip(chain_starts, chain_ends):
        conf = data[st:end,:]
        x,y = polymer_analyses.contact_scaling(conf, bins0=bin_edges, cutoff=cutoff)
        if bins is None:
            bins = x
        if contacts is None:
            contacts = y
        else:
            contact = contacts + y

    ser['Ps'] = [(bins, contacts)]
    return pd.DataFrame(ser)

def save_scalings_all_blocks(basepath, sim, bin_edges, chain_starts, 
                             chain_ends, blocks, savepath=None, cutoff_rad=2.0):
    #basepath = f'/net/levsha/share/deepti/simulations/active16_inactive1_act7.0/run{run}'
    #sim = f'inactive1_act7_run{run}'
    uris = list_URIs(f'{basepath}')
    if savepath is None:
        savepath = f'{basepath}/results'
    if not Path(savepath).is_dir():
        Path(savepath).mkdir()
    savename = f'{sim}_cutoff{cutoff_rad:.1f}.npy'
    f = partial(process, blocks=blocks, bin_edges=bin_edges, chain_starts=chain_starts, 
                chain_ends=chain_ends, cutoff=cutoff_rad)
    with mp.Pool(20) as p:
        results = p.imap_unordered(f, uris, chunksize=5)
        df = polymer_analyses.streaming_ndarray_agg(results, 
                                                    chunksize=5000,
                                                    ndarray_cols=['Ps'], 
                                                    aggregate_cols=['chunk'], 
                                                    add_count_col=True, divide_by_count=True
                                                    )

    result = []
    for _, row in df.iterrows():
        arr = row['Ps']
        arr[1,0] = 1
        result.append(arr)

    result = np.dstack(result)
    np.save(f'{savepath}/{savename}', result)
    print(savename)
    
def calculate_scalings(basepath='/net/levsha/share/deepti/simulations/active16_inactive1_act19.0_center',
                      inactive_loc="center"):
    basepath = Path(basepath)
    runs = [d for d in basepath.glob('run*')]
    with h5py.File(runs[0]/"initArgs_0.h5", 'r') as f:
        N = f.attrs['N']
    #total number of monomers
    #chromosome size
    mapN = 1000
    nchroms = int(N / mapN)
    if inactive_loc == "center":
        inactive_chain_starts = [0]
        inactive_chain_ends = [mapN]
        active_chain_starts = np.arange(mapN, N, mapN)
        active_chain_ends = np.arange(mapN + mapN, N+1, mapN)
        for st, end in zip(active_chain_starts, active_chain_ends):
            print(f'{st}, {end}')
    else:
        active_chain_starts = np.arange(0, (N - mapN), mapN)
        active_chain_ends = np.arange(mapN, (N - mapN)+1, mapN)
        inactive_chain_ends = np.arange(N - mapN, N, mapN)
        inactive_chain_ends = [N]
        
    T_blocks = 20000
    integrations_per_save = 2000
    bin_edges = numutils._logbins_numba(1, mapN, ratio=1.2, prepend_zero=True)
    start = int(0.005*T_blocks)
    end = T_blocks
    block_ratio = 1.25
    blocks = numutils._logbins_numba(start, end, ratio=block_ratio, prepend_zero=False)
    savepath = basepath / 'scalings'
    for run in runs:
        #if not (savepath / f'inactive1_act7_{run.name}_cutoff2.0.npy').exists():
        save_scalings_all_blocks(run, f'inactive1_act19_{run.name}', bin_edges, inactive_chain_starts, 
                                 inactive_chain_ends, blocks, savepath=savepath, cutoff_rad=2.0)
        #if not (savepath / f'active16_act7_{run.name}_cutoff2.0.npy').exists():
        save_scalings_all_blocks(run, f'active16_act19_{run.name}', bin_edges, active_chain_starts, 
                                 active_chain_ends, blocks, savepath=savepath, cutoff_rad=2.0)

