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
from post_processing.compscores import comp_score_1
from post_processing.msd import *
from post_processing.scoring import compute_AB_density_scores

def process_param_sweep(simdir=Path('/net/levsha/share/deepti/simulations/chr21_Su2020')):
    savepath = Path('/net/levsha/share/deepti/data') 
    basepaths = [d for d in simdir.glob('*rep3.0*')]
    simstrings = [str(d.name) for d in simdir.glob('*rep3.0*')]
    radius_of_gyration = []
    for i, basepath in enumerate(basepaths):
        mapN = 1302
        conf_file = savepath / f"conformations/conformations_{simstrings[i]}.npy"
        if not conf_file.is_file() and (basepath/'runs20000_2000_20copies').is_dir():
            conformations = extract(basepath/'runs20000_2000_20copies', start=10000, every_other=100)
            print(f"Extracted {len(conformations)} conformations for simulation {simstrings[i]}")
            np.save(conf_file, conformations)
            Rg2 = mean_squared_separation(conformations, savepath/'distance_maps', simstrings[i], metric='euclidean', N=mapN)
            radius_of_gyration.append(Rg2)
            mat = contactmaps.monomerResolutionContactMapSubchains(
                filenames=conformations, mapStarts=[i*mapN for i in range(20)], mapN=mapN, cutoff=2.0
            )
            mat2 = mat / (len(conformations)*20)
            # save cutoff radius = 2.0 contact map
            np.save(savepath/f"contact_maps/contact_map_{simstrings[i]}_cutoff2.0.npy", mat2)
    return radius_of_gyration
    #save radius of gyration
    #df = pd.DataFrame()
    #df['sim'] = simstrings
    #df['Rg2'] = radius_of_gyration
    #df.to_csv(savepath/"distance_maps/radius_of_gyration_chr21_Su2020.csv", index=False)

def linear_relaxation(basepath=Path("/net/levsha/share/deepti/data/linear_relaxation/chr21_Su2020")):
    """ Compute comp scores over time for each hybrid linear model in this folder. """
    for model in basepath.iterdir():
        if model.is_dir() and not (model/f"comp_score_1_dynamics_{str(model.name)}.csv").is_file():
            print(model)
            simstring = str(model.name)
            dict_list = []
            contact_map_files = [filename for filename in model.glob('contact*.csv') if filename.is_file()]
            for filename in tqdm(contact_map_files):
                t = float(str(filename).split('_')[-1][1:-4])
                contact_map = pd.read_csv(filename).to_numpy()
                #compute comp score over time
                cs, csA, csB = comp_score_1(contact_map)
                dict_list.append({"t" : t, "cs1_B" : np.nanmean(csB[200:1000])})
            df = pd.DataFrame.from_dict(dict_list)
            df.to_csv(model/f"comp_score_1_dynamics_{simstring}.csv", index=False)
    
def comp_scores_over_time(basepath,
    simstring,
    timepoints=None,
    ntimepoints=None,
    traj_length=None,
    mapN=1302,
    nchains=20,
    time_between_snapshots=1,
    time_window=10,
    save_maps=False,
    savepath=Path("/net/levsha/share/deepti/data"),
):
    """Plot an ensemble-averaged contact map at multiple `timepoints` in a simulation trajectory.
    Use 10 conformations centered around each time point from each simulation to get better
    statistics.

    Parameters
    ----------
    basepath : str or Path
        parent directory where each subdirectory is a simulation replicate for one set of parameters
    simstring : str
        tag associated with simulation
    ntimepoints : int
        number of time points at which to construct contact maps
    traj_length : int
        max time of simulation trajectory (in time blocks)
    time_between_snapshots : int
        number of blocks between correlated snapshots included in ensemble for each time point.
        Defaults to 1.
    time_window : int
        window (in number of time blocks) within which to take correlated snapshots for each time point
        ex: time_window of 10 implies snapshots will be taken between [t-5, t+5] for each time point t.
    savepath : str or Path
        path to directory where mean squared separation file will be written

    """

    # time spacing
    if timepoints is None:
        DT = traj_length / ntimepoints
        timepoints = np.arange(DT, (ntimepoints + 1) * DT, DT)
    half_window = time_window // 2
    dict_list = []
    
    for t in timepoints:
        # take 11 snapshots centered at t (5 before, 5 after) to average over
        start = int(t - half_window)
        end = int(t + half_window + 1)
        conformations = extract(
                basepath, start=start, end=end, every_other=time_between_snapshots
            )
        mat = contactmaps.monomerResolutionContactMapSubchains(
                filenames=conformations, mapStarts=[i*mapN for i in range(nchains)], mapN=mapN, cutoff=2.0
            )
        mat2 = mat / (len(conformations)*nchains)
        # save cutoff radius = 2.0 contact map
        if save_maps:
                np.save(savepath/f"comp_score_dynamics/contact_map_{simstring}_t{int(t)}_window{time_window}_snapshotDT_{time_between_snapshots}_cutoff2.0.npy", mat2)
        #compute comp score over time
        cs, csA, csB = comp_score_1(mat2)
        dict_list.append({"t" : t, "cs1" : np.nanmean(cs[200:1000]),
                   "cs1_A" : np.nanmean(csA[200:1000]), "cs1_B" : np.nanmean(csB[200:1000])})
    df = pd.DataFrame.from_dict(dict_list)
    df.to_csv(savepath/f"comp_score_dynamics/comp_score_1_{simstring}_logspaced.csv", index=False)
        
def comp_scores_dynamics(E0s, acts, savepath=Path("/net/levsha/share/deepti/data")):
    """ Extract conformations at different time points from each simulation."""
    
    basepath = Path("/net/levsha/share/deepti/simulations/chr21_Su2020")
    simpaths = []
    #compute comp score at 100 time points log-spaced (ish) between t=0 and t=10000*2000
    times = np.logspace(np.log10(50), np.log10(10000*2000), 100)
    early_block_nums = np.unique(np.rint(times / 50)[times <= 2000*2000]).astype(int)
    late_block_times = times[times >= (2000*2000 + 5000)]
    late_block_nums = (np.rint((late_block_times - 2000*2000) / 5000) + 80000).astype(int)
    #block numbers of desired time points
    timepoints = np.concatenate((early_block_nums, late_block_nums))
    
    for E0, act in tqdm(list(zip(E0s, acts))):
        sim = basepath/f"stickyBB_{E0}_act{act:.0f}_rep3.0_1302_sphwellarray_width20_depth20"
        filename = savepath/f"comp_score_dynamics/comp_score_1_stickyBB_{E0}_act{act:.0f}_logspaced.csv"
        if (sim/'runs20000_2000_200copies').is_dir() and not filename.is_file():
            print(f'Computing comp score dynamics for E0={E0}, act={act}')
            if act==1 and E0 in [0.15, 0.1, 0.05]:
                #log stepping dynamics -- only 100 blocks saved, use all time points
                comp_scores_over_time(sim/'runs20000_2000_200copies', 
                                   f'stickyBB_{E0}_act{act:.0f}', 
                                      time_window=0, timepoints=np.arange(0, 100, 1), nchains=200)   
            else:
                comp_scores_over_time(sim/'runs20000_2000_200copies', 
                                  f'stickyBB_{E0}_act{act:.0f}', timepoints=timepoints, nchains=200)

def save_Rg2_over_time(start=1000, every_other=10, end=None):
    """ Compute squared radius of gyration averaged over subchains in single
    simulations. """

    simdir = Path('/net/levsha/share/deepti/simulations/chr21_Su2020')
    savepath = Path('/net/levsha/share/deepti/data') 

    for d in simdir.glob('*rep3.0*'):
        simstring = str(d.name)
        #N = int(simstring[-4:])
        N = 1302
        Rg2file = savepath/f'msds/Rg2_over_time_{simstring}_20chain_ave.csv'
        if not Rg2file.is_file() and (d/'runs20000_2000_20copies').is_dir():
            Rg2 = compute_single_trajectory_Rg2(d/'runs20000_2000_20copies', 
                                                start=start, every_other=every_other,
                                                end=end, N=N)
            df = pd.DataFrame()
            df["Time"] = np.arange(0, len(Rg2)) * every_other
            df["Rg2"] = Rg2
            df.to_csv(Rg2file, index=False)

def save_msd_over_time(start=0, every_other=10, end=None):
    """ Compute squared radius of gyration averaged over subchains in single
    simulations. """

    simdir = Path('/net/levsha/share/deepti/simulations/chr21_Su2020')
    savepath = Path('/net/levsha/share/deepti/data') 
    for d in simdir.glob('*rep3.0*'):
        simstring = str(d.name)
        N = 1302
        monomers_per_locus = int(N / 651)
        ids = np.load(savepath/f'ABidentities_chr21_Su2020_{monomers_per_locus}perlocus.npy')
        msdfile = savepath/f'msds/msds_{simstring}_ens_ave.csv'
        if (d/'runs600_2000_20copies').is_dir():
            msd = compute_single_trajectory_msd(d/'runs600_2000_20copies', 
                                                start=start, every_other=every_other,
                                                end=end, N=N)
            df = pd.DataFrame()
            df["Time"] = np.arange(0, len(msd)) * every_other
            df["active_MSD"] = msd[:, ids==1].mean(axis=1)
            df["inactive_MSD"] = msd[:, ids==0].mean(axis=1)
            print(df)
            df.to_csv(msdfile, index=False)

def ABsegregation_param_sweep(simdir=Path('/net/levsha/share/deepti/simulations/chr21_Su2020'),
                             ncores=25):
    savepath = Path('/net/levsha/share/deepti/data') 
    conf_files = []
    simstrings = []
    for d in simdir.glob('*rep3.0*'):
        simstring = str(d.name)
        conf_file = savepath / f"conformations/conformations_{simstring}.npy"
        var_file = savepath/f'variability/ABsegregation_{simstring}.csv'
        if not var_file.is_file() and conf_file.is_file():
            print(simstring)
            conf_files.append(conf_file)
            simstrings.append(simstring)
    i = 0
    for conf in tqdm(conf_files):
        df = compute_AB_density_scores(conf)
        df.to_csv(savepath/f'variability/ABsegregation_{simstrings[i]}.csv', index=False)
        i += 1
        
def save_MSD_param_sweep(ids=None, ncores=25, start=10000, every_other=10):
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
    simdir = Path('/net/levsha/share/deepti/simulations/chr21_Su2020')
    savepath = Path('/net/levsha/share/deepti/data') 
    basepaths = []
    simstrings = []
    for d in simdir.glob('*rep3.0*'):
        simstring = str(d.name)
        msdfile = savepath/f'msds/msds_{simstring}_ens_ave.csv'
        if not msdfile.is_file() and (d/'runs20000_2000_20copies').is_dir():
            print(simstring)
            basepaths.append(d/'runs20000_2000_20copies')
            simstrings.append(simstring)
    print(simstrings)
    # 0 is cold (B) and 1 is hot (A)
    if ids is None:
        ids = np.load(savepath/'ABidentities_chr21_Su2020_2perlocus.npy')
    N = len(ids)
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
       df.to_csv(savepath/f'msds/msds_{simstrings[i]}_ens_ave.csv', index=False)
    return msds

if __name__ == "__main__":
    #dfs = ABsegregation_param_sweep()
    #comp_scores_dynamics([0.0, 0.15, 0.05, 0.10, 0.25], [5, 2, 4, 3, 1]) 
    #comp_scores_dynamics([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1, 2, 3, 4, 5, 7, 8, 9, 10])
    #comp_scores_dynamics([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [1, 3, 5, 7, 8, 9, 10])
    #Rg2 = process_param_sweep()
    linear_relaxation()
