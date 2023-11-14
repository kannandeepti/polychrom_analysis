r"""
Extract structural observables from polychrom simulations
=========================================================

The functions in this script assume a directory structure where each parameter
sweep is stored in a parent directory and all subdirectories are titled 'run0', 'run1', etc.
In Goychuk et al. PNAS (2023), we ran 200 independent runs per parameter set to compute
ensemble-averaged observables such as the mean squared separation map, contact map, and contour
correlation. Functions to compute all of these are available here.

Deepti Kannan. 2022
"""

import itertools
import multiprocessing as mp
import sys
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import polychrom
except:
    sys.path.append("/home/dkannan/git-remotes/polychrom/")
    import polychrom

from polychrom import contactmaps
from polychrom.hdf5_format import list_URIs, load_URI
from scipy.spatial.distance import pdist, squareform


def extract(path, start=10000, every_other=100, end=20000):
    """Extract independent snapshots from a single simulation.

    Parameters
    ----------
    path : str
        path to simulation directory
    start : int
        block number to start extracting files from
    every_other : int
        skip this number of blocks in between snapshots (should be large enough
        so that snapshots are decorrelated)
    end : int
        last block to include in trajectory.
    """
    try:
        confs = list_URIs(path)
        if end:
            uris = confs[start:end:every_other]
        else:
            uris = confs[start::every_other]
    except:
        raise Exception("Exception! Something went wrong")
        uris = []
    return uris


def extract_conformations(basepath, ncores=24, chain=True, **kwargs):
    """Extract conformations from multiple simulation replicates to be included in
    ensemble-averaged observables.

    Parameters
    ----------
    basepath : str or Path
        parent directory where each subdirectory is a simulation replicate for one set of parameters
    ncores : int
        number of cores available for parallelization
    chain : bool
        whether to aggregate conformations from multiple simulations into one list.
        Defaults to True.

    Returns
    -------
    conformations : list
        If chain is True, list of hdf5 filenames containing polymer conformations.
        If chain is False, list of lists, where each sublist is from a separate simulation run.
    """
    basepath = Path(basepath)
    rundirs = [f for f in basepath.iterdir() if f.is_dir()]
    runs = len(rundirs)
    extract_func = partial(extract, **kwargs)
    with mp.Pool(ncores) as p:
        confs = p.map(extract_func, rundirs)
    if chain:
        conformations = list(itertools.chain.from_iterable(confs))
        print(f"Number of simulations in directory: {runs}")
        print(f"Number of conformations extracted: {len(conformations)}")
        return conformations, runs
    else:
        return confs, runs


def mean_squared_separation(
    conformations, savepath, simstring, rsquared=False, metric="sqeuclidean", N=1000
):
    """Compute mean squared separation between all pairs of monomers averaged over all
    conformations. Saves N x N matrix to csv file. Also saves mean squared distance
    from each monomer to the origin to csv file.

    Parameters
    ----------
    conformations : list of str
        list of paths to hdf5 files (output of list_URIs)
    savepath : str or Path
        path to directory where mean squared separation file will be written
    simstring : str
        tag associated with simulation
    N : int
        number of monomers

    """
    # mean squared separation between all pairs of monomers
    msd = np.zeros((N, N))
    # mean radius of gyration of subchains
    Rg2 = 0.0
    # mean squared separation between each monomer and origin
    if rsquared:
        rsquared = np.zeros((N,))
    num_confs = 0
    for conformation in conformations:
        pos = load_URI(conformation)["pos"]
        ncopies = pos.shape[0] // N
        for i in range(ncopies):
            posN = pos[N * i : N * (i + 1)]
            if rsquared:
                rsquared += np.sum(posN**2, axis=1)
            Rg2 += np.mean((posN - np.mean(posN, axis=0)) ** 2) * 3
            dist = pdist(posN, metric=metric)
            Y = squareform(dist)
            msd += Y
        num_confs += ncopies
    msd /= num_confs
    Rg2 /= num_confs
    if rsquared:
        rsquared /= num_confs
        df2 = pd.DataFrame(rsquared)
        df2.to_csv(Path(savepath) / f"rsquared_{simstring}.csv", index=False)
    df = pd.DataFrame(msd)
    df.to_csv(Path(savepath) / f"mean_{metric}_distance_{simstring}.csv", index=False)
    return Rg2


def contour_alignment(savepath, simstring):
    """Calculate contour alignment :math:`\langle (r_{i+1} - r_i) \cdot (r_{j+1}-r_{j}) \rangle`
    using presaved mean squared separation matrix and mean squared distances of all monomers to the
    origin. Save N x N matrix to csv file in savepath.

    Parameters
    ----------
    savepath : str or Path
        path to directory where contour alignment file will be written
    simstring : str
        tag associated with simulation

    """

    # load mean squared separation <(r_i - r_j)^2>
    msd = pd.read_csv(
        Path(savepath) / f"mean_squared_separation_{simstring}.csv"
    ).to_numpy()
    N, _ = msd.shape
    # load <r^2>
    rsquared = pd.read_csv(Path(savepath) / f"rsquared_{simstring}.csv").to_numpy()
    r2i = np.tile(rsquared, (1, N))
    r2j = np.tile(rsquared.T, (N, 1))
    rirj = (r2i + r2j - msd) / 2
    contour_corr = rirj[:-1, :-1] + rirj[1:, 1:] - rirj[1:, :-1] - rirj[:-1, 1:]
    df = pd.DataFrame(contour_corr)
    df.to_csv(Path(savepath) / f"contour_alignment_{simstring}.csv")


def contact_maps_over_time(
    basepath,
    simstring,
    ntimepoints,
    traj_length,
    time_between_snapshots=1,
    time_window=10,
    save_confs=False,
    savepath=Path("data"),
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
    DT = traj_length / ntimepoints
    timepoints = np.arange(DT, (ntimepoints + 1) * DT, DT)
    print(timepoints)
    half_window = time_window // 2

    for t in timepoints:
        # take 11 snapshots centered at t (5 before, 5 after) to average over
        start = int(t - half_window)
        end = int(t + half_window + 1)
        conf_file = savepath / f"conformations_{simstring}_t{int(t)}.npy"
        if conf_file.is_file():
            conformations = np.load(conf_file)
            runs = len([f for f in basepath.iterdir() if f.is_dir()])
        else:
            conformations, runs = extract_conformations(
                basepath, start=start, end=end, every_other=time_between_snapshots
            )
            if save_confs:
                np.save(conf_file, conformations)

        mat = contactmaps.monomerResolutionContactMap(
            filenames=conformations, cutoff=2.0
        )
        mat2 = mat / len(conformations)
        # save cutoff radius = 2.0 contact map
        np.save(
            savepath
            / f"linear_relaxation/contact_map_{simstring}_t{int(t)}_window{time_window}_snapshotDT_{time_between_snapshots}_cutoff2.0.npy",
            mat2,
        )


def process_existing_simulations(simdir=None, savepath=Path("data")):
    """Script to look inside a simulation directory, find all parameter sweeps that have
    been done so far, extract conformations, calculate mean squared separations, and
    save contact maps."""

    basepaths = [d for d in simdir.iterdir()]
    simstrings = [str(d.name) for d in simdir.iterdir()]
    print(simstrings)
    for i, basepath in enumerate(basepaths):
        conf_file = savepath / f"conformations/conformations_{simstrings[i]}.npy"
        if conf_file.is_file():
            conformations = np.load(conf_file)
            runs = len([f for f in basepath.iterdir() if f.is_dir()])
        else:
            conformations, runs = extract_conformations(basepath)
            print(f"Extract conformations for simulation {simstrings[i]}")
            np.save(conf_file, conformations)
        if not (savepath / f"mean_squared_separation_{simstrings[i]}.csv").is_file():
            mean_squared_separation(conformations, savepath, simstrings[i])
            print(f"Computed mean squared separation for simulation")
        if not (savepath / f"contact_map_{simstrings[i]}_cutoff2.0.npy").is_file():
            mat = contactmaps.monomerResolutionContactMap(
                filenames=conformations, cutoff=2.0
            )
            mat2 = mat / len(conformations)
            # save cutoff radius = 2.0 contact map
            np.save(f"data/contact_map_{simstrings[i]}_cutoff2.0.npy", mat2)
