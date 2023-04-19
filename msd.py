r"""
Computing MSDS from polychrom simulations
=========================================

Script to calculate monomer mean squared displacements over time
from output of polychrom simulations. MSDs can either be computed
by (1) averaging over an ensemble of trajectories or (2) time lag averaging
using a single trajectory.

Deepti Kannan. 2022
"""

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from numba import jit
from polychrom.hdf5_format import list_URIs, load_hdf5_file, load_URI


def extract_hot_cold(simdir, D, start=100000, every_other=10):
    """Load conformations from a simulation trajectory stored in the hdf5 files in simdir
    and store in two matrices, one for the `A` type monomers, and one for the `B` monomers.

    Parameters
    ----------
    simdir : str or Path
        path to simulation directory containing .h5 files
    D : np.ndarray
        array of monomer diffusion coefficients.
        Assumes there are only 2 values: D.min() and D.max().
    start : int
        which time block to start loading conformations from
    every_other : int
        skip every_other time steps when loading conformations

    Returns
    -------
    Xhot : array_like (num_t, N_A, 3)
        x, y, z positions of all N_A active (hot) monomers over time
    Xcold : array-like (num_t, N_B, 3)
        x, y, z positions of all N_B inactive (cold) monomers over time

    """
    X = []
    data = list_URIs(simdir)
    if start == 0:
        starting_pos = load_hdf5_file(Path(simdir) / "starting_conformation_0.h5")[
            "pos"
        ]
        X.append(starting_pos)
    for conformation in data[start::every_other]:
        pos = load_URI(conformation)["pos"]
        X.append(pos)
    X = np.array(X)
    Xcold = X[:, D == D.min(), :]
    Xhot = X[:, D == D.max(), :]
    return Xhot, Xcold


@jit(nopython=True)
def get_bead_msd_time_ave(Xhot, Xcold):
    """Calculate time lag averaged monomer MSDs for active (hot) and inactive(cold)
    regions from a single simulation trajectory stored in Xhot, Xcold.

    Parameters
    ----------
    Xhot : np.ndarray (num_t, num_hot, d)
        trajectory of hot monomer positions in d dimensions over num_t timepoints
    Xcold : np.ndarray (num_t, num_cold, d)
        trajectory of cold monomer positions in d dimensions over num_t timepoints

    Returns
    -------
    hot_msd_ave : (num_t - 1,)
        time lag averaged MSD averaged over all hot monomers
    cold_msd_ave : (num_t - 1,)
        time lag averaged MSD averaged over all cold monomers

    """
    num_t, num_hot, d = Xhot.shape
    hot_msd = np.zeros((num_t - 1,))
    cold_msd = np.zeros((num_t - 1,))
    count = np.zeros((num_t - 1,))
    for i in range(num_t - 1):
        for j in range(i, num_t - 1):
            diff = Xhot[j] - Xhot[i]
            hot_msd[j - i] += np.mean(np.sum(diff * diff, axis=-1))
            diff = Xcold[j] - Xcold[i]
            cold_msd[j - i] += np.mean(np.sum(diff * diff, axis=-1))
            count[j - i] += 1
    hot_msd_ave = hot_msd / count
    cold_msd_ave = cold_msd / count
    return hot_msd_ave, cold_msd_ave


def compute_single_trajectory_msd(simdir, start=100000, every_other=1):
    """Compute MSDs for all N monomers over time using `ever_other` conformations
    starting at time point `start` from a single simulation in `simdir`.

    Returns
    -------
    dxs : (n_timesteps, N)
        MSDs (columns) over time (rows) of each of the N monomers

    """

    simdir = Path(simdir)
    data = list_URIs(simdir)
    if start == 0:
        starting_pos = load_hdf5_file(simdir / "starting_conformation_0.h5")["pos"]
    else:
        starting_pos = load_URI(data[start])["pos"]
    dxs = []
    for conformation in data[start::every_other]:
        pos = load_URI(conformation)["pos"]
        dx_squared = np.sum((pos - starting_pos) ** 2, axis=-1)
        dxs.append(dx_squared)
    dxs = np.array(dxs)
    print(simdir)
    return dxs


def save_MSD_ensemble_ave(basepath, savefile, every_other=1, ncores=25):
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
    # 0 is cold (B) and 1 is hot (A)
    ABids = np.loadtxt(
        "data/ABidentities_Zhang_Blobel2021_chr2_35Mb_60Mb.csv", dtype=str
    )
    ids = (ABids == "A").astype(int)
    basepath = Path(basepath)
    rundirs = [f for f in basepath.iterdir() if f.is_dir()]
    with mp.Pool(ncores) as p:
        msds = p.map(compute_single_trajectory_msd, rundirs)
    msds = np.array(msds)  # shape: (#runs, #timesteps, #monomers)
    # average over ensemble and over all monomers that have the same activity
    hot_msd_ave = np.mean(msds[:, :, ids == 1], axis=(0, -1))
    cold_msd_ave = np.mean(msds[:, :, ids == 0], axis=(0, -1))
    df = pd.DataFrame(columns=["Time", "active_MSD", "inactive_MSD"])
    # in units of blocks of 100 time steps
    df["Time"] = np.arange(0, len(hot_msd_ave)) * every_other
    df["active_MSD"] = hot_msd_ave
    df["inactive_MSD"] = cold_msd_ave
    df.to_csv(savefile, index=False)


def save_MSD_time_ave(simpath, D, savepath, every_other=10):
    """Compute time lag averaged MSDs averaged over active and inactive regions
    from a single simulation trajectory in simpath. Takes ~30 min for a simulation with
    10,000 conformations.

    Parameters
    ----------
    simpath : str or Path
        path to simulation directory
    D : array-like
        array where D==D.max() selects out A monomers and D==D.min() selects B monomers
    savepath : str or Path
        path to .csv file where MSDs will be saved
    every_other : int
        skip every_other conformation when loading conformations for MSD computation
    """
    Xhot, Xcold = extract_hot_cold(
        Path(simpath), D, start=100000, every_other=every_other
    )
    hot_msd, cold_msd = get_bead_msd_time_ave(Xhot, Xcold)
    df_msd = pd.DataFrame()
    df_msd["Times"] = np.arange(0, len(hot_msd)) * every_other
    df_msd["MSD_A"] = hot_msd
    df_msd["MSD_B"] = cold_msd
    df_msd.to_csv(savepath)
