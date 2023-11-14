""" Script to visualize simulation snapshots / make animations.

Requires nglutils (pip install -U git+https://github.com/mirnylab/nglutils)
and nglview (conda install nglview -c conda-forge)

For best results, run in a clean conda environment with provided install_visualization.sh script.

Deepti Kannan, 2023"""

from pathlib import Path
from string import ascii_uppercase

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import nglutils as ngu
import nglview as nv
import numpy as np
import polychrom
from polychrom.hdf5_format import list_URIs, load_hdf5_file, load_URI


def get_monomer_names_from_ids(ids):
    """Convert an array of integer monomer types (0 for B, 1 for A)
    to atom names."""
    assert np.all(np.logical_or(ids == 0, ids == 1))
    N = len(ids)
    monomer_ids = np.zeros((N,), dtype=int)
    monomer_ids[ids == 0] = 1  # type B(cold)
    monomer_ids[ids == 1] = 0  # type A (hot)
    # convert integer monomer types to atom namesgt
    monomer_names = ngu.intlist_to_alpha(monomer_ids)
    return monomer_names


def get_chrom_names(nchains, mapN):
    """Assign monomers of each chromosome a different atom name
    to visualize chains in different colors.

    Parameters
    ----------
    nchains : int
        number of chains
    mapN : int
        number of monomers in each chain

    """
    chromIDs = np.repeat(np.arange(0, nchains), mapN)
    chrom_names = ngu.intlist_to_alpha(chromIDs)
    return chrom_names


def visualize(X, nchains, mapN=None, ids=None, chrom=None, color_by="monomer_id"):
    """Visualize a simulation snapshot or animation, coloring monomers either by A/B identity or
    by chromosomes number.

    Parameters
    ----------
    X : np.ndarray[float] (totalN, 3) or (num_t, totalN, 3)
        matrix of monomer positions. if 3-dimensional, first dimension is time.
    nchains : int
        number of chains in simulation
    ids : array-like (mapN,)
        array of monomer types (0 for B, 1 for A)
    mapN : int
        number of monomers per chain. If None, infers from len(ids).
    chrom : int
        Defaults to None. Chromosome to visualize.
    color_by : str
        if "monomer_id", color by A/B identity.
        if "chrom_id", color by chromosomes.
    """
    if (X.ndim != 2 and X.ndim != 3) or X.shape[-1] != 3:
        raise ValueError("X should have shape (num_t, N, 3) or (N, 3)")

    monomer_names = get_monomer_names_from_ids(ids)
    # number of monomers in one subchain
    if mapN is None:
        mapN = len(monomer_names)
    # copy the list nchains times
    try : 
        all_monomer_names = monomer_names * nchains
    except:
        raise TypeError("nchains must be an integer")
    chrom_names = get_chrom_names(nchains, mapN)
    # total N = number of monomers in one subchain * nchains
    totalN = nchains * mapN

    if chrom is not None:
        # visualize a single chromosome
        # chrom should be an integer between 0 and nchains - 1
        assert chrom in np.arange(0, nchains, dtype=int)
        if color_by == "monomer_id":
            top = ngu.mdtop_for_polymer(mapN, atom_names=monomer_names)
        else:
            top = ngu.mdtop_for_polymer(
                mapN, atom_names=chrom_names[chrom * mapN : (chrom + 1) * mapN]
            )
        if X.ndim == 3:
            # this is an animation! first dimension is time
            view = ngu.xyz2nglview(X[:, chrom * mapN : (chrom + 1) * mapN, :], top=top)
        else:
            # this is a snapshot
            view = ngu.xyz2nglview(X[chrom * mapN : (chrom + 1) * mapN], top=top)
    else:
        # visualize all `nchains` chromosomes
        if color_by == "monomer_id":
            top = ngu.mdtop_for_polymer(
                totalN,
                atom_names=all_monomer_names,
                chains=[(i * mapN, i * mapN + mapN, False) for i in range(0, nchains)],
            )
        else:
            top = ngu.mdtop_for_polymer(
                totalN,
                atom_names=chrom_names,
                chains=[(i * mapN, i * mapN + mapN, False) for i in range(0, nchains)],
            )
        view = ngu.xyz2nglview(X, top=top)
    view.center()
    if color_by == "monomer_id":
        view.add_representation(
            "ball+stick", selection=".A", colorScheme="uniform", colorValue=0xFF4242
        )

        view.add_representation(
            "ball+stick", selection=".B", colorScheme="uniform", colorValue=0x475FD0
        )
    elif color_by == "chrom_id":
        cmap = cm.get_cmap("tab20")
        color_codes = [cmap(i % 20) for i in range(nchains)]
        hex_color_codes = [mcolors.to_hex(color[:3]) for color in color_codes]
        for i in range(nchains):
            view.add_representation(
                "ball+stick",
                selection=f".{ascii_uppercase[i % 26]}",
                colorScheme="uniform",
                colorValue=hex_color_codes[i],
            )
    return view


def extract_trajectory(simdir, start=0, end=-1, every_other=10):
    """Load conformations from a simulation trajectory stored in the hdf5 files in simdir.

    Parameters
    ----------
    simdir : str or Path
        path to simulation directory containing .h5 files
    start : int
        which time block to start loading conformations from
    end : int
        which time block to stop loading conformations from
    every_other : int
        skip every_other time steps when loading conformations

    Returns
    -------
    X : array_like (num_t, N, 3)
        x, y, z positions of all monomers over time

    """
    X = []
    data = list_URIs(simdir)
    if start == 0:
        starting_pos = load_hdf5_file(Path(simdir) / "starting_conformation_0.h5")[
            "pos"
        ]
        X.append(starting_pos)
    for conformation in data[start:end:every_other]:
        pos = load_URI(conformation)["pos"]
        X.append(pos)
    X = np.array(X)
    return X
