""" Script to visualize simulation snapshots

Deepti Kannan, 2023"""

import os
from pathlib import Path
import importlib as imp
from collections import defaultdict
import h5py
import json
from copy import deepcopy
import multiprocessing as mp

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import polychrom
from polychrom.hdf5_format import list_URIs, load_URI, load_hdf5_file

import nglutils as ngu
import nglview as nv

from string import ascii_uppercase
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def get_monomer_names_from_ids(ids):
    """ Convert an array of integer monomer types (0 for B, 1 for A)
    to atom names."""
    assert(np.all(np.logical_or(ids == 0, ids == 1)))
    N = len(ids)
    monomer_ids = np.zeros((N,), dtype=int)
    monomer_ids[ids==0] = 1 #type B(cold)
    monomer_ids[ids==1] = 0 #type A (hot)
    #convert integer monomer types to atom names
    monomer_names = ngu.intlist_to_alpha(monomer_ids)
    return monomer_names

def get_chrom_names(nchains, mapN):
    """ Assign monomers of each chromosome a different atom name
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

def visualize_snapshot(X, nchains, mapN=None, ids=None, run=-1, chrom=None, color_by="monomer_id"):
    """ Visualize a simulation snapshot, coloring monomers either by A/B identity or
    by chromosomes number.
    
    Parameters
    ----------
    X : np.ndarray[float] (totalN, 3)
        matrix of monomer positions
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
    monomer_names = get_monomer_names_from_ids(ids)
    #number of monomers in one subchain
    if mapN is None:
        mapN = len(monomer_names)
    #copy the list nchains times
    all_monomer_names = monomer_names * nchains
    chrom_names = get_chrom_names(nchains, mapN)
    #total N = number of monomers in one subchain * nchains
    totalN = nchains * mapN

    if chrom is not None:
        #visualize a single chromosome
        #chrom should be an integer between 0 and nchains - 1
        assert(chrom in np.arange(0, nchains, dtype=int))
        if color_by=="monomer_id":
            top = ngu.mdtop_for_polymer(mapN, atom_names=monomer_names)
        else:
            top = ngu.mdtop_for_polymer(mapN, atom_names=chrom_names[chrom*mapN : (chrom+1)*mapN])
        view = ngu.xyz2nglview(X[chrom * mapN : (chrom + 1) * mapN], top=top)
    else:
        #visualize all `nchains` chromosomes
        if color_by=="monomer_id":
            top = ngu.mdtop_for_polymer(totalN, atom_names=all_monomer_names)
        else:
            top = ngu.mdtop_for_polymer(totalN, atom_names=chrom_names)
        view = ngu.xyz2nglview(X, top=top)
    view.center()
    if color_by == "monomer_id":
        view.add_representation('ball+stick', selection='.A',
                                            colorScheme='uniform',
                                            colorValue=0xff4242)

        view.add_representation('ball+stick', selection='.B',
                                              colorScheme='uniform',
                                              colorValue=0x475FD0)
    elif color_by == "chrom_id":
        cmap = cm.get_cmap('tab20')
        color_codes = [cmap(i % 20) for i in range(nchains)]
        hex_color_codes = [mcolors.to_hex(color[:3]) for color in color_codes]
        for i in range(nchains):
            view.add_representation('ball+stick', selection=f'.{ascii_uppercase[i % 26]}',
                                            colorScheme='uniform',
                                            colorValue=hex_color_codes[i])
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
        starting_pos = load_hdf5_file(Path(simdir)/"starting_conformation_0.h5")['pos']
        X.append(starting_pos)
    for conformation in data[start:end:every_other]:
        pos = load_URI(conformation)['pos']
        X.append(pos)
    X = np.array(X)
    return X

def make_animation(X, nchains=None, mapN=None, ids=None, color_by='monomer_id'):
    """ Make an animation with NGLview of a simulation trajectory.
    
    Parameters
    ----------
    X : np.ndarray[float] (totalN, 3)
        matrix of monomer positions
    nchains : int
        number of chains in simulation
    ids : array-like (mapN,)
        array of monomer types (0 for B, 1 for A)
    mapN : int
        number of monomers per chain. If None, infers from len(ids).
    color_by : str
        if "monomer_id", color by A/B identity.
        if "chrom_id", color by chromosome
    
    """
    if color_by=='monomer_id':
        monomer_names = get_monomer_names_from_ids(ids)
        monomer_names = monomer_names * nchains
    else:
        monomer_names = get_chrom_names(nchains, mapN)
    N = len(monomer_names)
    top = ngu.mdtop_for_polymer(N, atom_names=monomer_names,
                               chains=[(i*mapN, i*mapN + mapN, False) for i in range(0, nchains)])
    view_anim = ngu.xyz2nglview(X, top=top)
    view_anim.center()
    if color_by=='monomer_id':
        view_anim.add_representation('ball+stick', selection='.A',
                                                colorScheme='uniform',
                                                colorValue=0xff4242)

        view_anim.add_representation('ball+stick', selection='.B',
                                                  colorScheme='uniform',
                                                  colorValue=0x475FD0)
    else:
        cmap = cm.get_cmap('tab20')
        color_codes = [cmap(i % 20) for i in range(nchains)]
        hex_color_codes = [mcolors.to_hex(color[:3]) for color in color_codes]
        for i in range(nchains):
            view_anim.add_representation('ball+stick', selection=f'.{ascii_uppercase[i % 26]}',
                                            colorScheme='uniform',
                                            colorValue=hex_color_codes[i])
    return view_anim
