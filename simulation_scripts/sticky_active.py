"""
Script to run hybrid simulations with active forces and sticky B-B attractions 

A/B identities inferred from q-arm of chr 2 in chromatin tracing data of Su et al. (2020).

Deepti Kannan, 2023
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
from pathlib import Path

import openmm
import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from polychrom.hdf5_format import HDF5Reporter
from scipy.spatial.distance import pdist
from simtk import unit

from contrib.forces import spherical_well_array
from contrib.forces import spherical_confinement_array
from contrib.initialize_chains import initialize_territories
from contrib.integrators import ActiveBrownianIntegrator

def run_sticky_sim(
    gpuid,
    N,
    ncopies,
    E0,
    activity_ratio,
    volume_fraction=0.2,
    width=10.0,
    depth=5.0,  # spherical well array parameters
    confine="many",
    timestep=170,
    nblocks=20000,
    blocksize=2000,
    time_stepping_fn=None,
):
    """Run a single simulation on a GPU of a hetero-polymer with A monomers and B monomers. A monomers
    have a larger diffusion coefficient than B monomers, with an activity ratio of D_A / D_B.

    Parameters
    ----------
    gpuid : int
        which GPU to use. If on Mirny Lab machine, should be 0, 1, 2, or 3.
    N : int
        number of monomers in each subchain
    ncopies : int
        number of subchains in system
    E0 : float
        selective B-B attractive energy
    activity_ratio : float
        ratio of D_A to D_B
    volume_fraction : float
        volume fraction of monomers  within the confinement ((N * ncopies * volume of monomer) / volume)
    confine : str
        if "single", put all chains in a single spherical confinement with provided volume fraction.
        if "many", put each chain in its own spherical well where chains are arranged on a lattice.
        lattice spacing is 5*r, where r the radius of each mini sphere determined based on density.
    width : float
        Width of spherical well. Defaults to 10.0. See polychrom.forces module.
    depth : float
        Depth of spherical well. Defaults to 5.0. See polychrom.forces module.
    timestep : int
        timestep to feed the Brownian integrator (in femtoseconds)
    nblocks : int
        number of blocks to run the simulation for. For a chain of 1000 monomers, need ~100000 blocks of
        100 timesteps to equilibrate.
    blocksize : int
        number of time steps in a block
    time_stepping_fn : function(polychrom.Simulation)
        Function that calls sim.do_block() at perscribed time intervals. Defaults to None.

    """
    ran_sim = False
    particle_inds = np.arange(0, N * ncopies, dtype="int")
    sticky_inds = particle_inds[np.tile(flipped_ids, ncopies)]
    D = np.ones((N, 3))
    if activity_ratio != 1:
        Ddiff = (activity_ratio - 1) / (activity_ratio + 1)
        D[ids == 0, :] = 1.0 - Ddiff
        D[ids == 1, :] = 1.0 + Ddiff
    # vertically stack ncopies of this array
    D = np.tile(D, (ncopies, 1))  # shape (N*ncopies, 3)
    # monomer density in confinement in units of monomers/volume (25%)
    r_chain = ((N * (0.5) ** 3) / volume_fraction) ** (1 / 3)
    r = ((N * ncopies * (0.5) ** 3) / volume_fraction) ** (1 / 3)
    print(f"Radius of confinement: {r}") # 25.3
    print(f"Radius of confined chain: {r_chain}") # 9.3
    # the monomer diffusion coefficient should be in units of kT / friction, where friction = mass*collision_rate
    collision_rate = 2.0
    mass = 100 * unit.amu
    friction = collision_rate * (1.0 / unit.picosecond) * mass
    temperature = 300
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kT = kB * temperature * unit.kelvin
    particleD = unit.Quantity(D, kT / friction)
    integrator = ActiveBrownianIntegrator(timestep, collision_rate, particleD)
    gpuid = f"{gpuid}"
    traj = (
        basepath
        / f"stickyBB_{E0}_act{activity_ratio}/runs{nblocks}_{blocksize}_{ncopies}copies"
    )
    try:
        Path(traj).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"E0={E0}, activity ratio={act_ratio}")
        print(f"{traj}")
        return ran_sim

    reporter = HDF5Reporter(folder=traj, max_data_length=100, overwrite=True)
    sim = simulation.Simulation(
        platform="CUDA",
        integrator=(integrator, "brownian"),
        timestep=timestep,
        temperature=temperature,
        GPU=gpuid,
        collision_rate=collision_rate,
        N=N * ncopies,
        save_decimals=2,
        PBCbox=False,
        reporters=[reporter],
    )
    # set lattice size to be 5 times the radius of a confined chain so that the chains
    # stay far apart from each other and don't interaction
    if confine == "many":
        polymer = initialize_territories(
            volume_fraction, N, ncopies, lattice="square", rs=5 * r_chain
        )
    elif confine == "single":
        polymer = starting_conformations.grow_cubic(N * ncopies, 2 * int(np.ceil(r)))
    #sim.set_data(polymer, center=True)  # loads a polymer, puts a center of mass at zero
    sim.set_data(polymer, center=False, random_offset = 0.0) # MODIFIED HERE, set to False
    polymer2 = sim.get_data()
    L = 5*r_chain
    polymer3 = polymer2.copy()
    for i in range(3):
        polymer3[:, i] -= L*np.floor(polymer3[:,i]/L)
    compare_com(polymer3, polymer2, ncopies)
    
    #assert(np.all(polymer == polymer2))
    sim.set_velocities(
        v=np.zeros((N * ncopies, 3))
    )  # initializes velocities of all monomers to zero (no inertia)
    f_sticky = forces.selective_SSW(
        sim,
        sticky_inds,
        extraHardParticlesIdxs=[],  # don't make any particles extra hard
        repulsionEnergy=3.0,  # base repulsion energy for all particles (same as polynomial_repulsive)
        attractionEnergy=0.0,  # base attraction energy for all particles
        selectiveAttractionEnergy=E0,
    )
    sim.add_force(f_sticky)
    if confine == "single":
        sim.add_force(forces.spherical_confinement(sim, r=r, k=5.0))
    elif confine == "many":
        sim.add_force(
            spherical_confinement_array(
                sim, cell_size=5 * r_chain, r=r_chain
            )
        )

    sim.add_force(
        forcekits.polymer_chains(
            sim,
            chains=[(i * N, i * N + N, False) for i in range(0, ncopies)],
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                "bondLength": 1.0,
                "bondWiggleDistance": 0.1,  # Bond distance will fluctuate +- 0.05 on average
            },
            angle_force_func=None,
            angle_force_kwargs={},
            nonbonded_force_func=None,
            nonbonded_force_kwargs={},
            except_bonds=True,
        )
    )
    tic = time.perf_counter()
    if time_stepping_fn:
        print("timestep_FN")
        time_stepping_fn(sim)
    else:
        for _ in range(nblocks):  # Do 10 blocks
            sim.do_block(
                blocksize
            )  # Of 100 timesteps each. Data is saved automatically.
    toc = time.perf_counter()
    print(f"Ran simulation in {(toc - tic):0.4f}s")
    sim.print_stats()  # In the end, print very simple statistics
    reporter.dump_data()  # always need to run in the end to dump the block cache to the disk
    ran_sim = True
    return ran_sim

def compare_com(array1, array2, ncopies):
    N1, _ = np.shape(array1)
    N2, _ = np.shape(array2)
    assert(N1 == N2)
    mapN = int(N1 / ncopies)
    for i in range(ncopies):
        pos_test1 = np.mean(array1[i*mapN : (i+1)*mapN], axis=0)
        pos_test2 = np.mean(array2[i*mapN : (i+1)*mapN], axis=0)
        print(f"first array COM: {pos_test1}")
        print(f"second array COM: {pos_test2}")

def short_time_dynamics(
    sim, stop1=2000 * 2000, block1=50, stop2=10000 * 2000, block2=5000
):
    """Step until t=stop1 time steps with block size `block1`, and then step
    until `stop2` time steps  with block size `block2`."""
    nblocks = int(stop1 // block1)
    for _ in range(nblocks):
        sim.do_block(block1)
    nblocks = int((stop2 - stop1) // block2)
    for _ in range(nblocks):
        sim.do_block(block2)


def log_time_stepping(sim, ntimepoints=100, mint=50, maxt=10000 * 2000):
    """Save data at time points that are log-spaced between t=mint and t=maxt."""
    timepoints = np.rint(np.logspace(np.log10(mint), np.log10(maxt), ntimepoints))
    blocks = np.concatenate(([timepoints[0]], np.diff(timepoints))).astype(int)
    for block in blocks:
        if block >= 1:
            sim.do_block(block)
            
def clustered_log_time_stepping(sim, ntimepoints=100, mint=100, maxt=10000*2000):
    """Save data at time points that are log-spaced between t-mint and t=maxt, and
    further save data at ten linear points in the vicinity of these log-spaced points."""
    timepoints = np.rint(np.logspace(np.log10(mint), np.log10(maxt), ntimepoints))
    timepoints = list(timepoints.astype(int))
    prev_time = mint-15
    new_timepoints = []
    for i in range(len(timepoints)):
        time_diff = timepoints[i] - prev_time
        rounded_step = round(time_diff/12)
        if (rounded_step > 1000):
            rounded_step = 1000
        ith_window = [timepoints[i] - 5*rounded_step, timepoints[i] - 4*rounded_step, timepoints[i] - 3*rounded_step, timepoints[i] - 2*rounded_step, timepoints[i] - rounded_step, timepoints[i], timepoints[i] + rounded_step, timepoints[i] + 2*rounded_step,
                      timepoints[i] + 3*rounded_step, timepoints[i] + 4*rounded_step, timepoints[i] + 5*rounded_step]
        new_timepoints.extend(ith_window)
        prev_time = timepoints[i]
    blocks = np.concatenate(([new_timepoints[0]], np.diff(new_timepoints))).astype(int)
    for block in blocks:
        if block >= 1:
            sim.do_block(block)

if __name__ == "__main__":
    my_task_id = int(sys.argv[1]) - 1
    num_tasks = int(sys.argv[2])
    # 1 is A, 0 is B
    ABids = np.loadtxt(sys.argv[3], dtype=str)
    ids = (ABids == "A").astype(int)
    flipped_ids = (1 - ids).astype(bool)
    N = len(ids)
    print(f"Number of monomers: {N}")
    basepath = Path(sys.argv[4])
    
    # for parameter sweeps
    param_set = []
    act_ratio = [1, 2, 3, 4, 5]
    e0 = [0, 0.075, 0.15, 0.225, 0.3]
    for a in act_ratio:
        for e in e0:
            param_set.append((a, e))
    
    param_set = [(7.6, 0)] #MODIFIED HERE
    # for contour parameter setting
    select_act = [1, 2, 3, 4, 5]
    select_e0 = [0.23, 0.15, 0.10, 0.05, 0]
    contour_param_set = list(zip(select_act, select_e0)) # modify for each chromo
    
    acts_per_task = param_set[my_task_id : len(param_set) : num_tasks] # change contour param set if necessary
    
    tic = time.time()
    sims_ran = 0
    for (act_ratio, E0) in acts_per_task:
        ran_sim = run_sticky_sim(0, N, 20, E0, act_ratio, nblocks=2, blocksize=0, time_stepping_fn=clustered_log_time_stepping) # 200 chains, 1100 nblocks, MODIFIED HERE
        print((act_ratio, E0))
        if ran_sim:
            sims_ran += 1
    toc = time.time()
    nsecs = toc - tic
    nhours = int(np.floor(nsecs // 3600))
    nmins = int((nsecs % 3600) // 60)
    nsecs = int(nsecs % 60)
    print(f"Ran {sims_ran} simulations in {nhours}h {nmins}m {nsecs}s")