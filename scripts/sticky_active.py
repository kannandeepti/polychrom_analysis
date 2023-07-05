"""
Script to run equilibrium simulations of polymer with B-B attractive interactions

A/B identities inferred from Hi-C data in Zhang, ..., Blobel (2021), from a portion
of chromosome 2.

Deepti Kannan, 2023
"""
import time
import numpy as np
import os, sys
try:
    import polychrom
except:
    sys.path.append("/home/dkannan/git-remotes/polychrom/")
    import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from polychrom.contrib.integrators import ActiveBrownianIntegrator
from polychrom.hdf5_format import HDF5Reporter
import openmm
from simtk import unit
from pathlib import Path

basepath = Path("/net/levsha/share/deepti/simulations/chr2_Su2020")
#0 is B, 1 is A
ids = np.load('/net/levsha/share/deepti/data/ABidentities_chr2_q_Su2020_2perlocus.npy')
N=len(ids)
#1 is B, 0 is A
flipped_ids = (1 - ids).astype(bool)

def run_sticky_sim(gpuid, run_number, N, ncopies, E0, activity_ratio, timestep=170, nblocks=20000, blocksize=1000):
    """Run a single simulation on a GPU of a hetero-polymer with A monomers and B monomers. A monomers
    have a larger diffusion coefficient than B monomers, with an activity ratio of D_A / D_B.

    Parameters
    ----------
    gpuid : int
        which GPU to use. If on Mirny Lab machine, should be 0, 1, 2, or 3.
    run_number : int
        replicate number for this parameter set
    N : int
        number of monomers in each subchain
    ncopies : int
        number of subchains in system
    sticky_ids : array-like
        indices of sticky monomers
    E0 : float
        selective B-B attractive energy
    timestep : int
        timestep to feed the Brownian integrator (in femtoseconds)
    nblocks : int
        number of blocks to run the simulation for. For a chain of 1000 monomers, need ~100000 blocks of
        100 timesteps to equilibrate.
    blocksize : int
        number of time steps in a block

    """
    particle_inds = np.arange(0, N*ncopies, dtype="int")
    sticky_inds = particle_inds[np.tile(flipped_ids, ncopies)]
    D = np.ones((N, 3))
    Ddiff = (activity_ratio - 1) / (activity_ratio + 1)
    D[ids==0, :] = 1.0 - Ddiff
    D[ids==1, :] = 1.0 + Ddiff
    #vertically stack ncopies of this array
    D = np.tile(D, (ncopies, 1)) #shape (N*ncopies, 3)
    # monomer density in confinement in units of monomers/volume (25%)
    density = 0.477
    r = (3 * N * ncopies / (4 * 3.141592 * density)) ** (1 / 3)
    print(f"Radius of confinement: {r}")
    timestep = timestep
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
    traj = basepath/f"stickyBB_{E0}_act{activity_ratio}/runs{nblocks}_{blocksize}_{ncopies}copies"
    Path(traj).mkdir(parents=True, exist_ok=True)
    reporter = HDF5Reporter(folder=traj, max_data_length=100, overwrite=True)
    sim = simulation.Simulation(
        platform="CUDA", 
        integrator=(integrator, "brownian"),
        timestep=timestep,
        temperature=temperature,
        GPU=gpuid,
        collision_rate=collision_rate,
        N=N*ncopies,
        save_decimals=2,
        PBCbox=False,
        reporters=[reporter],
    )

    polymer = starting_conformations.grow_cubic(N*ncopies, 2 * int(np.ceil(r)))
    sim.set_data(polymer, center=True)  # loads a polymer, puts a center of mass at zero
    sim.set_velocities(v=np.zeros((N*ncopies, 3)))  # initializes velocities of all monomers to zero (no inertia)
    f_sticky = forces.selective_SSW(sim, 
                                       sticky_inds, 
                                       extraHardParticlesIdxs=[], #don't make any particles extra hard
                                       repulsionEnergy=3.0, #base repulsion energy for all particles (same as polynomial_repulsive)
                                       attractionEnergy=0.0, #base attraction energy for all particles
                                       selectiveAttractionEnergy=E0)
    sim.add_force(f_sticky)
    sim.add_force(forces.spherical_confinement(sim, density=density, k=5.0))
    sim.add_force(
        forcekits.polymer_chains(
            sim,
            chains=[(i*N, i*N + N, False) for i in range(0, ncopies)],
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
    for _ in range(nblocks):  # Do 10 blocks
        sim.do_block(blocksize)  # Of 100 timesteps each. Data is saved automatically.
    toc = time.perf_counter()
    print(f"Ran simulation in {(toc - tic):0.4f}s")
    sim.print_stats()  # In the end, print very simple statistics
    reporter.dump_data()  # always need to run in the end to dump the block cache to the disk


if __name__ == '__main__':
    gpuid = int(sys.argv[1])
    for E0 in [0.05, 0.06]:
        for act_ratio in [3, 4, 5, 6, 7]:
            run_sticky_sim(gpuid, 0, N, 20, E0, act_ratio)

    for E0 in [0.02, 0.03, 0.04]:
        for act_ratio in [6, 7]:
            run_sticky_sim(gpuid, 0, N, 20, E0, act_ratio)

