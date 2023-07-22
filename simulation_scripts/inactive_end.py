"""
Script to run simulations of  mixtures of active and inactive homopolymers in confinement.

Deepti Kannan, 2023
"""
import time
import numpy as np
import os, sys
sys.path.append(os.getcwd())
import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from contrib.integrators import ActiveBrownianIntegrator
from polychrom.hdf5_format import HDF5Reporter
import openmm
from simtk import unit
from pathlib import Path


basepath = Path("/net/levsha/share/deepti/simulations")

def initialize_inactive_center(density=0.477, mapN=1000, ncopies=17):
    """ Initialize ncopies random walks within a spherical confinement. The first chain
    is initialized within an inner sphere centered at the origin, while all remaining chains
    perform a random walk between the inner sphere and outer sphere.
    
    Parameters
    ----------
    density : float
        monomer density within confinement
    mapN : int
        number of monomers per subchain
    ncopies : int
        number of subchains
    
    """
    
    r_inactive = (3 * mapN / (4 * 3.141592 * density)) ** (1/3)
    r_confinement = (3 * mapN * ncopies / (4 * 3.141592 * density)) ** (1/3)
    print(r_inactive)
    print(r_confinement)
    def confine_inactive(pos):
        x, y, z = pos
        #reject position if it's more than 5% outside of the spherical radius
        return ((np.sqrt(x**2 + y**2 + z**2) - r_inactive) <= 0.05*r_inactive)
    
    def confine_active(pos):
        x, y, z = pos
        outside_inner_sphere = (0.95*r_inactive <= np.sqrt(x**2 + y**2 + z**2))
        inside_outer_sphere = ((np.sqrt(x**2 + y**2 + z**2) - r_confinement) <= 0.05*r_confinement)
        return (inside_outer_sphere and outside_inner_sphere)
    
    inactive_pos = starting_conformations.create_constrained_random_walk(mapN, confine_inactive, starting_point=(0., 0., 0.))
    #choose a random starting point within the two spheres.
    theta = np.random.uniform(0.0, 1.0)
    theta = 2.0 * np.pi * theta
    u = np.random.uniform(0.0, 1.0)
    u = 2.0 * u - 1.0
    r = np.random.uniform(0.0, 1.0)
    r = (r_inactive + (r_confinement - r_inactive)*r**(1/3))
    x = r * np.sqrt(1.0 - u*u) * np.cos(theta)
    y = r * np.sqrt(1.0 - u*u) * np.sin(theta)
    z = r * u
    active_pos = starting_conformations.create_constrained_random_walk((ncopies - 1)*mapN, confine_active, 
                                                starting_point=(x, y, z))
    return np.concatenate((inactive_pos, active_pos))

def run_sim(gpuid, run_number, N, nhot, activity_ratio, ncold=1, E0=None, 
            inactive_loc="center", timestep=170, nblocks=20000, blocksize=2000):
    """Run a single simulation on a GPU of a collection of `nhot` active homopolymers and `ncold` inactive
    homopolymers of length N onomers in a spherical confinement.

    Parameters
    ----------
    gpuid : int
        which GPU to use. If on Mirny Lab machine, should be 0, 1, 2, or 3.
    run_number : int
        replicate number for this parameter set
    N : int
        number of monomers in each subchain
    nhot : int
        number of "active" subchains in system
    ncold : int
        number of "inactive" subchains in system
    activity_ratio : float
        ratio of activities between the active and inactive subchains
    timestep : int
        timestep to feed the Brownian integrator (in femtoseconds)
    nblocks : int
        number of blocks to run the simulation for. For a chain of 1000 monomers, need ~100000 blocks of
        100 timesteps to equilibrate.
    blocksize : int
        number of time steps in a block

    """
    ncopies = nhot + ncold
    Dhot = np.ones((N, 3))
    Dcold = np.ones((N, 3))
    if activity_ratio != 1:
        Ddiff = (activity_ratio - 1) / (activity_ratio + 1)
        Dcold[:, :] = 1.0 - Ddiff
        Dhot[:, :] = 1.0 + Ddiff
    #vertically stack ncopies of this array
    D = np.concatenate((np.tile(Dcold, (ncold, 1)), np.tile(Dhot, (nhot, 1))), axis=0) #shape (N*ncopies, 3)
    assert(D.shape[0] == (nhot + ncold)*N and D.shape[1] == 3)
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
    if E0:
        traj = basepath/f"active{nhot}_inactive{ncold}_act{activity_ratio}_E0{E0}_center/run{run_number}"
    else:
        traj = basepath/f"active{nhot}_inactive{ncold}_act{activity_ratio}_center/run{run_number}"
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
    
    if inactive_loc == "center":
        polymer = initialize_inactive_center(density, N, ncopies)
    else:
        polymer = starting_conformations.grow_cubic(N*ncopies, 2 * int(np.ceil(r)))
    sim.set_data(polymer, center=True)  # loads a polymer, puts a center of mass at zero
    sim.set_velocities(v=np.zeros((N*ncopies, 3)))  # initializes velocities of all monomers to zero (no inertia)
    if E0:
        sticky_inds = np.arange(0, N*ncold, dtype="int")
    else:
        sticky_inds = []
        E0 = 0.0
    f_sticky = forces.selective_SSW(sim, 
                                       sticky_inds, 
                                       extraHardParticlesIdxs=[], #don't make any particles extra hard
                                       repulsionEnergy=3.0, #base repulsion energy for all particles (same as polynomial_repulsive)
                                       attractionEnergy=0.05, #base attraction energy for all particles
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
    #16 active chromosomes, 1 inactive chromosomes, each with 1000 monomers
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        run_sim(gpuid, i, 1000, 16, 1.0, E0=0.5)

