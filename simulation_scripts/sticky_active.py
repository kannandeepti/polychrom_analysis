"""
Script to run hybrid simulations with active forces and sticky B-B attractions 

A/B identities inferred from q-arm of chr 2 in chromatin tracing data of Su et al. (2020).

Deepti Kannan, 2023
"""
import time
import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.getcwd())
import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from contrib.integrators import ActiveBrownianIntegrator
from polychrom.hdf5_format import HDF5Reporter
import openmm
from simtk import unit
from pathlib import Path

basepath = Path("/net/levsha/share/deepti/simulations/chr21_Su2020")
#0 is B, 1 is A
ids = np.load('/net/levsha/share/deepti/data/ABidentities_chr21_Su2020_2perlocus.npy')
N=len(ids)
print(f'Number of monomers: {N}')
#1 is B, 0 is A
flipped_ids = (1 - ids).astype(bool)

def hcp(n):
    dim = 3
    k, j, i = [v.flatten()
               for v in np.meshgrid(*([range(n)] * dim), indexing='ij')]
    df = pd.DataFrame({
        'x': 2 * i + (j + k) % 2,
        'y': np.sqrt(3) * (j + 1/3 * (k % 2)),
        'z': 2 * np.sqrt(6) / 3 * k,
    })
    return df

def square_lattice(n):
    dim = 3
    k, j, i = [v.flatten()
               for v in np.meshgrid(*([range(n)] * dim), indexing='ij')]
    df = pd.DataFrame({
        'x': i,
        'y': j,
        'z': k,
    })
    return df

def initialize_territories(density=0.477, mapN=1000, nchains=20, lattice='hcp',
                          rs=None):
    r_chain = (3 * mapN / (4 * 3.141592 * density)) ** (1/3)
    r_confinement = (3 * mapN * nchains / (4 * 3.141592 * density)) ** (1/3)
    print(r_chain)
    print(r_confinement)
    #first calculate centroid positions of chains
    n_lattice_points = [i**3 for i in range(10)]
    lattice_size = np.searchsorted(n_lattice_points, chains)
    if lattice=='hcp':
        df = hcp(lattice_size)
    if lattice=='square':
        df = square_lattice(lattice_size)
    else:
        raise ValueError('only hcp and square lattices implemented so far')
    df['radial_distance'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    df.sort_values('radial_distance', inplace=True)
    positions = df.to_numpy()[:, :3][:nchains]
    if rs is None:
        #assume dense packing of spheres (rs is maximum allowed)
        # in units of sphere radii
        max_diameter = pdist(positions).max()
        #mini sphere size
        rs = np.floor(2*r_confinement / max_diameter)
        positions *= rs
    else:
        #rs is the desired cell size of each square in lattice
        positions *= rs
        #now set radius of sphere to be that of individual chain
    starting_conf = []
    for i in range(nchains):
        centroid = positions[i]
        def confine_chrom(pos):
            x, y, z = pos
            #reject position if it's more than 5% outside of the spherical radius
            return ((np.sqrt((x - centroid[0])**2 + (y-centroid[1])**2 + (z-centroid[2])**2)) <= rs)
        chrom_pos = starting_conformations.create_constrained_random_walk(mapN, confine_chrom, starting_point=(centroid[0], centroid[1], centroid[2]))
        starting_conf.append(chrom_pos)
    starting_conf = np.array(starting_conf).reshape((nchains*mapN, 3))
    return starting_conf

def spherical_well_array(sim_object, r, cell_size, particles=None,
                         width=1, depth=1, name="spherical_well_array"):
    """
    An (array of) spherical potential wells. Uses floor functions to map
    particle positions to the coordinates of the well.

    Parameters
    ----------

    r : float
        Radius of the nucleus
    cell_size : float
        width of cell in lattice of spherical wells
    particles : list of int or np.array
        indices of particles that are attracted
    width : float, optional
        Width of attractive well, nm.
    depth : float, optional
        Depth of attractive potential in kT
        Positive means the walls are repulsive (i.e chain confined within lamina).
        Negative means walls are attractive (i.e. attraction to lamina)
    """

    force = openmm.CustomExternalForce(
        "step(1+d) * step(1-d) * SPHWELLdepth * (1 + cos(3.1415926536*d)) / 2;"
        "d = (sqrt((x1-SPHWELLx)^2 + (y1-SPHWELLy)^2 + (z1-SPHWELLz)^2) - SPHWELLradius) / SPHWELLwidth;"
        "x1 = x - L*floor(x/L);"
        "y1 = y - L*floor(y/L);"
        "z1 = z - L*floor(z/L);"
    )
    force.name = name
    particles = range(sim_object.N) if particles is None else particles
    center = 3 * [cell_size/2]
    
    force.addGlobalParameter("SPHWELLradius", r * sim_object.conlen)
    force.addGlobalParameter("SPHWELLwidth", width * sim_object.conlen)
    force.addGlobalParameter("SPHWELLdepth", depth * sim_object.kT)
    force.addGlobalParameter("L", cell_size * sim_object.conlen)
    force.addGlobalParameter("SPHWELLx", center[0] * sim_object.conlen)
    force.addGlobalParameter("SPHWELLy", center[1] * sim_object.conlen)
    force.addGlobalParameter("SPHWELLz", center[2] * sim_object.conlen)

    # adding all the particles on which force acts
    for i in particles:
        # NOTE: the explicit type cast seems to be necessary if we have an np.array...
        force.addParticle(int(i), [])

    return force
    
    
def run_sticky_sim(gpuid, run_number, N, ncopies, E0, activity_ratio, density=0.477,
                   width=10.0, depth=5.0, #spherical well array parameters
                   confine="single", timestep=170, nblocks=20000, blocksize=2000):
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
    E0 : float
        selective B-B attractive energy
    activity_ratio : float
        ratio of D_A to D_B
    density : float
        monomer density within the confinement (# monomers / volume)
    confine : str
        if "single", put all chains in a single spherical confinement with provided density/
        if "many", put each chain in its own spherical well where chains are arranged on a lattice.
        lattice spacing is 5*r, where r the radius of each mini sphere determined based on density.
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
    if activity_ratio != 1:
        Ddiff = (activity_ratio - 1) / (activity_ratio + 1)
        D[ids==0, :] = 1.0 - Ddiff
        D[ids==1, :] = 1.0 + Ddiff
    #vertically stack ncopies of this array
    D = np.tile(D, (ncopies, 1)) #shape (N*ncopies, 3)
    # monomer density in confinement in units of monomers/volume (25%)
    r_chain = (3 * mapN / (4 * 3.141592 * density)) ** (1/3)
    r = (3 * N * ncopies / (4 * 3.141592 * density)) ** (1 / 3)
    print(f"Radius of confinement: {r}")
    print(f"Radius of confined chain: {r_chain}")
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
    traj = basepath/f"stickyBB_{E0}_act{activity_ratio}_rep5.0_{N}/runs{nblocks}_{blocksize}_{ncopies}copies"
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
    #set lattice size to be 5 times the radius of a confined chain so that the chains
    #stay far apart from each other and don't interact
    polymer = initialize_territories(density=density, lattice='square', rs=5*r_chain)
    #polymer = starting_conformations.grow_cubic(N*ncopies, 2 * int(np.ceil(r)))
    sim.set_data(polymer, center=True)  # loads a polymer, puts a center of mass at zero
    sim.set_velocities(v=np.zeros((N*ncopies, 3)))  # initializes velocities of all monomers to zero (no inertia)
    f_sticky = forces.selective_SSW(sim, 
                                       sticky_inds, 
                                       extraHardParticlesIdxs=[], #don't make any particles extra hard
                                       repulsionEnergy=5.0, #base repulsion energy for all particles (same as polynomial_repulsive)
                                       attractionEnergy=0.0, #base attraction energy for all particles
                                       selectiveAttractionEnergy=E0)
    sim.add_force(f_sticky)
    if confine == "single":
        sim.add_force(forces.spherical_confinement(sim, density=density, k=5.0))
    elif confine == "many":
        sim.add_force(spherical_well_array(sim, cell_size=5*r_chain, r=width+r_chain, width=width, depth=depth))
        
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
    for act_ratio in [1]: 
        for E0 in [0.5]:
            run_sticky_sim(gpuid, 0, N, 20, E0, act_ratio, confine="many", width=10.0, depth=5.0)

