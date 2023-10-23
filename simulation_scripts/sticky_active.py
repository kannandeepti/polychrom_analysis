"""
Script to run hybrid simulations with active forces and sticky B-B attractions 

A/B identities inferred from q-arm of chr 2 in chromatin tracing data of Su et al. (2020).

Deepti Kannan, 2023
"""
import time
import numpy as np
import pandas as pd
import os, sys
from scipy.spatial.distance import pdist
sys.path.append(os.getcwd())
import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from contrib.integrators import ActiveBrownianIntegrator
from polychrom.hdf5_format import HDF5Reporter
import openmm
from simtk import unit
from pathlib import Path

#insert path to simulation directory where results should be stored
basepath = Path("./chr21_Su2020")
# 1 is A, 0 is B
ABids = np.loadtxt("data/ABidentities_chr21_Su2020_2perlocus.csv", dtype=str)
ids = (ABids == "A").astype(int)
N = len(ids)
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

def initialize_territories(volume_fraction, mapN, nchains, lattice='hcp',
                          rs=None):
    """ Initialize each chain as a constrained random walk within a sphere.
    Spheres are located on a lattice (hcp or square) within simulation volume.

    Parameters
    ----------
    volume_fraction : float
        fraction of simulation volume that monomers occupy.
    mapN : int
        number of monomers per chain
    nchains : int
        number of chains
    lattice : 'hcp' or 'square'
        lattice to put spheres in
    rs : float
        desired cell size of each cell in lattice.
        If None, assumes a dense packing of squares.
    
    Returns
    -------
    starting_conf : np.ndarray[float] (mapN * nchains, 3)
        initial x,y,z positions of all monomers

    """
    r_chain = ((mapN * (0.5)**3) / volume_fraction) ** (1/3)
    r_confinement = ((nchains * mapN * (0.5)**3) / volume_fraction) ** (1/3)
    print(r_chain)
    print(r_confinement)
    #first calculate centroid positions of chains
    n_lattice_points = [i**3 for i in range(10)]
    lattice_size = np.searchsorted(n_lattice_points, nchains)
    print(f'Lattice size = {lattice_size}')
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
        rs = r_chain
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
    
    
def run_sticky_sim(gpuid, N, ncopies, E0, activity_ratio, volume_fraction=0.2,
                   width=10.0, depth=5.0, #spherical well array parameters
                   confine="many", timestep=170, nblocks=20000, blocksize=2000,
                   time_stepping_fn=None):
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
    r_chain = ((N * (0.5)**3) / volume_fraction) ** (1/3)
    r = ((N * ncopies * (0.5)**3) / volume_fraction) ** (1/3)
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
    traj = basepath/f"stickyBB_{E0}_act{activity_ratio}/runs{nblocks}_{blocksize}_{ncopies}copies"
    try:
        Path(traj).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"E0={E0}, activity ratio={act_ratio} already exists")
        return ran_sim

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
    #stay far apart from each other and don't interaction
    if confine == "many":
        polymer = initialize_territories(volume_fraction, N, ncopies, lattice='square', rs=5*r_chain)
    elif confine == "single":
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
    if confine == "single":
        sim.add_force(forces.spherical_confinement(sim, r=r, k=5.0))
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
    if time_stepping_fn:
        time_stepping_fn(sim)
    else:
        for _ in range(nblocks):  # Do 10 blocks
            sim.do_block(blocksize)  # Of 100 timesteps each. Data is saved automatically.
    toc = time.perf_counter()
    print(f"Ran simulation in {(toc - tic):0.4f}s")
    sim.print_stats()  # In the end, print very simple statistics
    reporter.dump_data()  # always need to run in the end to dump the block cache to the disk
    ran_sim = True
    return ran_sim

def short_time_dynamics(sim, stop1=2000*2000, block1=50, stop2=10000*2000, block2=5000):
    """Step until t=stop1 time steps with block size `block1`, and then step
    until `stop2` time steps  with block size `block2`."""
    nblocks = int(stop1 // block1)
    for _ in range(nblocks):
        sim.do_block(block1)
    nblocks = int((stop2 - stop1) // block2)
    for _ in range(nblocks):
        sim.do_block(block2)

def log_time_stepping(sim, ntimepoints=100, mint=50, maxt=10000*2000):
    """ Save data at time points that are log-spaced between t=mint and t=maxt."""
    timepoints = np.rint(np.logspace(np.log10(mint), np.log10(maxt), ntimepoints))
    blocks = np.concatenate(([timepoints[0]], np.diff(timepoints))).astype(int)
    for block in blocks:
        if block >= 1:
            sim.do_block(block)

if __name__ == '__main__':
    gpuid = int(sys.argv[1])
    #range of models with cs1 = 1.0
    param_set_1 = [(1, 0.5), (3, 0.5), (5, 0.5), (7, 0.5),
                   (8, 0.5), (9, 0.5), (10, 0.5)]
    #range of models with cs1 = 0.6
    param_set_2 = [(1, 0.25), (2, 0.15), (3, 0.1), (4, 0.05), (5, 0.0)]
    acts_only = [(2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0), (7, 0.0), (8, 0.0), (9, 0.0), (10, 0.0)]
    E0_only = [(1, 0.15), (1, 0.1), (1, 0.05)]
    tic = time.time()
    sims_ran = 0
    all_params = param_set_1 + param_set_2 + acts_only + E0_only
    test_params = [(1, 0.0), (2, 0.4)]
    #print(all_params)
    #for the sensitive region of parameter space, run simulations with 200 chains
    #for act_ratio in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
    #    for E0 in [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]:
    for (E0, act_ratio) in test_params:
        ran_sim = run_sticky_sim(gpuid, N, 20, E0, act_ratio, nblocks=1000, blocksize=100)
        if ran_sim:
            sims_ran += 1
    toc = time.time()
    nsecs = toc - tic
    nhours = int(np.floor(nsecs // 3600))
    nmins = int((nsecs % 3600) // 60)
    nsecs = int(nsecs % 60)
    print(f"Ran {sims_ran} simulations in {nhours}h {nmins}m {nsecs}s") 
