"""
This is a sample simulation that does not represent any particular biological system. It is just a showcase 
of how create a Simulation object, add forces, and initialize the reporter. 

In this simulation, a simple polymer chain of 10,000 monomers is 
"""
import time
import numpy as np
import os, sys
import polychrom
from polychrom import simulation, starting_conformations, forces, forcekits
import openmm
from polychrom.hdf5_format import HDF5Reporter
from simtk import unit
from pathlib import Path

sys.path.append(os.getcwd())

from contrib.integrators import ActiveBrownianIntegrator

total_runs = 20
runs_per_gpu = 10

def harmonic_well(sim_object, r, particles=None, center=[0, 0, 0], depth=1, name="harmonic_well"):
    """
    A flat-bottomed harmonic potential well, suited for example to confine a chromosome to its territory.

    Parameters
    ----------

    particles : list of int or np.array
        indices of particles that are attracted
    r : float
        Radius of the nucleus
    center : vector, optional
        center position of the sphere. This parameter is useful when confining
        chromosomes to their territory.
    depth : float, optional
        Depth of attractive potential in kT
        NOTE: switched sign from openmm-polymer, because it was confusing. Now
        this parameter is really the depth of the well, i.e. positive =
        attractive, negative = repulsive
    """

    force = openmm.CustomExternalForce(
        "step(d) * step(-d) * SPHWELLdepth * d^2;"
        "d = (sqrt((x-SPHWELLx)^2 + (y-SPHWELLy)^2 + (z-SPHWELLz)^2) - SPHWELLradius)"
    )

    force.name = name

    force.addGlobalParameter("SPHWELLradius", r * sim_object.conlen)
    force.addGlobalParameter("SPHWELLdepth", depth * sim_object.kT)
    force.addGlobalParameter("SPHWELLx", center[0] * sim_object.conlen)
    force.addGlobalParameter("SPHWELLy", center[1] * sim_object.conlen)
    force.addGlobalParameter("SPHWELLz", center[2] * sim_object.conlen)
    
    particles = range(sim_object.N) if particles is None else particles
    # adding all the particles on which force acts
    for i in particles:
        # NOTE: the explicit type cast seems to be necessary if we have an np.array...
        force.addParticle(int(i), [])

    return force


def run_sim(gpuid, run_number, activity_ratio, width=None, depth=5.0, 
        timestep=170, ntimesteps=10000, blocksize=2000):
    """ Run a single simulation on GPU i."""
    ids = np.load('/net/levsha/share/deepti/data/ABidentities_chr21_Su2020_2perlocus.npy')
    N=len(ids)
    #0 is cold, 1 is hot
    D = np.ones((N, 3))
    Ddiff = (activity_ratio - 1) / (activity_ratio + 1)
    D[ids==0, :] = 1.0 - Ddiff
    D[ids==1, :] = 1.0 + Ddiff
    volume_fraction = 0.2
    r = ((N * (0.5)**3) / volume_fraction)**(1/3)
    print(f"Radius of confinement: {r}")
    timestep = timestep 
    collision_rate = 2.0
    friction = collision_rate * (1.0/unit.picosecond)
    conlen = 1.0 * unit.nanometer
    mass = 100 * unit.amu
    temperature = 300
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kT = kB * temperature * unit.kelvin
    particleD = unit.Quantity(D, kT/(friction * mass))
    integrator = ActiveBrownianIntegrator(timestep, collision_rate, particleD)
    gpuid = f"{gpuid}"
    if width:
        traj = f"/net/levsha/share/deepti/simulations/spherical_well_test/spherical_well_depth{depth:.0f}_width{width:.0f}"
    else:
        traj = f"/net/levsha/share/deepti/simulations/spherical_well_test/harmonic_well_depth{depth:.0f}"
    Path(traj).mkdir(parents=True, exist_ok=True)
    reporter = HDF5Reporter(folder=traj, max_data_length=100, overwrite=True)
    sim = simulation.Simulation(
        platform="CUDA", 
        integrator=(integrator, "brownian"),
        timestep=timestep,
        temperature=temperature,
        GPU=gpuid,
        collision_rate=collision_rate,
        N=N,
        save_decimals=2,
        PBCbox=False,
        reporters=[reporter],
    )

    polymer = starting_conformations.grow_cubic(N, int(1.5*np.ceil(r)))
    sim.set_data(polymer, center=True)  # loads a polymer, puts a center of mass at zero
    sim.set_velocities(v=np.zeros((N,3)))
    particles = range(N)
    #want wall to start AFTER desired radius
    #radius of spherical well - width = desired radius for given particle density (point where wall starts)
    #want wall to start AFTER desired radius
    if width:
        sim.add_force(forces.spherical_well(sim, particles, r=r+width, width=width, depth=depth))
    else:
        sim.add_force(harmonic_well(sim, r=r, depth=depth))
    sim.add_force(
        forcekits.polymer_chains(
            sim,
            chains=[(0, None, False)],
            # By default the library assumes you have one polymer chain
            # If you want to make it a ring, or more than one chain, use self.setChains
            # self.setChains([(0,50,True),(50,None,False)]) will set a 50-monomer ring and a chain from monomer 50 to the end
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                "bondLength": 1.0,
                "bondWiggleDistance": 0.1,  # Bond distance will fluctuate +- 0.05 on average
            },
            angle_force_func=None,
            angle_force_kwargs={},
            nonbonded_force_func=forces.polynomial_repulsive,
            nonbonded_force_kwargs={
                "trunc": 3.0,  # this will let chains cross sometimes
                #'trunc':10.0, # this will resolve chain crossings and will not let chain cross anymore
            },
            except_bonds=True,
        )
    )
    tic = time.perf_counter()
    for _ in range(ntimesteps):  # Do 10 blocks
        sim.do_block(blocksize)  # Of 100 timesteps each. Data is saved automatically. 
    toc = time.perf_counter()
    print(f'Ran simulation in {(toc - tic):0.4f}s')
    sim.print_stats()  # In the end, print very simple statistics
    reporter.dump_data()  # always need to run in the end to dump the block cache to the disk


if __name__ == '__main__':
    #run 8 simulations, one on each gpu, for the same parameters
    gpuid = int(sys.argv[1])
    #run_number = int(sys.argv[2])
    #run_sim(gpuid, run_number)
    #for act_ratio in [10, 19, 25 + 2/3, 39]:
    #for i in range(gpuid*runs_per_gpu, (gpuid + 1)*runs_per_gpu):
    run_sim(gpuid, 1, 7, width=4.0)
    run_sim(gpuid, 1, 7, width=10.0)
