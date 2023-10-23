"""
Initializing chromosome territories
===================================

Provides a tiny wrapper around polychrom.starting_conformations to initialize
multiple chains in ``territories". Each chain is initialized as a constrained
random walk within a sphere whose radius is chosen from the volume fraction
of monomers within the confinement. These spheres are then arranged on a lattice.

To simulate territories, choose a dense sphere packing and combine with
polychrom.forces.spherical_confinement() such that all chains are within a larger
spherical confinement representing the nucleus. Chains will naturally relax
and will not remain in their initialized territories unless they are sufficiently
long (>10K monomers) along with topological constraints (chain passing energy > 5kT).

Alternatively, the lattice spacing can be chosen to be large enough so that the
chains do not interact. When combined with contrib.forces.spherical_well_array(),
this starting conformation can be used to simulate many copies of a single chain
in the same simulation for ensemble averages.

Deepti Kannan, 2023
"""
import numpy as np
import pandas as pd
from polychrom import starting_conformations


def hcp(n):
    dim = 3
    k, j, i = [v.flatten() for v in np.meshgrid(*([range(n)] * dim), indexing="ij")]
    df = pd.DataFrame(
        {
            "x": 2 * i + (j + k) % 2,
            "y": np.sqrt(3) * (j + 1 / 3 * (k % 2)),
            "z": 2 * np.sqrt(6) / 3 * k,
        }
    )
    return df


def square_lattice(n):
    dim = 3
    k, j, i = [v.flatten() for v in np.meshgrid(*([range(n)] * dim), indexing="ij")]
    df = pd.DataFrame(
        {
            "x": i,
            "y": j,
            "z": k,
        }
    )
    return df


def initialize_territories(volume_fraction, mapN, nchains, lattice="hcp", rs=None):
    """Initialize each chain as a constrained random walk within a sphere.
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
    r_chain = ((mapN * (0.5) ** 3) / volume_fraction) ** (1 / 3)
    r_confinement = ((nchains * mapN * (0.5) ** 3) / volume_fraction) ** (1 / 3)
    print(r_chain)
    print(r_confinement)
    # first calculate centroid positions of chains
    n_lattice_points = [i**3 for i in range(10)]
    lattice_size = np.searchsorted(n_lattice_points, nchains)
    print(f"Lattice size = {lattice_size}")
    if lattice == "hcp":
        df = hcp(lattice_size)
    if lattice == "square":
        df = square_lattice(lattice_size)
    else:
        raise ValueError("only hcp and square lattices implemented so far")
    df["radial_distance"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    df.sort_values("radial_distance", inplace=True)
    positions = df.to_numpy()[:, :3][:nchains]
    if rs is None:
        # assume dense packing of spheres (rs is maximum allowed)
        # in units of sphere radii
        max_diameter = pdist(positions).max()
        # mini sphere size
        rs = np.floor(2 * r_confinement / max_diameter)
        positions *= rs
    else:
        # rs is the desired cell size of each square in lattice
        positions *= rs
        # now set radius of sphere to be that of individual chain
        rs = r_chain
    starting_conf = []
    for i in range(nchains):
        centroid = positions[i]

        def confine_chrom(pos):
            x, y, z = pos
            # reject position if it's more than 5% outside of the spherical radius
            return (
                np.sqrt(
                    (x - centroid[0]) ** 2
                    + (y - centroid[1]) ** 2
                    + (z - centroid[2]) ** 2
                )
            ) <= rs

        chrom_pos = starting_conformations.create_constrained_random_walk(
            mapN, confine_chrom, starting_point=(centroid[0], centroid[1], centroid[2])
        )
        starting_conf.append(chrom_pos)
    starting_conf = np.array(starting_conf).reshape((nchains * mapN, 3))
    return starting_conf
