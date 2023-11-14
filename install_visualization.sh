#!/usr/bin/env bash
<<comment
Script to create a clean conda environment solely for the
purposes of visualizing polychrom snapshots with NGLview
and post-processing analysis. 

Why is this separate from the `polychrom` environment?
- due to version incompatibilities with openmm
- TODO: figure out an all encompasing environment for simulations + analysis

comment

conda create -n polyvis python=3.9 matplotlib numpy scipy jupyter ipython pandas h5py joblib seaborn numba cmasher
source activate polyvis
pip install -U git+https://github.com/open2c/polychrom
conda install nglview -c conda-forge
pip install -U git+https://github.com/mirnylab/nglutils
python -m ipykernel install --sys-prefix
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter nbextension enable --py --sys-prefix nglview
python -m ipykernel install --user --name=polyvis
pip install -U git+https://github.com/open2c/cooltools
