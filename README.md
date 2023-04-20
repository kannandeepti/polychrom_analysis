# Analysis of polychrom simulations

This folder contains scripts to interface with the open2c/polychrom simulation software package in order to run the simulations in [^1]. To use these scripts, first create a clean conda environment and install openMM and other dependencies using the install.sh script.
```
conda create -n polychrom python=3.9
conda activate polychrom
bash install.sh
```

The `contrib` folder contains custom Brownian dynamics integrators for polymers driven by active 
forces that vary in magnitude and can potentially be correlated. The `examples` folder contains 
the scripts used to run polychrom simulations using these custom integrators. 
The `post_processing` folder contains scripts for analyzing the output of polychrom simulations. 
The `data` folder contains the A/B identities of a 1000-mer chain derived from Hi-C data for the 
35-60Mb region of  chromosome 2 in murine erythroblast cells (Zhang et al. *Nat. Commun.* 2021).

[^1] A. Goychuk, D. Kannan, A. K. Chakraborty, and M. Kardar. Polymer folding through active processes recreates features of genome organization. *bioRxiv* (2022) https://doi.org/10.1101/2022.12.24.521789


