# Analysis of polychrom simulations

This folder contains scripts to interface with the open2c/polychrom simulation software package. To use these scripts, first create a clean conda environment and install openMM and other dependencies using the install.sh script.

The `contrib` folder contains custom Brownian dynamics integrators for polymers driven by active forces that vary in magnitude and can potentially be correlated. The `examples` folder contains scripts for running polychrom simulations using these custom integrators. The `post_processing` folder contains scripts for analyzing the output of polychrom simulations.

A. Goychuk, D. Kannan, A. K. Chakraborty, and M. Kardar. Polymer folding through active processes recreates features of genome organization. *bioRxiv* (2022) https://doi.org/10.1101/2022.12.24.521789


