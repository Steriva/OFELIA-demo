# DEMO on OFELIA

This repository contains the code for the DEMO on OFELIA, presented at the OSSFE Online Conference (18th March 2025).

**Authors**: Lorenzo Loi & Stefano Riva, Carolina Introini, Antonio Cammi

[![Reference Paper](https://img.shields.io/badge/Reference%20Paper-Loi%20et%20al.%20(2025)-blue)](https://doi.org/10.1016/j.nucengdes.2024.113480) [![Reference Repo](https://img.shields.io/badge/Reference%20Github-OFELIA-red)](https://github.com/ERMETE-Lab/MP-OFELIA)

Contacts: lorenzo.loi@polimi.it, stefano.riva@polimi.it

## Install Requirements
OFELIA is a framework aiming at coupling OpenMC for neutroncs simulation and FEniCSx for thermal-hydraulics simulation, to perform enhanced neutronics calculations in nuclear reactors.

The current version works with [OpenMC (v 0.13.2)](https://openmc.org/) and [FEniCSx (v. 0.6.0)](https://fenicsproject.org/).

**Be sure to have Anaconda or Miniconda available on your machine.** To ensure a quicker and easier installation, it is suggested to change the *conda-solver* to `libmamba`:
```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

To avoid any incompatibility issue (we are aware it is not ideal, we are working to provide a better way), we suggest to follow these steps (check on Ubuntu 22.04)
```bash
conda create --name ofelia
conda activate ofelia
conda install python=3.10.12 numpy=1.23.5 ipykernel
python -m pip install gmsh gmsh-api
```
Now, we can install OpenMC
```bash
conda install mamba
mamba search openmc
mamba install openmc=0.13.2
```
Then, install dolfinx (version 0.6.0), downgrade setuptools to 62.0.0, and install the remaining packages
```bash
conda install fenics-dolfinx=0.6.0 petsc mpich pyvista tqdm
python -m pip install setuptools==62.0.0
conda install numpy=1.23.5 # needed?
# python -m pip install gmsh gmsh-api pyXSteam # needed?
python -m pip install pyXSteam
```

Set up .bashrc for XS???
