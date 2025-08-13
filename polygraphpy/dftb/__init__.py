"""
polygraphpy.dftb
==================

This package provides tools for setting up and running DFTB+ simulations,
specifically tailored for calculating the static polarizability of organic
molecules and polymers.

The main functionalities include:
- **SMILES to XYZ Conversion**: Generating 3D molecular structures from SMILES strings.
- **DFTB+ Input Generation**: Creating input files for DFTB+ simulations.
- **DFTB+ Simulation Execution**: Running the DFTB+ software to obtain output files.
- **Polarizability Trace Calculation**: Post-processing DFTB+ output to extract
  and compute the polarizability trace.

The pipelines are designed to be automated, allowing users to go from a
CSV file of SMILES strings to a dataset of calculated polarizabilities.
"""