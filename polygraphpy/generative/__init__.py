"""
polygraphpy.generative
======================

This package contains modules for the generative design of novel polymer
monomers. It provides two distinct approaches for molecule generation:

- **Genetic Algorithm (`ga.py`)**: An evolutionary approach that uses a
  pre-trained GNN model for fitness-based selection. It combines molecular
  fragments to create new, optimized structures.

- **GPT-based Model (`gpt.py`)**: A language-model-based approach that
  fine-tunes a GPT-2 model to generate molecular structures (in SELFIES format)
  based on a desired target property, such as polarizability.
"""