Installation
=======

Before installing and running this project, ensure that [DFTB+](https://www.dftbplus.org/) is installed on your system, as it is required for all quantum mechanical simulations. After installation, find the `dftb+` executable. This path will be required by the package to run simulations.

Install PyTorch and PyTorch Geometric dependencies first:

```bash
pip install torch torch_geometric
```

### Option 1: Install via PyPI

```bash
pip install polygraphpy
```

### Option 2: Clone Repository and Use as Standalone Code

You can also clone the repository to use it without installing via PyPI:

```bash
git clone https://github.com/yourusername/polygraphpy.git
cd polygraphpy
```

This allows running the code directly from the cloned repository.