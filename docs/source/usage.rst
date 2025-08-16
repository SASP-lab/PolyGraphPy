Usage
=======

After installing `polygraphpy` or cloning the repository and setting the path to the DFTB+ executable, you can run simulations, train models, and generate molecules using the following commands.

### 1. Run DFTB+ Simulations

* **Example data for monomers:**

```bash
python -m polygraphpy --run-dftb --use-example-data --dftbplus-path /home/jgduarte/psi4conda/bin
```

This runs the DFTB+ simulation using the included example data.

* **Copolymer simulations:**

```bash
python -m polygraphpy --run-dftb --input-csv polygraphpy/data/original_dataset.csv --dftbplus-path /home/jgduarte/psi4conda/bin/ --is-polymer --polymer-chain-size 1 --polymer-type copolymer
```

Runs DFTB+ simulations for copolymers with chain size = 1 using the specified dataset.

---

### 2. Train GNN Models

* **Homopolymers (and monomers):**

```bash
python -m polygraphpy --train-gnn-prediction --epochs 250 --is-polymer --polymer-type homopolymer
```

* **Copolymer GNN training:**

```bash
python -m polygraphpy --train-gnn-prediction --epochs 250 --is-polymer --polymer-type copolymer
```

---

### 3. Run Generative Models

* **GPT-based generative model:**

```bash
python -m polygraphpy --run-generative --generative-model gpt --monomers-number-per-target 5
```

Generates new molecules using the GPT model.
For the acrylate dataset, the pretrained GPT model can be stored in `polygraphpy/data/generative_model/` as `gpt_selfies.pt`.

* **Genetic algorithm generative model:**

```bash
python -m polygraphpy --run-generative --generative-model ga --target-polarizability 0.2345
```

Generates molecules for a specific target polarizability using the GA.

If no target polarizability is provided, the code will generate a `linspace(0,1,200)` for the GPT model and `linspace(0,1,100)` for the GA model as default targets.