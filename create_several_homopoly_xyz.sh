#!/bin/bash

for i in {1,2,3,4,5}
do
    python -m polygraphpy  --polymer --input-csv polygraphpy/data/full_dataset.csv --polymer-chain-size $i --prediction-target static_polarizability --dftbplus-path /home/jgduarte/psi4conda/bin
done