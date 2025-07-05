#!/bin/bash

for i in {1,2,3,4,5,6,7,8,9,10}
do
    python -m polygraphpy --use-example-data --polymer --polymer-chain-size $i --prediction-target static_polarizability --dftbplus-path /home/jgduarte/psi4conda/bin
done