This repository contains a simple split-step Fourier method (SSFM) for simulating the coupled nonlinear Schrodinger equation (CNLSE). This implementation is appropriate for both long traces and batched operation, and can be run on either the CPU or GPU.

main.py is a simple script which demonstrates how this repository can be used.
ssfm.py contains the main SSFM channel, including parameter initialisation and simulation.
utils.py contains useful utility functions, such as signal normalisation and pulse shaping.

Running this repository requires pytorch, numpy, and matplotlib installed.