=====
pyFEL
=====

A shameless python clone of Genesis 1.3 V4

Description
===========

The project aims to develop a package that is almost 100% compatible to Genesis 1.3 V4.


Several enhancements are planned:

- Opencl backend to speed-up the simulation when an compute accelerator (like GPUs, multicore CPUs) is present.
- Add the possibility to describe input files in TOML and JSON format
- Plugin interface for lattice elements to simplify the creation of new beam transformations.
- Accelerating cavities and multipolar/combined function magnets
- Parallel random generator (PCG64) to allow reproducible simulations at arbitrary number of MPI nodes.
- Tests!!!
