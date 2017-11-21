[Introduction](https://gitlab.msu.edu/SParTA/PuReMD#introduction) |
[Documentation](https://gitlab.msu.edu/SParTA/PuReMD/doc) |
[Wiki](https://gitlab.msu.edu/SParTA/PuReMD/wikis/home)

# Introduction

This repository contains the development version of the
[Purdue Reactive Molecular Dynamics](https://www.cs.purdue.edu/puremd) (PuReMD) project.

# Build Instructions

## Developer

```bash
	git clone https://gitlab.msu.edu/SParTA/PuReMD.git
	cd PuReMD
	git submodule init
	git submodule update
	autoreconf -ivf
	./configure
	make
```

To build tarball releases after configuring a specific build target, run the following:

```bash
	make dist
```

## User

```bash
	tar -xvf puremd-1.0.tar.gz
	cd puremd-1.0
	./configure
	make
```

By default, the shared memory version with OpenMP support will be built. For other build targets,
run ./configure --help and consult the documentation. An example of building the MPI+CUDA version
is given below.

```bash
	./configure --enable-openmp=no --enable-mpi-gpu=yes
```

# References

Shared Memory:
- [Serial](https://www.cs.purdue.edu/puremd/docs/80859.pdf)
- [CUDA (single GPU)](http://dx.doi.org/10.1016/j.jcp.2014.04.035)
- [Charge Method Optimizations with OpenMP](https://doi.org/10.1109/ScalA.2016.006)

Distributed Memory:
- [MPI (message passing interface)](https://www.cs.purdue.edu/puremd/docs/Parallel-Reactive-Molecular-Dynamics.pdf)
- [CUDA+MPI (multi-GPU)](https://www.cs.purdue.edu/puremd/docs/pgpuremd.pdf)
