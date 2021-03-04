[![pipeline status](https://gitlab.com/ohearnk/PuReMD/badges/master/pipeline.svg)](https://gitlab.com/ohearnk/PuReMD/commits/master)

[Introduction](https://gitlab.msu.edu/SParTA/PuReMD#introduction) |
[Documentation](https://gitlab.msu.edu/SParTA/PuReMD/doc) |
[Wiki](https://gitlab.msu.edu/SParTA/PuReMD/wikis/home)

# Introduction

This repository contains the development version of the **Pu**ReMD **Re**active
**M**olecular **D**ynamics (PuReMD) project.  PuReMD is an open-source highly
performant range-limited atomic-level molecular dynamics code which implements
the the reactive force field (ReaxFF) method coupled with a global atomic
charge model.  Supported charge models include charge equilibration,
electronegativity equilization, and atom-condensed Kohn-Sham approximated to
second order.

# Build Instructions

## User

To build, the following versions of software are required:

- GNU make
- C compiler with support for the c11 standard or newer and optionally OpenMP v4.0+ (shared-memory code)
- C++ compiler with support for the c++14 standard or newer (CUDA, MPI+CUDA versions)
- MPI v2+ compliant library (MPI, MPI+CUDA versions)
- CUDA v6.0+ (CUDA, MPI+CUDA versions)
- zlib v1.2.x or newer

```bash
# Download release tarball
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

## Developer

To build, the following versions of software are required:

- git
- Autoconf v2.69 or newer
- Automake v1.15 or newer
- libtool v2.2 or newer
- GNU make
- C compiler with support for the c11 standard or newer and optionally OpenMP v4.0+ (shared-memory code)
- C++ compiler with support for the c++14 standard or newer (CUDA, MPI+CUDA versions)
- MPI v2+ compliant library (MPI, MPI+CUDA versions)
- CUDA v6.0+ (CUDA, MPI+CUDA versions)
- zlib v1.2.x or newer

Instructions:
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

# References

Shared-Memory Versions:
- [Serial](https://www.cs.purdue.edu/puremd/docs/80859.pdf)
- [CUDA (single GPU)](http://dx.doi.org/10.1016/j.jcp.2014.04.035)
- [Charge Model Optimizations with OpenMP](https://doi.org/10.1137/18M1224684)

Distributed-Memory Versions:
- [MPI (message passing interface)](https://www.cs.purdue.edu/puremd/docs/Parallel-Reactive-Molecular-Dynamics.pdf)
- [Hybrid MPI+OpenMP optimization](https://doi.org/10.1177/1094342017746221)
- [Charge Model Optimizations with MPI](https://doi.org/10.1145/3330345.3330359)
- [CUDA+MPI (multi-GPU)](https://www.cs.purdue.edu/puremd/docs/pgpuremd.pdf)
