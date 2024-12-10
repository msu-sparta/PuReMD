<p align="right">
 <img src="https://github.com/msu-sparta/PuReMD/actions/workflows/build_test_serial.yml/badge.svg?branch=main">
</p>
<p align="right">
 <img src="https://github.com/msu-sparta/PuReMD/actions/workflows/build_test_mpi.yml/badge.svg?branch=main">
</p>

[Introduction](https://github.com/MSU-SParTA/PuReMD#introduction) |
[Documentation](https://github.com/MSU-SParTA/PuReMD/doc) |
[Wiki](https://github.com/MSU-SParTA/PuReMD/wikis/home)

# Introduction

This repository contains the development version of the **Pu**ReMD **Re**active
**M**olecular **D**ynamics (PuReMD) project.  PuReMD is an open-source, highly
performant, range-limited atomic-level molecular dynamics code which implements
the reactive force field (ReaxFF) method coupled with a global atomic charge
model for accurate electrostatics.  Supported dynamic charge models include
charge equilibration (QEq), electronegativity equilization (EE), and
atom-condensed Kohn-Sham approximated to second order (ACKS2).

# Build Instructions

## User

To build, the following software dependencies are required:

- GNU make
- C compiler with support for the c11 standard or newer and optionally OpenMP v4.0+ (shared-memory code)
- C++ compiler with support for the c++14 standard or newer (CUDA, MPI+CUDA versions)
- Message Passing Interface (MPI) v2+ compliant implementation (MPI, MPI+X versions)
- CUDA v6.0+ (CUDA, MPI+CUDA versions)
- RocM v5.0+ (HIP, MPI+HIP versions)
- zlib v1.2.x or newer

```bash
# Download latest release tarball
tar -xvf puremd-1.0.tar.gz
cd puremd-1.0
./configure
make
```

By default, the shared memory version with OpenMP support will be built. For other build targets,
run ./configure --help and consult the documentation. An example of building the MPI+CUDA version
is given below.

```bash
./configure --enable-serial=no --enable-mpi-cuda=yes
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
-- Older CUDA versions (prior to v11.0) require downloading NVIDIA CUB manually (provided as a git submodule)
- RocM v5.4+ (HIP, MPI+HIP versions)
- zlib v1.2.x or newer

Instructions:
```bash
git clone https://github.com/msu-sparta/PuReMD.git
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

# Selected Publications

Shared-Memory Versions:
- Serial, [DOI: 10.1137/100808599](https://doi.org/10.1137/100808599)
- CUDA (single GPU), [DOI: 10.1016/j.jcp.2014.04.035](http://dx.doi.org/10.1016/j.jcp.2014.04.035)
- OpenMP with charge model optimizations, [DOI: 10.1137/18M1224684](https://doi.org/10.1137/18M1224684)

Distributed-Memory Versions:
- MPI, [DOI: 10.1016/j.parco.2011.08.005](https://doi.org/10.1016/j.parco.2011.08.005)
- Hybrid MPI+OpenMP, [DOI: 10.1177/1094342017746221](https://doi.org/10.1177/1094342017746221)
- ReaxFF and charge model optimizations (MPI), [DOI: 10.1145/3330345.3330359](https://doi.org/10.1145/3330345.3330359)
- CUDA+MPI, [Purdue e-Pubs](https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2769&context=cstech)

Applications and Integrations:
- ReaxFF force field optimization with gradients using JAX, [DOI: 10.1021/acs.jctc.2c00363](https://doi.org/10.1021/acs.jctc.2c00363)
- Embedded QM/MM calcuations in Amber (Amber/ReaxFF in AmberTools), [DOI: 10.1021/acs.jpclett.2c01279](https://doi.org/10.1021/acs.jpclett.2c01279)
