if BUILD_CUDA
SUFFIXES = .cu
include cuda.am
endif

AM_CFLAGS = -Wall -O3 -funroll-loops -fstrict-aliasing
AM_CPPFLAGS = -I$(srcdir)
AM_LDFLAGS  = 

if BUILD_CUDA
AM_CFLAGS += $(CUDA_CFLAGS)
AM_CPPFLAGS += $(CUDA_CFLAGS)

# default CUDA nvcc flags
#   Note: cc 13 for Tesla
#   Note: cc 20 for Fermi
#   Note: cc 30 for Kepler K10
#   Note: cc 35 for Kepler K20
NVCCFLAGS +=  -D__USE_GPU__ -use_fast_math 
NVCCFLAGS += -gencode arch=compute_35,code=sm_35 --compiler-options "-O3 -funroll-loops -fstrict-aliasing" -O3
NVCCFLAGS += --ptxas-options -v
NVCCFLAGS += --compiler-options -I$(srcdir) 
NVCCFLAGS += --compiler-options -fno-strict-aliasing
NVCCFLAGS += --compiler-options -Wno-unused-function
NVCCFLAGS += --compiler-options -Wno-unused-parameter
NVCCFLAGS += --compiler-options "$(MPI_CFLAGS)"

#NVCC_LIBS = -lcudart -m64
endif


bin_PROGRAMS = pg-puremd
pg_puremd_SOURCES = src/allocate.c src/basic_comm.c src/ffield.c src/grid.c src/list.c \
	src/lookup.c src/io_tools.c src/reset_tools.c src/restart.c src/random.c \
	src/tool_box.c src/traj.c src/vector.c src/analyze.c src/box.c src/system_props.c \
	src/control.c src/comm_tools.c src/geo_tools.c src/linear_solvers.c src/neighbors.c \
	src/qEq.c src/bond_orders.c src/multi_body.c src/bonds.c src/valence_angles.c \
	src/hydrogen_bonds.c src/torsion_angles.c src/nonbonded.c src/forces.c \
	src/integrate.c src/init_md.c src/parallelreax.c

if BUILD_CUDA
pg_puremd_SOURCES += src/cuda_utils.cu src/dev_alloc.cu src/cuda_environment.cu \
      src/dev_system_props.cu src/reduction.cu src/center_mass.cu \
      src/cuda_copy.cu src/cuda_reset_tools.cu src/dev_list.cu \
      src/cuda_neighbors.cu src/cuda_bond_orders.cu src/cuda_bonds.cu \
      src/cuda_multi_body.cu src/cuda_valence_angles.cu \
      src/cuda_torsion_angles.cu src/cuda_hydrogen_bonds.cu src/cuda_forces.cu \
      src/cuda_qEq.cu src/cuda_linear_solvers.cu src/matvec.cu src/dual_matvec.cu \
      src/cuda_nonbonded.cu src/cuda_integrate.cu src/cuda_post_evolve.cu \
      src/cuda_init_md.cu src/validation.cu src/cuda_lookup.cu

# dummy source to cause C linking
nodist_EXTRA_pg_puremd_SOURCES = src/dummy.c
endif


pg_puremd_CFLAGS = $(AM_CFLAGS) $(MPI_CFLAGS)
pg_puremd_LDFLAGS = $(AM_LDFLAGS) $(MPI_LDFLAGS)

if BUILD_CUDA
	pg_puremd_LDFLAGS += $(CUDA_LIBS)
endif
