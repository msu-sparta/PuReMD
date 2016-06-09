SUFFIXES = .cu
include cuda.am

AM_CFLAGS = -Wall -O3 -funroll-loops -fstrict-aliasing $(CUDA_CFLAGS)
AM_CPPFLAGS = -I$(srcdir) $(CUDA_CFLAGS)
AM_LDFLAGS  = 

# default CUDA nvcc flags
#   Note: cc 13 for Tesla
#   Note: cc 20 for Fermi
#   Note: cc 30 for Kepler K10
#   Note: cc 35 for Kepler K20
NVCCFLAGS =  -D__USE_GPU__ -use_fast_math 
NVCCFLAGS += -gencode arch=compute_35,code=sm_35 --compiler-options "-O3 -funroll-loops -fstrict-aliasing" -O3
NVCCFLAGS += --ptxas-options -v
NVCCFLAGS += --compiler-options -I$(srcdir) 
NVCCFLAGS += --compiler-options -fno-strict-aliasing
NVCCFLAGS += --compiler-options -Wno-unused-function
NVCCFLAGS += --compiler-options -Wno-unused-parameter
NVCCFLAGS += --compiler-options "$(MPI_CFLAGS)"

#NVCC_LIBS = -lcudart -m64

bin_PROGRAMS = pg-puremd
pg_puremd_SOURCES = allocate.c basic_comm.c ffield.c grid.c list.c lookup.c \
      io_tools.c reset_tools.c restart.c random.c tool_box.c traj.c \
      vector.c analyze.c box.c system_props.c control.c comm_tools.c \
      geo_tools.c linear_solvers.c neighbors.c qEq.c bond_orders.c multi_body.c \
      bonds.c valence_angles.c hydrogen_bonds.c torsion_angles.c nonbonded.c forces.c \
      integrate.c init_md.c parallelreax.c cuda_utils.cu dev_alloc.cu cuda_environment.cu \
      dev_system_props.cu reduction.cu center_mass.cu cuda_copy.cu cuda_reset_tools.cu \
      dev_list.cu cuda_neighbors.cu cuda_bond_orders.cu cuda_bonds.cu cuda_multi_body.cu \
      cuda_valence_angles.cu cuda_torsion_angles.cu cuda_hydrogen_bonds.cu cuda_forces.cu \
      cuda_qEq.cu cuda_linear_solvers.cu matvec.cu dual_matvec.cu cuda_nonbonded.cu \
      cuda_integrate.cu cuda_post_evolve.cu cuda_init_md.cu validation.cu cuda_lookup.cu

# dummy source to cause C linking
nodist_EXTRA_pg_puremd_SOURCES = dummy.c

pg_puremd_CFLAGS = $(AM_CFLAGS) $(MPI_CFLAGS)
pg_puremd_LDFLAGS = $(AM_LDFLAGS) $(MPI_LDFLAGS) $(CUDA_LIBS)
