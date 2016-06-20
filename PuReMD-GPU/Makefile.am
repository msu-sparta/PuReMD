SUFFIXES = .cu
include ../cuda.am

AM_CFLAGS = -Wall -O3 -funroll-loops -fstrict-aliasing
AM_CPPFLAGS =
AM_LDFLAGS =

# default CUDA nvcc flags
#   Note: cc 13 for Tesla
#   Note: cc 20 for Fermi
#   Note: cc 30 for Kepler K10
#   Note: cc 35 for Kepler K20
NVCCFLAGS += -use_fast_math 
NVCCFLAGS += -gencode arch=compute_35,code=sm_35
NVCCFLAGS += --compiler-options "$(DEFS) -D__SM_35__ -O3 -funroll-loops -fstrict-aliasing"
#NVCCFLAGS += --ptxas-options -v

bin_PROGRAMS = bin/puremd-gpu
bin_puremd_gpu_SOURCES = src/analyze.c src/print_utils.c src/reset_utils.c src/param.c src/pdb_tools.c \
	src/GMRES.cu src/QEq.cu src/allocate.cu src/bond_orders.cu \
	src/box.cu src/forces.cu src/four_body_interactions.cu \
	src/grid.cu src/init_md.cu src/integrate.cu src/list.cu \
	src/lookup.cu src/neighbors.cu \
	src/restart.cu src/single_body_interactions.cu \
	src/system_props.cu src/three_body_interactions.cu \
	src/traj.cu src/two_body_interactions.cu src/vector.cu \
	src/testmd.cu \
	src/cuda_utils.cu src/cuda_copy.cu src/cuda_init.cu src/reduction.cu \
	src/center_mass.cu src/helpers.cu src/validation.cu src/matvec.cu

# dummy source to cause C linking
nodist_EXTRA_bin_puremd_gpu_SOURCES = src/dummy.c

bin_puremd_gpu_CFLAGS = $(AM_CFLAGS) $(CFLAGS) $(CUDA_CFLAGS)
bin_puremd_gpu_CPPFLAGS = $(AM_CPPFLAGS) $(CPPFLAGS)
bin_puremd_gpu_LDFLAGS = $(AM_LDFLAGS) $(LDFLAGS) $(CUDA_LIBS)
