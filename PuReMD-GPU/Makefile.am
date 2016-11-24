ACLOCAL_AMFLAGS = -I ../m4

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
bin_puremd_gpu_SOURCES = src/analyze.c src/print_utils.c src/restart.c src/param.c src/pdb_tools.c \
	src/GMRES.cu src/QEq.cu src/allocate.cu src/bond_orders.cu \
	src/box.cu src/forces.cu src/four_body_interactions.cu \
	src/grid.cu src/init_md.cu src/integrate.cu src/list.cu \
	src/lookup.cu src/neighbors.cu \
	src/reset_utils.cu src/single_body_interactions.cu \
	src/system_props.cu src/three_body_interactions.cu \
	src/traj.cu src/two_body_interactions.cu src/vector.cu \
	src/testmd.cu \
	src/cuda_utils.cu src/cuda_copy.cu src/cuda_init.cu src/reduction.cu \
	src/center_mass.cu src/helpers.cu src/validation.cu src/matvec.cu

include_HEADERS = src/mytypes.h src/analyze.h src/print_utils.h \
        src/restart.h src/param.h src/pdb_tools.h \
	src/GMRES.h src/QEq.h src/allocate.h src/bond_orders.h \
	src/box.h src/forces.h src/four_body_interactions.h \
	src/grid.h src/init_md.h src/integrate.h src/list.h \
	src/lookup.h src/neighbors.h \
	src/reset_utils.h src/single_body_interactions.h \
	src/system_props.h src/three_body_interactions.h \
	src/traj.h src/two_body_interactions.h src/vector.h \
	src/cuda_utils.h src/cuda_copy.h src/cuda_init.h src/reduction.h \
	src/center_mass.h src/helpers.h src/validation.h src/matvec.h

# dummy source to cause C linking
nodist_EXTRA_bin_puremd_gpu_SOURCES = src/dummy.c

bin_puremd_gpu_CFLAGS = $(AM_CFLAGS) $(CFLAGS) $(CUDA_CFLAGS)
bin_puremd_gpu_CPPFLAGS = $(AM_CPPFLAGS) $(CPPFLAGS)
bin_puremd_gpu_LDFLAGS = $(AM_LDFLAGS) $(LDFLAGS) $(CUDA_LIBS)
