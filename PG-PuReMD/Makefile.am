if USE_CUDA
SUFFIXES = .cu
include ../cuda.am
endif

AM_CFLAGS = -Wall -O3 -funroll-loops -fstrict-aliasing $(MPI_CFLAGS)
AM_CPPFLAGS =
AM_LDFLAGS = $(MPI_LDFLAGS)

if USE_CUDA
# default CUDA nvcc flags
#   Note: cc 13 for Tesla
#   Note: cc 20 for Fermi
#   Note: cc 30 for Kepler K10
#   Note: cc 35 for Kepler K20
NVCCFLAGS += -use_fast_math 
NVCCFLAGS += -gencode arch=compute_35,code=sm_35
NVCCFLAGS += --compiler-options "$(DEFS) -D__SM_35__ -O3 -funroll-loops -fstrict-aliasing $(MPI_CFLAGS)"
#NVCCFLAGS += --ptxas-options -v
endif


bin_PROGRAMS = bin/pg-puremd
bin_pg_puremd_SOURCES = src/allocate.c src/basic_comm.c src/ffield.c src/grid.c src/list.c \
	src/lookup.c src/io_tools.c src/reset_tools.c src/restart.c src/random.c \
	src/tool_box.c src/traj.c src/analyze.c src/box.c src/system_props.c \
	src/control.c src/comm_tools.c src/geo_tools.c src/linear_solvers.c src/neighbors.c \
	src/qEq.c src/bond_orders.c src/multi_body.c src/bonds.c src/valence_angles.c \
	src/hydrogen_bonds.c src/torsion_angles.c src/nonbonded.c src/forces.c \
	src/integrate.c src/init_md.c src/parallelreax.c
include_HEADERS = src/reax_defs.h src/reax_types.h \
        src/allocate.h src/basic_comm.h src/ffield.h src/grid.h src/list.h \
	src/lookup.h src/io_tools.h src/reset_tools.h src/restart.h src/random.h \
	src/tool_box.h src/traj.h src/analyze.h src/box.h src/system_props.h \
	src/control.h src/comm_tools.h src/geo_tools.h src/linear_solvers.h src/neighbors.h \
	src/qEq.h src/bond_orders.h src/multi_body.h src/bonds.h src/valence_angles.h \
	src/hydrogen_bonds.h src/torsion_angles.h src/nonbonded.h src/forces.h \
	src/integrate.h src/init_md.h
bin_pg_puremd_LDADD = src/vector.o

if USE_CUDA
bin_pg_puremd_SOURCES += src/cuda_utils.cu src/dev_alloc.cu src/cuda_environment.cu \
      src/dev_system_props.cu src/reduction.cu src/center_mass.cu \
      src/cuda_copy.cu src/cuda_reset_tools.cu src/dev_list.cu \
      src/cuda_neighbors.cu src/cuda_bond_orders.cu src/cuda_bonds.cu \
      src/cuda_multi_body.cu src/cuda_valence_angles.cu \
      src/cuda_torsion_angles.cu src/cuda_hydrogen_bonds.cu src/cuda_forces.cu \
      src/cuda_qEq.cu src/cuda_linear_solvers.cu src/matvec.cu src/dual_matvec.cu \
      src/cuda_nonbonded.cu src/cuda_integrate.cu src/cuda_post_evolve.cu \
      src/cuda_init_md.cu src/validation.cu src/cuda_lookup.cu
include_HEADERS += src/cuda_utils.h src/dev_alloc.h src/cuda_environment.h \
      src/dev_system_props.h src/reduction.h src/center_mass.h \
      src/cuda_copy.h src/cuda_reset_tools.h src/dev_list.h \
      src/cuda_neighbors.h src/cuda_bond_orders.h src/cuda_bonds.h \
      src/cuda_multi_body.h src/cuda_valence_angles.h \
      src/cuda_torsion_angles.h src/cuda_hydrogen_bonds.h src/cuda_forces.h \
      src/cuda_qEq.h src/cuda_linear_solvers.h src/matvec.h src/dual_matvec.h \
      src/cuda_nonbonded.h src/cuda_integrate.h src/cuda_post_evolve.h \
      src/cuda_init_md.h src/validation.h src/cuda_lookup.h

# dummy source to cause C linking
nodist_EXTRA_bin_pg_puremd_SOURCES = src/dummy.c

src_vector.o:
	$(AM_V_NVCC)$(NVCC) $(NVCCFLAGS) -maxrregcount=$(MAX_REG_COUNT) -o src/vector.o -c src/vector.c
else
src_vector.o:
	$(AM_V_CC)$(CC) $(DEFS) $(AM_CPPFLAGS) $(CPPFLAGS) $(AM_CFLAGS) $(CCFLAGS) \
	-maxrregcount=$(MAX_REG_COUNT) -o src/vector.o -c src/vector.c
endif


bin_pg_puremd_CFLAGS = $(AM_CFLAGS) $(CFLAGS)
bin_pg_puremd_CPPFLAGS = $(AM_CPPFLAGS) $(CPPFLAGS)
bin_pg_puremd_LDFLAGS = $(AM_LDFLAGS) $(LDFLAGS)

if USE_CUDA
bin_pg_puremd_CFLAGS += $(CUDA_CFLAGS)
bin_pg_puremd_LDFLAGS += $(CUDA_LIBS)
endif
