ACLOCAL_AMFLAGS = -I ../m4

if USE_CUDA
SUFFIXES = .cu
include ../cuda.am
endif

if USE_CUDA
# flags for CUDA compilation via NVCC (see cuda.am)
#   Note: cc 13 for Tesla, cc 20 for Fermi, cc 30 for Kepler K10, cc 35 for Kepler K20/K40, etc.
#NVCCFLAGS += -use_fast_math -gencode arch=compute_35,code=sm_35
NVCCFLAGS += -use_fast_math $(NVCC_OPT_CODE)
#NVCCFLAGS += --compiler-options "$(DEFS) -D__SM_35__ -O3 -funroll-loops -fstrict-aliasing $(MPI_CFLAGS)"
NVCCFLAGS += --compiler-options "$(DEFS) $(NVCC_OPT_CODE_DEFS) -O3 -funroll-loops -fstrict-aliasing $(MPI_CFLAGS)"
#NVCCFLAGS += --ptxas-options -v
endif


lib_LTLIBRARIES = lib/libpuremd.la
lib_libpuremd_la_SOURCES = src/allocate.c src/basic_comm.c src/ffield.c src/grid.c src/list.c \
	src/lookup.c src/io_tools.c src/reset_tools.c src/restart.c src/random.c \
	src/tool_box.c src/traj.c src/analyze.c src/box.c src/system_props.c \
	src/control.c src/comm_tools.c src/geo_tools.c src/lin_alg.c src/neighbors.c \
	src/charges.c src/bond_orders.c src/multi_body.c src/bonds.c src/valence_angles.c \
	src/hydrogen_bonds.c src/torsion_angles.c src/nonbonded.c src/forces.c \
	src/integrate.c src/init_md.c src/puremd.c
lib_libpuremd_la_SOURCES += src/reax_types.h src/index_utils.h \
        src/allocate.h src/basic_comm.h src/ffield.h src/grid.h src/list.h \
	src/lookup.h src/io_tools.h src/reset_tools.h src/restart.h src/random.h src/vector.h \
	src/tool_box.h src/traj.h src/analyze.h src/box.h src/system_props.h \
	src/control.h src/comm_tools.h src/geo_tools.h src/lin_alg.h src/neighbors.h \
	src/charges.h src/bond_orders.h src/multi_body.h src/bonds.h src/valence_angles.h \
	src/hydrogen_bonds.h src/torsion_angles.h src/nonbonded.h src/forces.h \
	src/integrate.h src/init_md.h
include_HEADERS = src/puremd.h

if USE_CUDA
lib_libpuremd_la_SOURCES += src/cuda/cuda_utils.cu src/cuda/cuda_allocate.cu src/cuda/cuda_environment.cu \
      src/cuda/cuda_system_props.cu src/cuda/cuda_reduction.cu src/cuda/cuda_box.cu src/cuda/cuda_list.cu \
      src/cuda/cuda_copy.cu src/cuda/cuda_reset_tools.cu src/cuda/cuda_random.cu \
      src/cuda/cuda_neighbors.cu src/cuda/cuda_bond_orders.cu src/cuda/cuda_bonds.cu \
      src/cuda/cuda_multi_body.cu src/cuda/cuda_valence_angles.cu \
      src/cuda/cuda_torsion_angles.cu src/cuda/cuda_hydrogen_bonds.cu src/cuda/cuda_forces.cu \
      src/cuda/cuda_charges.cu src/cuda/cuda_lin_alg.cu \
      src/cuda/cuda_nonbonded.cu src/cuda/cuda_integrate.cu src/cuda/cuda_post_evolve.cu \
      src/cuda/cuda_init_md.cu src/cuda/cuda_lookup.cu
lib_libpuremd_la_SOURCES += src/cuda/cuda_helpers.h src/cuda/cuda_shuffle.h \
      src/cuda/cuda_utils.h src/cuda/cuda_allocate.h src/cuda/cuda_environment.h \
      src/cuda/cuda_system_props.h src/cuda/cuda_reduction.h src/cuda/cuda_box.h src/cuda/cuda_list.h \
      src/cuda/cuda_copy.h src/cuda/cuda_reset_tools.h src/cuda/cuda_random.h src/cuda/cuda_vector.h \
      src/cuda/cuda_neighbors.h src/cuda/cuda_bond_orders.h src/cuda/cuda_bonds.h \
      src/cuda/cuda_multi_body.h src/cuda/cuda_valence_angles.h \
      src/cuda/cuda_torsion_angles.h src/cuda/cuda_hydrogen_bonds.h src/cuda/cuda_forces.h \
      src/cuda/cuda_charges.h src/cuda/cuda_lin_alg.h \
      src/cuda/cuda_nonbonded.h src/cuda/cuda_integrate.h src/cuda/cuda_post_evolve.h \
      src/cuda/cuda_init_md.h src/cuda/cuda_lookup.h

if DEBUG
lib_libpuremd_la_SOURCES += src/cuda/cuda_validation.cu
lib_libpuremd_la_SOURCES += src/cuda/cuda_validation.h
endif

# dummy source to cause C linking
nodist_EXTRA_lib_libpuremd_la_SOURCES = src/dummy.c
endif

lib_libpuremd_la_CFLAGS = $(AM_CFLAGS) $(MPI_CFLAGS) $(CFLAGS)
lib_libpuremd_la_CPPFLAGS = $(AM_CPPFLAGS) $(CPPFLAGS)
lib_libpuremd_la_LIBADD = $(AM_LDADD) $(MPI_LIBS) $(LDADD) -lstdc++
lib_libpuremd_la_LDFLAGS = -version-info 1:0:0
if USE_CUDA
lib_libpuremd_la_CFLAGS += $(CUDA_CFLAGS)
lib_libpuremd_la_LIBADD += $(CUDA_LIBS)
endif

bin_PROGRAMS = bin/pg-puremd
bin_pg_puremd_SOURCES = src/driver.c
bin_pg_puremd_CFLAGS = $(AM_CFLAGS) $(MPI_CFLAGS) $(CFLAGS)
bin_pg_puremd_CPPFLAGS = -I src $(AM_CPPFLAGS) $(CPPFLAGS)
bin_pg_puremd_LDADD = lib/libpuremd.la $(AM_LDADD) $(MPI_LIBS) $(LDADD) -lstdc++
if USE_CUDA
bin_pg_puremd_CFLAGS += $(CUDA_CFLAGS)
bin_pg_puremd_LDADD += $(CUDA_LIBS)
endif
