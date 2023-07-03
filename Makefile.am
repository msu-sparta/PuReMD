ACLOCAL_AMFLAGS = -I m4

AM_CFLAGS = -O2 -funroll-loops -Wall

SUBDIRS =
DIST_SUBDIRS =
EXTRA_DIST = data environ tools

# sPuReMD
if BUILD_S_OMP
lib_LTLIBRARIES = sPuReMD/lib/libspuremd.la
sPuReMD_lib_libspuremd_la_SOURCES = sPuReMD/src/allocate.c \
				    sPuReMD/src/allocate.h \
				    sPuReMD/src/analyze.c \
				    sPuReMD/src/analyze.h \
				    sPuReMD/src/bond_orders.c \
				    sPuReMD/src/bond_orders.h \
				    sPuReMD/src/bonds.c \
				    sPuReMD/src/bonds.h \
				    sPuReMD/src/box.c \
				    sPuReMD/src/box.h \
				    sPuReMD/src/charges.c \
				    sPuReMD/src/charges.h \
				    sPuReMD/src/control.c \
				    sPuReMD/src/control.h \
				    sPuReMD/src/ffield.c \
				    sPuReMD/src/ffield.h \
				    sPuReMD/src/forces.c \
				    sPuReMD/src/forces.h \
				    sPuReMD/src/hydrogen_bonds.c \
				    sPuReMD/src/hydrogen_bonds.h \
				    sPuReMD/src/geo_tools.c \
				    sPuReMD/src/geo_tools.h \
				    sPuReMD/src/grid.c \
				    sPuReMD/src/grid.h \
				    sPuReMD/src/init_md.c \
				    sPuReMD/src/init_md.h \
				    sPuReMD/src/io_tools.c \
				    sPuReMD/src/io_tools.h \
				    sPuReMD/src/integrate.c \
				    sPuReMD/src/integrate.h \
				    sPuReMD/src/lin_alg.c \
				    sPuReMD/src/lin_alg.h \
				    sPuReMD/src/list.c \
				    sPuReMD/src/list.h \
				    sPuReMD/src/lookup.c \
				    sPuReMD/src/lookup.h \
				    sPuReMD/src/multi_body.c \
				    sPuReMD/src/multi_body.h \
				    sPuReMD/src/neighbors.c \
				    sPuReMD/src/neighbors.h \
				    sPuReMD/src/nonbonded.c \
				    sPuReMD/src/nonbonded.h \
				    sPuReMD/src/random.c \
				    sPuReMD/src/random.h \
				    sPuReMD/src/reax_types.h \
				    sPuReMD/src/reset_tools.c \
				    sPuReMD/src/reset_tools.h \
				    sPuReMD/src/restart.c \
				    sPuReMD/src/restart.h \
				    sPuReMD/src/spuremd.c \
				    sPuReMD/src/spuremd.h \
				    sPuReMD/src/system_props.c \
				    sPuReMD/src/system_props.h \
				    sPuReMD/src/torsion_angles.c \
				    sPuReMD/src/torsion_angles.h \
				    sPuReMD/src/tool_box.c \
				    sPuReMD/src/tool_box.h \
				    sPuReMD/src/traj.c \
				    sPuReMD/src/traj.h \
				    sPuReMD/src/valence_angles.c \
				    sPuReMD/src/valence_angles.h \
				    sPuReMD/src/vector.c \
				    sPuReMD/src/vector.h
sPuReMD_lib_libspuremd_la_CFLAGS = @DFT_CFLAGS@ @CFLAGS_EXTRA@ @OMP_CFLAGS@
sPuReMD_lib_libspuremd_la_CPPFLAGS = -I sPuReMD/src @CPPFLAGS_EXTRA@
sPuReMD_lib_libspuremd_la_LDFLAGS = -version-info 1:0:0
sPuReMD_lib_libspuremd_la_LIBADD = @OMP_LIBS@ @Z_LIBS@ @L_LIBS@ @T_LIBS@ @G_LIBS@

bin_PROGRAMS = sPuReMD/bin/spuremd
sPuReMD_bin_spuremd_SOURCES = sPuReMD/src/driver.c
sPuReMD_bin_spuremd_CFLAGS = @DFT_CFLAGS@ @CFLAGS_EXTRA@ @OMP_CFLAGS@
sPuReMD_bin_spuremd_CPPFLAGS = -I sPuReMD/src @CPPFLAGS_EXTRA@
sPuReMD_bin_spuremd_LDADD = sPuReMD/lib/libspuremd.la @OMP_LIBS@ @Z_LIBS@ @L_LIBS@ @T_LIBS@ @G_LIBS@

check_PROGRAMS =
TESTS =

if BUILD_TEST
check_PROGRAMS += sPuReMD/tests/test_spuremd
check_PROGRAMS += sPuReMD/tests/test_lin_alg
check_PROGRAMS += sPuReMD/tests/test_vector
TESTS += $(check_PROGRAMS)

sPuReMD_tests_test_spuremd_SOURCES = sPuReMD/tests/test_spuremd.cpp
sPuReMD_tests_test_spuremd_CPPFLAGS = -I sPuReMD/src $(GTEST_CPPFLAGS)
sPuReMD_tests_test_spuremd_CXXFLAGS = $(GTEST_CXXFLAGS)
sPuReMD_tests_test_spuremd_LDFLAGS = $(GTEST_LDFLAGS) $(GTEST_LIBS)
sPuReMD_tests_test_spuremd_LDADD = sPuReMD/lib/libspuremd.la -lgtest

sPuReMD_tests_test_lin_alg_SOURCES = sPuReMD/tests/test_lin_alg.cpp
sPuReMD_tests_test_lin_alg_CPPFLAGS = -I sPuReMD/src $(GTEST_CPPFLAGS)
sPuReMD_tests_test_lin_alg_CXXFLAGS = $(GTEST_CXXFLAGS)
sPuReMD_tests_test_lin_alg_LDFLAGS = $(GTEST_LDFLAGS) $(GTEST_LIBS)
sPuReMD_tests_test_lin_alg_LDADD = -lgtest

sPuReMD_tests_test_vector_SOURCES = sPuReMD/tests/test_vector.cpp
sPuReMD_tests_test_vector_CPPFLAGS = -I sPuReMD/src $(GTEST_CPPFLAGS)
sPuReMD_tests_test_vector_CXXFLAGS = $(GTEST_CXXFLAGS)
sPuReMD_tests_test_vector_LDFLAGS = $(GTEST_LDFLAGS) $(GTEST_LIBS)
sPuReMD_tests_test_vector_LDADD = -lgtest
endif
endif


if BUILD_MPI_GPU
EXTRA_DIST += PG-PuReMD/src/cub

if USE_CUDA
SUFFIXES = .cu
include cuda.am

AM_NVCCFLAGS = --compiler-options="$(DEFS) $(AM_CFLAGS) -std=c++14 @CFLAGS_EXTRA@ $(CFLAGS) @OMP_CFLAGS@ @M_CFLAGS@" -std=c++14 @NFLAGS@ @NFLAGS_EXTRA@
endif

if USE_HIP
SUFFIXES = .cu
include hip.am

AM_HIPCCFLAGS = $(DEFS) $(AM_CFLAGS) -std=c++14 @CFLAGS_EXTRA@ $(CFLAGS) @OMP_CFLAGS@ @M_CFLAGS@
endif


bin_PROGRAMS = PG-PuReMD/bin/pg-puremd
PG_PuReMD_bin_pg_puremd_SOURCES = PG-PuReMD/src/allocate.c \
				  PG-PuReMD/src/allocate.h \
				  PG-PuReMD/src/analyze.c \
				  PG-PuReMD/src/analyze.h \
				  PG-PuReMD/src/basic_comm.c \
				  PG-PuReMD/src/basic_comm.h \
				  PG-PuReMD/src/bond_orders.c \
				  PG-PuReMD/src/bond_orders.h \
				  PG-PuReMD/src/bonds.c \
				  PG-PuReMD/src/bonds.h \
				  PG-PuReMD/src/box.c \
				  PG-PuReMD/src/box.h \
				  PG-PuReMD/src/charges.c \
				  PG-PuReMD/src/charges.h \
				  PG-PuReMD/src/comm_tools.c \
				  PG-PuReMD/src/comm_tools.h \
				  PG-PuReMD/src/control.c \
				  PG-PuReMD/src/control.h \
				  PG-PuReMD/src/driver.c \
				  PG-PuReMD/src/ffield.c \
				  PG-PuReMD/src/ffield.h \
				  PG-PuReMD/src/forces.c \
				  PG-PuReMD/src/forces.h \
				  PG-PuReMD/src/geo_tools.c \
				  PG-PuReMD/src/geo_tools.h \
				  PG-PuReMD/src/grid.c \
				  PG-PuReMD/src/grid.h \
				  PG-PuReMD/src/hydrogen_bonds.c \
				  PG-PuReMD/src/hydrogen_bonds.h \
				  PG-PuReMD/src/index_utils.h \
				  PG-PuReMD/src/init_md.c \
				  PG-PuReMD/src/init_md.h \
				  PG-PuReMD/src/integrate.c \
				  PG-PuReMD/src/integrate.h \
				  PG-PuReMD/src/io_tools.c \
				  PG-PuReMD/src/io_tools.h \
				  PG-PuReMD/src/lin_alg.c \
				  PG-PuReMD/src/lin_alg.h \
				  PG-PuReMD/src/list.c \
				  PG-PuReMD/src/list.h \
				  PG-PuReMD/src/lookup.c \
				  PG-PuReMD/src/lookup.h \
				  PG-PuReMD/src/multi_body.c \
				  PG-PuReMD/src/multi_body.h \
				  PG-PuReMD/src/neighbors.c \
				  PG-PuReMD/src/neighbors.h \
				  PG-PuReMD/src/nonbonded.c \
				  PG-PuReMD/src/nonbonded.h \
				  PG-PuReMD/src/puremd.c \
				  PG-PuReMD/src/puremd.h \
				  PG-PuReMD/src/random.c \
				  PG-PuReMD/src/random.h \
				  PG-PuReMD/src/reax_types.h \
				  PG-PuReMD/src/reset_tools.c \
				  PG-PuReMD/src/reset_tools.h \
				  PG-PuReMD/src/restart.c \
				  PG-PuReMD/src/restart.h \
				  PG-PuReMD/src/system_props.c \
				  PG-PuReMD/src/system_props.h \
				  PG-PuReMD/src/tool_box.c \
				  PG-PuReMD/src/tool_box.h \
				  PG-PuReMD/src/torsion_angles.c \
				  PG-PuReMD/src/torsion_angles.h \
				  PG-PuReMD/src/traj.c \
				  PG-PuReMD/src/traj.h \
				  PG-PuReMD/src/valence_angles.c \
				  PG-PuReMD/src/valence_angles.h \
				  PG-PuReMD/src/vector.h
if USE_CUDA
PG_PuReMD_bin_pg_puremd_SOURCES += PG-PuReMD/src/cuda/cuda_allocate.cu \
				   PG-PuReMD/src/cuda/cuda_allocate.h \
				   PG-PuReMD/src/cuda/cuda_basic_comm.cu \
				   PG-PuReMD/src/cuda/cuda_basic_comm.h \
				   PG-PuReMD/src/cuda/cuda_bond_orders.cu \
				   PG-PuReMD/src/cuda/cuda_bond_orders.h \
				   PG-PuReMD/src/cuda/cuda_bonds.cu \
				   PG-PuReMD/src/cuda/cuda_bonds.h \
				   PG-PuReMD/src/cuda/cuda_box.cu \
				   PG-PuReMD/src/cuda/cuda_box.h \
				   PG-PuReMD/src/cuda/cuda_charges.cu \
				   PG-PuReMD/src/cuda/cuda_charges.h \
				   PG-PuReMD/src/cuda/cuda_copy.cu \
				   PG-PuReMD/src/cuda/cuda_copy.h \
				   PG-PuReMD/src/cuda/cuda_dense_lin_alg.cu \
				   PG-PuReMD/src/cuda/cuda_dense_lin_alg.h \
				   PG-PuReMD/src/cuda/cuda_environment.cu \
				   PG-PuReMD/src/cuda/cuda_environment.h \
				   PG-PuReMD/src/cuda/cuda_forces.cu \
				   PG-PuReMD/src/cuda/cuda_forces.h \
				   PG-PuReMD/src/cuda/cuda_helpers.h \
				   PG-PuReMD/src/cuda/cuda_hydrogen_bonds.cu \
				   PG-PuReMD/src/cuda/cuda_hydrogen_bonds.h \
				   PG-PuReMD/src/cuda/cuda_init_md.cu \
				   PG-PuReMD/src/cuda/cuda_init_md.h \
				   PG-PuReMD/src/cuda/cuda_integrate.cu \
				   PG-PuReMD/src/cuda/cuda_integrate.h \
				   PG-PuReMD/src/cuda/cuda_list.cu \
				   PG-PuReMD/src/cuda/cuda_list.h \
				   PG-PuReMD/src/cuda/cuda_lookup.cu \
				   PG-PuReMD/src/cuda/cuda_lookup.h \
				   PG-PuReMD/src/cuda/cuda_multi_body.cu \
				   PG-PuReMD/src/cuda/cuda_multi_body.h \
				   PG-PuReMD/src/cuda/cuda_neighbors.cu \
				   PG-PuReMD/src/cuda/cuda_neighbors.h \
				   PG-PuReMD/src/cuda/cuda_nonbonded.cu \
				   PG-PuReMD/src/cuda/cuda_nonbonded.h \
				   PG-PuReMD/src/cuda/cuda_post_evolve.cu \
				   PG-PuReMD/src/cuda/cuda_post_evolve.h \
				   PG-PuReMD/src/cuda/cuda_random.cu \
				   PG-PuReMD/src/cuda/cuda_random.h \
				   PG-PuReMD/src/cuda/cuda_reduction.cu \
				   PG-PuReMD/src/cuda/cuda_reduction.h \
				   PG-PuReMD/src/cuda/cuda_reset_tools.cu \
				   PG-PuReMD/src/cuda/cuda_reset_tools.h \
				   PG-PuReMD/src/cuda/cuda_spar_lin_alg.cu \
				   PG-PuReMD/src/cuda/cuda_spar_lin_alg.h \
				   PG-PuReMD/src/cuda/cuda_system_props.cu \
				   PG-PuReMD/src/cuda/cuda_system_props.h \
				   PG-PuReMD/src/cuda/cuda_torsion_angles.cu \
				   PG-PuReMD/src/cuda/cuda_torsion_angles.h \
				   PG-PuReMD/src/cuda/cuda_utils.cu \
				   PG-PuReMD/src/cuda/cuda_utils.h \
				   PG-PuReMD/src/cuda/cuda_valence_angles.cu \
				   PG-PuReMD/src/cuda/cuda_valence_angles.h \
				   PG-PuReMD/src/cuda/cuda_vector.h
# dummy source to trigger C linking
nodist_EXTRA_PG_PuReMD_bin_pg_puremd_SOURCES = src/dummy.c
endif
if USE_HIP
PG_PuReMD_bin_pg_puremd_SOURCES += PG-PuReMD/src/hip/hip_allocate.cu \
				   PG-PuReMD/src/hip/hip_allocate.h \
				   PG-PuReMD/src/hip/hip_basic_comm.cu \
				   PG-PuReMD/src/hip/hip_basic_comm.h \
				   PG-PuReMD/src/hip/hip_bond_orders.cu \
				   PG-PuReMD/src/hip/hip_bond_orders.h \
				   PG-PuReMD/src/hip/hip_bonds.cu \
				   PG-PuReMD/src/hip/hip_bonds.h \
				   PG-PuReMD/src/hip/hip_box.cu \
				   PG-PuReMD/src/hip/hip_box.h \
				   PG-PuReMD/src/hip/hip_charges.cu \
				   PG-PuReMD/src/hip/hip_charges.h \
				   PG-PuReMD/src/hip/hip_copy.cu \
				   PG-PuReMD/src/hip/hip_copy.h \
				   PG-PuReMD/src/hip/hip_dense_lin_alg.cu \
				   PG-PuReMD/src/hip/hip_dense_lin_alg.h \
				   PG-PuReMD/src/hip/hip_environment.cu \
				   PG-PuReMD/src/hip/hip_environment.h \
				   PG-PuReMD/src/hip/hip_forces.cu \
				   PG-PuReMD/src/hip/hip_forces.h \
				   PG-PuReMD/src/hip/hip_helpers.h \
				   PG-PuReMD/src/hip/hip_hydrogen_bonds.cu \
				   PG-PuReMD/src/hip/hip_hydrogen_bonds.h \
				   PG-PuReMD/src/hip/hip_init_md.cu \
				   PG-PuReMD/src/hip/hip_init_md.h \
				   PG-PuReMD/src/hip/hip_integrate.cu \
				   PG-PuReMD/src/hip/hip_integrate.h \
				   PG-PuReMD/src/hip/hip_list.cu \
				   PG-PuReMD/src/hip/hip_list.h \
				   PG-PuReMD/src/hip/hip_lookup.cu \
				   PG-PuReMD/src/hip/hip_lookup.h \
				   PG-PuReMD/src/hip/hip_multi_body.cu \
				   PG-PuReMD/src/hip/hip_multi_body.h \
				   PG-PuReMD/src/hip/hip_neighbors.cu \
				   PG-PuReMD/src/hip/hip_neighbors.h \
				   PG-PuReMD/src/hip/hip_nonbonded.cu \
				   PG-PuReMD/src/hip/hip_nonbonded.h \
				   PG-PuReMD/src/hip/hip_post_evolve.cu \
				   PG-PuReMD/src/hip/hip_post_evolve.h \
				   PG-PuReMD/src/hip/hip_random.cu \
				   PG-PuReMD/src/hip/hip_random.h \
				   PG-PuReMD/src/hip/hip_reduction.cu \
				   PG-PuReMD/src/hip/hip_reduction.h \
				   PG-PuReMD/src/hip/hip_reset_tools.cu \
				   PG-PuReMD/src/hip/hip_reset_tools.h \
				   PG-PuReMD/src/hip/hip_spar_lin_alg.cu \
				   PG-PuReMD/src/hip/hip_spar_lin_alg.h \
				   PG-PuReMD/src/hip/hip_system_props.cu \
				   PG-PuReMD/src/hip/hip_system_props.h \
				   PG-PuReMD/src/hip/hip_torsion_angles.cu \
				   PG-PuReMD/src/hip/hip_torsion_angles.h \
				   PG-PuReMD/src/hip/hip_utils.cu \
				   PG-PuReMD/src/hip/hip_utils.h \
				   PG-PuReMD/src/hip/hip_valence_angles.cu \
				   PG-PuReMD/src/hip/hip_valence_angles.h \
				   PG-PuReMD/src/hip/hip_vector.h
# dummy source to trigger C linking
nodist_EXTRA_PG_PuReMD_bin_pg_puremd_SOURCES = src/dummy.c
endif

PG_PuReMD_bin_pg_puremd_CFLAGS = -std=c11 @CFLAGS_EXTRA@ @OMP_CFLAGS@ @M_CFLAGS@
PG_PuReMD_bin_pg_puremd_CPPFLAGS = -I PG-PuReMD/src @CPPFLAGS_EXTRA@
PG_PuReMD_bin_pg_puremd_LDADD = @OMP_LIBS@ @M_LIBS@ @L_LIBS@ -lstdc++
if USE_CUDA
PG_PuReMD_bin_pg_puremd_CFLAGS += @CUDA_CFLAGS@
PG_PuReMD_bin_pg_puremd_LDADD += @CUDA_LIBS@
endif
if USE_HIP
PG_PuReMD_bin_pg_puremd_CFLAGS += @HIP_CFLAGS@
PG_PuReMD_bin_pg_puremd_LDADD += @HIP_LIBS@
endif
endif

if BUILD_DOC
SUBDIRS += doc
DIST_SUBDIRS += doc
endif

dist-hook:
	rm -rf `find $(distdir) -name .git`
