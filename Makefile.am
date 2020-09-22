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
sPuReMD_lib_libspuremd_la_LIBADD = @OMP_LIBS@ @L_LIBS@ @T_LIBS@ @G_LIBS@

bin_PROGRAMS = sPuReMD/bin/spuremd
sPuReMD_bin_spuremd_SOURCES = sPuReMD/src/driver.c
sPuReMD_bin_spuremd_CFLAGS = @DFT_CFLAGS@ @CFLAGS_EXTRA@ @OMP_CFLAGS@
sPuReMD_bin_spuremd_CPPFLAGS = -I sPuReMD/src @CPPFLAGS_EXTRA@
sPuReMD_bin_spuremd_LDADD = sPuReMD/lib/libspuremd.la @OMP_LIBS@ @L_LIBS@ @T_LIBS@ @G_LIBS@

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


if BUILD_MPI
bin_PROGRAMS = PuReMD/bin/puremd
PuReMD_bin_puremd_SOURCES = PuReMD/src/allocate.c \
			    PuReMD/src/allocate.h \
			    PuReMD/src/analyze.c \
			    PuReMD/src/analyze.h \
			    PuReMD/src/basic_comm.c \
			    PuReMD/src/basic_comm.h \
			    PuReMD/src/bond_orders.c \
			    PuReMD/src/bond_orders.h \
			    PuReMD/src/bonds.c \
			    PuReMD/src/bonds.h \
			    PuReMD/src/box.c \
			    PuReMD/src/box.h \
			    PuReMD/src/charges.c \
			    PuReMD/src/charges.h \
			    PuReMD/src/comm_tools.c \
			    PuReMD/src/comm_tools.h \
			    PuReMD/src/control.c \
			    PuReMD/src/control.h \
			    PuReMD/src/ffield.c \
			    PuReMD/src/ffield.h \
			    PuReMD/src/forces.c \
			    PuReMD/src/forces.h \
			    PuReMD/src/geo_tools.c \
			    PuReMD/src/geo_tools.h \
			    PuReMD/src/grid.c \
			    PuReMD/src/grid.h \
			    PuReMD/src/hydrogen_bonds.c \
			    PuReMD/src/hydrogen_bonds.h \
			    PuReMD/src/init_md.c \
			    PuReMD/src/init_md.h \
			    PuReMD/src/integrate.c \
			    PuReMD/src/integrate.h \
			    PuReMD/src/io_tools.c \
			    PuReMD/src/io_tools.h \
			    PuReMD/src/linear_solvers.c \
			    PuReMD/src/linear_solvers.h \
			    PuReMD/src/list.c \
			    PuReMD/src/list.h \
			    PuReMD/src/lookup.c \
			    PuReMD/src/lookup.h \
			    PuReMD/src/multi_body.c \
			    PuReMD/src/multi_body.h \
			    PuReMD/src/neighbors.c \
			    PuReMD/src/neighbors.h \
			    PuReMD/src/nonbonded.c \
			    PuReMD/src/nonbonded.h \
			    PuReMD/src/parallelreax.c \
			    PuReMD/src/random.c \
			    PuReMD/src/random.h \
			    PuReMD/src/reax_types.h \
			    PuReMD/src/reset_tools.c \
			    PuReMD/src/reset_tools.h \
			    PuReMD/src/restart.c \
			    PuReMD/src/restart.h \
			    PuReMD/src/system_props.c \
			    PuReMD/src/system_props.h \
			    PuReMD/src/tool_box.c \
			    PuReMD/src/tool_box.h \
			    PuReMD/src/torsion_angles.c \
			    PuReMD/src/torsion_angles.h \
			    PuReMD/src/traj.c \
			    PuReMD/src/traj.h \
			    PuReMD/src/valence_angles.c \
			    PuReMD/src/valence_angles.h \
			    PuReMD/src/vector.c \
			    PuReMD/src/vector.h

PuReMD_bin_puremd_CFLAGS = @DFT_CFLAGS@ @CFLAGS_EXTRA@ @M_CFLAGS@
PuReMD_bin_puremd_CPPFLAGS = -I PuReMD/src @CPPFLAGS_EXTRA@
PuReMD_bin_puremd_LDADD = @M_LIBS@ @L_LIBS@
endif


if BUILD_GPU
if USE_CUDA
SUFFIXES = .cu
include cuda.am
endif

if USE_CUDA
AM_NVCCFLAGS = --compiler-options="$(DEFS) $(AM_CFLAGS) @CFLAGS_EXTRA@ $(CFLAGS)" @NFLAGS@ @NFLAGS_EXTRA@
endif

bin_PROGRAMS = PuReMD-GPU/bin/spuremd
PuReMD_GPU_bin_spuremd_SOURCES = PuReMD-GPU/src/allocate.c \
				 PuReMD-GPU/src/allocate.h \
				 PuReMD-GPU/src/analyze.c \
				 PuReMD-GPU/src/analyze.h \
				 PuReMD-GPU/src/bond_orders.c \
				 PuReMD-GPU/src/bond_orders.h \
				 PuReMD-GPU/src/box.c \
				 PuReMD-GPU/src/box.h \
				 PuReMD-GPU/src/control.c \
				 PuReMD-GPU/src/control.h \
				 PuReMD-GPU/src/ffield.c \
				 PuReMD-GPU/src/ffield.c \
				 PuReMD-GPU/src/forces.c \
				 PuReMD-GPU/src/forces.h \
				 PuReMD-GPU/src/four_body_interactions.c \
				 PuReMD-GPU/src/four_body_interactions.h \
				 PuReMD-GPU/src/geo_tools.c \
				 PuReMD-GPU/src/geo_tools.h \
				 PuReMD-GPU/src/grid.c \
				 PuReMD-GPU/src/grid.h \
				 PuReMD-GPU/src/init_md.c \
				 PuReMD-GPU/src/init_md.h \
				 PuReMD-GPU/src/integrate.c \
				 PuReMD-GPU/src/integrate.h \
				 PuReMD-GPU/src/lin_alg.c \
				 PuReMD-GPU/src/lin_alg.h \
				 PuReMD-GPU/src/list.c \
				 PuReMD-GPU/src/list.h \
				 PuReMD-GPU/src/lookup.c \
				 PuReMD-GPU/src/lookup.h \
				 PuReMD-GPU/src/mytypes.h \
				 PuReMD-GPU/src/neighbors.c \
				 PuReMD-GPU/src/neighbors.h \
				 PuReMD-GPU/src/print_utils.c \
				 PuReMD-GPU/src/print_utils.h \
				 PuReMD-GPU/src/qeq.c \
				 PuReMD-GPU/src/qeq.h \
				 PuReMD-GPU/src/reset_utils.c \
				 PuReMD-GPU/src/reset_utils.h \
				 PuReMD-GPU/src/restart.c \
				 PuReMD-GPU/src/restart.h \
				 PuReMD-GPU/src/single_body_interactions.c \
				 PuReMD-GPU/src/single_body_interactions.h \
				 PuReMD-GPU/src/system_props.c \
				 PuReMD-GPU/src/system_props.h \
				 PuReMD-GPU/src/testmd.c \
				 PuReMD-GPU/src/three_body_interactions.c \
				 PuReMD-GPU/src/three_body_interactions.h \
				 PuReMD-GPU/src/tool_box.c \
				 PuReMD-GPU/src/tool_box.c \
				 PuReMD-GPU/src/traj.c \
				 PuReMD-GPU/src/traj.h \
				 PuReMD-GPU/src/two_body_interactions.c \
				 PuReMD-GPU/src/two_body_interactions.h \
				 PuReMD-GPU/src/vector.c \
				 PuReMD-GPU/src/vector.h
if USE_CUDA
PuReMD_GPU_bin_spuremd_SOURCES += PuReMD-GPU/src/cuda_allocate.cu \
				  PuReMD-GPU/src/cuda_allocate.h \
				  PuReMD-GPU/src/cuda_bond_orders.cu \
				  PuReMD-GPU/src/cuda_bond_orders.h \
				  PuReMD-GPU/src/cuda_box.cu \
				  PuReMD-GPU/src/cuda_box.h \
				  PuReMD-GPU/src/cuda_center_mass.cu \
				  PuReMD-GPU/src/cuda_center_mass.h \
				  PuReMD-GPU/src/cuda_copy.cu \
				  PuReMD-GPU/src/cuda_copy.h \
				  PuReMD-GPU/src/cuda_environment.cu \
				  PuReMD-GPU/src/cuda_environment.h \
				  PuReMD-GPU/src/cuda_forces.cu \
				  PuReMD-GPU/src/cuda_forces.h \
				  PuReMD-GPU/src/cuda_four_body_interactions.cu \
				  PuReMD-GPU/src/cuda_four_body_interactions.h \
				  PuReMD-GPU/src/cuda_grid.cu \
				  PuReMD-GPU/src/cuda_grid.h \
				  PuReMD-GPU/src/cuda_init.cu \
				  PuReMD-GPU/src/cuda_init.h \
				  PuReMD-GPU/src/cuda_init_md.cu \
				  PuReMD-GPU/src/cuda_init_md.h \
				  PuReMD-GPU/src/cuda_integrate.cu \
				  PuReMD-GPU/src/cuda_integrate.h \
				  PuReMD-GPU/src/cuda_lin_alg.cu \
				  PuReMD-GPU/src/cuda_lin_alg.h \
				  PuReMD-GPU/src/cuda_list.cu \
				  PuReMD-GPU/src/cuda_list.h \
				  PuReMD-GPU/src/cuda_lookup.cu \
				  PuReMD-GPU/src/cuda_lookup.h \
				  PuReMD-GPU/src/cuda_neighbors.cu \
				  PuReMD-GPU/src/cuda_neighbors.h \
				  PuReMD-GPU/src/cuda_post_evolve.cu \
				  PuReMD-GPU/src/cuda_post_evolve.h \
				  PuReMD-GPU/src/cuda_qeq.cu \
				  PuReMD-GPU/src/cuda_qeq.h \
				  PuReMD-GPU/src/cuda_reduction.cu \
				  PuReMD-GPU/src/cuda_reduction.h \
				  PuReMD-GPU/src/cuda_reset_utils.cu \
				  PuReMD-GPU/src/cuda_reset_utils.h \
				  PuReMD-GPU/src/cuda_single_body_interactions.cu \
				  PuReMD-GPU/src/cuda_single_body_interactions.h \
				  PuReMD-GPU/src/cuda_system_props.cu \
				  PuReMD-GPU/src/cuda_system_props.h \
				  PuReMD-GPU/src/cuda_three_body_interactions.cu \
				  PuReMD-GPU/src/cuda_three_body_interactions.h \
				  PuReMD-GPU/src/cuda_two_body_interactions.cu \
				  PuReMD-GPU/src/cuda_two_body_interactions.h \
				  PuReMD-GPU/src/cuda_utils.cu \
				  PuReMD-GPU/src/cuda_utils.h \
				  PuReMD-GPU/src/cuda_validation.cu \
				  PuReMD-GPU/src/cuda_validation.h
# dummy source to cause C linking
nodist_EXTRA_PuReMD_GPU_bin_spuremd_SOURCES = PuReMD-GPU/src/dummy.c
endif

PuReMD_GPU_bin_spuremd_CFLAGS = @DFT_CFLAGS@ @CFLAGS_EXTRA@
PuReMD_GPU_bin_spuremd_CPPFLAGS = -I PuReMD-GPU/src @CPPFLAGS_EXTRA@

if USE_CUDA
PuReMD_GPU_bin_spuremd_CFLAGS += @CUDA_CFLAGS@
PuReMD_GPU_bin_spuremd_LDFLAGS = @CUDA_LIBS@
endif
endif


if BUILD_MPI_GPU
EXTRA_DIST += PG-PuReMD/src/cub

if USE_CUDA
SUFFIXES = .cu
include cuda.am
endif

if USE_CUDA
AM_NVCCFLAGS = --compiler-options="$(DEFS) $(AM_CFLAGS) -std=c++11 @CFLAGS_EXTRA@ $(CFLAGS) @M_CFLAGS@" -std=c++11 @NFLAGS@ @NFLAGS_EXTRA@
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

PG_PuReMD_bin_pg_puremd_CFLAGS = -std=c11 @CFLAGS_EXTRA@ @M_CFLAGS@
PG_PuReMD_bin_pg_puremd_CPPFLAGS = -I PG-PuReMD/src @CPPFLAGS_EXTRA@
PG_PuReMD_bin_pg_puremd_LDADD = @M_LIBS@ @L_LIBS@ -lstdc++
if USE_CUDA
PG_PuReMD_bin_pg_puremd_CFLAGS += @CUDA_CFLAGS@
PG_PuReMD_bin_pg_puremd_LDADD += @CUDA_LIBS@
endif
endif

if BUILD_DOC
SUBDIRS += doc
DIST_SUBDIRS += doc
endif

dist-hook:
	rm -rf `find $(distdir) -name .git`
