ACLOCAL_AMFLAGS = -I m4

AM_CFLAGS = -O2 -funroll-loops -Wall

bin_PROGRAMS = 
lib_LTLIBRARIES =
check_PROGRAMS =
TESTS =
SUBDIRS =
DIST_SUBDIRS =
EXTRA_DIST = data environ tools

if BUILD_SERIAL
lib_LTLIBRARIES += lib/libpuremd.la
lib_libpuremd_la_SOURCES = src/shared_mem/allocate.c \
			   src/shared_mem/allocate.h \
			   src/shared_mem/analyze.c \
			   src/shared_mem/analyze.h \
			   src/shared_mem/bond_orders.c \
			   src/shared_mem/bond_orders.h \
			   src/shared_mem/bonds.c \
			   src/shared_mem/bonds.h \
			   src/shared_mem/box.c \
			   src/shared_mem/box.h \
			   src/shared_mem/charges.c \
			   src/shared_mem/charges.h \
			   src/shared_mem/control.c \
			   src/shared_mem/control.h \
			   src/shared_mem/ffield.c \
			   src/shared_mem/ffield.h \
			   src/shared_mem/forces.c \
			   src/shared_mem/forces.h \
			   src/shared_mem/hydrogen_bonds.c \
			   src/shared_mem/hydrogen_bonds.h \
			   src/shared_mem/geo_tools.c \
			   src/shared_mem/geo_tools.h \
			   src/shared_mem/grid.c \
			   src/shared_mem/grid.h \
			   src/shared_mem/init_md.c \
			   src/shared_mem/init_md.h \
			   src/shared_mem/io_tools.c \
			   src/shared_mem/io_tools.h \
			   src/shared_mem/integrate.c \
			   src/shared_mem/integrate.h \
			   src/shared_mem/lin_alg.c \
			   src/shared_mem/lin_alg.h \
			   src/shared_mem/list.c \
			   src/shared_mem/list.h \
			   src/shared_mem/lookup.c \
			   src/shared_mem/lookup.h \
			   src/shared_mem/multi_body.c \
			   src/shared_mem/multi_body.h \
			   src/shared_mem/neighbors.c \
			   src/shared_mem/neighbors.h \
			   src/shared_mem/nonbonded.c \
			   src/shared_mem/nonbonded.h \
			   src/shared_mem/random.c \
			   src/shared_mem/random.h \
			   src/shared_mem/reax_types.h \
			   src/shared_mem/reset_tools.c \
			   src/shared_mem/reset_tools.h \
			   src/shared_mem/restart.c \
			   src/shared_mem/restart.h \
			   src/shared_mem/puremd.c \
			   src/shared_mem/puremd.h \
			   src/shared_mem/system_props.c \
			   src/shared_mem/system_props.h \
			   src/shared_mem/torsion_angles.c \
			   src/shared_mem/torsion_angles.h \
			   src/shared_mem/tool_box.c \
			   src/shared_mem/tool_box.h \
			   src/shared_mem/traj.c \
			   src/shared_mem/traj.h \
			   src/shared_mem/valence_angles.c \
			   src/shared_mem/valence_angles.h \
			   src/shared_mem/vector.c \
			   src/shared_mem/vector.h
lib_libpuremd_la_CFLAGS = @DFT_CFLAGS@ @CFLAGS_EXTRA@
lib_libpuremd_la_CPPFLAGS = -I src/shared_mem @CPPFLAGS_EXTRA@
lib_libpuremd_la_LDFLAGS = -version-info 1:0:0
lib_libpuremd_la_LIBADD = @Z_LIBS@ @L_LIBS@ @T_LIBS@ @G_LIBS@

bin_PROGRAMS += bin/puremd
bin_puremd_SOURCES = src/shared_mem/driver.c
bin_puremd_CFLAGS = @DFT_CFLAGS@ @CFLAGS_EXTRA@
bin_puremd_CPPFLAGS = -I src/shared_mem @CPPFLAGS_EXTRA@
bin_puremd_LDADD = lib/libpuremd.la @Z_LIBS@ @L_LIBS@ @T_LIBS@ @G_LIBS@

if BUILD_TEST
check_PROGRAMS += test/shared_mem/test_puremd
check_PROGRAMS += test/shared_mem/test_lin_alg
check_PROGRAMS += test/shared_mem/test_vector
TESTS += $(check_PROGRAMS)

test_shared_mem_test_puremd_SOURCES = test/shared_mem/test_puremd.cpp
test_shared_mem_test_puremd_CPPFLAGS = -I src/shared_mem $(GTEST_CPPFLAGS)
test_shared_mem_test_puremd_CXXFLAGS = $(GTEST_CXXFLAGS)
test_shared_mem_test_puremd_LDFLAGS = $(GTEST_LDFLAGS) $(GTEST_LIBS)
test_shared_mem_test_puremd_LDADD = lib/libpuremd.la -lgtest

test_shared_mem_test_lin_alg_SOURCES = test/shared_mem/test_lin_alg.cpp
test_shared_mem_test_lin_alg_CPPFLAGS = -I src/shared_mem $(GTEST_CPPFLAGS)
test_shared_mem_test_lin_alg_CXXFLAGS = $(GTEST_CXXFLAGS)
test_shared_mem_test_lin_alg_LDFLAGS = $(GTEST_LDFLAGS) $(GTEST_LIBS)
test_shared_mem_test_lin_alg_LDADD = -lgtest

test_shared_mem_test_vector_SOURCES = test/shared_mem/test_vector.cpp
test_shared_mem_test_vector_CPPFLAGS = -I src/shared_mem $(GTEST_CPPFLAGS)
test_shared_mem_test_vector_CXXFLAGS = $(GTEST_CXXFLAGS)
test_shared_mem_test_vector_LDFLAGS = $(GTEST_LDFLAGS) $(GTEST_LIBS)
test_shared_mem_test_vector_LDADD = -lgtest
endif
endif


if BUILD_OPENMP
lib_LTLIBRARIES += lib/libpuremd-openmp.la
lib_libpuremd_openmp_la_SOURCES = src/shared_mem/allocate.c \
				  src/shared_mem/allocate.h \
				  src/shared_mem/analyze.c \
				  src/shared_mem/analyze.h \
				  src/shared_mem/bond_orders.c \
				  src/shared_mem/bond_orders.h \
				  src/shared_mem/bonds.c \
				  src/shared_mem/bonds.h \
				  src/shared_mem/box.c \
				  src/shared_mem/box.h \
				  src/shared_mem/charges.c \
				  src/shared_mem/charges.h \
				  src/shared_mem/control.c \
				  src/shared_mem/control.h \
				  src/shared_mem/ffield.c \
				  src/shared_mem/ffield.h \
				  src/shared_mem/forces.c \
				  src/shared_mem/forces.h \
				  src/shared_mem/hydrogen_bonds.c \
				  src/shared_mem/hydrogen_bonds.h \
				  src/shared_mem/geo_tools.c \
				  src/shared_mem/geo_tools.h \
				  src/shared_mem/grid.c \
				  src/shared_mem/grid.h \
				  src/shared_mem/init_md.c \
				  src/shared_mem/init_md.h \
				  src/shared_mem/io_tools.c \
				  src/shared_mem/io_tools.h \
				  src/shared_mem/integrate.c \
				  src/shared_mem/integrate.h \
				  src/shared_mem/lin_alg.c \
				  src/shared_mem/lin_alg.h \
				  src/shared_mem/list.c \
				  src/shared_mem/list.h \
				  src/shared_mem/lookup.c \
				  src/shared_mem/lookup.h \
				  src/shared_mem/multi_body.c \
				  src/shared_mem/multi_body.h \
				  src/shared_mem/neighbors.c \
				  src/shared_mem/neighbors.h \
				  src/shared_mem/nonbonded.c \
				  src/shared_mem/nonbonded.h \
				  src/shared_mem/random.c \
				  src/shared_mem/random.h \
				  src/shared_mem/reax_types.h \
				  src/shared_mem/reset_tools.c \
				  src/shared_mem/reset_tools.h \
				  src/shared_mem/restart.c \
				  src/shared_mem/restart.h \
				  src/shared_mem/puremd.c \
				  src/shared_mem/puremd.h \
				  src/shared_mem/system_props.c \
				  src/shared_mem/system_props.h \
				  src/shared_mem/torsion_angles.c \
				  src/shared_mem/torsion_angles.h \
				  src/shared_mem/tool_box.c \
				  src/shared_mem/tool_box.h \
				  src/shared_mem/traj.c \
				  src/shared_mem/traj.h \
				  src/shared_mem/valence_angles.c \
				  src/shared_mem/valence_angles.h \
				  src/shared_mem/vector.c \
				  src/shared_mem/vector.h
lib_libpuremd_openmp_la_CFLAGS = @DFT_CFLAGS@ @CFLAGS_EXTRA@ @OMP_CFLAGS@
lib_libpuremd_openmp_la_CPPFLAGS = -I src/shared_mem @CPPFLAGS_EXTRA@
lib_libpuremd_openmp_la_LDFLAGS = -version-info 1:0:0
lib_libpuremd_openmp_la_LIBADD = @OMP_LIBS@ @Z_LIBS@ @L_LIBS@ @T_LIBS@ @G_LIBS@

bin_PROGRAMS += bin/puremd-openmp
bin_puremd_openmp_SOURCES = src/shared_mem/driver.c
bin_puremd_openmp_CFLAGS = @DFT_CFLAGS@ @CFLAGS_EXTRA@ @OMP_CFLAGS@
bin_puremd_openmp_CPPFLAGS = -I src/shared_mem @CPPFLAGS_EXTRA@
bin_puremd_openmp_LDADD = lib/libpuremd-openmp.la @OMP_LIBS@ @Z_LIBS@ @L_LIBS@ @T_LIBS@ @G_LIBS@

if BUILD_TEST
check_PROGRAMS += test/shared_mem/test_puremd-openmp
check_PROGRAMS += test/shared_mem/test_lin_alg-openmp
check_PROGRAMS += test/shared_mem/test_vector-openmp
TESTS += $(check_PROGRAMS)

test_shared_mem_test_puremd_openmp_SOURCES = test/shared_mem/test_puremd.cpp
test_shared_mem_test_puremd_openmp_CPPFLAGS = -I src/shared_mem $(GTEST_CPPFLAGS)
test_shared_mem_test_puremd_openmp_CXXFLAGS = $(GTEST_CXXFLAGS)
test_shared_mem_test_puremd_openmp_LDFLAGS = $(GTEST_LDFLAGS) $(GTEST_LIBS)
test_shared_mem_test_puremd_openmp_LDADD = lib/libpuremd-openmp.la -lgtest

test_shared_mem_test_lin_alg_openmp_SOURCES = test/shared_mem/test_lin_alg.cpp
test_shared_mem_test_lin_alg_openmp_CPPFLAGS = -I src/shared_mem $(GTEST_CPPFLAGS)
test_shared_mem_test_lin_alg_openmp_CXXFLAGS = $(GTEST_CXXFLAGS)
test_shared_mem_test_lin_alg_openmp_LDFLAGS = $(GTEST_LDFLAGS) $(GTEST_LIBS)
test_shared_mem_test_lin_alg_openmp_LDADD = -lgtest

test_shared_mem_test_vector_openmp_SOURCES = test/shared_mem/test_vector.cpp
test_shared_mem_test_vector_openmp_CPPFLAGS = -I src/shared_mem $(GTEST_CPPFLAGS)
test_shared_mem_test_vector_openmp_CXXFLAGS = $(GTEST_CXXFLAGS)
test_shared_mem_test_vector_openmp_LDFLAGS = $(GTEST_LDFLAGS) $(GTEST_LIBS)
test_shared_mem_test_vector_openmp_LDADD = -lgtest
endif
endif


if BUILD_MPI
bin_PROGRAMS += bin/puremd-mpi
bin_puremd_mpi_SOURCES = src/dist_mem/allocate.c \
			 src/dist_mem/allocate.h \
			 src/dist_mem/analyze.c \
			 src/dist_mem/analyze.h \
			 src/dist_mem/basic_comm.c \
			 src/dist_mem/basic_comm.h \
			 src/dist_mem/bond_orders.c \
			 src/dist_mem/bond_orders.h \
			 src/dist_mem/bonds.c \
			 src/dist_mem/bonds.h \
			 src/dist_mem/box.c \
			 src/dist_mem/box.h \
			 src/dist_mem/charges.c \
			 src/dist_mem/charges.h \
			 src/dist_mem/comm_tools.c \
			 src/dist_mem/comm_tools.h \
			 src/dist_mem/control.c \
			 src/dist_mem/control.h \
			 src/dist_mem/driver.c \
			 src/dist_mem/ffield.c \
			 src/dist_mem/ffield.h \
			 src/dist_mem/forces.c \
			 src/dist_mem/forces.h \
			 src/dist_mem/geo_tools.c \
			 src/dist_mem/geo_tools.h \
			 src/dist_mem/grid.c \
			 src/dist_mem/grid.h \
			 src/dist_mem/hydrogen_bonds.c \
			 src/dist_mem/hydrogen_bonds.h \
			 src/dist_mem/index_utils.h \
			 src/dist_mem/init_md.c \
			 src/dist_mem/init_md.h \
			 src/dist_mem/integrate.c \
			 src/dist_mem/integrate.h \
			 src/dist_mem/io_tools.c \
			 src/dist_mem/io_tools.h \
			 src/dist_mem/lin_alg.c \
			 src/dist_mem/lin_alg.h \
			 src/dist_mem/list.c \
			 src/dist_mem/list.h \
			 src/dist_mem/lookup.c \
			 src/dist_mem/lookup.h \
			 src/dist_mem/multi_body.c \
			 src/dist_mem/multi_body.h \
			 src/dist_mem/neighbors.c \
			 src/dist_mem/neighbors.h \
			 src/dist_mem/nonbonded.c \
			 src/dist_mem/nonbonded.h \
			 src/dist_mem/puremd.c \
			 src/dist_mem/puremd.h \
			 src/dist_mem/random.c \
			 src/dist_mem/random.h \
			 src/dist_mem/reax_types.h \
			 src/dist_mem/reset_tools.c \
			 src/dist_mem/reset_tools.h \
			 src/dist_mem/restart.c \
			 src/dist_mem/restart.h \
			 src/dist_mem/system_props.c \
			 src/dist_mem/system_props.h \
			 src/dist_mem/tool_box.c \
			 src/dist_mem/tool_box.h \
			 src/dist_mem/torsion_angles.c \
			 src/dist_mem/torsion_angles.h \
			 src/dist_mem/traj.c \
			 src/dist_mem/traj.h \
			 src/dist_mem/valence_angles.c \
			 src/dist_mem/valence_angles.h \
			 src/dist_mem/vector.h

bin_puremd_mpi_CFLAGS = -std=c11 @CFLAGS_EXTRA@ @OMP_CFLAGS@ @M_CFLAGS@
bin_puremd_mpi_CPPFLAGS = -I src/dist_mem @CPPFLAGS_EXTRA@
bin_puremd_mpi_LDADD = @OMP_LIBS@ @M_LIBS@ @L_LIBS@ -lstdc++
endif


if BUILD_MPI_CUDA
EXTRA_DIST += src/dist_mem/cub

SUFFIXES = .cu
include cuda.am

AM_NVCCFLAGS = --compiler-options="$(DEFS) $(AM_CFLAGS) -std=c++14 @CFLAGS_EXTRA@ $(CFLAGS) @OMP_CFLAGS@ @M_CFLAGS@" -std=c++14 @NFLAGS@ @NFLAGS_EXTRA@

bin_PROGRAMS += bin/puremd-mpi-cuda
bin_puremd_mpi_cuda_SOURCES = src/dist_mem/allocate.c \
			      src/dist_mem/allocate.h \
			      src/dist_mem/analyze.c \
			      src/dist_mem/analyze.h \
			      src/dist_mem/basic_comm.c \
			      src/dist_mem/basic_comm.h \
			      src/dist_mem/bond_orders.c \
			      src/dist_mem/bond_orders.h \
			      src/dist_mem/bonds.c \
			      src/dist_mem/bonds.h \
			      src/dist_mem/box.c \
			      src/dist_mem/box.h \
			      src/dist_mem/charges.c \
			      src/dist_mem/charges.h \
			      src/dist_mem/comm_tools.c \
			      src/dist_mem/comm_tools.h \
			      src/dist_mem/control.c \
			      src/dist_mem/control.h \
			      src/dist_mem/driver.c \
			      src/dist_mem/ffield.c \
			      src/dist_mem/ffield.h \
			      src/dist_mem/forces.c \
			      src/dist_mem/forces.h \
			      src/dist_mem/geo_tools.c \
			      src/dist_mem/geo_tools.h \
			      src/dist_mem/grid.c \
			      src/dist_mem/grid.h \
			      src/dist_mem/hydrogen_bonds.c \
			      src/dist_mem/hydrogen_bonds.h \
			      src/dist_mem/index_utils.h \
			      src/dist_mem/init_md.c \
			      src/dist_mem/init_md.h \
			      src/dist_mem/integrate.c \
			      src/dist_mem/integrate.h \
			      src/dist_mem/io_tools.c \
			      src/dist_mem/io_tools.h \
			      src/dist_mem/lin_alg.c \
			      src/dist_mem/lin_alg.h \
			      src/dist_mem/list.c \
			      src/dist_mem/list.h \
			      src/dist_mem/lookup.c \
			      src/dist_mem/lookup.h \
			      src/dist_mem/multi_body.c \
			      src/dist_mem/multi_body.h \
			      src/dist_mem/neighbors.c \
			      src/dist_mem/neighbors.h \
			      src/dist_mem/nonbonded.c \
			      src/dist_mem/nonbonded.h \
			      src/dist_mem/puremd.c \
			      src/dist_mem/puremd.h \
			      src/dist_mem/random.c \
			      src/dist_mem/random.h \
			      src/dist_mem/reax_types.h \
			      src/dist_mem/reset_tools.c \
			      src/dist_mem/reset_tools.h \
			      src/dist_mem/restart.c \
			      src/dist_mem/restart.h \
			      src/dist_mem/system_props.c \
			      src/dist_mem/system_props.h \
			      src/dist_mem/tool_box.c \
			      src/dist_mem/tool_box.h \
			      src/dist_mem/torsion_angles.c \
			      src/dist_mem/torsion_angles.h \
			      src/dist_mem/traj.c \
			      src/dist_mem/traj.h \
			      src/dist_mem/valence_angles.c \
			      src/dist_mem/valence_angles.h \
			      src/dist_mem/vector.h \
			      src/dist_mem/cuda/gpu_allocate.cu \
			      src/dist_mem/cuda/gpu_allocate.h \
			      src/dist_mem/cuda/gpu_basic_comm.cu \
			      src/dist_mem/cuda/gpu_basic_comm.h \
			      src/dist_mem/cuda/gpu_bond_orders.cu \
			      src/dist_mem/cuda/gpu_bond_orders.h \
			      src/dist_mem/cuda/gpu_bonds.cu \
			      src/dist_mem/cuda/gpu_bonds.h \
			      src/dist_mem/cuda/gpu_box.cu \
			      src/dist_mem/cuda/gpu_box.h \
			      src/dist_mem/cuda/gpu_charges.cu \
			      src/dist_mem/cuda/gpu_charges.h \
			      src/dist_mem/cuda/gpu_copy.cu \
			      src/dist_mem/cuda/gpu_copy.h \
			      src/dist_mem/cuda/gpu_dense_lin_alg.cu \
			      src/dist_mem/cuda/gpu_dense_lin_alg.h \
			      src/dist_mem/cuda/gpu_environment.cu \
			      src/dist_mem/cuda/gpu_environment.h \
			      src/dist_mem/cuda/gpu_forces.cu \
			      src/dist_mem/cuda/gpu_forces.h \
			      src/dist_mem/cuda/gpu_helpers.h \
			      src/dist_mem/cuda/gpu_hydrogen_bonds.cu \
			      src/dist_mem/cuda/gpu_hydrogen_bonds.h \
			      src/dist_mem/cuda/gpu_init_md.cu \
			      src/dist_mem/cuda/gpu_init_md.h \
			      src/dist_mem/cuda/gpu_integrate.cu \
			      src/dist_mem/cuda/gpu_integrate.h \
			      src/dist_mem/cuda/gpu_list.cu \
			      src/dist_mem/cuda/gpu_list.h \
			      src/dist_mem/cuda/gpu_lookup.cu \
			      src/dist_mem/cuda/gpu_lookup.h \
			      src/dist_mem/cuda/gpu_multi_body.cu \
			      src/dist_mem/cuda/gpu_multi_body.h \
			      src/dist_mem/cuda/gpu_neighbors.cu \
			      src/dist_mem/cuda/gpu_neighbors.h \
			      src/dist_mem/cuda/gpu_nonbonded.cu \
			      src/dist_mem/cuda/gpu_nonbonded.h \
			      src/dist_mem/cuda/gpu_post_evolve.cu \
			      src/dist_mem/cuda/gpu_post_evolve.h \
			      src/dist_mem/cuda/gpu_random.cu \
			      src/dist_mem/cuda/gpu_random.h \
			      src/dist_mem/cuda/gpu_reduction.cu \
			      src/dist_mem/cuda/gpu_reduction.h \
			      src/dist_mem/cuda/gpu_reset_tools.cu \
			      src/dist_mem/cuda/gpu_reset_tools.h \
			      src/dist_mem/cuda/gpu_spar_lin_alg.cu \
			      src/dist_mem/cuda/gpu_spar_lin_alg.h \
			      src/dist_mem/cuda/gpu_system_props.cu \
			      src/dist_mem/cuda/gpu_system_props.h \
			      src/dist_mem/cuda/gpu_torsion_angles.cu \
			      src/dist_mem/cuda/gpu_torsion_angles.h \
			      src/dist_mem/cuda/gpu_utils.cu \
			      src/dist_mem/cuda/gpu_utils.h \
			      src/dist_mem/cuda/gpu_valence_angles.cu \
			      src/dist_mem/cuda/gpu_valence_angles.h \
			      src/dist_mem/cuda/gpu_vector.h
# dummy source to trigger C linking
nodist_EXTRA_bin_puremd_mpi_cuda_SOURCES = src/dist_mem/dummy.c

bin_puremd_mpi_cuda_CFLAGS = -std=c11 @CFLAGS_EXTRA@ @OMP_CFLAGS@ @M_CFLAGS@ @CUDA_CFLAGS@
bin_puremd_mpi_cuda_CPPFLAGS = -I src/dist_mem @CPPFLAGS_EXTRA@
bin_puremd_mpi_cuda_LDADD = @OMP_LIBS@ @M_LIBS@ @L_LIBS@ -lstdc++ @CUDA_LIBS@
endif


if BUILD_MPI_HIP
SUFFIXES = .cu
include hip.am

AM_HIPCCFLAGS = $(DEFS) $(AM_CFLAGS) -std=c++14 @CFLAGS_EXTRA@ $(CFLAGS) @OMP_CFLAGS@ @M_CFLAGS@


bin_PROGRAMS += bin/puremd-mpi-hip
bin_puremd_mpi_hip_SOURCES = src/dist_mem/allocate.c \
			     src/dist_mem/allocate.h \
			     src/dist_mem/analyze.c \
			     src/dist_mem/analyze.h \
			     src/dist_mem/basic_comm.c \
			     src/dist_mem/basic_comm.h \
			     src/dist_mem/bond_orders.c \
			     src/dist_mem/bond_orders.h \
			     src/dist_mem/bonds.c \
			     src/dist_mem/bonds.h \
			     src/dist_mem/box.c \
			     src/dist_mem/box.h \
			     src/dist_mem/charges.c \
			     src/dist_mem/charges.h \
			     src/dist_mem/comm_tools.c \
			     src/dist_mem/comm_tools.h \
			     src/dist_mem/control.c \
			     src/dist_mem/control.h \
			     src/dist_mem/driver.c \
			     src/dist_mem/ffield.c \
			     src/dist_mem/ffield.h \
			     src/dist_mem/forces.c \
			     src/dist_mem/forces.h \
			     src/dist_mem/geo_tools.c \
			     src/dist_mem/geo_tools.h \
			     src/dist_mem/grid.c \
			     src/dist_mem/grid.h \
			     src/dist_mem/hydrogen_bonds.c \
			     src/dist_mem/hydrogen_bonds.h \
			     src/dist_mem/index_utils.h \
			     src/dist_mem/init_md.c \
			     src/dist_mem/init_md.h \
			     src/dist_mem/integrate.c \
			     src/dist_mem/integrate.h \
			     src/dist_mem/io_tools.c \
			     src/dist_mem/io_tools.h \
			     src/dist_mem/lin_alg.c \
			     src/dist_mem/lin_alg.h \
			     src/dist_mem/list.c \
			     src/dist_mem/list.h \
			     src/dist_mem/lookup.c \
			     src/dist_mem/lookup.h \
			     src/dist_mem/multi_body.c \
			     src/dist_mem/multi_body.h \
			     src/dist_mem/neighbors.c \
			     src/dist_mem/neighbors.h \
			     src/dist_mem/nonbonded.c \
			     src/dist_mem/nonbonded.h \
			     src/dist_mem/puremd.c \
			     src/dist_mem/puremd.h \
			     src/dist_mem/random.c \
			     src/dist_mem/random.h \
			     src/dist_mem/reax_types.h \
			     src/dist_mem/reset_tools.c \
			     src/dist_mem/reset_tools.h \
			     src/dist_mem/restart.c \
			     src/dist_mem/restart.h \
			     src/dist_mem/system_props.c \
			     src/dist_mem/system_props.h \
			     src/dist_mem/tool_box.c \
			     src/dist_mem/tool_box.h \
			     src/dist_mem/torsion_angles.c \
			     src/dist_mem/torsion_angles.h \
			     src/dist_mem/traj.c \
			     src/dist_mem/traj.h \
			     src/dist_mem/valence_angles.c \
			     src/dist_mem/valence_angles.h \
			     src/dist_mem/vector.h \
			     src/dist_mem/hip/gpu_allocate.cu \
			     src/dist_mem/hip/gpu_allocate.h \
			     src/dist_mem/hip/gpu_basic_comm.cu \
			     src/dist_mem/hip/gpu_basic_comm.h \
			     src/dist_mem/hip/gpu_bond_orders.cu \
			     src/dist_mem/hip/gpu_bond_orders.h \
			     src/dist_mem/hip/gpu_bonds.cu \
			     src/dist_mem/hip/gpu_bonds.h \
			     src/dist_mem/hip/gpu_box.cu \
			     src/dist_mem/hip/gpu_box.h \
			     src/dist_mem/hip/gpu_charges.cu \
			     src/dist_mem/hip/gpu_charges.h \
			     src/dist_mem/hip/gpu_copy.cu \
			     src/dist_mem/hip/gpu_copy.h \
			     src/dist_mem/hip/gpu_dense_lin_alg.cu \
			     src/dist_mem/hip/gpu_dense_lin_alg.h \
			     src/dist_mem/hip/gpu_environment.cu \
			     src/dist_mem/hip/gpu_environment.h \
			     src/dist_mem/hip/gpu_forces.cu \
			     src/dist_mem/hip/gpu_forces.h \
			     src/dist_mem/hip/gpu_helpers.h \
			     src/dist_mem/hip/gpu_hydrogen_bonds.cu \
			     src/dist_mem/hip/gpu_hydrogen_bonds.h \
			     src/dist_mem/hip/gpu_init_md.cu \
			     src/dist_mem/hip/gpu_init_md.h \
			     src/dist_mem/hip/gpu_integrate.cu \
			     src/dist_mem/hip/gpu_integrate.h \
			     src/dist_mem/hip/gpu_list.cu \
			     src/dist_mem/hip/gpu_list.h \
			     src/dist_mem/hip/gpu_lookup.cu \
			     src/dist_mem/hip/gpu_lookup.h \
			     src/dist_mem/hip/gpu_multi_body.cu \
			     src/dist_mem/hip/gpu_multi_body.h \
			     src/dist_mem/hip/gpu_neighbors.cu \
			     src/dist_mem/hip/gpu_neighbors.h \
			     src/dist_mem/hip/gpu_nonbonded.cu \
			     src/dist_mem/hip/gpu_nonbonded.h \
			     src/dist_mem/hip/gpu_post_evolve.cu \
			     src/dist_mem/hip/gpu_post_evolve.h \
			     src/dist_mem/hip/gpu_random.cu \
			     src/dist_mem/hip/gpu_random.h \
			     src/dist_mem/hip/gpu_reduction.cu \
			     src/dist_mem/hip/gpu_reduction.h \
			     src/dist_mem/hip/gpu_reset_tools.cu \
			     src/dist_mem/hip/gpu_reset_tools.h \
			     src/dist_mem/hip/gpu_spar_lin_alg.cu \
			     src/dist_mem/hip/gpu_spar_lin_alg.h \
			     src/dist_mem/hip/gpu_system_props.cu \
			     src/dist_mem/hip/gpu_system_props.h \
			     src/dist_mem/hip/gpu_torsion_angles.cu \
			     src/dist_mem/hip/gpu_torsion_angles.h \
			     src/dist_mem/hip/gpu_utils.cu \
			     src/dist_mem/hip/gpu_utils.h \
			     src/dist_mem/hip/gpu_valence_angles.cu \
			     src/dist_mem/hip/gpu_valence_angles.h \
			     src/dist_mem/hip/gpu_vector.h
# dummy source to trigger C linking
nodist_EXTRA_bin_puremd_mpi_hip_SOURCES = src/dist_mem/dummy.c

bin_puremd_mpi_hip_CFLAGS = -std=c11 @CFLAGS_EXTRA@ @OMP_CFLAGS@ @M_CFLAGS@ @HIP_CFLAGS@
bin_puremd_mpi_hip_CPPFLAGS = -I src/dist_mem @CPPFLAGS_EXTRA@
bin_puremd_mpi_hip_LDADD = @OMP_LIBS@ @M_LIBS@ @L_LIBS@ -lstdc++ @HIP_LIBS@
endif


if BUILD_DOC
SUBDIRS += doc
DIST_SUBDIRS += doc
endif


dist-hook:
	rm -rf `find $(distdir) -name .git`
