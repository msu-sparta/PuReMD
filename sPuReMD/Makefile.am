ACLOCAL_AMFLAGS = -I ../m4

lib_LTLIBRARIES = lib/libspuremd.la
lib_libspuremd_la_SOURCES = src/ffield.c src/grid.c src/list.c src/lookup.c src/print_utils.c \
		  src/reset_utils.c src/restart.c src/random.c src/tool_box.c src/traj.c \
		  src/vector.c src/allocate.c src/analyze.c src/box.c src/system_props.c src/control.c \
		  src/geo_tools.c src/neighbors.c src/lin_alg.c src/charges.c src/bond_orders.c \
		  src/single_body_interactions.c src/two_body_interactions.c \
		  src/three_body_interactions.c src/four_body_interactions.c src/forces.c \
		  src/integrate.c src/init_md.c src/spuremd.c
lib_libspuremd_la_CFLAGS = -I src
lib_libspuremd_la_LDFLAGS = -version-info 1:0:0

bin_PROGRAMS = bin/spuremd
bin_spuremd_SOURCES = src/driver.c
bin_spuremd_CPPFLAGS = -I src
bin_spuremd_LDADD = lib/libspuremd.la

check_PROGRAMS =
TESTS =

if BUILD_TEST
check_PROGRAMS += tests/test_spuremd
check_PROGRAMS += tests/test_lin_alg
check_PROGRAMS += tests/test_vector
TESTS += $(check_PROGRAMS)

tests_test_spuremd_SOURCES = tests/test_spuremd.cpp
tests_test_spuremd_CPPFLAGS = -Isrc $(GTEST_CPPFLAGS)
tests_test_spuremd_CXXFLAGS = $(GTEST_CXXFLAGS)
tests_test_spuremd_LDFLAGS = $(GTEST_LDFLAGS) $(GTEST_LIBS)
tests_test_spuremd_LDADD = lib/libspuremd.la -lgtest

tests_test_lin_alg_SOURCES = tests/test_lin_alg.cpp
tests_test_lin_alg_CPPFLAGS = -Isrc $(GTEST_CPPFLAGS)
tests_test_lin_alg_CXXFLAGS = $(GTEST_CXXFLAGS)
tests_test_lin_alg_LDFLAGS = $(GTEST_LDFLAGS) $(GTEST_LIBS)
tests_test_lin_alg_LDADD = -lgtest

tests_test_vector_SOURCES = tests/test_vector.cpp
tests_test_vector_CPPFLAGS = -Isrc $(GTEST_CPPFLAGS)
tests_test_vector_CXXFLAGS = $(GTEST_CXXFLAGS)
tests_test_vector_LDFLAGS = $(GTEST_LDFLAGS) $(GTEST_LIBS)
tests_test_vector_LDADD = -lgtest
endif
