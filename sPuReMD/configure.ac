#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])

AC_INIT([sPuReMD], [1.0], [ohearnku@msu.edu hma@msu.edu])
: ${CFLAGS=""}
AM_INIT_AUTOMAKE([1.15 subdir-objects -Wall -Werror foreign])
# Enable silent build rules by default.
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])], [AC_SUBST([AM_DEFAULT_VERBOSITY],[1])])
AM_PROG_AR
LT_PREREQ([2.2])
LT_INIT([dlopen])

AC_LANG([C])

AC_CONFIG_SRCDIR([src/traj.h])
AC_CONFIG_HEADERS([src/config.h])

# Checks for programs.
AC_PROG_CC([icc gcc cc])
AC_PROG_CPP

# Checks for libraries.
AC_SEARCH_LIBS([exp], [m])
AC_SEARCH_LIBS([sqrt], [m])
AC_SEARCH_LIBS([pow], [m])
AC_SEARCH_LIBS([acos], [m])
AC_SEARCH_LIBS([cos], [m])
AC_SEARCH_LIBS([sin], [m])
AC_SEARCH_LIBS([tan], [m])
AC_SEARCH_LIBS([fabs], [m])
AC_SEARCH_LIBS([gzopen], [z])
AC_SEARCH_LIBS([gzeof], [z])
AC_SEARCH_LIBS([gzgets], [z])
AC_SEARCH_LIBS([gzseek], [z])
AC_SEARCH_LIBS([gzclose], [z])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE
AC_CHECK_TYPES([gzFile], [], [AC_MSG_FAILURE([gzFile type not found in zlib.h], [1])], [#include<zlib.h>])

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([gettimeofday memset])

# Check for compiler vendor
AX_COMPILER_VENDOR
if test "x$ax_cv_c_compiler_vendor" = "xgnu"; then
	if test "x$BUILD_DEBUG" = "x"; then
		CFLAGS="$CFLAGS -Wall -O3 -funroll-loops -fstrict-aliasing"
	else
		CFLAGS="$CFLAGS -Wall -O0 -g"
	fi
fi
if test "x$ax_cv_c_compiler_vendor" = "xintel"; then
	if test "x$BUILD_DEBUG" = "x"; then
		CFLAGS="$CFLAGS -fast"
	fi
fi

# Check for OpenMP support.
if test "x${BUILD_OPENMP}" = "xyes"; then
	AC_OPENMP
	if test "x${OPENMP_CFLAGS}" = "x"; then
		AC_MSG_WARN([
  -----------------------------------------------
   Unable to find OpenMP support on this system.
   Building a single-threaded version.
  -----------------------------------------------])
	else
		# bug due to recent Intel compiler change (?)
		if test "x${ax_cv_c_compiler_vendor}" = "xintel"; then
			OPENMP_CFLAGS="-qopenmp"
		fi
		AC_SUBST(AM_CFLAGS, "$OPENMP_CFLAGS")
		AC_SUBST(AM_CPPFLAGS, "$OPENMP_CFLAGS")
		LIBS="${LIBS} -lgomp"
	fi
fi

if test "x$BUILD_DEBUG" != "x"
then
	CFLAGS="${CFLAGS} ${DEBUG_FLAGS}"
fi

if test "x$BUILD_GPROF" != "x"
then
	CFLAGS="${CFLAGS} ${GPROF_FLAGS}"
fi

# Check for LAPACKE
AC_CHECK_HEADERS([mkl.h], [MKL_FOUND_HEADERS="yes"])
if test "x${MKL_FOUND_HEADERS}" = "xyes"
then
	AC_SEARCH_LIBS([LAPACKE_dgels], [mkl_intel_ilp64],
		       [MKL_FOUND_LIBS="yes"], [MKL_FOUND_LIBS="no"],
		       [-lmkl_sequential -lmkl_core -lpthread -lm -ldl])
	AS_IF([test "x${MKL_FOUND_LIBS}" != "xyes"],
	      [AC_MSG_ERROR([Unable to find MKL LAPACKE library.])])
	LIBS="${LIBS} -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"
	AC_DEFINE([HAVE_LAPACKE_MKL], [1], [Define to 1 if you have MKL LAPACKE support enabled.])
else
	AC_CHECK_HEADERS([lapacke.h], [LAPACKE_FOUND_HEADERS="yes"])
	if test "x${LAPACKE_FOUND_HEADERS}" = "xyes"
	then
		AC_SEARCH_LIBS([LAPACKE_dgels], [lapacke],
			       [LAPACKE_FOUND_LIBS="yes"], [LAPACKE_FOUND_LIBS="no"],
			       [-llapack])
		AS_IF([test "x${LAPACKE_FOUND_LIBS}" != "xyes"],
		      [AC_MSG_ERROR([Unable to find LAPACKE library.])])
		LIBS="${LIBS} -llapack"
		AC_DEFINE([HAVE_LAPACKE], [1], [Define to 1 if you have LAPACKE support enabled.])
	else
		AC_MSG_WARN([
  -----------------------------------------------
   Unable to find LAPACKE on this system.
   Disabling support for dependent methods.
  -----------------------------------------------])
	fi
fi

# Tests using Google C++ testing framework (gtest)
AC_LANG_PUSH([C++])
AC_PROG_CXX([icpc g++ clang++ CC])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
  [[#ifndef __cplusplus
    #error "broken C++"
    #endif]])],,
  [CXX=;])
if test "x${CXX}" != "x"
then
	AC_CHECK_HEADER([gtest/gtest.h], [HAVE_GTEST_H="yes"])
	if test "x${HAVE_GTEST_H}" != "xyes"
	then
		AC_MSG_WARN([gtest.h not found. Unable to build tests. Continuing...])
	else
		ACX_PTHREAD([have_pthread=yes])

		# define gtest variables 
		if test "x${have_pthread}" = "xyes"; then
			GTEST_CPPFLAGS="-DGTEST_HAS_PTHREAD=1"
			GTEST_CXXFLAGS="$PTHREAD_CFLAGS"
			GTEST_LDFLAGS=
			GTEST_LIBS="$PTHREAD_LIBS"
		else
			GTEST_CPPFLAGS="-DGTEST_HAS_PTHREAD=0"
			GTEST_CXXFLAGS=
			GTEST_LDFLAGS=
			GTEST_LIBS=
		fi
		AC_SUBST([GTEST_CPPFLAGS])
		AC_SUBST([GTEST_CXXFLAGS])
		AC_SUBST([GTEST_LDFLAGS])
		AC_SUBST([GTEST_LIBS])
	fi
else
	AC_MSG_WARN([C++ compiler not found. Unable to build tests. Continuing...])
fi
AM_CONDITIONAL([BUILD_TEST], [test "x${HAVE_GTEST_H}" = "xyes"])
AC_LANG_POP([C++])

AC_CONFIG_FILES([Makefile])

AC_OUTPUT
