#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])

AC_INIT([PuReMD-GPU], [1.0], [ohearnku@msu.edu hma@msu.edu])
# Do not allow AC_PROG_CC to set CFLAGS (this line must be after AC_INIT but before AC_PROG_CC)
save_CFLAGS="${CFLAGS}"
: ${CFLAGS=""}
AM_INIT_AUTOMAKE([1.15 subdir-objects -Wall -Werror foreign])
# Enable silent build rules by default.
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])], [AC_SUBST([AM_DEFAULT_VERBOSITY],[1])])
#LT_PREREQ([2.2])
#LT_INIT([dlopen])

AC_CONFIG_MACRO_DIR([../m4])

# Headline formatter
AC_DEFUN([CONFIGURE_HEADLINE],
[
        echo; echo "+++ $1 +++"
])

AC_LANG([C])

AC_CONFIG_SRCDIR([src/analyze.c])
AC_CONFIG_HEADERS([src/config.h])

# Checks for programs.
AC_PROG_CC([icc gcc cc])
AC_PROG_CPP
CFLAGS="${save_CFLAGS}"

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
AC_SEARCH_LIBS([gzclose, [z]])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T
AC_CHECK_TYPES([gzFile], [], 
	       [AC_MSG_FAILURE([gzFile type not found in zlib.h], [1])], [#include<zlib.h>])

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([gettimeofday memset pow sqrt])

# Check for compiler vendor. If the compiler is recognized,
#   the variable ax_cv_c_compiler_vendor is set accordingly.
AX_COMPILER_VENDOR
if test "x${ax_cv_c_compiler_vendor}" = "xgnu"; then
	if test "x${BUILD_DEBUG}" = "xyes"; then
		CFLAGS="-Wall -O0 -g2 -D_GLIBCXX_DEBUG"
	else
		CFLAGS="-Wall -O2 -funroll-loops"
	fi
	if test "x${BUILD_GPROF}" = "xyes"; then
		CFLAGS="${CFLAGS} -pg"
	fi
elif test "x${ax_cv_c_compiler_vendor}" = "xintel"; then
	if test "x${BUILD_DEBUG}" = "xyes"; then
		CFLAGS="-Wall -O0"
	else
		CFLAGS="-Wall -O2 -funroll-loops"
	fi
	if test "x${BUILD_GPROF}" = "xyes"; then
		CFLAGS="${CFLAGS} -p"
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
		AC_SUBST(CFLAGS, "${OPENMP_CFLAGS}")
		AC_SUBST(CPPFLAGS, "${OPENMP_CFLAGS}")
		LIBS="${LIBS} -lgomp"
	fi
fi

# Check for CUDA support.
if test "x${BUILD_GPU}" = "xyes"
then
	CONFIGURE_HEADLINE([ CUDA support ])
	AX_CUDA
	NVCCFLAGS=
	if test "x${BUILD_DEBUG}" = "xyes"
	then
		NVCCFLAGS+=" -g -G"
	fi
	AC_DEFINE([HAVE_CUDA], [1], [Define to 1 if you have CUDA support enabled.])

	AC_CHECK_LIB([cuda], [cuGetErrorString])
	AC_CHECK_LIB([cudart], [cudaMalloc])
	AC_CHECK_LIB([cublas], [cublasDnrm2])
	AC_CHECK_LIB([cusparse], [cusparseCreateMatDescr])
#	AC_SEARCH_LIBS([cublasDaxpy], [cublas])
#	AC_SEARCH_LIBS([cublasDscal], [cublas])
#	AC_SEARCH_LIBS([cublasDdot], [cublas])
#	AC_SEARCH_LIBS([cudaThreadSynchronize], [cudart])
#	AC_SEARCH_LIBS([cudaGetLastError], [cudart])
#	AC_SEARCH_LIBS([cusparseCreateMatDescr], [cusparse])
#	AC_SEARCH_LIBS([cusparseSetMatType], [cusparse])
#	AC_SEARCH_LIBS([cusparseSetMatIndexBase], [cusparse])
#
#	AC_SEARCH_LIBS([cublasDnrm2], [cublas],
#		[CUBLAS_FOUND_LIBS="yes"], [CUBLAS_FOUND_LIBS="no"], [-lcublas])
#	AS_IF([test "x${CUBLAS_FOUND_LIBS}" != "xyes"],
#		[AC_MSG_ERROR([Unable to find CUBLAS library.])])
#
#	AC_SEARCH_LIBS([cusparseSetMatType], [cusparse],
#		[CUSPARSE_FOUND_LIBS="yes"], [CUSPARSE_FOUND_LIBS="no"], [-lcusparse])
#	AS_IF([test "x${CUSPARSE_FOUND_LIBS}" != "xyes"],
#		[AC_MSG_ERROR([Unable to find CUSPARSE library.])])
#
#	AC_CHECK_TYPES([cublasHandle_t], [], 
#		       [AC_MSG_FAILURE([cublasHandle_t type not found in cublas.h], [1])], [#include<cublas_v2.h>])
#	AC_CHECK_TYPES([cusparseHandle_t], [], 
#		       [AC_MSG_FAILURE([cusparseHandle_t type not found in cusparse.h], [1])], [#include<cusparse_v2.h>])
#	AC_CHECK_TYPES([cusparseMatDescr_t], [], 
#		       [AC_MSG_FAILURE([cusparseMatDescr_t type not found in cusparse.h], [1])], [#include<cusparse_v2.h>])
else
	AM_CONDITIONAL(USE_CUDA, test "x" = "xyes")
fi

if test "x${BUILD_GPROF}" = "xtrue"
then
	NVCCFLAGS+=" --compiler-options ${gprof_flags}"
fi


AC_CONFIG_FILES([Makefile])

AC_OUTPUT
