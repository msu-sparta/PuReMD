#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])

AC_INIT([PuReMD], [1.0], [ohearnku@msu.edu hma@msu.edu])
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

# Checks for programs.
AC_PROG_CC([cc icc gcc])
AC_PROG_CPP
CFLAGS="${save_CFLAGS}"

AC_CONFIG_SRCDIR([src/torsion_angles.h])
AC_CONFIG_HEADERS([src/config.h])

# Checks for libraries.
AC_SEARCH_LIBS([exp], [m])
AC_SEARCH_LIBS([sqrt], [m])
AC_SEARCH_LIBS([pow], [m])
AC_SEARCH_LIBS([acos], [m])
AC_SEARCH_LIBS([cos], [m])
AC_SEARCH_LIBS([sin], [m])
AC_SEARCH_LIBS([tan], [m])
AC_SEARCH_LIBS([fabs], [m])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([gettimeofday memset pow sqrt])

# Check for compiler vendor. If the compiler is recognized,
#   the variable ax_cv_c_compiler_vendor is set accordingly.
AX_COMPILER_VENDOR
if test "x${ax_cv_c_compiler_vendor}" = "xgnu"; then
	if test "x${BUILD_DEBUG}" = "x"; then
		CFLAGS="-Wall -O2 -funroll-loops"
	else
		CFLAGS="-Wall -O0 -g2 -D_GLIBCXX_DEBUG"
	fi
elif test "x${ax_cv_c_compiler_vendor}" = "xintel"; then
	if test "x${BUILD_DEBUG}" = "x"; then
		CFLAGS="-Wall -O2 -funroll-loops"
	else
		CFLAGS="-Wall -O0"
	fi
fi

# Check for MPI support. If found, sets the following variables:
#   Compiler-related: MPICC, MPICXX, MPIF77
#   Library linking: MPILIBS
#   Compilation support (via preprocessor def): HAVE_MPI
CONFIGURE_HEADLINE([ MPI compiler ])
ACX_MPI([], [AC_MSG_ERROR([could not find mpi library])])
AC_CHECK_PROG(MPIRUN, mpirun, mpirun)
AC_SUBST(MPIRUN)

# try to find if we are using OpenMPI / MPICH by looking inside mpi.h
save_CC="${CC}"
save_CFLAGS="${CFLAGS}"
CC="${MPICC}"
CFLAGS=""
AC_CHECK_DECL([OPEN_MPI], [mpi_vendor="OpenMPI"],
	     [], [#include "mpi.h"])
# MPICH v2
AC_CHECK_DECL([MPICH2], [mpi_vendor="MPICH2"],
	     [], [#include "mpi.h"])
# MPICH v3
AC_CHECK_DECL([MPICH_VERSION], [mpi_vendor="MPICH3"],
	     [], [#include "mpi.h"])
CC="${save_CC}"
CFLAGS="${save_CFLAGS}"

# try to set MPI_CFLAGS and MPI_LIBS
MPI_CFLAGS=
MPI_LIBS=
if test "x${mpi_vendor}" = "xOpenMPI"
then 
	MPI_CFLAGS=`${MPICC} --showme:compile`
	MPI_LIBS=`${MPICC} --showme:link`
	AC_MSG_NOTICE([OpenMPI found])
	AC_MSG_NOTICE([MPI_CFLAGS=${MPI_CFLAGS}])
	AC_MSG_NOTICE([MPI_LIBS=${MPI_LIBS}])
elif test "x${mpi_vendor}" = "xMPICH2" || test "x${mpi_vendor}" = "xMPICH3"
then
	# build MPI_CFLAGS
	tmp=`${MPICC} -compile-info | awk '{$1=""; print $0 }'`
	MPI_CFLAGS=
	for i in $tmp
	do 
		case ${i} in 
			-[[DIUbi]]*)
				MPI_CFLAGS="${MPI_CFLAGS} ${i}"
				;;
		esac
	done
	# build MPI_LIBS
	tmp=`${MPICC} -link-info | awk '{$1=""; print $0 }'`
	for i in $tmp
	do 
		case ${i} in 
			[[\\/]]*.a | ?:[[\\/]]*.a | -[[lLRu]]* | -Wl* )
				MPI_LIBS="${MPI_LIBS} ${i}"
				;;
		esac
	done
	AC_MSG_NOTICE([MPICH found])
	AC_MSG_NOTICE([MPI_CFLAGS=${MPI_CFLAGS}])
	AC_MSG_NOTICE([MPI_LIBS=${MPI_LIBS}])
else
	AC_MSG_WARN([Neither OpenMPI and MPICH have been recognized...])
fi
AC_SUBST(MPI_CFLAGS)
AC_SUBST(MPI_LIBS)

# Check for LAPACKE
AC_CHECK_HEADERS([mkl.h], [MKL_FOUND_HEADERS="yes"])
if test "x${MKL_FOUND_HEADERS}" = "xyes"
then
	AC_SEARCH_LIBS([LAPACKE_dgels], [mkl_intel_ilp64],
		       [MKL_FOUND_LIBS="yes"], [MKL_FOUND_LIBS="no"],
		       [-lmkl_sequential -lmkl_core -lpthread -lm -ldl])
	AS_IF([test "x${MKL_FOUND_LIBS}" != "xyes"],
	      [AC_MSG_ERROR([Unable to find MKL LAPACKE library.])])
	# dynamic linking
	#LIBS="${LIBS} -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"
	# static linking
	LIBS="${LIBS} -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl"
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

AC_CONFIG_FILES([Makefile])

AC_OUTPUT
