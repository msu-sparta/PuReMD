#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])

AC_INIT([PG-PuReMD], [1.0], [ohearnku@msu.edu hma@msu.edu])
# Do not allow AC_PROG_CC to set CFLAGS (this line must be after AC_INIT but before AC_PROG_CC)
sav_CFLAGS="$CFLAGS"
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
AC_PROG_CC([icc gcc clang cc])
AC_PROG_CPP
CFLAGS="$sav_CFLAGS"

AX_COMPILER_VENDOR

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
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([gettimeofday memset])

# Check for MPI support.
CONFIGURE_HEADLINE([ MPI compiler ])
ACX_MPI([], [AC_MSG_ERROR([could not find MPI library])])
AC_CHECK_PROG(MPIRUN, mpirun, mpirun)
AC_SUBST(MPIRUN)

# try to find if we are using OpenMPI / MPICH by looking inside mpi.h
sav_CC="$CC"
sav_CFLAGS="$CFLAGS"
CC="$MPICC"
CFLAGS="$CFLAGS"
AC_CHECK_DECL([OPEN_MPI], [mpi_vendor="OpenMPI"],
	     [], [#include "mpi.h"])
# MPICH v2
AC_CHECK_DECL([MPICH2], [mpi_vendor="MPICH2"],
	     [], [#include "mpi.h"])
# MPICH v3
AC_CHECK_DECL([MPICH_VERSION], [mpi_vendor="MPICH3"],
	     [], [#include "mpi.h"])
CC="$sav_CC"
CFLAGS="$sav_CFLAGS"

# try to set MPI_CFLAGS and MPI_LIBS
MPI_CFLAGS=
MPI_LIBS=
if test "$mpi_vendor" = "OpenMPI"
then 
	MPI_CFLAGS=`$MPICC --showme:compile`
	MPI_LIBS=`$MPICC --showme:link`
	AC_MSG_NOTICE([OpenMPI found])
	AC_MSG_NOTICE([MPI_CFLAGS=$MPI_CFLAGS])
	AC_MSG_NOTICE([MPI_LIBS=$MPI_LIBS])
elif test "$mpi_vendor" = "MPICH2" || test "$mpi_vendor" = "MPICH3"
then
	# build MPI_CFLAGS
	tmp=`$MPICC -compile-info | awk '{$1=""; print $0 }'`
	MPI_CFLAGS=
	for i in $tmp
	do 
		case $i in 
			-[[DIUbi]]*)
				MPI_CFLAGS="$MPI_CFLAGS $i"
				;;
		esac
	done
	# build MPI_LIBS
	tmp=`$MPICC -link-info | awk '{$1=""; print $0 }'`
	for i in $tmp
	do 
		case $i in 
			[[\\/]]*.a | ?:[[\\/]]*.a | -[[lLRu]]* | -Wl* )
				MPI_LIBS="$MPI_LIBS $i"
				;;
		esac
	done
	AC_MSG_NOTICE([MPICH found])
	AC_MSG_NOTICE([MPI_CFLAGS=$MPI_CFLAGS])
	AC_MSG_NOTICE([MPI_LIBS=$MPI_LIBS])
else
	AC_MSG_WARN([Neither OpenMPI and MPICH have been recognized...])
fi
AC_SUBST(MPI_CFLAGS)
AC_SUBST(MPI_LIBS)

AC_LANG([C++])

# Checks for programs.
AC_PROG_CXX([icpc g++ clang++ c++])
AC_PROG_CXXCPP

AX_COMPILER_VENDOR

# Check for CUDA support.
if test "x$BUILD_GPU" = "xyes"; then
	CONFIGURE_HEADLINE([ CUDA support ])
	AX_CUDA

        NVCCFLAGS=
	NVCCFLAGS+=" -ccbin=$CXX"
	if test "BUILD_DEBUG" = "true"
	then
		NVCCFLAGS+=" -g -G"
	fi
	AC_DEFINE([HAVE_CUDA], [1], [Define to 1 if you have CUDA support enabled.])
else
	AM_CONDITIONAL(USE_CUDA, test "x" = "xyes")
fi

if test "BUILD_PROF" = "true"
then
	if test "x$BUILD_GPU" = "xyes"; then
		NVCCFLAGS+=" --compiler-options ${gprof_flags}"
	else
		CFLAGS+=" ${gprof_flags}"
	fi
fi

AC_CONFIG_FILES([Makefile])

AC_OUTPUT
