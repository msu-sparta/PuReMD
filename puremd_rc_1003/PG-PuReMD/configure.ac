#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])

AC_INIT([PG-PuReMD], [1.0], [ohearnku@msu.edu hma@msu.edu])
AM_INIT_AUTOMAKE([1.15 subdir-objects -Wall -Werror foreign])
#LT_PREREQ([2.2])
#LT_INIT([dlopen])

AC_CONFIG_MACRO_DIR([../m4])

AC_LANG([C])

AC_CONFIG_SRCDIR([src/torsion_angles.h])
AC_CONFIG_HEADERS([src/config.h])

# Headline formatter
AC_DEFUN([CONFIGURE_HEADLINE],
[
        echo; echo "+++ $1 +++"
])

# Checks for programs.
AC_PROG_CC
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

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([gettimeofday memset])

# Check for CUDA support.
if test "x$BUILD_GPU" = "xyes"; then
	CONFIGURE_HEADLINE([ CUDA support ])
	AX_CUDA
	if test "BUILD_DEBUG" = "true"
	then
		NVCCFLAGS+=" -g -G"
	else
		NVCCFLAGS+=" -O3"
	fi

	AC_DEFINE([HAVE_CUDA], [1], [Define to 1 if you have CUDA support enabled.])
fi
AM_CONDITIONAL([BUILD_CUDA], [test "x${BUILD_GPU}" = "xyes"])

if test "BUILD_PROF" = "true"
then
	if test "x$BUILD_GPU" = "xyes"; then
		NVCCFLAGS+=" --compiler-options ${gprof_flags}"
	else
		CFLAGS+=" --compiler-options ${gprof_flags}"
	fi
fi

# Check for MPI support.
CONFIGURE_HEADLINE([ MPI compiler ])
ACX_MPI([], [AC_MSG_ERROR([could not find mpi library])])
AC_CHECK_PROG(MPIRUN, mpirun, mpirun)
AC_SUBST(MPIRUN)

# try to find if we are using OpenMPI / MPICH by looking inside mpi.h
sav_CC="$CC"
sav_CFLAGS="$CFLAGS"
CC="$MPICC"
CFLAGS="$CFLAGS"
AC_CHECK_DECL([OPEN_MPI], [mpi_vendor="OpenMPI"],
	     [], [#include "mpi.h"])
AC_CHECK_DECL([MPICH2], [mpi_vendor="MPICH"],
	     [], [#include "mpi.h"])
CC="$sav_CC"
CFLAGS="$sav_CFLAGS"

#
# try to set MPI_CFLAGS and MPI_LDFLAGS
#
MPI_CFLAGS=
MPI_LDFLAGS=
if test "$mpi_vendor" = "OpenMPI"
then 
	MPI_CFLAGS=`$MPICC --showme:compile`
	MPI_LDFLAGS=`$MPICC --showme:link`
	AC_MSG_NOTICE([OpenMPI found])
	AC_MSG_NOTICE([MPI_CFLAGS=$MPI_CFLAGS])
	AC_MSG_NOTICE([MPI_LDFLAGS=$MPI_LDFLAGS])
elif test "$mpi_vendor" = "MPICH"
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
	# build MPI_LDFLAGS
	tmp=`$MPICC -link-info | awk '{$1=""; print $0 }'`
	for i in $tmp
	do 
		case $i in 
			[[\\/]]*.a | ?:[[\\/]]*.a | -[[lLRu]]* | -Wl* )
				MPI_LDFLAGS="$MPI_LDFLAGS $i"
				;;
		esac
	done
	AC_MSG_NOTICE([MPICH found])
	AC_MSG_NOTICE([MPI_CFLAGS=$MPI_CFLAGS])
	AC_MSG_NOTICE([MPI_LDFLAGS=$MPI_LDFLAGS])
else
	AC_MSG_WARN([Neither OpenMPI and MPICH have been recognized...])
fi
AC_SUBST(MPI_CFLAGS)
AC_SUBST(MPI_LDFLAGS)

AC_CONFIG_FILES([Makefile])

AC_OUTPUT
