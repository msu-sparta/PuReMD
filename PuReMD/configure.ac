#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])

AC_INIT([PuReMD], [1.0], [ohearnku@msu.edu hma@msu.edu])
: ${CFLAGS=""}
AM_INIT_AUTOMAKE([1.15 subdir-objects -Wall -Werror foreign])
# Enable silent build rules by default.
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])], [AC_SUBST([AM_DEFAULT_VERBOSITY],[1])])
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

AC_LANG([C++])

# Checks for programs.
AC_PROG_CC([cc icc gcc])
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
AC_C_INLINE

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([gettimeofday memset pow sqrt])

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
AC_CHECK_DECL([MPICH2], [mpi_vendor="MPICH2"],
	     [], [#include "mpi.h"])
AC_CHECK_DECL([MPICH_VERSION], [mpi_vendor="MPICH3"],
	     [], [#include "mpi.h"])
CC="$sav_CC"
CFLAGS="$sav_CFLAGS"

#
# try to set MPI_CFLAGS and MPI_LDFLAGS
#
MPI_CFLAGS=
MPI_LDFLAGS=
if test "x$mpi_vendor" = "xOpenMPI"
then 
	MPI_CFLAGS=`$MPICC --showme:compile`
	MPI_LDFLAGS=`$MPICC --showme:link`
	AC_MSG_NOTICE([OpenMPI found])
	AC_MSG_NOTICE([MPI_CFLAGS=$MPI_CFLAGS])
	AC_MSG_NOTICE([MPI_LDFLAGS=$MPI_LDFLAGS])
elif test "x$mpi_vendor" = "xMPICH2" || test "x$mpi_vendor" = "xMPICH3"
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
