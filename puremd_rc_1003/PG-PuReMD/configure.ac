#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])

AC_INIT([PG-PuReMD], [1.0], [ohearnku@msu.edu hma@msu.edu])
AM_INIT_AUTOMAKE([1.15])
LT_PREREQ([2.2])
LT_INIT([dlopen])

AC_LANG([C])

AC_CONFIG_SRCDIR([src/torsion_angles.h])
AC_CONFIG_HEADERS([src/config.h])

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
CONFIGURE_HEADLINE([ CUDA support ])
AX_CUDA
if test "BUILD_DEBUG" = "true"
then
	NVCCFLAGS+=" -g -G"
else
	NVCCFLAGS+=" -O3"
fi

if test "BUILD_PROF" = "true"
then
	NVCCFLAGS+=" --compiler-options ${gprof_flags}"
fi

# Check for MPI support.
CONFIGURE_HEADLINE([ MPI compiler ])
AC_ARG_ENABLE(mpi, [AC_HELP_STRING([--enable-mpi],[MPI library required])], enable_mpi=$enableval, enable_mpi=no)

if test "$enable_mpi" = "yes"; then
	ACX_MPI([],[AC_MSG_ERROR([could not find mpi library for --enable-mpi])])
	AC_CHECK_PROG(MPIRUN, mpirun, mpirun)
	AC_SUBST(MPIRUN)
else
	AC_MSG_NOTICE([MPI is currently disabled; use option --enable-mpi to enable])
fi
AM_CONDITIONAL(USE_MPI, test "$enable_mpi" = "yes")

# try to find if we are using OpenMPI / MPICH by looking inside mpi.h
AS_IF([test "$enable_mpi" = "yes"],
      [
       sav_CXX=$CXX
       sav_CXXFLAGS=$CXXFLAGS
       CXX=$MPICXX
       CXXFLAGS="$CXXFLAGS"
       AC_CHECK_DECL([OPEN_MPI], [mpi_vendor="OpenMPI"],
		     [], [#include "mpi.h"])
       AC_CHECK_DECL([MPICH2], [mpi_vendor="MPICH"],
		     [], [#include "mpi.h"])
       CXX=$sav_CXX
       CXXFLAGS=$sav_CXXFLAGS
       ])

#
# try to set MPI_CXXFLAGS and MPI_LDFLAGS
#
MPI_CXXFLAGS=
MPI_LDFLAGS=
if test "$enable_mpi" = "yes" -a "$mpi_vendor" = "OpenMPI"
then 
	MPI_CXXFLAGS=`$MPICXX --showme:compile`
	MPI_LDFLAGS=`$MPICXX --showme:link`
	AC_MSG_NOTICE([OpenMPI found])
	AC_MSG_NOTICE([MPI_CXXFLAGS=$MPI_CXXFLAGS])
	AC_MSG_NOTICE([MPI_LDFLAGS=$MPI_LDFLAGS])
elif test "$enable_mpi" = "yes" -a "$mpi_vendor" = "MPICH"
then
	# first grab CXXFLAGS (awk script is used to remove CXX at front)
	tmp=`$MPICXX -compile-info | awk '{$1=""; print $0 }'`
	MPI_CXXFLAGS=
	for i in $tmp
	do 
		case $i in 
			-[[DIUbi]]*)
				MPI_CXXFLAGS="$MPI_CXXFLAGS $i"
				;;
		esac
	done
	# second grab LDFLAGS (awk script is used to remove CXX at front)
	tmp=`$MPICXX -link-info | awk '{$1=""; print $0 }'`
	for i in $tmp
	do 
		case $i in 
			[[\\/]]*.a | ?:[[\\/]]*.a | -[[lLRu]]* | -Wl* )
				MPI_LDFLAGS="$MPI_LDFLAGS $i"
				;;
		esac
	done
	AC_MSG_NOTICE([MPICH found])
	AC_MSG_NOTICE([MPI_CXXFLAGS=$MPI_CXXFLAGS])
	AC_MSG_NOTICE([MPI_LDFLAGS=$MPI_LDFLAGS])
else
	AC_MSG_WARN([Neither OpenMPI and MPICH have been recognized...])
fi
AC_SUBST(MPI_CXXFLAGS)
AC_SUBST(MPI_LDFLAGS)

AC_CONFIG_FILES([Makefile src/Makefile])

AC_OUTPUT
