#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])

AC_INIT([sPuReMD], [1.0], [ohearnku@msu.edu hma@msu.edu])
AM_INIT_AUTOMAKE([1.15 subdir-objects -Wall -Werror foreign])
AM_SILENT_RULES([yes])
#LT_PREREQ([2.2])
#LT_INIT([dlopen])

AC_LANG([C])

AC_CONFIG_SRCDIR([src/traj.h])
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
AC_SEARCH_LIBS([gzopen], [z])
AC_SEARCH_LIBS([gzeof], [z])
AC_SEARCH_LIBS([gzgets], [z])
AC_SEARCH_LIBS([gzseek], [z])
AC_SEARCH_LIBS([gzclose, [z]])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE
AC_CHECK_TYPES([gzFile], [], [AC_MSG_FAILURE([gzFile type not found in zlib.h], [1])], [#include<zlib.h>])

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([gettimeofday memset])

# Check for OpenMP support.
if test "x$BUILD_OPENMP" = "xyes"; then
	AC_OPENMP
	if test "x${OPENMP_CFLAGS}" = "x"; then
		AC_MSG_WARN([
	  -----------------------------------------------
	   Unable to find OpenMP support on this system.
	   Building a single-threaded version.
	  -----------------------------------------------])
	else
		AC_SUBST(AM_CFLAGS, "$OPENMP_CFLAGS")
		AC_SUBST(AM_CPPFLAGS, "$OPENMP_CFLAGS")
	fi
fi

AC_CONFIG_FILES([Makefile])

AC_OUTPUT
