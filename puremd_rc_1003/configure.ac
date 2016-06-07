#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])

AC_INIT([Purdue Molecular Dynamics Suite], [1.0], [ohearnku@msu.edu hma@msu.edu])
AM_INIT_AUTOMAKE([1.15 -Wall -Werror])
LT_PREREQ([2.2])
LT_INIT([dlopen])

AC_CONFIG_MACRO_DIR([m4])

# Enable silent build rules by default.
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])], [AC_SUBST([AM_DEFAULT_VERBOSITY],[1])])

AC_ARG_ENABLE([serial],
	      [AS_HELP_STRING([--enable-serial],
			      [enable serial support @<:@default: no@:>@])],
	      [package_serial_enabled=${enableval}], [package_serial_enabled=no])
AC_ARG_ENABLE([openmp],
	      [AS_HELP_STRING([--enable-openmp],
			      [enable OpenMP support @<:@default: yes@:>@])],
	      [package_openmp_enabled=${enableval}], [package_openmp_enabled=yes])
AC_ARG_ENABLE([mpi],
	      [AS_HELP_STRING([--enable-mpi],
			      [enable MPI support @<:@default: no@:>@])],
	      [package_mpi_enabled=${enableval}], [package_mpi_enabled=no])
AC_ARG_ENABLE([mpi-single-gpu],
	      [AS_HELP_STRING([--enable-mpi-single-gpu],
			      [enable MPI+CUDA (single GPU) support @<:@default: no@:>@])],
	      [package_mpi_single_gpu_enabled=${enableval}], [package_mpi_single_gpu_enabled=no])
AC_ARG_ENABLE([mpi-multi-gpu],
	      [AS_HELP_STRING([--enable-mpi-multi-gpu],
			      [enable MPI+CUDA (multi GPU) support @<:@default: no@:>@])],
	      [package_mpi_multi_gpu_enabled=${enableval}], [package_mpi_multi_gpu_enabled=no])

if test "x${package_serial_enabled}" = "xyes" || test "x${package_openmp_enabled}" = "xyes"; then
	AC_CONFIG_SUBDIRS([sPuReMD])
	if test "x${package_serial_enabled}" = "xyes" || test "x${package_openmp_enabled}" != "xyes"; then
		export BUILD_OPENMP="no"
	else
		export BUILD_OPENMP="yes"
	fi
fi
AM_CONDITIONAL([BUILD_S_OMP], [test "x${package_serial_enabled}" = "xyes" || test "x${package_openmp_enabled}" = "xyes"])
if test "x${package_mpi_enabled}" = "xyes"; then
	AC_CONFIG_SUBDIRS([PuReMD])
fi
AM_CONDITIONAL([BUILD_MPI], [test "x${package_mpi_enabled}" = "xyes"])
if test "x${package_mpi_single_gpu_enabled}" = "xyes"; then
	AC_CONFIG_SUBDIRS([PuReMD-GPU])
fi
AM_CONDITIONAL([BUILD_SINGLE_GPU], [test "x${package_mpi_single_gpu_enabled}" = "xyes"])
if test "x${package_mpi_multi_gpu_enabled}" = "xyes"; then
	AC_CONFIG_SUBDIRS([PG-PuReMD])
fi
AM_CONDITIONAL([BUILD_MULTI_GPU], [test "x${package_mpi_multi_gpu_enabled}" = "xyes"])

# Provides debug compilation mode.
AC_ARG_ENABLE([debug],
	      [AS_HELP_STRING([--enable-debug],
			      [enable debug support @<:@default: no@:>@])],
	      [DEBUG=${enableval}], [DEBUG=no]
	      )
if test "x$DEBUG" = "xyes"
then
	#TODO: fix exporting to subdirs
	# See: http://stackoverflow.com/questions/34124337/changing-flags-in-configure-ac-vs-caching-with-subprojects
	CFLAGS="-g3 -O0 -D_GLIBCXX_DEBUG ${CFLAGS}"
	export BUILD_DEBUG="true"
fi

# gprof flags.
AC_ARG_ENABLE([gprof],
	[AS_HELP_STRING([--enable-gprof],
		[enable support for profiling with gprof @<:@default: no@:>@])],
	[case "${enableval}" in
		gnu | yes) 
			gprof_enabled="yes" 
			gprof_compiler="gnu compiler"
			gprof_flags="-pg"
			;;
		intel)
			gprof_enabled="yes"
			gprof_compiler="intel compiler"
			gprof_flags="-p"
			;;
		no)
			gprof_enabled="no" 
			gprof_compiler="none"
			gprof_flags=""
			;;
		*)
			AC_MSG_ERROR([bad value ${enableval} for --enable-gprof (only yes, gnu or intel are possible)]) ;;
	esac],
	[gprof_enabled="no"
	 gprof_compiler="none"
	 gprof_flags=""
	]	      
)
if test "x$gprof_enabled" = "xyes"
then
	#TODO: fix exporting to subdirs
	CPPFLAGS="${CPPFLAGS} ${gprof_flags}"
	LDFLAGS="${LDFLAGS} ${gprof_flags}"
	export BUILD_PROF="yes"
fi

# Timing measurements.
AC_ARG_ENABLE([timing],
	[AS_HELP_STRING([--enable-timing],
		[enable timing measurements and reporting @<:@default: no@:>@])],
	[timing_enabled=${enableval}], [timing_enabled=no]
)
if test "x$timing_enabled" = "xyes"
then
	export BUILD_TIMING="yes"
fi

AC_CONFIG_FILES([Makefile])

AC_OUTPUT
