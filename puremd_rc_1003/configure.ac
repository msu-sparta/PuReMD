#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])

AC_INIT([Purdue Molecular Dynamics Suite], [1.0], [ohearnku@msu.edu hma@msu.edu])
AM_INIT_AUTOMAKE([1.15])
LT_PREREQ([2.2])
LT_INIT([dlopen])

AC_CONFIG_MACRO_DIR([m4])

# Enable silent build rules by default.
m4_ifdef([AM_SILENT_RULES],[AM_SILENT_RULES([yes])],[AC_SUBST([AM_DEFAULT_VERBOSITY], [1])])

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

AC_CONFIG_SUBDIRS([sPuReMD])
#TODO: find a way to enable/disable serial/OpenMP builds
#if test "x${package_serial_enabled}" = "xyes" || test "x${package_openmp_enabled}" = "xyes"; then
#	AC_CONFIG_SUBDIRS([sPuReMD])
#	if test "x${package_serial_enabled}" = "xyes"; then
#
#	fi
#fi
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
	      [  --enable-debug    Turn on debugging],
	      [case "${enableval}" in
		       yes) DEBUG=true ;;
		       no)  DEBUG=false ;;
		       *)   AC_MSG_ERROR([bad value ${enableval} for --enable-debug]) ;;
	       esac],
	       [DEBUG=false]
	       )
if test "x$DEBUG" = "xtrue"
then
	CFLAGS="-g3 -O0 -D_GLIBCXX_DEBUG ${CFLAGS}"
	BUILD_DEBUG="true"
else
	CFLAGS="-O3 ${CFLAGS}"
fi

# gprof flags.
AC_ARG_ENABLE([gprof],
	[  --enable-gprof    Turn on profiling with gprof (pass gnu or intel as argument)],
	[case "${enableval}" in
		gnu | yes) 
			gprof_enabled=true 
			gprof_compiler="gnu compiler"
			gprof_flags="-pg"
			;;
		intel)
			gprof_enabled=true
			gprof_compiler="intel compiler"
			gprof_flags="-p"
			;;
		no)
			gprof_enabled=false 
			gprof_compiler="none"
			gprof_flags=""
			;;
		*)
			AC_MSG_ERROR([bad value ${enableval} for --enable-gprof (only yes, gnu or intel are possible)]) ;;
	esac],
	[gprof_enabled=false
	 gprof_compiler="none"
	 gprof_flags=""
	]	      
)
if test "x$gprof_enabled" = "xtrue"
then
	CPPFLAGS="${CPPFLAGS} ${gprof_flags}"
	LDFLAGS="${LDFLAGS} ${gprof_flags}"
	BUILD_PROF="true"
fi

# Timing measurements.
AC_ARG_ENABLE([timing],
	[  --enable-timing    Turn on timing measurement and report],
	[case "${enableval}" in
		yes) 
			timing_enabled=true 
			;;
		no)
			timing_enabled=false 
			;;
		*)
			AC_MSG_ERROR([bad value ${enableval} for --enable-timing]) ;;
	esac],
	[timing_enabled=false]
)
if test "x$gprof_enabled" = "xtrue"
then
	BUILD_TIMING="true"
fi

AC_CONFIG_FILES([Makefile])

AC_OUTPUT
