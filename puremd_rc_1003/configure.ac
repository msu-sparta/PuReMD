#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])

AC_INIT([Purdue Molecular Dynamics Suite], [1.0], [ohearnku@msu.edu hma@msu.edu])
AM_INIT_AUTOMAKE([1.15])
LT_PREREQ([2.2])
LT_INIT([dlopen])

AC_CONFIG_MACRO_DIR([m4])

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

AC_CONFIG_FILES([Makefile])

AC_OUTPUT
