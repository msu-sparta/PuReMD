#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])

AC_INIT([PuReMD], [1.0], [ohearnku@msu.edu hma@msu.edu])
# Do not allow AC_PROG_CC to set CFLAGS (this line must be after AC_INIT but before AC_PROG_CC)
save_CFLAGS="${CFLAGS}"
: ${CFLAGS=""}
AM_INIT_AUTOMAKE([1.15 subdir-objects -Wall -Werror])
# Enable silent build rules by default.
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])], [AC_SUBST([AM_DEFAULT_VERBOSITY],[1])])
AM_PROG_AR
LT_PREREQ([2.2])
LT_INIT([dlopen])

AC_CONFIG_MACRO_DIR([m4])

# Headline formatter
AC_DEFUN([CONFIGURE_HEADLINE],
[
        echo; echo "+++ $1 +++"
])

AC_DEFUN([AC_PROG_PDFLATEX],
	 [AC_ARG_VAR([PDFLATEX], [LaTeX PDF generation program])dnl
	 AC_CHECK_PROGS([PDFLATEX], [pdflatex])
	 m4_ifval([$1],,
	 	  [if test -z "$PDFLATEX"; then
		   AC_MSG_WARN([pdflatex not found. Unable to build documentation. Continuing...])
		   fi])])

AC_ARG_ENABLE([serial],
	      [AS_HELP_STRING([--enable-serial],
			      [enable serial support @<:@default: no@:>@])],
	      [pack_serial_enabled=${enableval}], [pack_serial_enabled=no])
AC_ARG_ENABLE([openmp],
	      [AS_HELP_STRING([--enable-openmp],
			      [enable OpenMP support @<:@default: yes@:>@])],
	      [pack_openmp_enabled=${enableval}], [pack_openmp_enabled=yes])
AC_ARG_ENABLE([mpi-old],
	      [AS_HELP_STRING([--enable-mpi-old],
			      [enable MPI support (old) @<:@default: no@:>@])],
	      [pack_mpi_old_enabled=${enableval}], [pack_mpi_old_enabled=no])
AC_ARG_ENABLE([gpu],
	      [AS_HELP_STRING([--enable-gpu],
			      [enable CUDA (single GPU) support @<:@default: no@:>@])],
	      [pack_gpu_enabled=${enableval}], [pack_gpu_enabled=no])
AC_ARG_ENABLE([mpi],
	      [AS_HELP_STRING([--enable-mpi],
			      [enable MPI support @<:@default: no@:>@])],
	      [pack_mpi_enabled=${enableval}], [pack_mpi_enabled=no])
AC_ARG_ENABLE([mpi-gpu],
	      [AS_HELP_STRING([--enable-mpi-gpu],
			      [enable MPI+CUDA (multi GPU) support @<:@default: no@:>@])],
	      [pack_mpi_gpu_enabled=${enableval}], [pack_mpi_gpu_enabled=no])

# Provides debug compilation mode.
AC_ARG_ENABLE([debug],
	      [AS_HELP_STRING([--enable-debug],
			      [enable debug support @<:@default: no@:>@])],
	      [DEBUG=${enableval}], [DEBUG=no]
	      )
if test "x${DEBUG}" = "xyes"
then
#	#TODO: fix exporting to subdirs
#	# See: http://stackoverflow.com/questions/34124337/changing-flags-in-configure-ac-vs-caching-with-subprojects
	export BUILD_DEBUG="yes"
fi

# gprof flags.
AC_ARG_ENABLE([gprof],
	[AS_HELP_STRING([--enable-gprof],
		[enable support for profiling with gprof @<:@default: no@:>@])],
	[case "${enableval}" in
		yes) 
			export BUILD_GPROF="yes"
			;;
		no)
			;;
		*)
			AC_MSG_ERROR([bad value ${enableval} for --enable-gprof (only yes or no are possible)]) ;;
	esac],
	[]	      
)

# Timing measurements.
AC_ARG_ENABLE([timing],
	[AS_HELP_STRING([--enable-timing],
		[enable timing measurements and reporting @<:@default: no@:>@])],
	[timing_enabled=${enableval}], [timing_enabled=no]
)
if test "x${timing_enabled}" = "xyes"
then
	export BUILD_TIMING="yes"
fi

AC_ARG_ENABLE([doc],
	      [AS_HELP_STRING([--enable-doc],
			      [enable documentation generation @<:@default: no@:>@])],
	      [pack_doc_enabled=${enableval}], [pack_doc_enabled=no])
AC_PROG_PDFLATEX
AM_CONDITIONAL([BUILD_DOC], [test "x${pack_doc_enabled}" = "xyes" && test "x${PDFLATEX}" != "x"])
if test "x${pack_doc_enabled}" = "xyes"; then
	AC_CONFIG_SUBDIRS([doc])
fi


# Checks for compilers.
AC_LANG([C])
AC_PROG_CC([icc gcc clang cc])
AC_PROG_CPP
CFLAGS="${save_CFLAGS}"

AC_LANG([C++])
AC_PROG_CXX([icpc g++ clang++ CC c++])
AC_PROG_CXXCPP

AC_CONFIG_HEADERS([common/include/config.h])

# sPuReMD
if test "x${pack_serial_enabled}" = "xyes" || test "x${pack_openmp_enabled}" = "xyes"; then
	if test "x${pack_serial_enabled}" = "xyes" || test "x${pack_openmp_enabled}" != "xyes"; then
		BUILD_OPENMP="no"
	else
		BUILD_OPENMP="yes"
	fi

	AC_LANG_PUSH([C])

	AC_CONFIG_SRCDIR([sPuReMD/src/traj.h])

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
	AC_SEARCH_LIBS([gzclose], [z])

	# Checks for typedefs, structures, and compiler characteristics.
	AC_C_INLINE
	AC_CHECK_TYPES([gzFile], [],
		       [AC_MSG_FAILURE([gzFile type not found in zlib.h], [1])], [#include<zlib.h>])

	# Checks for library functions.
	AC_FUNC_MALLOC
	AC_FUNC_REALLOC
	AC_FUNC_STRTOD
	AC_CHECK_FUNCS([gettimeofday memset])

	# Check for compiler vendor. If the compiler is recognized,
	#   the variable ax_cv_c_compiler_vendor is set accordingly.
	AX_COMPILER_VENDOR
	if test "x${ax_cv_c_compiler_vendor}" = "xgnu"; then
		if test "x${BUILD_DEBUG}" = "xyes"; then
			DEFAULT_CFLAGS="-Wall -O0 -g2 -D_GLIBCXX_DEBUG"
			AC_DEFINE([DEBUG], [1], [Define to 1 if you have debug support enabled.])
		else
			DEFAULT_CFLAGS="-Wall -O3"
		fi
		if test "x${BUILD_GPROF}" = "xyes"; then
			DEFAULT_CFLAGS+=" -pg"
		fi
	elif test "x${ax_cv_c_compiler_vendor}" = "xintel"; then
		if test "x${BUILD_DEBUG}" = "xyes"; then
			DEFAULT_CFLAGS="-Wall -O0"
			AC_DEFINE([DEBUG], [1], [Define to 1 if you have debug support enabled.])
		else
			DEFAULT_CFLAGS="-Wall -O3"
		fi
		if test "x${BUILD_GPROF}" = "xyes"; then
			DEFAULT_CFLAGS+=" -p"
		fi
	fi
	AC_SUBST([DFT_CFLAGS], ["${DEFAULT_CFLAGS}"])

	# Check for OpenMP support.
	if test "x${BUILD_OPENMP}" = "xyes"; then
		AX_OPENMP

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
			OPENMP_LIBS="-lgomp"
		fi

	#	AC_MSG_NOTICE([OPENMP_CFLAGS=${OPENMP_CFLAGS}])
	#	AC_MSG_NOTICE([OPENMP_LIBS=${OPENMP_LIBS}])
		AC_SUBST([OMP_CFLAGS], ["${OPENMP_CFLAGS}"])
		AC_SUBST([OMP_LIBS], ["${OPENMP_LIBS}"])
	fi

	# Check for LAPACKE
	LAPACKE_LIBS=
	AC_CHECK_HEADERS([mkl.h], [MKL_FOUND_HEADERS="yes"])
	if test "x${MKL_FOUND_HEADERS}" = "xyes"
	then
		AC_SEARCH_LIBS([LAPACKE_dgels], [mkl_intel_ilp64],
			       [MKL_FOUND_LIBS="yes"], [MKL_FOUND_LIBS="no"],
			       [-lmkl_sequential -lmkl_core -lpthread -lm -ldl])
		AS_IF([test "x${MKL_FOUND_LIBS}" != "xyes"],
		      [AC_MSG_ERROR([Unable to find MKL LAPACKE library.])])
		LAPACKE_LIBS="-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"
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
			LAPACKE_LIBS="-llapack"
			AC_DEFINE([HAVE_LAPACKE], [1], [Define to 1 if you have LAPACKE support enabled.])
		else
			AC_MSG_WARN([
-----------------------------------------------
 Unable to find LAPACKE on this system.
 Disabling support for dependent methods.
-----------------------------------------------])
		fi
	fi
	AC_SUBST([L_LIBS], ["${LAPACKE_LIBS}"])

	AC_LANG_POP([C])
fi
AM_CONDITIONAL([BUILD_S_OMP], [test "x${pack_serial_enabled}" = "xyes" || test "x${pack_openmp_enabled}" = "xyes"])


# PuReMD-GPU
if test "x${pack_gpu_enabled}" = "xyes"; then
	BUILD_GPU="yes"

	AC_LANG_PUSH([C])

	AC_CONFIG_SRCDIR([PuReMD-GPU/src/analyze.c])

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
	AC_CHECK_FUNCS([gettimeofday memset])

	# Check for compiler vendor. If the compiler is recognized,
	#   the variable ax_cv_c_compiler_vendor is set accordingly.
	AX_COMPILER_VENDOR
	if test "x${ax_cv_c_compiler_vendor}" = "xgnu"; then
		if test "x${BUILD_DEBUG}" = "xyes"; then
			DEFAULT_CFLAGS="-Wall -O0 -g2 -D_GLIBCXX_DEBUG"
			AC_DEFINE([DEBUG], [1], [Define to 1 if you have debug support enabled.])
		else
			DEFAULT_CFLAGS="-Wall -O3"
		fi
		if test "x${BUILD_GPROF}" = "xyes"; then
			DEFAULT_CFLAGS+=" -pg"
		fi
	elif test "x${ax_cv_c_compiler_vendor}" = "xintel"; then
		if test "x${BUILD_DEBUG}" = "xyes"; then
			DEFAULT_CFLAGS="-Wall -O0"
			AC_DEFINE([DEBUG], [1], [Define to 1 if you have debug support enabled.])
		else
			DEFAULT_CFLAGS="-Wall -O3"
		fi
		if test "x${BUILD_GPROF}" = "xyes"; then
			DEFAULT_CFLAGS+=" -p"
		fi
	fi

	# Check for CUDA support.
	CONFIGURE_HEADLINE([ CUDA support ])
	AX_CUDA
	NVCCFLAGS=
	if test "x${BUILD_DEBUG}" = "xyes"
	then
		NVCCFLAGS="-g -G"
	fi
	if test "x${BUILD_GPROF}" = "xtrue"
	then
		NVCCFLAGS+=" --compiler-options ${gprof_flags}"
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

	AC_LANG_POP([C])
fi
AM_CONDITIONAL([BUILD_GPU], [test "x${pack_gpu_enabled}" = "xyes"])


# PuReMD
if test "x${pack_mpi_old_enabled}" = "xyes"; then
	AC_LANG_PUSH([C])

	AC_CONFIG_SRCDIR([PuReMD/src/torsion_angles.h])

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
		if test "x${BUILD_DEBUG}" = "xyes"; then
			DEFAULT_CFLAGS="-Wall -O0 -g2 -D_GLIBCXX_DEBUG"
			AC_DEFINE([DEBUG], [1], [Define to 1 if you have debug support enabled.])
		else
			DEFAULT_CFLAGS="-Wall -O3"
		fi
		if test "x${BUILD_GPROF}" = "xyes"; then
			DEFAULT_CFLAGS+=" -pg"
		fi
	elif test "x${ax_cv_c_compiler_vendor}" = "xintel"; then
		if test "x${BUILD_DEBUG}" = "xyes"; then
			DEFAULT_CFLAGS="-Wall -O0"
			AC_DEFINE([DEBUG], [1], [Define to 1 if you have debug support enabled.])
		else
			DEFAULT_CFLAGS="-Wall -O3"
		fi
		if test "x${BUILD_GPROF}" = "xyes"; then
			DEFAULT_CFLAGS+=" -p"
		fi
	fi
	AC_SUBST([DFT_CFLAGS], ["${DEFAULT_CFLAGS}"])

	# Check for MPI support. If found, sets the following variables:
	#   Compiler-related: MPICC, MPICXX, MPIF77
	#   Library linking: MPILIBS
	#   Compilation support (via preprocessor def): HAVE_MPI
	CONFIGURE_HEADLINE([ MPI compiler ])
	ACX_MPI([], [AC_MSG_ERROR([could not find mpi library])])
	AC_CHECK_PROG(MPIRUN, mpirun, mpirun)

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
		AC_MSG_NOTICE([OpenMPI support found])
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
		AC_MSG_NOTICE([MPICH support found])
		AC_MSG_NOTICE([MPI_CFLAGS=${MPI_CFLAGS}])
		AC_MSG_NOTICE([MPI_LIBS=${MPI_LIBS}])
	else
		AC_MSG_WARN([Neither OpenMPI and MPICH support recognized...])
	fi
	AC_SUBST([M_CFLAGS], ["${MPI_CFLAGS}"])
	AC_SUBST([M_LIBS], ["${MPI_LIBS}"])

	# Check for LAPACKE
	LAPACKE_LIBS=
	AC_CHECK_HEADERS([mkl.h], [MKL_FOUND_HEADERS="yes"])
	if test "x${MKL_FOUND_HEADERS}" = "xyes"
	then
		AC_SEARCH_LIBS([LAPACKE_dgels], [mkl_intel_ilp64],
			       [MKL_FOUND_LIBS="yes"], [MKL_FOUND_LIBS="no"],
			       [-lmkl_sequential -lmkl_core -lpthread -lm -ldl])
		AS_IF([test "x${MKL_FOUND_LIBS}" != "xyes"],
		      [AC_MSG_ERROR([Unable to find MKL LAPACKE library.])])
		# dynamic linking
		#LAPACKE_LIBS="-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"
		# static linking
		LAPACKE_LIBS="-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl"
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
			LAPACKE_LIBS="-llapack"
			AC_DEFINE([HAVE_LAPACKE], [1], [Define to 1 if you have LAPACKE support enabled.])
		else
			AC_MSG_WARN([
-----------------------------------------------
 Unable to find LAPACKE on this system.
 Disabling support for dependent methods.
-----------------------------------------------])
		fi
	fi
	AC_SUBST([L_LIBS], ["${LAPACKE_LIBS}"])
fi

	AC_LANG_POP([C])
AM_CONDITIONAL([BUILD_MPI], [test "x${pack_mpi_old_enabled}" = "xyes"])


# PG-PuReMD
if test "x${pack_mpi_enabled}" = "xyes" || test "x${pack_mpi_gpu_enabled}" = "xyes"; then
	if test "x${pack_mpi_enabled}" = "xyes" || test "x${pack_mpi_gpu_enabled}" != "xyes"; then
		export BUILD_GPU="no"
	else
		export BUILD_GPU="yes"
	fi

	AC_LANG_PUSH([C])

	AC_CONFIG_SRCDIR([PG-PuReMD/src/torsion_angles.h])

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

	# Check for compiler vendor. If the compiler is recognized,
	#   the variable ax_cv_c_compiler_vendor is set accordingly.
	AX_COMPILER_VENDOR
	if test "x${ax_cv_c_compiler_vendor}" = "xgnu"; then
		if test "x${BUILD_DEBUG}" = "xyes"; then
			DEFAULT_CFLAGS="-Wall -O0 -g2 -D_GLIBCXX_DEBUG"
			AC_DEFINE([DEBUG], [1], [Define to 1 if you have debug support enabled.])
		else
			DEFAULT_CFLAGS="-Wall -O3"
		fi
		if test "x${BUILD_GPROF}" = "xyes"; then
			DEFAULT_CFLAGS+=" -pg"
		fi
	elif test "x${ax_cv_c_compiler_vendor}" = "xintel"; then
		if test "x${BUILD_DEBUG}" = "xyes"; then
			DEFAULT_CFLAGS="-Wall -O0"
			AC_DEFINE([DEBUG], [1], [Define to 1 if you have debug support enabled.])
		else
			DEFAULT_CFLAGS="-Wall -O3"
		fi
		if test "x${BUILD_GPROF}" = "xyes"; then
			DEFAULT_CFLAGS+=" -p"
		fi
	fi
	AC_SUBST([DFT_CFLAGS], ["${DEFAULT_CFLAGS}"])

	# Check for MPI support.
	CONFIGURE_HEADLINE([ MPI compiler ])
	ACX_MPI([], [AC_MSG_ERROR([could not find MPI library])])
	AC_CHECK_PROG(MPIRUN, mpirun, mpirun)
	AC_SUBST(MPIRUN)

	# try to find if we are using OpenMPI / MPICH by looking inside mpi.h
	save_CC="${CC}"
	save_CFLAGS="${CFLAGS}"
	CC="${MPICC}"
	CFLAGS="${CFLAGS}"
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
		for i in ${tmp}
		do 
			case ${i} in 
				-[[DIUbi]]*)
					MPI_CFLAGS="${MPI_CFLAGS} ${i}"
					;;
			esac
		done
		# build MPI_LIBS
		tmp=`${MPICC} -link-info | awk '{$1=""; print $0 }'`
		for i in ${tmp}
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
	AC_SUBST([M_CFLAGS], ["${MPI_CFLAGS}"])
	AC_SUBST([M_LIBS], ["${MPI_LIBS}"])

	# Check for LAPACKE
	LAPACKE_LIBS=
	AC_CHECK_HEADERS([mkl.h], [MKL_FOUND_HEADERS="yes"])
	if test "x${MKL_FOUND_HEADERS}" = "xyes"
	then
		AC_SEARCH_LIBS([LAPACKE_dgels], [mkl_intel_ilp64],
			       [MKL_FOUND_LIBS="yes"], [MKL_FOUND_LIBS="no"],
			       [-lmkl_sequential -lmkl_core -lpthread -lm -ldl])
		AS_IF([test "x${MKL_FOUND_LIBS}" != "xyes"],
		      [AC_MSG_ERROR([Unable to find MKL LAPACKE library.])])
		LAPACKE_LIBS="-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"
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
			LAPACKE_LIBS="-llapack"
			AC_DEFINE([HAVE_LAPACKE], [1], [Define to 1 if you have LAPACKE support enabled.])
		else
			AC_MSG_WARN([
-----------------------------------------------
 Unable to find LAPACKE on this system.
 Disabling support for dependent methods.
-----------------------------------------------])
		fi
	fi
	AC_SUBST([L_LIBS], ["${LAPACKE_LIBS}"])

	AC_LANG_POP([C])

	AC_LANG_PUSH([C++])

	# Check for CUDA support.
	if test "x${BUILD_GPU}" = "xyes"; then
		CONFIGURE_HEADLINE([ CUDA support ])
		AX_CUDA

		NVCCFLAGS=
		NVCCFLAGS+=" -ccbin=${CXX}"
		if test "x${BUILD_DEBUG}" = "xyes"
		then
			NVCCFLAGS+=" -g -G"

		fi
		AC_DEFINE([HAVE_CUDA], [1], [Define to 1 if you have CUDA support enabled.])
	else
		AM_CONDITIONAL(USE_CUDA, test "x" = "xyes")
	fi

	if test "x${BUILD_GPROF}" = "xyes"
	then
		if test "x${BUILD_GPU}" = "xyes"; then
			NVCCFLAGS+=" --compiler-options ${gprof_flags}"
		else
			CFLAGS+=" ${gprof_flags}"
		fi
	fi

	AC_LANG_POP([C++])
fi
AM_CONDITIONAL([BUILD_MPI_GPU], [test "x${pack_mpi_enabled}" = "xyes" || test "x${pack_mpi_gpu_enabled}" = "xyes"])
# Moved from m4/ax_cuda.am to avoid conditionally calling inside AX_CUDA
AM_CONDITIONAL(USE_CUDA, [test "x${pack_gpu_enabled}" = "xyes" || test "x${pack_mpi_gpu_enabled}" = "xyes"])
AM_CONDITIONAL(CUDA_DEBUG, [(test "x${pack_gpu_enabled}" = "xyes" || test "x${pack_mpi_gpu_enabled}" = "xyes") && test "x${BUILD_DEBUG}" = "xyes"])


# Tests using Google C++ testing framework (gtest)
AC_LANG_PUSH([C++])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
  [[#ifndef __cplusplus
    #error "broken C++"
    #endif]])],,
  [CXX=;])
if test "x${CXX}" != "x"
then
	AC_CHECK_HEADER([gtest/gtest.h], [HAVE_GTEST_H="yes"])
	if test "x${HAVE_GTEST_H}" != "xyes"
	then
		AC_MSG_WARN([gtest.h not found. Unable to build tests. Continuing...])
	else
		ACX_PTHREAD([have_pthread=yes])

		# define gtest variables 
		if test "x${have_pthread}" = "xyes"; then
			GTEST_CPPFLAGS="-DGTEST_HAS_PTHREAD=1"
			GTEST_CXXFLAGS="${PTHREAD_CFLAGS}"
			GTEST_LDFLAGS=
			GTEST_LIBS="${PTHREAD_LIBS}"
		else
			GTEST_CPPFLAGS="-DGTEST_HAS_PTHREAD=0"
			GTEST_CXXFLAGS=
			GTEST_LDFLAGS=
			GTEST_LIBS=
		fi
	fi
else
	AC_MSG_WARN([C++ compiler not found. Unable to build tests. Continuing...])
fi
AM_CONDITIONAL([BUILD_TEST], [test "x${HAVE_GTEST_H}" = "xyes"])
AC_LANG_POP([C++])


AC_CONFIG_FILES([Makefile])

AC_OUTPUT
