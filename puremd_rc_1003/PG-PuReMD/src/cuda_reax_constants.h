
#ifndef	__CUDA_REAX_CONSTANTS_H__
#define  __CUDA_REAX_CONSTANTS_H__

#include "cuda.h"

#ifdef __USE_GPU__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_GLOBAL __global__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST
#define CUDA_DEVICE
#define CUDA_GLOBAL
#define CUDA_HOST_DEVICE
#endif

//BLOCK SIZES for kernels
//
#define				HB_SYM_BLOCK_SIZE					64
#define				HB_KER_SYM_THREADS_PER_ATOM			16
#define				HB_POST_PROC_BLOCK_SIZE				256
#define				HB_POST_PROC_KER_THREADS_PER_ATOM	32

#if defined( __INIT_BLOCK_SIZE__)
#define				DEF_BLOCK_SIZE						__INIT_BLOCK_SIZE__    /* all utility functions and all */
#define				CUDA_BLOCK_SIZE						__INIT_BLOCK_SIZE__    	/* init forces */
#define				ST_BLOCK_SIZE						__INIT_BLOCK_SIZE__    
#else
#define				DEF_BLOCK_SIZE						256						/* all utility functions and all */
#define				CUDA_BLOCK_SIZE						256						/* init forces */
#define				ST_BLOCK_SIZE						256	
#endif


#if defined( __NBRS_THREADS_PER_ATOM__ )
#define				NB_KER_THREADS_PER_ATOM				__NBRS_THREADS_PER_ATOM__
#else
#define				NB_KER_THREADS_PER_ATOM				16
#endif

#if defined( __NBRS_BLOCK_SIZE__)
#define				NBRS_BLOCK_SIZE						__NBRS_BLOCK_SIZE__
#else
#define				NBRS_BLOCK_SIZE						256
#endif


#if defined( __HB_THREADS_PER_ATOM__)
#define				HB_KER_THREADS_PER_ATOM				__HB_THREADS_PER_ATOM__
#else
#define				HB_KER_THREADS_PER_ATOM				32
#endif

#if defined(__HB_BLOCK_SIZE__)
#define				HB_BLOCK_SIZE					__HB_BLOCK_SIZE__
#else
#define				HB_BLOCK_SIZE						256
#endif


#if defined( __VDW_THREADS_PER_ATOM__ )
#define				VDW_KER_THREADS_PER_ATOM			__VDW_THREADS_PER_ATOM__
#else
#define				VDW_KER_THREADS_PER_ATOM			32
#endif

#if defined( __VDW_BLOCK_SIZE__)
#define				VDW_BLOCK_SIZE						__VDW_BLOCK_SIZE__
#else
#define				VDW_BLOCK_SIZE						256
#endif


#if defined( __MATVEC_THREADS_PER_ROW__ )
#define 			MATVEC_KER_THREADS_PER_ROW		__MATVEC_THREADS_PER_ROW__
#else
#define 			MATVEC_KER_THREADS_PER_ROW		32
#endif


#if defined( __MATVEC_BLOCK_SIZE__)
#define				MATVEC_BLOCK_SIZE					__MATVEC_BLOCK_SIZE__
#else
#define				MATVEC_BLOCK_SIZE					512
#endif






//Validation
#define				GPU_TOLERANCE				1e-5

#endif
