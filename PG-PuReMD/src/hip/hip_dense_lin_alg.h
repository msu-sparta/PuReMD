#ifndef __HIP_DENSE_LIN_ALG_H_
#define __HIP_DENSE_LIN_ALG_H_

#include "../reax_types.h"


void Vector_Copy( real * const, real const * const, unsigned int, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t
#else
        int, hipStream_t
#endif
        );

void Vector_Copy_rvec2( rvec2 * const, rvec2 const * const, unsigned int, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t
#else
        int, hipStream_t
#endif
        );

void Vector_Copy_From_rvec2( real * const, rvec2 const * const, int, int, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t
#else
        int, hipStream_t
#endif
        );

void Vector_Copy_To_rvec2( rvec2 * const, real const * const, int, int, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t
#else
        int, hipStream_t
#endif
        );

void Vector_Sum( real * const, real, real const * const,
        real, real const * const, unsigned int, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t
#else
        int, hipStream_t
#endif
        );

void Vector_Sum_rvec2( rvec2 * const, real, real, rvec2 const * const,
        real, real, rvec2 const * const, unsigned int, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t
#else
        int, hipStream_t
#endif
        );

void Vector_Add( real * const, real, real const * const, unsigned int, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t
#else
        int, hipStream_t
#endif
        );

void Vector_Add_rvec2( rvec2 * const, real, real, rvec2 const * const, unsigned int, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t
#else
        int, hipStream_t
#endif
        );

real Dot( storage * const, real const * const, real const * const,
        unsigned int, MPI_Comm,
#if defined(USE_HIPBLAS)
        hipblasHandle_t
#else
        int, hipStream_t
#endif
        );

real Dot_local( storage * const, real const * const, real const * const,
        unsigned int, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t
#else
        int, hipStream_t
#endif
        );

void Dot_local_rvec2( storage * const, rvec2 const * const, rvec2 const * const,
        unsigned int, real *, real *, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t
#else
        int, hipStream_t
#endif
        );


#endif
