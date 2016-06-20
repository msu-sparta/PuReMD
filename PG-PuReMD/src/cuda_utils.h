#ifndef __CUDA_UTILS_H_
#define __CUDA_UTILS_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include "stdlib.h"
#include "stdio.h"

#include "reax_types.h"

#ifdef __cplusplus
extern "C"  {
#endif

void compute_blocks (int *, int *, int);
void compute_nearest_pow_2 (int blocks, int *result);
void compute_matvec_blocks (int *, int);


void cuda_malloc (void **, int , int , char *);
void cuda_free (void *, char *);
void cuda_memset (void *, int , size_t , char *);
void copy_host_device (void *, void *, int , enum cudaMemcpyKind, char *);
void copy_device (void *, void *, int , char *);

void print_device_mem_usage ();

#define cudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line )
{
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf (stderr, "Failed .. %s:%d -- gpu erro code %d\n", file, line, err );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    /*
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
       exit( -1 );
    }
    */
    return;
}

#endif

#ifdef __cplusplus
}
#endif
