#ifndef __CUDA_HELPERS__
#define __CUDA_HELPERS__

#include "../reax_types.h"


CUDA_DEVICE static inline int Cuda_strncmp( const char * a,
        const char * b, int len )
{
    int i, ret;
    char *src, *dst;

    ret = 0;
    src = (char *) a;
    dst = (char *) b;

    for ( i = 0; i < len; i++ )
    {
        if ( *src != *dst )
        {
            ret = *src > *dst ? 1 : -1;
            break;
        }
        else if ( *dst == '\0' )
        {
            break;
        }

        src++;
        dst++;
    }

    return ret;
}


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
CUDA_DEVICE static inline double atomicAdd( double* address, double val )
{
    unsigned long long int *address_as_ull, old, assumed;

    address_as_ull = (unsigned long long int*)address;
    old = *address_as_ull;

    do
    {
        assumed = old;
        old = atomicCAS( address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)) );
    }
    while ( assumed != old );

    return __longlong_as_double( old );
}
#endif


CUDA_DEVICE static inline void atomic_rvecAdd( rvec ret, rvec v )
{
    atomicAdd( &ret[0], v[0] );
    atomicAdd( &ret[1], v[1] );
    atomicAdd( &ret[2], v[2] );
}


CUDA_DEVICE static inline void atomic_rvecScaledAdd( rvec ret, real c, rvec v )
{
    atomicAdd( &ret[0], c * v[0] );
    atomicAdd( &ret[1], c * v[1] );
    atomicAdd( &ret[2], c * v[2] );
}

#endif
