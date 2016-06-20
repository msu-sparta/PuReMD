
#ifndef __CUDA_HELPERS__
#define __CUDA_HELPERS__

#include "reax_types.h"

CUDA_DEVICE inline int cuda_strcmp (char *a, char *b, int len)
{
   char *src, *dst;

   src = a; 
   dst = b; 

   for (int i = 0; i < len; i++) {

      if (*dst == '\0')
         return 0; 

      if (*src != *dst)  return 1; 

      src ++;
      dst ++;
   }  

   return 0; 
}

CUDA_DEVICE inline real atomicAdd(real* address, real val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        	assumed = old;
			old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

CUDA_DEVICE inline void atomic_rvecAdd( rvec ret, rvec v ) 
{
  atomicAdd ( &ret[0], v[0] );
  atomicAdd ( &ret[1], v[1] ); 
  atomicAdd ( &ret[2], v[2] );
}

CUDA_DEVICE inline void atomic_rvecScaledAdd( rvec ret, real c, rvec v ) 
{
  atomicAdd ( &ret[0], c * v[0] );
  atomicAdd ( &ret[1], c * v[1] );
  atomicAdd ( &ret[2], c * v[2] );
}

#endif
