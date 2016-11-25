/*----------------------------------------------------------------------
  PuReMD-GPU - Reax Force Field Simulator

  Copyright (2014) Purdue University
  Sudhir Kylasa, skylasa@purdue.edu
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
  Ananth Y Grama, ayg@cs.purdue.edu

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License as
  published by the Free Software Foundation; either version 2 of
  the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details:
  <http://www.gnu.org/licenses/>.
  ----------------------------------------------------------------------*/

#ifndef __CUDA_HELPERS__
#define __CUDA_HELPERS__


#include "mytypes.h"


DEVICE inline int cuda_strcmp(char *a, char *b, int len)
{
    char *src, *dst;

    src = a;
    dst = b;

    for (int i = 0; i < len; i++)
    {
        if (*dst == '\0')
        {
            return 0;
        }

        if (*src != *dst)
        {
            return 1;
        }

        src++;
        dst++;
    }

    return 0;
}


DEVICE inline real myAtomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    }
    while (assumed != old);
    return __longlong_as_double(old);
}


DEVICE inline void atomic_rvecAdd( rvec ret, rvec v )
{
    MYATOMICADD( (double*)&ret[0], (double)v[0] );
    MYATOMICADD( (double*)&ret[1], (double)v[1] );
    MYATOMICADD( (double*)&ret[2], (double)v[2] );
}


DEVICE inline void atomic_rvecScaledAdd( rvec ret, real c, rvec v )
{
    MYATOMICADD( (double*)&ret[0], (double)(c * v[0]) );
    MYATOMICADD( (double*)&ret[1], (double)(c * v[1]) );
    MYATOMICADD( (double*)&ret[2], (double)(c * v[2]) );
}


#endif
