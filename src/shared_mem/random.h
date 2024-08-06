/*----------------------------------------------------------------------
  SerialReax - Reax Force Field Simulator

  Copyright (2010) Purdue University
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
  Joseph Fogarty, jcfogart@mail.usf.edu
  Sagar Pandit, pandit@usf.edu
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

#ifndef __RANDOM_H_
#define __RANDOM_H_

#include "reax_types.h"

#include <stdlib.h>
#include <time.h>


/* initialize and seed the system pseudo random number generator with
 * current time (use once) */
void Randomize( );


/* generate a random number in range [0.0, limit] */
static inline double Random( double limit )
{
    int divisor, ret;

    divisor = RAND_MAX / (limit + 1);

    do
    {
        ret = rand() / divisor;
    } while (ret > limit);

    return ret;
}


/* generate a random number from a Gaussian distribution with
 * prescribed mean and standard deviation */
static inline double GRandom( double mean, double sigma )
{
    double v1 = Random(2.0) - 1.0;
    double v2 = Random(2.0) - 1.0;
    double rsq = v1 * v1 + v2 * v2;

    while (rsq >= 1.0 || rsq == 0.0)
    {
        v1 = Random(2.0) - 1.0;
        v2 = Random(2.0) - 1.0;
        rsq = v1 * v1 + v2 * v2;
    }

    return mean + v1 * sigma * SQRT(-2.0 * LOG(rsq) / rsq);
}


#endif
