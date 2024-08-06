/*----------------------------------------------------------------------
  PuReMD - Purdue ReaxFF Molecular Dynamics Program

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

#ifndef __GPU_VALENCE_ANGLES_H_
#define __GPU_VALENCE_ANGLES_H_

#include "../reax_types.h"

#include "../vector.h"


int GPU_Compute_Valence_Angles( reax_system * const,
        control_params const * const, simulation_data * const, storage * const ,
        reax_list **, output_controls const * const );


/* calculates the theta angle between atom triplet i-j-k */
GPU_DEVICE static inline void Calculate_Theta( const rvec dvec_ji, real d_ji,
        const rvec dvec_jk, real d_jk, real * const theta, real * const cos_theta )
{
    assert( d_ji > 0.0 );
    assert( d_jk > 0.0 );

    *cos_theta = rvec_Dot( dvec_ji, dvec_jk ) / ( d_ji * d_jk );

    if ( *cos_theta > 1.0 )
    {
        *cos_theta = 1.0;
    }
    if ( *cos_theta < -1.0 )
    {
        *cos_theta = -1.0;
    }

    (*theta) = ACOS( *cos_theta );
}


/* calculates the derivative of the cosine of the angle between atom triplet i-j-k */
GPU_DEVICE static inline void Calculate_dCos_Theta( const rvec dvec_ji,
        real d_ji, const rvec dvec_jk, real d_jk, rvec * const dcos_theta_di,
        rvec * const dcos_theta_dj, rvec * const dcos_theta_dk )
{
    int t;
    real sqr_d_ji, sqr_d_jk, inv_dists, inv_dists3, dot_dvecs, Cdot_inv3;

    assert( d_ji > 0.0 );
    assert( d_jk > 0.0 );

    sqr_d_ji = SQR( d_ji );
    sqr_d_jk = SQR( d_jk );
    inv_dists = 1.0 / (d_ji * d_jk);
    inv_dists3 = CUBE( inv_dists );
    dot_dvecs = rvec_Dot( dvec_ji, dvec_jk );
    Cdot_inv3 = dot_dvecs * inv_dists3;

    for ( t = 0; t < 3; ++t )
    {
        (*dcos_theta_di)[t] = dvec_jk[t] * inv_dists
            - Cdot_inv3 * sqr_d_jk * dvec_ji[t];

        (*dcos_theta_dj)[t] = -1.0 * (dvec_jk[t] + dvec_ji[t]) * inv_dists
            + Cdot_inv3 * ( sqr_d_jk * dvec_ji[t] + sqr_d_ji * dvec_jk[t] );

        (*dcos_theta_dk)[t] = dvec_ji[t] * inv_dists
            - Cdot_inv3 * sqr_d_ji * dvec_jk[t];
    }
}


#endif
