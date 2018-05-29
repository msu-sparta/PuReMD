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

#ifndef __BOND_ORDERS_H_
#define __BOND_ORDERS_H_

#include "reax_types.h"


typedef struct
{
    real C1dbo;
    real C2dbo;
    real C3dbo;
    real C1dbopi;
    real C2dbopi;
    real C3dbopi;
    real C4dbopi;
    real C1dbopi2;
    real C2dbopi2;
    real C3dbopi2;
    real C4dbopi2;
    real C1dDelta;
    real C2dDelta;
    real C3dDelta;
} dbond_coefficients;


#ifdef __cplusplus
extern "C" {
#endif

void Add_dBond_to_Forces( int, int, storage * const, reax_list ** const );

void Add_dBond_to_Forces_NPT( int, int, simulation_data * const, storage * const,
        reax_list ** const );

int BOp( storage * const, reax_list * const, real, int, int, const far_neighbor_data * const,
        const single_body_parameters * const, const single_body_parameters * const,
        const two_body_parameters * const );

void BO( reax_system * const, control_params * const, simulation_data * const,
        storage * const, reax_list ** const, output_controls * const );

#ifdef TEST_FORCES
void Add_dBO( reax_system * const, reax_list ** const,
        int, int, real, rvec * const );

void Add_dBOpinpi2( reax_system * const, reax_list ** const,
        int, int, real, real, rvec * const, rvec * const );

void Add_dBO_to_Forces( reax_system * const, reax_list ** const,
        int, int, real );

void Add_dBOpinpi2_to_Forces( reax_system * const, reax_list ** const,
        int, int, real, real );

void Add_dDelta( reax_system * const, reax_list ** const, int, real,
        rvec * const );

void Add_dDelta_to_Forces( reax_system * const, reax_list ** const,
        int, real );
#endif

#ifdef __cplusplus
}
#endif


#endif
