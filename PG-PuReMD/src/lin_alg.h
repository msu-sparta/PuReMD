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

#ifndef __LIN_ALG_H_
#define __LIN_ALG_H_

#include "reax_types.h"


#ifdef __cplusplus
extern "C" {
#endif


//real diag_pre_comp( const sparse_matrix * const, real * const );
real diag_pre_comp( const reax_system * const, real * const );

int dual_CG( const reax_system * const, const control_params * const,
        const storage * const, const simulation_data * const,
        mpi_datatypes * const,
        const sparse_matrix * const, const rvec2 * const,
        const real, rvec2 * const, const int );

int CG( const reax_system * const, const control_params * const,
        const storage * const, const simulation_data * const,
        mpi_datatypes * const,
        const sparse_matrix * const, const real * const,
        const real, real * const, const int );


#ifdef __cplusplus
}
#endif


#endif
