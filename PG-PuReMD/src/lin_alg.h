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


void Sort_Matrix_Rows( sparse_matrix * const );

real setup_sparse_approx_inverse( reax_system const * const,
        simulation_data * const,
        storage * const, mpi_datatypes * const, 
        sparse_matrix * const, sparse_matrix *, int, double );

real sparse_approx_inverse( reax_system const * const,
        simulation_data * const,
        storage * const, mpi_datatypes * const, 
        sparse_matrix * const, sparse_matrix * const, sparse_matrix **, int );

int SDM( reax_system const * const, control_params const * const,
        simulation_data * const,
        storage * const, sparse_matrix * const, real * const,
        real, real * const, mpi_datatypes * const );

int dual_CG( reax_system const * const, control_params const * const,
        simulation_data * const,
        storage * const, sparse_matrix * const,
        rvec2 * const, real, rvec2 * const, mpi_datatypes * const );

int CG( reax_system const * const, control_params const * const,
        simulation_data * const,
        storage * const, sparse_matrix * const, real * const,
        real, real * const, mpi_datatypes * const );

int BiCGStab( reax_system const * const, control_params const * const,
        simulation_data * const,
        storage * const, sparse_matrix * const, real * const,
        real, real * const, mpi_datatypes * const );

int dual_PIPECG( reax_system const * const, control_params const * const,
        simulation_data * const,
        storage * const, sparse_matrix * const,
        rvec2 * const, real, rvec2 * const, mpi_datatypes * const );

int PIPECG( reax_system const * const, control_params const * const,
        simulation_data * const,
        storage * const, sparse_matrix * const, real * const,
        real, real * const, mpi_datatypes * const );

int PIPECR( reax_system const * const, control_params const * const,
        simulation_data * const,
        storage * const, sparse_matrix * const, real * const,
        real, real * const, mpi_datatypes * const );


#ifdef __cplusplus
}
#endif


#endif
