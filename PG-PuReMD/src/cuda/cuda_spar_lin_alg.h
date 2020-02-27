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

#ifndef __CUDA_LIN_ALG_H_
#define __CUDA_LIN_ALG_H_

#include "../reax_types.h"


#ifdef __cplusplus
extern "C" {
#endif

int Cuda_dual_CG( reax_system const * const, control_params const * const,
        simulation_data * const, storage * const,
        sparse_matrix const * const, rvec2 const * const, real,
        rvec2 * const, mpi_datatypes * const, FILE * );

int Cuda_CG( reax_system const * const, control_params const * const,
        simulation_data * const, storage * const,
        sparse_matrix const * const, real const * const, real,
        real * const, mpi_datatypes * const );

#ifdef __cplusplus
}
#endif


#endif
