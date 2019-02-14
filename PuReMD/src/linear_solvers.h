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

#ifndef __LINEAR_SOLVERS_H_
#define __LINEAR_SOLVERS_H_

#include "reax_types.h"


real setup_sparse_approx_inverse( reax_system*, simulation_data*, storage*, mpi_datatypes*, 
        sparse_matrix *, sparse_matrix **, int, double );

real sparse_approx_inverse( reax_system*, simulation_data*, storage*, mpi_datatypes*, 
        sparse_matrix*, sparse_matrix*, sparse_matrix**, int );

int dual_CG( reax_system*, control_params*, simulation_data*, storage*, sparse_matrix*,
             rvec2*, real, rvec2*, mpi_datatypes* );

int CG( reax_system*, control_params*, simulation_data*, storage*, sparse_matrix*,
        real*, real, real*, mpi_datatypes* );

int PIPECG( reax_system*, control_params*, simulation_data*, storage*, sparse_matrix*,
        real*, real, real*, mpi_datatypes* );

int PIPECR( reax_system*, control_params*, simulation_data*, storage*, sparse_matrix*,
        real*, real, real*, mpi_datatypes* );


#endif
