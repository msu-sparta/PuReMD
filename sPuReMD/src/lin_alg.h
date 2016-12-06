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

#ifndef __GMRES_H_
#define __GMRES_H_

#include "mytypes.h"


void Transpose( const sparse_matrix const *, sparse_matrix const * );
void Transpose_I( sparse_matrix * const );

sparse_matrix * setup_graph_coloring( sparse_matrix * const );

int GMRES( const static_storage * const, const control_params * const,
        simulation_data * const, const sparse_matrix * const,
        const real * const, const real, real * const,
        const FILE * const, const int );

int GMRES_HouseHolder( const static_storage * const, const control_params * const,
        simulation_data * const, const sparse_matrix * const,
        const real * const, const real, real * const,
        const FILE * const, const int );

int CG( static_storage*, sparse_matrix*,
        real*, real, real*, FILE* );

int SDM( static_storage*, sparse_matrix*,
         real*, real, real*, FILE* );

real condest( const sparse_matrix * const, const sparse_matrix * const );

#endif
