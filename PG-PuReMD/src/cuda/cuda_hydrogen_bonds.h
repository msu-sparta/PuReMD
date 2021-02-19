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

#ifndef __CUDA_HBONDS_H_
#define __CUDA_HBONDS_H_

#include "../reax_types.h"


CUDA_GLOBAL void Cuda_Hydrogen_Bonds_Part1( reax_atom *, single_body_parameters *,
        hbond_parameters *, global_parameters,
        control_params *, storage ,
        reax_list, reax_list, reax_list, int,
        int, real *, rvec *, int );

CUDA_GLOBAL void Cuda_Hydrogen_Bonds_Part1_opt( reax_atom *, single_body_parameters *,
        hbond_parameters *, global_parameters, control_params *, storage,
        reax_list, reax_list, reax_list, int,
        int, real *, rvec * );

#if !defined(CUDA_ACCUM_ATOMIC)
CUDA_GLOBAL void Cuda_Hydrogen_Bonds_Part2( reax_atom *,
        storage, reax_list, int );

CUDA_GLOBAL void Cuda_Hydrogen_Bonds_Part2_opt( reax_atom *,
        storage, reax_list, int );

CUDA_GLOBAL void Cuda_Hydrogen_Bonds_Part3( reax_atom *,
        storage, reax_list, int );

CUDA_GLOBAL void Cuda_Hydrogen_Bonds_Part3_opt( reax_atom *,
        storage, reax_list, int );
#endif


#endif
