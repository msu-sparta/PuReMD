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

#ifndef __CUDA_INTEGRATE_H_
#define __CUDA_INTEGRATE_H_

#include "mytypes.h"


#ifdef __cplusplus
extern "C"  {
#endif

void Cuda_Velocity_Verlet_NVE( reax_system*, control_params*, simulation_data*,
        static_storage*, list**, output_controls* );
void Cuda_Velocity_Verlet_Nose_Hoover_NVT_Klein( reax_system*, control_params*,
        simulation_data*, static_storage*,
        list**, output_controls* );
void Cuda_Velocity_Verlet_Berendsen_NVT( reax_system* , control_params* ,
        simulation_data *, static_storage *,
        list **, output_controls * );

#ifdef __cplusplus
}
#endif


#endif

