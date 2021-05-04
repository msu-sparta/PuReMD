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

#include "cuda_box.h"

#include "cuda_integrate.h"
#include "cuda_system_props.h"
#include "cuda_utils.h"

#include "../box.h"
#include "../comm_tools.h"


void Cuda_Scale_Box( reax_system *system, control_params *control,
        storage *workspace, simulation_data *data, mpi_datatypes *mpi_data )
{
    int d;
    real dt, lambda;
    rvec mu = {0.0, 0.0, 0.0};

    dt = control->dt;

    /* pressure scaler */
    if ( control->ensemble == iNPT )
    {
        mu[0] = POW( 1.0 + (dt / control->Tau_P[0]) * (data->iso_bar.P - control->P[0]),
                1.0 / 3.0 );

        if ( mu[0] < MIN_dV )
        {
            mu[0] = MIN_dV;
        }
        else if ( mu[0] > MAX_dV )
        {
            mu[0] = MAX_dV;
        }

        mu[1] = mu[0];
        mu[2] = mu[1];
    }
    else if ( control->ensemble == sNPT )
    {
        for ( d = 0; d < 3; ++d )
        {
            mu[d] = POW(1.0 + (dt / control->Tau_P[d]) * (data->tot_press[d] - control->P[d]),
                        1. / 3 );

            if ( mu[d] < MIN_dV )
            {
                mu[d] = MIN_dV;
            }
            else if ( mu[d] > MAX_dV )
            {
                mu[d] = MAX_dV;
            }
        }
    }

    /* temperature scaler */
    lambda = 1.0 + (dt / control->Tau_T) * (control->T / data->therm.T - 1.0);
    if ( lambda < MIN_dT )
    {
        lambda = MIN_dT;
    }
    else if (lambda > MAX_dT )
    {
        lambda = MAX_dT;
    }
    lambda = SQRT( lambda );

    /* Scale velocities and positions at t+dt */
    Cuda_Scale_Velocities_NPT( system, control, lambda, mu );

    Cuda_Compute_Kinetic_Energy( system, control, workspace, data, mpi_data->comm_mesh3D );

    /* update box & grid */
    system->big_box.box[0][0] *= mu[0];
    system->big_box.box[1][1] *= mu[1];
    system->big_box.box[2][2] *= mu[2];

    Make_Consistent( &system->big_box );
    Setup_My_Box( system, control );
    Setup_My_Ext_Box( system, control );
    Update_Comm( system );

    sCudaMemcpy( &system->d_big_box, &system->big_box,
            sizeof(simulation_box), cudaMemcpyHostToDevice, __FILE__, __LINE__ );
    sCudaMemcpy( &system->d_my_box, &system->my_box,
            sizeof(simulation_box), cudaMemcpyHostToDevice, __FILE__, __LINE__ );
    sCudaMemcpy( &system->d_my_ext_box, &system->my_ext_box,
            sizeof(simulation_box), cudaMemcpyHostToDevice, __FILE__, __LINE__ );
}
