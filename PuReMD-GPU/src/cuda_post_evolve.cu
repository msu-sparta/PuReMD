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

#include "cuda_post_evolve.h"

#include "vector.h"

#include "cuda_utils.h"
#include "cuda_copy.h"
#include "cuda_system_props.h"


void Cuda_Setup_Evolve( reax_system* system, control_params* control, 
        simulation_data* data, static_storage* workspace, 
        list** lists, output_controls *out_control )
{
    //fprintf (stderr, "Begin ... \n");
    //to Sync step to the device.
    //Sync_Host_Device (&data, (simulation_data *)data.d_simulation_data, cudaMemcpyHostToDevice );
    copy_host_device( &data->step, &((simulation_data *)data->d_simulation_data)->step, 
            INT_SIZE, cudaMemcpyHostToDevice, RES_SIMULATION_DATA );

}


void Cuda_Setup_Output( reax_system* system, simulation_data* data )
{
    // Here sync the simulation data, because it has been changed.
    Prep_Device_For_Output( system, data );
}


void Cuda_Sync_Temp( control_params* control )
{
    Sync_Host_Device( control, (control_params*)control->d_control, cudaMemcpyHostToDevice );
}


GLOBAL void Update_Atoms_Post_Evolve (reax_atom *atoms, simulation_data *data, int N)
{
    rvec diff, cross;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    //for( i = 0; i < system->N; i++ ) {
    // remove translational
    rvec_ScaledAdd( atoms[i].v, -1., data->vcm ); 

    // remove rotational
    rvec_ScaledSum( diff, 1., atoms[i].x, -1., data->xcm );
    rvec_Cross( cross, data->avcm, diff );
    rvec_ScaledAdd( atoms[i].v, -1., cross );
    //}
}


void Cuda_Post_Evolve( reax_system* system, control_params* control, 
        simulation_data* data, static_storage* workspace, 
        list** lists, output_controls *out_control )
{
    int i;
    rvec diff, cross;

    /* compute kinetic energy of the system */
    /*
       real *results = (real *) scratch;
       cuda_memset (results, 0, REAL_SIZE * BLOCKS_POW_2, RES_SCRATCH);
       Compute_Kinetic_Energy <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE>>>
       (system->reaxprm.d_sbp, system->d_atoms, system->N, 
       (simulation_data *)data->d_simulation_data, (real *) results);
       cudaThreadSynchronize ();
       cudaCheckError ();
     */

    //fprintf (stderr, "Cuda_Post_Evolve: Begin\n");
    Cuda_Compute_Kinetic_Energy( system, data );
    //fprintf (stderr, " Cuda_Compute_Kinetic_Energy done.... \n");

    /* remove rotational and translational velocity of the center of mass */
    if( control->ensemble != NVE && 
            control->remove_CoM_vel && 
            data->step && data->step % control->remove_CoM_vel == 0 ) {

        /*
           rvec t_xcm, t_vcm, t_avcm;
           rvec_MakeZero (t_xcm);
           rvec_MakeZero (t_vcm);
           rvec_MakeZero (t_avcm);

           rvec_Copy (t_xcm, data->xcm);
           rvec_Copy (t_vcm, data->vcm);
           rvec_Copy (t_avcm, data->avcm);
         */

        /* compute velocity of the center of mass */
        Cuda_Compute_Center_of_Mass( system, data, out_control->prs );
        //fprintf (stderr, "Cuda_Compute_Center_of_Mass done... \n");
        /*
           fprintf (stderr, "center of mass done on the device \n");

           fprintf (stderr, "xcm --> %4.10f %4.10f \n", t_xcm, data->xcm );
           fprintf (stderr, "vcm --> %4.10f %4.10f \n", t_vcm, data->vcm );
           fprintf (stderr, "avcm --> %4.10f %4.10f \n", t_avcm, data->avcm );

           if (check_zero (t_xcm, data->xcm) || 
           check_zero (t_vcm, data->vcm) ||
           check_zero (t_avcm, data->avcm)){
           fprintf (stderr, "SimulationData (xcm, vcm, avcm) does not match between device and host \n");
           exit (0);
           }
         */

        //xcm, avcm, 
        copy_host_device( data->vcm,
            ((simulation_data *)data->d_simulation_data)->vcm, RVEC_SIZE, cudaMemcpyHostToDevice, RES_SIMULATION_DATA );
        copy_host_device( data->xcm,
            ((simulation_data *)data->d_simulation_data)->xcm, RVEC_SIZE, cudaMemcpyHostToDevice, RES_SIMULATION_DATA );
        copy_host_device( data->avcm,
            ((simulation_data *)data->d_simulation_data)->avcm, RVEC_SIZE, cudaMemcpyHostToDevice, RES_SIMULATION_DATA );

        //fprintf (stderr, "data copied.... \n");

        Update_Atoms_Post_Evolve<<< BLOCKS, BLOCK_SIZE >>>
            (system->d_atoms, (simulation_data *)data->d_simulation_data, system->N);
        cudaThreadSynchronize( );
        cudaCheckError( );

        //fprintf (stderr, " Cuda_Post_Evolve:End \n");

    }
}
