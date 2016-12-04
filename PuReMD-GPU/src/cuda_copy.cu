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

#include "cuda_copy.h"

#include "cuda_list.h"

#include "vector.h"


void Sync_Host_Device( grid *host, grid *dev, enum cudaMemcpyKind dir )
{
    copy_host_device( host->top, dev->top, 
            INT_SIZE * host->ncell[0]*host->ncell[1]*host->ncell[2], dir, RES_GRID_TOP );

    copy_host_device( host->mark, dev->mark, 
            INT_SIZE * host->ncell[0]*host->ncell[1]*host->ncell[2], dir, RES_GRID_MARK );

    copy_host_device( host->start, dev->start, 
            INT_SIZE * host->ncell[0]*host->ncell[1]*host->ncell[2], dir, RES_GRID_START );

    copy_host_device( host->end, dev->end, 
            INT_SIZE * host->ncell[0]*host->ncell[1]*host->ncell[2], dir, RES_GRID_END );

    copy_host_device( host->atoms, dev->atoms, 
            INT_SIZE * host->max_atoms*host->ncell[0]*host->ncell[1]*host->ncell[2], dir, RES_GRID_ATOMS );

    copy_host_device( host->nbrs, dev->nbrs, 
            IVEC_SIZE * host->max_nbrs*host->ncell[0]*host->ncell[1]*host->ncell[2], dir, RES_GRID_NBRS );

    copy_host_device( host->nbrs_cp, dev->nbrs_cp, 
            RVEC_SIZE * host->max_nbrs*host->ncell[0]*host->ncell[1]*host->ncell[2], dir, RES_GRID_NBRS_CP );
}


void Sync_Host_Device( reax_system *sys, enum cudaMemcpyKind dir )
{

    copy_host_device( sys->atoms, sys->d_atoms, 
            REAX_ATOM_SIZE * sys->N, dir, RES_SYSTEM_ATOMS );

    copy_host_device( &(sys->box), sys->d_box, SIMULATION_BOX_SIZE, dir, RES_SYSTEM_SIMULATION_BOX );

    //synch bonds here.
    copy_host_device( sys->reaxprm.sbp, sys->reaxprm.d_sbp, SBP_SIZE * sys->reaxprm.num_atom_types, 
            dir, RES_REAX_INT_SBP );
    copy_host_device( sys->reaxprm.tbp, sys->reaxprm.d_tbp, TBP_SIZE * pow (sys->reaxprm.num_atom_types, 2), 
            dir, RES_REAX_INT_TBP );
    copy_host_device( sys->reaxprm.thbp, sys->reaxprm.d_thbp, THBP_SIZE * pow (sys->reaxprm.num_atom_types, 3), 
            dir, RES_REAX_INT_THBP );
    copy_host_device( sys->reaxprm.hbp, sys->reaxprm.d_hbp, HBP_SIZE * pow (sys->reaxprm.num_atom_types, 3), 
            dir, RES_REAX_INT_HBP );
    copy_host_device( sys->reaxprm.fbp, sys->reaxprm.d_fbp, FBP_SIZE * pow (sys->reaxprm.num_atom_types, 4),
            dir, RES_REAX_INT_FBP );

    copy_host_device( sys->reaxprm.gp.l, sys->reaxprm.d_gp.l, REAL_SIZE * sys->reaxprm.gp.n_global, 
            dir, RES_GLOBAL_PARAMS );

    sys->reaxprm.d_gp.n_global = sys->reaxprm.gp.n_global; 
    sys->reaxprm.d_gp.vdw_type = sys->reaxprm.gp.vdw_type; 
}


void Sync_Host_Device( simulation_data *host, simulation_data *dev, enum cudaMemcpyKind dir )
{
    copy_host_device( host, dev, SIMULATION_DATA_SIZE, dir, RES_SIMULATION_DATA );
}


void Sync_Host_Device( sparse_matrix *L, sparse_matrix *U, enum cudaMemcpyKind dir )
{
    copy_host_device( L->start, dev_workspace->L.start, INT_SIZE * (L->n + 1), dir, RES_SPARSE_MATRIX_INDEX );
    copy_host_device( L->end, dev_workspace->L.end, INT_SIZE * (L->n + 1), dir, RES_SPARSE_MATRIX_INDEX );
    copy_host_device( L->entries, dev_workspace->L.entries, SPARSE_MATRIX_ENTRY_SIZE * L->m, dir, RES_SPARSE_MATRIX_ENTRY );

    copy_host_device( U->start, dev_workspace->U.start, INT_SIZE * (U->n + 1), dir, RES_SPARSE_MATRIX_INDEX );
    copy_host_device( U->end, dev_workspace->U.end, INT_SIZE * (U->n + 1), dir, RES_SPARSE_MATRIX_INDEX );
    copy_host_device( U->entries, dev_workspace->U.entries, SPARSE_MATRIX_ENTRY_SIZE * U->m, dir, RES_SPARSE_MATRIX_ENTRY );
}


void Sync_Host_Device( output_controls *, control_params *, enum cudaMemcpyKind )
{
}


void Sync_Host_Device( control_params *host, control_params *device, enum cudaMemcpyKind )
{
    copy_host_device( host, device, CONTROL_PARAMS_SIZE, cudaMemcpyHostToDevice, RES_CONTROL_PARAMS );
}


void Prep_Device_For_Output( reax_system *system, simulation_data *data )
{
    //int size = sizeof (simulation_data) - (2*sizeof (reax_timing) + sizeof (void *));
    //unsigned long start_address = (unsigned long)data->d_simulation_data + (unsigned long) (2 * INT_SIZE + REAL_SIZE);

    //fprintf (stderr, "Address of Simulation data (address) --> %ld \n", data->d_simulation_data );
    //fprintf (stderr, "Size of simulation_data --> %d \n", sizeof (simulation_data));
    //fprintf (stderr, "size to copy --> %d \n", size );
    //copy_host_device (data, (simulation_data *)data->d_simulation_data, size, cudaMemcpyDeviceToHost, RES_SIMULATION_DATA );

    //Sync_Host_Device (data, (simulation_data *)data->d_simulation_data, cudaMemcpyDeviceToHost );
    /*
       copy_host_device (&data->E_BE, &((simulation_data *)data->d_simulation_data)->E_BE, 
       REAL_SIZE * 13, cudaMemcpyDeviceToHost, RES_SIMULATION_DATA );
       copy_host_device (&data->E_Kin, &((simulation_data *)data->d_simulation_data)->E_Kin, 
       REAL_SIZE, cudaMemcpyDeviceToHost, RES_SIMULATION_DATA );
       copy_host_device (&data->int_press, &((simulation_data *)data->d_simulation_data)->int_press, 
       3*(RVEC_SIZE) + REAL_SIZE, cudaMemcpyDeviceToHost, RES_SIMULATION_DATA );

       copy_host_device (&data->therm.T, &((simulation_data *)data->d_simulation_data)->therm.T, 
       REAL_SIZE, cudaMemcpyDeviceToHost, RES_SIMULATION_DATA );
     */

    simulation_data local_data;
    copy_host_device( &local_data, (simulation_data *)data->d_simulation_data, 
            SIMULATION_DATA_SIZE, cudaMemcpyDeviceToHost, RES_SIMULATION_DATA );
    data->E_BE = local_data.E_BE;
    data->E_Ov = local_data.E_Ov;
    data->E_Un = local_data.E_Un;
    data->E_Lp = local_data.E_Lp;
    data->E_Ang = local_data.E_Ang;
    data->E_Pen = local_data.E_Pen;
    data->E_Coa = local_data.E_Coa;
    data->E_HB = local_data.E_HB;
    data->E_Tor = local_data.E_Tor;
    data->E_Con = local_data.E_Con;
    data->E_vdW = local_data.E_vdW;
    data->E_Ele = local_data.E_Ele;
    data->E_Kin = local_data.E_Kin;
    rvec_Copy( data->int_press, local_data.int_press);
    rvec_Copy( data->ext_press, local_data.ext_press);
    data->kin_press =  local_data.kin_press;
    data->therm.T = local_data.therm.T;

    //Sync_Host_Device (&system.g, &system.d_g, cudaMemcpyDeviceToHost );
    Sync_Host_Device( system, cudaMemcpyDeviceToHost );
}


void Sync_Host_Device( list *host, list *device, int type )
{
    //list is already allocated -- discard it first
    if (host->n > 0)
        Cuda_Delete_List( host );

    //memory is allocated on the host
    Cuda_Make_List( device->n, device->num_intrs, type, host );

    //memcpy the entries from device to host
    copy_host_device( host->index, device->index, INT_SIZE * device->n, cudaMemcpyDeviceToHost, LIST_INDEX );
    copy_host_device( host->end_index, device->end_index, INT_SIZE * device->n, cudaMemcpyDeviceToHost, LIST_END_INDEX );

    switch (type)
    {
        case TYP_BOND:
            copy_host_device( host->select.bond_list, device->select.bond_list, 
                    BOND_DATA_SIZE * device->num_intrs, cudaMemcpyDeviceToHost, LIST_BOND_DATA );
            break;

        case TYP_THREE_BODY:
            copy_host_device( host->select.three_body_list, device->select.three_body_list, 
                    sizeof( three_body_interaction_data ) * device->num_intrs, cudaMemcpyDeviceToHost, LIST_THREE_BODY_DATA );
            break;

        default:
            fprintf( stderr, "Unknown list synching from device to host ---- > %d \n", type );
            exit( 1 );
            break;
    }
}
