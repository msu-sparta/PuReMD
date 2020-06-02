
#include "cuda_copy.h"

#include "cuda_utils.h"

#include "../list.h"
#include "../vector.h"


/* Copy grid info from host to device */
void Cuda_Copy_Grid_Host_to_Device( grid *host, grid *device )
{
    int total;

    total = host->ncells[0] * host->ncells[1] * host->ncells[2];

    ivec_Copy( device->ncells, host->ncells );
    rvec_Copy( device->cell_len, host->cell_len );
    rvec_Copy( device->inv_len, host->inv_len );

    ivec_Copy( device->bond_span, host->bond_span );
    ivec_Copy( device->nonb_span, host->nonb_span );
    ivec_Copy( device->vlist_span, host->vlist_span );

    ivec_Copy( device->native_cells, host->native_cells );
    ivec_Copy( device->native_str, host->native_str );
    ivec_Copy( device->native_end, host->native_end );

    device->ghost_cut = host->ghost_cut;
    ivec_Copy( device->ghost_span, host->ghost_span );
    ivec_Copy( device->ghost_nonb_span, host->ghost_nonb_span );
    ivec_Copy( device->ghost_hbond_span, host->ghost_hbond_span );
    ivec_Copy( device->ghost_bond_span, host->ghost_bond_span );

    copy_host_device( host->str, device->str, sizeof(int) * total,
            cudaMemcpyHostToDevice, "grid:str" );
    copy_host_device( host->end, device->end, sizeof(int) * total,
            cudaMemcpyHostToDevice, "grid:end" );
    copy_host_device( host->cutoff, device->cutoff, sizeof(real) * total,
            cudaMemcpyHostToDevice, "grid:cutoff" );
    copy_host_device( host->nbrs_x, device->nbrs_x, sizeof(ivec) * total
            * host->max_nbrs, cudaMemcpyHostToDevice, "grid:nbrs_x" );
    copy_host_device( host->nbrs_cp, device->nbrs_cp, sizeof(rvec) * total
            * host->max_nbrs, cudaMemcpyHostToDevice, "grid:nbrs_cp" );

    copy_host_device( host->rel_box, device->rel_box, sizeof(ivec) * total,
            cudaMemcpyHostToDevice, "grid:rel_box" );

    device->max_nbrs = host->max_nbrs;
}


/* Copy atom info from host to device */
void Cuda_Copy_Atoms_Host_to_Device( reax_system *system )
{
    copy_host_device( system->my_atoms, system->d_my_atoms,
            sizeof(reax_atom) * system->N,
            cudaMemcpyHostToDevice, "Cuda_Copy_Atoms_Host_to_Device::system->my_atoms" );
}


/* Copy atomic system info from host to device */
void Cuda_Copy_System_Host_to_Device( reax_system *system )
{
    Cuda_Copy_Atoms_Host_to_Device( system );

    copy_host_device( &system->my_box, system->d_my_box, sizeof(simulation_box),
            cudaMemcpyHostToDevice, "Cuda_Copy_System_Host_to_Device::system->my_box" );

    copy_host_device( &system->my_ext_box, system->d_my_ext_box,
            sizeof(simulation_box), cudaMemcpyHostToDevice,
            "Cuda_Copy_System_Host_to_Device::system->my_ext_box" );

    copy_host_device( system->reax_param.sbp, system->reax_param.d_sbp,
            sizeof(single_body_parameters) * system->reax_param.num_atom_types,
            cudaMemcpyHostToDevice, "Cuda_Copy_System_Host_to_Device::system->sbp" );
    copy_host_device( system->reax_param.tbp, system->reax_param.d_tbp,
            sizeof(two_body_parameters) * POW(system->reax_param.num_atom_types, 2),
            cudaMemcpyHostToDevice, "Cuda_Copy_System_Host_to_Device::system->tbp" );
    copy_host_device( system->reax_param.thbp, system->reax_param.d_thbp,
            sizeof(three_body_header) * POW(system->reax_param.num_atom_types, 3),
            cudaMemcpyHostToDevice, "Cuda_Copy_System_Host_to_Device::system->thbh" );
    copy_host_device( system->reax_param.hbp, system->reax_param.d_hbp,
            sizeof(hbond_parameters) * POW(system->reax_param.num_atom_types, 3),
            cudaMemcpyHostToDevice, "Cuda_Copy_System_Host_to_Device::system->hbond" );
    copy_host_device( system->reax_param.fbp, system->reax_param.d_fbp, 
            sizeof(four_body_header) * POW(system->reax_param.num_atom_types, 4),
            cudaMemcpyHostToDevice, "Cuda_Copy_System_Host_to_Device::system->four_header" );

    copy_host_device( system->reax_param.gp.l, system->reax_param.d_gp.l,
            sizeof(real) * system->reax_param.gp.n_global, cudaMemcpyHostToDevice,
            "Cuda_Copy_System_Host_to_Device::system->global_parameters" );

    system->reax_param.d_gp.n_global = system->reax_param.gp.n_global; 
    system->reax_param.d_gp.vdw_type = system->reax_param.gp.vdw_type; 
}


/* Copy atom info from device to host */
void Cuda_Copy_Atoms_Device_to_Host( reax_system *system )
{
    copy_host_device( system->my_atoms, system->d_my_atoms,
            sizeof(reax_atom) * system->N,
            cudaMemcpyDeviceToHost, "Cuda_Copy_Atoms_Device_to_Host::my_atoms" );
}


/* Copy simulation data from device to host */
void Output_Sync_Simulation_Data( simulation_data *host, simulation_data *dev )
{
    copy_host_device( &host->my_en, &dev->my_en, sizeof(energy_data), 
            cudaMemcpyDeviceToHost, "simulation_data:energy_data" );
    copy_host_device( &host->kin_press, &dev->kin_press, sizeof(real), 
            cudaMemcpyDeviceToHost, "simulation_data:kin_press" );
    copy_host_device( host->int_press, dev->int_press, sizeof(rvec), 
            cudaMemcpyDeviceToHost, "simulation_data:int_press" );
    copy_host_device( host->ext_press, dev->ext_press, sizeof(rvec), 
            cudaMemcpyDeviceToHost, "simulation_data:ext_press" );
}


/* Copy interaction lists from device to host,
 * with allocation for the host list */
void Output_Sync_Lists( reax_list *host_list, reax_list *device_list, int type )
{
    int format;

//    assert( device_list != NULL );
//    assert( device_list->allocated == TRUE );

    format = host_list->format;

    if ( host_list != NULL && host_list->allocated == TRUE )
    {
        Delete_List( host_list );
    }
    Make_List( device_list->n, device_list->max_intrs, type, format, host_list );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, " [INFO] trying to copy %d list from device to host\n", type );
#endif

    copy_host_device( host_list->index, device_list->index, sizeof(int) * device_list->n,
            cudaMemcpyDeviceToHost, "Output_Sync_Lists::list->index" );
    copy_host_device( host_list->end_index, device_list->end_index, sizeof(int) *
            device_list->n, cudaMemcpyDeviceToHost, "Output_Sync_Lists::list->end_index" );

    switch ( type )
    {   
        case TYP_FAR_NEIGHBOR:
            copy_host_device( host_list->far_nbr_list.nbr, device_list->far_nbr_list.nbr,
                    sizeof(int) * device_list->max_intrs,
                    cudaMemcpyDeviceToHost, "Output_Sync_Lists::far_neighbor_list.nbr" );
            copy_host_device( host_list->far_nbr_list.rel_box, device_list->far_nbr_list.rel_box,
                    sizeof(ivec) * device_list->max_intrs,
                    cudaMemcpyDeviceToHost, "Output_Sync_Lists::far_neighbor_list.rel_box" );
            copy_host_device( host_list->far_nbr_list.d, device_list->far_nbr_list.d,
                    sizeof(real) * device_list->max_intrs,
                    cudaMemcpyDeviceToHost, "Output_Sync_Lists::far_neighbor_list.d" );
            copy_host_device( host_list->far_nbr_list.dvec, device_list->far_nbr_list.dvec,
                    sizeof(rvec) * device_list->max_intrs,
                    cudaMemcpyDeviceToHost, "Output_Sync_Lists::far_neighbor_list.dvec" );
            break;

        case TYP_BOND:
            copy_host_device( host_list->bond_list, device_list->bond_list,
                    sizeof(bond_data) * device_list->max_intrs,
                    cudaMemcpyDeviceToHost, "Output_Sync_Lists::bond_list" );
            break;

        case TYP_HBOND:
            copy_host_device( host_list->hbond_list, device_list->hbond_list,
                    sizeof(hbond_data) * device_list->max_intrs,
                    cudaMemcpyDeviceToHost, "Output_Sync_Lists::hbond_list" );
            break;

        case TYP_THREE_BODY:
            copy_host_device( host_list->three_body_list,
                    device_list->three_body_list,
                    sizeof(three_body_interaction_data ) * device_list->max_intrs,
                    cudaMemcpyDeviceToHost, "Output_Sync_Lists::three_body_list" );
            break;

        default:
            fprintf( stderr, "[ERROR] Unknown list synching from device to host (%d)\n",
                    type );
            exit( INVALID_INPUT );
            break;
    }  
}
