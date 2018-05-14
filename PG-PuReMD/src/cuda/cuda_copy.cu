
#include "cuda_copy.h"

#include "cuda_utils.h"

#include "../list.h"
#include "../vector.h"


/* Copy grid info from host to device */
void Sync_Grid( grid *host, grid *device )
{
    int total;

    total = host->ncells[0] * host->ncells[1] * host->ncells[2];

    ivec_Copy( device->ncells, host->ncells);
    rvec_Copy( device->cell_len, host->cell_len);
    rvec_Copy( device->inv_len, host->inv_len);

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
    copy_host_device( host->nbrs_x, device->nbrs_x, sizeof(ivec) * total *
            host->max_nbrs, cudaMemcpyHostToDevice, "grid:nbrs_x" );
    copy_host_device( host->nbrs_cp, device->nbrs_cp, sizeof(rvec) * total *
            host->max_nbrs, cudaMemcpyHostToDevice, "grid:nbrs_cp" );

    copy_host_device( host->rel_box, device->rel_box, sizeof(ivec) * total,
            cudaMemcpyHostToDevice, "grid:rel_box" );

    device->max_nbrs = host->max_nbrs;
}


/* Copy atom info from host to device */
void Sync_Atoms( reax_system *sys )
{
    //TODO METIN FIX, coredump on his machine
//    copy_host_device( sys->my_atoms, sys->d_my_atoms, sizeof(reax_atom) * sys->total_cap,
//            cudaMemcpyHostToDevice, "Sync_Atoms::system->my_atoms" );

#if defined(__CUDA_DEBUG_LOG__)
    fprintf( stderr, "p:%d - Synching atoms: n: %d N: %d, total_cap: %d \n", 
            sys->my_rank, sys->n, sys->N, sys->total_cap );
#endif

    copy_host_device( sys->my_atoms, sys->d_my_atoms, sizeof(reax_atom) * sys->N,
            cudaMemcpyHostToDevice, "Sync_Atoms::system->my_atoms" );
    //TODO METIN FIX, coredump on his machine
}


/* Copy atomic system info from host to device */
void Sync_System( reax_system *sys )
{
    Sync_Atoms( sys );

    copy_host_device( &sys->my_box, sys->d_my_box, sizeof(simulation_box),
            cudaMemcpyHostToDevice, "Sync_System::system->my_box" );

    copy_host_device( &sys->my_ext_box, sys->d_my_ext_box,
            sizeof(simulation_box), cudaMemcpyHostToDevice,
            "Sync_System::system->my_ext_box" );

    copy_host_device( sys->reax_param.sbp, sys->reax_param.d_sbp,
            sizeof(single_body_parameters) * sys->reax_param.num_atom_types,
            cudaMemcpyHostToDevice, "Sync_System::system->sbp" );
    copy_host_device( sys->reax_param.tbp, sys->reax_param.d_tbp,
            sizeof(two_body_parameters) * POW(sys->reax_param.num_atom_types, 2),
            cudaMemcpyHostToDevice, "Sync_System::system->tbp" );
    copy_host_device( sys->reax_param.thbp, sys->reax_param.d_thbp,
            sizeof(three_body_header) * POW(sys->reax_param.num_atom_types, 3),
            cudaMemcpyHostToDevice, "Sync_System::system->thbh" );
    copy_host_device( sys->reax_param.hbp, sys->reax_param.d_hbp,
            sizeof(hbond_parameters) * POW(sys->reax_param.num_atom_types, 3),
            cudaMemcpyHostToDevice, "Sync_System::system->hbond" );
    copy_host_device( sys->reax_param.fbp, sys->reax_param.d_fbp, 
            sizeof(four_body_header) * POW(sys->reax_param.num_atom_types, 4),
            cudaMemcpyHostToDevice, "Sync_System::system->four_header" );

    copy_host_device( sys->reax_param.gp.l, sys->reax_param.d_gp.l,
            sizeof(real) * sys->reax_param.gp.n_global, cudaMemcpyHostToDevice,
            "Sync_System::system->global_parameters" );

    sys->reax_param.d_gp.n_global = sys->reax_param.gp.n_global; 
    sys->reax_param.d_gp.vdw_type = sys->reax_param.gp.vdw_type; 
}


/* Copy atom info from device to host */
void Output_Sync_Atoms( reax_system *sys )
{
    copy_host_device( sys->my_atoms, sys->d_my_atoms, sizeof(reax_atom) *
            sys->total_cap, cudaMemcpyDeviceToHost, "system:my_atoms" );
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


/* Copy interaction lists from device to host */
void Output_Sync_Lists( reax_list *host, reax_list *device, int type )
{
#if defined(DEBUG)
    fprintf( stderr, " Trying to copy *%d* list from device to host \n", type );
#endif

    if ( host->allocated == TRUE )
    {
        Delete_List( host );
    }
    Make_List( device->n, device->max_intrs, type, host );

    copy_host_device( host->index, device->index, sizeof(int) * device->n,
            cudaMemcpyDeviceToHost, "Output_Sync_Lists::list->index" );
    copy_host_device( host->end_index, device->end_index, sizeof(int) *
            device->n, cudaMemcpyDeviceToHost, "Output_Sync_Lists::list->end_index" );

    switch ( type )
    {   
        case TYP_FAR_NEIGHBOR:
            copy_host_device( host->select.far_nbr_list, device->select.far_nbr_list,
                    sizeof(far_neighbor_data) * device->max_intrs,
                    cudaMemcpyDeviceToHost, "Output_Sync_Lists::far_neighbor_list" );
            break;

        case TYP_BOND:
            copy_host_device( host->select.bond_list, device->select.bond_list,
                    sizeof(bond_data) * device->max_intrs,
                    cudaMemcpyDeviceToHost, "Output_Sync_Lists::bond_list" );
            break;

        case TYP_HBOND:
            copy_host_device( host->select.hbond_list, device->select.hbond_list,
                    sizeof(hbond_data) * device->max_intrs,
                    cudaMemcpyDeviceToHost, "Output_Sync_Lists::hbond_list" );
            break;

        case TYP_THREE_BODY:
            copy_host_device( host->select.three_body_list,
                    device->select.three_body_list,
                    sizeof(three_body_interaction_data )* device->max_intrs,
                    cudaMemcpyDeviceToHost, "Output_Sync_Lists::three_body_list" );
            break;

        default:
            fprintf( stderr, "Unknown list synching from device to host ---- > %d \n",
                    type );
            exit( INVALID_INPUT );
            break;
    }  
}
