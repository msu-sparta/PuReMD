#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#include "cuda_copy.h"

//#include "list.h"
#include "cuda_utils.h"
#include "vector.h"

//#ifdef __cplusplus
//extern "C"  {  
//#endif

extern "C" void Make_List( int, int, int, reax_list* );
extern "C" void Delete_List( reax_list* );


void Sync_Grid( grid *host, grid *device )
{
    int total;
    grid_cell local_cell;
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
            cudaMemcpyHostToDevice, "grid:str");
    copy_host_device( host->end, device->end, sizeof(int) * total,
            cudaMemcpyHostToDevice, "grid:end");
    copy_host_device( host->cutoff, device->cutoff, sizeof(real) * total,
            cudaMemcpyHostToDevice, "grid:cutoff");
    copy_host_device( host->nbrs_x, device->nbrs_x, sizeof(ivec) * total *
            host->max_nbrs, cudaMemcpyHostToDevice, "grid:nbrs_x");
    copy_host_device( host->nbrs_cp, device->nbrs_cp, sizeof(rvec) * total *
            host->max_nbrs, cudaMemcpyHostToDevice, "grid:nbrs_cp");

    copy_host_device( host->rel_box, device->rel_box, sizeof(ivec) * total,
            cudaMemcpyHostToDevice, "grid:rel_box");

    device->max_nbrs = host->max_nbrs;

    /*
       for (int i = 0; i < total; i++) {

       copy_host_device (&local_cell, &device->cells[i], sizeof (grid_cell), cudaMemcpyDeviceToHost, "grid:cell-cuda_copy");

    //fprintf (stderr, " Atoms address %ld (%d) \n", local_cell.atoms, host->max_atoms );
    //cuda_memset (local_cell.atoms, 0, sizeof (int) * host->max_atoms, "grid:cell:atoms-memset");
    //fprintf (stderr, "host native atoms -> %d %d \n", host->native_str[0], host->native_end[0]);
    //fprintf (stderr, "host atoms -> %d \n", host->cells[i].atoms[i]);
    //fprintf (stderr, "Host Max atoms : %d \n", host->max_atoms ); 
    //copy_host_device (host->cells[i].atoms, 
    //        (local_cell.atoms), sizeof (int) * host->max_atoms, cudaMemcpyHostToDevice, "grid:cell:atoms");

    ////////////////////////////////////////////
    //No need to copy atoms from the cells from host to device. 
    // str and end has positions in the d_my_atoms list, which are just indexes into this list
    // this index is used in the cuda_neighbors to compute the neighbors. 
    // This is the only place where atoms is used. 
    ////////////////////////////////////////////////

    //fprintf (stderr, " cells:nbrs_x %ld \n", local_cell.nbrs_x);
    copy_host_device (host->cells[i].nbrs_x, 
    local_cell.nbrs_x, sizeof (ivec) * host->max_nbrs, cudaMemcpyHostToDevice, "grid:nbrs_x");

    //fprintf (stderr, " Atoms address %ld \n", local_cell.nbrs_cp);
    copy_host_device (host->cells[i].nbrs_cp, 
    local_cell.nbrs_cp, sizeof (rvec) * host->max_nbrs, cudaMemcpyHostToDevice, "grid:nbrs_cp");

    //no need to copy pointers for device->cells[i].nbrs. 
    // we can extract the pointer by nbrs_x (ivec) into the cells array. 
    // This makes nbrs member redundant on the device

    local_cell.cutoff = host->cells[i].cutoff;
    rvec_Copy (local_cell.min, host->cells[i].min);
    rvec_Copy (local_cell.max, host->cells[i].max);
    ivec_Copy (local_cell.rel_box, host->cells[i].rel_box);

    local_cell.mark = host->cells[i].mark;
    local_cell.type = host->cells[i].type;
    local_cell.str = host->cells[i].str;
    local_cell.end = host->cells[i].end;
    local_cell.top = host->cells[i].top;

    copy_host_device (&local_cell, &device->cells[i], sizeof (grid_cell), 
    cudaMemcpyHostToDevice, "grid:cell-cuda_copy");
    }
     */
}


void Sync_Atoms( reax_system *sys )
{
    //TODO METIN FIX, coredump on his machine
    //copy_host_device( sys->my_atoms, sys->d_my_atoms, sizeof(reax_atom) * sys->total_cap, cudaMemcpyHostToDevice, "system:my_atoms" );

#if defined(__CUDA_DEBUG_LOG__)
    fprintf( stderr, "p:%d - Synching atoms: n: %d N: %d, total_cap: %d \n", 
            sys->my_rank, sys->n, sys->N, sys->total_cap );
#endif

    copy_host_device( sys->my_atoms, sys->d_my_atoms, sizeof(reax_atom) *
            sys->N, cudaMemcpyHostToDevice, "system:my_atoms" );
    //TODO METIN FIX, coredump on his machine
}


void Sync_System( reax_system *sys )
{
    //fprintf (stderr, "p:%d - trying to copy atoms : %d \n", sys->my_rank, sys->local_cap);
    Sync_Atoms (sys);

    copy_host_device (&(sys->my_box), sys->d_my_box, sizeof(simulation_box),
            cudaMemcpyHostToDevice, "system:my_box");

    copy_host_device (&(sys->my_ext_box), sys->d_my_ext_box,
            sizeof(simulation_box), cudaMemcpyHostToDevice,
            "system:my_ext_box");

    copy_host_device (sys->reax_param.sbp, sys->reax_param.d_sbp,
            sizeof(single_body_parameters) * sys->reax_param.num_atom_types,
            cudaMemcpyHostToDevice, "system:sbp");
    copy_host_device (sys->reax_param.tbp, sys->reax_param.d_tbp,
            sizeof(two_body_parameters) * pow (sys->reax_param.num_atom_types,
                2), cudaMemcpyHostToDevice, "system:tbp");
    copy_host_device (sys->reax_param.thbp, sys->reax_param.d_thbp,
            sizeof(three_body_header) * pow (sys->reax_param.num_atom_types,
                3), cudaMemcpyHostToDevice, "system:thbh");
    copy_host_device (sys->reax_param.hbp, sys->reax_param.d_hbp,
            sizeof(hbond_parameters) * pow (sys->reax_param.num_atom_types, 3),
            cudaMemcpyHostToDevice, "system:hbond");
    copy_host_device (sys->reax_param.fbp, sys->reax_param.d_fbp, 
            sizeof(four_body_header) * pow (sys->reax_param.num_atom_types, 4),
            cudaMemcpyHostToDevice, "system:four_header");

    copy_host_device (sys->reax_param.gp.l, sys->reax_param.d_gp.l,
            sizeof(real) * sys->reax_param.gp.n_global, cudaMemcpyHostToDevice,
            "system:global_parameters");

    sys->reax_param.d_gp.n_global = sys->reax_param.gp.n_global; 
    sys->reax_param.d_gp.vdw_type = sys->reax_param.gp.vdw_type; 
}


void Output_Sync_Atoms( reax_system *sys )
{
    //TODO changed this from sys->n to sys->N
    copy_host_device( sys->my_atoms, sys->d_my_atoms, sizeof(reax_atom) *
            sys->total_cap, cudaMemcpyDeviceToHost, "system:my_atoms" );
}


void Output_Sync_Simulation_Data( simulation_data *host, simulation_data *dev )
{
    copy_host_device (&host->my_en, &dev->my_en, sizeof(energy_data), 
            cudaMemcpyDeviceToHost, "simulation_data:energy_data");
    copy_host_device (&host->kin_press, &dev->kin_press, sizeof(real), 
            cudaMemcpyDeviceToHost, "simulation_data:kin_press");
    copy_host_device (host->int_press, dev->int_press, sizeof(rvec), 
            cudaMemcpyDeviceToHost, "simulation_data:int_press");
    copy_host_device (host->ext_press, dev->ext_press, sizeof(rvec), 
            cudaMemcpyDeviceToHost, "simulation_data:ext_press");
}


void Output_Sync_Lists( reax_list *host, reax_list *device, int type )
{
    //fprintf (stderr, " Trying to copy *%d* list from device to host \n", type);

    //list is already allocated -- discard it first
    //if (host->n > 0)
    //if (host->allocated > 0)
    //  Delete_List (host);

    //memory is allocated on the host
    //Make_List(device->n, device->num_intrs, type, host);

    //memcpy the entries from device to host
    copy_host_device (host->index, device->index, sizeof(int) * device->n,
            cudaMemcpyDeviceToHost, "output_sync_list:list:index");
    copy_host_device (host->end_index, device->end_index, sizeof(int) *
            device->n, cudaMemcpyDeviceToHost, "output_sync:list:end_index");

    switch (type)
    {   
        case TYP_BOND:
            copy_host_device (host->select.bond_list, device->select.bond_list,
                    sizeof(bond_data) * device->num_intrs,
                    cudaMemcpyDeviceToHost, "bond_list");
            break;

        case TYP_THREE_BODY:
            copy_host_device (host->select.three_body_list,
                    device->select.three_body_list,
                    sizeof(three_body_interaction_data )* device->num_intrs,
                    cudaMemcpyDeviceToHost, "three_body_list");
            break;

        default:
            fprintf( stderr, "Unknown list synching from device to host ---- > %d \n", type );
            exit( 1 );
            break;
    }  
}

//#ifdef __cplusplus
//}
//#endif
