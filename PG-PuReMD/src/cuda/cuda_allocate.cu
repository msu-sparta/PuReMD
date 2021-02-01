
#include "cuda_allocate.h"

#include "cuda_allocate.h"
#include "cuda_forces.h"
#include "cuda_list.h"
#include "cuda_neighbors.h"
#include "cuda_utils.h"

#include "../allocate.h"
#include "../index_utils.h"
#include "../tool_box.h"
#include "../vector.h"


CUDA_GLOBAL void k_init_nbrs( ivec *nbrs, int N )
{
    int i;
   
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    nbrs[i][0] = -1; 
    nbrs[i][1] = -1; 
    nbrs[i][2] = -1; 
}


void Cuda_Allocate_Control( control_params *control )
{
    cuda_malloc( (void **)&control->d_control_params,
            sizeof(control_params), TRUE, "control_params" );
    copy_host_device( control, control->d_control_params,
            sizeof(control_params), cudaMemcpyHostToDevice, "control_params" );
}


void Cuda_Allocate_Grid( reax_system *system )
{
    int total;
//    grid_cell local_cell;
    grid *host = &system->my_grid;
    grid *device = &system->d_my_grid;
//    ivec *nbrs_x = (ivec *) workspace->scratch;

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

    cuda_malloc( (void **) &device->str, sizeof(int) * total, TRUE,
            "Cuda_Allocate_Grid::grid->str" );
    cuda_malloc( (void **) &device->end, sizeof(int) * total, TRUE,
            "Cuda_Allocate_Grid::grid->end" );
    cuda_malloc( (void **) &device->cutoff, sizeof(real) * total, TRUE,
            "Cuda_Allocate_Grid::grid->cutoff" );

    cuda_malloc( (void **) &device->nbrs_x, sizeof(ivec) * total * host->max_nbrs,
            TRUE, "Cuda_Allocate_Grid::grid->nbrs_x" );
    cuda_malloc( (void **) &device->nbrs_cp, sizeof(rvec) * total * host->max_nbrs,
            TRUE, "Cuda_Allocate_Grid::grid->nbrs_cp" );
    cuda_malloc( (void **) &device->rel_box, sizeof(ivec) * total,
            TRUE, "Cuda_Allocate_Grid::grid->rel_box" );

//    int block_size = 512;
//    int blocks = (host->max_nbrs) / block_size + ((host->max_nbrs) % block_size == 0 ? 0 : 1); 
//
//    k_init_nbrs <<< blocks, block_size >>>
//        ( nbrs_x, host->max_nbrs );
//    cudaCheckError( );
//
//    cuda_malloc( (void **)& device->cells, sizeof(grid_cell) * total,
//            TRUE, "grid:cells");
//    fprintf( stderr, " Device cells address --> %ld \n", device->cells );
//    cuda_malloc( (void **) &device->order,
//            sizeof(ivec) * (host->total + 1), TRUE, "grid:order" );
//
//    local_cell.top = local_cell.mark = local_cell.str = local_cell.end = 0;
//    fprintf( stderr, "Total cells to be allocated -- > %d \n", total );
//    for (int i = 0; i < total; i++)
//    {
//        //fprintf( stderr, "Address of the local atom -> %ld  \n", &local_cell );
//
//        cuda_malloc( (void **) &local_cell.atoms, sizeof(int) * host->max_atoms,
//                TRUE, "alloc:grid:cells:atoms" );
//        //fprintf( stderr, "Allocated address of the atoms --> %ld  (%d)\n", local_cell.atoms, host->max_atoms );
//
//        cuda_malloc( (void **) &local_cell.nbrs_x, sizeof(ivec) * host->max_nbrs,
//                TRUE, "alloc:grid:cells:nbrs_x" );
//        copy_device( local_cell.nbrs_x, nbrs_x, host->max_nbrs * sizeof(ivec), "grid:nbrs_x" );
//        //fprintf( stderr, "Allocated address of the nbrs_x--> %ld \n", local_cell.nbrs_x );
//
//        cuda_malloc( (void **) &local_cell.nbrs_cp, sizeof(rvec) * host->max_nbrs,
//                TRUE, "alloc:grid:cells:nbrs_cp" );
//        //fprintf( stderr, "Allocated address of the nbrs_cp--> %ld \n", local_cell.nbrs_cp );
//
//        //cuda_malloc( (void **) &local_cell.nbrs, sizeof(grid_cell *) * host->max_nbrs,
//        //                TRUE, "alloc:grid:cells:nbrs" );
//        //fprintf( stderr, "Allocated address of the nbrs--> %ld \n", local_cell.nbrs );
//
//        copy_host_device( &local_cell, &device->cells[i], sizeof(grid_cell),
//                cudaMemcpyHostToDevice, "grid:cell-alloc" );
//    }
}


void Cuda_Deallocate_Grid_Cell_Atoms( reax_system *system )
{
    int i, total;
    grid_cell local_cell;
    grid *host, *device;

    host = &system->my_grid;
    device = &system->d_my_grid;
    total = host->ncells[0] * host->ncells[1] * host->ncells[2];

    for ( i = 0; i < total; ++i )
    {
        copy_host_device( &local_cell, &device->cells[i], 
                sizeof(grid_cell), cudaMemcpyDeviceToHost,
                "Cuda_Deallocate_Grid_Cell_Atoms::grid" );

        cuda_free( local_cell.atoms,
                "Cuda_Deallocate_Grid_Cell_Atoms::grid_cell.atoms" );
    }
}


void Cuda_Allocate_Grid_Cell_Atoms( reax_system *system, int cap )
{
    int i, total;
    grid_cell local_cell;
    grid *host, *device;

    host = &system->my_grid;
    device = &system->d_my_grid;
    total = host->ncells[0] * host->ncells[1] * host->ncells[2];

    for ( i = 0; i < total; i++ )
    {
        copy_host_device( &local_cell, &device->cells[i], 
                sizeof(grid_cell), cudaMemcpyDeviceToHost, "grid:cell-dealloc" );
        cuda_malloc( (void **)&local_cell.atoms, sizeof(int) * cap, 
                TRUE, "realloc:grid:cells:atoms" );
        copy_host_device( &local_cell, &device->cells[i], 
                sizeof(grid_cell), cudaMemcpyHostToDevice, "grid:cell-realloc" );
    }
}


void Cuda_Allocate_System( reax_system *system )
{
    /* atoms */
    cuda_malloc( (void **) &system->d_my_atoms,
            system->total_cap * sizeof(reax_atom),
            TRUE, "system:d_my_atoms" );
    cuda_malloc( (void **) &system->d_numH, sizeof(int), TRUE, "system:d_numH" );

    /* list management */
    cuda_malloc( (void **) &system->d_far_nbrs,
            system->total_cap * sizeof(int), TRUE, "system:d_far_nbrs" );
    cuda_malloc( (void **) &system->d_max_far_nbrs,
            system->total_cap * sizeof(int), TRUE, "system:d_max_far_nbrs" );
    cuda_malloc( (void **) &system->d_total_far_nbrs,
            sizeof(int), TRUE, "system:d_total_far_nbrs" );
    cuda_malloc( (void **) &system->d_realloc_far_nbrs,
            sizeof(int), TRUE, "system:d_realloc_far_nbrs" );

    cuda_malloc( (void **) &system->d_bonds,
            system->total_cap * sizeof(int), TRUE, "system:d_bonds" );
    cuda_malloc( (void **) &system->d_max_bonds,
            system->total_cap * sizeof(int), TRUE, "system:d_max_bonds" );
    cuda_malloc( (void **) &system->d_total_bonds,
            sizeof(int), TRUE, "system:d_total_bonds" );
    cuda_malloc( (void **) &system->d_realloc_bonds,
            sizeof(int), TRUE, "system:d_realloc_bonds" );

    cuda_malloc( (void **) &system->d_hbonds,
            system->total_cap * sizeof(int), TRUE, "system:d_hbonds" );
    cuda_malloc( (void **) &system->d_max_hbonds,
            system->total_cap * sizeof(int), TRUE, "system:d_max_hbonds" );
    cuda_malloc( (void **) &system->d_total_hbonds,
            sizeof(int), TRUE, "system:d_total_hbonds" );
    cuda_malloc( (void **) &system->d_realloc_hbonds,
            sizeof(int), TRUE, "system:d_realloc_hbonds" );

    cuda_malloc( (void **) &system->d_cm_entries,
            system->local_cap * sizeof(int), TRUE, "system:d_cm_entries" );
    cuda_malloc( (void **) &system->d_max_cm_entries,
            system->local_cap * sizeof(int), TRUE, "system:d_max_cm_entries" );
    cuda_malloc( (void **) &system->d_total_cm_entries,
            sizeof(int), TRUE, "system:d_total_cm_entries" );
    cuda_malloc( (void **) &system->d_realloc_cm_entries,
            sizeof(int), TRUE, "system:d_realloc_cm_entries" );

    cuda_malloc( (void **) &system->d_total_thbodies,
            sizeof(int), TRUE, "system:d_total_thbodies" );

    /* simulation boxes */
    cuda_malloc( (void **) &system->d_big_box,
            sizeof(simulation_box), TRUE, "system:d_big_box" );
    cuda_malloc( (void **) &system->d_my_box,
            sizeof(simulation_box), TRUE, "system:d_my_box" );
    cuda_malloc( (void **) &system->d_my_ext_box,
            sizeof(simulation_box), TRUE, "d_my_ext_box" );

    /* interaction parameters */
    cuda_malloc( (void **) &system->reax_param.d_sbp,
            system->reax_param.num_atom_types * sizeof(single_body_parameters),
            TRUE, "system:d_sbp" );

    cuda_malloc( (void **) &system->reax_param.d_tbp,
            POW( system->reax_param.num_atom_types, 2.0 ) * sizeof(two_body_parameters), 
            TRUE, "system:d_tbp" );

    cuda_malloc( (void **) &system->reax_param.d_thbp,
            POW( system->reax_param.num_atom_types, 3.0 ) * sizeof(three_body_header),
            TRUE, "system:d_thbp" );

    cuda_malloc( (void **) &system->reax_param.d_hbp,
            POW( system->reax_param.num_atom_types, 3.0 ) * sizeof(hbond_parameters),
            TRUE, "system:d_hbp" );

    cuda_malloc( (void **) &system->reax_param.d_fbp,
            POW( system->reax_param.num_atom_types, 4.0 ) * sizeof(four_body_header),
            TRUE, "system:d_fbp" );

    cuda_malloc( (void **) &system->reax_param.d_gp.l,
            system->reax_param.gp.n_global * sizeof(real), TRUE, "system:d_gp.l" );

    system->reax_param.d_gp.n_global = 0;
    system->reax_param.d_gp.vdw_type = 0;
}


static void Cuda_Reallocate_System_Part1( reax_system *system, storage *workspace,
        int local_cap_old )
{
    int *temp;

    cuda_check_malloc( &workspace->scratch, &workspace->scratch_size,
            sizeof(int) * local_cap_old,
            "Cuda_Reallocate_System_Part1::workspace->scratch" );
    temp = (int *) workspace->scratch;

    copy_device( temp, system->d_cm_entries, sizeof(int) * local_cap_old,
            "Cuda_Reallocate_System_Part1::temp" );
    cuda_free( system->d_cm_entries, "Cuda_Reallocate_System_Part1::d_cm_entries" );
    cuda_malloc( (void **) &system->d_cm_entries,
            sizeof(int) * system->local_cap, TRUE, "Cuda_Reallocate_System_Part1::d_cm_entries" );
    copy_device( system->d_cm_entries, temp, sizeof(int) * local_cap_old,
            "Cuda_Reallocate_System_Part1::temp" );

    copy_device( temp, system->d_max_cm_entries, sizeof(int) * local_cap_old,
            "Cuda_Reallocate_System_Part1::temp" );
    cuda_free( system->d_max_cm_entries, "Cuda_Reallocate_System_Part1::d_max_cm_entries" );
    cuda_malloc( (void **) &system->d_max_cm_entries,
            sizeof(int) * system->local_cap, TRUE, "Cuda_Reallocate_System_Part1::d_max_cm_entries" );
    copy_device( system->d_max_cm_entries, temp, sizeof(int) * local_cap_old,
            "Cuda_Reallocate_System_Part1::temp" );
}


static void Cuda_Reallocate_System_Part2( reax_system *system, storage *workspace,
        int total_cap_old )
{
    int *temp;
    reax_atom *temp_atom;

    cuda_check_malloc( &workspace->scratch, &workspace->scratch_size,
            MAX( sizeof(reax_atom), sizeof(int) ) * total_cap_old,
            "Cuda_Reallocate_System_Part2::workspace->scratch" );
    temp = (int *) workspace->scratch;
    temp_atom = (reax_atom *) workspace->scratch;

    /* free the existing storage for atoms, leave other info allocated */
    copy_device( temp_atom, system->d_my_atoms, sizeof(reax_atom) * total_cap_old,
            "Cuda_Reallocate_System_Part2::temp_atom" );
    cuda_free( system->d_my_atoms, "system::d_my_atoms" );
    cuda_malloc( (void **) &system->d_my_atoms,
            sizeof(reax_atom) * system->total_cap, TRUE,
            "Cuda_Reallocate_System_Part2::d_my_atoms" );
    copy_device( system->d_my_atoms, temp_atom, sizeof(reax_atom) * total_cap_old,
            "Cuda_Reallocate_System_Part2::temp_atom" );

    /* list management */
    copy_device( temp, system->d_far_nbrs, sizeof(int) * total_cap_old,
            "Cuda_Reallocate_System_Part2::temp" );
    cuda_free( system->d_far_nbrs, "Cuda_Reallocate_System_Part2::d_far_nbrs" );
    cuda_malloc( (void **) &system->d_far_nbrs,
            sizeof(int) * system->total_cap, TRUE,
            "Cuda_Reallocate_System_Part2::d_far_nbrs" );
    copy_device( system->d_far_nbrs, temp, sizeof(int) * total_cap_old,
            "Cuda_Reallocate_System_Part2::temp" );

    copy_device( temp, system->d_max_far_nbrs, sizeof(int) * total_cap_old,
            "Cuda_Reallocate_System_Part2::temp" );
    cuda_free( system->d_max_far_nbrs, "Cuda_Reallocate_System_Part2::d_max_far_nbrs" );
    cuda_malloc( (void **) &system->d_max_far_nbrs,
            sizeof(int) * system->total_cap, TRUE,
            "Cuda_Reallocate_System_Part2::d_max_far_nbrs" );
    copy_device( system->d_max_far_nbrs, temp, sizeof(int) * total_cap_old,
            "Cuda_Reallocate_System_Part2::temp" );

    copy_device( temp, system->d_bonds, sizeof(int) * total_cap_old,
            "Cuda_Reallocate_System_Part2::temp" );
    cuda_free( system->d_bonds, "Cuda_Reallocate_System_Part2::d_bonds" );
    cuda_malloc( (void **) &system->d_bonds,
            sizeof(int) * system->total_cap, TRUE,
            "Cuda_Reallocate_System_Part2::d_bonds" );
    copy_device( system->d_bonds, temp, sizeof(int) * total_cap_old,
            "Cuda_Reallocate_System_Part2::temp" );

    copy_device( temp, system->d_max_bonds, sizeof(int) * total_cap_old,
            "Cuda_Reallocate_System_Part2::temp" );
    cuda_free( system->d_max_bonds, "Cuda_Reallocate_System_Part2::d_max_bonds" );
    cuda_malloc( (void **) &system->d_max_bonds,
            sizeof(int) * system->total_cap, TRUE,
            "Cuda_Reallocate_System_Part2::d_max_bonds" );
    copy_device( system->d_max_bonds, temp, sizeof(int) * total_cap_old,
            "Cuda_Reallocate_System_Part2::temp" );

    copy_device( temp, system->d_hbonds, sizeof(int) * total_cap_old,
            "Cuda_Reallocate_System_Part2::temp" );
    cuda_free( system->d_hbonds, "system::d_hbonds" );
    cuda_malloc( (void **) &system->d_hbonds,
            sizeof(int) * system->total_cap, TRUE,
            "Cuda_Reallocate_System_Part2::d_hbonds" );
    copy_device( system->d_hbonds, temp, sizeof(int) * total_cap_old,
            "Cuda_Reallocate_System_Part2::temp" );

    copy_device( temp, system->d_max_hbonds, sizeof(int) * total_cap_old,
            "Cuda_Reallocate_System_Part2::temp" );
    cuda_free( system->d_max_hbonds, "system::d_max_hbonds" );
    cuda_malloc( (void **) &system->d_max_hbonds,
            sizeof(int) * system->total_cap, TRUE,
            "Cuda_Reallocate_System_Part2::d_max_hbonds" );
    copy_device( system->d_max_hbonds, temp, sizeof(int) * total_cap_old,
            "Cuda_Reallocate_System_Part2::temp" );
}


void Cuda_Allocate_Simulation_Data( simulation_data *data )
{
    cuda_malloc( (void **) &data->d_simulation_data,
            sizeof(simulation_data), TRUE, "simulation_data" );
}


void Cuda_Allocate_Workspace_Part1( reax_system *system, control_params *control, 
        storage *workspace, int local_cap )
{
    int local_rvec;

    local_rvec = sizeof(rvec) * local_cap;

    /* integrator storage */
    if ( control->ensemble == nhNVT )
    {
        cuda_malloc( (void **) &workspace->v_const, local_rvec, TRUE, "v_const" );
    }

    /* storage for analysis */
    if ( control->molecular_analysis || control->diffusion_coef )
    {
        cuda_malloc( (void **) &workspace->mark, local_cap * sizeof(int), TRUE, "mark" );
        cuda_malloc( (void **) &workspace->old_mark, local_cap * sizeof(int), TRUE, "old_mark" );
    }
    else
    {
        workspace->mark = NULL;
        workspace->old_mark = NULL;
    }

    if ( control->diffusion_coef )
    {
        cuda_malloc( (void **) &workspace->x_old, local_cap * sizeof(rvec), TRUE, "x_old" );
    }
    else
    {
        workspace->x_old = NULL;
    }
}


void Cuda_Allocate_Workspace_Part2( reax_system *system, control_params *control, 
        storage *workspace, int total_cap )
{
    int total_real, total_rvec;
#if defined(DUAL_SOLVER)
    int total_rvec2;
#endif

    total_real = sizeof(real) * total_cap;
    total_rvec = sizeof(rvec) * total_cap;
#if defined(DUAL_SOLVER)
    total_rvec2 = sizeof(rvec2) * total_cap;
#endif

    /* bond order related storage  */
    cuda_malloc( (void **) &workspace->total_bond_order, total_real, TRUE, "total_bo" );
    cuda_malloc( (void **) &workspace->Deltap, total_real, TRUE, "Deltap" );
    cuda_malloc( (void **) &workspace->Deltap_boc, total_real, TRUE, "Deltap_boc" );
    cuda_malloc( (void **) &workspace->dDeltap_self, total_rvec, TRUE, "dDeltap_self" );
    cuda_malloc( (void **) &workspace->Delta, total_real, TRUE, "Delta" );
    cuda_malloc( (void **) &workspace->Delta_lp, total_real, TRUE, "Delta_lp" );
    cuda_malloc( (void **) &workspace->Delta_lp_temp, total_real, TRUE, "Delta_lp_temp" );
    cuda_malloc( (void **) &workspace->dDelta_lp, total_real, TRUE, "Delta_lp_temp" );
    cuda_malloc( (void **) &workspace->dDelta_lp_temp, total_real, TRUE, "dDelta_lp_temp" );
    cuda_malloc( (void **) &workspace->Delta_e, total_real, TRUE, "Delta_e" );
    cuda_malloc( (void **) &workspace->Delta_boc, total_real, TRUE, "Delta_boc" );
    cuda_malloc( (void **) &workspace->nlp, total_real, TRUE, "nlp" );
    cuda_malloc( (void **) &workspace->nlp_temp, total_real, TRUE, "nlp_temp" );
    cuda_malloc( (void **) &workspace->Clp, total_real, TRUE, "Clp" );
    cuda_malloc( (void **) &workspace->vlpex, total_real, TRUE, "vlpex" );
    cuda_malloc( (void **) &workspace->bond_mark, total_real, TRUE, "bond_mark" );

    /* charge matrix storage */
    if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        cuda_malloc( (void **) &workspace->Hdia_inv, total_real, TRUE, "Hdia_inv" );
    }
    if ( control->cm_solver_pre_comp_type == ICHOLT_PC
            || control->cm_solver_pre_comp_type == ILUT_PC
            || control->cm_solver_pre_comp_type == ILUTP_PC
            || control->cm_solver_pre_comp_type == FG_ILUT_PC )
    {
        cuda_malloc( (void **) &workspace->droptol, total_real, TRUE, "droptol" );
    }
    cuda_malloc( (void **) &workspace->b_s, total_real, TRUE, "b_s" );
    cuda_malloc( (void **) &workspace->b_t, total_real, TRUE, "b_t" );
    cuda_malloc( (void **) &workspace->s, total_real, TRUE, "s" );
    cuda_malloc( (void **) &workspace->t, total_real, TRUE, "t" );
#if defined(DUAL_SOLVER)
    cuda_malloc( (void **) &workspace->b, total_rvec2, TRUE, "b" );
    cuda_malloc( (void **) &workspace->x, total_rvec2, TRUE, "x" );
#endif

    switch ( control->cm_solver_type )
    {
    case GMRES_S:
    case GMRES_H_S:
        cuda_malloc( (void **) &workspace->b_prc,
                total_real, TRUE, "b_prc" );
        cuda_malloc( (void **) &workspace->b_prm,
                total_real, TRUE, "b_prm" );
        cuda_malloc( (void **) &workspace->y,
                (control->cm_solver_restart + 1) * sizeof(real), TRUE, "y" );
        cuda_malloc( (void **) &workspace->z,
                (control->cm_solver_restart + 1) * sizeof(real), TRUE, "z" );
        cuda_malloc( (void **) &workspace->g,
                (control->cm_solver_restart + 1) * sizeof(real), TRUE, "g" );
        cuda_malloc( (void **) &workspace->h,
                SQR(control->cm_solver_restart + 1) * sizeof(real), TRUE, "h" );
        cuda_malloc( (void **) &workspace->hs,
                (control->cm_solver_restart + 1) * sizeof(real), TRUE, "hs" );
        cuda_malloc( (void **) &workspace->hc,
                (control->cm_solver_restart + 1) * sizeof(real), TRUE, "hc" );
        cuda_malloc( (void **) &workspace->v,
                SQR(control->cm_solver_restart + 1) * sizeof(real), TRUE, "v" );
        break;

    case SDM_S:
        cuda_malloc( (void **) &workspace->r, total_real, TRUE, "r" );
        cuda_malloc( (void **) &workspace->d, total_real, TRUE, "d" );
        cuda_malloc( (void **) &workspace->q, total_real, TRUE, "q" );
        cuda_malloc( (void **) &workspace->p, total_real, TRUE, "p" );
        break;

    case CG_S:
        cuda_malloc( (void **) &workspace->r, total_real, TRUE, "r" );
        cuda_malloc( (void **) &workspace->d, total_real, TRUE, "d" );
        cuda_malloc( (void **) &workspace->q, total_real, TRUE, "q" );
        cuda_malloc( (void **) &workspace->p, total_real, TRUE, "p" );
#if defined(DUAL_SOLVER)
        cuda_malloc( (void **) &workspace->r2, total_rvec2, TRUE, "r2" );
        cuda_malloc( (void **) &workspace->d2, total_rvec2, TRUE, "d2" );
        cuda_malloc( (void **) &workspace->q2, total_rvec2, TRUE, "q2" );
        cuda_malloc( (void **) &workspace->p2, total_rvec2, TRUE, "p2" );
#endif
        break;

    case BiCGStab_S:
        cuda_malloc( (void **) &workspace->y, total_real, TRUE, "y" );
        cuda_malloc( (void **) &workspace->g, total_real, TRUE, "g" );
        cuda_malloc( (void **) &workspace->z, total_real, TRUE, "z" );
        cuda_malloc( (void **) &workspace->r, total_real, TRUE, "r" );
        cuda_malloc( (void **) &workspace->d, total_real, TRUE, "d" );
        cuda_malloc( (void **) &workspace->q, total_real, TRUE, "q" );
        cuda_malloc( (void **) &workspace->p, total_real, TRUE, "p" );
        cuda_malloc( (void **) &workspace->r_hat, total_real, TRUE, "r_hat" );
        cuda_malloc( (void **) &workspace->q_hat, total_real, TRUE, "q_hat" );
#if defined(DUAL_SOLVER)
        cuda_malloc( (void **) &workspace->y2, total_rvec2, TRUE, "y" );
        cuda_malloc( (void **) &workspace->g2, total_rvec2, TRUE, "g" );
        cuda_malloc( (void **) &workspace->z2, total_rvec2, TRUE, "z" );
        cuda_malloc( (void **) &workspace->r2, total_rvec2, TRUE, "r" );
        cuda_malloc( (void **) &workspace->d2, total_rvec2, TRUE, "d" );
        cuda_malloc( (void **) &workspace->q2, total_rvec2, TRUE, "q" );
        cuda_malloc( (void **) &workspace->p2, total_rvec2, TRUE, "p" );
        cuda_malloc( (void **) &workspace->r_hat2, total_rvec2, TRUE, "r_hat" );
        cuda_malloc( (void **) &workspace->q_hat2, total_rvec2, TRUE, "q_hat" );
#endif
        break;

    case PIPECG_S:
        break;

    case PIPECR_S:
        break;

    default:
        fprintf( stderr, "[ERROR] Unknown charge method linear solver type. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    /* force related storage */
    cuda_malloc( (void **) &workspace->f, sizeof(rvec) * total_cap, TRUE, "f" );
    cuda_malloc( (void **) &workspace->CdDelta, sizeof(rvec) * total_cap, TRUE, "CdDelta" );
}


void Cuda_Deallocate_Workspace_Part1( control_params *control, storage *workspace )
{
    /* Nose-Hoover integrator */
    if ( control->ensemble == nhNVT )
    {
        cuda_free( workspace->v_const, "v_const" );
    }

    /* storage for analysis */
    if ( control->molecular_analysis || control->diffusion_coef )
    {
        cuda_free( workspace->mark, "mark" );
        cuda_free( workspace->old_mark, "old_mark" );
    }
    else
    {
        workspace->mark = NULL;
        workspace->old_mark = NULL;
    }

    if ( control->diffusion_coef )
    {
        cuda_free( workspace->x_old, "x_old" );
    }
    else
    {
        workspace->x_old = NULL;
    }
}


void Cuda_Deallocate_Workspace_Part2( control_params *control, storage *workspace )
{
    /* bond order related storage  */
    cuda_free( workspace->total_bond_order, "total_bo" );
    cuda_free( workspace->Deltap, "Deltap" );
    cuda_free( workspace->Deltap_boc, "Deltap_boc" );
    cuda_free( workspace->dDeltap_self, "dDeltap_self" );
    cuda_free( workspace->Delta, "Delta" );
    cuda_free( workspace->Delta_lp, "Delta_lp" );
    cuda_free( workspace->Delta_lp_temp, "Delta_lp_temp" );
    cuda_free( workspace->dDelta_lp, "Delta_lp_temp" );
    cuda_free( workspace->dDelta_lp_temp, "dDelta_lp_temp" );
    cuda_free( workspace->Delta_e, "Delta_e" );
    cuda_free( workspace->Delta_boc, "Delta_boc" );
    cuda_free( workspace->nlp, "nlp" );
    cuda_free( workspace->nlp_temp, "nlp_temp" );
    cuda_free( workspace->Clp, "Clp" );
    cuda_free( workspace->vlpex, "vlpex" );
    cuda_free( workspace->bond_mark, "bond_mark" );

    /* charge matrix storage */
    if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        cuda_free( workspace->Hdia_inv, "Hdia_inv" );
    }
    if ( control->cm_solver_pre_comp_type == ICHOLT_PC
            || control->cm_solver_pre_comp_type == ILUT_PC
            || control->cm_solver_pre_comp_type == ILUTP_PC
            || control->cm_solver_pre_comp_type == FG_ILUT_PC )
    {
        cuda_free( workspace->droptol, "droptol" );
    }
    cuda_free( workspace->b_s, "b_s" );
    cuda_free( workspace->b_t, "b_t" );
    cuda_free( workspace->s, "s" );
    cuda_free( workspace->t, "t" );
#if defined(DUAL_SOLVER)
    cuda_free( workspace->b, "b" );
    cuda_free( workspace->x, "x" );
#endif

    switch ( control->cm_solver_type )
    {
        case GMRES_S:
        case GMRES_H_S:
            cuda_free( workspace->b_prc, "b_prc" );
            cuda_free( workspace->b_prm, "b_prm" );
            cuda_free( workspace->y, "y" );
            cuda_free( workspace->z, "z" );
            cuda_free( workspace->g, "g" );
            cuda_free( workspace->h, "h" );
            cuda_free( workspace->hs, "hs" );
            cuda_free( workspace->hc, "hc" );
            cuda_free( workspace->v, "v" );
            break;

        case CG_S:
            cuda_free( workspace->r, "r" );
            cuda_free( workspace->d, "d" );
            cuda_free( workspace->q, "q" );
            cuda_free( workspace->p, "p" );
#if defined(DUAL_SOLVER)
            cuda_free( workspace->r2, "r2" );
            cuda_free( workspace->d2, "d2" );
            cuda_free( workspace->q2, "q2" );
            cuda_free( workspace->p2, "p2" );
#endif
            break;

        case SDM_S:
            cuda_free( workspace->r, "r" );
            cuda_free( workspace->d, "d" );
            cuda_free( workspace->q, "q" );
            cuda_free( workspace->p, "p" );
            break;

        case BiCGStab_S:
            cuda_free( workspace->y, "y" );
            cuda_free( workspace->g, "g" );
            cuda_free( workspace->z, "z" );
            cuda_free( workspace->r, "r" );
            cuda_free( workspace->d, "d" );
            cuda_free( workspace->q, "q" );
            cuda_free( workspace->p, "p" );
            cuda_free( workspace->r_hat, "r_hat" );
            cuda_free( workspace->q_hat, "q_hat" );
#if defined(DUAL_SOLVER)
            cuda_free( workspace->y2, "y2" );
            cuda_free( workspace->g2, "g2" );
            cuda_free( workspace->z2, "z2" );
            cuda_free( workspace->r2, "r2" );
            cuda_free( workspace->d2, "d2" );
            cuda_free( workspace->q2, "q2" );
            cuda_free( workspace->p2, "p2" );
            cuda_free( workspace->r_hat2, "r_hat2" );
            cuda_free( workspace->q_hat2, "q_hat2" );
#endif
            break;

        case PIPECG_S:
            break;

        case PIPECR_S:
            break;

        default:
            fprintf( stderr, "[ERROR] Unknown charge method linear solver type. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }

    /* force related storage */
    cuda_free( workspace->f, "f" );
    cuda_free( workspace->CdDelta, "CdDelta" );
}


/* Allocate sparse matrix struc
 *
 * H: pointer to struct
 * n: currently utilized number of rows
 * n_max: max number of rows allocated
 * m: max number of entries allocated
 * format: sparse matrix format
 */
void Cuda_Allocate_Matrix( sparse_matrix * const H, int n, int n_max, int m,
       int format )
{
    H->allocated = TRUE;
    H->n = n;
    H->n_max = n_max;
    H->m = m;
    H->format = format;

    cuda_malloc( (void **) &H->start, sizeof(int) * n_max, TRUE,
            "Cuda_Allocate_Matrix::H->start" );
    cuda_malloc( (void **) &H->end, sizeof(int) * n_max, TRUE,
            "Cuda_Allocate_Matrix::H->end" );
    cuda_malloc( (void **) &H->j, sizeof(int) * m, TRUE,
            "Cuda_Allocate_Matrix::H->j" );
    cuda_malloc( (void **) &H->val, sizeof(real) * m, TRUE,
            "Cuda_Allocate_Matrix::H->val" );
}


void Cuda_Deallocate_Matrix( sparse_matrix *H )
{
    H->allocated = FALSE;
    H->n = 0;
    H->n_max = 0;
    H->m = 0;

    cuda_free( H->start, "Cuda_Deallocate_Matrix::start" );
    cuda_free( H->end, "Cuda_Deallocate_Matrix::end" );
    cuda_free( H->j, "Cuda_Deallocate_Matrix::j" );
    cuda_free( H->val, "Cuda_Deallocate_Matrix::val" );
}


void Cuda_Reallocate_Neighbor_List( reax_list *far_nbr_list, size_t n, size_t max_intrs )
{
    Cuda_Delete_List( far_nbr_list );
    Cuda_Make_List( n, max_intrs, TYP_FAR_NEIGHBOR, far_nbr_list );
}


void Cuda_Reallocate_HBonds_List( reax_list *hbond_list, size_t n, size_t max_intrs )
{
    Cuda_Delete_List( hbond_list );
    Cuda_Make_List( n, max_intrs, TYP_HBOND, hbond_list );
}


void Cuda_Reallocate_Bonds_List( reax_list *bond_list, size_t n, size_t max_intrs )
{
    Cuda_Delete_List( bond_list );
    Cuda_Make_List( n, max_intrs, TYP_BOND, bond_list );
}


void Cuda_Reallocate_Thbodies_List( reax_list *thbodies, size_t n, size_t max_intrs )
{
    Cuda_Delete_List( thbodies );
    Cuda_Make_List( n, max_intrs, TYP_THREE_BODY, thbodies );

}


void Cuda_Reallocate_Part1( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        mpi_datatypes *mpi_data )
{
    int i, j, k, renbr;
    reallocate_data *realloc;
    grid *g;

    realloc = &workspace->d_workspace->realloc;
    g = &system->my_grid;
    renbr = (data->step - data->prev_steps) % control->reneighbor == 0 ? TRUE : FALSE;

    /* grid */
    if ( renbr == TRUE && realloc->gcell_atoms > -1 )
    {
        for ( i = g->native_str[0]; i < g->native_end[0]; i++ )
        {
            for ( j = g->native_str[1]; j < g->native_end[1]; j++ )
            {
                for ( k = g->native_str[2]; k < g->native_end[2]; k++ )
                {
                    sfree( g->cells[ index_grid_3d(i,j,k,g) ].atoms, "g:atoms" );
                    g->cells[ index_grid_3d(i,j,k,g) ].atoms = (int *)
                            scalloc( realloc->gcell_atoms, sizeof(int), "g:atoms" );
                }
            }
        }

        fprintf( stderr, "p:%d - *** Reallocating Grid Cell Atoms *** Step:%d\n", system->my_rank, data->step );

//        Cuda_Deallocate_Grid_Cell_Atoms( system );
//        Cuda_Allocate_Grid_Cell_Atoms( system, realloc->gcell_atoms );
        realloc->gcell_atoms = -1;
    }
}


void Cuda_Reallocate_Part2( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        mpi_datatypes *mpi_data )
{
    int nflag, Nflag, local_cap_old, total_cap_old, renbr, format;
    reallocate_data *realloc;
    sparse_matrix *H;

    realloc = &workspace->d_workspace->realloc;
    H = &workspace->d_workspace->H;
    renbr = (data->step - data->prev_steps) % control->reneighbor == 0 ? TRUE : FALSE;

    /* IMPORTANT: LOOSE ZONES CHECKS ARE DISABLED FOR NOW BY &&'ing with FALSE!!! */
    nflag = FALSE;
    if ( system->n >= (int) CEIL( DANGER_ZONE * system->local_cap )
            || (FALSE && system->n <= (int) CEIL( LOOSE_ZONE * system->local_cap )) )
    {
        nflag = TRUE;
        local_cap_old = system->local_cap;
        system->local_cap = (int) CEIL( system->n * SAFE_ZONE );
    }

    Nflag = FALSE;
    if ( system->N >= (int) CEIL( DANGER_ZONE * system->total_cap )
            || (FALSE && system->N <= (int) CEIL( LOOSE_ZONE * system->total_cap )) )
    {
        Nflag = TRUE;
        total_cap_old = system->total_cap;
        system->total_cap = (int) CEIL( system->N * SAFE_ZONE );
    }

    if ( nflag == TRUE )
    {
//        fprintf( stderr, "[INFO] Cuda_Reallocate_Part2: p%d, local_cap_old = %d\n, local_cap = %d", system->my_rank, local_cap_old, system->local_cap );
//        fflush( stderr );
        Cuda_Reallocate_System_Part1( system, workspace, local_cap_old );

        Cuda_Deallocate_Workspace_Part1( control, workspace );
        Cuda_Allocate_Workspace_Part1( system, control, workspace,
                system->local_cap );
    }

    if ( Nflag == TRUE )
    {
//        fprintf( stderr, "[INFO] Cuda_Reallocate_Part2: p%d, total_cap_old = %d\n, total_cap = %d", system->my_rank, total_cap_old, system->total_cap );
//        fflush( stderr );
        Cuda_Reallocate_System_Part2( system, workspace, total_cap_old );

        Cuda_Deallocate_Workspace_Part2( control, workspace );
        Cuda_Allocate_Workspace_Part2( system, control, workspace,
                system->total_cap );
    }

    /* far neighbors */
    if ( renbr == TRUE && (Nflag == TRUE || realloc->far_nbrs == TRUE) )
    {
        Cuda_Reallocate_Neighbor_List( lists[FAR_NBRS],
                system->total_cap, system->total_far_nbrs );
        Cuda_Init_Neighbor_Indices( system, lists[FAR_NBRS] );
        realloc->far_nbrs = FALSE;
    }

    /* charge matrix */
    if ( nflag == TRUE || realloc->cm == TRUE )
    {
        format = H->format;

        Cuda_Deallocate_Matrix( H );
        Cuda_Allocate_Matrix( H, system->n, system->local_cap,
                system->total_cm_entries, format );
        Cuda_Init_Sparse_Matrix_Indices( system, H );
        realloc->cm = FALSE;
    }

    /* hydrogen bonds list */
    if ( control->hbond_cut > 0.0 && system->numH > 0
            && (Nflag == TRUE || realloc->hbonds == TRUE) )
    {
        Cuda_Reallocate_HBonds_List( lists[HBONDS],
                system->total_cap, system->total_hbonds );
        Cuda_Init_HBond_Indices( system, workspace, lists[HBONDS] );
        realloc->hbonds = FALSE;
    }

    /* bonds list */
    if ( Nflag == TRUE || realloc->bonds == TRUE )
    {
        Cuda_Reallocate_Bonds_List( lists[BONDS],
                system->total_cap, system->total_bonds );
        Cuda_Init_Bond_Indices( system, lists[BONDS] );
        realloc->bonds = FALSE;
    }

    /* 3-body list */
    if ( Nflag == TRUE || realloc->thbody == TRUE )
    {
        Cuda_Reallocate_Thbodies_List( lists[THREE_BODIES],
                system->total_thbodies_indices, system->total_thbodies );
        realloc->thbody = FALSE;
    }
}
