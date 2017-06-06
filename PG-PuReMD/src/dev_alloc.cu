
#include "dev_alloc.h"
#include "cuda_utils.h"

#include "vector.h"

extern "C"
{


void dev_alloc_control( control_params *control )
{
    cuda_malloc( (void **)&control->d_control_params,
            sizeof(control_params), TRUE, "control_params" );
    copy_host_device( control, control->d_control_params,
            sizeof(control_params), cudaMemcpyHostToDevice, "control_params" );
}


CUDA_GLOBAL void Init_Nbrs( ivec *nbrs, int N )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
    {
        return;
    }

    nbrs[index][0] = -1; 
    nbrs[index][1] = -1; 
    nbrs[index][2] = -1; 
}


void dev_alloc_grid( reax_system *system )
{
    int total;
    grid_cell local_cell;
    grid *host = &system->my_grid;
    grid *device = &system->d_my_grid;
    ivec *nbrs_x = (ivec *) scratch;

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

    cuda_malloc( (void **) &device->str, sizeof(int) * total, TRUE, "grid:str" );
    cuda_malloc( (void **) &device->end, sizeof(int) * total, TRUE, "grid:end" );
    cuda_malloc( (void **) &device->cutoff, sizeof(real) * total, TRUE, "grid:cutoff" );

    cuda_malloc( (void **) &device->nbrs_x, sizeof(ivec) * total * host->max_nbrs, TRUE, "grid:nbrs_x" );
    cuda_malloc( (void **) &device->nbrs_cp, sizeof(rvec) * total * host->max_nbrs, TRUE, "grid:nbrs_cp" );
    cuda_malloc( (void **) &device->rel_box, sizeof(ivec) * total, TRUE, "grid:rel_box" );

    /*
       int block_size = 512;
       int blocks = (host->max_nbrs) / block_size + ((host->max_nbrs) % block_size == 0 ? 0 : 1); 

       Init_Nbrs <<<blocks, block_size>>>
       (nbrs_x, host->max_nbrs );
       cudaThreadSynchronize (); 
       cudaCheckError ();

       cuda_malloc ((void **)& device->cells, 
       sizeof (grid_cell) * total, 
       TRUE, "grid:cells");
       fprintf (stderr, " Device cells address --> %ld \n", device->cells );
       cuda_malloc ((void **) &device->order, sizeof (ivec) * (host->total + 1), TRUE, "grid:order");

       local_cell.top = local_cell.mark = local_cell.str = local_cell.end = 0;
       fprintf (stderr, "Total cells to be allocated -- > %d \n", total );
       for (int i = 0; i < total; i++) {
    //fprintf (stderr, "Address of the local atom -> %ld  \n", &local_cell);

    cuda_malloc ((void **) &local_cell.atoms, sizeof (int) * host->max_atoms, 
    TRUE, "alloc:grid:cells:atoms");
    //fprintf (stderr, "Allocated address of the atoms --> %ld  (%d)\n", local_cell.atoms, host->max_atoms );

    cuda_malloc ((void **) &local_cell.nbrs_x, sizeof (ivec) * host->max_nbrs, 
    TRUE, "alloc:grid:cells:nbrs_x" );
    copy_device (local_cell.nbrs_x, nbrs_x, host->max_nbrs * sizeof (ivec), "grid:nbrs_x");    
    //fprintf (stderr, "Allocated address of the nbrs_x--> %ld \n", local_cell.nbrs_x);

    cuda_malloc ((void **) &local_cell.nbrs_cp, sizeof (rvec) * host->max_nbrs, 
    TRUE, "alloc:grid:cells:nbrs_cp" );
    //fprintf (stderr, "Allocated address of the nbrs_cp--> %ld \n", local_cell.nbrs_cp);

    //cuda_malloc ((void **) &local_cell.nbrs, sizeof (grid_cell *) * host->max_nbrs , 
    //                TRUE, "alloc:grid:cells:nbrs" );
    //fprintf (stderr, "Allocated address of the nbrs--> %ld \n", local_cell.nbrs);

    copy_host_device (&local_cell, &device->cells[i], sizeof (grid_cell), cudaMemcpyHostToDevice, "grid:cell-alloc");
    }
     */
}


void dev_dealloc_grid_cell_atoms( reax_system *system )
{
    int total;
    grid_cell local_cell;
    grid *host = &system->my_grid;
    grid *device = &system->d_my_grid;

    total = host->ncells[0] * host->ncells[1] * host->ncells[2];

    for (int i = 0; i < total; i++)
    {
        copy_host_device( &local_cell, &device->cells[i], 
                sizeof(grid_cell), cudaMemcpyDeviceToHost, "grid:cell-dealloc" );
        cuda_free( local_cell.atoms, "grid_cell:atoms" );
    }
}


void dev_alloc_grid_cell_atoms( reax_system *system, int cap )
{
    int i, total;
    grid_cell local_cell;
    grid *host = &system->my_grid;
    grid *device = &system->d_my_grid;

    total = host->ncells[0] * host->ncells[1] * host->ncells[2];

    for (i = 0; i < total; i++)
    {
        copy_host_device( &local_cell, &device->cells[i], 
                sizeof(grid_cell), cudaMemcpyDeviceToHost, "grid:cell-dealloc" );
        cuda_malloc( (void **)&local_cell.atoms, sizeof(int) * cap, 
                TRUE, "realloc:grid:cells:atoms" );
        copy_host_device( &local_cell, &device->cells[i], 
                sizeof(grid_cell), cudaMemcpyHostToDevice, "grid:cell-realloc" );
    }
}


void dev_alloc_system( reax_system *system )
{
    /* atoms */
    cuda_malloc( (void **) &system->d_my_atoms, system->total_cap * sizeof(reax_atom),
            TRUE, "system:d_my_atoms" );

    /* simulation boxes */
    cuda_malloc( (void **) &system->d_big_box, sizeof(simulation_box), TRUE, "system:d_big_box" );
    cuda_malloc( (void **) &system->d_my_box, sizeof(simulation_box), TRUE, "system:d_my_box" );
    cuda_malloc( (void **) &system->d_my_ext_box, sizeof(simulation_box), TRUE, "d_my_ext_box" );

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


void dev_realloc_system( reax_system *system, int local_cap, int total_cap, char *msg )
{
    /* free the existing storage for atoms, leave other info allocated */
    cuda_free( system->d_my_atoms, "system:d_my_atoms" );

    cuda_malloc( (void **) &system->d_my_atoms, sizeof(reax_atom) * total_cap, 
            TRUE, "system:d_my_atoms" );
}


void dev_alloc_simulation_data( simulation_data *data )
{
    cuda_malloc( (void **) &(data->d_simulation_data), sizeof(simulation_data), TRUE, "simulation_data" );
}


void dev_alloc_workspace( reax_system *system, control_params *control, 
        storage *workspace, int local_cap, int total_cap, char *msg )
{
    int i, total_real, total_rvec, local_int, local_real, local_rvec;

    workspace->allocated = TRUE;

    total_real = total_cap * sizeof(real);
    total_rvec = total_cap * sizeof(rvec);
    local_int = local_cap * sizeof(int);
    local_real = local_cap * sizeof(real);
    local_rvec = local_cap * sizeof(rvec);

    /* communication storage */  
    /*
       workspace->tmp_dbl = NULL;
       workspace->tmp_rvec = NULL;
       workspace->tmp_rvec2 = NULL;
     */

    /* bond order related storage  */
    cuda_malloc( (void **) &workspace->within_bond_box, total_cap * sizeof (int), TRUE, "skin" );
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
    cuda_malloc( (void **) &workspace->done_after, total_real, TRUE, "done_after" );


    /* QEq storage */
    cuda_malloc( (void **) &workspace->Hdia_inv, total_cap * sizeof(real), TRUE, "Hdia_inv" );
    cuda_malloc( (void **) &workspace->b_s, total_cap * sizeof(real), TRUE, "b_s" );
    cuda_malloc( (void **) &workspace->b_t, total_cap * sizeof(real), TRUE, "b_t" );
    cuda_malloc( (void **) &workspace->b_prc, total_cap * sizeof(real), TRUE, "b_prc" );
    cuda_malloc( (void **) &workspace->b_prm, total_cap * sizeof(real), TRUE, "b_prm" );
    cuda_malloc( (void **) &workspace->s, total_cap * sizeof(real), TRUE, "s" );
    cuda_malloc( (void **) &workspace->t, total_cap * sizeof(real), TRUE, "t" );
    cuda_malloc( (void **) &workspace->droptol, total_cap * sizeof(real), TRUE, "droptol" );
    cuda_malloc( (void **) &workspace->b, total_cap * sizeof(rvec2), TRUE, "b" );
    cuda_malloc( (void **) &workspace->x, total_cap * sizeof(rvec2), TRUE, "x" );

    /* GMRES storage */
    cuda_malloc( (void **) &workspace->y, (RESTART+1)*sizeof(real), TRUE, "y" );
    cuda_malloc( (void **) &workspace->z, (RESTART+1)*sizeof(real), TRUE, "z" );
    cuda_malloc( (void **) &workspace->g, (RESTART+1)*sizeof(real), TRUE, "g" );
    cuda_malloc( (void **) &workspace->h, (RESTART+1)*(RESTART+1)*sizeof(real), TRUE, "h" );
    cuda_malloc( (void **) &workspace->hs, (RESTART+1)*sizeof(real), TRUE, "hs" );
    cuda_malloc( (void **) &workspace->hc, (RESTART+1)*sizeof(real), TRUE, "hc" );
    cuda_malloc( (void **) &workspace->v, (RESTART+1)*(RESTART+1)*sizeof(real), TRUE, "v" );

    /* CG storage */
    cuda_malloc( (void **) &workspace->r, total_cap * sizeof(real), TRUE, "r" );
    cuda_malloc( (void **) &workspace->d, total_cap * sizeof(real), TRUE, "d" );
    cuda_malloc( (void **) &workspace->q, total_cap * sizeof(real), TRUE, "q" );
    cuda_malloc( (void **) &workspace->p, total_cap * sizeof(real), TRUE, "p" );
    cuda_malloc( (void **) &workspace->r2, total_cap * sizeof(rvec2), TRUE, "r2" );
    cuda_malloc( (void **) &workspace->d2, total_cap * sizeof(rvec2), TRUE, "d2" );
    cuda_malloc( (void **) &workspace->q2, total_cap * sizeof(rvec2), TRUE, "q2" );
    cuda_malloc( (void **) &workspace->p2, total_cap * sizeof(rvec2), TRUE, "p2" );

    /* integrator storage */
    cuda_malloc( (void **) &workspace->v_const, local_rvec, TRUE, "v_const" );

    /* storage for analysis */
    if( control->molecular_analysis || control->diffusion_coef )
    {
        cuda_malloc( (void **) &workspace->mark, local_cap * sizeof(int), TRUE, "mark" );
        cuda_malloc( (void **) &workspace->old_mark, local_cap * sizeof(int), TRUE, "old_mark" );
    }
    else
    {
        workspace->mark = workspace->old_mark = NULL;
    }

    if( control->diffusion_coef )
    {
        cuda_malloc( (void **) &workspace->x_old, local_cap * sizeof(rvec), TRUE, "x_old" );
    }
    else
    {
        workspace->x_old = NULL;
    }

    /* force related storage */
    cuda_malloc( (void **) &workspace->f, total_cap * sizeof(rvec), TRUE, "f" );
    cuda_malloc( (void **) &workspace->CdDelta, total_cap * sizeof(rvec), TRUE, "CdDelta" );

    /* Taper params */
    cuda_malloc( (void **) &workspace->Tap, 8 * sizeof(real), TRUE, "Tap" );
}


void dev_dealloc_workspace( control_params *control, storage *workspace )
{
    if ( workspace->allocated == FALSE )
    {
        return;
    }

    workspace->allocated = FALSE;

    /* communication storage */  
    /*
       workspace->tmp_dbl = NULL;
       workspace->tmp_rvec = NULL;
       workspace->tmp_rvec2 = NULL;
     */

    /* bond order related storage  */
    cuda_free( workspace->within_bond_box, "skin" );
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
    cuda_free( workspace->done_after, "done_after" );

    /* QEq storage */
    cuda_free( workspace->Hdia_inv, "Hdia_inv" );
    cuda_free( workspace->b_s, "b_s" );
    cuda_free( workspace->b_t, "b_t" );
    cuda_free( workspace->b_prc, "b_prc" );
    cuda_free( workspace->b_prm, "b_prm" );
    cuda_free( workspace->s, "s" );
    cuda_free( workspace->t, "t" );
    cuda_free( workspace->droptol, "droptol" );
    cuda_free( workspace->b, "b" );
    cuda_free( workspace->x, "x" );

    /* GMRES storage */
    cuda_free( workspace->y, "y" );
    cuda_free( workspace->z, "z" );
    cuda_free( workspace->g, "g" );
    cuda_free( workspace->h, "h" );
    cuda_free( workspace->hs, "hs" );
    cuda_free( workspace->hc, "hc" );
    cuda_free( workspace->v, "v" );

    /* CG storage */
    cuda_free( workspace->r, "r" );
    cuda_free( workspace->d, "d" );
    cuda_free( workspace->q, "q" );
    cuda_free( workspace->p, "p" );
    cuda_free( workspace->r2, "r2" );
    cuda_free( workspace->d2, "d2" );
    cuda_free( workspace->q2, "q2" );
    cuda_free( workspace->p2, "p2" );

    /* integrator storage */
    cuda_free( workspace->v_const, "v_const" );

    /* storage for analysis */
    if( control->molecular_analysis || control->diffusion_coef )
    {
        cuda_free( workspace->mark, "mark" );
        cuda_free( workspace->old_mark, "old_mark" );
    }
    else
    {
        workspace->mark = workspace->old_mark = NULL;
    }

    if( control->diffusion_coef )
    {
        cuda_free (workspace->x_old, "x_old");
    }
    else
    {
        workspace->x_old = NULL;
    }

    /* force related storage */
    cuda_free( workspace->f, "f" );
    cuda_free( workspace->CdDelta, "CdDelta" );

    /* Taper params */
    cuda_free( workspace->Tap, "Tap" );
}


void dev_alloc_matrix( sparse_matrix *H, int cap, int m )
{
    //sparse_matrix *H;
    //H = *pH;

    H->cap = cap;
    H->m = m;
    cuda_malloc( (void **) &H->start, sizeof(int) * cap, TRUE, "matrix_start" );
    cuda_malloc( (void **) &H->end, sizeof(int) * cap, TRUE, "matrix_end" );
    cuda_malloc( (void **) &H->entries, sizeof(sparse_matrix_entry) * m, TRUE, "matrix_entries" );
}


void dev_dealloc_matrix( sparse_matrix *H )
{
    cuda_free( H->start, "matrix_start" );
    cuda_free( H->end, "matrix_end" );
    cuda_free( H->entries, "matrix_entries" );
}


}
