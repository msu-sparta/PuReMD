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

#include "cuda_init.h"

#include "cuda_utils.h"
#include "cuda_copy.h"
#include "cuda_reset_utils.h"

#include "vector.h"


void Cuda_Init_System( reax_system *system)
{
    cuda_malloc( (void **) &system->d_atoms, system->N * REAX_ATOM_SIZE, 1, RES_SYSTEM_ATOMS );    

    cuda_malloc( (void **) &system->d_box, sizeof (simulation_box), 1, RES_SYSTEM_SIMULATION_BOX );

    //interaction parameters
    cuda_malloc((void **) &system->reaxprm.d_sbp, system->reaxprm.num_atom_types * SBP_SIZE,
            1, RES_REAX_INT_SBP );

    cuda_malloc((void **) &system->reaxprm.d_tbp, pow (system->reaxprm.num_atom_types, 2) * TBP_SIZE, 
            1, RES_REAX_INT_TBP );

    cuda_malloc((void **) &system->reaxprm.d_thbp, pow (system->reaxprm.num_atom_types, 3) * THBP_SIZE,
            1, RES_REAX_INT_THBP );

    cuda_malloc((void **) &system->reaxprm.d_hbp, pow (system->reaxprm.num_atom_types, 3) * HBP_SIZE,
            1, RES_REAX_INT_HBP );

    cuda_malloc((void **) &system->reaxprm.d_fbp, pow (system->reaxprm.num_atom_types, 4) * FBP_SIZE,
            1, RES_REAX_INT_FBP );

    cuda_malloc((void **) &system->reaxprm.d_gp.l, REAL_SIZE * system->reaxprm.gp.n_global, 1, RES_GLOBAL_PARAMS );

    system->reaxprm.d_gp.n_global = 0;
    system->reaxprm.d_gp.vdw_type = 0;
}


void Cuda_Init_Control(control_params *control)
{
    cuda_malloc((void **)&control->d_control, CONTROL_PARAMS_SIZE, 1, RES_CONTROL_PARAMS );
    copy_host_device(control, control->d_control, CONTROL_PARAMS_SIZE, cudaMemcpyHostToDevice, RES_CONTROL_PARAMS );
}


void Cuda_Init_Simulation_Data (simulation_data *data)
{
    cuda_malloc((void **) &(data->d_simulation_data), SIMULATION_DATA_SIZE, 1, RES_SIMULATION_DATA );
}


GLOBAL void Initialize_Grid(ivec *nbrs, rvec *nbrs_cp, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
    {
        return;
    }

    nbrs[index][0] = -1;
    nbrs[index][1] = -1;
    nbrs[index][2] = -1;
    nbrs_cp[index][0] = -1;
    nbrs_cp[index][1] = -1;
    nbrs_cp[index][2] = -1;
}


void Cuda_Init_Grid (grid *host, grid *dev)
{
    int total = host->ncell[0] * host->ncell[1] * host->ncell[2];
    dev->max_atoms = host->max_atoms;
    dev->max_nbrs = host->max_nbrs;
    dev->total = host->total;
    dev->max_cuda_nbrs = host->max_cuda_nbrs;
    dev->cell_size = host->cell_size;

    ivec_Copy( dev->spread, host->spread );
    ivec_Copy( dev->ncell, host->ncell );
    rvec_Copy( dev->len, host->len );
    rvec_Copy( dev->inv_len, host->inv_len );

    cuda_malloc((void **) &dev->top, INT_SIZE * total , 1, RES_GRID_TOP );
    cuda_malloc((void **) &dev->mark, INT_SIZE * total , 1, RES_GRID_MARK );
    cuda_malloc((void **) &dev->start, INT_SIZE * total , 1, RES_GRID_START );
    cuda_malloc((void **) &dev->end, INT_SIZE * total , 1, RES_GRID_END );

    cuda_malloc((void **) &dev->atoms, INT_SIZE * total * host->max_atoms, 1, RES_GRID_ATOMS );
    cuda_malloc((void **) &dev->nbrs, IVEC_SIZE * total * host->max_nbrs, 0, RES_GRID_NBRS );
    cuda_malloc((void **) &dev->nbrs_cp, RVEC_SIZE * total * host->max_nbrs, 0, RES_GRID_NBRS_CP );

    int block_size = 512;
    int blocks = (total*dev->max_nbrs) / block_size + ((total*dev->max_nbrs) % block_size == 0 ? 0 : 1);

    Initialize_Grid<<<blocks, block_size>>>
        ( dev->nbrs, dev->nbrs_cp, total * host->max_nbrs );
    cudaThreadSynchronize( );
    cudaCheckError( );
}


GLOBAL void Init_Workspace_Arrays(single_body_parameters *sbp, reax_atom *atoms, 
        static_storage workspace, int N)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;

    workspace.Hdia_inv[i] = 1./sbp[atoms[i].type].eta;
    workspace.b_s[i] = -sbp[ atoms[i].type ].chi;
    workspace.b_t[i] = -1.0;

    workspace.b[i] = -sbp[ atoms[i].type ].chi;
    workspace.b[i+N] = -1.0;
}


GLOBAL void Init_Map_Serials (int *input, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    input[i] = -1;
}


void Cuda_Init_Workspace_System (reax_system *system, static_storage *workspace )
{
    int blocks, block_size = BLOCK_SIZE;
    compute_blocks (&blocks, &block_size, MAX_ATOM_ID );

    cuda_malloc ( (void **) &workspace->map_serials, INT_SIZE * MAX_ATOM_ID, 0, RES_STORAGE_MAP_SERIALS );

    Init_Map_Serials <<< blocks, block_size >>> 
        ( workspace->map_serials, MAX_ATOM_ID );
    cudaThreadSynchronize ();
    cudaCheckError ();

    cuda_malloc ( (void **) &workspace->orig_id, INT_SIZE * system->N, 0, RES_STORAGE_ORIG_ID );
    cuda_malloc ( (void **) &workspace->restricted, INT_SIZE * system->N, 0, RES_STORAGE_RESTRICTED );
    cuda_malloc ( (void **) &workspace->restricted_list, system->N * MAX_RESTRICT * INT_SIZE, 0, RES_STORAGE_RESTRICTED_LIST );
}


void Cuda_Init_Workspace( reax_system *system, control_params *control,
        static_storage *workspace )
{
    int i;

    /* Allocate space for hydrogen bond list */
    cuda_malloc ((void **) &workspace->hbond_index,             system->N * INT_SIZE, 0, RES_STORAGE_HBOND_INDEX );

    /* bond order related storage  */
    cuda_malloc ((void **) &workspace->total_bond_order,     system->N * REAL_SIZE, 0, RES_STORAGE_TOTAL_BOND_ORDER );
    cuda_malloc ((void **) &workspace->Deltap,                 system->N * REAL_SIZE, 0, RES_STORAGE_DELTAP );
    cuda_malloc ((void **) &workspace->Deltap_boc,             system->N * REAL_SIZE, 0, RES_STORAGE_DELTAP_BOC );
    cuda_malloc ((void **) &workspace->dDeltap_self,          system->N * RVEC_SIZE, 0, RES_STORAGE_DDELTAP_SELF );

    cuda_malloc ((void **) &workspace->Delta,                  system->N * REAL_SIZE, 0, RES_STORAGE_DELTA );
    cuda_malloc ((void **) &workspace->Delta_lp,           system->N * REAL_SIZE, 0, RES_STORAGE_DELTA_LP );
    cuda_malloc ((void **) &workspace->Delta_lp_temp,      system->N * REAL_SIZE, 0, RES_STORAGE_DELTA_LP_TEMP );
    cuda_malloc ((void **) &workspace->dDelta_lp,          system->N * REAL_SIZE, 0, RES_STORAGE_DDELTA_LP );
    cuda_malloc ((void **) &workspace->dDelta_lp_temp,     system->N * REAL_SIZE, 0, RES_STORAGE_DDELTA_LP_TEMP );
    cuda_malloc ((void **) &workspace->Delta_e,            system->N * REAL_SIZE, 0, RES_STORAGE_DELTA_E );
    cuda_malloc ((void **) &workspace->Delta_boc,          system->N * REAL_SIZE, 0, RES_STORAGE_DELTA_BOC );
    cuda_malloc ((void **) &workspace->nlp,                system->N * REAL_SIZE, 0, RES_STORAGE_NLP );
    cuda_malloc ((void **) &workspace->nlp_temp,           system->N * REAL_SIZE, 0, RES_STORAGE_NLP_TEMP );
    cuda_malloc ((void **) &workspace->Clp,                system->N * REAL_SIZE, 0, RES_STORAGE_CLP );
    cuda_malloc ((void **) &workspace->CdDelta,            system->N * REAL_SIZE, 0, RES_STORAGE_CDDELTA );
    cuda_malloc ((void **) &workspace->vlpex,              system->N * REAL_SIZE, 0, RES_STORAGE_VLPEX );

    /* QEq storage */
    workspace->H.start        = NULL;
    //cuda_malloc ((void **) &workspace->H.start,              (system->N+1)* INT_SIZE, 0, RES_SPARSE_MATRIX_INDEX );
    workspace->L.start        = NULL;
    //cuda_malloc ((void **) &workspace->L.start,              (system->N+1)* INT_SIZE, 0, RES_SPARSE_MATRIX_INDEX );
    workspace->U.start        = NULL;
    //cuda_malloc ((void **) &workspace->U.start,              (system->N+1)* INT_SIZE, 0, RES_SPARSE_MATRIX_INDEX );

    workspace->H.end            = NULL;
    //cuda_malloc ((void **) &workspace->H.end,              (system->N+1)* INT_SIZE, 0, RES_SPARSE_MATRIX_INDEX );
    workspace->L.end            = NULL;
    //cuda_malloc ((void **) &workspace->L.end,              (system->N+1)* INT_SIZE, 0, RES_SPARSE_MATRIX_INDEX );
    workspace->U.end            = NULL;
    //cuda_malloc ((void **) &workspace->U.end,              (system->N+1)* INT_SIZE, 0, RES_SPARSE_MATRIX_INDEX );

    workspace->H.entries     = NULL;
    workspace->L.entries     = NULL;
    workspace->U.entries     = NULL;

    cuda_malloc ((void **) &workspace->droptol,  system->N * REAL_SIZE, 1, RES_STORAGE_DROPTOL );
    cuda_malloc ((void **) &workspace->w,        system->N * REAL_SIZE, 1, RES_STORAGE_W );
    cuda_malloc ((void **) &workspace->Hdia_inv, system->N * REAL_SIZE, 1, RES_STORAGE_HDIA_INV );
    cuda_malloc ((void **) &workspace->b,        system->N * 2 * REAL_SIZE, 1, RES_STORAGE_B );
    cuda_malloc ((void **) &workspace->b_s,      system->N * REAL_SIZE, 1, RES_STORAGE_B_S );
    cuda_malloc ((void **) &workspace->b_t,      system->N * REAL_SIZE, 1, RES_STORAGE_B_T );
    cuda_malloc ((void **) &workspace->b_prc,    system->N * 2 * REAL_SIZE, 1, RES_STORAGE_B_PRC );
    cuda_malloc ((void **) &workspace->b_prm,    system->N * 2 * REAL_SIZE, 1, RES_STORAGE_B_PRM );
    cuda_malloc ((void **) &workspace->s_t,      system->N * 2 * REAL_SIZE, 1, RES_STORAGE_S_T );
    cuda_malloc ((void **) &workspace->s,        5 * system->N * REAL_SIZE, 1, RES_STORAGE_S );
    cuda_malloc ((void **) &workspace->t,        5 * system->N * REAL_SIZE, 1, RES_STORAGE_T );

    Init_Workspace_Arrays  <<<BLOCKS, BLOCK_SIZE>>>
        (system->reaxprm.d_sbp, system->d_atoms, *workspace, system->N );
    cudaThreadSynchronize ();
    cudaCheckError ();

    /* GMRES storage */
    cuda_malloc ((void **) &workspace->y,  (RESTART+1) * REAL_SIZE, 1, RES_STORAGE_Y );
    cuda_malloc ((void **) &workspace->z,  (RESTART+1) * REAL_SIZE, 1, RES_STORAGE_Z );
    cuda_malloc ((void **) &workspace->g,  (RESTART+1) * REAL_SIZE, 1, RES_STORAGE_G );
    cuda_malloc ((void **) &workspace->hs, (RESTART+1) * REAL_SIZE, 1, RES_STORAGE_HS );
    cuda_malloc ((void **) &workspace->hc, (RESTART+1) * REAL_SIZE, 1, RES_STORAGE_HC );

    cuda_malloc ((void **) &workspace->rn, (RESTART+1)*system->N * 2 * REAL_SIZE, 1, RES_STORAGE_RN );
    cuda_malloc ((void **) &workspace->v,  (RESTART+1)*system->N * REAL_SIZE, 1, RES_STORAGE_V );
    cuda_malloc ((void **) &workspace->h,  (RESTART+1)*(RESTART+1) * REAL_SIZE, 1, RES_STORAGE_H );

    /* CG storage */
    cuda_malloc ((void **) &workspace->r, system->N * REAL_SIZE, 1, RES_STORAGE_R );
    cuda_malloc ((void **) &workspace->d, system->N * REAL_SIZE, 1, RES_STORAGE_D );
    cuda_malloc ((void **) &workspace->q, system->N * REAL_SIZE, 1, RES_STORAGE_Q );
    cuda_malloc ((void **) &workspace->p, system->N * REAL_SIZE, 1, RES_STORAGE_P );


    /* integrator storage */
    cuda_malloc ((void **) &workspace->a,       system->N * RVEC_SIZE, 1, RES_STORAGE_A );
    cuda_malloc ((void **) &workspace->f_old,  system->N * RVEC_SIZE, 1, RES_STORAGE_F_OLD );
    cuda_malloc ((void **) &workspace->v_const,system->N * RVEC_SIZE, 1, RES_STORAGE_V_CONST );

    /* storage for analysis */
    if( control->molec_anal || control->diffusion_coef )
    {
        cuda_malloc ((void **) &workspace->mark,       system->N * INT_SIZE, 1, RES_STORAGE_MARK );
        cuda_malloc ((void **) &workspace->old_mark, system->N * INT_SIZE, 1, RES_STORAGE_OLD_MARK);
    }
    else
        workspace->mark = workspace->old_mark = NULL;

    if( control->diffusion_coef )
        cuda_malloc ((void **) &workspace->x_old,  system->N * RVEC_SIZE, 1, RES_STORAGE_X_OLD );
    else workspace->x_old = NULL;

    workspace->realloc.num_far = -1;
    workspace->realloc.Htop = -1;
    workspace->realloc.hbonds = -1;
    workspace->realloc.bonds = -1;
    workspace->realloc.num_3body = -1;
    workspace->realloc.gcell_atoms = -1;

    Cuda_Reset_Workspace( system, workspace );
}


void Cuda_Init_Workspace_Device ( static_storage *workspace )
{
    workspace->realloc.estimate_nbrs = -1;
    workspace->realloc.num_far = -1;
    workspace->realloc.Htop = -1;
    workspace->realloc.hbonds = -1;
    workspace->realloc.bonds = -1;
    workspace->realloc.num_3body = -1;
    workspace->realloc.gcell_atoms = -1;
}


void Cuda_Init_Sparse_Matrix (sparse_matrix *matrix, int entries, int N)
{
    cuda_malloc ((void **) &matrix->start, INT_SIZE * (N + 1), 1, RES_SPARSE_MATRIX_INDEX );
    cuda_malloc ((void **) &matrix->end, INT_SIZE * (N + 1), 1, RES_SPARSE_MATRIX_INDEX );
    cuda_malloc ((void **) &matrix->entries, SPARSE_MATRIX_ENTRY_SIZE * entries, 1, RES_SPARSE_MATRIX_ENTRY );

    cuda_malloc ((void **) &matrix->j, INT_SIZE * entries, 1, RES_SPARSE_MATRIX_ENTRY );
    cuda_malloc ((void **) &matrix->val, REAL_SIZE * entries, 1, RES_SPARSE_MATRIX_ENTRY );

}


void Cuda_Init_Scratch()
{
    cuda_malloc ((void **) &scratch, SCRATCH_SIZE, 0, RES_SCRATCH );

    /*
       cudaError_t retval = cudaErrorInvalidDevice;

       retval = cudaMallocHost ( (void **) &scratch, SCRATCH_SIZE );
    //retval = cudaHostAlloc ((void **) &scratch, SCRATCH_SIZE, cudaHostAllocDefault );
    if (retval != cudaSuccess)
    {
    fprintf (stderr, "Error allocating the scratch area on the device \n");
    exit (0);
    }
     */
}
