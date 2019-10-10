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

#include "cuda_charges.h"

#include "cuda_lin_alg.h"
#include "cuda_reduction.h"
#include "cuda_utils.h"

#include "../basic_comm.h"


CUDA_GLOBAL void k_init_matvec( reax_atom *my_atoms, single_body_parameters
        *sbp, storage p_workspace, int n  )
{
    int i;
    storage *workspace;
    reax_atom *atom;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    workspace = &p_workspace;
    atom = &my_atoms[i];

    /* init pre-conditioner for H and init solution vectors */
    workspace->Hdia_inv[i] = 1. / sbp[ atom->type ].eta;
    workspace->b_s[i] = -sbp[ atom->type ].chi;
    workspace->b_t[i] = -1.0;
    workspace->b[i][0] = -sbp[ atom->type ].chi;
    workspace->b[i][1] = -1.0;

    workspace->x[i][1] = atom->t[2] + 3 * ( atom->t[0] - atom->t[1] );

    /* cubic extrapolation for s and t */
    workspace->x[i][0] = 4*(atom->s[0]+atom->s[2])-(6*atom->s[1]+atom->s[3]);
}


void Cuda_Init_MatVec( reax_system *system, storage *workspace )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE
        + (( system->n % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);

    k_init_matvec <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, 
          *dev_workspace, system->n );
    cudaDeviceSynchronize();
    cudaCheckError();
}


void cuda_charges_x( reax_system *system, rvec2 my_sum )
{
    int blocks;
    rvec2 *output = (rvec2 *) scratch;

    cuda_memset( output, 0, sizeof(rvec2) * 2 * system->n, "cuda_charges_x:q" );

    blocks = system->n / DEF_BLOCK_SIZE
        + (( system->n % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);

    k_reduction_rvec2 <<< blocks, DEF_BLOCK_SIZE, sizeof(rvec2) * DEF_BLOCK_SIZE >>>
        ( dev_workspace->x, output, system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    k_reduction_rvec2 <<< 1, BLOCKS_POW_2, sizeof(rvec2) * BLOCKS_POW_2 >>>
        ( output, output + system->n, blocks );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    copy_host_device( my_sum, output + system->n,
            sizeof(rvec2), cudaMemcpyDeviceToHost, "charges:x" );
}


CUDA_GLOBAL void k_calculate_st( reax_atom *my_atoms, storage p_workspace, 
        real u, real *q, int n )
{
    storage *workspace;
    reax_atom *atom;
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    workspace = &p_workspace;
    atom = &my_atoms[i];

    //atom->q = workspace->s[i] - u * workspace->t[i];
    q[i] = atom->q = workspace->x[i][0] - u * workspace->x[i][1];

    atom->s[3] = atom->s[2];
    atom->s[2] = atom->s[1];
    atom->s[1] = atom->s[0];
    //atom->s[0] = workspace->s[i];
    atom->s[0] = workspace->x[i][0];

    atom->t[3] = atom->t[2];
    atom->t[2] = atom->t[1];
    atom->t[1] = atom->t[0];
    //atom->t[0] = workspace->t[i];
    atom->t[0] = workspace->x[i][1];
}


extern "C" void cuda_charges_st( reax_system *system, storage *workspace,
        real *output, real u )
{
    int blocks;
    real *tmp = (real *) scratch;
    real *tmp_output = (real *) host_scratch;

    cuda_memset( tmp, 0, sizeof(real) * system->n, "charges:q" );
    memset( tmp_output, 0, sizeof(real) * system->n );

    blocks = system->n / DEF_BLOCK_SIZE
        + (( system->n % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);

    k_calculate_st <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, *dev_workspace, u, tmp, system->n);
    cudaDeviceSynchronize( );
    cudaCheckError( );

    copy_host_device( output, tmp, sizeof(real) * system->n, 
            cudaMemcpyDeviceToHost, "charges:q" );
}


CUDA_GLOBAL void k_update_q( reax_atom *my_atoms, real *q, int n, int N )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= (N - n) )
    {
        return;
    }

    my_atoms[i + n].q = q[i + n];
}


void cuda_charges_updateq( reax_system *system, real *q )
{
    int blocks;
    real *dev_q = (real *) scratch;

    copy_host_device( q, dev_q, system->N * sizeof(real),
            cudaMemcpyHostToDevice, "charges:q" );

    blocks = (system->N - system->n) / DEF_BLOCK_SIZE
        + (( (system->N - system->n) % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);

    k_update_q <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, dev_q, system->n, system->N );
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


void Cuda_Calculate_Charges( reax_system *system, storage *workspace,
        mpi_datatypes *mpi_data )
{
    real u;//, s_sum, t_sum;
    rvec2 my_sum, all_sum;
    real *q;

    my_sum[0] = 0.0;
    my_sum[1] = 0.0;
    q = (real *) host_scratch;
    memset( q, 0, system->N * sizeof(real) );

    cuda_charges_x( system, my_sum );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "Device: my_sum[0]: %f, my_sum[1]: %f\n",
            my_sum[0], my_sum[1] );
#endif

    MPI_Allreduce( &my_sum, &all_sum, 2, MPI_DOUBLE, MPI_SUM, mpi_data->world );

    u = all_sum[0] / all_sum[1];

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "Device: u: %f \n", u );
#endif

    cuda_charges_st( system, workspace, q, u );

    Dist( system, mpi_data, q, REAL_PTR_TYPE, MPI_DOUBLE );

    cuda_charges_updateq( system, q );
}


void Cuda_QEq( reax_system *system, control_params *control, simulation_data
        *data, storage *workspace, output_controls *out_control, mpi_datatypes
        *mpi_data )
{
    int iters;

    Cuda_Init_MatVec( system, workspace );

    //if (data->step > 0) {
    //    compare_rvec2 (workspace->b, dev_workspace->b, system->n, "b");
    //    compare_rvec2 (workspace->x, dev_workspace->x, system->n, "x");
    // compare_array (workspace->b_s, dev_workspace->b_s, system->n, "b_s");
    // compare_array (workspace->b_t, dev_workspace->b_t, system->n, "b_t");
    //}

    switch ( control->cm_solver_type )
    {
    case GMRES_S:
    case GMRES_H_S:
    case SDM_S:
        fprintf( stderr, "Unsupported QEq solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;

    case CG_S:
        iters = Cuda_dual_CG( system, control, workspace, &dev_workspace->H,
                dev_workspace->b, control->cm_solver_q_err, dev_workspace->x, mpi_data,
                out_control->log, data );
        break;


    default:
        fprintf( stderr, "Unrecognized QEq solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    Cuda_Calculate_Charges( system, workspace, mpi_data );

#if defined(LOG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        data->timing.cm_solver_iters += iters;
    }
#endif
}


void Cuda_EE( reax_system *system, control_params *control, simulation_data
        *data, storage *workspace, output_controls *out_control, mpi_datatypes
        *mpi_data )
{
    int iters;

    Cuda_Init_MatVec( system, workspace );

    switch ( control->cm_solver_type )
    {
    case GMRES_S:
    case GMRES_H_S:
    case SDM_S:
        fprintf( stderr, "Unsupported QEq solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;

    case CG_S:
        iters = Cuda_CG( system, control, workspace, &dev_workspace->H,
                dev_workspace->b_s, control->cm_solver_q_err, dev_workspace->s, mpi_data );
        break;


    default:
        fprintf( stderr, "Unrecognized QEq solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    Cuda_Calculate_Charges( system, workspace, mpi_data );

#if defined(LOG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        data->timing.cm_solver_iters += iters;
    }
#endif
}


void Cuda_ACKS2( reax_system *system, control_params *control, simulation_data
        *data, storage *workspace, output_controls *out_control, mpi_datatypes
        *mpi_data )
{
    int iters;

    Cuda_Init_MatVec( system, workspace );

    switch ( control->cm_solver_type )
    {
    case GMRES_S:
    case GMRES_H_S:
    case SDM_S:
        fprintf( stderr, "Unsupported QEq solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;

    case CG_S:
        iters = Cuda_CG( system, control, workspace, &dev_workspace->H,
                dev_workspace->b_s, control->cm_solver_q_err, dev_workspace->s, mpi_data );
        break;


    default:
        fprintf( stderr, "[ERROR] Unrecognized QEq solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    Cuda_Calculate_Charges( system, workspace, mpi_data );

#if defined(LOG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        data->timing.cm_solver_iters += iters;
    }
#endif
}


void Cuda_Compute_Charges( reax_system *system, control_params *control, simulation_data
        *data, storage *workspace, output_controls *out_control, mpi_datatypes
        *mpi_data )
{
    switch ( control->charge_method )
    {
    case QEQ_CM:
        Cuda_QEq( system, control, data, workspace, out_control, mpi_data );
        break;

    case EE_CM:
        Cuda_EE( system, control, data, workspace, out_control, mpi_data );
        break;

    case ACKS2_CM:
        Cuda_ACKS2( system, control, data, workspace, out_control, mpi_data );
        break;

    default:
        fprintf( stderr, "Invalid charge method. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }
}
