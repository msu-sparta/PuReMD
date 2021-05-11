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

#include "cuda_reduction.h"
#include "cuda_spar_lin_alg.h"
#include "cuda_utils.h"

#include "../basic_comm.h"
#include "../charges.h"
#include "../comm_tools.h"
#include "../tool_box.h"

#include "../cub/cub/device/device_radix_sort.cuh"
#include "../cub/cub/block/block_reduce.cuh"
//#include <cub/device/device_radix_sort.cuh>
//#include <cub/block/block_reduce.cuh>


//TODO: move k_jacob and jacboi to cuda_lin_alg.cu
CUDA_GLOBAL void k_jacobi( reax_atom const * const my_atoms,
        single_body_parameters const * const sbp,
        storage workspace, int n  )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    workspace.Hdia_inv[i] = 1.0 / sbp[ my_atoms[i].type ].eta;
}



static void jacobi( reax_system const * const system,
        control_params const * const control, storage const * const workspace )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE
        + ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_jacobi <<< blocks, DEF_BLOCK_SIZE, 0, control->streams[4] >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, 
          *(workspace->d_workspace), system->n );
    cudaCheckError( );
}


/* Routine used for sorting nonzeros within a sparse matrix row;
 *  internally, a combination of qsort and manual sorting is utilized
 *
 * A: sparse matrix for which to sort nonzeros within a row, stored in CSR format
 */
void Sort_Matrix_Rows( sparse_matrix * const A, reax_system const * const system,
       control_params const * const control )
{
    int i, num_entries, *start, *end, *d_j_temp;
    real *d_val_temp;
    void *d_temp_storage;
    size_t temp_storage_bytes, max_temp_storage_bytes;

    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    max_temp_storage_bytes = 0;

    /* copy row indices from device */
    start = (int *) smalloc( sizeof(int) * system->total_cap, "Sort_Matrix_Rows::start" );
    end = (int *) smalloc( sizeof(int) * system->total_cap, "Sort_Matrix_Rows::end" );
    sCudaMemcpyAsync( start, A->start, sizeof(int) * system->total_cap, 
            cudaMemcpyDeviceToHost, control->streams[4], __FILE__, __LINE__ );
    sCudaMemcpyAsync( end, A->end, sizeof(int) * system->total_cap, 
            cudaMemcpyDeviceToHost, control->streams[4], __FILE__, __LINE__ );

    /* make copies of column indices and non-zero values */
    sCudaMalloc( (void **) &d_j_temp, sizeof(int) * system->total_cm_entries,
            __FILE__, __LINE__ );
    sCudaMalloc( (void **) &d_val_temp, sizeof(real) * system->total_cm_entries,
            __FILE__, __LINE__ );
    sCudaMemcpyAsync( d_j_temp, A->j, sizeof(int) * system->total_cm_entries,
            cudaMemcpyDeviceToDevice, control->streams[4], __FILE__, __LINE__ );
    sCudaMemcpyAsync( d_val_temp, A->val, sizeof(real) * system->total_cm_entries,
            cudaMemcpyDeviceToDevice, control->streams[4], __FILE__, __LINE__ );

    cudaStreamSynchronize( control->streams[4] );

    for ( i = 0; i < system->n; ++i )
    {
        num_entries = end[i] - start[i];

        /* determine temporary device storage requirements */
        cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes,
                &d_j_temp[start[i]], &A->j[start[i]],
                &d_val_temp[start[i]], &A->val[start[i]], num_entries );

        if ( d_temp_storage == NULL )
        {
            /* allocate temporary storage */
            sCudaMalloc( &d_temp_storage, temp_storage_bytes,
                    __FILE__, __LINE__ );
        }
        else if ( max_temp_storage_bytes < temp_storage_bytes )
        {
            /* deallocate temporary storage */
            sCudaFree( d_temp_storage, __FILE__, __LINE__ );

            /* allocate temporary storage */
            sCudaMalloc( &d_temp_storage, temp_storage_bytes,
                    __FILE__, __LINE__ );

            max_temp_storage_bytes = temp_storage_bytes;
        }

        /* run sorting operation */
        cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes,
                &d_j_temp[start[i]], &A->j[start[i]],
                &d_val_temp[start[i]], &A->val[start[i]], num_entries );
        cudaCheckError( );
    }

    /* deallocate temporary storage */
    sCudaFree( d_temp_storage, __FILE__, __LINE__ );
    sCudaFree( d_j_temp, __FILE__, __LINE__ );
    sCudaFree( d_val_temp, __FILE__, __LINE__ );
    sfree( start, "Sort_Matrix_Rows::start" );
    sfree( end, "Sort_Matrix_Rows::end" );
}


CUDA_GLOBAL void k_spline_extrapolate_charges_qeq( reax_atom const * const my_atoms,
        single_body_parameters const * const sbp, control_params const * const control,
        storage workspace, int n  )
{
    int i;
    real s_tmp, t_tmp;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    /* RHS vectors for linear system */
    workspace.b_s[i] = -1.0 * sbp[ my_atoms[i].type ].chi;
    workspace.b_t[i] = -1.0;
#if defined(DUAL_SOLVER)
    workspace.b[i][0] = -1.0 * sbp[ my_atoms[i].type ].chi;
    workspace.b[i][1] = -1.0;
#endif

    /* no extrapolation, previous solution as initial guess */
    if ( control->cm_init_guess_extrap1 == 0 )
    {
        s_tmp = my_atoms[i].s[0];
    }
    /* linear */
    else if ( control->cm_init_guess_extrap1 == 1 )
    {
        s_tmp = 2.0 * my_atoms[i].s[0] - my_atoms[i].s[1];
    }
    /* quadratic */
    else if ( control->cm_init_guess_extrap1 == 2 )
    {
        s_tmp = my_atoms[i].s[2] + 3.0 * (my_atoms[i].s[0] - my_atoms[i].s[1]);
    }
    /* cubic */
    else if ( control->cm_init_guess_extrap1 == 3 )
    {
        s_tmp = 4.0 * (my_atoms[i].s[0] + my_atoms[i].s[2])
            - (6.0 * my_atoms[i].s[1] + my_atoms[i].s[3]);
    }
    else
    {
        s_tmp = 0.0;
    }

    /* no extrapolation, previous solution as initial guess */
    if ( control->cm_init_guess_extrap1 == 0 )
    {
        t_tmp = my_atoms[i].t[0];
    }
    /* linear */
    else if ( control->cm_init_guess_extrap1 == 1 )
    {
        t_tmp = 2.0 * my_atoms[i].t[0] - my_atoms[i].t[1];
    }
    /* quadratic */
    else if ( control->cm_init_guess_extrap1 == 2 )
    {
        t_tmp = my_atoms[i].t[2] + 3.0 * (my_atoms[i].t[0] - my_atoms[i].t[1]);
    }
    /* cubic */
    else if ( control->cm_init_guess_extrap1 == 3 )
    {
        t_tmp = 4.0 * (my_atoms[i].t[0] + my_atoms[i].t[2])
            - (6.0 * my_atoms[i].t[1] + my_atoms[i].t[3]);
    }
    else
    {
        t_tmp = 0.0;
    }

#if defined(DUAL_SOLVER)
    workspace.x[i][0] = s_tmp;
    workspace.x[i][1] = t_tmp;
#else
    workspace.s[i] = s_tmp;
    workspace.t[i] = t_tmp;
#endif
}


static void Spline_Extrapolate_Charges_QEq( reax_system const * const system,
        control_params const * const control,
        simulation_data const * const data,
        storage const * const workspace,
        mpi_datatypes const * const mpi_data )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE
        + ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_spline_extrapolate_charges_qeq <<< blocks, DEF_BLOCK_SIZE, 0,
                                     control->streams[4] >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, 
          (control_params *)control->d_control_params,
          *(workspace->d_workspace), system->n );
    cudaCheckError( );
}


static void Spline_Extrapolate_Charges_EE( reax_system const * const system,
        control_params const * const control,
        simulation_data const * const data,
        storage const * const workspace,
        mpi_datatypes const * const mpi_data )
{
}


static void Setup_Preconditioner_QEq( reax_system const * const system,
        control_params const * const control,
        simulation_data * const data, storage * const workspace,
        mpi_datatypes * const mpi_data )
{
//    int ret;
#if defined(LOG_PERFORMANCE)
    real time;

    time = Get_Time( );
#endif

    /* sort H needed for SpMV's in linear solver, H or H_sp needed for preconditioning */
//    Sort_Matrix_Rows( &workspace->d_workspace->H, system, control );
    
#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_sort );
#endif

    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case JACOBI_PC:
            break;

        case ICHOLT_PC:
        case ILUT_PC:
        case ILUTP_PC:
        case FG_ILUT_PC:
            fprintf( stderr, "[ERROR] Unsupported preconditioner computation method (%d). Terminating...\n",
                   control->cm_solver_pre_comp_type );
            exit( INVALID_INPUT );
            break;

        case SAI_PC:
//            setup_sparse_approx_inverse( system, data, workspace, mpi_data,
//                    &workspace->H, &workspace->H_spar_patt, 
//                    control->nprocs, control->cm_solver_pre_comp_sai_thres );
            break;

        default:
            fprintf( stderr, "[ERROR] Unrecognized preconditioner computation method (%d). Terminating...\n",
                   control->cm_solver_pre_comp_type );
            exit( INVALID_INPUT );
            break;
    }

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_pre_comp );
#endif
}


static void Setup_Preconditioner_EE( reax_system const * const system,
        control_params const * const control,
        simulation_data * const data, storage * const workspace,
        mpi_datatypes const * const mpi_data )
{
}


static void Setup_Preconditioner_ACKS2( reax_system const * const system,
        control_params const * const control,
        simulation_data * const data, storage * const workspace,
        mpi_datatypes const * const mpi_data )
{
}


static void Compute_Preconditioner_QEq( reax_system const * const system,
        control_params const * const control,
        simulation_data * const data, storage * const workspace,
        mpi_datatypes const * const mpi_data )
{
#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
    real t_pc, total_pc;
#endif

    if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        jacobi( system, control, workspace );
    }
    else if ( control->cm_solver_pre_comp_type == SAI_PC )
    {
#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
//        t_pc = sparse_approx_inverse( system, data, workspace, mpi_data,
//                &workspace->H, workspace->H_spar_patt, &workspace->H_app_inv, control->nprocs );
//
//        ret = MPI_Reduce( &t_pc, &total_pc, 1, MPI_DOUBLE, MPI_SUM, MASTER_NODE, MPI_COMM_WORLD );
//        Check_MPI_Error( ret, __FILE__, __LINE__ );
//
//        if( system->my_rank == MASTER_NODE )
//        {
//            data->timing.cm_solver_pre_comp += total_pc / control->nprocs;
//        }
#else
        fprintf( stderr, "[ERROR] LAPACKE support disabled. Re-compile before enabling. Terminating...\n" );
        exit( INVALID_INPUT );
#endif
    }
}


CUDA_GLOBAL void k_extrapolate_charges_qeq_part2( reax_atom *my_atoms,
        storage workspace, real u, real *q, int n )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    /* compute charge based on s & t */
#if defined(DUAL_SOLVER)
    my_atoms[i].q = workspace.x[i][0] - u * workspace.x[i][1];
#else
    my_atoms[i].q = workspace.s[i] - u * workspace.t[i];
#endif
    q[i] = my_atoms[i].q;

    my_atoms[i].s[3] = my_atoms[i].s[2];
    my_atoms[i].s[2] = my_atoms[i].s[1];
    my_atoms[i].s[1] = my_atoms[i].s[0];
#if defined(DUAL_SOLVER)
    my_atoms[i].s[0] = workspace.x[i][0];
#else
    my_atoms[i].s[0] = workspace.s[i];
#endif

    my_atoms[i].t[3] = my_atoms[i].t[2];
    my_atoms[i].t[2] = my_atoms[i].t[1];
    my_atoms[i].t[1] = my_atoms[i].t[0];
#if defined(DUAL_SOLVER)
    my_atoms[i].t[0] = workspace.x[i][1];
#else
    my_atoms[i].t[0] = workspace.t[i];
#endif
}


static void Extrapolate_Charges_QEq_Part2( reax_system const * const system,
        control_params const * const control, storage * const workspace,
        real * const q, real u )
{
    int blocks;
    real *spad;

    blocks = system->n / DEF_BLOCK_SIZE
        + ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    sCudaCheckMalloc( &workspace->scratch[4], &workspace->scratch_size[4],
            sizeof(real) * system->n, __FILE__, __LINE__ );
    spad = (real *) workspace->scratch[4];
    sCudaMemsetAsync( spad, 0, sizeof(real) * system->n,
            control->streams[4], __FILE__, __LINE__ );

    k_extrapolate_charges_qeq_part2 <<< blocks, DEF_BLOCK_SIZE, 0,
                                    control->streams[4] >>>
        ( system->d_my_atoms, *(workspace->d_workspace), u, spad, system->n );
    cudaCheckError( );

    sCudaMemcpyAsync( q, spad, sizeof(real) * system->n, 
            cudaMemcpyDeviceToHost, control->streams[4], __FILE__, __LINE__ );

    cudaStreamSynchronize( control->streams[4] );
}


CUDA_GLOBAL void k_update_ghost_atom_charges( reax_atom *my_atoms, real *q,
        int n, int N )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= (N - n) )
    {
        return;
    }

    my_atoms[n + i].q = q[i];
}


static void Update_Ghost_Atom_Charges( reax_system const * const system,
        control_params const * const control, storage * const workspace,
        real * const q )
{
    int blocks;
    real *spad;

    blocks = (system->N - system->n) / DEF_BLOCK_SIZE
        + (((system->N - system->n) % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    sCudaCheckMalloc( &workspace->scratch[4], &workspace->scratch_size[4],
            sizeof(real) * (system->N - system->n), __FILE__, __LINE__ );
    spad = (real *) workspace->scratch[4];

    sCudaMemcpyAsync( spad, &q[system->n], sizeof(real) * (system->N - system->n),
            cudaMemcpyHostToDevice, control->streams[4], __FILE__, __LINE__ );

    cudaStreamSynchronize( control->streams[4] );

    k_update_ghost_atom_charges <<< blocks, DEF_BLOCK_SIZE, 0,
                                control->streams[4] >>>
        ( system->d_my_atoms, spad, system->n, system->N );
    cudaCheckError( );
}


static void Calculate_Charges_QEq( reax_system const * const system,
        control_params const * const control, storage * const workspace,
        mpi_datatypes * const mpi_data )
{
    int ret;
    size_t s;
    real u, *q;
    rvec2 my_sum, all_sum;
#if defined(DUAL_SOLVER)
    int blocks;
    rvec2 *spad;
#else
    real *spad;
#endif

#if defined(DUAL_SOLVER)
    blocks = system->n / DEF_BLOCK_SIZE
        + ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

#if !defined(CUDA_ACCUM_ATOMIC)
    s = sizeof(rvec2) * (blocks + 1);
#else
    s = sizeof(rvec2);
#endif

    sCudaCheckMalloc( &workspace->scratch[4], &workspace->scratch_size[4],
            s, __FILE__, __LINE__ );
    spad = (rvec2 *) workspace->scratch[4];

    sCudaMemsetAsync( spad, 0, s, control->streams[4], __FILE__, __LINE__ );

    /* compute local sums of pseudo-charges in s and t on device */
    k_reduction_rvec2 <<< blocks, DEF_BLOCK_SIZE,
                      sizeof(cub::BlockReduce<double, DEF_BLOCK_SIZE>::TempStorage),
                      control->streams[4] >>>
        ( workspace->d_workspace->x, spad, system->n );
    cudaCheckError( );

#if !defined(CUDA_ACCUM_ATOMIC)
    k_reduction_rvec2 <<< 1, ((blocks + 31) / 32) * 32,
                      sizeof(cub::BlockReduce<double, DEF_BLOCK_SIZE>::TempStorage),
                      control->streams[4] >>>
        ( spad, &spad[blocks], blocks );
    cudaCheckError( );
#endif

    sCudaMemcpyAsync( &my_sum,
#if !defined(CUDA_ACCUM_ATOMIC)
            &spad[blocks],
#else
            spad,
#endif
            sizeof(rvec2), cudaMemcpyDeviceToHost,
            control->streams[4], __FILE__, __LINE__ );

    cudaStreamSynchronize( control->streams[4] );
#else
    sCudaCheckMalloc( &workspace->scratch[4], &workspace->scratch_size[4],
            sizeof(real) * 2, __FILE__, __LINE__ );
    spad = (real *) workspace->scratch[4];

    /* local reductions (sums) on device */
    Cuda_Reduction_Sum( workspace->d_workspace->s, &spad[0], system->n,
		    control->streams[4] );
    Cuda_Reduction_Sum( workspace->d_workspace->t, &spad[1], system->n,
		    control->streams[4] );

    sCudaMemcpyAsync( my_sum, spad, sizeof(real) * 2,
            cudaMemcpyDeviceToHost, control->streams[4], __FILE__, __LINE__ );

    cudaStreamSynchronize( control->streams[4] );
#endif

    /* global reduction on pseudo-charges for s and t */
    ret = MPI_Allreduce( &my_sum, &all_sum, 2, MPI_DOUBLE,
            MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    u = all_sum[0] / all_sum[1];

    check_smalloc( &workspace->host_scratch, &workspace->host_scratch_size,
            sizeof(real) * system->n, TRUE, SAFE_ZONE,
            "Calculate_Charges_QEq::workspace->host_scratch" );
    q = (real *) workspace->host_scratch;

    /* derive atomic charges from pseudo-charges
     * and set up extrapolation for next time step */
    Extrapolate_Charges_QEq_Part2( system, control, workspace, q, u );

    Dist_FS( system, mpi_data, q, REAL_PTR_TYPE, MPI_DOUBLE );

    /* copy atomic charges to ghost atoms in case of ownership transfer */
    Update_Ghost_Atom_Charges( system, control, workspace, q );
}


static void Calculate_Charges_EE( reax_system const * const system,
        control_params const * const control,
        storage const * const workspace,
        mpi_datatypes * const mpi_data )
{
}


static void Calculate_Charges_ACKS2( reax_system const * const system,
        control_params const * const control,
        storage const * const workspace,
        mpi_datatypes * const mpi_data )
{
}


/* Main driver method for QEq kernel
 *  1) init / setup routines for preconditioning of linear solver
 *  2) compute preconditioner
 *  3) extrapolate charges
 *  4) perform 2 linear solves
 *  5) compute atomic charges based on output of (4)
 */
void QEq( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        output_controls const * const out_control,
        mpi_datatypes * const mpi_data )
{
    int iters;

    iters = 0;

//    if ( is_refactoring_step( control, data ) == TRUE )
    {
        Setup_Preconditioner_QEq( system, control, data, workspace, mpi_data );

        Compute_Preconditioner_QEq( system, control, data, workspace, mpi_data );
    }

//    switch ( control->cm_init_guess_type )
//    {
//    case SPLINE:
        Spline_Extrapolate_Charges_QEq( system, control, data, workspace, mpi_data );
//        break;
//
//    case TF_FROZEN_MODEL_LSTM:
//#if defined(HAVE_TENSORFLOW)
//        if ( data->step < control->cm_init_guess_win_size )
//        {
//            Spline_Extrapolate_Charges_QEq( system, control, data, workspace, mpi_data );
//        }
//        else
//        {
//            Predict_Charges_TF_LSTM( system, control, data, workspace );
//        }
//#else
//        fprintf( stderr, "[ERROR] Tensorflow support disabled. Re-compile to enable. Terminating...\n" );
//        exit( INVALID_INPUT );
//#endif
//        break;
//
//    default:
//        fprintf( stderr, "[ERROR] Unrecognized solver initial guess type (%d). Terminating...\n",
//              control->cm_init_guess_type );
//        exit( INVALID_INPUT );
//        break;
//    }

    switch ( control->cm_solver_type )
    {
    case GMRES_S:
    case GMRES_H_S:
        fprintf( stderr, "[ERROR] Unsupported solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;

    case CG_S:
#if defined(DUAL_SOLVER)
        iters = Cuda_dual_CG( system, control, data, workspace,
                &workspace->d_workspace->H, workspace->d_workspace->b,
                control->cm_solver_q_err, workspace->d_workspace->x, mpi_data );
#else
        iters = Cuda_CG( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_s, control->cm_solver_q_err, workspace->d_workspace->s, mpi_data );
        iters += Cuda_CG( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_t, control->cm_solver_q_err, workspace->d_workspace->t, mpi_data );
#endif
        break;

    case SDM_S:
#if defined(DUAL_SOLVER)
        iters = Cuda_dual_SDM( system, control, data, workspace,
                &workspace->d_workspace->H, workspace->d_workspace->b,
                control->cm_solver_q_err, workspace->d_workspace->x, mpi_data );
#else
        iters = Cuda_SDM( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_s, control->cm_solver_q_err, workspace->d_workspace->s, mpi_data );
        iters += Cuda_SDM( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_t, control->cm_solver_q_err, workspace->d_workspace->t, mpi_data );
#endif
        break;

    case BiCGStab_S:
#if defined(DUAL_SOLVER)
        iters = Cuda_dual_BiCGStab( system, control, data, workspace,
                &workspace->d_workspace->H, workspace->d_workspace->b,
                control->cm_solver_q_err, workspace->d_workspace->x, mpi_data );
#else
        iters = Cuda_BiCGStab( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_s, control->cm_solver_q_err, workspace->d_workspace->s, mpi_data );
        iters += Cuda_BiCGStab( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_t, control->cm_solver_q_err, workspace->d_workspace->t, mpi_data );
#endif
        break;

    case PIPECG_S:
#if defined(DUAL_SOLVER)
        iters = Cuda_dual_PIPECG( system, control, data, workspace,
                &workspace->d_workspace->H, workspace->d_workspace->b,
                control->cm_solver_q_err, workspace->d_workspace->x, mpi_data );
#else
        iters = Cuda_PIPECG( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_s, control->cm_solver_q_err, workspace->d_workspace->s, mpi_data );
        iters += Cuda_PIPECG( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_t, control->cm_solver_q_err, workspace->d_workspace->t, mpi_data );
#endif
        break;

    case PIPECR_S:
#if defined(DUAL_SOLVER)
        iters = Cuda_dual_PIPECR( system, control, data, workspace,
                &workspace->d_workspace->H, workspace->d_workspace->b,
                control->cm_solver_q_err, workspace->d_workspace->x, mpi_data );
#else
        iters = Cuda_PIPECR( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_s, control->cm_solver_q_err, workspace->d_workspace->s, mpi_data );
        iters += Cuda_PIPECR( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_t, control->cm_solver_q_err, workspace->d_workspace->t, mpi_data );
#endif
        break;

    default:
        fprintf( stderr, "[ERROR] Unrecognized solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    Calculate_Charges_QEq( system, control, workspace, mpi_data );

#if defined(LOG_PERFORMANCE)
    data->timing.cm_solver_iters += iters;
#endif
}


void EE( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        output_controls const * const out_control,
        mpi_datatypes * const mpi_data )
{
    fprintf( stderr, "[ERROR] Unsupported charge model (EE). Terminating...\n" );
    exit( INVALID_INPUT );
}


void ACKS2( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        output_controls const * const out_control,
        mpi_datatypes * const mpi_data )
{
    fprintf( stderr, "[ERROR] Unsupported charge model (ACKS2). Terminating...\n" );
    exit( INVALID_INPUT );
}


void Cuda_Compute_Charges( reax_system const * const system,
        control_params const * const control,
        simulation_data * const data,
        storage * const workspace,
        output_controls const * const out_control,
        mpi_datatypes * const mpi_data )
{
    switch ( control->charge_method )
    {
    case QEQ_CM:
        QEq( system, control, data, workspace, out_control, mpi_data );
        break;

    case EE_CM:
        EE( system, control, data, workspace, out_control, mpi_data );
        break;

    case ACKS2_CM:
        ACKS2( system, control, data, workspace, out_control, mpi_data );
        break;

    default:
        fprintf( stderr, "[ERROR] Invalid charge method. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }
}
