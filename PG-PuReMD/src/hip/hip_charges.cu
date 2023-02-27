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

#include "hip_charges.h"

#include "hip_allocate.h"
#include "hip_copy.h"
#include "hip_reduction.h"
#include "hip_spar_lin_alg.h"
#include "hip_utils.h"

#include "../allocate.h"
#include "../charges.h"
#include "../comm_tools.h"
#include "../lin_alg.h"
#include "../tool_box.h"

#if !defined(OMPI_HAVE_MPI_EXT_ROCM) || !OMPI_HAVE_MPI_EXT_ROCM
  #include "../basic_comm.h"
#else
  #include "hip_basic_comm.h"
#endif

#include <hipcub/device/device_radix_sort.hpp>


//TODO: move k_jacob and jacboi to hip_lin_alg.cu
GPU_GLOBAL void k_jacobi( reax_atom const * const my_atoms,
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


GPU_GLOBAL void k_spline_extrapolate_charges_qeq( reax_atom const * const my_atoms,
        single_body_parameters const * const sbp, control_params const * const control,
        storage workspace, int n )
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


GPU_GLOBAL void k_extrapolate_charges_qeq_part2( reax_atom *my_atoms,
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


GPU_GLOBAL void k_update_ghost_atom_charges( reax_atom *my_atoms, real *q,
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


static void jacobi( reax_system const * const system,
        control_params const * const control, storage const * const workspace,
        hipStream_t s )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE
        + ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_jacobi <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, 
          *(workspace->d_workspace), system->n );
    hipCheckError( );
}


/* Routine used for sorting nonzeros within a sparse matrix row;
 *  internally, a combination of qsort and manual sorting is utilized
 *
 * A: sparse matrix for which to sort nonzeros within a row, stored in CSR format
 */
void Sort_Matrix_Rows( sparse_matrix * const A, storage * const workspace,
        hipStream_t s )
{
    int i, *start, *end, *d_j_temp;
    real *d_val_temp;
    void *d_temp_storage;
    size_t temp_storage_bytes, max_temp_storage_bytes;

    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    max_temp_storage_bytes = 0;

    /* copy row indices from device */
    start = (int *) smalloc( sizeof(int) * A->n_max, __FILE__, __LINE__ );
    end = (int *) smalloc( sizeof(int) * A->n_max, __FILE__, __LINE__ );
    sHipMemcpyAsync( start, A->start, sizeof(int) * A->n_max, 
            hipMemcpyDeviceToHost, s, __FILE__, __LINE__ );
    sHipMemcpyAsync( end, A->end, sizeof(int) * A->n_max, 
            hipMemcpyDeviceToHost, s, __FILE__, __LINE__ );

    /* make copies of column indices and non-zero values */
    sHipCheckMalloc( &workspace->scratch[5], &workspace->scratch_size[5],
            (sizeof(int) + sizeof(real)) * A->m, __FILE__, __LINE__ );
    d_j_temp = (int *) workspace->scratch[5];
    d_val_temp = (real *) &((int *) workspace->scratch[5])[A->m];
    sHipMemcpyAsync( d_j_temp, A->j, sizeof(int) * A->m,
            hipMemcpyDeviceToDevice, s, __FILE__, __LINE__ );
    sHipMemcpyAsync( d_val_temp, A->val, sizeof(real) * A->m,
            hipMemcpyDeviceToDevice, s, __FILE__, __LINE__ );

    for ( i = 0; i < A->n; ++i )
    {
        /* determine temporary device storage requirements */
        hipcub::DeviceRadixSort::SortPairs( NULL, temp_storage_bytes,
                &d_j_temp[start[i]], &A->j[start[i]],
                &d_val_temp[start[i]], &A->val[start[i]],
                end[i] - start[i], 0, sizeof(int) * 8, s );
        hipCheckError( );

        if ( d_temp_storage == NULL )
        {
            /* allocate temporary storage */
            sHipMalloc( &d_temp_storage, temp_storage_bytes,
                    __FILE__, __LINE__ );

            max_temp_storage_bytes = temp_storage_bytes;
        }
        else if ( max_temp_storage_bytes < temp_storage_bytes )
        {
            /* deallocate temporary storage */
            sHipFree( d_temp_storage, __FILE__, __LINE__ );

            /* allocate temporary storage */
            sHipMalloc( &d_temp_storage, temp_storage_bytes,
                    __FILE__, __LINE__ );

            max_temp_storage_bytes = temp_storage_bytes;
        }

        /* run sorting operation */
        hipcub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes,
                &d_j_temp[start[i]], &A->j[start[i]],
                &d_val_temp[start[i]], &A->val[start[i]],
                end[i] - start[i], 0, sizeof(int) * 8, s);
        hipCheckError( );
    }

    /* deallocate temporary storage */
    sHipFree( d_temp_storage, __FILE__, __LINE__ );
    sfree( start, __FILE__, __LINE__ );
    sfree( end, __FILE__, __LINE__ );
}


static void Spline_Extrapolate_Charges_QEq( reax_system const * const system,
        control_params const * const control,
        simulation_data const * const data,
        storage const * const workspace,
        mpi_datatypes const * const mpi_data, hipStream_t s )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE
        + ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_spline_extrapolate_charges_qeq <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, 
          (control_params *)control->d_control_params,
          *(workspace->d_workspace), system->n );
    hipCheckError( );
}


static void Spline_Extrapolate_Charges_EE( reax_system const * const system,
        control_params const * const control,
        simulation_data const * const data,
        storage const * const workspace,
        mpi_datatypes const * const mpi_data, hipStream_t s )
{
}


static void Setup_Preconditioner_QEq( reax_system const * const system,
        control_params const * const control,
        simulation_data * const data, storage * const workspace,
        mpi_datatypes * const mpi_data, hipStream_t s )
{
//    int ret;
#if defined(LOG_PERFORMANCE)
    real time;

    time = Get_Time( );
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
            if ( workspace->H.allocated == FALSE )
            {
                Allocate_Matrix( &workspace->H,
                        workspace->d_workspace->H.n, workspace->d_workspace->H.n_max,
                        workspace->d_workspace->H.m, workspace->d_workspace->H.format );
            }
            else if ( workspace->H.m < workspace->d_workspace->H.m
                   || workspace->H.n_max < workspace->d_workspace->H.n_max )
            {
                Deallocate_Matrix( &workspace->H );
                Allocate_Matrix( &workspace->H,
                        workspace->d_workspace->H.n, workspace->d_workspace->H.n_max,
                        workspace->d_workspace->H.m, workspace->d_workspace->H.format );
            }

            Hip_Copy_Matrix_Device_to_Host( &workspace->H,
                    &workspace->d_workspace->H, s );

            setup_sparse_approx_inverse( system, data,
                    &workspace->H, &workspace->H_spar_patt, 
                    control->nprocs, control->cm_solver_pre_comp_sai_thres );
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
        mpi_datatypes const * const mpi_data, hipStream_t s )
{
}


static void Setup_Preconditioner_ACKS2( reax_system const * const system,
        control_params const * const control,
        simulation_data * const data, storage * const workspace,
        mpi_datatypes const * const mpi_data, hipStream_t s )
{
}


static void Compute_Preconditioner_QEq( reax_system const * const system,
        control_params const * const control,
        simulation_data * const data, storage * const workspace,
        mpi_datatypes * const mpi_data, hipStream_t s )
{
#if defined(LOG_PERFORMANCE)
    real time;

    time = Get_Time( );
#endif

    if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        jacobi( system, control, workspace, s );
    }
    else if ( control->cm_solver_pre_comp_type == SAI_PC )
    {
#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
        sparse_approx_inverse( system, data, mpi_data,
                &workspace->H, &workspace->H_spar_patt,
                &workspace->H_app_inv, control->nprocs );

        if ( workspace->d_workspace->H_app_inv.allocated == FALSE )
        {
            Hip_Allocate_Matrix( &workspace->d_workspace->H_app_inv,
                    workspace->H_app_inv.n, workspace->H_app_inv.n_max,
                    workspace->H_app_inv.m, workspace->H_app_inv.format, s );
        }
        else if ( workspace->d_workspace->H_app_inv.m < workspace->H_app_inv.m
               || workspace->d_workspace->H_app_inv.n_max < workspace->H_app_inv.n_max )
        {
            Hip_Deallocate_Matrix( &workspace->d_workspace->H_app_inv );
            Hip_Allocate_Matrix( &workspace->d_workspace->H_app_inv,
                    workspace->H_app_inv.n, workspace->H_app_inv.n_max,
                    workspace->H_app_inv.m, workspace->H_app_inv.format, s );
        }

        Hip_Copy_Matrix_Host_to_Device( &workspace->H_app_inv,
                &workspace->d_workspace->H_app_inv, s );

#else
        fprintf( stderr, "[ERROR] LAPACKE support disabled. Re-compile before enabling. Terminating...\n" );
        exit( INVALID_INPUT );
#endif
    }

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_pre_comp );
#endif
}


static void Extrapolate_Charges_QEq_Part2( reax_system const * const system,
        control_params const * const control, storage * const workspace,
        real * const q, real u, hipStream_t s )
{
    int blocks;
#if !defined(OMPI_HAVE_MPI_EXT_ROCM) || !OMPI_HAVE_MPI_EXT_ROCM
    real *spad;
#endif

    blocks = system->n / DEF_BLOCK_SIZE
        + ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

#if !defined(OMPI_HAVE_MPI_EXT_ROCM) || !OMPI_HAVE_MPI_EXT_ROCM
    sHipCheckMalloc( &workspace->scratch[5], &workspace->scratch_size[5],
            sizeof(real) * system->n, __FILE__, __LINE__ );
    spad = (real *) workspace->scratch[5];
    sHipMemsetAsync( spad, 0, sizeof(real) * system->n,
            s, __FILE__, __LINE__ );
#endif

    k_extrapolate_charges_qeq_part2 <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( system->d_my_atoms, *(workspace->d_workspace), u,
#if !defined(OMPI_HAVE_MPI_EXT_ROCM) || !OMPI_HAVE_MPI_EXT_ROCM
          spad,
#else
          q,
#endif
          system->n );
    hipCheckError( );

#if !defined(OMPI_HAVE_MPI_EXT_ROCM) || !OMPI_HAVE_MPI_EXT_ROCM
    sHipMemcpyAsync( q, spad, sizeof(real) * system->n, 
            hipMemcpyDeviceToHost, s, __FILE__, __LINE__ );

    hipStreamSynchronize( s );
#endif
}


static void Update_Ghost_Atom_Charges( reax_system const * const system,
        control_params const * const control, storage * const workspace,
        real * const q, hipStream_t s )
{
    int blocks;
#if !defined(OMPI_HAVE_MPI_EXT_ROCM) || !OMPI_HAVE_MPI_EXT_ROCM
    real *spad;
#endif

    blocks = (system->N - system->n) / DEF_BLOCK_SIZE
        + (((system->N - system->n) % DEF_BLOCK_SIZE == 0) ? 0 : 1);

#if !defined(OMPI_HAVE_MPI_EXT_ROCM) || !OMPI_HAVE_MPI_EXT_ROCM
    sHipCheckMalloc( &workspace->scratch[5], &workspace->scratch_size[5],
            sizeof(real) * (system->N - system->n), __FILE__, __LINE__ );
    spad = (real *) workspace->scratch[5];

    sHipMemcpyAsync( spad, &q[system->n], sizeof(real) * (system->N - system->n),
            hipMemcpyHostToDevice, s, __FILE__, __LINE__ );

    hipStreamSynchronize( s );
#endif

    k_update_ghost_atom_charges <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( system->d_my_atoms,
#if !defined(OMPI_HAVE_MPI_EXT_ROCM) || !OMPI_HAVE_MPI_EXT_ROCM
          spad,
#else
          &q[system->n],
#endif
          system->n, system->N );
    hipCheckError( );
}


static void Calculate_Charges_QEq( reax_system const * const system,
        control_params const * const control, storage * const workspace,
        mpi_datatypes * const mpi_data, hipStream_t s )
{
    int ret;
    real u;
    rvec2 my_sum, all_sum;
#if defined(DUAL_SOLVER)
    rvec2 *spad;
#else
    real *spad;
#endif

#if defined(DUAL_SOLVER)
    sHipCheckMalloc( &workspace->scratch[5], &workspace->scratch_size[5],
            sizeof(rvec2), __FILE__, __LINE__ );
    spad = (rvec2 *) workspace->scratch[5];

    /* compute local sums of pseudo-charges in s and t on device */
    Hip_Reduction_Sum( workspace->d_workspace->x, spad, system->n, 5, s );

    sHipMemcpyAsync( &my_sum, spad, sizeof(rvec2), hipMemcpyDeviceToHost,
            s, __FILE__, __LINE__ );
#else
    sHipCheckMalloc( &workspace->scratch[5], &workspace->scratch_size[5],
            sizeof(real) * 2, __FILE__, __LINE__ );
    spad = (real *) workspace->scratch[5];

    /* local reductions (sums) on device */
    Hip_Reduction_Sum( workspace->d_workspace->s, &spad[0], system->n, 5, s );
    Hip_Reduction_Sum( workspace->d_workspace->t, &spad[1], system->n, 5, s );

    sHipMemcpyAsync( my_sum, spad, sizeof(real) * 2,
            hipMemcpyDeviceToHost, s, __FILE__, __LINE__ );
#endif

    hipStreamSynchronize( s );

    /* global reduction on pseudo-charges for s and t */
    ret = MPI_Allreduce( &my_sum, &all_sum, 2, MPI_DOUBLE,
            MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    u = all_sum[0] / all_sum[1];

#if !defined(OMPI_HAVE_MPI_EXT_ROCM) || !OMPI_HAVE_MPI_EXT_ROCM
    sHipHostAllocCheck( &workspace->host_scratch, &workspace->host_scratch_size,
            sizeof(real) * system->N, hipHostAllocPortable, TRUE, SAFE_ZONE,
            __FILE__, __LINE__ );
#else
    sHipCheckMalloc( &workspace->scratch[5], &workspace->scratch_size[5],
            sizeof(real) * system->N, __FILE__, __LINE__ );
#endif

    /* derive atomic charges from pseudo-charges
     * and set up extrapolation for next time step */
    Extrapolate_Charges_QEq_Part2( system, control, workspace,
#if !defined(OMPI_HAVE_MPI_EXT_ROCM) || !OMPI_HAVE_MPI_EXT_ROCM
            (real *) workspace->host_scratch,
#else
            (real *) workspace->scratch[5],
#endif
            u, s );

#if !defined(OMPI_HAVE_MPI_EXT_ROCM) || !OMPI_HAVE_MPI_EXT_ROCM
    Dist( system, mpi_data, workspace->host_scratch, REAL_PTR_TYPE,
            MPI_DOUBLE );
#else
    Hip_Dist( system, workspace, mpi_data, workspace->scratch[5],
            REAL_PTR_TYPE, MPI_DOUBLE, s );
#endif

    /* copy atomic charges to ghost atoms in case of ownership transfer */
    Update_Ghost_Atom_Charges( system, control, workspace,
#if !defined(OMPI_HAVE_MPI_EXT_ROCM) || !OMPI_HAVE_MPI_EXT_ROCM
            (real *) workspace->host_scratch,
#else
            (real *) workspace->scratch[5],
#endif
            s );
}


static void Calculate_Charges_EE( reax_system const * const system,
        control_params const * const control,
        storage const * const workspace,
        mpi_datatypes * const mpi_data, hipStream_t s )
{
}


static void Calculate_Charges_ACKS2( reax_system const * const system,
        control_params const * const control,
        storage const * const workspace,
        mpi_datatypes * const mpi_data, hipStream_t s )
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
        mpi_datatypes * const mpi_data, hipStream_t s )
{
    int iters, refactor;

    iters = 0;
    refactor = is_refactoring_step( control, data );

    if ( refactor == TRUE )
    {
        Setup_Preconditioner_QEq( system, control, data, workspace, mpi_data, s );

        Compute_Preconditioner_QEq( system, control, data, workspace, mpi_data, s );
    }

//    switch ( control->cm_init_guess_type )
//    {
//    case SPLINE:
        Spline_Extrapolate_Charges_QEq( system, control, data, workspace, mpi_data, s );
//        break;
//
//    case TF_FROZEN_MODEL_LSTM:
//#if defined(HAVE_TENSORFLOW)
//        if ( data->step < control->cm_init_guess_win_size )
//        {
//            Spline_Extrapolate_Charges_QEq( system, control, data, workspace, mpi_data, s );
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
        iters = Hip_dual_CG( system, control, data, workspace,
                &workspace->d_workspace->H, workspace->d_workspace->b,
                control->cm_solver_q_err, workspace->d_workspace->x, mpi_data,
                refactor, s );
#else
        iters = Hip_CG( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_s, control->cm_solver_q_err, workspace->d_workspace->s,
                mpi_data, refactor, s );
        iters += Hip_CG( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_t, control->cm_solver_q_err, workspace->d_workspace->t,
                mpi_data, FALSE, s );
#endif
        break;

    case SDM_S:
#if defined(DUAL_SOLVER)
        iters = Hip_dual_SDM( system, control, data, workspace,
                &workspace->d_workspace->H, workspace->d_workspace->b,
                control->cm_solver_q_err, workspace->d_workspace->x, mpi_data,
                refactor, s );
#else
        iters = Hip_SDM( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_s, control->cm_solver_q_err, workspace->d_workspace->s,
                mpi_data, refactor, s );
        iters += Hip_SDM( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_t, control->cm_solver_q_err, workspace->d_workspace->t,
                mpi_data, FALSE, s );
#endif
        break;

    case BiCGStab_S:
#if defined(DUAL_SOLVER)
        iters = Hip_dual_BiCGStab( system, control, data, workspace,
                &workspace->d_workspace->H, workspace->d_workspace->b,
                control->cm_solver_q_err, workspace->d_workspace->x, mpi_data,
                refactor, s );
#else
        iters = Hip_BiCGStab( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_s, control->cm_solver_q_err, workspace->d_workspace->s,
                mpi_data, refactor, s );
        iters += Hip_BiCGStab( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_t, control->cm_solver_q_err, workspace->d_workspace->t,
                mpi_data, FALSE, s );
#endif
        break;

    case PIPECG_S:
#if defined(DUAL_SOLVER)
        iters = Hip_dual_PIPECG( system, control, data, workspace,
                &workspace->d_workspace->H, workspace->d_workspace->b,
                control->cm_solver_q_err, workspace->d_workspace->x, mpi_data,
                refactor, s );
#else
        iters = Hip_PIPECG( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_s, control->cm_solver_q_err, workspace->d_workspace->s,
                mpi_data, refactor, s );
        iters += Hip_PIPECG( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_t, control->cm_solver_q_err, workspace->d_workspace->t,
                mpi_data, FALSE, s );
#endif
        break;

    case PIPECR_S:
#if defined(DUAL_SOLVER)
        iters = Hip_dual_PIPECR( system, control, data, workspace,
                &workspace->d_workspace->H, workspace->d_workspace->b,
                control->cm_solver_q_err, workspace->d_workspace->x, mpi_data,
                refactor, s );
#else
        iters = Hip_PIPECR( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_s, control->cm_solver_q_err, workspace->d_workspace->s,
                mpi_data, refactor, s );
        iters += Hip_PIPECR( system, control, data, workspace, &workspace->d_workspace->H,
                workspace->d_workspace->b_t, control->cm_solver_q_err, workspace->d_workspace->t,
                mpi_data, FALSE, s );
#endif
        break;

    default:
        fprintf( stderr, "[ERROR] Unrecognized solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    Calculate_Charges_QEq( system, control, workspace, mpi_data, s );

#if defined(LOG_PERFORMANCE)
    data->timing.cm_solver_iters += iters;
#endif
}


void EE( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        output_controls const * const out_control,
        mpi_datatypes * const mpi_data, hipStream_t s )
{
    fprintf( stderr, "[ERROR] Unsupported charge model (EE). Terminating...\n" );
    exit( INVALID_INPUT );
}


void ACKS2( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        output_controls const * const out_control,
        mpi_datatypes * const mpi_data, hipStream_t s )
{
    fprintf( stderr, "[ERROR] Unsupported charge model (ACKS2). Terminating...\n" );
    exit( INVALID_INPUT );
}


void Hip_Compute_Charges( reax_system const * const system,
        control_params const * const control,
        simulation_data * const data,
        storage * const workspace,
        output_controls const * const out_control,
        mpi_datatypes * const mpi_data, hipStream_t s )
{
    switch ( control->charge_method )
    {
    case QEQ_CM:
        QEq( system, control, data, workspace, out_control, mpi_data, s );
        break;

    case EE_CM:
        EE( system, control, data, workspace, out_control, mpi_data, s );
        break;

    case ACKS2_CM:
        ACKS2( system, control, data, workspace, out_control, mpi_data, s );
        break;

    default:
        fprintf( stderr, "[ERROR] Invalid charge method. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }
}
