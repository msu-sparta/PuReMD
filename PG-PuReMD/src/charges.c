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

#include "charges.h"

#include "allocate.h"
#include "basic_comm.h"
#include "io_tools.h"
#include "lin_alg.h"
#include "tool_box.h"


int is_refactoring_step( control_params const * const control,
        simulation_data * const data )
{
    if ( control->cm_solver_pre_comp_refactor != -1 )
    {
        if ( control->cm_solver_pre_comp_refactor > 0
                && ((data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) )
        {
            return TRUE;
        }
        else
        {
            return FALSE;
        }
    }
    else
    {
        return data->refactor;
    }
}


static void Spline_Extrapolate_Charges_QEq( reax_system const * const system,
        control_params const * const control,
        simulation_data const * const data,
        storage const * const workspace,
        mpi_datatypes const * const mpi_data )
{
    int i;
    real s_tmp, t_tmp;

    /* RHS vectors for linear system */
    for ( i = 0; i < system->n; ++i )
    {
        workspace->b_s[i] = -system->reax_param.sbp[ system->my_atoms[i].type ].chi;
    }
    for ( i = 0; i < system->n; ++i )
    {
        workspace->b_t[i] = -1.0;
    }
    for ( i = 0; i < system->n; ++i )
    {
        workspace->b[i][0] = -system->reax_param.sbp[ system->my_atoms[i].type ].chi;
        workspace->b[i][1] = -1.0;
    }

    /* spline extrapolation for s & t */
    for ( i = 0; i < system->n; ++i )
    {
        /* no extrapolation, previous solution as initial guess */
        if ( control->cm_init_guess_extrap1 == 0 )
        {
            s_tmp = system->my_atoms[i].s[0];
        }
        /* linear */
        else if ( control->cm_init_guess_extrap1 == 1 )
        {
            s_tmp = 2.0 * system->my_atoms[i].s[0] - system->my_atoms[i].s[1];
        }
        /* quadratic */
        else if ( control->cm_init_guess_extrap1 == 2 )
        {
            s_tmp = system->my_atoms[i].s[2] + 3.0 * (system->my_atoms[i].s[0] - system->my_atoms[i].s[1]);
        }
        /* cubic */
        else if ( control->cm_init_guess_extrap1 == 3 )
        {
            s_tmp = 4.0 * (system->my_atoms[i].s[0] + system->my_atoms[i].s[2])
                - (6.0 * system->my_atoms[i].s[1] + system->my_atoms[i].s[3]);
        }
        else
        {
            s_tmp = 0.0;
        }

        /* no extrapolation, previous solution as initial guess */
        if ( control->cm_init_guess_extrap1 == 0 )
        {
            t_tmp = system->my_atoms[i].t[0];
        }
        /* linear */
        else if ( control->cm_init_guess_extrap1 == 1 )
        {
            t_tmp = 2.0 * system->my_atoms[i].t[0] - system->my_atoms[i].t[1];
        }
        /* quadratic */
        else if ( control->cm_init_guess_extrap1 == 2 )
        {
            t_tmp = system->my_atoms[i].t[2] + 3.0 * (system->my_atoms[i].t[0] - system->my_atoms[i].t[1]);
        }
        /* cubic */
        else if ( control->cm_init_guess_extrap1 == 3 )
        {
            t_tmp = 4.0 * (system->my_atoms[i].t[0] + system->my_atoms[i].t[2])
                - (6.0 * system->my_atoms[i].t[1] + system->my_atoms[i].t[3]);
        }
        else
        {
            t_tmp = 0.0;
        }
        
        workspace->x[i][0] = s_tmp;
        workspace->x[i][1] = t_tmp;
    }
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
    real time, t_sort, t_pc, total_sort, total_pc;
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = &workspace->H_sp;
    }
    else
    {
        Hptr = &workspace->H;
    }

    /* sort H needed for SpMV's in linear solver, H or H_sp needed for preconditioning */
    time = MPI_Wtime( );
    Sort_Matrix_Rows( &workspace->H );
    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Sort_Matrix_Rows( &workspace->H_sp );
    }
    t_sort = MPI_Wtime( ) - time;

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
            t_pc = setup_sparse_approx_inverse( system, data, workspace, mpi_data,
                    &workspace->H, &workspace->H_spar_patt, 
                    control->nprocs, control->cm_solver_pre_comp_sai_thres );

            MPI_Reduce( &t_sort, &total_sort, 1, MPI_DOUBLE, MPI_SUM,
                    MASTER_NODE, mpi_data->world );
            MPI_Reduce( &t_pc, &total_pc, 1, MPI_DOUBLE, MPI_SUM,
                    MASTER_NODE, mpi_data->world );

            if ( system->my_rank == MASTER_NODE )
            {
                data->timing.cm_sort += total_sort / control->nprocs;
                data->timing.cm_solver_pre_comp += total_pc / control->nprocs;
            }
            break;

        default:
            fprintf( stderr, "[ERROR] Unrecognized preconditioner computation method (%d). Terminating...\n",
                   control->cm_solver_pre_comp_type );
            exit( INVALID_INPUT );
            break;
    }
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
    int i;
#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
    real t_pc, total_pc;
#endif

    if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        for ( i = 0; i < system->n; ++i )
        {
            workspace->Hdia_inv[i] = 1.0 / system->reax_param.sbp[ system->my_atoms[i].type ].eta;
        }
    }
    else if ( control->cm_solver_pre_comp_type == SAI_PC )
    {
#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
        t_pc = sparse_approx_inverse( system, data, workspace, mpi_data,
                &workspace->H, workspace->H_spar_patt, &workspace->H_app_inv, control->nprocs );

        MPI_Reduce( &t_pc, &total_pc, 1, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );

        if( system->my_rank == MASTER_NODE )
        {
            data->timing.cm_solver_pre_comp += total_pc / control->nprocs;
        }
#else
        fprintf( stderr, "[ERROR] LAPACKE support disabled. Re-compile before enabling. Terminating...\n" );
        exit( INVALID_INPUT );
#endif
    }
}


static void Compute_Preconditioner_EE( reax_system const * const system,
        control_params const * const control,
        simulation_data * const data, storage * const workspace,
        mpi_datatypes const * const mpi_data )
{
}


static void Compute_Preconditioner_ACKS2( reax_system const * const system,
        control_params const * const control,
        simulation_data * const data, storage * const workspace,
        mpi_datatypes const * const mpi_data )
{
}


static void Calculate_Charges_QEq( reax_system const * const system,
        storage const * const workspace,
        mpi_datatypes * const mpi_data )
{
    int i;
    real u;
    rvec2 my_sum, all_sum;
    reax_atom *atom;
    real *q;

    q = smalloc( sizeof(real) * system->N, "Calculate_Charges_QEq::q" );

    my_sum[0] = 0.0;
    my_sum[1] = 0.0;
    for ( i = 0; i < system->n; ++i )
    {
        my_sum[0] += workspace->x[i][0];
        my_sum[1] += workspace->x[i][1];
    }
    MPI_Allreduce( &my_sum, &all_sum, 2, MPI_DOUBLE, MPI_SUM, mpi_data->world );

    u = all_sum[0] / all_sum[1];
    for ( i = 0; i < system->n; ++i )
    {
        atom = &system->my_atoms[i];

        /* compute charge based on s & t */
        q[i] = atom->q = workspace->x[i][0] - u * workspace->x[i][1];

        /* update previous solutions in s & t */
        atom->s[3] = atom->s[2];
        atom->s[2] = atom->s[1];
        atom->s[1] = atom->s[0];
        atom->s[0] = workspace->x[i][0];

        atom->t[3] = atom->t[2];
        atom->t[2] = atom->t[1];
        atom->t[1] = atom->t[0];
        atom->t[0] = workspace->x[i][1];
    }

    Dist_FS( system, mpi_data, q, REAL_PTR_TYPE, MPI_DOUBLE );

    for ( i = system->n; i < system->N; ++i )
    {
        system->my_atoms[i].q = q[i];
    }

    sfree( q, "Calculate_Charges_QEq::q" );
}


static void Calculate_Charges_EE( reax_system const * const system,
        storage const * const workspace,
        mpi_datatypes * const mpi_data )
{
}


static void Calculate_Charges_ACKS2( reax_system const * const system,
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
static void QEq( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        output_controls const * const out_control,
        mpi_datatypes * const mpi_data )
{
    int iters;
#if !defined(DUAL_SOLVER)
    int j;
#endif

    iters = 0;

    if ( is_refactoring_step( control, data ) == TRUE )
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
        iters = dual_CG( system, control, data, workspace, &workspace->H, workspace->b,
                control->cm_solver_q_err, workspace->x, mpi_data );
#else
        for ( j = 0; j < system->n; ++j )
        {
            workspace->s[j] = workspace->x[j][0];
        }

        iters = CG( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data );

        for ( j = 0; j < system->n; ++j )
        {
            workspace->x[j][0] = workspace->s[j];
        }

        for ( j = 0; j < system->n; ++j )
        {
            workspace->t[j] = workspace->x[j][1];
        }

        iters += CG( system, control, data, workspace, &workspace->H, workspace->b_t,
                control->cm_solver_q_err, workspace->t, mpi_data );

        for ( j = 0; j < system->n; ++j )
        {
            workspace->x[j][1] = workspace->t[j];
        }
#endif
        break;

    case SDM_S:
#if defined(DUAL_SOLVER)
        fprintf( stderr, "[ERROR] Dual SDM solver for QEq not yet implemented. Terminating...\n" );
        exit( INVALID_INPUT );
//        iters = dual_SDM( system, control, data, workspace, &workspace->H, workspace->b,
//                control->cm_solver_q_err, workspace->x, mpi_data );
#else
        for ( j = 0; j < system->n; ++j )
        {
            workspace->s[j] = workspace->x[j][0];
        }

        iters = SDM( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data );

        for ( j = 0; j < system->n; ++j )
        {
            workspace->x[j][0] = workspace->s[j];
        }

        for ( j = 0; j < system->n; ++j )
        {
            workspace->t[j] = workspace->x[j][1];
        }

        iters += SDM( system, control, data, workspace, &workspace->H, workspace->b_t,
                control->cm_solver_q_err, workspace->t, mpi_data );

        for ( j = 0; j < system->n; ++j )
        {
            workspace->x[j][1] = workspace->t[j];
        }
#endif
        break;

    case BiCGStab_S:
#if defined(DUAL_SOLVER)
        fprintf( stderr, "[ERROR] Dual BiCGStab solver for QEq not yet implemented. Terminating...\n" );
        exit( INVALID_INPUT );
//        iters = dual_BiCGStab( system, control, data, workspace, &workspace->H, workspace->b,
//                control->cm_solver_q_err, workspace->x, mpi_data );
#else
        for ( j = 0; j < system->n; ++j )
        {
            workspace->s[j] = workspace->x[j][0];
        }

        iters = BiCGStab( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data );

        for ( j = 0; j < system->n; ++j )
        {
            workspace->x[j][0] = workspace->s[j];
        }

        for ( j = 0; j < system->n; ++j )
        {
            workspace->t[j] = workspace->x[j][1];
        }

        iters += BiCGStab( system, control, data, workspace, &workspace->H, workspace->b_t,
                control->cm_solver_q_err, workspace->t, mpi_data );

        for ( j = 0; j < system->n; ++j )
        {
            workspace->x[j][1] = workspace->t[j];
        }
#endif
        break;

    case PIPECG_S:
#if defined(DUAL_SOLVER)
        iters = dual_PIPECG( system, control, data, workspace, &workspace->H, workspace->b,
                control->cm_solver_q_err, workspace->x, mpi_data );
#else
        for ( j = 0; j < system->n; ++j )
        {
            workspace->s[j] = workspace->x[j][0];
        }

        iters = PIPECG( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data );

        for ( j = 0; j < system->n; ++j )
        {
            workspace->x[j][0] = workspace->s[j];
        }

        for ( j = 0; j < system->n; ++j )
        {
            workspace->t[j] = workspace->x[j][1];
        }

        iters += PIPECG( system, control, data, workspace, &workspace->H, workspace->b_t,
                control->cm_solver_q_err, workspace->t, mpi_data );

        for ( j = 0; j < system->n; ++j )
        {
            workspace->x[j][1] = workspace->t[j];
        }
#endif
        break;

    case PIPECR_S:
#if defined(DUAL_SOLVER)
        fprintf( stderr, "[ERROR] Dual PIPECR solver for QEq not yet implemented. Terminating...\n" );
        exit( INVALID_INPUT );
//        iters = dual_PIPECR( system, control, data, workspace, &workspace->H, workspace->b,
//                control->cm_solver_q_err, workspace->x, mpi_data );
#else
        for ( j = 0; j < system->n; ++j )
        {
            workspace->s[j] = workspace->x[j][0];
        }

        iters = PIPECR( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data );

        for ( j = 0; j < system->n; ++j )
        {
            workspace->x[j][0] = workspace->s[j];
        }

        for ( j = 0; j < system->n; ++j )
        {
            workspace->t[j] = workspace->x[j][1];
        }

        iters += PIPECR( system, control, data, workspace, &workspace->H, workspace->b_t,
                control->cm_solver_q_err, workspace->t, mpi_data );

        for ( j = 0; j < system->n; ++j )
        {
            workspace->x[j][1] = workspace->t[j];
        }
#endif
        break;

    default:
        fprintf( stderr, "[ERROR] Unrecognized solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    Calculate_Charges_QEq( system, workspace, mpi_data );

#if defined(LOG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        data->timing.cm_solver_iters += iters;
    }
#endif
}


/* Main driver method for EE kernel
 *  1) init / setup routines for preconditioning of linear solver
 *  2) compute preconditioner
 *  3) extrapolate charges
 *  4) perform 1 linear solve
 *  5) compute atomic charges based on output of (4)
 */
static void EE( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        output_controls const * const out_control,
        mpi_datatypes * const mpi_data )
{
    int iters;

    iters = 0;

    if ( is_refactoring_step( control, data ) == TRUE )
    {
        Setup_Preconditioner_EE( system, control, data, workspace, mpi_data );

        Compute_Preconditioner_EE( system, control, data, workspace, mpi_data );
    }

//    switch ( control->cm_init_guess_type )
//    {
//    case SPLINE:
        Spline_Extrapolate_Charges_EE( system, control, data, workspace, mpi_data );
//        break;
//
//    case TF_FROZEN_MODEL_LSTM:
//#if defined(HAVE_TENSORFLOW)
//        if ( data->step < control->cm_init_guess_win_size )
//        {
//            Spline_Extrapolate_Charges_EE( system, control, data, workspace, mpi_data );
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
        iters = CG( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data );
        break;

    case SDM_S:
        iters = SDM( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data );
        break;

    case BiCGStab_S:
        iters = BiCGStab( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data );
        break;

    case PIPECG_S:
        iters = PIPECG( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data );
        break;

    case PIPECR_S:
        iters = PIPECR( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data );
        break;

    default:
        fprintf( stderr, "[ERROR] Unrecognized solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    Calculate_Charges_EE( system, workspace, mpi_data );

#if defined(LOG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        data->timing.cm_solver_iters += iters;
    }
#endif
}


/* Main driver method for ACKS2 kernel
 *  1) init / setup routines for preconditioning of linear solver
 *  2) compute preconditioner
 *  3) extrapolate charges
 *  4) perform 1 linear solve
 *  5) compute atomic charges based on output of (4)
 */
static void ACKS2( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        output_controls const * const out_control,
        mpi_datatypes * const mpi_data )
{
    int iters;

    iters = 0;

    if ( is_refactoring_step( control, data ) == TRUE )
    {
        Setup_Preconditioner_ACKS2( system, control, data, workspace, mpi_data );

        Compute_Preconditioner_ACKS2( system, control, data, workspace, mpi_data );
    }

//    switch ( control->cm_init_guess_type )
//    {
//    case SPLINE:
        Spline_Extrapolate_Charges_EE( system, control, data, workspace, mpi_data );
//        break;
//
//    case TF_FROZEN_MODEL_LSTM:
//#if defined(HAVE_TENSORFLOW)
//        if ( data->step < control->cm_init_guess_win_size )
//        {
//            Spline_Extrapolate_Charges_EE( system, control, data, workspace, mpi_data );
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
        iters = CG( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data );
        break;

    case SDM_S:
        iters = SDM( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data );
        break;

    case BiCGStab_S:
        iters = BiCGStab( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data );
        break;

    case PIPECG_S:
        iters = PIPECG( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data );
        break;

    case PIPECR_S:
        iters = PIPECR( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data );
        break;

    default:
        fprintf( stderr, "[ERROR] Unrecognized solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    Calculate_Charges_ACKS2( system, workspace, mpi_data );

#if defined(LOG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        data->timing.cm_solver_iters += iters;
    }
#endif
}


void Compute_Charges( reax_system const * const system,
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
        exit( UNKNOWN_OPTION );
        break;
    }
}
