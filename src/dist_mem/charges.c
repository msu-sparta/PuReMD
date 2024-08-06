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
#include "comm_tools.h"
#include "io_tools.h"
#include "lin_alg.h"
#include "tool_box.h"


int is_refactoring_step( control_params const * const control,
        simulation_data * const data )
{
    int ret;

    if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        ret = TRUE;
    }
    else if ( control->cm_solver_pre_comp_refactor != -1 )
    {
        if ( control->cm_solver_pre_comp_refactor > 0
                && ((data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) )
        {
            ret = TRUE;
        }
        else
        {
            ret = FALSE;
        }
    }
    else
    {
        ret = data->refactor;
    }

    return ret;
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
        workspace->b_s[i] = -1.0 * system->reax_param.sbp[ system->my_atoms[i].type ].chi;
    }
    for ( i = 0; i < system->n; ++i )
    {
        workspace->b_t[i] = -1.0;
    }
#if defined(DUAL_SOLVER)
    for ( i = 0; i < system->n; ++i )
    {
        workspace->b[i][0] = -1.0 * system->reax_param.sbp[ system->my_atoms[i].type ].chi;
        workspace->b[i][1] = -1.0;
    }
#endif

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
        
#if defined(DUAL_SOLVER)
        workspace->x[i][0] = s_tmp;
        workspace->x[i][1] = t_tmp;
#else
        workspace->s[i] = s_tmp;
        workspace->t[i] = t_tmp;
#endif
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
    real time;
    sparse_matrix *Hptr;

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = &workspace->H_sp;
    }
    else
    {
        Hptr = &workspace->H;
    }

    /* sort H needed for SpMV's in linear solver, H or H_sp needed for preconditioning */
    Sort_Matrix_Rows( &workspace->H );
    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Sort_Matrix_Rows( &workspace->H_sp );
    }

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
#if defined(LOG_PERFORMANCE)
    real time;
#endif
#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
    int ret;
#endif

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif


    if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        jacobi( &workspace->H, workspace->Hdia_inv );
    }
    else if ( control->cm_solver_pre_comp_type == SAI_PC )
    {
#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
        sparse_approx_inverse( system, data, mpi_data,
                &workspace->H, &workspace->H_spar_patt, &workspace->H_app_inv,
                control->nprocs );
#else
        fprintf( stderr, "[ERROR] LAPACKE support disabled. Re-compile before enabling. Terminating...\n" );
        exit( INVALID_INPUT );
#endif
    }

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_pre_comp );
#endif
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
    int i, ret;
    real u, *q;
    rvec2 my_sum, all_sum;
    reax_atom *atom;

    q = smalloc( sizeof(real) * system->N, __FILE__, __LINE__ );

    my_sum[0] = 0.0;
    my_sum[1] = 0.0;
#if defined(DUAL_SOLVER)
    for ( i = 0; i < system->n; ++i )
    {
        my_sum[0] += workspace->x[i][0];
        my_sum[1] += workspace->x[i][1];
    }
#else
    for ( i = 0; i < system->n; ++i )
    {
        my_sum[0] += workspace->s[i];
    }
    for ( i = 0; i < system->n; ++i )
    {
        my_sum[1] += workspace->t[i];
    }
#endif

    ret = MPI_Allreduce( &my_sum, &all_sum, 2, MPI_DOUBLE,
            MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    u = all_sum[0] / all_sum[1];
    for ( i = 0; i < system->n; ++i )
    {
        atom = &system->my_atoms[i];

        /* compute charge based on s & t */
#if defined(DUAL_SOLVER)
        atom->q = workspace->x[i][0] - u * workspace->x[i][1];
#else
        atom->q = workspace->s[i] - u * workspace->t[i];
#endif
        q[i] = atom->q;

        /* update previous solutions in s & t */
        atom->s[3] = atom->s[2];
        atom->s[2] = atom->s[1];
        atom->s[1] = atom->s[0];
#if defined(DUAL_SOLVER)
        atom->s[0] = workspace->x[i][0];
#else
        atom->s[0] = workspace->s[i];
#endif

        atom->t[3] = atom->t[2];
        atom->t[2] = atom->t[1];
        atom->t[1] = atom->t[0];
#if defined(DUAL_SOLVER)
        atom->t[0] = workspace->x[i][1];
#else
        atom->t[0] = workspace->t[i];
#endif
    }

    Dist( system, mpi_data, q, REAL_PTR_TYPE, MPI_DOUBLE );

    /* copy charges of received ghost atoms */
    for ( i = system->n; i < system->N; ++i )
    {
        system->my_atoms[i].q = q[i];
    }

    sfree( q, __FILE__, __LINE__ );
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
    int iters, refactor;

    iters = 0;
    refactor = is_refactoring_step( control, data );

    if ( refactor == TRUE )
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
                control->cm_solver_q_err, workspace->x, mpi_data, refactor );
#else
        iters = CG( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data, refactor );

        iters += CG( system, control, data, workspace, &workspace->H, workspace->b_t,
                control->cm_solver_q_err, workspace->t, mpi_data, FALSE );
#endif
        break;

    case SDM_S:
#if defined(DUAL_SOLVER)
        iters = dual_SDM( system, control, data, workspace, &workspace->H, workspace->b,
                control->cm_solver_q_err, workspace->x, mpi_data, refactor );
#else
        iters = SDM( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data, refactor );

        iters += SDM( system, control, data, workspace, &workspace->H, workspace->b_t,
                control->cm_solver_q_err, workspace->t, mpi_data, FALSE );
#endif
        break;

    case BiCGStab_S:
#if defined(DUAL_SOLVER)
        iters = dual_BiCGStab( system, control, data, workspace, &workspace->H, workspace->b,
                control->cm_solver_q_err, workspace->x, mpi_data, refactor );
#else
        iters = BiCGStab( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data, refactor );

        iters += BiCGStab( system, control, data, workspace, &workspace->H, workspace->b_t,
                control->cm_solver_q_err, workspace->t, mpi_data, FALSE );
#endif
        break;

    case PIPECG_S:
#if defined(DUAL_SOLVER)
        iters = dual_PIPECG( system, control, data, workspace, &workspace->H, workspace->b,
                control->cm_solver_q_err, workspace->x, mpi_data, refactor );
#else
        iters = PIPECG( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data, refactor );

        iters += PIPECG( system, control, data, workspace, &workspace->H, workspace->b_t,
                control->cm_solver_q_err, workspace->t, mpi_data, FALSE );
#endif
        break;

    case PIPECR_S:
#if defined(DUAL_SOLVER)
        iters = dual_PIPECR( system, control, data, workspace, &workspace->H, workspace->b,
                control->cm_solver_q_err, workspace->x, mpi_data, refactor );
#else
        iters = PIPECR( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data, refactor );

        iters += PIPECR( system, control, data, workspace, &workspace->H, workspace->b_t,
                control->cm_solver_q_err, workspace->t, mpi_data, FALSE );
#endif
        break;

    default:
        fprintf( stderr, "[ERROR] Unrecognized solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    Calculate_Charges_QEq( system, workspace, mpi_data );

#if defined(LOG_PERFORMANCE)
    data->timing.cm_solver_iters += iters;
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
    int iters, refactor;

    iters = 0;
    refactor = is_refactoring_step( control, data );

    if ( refactor == TRUE )
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
                control->cm_solver_q_err, workspace->s, mpi_data, refactor );
        break;

    case SDM_S:
        iters = SDM( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data, refactor );
        break;

    case BiCGStab_S:
        iters = BiCGStab( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data, refactor );
        break;

    case PIPECG_S:
        iters = PIPECG( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data, refactor );
        break;

    case PIPECR_S:
        iters = PIPECR( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data, refactor );
        break;

    default:
        fprintf( stderr, "[ERROR] Unrecognized solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    Calculate_Charges_EE( system, workspace, mpi_data );

#if defined(LOG_PERFORMANCE)
    data->timing.cm_solver_iters += iters;
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
    int iters, refactor;

    iters = 0;
    refactor = is_refactoring_step( control, data );

    if ( refactor == TRUE )
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
                control->cm_solver_q_err, workspace->s, mpi_data, refactor );
        break;

    case SDM_S:
        iters = SDM( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data, refactor );
        break;

    case BiCGStab_S:
        iters = BiCGStab( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data, refactor );
        break;

    case PIPECG_S:
        iters = PIPECG( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data, refactor );
        break;

    case PIPECR_S:
        iters = PIPECR( system, control, data, workspace, &workspace->H, workspace->b_s,
                control->cm_solver_q_err, workspace->s, mpi_data, refactor );
        break;

    default:
        fprintf( stderr, "[ERROR] Unrecognized solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    Calculate_Charges_ACKS2( system, workspace, mpi_data );

#if defined(LOG_PERFORMANCE)
    data->timing.cm_solver_iters += iters;
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
