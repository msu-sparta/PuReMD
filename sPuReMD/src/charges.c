/*----------------------------------------------------------------------
  SerialReax - Reax Force Field Simulator

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
#include "list.h"
#include "lin_alg.h"
#include "print_utils.h"
#include "tool_box.h"
#include "vector.h"
#if defined(HAVE_SUPERLU_MT)
  #include "slu_mt_ddefs.h"
#endif


static void Extrapolate_Charges_QEq( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace )
{
    int i;
    real s_tmp, t_tmp;

    /* extrapolation for s & t */
    //TODO: good candidate for vectorization, avoid moving data with head pointer and circular buffer
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) \
        default(none) private(i, s_tmp, t_tmp)
#endif
    for ( i = 0; i < system->N_cm; ++i )
    {
        // no extrapolation
        //s_tmp = workspace->s[0][i];
        //t_tmp = workspace->t[0][i];

        // linear
        //s_tmp = 2 * workspace->s[0][i] - workspace->s[1][i];
        //t_tmp = 2 * workspace->t[0][i] - workspace->t[1][i];

        // quadratic
        //s_tmp = workspace->s[2][i] + 3 * (workspace->s[0][i]-workspace->s[1][i]);
        t_tmp = workspace->t[2][i] + 3 * (workspace->t[0][i] - workspace->t[1][i]);

        // cubic
        s_tmp = 4 * (workspace->s[0][i] + workspace->s[2][i]) -
                (6 * workspace->s[1][i] + workspace->s[3][i] );
        //t_tmp = 4 * (workspace->t[0][i] + workspace->t[2][i]) -
        //  (6 * workspace->t[1][i] + workspace->t[3][i] );

        // 4th order
        //s_tmp = 5 * (workspace->s[0][i] - workspace->s[3][i]) +
        //  10 * (-workspace->s[1][i] + workspace->s[2][i] ) + workspace->s[4][i];
        //t_tmp = 5 * (workspace->t[0][i] - workspace->t[3][i]) +
        //  10 * (-workspace->t[1][i] + workspace->t[2][i] ) + workspace->t[4][i];

        workspace->s[4][i] = workspace->s[3][i];
        workspace->s[3][i] = workspace->s[2][i];
        workspace->s[2][i] = workspace->s[1][i];
        workspace->s[1][i] = workspace->s[0][i];
        workspace->s[0][i] = s_tmp;

        workspace->t[4][i] = workspace->t[3][i];
        workspace->t[3][i] = workspace->t[2][i];
        workspace->t[2][i] = workspace->t[1][i];
        workspace->t[1][i] = workspace->t[0][i];
        workspace->t[0][i] = t_tmp;
    }
}


static void Extrapolate_Charges_EE( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace )
{
    int i;
    real s_tmp;

    /* extrapolation for s */
    //TODO: good candidate for vectorization, avoid moving data with head pointer and circular buffer
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) \
        default(none) private(i, s_tmp)
#endif
    for ( i = 0; i < system->N_cm; ++i )
    {
        // no extrapolation
        //s_tmp = workspace->s[0][i];

        // linear
        //s_tmp = 2 * workspace->s[0][i] - workspace->s[1][i];

        // quadratic
        //s_tmp = workspace->s[2][i] + 3 * (workspace->s[0][i]-workspace->s[1][i]);

        // cubic
        s_tmp = 4 * (workspace->s[0][i] + workspace->s[2][i]) -
                (6 * workspace->s[1][i] + workspace->s[3][i] );

        // 4th order
        //s_tmp = 5 * (workspace->s[0][i] - workspace->s[3][i]) +
        //  10 * (-workspace->s[1][i] + workspace->s[2][i] ) + workspace->s[4][i];

        workspace->s[4][i] = workspace->s[3][i];
        workspace->s[3][i] = workspace->s[2][i];
        workspace->s[2][i] = workspace->s[1][i];
        workspace->s[1][i] = workspace->s[0][i];
        workspace->s[0][i] = s_tmp;
    }
}


/* Compute preconditioner for QEq
 */
static void Compute_Preconditioner_QEq( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs )
{
    real time;
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = workspace->H_sp;
    }
    else
    {
        Hptr = workspace->H;
    }

#if defined(TEST_MAT)
    Hptr = create_test_mat( );
#endif

    time = Get_Time( );
    if ( control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA )
    {
        setup_graph_coloring( control, workspace, Hptr, &workspace->H_full, &workspace->H_p );
        Sort_Matrix_Rows( workspace->H_p );
        Hptr = workspace->H_p;
    }
    data->timing.cm_sort_mat_rows += Get_Timing_Info( time );

    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case DIAG_PC:
            data->timing.cm_solver_pre_comp +=
                diag_pre_comp( Hptr, workspace->Hdia_inv );
            break;

        case ICHOLT_PC:
            data->timing.cm_solver_pre_comp +=
                ICHOLT( Hptr, workspace->droptol, workspace->L, workspace->U );
            break;

        case ILU_PAR_PC:
            data->timing.cm_solver_pre_comp +=
                ILU_PAR( Hptr, control->cm_solver_pre_comp_sweeps, workspace->L, workspace->U );
            break;

        case ILUT_PAR_PC:
            data->timing.cm_solver_pre_comp +=
                ILUT_PAR( Hptr, workspace->droptol, control->cm_solver_pre_comp_sweeps,
                        workspace->L, workspace->U );
            break;

        case ILU_SUPERLU_MT_PC:
#if defined(HAVE_SUPERLU_MT)
            data->timing.cm_solver_pre_comp +=
                SuperLU_Factorize( Hptr, workspace->L, workspace->U );
#else
            fprintf( stderr, "SuperLU MT support disabled. Re-compile before enabling. Terminating...\n" );
            exit( INVALID_INPUT );
#endif
            break;

        case SAI_PC:
#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
            data->timing.cm_solver_pre_comp +=
                sparse_approx_inverse( workspace->H_full, workspace->H_spar_patt_full,
                        &workspace->H_app_inv );
#else
            fprintf( stderr, "LAPACKE support disabled. Re-compile before enabling. Terminating...\n" );
            exit( INVALID_INPUT );
#endif
            break;

        default:
            fprintf( stderr, "Unrecognized preconditioner computation method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }

#if defined(DEBUG)
#define SIZE (1000)
    char fname[SIZE];

    if ( control->cm_solver_pre_comp_type != NONE_PC && 
            control->cm_solver_pre_comp_type != DIAG_PC )
    {
        fprintf( stderr, "condest = %f\n", condest(workspace->L, workspace->U) );

#if defined(DEBUG_FOCUS)
        snprintf( fname, SIZE + 10, "%s.L%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->L, fname, NULL );
        snprintf( fname, SIZE + 10, "%s.U%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->U, fname, NULL );
#endif
    }
#undef SIZE
#endif
}


/* Compute preconditioner for EE
 */
//static void Compute_Preconditioner_EE( const reax_system * const system,
//        const control_params * const control,
//        simulation_data * const data, static_storage * const workspace,
//        const reax_list * const far_nbrs )
//{
//    int i, top;
//    static real * ones = NULL, * x = NULL, * y = NULL;
//    sparse_matrix *Hptr;
//
//    Hptr = workspace->H_EE;
//
//#if defined(TEST_MAT)
//    Hptr = create_test_mat( );
//#endif
//
//    if ( ones == NULL )
//    {
//        ones = (real*) smalloc( system->N * sizeof(real), "Compute_Preconditioner_EE::ones" );
//        x = (real*) smalloc( system->N * sizeof(real), "Compute_Preconditioner_EE::x" );
//        y = (real*) smalloc( system->N * sizeof(real), "Compute_Preconditioner_EE::y" );
//
//        for ( i = 0; i < system->N; ++i )
//        {
//            ones[i] = 1.0;
//        }
//    }
//
//    switch ( control->cm_solver_pre_comp_type )
//    {
//    case DIAG_PC:
//        data->timing.cm_solver_pre_comp +=
//            diag_pre_comp( Hptr, workspace->Hdia_inv );
//        break;
//
//    case ICHOLT_PC:
//        data->timing.cm_solver_pre_comp +=
//            ICHOLT( Hptr, workspace->droptol, workspace->L_EE, workspace->U_EE );
//        break;
//
//    case ILU_PAR_PC:
//        data->timing.cm_solver_pre_comp +=
//            ILU_PAR( Hptr, control->cm_solver_pre_comp_sweeps, workspace->L_EE, workspace->U_EE );
//        break;
//
//    case ILUT_PAR_PC:
//        data->timing.cm_solver_pre_comp +=
//            ILUT_PAR( Hptr, workspace->droptol, control->cm_solver_pre_comp_sweeps,
//                    workspace->L_EE, workspace->U_EE );
//        break;
//
//    case ILU_SUPERLU_MT_PC:
//#if defined(HAVE_SUPERLU_MT)
//        data->timing.cm_solver_pre_comp +=
//            SuperLU_Factorize( Hptr, workspace->L_EE, workspace->U_EE );
//#else
//        fprintf( stderr, "SuperLU MT support disabled. Re-compile before enabling. Terminating...\n" );
//        exit( INVALID_INPUT );
//#endif
//        break;
//
//    default:
//        fprintf( stderr, "Unrecognized preconditioner computation method. Terminating...\n" );
//        exit( INVALID_INPUT );
//        break;
//    }
//
//    if ( control->cm_solver_pre_comp_type != DIAG_PC )
//    {
//        switch ( control->cm_solver_pre_app_type )
//        {
//            case TRI_SOLVE_PA:
//                tri_solve( workspace->L_EE, ones, x, workspace->L_EE->n, LOWER );
//                Transpose_I( workspace->U_EE );
//                tri_solve( workspace->U_EE, ones, y, workspace->U_EE->n, LOWER );
//                Transpose_I( workspace->U_EE );
//
//                memcpy( workspace->L->start, workspace->L_EE->start, sizeof(unsigned int) * (system->N + 1) );
//                memcpy( workspace->L->j, workspace->L_EE->j, sizeof(unsigned int) * workspace->L_EE->start[workspace->L_EE->n] );
//                memcpy( workspace->L->val, workspace->L_EE->val, sizeof(real) * workspace->L_EE->start[workspace->L_EE->n] );
//
//                top = workspace->L->start[system->N];
//                for ( i = 0; i < system->N; ++i )
//                {
//                    workspace->L->j[top] = i;
//                    workspace->L->val[top] = x[i];
//                    ++top;
//                }
//
//                workspace->L->j[top] = system->N_cm - 1;
//                workspace->L->val[top] = 1.0;
//                ++top;
//
//                workspace->L->start[system->N_cm] = top;
//
//                top = 0;
//                for ( i = 0; i < system->N; ++i )
//                {
//                    workspace->U->start[i] = top;
//                    memcpy( workspace->U->j + top, workspace->U_EE->j + workspace->U_EE->start[i],
//                            sizeof(unsigned int) * (workspace->U_EE->start[i + 1] - workspace->U_EE->start[i]) );
//                    memcpy( workspace->U->val + top, workspace->U_EE->val + workspace->U_EE->start[i],
//                            sizeof(real) * (workspace->U_EE->start[i + 1] - workspace->U_EE->start[i]) );
//                    top += (workspace->U_EE->start[i + 1] - workspace->U_EE->start[i]);
//
//                    workspace->U->j[top] = system->N_cm - 1;
//                    workspace->U->val[top] = y[i];
//                    ++top;
//                }
//
//                workspace->U->start[system->N_cm - 1] = top;
//
//                workspace->U->j[top] = system->N_cm - 1;
//                workspace->U->val[top] = -Dot( x, y, system->N );
//                ++top;
//
//                workspace->U->start[system->N_cm] = top;
//                break;
//
//            case TRI_SOLVE_LEVEL_SCHED_PA:
//                tri_solve_level_sched( workspace->L_EE, ones, x, workspace->L_EE->n, LOWER, TRUE );
//                Transpose_I( workspace->U_EE );
//                tri_solve_level_sched( workspace->U_EE, ones, y, workspace->U_EE->n, LOWER, TRUE );
//                Transpose_I( workspace->U_EE );
//
//                memcpy( workspace->L->start, workspace->L_EE->start, sizeof(unsigned int) * (system->N + 1) );
//                memcpy( workspace->L->j, workspace->L_EE->j, sizeof(unsigned int) * workspace->L_EE->start[workspace->L_EE->n] );
//                memcpy( workspace->L->val, workspace->L_EE->val, sizeof(real) * workspace->L_EE->start[workspace->L_EE->n] );
//
//                top = workspace->L->start[system->N];
//                for ( i = 0; i < system->N; ++i )
//                {
//                    workspace->L->j[top] = i;
//                    workspace->L->val[top] = x[i];
//                    ++top;
//                }
//
//                workspace->L->j[top] = system->N_cm - 1;
//                workspace->L->val[top] = 1.0;
//                ++top;
//
//                workspace->L->start[system->N_cm] = top;
//
//                top = 0;
//                for ( i = 0; i < system->N; ++i )
//                {
//                    workspace->U->start[i] = top;
//                    memcpy( workspace->U->j + top, workspace->U_EE->j + workspace->U_EE->start[i],
//                            sizeof(unsigned int) * (workspace->U_EE->start[i + 1] - workspace->U_EE->start[i]) );
//                    memcpy( workspace->U->val + top, workspace->U_EE->val + workspace->U_EE->start[i],
//                            sizeof(real) * (workspace->U_EE->start[i + 1] - workspace->U_EE->start[i]) );
//                    top += (workspace->U_EE->start[i + 1] - workspace->U_EE->start[i]);
//
//                    workspace->U->j[top] = system->N_cm - 1;
//                    workspace->U->val[top] = y[i];
//                    ++top;
//                }
//
//                workspace->U->start[system->N_cm - 1] = top;
//
//                workspace->U->j[top] = system->N_cm - 1;
//                workspace->U->val[top] = -Dot( x, y, system->N );
//                ++top;
//
//                workspace->U->start[system->N_cm] = top;
//                break;
//
//            //TODO: add Jacobi iter, etc.?
//            default:
//                tri_solve( workspace->L_EE, ones, x, workspace->L_EE->n, LOWER );
//                Transpose_I( workspace->U_EE );
//                tri_solve( workspace->U_EE, ones, y, workspace->U_EE->n, LOWER );
//                Transpose_I( workspace->U_EE );
//
//                memcpy( workspace->L->start, workspace->L_EE->start, sizeof(unsigned int) * (system->N + 1) );
//                memcpy( workspace->L->j, workspace->L_EE->j, sizeof(unsigned int) * workspace->L_EE->start[workspace->L_EE->n] );
//                memcpy( workspace->L->val, workspace->L_EE->val, sizeof(real) * workspace->L_EE->start[workspace->L_EE->n] );
//
//                top = workspace->L->start[system->N];
//                for ( i = 0; i < system->N; ++i )
//                {
//                    workspace->L->j[top] = i;
//                    workspace->L->val[top] = x[i];
//                    ++top;
//                }
//
//                workspace->L->j[top] = system->N_cm - 1;
//                workspace->L->val[top] = 1.0;
//                ++top;
//
//                workspace->L->start[system->N_cm] = top;
//
//                top = 0;
//                for ( i = 0; i < system->N; ++i )
//                {
//                    workspace->U->start[i] = top;
//                    memcpy( workspace->U->j + top, workspace->U_EE->j + workspace->U_EE->start[i],
//                            sizeof(unsigned int) * (workspace->U_EE->start[i + 1] - workspace->U_EE->start[i]) );
//                    memcpy( workspace->U->val + top, workspace->U_EE->val + workspace->U_EE->start[i],
//                            sizeof(real) * (workspace->U_EE->start[i + 1] - workspace->U_EE->start[i]) );
//                    top += (workspace->U_EE->start[i + 1] - workspace->U_EE->start[i]);
//
//                    workspace->U->j[top] = system->N_cm - 1;
//                    workspace->U->val[top] = y[i];
//                    ++top;
//                }
//
//                workspace->U->start[system->N_cm - 1] = top;
//
//                workspace->U->j[top] = system->N_cm - 1;
//                workspace->U->val[top] = -Dot( x, y, system->N );
//                ++top;
//
//                workspace->U->start[system->N_cm] = top;
//                break;
//        }
//    }
//
//#if defined(DEBUG)
//#define SIZE (1000)
//    char fname[SIZE];
//
//    if ( control->cm_solver_pre_comp_type != DIAG_PC )
//    {
//        fprintf( stderr, "condest = %f\n", condest(workspace->L) );
//
//#if defined(DEBUG_FOCUS)
//        snprintf( fname, SIZE + 10, "%s.L%d.out", control->sim_name, data->step );
//        Print_Sparse_Matrix2( workspace->L, fname, NULL );
//        snprintf( fname, SIZE + 10, "%s.U%d.out", control->sim_name, data->step );
//        Print_Sparse_Matrix2( workspace->U, fname, NULL );
//
//        fprintf( stderr, "icholt-" );
//        snprintf( fname, SIZE + 10, "%s.L%d.out", control->sim_name, data->step );
//        Print_Sparse_Matrix2( workspace->L, fname, NULL );
//        Print_Sparse_Matrix( U );
//#endif
//    }
//#undef SIZE
//#endif
//}


/* Compute preconditioner for EE
 */
static void Compute_Preconditioner_EE( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs )
{
    real time;
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = workspace->H_sp;
    }
    else
    {
        Hptr = workspace->H;
    }

#if defined(TEST_MAT)
    Hptr = create_test_mat( );
#endif

    if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
            control->cm_solver_pre_comp_type == ILU_PAR_PC ||
            control->cm_solver_pre_comp_type == ILUT_PAR_PC )
    {
        Hptr->val[Hptr->start[system->N + 1] - 1] = 1.0;
    }

    time = Get_Time( );
    if ( control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA )
    {
        setup_graph_coloring( control, workspace, Hptr, &workspace->H_full, &workspace->H_p );
        Sort_Matrix_Rows( workspace->H_p );
        Hptr = workspace->H_p;
    }
    data->timing.cm_sort_mat_rows += Get_Timing_Info( time );
    
    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case DIAG_PC:
            data->timing.cm_solver_pre_comp +=
                diag_pre_comp( Hptr, workspace->Hdia_inv );
            break;

        case ICHOLT_PC:
            data->timing.cm_solver_pre_comp +=
                ICHOLT( Hptr, workspace->droptol, workspace->L, workspace->U );
            break;

        case ILU_PAR_PC:
            data->timing.cm_solver_pre_comp +=
                ILU_PAR( Hptr, control->cm_solver_pre_comp_sweeps, workspace->L, workspace->U );
            break;

        case ILUT_PAR_PC:
            data->timing.cm_solver_pre_comp +=
                ILUT_PAR( Hptr, workspace->droptol, control->cm_solver_pre_comp_sweeps,
                        workspace->L, workspace->U );
            break;

        case ILU_SUPERLU_MT_PC:
#if defined(HAVE_SUPERLU_MT)
            data->timing.cm_solver_pre_comp +=
                SuperLU_Factorize( Hptr, workspace->L, workspace->U );
#else
            fprintf( stderr, "SuperLU MT support disabled. Re-compile before enabling. Terminating...\n" );
            exit( INVALID_INPUT );
#endif
            break;

        case SAI_PC:
#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
            data->timing.cm_solver_pre_comp +=
                sparse_approx_inverse( workspace->H_full, workspace->H_spar_patt_full,
                        &workspace->H_app_inv );
#else
            fprintf( stderr, "LAPACKE support disabled. Re-compile before enabling. Terminating...\n" );
            exit( INVALID_INPUT );
#endif
            break;

        default:
            fprintf( stderr, "Unrecognized preconditioner computation method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }

    if ( control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA )
    {
        if ( control->cm_domain_sparsify_enabled == TRUE )
        {
            Hptr = workspace->H_sp;
        }
        else
        {
            Hptr = workspace->H;
        }
    }

    if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
            control->cm_solver_pre_comp_type == ILU_PAR_PC ||
            control->cm_solver_pre_comp_type == ILUT_PAR_PC )
    {
        Hptr->val[Hptr->start[system->N + 1] - 1] = 0.0;
    }

#if defined(DEBUG)
#define SIZE (1000)
    char fname[SIZE];

    if ( control->cm_solver_pre_comp_type != NONE_PC && 
            control->cm_solver_pre_comp_type != DIAG_PC )
    {
        fprintf( stderr, "condest = %f\n", condest(workspace->L, workspace->U) );

#if defined(DEBUG_FOCUS)
        snprintf( fname, SIZE + 10, "%s.L%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->L, fname, NULL );
        snprintf( fname, SIZE + 10, "%s.U%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->U, fname, NULL );
#endif
    }
#undef SIZE
#endif
}


/* Compute preconditioner for ACKS2
 */
static void Compute_Preconditioner_ACKS2( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs )
{
    real time;
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = workspace->H_sp;
    }
    else
    {
        Hptr = workspace->H;
    }

#if defined(TEST_MAT)
    Hptr = create_test_mat( );
#endif

    if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
            control->cm_solver_pre_comp_type == ILU_PAR_PC ||
            control->cm_solver_pre_comp_type == ILUT_PAR_PC )
    {
        Hptr->val[Hptr->start[system->N + 1] - 1] = 1.0;
        Hptr->val[Hptr->start[system->N_cm] - 1] = 1.0;
    }

    time = Get_Time( );
    if ( control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA )
    {
        setup_graph_coloring( control, workspace, Hptr, &workspace->H_full, &workspace->H_p );
        Sort_Matrix_Rows( workspace->H_p );
        Hptr = workspace->H_p;
    }
    data->timing.cm_sort_mat_rows += Get_Timing_Info( time );
    
    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case DIAG_PC:
            data->timing.cm_solver_pre_comp +=
                diag_pre_comp( Hptr, workspace->Hdia_inv );
            break;

        case ICHOLT_PC:
            data->timing.cm_solver_pre_comp +=
                ICHOLT( Hptr, workspace->droptol, workspace->L, workspace->U );
            break;

        case ILU_PAR_PC:
            data->timing.cm_solver_pre_comp +=
                ILU_PAR( Hptr, control->cm_solver_pre_comp_sweeps, workspace->L, workspace->U );
            break;

        case ILUT_PAR_PC:
            data->timing.cm_solver_pre_comp +=
                ILUT_PAR( Hptr, workspace->droptol, control->cm_solver_pre_comp_sweeps,
                        workspace->L, workspace->U );
            break;

        case ILU_SUPERLU_MT_PC:
#if defined(HAVE_SUPERLU_MT)
            data->timing.cm_solver_pre_comp +=
                SuperLU_Factorize( Hptr, workspace->L, workspace->U );
#else
            fprintf( stderr, "SuperLU MT support disabled. Re-compile before enabling. Terminating...\n" );
            exit( INVALID_INPUT );
#endif
            break;

        case SAI_PC:
#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
            data->timing.cm_solver_pre_comp +=
                sparse_approx_inverse( workspace->H_full, workspace->H_spar_patt_full,
                        &workspace->H_app_inv );
#else
            fprintf( stderr, "LAPACKE support disabled. Re-compile before enabling. Terminating...\n" );
            exit( INVALID_INPUT );
#endif
            break;

        default:
            fprintf( stderr, "Unrecognized preconditioner computation method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }

    if ( control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA )
    {
        if ( control->cm_domain_sparsify_enabled == TRUE )
        {
            Hptr = workspace->H_sp;
        }
        else
        {
            Hptr = workspace->H;
        }
    }

    if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
            control->cm_solver_pre_comp_type == ILU_PAR_PC ||
            control->cm_solver_pre_comp_type == ILUT_PAR_PC )
    {
        Hptr->val[Hptr->start[system->N + 1] - 1] = 0.0;
        Hptr->val[Hptr->start[system->N_cm] - 1] = 0.0;
    }

#if defined(DEBUG)
#define SIZE (1000)
    char fname[SIZE];

    if ( control->cm_solver_pre_comp_type != NONE_PC || 
            control->cm_solver_pre_comp_type != DIAG_PC )
    {
        fprintf( stderr, "condest = %f\n", condest(workspace->L, workspace->U) );

#if defined(DEBUG_FOCUS)
        snprintf( fname, SIZE + 10, "%s.L%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->L, fname, NULL );
        snprintf( fname, SIZE + 10, "%s.U%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->U, fname, NULL );
#endif
    }
#undef SIZE
#endif
}


static void Setup_Preconditioner_QEq( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs )
{
    int fillin;
    real time;
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = workspace->H_sp;
    }
    else
    {
        Hptr = workspace->H;
    }

    /* sort H needed for SpMV's in linear solver, H or H_sp needed for preconditioning */
    time = Get_Time( );
    Sort_Matrix_Rows( workspace->H );
    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Sort_Matrix_Rows( workspace->H_sp );
    }
    data->timing.cm_sort_mat_rows += Get_Timing_Info( time );

    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case DIAG_PC:
            if ( workspace->Hdia_inv == NULL )
            {
                workspace->Hdia_inv = (real *) scalloc( Hptr->n, sizeof( real ),
                        "Setup_Preconditioner_QEq::workspace->Hdiv_inv" );
            }
            break;

        case ICHOLT_PC:
            Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );

            fillin = Estimate_LU_Fill( Hptr, workspace->droptol );

            if ( workspace->L == NULL )
            {
                Allocate_Matrix( &(workspace->L), Hptr->n, fillin );
                Allocate_Matrix( &(workspace->U), Hptr->n, fillin );
            }
            else if ( workspace->L->m < fillin )
            {
                Deallocate_Matrix( workspace->L );
                Deallocate_Matrix( workspace->U );

                Allocate_Matrix( &(workspace->L), Hptr->n, fillin );
                Allocate_Matrix( &(workspace->U), Hptr->n, fillin );
            }
            break;

        case ILU_PAR_PC:
            if ( workspace->L == NULL )
            {
                /* factors have sparsity pattern as H */
                Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m );
                Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m );
            }
            else if ( workspace->L->m < Hptr->m )
            {
                Deallocate_Matrix( workspace->L );
                Deallocate_Matrix( workspace->U );

                /* factors have sparsity pattern as H */
                Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m );
                Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m );
            }
            break;

        case ILUT_PAR_PC:
            Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );

            if ( workspace->L == NULL )
            {
                /* TODO: safest storage estimate is ILU(0)
                 * (same as lower triangular portion of H), could improve later */
                Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m );
                Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m );
            }
            else if ( workspace->L->m < Hptr->m )
            {
                Deallocate_Matrix( workspace->L );
                Deallocate_Matrix( workspace->U );

                /* TODO: safest storage estimate is ILU(0)
                 * (same as lower triangular portion of H), could improve later */
                Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m );
                Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m );
            }
            break;

        case ILU_SUPERLU_MT_PC:
            if ( workspace->L == NULL )
            {
                /* factors have sparsity pattern as H */
                Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m );
                Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m );
            }
            else if ( workspace->L->m < Hptr->m )
            {
                Deallocate_Matrix( workspace->L );
                Deallocate_Matrix( workspace->U );

                /* factors have sparsity pattern as H */
                Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m );
                Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m );
            }
            break;

        case SAI_PC:
            setup_sparse_approx_inverse( Hptr, &workspace->H_full, &workspace->H_spar_patt,
                    &workspace->H_spar_patt_full, &workspace->H_app_inv,
                    control->cm_solver_pre_comp_sai_thres );
            break;

        default:
            fprintf( stderr, "[ERROR] Unrecognized preconditioner computation method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }
}


/* Setup routines before computing the preconditioner for EE
 */
static void Setup_Preconditioner_EE( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs )
{
    int fillin;
    real time;
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = workspace->H_sp;
    }
    else
    {
        Hptr = workspace->H;
    }

    /* sorted H needed for SpMV's in linear solver, H or H_sp needed for preconditioning */
    time = Get_Time( );
    Sort_Matrix_Rows( workspace->H );
    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Sort_Matrix_Rows( workspace->H_sp );
    }
    data->timing.cm_sort_mat_rows += Get_Timing_Info( time );

    if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
            control->cm_solver_pre_comp_type == ILU_PAR_PC ||
            control->cm_solver_pre_comp_type == ILUT_PAR_PC )
    {
        Hptr->val[Hptr->start[system->N + 1] - 1] = 1.0;
    }

    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case DIAG_PC:
            if ( workspace->Hdia_inv == NULL )
            {
                workspace->Hdia_inv = (real *) scalloc( system->N_cm, sizeof( real ),
                        "Setup_Preconditioner_QEq::workspace->Hdiv_inv" );
            }
            break;

        case ICHOLT_PC:
            Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );

            fillin = Estimate_LU_Fill( Hptr, workspace->droptol );

            if ( workspace->L == NULL )
            {
                Allocate_Matrix( &(workspace->L), system->N_cm, fillin + system->N_cm );
                Allocate_Matrix( &(workspace->U), system->N_cm, fillin + system->N_cm );
            }
            else if ( workspace->L->m < fillin + system->N_cm )
            {
                Deallocate_Matrix( workspace->L );
                Deallocate_Matrix( workspace->U );

                Allocate_Matrix( &(workspace->L), system->N_cm, fillin + system->N_cm );
                Allocate_Matrix( &(workspace->U), system->N_cm, fillin + system->N_cm );
            }
            break;

        case ILU_PAR_PC:
            if ( workspace->L == NULL )
            {
                /* factors have sparsity pattern as H */
                Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m );
                Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m );
            }
            else if ( workspace->L->m < Hptr->m )
            {
                Deallocate_Matrix( workspace->L );
                Deallocate_Matrix( workspace->U );

                /* factors have sparsity pattern as H */
                Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m );
                Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m );
            }
            break;

        case ILUT_PAR_PC:
            Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );

            if ( workspace->L == NULL )
            {
                /* TODO: safest storage estimate is ILU(0)
                 * (same as lower triangular portion of H), could improve later */
                Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m );
                Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m );
            }
            else if ( workspace->L->m < Hptr->m )
            {
                Deallocate_Matrix( workspace->L );
                Deallocate_Matrix( workspace->U );

                /* TODO: safest storage estimate is ILU(0)
                 * (same as lower triangular portion of H), could improve later */
                Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m );
                Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m );
            }
            break;

        case ILU_SUPERLU_MT_PC:
            if ( workspace->L == NULL )
            {
                /* factors have sparsity pattern as H */
                Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m );
                Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m );
            }
            else if ( workspace->L->m < Hptr->m )
            {
                Deallocate_Matrix( workspace->L );
                Deallocate_Matrix( workspace->U );

                /* factors have sparsity pattern as H */
                Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m );
                Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m );
                {
                    fprintf( stderr, "[ERROR] not enough memory for preconditioning matrices. terminating.\n" );
                    exit( INSUFFICIENT_MEMORY );
                }
            }
            break;

        case SAI_PC:
            setup_sparse_approx_inverse( Hptr, &workspace->H_full, &workspace->H_spar_patt,
                    &workspace->H_spar_patt_full, &workspace->H_app_inv,
                    control->cm_solver_pre_comp_sai_thres );
            break;

        default:
            fprintf( stderr, "[ERROR] Unrecognized preconditioner computation method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }

    if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
            control->cm_solver_pre_comp_type == ILU_PAR_PC ||
            control->cm_solver_pre_comp_type == ILUT_PAR_PC )
    {
        Hptr->val[Hptr->start[system->N + 1] - 1] = 0.0;
    }
}


/* Setup routines before computing the preconditioner for ACKS2
 */
static void Setup_Preconditioner_ACKS2( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs )
{
    int fillin;
    real time;
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = workspace->H_sp;
    }
    else
    {
        Hptr = workspace->H;
    }

    /* sort H needed for SpMV's in linear solver, H or H_sp needed for preconditioning */
    time = Get_Time( );
    Sort_Matrix_Rows( workspace->H );
    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Sort_Matrix_Rows( workspace->H_sp );
    }
    data->timing.cm_sort_mat_rows += Get_Timing_Info( time );

    if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
            control->cm_solver_pre_comp_type == ILU_PAR_PC ||
            control->cm_solver_pre_comp_type == ILUT_PAR_PC )
    {
        Hptr->val[Hptr->start[system->N + 1] - 1] = 1.0;
        Hptr->val[Hptr->start[system->N_cm] - 1] = 1.0;
    }

    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case DIAG_PC:
            if ( workspace->Hdia_inv == NULL )
            {
                workspace->Hdia_inv = (real *) scalloc( Hptr->n, sizeof( real ),
                        "Setup_Preconditioner_QEq::workspace->Hdiv_inv" );
            }
            break;

        case ICHOLT_PC:
            Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );

            fillin = Estimate_LU_Fill( Hptr, workspace->droptol );

            if ( workspace->L == NULL )
            {
                Allocate_Matrix( &(workspace->L), Hptr->n, fillin );
                Allocate_Matrix( &(workspace->U), Hptr->n, fillin );
            }
            else if ( workspace->L->m < fillin )
            {
                Deallocate_Matrix( workspace->L );
                Deallocate_Matrix( workspace->U );

                /* factors have sparsity pattern as H */
                Allocate_Matrix( &(workspace->L), Hptr->n, fillin );
                Allocate_Matrix( &(workspace->U), Hptr->n, fillin );
            }
            break;

        case ILU_PAR_PC:
            if ( workspace->L == NULL )
            {
                /* factors have sparsity pattern as H */
                Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m );
                Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m );
            }
            else if ( workspace->L->m < Hptr->m )
            {
                Deallocate_Matrix( workspace->L );
                Deallocate_Matrix( workspace->U );

                /* factors have sparsity pattern as H */
                Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m );
                Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m );
            }
            break;

        case ILUT_PAR_PC:
            Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );

            if ( workspace->L == NULL )
            {
                /* TODO: safest storage estimate is ILU(0)
                 * (same as lower triangular portion of H), could improve later */
                Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m );
                Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m );
            }
            else if ( workspace->L->m < Hptr->m )
            {
                Deallocate_Matrix( workspace->L );
                Deallocate_Matrix( workspace->U );

                /* factors have sparsity pattern as H */
                Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m );
                Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m );
            }
            break;

        case ILU_SUPERLU_MT_PC:
            if ( workspace->L == NULL )
            {
                /* factors have sparsity pattern as H */
                Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m );
                Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m );
            }
            else if ( workspace->L->m < Hptr->m )
            {
                Deallocate_Matrix( workspace->L );
                Deallocate_Matrix( workspace->U );

                /* factors have sparsity pattern as H */
                Allocate_Matrix( &(workspace->L), Hptr->n, Hptr->m );
                Allocate_Matrix( &(workspace->U), Hptr->n, Hptr->m );
            }
            break;

        case SAI_PC:
            setup_sparse_approx_inverse( Hptr, &workspace->H_full, &workspace->H_spar_patt,
                    &workspace->H_spar_patt_full, &workspace->H_app_inv,
                    control->cm_solver_pre_comp_sai_thres );
            break;

        default:
            fprintf( stderr, "[ERROR] Unrecognized preconditioner computation method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }

    if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
            control->cm_solver_pre_comp_type == ILU_PAR_PC ||
            control->cm_solver_pre_comp_type == ILUT_PAR_PC )
    {
        Hptr->val[Hptr->start[system->N + 1] - 1] = 0.0;
        Hptr->val[Hptr->start[system->N_cm] - 1] = 0.0;
    }
}


/* Combine ficticious charges s and t to get atomic charge q for QEq method
 */
static void Calculate_Charges_QEq( const reax_system * const system,
        static_storage * const workspace )
{
    int i;
    real u, s_sum, t_sum;

    s_sum = 0.0;
    t_sum = 0.0;
    for ( i = 0; i < system->N_cm; ++i )
    {
        s_sum += workspace->s[0][i];
        t_sum += workspace->t[0][i];
    }

    u = s_sum / t_sum;
    for ( i = 0; i < system->N_cm; ++i )
    {
        system->atoms[i].q = workspace->s[0][i] - u * workspace->t[0][i];

#if defined(DEBUG_FOCUS)
        printf("atom %4d: %f\n", i, system->atoms[i].q);
        printf("  x[0]: %10.5f, x[1]: %10.5f, x[2]:  %10.5f\n",
                system->atoms[i].x[0], system->atoms[i].x[1], system->atoms[i].x[2]);
#endif
    }
}


/* Get atomic charge q for EE method
 */
static void Calculate_Charges_EE( const reax_system * const system,
        static_storage * const workspace )
{
    int i;

    for ( i = 0; i < system->N; ++i )
    {
        system->atoms[i].q = workspace->s[0][i];

#if defined(DEBUG_FOCUS)
        printf( "atom %4d: %f\n", i, system->atoms[i].q );
        printf( "  x[0]: %10.5f, x[1]: %10.5f, x[2]:  %10.5f\n",
               system->atoms[i].x[0], system->atoms[i].x[1], system->atoms[i].x[2] );
#endif
    }
}


/* Main driver method for QEq kernel
 *  1) init / setup routines for preconditioning of linear solver
 *  2) compute preconditioner
 *  3) extrapolate charges
 *  4) perform 2 linear solves
 *  5) compute atomic charges based on output of (4)
 */
static void QEq( reax_system * const system, control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs, const output_controls * const out_control )
{
    int iters;

    if ( control->cm_solver_pre_comp_refactor > 0 &&
            ((data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) )
        
    {
        Setup_Preconditioner_QEq( system, control, data, workspace, far_nbrs );

        Compute_Preconditioner_QEq( system, control, data, workspace, far_nbrs );
    }

    Extrapolate_Charges_QEq( system, control, data, workspace );

    switch ( control->cm_solver_type )
    {
    case GMRES_S:
        iters = GMRES( workspace, control, data, workspace->H,
                workspace->b_s, control->cm_solver_q_err, workspace->s[0],
                (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE );
        iters += GMRES( workspace, control, data, workspace->H,
                workspace->b_t, control->cm_solver_q_err, workspace->t[0], FALSE );
        break;

    case GMRES_H_S:
        iters = GMRES_HouseHolder( workspace, control, data, workspace->H,
                workspace->b_s, control->cm_solver_q_err, workspace->s[0],
                (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE );
        iters += GMRES_HouseHolder( workspace, control, data, workspace->H,
                workspace->b_t, control->cm_solver_q_err, workspace->t[0], 0 );
        break;

    case CG_S:
        iters = CG( workspace, control, data, workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE ) + 1;
        iters += CG( workspace, control, data, workspace->H, workspace->b_t, control->cm_solver_q_err,
                workspace->t[0], FALSE ) + 1;
        break;

    case SDM_S:
        iters = SDM( workspace, control, data, workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE ) + 1;
        iters += SDM( workspace, control, data, workspace->H, workspace->b_t, control->cm_solver_q_err,
                      workspace->t[0], FALSE ) + 1;
        break;

    default:
        fprintf( stderr, "Unrecognized QEq solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    data->timing.cm_solver_iters += iters;

    Calculate_Charges_QEq( system, workspace );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "%d %.9f %.9f %.9f %.9f %.9f %.9f\n", data->step,
       workspace->s[0][0], workspace->t[0][0],
       workspace->s[0][1], workspace->t[0][1],
       workspace->s[0][2], workspace->t[0][2] );
    if( data->step == control->nsteps )
    {
        Print_Charges( system, control, workspace, data->step );
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
static void EE( reax_system * const system, control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs, const output_controls * const out_control )
{
    int iters;

    if ( control->cm_solver_pre_comp_refactor > 0 &&
            ((data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) )
    {
        Setup_Preconditioner_EE( system, control, data, workspace, far_nbrs );

        Compute_Preconditioner_EE( system, control, data, workspace, far_nbrs );
    }

    Extrapolate_Charges_EE( system, control, data, workspace );

    switch ( control->cm_solver_type )
    {
    case GMRES_S:
        iters = GMRES( workspace, control, data, workspace->H,
                workspace->b_s, control->cm_solver_q_err, workspace->s[0],
                (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE );
        break;

    case GMRES_H_S:
        iters = GMRES_HouseHolder( workspace, control, data,workspace->H,
                workspace->b_s, control->cm_solver_q_err, workspace->s[0],
                control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0 );
        break;

    case CG_S:
        iters = CG( workspace, control, data, workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE ) + 1;
        break;

    case SDM_S:
        iters = SDM( workspace, control, data, workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE ) + 1;
        break;

    default:
        fprintf( stderr, "Unrecognized EE solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    data->timing.cm_solver_iters += iters;

    Calculate_Charges_EE( system, workspace );

    // if( data->step == control->nsteps )
    //Print_Charges( system, control, workspace, data->step );
}


/* Main driver method for ACKS2 kernel
 *  1) init / setup routines for preconditioning of linear solver
 *  2) compute preconditioner
 *  3) extrapolate charges
 *  4) perform 1 linear solve
 *  5) compute atomic charges based on output of (4)
 */
static void ACKS2( reax_system * const system, control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs, const output_controls * const out_control )
{
    int iters;

    if ( control->cm_solver_pre_comp_refactor > 0 &&
            ((data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) )
    {
        Setup_Preconditioner_ACKS2( system, control, data, workspace, far_nbrs );

        Compute_Preconditioner_ACKS2( system, control, data, workspace, far_nbrs );
    }

//   Print_Linear_System( system, control, workspace, data->step );

    Extrapolate_Charges_EE( system, control, data, workspace );

#if defined(DEBUG_FOCUS)
#define SIZE (200)
    char fname[SIZE];
    FILE * fp;

    if ( data->step % 10 == 0 )
    {
        snprintf( fname, SIZE + 11, "s_%d_%s.out", data->step, control->sim_name );
        fp = fopen( fname, "w" );
        Vector_Print( fp, NULL, workspace->s[0], system->N_cm );
        fclose( fp );
    }
#undef SIZE
#endif

    switch ( control->cm_solver_type )
    {
    case GMRES_S:
        iters = GMRES( workspace, control, data, workspace->H,
                workspace->b_s, control->cm_solver_q_err, workspace->s[0],
                (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE );
        break;

    case GMRES_H_S:
        iters = GMRES_HouseHolder( workspace, control, data,workspace->H,
                workspace->b_s, control->cm_solver_q_err, workspace->s[0],
                control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0 );
        break;

    case CG_S:
        iters = CG( workspace, control, data, workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE ) + 1;
        break;

    case SDM_S:
        iters = SDM( workspace, control, data, workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], (control->cm_solver_pre_comp_refactor > 0 &&
                 (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE ) + 1;
        break;

    default:
        fprintf( stderr, "Unrecognized ACKS2 solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    data->timing.cm_solver_iters += iters;

    Calculate_Charges_EE( system, workspace );
}


void Compute_Charges( reax_system * const system, control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const reax_list * const far_nbrs, const output_controls * const out_control )
{
#if defined(DEBUG_FOCUS)
#define SIZE (200)
    char fname[SIZE];
    FILE * fp;

    if ( data->step % 10 == 0 )
    {
        snprintf( fname, SIZE + 11, "H_%d_%s.out", data->step, control->sim_name );
        Print_Sparse_Matrix2( workspace->H, fname, NULL );
//        Print_Sparse_Matrix_Binary( workspace->H, fname );

        snprintf( fname, SIZE + 11, "b_s_%d_%s.out", data->step, control->sim_name );
        fp = fopen( fname, "w" );
        Vector_Print( fp, NULL, workspace->b_s, system->N_cm );
        fclose( fp );

//        snprintf( fname, SIZE + 11, "b_t_%d_%s.out", data->step, control->sim_name );
//        fp = fopen( fname, "w" );
//        Vector_Print( fp, NULL, workspace->b_t, system->N_cm );
//        fclose( fp );
    }
#undef SIZE
#endif

    switch ( control->charge_method )
    {
    case QEQ_CM:
        QEq( system, control, data, workspace, far_nbrs, out_control );
        break;

    case EE_CM:
        EE( system, control, data, workspace, far_nbrs, out_control );
        break;

    case ACKS2_CM:
        ACKS2( system, control, data, workspace, far_nbrs, out_control );
        break;

    default:
        fprintf( stderr, "Invalid charge method. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }
}
