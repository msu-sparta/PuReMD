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
#if defined(DEBUG_FOCUS)
  #include "io_tools.h"
#endif
#include "list.h"
#include "lin_alg.h"
#include "tool_box.h"
#include "vector.h"

#if defined(HAVE_TENSORFLOW)
  #include "tensorflow_regressor.c"   
#endif
#include "ELM_regressor.c"

int is_refactoring_step( control_params * const control,
        simulation_data * const data )
{
    int ret;

    if ( control->cm_solver_pre_comp_refactor != -1 )
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
        /* cases:
         *  - first overall MD step for non-restarted OR restarted MD run
         *  - total losses from degradation of prec. outweight costs of recomputing prec.
         *  */
        if ( data->step - data->prev_steps == 0
                || data->timing.cm_total_loss > data->timing.cm_last_pre_comp )
        {
            ret = TRUE;
        }
        else
        {
            ret = FALSE;
        }
    }

    return ret;
}

static void Spline_Extrapolate_Charges_QEq( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace )
{
    int i;
    real s_tmp, t_tmp;

    /* spline extrapolation for s & t */
    //TODO: good candidate for vectorization, avoid moving data with head pointer and circular buffer
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) \
        default(none) private(i, s_tmp, t_tmp) firstprivate(system, control, workspace)
#endif
    for ( i = 0; i < system->N_cm; ++i )
    {
        /* no extrapolation, previous solution as initial guess */
        if ( control->cm_init_guess_extrap1 == 0 )
        {
            s_tmp = workspace->s[0][i];
        }
        /* linear */
        else if ( control->cm_init_guess_extrap1 == 1 )
        {
            s_tmp = 2.0 * workspace->s[0][i] - workspace->s[1][i];
        }
        /* quadratic */
        else if ( control->cm_init_guess_extrap1 == 2 )
        {
            s_tmp = workspace->s[2][i] + 3.0 * (workspace->s[0][i] - workspace->s[1][i]);
        }
        /* cubic */
        else if ( control->cm_init_guess_extrap1 == 3 )
        {
            s_tmp = 4.0 * (workspace->s[0][i] + workspace->s[2][i])
                - (6.0 * workspace->s[1][i] + workspace->s[3][i]);
        }
        /* 4th order */
        else if ( control->cm_init_guess_extrap1 == 4 )
        {
            s_tmp = 5.0 * (workspace->s[0][i] - workspace->s[3][i])
                + 10.0 * (-1.0 * workspace->s[1][i] + workspace->s[2][i]) + workspace->s[4][i];
        }
        else
        {
            s_tmp = 0.0;
        }

        /* no extrapolation, previous solution as initial guess */
        if ( control->cm_init_guess_extrap2 == 0 )
        {
            t_tmp = workspace->t[0][i];
        }
        /* linear */
        else if ( control->cm_init_guess_extrap2 == 1 )
        {
            t_tmp = 2.0 * workspace->t[0][i] - workspace->t[1][i];
        }
        /* quadratic */
        else if ( control->cm_init_guess_extrap2 == 2 )
        {
            t_tmp = workspace->t[2][i] + 3.0 * (workspace->t[0][i] - workspace->t[1][i]);
        }
        /* cubic */
        else if ( control->cm_init_guess_extrap2 == 3 )
        {
            t_tmp = 4.0 * (workspace->t[0][i] + workspace->t[2][i]) -
                (6.0 * workspace->t[1][i] + workspace->t[3][i]);
        }
        /* 4th order */
        else if ( control->cm_init_guess_extrap2 == 4 )
        {
            t_tmp = 5.0 * (workspace->t[0][i] - workspace->t[3][i]) +
                10.0 * (-1.0 * workspace->t[1][i] + workspace->t[2][i]) + workspace->t[4][i];
        }
        else
        {
            t_tmp = 0.0;
        }

        // shifting
        // TODO: Check this part after working on WINDOW_SIZE variable
        int j = 0;
        for (j = WINDOW_SIZE-1;j > 0 ; j--) {
            workspace->s[j][i] = workspace->s[j-1][i];
            workspace->t[j][i] = workspace->t[j-1][i];
        }
        workspace->s[0][i] = s_tmp;
        workspace->t[0][i] = t_tmp;
    }
}


static void Spline_Extrapolate_Charges_EE( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace )
{
    int i,j;
    real s_tmp;
    double limit = WINDOW_SIZE;
    // TODO: this is a temporary solution decrease the cost of shifting after training is done
    if (data->step > 501) {
        limit = 10;
    }



    /* spline extrapolation for s */
    //TODO: good candidate for vectorization, avoid moving data with head pointer and circular buffer
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) \
        default(none) private(i, s_tmp, j) firstprivate(system, control, workspace)
#endif
    for ( i = 0; i < system->N_cm; ++i )
    {
        //for (j = 0; j < 5; j++) {
        //    diff[j] = workspace->s[j][i] - workspace->s[j+1][i];
        //}
        /* no extrapolation */
        if ( control->cm_init_guess_extrap1 == 0 )
        {
            s_tmp = workspace->s[0][i];
            //s_tmp = diff[0];
        }
        /* linear */
        else if ( control->cm_init_guess_extrap1 == 1 )
        {
            s_tmp = 2.0 * workspace->s[0][i] - workspace->s[1][i];
            //s_tmp = 2.0 * diff[0] - diff[1];
        }
        /* quadratic */
        else if ( control->cm_init_guess_extrap1 == 2 )
        {
            s_tmp = workspace->s[2][i] + 3.0 * (workspace->s[0][i]-workspace->s[1][i]);
            //s_tmp = diff[2] + 3.0 * (diff[0]-diff[1]);
        }
        /* cubic */
        else if ( control->cm_init_guess_extrap1 == 3 )
        {
            s_tmp = 4.0 * (workspace->s[0][i] + workspace->s[2][i]) -
                    (6.0 * workspace->s[1][i] + workspace->s[3][i] );

            //s_tmp = 4.0 * (diff[0] + diff[2]) -
            //        (6.0 * diff[1] + diff[3] );
        }
        /* 4th order */
        else if ( control->cm_init_guess_extrap1 == 4 )
        {
            s_tmp = 5.0 * (workspace->s[0][i] - workspace->s[3][i]) +
                10.0 * (-workspace->s[1][i] + workspace->s[2][i] ) + workspace->s[4][i];
            //s_tmp = 5.0 * (diff[0] - diff[3]) +
            //   10.0 * (-diff[1] + diff[2]) + diff[4];
        }
        else
        {
            s_tmp = 0.0;
        }

        // shifting
        // TODO: Check this part after working on WINDOW_SIZE variable
        int j = 0;
        for (j = limit-1;j > 0 ; j--) {
            workspace->s[j][i] = workspace->s[j-1][i];
        }
        workspace->s[0][i] = s_tmp;
        //workspace->s[0][i] = s_tmp + workspace->s[1][i];
    }
}


/* Compute preconditioner for QEq
 */
static void Compute_Preconditioner_QEq( const reax_system * const system,
        const control_params * const control, simulation_data * const data,
        static_storage * const workspace, int realloc )
{
    real time;
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = &workspace->H_sp;
    }
    else
    {
        Hptr = &workspace->H;
    }

#if defined(TEST_MAT)
    Hptr = create_test_mat( );
#endif

    time = Get_Time( );
    if ( control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA )
    {
        setup_graph_coloring( control, workspace, Hptr, &workspace->H_full,
                &workspace->H_p, realloc );
        Sort_Matrix_Rows( &workspace->H_p );
        Hptr = &workspace->H_p;
    }
    data->timing.cm_sort_mat_rows += Get_Timing_Info( time );

    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case JACOBI_PC:
            data->timing.cm_solver_pre_comp += jacobi( Hptr, workspace->Hdia_inv );
            break;

        case ICHOLT_PC:
            data->timing.cm_solver_pre_comp +=
                ICHOLT( Hptr, workspace->droptol, &workspace->L, &workspace->U );
            break;

        case ILUT_PC:
            if ( control->cm_solver_pre_comp_droptol > 0.0 )
            {
                data->timing.cm_solver_pre_comp +=
                    ILUT( Hptr, workspace->droptol, &workspace->L, &workspace->U );
            }
            else
            {
                data->timing.cm_solver_pre_comp +=
                    ILU( Hptr, &workspace->L, &workspace->U );
            }
            break;

        case ILUTP_PC:
            data->timing.cm_solver_pre_comp +=
                ILUTP( Hptr, workspace->droptol, &workspace->L, &workspace->U );
            break;

        case FG_ILUT_PC:
            if ( control->charge_method == QEQ_CM )
            {
                data->timing.cm_solver_pre_comp +=
                    FG_ICHOLT( Hptr, workspace->droptol, control->cm_solver_pre_comp_sweeps,
                            &workspace->L, &workspace->U );
            }
            else
            {
                data->timing.cm_solver_pre_comp +=
                    FG_ILUT( Hptr, workspace->droptol, control->cm_solver_pre_comp_sweeps,
                            &workspace->L, &workspace->U );
            }
            break;

        case SAI_PC:
#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
            data->timing.cm_solver_pre_comp +=
                sparse_approx_inverse( &workspace->H_full, &workspace->H_spar_patt_full,
                        &workspace->H_app_inv );
#else
            fprintf( stderr, "[ERROR] LAPACKE support disabled. Re-compile to enable. Terminating...\n" );
            exit( INVALID_INPUT );
#endif
            break;

        default:
            fprintf( stderr, "[ERROR] Unrecognized preconditioner computation method (%d). Terminating...\n",
                    control->cm_solver_pre_comp_type );
            exit( INVALID_INPUT );
            break;
    }

    //if ( control->cm_solver_pre_comp_refactor == -1 )
    //{
    //    data->timing.cm_last_pre_comp = data->timing.cm_solver_pre_comp;
    //    data->timing.cm_total_loss = 0.0;
    //}

#if defined(DEBUG)
    if ( control->cm_solver_pre_comp_type == ICHOLT_PC
            || control->cm_solver_pre_comp_type == ILUT_PC
            || control->cm_solver_pre_comp_type == ILUTP_PC
            || control->cm_solver_pre_comp_type == FG_ILUT_PC )
    {
        fprintf( stderr, "[INFO] condest = %f\n", condest(&workspace->L, &workspace->U) );

#if defined(DEBUG_FOCUS)
#define SIZE (1000)
        char fname[SIZE];
        snprintf( fname, SIZE, "%s.L%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->L, fname, NULL );
        snprintf( fname, SIZE, "%s.U%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->U, fname, NULL );
#undef SIZE
#endif
    }
#endif
}


/* Compute preconditioner for EE
 */
static void Compute_Preconditioner_EE( const reax_system * const system,
        const control_params * const control, simulation_data * const data,
        static_storage * const workspace, int realloc )
{
    real time;
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = &workspace->H_sp;
    }
    else
    {
        Hptr = &workspace->H;
    }

#if defined(TEST_MAT)
    Hptr = create_test_mat( );
#endif

    if ( control->cm_solver_pre_comp_type == ILUT_PC
            || control->cm_solver_pre_comp_type == FG_ILUT_PC )
    {
        Hptr->val[Hptr->start[system->N_cm] - 1] = 1.0;
    }

    time = Get_Time( );
    if ( control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA )
    {
        setup_graph_coloring( control, workspace, Hptr, &workspace->H_full,
                &workspace->H_p, realloc );
        Sort_Matrix_Rows( &workspace->H_p );
        Hptr = &workspace->H_p;
    }
    data->timing.cm_sort_mat_rows += Get_Timing_Info( time );
    
    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case JACOBI_PC:
            data->timing.cm_solver_pre_comp += jacobi( Hptr, workspace->Hdia_inv );
            break;

        case ICHOLT_PC:
            fprintf( stderr, "[ERROR] ICHOLT is not supported for indefinite, symmetric matrices of EE. Terminating...\n" );
            exit( INVALID_INPUT );
            break;

        case ILUT_PC:
            if ( control->cm_solver_pre_comp_droptol > 0.0 )
            {
                data->timing.cm_solver_pre_comp +=
                    ILUT( Hptr, workspace->droptol, &workspace->L, &workspace->U );
            }
            else
            {
                data->timing.cm_solver_pre_comp +=
                    ILU( Hptr, &workspace->L, &workspace->U );
            }
            break;

        case ILUTP_PC:
            data->timing.cm_solver_pre_comp +=
                ILUTP( Hptr, workspace->droptol, &workspace->L, &workspace->U );
            break;

        case FG_ILUT_PC:
            data->timing.cm_solver_pre_comp +=
                FG_ILUT( Hptr, workspace->droptol, control->cm_solver_pre_comp_sweeps,
                        &workspace->L, &workspace->U );
            break;

        case SAI_PC:
#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
            data->timing.cm_solver_pre_comp +=
                sparse_approx_inverse( &workspace->H_full, &workspace->H_spar_patt_full,
                        &workspace->H_app_inv );
#else
            fprintf( stderr, "[ERROR] LAPACKE support disabled. Re-compile to enable. Terminating...\n" );
            exit( INVALID_INPUT );
#endif
            break;

        default:
            fprintf( stderr, "[ERROR] Unrecognized preconditioner computation method (%d). Terminating...\n",
                   control->cm_solver_pre_comp_type );
            exit( INVALID_INPUT );
            break;
    }

    //if ( control->cm_solver_pre_comp_refactor == -1 )
    //{
    //    data->timing.cm_last_pre_comp = data->timing.cm_solver_pre_comp;
    //    data->timing.cm_total_loss = 0.0;
    //}

    if ( control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA )
    {
        if ( control->cm_domain_sparsify_enabled == TRUE )
        {
            Hptr = &workspace->H_sp;
        }
        else
        {
            Hptr = &workspace->H;
        }
    }

    if ( control->cm_solver_pre_comp_type == ILUT_PC
            || control->cm_solver_pre_comp_type == FG_ILUT_PC )
    {
        Hptr->val[Hptr->start[system->N_cm] - 1] = 0.0;
    }

#if defined(DEBUG)
    if ( control->cm_solver_pre_comp_type == ICHOLT_PC
            || control->cm_solver_pre_comp_type == ILUT_PC
            || control->cm_solver_pre_comp_type == ILUTP_PC
            || control->cm_solver_pre_comp_type == FG_ILUT_PC )
    {
        fprintf( stderr, "[INFO] condest = %f\n", condest(&workspace->L, &workspace->U) );

#if defined(DEBUG_FOCUS)
#define SIZE (1000)
        char fname[SIZE];
        snprintf( fname, SIZE, "%s.L%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->L, fname, NULL );
        snprintf( fname, SIZE, "%s.U%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->U, fname, NULL );
#undef SIZE
#endif
    }
#endif
}


/* Compute preconditioner for ACKS2
 */
static void Compute_Preconditioner_ACKS2( const reax_system * const system,
        const control_params * const control, simulation_data * const data,
        static_storage * const workspace, int realloc )
{
    real time;
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = &workspace->H_sp;
    }
    else
    {
        Hptr = &workspace->H;
    }

#if defined(TEST_MAT)
    Hptr = create_test_mat( );
#endif

    if ( control->cm_solver_pre_comp_type == ILUT_PC
            || control->cm_solver_pre_comp_type == FG_ILUT_PC )
    {
        Hptr->val[Hptr->start[system->N_cm - 1] - 1] = 1.0;
        Hptr->val[Hptr->start[system->N_cm] - 1] = 1.0;
    }

    time = Get_Time( );
    if ( control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA )
    {
        setup_graph_coloring( control, workspace, Hptr, &workspace->H_full,
                &workspace->H_p, realloc );
        Sort_Matrix_Rows( &workspace->H_p );
        Hptr = &workspace->H_p;
    }
    data->timing.cm_sort_mat_rows += Get_Timing_Info( time );
    
    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case JACOBI_PC:
            data->timing.cm_solver_pre_comp += jacobi( Hptr, workspace->Hdia_inv );
            break;

        case ICHOLT_PC:
            fprintf( stderr, "[ERROR] ICHOLT is not supported for indefinite, symmetric matrices of ACKS2. Terminating...\n" );
            exit( INVALID_INPUT );
            break;

        case ILUT_PC:
            if ( control->cm_solver_pre_comp_droptol > 0.0 )
            {
                data->timing.cm_solver_pre_comp +=
                    ILUT( Hptr, workspace->droptol, &workspace->L, &workspace->U );
            }
            else
            {
                data->timing.cm_solver_pre_comp +=
                    ILU( Hptr, &workspace->L, &workspace->U );
            }
            break;

        case ILUTP_PC:
            data->timing.cm_solver_pre_comp +=
                ILUTP( Hptr, workspace->droptol, &workspace->L, &workspace->U );
            break;

        case FG_ILUT_PC:
            data->timing.cm_solver_pre_comp +=
                FG_ILUT( Hptr, workspace->droptol, control->cm_solver_pre_comp_sweeps,
                        &workspace->L, &workspace->U );
            break;

        case SAI_PC:
#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
            data->timing.cm_solver_pre_comp +=
                sparse_approx_inverse( &workspace->H_full, &workspace->H_spar_patt_full,
                        &workspace->H_app_inv );
#else
            fprintf( stderr, "[ERROR] LAPACKE support disabled. Re-compile to enable. Terminating...\n" );
            exit( INVALID_INPUT );
#endif
            break;

        default:
            fprintf( stderr, "[ERROR] Unrecognized preconditioner computation method (%d). Terminating...\n",
                   control->cm_solver_pre_comp_type );
            exit( INVALID_INPUT );
            break;
    }

    //if ( control->cm_solver_pre_comp_refactor == -1 )
    //{
    //    data->timing.cm_last_pre_comp = data->timing.cm_solver_pre_comp;
    //    data->timing.cm_total_loss = 0.0;
    //}

    if ( control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA )
    {
        if ( control->cm_domain_sparsify_enabled == TRUE )
        {
            Hptr = &workspace->H_sp;
        }
        else
        {
            Hptr = &workspace->H;
        }
    }

    if ( control->cm_solver_pre_comp_type == ILUT_PC
            || control->cm_solver_pre_comp_type == FG_ILUT_PC )
    {
        Hptr->val[Hptr->start[system->N_cm - 1] - 1] = 0.0;
        Hptr->val[Hptr->start[system->N_cm] - 1] = 0.0;
    }

#if defined(DEBUG)
    if ( control->cm_solver_pre_comp_type == ICHOLT_PC
            || control->cm_solver_pre_comp_type == ILUT_PC
            || control->cm_solver_pre_comp_type == ILUTP_PC
            || control->cm_solver_pre_comp_type == FG_ILUT_PC )
    {
        fprintf( stderr, "[INFO] condest = %f\n", condest(&workspace->L, &workspace->U) );

#if defined(DEBUG_FOCUS)
#define SIZE (1000)
        char fname[SIZE];
        snprintf( fname, SIZE, "%s.L%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->L, fname, NULL );
        snprintf( fname, SIZE, "%s.U%d.out", control->sim_name, data->step );
        Print_Sparse_Matrix2( workspace->U, fname, NULL );
#undef SIZE
#endif
    }
#endif
}


static void Setup_Preconditioner_QEq( const reax_system * const system,
        const control_params * const control, simulation_data * const data,
        static_storage * const workspace, int realloc )
{
    int fillin;
    real time;
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
    time = Get_Time( );
    Sort_Matrix_Rows( &workspace->H );
    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Sort_Matrix_Rows( &workspace->H_sp );
    }
    data->timing.cm_sort_mat_rows += Get_Timing_Info( time );

    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case JACOBI_PC:
            if ( workspace->Hdia_inv == NULL )
            {
                workspace->Hdia_inv = scalloc( Hptr->n_max, sizeof( real ),
                        "Setup_Preconditioner_QEq::workspace->Hdia_inv" );
            }
            else if ( realloc == TRUE )
            {
                workspace->Hdia_inv = srealloc( workspace->Hdia_inv,
                        sizeof( real ) * Hptr->n_max,
                        "Setup_Preconditioner_QEq::workspace->Hdia_inv" );
            }
            break;

        case ICHOLT_PC:
            Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );

            fillin = Estimate_LU_Fill( Hptr, workspace->droptol );

            if ( workspace->L.allocated == FALSE )
            {
                Allocate_Matrix( &workspace->L, Hptr->n, Hptr->n_max, fillin );
                Allocate_Matrix( &workspace->U, Hptr->n, Hptr->n_max, fillin );
            }
            else if ( workspace->L.m < fillin || realloc == TRUE )
            {
                Deallocate_Matrix( &workspace->L );
                Deallocate_Matrix( &workspace->U );

                Allocate_Matrix( &workspace->L, Hptr->n, Hptr->n_max, fillin );
                Allocate_Matrix( &workspace->U, Hptr->n, Hptr->n_max, fillin );
            }
            break;

        case ILUT_PC:
            if ( control->cm_solver_pre_comp_droptol > 0.0 )
            {
                Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );
            }

            if ( workspace->L.allocated == FALSE )
            {
                Allocate_Matrix( &workspace->L, Hptr->n, Hptr->n_max, Hptr->m );
                Allocate_Matrix( &workspace->U, Hptr->n, Hptr->n_max, Hptr->m );
            }
            else if ( workspace->L.m < Hptr->m || realloc == TRUE )
            {
                Deallocate_Matrix( &workspace->L );
                Deallocate_Matrix( &workspace->U );

                Allocate_Matrix( &workspace->L, Hptr->n, Hptr->n_max, Hptr->m );
                Allocate_Matrix( &workspace->U, Hptr->n, Hptr->n_max, Hptr->m );
            }
            break;

        case ILUTP_PC:
        case FG_ILUT_PC:
            Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );

            if ( workspace->L.allocated == FALSE )
            {
                /* safest storage estimate is ILU(0) (same as
                 * lower triangular portion of H), could improve later */
                Allocate_Matrix( &workspace->L, Hptr->n, Hptr->n_max, Hptr->m );
                Allocate_Matrix( &workspace->U, Hptr->n, Hptr->n_max, Hptr->m );
            }
            else if ( workspace->L.m < Hptr->m || realloc == TRUE )
            {
                Deallocate_Matrix( &workspace->L );
                Deallocate_Matrix( &workspace->U );

                /* safest storage estimate is ILU(0) (same as
                 * lower triangular portion of H), could improve later */
                Allocate_Matrix( &workspace->L, Hptr->n, Hptr->n_max, Hptr->m );
                Allocate_Matrix( &workspace->U, Hptr->n, Hptr->n_max, Hptr->m );
            }
            break;

        case SAI_PC:
            setup_sparse_approx_inverse( Hptr, &workspace->H_full,
                    &workspace->H_spar_patt, &workspace->H_spar_patt_full,
                    &workspace->H_app_inv, control->cm_solver_pre_comp_sai_thres,
                    realloc );
            break;

        default:
            fprintf( stderr, "[ERROR] Unrecognized preconditioner computation method (%d). Terminating...\n",
                   control->cm_solver_pre_comp_type );
            exit( INVALID_INPUT );
            break;
    }
}


/* Setup routines before computing the preconditioner for EE
 */
static void Setup_Preconditioner_EE( const reax_system * const system,
        const control_params * const control, simulation_data * const data,
        static_storage * const workspace, int realloc )
{
    real time;
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = &workspace->H_sp;
    }
    else
    {
        Hptr = &workspace->H;
    }

    /* sorted H needed for SpMV's in linear solver, H or H_sp needed for preconditioning */
    time = Get_Time( );
    Sort_Matrix_Rows( &workspace->H );
    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Sort_Matrix_Rows( &workspace->H_sp );
    }
    data->timing.cm_sort_mat_rows += Get_Timing_Info( time );

    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case JACOBI_PC:
            if ( workspace->Hdia_inv == NULL )
            {
                workspace->Hdia_inv = scalloc( Hptr->n_max, sizeof( real ),
                        "Setup_Preconditioner_EE::workspace->Hdiv_inv" );
            }
            else if ( realloc == TRUE )
            {
                workspace->Hdia_inv = srealloc( workspace->Hdia_inv,
                        sizeof( real ) * Hptr->n_max,
                        "Setup_Preconditioner_EE::workspace->Hdiv_inv" );
            }
            break;

        case ICHOLT_PC:
            fprintf( stderr, "[ERROR] ICHOLT is not supported for indefinite, symmetric matrices of EE. Terminating...\n" );
            exit( INVALID_INPUT );
            break;

        case ILUT_PC:
            if ( control->cm_solver_pre_comp_droptol > 0.0 )
            {
                /* replace zeros on diagonal with non-zero values */
                Hptr->val[Hptr->start[system->N_cm] - 1] = 1.0;

                Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );

                /* put zeros back */
                Hptr->val[Hptr->start[system->N_cm] - 1] = 0.0;
            }

            if ( workspace->L.allocated == FALSE )
            {
                Allocate_Matrix( &workspace->L, Hptr->n, Hptr->n_max, Hptr->m );
                Allocate_Matrix( &workspace->U, Hptr->n, Hptr->n_max, Hptr->m );
            }
            else if ( workspace->L.m < Hptr->m || realloc == TRUE )
            {
                Deallocate_Matrix( &workspace->L );
                Deallocate_Matrix( &workspace->U );

                Allocate_Matrix( &workspace->L, Hptr->n, Hptr->n_max, Hptr->m );
                Allocate_Matrix( &workspace->U, Hptr->n, Hptr->n_max, Hptr->m );
            }
            break;

        case ILUTP_PC:
        case FG_ILUT_PC:
            /* replace zeros on diagonal with non-zero values */
            Hptr->val[Hptr->start[system->N_cm] - 1] = 1.0;

            Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );

            /* put zeros back */
            Hptr->val[Hptr->start[system->N_cm] - 1] = 0.0;

            if ( workspace->L.allocated == FALSE )
            {
                /* safest storage estimate is ILU(0) (same as
                 * lower triangular portion of H), could improve later */
                Allocate_Matrix( &workspace->L, Hptr->n, Hptr->n_max, Hptr->m );
                Allocate_Matrix( &workspace->U, Hptr->n, Hptr->n_max, Hptr->m );
            }
            else if ( workspace->L.m < Hptr->m || realloc == TRUE )
            {
                Deallocate_Matrix( &workspace->L );
                Deallocate_Matrix( &workspace->U );

                /* safest storage estimate is ILU(0) (same as
                 * lower triangular portion of H), could improve later */
                Allocate_Matrix( &workspace->L, Hptr->n, Hptr->n_max, Hptr->m );
                Allocate_Matrix( &workspace->U, Hptr->n, Hptr->n_max, Hptr->m );
            }
            break;

        case SAI_PC:
            setup_sparse_approx_inverse( Hptr, &workspace->H_full,
                    &workspace->H_spar_patt, &workspace->H_spar_patt_full,
                    &workspace->H_app_inv, control->cm_solver_pre_comp_sai_thres,
                    realloc );
            break;

        default:
            fprintf( stderr, "[ERROR] Unrecognized preconditioner computation method (%d). Terminating...\n",
                   control->cm_solver_pre_comp_type );
            exit( INVALID_INPUT );
            break;
    }
}


/* Setup routines before computing the preconditioner for ACKS2
 */
static void Setup_Preconditioner_ACKS2( const reax_system * const system,
        const control_params * const control, simulation_data * const data,
        static_storage * const workspace, int realloc )
{
    real time;
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
    time = Get_Time( );
    Sort_Matrix_Rows( &workspace->H );
    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Sort_Matrix_Rows( &workspace->H_sp );
    }
    data->timing.cm_sort_mat_rows += Get_Timing_Info( time );

    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case JACOBI_PC:
            if ( workspace->Hdia_inv == NULL )
            {
                workspace->Hdia_inv = scalloc( Hptr->n_max, sizeof( real ),
                        "Setup_Preconditioner_ACKS2::workspace->Hdiv_inv" );
            }
            else if ( realloc == TRUE )
            {
                workspace->Hdia_inv = srealloc( workspace->Hdia_inv,
                        sizeof( real ) * Hptr->n_max,
                        "Setup_Preconditioner_ACKS2::workspace->Hdiv_inv" );
            }
            break;

        case ICHOLT_PC:
            fprintf( stderr, "[ERROR] ICHOLT is not supported for indefinite, symmetric matrices of ACKS2. Terminating...\n" );
            exit( INVALID_INPUT );
            break;

        case ILUT_PC:
            if ( control->cm_solver_pre_comp_droptol > 0.0 )
            {
                /* replace zeros on diagonal with non-zero values */
                Hptr->val[Hptr->start[system->N_cm - 1] - 1] = 1.0;
                Hptr->val[Hptr->start[system->N_cm] - 1] = 1.0;

                Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );

                /* put zeros back */
                Hptr->val[Hptr->start[system->N_cm - 1] - 1] = 0.0;
                Hptr->val[Hptr->start[system->N_cm] - 1] = 0.0;
            }

            if ( workspace->L.allocated == FALSE )
            {
                Allocate_Matrix( &workspace->L, Hptr->n, Hptr->n_max, Hptr->m );
                Allocate_Matrix( &workspace->U, Hptr->n, Hptr->n_max, Hptr->m );
            }
            else if ( workspace->L.m < Hptr->m || realloc == TRUE )
            {
                Deallocate_Matrix( &workspace->L );
                Deallocate_Matrix( &workspace->U );

                Allocate_Matrix( &workspace->L, Hptr->n, Hptr->n_max, Hptr->m );
                Allocate_Matrix( &workspace->U, Hptr->n, Hptr->n_max, Hptr->m );
            }
            break;

        case ILUTP_PC:
        case FG_ILUT_PC:
            /* replace zeros on diagonal with non-zero values */
            Hptr->val[Hptr->start[system->N_cm - 1] - 1] = 1.0;
            Hptr->val[Hptr->start[system->N_cm] - 1] = 1.0;

            Calculate_Droptol( Hptr, workspace->droptol, control->cm_solver_pre_comp_droptol );

            /* put zeros back */
            Hptr->val[Hptr->start[system->N_cm - 1] - 1] = 0.0;
            Hptr->val[Hptr->start[system->N_cm] - 1] = 0.0;

            if ( workspace->L.allocated == FALSE )
            {
                /* safest storage estimate is ILU(0) (same as
                 * lower triangular portion of H), could improve later */
                Allocate_Matrix( &workspace->L, Hptr->n, Hptr->n_max, Hptr->m );
                Allocate_Matrix( &workspace->U, Hptr->n, Hptr->n_max, Hptr->m );
            }
            else if ( workspace->L.m < Hptr->m || realloc == TRUE )
            {
                Deallocate_Matrix( &workspace->L );
                Deallocate_Matrix( &workspace->U );

                /* factors have sparsity pattern as H */
                Allocate_Matrix( &workspace->L, Hptr->n, Hptr->n_max, Hptr->m );
                Allocate_Matrix( &workspace->U, Hptr->n, Hptr->n_max, Hptr->m );
            }
            break;

        case SAI_PC:
            setup_sparse_approx_inverse( Hptr, &workspace->H_full,
                    &workspace->H_spar_patt, &workspace->H_spar_patt_full,
                    &workspace->H_app_inv, control->cm_solver_pre_comp_sai_thres,
                    realloc );
            break;

        default:
            fprintf( stderr, "[ERROR] Unrecognized preconditioner computation method (%d). Terminating...\n",
                   control->cm_solver_pre_comp_type );
            exit( INVALID_INPUT );
            break;
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


/* Get atomic charge q for ACKS2 method
 */
static void Calculate_Charges_ACKS2( const reax_system * const system,
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
        const output_controls * const out_control, int realloc )
{
    int iters, refactor;
    real time;
    /* Finetune the model after reaching a certain step */
    /* TODO: define control variables to control finetuning */
    #if defined(HAVE_TENSORFLOW)
    if (control->cm_init_guess_type == TF_FROZEN_MODEL_LSTM 
        && control->cm_init_guess_training == 1 && 
        data->step % control->cm_init_guess_training_step == 0 && data->step != 0 && data->step < 501) {
       // #pragma omp single{
        //train_and_save(workspace, system, control);
       // }
    }
    #endif

    refactor = is_refactoring_step( control, data );

    if ( refactor == TRUE )    
    {
        Setup_Preconditioner_QEq( system, control, data, workspace, realloc );

        Compute_Preconditioner_QEq( system, control, data, workspace, realloc );
    }
    // Time init. guess method
    time = Get_Time( );
    switch ( control->cm_init_guess_type )
    {
    case SPLINE:
        Spline_Extrapolate_Charges_QEq( system, control, data, workspace );
        break;

    case TF_FROZEN_MODEL_LSTM:
#if defined(HAVE_TENSORFLOW)
        if ( data->step < control->cm_init_guess_win_size + 5 )
        {
            Spline_Extrapolate_Charges_QEq( system, control, data, workspace );
        }
        else
        {
            // TODO: This part is not tested
            if (data->step == 0)
            {
                fprintf(stdout, "[WARNING] Tensorflow based initial guess method is not tested for QEQ!\n");
            }
            Spline_Extrapolate_Charges_QEq( system, control, data, workspace );
            //Predict_Charges_TF_LSTM( system, control, data, workspace );
        }
#else
        fprintf( stderr, "[ERROR] Tensorflow support disabled. Re-compile to enable. Terminating...\n" );
        exit( INVALID_INPUT );
#endif
        break;

    default:
        fprintf( stderr, "[ERROR] Unrecognized solver initial guess type (%d). Terminating...\n",
              control->cm_init_guess_type );
        exit( INVALID_INPUT );
        break;
    }
    data->timing.cm_prediction_overall = Get_Timing_Info( time );

#if defined(QMMM)
    fprintf( stderr, "[ERROR] QEq charge method is not supported in QM/MM mode. Use EEM instead. Terminating...\n" );
    exit( INVALID_INPUT );
#endif

    switch ( control->cm_solver_type )
    {
    case GMRES_S:
        iters = GMRES( workspace, control, data, &workspace->H,
                workspace->b_s, control->cm_solver_q_err, workspace->s[0], refactor );
        iters += GMRES( workspace, control, data, &workspace->H,
                workspace->b_t, control->cm_solver_q_err, workspace->t[0], FALSE );
        break;

    case GMRES_H_S:
        iters = GMRES_HouseHolder( workspace, control, data, &workspace->H,
                workspace->b_s, control->cm_solver_q_err, workspace->s[0], refactor );
        iters += GMRES_HouseHolder( workspace, control, data, &workspace->H,
                workspace->b_t, control->cm_solver_q_err, workspace->t[0], FALSE );
        break;

    case CG_S:
        iters = CG( workspace, control, data, &workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], refactor ) + 1;
        iters += CG( workspace, control, data, &workspace->H, workspace->b_t, control->cm_solver_q_err,
                workspace->t[0], FALSE ) + 1;
        break;

    case SDM_S:
        iters = SDM( workspace, control, data, &workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], refactor ) + 1;
        iters += SDM( workspace, control, data, &workspace->H, workspace->b_t, control->cm_solver_q_err,
                      workspace->t[0], FALSE ) + 1;
        break;

    case BiCGStab_S:
        iters = BiCGStab( workspace, control, data, &workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], refactor ) + 1;
        iters += BiCGStab( workspace, control, data, &workspace->H, workspace->b_t, control->cm_solver_q_err,
                workspace->t[0], FALSE ) + 1;
        break;

    default:
        fprintf( stderr, "[ERROR] Unrecognized solver selection (%d). Terminating...\n",
              control->cm_solver_type );
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
        const output_controls * const out_control, int realloc )
{
    int iters, refactor;
    real time;
    /* Finetune the model after reaching a certain step */
    /* TODO: define control variables to control finetuning */
    #if defined(HAVE_TENSORFLOW)
    if (control->cm_init_guess_type == TF_FROZEN_MODEL_LSTM 
        && control->cm_init_guess_training == 1 && 
        data->step % control->cm_init_guess_training_step == 0 && data->step != 0 && data->step < 501) {
       // #pragma omp single{
        //train_and_save_EE(workspace, system, control);
       // }
    }
    #endif

    refactor = is_refactoring_step( control, data );

    if ( refactor == TRUE )
    {
        Setup_Preconditioner_EE( system, control, data, workspace, realloc );

        Compute_Preconditioner_EE( system, control, data, workspace, realloc );
    }
    // Time init. guess method
    time = Get_Time( );
    switch ( control->cm_init_guess_type )
    {
    case SPLINE:
        Spline_Extrapolate_Charges_EE( system, control, data, workspace );
        break;

    case TF_FROZEN_MODEL_LSTM:
#if defined(HAVE_TENSORFLOW)
        if ( data->step < control->cm_init_guess_win_size + 5 )
        {
            Spline_Extrapolate_Charges_EE( system, control, data, workspace );
        }
        else
        {
            // Preemptively call the spline method in case some predictions are skipped by the following method
            // (Charge predictions can be made by the spline method solely and LSTM can predict the rest)
            // TODO: Remove this call after everything is finalized
            Spline_Extrapolate_Charges_EE( system, control, data, workspace );
            //Predict_Charges_TF_LSTM_EE( system, control, data, workspace );
            Predict_Charges_ELM_EE( system, control, data, workspace );
        }
#else
        fprintf( stderr, "[ERROR] Tensorflow support disabled. Re-compile to enable. Terminating...\n" );
        exit( INVALID_INPUT );
#endif
        break;

    default:
        fprintf( stderr, "[ERROR] Unrecognized solver initial guess type (%d). Terminating...\n",
              control->cm_init_guess_type );
        exit( INVALID_INPUT );
        break;
    }
    data->timing.cm_prediction_overall = Get_Timing_Info( time );

#if defined(QMMM)
    for ( int i = 0; i < system->N_qm; ++i )
    {
        workspace->mask_qmmm[i] = system->atoms[i].qmmm_mask;
    }
    for ( int i = system->N_qm; i < system->N; ++i )
    {
        workspace->s[0][i] = system->atoms[i].q_init;
        workspace->mask_qmmm[i] = system->atoms[i].qmmm_mask;
    }
    workspace->mask_qmmm[system->N_cm - 1] = 1;
#endif

    switch ( control->cm_solver_type )
    {
    case GMRES_S:
        iters = GMRES( workspace, control, data, &workspace->H,
                workspace->b_s, control->cm_solver_q_err, workspace->s[0], refactor );
        break;

    case GMRES_H_S:
        iters = GMRES_HouseHolder( workspace, control, data, &workspace->H,
                workspace->b_s, control->cm_solver_q_err, workspace->s[0], refactor );
        break;

    case CG_S:
        iters = CG( workspace, control, data, &workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], refactor ) + 1;
        break;

    case SDM_S:
        iters = SDM( workspace, control, data, &workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], refactor ) + 1;
        break;

    case BiCGStab_S:
        iters = BiCGStab( workspace, control, data, &workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], refactor ) + 1;
        break;

    default:
        fprintf( stderr, "[ERROR] Unrecognized solver selection (%d). Terminating...\n",
              control->cm_solver_type );
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
        const output_controls * const out_control, int realloc )
{
    int iters, refactor;
    real time;

    /* Finetune the model after reaching a certain step */
    /* TODO: define control variables to control finetuning */
    #if defined(HAVE_TENSORFLOW)
    if (control->cm_init_guess_type == TF_FROZEN_MODEL_LSTM 
        && control->cm_init_guess_training == 1 && 
        data->step % control->cm_init_guess_training_step == 0 && data->step != 0 && data->step < 501) {
       // #pragma omp single{
        //train_and_save_ACKS2(workspace, system, control);
       // }
    }
    #endif

    refactor = is_refactoring_step( control, data );

    if ( refactor == TRUE )
    {
        Setup_Preconditioner_ACKS2( system, control, data, workspace, realloc );

        Compute_Preconditioner_ACKS2( system, control, data, workspace, realloc );
    }

//   Print_Linear_System( system, control, workspace, data->step );
    // Time init. guess method
    time = Get_Time( );

    switch ( control->cm_init_guess_type )
    {
    case SPLINE:
        Spline_Extrapolate_Charges_EE( system, control, data, workspace );
        break;

    case TF_FROZEN_MODEL_LSTM:
#if defined(HAVE_TENSORFLOW)
        if ( data->step < control->cm_init_guess_win_size + 5 )
        {
            Spline_Extrapolate_Charges_EE( system, control, data, workspace );
        }
        else
        {
            // Preemptively call the spline method in case some predictions are skipped by the following method
            // (Charge predictions can be made by the spline method solely and LSTM can predict the rest)
            // TODO: Remove this call after everything is finalized
            Spline_Extrapolate_Charges_EE( system, control, data, workspace );
            //Predict_Charges_TF_LSTM_ACKS2( system, control, data, workspace );
            Predict_Charges_ELM_ACKS2( system, control, data, workspace );
        }
#else
        fprintf( stderr, "[ERROR] Tensorflow support disabled. Re-compile to enable. Terminating...\n" );
        exit( INVALID_INPUT );
#endif
        break;

    default:
        fprintf( stderr, "[ERROR] Unrecognized solver initial guess type (%d). Terminating...\n",
              control->cm_init_guess_type );
        exit( INVALID_INPUT );
        break;
    }

    data->timing.cm_prediction_overall = Get_Timing_Info( time );

#if defined(DEBUG_FOCUS)
#define SIZE (200)
    char fname[SIZE];
    FILE * fp;

    if ( data->step % 10 == 0 )
    {
        snprintf( fname, SIZE, "s_%d_%s.out", data->step, control->sim_name );
        fp = sfopen( fname, "w" );
        Vector_Print( fp, NULL, workspace->s[0], system->N_cm );
        sfclose( fp, "ACKS2::fp" );
    }
#undef SIZE
#endif

#if defined(QMMM)
    /* TODO: further testing needed for QM/MM mode with ACKS2 */
    for ( int i = 0; i < system->N_qm; ++i )
    {
        workspace->mask_qmmm[i] = system->atoms[i].qmmm_mask;
    }
    for ( int i = system->N_qm; i < system->N; ++i )
    {
        workspace->s[0][i] = system->atoms[i].q_init;
        workspace->mask_qmmm[i] = system->atoms[i].qmmm_mask;
    }
    for ( int i = system->N; i < 2 * system->N; ++i )
    {
        workspace->mask_qmmm[i] = system->atoms[i - system->N].qmmm_mask;
    }
    workspace->mask_qmmm[2 * system->N_cm] = 1;
    workspace->mask_qmmm[2 * system->N_cm + 1] = 1;
#endif

    switch ( control->cm_solver_type )
    {
    case GMRES_S:
        iters = GMRES( workspace, control, data, &workspace->H,
                workspace->b_s, control->cm_solver_q_err, workspace->s[0], refactor );
        break;

    case GMRES_H_S:
        iters = GMRES_HouseHolder( workspace, control, data, &workspace->H,
                workspace->b_s, control->cm_solver_q_err, workspace->s[0], refactor );
        break;

    case CG_S:
        iters = CG( workspace, control, data, &workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], refactor ) + 1;
        break;

    case SDM_S:
        iters = SDM( workspace, control, data, &workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], refactor ) + 1;
        break;

    case BiCGStab_S:
        iters = BiCGStab( workspace, control, data, &workspace->H, workspace->b_s, control->cm_solver_q_err,
                workspace->s[0], refactor ) + 1;
        break;

    default:
        fprintf( stderr, "[ERROR] Unrecognized solver selection (%d). Terminating...\n",
              control->cm_solver_type );
        exit( INVALID_INPUT );
        break;
    }

    data->timing.cm_solver_iters += iters;

    Calculate_Charges_ACKS2( system, workspace );
}


void Compute_Charges( reax_system * const system, control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        const output_controls * const out_control, int realloc )
{
#if defined(DEBUG_FOCUS)
#define SIZE (200)
    char fname[SIZE];
    FILE * fp;

    if ( data->step % 10 == 0 )
    {
        snprintf( fname, SIZE, "H_%d_%s.out", data->step, control->sim_name );
        Print_Sparse_Matrix2( workspace->H, fname, NULL );
//        Print_Sparse_Matrix_Binary( workspace->H, fname );

        snprintf( fname, SIZE, "b_s_%d_%s.out", data->step, control->sim_name );
        fp = sfopen( fname, "w" );
        Vector_Print( fp, NULL, workspace->b_s, system->N_cm );
        sfclose( fp, "Compute_Charges::fp" );

//        snprintf( fname, SIZE, "b_t_%d_%s.out", data->step, control->sim_name );
//        fp = sfopen( fname, "w" );
//        Vector_Print( fp, NULL, workspace->b_t, system->N_cm );
//        sfclose( fp, "Compute_Charges::fp" );
    }
#undef SIZE
#endif

    switch ( control->charge_method )
    {
    case QEQ_CM:
        QEq( system, control, data, workspace, out_control, realloc );
        break;

    case EE_CM:
        EE( system, control, data, workspace, out_control, realloc );
        break;

    case ACKS2_CM:
        ACKS2( system, control, data, workspace, out_control, realloc );
        break;

    default:
        fprintf( stderr, "[ERROR] Invalid charge method. Terminating...\n" );
        exit( UNKNOWN_OPTION );
        break;
    }
}
