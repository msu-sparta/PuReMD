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

#include "reax_types.h"

#include "charges.h"

#include "allocate.h"
#include "basic_comm.h"
#include "io_tools.h"
#include "lin_alg.h"
#include "tool_box.h"


int compare_matrix_entry( const void *v1, const void *v2 )
{
    return ((sparse_matrix_entry *) v1)->j - ((sparse_matrix_entry *) v2)->j;
}


void Sort_Matrix_Rows( sparse_matrix *A )
{
    int i, si, ei;

    for ( i = 0; i < A->n; ++i )
    {
        si = A->start[i];
        ei = A->end[i];
        qsort( &A->entries[si], ei - si,
                sizeof(sparse_matrix_entry), compare_matrix_entry );
    }
}


static void Init_Linear_Solver( reax_system *system, simulation_data *data,
        control_params *control, storage *workspace, mpi_datatypes *mpi_data )
{
    int i;
    reax_atom *atom;

    /* reset size of locally owned portion of charge matrix */
    switch ( control->charge_method )
    {
        case QEQ_CM:
            system->n_cm = system->N;
            break;
        case EE_CM:
            system->n_cm = system->N + 1;
            break;
        case ACKS2_CM:
            system->n_cm = 2 * system->N + 2;
            break;
        default:
            fprintf( stderr, "[ERROR] Unknown charge method type. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }

    /* initialize solution vectors for linear solves in charge method */
    switch ( control->charge_method )
    {
        case QEQ_CM:
            for ( i = 0; i < system->n; ++i )
            {
                atom = &system->my_atoms[i];

                workspace->b_s[i] = -1.0 * system->reax_param.sbp[ atom->type ].chi;
                workspace->b_t[i] = -1.0;
                workspace->b[i][0] = -1.0 * system->reax_param.sbp[ atom->type ].chi;
                workspace->b[i][1] = -1.0;
            }
            break;

        case EE_CM:
            for ( i = 0; i < system->n; ++i )
            {
                atom = &system->my_atoms[i];

                workspace->b_s[i] = -1.0 * system->reax_param.sbp[ atom->type ].chi;

                //TODO: check if unused (redundant)
                workspace->b[i][0] = -1.0 * system->reax_param.sbp[ atom->type ].chi;
            }

            if ( system->my_rank == 0 )
            {
                workspace->b_s[system->n] = control->cm_q_net;
                workspace->b[system->n][0] = control->cm_q_net;
            }
            break;

        case ACKS2_CM:
            for ( i = 0; i < system->n; ++i )
            {
                atom = &system->my_atoms[i];

                workspace->b_s[i] = -1.0 * system->reax_param.sbp[ atom->type ].chi;

                //TODO: check if unused (redundant)
                workspace->b[i][0] = -1.0 * system->reax_param.sbp[ atom->type ].chi;
            }

            if ( system->my_rank == 0 )
            {
                workspace->b_s[system->n] = control->cm_q_net;
                workspace->b[system->n][0] = control->cm_q_net;
            }

            for ( i = system->n + 1; i < system->n_cm; ++i )
            {
                atom = &system->my_atoms[i];

                workspace->b_s[i] = 0.0;

                //TODO: check if unused (redundant)
                workspace->b[i][0] = 0.0;
            }

            if ( system->my_rank == 0 )
            {
                workspace->b_s[system->n] = control->cm_q_net;
                workspace->b[system->n][0] = control->cm_q_net;
            }
            break;

        default:
            fprintf( stderr, "[ERROR] Unknown charge method type. Terminating...\n" );
            exit( UNKNOWN_OPTION );
            break;
    }
}


static void Extrapolate_Charges_QEq( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, storage * const workspace )
{
    int i;
    real s_tmp, t_tmp;

    /* spline extrapolation for s & t */
    //TODO: good candidate for vectorization, avoid moving data with head pointer and circular buffer
    for ( i = 0; i < system->n_cm; ++i )
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
            s_tmp = 4.0 * (system->my_atoms[i].s[0] + system->my_atoms[i].s[2]) -
                    (6.0 * system->my_atoms[i].s[1] + system->my_atoms[i].s[3]);
        }
        else
        {
            s_tmp = 0.0;
        }

        /* x is used as initial guess to solver */
        workspace->x[i][0] = s_tmp;
    }

    for ( i = 0; i < system->n_cm; ++i )
    {
        /* no extrapolation, previous solution as initial guess */
        if ( control->cm_init_guess_extrap2 == 0 )
        {
            t_tmp = system->my_atoms[i].t[0];
        }
        /* linear */
        else if ( control->cm_init_guess_extrap2 == 1 )
        {
            t_tmp = 2.0 * system->my_atoms[i].t[0] - system->my_atoms[i].t[1];
        }
        /* quadratic */
        else if ( control->cm_init_guess_extrap2 == 2 )
        {
            t_tmp = system->my_atoms[i].t[2] + 3.0 * (system->my_atoms[i].t[0] - system->my_atoms[i].t[1]);
        }
        /* cubic */
        else if ( control->cm_init_guess_extrap2 == 3 )
        {
            t_tmp = 4.0 * (system->my_atoms[i].t[0] + system->my_atoms[i].t[2]) -
                (6.0 * system->my_atoms[i].t[1] + system->my_atoms[i].t[3]);
        }
        else
        {
            t_tmp = 0.0;
        }

        /* x is used as initial guess to solver */
        workspace->x[i][1] = t_tmp;
    }
}


static void Extrapolate_Charges_QEq_Part2( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, storage * const workspace )
{
    int i;

    /* spline extrapolation for s & t */
    //TODO: good candidate for vectorization, avoid moving data with head pointer and circular buffer
    for ( i = 0; i < system->n_cm; ++i )
    {
        system->my_atoms[i].s[3] = system->my_atoms[i].s[2];
        system->my_atoms[i].s[2] = system->my_atoms[i].s[1];
        system->my_atoms[i].s[1] = system->my_atoms[i].s[0];
        system->my_atoms[i].s[0] = workspace->x[i][0];

        system->my_atoms[i].t[3] = system->my_atoms[i].t[2];
        system->my_atoms[i].t[2] = system->my_atoms[i].t[1];
        system->my_atoms[i].t[1] = system->my_atoms[i].t[0];
        system->my_atoms[i].t[0] = workspace->x[i][1];
    }
}


static void Extrapolate_Charges_EE( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, storage * const workspace )
{
    int i;
    real s_tmp;

    /* spline extrapolation for s */
    //TODO: good candidate for vectorization, avoid moving data with head pointer and circular buffer
    for ( i = 0; i < system->n_cm; ++i )
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
            s_tmp = 4.0 * (system->my_atoms[i].s[0] + system->my_atoms[i].s[2]) -
                    (6.0 * system->my_atoms[i].s[1] + system->my_atoms[i].s[3]);
        }
        else
        {
            s_tmp = 0.0;
        }

        workspace->x[i][0] = s_tmp;
    }
}


static void Extrapolate_Charges_EE_Part2( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, storage * const workspace )
{
    int i;

    /* spline extrapolation for s */
    //TODO: good candidate for vectorization, avoid moving data with head pointer and circular buffer
    for ( i = 0; i < system->n_cm; ++i )
    {
        system->my_atoms[i].s[4] = system->my_atoms[i].s[3];
        system->my_atoms[i].s[3] = system->my_atoms[i].s[2];
        system->my_atoms[i].s[2] = system->my_atoms[i].s[1];
        system->my_atoms[i].s[1] = system->my_atoms[i].s[0];
        system->my_atoms[i].s[0] = workspace->x[i][0];
    }
}


/* Compute preconditioner for QEq
 */
static void Compute_Preconditioner_QEq( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, storage * const workspace )
{
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = &workspace->H_sp;
    }
    else
    {
        Hptr = &workspace->H;
    }

    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case DIAG_PC:
            data->timing.cm_solver_pre_comp +=
                diag_pre_comp( system, workspace->Hdia_inv );
//                diag_pre_comp( Hptr, workspace->Hdia_inv );
            break;

        case ICHOLT_PC:
        case ILU_PAR_PC:
        case ILUT_PAR_PC:
        case ILU_SUPERLU_MT_PC:
            fprintf( stderr, "[ERROR] Unsupported preconditioner computation method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;

        case SAI_PC:
#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
            //TODO: implement
//            data->timing.cm_solver_pre_comp +=
//                sparse_approx_inverse( workspace->H_full, workspace->H_spar_patt_full,
//                        &workspace->H_app_inv );
#else
            fprintf( stderr, "[ERROR] LAPACKE support disabled. Re-compile before enabling. Terminating...\n" );
            exit( INVALID_INPUT );
#endif
            break;

        default:
            fprintf( stderr, "[ERROR] Unrecognized preconditioner computation method. Terminating...\n" );
            exit( UNKNOWN_OPTION );
            break;
    }
}


/* Compute preconditioner for EE
 */
static void Compute_Preconditioner_EE( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, storage * const workspace )
{
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = &workspace->H_sp;
    }
    else
    {
        Hptr = &workspace->H;
    }

    if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
            control->cm_solver_pre_comp_type == ILU_PAR_PC ||
            control->cm_solver_pre_comp_type == ILUT_PAR_PC )
    {
        Hptr->entries[Hptr->start[system->N + 1] - 1].val = 1.0;
    }
    
    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case DIAG_PC:
            data->timing.cm_solver_pre_comp +=
                diag_pre_comp( system, workspace->Hdia_inv );
//                diag_pre_comp( Hptr, workspace->Hdia_inv );
            break;

        case ICHOLT_PC:
        case ILU_PAR_PC:
        case ILUT_PAR_PC:
        case ILU_SUPERLU_MT_PC:
            fprintf( stderr, "[ERROR] Unsupported preconditioner computation method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;

        case SAI_PC:
#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
            //TODO: implement
//            data->timing.cm_solver_pre_comp +=
//                sparse_approx_inverse( workspace->H_full, workspace->H_spar_patt_full,
//                        &workspace->H_app_inv );
#else
            fprintf( stderr, "[ERROR] LAPACKE support disabled. Re-compile before enabling. Terminating...\n" );
            exit( INVALID_INPUT );
#endif
            break;

        default:
            fprintf( stderr, "[ERROR] Unrecognized preconditioner computation method. Terminating...\n" );
            exit( UNKNOWN_OPTION );
            break;
    }

    if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
            control->cm_solver_pre_comp_type == ILU_PAR_PC ||
            control->cm_solver_pre_comp_type == ILUT_PAR_PC )
    {
        Hptr->entries[Hptr->start[system->N + 1] - 1].val = 0.0;
    }
}


/* Compute preconditioner for ACKS2
 */
static void Compute_Preconditioner_ACKS2( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, storage * const workspace )
{
    sparse_matrix *Hptr;

    if ( control->cm_domain_sparsify_enabled == TRUE )
    {
        Hptr = &workspace->H_sp;
    }
    else
    {
        Hptr = &workspace->H;
    }

    if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
            control->cm_solver_pre_comp_type == ILU_PAR_PC ||
            control->cm_solver_pre_comp_type == ILUT_PAR_PC )
    {
        Hptr->entries[Hptr->start[system->N + 1] - 1].val = 1.0;
        Hptr->entries[Hptr->start[system->n_cm] - 1].val = 1.0;
    }
    
    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case DIAG_PC:
            data->timing.cm_solver_pre_comp +=
                diag_pre_comp( system, workspace->Hdia_inv );
//                diag_pre_comp( Hptr, workspace->Hdia_inv );
            break;

        case ICHOLT_PC:
        case ILU_PAR_PC:
        case ILUT_PAR_PC:
        case ILU_SUPERLU_MT_PC:
            fprintf( stderr, "[ERROR] Unsupported preconditioner computation method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;

        case SAI_PC:
#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
            //TODO: implement
//            data->timing.cm_solver_pre_comp +=
//                sparse_approx_inverse( workspace->H_full, workspace->H_spar_patt_full,
//                        &workspace->H_app_inv );
#else
            fprintf( stderr, "[ERROR] LAPACKE support disabled. Re-compile before enabling. Terminating...\n" );
            exit( INVALID_INPUT );
#endif
            break;

        default:
            fprintf( stderr, "[ERROR] Unrecognized preconditioner computation method. Terminating...\n" );
            exit( UNKNOWN_OPTION );
            break;
    }

    if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
            control->cm_solver_pre_comp_type == ILU_PAR_PC ||
            control->cm_solver_pre_comp_type == ILUT_PAR_PC )
    {
        Hptr->entries[Hptr->start[system->N + 1] - 1].val = 0.0;
        Hptr->entries[Hptr->start[system->n_cm] - 1].val = 0.0;
    }
}


static void Setup_Preconditioner_QEq( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, storage * const workspace )
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

        case DIAG_PC:
            if ( workspace->Hdia_inv == NULL )
            {
//                workspace->Hdia_inv = scalloc( Hptr->n, sizeof( real ),
//                        "Setup_Preconditioner_QEq::workspace->Hdiv_inv" );
            }
            break;

        case ICHOLT_PC:
        case ILU_PAR_PC:
        case ILUT_PAR_PC:
        case ILU_SUPERLU_MT_PC:
            fprintf( stderr, "[ERROR] Unsupported preconditioner computation method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;

        case SAI_PC:
            //TODO: implement
//            setup_sparse_approx_inverse( Hptr, &workspace->H_full, &workspace->H_spar_patt,
//                    &workspace->H_spar_patt_full, &workspace->H_app_inv,
//                    control->cm_solver_pre_comp_sai_thres );
            break;

        default:
            fprintf( stderr, "[ERROR] Unrecognized preconditioner computation method. Terminating...\n" );
            exit( UNKNOWN_OPTION );
            break;
    }
}


/* Setup routines before computing the preconditioner for EE
 */
static void Setup_Preconditioner_EE( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, storage * const workspace )
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

    if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
            control->cm_solver_pre_comp_type == ILU_PAR_PC ||
            control->cm_solver_pre_comp_type == ILUT_PAR_PC )
    {
        Hptr->entries[Hptr->start[system->N + 1] - 1].val = 1.0;
    }

    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case DIAG_PC:
            if ( workspace->Hdia_inv == NULL )
            {
//                workspace->Hdia_inv = scalloc( system->n_cm, sizeof( real ),
//                        "Setup_Preconditioner_QEq::workspace->Hdiv_inv" );
            }
            break;

        case ICHOLT_PC:
        case ILU_PAR_PC:
        case ILUT_PAR_PC:
        case ILU_SUPERLU_MT_PC:
            fprintf( stderr, "[ERROR] Unsupported preconditioner computation method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;

        case SAI_PC:
            //TODO: implement
//            setup_sparse_approx_inverse( Hptr, &workspace->H_full, &workspace->H_spar_patt,
//                    &workspace->H_spar_patt_full, &workspace->H_app_inv,
//                    control->cm_solver_pre_comp_sai_thres );
            break;

        default:
            fprintf( stderr, "[ERROR] Unrecognized preconditioner computation method. Terminating...\n" );
            exit( UNKNOWN_OPTION );
            break;
    }

    if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
            control->cm_solver_pre_comp_type == ILU_PAR_PC ||
            control->cm_solver_pre_comp_type == ILUT_PAR_PC )
    {
        Hptr->entries[Hptr->start[system->N + 1] - 1].val = 0.0;
    }
}


/* Setup routines before computing the preconditioner for ACKS2
 */
static void Setup_Preconditioner_ACKS2( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, storage * const workspace )
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

    if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
            control->cm_solver_pre_comp_type == ILU_PAR_PC ||
            control->cm_solver_pre_comp_type == ILUT_PAR_PC )
    {
        Hptr->entries[Hptr->start[system->N + 1] - 1].val = 1.0;
        Hptr->entries[Hptr->start[system->n_cm] - 1].val = 1.0;
    }

    switch ( control->cm_solver_pre_comp_type )
    {
        case NONE_PC:
            break;

        case DIAG_PC:
            if ( workspace->Hdia_inv == NULL )
            {
//                workspace->Hdia_inv = scalloc( Hptr->n, sizeof( real ),
//                        "Setup_Preconditioner_QEq::workspace->Hdiv_inv" );
            }
            break;

        case ICHOLT_PC:
        case ILU_PAR_PC:
        case ILUT_PAR_PC:
        case ILU_SUPERLU_MT_PC:
            fprintf( stderr, "[ERROR] Unsupported preconditioner computation method. Terminating...\n" );
            exit( INVALID_INPUT );
            break;

        case SAI_PC:
            //TODO: implement
//            setup_sparse_approx_inverse( Hptr, &workspace->H_full, &workspace->H_spar_patt,
//                    &workspace->H_spar_patt_full, &workspace->H_app_inv,
//                    control->cm_solver_pre_comp_sai_thres );
            break;

        default:
            fprintf( stderr, "[ERROR] Unrecognized preconditioner computation method. Terminating...\n" );
            exit( UNKNOWN_OPTION );
            break;
    }

    if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
            control->cm_solver_pre_comp_type == ILU_PAR_PC ||
            control->cm_solver_pre_comp_type == ILUT_PAR_PC )
    {
        Hptr->entries[Hptr->start[system->N + 1] - 1].val = 0.0;
        Hptr->entries[Hptr->start[system->n_cm] - 1].val = 0.0;
    }
}


/* Combine ficticious charges s and t to get atomic charge q for QEq method
 */
static void Calculate_Charges_QEq( const reax_system * const system,
        storage * const workspace, const mpi_datatypes * const mpi_data )
{
    int i;
    real u;
    rvec2 my_sum, all_sum;
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
        system->my_atoms[i].q  = workspace->x[i][0] - u * workspace->x[i][1];
        q[i] = system->my_atoms[i].q;
    }

    Dist( system, mpi_data, q, REAL_PTR_TYPE, MPI_DOUBLE );

    for ( i = system->n; i < system->N; ++i )
    {
        system->my_atoms[i].q = q[i];
    }

    sfree( q, "Calculate_Charges_QEq::q" );
}


/* Get atomic charge q for EE method
 */
static void Calculate_Charges_EE( const reax_system * const system,
        storage * const workspace, const mpi_datatypes * const mpi_data )
{
    int i;

    for ( i = 0; i < system->N; ++i )
    {
        system->my_atoms[i].q = workspace->s[i];
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
        simulation_data * const data, storage * const workspace,
        const output_controls * const out_control,
        const mpi_datatypes * const mpi_data )
{
    int iters;

    Init_Linear_Solver( system, data, control, workspace, mpi_data );

    if ( control->cm_solver_pre_comp_refactor > 0 &&
            ((data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) )
        
    {
        Setup_Preconditioner_QEq( system, control, data, workspace );

        Compute_Preconditioner_QEq( system, control, data, workspace );
    }

    Extrapolate_Charges_QEq( system, control, data, workspace );

    switch ( control->cm_solver_type )
    {
    case CG_S:
        iters = dual_CG( system, control, workspace, data, mpi_data,
                &workspace->H, workspace->b, control->cm_solver_q_err,
                workspace->x, (control->cm_solver_pre_comp_refactor > 0
                    && (data->step - data->prev_steps)
                    % control->cm_solver_pre_comp_refactor == 0) ? TRUE : FALSE );
        break;

    case GMRES_S:
    case GMRES_H_S:
    case SDM_S:
    case BiCGStab_S:
    default:
        fprintf( stderr, "[ERROR] Unsupported solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

#if defined(LOG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        data->timing.cm_solver_iters += iters;
    }
#endif

    Calculate_Charges_QEq( system, workspace, mpi_data );

    Extrapolate_Charges_QEq_Part2( system, control, data, workspace );
}


/* Main driver method for EE kernel
 *  1) init / setup routines for preconditioning of linear solver
 *  2) compute preconditioner
 *  3) extrapolate charges
 *  4) perform 1 linear solve
 *  5) compute atomic charges based on output of (4)
 */
static void EE( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        const output_controls * const out_control,
        const mpi_datatypes * const mpi_data )
{
    int iters;

    Init_Linear_Solver( system, data, control, workspace, mpi_data );

    if ( control->cm_solver_pre_comp_refactor > 0 &&
            ((data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) )
    {
        Setup_Preconditioner_EE( system, control, data, workspace );

        Compute_Preconditioner_EE( system, control, data, workspace );
    }

    Extrapolate_Charges_EE( system, control, data, workspace );

    switch ( control->cm_solver_type )
    {
    case GMRES_S:
    case GMRES_H_S:
    case CG_S:
    case SDM_S:
    case BiCGStab_S:
    default:
        fprintf( stderr, "[ERROR] Unrecognized EE solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    data->timing.cm_solver_iters += iters;

    Calculate_Charges_EE( system, workspace, mpi_data );

    Extrapolate_Charges_EE_Part2( system, control, data, workspace );
}


/* Main driver method for ACKS2 kernel
 *  1) init / setup routines for preconditioning of linear solver
 *  2) compute preconditioner
 *  3) extrapolate charges
 *  4) perform 1 linear solve
 *  5) compute atomic charges based on output of (4)
 */
static void ACKS2( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        const output_controls * const out_control,
        const mpi_datatypes * const mpi_data )
{
    int iters;

    Init_Linear_Solver( system, data, control, workspace, mpi_data );

    if ( control->cm_solver_pre_comp_refactor > 0 &&
            ((data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0) )
    {
        Setup_Preconditioner_ACKS2( system, control, data, workspace );

        Compute_Preconditioner_ACKS2( system, control, data, workspace );
    }

    Extrapolate_Charges_EE( system, control, data, workspace );

    switch ( control->cm_solver_type )
    {
    case GMRES_S:
    case GMRES_H_S:
    case CG_S:
    case SDM_S:
    case BiCGStab_S:
    default:
        fprintf( stderr, "[ERROR] Unrecognized ACKS2 solver selection. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    data->timing.cm_solver_iters += iters;

    Calculate_Charges_EE( system, workspace, mpi_data );

    Extrapolate_Charges_EE_Part2( system, control, data, workspace );
}


void Compute_Charges( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        const output_controls * const out_control,
        const mpi_datatypes * const mpi_data )
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
