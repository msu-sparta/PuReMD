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

#include "qEq.h"
#include "allocate.h"
#include "basic_comm.h"
#include "io_tools.h"
#include "linear_solvers.h"
#include "tool_box.h"

int compare_matrix_entry(const void *v1, const void *v2)
{
    return ((sparse_matrix_entry *)v1)->j - ((sparse_matrix_entry *)v2)->j;
}


void Sort_Matrix_Rows( sparse_matrix *A )
{
    int i, si, ei;

    for ( i = 0; i < A->n; ++i )
    {
        si = A->start[i];
        ei = A->end[i];
        qsort( &(A->entries[si]), ei - si,
                sizeof(sparse_matrix_entry), compare_matrix_entry );
    }
}


void Calculate_Droptol( sparse_matrix *A, real *droptol, real dtol )
{
    int i, j, k;
    real val;

    /* init droptol to 0 - not necessary for an upper-triangular A */
    for ( i = 0; i < A->n; ++i )
        droptol[i] = 0;

    /* calculate sqaure of the norm of each row */
    for ( i = 0; i < A->n; ++i )
    {
        val = A->entries[A->start[i]].val; // diagonal entry
        droptol[i] += val * val;
        // only within my block
        for ( k = A->start[i] + 1; A->entries[k].j < A->n; ++k )
        {
            j = A->entries[k].j;
            val = A->entries[k].val;

            droptol[i] += val * val;
            droptol[j] += val * val;
        }
    }

    /* calculate local droptol for each row */
    //fprintf( stderr, "droptol: " );
    for ( i = 0; i < A->n; ++i )
    {
        droptol[i] = sqrt( droptol[i] ) * dtol;
        //fprintf( stderr, "%f\n", droptol[i] );
    }
    //fprintf( stderr, "\n" );
}


int Estimate_LU_Fill( sparse_matrix *A, real *droptol )
{
    int i, j, pj;
    int fillin;
    real val;

    fillin = 0;
    for ( i = 0; i < A->n; ++i )
        for ( pj = A->start[i] + 1; A->entries[pj].j < A->n; ++pj )
        {
            j = A->entries[pj].j;
            val = A->entries[pj].val;
            if ( fabs(val) > droptol[i] )
                ++fillin;
        }

    return fillin + A->n;
}


void ICHOLT( sparse_matrix *A, real *droptol,
        sparse_matrix *L, sparse_matrix *U )
{
    sparse_matrix_entry tmp[1000];
    int i, j, pj, k1, k2, tmptop, Utop;
    real val, dval;
    int *Ltop;

    Ltop = (int*) malloc((A->n) * sizeof(int));

    // clear data structures
    Utop = 0;
    tmptop = 0;
    for ( i = 0; i < A->n; ++i )
        U->start[i] = L->start[i] = L->end[i] = Ltop[i] = 0;

    for ( i = A->n - 1; i >= 0; --i )
    {
        U->start[i] = Utop;
        tmptop = 0;

        for ( pj = A->end[i] - 1; A->entries[pj].j >= A->n; --pj ); // skip ghosts
        for ( ; pj > A->start[i]; --pj )
        {
            j = A->entries[pj].j;
            val = A->entries[pj].val;
            fprintf( stderr, "i: %d, j: %d  val=%f ", i, j, val );
            //fprintf( stdout, "%d %d %24.16f\n", 6540-i, 6540-j, val );
            //fprintf( stdout, "%d %d %24.16f\n", 6540-j, 6540-i, val );

            if ( fabs(val) > droptol[i] )
            {
                k1 = tmptop - 1;
                k2 = U->start[j] + 1;
                while ( k1 >= 0 && k2 < U->end[j] )
                {
                    if ( tmp[k1].j < U->entries[k2].j )
                        k1--;
                    else if ( tmp[k1].j > U->entries[k2].j )
                        k2++;
                    else
                        val -= (tmp[k1--].val * U->entries[k2++].val);
                }

                // U matrix is upper triangular
                val /= U->entries[U->start[j]].val;
                fprintf( stderr, " newval=%f", val );
                tmp[tmptop].j = j;
                tmp[tmptop].val = val;
                tmptop++;
            }
            fprintf( stderr, "\n" );
        }
        //fprintf( stderr, "i = %d - tmptop = %d\n", i, tmptop );

        // compute the ith diagonal in U
        dval = A->entries[A->start[i]].val;
        //fprintf( stdout, "%d %d %24.16f\n", 6540-i, 6540-i, dval );
        for ( k1 = 0; k1 < tmptop; ++k1 )
            //if( fabs(tmp[k1].val) > droptol[i] )
            dval -= SQR(tmp[k1].val);
        dval = sqrt(dval);
        // keep the diagonal in any case
        U->entries[Utop].j = i;
        U->entries[Utop].val = dval;
        Utop++;

        fprintf(stderr, "row%d: droptol=%.15f val=%.15f\n", i, droptol[i], dval);
        for ( k1 = tmptop - 1; k1 >= 0; --k1 )
        {
            // apply the dropping rule once again
            if ( fabs(tmp[k1].val) > droptol[i] / dval )
            {
                U->entries[Utop].j = tmp[k1].j;
                U->entries[Utop].val = tmp[k1].val;
                Utop++;
                Ltop[tmp[k1].j]++;
                fprintf( stderr, "%d(%.15f)\n", tmp[k1].j, tmp[k1].val );
            }
        }

        U->end[i] = Utop;
        //fprintf( stderr, "i = %d - Utop = %d\n", i, Utop );
    }
    // print matrix U
#if defined(DEBUG)
    fprintf( stderr, "nnz(U): %d\n", Utop );
    for ( i = 0; i < U->n; ++i )
    {
        fprintf( stderr, "row%d: ", i );
        for ( pj = U->start[i]; pj < U->end[i]; ++pj )
            fprintf( stderr, "%d ", U->entries[pj].j );
        fprintf( stderr, "\n" );
    }
#endif

    // transpose matrix U into L
    L->start[0] = L->end[0] = 0;
    for ( i = 1; i < L->n; ++i )
    {
        L->start[i] = L->end[i] = L->start[i - 1] + Ltop[i - 1] + 1;
        //fprintf( stderr, "i=%d  L->start[i]=%d\n", i, L->start[i] );
    }

    for ( i = 0; i < U->n; ++i )
        for ( pj = U->start[i]; pj < U->end[i]; ++pj )
        {
            j = U->entries[pj].j;
            L->entries[L->end[j]].j = i;
            L->entries[L->end[j]].val = U->entries[pj].val;
            L->end[j]++;
        }

    // print matrix L
#if defined(DEBUG)
    fprintf( stderr, "nnz(L): %d\n", L->end[L->n - 1] );
    for ( i = 0; i < L->n; ++i )
    {
        fprintf( stderr, "row%d: ", i );
        for ( pj = L->start[i]; pj < L->end[i]; ++pj )
            fprintf( stderr, "%d ", L->entries[pj].j );
        fprintf( stderr, "\n" );
    }
#endif
    sfree( Ltop, "Ltop" );
}


void Init_MatVec( reax_system *system, simulation_data *data,
        control_params *control,  storage *workspace,
        mpi_datatypes *mpi_data )
{
    int i; //, fillin;
    reax_atom *atom;

    /*if( (data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0 ||
      workspace->L == NULL ) 
    {
        //Print_Linear_System( system, control, workspace, data->step );
        Sort_Matrix_Rows( workspace->H );
        fprintf( stderr, "H matrix sorted\n" );
        Calculate_Droptol( workspace->H, workspace->droptol, control->cm_solver_pre_comp_droptol );
        fprintf( stderr, "drop tolerances calculated\n" );
        if( workspace->L == NULL ) 
        {
            fillin = Estimate_LU_Fill( workspace->H, workspace->droptol );

            if( Allocate_Matrix( &(workspace->L), workspace->H->cap, fillin, FULL_MATRIX, comm ) == 0 ||
            Allocate_Matrix( &(workspace->U), workspace->H->cap, fillin, FULL_MATRIX, comm ) == 0 ) 
            {
                fprintf( stderr, "not enough memory for LU matrices. terminating.\n" );
                MPI_Abort( mpi_data->world, INSUFFICIENT_MEMORY );
            }

            workspace->L->n = workspace->H->n;
            workspace->U->n = workspace->H->n;
#if defined(DEBUG_FOCUS)
            fprintf( stderr, "p%d: n=%d, fillin = %d\n",
            system->my_rank, workspace->L->n, fillin );
            fprintf( stderr, "p%d: allocated memory: L = U = %ldMB\n",
            system->my_rank,fillin*sizeof(sparse_matrix_entry)/(1024*1024) );
#endif
        }

        ICHOLT( workspace->H, workspace->droptol, workspace->L, workspace->U );
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: icholt finished\n", system->my_rank );
        //sprintf( fname, "%s.L%d.out", control->sim_name, data->step );
        //Print_Sparse_Matrix2( workspace->L, fname );
        //Print_Sparse_Matrix( U );
#endif
    }*/

    //TODO: fill in code for setting up and computing SAI, see sPuReMD code,
    //  and remove diagonal preconditioner computation below (workspace->Hdia_inv)
    //    setup_sparse_approx_inverse( Hptr, &workspace->H_full, &workspace->H_spar_patt,
    //            &workspace->H_spar_patt_full, &workspace->H_app_inv,
    //            control->cm_solver_pre_comp_sai_thres );

    for ( i = 0; i < system->n; ++i )
    {
        atom = &( system->my_atoms[i] );

        /* init pre-conditioner for H and init solution vectors */
        workspace->Hdia_inv[i] = 1. / system->reax_param.sbp[ atom->type ].eta;
        workspace->b_s[i] = -system->reax_param.sbp[ atom->type ].chi;
        workspace->b_t[i] = -1.0;
        workspace->b[i][0] = -system->reax_param.sbp[ atom->type ].chi;
        workspace->b[i][1] = -1.0;

        /* linear extrapolation for s and for t */
        // newQEq: no extrapolation!
        //workspace->s[i] = 2 * atom->s[0] - atom->s[1]; //0;
        //workspace->t[i] = 2 * atom->t[0] - atom->t[1]; //0;
        //workspace->x[i][0] = 2 * atom->s[0] - atom->s[1]; //0;
        //workspace->x[i][1] = 2 * atom->t[0] - atom->t[1]; //0;

        /* quadratic extrapolation for s and t */
        // workspace->s[i] = atom->s[2] + 3 * ( atom->s[0] - atom->s[1] );
        // workspace->t[i] = atom->t[2] + 3 * ( atom->t[0] - atom->t[1] );
        //workspace->x[i][0] = atom->s[2] + 3 * ( atom->s[0] - atom->s[1] );
        workspace->x[i][1] = atom->t[2] + 3 * ( atom->t[0] - atom->t[1] );

        /* cubic extrapolation for s and t */
        workspace->x[i][0] = 4 * (atom->s[0] + atom->s[2]) - (6 * atom->s[1] + atom->s[3]);
        //workspace->x[i][1] = 4*(atom->t[0]+atom->t[2])-(6*atom->t[1]+atom->t[3]);

        // fprintf(stderr, "i=%d s=%f t=%f\n", i, workspace->s[i], workspace->t[i]);
    }
}



void Calculate_Charges( reax_system *system, storage *workspace,
        mpi_datatypes *mpi_data )
{
    int        i, scale;
    real       u;//, s_sum, t_sum;
    rvec2      my_sum, all_sum;
    reax_atom *atom;
    real *q;

    scale = sizeof(real) / sizeof(void);
    q = (real*) malloc(system->N * sizeof(real));

    //s_sum = Parallel_Vector_Acc(workspace->s, system->n, mpi_data->world);
    //t_sum = Parallel_Vector_Acc(workspace->t, system->n, mpi_data->world);
    my_sum[0] = my_sum[1] = 0;
    for ( i = 0; i < system->n; ++i )
    {
        my_sum[0] += workspace->x[i][0];
        my_sum[1] += workspace->x[i][1];
    }
    MPI_Allreduce( &my_sum, &all_sum, 2, MPI_DOUBLE, MPI_SUM, mpi_data->world );

    u = all_sum[0] / all_sum[1];
    for ( i = 0; i < system->n; ++i )
    {
        atom = &( system->my_atoms[i] );

        /* compute charge based on s & t */
        //atom->q = workspace->s[i] - u * workspace->t[i];
        q[i] = atom->q = workspace->x[i][0] - u * workspace->x[i][1];

        /* backup s & t */
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

    Dist( system, mpi_data, q, MPI_DOUBLE, scale, real_packer );
    for ( i = system->n; i < system->N; ++i )
        system->my_atoms[i].q = q[i];

    sfree(q, "q");
}


static void Setup_Preconditioner_QEq( reax_system *system, control_params *control, 
        simulation_data *data, storage *workspace, mpi_datatypes *mpi_data )
{
    real time, t_sort, t_pc, total_sort, total_pc;

    /* sort H needed for SpMV's in linear solver, H or H_sp needed for preconditioning */
    time = MPI_Wtime();
    Sort_Matrix_Rows( workspace->H );
    t_sort = MPI_Wtime() - time;

    t_pc = setup_sparse_approx_inverse( system, data, workspace, mpi_data, workspace->H, &workspace->H_spar_patt, 
            control->nprocs, control->cm_solver_pre_comp_sai_thres );


    MPI_Reduce(&t_sort, &total_sort, 1, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world);
    MPI_Reduce(&t_pc, &total_pc, 1, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world);

    if( system->my_rank == MASTER_NODE )
    {
        data->timing.init_qeq += total_sort / control->nprocs;
        data->timing.cm_solver_pre_comp += total_pc / control->nprocs;
    }
}

static void Compute_Preconditioner_QEq( reax_system *system, control_params *control, 
        simulation_data *data, storage *workspace, mpi_datatypes *mpi_data )
{
    real t_pc, total_pc;
#if defined(HAVE_LAPACKE) || defined(HAVE_LAPACKE_MKL)
    t_pc = sparse_approx_inverse( system, data, workspace, mpi_data,
            workspace->H, workspace->H_spar_patt, &workspace->H_app_inv, control->nprocs );

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


void QEq( reax_system *system, control_params *control, simulation_data *data,
        storage *workspace, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    int j, iters;

    Init_MatVec( system, data, control, workspace, mpi_data );

#if defined(DEBUG)
    fprintf( stderr, "p%d: initialized qEq\n", system->my_rank );
    //Print_Linear_System( system, control, workspace, data->step );
#endif

    if( control->cm_solver_pre_comp_type == SAI_PC )
    {
        if( control->cm_solver_pre_comp_refactor > 0 && ((data->step - data->prev_steps) % control->cm_solver_pre_comp_refactor == 0))
        {
            Setup_Preconditioner_QEq( system, control, data, workspace, mpi_data );

            Compute_Preconditioner_QEq( system, control, data, workspace, mpi_data );
        }
    }
    
    //TODO: used for timing to sync processors going into the linear solve, but remove for production code
    MPI_Barrier( mpi_data->world ); 
    for ( j = 0; j < system->n; ++j )
        workspace->s[j] = workspace->x[j][0];
    iters = CG(system, control, data, workspace, workspace->H, workspace->b_s,
            control->cm_solver_q_err, workspace->s, mpi_data, out_control->log , control->nprocs );
    for ( j = 0; j < system->n; ++j )
        workspace->x[j][0] = workspace->s[j];

#if defined(DEBUG)
    fprintf( stderr, "p%d: first CG completed\n", system->my_rank );
#endif

    for ( j = 0; j < system->n; ++j )
        workspace->t[j] = workspace->x[j][1];
    iters += CG(system, control, data, workspace, workspace->H, workspace->b_t,//newQEq sCG
            control->cm_solver_q_err, workspace->t, mpi_data, out_control->log, control->nprocs );
    for ( j = 0; j < system->n; ++j )
        workspace->x[j][1] = workspace->t[j];

#if defined(DEBUG)
    fprintf( stderr, "p%d: second CG completed\n", system->my_rank );
#endif

    Calculate_Charges( system, workspace, mpi_data );
#if defined(DEBUG)
    fprintf( stderr, "p%d: computed charges\n", system->my_rank );
    //Print_Charges( system );
#endif

#if defined(LOG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        data->timing.cm_solver_iters += iters;
    }
#endif
}
