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

#include "QEq.h"

#include "allocate.h"
#include "lin_alg.h"
#include "list.h"
#include "print_utils.h"
#include "index_utils.h"
#include "system_props.h"

#include "sort.h"


int compare_matrix_entry(const void *v1, const void *v2)
{
    return ((sparse_matrix_entry *)v1)->j - ((sparse_matrix_entry *)v2)->j;
}


void Sort_Matrix_Rows( sparse_matrix *A )
{
    int i, si, ei;

    for( i = 0; i < A->n; ++i ) {
        si = A->start[i];
        ei = A->start[i+1];
        qsort( &(A->entries[si]), ei - si, 
                sizeof(sparse_matrix_entry), compare_matrix_entry );
    }
}


void Calculate_Droptol( sparse_matrix *A, real *droptol, real dtol )
{
    int i, j, k;
    real val;

    /* init droptol to 0 */
    for( i = 0; i < A->n; ++i )
        droptol[i] = 0;

    /* calculate sqaure of the norm of each row */
    for( i = 0; i < A->n; ++i ) {
        for( k = A->start[i]; k < A->start[i+1]-1; ++k ) {
            j = A->entries[k].j;
            val = A->entries[k].val;

            droptol[i] += val*val;
            droptol[j] += val*val;
        }

        val = A->entries[k].val; // diagonal entry
        droptol[i] += val*val;
    }

    /* calculate local droptol for each row */
    //fprintf( stderr, "droptol: " );
    for( i = 0; i < A->n; ++i ) {
        //fprintf( stderr, "%f-->", droptol[i] );
        droptol[i] = SQRT( droptol[i] ) * dtol;
        //fprintf( stderr, "%f  ", droptol[i] );
    }
    //fprintf( stderr, "\n" );
}


int Estimate_LU_Fill( sparse_matrix *A, real *droptol )
{
    int i, j, pj;
    int fillin;
    real val;

    fillin = 0;

    //fprintf( stderr, "n: %d\n", A->n );
    for( i = 0; i < A->n; ++i )
        for( pj = A->start[i]; pj < A->start[i+1]-1; ++pj ){
            j = A->entries[pj].j;
            val = A->entries[pj].val;
            //fprintf( stderr, "i: %d, j: %d", i, j );

            if( fabs(val) > droptol[i] )
                ++fillin;
        }

    return fillin + A->n;
}


void ICHOLT( sparse_matrix *A, real *droptol, 
        sparse_matrix *L, sparse_matrix *U )
{
    sparse_matrix_entry tmp[1000];
    int i, j, pj, k1, k2, tmptop, Ltop;
    real val;
    int *Utop;

    Utop = (int*) malloc((A->n+1) * sizeof(int));

    // clear variables
    Ltop = 0;
    tmptop = 0;
    for( i = 0; i <= A->n; ++i )
        L->start[i] = U->start[i] = 0;

    for( i = 0; i < A->n; ++i )
        Utop[i] = 0;

    //fprintf( stderr, "n: %d\n", A->n );
    for( i = 0; i < A->n; ++i ){
        L->start[i] = Ltop;
        tmptop = 0;

        for( pj = A->start[i]; pj < A->start[i+1]-1; ++pj ){
            j = A->entries[pj].j;
            val = A->entries[pj].val;
            //fprintf( stderr, "i: %d, j: %d", i, j );

            if( fabs(val) > droptol[i] ){
                k1 = 0;
                k2 = L->start[j];
                while( k1 < tmptop && k2 < L->start[j+1] ){
                    if( tmp[k1].j < L->entries[k2].j )
                        ++k1;
                    else if( tmp[k1].j > L->entries[k2].j )
                        ++k2;
                    else
                        val -= (tmp[k1++].val * L->entries[k2++].val);
                }

                // L matrix is lower triangular, 
                // so right before the start of next row comes jth diagonal
                val /= L->entries[L->start[j+1]-1].val;

                tmp[tmptop].j = j;
                tmp[tmptop].val = val;
                ++tmptop;
            }
            //fprintf( stderr, " -- done\n" );
        }

        // compute the ith diagonal in L
        // sanity check
        if( A->entries[pj].j != i ) {
            fprintf( stderr, "i=%d, badly built A matrix!\n", i );
            exit(999);
        }

        val = A->entries[pj].val;
        for( k1 = 0; k1 < tmptop; ++k1 )
            val -= (tmp[k1].val * tmp[k1].val);

        tmp[tmptop].j = i;
        tmp[tmptop].val = SQRT(val);

        // apply the dropping rule once again
        //fprintf( stderr, "row%d: tmptop: %d\n", i, tmptop );
        //for( k1 = 0; k1<= tmptop; ++k1 )
        //  fprintf( stderr, "%d(%f)  ", tmp[k1].j, tmp[k1].val );
        //fprintf( stderr, "\n" );
        //fprintf( stderr, "row(%d): droptol=%.4f\n", i+1, droptol[i] );
        for( k1 = 0; k1 < tmptop; ++k1 )
            if( fabs(tmp[k1].val) > droptol[i] / tmp[tmptop].val ){
                L->entries[Ltop].j = tmp[k1].j;
                L->entries[Ltop].val = tmp[k1].val;
                U->start[tmp[k1].j+1]++;
                ++Ltop;
                //fprintf( stderr, "%d(%.4f)  ", tmp[k1].j+1, tmp[k1].val );
            }
        // keep the diagonal in any case
        L->entries[Ltop].j = tmp[k1].j;
        L->entries[Ltop].val = tmp[k1].val;
        ++Ltop;
        //fprintf( stderr, "%d(%.4f)\n", tmp[k1].j+1,  tmp[k1].val );
    }

    L->start[i] = Ltop;
    //fprintf( stderr, "nnz(L): %d, max: %d\n", Ltop, L->n * 50 );

    for( i = 1; i <= U->n; ++i )
        Utop[i] = U->start[i] = U->start[i] + U->start[i-1] + 1;

    for( i = 0; i < L->n; ++i )
        for( pj = L->start[i]; pj < L->start[i+1]; ++pj ){
            j = L->entries[pj].j;
            U->entries[Utop[j]].j = i;
            U->entries[Utop[j]].val = L->entries[pj].val;
            Utop[j]++;
        }

    //fprintf( stderr, "nnz(U): %d, max: %d\n", Utop[U->n], U->n * 50 );
}


void Init_MatVec( reax_system *system, control_params *control, 
        simulation_data *data, static_storage *workspace, 
        list *far_nbrs )
{
    int i, fillin;
    real s_tmp, t_tmp;
    //char fname[100];

    if(control->refactor > 0 && 
            ((data->step-data->prev_steps)%control->refactor==0 || workspace->L.entries==NULL))
    {
        //Print_Linear_System( system, control, workspace, data->step );
        Sort_Matrix_Rows( &workspace->H );

        //fprintf( stderr, "H matrix sorted\n" );

        Calculate_Droptol( &workspace->H, workspace->droptol, control->droptol ); 
        //fprintf( stderr, "drop tolerances calculated\n" );

        if( workspace->L.entries == NULL )
        {
            fillin = Estimate_LU_Fill( &workspace->H, workspace->droptol );

#ifdef __DEBUG_CUDA__
            fprintf( stderr, "fillin = %d\n", fillin );
#endif

            if( Allocate_Matrix( &(workspace->L), far_nbrs->n, fillin ) == 0 ||
                    Allocate_Matrix( &(workspace->U), far_nbrs->n, fillin ) == 0 )
            {
                fprintf( stderr, "not enough memory for LU matrices. terminating.\n" );
                exit(INSUFFICIENT_SPACE);
            }

#if defined(DEBUG_FOCUS)
            fprintf( stderr, "fillin = %d\n", fillin );
            fprintf( stderr, "allocated memory: L = U = %ldMB\n",
                    fillin * sizeof(sparse_matrix_entry) / (1024*1024) );
#endif
        }

        ICHOLT( &workspace->H, workspace->droptol, &workspace->L, &workspace->U );

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "icholt-" );
        //sprintf( fname, "%s.L%d.out", control->sim_name, data->step );
        //Print_Sparse_Matrix2( workspace->L, fname );
        //Print_Sparse_Matrix( U );
#endif
    }

    /* extrapolation for s & t */
    for( i = 0; i < system->N; ++i ) {
        // no extrapolation
        //s_tmp = workspace->s[0][i];
        //t_tmp = workspace->t[0][i];

        // linear
        //s_tmp = 2 * workspace->s[0][i] - workspace->s[1][i];
        //t_tmp = 2 * workspace->t[0][i] - workspace->t[1][i];

        // quadratic
        //s_tmp = workspace->s[2][i] + 3 * (workspace->s[0][i]-workspace->s[1][i]);
        t_tmp = workspace->t[index_wkspace_sys(2,i,system->N)] + 3*(workspace->t[index_wkspace_sys(0,i,system->N)]-workspace->t[index_wkspace_sys(1,i,system->N)]);

        // cubic
        s_tmp = 4 * (workspace->s[index_wkspace_sys(0,i,system->N)] + workspace->s[index_wkspace_sys(2,i,system->N)]) - 
            (6 * workspace->s[index_wkspace_sys(1,i,system->N)] + workspace->s[index_wkspace_sys(3,i,system->N)] );
        //t_tmp = 4 * (workspace->t[0][i] + workspace->t[2][i]) - 
        //  (6 * workspace->t[1][i] + workspace->t[3][i] );

        // 4th order
        //s_tmp = 5 * (workspace->s[0][i] - workspace->s[3][i]) + 
        //  10 * (-workspace->s[1][i] + workspace->s[2][i] ) + workspace->s[4][i];
        //t_tmp = 5 * (workspace->t[0][i] - workspace->t[3][i]) + 
        //  10 * (-workspace->t[1][i] + workspace->t[2][i] ) + workspace->t[4][i];

        workspace->s[index_wkspace_sys(4,i,system->N)] = workspace->s[index_wkspace_sys(3,i,system->N)];
        workspace->s[index_wkspace_sys(3,i,system->N)] = workspace->s[index_wkspace_sys(2,i,system->N)]; 
        workspace->s[index_wkspace_sys(2,i,system->N)] = workspace->s[index_wkspace_sys(1,i,system->N)];
        workspace->s[index_wkspace_sys(1,i,system->N)] = workspace->s[index_wkspace_sys(0,i,system->N)];
        workspace->s[index_wkspace_sys(0,i,system->N)] = s_tmp;

        workspace->t[index_wkspace_sys(4,i,system->N)] = workspace->t[index_wkspace_sys(3,i,system->N)];
        workspace->t[index_wkspace_sys(3,i,system->N)] = workspace->t[index_wkspace_sys(2,i,system->N)]; 
        workspace->t[index_wkspace_sys(2,i,system->N)] = workspace->t[index_wkspace_sys(1,i,system->N)];
        workspace->t[index_wkspace_sys(1,i,system->N)] = workspace->t[index_wkspace_sys(0,i,system->N)];
        workspace->t[index_wkspace_sys(0,i,system->N)] = t_tmp;
    }
}


void Calculate_Charges( reax_system *system, static_storage *workspace )
{
    int i;
    real u, s_sum, t_sum;

    s_sum = t_sum = 0.;
    for( i = 0; i < system->N; ++i ) {
        s_sum += workspace->s[index_wkspace_sys(0,i,system->N)];
        t_sum += workspace->t[index_wkspace_sys(0,i,system->N)];
    }

    u = s_sum / t_sum;

#ifdef __DEBUG_CUDA__
    fprintf (stderr, "Host --->s %13.2f, t %13.f, u %13.2f \n", s_sum, t_sum, u );
#endif

    for( i = 0; i < system->N; ++i )
    {
        system->atoms[i].q = workspace->s[index_wkspace_sys(0,i,system->N)] - u * workspace->t[index_wkspace_sys(0,i,system->N)];
    }
}


void QEq( reax_system *system, control_params *control, simulation_data *data, 
        static_storage *workspace, list *far_nbrs, 
        output_controls *out_control )
{
    int matvecs;

    //real t_start, t_elapsed;

    //t_start = Get_Time ();
    Init_MatVec( system, control, data, workspace, far_nbrs );
    //t_elapsed = Get_Timing_Info ( t_start );

    //fprintf (stderr, " CPU Init_MatVec timing ----> %f \n", t_elapsed );

    //if( data->step % 10 == 0 )
    //  Print_Linear_System( system, control, workspace, far_nbrs, data->step );

    //t_start = Get_Time ( );
    matvecs = GMRES( workspace, &workspace->H, 
            workspace->b_s, control->q_err, &workspace->s[0], out_control->log, system );
    matvecs += GMRES( workspace, &workspace->H, 
            workspace->b_t, control->q_err, &workspace->t[0], out_control->log, system );
    //t_elapsed = Get_Timing_Info ( t_start );

    //fprintf (stderr, " CPU GMRES timing ---> %f \n", t_elapsed );

    //matvecs = GMRES_HouseHolder( workspace, workspace->H, 
    //    workspace->b_s, control->q_err, workspace->s[0], out_control->log );
    //matvecs += GMRES_HouseHolder( workspace, workspace->H,  
    //    workspace->b_t, control->q_err, workspace->t[0], out_control->log );

    //matvecs = PGMRES( workspace, &workspace->H, workspace->b_s, control->q_err,
    //  &workspace->L, &workspace->U, &workspace->s[index_wkspace_sys(0,0,system->N)], out_control->log, system );
    //matvecs += PGMRES( workspace, &workspace->H, workspace->b_t, control->q_err,
    //  &workspace->L, &workspace->U, &workspace->t[index_wkspace_sys(0,0,system->N)], out_control->log, system );

    //matvecs=PCG( workspace, workspace->H, workspace->b_s, control->q_err, 
    //      workspace->L, workspace->U, workspace->s[0], out_control->log ) + 1;
    ///matvecs+=PCG( workspace, workspace->H, workspace->b_t, control->q_err, 
    //     workspace->L, workspace->U, workspace->t[0], out_control->log ) + 1;

    //matvecs = CG( workspace, workspace->H, 
    // workspace->b_s, control->q_err, workspace->s[0], out_control->log ) + 1;
    //matvecs += CG( workspace, workspace->H, 
    // workspace->b_t, control->q_err, workspace->t[0], out_control->log ) + 1;

    //matvecs = SDM( workspace, workspace->H, 
    // workspace->b_s, control->q_err, workspace->s[0], out_control->log ) + 1;
    //matvecs += SDM( workspace, workspace->H, 
    // workspace->b_t, control->q_err, workspace->t[0], out_control->log ) + 1;

    //fprintf (stderr, " GMRES done with iterations %d \n", matvecs );

    data->timing.matvecs += matvecs;
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "linsolve-" );
#endif

    Calculate_Charges( system, workspace );
    //fprintf( stderr, "%d %.9f %.9f %.9f %.9f %.9f %.9f\n", 
    //   data->step, 
    //   workspace->s[0][0], workspace->t[0][0], 
    //   workspace->s[0][1], workspace->t[0][1], 
    //   workspace->s[0][2], workspace->t[0][2] );
    // if( data->step == control->nsteps )
    //Print_Charges( system, control, workspace, data->step );
}
