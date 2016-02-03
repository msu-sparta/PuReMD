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

#include "QEq.h"
#include "allocate.h"
#include "GMRES.h"
#include "list.h"
#include "print_utils.h"

int compare_matrix_entry(const void *v1, const void *v2)
{
    /* larger element has larger column index */
    return ((sparse_matrix_entry *)v1)->j - ((sparse_matrix_entry *)v2)->j;
}


void Sort_Matrix_Rows( sparse_matrix *A )
{
    int i, si, ei;

    /* sort each row of A using column indices */
    for ( i = 0; i < A->n; ++i )
    {
        si = A->start[i];
        ei = A->start[i + 1];
        /* polymorphic sort in standard C library */
        qsort( &(A->entries[si]), ei - si,
               sizeof(sparse_matrix_entry), compare_matrix_entry );
    }
}


void Calculate_Droptol( sparse_matrix *A, real *droptol, real dtol )
{
    int i, j, k;
    real val;

    /* init droptol to 0 */
    for ( i = 0; i < A->n; ++i )
        droptol[i] = 0;

    /* calculate sqaure of the norm of each row */
    for ( i = 0; i < A->n; ++i )
    {
        for ( k = A->start[i]; k < A->start[i + 1] - 1; ++k )
        {
            j = A->entries[k].j;
            val = A->entries[k].val;

            droptol[i] += val * val;
            droptol[j] += val * val;
        }

        val = A->entries[k].val; // diagonal entry
        droptol[i] += val * val;
    }

    /* calculate local droptol for each row */
    //fprintf( stderr, "droptol: " );
    for ( i = 0; i < A->n; ++i )
    {
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
    for ( i = 0; i < A->n; ++i )
        for ( pj = A->start[i]; pj < A->start[i + 1] - 1; ++pj )
        {
            j = A->entries[pj].j;
            val = A->entries[pj].val;
            //fprintf( stderr, "i: %d, j: %d", i, j );

            if ( fabs(val) > droptol[i] )
                ++fillin;
        }

    return fillin + A->n;
}


/* Incomplete Cholesky factorization with thresholding */
void ICHOLT( sparse_matrix *A, real *droptol,
             sparse_matrix *L, sparse_matrix *U )
{
    sparse_matrix_entry tmp[1000];
    int i, j, pj, k1, k2, tmptop, Ltop;
    real val;
    int *Utop;

    Utop = (int*) malloc((A->n + 1) * sizeof(int));

    // clear variables
    Ltop = 0;
    tmptop = 0;
    for ( i = 0; i <= A->n; ++i )
        L->start[i] = U->start[i] = 0;

    for ( i = 0; i < A->n; ++i )
        Utop[i] = 0;

    //fprintf( stderr, "n: %d\n", A->n );
    for ( i = 0; i < A->n; ++i )
    {
        L->start[i] = Ltop;
        tmptop = 0;

        for ( pj = A->start[i]; pj < A->start[i + 1] - 1; ++pj )
        {
            j = A->entries[pj].j;
            val = A->entries[pj].val;
            //fprintf( stderr, "i: %d, j: %d", i, j );

            if ( fabs(val) > droptol[i] )
            {
                k1 = 0;
                k2 = L->start[j];
                while ( k1 < tmptop && k2 < L->start[j + 1] )
                {
                    if ( tmp[k1].j < L->entries[k2].j )
                        ++k1;
                    else if ( tmp[k1].j > L->entries[k2].j )
                        ++k2;
                    else
                        val -= (tmp[k1++].val * L->entries[k2++].val);
                }

                // L matrix is lower triangular,
                // so right before the start of next row comes jth diagonal
                val /= L->entries[L->start[j + 1] - 1].val;

                tmp[tmptop].j = j;
                tmp[tmptop].val = val;
                ++tmptop;
            }
            //fprintf( stderr, " -- done\n" );
        }

        // compute the ith diagonal in L
        // sanity check
        if ( A->entries[pj].j != i )
        {
            fprintf( stderr, "i=%d, badly built A matrix!\n", i );
            exit(999);
        }

        val = A->entries[pj].val;
        for ( k1 = 0; k1 < tmptop; ++k1 )
            val -= (tmp[k1].val * tmp[k1].val);

        tmp[tmptop].j = i;
        tmp[tmptop].val = SQRT(val);

        // apply the dropping rule once again
        //fprintf( stderr, "row%d: tmptop: %d\n", i, tmptop );
        //for( k1 = 0; k1<= tmptop; ++k1 )
        //  fprintf( stderr, "%d(%f)  ", tmp[k1].j, tmp[k1].val );
        //fprintf( stderr, "\n" );
        //fprintf( stderr, "row(%d): droptol=%.4f\n", i+1, droptol[i] );
        for ( k1 = 0; k1 < tmptop; ++k1 )
            if ( fabs(tmp[k1].val) > droptol[i] / tmp[tmptop].val )
            {
                L->entries[Ltop].j = tmp[k1].j;
                L->entries[Ltop].val = tmp[k1].val;
                U->start[tmp[k1].j + 1]++;
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

    /* U = L^T (Cholesky factorization) */
    for ( i = 1; i <= U->n; ++i )
    {
        Utop[i] = U->start[i] = U->start[i] + U->start[i - 1] + 1;
    }
    for ( i = 0; i < L->n; ++i )
    {
        for ( pj = L->start[i]; pj < L->start[i + 1]; ++pj )
        {
            j = L->entries[pj].j;
            U->entries[Utop[j]].j = i;
            U->entries[Utop[j]].val = L->entries[pj].val;
            Utop[j]++;
        }
    }

    //fprintf( stderr, "nnz(U): %d, max: %d\n", Utop[U->n], U->n * 50 );

    free(Utop);
}


/* Parallel incomplete Cholesky factorization
 * 
 * Reference:
 * Edmond Chow and Aftab Patel
 * Fine-Grained Parallel Incomplete LU Factorization
 * SIAM J. Sci. Comp. */
static void ICHOL_PAR( const sparse_matrix * const A, const unsigned int sweeps,
             sparse_matrix * const U, sparse_matrix * const U_t )
{
    unsigned int i, j, k, pj, x = 0, y = 0, ei_x, ei_y;
    real sum;
    int *Utop;

    Utop = (int*) malloc((A->n + 1) * sizeof(int));
    for ( i = 0; i <= A->n; ++i )
    {
        U->start[i] = 0;
        Utop[i] = 0;
    }

    /* initial guesses for U^T */
    memcpy( U_t->start, A->start, sizeof(int) * (A->n+1) );
    memcpy( U_t->entries, A->entries, sizeof(sparse_matrix_entry) * (A->m) );

    /* to get convergence, A must have unit diagonal, so apply
     * transformation DAD, where D = diag(1./diag(A)),
     * to initial guess and apply ad hoc during sweeps */
    for ( i = 0; i < U_t->n; ++i )
    {
        for ( pj = U_t->start[i]; pj < U_t->start[i + 1] - 1; ++pj )
        {
            U_t->entries[pj].val /= U_t->entries[U_t->start[i + 1] - 1].val;
        }
        U_t->entries[pj].val = 1.;
    }

    for ( i = 0; i < sweeps; ++i )
    {
//        #pragma omp parallel for
        for ( j = 0; j < A->m; ++j )
        {
            sum = ZERO;

            /* determine row bounds of current nonzero */
            for ( k = 0; k < A->n; ++k )
            {
                if( A->start[k] > j )
                {
                    x = A->start[k - 1];
                    ei_x = A->start[k];
                    break;
                }
            }
            y = A->start[A->entries[j].j];
            ei_y = A->start[A->entries[j].j + 1];

            /* sparse dot product: dot( A(i,1:i-1), A(j,1:i-1) ) */
            while( A->entries[x].j < A->entries[j].j && 
                    A->entries[y].j < A->entries[j].j && 
                    x < ei_x && y < ei_y )
            {
                if( A->entries[x].j == A->entries[y].j )
                {
                    sum += (A->entries[x].val / A->entries[ei_x - 1].val) * 
                            (A->entries[y].val / A->entries[ei_y - 1].val);
                    ++x;
                    ++y;
                }
                else if( A->entries[x].j < A->entries[y].j )
                {
                    ++x;
                }
                else
                {
                    ++y;
                }
            }

            sum = (A->entries[j].val / A->entries[ei_x - 1].val) - sum;

            if( A->entries[j].j == A->start[k - 1] )
            {
                U_t->entries[j].val = sum / U_t->entries[A->start[k]-1].val;
            }
            else
            {
                U_t->entries[j].val = SQRT( sum );
            }
        }
    }


    /* transpose U^T and copy into U */
    for ( i = 0; i < U_t->n; ++i )
    {
        /* count nonzeros in each row of U, store in U->start */
        for ( pj = U_t->start[i]; pj < U_t->start[i + 1]; ++pj )
        {
            U->start[U_t->entries[pj].j + 1]++;
        }
    }
    for ( i = 1; i <= U->n; ++i )
    {
        Utop[i] = U->start[i] = U->start[i] + U->start[i - 1] + 1;
    }
    for ( i = 0; i < U_t->n; ++i )
    {
        for ( pj = U_t->start[i]; pj < U_t->start[i + 1]; ++pj )
        {
            j = U_t->entries[pj].j;
            U->entries[Utop[j]].j = i;
            U->entries[Utop[j]].val = U_t->entries[pj].val;
            Utop[j]++;
        }
    }

    free(Utop);
}
//% use entries of A as inital guesses for U
//U = triu(A);
//% find all non-zero elements
//[row,col] = find(U);
//for s=1:sweeps
//    for b=1:numel(row);
//        i = row(b);
//        j = col(b);
//%        stemp = 0.0;
//%        for k=1:(i-1)
//%          stemp = stemp + U(k,i)*U(k,j);
//%        end
//         OR
//%        for k=1:(j-1)
//%          stemp = stemp + U(i,k)*U(j,k);
//%        end
//%	stemp = A(i,j) - stemp;
//        % vectorized computations for sparse matrices
//	if (i ~= j)
//	    U(i,j) = (A(i,j) - dot(U(1:i-1,i),U(1:i-1,j))) / U(i,i);
//        else
//	    U(i,i) = sqrt(A(i,j) - dot(U(1:i-1,i),U(1:i-1,j)));
//	end
//    end
//end

void Init_MatVec( reax_system *system, control_params *control,
                  simulation_data *data, static_storage *workspace,
                  list *far_nbrs )
{
    int i, fillin;
    real s_tmp, t_tmp;
    //char fname[100];

    if (control->refactor > 0 &&
            ((data->step - data->prev_steps) % control->refactor == 0 || workspace->L == NULL))
    {
        //Print_Linear_System( system, control, workspace, data->step );
        Sort_Matrix_Rows( workspace->H );
        //fprintf( stderr, "H matrix sorted\n" );
//        Calculate_Droptol( workspace->H, workspace->droptol, control->droptol );
        //fprintf( stderr, "drop tolerances calculated\n" );
        if ( workspace->L == NULL )
        {
//            fillin = Estimate_LU_Fill( workspace->H, workspace->droptol );
//            if ( Allocate_Matrix( &(workspace->L), far_nbrs->n, fillin ) == 0 ||
//                    Allocate_Matrix( &(workspace->U), far_nbrs->n, fillin ) == 0 )
//            {
//                fprintf( stderr, "not enough memory for LU matrices. terminating.\n" );
//                exit(INSUFFICIENT_SPACE);
//            }
            // TODO: ilu_par & ichol_par contain same sparsity pattern as H,
            //   so allocate with same NNZ (workspace->H->m)
            if ( Allocate_Matrix( &(workspace->L), far_nbrs->n, workspace->H->m ) == 0 ||
                    Allocate_Matrix( &(workspace->U), far_nbrs->n, workspace->H->m ) == 0 )
            {
                fprintf( stderr, "not enough memory for preconditioning matrices. terminating.\n" );
                exit(INSUFFICIENT_SPACE);
            }
#if defined(DEBUG_FOCUS)
            fprintf( stderr, "fillin = %d\n", fillin );
            fprintf( stderr, "allocated memory: L = U = %ldMB\n",
                     fillin * sizeof(sparse_matrix_entry) / (1024 * 1024) );
#endif
        }

//        ICHOLT( workspace->H, workspace->droptol, workspace->L, workspace->U );
        // TODO: add parameters for sweeps to control file
        ICHOL_PAR( workspace->H, 3, workspace->L, workspace->U );
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "icholt-" );
        //sprintf( fname, "%s.L%d.out", control->sim_name, data->step );
        //Print_Sparse_Matrix2( workspace->L, fname );
        //Print_Sparse_Matrix( U );
#endif
    }

    /* extrapolation for s & t */
    for ( i = 0; i < system->N; ++i )
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



void Calculate_Charges( reax_system *system, static_storage *workspace )
{
    int i;
    real u, s_sum, t_sum;

    s_sum = t_sum = 0.;
    for ( i = 0; i < system->N; ++i )
    {
        s_sum += workspace->s[0][i];
        t_sum += workspace->t[0][i];
    }

    u = s_sum / t_sum;
    for ( i = 0; i < system->N; ++i )
        system->atoms[i].q = workspace->s[0][i] - u * workspace->t[0][i];
}


void QEq( reax_system *system, control_params *control, simulation_data *data,
          static_storage *workspace, list *far_nbrs,
          output_controls *out_control )
{
    int matvecs;

    Init_MatVec( system, control, data, workspace, far_nbrs );

    //if( data->step % 10 == 0 )
    //  Print_Linear_System( system, control, workspace, far_nbrs, data->step );

    //TODO: add parameters in control file for solver choice and options
//  matvecs = GMRES( workspace, workspace->H,
//    workspace->b_s, control->q_err, workspace->s[0], out_control->log );
//  matvecs += GMRES( workspace, workspace->H,
//    workspace->b_t, control->q_err, workspace->t[0], out_control->log );

    //matvecs = GMRES_HouseHolder( workspace, workspace->H,
    //    workspace->b_s, control->q_err, workspace->s[0], out_control->log );
    //matvecs += GMRES_HouseHolder( workspace, workspace->H,
    //    workspace->b_t, control->q_err, workspace->t[0], out_control->log );

//    matvecs = PGMRES( workspace, workspace->H, workspace->b_s, control->q_err,
//      workspace->L, workspace->U, workspace->s[0], out_control->log );
//    matvecs += PGMRES( workspace, workspace->H, workspace->b_t, control->q_err,
//      workspace->L, workspace->U, workspace->t[0], out_control->log );

    matvecs = PGMRES_Jacobi( workspace, workspace->H, workspace->b_s, control->q_err,
                             workspace->L, workspace->U, workspace->s[0], out_control->log );
    matvecs += PGMRES_Jacobi( workspace, workspace->H, workspace->b_t, control->q_err,
                              workspace->L, workspace->U, workspace->t[0], out_control->log );

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
