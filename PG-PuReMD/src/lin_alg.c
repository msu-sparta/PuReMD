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

#include "lin_alg.h"

#include "basic_comm.h"
#include "io_tools.h"
#include "tool_box.h"
#include "vector.h"

#if defined(HAVE_CUDA) && defined(DEBUG)
  #include "cuda/cuda_validation.h"
#endif

#if defined(CG_PERFORMANCE)
real t_start, t_elapsed, matvec_time, dot_time;
#endif


static void dual_Sparse_MatVec( const sparse_matrix * const A,
        const rvec2 * const x, rvec2 * const b, const int N )
{
    int  i, j, k, si;
    real H;

    for ( i = 0; i < N; ++i )
    {
        b[i][0] = 0;
        b[i][1] = 0;
    }

    /* perform multiplication */
    for ( i = 0; i < A->n; ++i )
    {
        si = A->start[i];
        b[i][0] += A->entries[si].val * x[i][0];
        b[i][1] += A->entries[si].val * x[i][1];

        for ( k = si + 1; k < A->end[i]; ++k )
        {
            j = A->entries[k].j;
            H = A->entries[k].val;

            b[i][0] += H * x[j][0];
            b[i][1] += H * x[j][1];

#if defined(HALF_LIST)
            // comment out for tryQEq
            //if( j < A->n ) {
            b[j][0] += H * x[i][0];
            b[j][1] += H * x[i][1];
            //}
#endif
        }
    }
}


/* Diagonal (Jacobi) preconditioner computation */
real diag_pre_comp( const reax_system * const system, real * const Hdia_inv )
{
    unsigned int i;
    real start;

    start = Get_Time( );

    for ( i = 0; i < system->n; ++i )
    {
//        if ( H->entries[H->start[i + 1] - 1].val != 0.0 )
//        {
//            Hdia_inv[i] = 1.0 / H->entries[H->start[i + 1] - 1].val;
            Hdia_inv[i] = 1.0 / system->reax_param.sbp[ system->my_atoms[i].type ].eta;
//        }
//        else
//        {
//            Hdia_inv[i] = 1.0;
//        }
    }

    return Get_Timing_Info( start );
}


int dual_CG( const reax_system * const system, const control_params * const control,
        const storage * const workspace, const simulation_data * const data,
        const mpi_datatypes * const mpi_data,
        const sparse_matrix * const H, const rvec2 * const b,
        const real tol, rvec2 * const x, const int fresh_pre )
{
    int i, j, n, N, iters;
    rvec2 tmp, alpha, beta;
    rvec2 my_sum, norm_sqr, b_norm, my_dot;
    rvec2 sig_old, sig_new;
    MPI_Comm comm;

    n = system->n;
    N = system->N;
    comm = mpi_data->world;
    iters = 0;

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        iters = 0;
        t_start = matvec_time = dot_time = 0;
        t_start = Get_Time( );
    }
#endif

#if defined(HAVE_CUDA) && defined(DEBUG)
    check_zeros_host( x, N, "x" );
#endif

    Dist( system, mpi_data, x, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );

#if defined(HAVE_CUDA) && defined(DEBUG)
    check_zeros_host( x, N, "x" );
#endif

    dual_Sparse_MatVec( H, x, workspace->q2, N );

//  if (data->step > 0) return;

#if defined(HALF_LIST)
    // tryQEq
    Coll( system, mpi_data, workspace->q2, RVEC2_PTR_TYPE,
            mpi_data->mpi_rvec2 );
#endif

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
        Update_Timing_Info( &t_start, &matvec_time );
#endif

    for ( j = 0; j < n; ++j )
    {
        /* residual */
        workspace->r2[j][0] = b[j][0] - workspace->q2[j][0];
        workspace->r2[j][1] = b[j][1] - workspace->q2[j][1];

        /* apply diagonal pre-conditioner */
        workspace->d2[j][0] = workspace->r2[j][0] * workspace->Hdia_inv[j];
        workspace->d2[j][1] = workspace->r2[j][1] * workspace->Hdia_inv[j];
    }

    //print_host_rvec2 (workspace->r2, n);

    /* 2-norm of b */
    my_sum[0] = 0.0;
    my_sum[1] = 0.0;
    for ( j = 0; j < n; ++j )
    {
        my_sum[0] += SQR( b[j][0] );
        my_sum[1] += SQR( b[j][1] );
    }
    MPI_Allreduce( &my_sum, &norm_sqr, 2, MPI_DOUBLE, MPI_SUM, comm );
    b_norm[0] = SQRT( norm_sqr[0] );
    b_norm[1] = SQRT( norm_sqr[1] );

    /* inner product of r and d */
    my_dot[0] = 0.0;
    my_dot[1] = 0.0;
    for ( j = 0; j < n; ++j )
    {
        my_dot[0] += workspace->r2[j][0] * workspace->d2[j][0];
        my_dot[1] += workspace->r2[j][1] * workspace->d2[j][1];
    }

    MPI_Allreduce( &my_dot, &sig_new, 2, MPI_DOUBLE, MPI_SUM, comm );

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &dot_time );
    }
#endif

    for ( i = 1; i < control->cm_solver_max_iters; ++i )
    {
        Dist( system, mpi_data, workspace->d2, RVEC2_PTR_TYPE,
                mpi_data->mpi_rvec2 );

        dual_Sparse_MatVec( H, workspace->d2, workspace->q2, N );

#if defined(HALF_LIST)
        // tryQEq
        Coll( system, mpi_data, workspace->q2, RVEC2_PTR_TYPE,
                mpi_data->mpi_rvec2 );
#endif

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
            Update_Timing_Info( &t_start, &matvec_time );
#endif

        /* inner product of d and q */
        my_dot[0] = 0.0;
        my_dot[1] = 0.0;
        for ( j = 0; j < n; ++j )
        {
            my_dot[0] += workspace->d2[j][0] * workspace->q2[j][0];
            my_dot[1] += workspace->d2[j][1] * workspace->q2[j][1];
        }
        MPI_Allreduce( &my_dot, &tmp, 2, MPI_DOUBLE, MPI_SUM, comm );

        alpha[0] = sig_new[0] / tmp[0];
        alpha[1] = sig_new[1] / tmp[1];
        my_dot[0] = 0.0;
        my_dot[1] = 0.0;
        for ( j = 0; j < n; ++j )
        {
            /* update x */
            x[j][0] += alpha[0] * workspace->d2[j][0];
            x[j][1] += alpha[1] * workspace->d2[j][1];

            /* update residual */
            workspace->r2[j][0] -= alpha[0] * workspace->q2[j][0];
            workspace->r2[j][1] -= alpha[1] * workspace->q2[j][1];

            /* apply diagonal pre-conditioner */
            workspace->p2[j][0] = workspace->r2[j][0] * workspace->Hdia_inv[j];
            workspace->p2[j][1] = workspace->r2[j][1] * workspace->Hdia_inv[j];

            /* dot product: r.p */
            my_dot[0] += workspace->r2[j][0] * workspace->p2[j][0];
            my_dot[1] += workspace->r2[j][1] * workspace->p2[j][1];
        }

        sig_old[0] = sig_new[0];
        sig_old[1] = sig_new[1];
        MPI_Allreduce( &my_dot, &sig_new, 2, MPI_DOUBLE, MPI_SUM, comm );

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
            Update_Timing_Info( &t_start, &dot_time );
#endif

        if ( SQRT(sig_new[0]) / b_norm[0] <= tol || SQRT(sig_new[1]) / b_norm[1] <= tol )
        {
            break;
        }

        beta[0] = sig_new[0] / sig_old[0];
        beta[1] = sig_new[1] / sig_old[1];
        for ( j = 0; j < n; ++j )
        {
            /* d = p + beta * d */
            workspace->d2[j][0] = workspace->p2[j][0] + beta[0] * workspace->d2[j][0];
            workspace->d2[j][1] = workspace->p2[j][1] + beta[1] * workspace->d2[j][1];
        }
    }

    if ( SQRT(sig_new[0]) / b_norm[0] <= tol )
    {
        for ( j = 0; j < n; ++j )
        {
            workspace->t[j] = workspace->x[j][1];
        }

        iters = CG( system, control, workspace, data, mpi_data,
                H, workspace->b_t, tol, workspace->t, fresh_pre );

        for ( j = 0; j < n; ++j )
        {
            workspace->x[j][1] = workspace->t[j];
        }
    }
    else if ( SQRT(sig_new[1]) / b_norm[1] <= tol )
    {
        for ( j = 0; j < n; ++j )
        {
            workspace->s[j] = workspace->x[j][0];
        }

        iters = CG( system, control, workspace, data, mpi_data, 
                H, workspace->b_s, tol, workspace->s, fresh_pre );

        for ( j = 0; j < n; ++j )
        {
            workspace->x[j][0] = workspace->s[j];
        }
    }

    if ( i >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] Dual CG convergence failed (%d iters)\n", i + 1 + iters );
    }

    return (i + 1) + iters;
}


const void Sparse_MatVec( const sparse_matrix * const A, const real * const x,
        real * const b, const int N )
{
    int i, j, k, si;
    real H;

    for ( i = 0; i < N; ++i )
    {
        b[i] = 0;
    }

    for ( i = 0; i < A->n; ++i )
    {
        si = A->start[i];

        b[i] += A->entries[si].val * x[i];

        for ( k = si + 1; k < A->end[i]; ++k )
        {
            j = A->entries[k].j;
            H = A->entries[k].val;

            b[i] += H * x[j];
#if defined(HALF_LIST)
            //if( j < A->n ) // comment out for tryQEq
            b[j] += H * x[i];
#endif
        }
    }
}


int CG( const reax_system * const system, const control_params * const control,
        const storage * const workspace, const simulation_data * const data,
        const mpi_datatypes * const mpi_data,
        const sparse_matrix * const H, const real * const b,
        const real tol, real * const x, const int fresh_pre )
{
    int i, j;
    real tmp, alpha, beta, b_norm;
    real sig_old, sig_new, sig0;

    Dist( system, mpi_data, x, REAL_PTR_TYPE, MPI_DOUBLE );
    Sparse_MatVec( H, x, workspace->q, system->N );

#if defined(HALF_LIST)
    // tryQEq
    Coll( system, mpi_data, workspace->q, REAL_PTR_TYPE, MPI_DOUBLE );
#endif

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &matvec_time );
    }
#endif

    Vector_Sum( workspace->r , 1.0,  b, -1.0, workspace->q, system->n );

    // preconditioner
    for ( j = 0; j < system->n; ++j )
    {
        workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j];
    }

    b_norm = Parallel_Norm( b, system->n, mpi_data->world );
    sig_new = Parallel_Dot( workspace->r, workspace->d, system->n, mpi_data->world );
    sig0 = sig_new;

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &dot_time );
    }
#endif

    for ( i = 1; i < control->cm_solver_max_iters && SQRT(sig_new) / b_norm > tol; ++i )
    {
        Dist( system, mpi_data, workspace->d, REAL_PTR_TYPE, MPI_DOUBLE );
        Sparse_MatVec( H, workspace->d, workspace->q, system->N );

#if defined(HALF_LIST)
        //tryQEq
        Coll( system, mpi_data, workspace->q, REAL_PTR_TYPE, MPI_DOUBLE );
#endif

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &matvec_time );
        }
#endif

        tmp = Parallel_Dot( workspace->d, workspace->q, system->n, mpi_data->world );
        alpha = sig_new / tmp;
        Vector_Add( x, alpha, workspace->d, system->n );
        Vector_Add( workspace->r, -alpha, workspace->q, system->n );

        /* preconditioner */
        for ( j = 0; j < system->n; ++j )
        {
            workspace->p[j] = workspace->r[j] * workspace->Hdia_inv[j];
        }

        sig_old = sig_new;
        sig_new = Parallel_Dot( workspace->r, workspace->p, system->n, mpi_data->world );
        beta = sig_new / sig_old;
        Vector_Sum( workspace->d, 1.0, workspace->p, beta, workspace->d, system->n );

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &dot_time );
        }
#endif
    }

    if ( i >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] CG convergence failed (%d iters)\n", i );
        fprintf( stderr, "  [INFO] Rel. residual errors: %f\n",
                SQRT(sig_new) / b_norm );
        return i;
    }

    return i;
}
