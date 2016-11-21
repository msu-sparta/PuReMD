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

#include "linear_solvers.h"
#include "basic_comm.h"
#include "io_tools.h"
#include "tool_box.h"
#include "vector.h"

#ifdef HAVE_CUDA
#include "validation.h"
#endif

#if defined(CG_PERFORMANCE)
real t_start, t_elapsed, matvec_time, dot_time;
#endif


void dual_Sparse_MatVec( sparse_matrix *A, rvec2 *x, rvec2 *b, int N )
{
    int  i, j, k, si;
    real H;

    for ( i = 0; i < N; ++i )
    {
        b[i][0] = b[i][1] = 0;
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

            // comment out for tryQEq
            //if( j < A->n ) {
            b[j][0] += H * x[i][0];
            b[j][1] += H * x[i][1];
            //}
        }
    }
}


int dual_CG( reax_system *system, storage *workspace, sparse_matrix *H,
             rvec2 *b, real tol, rvec2 *x, mpi_datatypes* mpi_data, FILE *fout, simulation_data *data )
{
    int  i, j, n, N, matvecs, scale;
    rvec2 tmp, alpha, beta;
    rvec2 my_sum, norm_sqr, b_norm, my_dot;
    rvec2 sig_old, sig_new;
    MPI_Comm comm;

    int a;

    n = system->n;
    N = system->N;
    comm = mpi_data->world;
    matvecs = 0;
    scale = sizeof(rvec2) / sizeof(void);

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        matvecs = 0;
        t_start = matvec_time = dot_time = 0;
        t_start = Get_Time( );
    }
#endif

#ifdef HAVE_CUDA
    check_zeros_host (x, system->N, "x");
#endif

    Dist( system, mpi_data, x, mpi_data->mpi_rvec2, scale, rvec2_packer );

#ifdef HAVE_CUDA
    check_zeros_host (x, system->N, "x");
#endif

    dual_Sparse_MatVec( H, x, workspace->q2, N );

//  if (data->step > 0) return;

    // tryQEq
    Coll(system, mpi_data, workspace->q2, mpi_data->mpi_rvec2, scale, rvec2_unpacker);

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
        Update_Timing_Info( &t_start, &matvec_time );
#endif

    for ( j = 0; j < system->n; ++j )
    {
        /* residual */
        workspace->r2[j][0] = b[j][0] - workspace->q2[j][0];
        workspace->r2[j][1] = b[j][1] - workspace->q2[j][1];

        /* apply diagonal pre-conditioner */
        workspace->d2[j][0] = workspace->r2[j][0] * workspace->Hdia_inv[j];
        workspace->d2[j][1] = workspace->r2[j][1] * workspace->Hdia_inv[j];
    }

    //print_host_rvec2 (workspace->r2, n);

    /* norm of b */
    my_sum[0] = my_sum[1] = 0;
    for ( j = 0; j < n; ++j )
    {
        my_sum[0] += SQR( b[j][0] );
        my_sum[1] += SQR( b[j][1] );
    }
    //fprintf (stderr, "cg: my_sum[ %f, %f] \n", my_sum[0], my_sum[1]);
    MPI_Allreduce( &my_sum, &norm_sqr, 2, MPI_DOUBLE, MPI_SUM, comm );
    b_norm[0] = SQRT( norm_sqr[0] );
    b_norm[1] = SQRT( norm_sqr[1] );
    //fprintf( stderr, "bnorm = %f %f\n", b_norm[0], b_norm[1] );

    /* dot product: r.d */
    my_dot[0] = my_dot[1] = 0;
    for ( j = 0; j < n; ++j )
    {
        my_dot[0] += workspace->r2[j][0] * workspace->d2[j][0];
        my_dot[1] += workspace->r2[j][1] * workspace->d2[j][1];
    }
    //fprintf( stderr, "my_dot: %f %f\n", my_dot[0], my_dot[1] );
    MPI_Allreduce( &my_dot, &sig_new, 2, MPI_DOUBLE, MPI_SUM, comm );
    //fprintf( stderr, "HOST:sig_new: %f %f\n", sig_new[0], sig_new[1] );

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &dot_time );
    }
#endif

    for ( i = 1; i < 300; ++i )
    {
        Dist(system, mpi_data, workspace->d2, mpi_data->mpi_rvec2, scale, rvec2_packer);
        //print_host_rvec2 (workspace->d2, N);

        dual_Sparse_MatVec( H, workspace->d2, workspace->q2, N );

        // tryQEq
        Coll(system, mpi_data, workspace->q2, mpi_data->mpi_rvec2, scale, rvec2_unpacker);

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
            Update_Timing_Info( &t_start, &matvec_time );
#endif

        /* dot product: d.q */
        my_dot[0] = my_dot[1] = 0;
        for ( j = 0; j < n; ++j )
        {
            my_dot[0] += workspace->d2[j][0] * workspace->q2[j][0];
            my_dot[1] += workspace->d2[j][1] * workspace->q2[j][1];
        }
        MPI_Allreduce( &my_dot, &tmp, 2, MPI_DOUBLE, MPI_SUM, comm );
        //fprintf( stderr, "tmp: %f %f\n", tmp[0], tmp[1] );

        alpha[0] = sig_new[0] / tmp[0];
        alpha[1] = sig_new[1] / tmp[1];
        my_dot[0] = my_dot[1] = 0;
        for ( j = 0; j < system->n; ++j )
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
        //fprintf( stderr, "HOST:sig_new: %f %f\n", sig_new[0], sig_new[1] );

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
        for ( j = 0; j < system->n; ++j )
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
        matvecs = CG( system, workspace, H, workspace->b_t, tol, workspace->t,
                      mpi_data, fout );
        fprintf (stderr, " CG1: iterations --> %d \n", matvecs );
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
        matvecs = CG( system, workspace, H, workspace->b_s, tol, workspace->s,
                      mpi_data, fout );
        fprintf (stderr, " CG2: iterations --> %d \n", matvecs );
        for ( j = 0; j < system->n; ++j )
        {
            workspace->x[j][0] = workspace->s[j];
        }
    }

    if ( i >= 300 )
    {
        fprintf( stderr, "CG convergence failed!\n" );
    }

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
        fprintf( fout, "QEq %d + %d iters. matvecs: %f  dot: %f\n",
                 i + 1, matvecs, matvec_time, dot_time );
#endif

    return (i + 1) + matvecs;
}


#ifdef HAVE_CUDA
int Cuda_dual_CG( reax_system *system, storage *workspace, sparse_matrix *H,
        rvec2 *b, real tol, rvec2 *x, mpi_datatypes* mpi_data, FILE *fout,
        simulation_data *data )
{
    int  i, j, n, N, matvecs, scale;
    rvec2 tmp, alpha, beta;
    rvec2 my_sum, norm_sqr, b_norm, my_dot;
    rvec2 sig_old, sig_new;
    MPI_Comm comm;
    rvec2 *spad = (rvec2 *) host_scratch;
    int a;

    n = system->n;
    N = system->N;
    comm = mpi_data->world;
    matvecs = 0;
    scale = sizeof(rvec2) / sizeof(void);

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        matvecs = 0;
        t_start = matvec_time = dot_time = 0;
        t_start = Get_Time( );
    }
#endif

    //MVAPICH2
//#ifdef __CUDA_DEBUG__
//  Dist( system, mpi_data, workspace->x, mpi_data->mpi_rvec2, scale, rvec2_packer );
//#endif

//  check_zeros_device( x, system->N, "x" );

    get_from_device( spad, x, sizeof (rvec2) * system->total_cap, "CG:x:get" );
    Dist( system, mpi_data, spad, mpi_data->mpi_rvec2, scale, rvec2_packer );
    put_on_device( spad, x, sizeof (rvec2) * system->total_cap, "CG:x:put" );

//  check_zeros_device( x, system->N, "x" );

//  compare_rvec2 (workspace->x, x, N, "x");
//  if (data->step > 0) {
//      compare_rvec2 (workspace->b, dev_workspace->b, system->N, "b");
//      compare_rvec2 (workspace->x, dev_workspace->x, system->N, "x");
//
//      exit (0);
//  }


//#ifdef __CUDA_DEBUG__
//  dual_Sparse_MatVec( &workspace->H, workspace->x, workspace->q2, N );
//#endif
    //originally we were using only H->n which was system->n (init_md.c)
    //Cuda_Dual_Matvec ( H, x, dev_workspace->q2, H->n, system->total_cap);
    
    Cuda_Dual_Matvec ( H, x, dev_workspace->q2, system->N, system->total_cap);

//  compare_rvec2 (workspace->q2, dev_workspace->q2, N, "q2");

//  if (data->step > 0) exit (0);

    // tryQEq
    //MVAPICH2
//#ifdef __CUDA_DEBUG__
//  Coll(system,mpi_data,workspace->q2,mpi_data->mpi_rvec2,scale,rvec2_unpacker);
//#endif
    
    get_from_device (spad, dev_workspace->q2, sizeof (rvec2) * system->total_cap, "CG:q2:get" );
    Coll(system, mpi_data, spad, mpi_data->mpi_rvec2, scale, rvec2_unpacker);
    put_on_device (spad, dev_workspace->q2, sizeof (rvec2) * system->total_cap, "CG:q2:put" );

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
        Update_Timing_Info( &t_start, &matvec_time );
#endif

//#ifdef __CUDA_DEBUG__
//  for( j = 0; j < system->n; ++j ) {
//    // residual
//    workspace->r2[j][0] = workspace->b[j][0] - workspace->q2[j][0];
//    workspace->r2[j][1] = workspace->b[j][1] - workspace->q2[j][1];
//    // apply diagonal pre-conditioner
//    workspace->d2[j][0] = workspace->r2[j][0] * workspace->Hdia_inv[j];
//    workspace->d2[j][1] = workspace->r2[j][1] * workspace->Hdia_inv[j];
//  }
//#endif
    
    Cuda_CG_Diagnol_Preconditioner (dev_workspace, b, system->n);

//  compare_rvec2 (workspace->r2, dev_workspace->r2, n, "r2");
//  compare_rvec2 (workspace->d2, dev_workspace->d2, n, "d2");

    /* norm of b */
//#ifdef __CUDA_DEBUG__
//  my_sum[0] = my_sum[1] = 0;
//  for( j = 0; j < n; ++j ) {
//    my_sum[0] += SQR( workspace->b[j][0] );
//    my_sum[1] += SQR( workspace->b[j][1] );
//  }
//  fprintf (stderr, "cg: my_sum[ %f, %f] \n", my_sum[0], my_sum[1]);
//#endif

    my_sum[0] = my_sum[1] = 0;
    Cuda_Norm (b, n, my_sum);

//  fprintf (stderr, "cg: my_sum[ %f, %f] \n", my_sum[0], my_sum[1]);

    MPI_Allreduce( &my_sum, &norm_sqr, 2, MPI_DOUBLE, MPI_SUM, comm );
    b_norm[0] = SQRT( norm_sqr[0] );
    b_norm[1] = SQRT( norm_sqr[1] );
    //fprintf( stderr, "bnorm = %f %f\n", b_norm[0], b_norm[1] );

    /* dot product: r.d */
//#ifdef __CUDA_DEBUG__
//  my_dot[0] = my_dot[1] = 0;
//  for( j = 0; j < n; ++j ) {
//    my_dot[0] += workspace->r2[j][0] * workspace->d2[j][0];
//    my_dot[1] += workspace->r2[j][1] * workspace->d2[j][1];
//  }
//  fprintf( stderr, "my_dot: %f %f\n", my_dot[0], my_dot[1] );
//#endif

    my_dot[0] = my_dot[1] = 0;
    Cuda_Dot (dev_workspace->r2, dev_workspace->d2, my_dot, n);

// fprintf( stderr, "my_dot: %f %f\n", my_dot[0], my_dot[1] );
    
    MPI_Allreduce( &my_dot, &sig_new, 2, MPI_DOUBLE, MPI_SUM, comm );

    //fprintf( stderr, "DEVICE:sig_new: %f %f\n", sig_new[0], sig_new[1] );

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &dot_time );
    }
#endif

    for ( i = 1; i < 300; ++i )
    {
        //MVAPICH2
//#ifdef __CUDA_DEBUG__
//    Dist(system,mpi_data,workspace->d2,mpi_data->mpi_rvec2,scale,rvec2_packer);
//#endif
        
        get_from_device( spad, dev_workspace->d2, sizeof (rvec2) * system->total_cap, "cg:d2:get" );
        Dist( system, mpi_data, spad, mpi_data->mpi_rvec2, scale, rvec2_packer );
        put_on_device( spad, dev_workspace->d2, sizeof (rvec2) * system->total_cap, "cg:d2:put" );

        //print_device_rvec2 (dev_workspace->d2, N);

//#ifdef __CUDA_DEBUG__
//    dual_Sparse_MatVec( &workspace->H, workspace->d2, workspace->q2, N );
//#endif
        
        Cuda_Dual_Matvec( H, dev_workspace->d2, dev_workspace->q2, system->N, system->total_cap );

        /*
        fprintf (stderr, "******************* Device sparse Matrix--------> %d \n", H->n );
        fprintf (stderr, " ******* HOST SPARSE MATRIX ******** \n");
        print_sparse_matrix_host (&workspace->H);
        fprintf (stderr, " ******* HOST Vector ***************\n");
        print_host_rvec2 (workspace->d2, system->N);
        fprintf (stderr, " ******* Device SPARSE MATRIX ******** \n");
        print_sparse_matrix (&dev_workspace->H);
        fprintf (stderr, " ******* Device Vector ***************\n");
        print_device_rvec2 (dev_workspace->d2, system->N);
        */
        //compare_rvec2 (workspace->q2, dev_workspace->q2, N, "q2");

        // tryQEq
        // MVAPICH2
//#ifdef __CUDA_DEBUG__
//    Coll(system,mpi_data,workspace->q2,mpi_data->mpi_rvec2,scale,rvec2_unpacker);
//#endif

        get_from_device( spad, dev_workspace->q2, sizeof (rvec2) * system->total_cap, "cg:q2:get" );
        Coll( system, mpi_data, spad, mpi_data->mpi_rvec2, scale, rvec2_unpacker );
        put_on_device( spad, dev_workspace->q2, sizeof (rvec2) * system->total_cap, "cg:q2:put" );

//       compare_rvec2 (workspace->q2, dev_workspace->q2, N, "q2");

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &matvec_time );
        }
#endif

        /* dot product: d.q */
//#ifdef __CUDA_DEBUG__
//    my_dot[0] = my_dot[1] = 0;
//    for( j = 0; j < n; ++j ) {
//      my_dot[0] += workspace->d2[j][0] * workspace->q2[j][0];
//      my_dot[1] += workspace->d2[j][1] * workspace->q2[j][1];
//    }
//       fprintf( stderr, "H:my_dot: %f %f\n", my_dot[0], my_dot[1] );
//#endif

        my_dot[0] = my_dot[1] = 0;
        Cuda_Dot (dev_workspace->d2, dev_workspace->q2, my_dot, n);
        //fprintf( stderr, "D:my_dot: %f %f\n", my_dot[0], my_dot[1] );

        MPI_Allreduce( &my_dot, &tmp, 2, MPI_DOUBLE, MPI_SUM, comm );
        //fprintf( stderr, "tmp: %f %f\n", tmp[0], tmp[1] );

        alpha[0] = sig_new[0] / tmp[0];
        alpha[1] = sig_new[1] / tmp[1];
        my_dot[0] = my_dot[1] = 0;

//#ifdef __CUDA_DEBUG__
//    for( j = 0; j < system->n; ++j ) {
//      // update x
//      workspace->x[j][0] += alpha[0] * workspace->d2[j][0];
//      workspace->x[j][1] += alpha[1] * workspace->d2[j][1];
//      // update residual
//      workspace->r2[j][0] -= alpha[0] * workspace->q2[j][0];
//      workspace->r2[j][1] -= alpha[1] * workspace->q2[j][1];
//      // apply diagonal pre-conditioner
//      workspace->p2[j][0] = workspace->r2[j][0] * workspace->Hdia_inv[j];
//      workspace->p2[j][1] = workspace->r2[j][1] * workspace->Hdia_inv[j];
//      // dot product: r.p
//      my_dot[0] += workspace->r2[j][0] * workspace->p2[j][0];
//      my_dot[1] += workspace->r2[j][1] * workspace->p2[j][1];
//    }
//       fprintf( stderr, "H:my_dot: %f %f\n", my_dot[0], my_dot[1] );
//#endif

        my_dot[0] = my_dot[1] = 0;
        Cuda_DualCG_Preconditioer( dev_workspace, x, alpha, system->n, my_dot );

        //fprintf( stderr, "D:my_dot: %f %f\n", my_dot[0], my_dot[1] );

//   compare_rvec2 (workspace->x, dev_workspace->x, N, "x");
//   compare_rvec2 (workspace->r2, dev_workspace->r2, N, "r2");
//   compare_rvec2 (workspace->p2, dev_workspace->p2, N, "p2");

        sig_old[0] = sig_new[0];
        sig_old[1] = sig_new[1];
        MPI_Allreduce( &my_dot, &sig_new, 2, MPI_DOUBLE, MPI_SUM, comm );

        //fprintf( stderr, "DEVICE:sig_new: %f %f\n", sig_new[0], sig_new[1] );

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &dot_time );
        }
#endif

        if ( SQRT(sig_new[0]) / b_norm[0] <= tol || SQRT(sig_new[1]) / b_norm[1] <= tol )
        {
            break;
        }

        beta[0] = sig_new[0] / sig_old[0];
        beta[1] = sig_new[1] / sig_old[1];

//#ifdef __CUDA_DEBUG__
//    for( j = 0; j < system->n; ++j ) {
//      // d = p + beta * d
//      workspace->d2[j][0] = workspace->p2[j][0] + beta[0] * workspace->d2[j][0];
//      workspace->d2[j][1] = workspace->p2[j][1] + beta[1] * workspace->d2[j][1];
//    }
//#endif

        Cuda_Vector_Sum_Rvec2( dev_workspace->d2, dev_workspace->p2, beta,
                dev_workspace->d2, system->n );

//       compare_rvec2 (workspace->d2, dev_workspace->d2, N, "q2");
    }


    if ( SQRT(sig_new[0]) / b_norm[0] <= tol )
    {
        //for( j = 0; j < n; ++j )
        //  workspace->t[j] = workspace->x[j][1];
        //fprintf (stderr, "Getting started with Cuda_CG1 \n");

        Cuda_RvecCopy_From( dev_workspace->t, dev_workspace->x, 1, system->n );

        //compare_array (workspace->b_t, dev_workspace->b_t, system->n, "b_t");
        //compare_array (workspace->t, dev_workspace->t, system->n, "t");

        matvecs = Cuda_CG( system, workspace, H, dev_workspace->b_t, tol, dev_workspace->t,
                mpi_data, fout );

        //fprintf (stderr, " Cuda_CG1: iterations --> %d \n", matvecs );
        //for( j = 0; j < n; ++j )
        //  workspace->x[j][1] = workspace->t[j];

        Cuda_RvecCopy_To( dev_workspace->x, dev_workspace->t, 1, system->n );
    }
    else if ( SQRT(sig_new[1]) / b_norm[1] <= tol )
    {
        //for( j = 0; j < n; ++j )
        //  workspace->s[j] = workspace->x[j][0];

        Cuda_RvecCopy_From( dev_workspace->s, dev_workspace->x, 0, system->n );

        //compare_array (workspace->s, dev_workspace->s, system->n, "s");
        //compare_array (workspace->b_s, dev_workspace->b_s, system->n, "b_s");

        //fprintf (stderr, "Getting started with Cuda_CG2 \n");

        matvecs = Cuda_CG( system, workspace, H, dev_workspace->b_s, tol, dev_workspace->s,
                mpi_data, fout );

        //fprintf (stderr, " Cuda_CG2: iterations --> %d \n", matvecs );
        //for( j = 0; j < system->n; ++j )
        //  workspace->x[j][0] = workspace->s[j];

        Cuda_RvecCopy_To( dev_workspace->x, dev_workspace->s, 0, system->n );
    }

    if ( i >= 300 )
    {
        fprintf( stderr, "Dual CG convergence failed! -> %d\n", i );
    }

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        fprintf( fout, "QEq %d + %d iters. matvecs: %f  dot: %f\n",
                i + 1, matvecs, matvec_time, dot_time );
    }
#endif

    return (i + 1) + matvecs;
}
#endif


void Sparse_MatVec( sparse_matrix *A, real *x, real *b, int N )
{
    int  i, j, k, si;
    real H;

    for ( i = 0; i < N; ++i )
    {
        b[i] = 0;
    }

    /* perform multiplication */
    for ( i = 0; i < A->n; ++i )
    {
        si = A->start[i];
        b[i] += A->entries[si].val * x[i];
        for ( k = si + 1; k < A->end[i]; ++k )
        {
            j = A->entries[k].j;
            H = A->entries[k].val;
            b[i] += H * x[j];
            //if( j < A->n ) // comment out for tryQEq
            b[j] += H * x[i];
        }
    }
}


int CG( reax_system *system, storage *workspace, sparse_matrix *H,
        real *b, real tol, real *x, mpi_datatypes* mpi_data, FILE *fout)
{
    int  i, j, scale;
    real tmp, alpha, beta, b_norm;
    real sig_old, sig_new, sig0;

    scale = sizeof(real) / sizeof(void);
    Dist( system, mpi_data, x, MPI_DOUBLE, scale, real_packer );
    Sparse_MatVec( H, x, workspace->q, system->N );

    // tryQEq
    Coll( system, mpi_data, workspace->q, MPI_DOUBLE, scale, real_unpacker );

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &matvec_time );
    }
#endif

    Vector_Sum( workspace->r , 1.,  b, -1., workspace->q, system->n );
    for ( j = 0; j < system->n; ++j )
    {
        workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j]; //pre-condition
    }

    b_norm = Parallel_Norm( b, system->n, mpi_data->world );
    sig_new = Parallel_Dot(workspace->r, workspace->d, system->n, mpi_data->world);
    sig0 = sig_new;

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &dot_time );
    }
#endif

    for ( i = 1; i < 300 && SQRT(sig_new) / b_norm > tol; ++i )
    {
        Dist( system, mpi_data, workspace->d, MPI_DOUBLE, scale, real_packer );
        Sparse_MatVec( H, workspace->d, workspace->q, system->N );

        //tryQEq
        Coll(system, mpi_data, workspace->q, MPI_DOUBLE, scale, real_unpacker);

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &matvec_time );
        }
#endif

        tmp = Parallel_Dot(workspace->d, workspace->q, system->n, mpi_data->world);
        alpha = sig_new / tmp;
        Vector_Add( x, alpha, workspace->d, system->n );
        Vector_Add( workspace->r, -alpha, workspace->q, system->n );
        /* pre-conditioning */
        for ( j = 0; j < system->n; ++j )
        {
            workspace->p[j] = workspace->r[j] * workspace->Hdia_inv[j];
        }

        sig_old = sig_new;
        sig_new = Parallel_Dot(workspace->r, workspace->p, system->n, mpi_data->world);
        //fprintf (stderr, "Host : sig_new: %f \n", sig_new );
        beta = sig_new / sig_old;
        Vector_Sum( workspace->d, 1., workspace->p, beta, workspace->d, system->n );

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &dot_time );
        }
#endif
    }

    if ( i >= 300 )
    {
        fprintf( stderr, "CG convergence failed!\n" );
        return i;
    }

    return i;
}


#ifdef HAVE_CUDA
int Cuda_CG( reax_system *system, storage *workspace, sparse_matrix *H,
             real *b, real tol, real *x, mpi_datatypes* mpi_data, FILE *fout )
{
    int  i, j, scale;
    real tmp, alpha, beta, b_norm;
    real sig_old, sig_new, sig0;
    real *spad = (real *) host_scratch;

    scale = sizeof(real) / sizeof(void);

    /* x is on the device */
    //MVAPICH2
    memset (spad, 0, sizeof (real) * system->total_cap);
    get_from_device (spad, x, sizeof (real) * system->total_cap, "cuda_cg:x:get");
    Dist( system, mpi_data, spad, MPI_DOUBLE, scale, real_packer );

    //MVAPICH2
    put_on_device (spad, x, sizeof (real) * system->total_cap , "cuda_cg:x:put");
    Cuda_Matvec( H, x, dev_workspace->q, system->N, system->total_cap );

    // tryQEq
    // MVAPICH2
    get_from_device (spad, dev_workspace->q, sizeof (real) * system->total_cap, "cuda_cg:q:get" );
    Coll( system, mpi_data, spad, MPI_DOUBLE, scale, real_unpacker );

    //MVAPICH2
    put_on_device (spad, dev_workspace->q, sizeof (real) * system->total_cap, "cuda_cg:q:put" );

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
        Update_Timing_Info( &t_start, &matvec_time );
#endif

    Cuda_Vector_Sum( dev_workspace->r , 1.,  b, -1., dev_workspace->q, system->n );
    //for( j = 0; j < system->n; ++j )
    //  workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j]; //pre-condition
    Cuda_CG_Preconditioner (dev_workspace->d, dev_workspace->r, dev_workspace->Hdia_inv, system->n);

    //TODO do the parallel_norm on the device for the local sum
    get_from_device (spad, b, sizeof (real) * system->n, "cuda_cg:b:get");
    b_norm = Parallel_Norm( spad, system->n, mpi_data->world );

    //TODO do the parallel dot on the device for the local sum
    get_from_device (spad, dev_workspace->r, sizeof (real) * system->total_cap, "cuda_cg:r:get");
    get_from_device (spad + system->total_cap, dev_workspace->d, sizeof (real) * system->total_cap, "cuda_cg:d:get");
    sig_new = Parallel_Dot(spad, spad + system->total_cap, system->n, mpi_data->world);

    sig0 = sig_new;

#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &dot_time );
    }
#endif

    for ( i = 1; i < 300 && SQRT(sig_new) / b_norm > tol; ++i )
    {
        //MVAPICH2
        get_from_device (spad, dev_workspace->d, sizeof (real) * system->total_cap, "cuda_cg:d:get");
        Dist( system, mpi_data, spad, MPI_DOUBLE, scale, real_packer );
        put_on_device (spad, dev_workspace->d, sizeof (real) * system->total_cap, "cuda_cg:d:put");

        Cuda_Matvec( H, dev_workspace->d, dev_workspace->q, system->N, system->total_cap );

        //tryQEq
        get_from_device (spad, dev_workspace->q, sizeof (real) * system->total_cap, "cuda_cg:q:get" );
        Coll(system, mpi_data, spad, MPI_DOUBLE, scale, real_unpacker);
        put_on_device (spad, dev_workspace->q, sizeof (real) * system->total_cap , "cuda_cg:q:get");

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &matvec_time );
        }
#endif

        //TODO do the parallel dot on the device for the local sum
        get_from_device (spad, dev_workspace->d, sizeof (real) * system->n, "cuda_cg:d:get");
        get_from_device (spad + system->n, dev_workspace->q, sizeof (real) * system->n, "cuda_cg:q:get");
        tmp = Parallel_Dot(spad, spad + system->n, system->n, mpi_data->world);

        alpha = sig_new / tmp;
        //Cuda_Vector_Add( x, alpha, dev_workspace->d, system->n );
        Cuda_Vector_Sum( x, alpha, dev_workspace->d, 1.0, x, system->n );

        //Cuda_Vector_Add( workspace->r, -alpha, workspace->q, system->n );
        Cuda_Vector_Sum( dev_workspace->r, -alpha, dev_workspace->q, 1.0, dev_workspace->r, system->n );
        /* pre-conditioning */
        //for( j = 0; j < system->n; ++j )
        //  workspace->p[j] = workspace->r[j] * workspace->Hdia_inv[j];
        Cuda_CG_Preconditioner (dev_workspace->p, dev_workspace->r, dev_workspace->Hdia_inv, system->n);

        sig_old = sig_new;

        //TODO do the parallel dot on the device for the local sum
        get_from_device (spad, dev_workspace->r, sizeof (real) * system->n, "cuda_cg:r:get");
        get_from_device (spad + system->n, dev_workspace->p, sizeof (real) * system->n, "cuda_cg:p:get");
        sig_new = Parallel_Dot(spad , spad + system->n, system->n, mpi_data->world);
        //fprintf (stderr, "Device: sig_new: %f \n", sig_new );

        beta = sig_new / sig_old;
        Cuda_Vector_Sum( dev_workspace->d, 1., dev_workspace->p, beta, dev_workspace->d, system->n );

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &dot_time );
        }
#endif
    }

    if ( i >= 300 )
    {
        fprintf( stderr, "CG convergence failed!\n" );
        return i;
    }

    return i;
}
#endif


int CG_test( reax_system *system, storage *workspace, sparse_matrix *H,
             real *b, real tol, real *x, mpi_datatypes* mpi_data, FILE *fout )
{
    int  i, j, scale;
    real tmp, alpha, beta, b_norm;
    real sig_old, sig_new, sig0;

    scale = sizeof(real) / sizeof(void);
    b_norm = Parallel_Norm( b, system->n, mpi_data->world );
#if defined(DEBUG)
    if ( system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "n=%d, N=%d\n", system->n, system->N );
        fprintf( stderr, "p%d CGinit: b_norm=%24.15e\n", system->my_rank, b_norm );
        //Vector_Print( stderr, "d", workspace->d, system->N );
        //Vector_Print( stderr, "q", workspace->q, system->N );
    }
    MPI_Barrier( mpi_data->world );
#endif

    Sparse_MatVec( H, x, workspace->q, system->N );
    //Coll( system, mpi_data, workspace->q, MPI_DOUBLE, real_unpacker );

    Vector_Sum( workspace->r , 1.,  b, -1., workspace->q, system->n );
    for ( j = 0; j < system->n; ++j )
        workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j]; //pre-condition

    sig_new = Parallel_Dot( workspace->r, workspace->d, system->n,
                            mpi_data->world );
    sig0 = sig_new;
#if defined(DEBUG)
    //if( system->my_rank == MASTER_NODE ) {
    fprintf( stderr, "p%d CG:sig_new=%24.15e,d_norm=%24.15e,q_norm=%24.15e\n",
             system->my_rank, sqrt(sig_new),
             Parallel_Norm(workspace->d, system->n, mpi_data->world),
             Parallel_Norm(workspace->q, system->n, mpi_data->world) );
    //Vector_Print( stderr, "d", workspace->d, system->N );
    //Vector_Print( stderr, "q", workspace->q, system->N );
    //}
    MPI_Barrier( mpi_data->world );
#endif

    for ( i = 1; i < 300 && SQRT(sig_new) / b_norm > tol; ++i )
    {
#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
            t_start = Get_Time( );
#endif
        Dist( system, mpi_data, workspace->d, MPI_DOUBLE, scale, real_packer );
        Sparse_MatVec( H, workspace->d, workspace->q, system->N );
        //tryQEq
        //Coll(system, mpi_data, workspace->q, MPI_DOUBLE, real_unpacker);
#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            t_elapsed = Get_Timing_Info( t_start );
            matvec_time += t_elapsed;
        }
#endif

#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
            t_start = Get_Time( );
#endif
        tmp = Parallel_Dot(workspace->d, workspace->q, system->n, mpi_data->world);
        alpha = sig_new / tmp;
#if defined(DEBUG)
        //if( system->my_rank == MASTER_NODE ){
        fprintf(stderr,
                "p%d CG iter%d:d_norm=%24.15e,q_norm=%24.15e,tmp = %24.15e\n",
                system->my_rank, i,
                //Parallel_Norm(workspace->d, system->n, mpi_data->world),
                //Parallel_Norm(workspace->q, system->n, mpi_data->world),
                Norm(workspace->d, system->n), Norm(workspace->q, system->n), tmp);
        //Vector_Print( stderr, "d", workspace->d, system->N );
        //for( j = 0; j < system->N; ++j )
        //  fprintf( stdout, "%d  %24.15e\n",
        //     system->my_atoms[j].orig_id, workspace->q[j] );
        //fprintf( stdout, "\n" );
        //}
        MPI_Barrier( mpi_data->world );
#endif

        Vector_Add( x, alpha, workspace->d, system->n );
        Vector_Add( workspace->r, -alpha, workspace->q, system->n );
        /* pre-conditioning */
        for ( j = 0; j < system->n; ++j )
            workspace->p[j] = workspace->r[j] * workspace->Hdia_inv[j];

        sig_old = sig_new;
        sig_new = Parallel_Dot(workspace->r, workspace->p, system->n, mpi_data->world);
        beta = sig_new / sig_old;
        Vector_Sum( workspace->d, 1., workspace->p, beta, workspace->d, system->n );
#if defined(DEBUG)
        if ( system->my_rank == MASTER_NODE )
            fprintf(stderr, "p%d CG iter%d: sig_new = %24.15e\n",
                    system->my_rank, i, sqrt(sig_new) );
        MPI_Barrier( mpi_data->world );
#endif
#if defined(CG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            t_elapsed = Get_Timing_Info( t_start );
            dot_time += t_elapsed;
        }
#endif
    }

#if defined(DEBUG)
    if ( system->my_rank == MASTER_NODE )
        fprintf( stderr, "CG took %d iterations\n", i );
#endif
#if defined(CG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
        fprintf( stderr, "%f  %f\n", matvec_time, dot_time );
#endif
    if ( i >= 300 )
    {
        fprintf( stderr, "CG convergence failed!\n" );
        return i;
    }

    return i;
}


void Forward_Subs( sparse_matrix *L, real *b, real *y )
{
    int i, pj, j, si, ei;
    real val;

    for ( i = 0; i < L->n; ++i )
    {
        y[i] = b[i];
        si = L->start[i];
        ei = L->end[i];
        for ( pj = si; pj < ei - 1; ++pj )
        {
            j = L->entries[pj].j;
            val = L->entries[pj].val;
            y[i] -= val * y[j];
        }
        y[i] /= L->entries[pj].val;
    }
}


void Backward_Subs( sparse_matrix *U, real *y, real *x )
{
    int i, pj, j, si, ei;
    real val;

    for ( i = U->n - 1; i >= 0; --i )
    {
        x[i] = y[i];
        si = U->start[i];
        ei = U->end[i];
        for ( pj = si + 1; pj < ei; ++pj )
        {
            j = U->entries[pj].j;
            val = U->entries[pj].val;
            x[i] -= val * x[j];
        }
        x[i] /= U->entries[si].val;
    }
}


int PCG( reax_system *system, storage *workspace,
         sparse_matrix *H, real *b, real tol,
         sparse_matrix *L, sparse_matrix *U, real *x,
         mpi_datatypes* mpi_data, FILE *fout )
{
    int  i, me, n, N, scale;
    real tmp, alpha, beta, b_norm, r_norm, sig_old, sig_new;
    MPI_Comm world;

    me = system->my_rank;
    n = system->n;
    N = system->N;
    world = mpi_data->world;
    scale = sizeof(real) / sizeof(void);
    b_norm = Parallel_Norm( b, n, world );
#if defined(DEBUG_FOCUS)
    if ( me == MASTER_NODE )
    {
        fprintf( stderr, "init_PCG: n=%d, N=%d\n", n, N );
        fprintf( stderr, "init_PCG: |b|=%24.15e\n", b_norm );
    }
    MPI_Barrier( world );
#endif

    Sparse_MatVec( H, x, workspace->q, N );
    //Coll( system, workspace, mpi_data, workspace->q );
    Vector_Sum( workspace->r , 1.,  b, -1., workspace->q, n );
    r_norm = Parallel_Norm( workspace->r, n, world );

    Forward_Subs( L, workspace->r, workspace->d );
    Backward_Subs( U, workspace->d, workspace->p );
    sig_new = Parallel_Dot( workspace->r, workspace->p, n, world );
#if defined(DEBUG_FOCUS)
    if ( me == MASTER_NODE )
    {
        fprintf( stderr, "init_PCG: sig_new=%.15e\n", r_norm );
        fprintf( stderr, "init_PCG: |d|=%.15e |q|=%.15e\n",
                 Parallel_Norm(workspace->d, n, world),
                 Parallel_Norm(workspace->q, n, world) );
    }
    MPI_Barrier( world );
#endif

    for ( i = 1; i < 100 && r_norm / b_norm > tol; ++i )
    {
        Dist( system, mpi_data, workspace->p, MPI_DOUBLE, scale, real_packer );
        Sparse_MatVec( H, workspace->p, workspace->q, N );
        // tryQEq
        //Coll(system,mpi_data,workspace->q, MPI_DOUBLE, real_unpacker);
        tmp = Parallel_Dot( workspace->q, workspace->p, n, world );
        alpha = sig_new / tmp;
        Vector_Add( x, alpha, workspace->p, n );
#if defined(DEBUG_FOCUS)
        if ( me == MASTER_NODE )
            fprintf(stderr, "iter%d: |p|=%.15e |q|=%.15e tmp=%.15e\n",
                    i, Parallel_Norm(workspace->p, n, world),
                    Parallel_Norm(workspace->q, n, world), tmp );
        MPI_Barrier( world );
#endif

        Vector_Add( workspace->r, -alpha, workspace->q, n );
        r_norm = Parallel_Norm( workspace->r, n, world );
#if defined(DEBUG_FOCUS)
        if ( me == MASTER_NODE )
            fprintf( stderr, "iter%d: res=%.15e\n", i, r_norm );
        MPI_Barrier( world );
#endif

        Forward_Subs( L, workspace->r, workspace->d );
        Backward_Subs( U, workspace->d, workspace->d );
        sig_old = sig_new;
        sig_new = Parallel_Dot( workspace->r, workspace->d, n, world );
        beta = sig_new / sig_old;
        Vector_Sum( workspace->p, 1., workspace->d, beta, workspace->p, n );
    }

#if defined(DEBUG_FOCUS)
    if ( me == MASTER_NODE )
        fprintf( stderr, "PCG took %d iterations\n", i );
#endif
    if ( i >= 100 )
        fprintf( stderr, "PCG convergence failed!\n" );

    return i;
}


#if defined(OLD_STUFF)
int sCG( reax_system *system, storage *workspace, sparse_matrix *H,
         real *b, real tol, real *x, mpi_datatypes* mpi_data, FILE *fout )
{
    int  i, j;
    real tmp, alpha, beta, b_norm;
    real sig_old, sig_new, sig0;

    b_norm = Norm( b, system->n );
#if defined(DEBUG)
    if ( system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "n=%d, N=%d\n", system->n, system->N );
        fprintf( stderr, "p%d CGinit: b_norm=%24.15e\n", system->my_rank, b_norm );
        //Vector_Print( stderr, "d", workspace->d, system->N );
        //Vector_Print( stderr, "q", workspace->q, system->N );
    }
    MPI_Barrier( mpi_data->world );
#endif

    Sparse_MatVec( H, x, workspace->q, system->N );
    //Coll_Vector( system, workspace, mpi_data, workspace->q );

    Vector_Sum( workspace->r , 1.,  b, -1., workspace->q, system->n );
    for ( j = 0; j < system->n; ++j )
        workspace->d[j] = workspace->r[j] * workspace->Hdia_inv[j]; //pre-condition

    sig_new = Dot( workspace->r, workspace->d, system->n );
    sig0 = sig_new;
#if defined(DEBUG)
    if ( system->my_rank == MASTER_NODE )
    {
        fprintf( stderr, "p%d CGinit:sig_new=%24.15e\n", system->my_rank, sig_new );
        //Vector_Print( stderr, "d", workspace->d, system->N );
        //Vector_Print( stderr, "q", workspace->q, system->N );
    }
    MPI_Barrier( mpi_data->world );
#endif

    for ( i = 1; i < 100 && SQRT(sig_new) / b_norm > tol; ++i )
    {
        //Dist_Vector( system, mpi_data, workspace->d );
        Sparse_MatVec( H, workspace->d, workspace->q, system->N );
        //Coll_Vector( system, workspace, mpi_data, workspace->q );

        tmp = Dot( workspace->d, workspace->q, system->n );
        alpha = sig_new / tmp;
#if defined(DEBUG)
        if ( system->my_rank == MASTER_NODE )
        {
            fprintf(stderr,
                    "p%d CG iter%d:d_norm=%24.15e,q_norm=%24.15e,tmp = %24.15e\n",
                    system->my_rank, i,
                    Parallel_Norm(workspace->d, system->n, mpi_data->world),
                    Parallel_Norm(workspace->q, system->n, mpi_data->world), tmp );
            //Vector_Print( stderr, "d", workspace->d, system->N );
            //Vector_Print( stderr, "q", workspace->q, system->N );
        }
        MPI_Barrier( mpi_data->world );
#endif

        Vector_Add( x, alpha, workspace->d, system->n );
        Vector_Add( workspace->r, -alpha, workspace->q, system->n );
        /* pre-conditioning */
        for ( j = 0; j < system->n; ++j )
            workspace->p[j] = workspace->r[j] * workspace->Hdia_inv[j];

        sig_old = sig_new;
        sig_new = Dot( workspace->r, workspace->p, system->n );

        beta = sig_new / sig_old;
        Vector_Sum( workspace->d, 1., workspace->p, beta, workspace->d, system->n );
#if defined(DEBUG)
        if ( system->my_rank == MASTER_NODE )
            fprintf(stderr, "p%d CG iter%d: sig_new = %24.15e\n",
                    system->my_rank, i, sig_new );
        MPI_Barrier( mpi_data->world );
#endif
    }

#if defined(DEBUG)
    if ( system->my_rank == MASTER_NODE )
        fprintf( stderr, "CG took %d iterations\n", i );
#endif
    if ( i >= 100 )
    {
        fprintf( stderr, "CG convergence failed!\n" );
        return i;
    }

    return i;
}


int GMRES( reax_system *system, storage *workspace, sparse_matrix *H,
           real *b, real tol, real *x, mpi_datatypes* mpi_data, FILE *fout )
{
    int i, j, k, itr, N;
    real cc, tmp1, tmp2, temp, bnorm;

    N = system->N;
    bnorm = Norm( b, N );

    /* apply the diagonal pre-conditioner to rhs */
    for ( i = 0; i < N; ++i )
        workspace->b_prc[i] = b[i] * workspace->Hdia_inv[i];

    /* GMRES outer-loop */
    for ( itr = 0; itr < MAX_ITR; ++itr )
    {
        /* calculate r0 */
        Sparse_MatVec( H, x, workspace->b_prm, N );
        for ( i = 0; i < N; ++i )
            workspace->b_prm[i] *= workspace->Hdia_inv[i]; // pre-conditioner

        Vector_Sum( workspace->v[0],
                    1.,  workspace->b_prc, -1., workspace->b_prm, N );
        workspace->g[0] = Norm( workspace->v[0], N );
        Vector_Scale( workspace->v[0],
                      1. / workspace->g[0], workspace->v[0], N );

        // fprintf( stderr, "%10.6f\n", workspace->g[0] );

        /* GMRES inner-loop */
        for ( j = 0; j < RESTART && fabs(workspace->g[j]) / bnorm > tol; j++ )
        {
            /* matvec */
            Sparse_MatVec( H, workspace->v[j], workspace->v[j + 1], N );

            for ( k = 0; k < N; ++k )
                workspace->v[j + 1][k] *= workspace->Hdia_inv[k]; // pre-conditioner
            // fprintf( stderr, "%d-%d: matvec done.\n", itr, j );

            /* apply modified Gram-Schmidt to orthogonalize the new residual */
            for ( i = 0; i <= j; i++ )
            {
                workspace->h[i][j] = Dot(workspace->v[i], workspace->v[j + 1], N);
                Vector_Add( workspace->v[j + 1],
                            -workspace->h[i][j], workspace->v[i], N );
            }

            workspace->h[j + 1][j] = Norm( workspace->v[j + 1], N );
            Vector_Scale( workspace->v[j + 1],
                          1. / workspace->h[j + 1][j], workspace->v[j + 1], N );
            // fprintf(stderr, "%d-%d: orthogonalization completed.\n", itr, j);

            /* Givens rotations on the H matrix to make it U */
            for ( i = 0; i <= j; i++ )
            {
                if ( i == j )
                {
                    cc = SQRT(SQR(workspace->h[j][j]) + SQR(workspace->h[j + 1][j]));
                    workspace->hc[j] = workspace->h[j][j] / cc;
                    workspace->hs[j] = workspace->h[j + 1][j] / cc;
                }

                tmp1 =  workspace->hc[i] * workspace->h[i][j] +
                        workspace->hs[i] * workspace->h[i + 1][j];
                tmp2 = -workspace->hs[i] * workspace->h[i][j] +
                       workspace->hc[i] * workspace->h[i + 1][j];

                workspace->h[i][j] = tmp1;
                workspace->h[i + 1][j] = tmp2;
            }

            /* apply Givens rotations to the rhs as well */
            tmp1 =  workspace->hc[j] * workspace->g[j];
            tmp2 = -workspace->hs[j] * workspace->g[j];
            workspace->g[j] = tmp1;
            workspace->g[j + 1] = tmp2;

            // fprintf( stderr, "%10.6f\n", fabs(workspace->g[j+1]) );
        }

        /* solve Hy = g.
           H is now upper-triangular, do back-substitution */
        for ( i = j - 1; i >= 0; i-- )
        {
            temp = workspace->g[i];
            for ( k = j - 1; k > i; k-- )
                temp -= workspace->h[i][k] * workspace->y[k];
            workspace->y[i] = temp / workspace->h[i][i];
        }

        /* update x = x_0 + Vy */
        for ( i = 0; i < j; i++ )
            Vector_Add( x, workspace->y[i], workspace->v[i], N );

        /* stopping condition */
        if ( fabs(workspace->g[j]) / bnorm <= tol )
            break;
    }

    /*Sparse_MatVec( system, H, x, workspace->b_prm, mpi_data );
      for( i = 0; i < N; ++i )
      workspace->b_prm[i] *= workspace->Hdia_inv[i];

      fprintf( fout, "\n%10s%15s%15s\n", "b_prc", "b_prm", "x" );
      for( i = 0; i < N; ++i )
      fprintf( fout, "%10.5f%15.12f%15.12f\n",
      workspace->b_prc[i], workspace->b_prm[i], x[i] );*/

    fprintf( fout, "GMRES outer: %d, inner: %d - |rel residual| = %15.10f\n",
             itr, j, fabs( workspace->g[j] ) / bnorm );

    if ( itr >= MAX_ITR )
    {
        fprintf( stderr, "GMRES convergence failed\n" );
        return FAILURE;
    }

    return SUCCESS;
}


int GMRES_HouseHolder( reax_system *system, storage *workspace,
                       sparse_matrix *H, real *b, real tol, real *x,
                       mpi_datatypes* mpi_data, FILE *fout )
{
    int  i, j, k, itr, N;
    real cc, tmp1, tmp2, temp, bnorm;
    real v[10000], z[RESTART + 2][10000], w[RESTART + 2];
    real u[RESTART + 2][10000];

    N = system->N;
    bnorm = Norm( b, N );

    /* apply the diagonal pre-conditioner to rhs */
    for ( i = 0; i < N; ++i )
        workspace->b_prc[i] = b[i] * workspace->Hdia_inv[i];

    /* GMRES outer-loop */
    for ( itr = 0; itr < MAX_ITR; ++itr )
    {
        /* compute z = r0 */
        Sparse_MatVec( H, x, workspace->b_prm, N );

        for ( i = 0; i < N; ++i )
            workspace->b_prm[i] *= workspace->Hdia_inv[i]; /* pre-conditioner */

        Vector_Sum( z[0], 1.,  workspace->b_prc, -1., workspace->b_prm, N );

        Vector_MakeZero( w, RESTART + 1 );
        w[0] = Norm( z[0], N );

        Vector_Copy( u[0], z[0], N );
        u[0][0] += ( u[0][0] < 0.0 ? -1 : 1 ) * w[0];
        Vector_Scale( u[0], 1 / Norm( u[0], N ), u[0], N );

        w[0]    *= ( u[0][0] < 0.0 ?  1 : -1 );
        // fprintf( stderr, "\n\n%12.6f\n", w[0] );

        /* GMRES inner-loop */
        for ( j = 0; j < RESTART && fabs( w[j] ) / bnorm > tol; j++ )
        {
            /* compute v_j */
            Vector_Scale( z[j], -2 * u[j][j], u[j], N );
            z[j][j] += 1.; /* due to e_j */

            for ( i = j - 1; i >= 0; --i )
                Vector_Add( z[j] + i, -2 * Dot( u[i] + i, z[j] + i, N - i ), u[i] + i, N - i );

            /* matvec */
            Sparse_MatVec( H, z[j], v, N );

            for ( k = 0; k < N; ++k )
                v[k] *= workspace->Hdia_inv[k]; /* pre-conditioner */

            for ( i = 0; i <= j; ++i )
                Vector_Add( v + i, -2 * Dot( u[i] + i, v + i, N - i ), u[i] + i, N - i );

            if ( !Vector_isZero( v + (j + 1), N - (j + 1) ) )
            {
                /* compute the HouseHolder unit vector u_j+1 */
                for ( i = 0; i <= j; ++i )
                    u[j + 1][i] = 0;

                Vector_Copy( u[j + 1] + (j + 1), v + (j + 1), N - (j + 1) );

                u[j + 1][j + 1] +=
                    ( v[j + 1] < 0.0 ? -1 : 1 ) * Norm( v + (j + 1), N - (j + 1) );

                Vector_Scale( u[j + 1], 1 / Norm( u[j + 1], N ), u[j + 1], N );

                /* overwrite v with P_m+1 * v */
                v[j + 1] -=
                    2 * Dot( u[j + 1] + (j + 1), v + (j + 1), N - (j + 1) ) * u[j + 1][j + 1];
                Vector_MakeZero( v + (j + 2), N - (j + 2) );
            }


            /* previous Givens rotations on H matrix to make it U */
            for ( i = 0; i < j; i++ )
            {
                tmp1 =  workspace->hc[i] * v[i] + workspace->hs[i] * v[i + 1];
                tmp2 = -workspace->hs[i] * v[i] + workspace->hc[i] * v[i + 1];

                v[i]   = tmp1;
                v[i + 1] = tmp2;
            }

            /* apply the new Givens rotation to H and right-hand side */
            if ( fabs(v[j + 1]) >= ALMOST_ZERO )
            {
                cc = SQRT( SQR( v[j] ) + SQR( v[j + 1] ) );
                workspace->hc[j] = v[j] / cc;
                workspace->hs[j] = v[j + 1] / cc;

                tmp1 =  workspace->hc[j] * v[j] + workspace->hs[j] * v[j + 1];
                tmp2 = -workspace->hs[j] * v[j] + workspace->hc[j] * v[j + 1];

                v[j]   = tmp1;
                v[j + 1] = tmp2;

                /* Givens rotations to rhs */
                tmp1 =  workspace->hc[j] * w[j];
                tmp2 = -workspace->hs[j] * w[j];
                w[j]   = tmp1;
                w[j + 1] = tmp2;
            }

            /* extend R */
            for ( i = 0; i <= j; ++i )
                workspace->h[i][j] = v[i];


            // fprintf( stderr, "h:" );
            // for( i = 0; i <= j+1 ; ++i )
            // fprintf( stderr, "%.6f ", h[i][j] );
            // fprintf( stderr, "\n" );
            // fprintf( stderr, "%12.6f\n", w[j+1] );
        }


        /* solve Hy = w.
           H is now upper-triangular, do back-substitution */
        for ( i = j - 1; i >= 0; i-- )
        {
            temp = w[i];
            for ( k = j - 1; k > i; k-- )
                temp -= workspace->h[i][k] * workspace->y[k];

            workspace->y[i] = temp / workspace->h[i][i];
        }

        for ( i = j - 1; i >= 0; i-- )
            Vector_Add( x, workspace->y[i], z[i], N );

        /* stopping condition */
        if ( fabs( w[j] ) / bnorm <= tol )
            break;
    }

    // Sparse_MatVec( system, H, x, workspace->b_prm );
    // for( i = 0; i < N; ++i )
    // workspace->b_prm[i] *= workspace->Hdia_inv[i];

    // fprintf( fout, "\n%10s%15s%15s\n", "b_prc", "b_prm", "x" );
    // for( i = 0; i < N; ++i )
    // fprintf( fout, "%10.5f%15.12f%15.12f\n",
    //          workspace->b_prc[i], workspace->b_prm[i], x[i] );

    fprintf( fout, "GMRES outer:%d  inner:%d iters, |rel residual| = %15.10f\n",
             itr, j, fabs( workspace->g[j] ) / bnorm );

    if ( itr >= MAX_ITR )
    {
        fprintf( stderr, "GMRES convergence failed\n" );
        return FAILURE;
    }

    return SUCCESS;
}
#endif
