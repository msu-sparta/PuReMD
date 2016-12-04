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

#include "cuda_lookup.h"

#include "index_utils.h"

#include "cuda_utils.h"
#include "cuda_two_body_interactions.h"


/* Fills solution into x. Warning: will modify c and d! */
DEVICE void Tridiagonal_Solve( const real *a, const real *b,
        real *c, real *d, real *x, unsigned int n){
    int i;
    real id;

    /* Modify the coefficients. */
    c[0] /= b[0];    /* Division by zero risk. */
    d[0] /= b[0];    /* Division by zero would imply a singular matrix. */
    for(i = 1; i < n; i++){
        id = (b[i] - c[i-1] * a[i]);  /* Division by zero risk. */
        c[i] /= id;            /* Last value calculated is redundant. */
        d[i] = (d[i] - d[i-1] * a[i])/id;
    }

    /* Now back substitute. */
    x[n - 1] = d[n - 1];
    for(i = n - 2; i >= 0; i--)
        x[i] = d[i] - c[i] * x[i + 1];
}


GLOBAL void Cuda_Tridiagonal_Solve (const real *a, const real *b, 
        real *c, real *d, real *x, unsigned int n)
{
    Tridiagonal_Solve ( a, b, c, d, x, n );
}


GLOBAL void cubic_spline_init_a ( real *a, const real *h, int n )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    if (i == 0 || i == 1 || i == (n-1)) {
        a[i] = 0;
    } else {
        a[i] = h[i-1];
    }
}


GLOBAL void cubic_spline_init_b (real *b, const real *h, int n )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    if (i == 0 || i == (n-1)) {
        b[i] = 0;
    } else {
        b[i] = 2 * (h[i-1] + h[i]);
    }
}


GLOBAL void cubic_spline_init_c (real *c, const real *h, int n )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    if (i == 0 || i == (n-1) || i == (n-2)) {
        c[i] = 0;
    } else {
        c[i] = h[i];
    }
}


GLOBAL void cubic_spline_init_d (real *d, const real *f, const real *h, int n )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    if ( i == 0 || i == (n-1) ) {
        d[i] = 0;
    } else {
        d[i] = 6 * ((f[i+1]-f[i])/h[i] - (f[i]-f[i-1])/h[i-1]);
    }
}


GLOBAL void calculate_cubic_spline_coef ( const real *f, real *v, const real *h, LR_lookup_table *data, int offset, int n )
{
    cubic_spline_coef *coef;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || i == 0) return;

    if (offset == SPLINE_H_OFFSET)
        coef = &data->H[1];
    else if(offset == SPLINE_CEVD_OFFSET)
        coef = &data->CEvd[1];
    else if (offset == SPLINE_CECLMB_OFFSET)
        coef = &data->CEclmb[1];
    else if (offset == SPLINE_VDW_OFFSET)
        coef = &data->vdW[1];
    else if (offset == SPLINE_ELE_OFFSET)
        coef = &data->ele[1];
    else
        coef = 0;

    coef[i-1].d = (v[i] - v[i-1]) / (6*h[i-1]);
    coef[i-1].c = v[i]/2;
    coef[i-1].b = (f[i]-f[i-1])/h[i-1] + h[i-1]*(2*v[i] + v[i-1])/6;
    coef[i-1].a = f[i];
}


void Cuda_Natural_Cubic_Spline( const real *h, const real *f, 
        LR_lookup_table *data, int offset, unsigned int n )
{
    int i;
    real *a, *b, *c, *d, *v;
    int blocks, block_size;

    ////fprintf (stderr, "Entering Cuda_Natural_Cubic_Spline ... \n");

    /* allocate space for the linear system */
    cuda_malloc ((void **) &a, REAL_SIZE * n, 0, __LINE__ );
    cuda_malloc ((void **) &b, REAL_SIZE * n, 0, __LINE__ );
    cuda_malloc ((void **) &c, REAL_SIZE * n, 0, __LINE__ );
    cuda_malloc ((void **) &d, REAL_SIZE * n, 0, __LINE__ );
    cuda_malloc ((void **) &v, REAL_SIZE * n, 1, __LINE__ );

    ////fprintf (stderr, "Mem allocation done... \n");

    /* build linear system */
    compute_blocks ( &blocks, &block_size, n);
    cubic_spline_init_a <<< blocks, block_size >>>
        ( a, h, n );
    cudaThreadSynchronize ();
    ////fprintf (stderr, "cubic_spline_init_a done.... -> %d \n", cudaGetLastError ());

    cubic_spline_init_b <<< blocks, block_size >>>
        ( b, h, n );
    cudaThreadSynchronize ();
    ////fprintf (stderr, "cubic_spline_init_b done.... -> %d \n", cudaGetLastError ());

    cubic_spline_init_c <<< blocks, block_size >>>
        ( c, h, n );
    cudaThreadSynchronize ();
    //fprintf (stderr, "cubic_spline_init_c done.... -> %d \n", cudaGetLastError ());

    cubic_spline_init_d <<< blocks, block_size >>>
        ( d, f, h, n );
    cudaThreadSynchronize ();
    //fprintf (stderr, "cubic_spline_init_d done.... -> %d \n", cudaGetLastError ());

    /*//fprintf( stderr, "i  a        b        c        d\n" );
      for( i = 0; i < n; ++i )
    //fprintf( stderr, "%d  %f  %f  %f  %f\n", i, a[i], b[i], c[i], d[i] );*/

    Cuda_Tridiagonal_Solve <<<1, 1>>>
        ( &(a[1]), &(b[1]), &(c[1]), &(d[1]), &(v[1]), n-2 );
    cudaThreadSynchronize ();
    //fprintf (stderr, "Tridiagonal_Solve done.... -> %d \n", cudaGetLastError ());

    calculate_cubic_spline_coef <<< blocks, block_size >>>
        ( f, v, h, data,offset, n );
    cudaThreadSynchronize ();
    //fprintf (stderr, "calculate_cubic_spline_coef done.... -> %d \n", cudaGetLastError ());

    /*//fprintf( stderr, "i  v  coef\n" );
      for( i = 0; i < n; ++i )
    //fprintf( stderr, "%d  %f  %f  %f  %f  %f\n", 
    i, v[i], coef[i].a, coef[i].b, coef[i].c, coef[i].d ); */
}


GLOBAL void complete_cubic_spline_init_a (real *a, const real *h, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    if (i == 0) a[0] = 0;
    else {
        a[i] = h[i];
    }
}


GLOBAL void complete_cubic_spline_init_b (real *b, const real *h, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    if (i == 0) b[0] = 2 * h[0];
    else {
        b[i] = 2 * (h[i-1] + h[i]); 
    }
}


GLOBAL void complete_cubic_spline_init_c (real *c, const real *h, int n )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    if (i == (n-1)) c[n-1] = 0;
    else {
        c[i] = h[i];
    }
}


GLOBAL void complete_cubic_spline_init_d (real *d, const real *f, const real *h, int v0_r, int vlast_r, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    real v0, vlast;
    if ( i >= n ) return;

    v0 = 0;
    vlast = 0;

    if (i == 0) {
        d[0] = 6 * (f[1]-f[0])/h[0] - 6 * v0;   
    }
    else if (i == (n-1)) {
        d[n-1] = 6 * vlast - 6 * (f[n-1]-f[n-2]/h[n-2]);
    }
    else
        d[i] = 6 * ((f[i+1]-f[i])/h[i] - (f[i]-f[i-1])/h[i-1]);
}


GLOBAL void calculate_complete_cubic_spline_coef (LR_lookup_table *data, int offset, real *v, const real *h, const real *f, int n)
{

    cubic_spline_coef *coef;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    if (offset == SPLINE_H_OFFSET)
        coef = &data->H[1];
    else if(offset == SPLINE_CEVD_OFFSET)
        coef = &data->CEvd[1];
    else if (offset == SPLINE_CECLMB_OFFSET)
        coef = &data->CEclmb[1];
    else if (offset == SPLINE_VDW_OFFSET)
        coef = &data->vdW[1];
    else if (offset == SPLINE_ELE_OFFSET)
        coef = &data->ele[1];
    else
        coef = 0;

    coef[i-1].d = (v[i] - v[i-1]) / (6*h[i-1]);
    coef[i-1].c = v[i]/2;
    coef[i-1].b = (f[i]-f[i-1])/h[i-1] + h[i-1]*(2*v[i] + v[i-1])/6;
    coef[i-1].a = f[i];
}


void Cuda_Complete_Cubic_Spline( const real *h, const real *f, int v0_r, int vlast_r,
        LR_lookup_table *data, int offset, unsigned int n )
{
    int i;
    real *a, *b, *c, *d, *v;

    int blocks, block_size;

    /* allocate space for the linear system */
    cuda_malloc ((void **) &a, REAL_SIZE * n, 0, __LINE__ );
    cuda_malloc ((void **) &b, REAL_SIZE * n, 0, __LINE__ );
    cuda_malloc ((void **) &c, REAL_SIZE * n, 0, __LINE__ );
    cuda_malloc ((void **) &d, REAL_SIZE * n, 0, __LINE__ );
    cuda_malloc ((void **) &v, REAL_SIZE * n, 1, __LINE__ );

    /* build the linear system */
    compute_blocks ( &blocks, &block_size, n );

    complete_cubic_spline_init_a <<< blocks, block_size >>>
        (a, h, n);
    cudaThreadSynchronize ();
    //fprintf (stderr, "complete_cubic_spline_init_a done.... -> %d \n", cudaGetLastError ());

    complete_cubic_spline_init_b <<< blocks, block_size >>>
        (b, h, n);
    cudaThreadSynchronize ();
    //fprintf (stderr, "complete_cubic_spline_init_b done.... -> %d \n", cudaGetLastError ());

    complete_cubic_spline_init_c <<< blocks, block_size >>>
        ( c, h, n );
    cudaThreadSynchronize ();
    //fprintf (stderr, "complete_cubic_spline_init_c done.... -> %d \n", cudaGetLastError ());

    complete_cubic_spline_init_d <<< blocks, block_size >>>
        (d, f, h, v0_r, vlast_r, n);
    cudaThreadSynchronize ();
    //fprintf (stderr, "complete_cubic_spline_init_d done.... -> %d \n", cudaGetLastError ());

    /*//fprintf( stderr, "i  a        b        c        d\n" );
      for( i = 0; i < n; ++i )
    //fprintf( stderr, "%d  %f  %f  %f  %f\n", i, a[i], b[i], c[i], d[i] );*/


    Cuda_Tridiagonal_Solve <<< 1, 1 >>>
        ( &(a[0]), &(b[0]), &(c[0]), &(d[0]), &(v[0]), n );
    cudaThreadSynchronize ();
    //fprintf (stderr, "Tridiagonal_Solve done.... -> %d \n", cudaGetLastError ());
    // Tridiagonal_Solve( &(a[1]), &(b[1]), &(c[1]), &(d[1]), &(v[1]), n-2 );


    calculate_complete_cubic_spline_coef <<< blocks, block_size >>>
        (data, offset, v, h, f, n);
    cudaThreadSynchronize ();
    //fprintf (stderr, " calculate_complete_cubic_spline_coef done.... -> %d \n", cudaGetLastError ());

    /*//fprintf( stderr, "i  v  coef\n" );
      for( i = 0; i < n; ++i )
    //fprintf( stderr, "%d  %f  %f  %f  %f  %f\n", 
    i, v[i], coef[i].a, coef[i].b, coef[i].c, coef[i].d ); */
}


void copy_LR_table_to_device (reax_system *system, control_params *control)
{
    int i, j, r;
    int num_atom_types;
    int existing_types[MAX_ATOM_TYPES];
    LR_data *d_y;
    cubic_spline_coef *temp;

    num_atom_types = system->reaxprm.num_atom_types;

    //fprintf (stderr, "Copying the LR Lookyp Table to the device ... \n");

    cuda_malloc ((void **) &d_LR, LR_LOOKUP_TABLE_SIZE * ( num_atom_types * num_atom_types ), 0, RES_LR_LOOKUP_TABLE );

    for( i = 0; i < MAX_ATOM_TYPES; ++i )
        existing_types[i] = 0;

    for( i = 0; i < system->N; ++i )
        existing_types[ system->atoms[i].type ] = 1;

    copy_host_device ( LR, d_LR, LR_LOOKUP_TABLE_SIZE * (num_atom_types * num_atom_types), cudaMemcpyHostToDevice, RES_LR_LOOKUP_TABLE );

    for( i = 0; i < num_atom_types; ++i )
        if( existing_types[i] )
            for( j = i; j < num_atom_types; ++j )

                if( existing_types[j] ) {

                    cuda_malloc ((void **) &d_y, LR_DATA_SIZE * (control->tabulate + 1), 0, RES_LR_LOOKUP_Y );
                    copy_host_device ( LR [ index_lr (i, j, num_atom_types) ].y, d_y, LR_DATA_SIZE * (control->tabulate + 1), cudaMemcpyHostToDevice, RES_LR_LOOKUP_Y );
                    copy_host_device ( &d_y, &d_LR [ index_lr (i, j, num_atom_types) ].y, LR_DATA_PTR_SIZE, cudaMemcpyHostToDevice, RES_LR_LOOKUP_Y );

                    cuda_malloc ((void **) &temp, CUBIC_SPLINE_COEF_SIZE * (control->tabulate + 1), 0, RES_LR_LOOKUP_H );
                    copy_host_device ( LR [ index_lr (i, j, num_atom_types) ].H, temp, CUBIC_SPLINE_COEF_SIZE * (control->tabulate + 1), cudaMemcpyHostToDevice, RES_LR_LOOKUP_H );
                    copy_host_device ( &temp, &d_LR [ index_lr (i, j, num_atom_types) ].H, CUBIC_SPLINE_COEF_PTR_SIZE, cudaMemcpyHostToDevice, RES_LR_LOOKUP_H );

                    cuda_malloc ((void **) &temp, CUBIC_SPLINE_COEF_SIZE * (control->tabulate + 1), 0, RES_LR_LOOKUP_VDW );
                    copy_host_device ( LR [ index_lr (i, j, num_atom_types) ].vdW, temp, CUBIC_SPLINE_COEF_SIZE * (control->tabulate + 1), cudaMemcpyHostToDevice, RES_LR_LOOKUP_VDW );
                    copy_host_device ( &temp, &d_LR [ index_lr (i, j, num_atom_types) ].vdW,CUBIC_SPLINE_COEF_PTR_SIZE, cudaMemcpyHostToDevice, RES_LR_LOOKUP_VDW );

                    cuda_malloc ((void **) &temp, CUBIC_SPLINE_COEF_SIZE * (control->tabulate + 1), 0, RES_LR_LOOKUP_CEVD );
                    copy_host_device ( LR [ index_lr (i, j, num_atom_types) ].CEvd, temp, CUBIC_SPLINE_COEF_SIZE * (control->tabulate + 1), cudaMemcpyHostToDevice, RES_LR_LOOKUP_CEVD );
                    copy_host_device ( &temp, &d_LR [ index_lr (i, j, num_atom_types) ].CEvd, CUBIC_SPLINE_COEF_PTR_SIZE, cudaMemcpyHostToDevice, RES_LR_LOOKUP_CEVD );

                    cuda_malloc ((void **) &temp, CUBIC_SPLINE_COEF_SIZE * (control->tabulate + 1), 0, RES_LR_LOOKUP_ELE );
                    copy_host_device ( LR [ index_lr (i, j, num_atom_types) ].ele, temp, CUBIC_SPLINE_COEF_SIZE * (control->tabulate + 1), cudaMemcpyHostToDevice, RES_LR_LOOKUP_ELE );
                    copy_host_device ( &temp, &d_LR [ index_lr (i, j, num_atom_types) ].ele, CUBIC_SPLINE_COEF_PTR_SIZE, cudaMemcpyHostToDevice, RES_LR_LOOKUP_ELE );

                    cuda_malloc ((void **) &temp, CUBIC_SPLINE_COEF_SIZE * (control->tabulate + 1), 0, RES_LR_LOOKUP_CECLMB );
                    copy_host_device ( LR [ index_lr (i, j, num_atom_types) ].CEclmb, temp, CUBIC_SPLINE_COEF_SIZE * (control->tabulate + 1), cudaMemcpyHostToDevice, RES_LR_LOOKUP_CECLMB );
                    copy_host_device ( &temp, &d_LR [ index_lr (i, j, num_atom_types) ].CEclmb, CUBIC_SPLINE_COEF_PTR_SIZE, cudaMemcpyHostToDevice, RES_LR_LOOKUP_CECLMB );
                }

    //fprintf (stderr, "Copy of the LR Lookup Table to the device complete ... \n");
}


GLOBAL void calculate_LR_Values ( LR_lookup_table *d_LR, real *h, real *fh, real *fvdw, real *fCEvd, real *fele, real *fCEclmb, 
        global_parameters g_params, two_body_parameters *tbp, 
        control_params *control, int i, 
        int j, int num_atom_types, LR_data *data, real dr, int count )
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if ( r == 0 || r > count ) return;

    LR_vdW_Coulomb( g_params, tbp, control, i, j, r * dr, &data[r], num_atom_types );

    h[r] = d_LR[ index_lr(i, j, num_atom_types) ].dx;
    fh[r] = d_LR[ index_lr(i, j, num_atom_types) ].y[r].H;
    fvdw[r] = d_LR[ index_lr(i, j, num_atom_types) ].y[r].e_vdW;
    fCEvd[r] = d_LR[ index_lr(i, j, num_atom_types) ].y[r].CEvd;
    fele[r] = d_LR[ index_lr(i, j, num_atom_types) ].y[r].e_ele;
    fCEclmb[r] = d_LR[ index_lr(i, j, num_atom_types) ].y[r].CEclmb;
}


GLOBAL void init_LR_values( LR_lookup_table *d_LR, control_params *control, real dr, int i, int j, int num_atom_types )
{
    d_LR[ index_lr (i, j, num_atom_types) ].xmin = 0;
    d_LR[ index_lr (i, j, num_atom_types) ].xmax = control->r_cut;
    d_LR[ index_lr (i, j, num_atom_types) ].n = control->tabulate + 1;
    d_LR[ index_lr (i, j, num_atom_types) ].dx = dr;
    d_LR[ index_lr (i, j, num_atom_types) ].inv_dx = control->tabulate / control->r_cut;
}


void Cuda_Make_LR_Lookup_Table( reax_system *system, control_params *control )
{
    int i, j, r;
    int num_atom_types;
    int existing_types[MAX_ATOM_TYPES];
    real dr;
    real *h, *fh, *fvdw, *fele, *fCEvd, *fCEclmb;

    int v0_vdw_r, v0_ele_r, vlast_vdw_r, vlast_ele_r;

    void *temp;
    LR_data *d_y;
    int blocks, block_size;

    /* initializations */
    vlast_ele_r = 0;
    vlast_vdw_r = 0;
    v0_ele_r = 0;
    v0_vdw_r = 0;

    num_atom_types = system->reaxprm.num_atom_types;
    dr = control->r_cut / control->tabulate;

    cuda_malloc ((void **) &h,             REAL_SIZE * (control->tabulate + 1), 0, RES_LR_LOOKUP_Y);
    cuda_malloc ((void **) &fh,         REAL_SIZE * (control->tabulate + 1), 0, RES_LR_LOOKUP_H);
    cuda_malloc ((void **) &fvdw,         REAL_SIZE * (control->tabulate + 1), 0, RES_LR_LOOKUP_VDW);
    cuda_malloc ((void **) &fCEvd,     REAL_SIZE * (control->tabulate + 1), 0, RES_LR_LOOKUP_CEVD);
    cuda_malloc ((void **) &fele,         REAL_SIZE * (control->tabulate + 1), 0, RES_LR_LOOKUP_ELE);
    cuda_malloc ((void **) &fCEclmb,     REAL_SIZE * (control->tabulate + 1), 0, RES_LR_LOOKUP_CECLMB);

    /* allocate Long-Range LookUp Table space based on 
       number of atom types in the ffield file */
    cuda_malloc ((void **) &d_LR, LR_LOOKUP_TABLE_SIZE * ( num_atom_types * num_atom_types ), 0, RES_LR_LOOKUP_TABLE );

    /* most atom types in ffield file will not exist in the current
       simulation. to avoid unnecessary lookup table space, determine
       the atom types that exist in the current simulation */
    for( i = 0; i < MAX_ATOM_TYPES; ++i )
        existing_types[i] = 0;

    for( i = 0; i < system->N; ++i )
        existing_types[ system->atoms[i].type ] = 1;

    /* fill in the lookup table entries for existing atom types.
       only lower half should be enough. */
    for( i = 0; i < num_atom_types; ++i )
        if( existing_types[i] )
            for( j = i; j < num_atom_types; ++j )
                if( existing_types[j] ) {

                    init_LR_values <<< 1, 1 >>>
                        ( d_LR, (control_params *)control->d_control, dr, i, j, num_atom_types );
                    cudaThreadSynchronize ();
                    //fprintf (stderr, "Done with init LR Values --> %d \n", cudaGetLastError ());

                    cuda_malloc ((void **) &d_y, LR_DATA_SIZE * (control->tabulate + 1), 0, RES_LR_LOOKUP_Y );
                    copy_host_device ( &d_y, &d_LR [ index_lr (i, j, num_atom_types) ].y, LR_DATA_PTR_SIZE, cudaMemcpyHostToDevice, RES_LR_LOOKUP_Y );

                    cuda_malloc ((void **) &temp, CUBIC_SPLINE_COEF_SIZE * (control->tabulate + 1), 0, RES_LR_LOOKUP_H );
                    copy_host_device ( &temp, &d_LR [ index_lr (i, j, num_atom_types) ].H, CUBIC_SPLINE_COEF_PTR_SIZE, cudaMemcpyHostToDevice, RES_LR_LOOKUP_H );

                    cuda_malloc ((void **) &temp, CUBIC_SPLINE_COEF_SIZE * (control->tabulate + 1), 0, RES_LR_LOOKUP_VDW );
                    copy_host_device ( &temp, &d_LR [ index_lr (i, j, num_atom_types) ].vdW,CUBIC_SPLINE_COEF_PTR_SIZE, cudaMemcpyHostToDevice, RES_LR_LOOKUP_VDW );

                    cuda_malloc ((void **) &temp, CUBIC_SPLINE_COEF_SIZE * (control->tabulate + 1), 0, RES_LR_LOOKUP_CEVD );
                    copy_host_device ( &temp, &d_LR [ index_lr (i, j, num_atom_types) ].CEvd, CUBIC_SPLINE_COEF_PTR_SIZE, cudaMemcpyHostToDevice, RES_LR_LOOKUP_CEVD );

                    cuda_malloc ((void **) &temp, CUBIC_SPLINE_COEF_SIZE * (control->tabulate + 1), 0, RES_LR_LOOKUP_ELE );
                    copy_host_device ( &temp, &d_LR [ index_lr (i, j, num_atom_types) ].ele, CUBIC_SPLINE_COEF_PTR_SIZE, cudaMemcpyHostToDevice, RES_LR_LOOKUP_ELE );

                    cuda_malloc ((void **) &temp, CUBIC_SPLINE_COEF_SIZE * (control->tabulate + 1), 0, RES_LR_LOOKUP_CECLMB );
                    copy_host_device ( &temp, &d_LR [ index_lr (i, j, num_atom_types) ].CEclmb, CUBIC_SPLINE_COEF_PTR_SIZE, cudaMemcpyHostToDevice, RES_LR_LOOKUP_CECLMB );

                    //TODO check the bounds
                    compute_blocks ( &blocks, &block_size, control->tabulate );
                    calculate_LR_Values <<<blocks, block_size>>>
                        ( d_LR, h, fh, fvdw, fCEvd, fele, fCEclmb, 
                          system->reaxprm.d_gp, system->reaxprm.d_tbp, 
                          (control_params *)control->d_control, i, j, system->reaxprm.num_atom_types, 
                          d_y, dr, control->tabulate );
                    cudaThreadSynchronize ();

                    //fprintf (stderr, "Done with LR Values Calculation --> %d \n", cudaGetLastError ());

                    /*//fprintf( stderr, "%-6s  %-6s  %-6s\n", "r", "h", "fh" );
                      for( r = 1; r <= control->tabulate; ++r )
                    //fprintf( stderr, "%f  %f  %f\n", r * dr, h[r], fh[r] ); */
                    Cuda_Natural_Cubic_Spline( h+1, fh+1, 
                            d_LR + index_lr (i,j,num_atom_types), SPLINE_H_OFFSET, control->tabulate+1 );

                    /*//fprintf( stderr, "%-6s  %-6s  %-6s\n", "r", "h", "fvdw" );
                      for( r = 1; r <= control->tabulate; ++r )
                    //fprintf( stderr, "%f  %f  %f\n", r * dr, h[r], fvdw[r] );
                    //fprintf( stderr, "v0_vdw: %f, vlast_vdw: %f\n", v0_vdw, vlast_vdw );
                     */

                    //TODO -- Pass the right v0 and vlast for the cubic spline
                    //Cuda_Complete_Cubic_Spline( &h[1], &fvdw[1], v0_vdw_r, vlast_vdw_r, 
                    //         &(LR[ index_lr (i,j,num_atom_types) ].vdW[1]), control->tabulate+1 );
                    //Cuda_Natural_Cubic_Spline( &h[1], &fCEvd[1], 
                    //        &(LR[ index_lr (i,j,num_atom_types) ].CEvd[1]), control->tabulate+1 );
                    Cuda_Complete_Cubic_Spline( &h[1], &fvdw[1], v0_vdw_r, vlast_vdw_r, 
                            d_LR + index_lr (i,j,num_atom_types) , SPLINE_VDW_OFFSET, control->tabulate+1 );
                    Cuda_Natural_Cubic_Spline( &h[1], &fCEvd[1], 
                            d_LR + index_lr (i,j,num_atom_types) , SPLINE_CEVD_OFFSET, control->tabulate+1 );

                    /*//fprintf( stderr, "%-6s  %-6s  %-6s\n", "r", "h", "fele" );
                      for( r = 1; r <= control->tabulate; ++r )
                    //fprintf( stderr, "%f  %f  %f\n", r * dr, h[r], fele[r] );
                    //fprintf( stderr, "v0_ele: %f, vlast_ele: %f\n", v0_ele, vlast_ele );
                     */
                    //Cuda_Complete_Cubic_Spline( &h[1], &fele[1], v0_ele_r, vlast_ele_r, 
                    //         &(LR[index_lr (i,j,num_atom_types) ].ele[1]), control->tabulate+1 );
                    //Cuda_Natural_Cubic_Spline( &h[1], &fCEclmb[1], 
                    //        &(LR[ index_lr (i,j,num_atom_types) ].CEclmb[1]), control->tabulate+1 );
                    Cuda_Complete_Cubic_Spline( &h[1], &fele[1], v0_ele_r, vlast_ele_r, 
                            d_LR + index_lr (i,j,num_atom_types) , SPLINE_ELE_OFFSET, control->tabulate+1 );
                    Cuda_Natural_Cubic_Spline( &h[1], &fCEclmb[1], 
                            d_LR + index_lr (i,j,num_atom_types) , SPLINE_CECLMB_OFFSET, control->tabulate+1 );
                }

    cuda_free(h, RES_LR_LOOKUP_Y);
    cuda_free(fh, RES_LR_LOOKUP_H);
    cuda_free(fvdw, RES_LR_LOOKUP_VDW);
    cuda_free(fCEvd, RES_LR_LOOKUP_CEVD);
    cuda_free(fele, RES_LR_LOOKUP_ELE);
    cuda_free(fCEclmb, RES_LR_LOOKUP_CECLMB);
}
