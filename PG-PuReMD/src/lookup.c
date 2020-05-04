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

#if defined(PURE_REAX)
  #include "lookup.h"
  #include "comm_tools.h"
  #include "nonbonded.h"
  #include "tool_box.h"
#elif defined(LAMMPS_REAX)
  #include "reax_lookup.h"
  #include "reax_comm_tools.h"
  #include "reax_nonbonded.h"
  #include "reax_tool_box.h"
#endif

#include "index_utils.h"

#if defined(HAVE_CUDA)
  #include "cuda/cuda_lookup.h"
#endif


/* Fills solution into x. Warning: will modify c and d! */
static void Tridiagonal_Solve( const real * const a, const real * const b,
        real * const c, real * const d, real * const x, unsigned int n )
{
    int i;
    real id;

    /* Modify the coefficients. */
    c[0] /= b[0]; /* Division by zero risk. */
    d[0] /= b[0]; /* Division by zero would imply a singular matrix. */
    for ( i = 1; i < n; i++ )
    {
        id = (b[i] - c[i - 1] * a[i]); /* Division by zero risk. */
        c[i] /= id;         /* Last value calculated is redundant. */
        d[i] = (d[i] - d[i - 1] * a[i]) / id;
    }

    /* Now back substitute. */
    x[n - 1] = d[n - 1];
    for ( i = n - 2; i >= 0; i-- )
    {
        x[i] = d[i] - c[i] * x[i + 1];
    }
}


static void Natural_Cubic_Spline( const real * const h, const real * const f,
        cubic_spline_coef * const coef, unsigned int n )
{
    int i;
    real *a, *b, *c, *d, *v;

    /* allocate space for the linear system */
    a = smalloc( sizeof(real) * n, "Natural_Cubic_Spline::a" );
    b = smalloc( sizeof(real) * n, "Natural_Cubic_Spline::b" );
    c = smalloc( sizeof(real) * n, "Natural_Cubic_Spline::c" );
    d = smalloc( sizeof(real) * n, "Natural_Cubic_Spline::d" );
    v = smalloc( sizeof(real) * n, "Natural_Cubic_Spline::v" );

    /* build the linear system */
    a[0] = 0.0;
    a[1] = 0.0;
    a[n - 1] = 0.0;
    for ( i = 2; i < n - 1; ++i )
    {
        a[i] = h[i - 1];
    }

    b[0] = 0.0;
    b[n - 1] = 0.0;
    for ( i = 1; i < n - 1; ++i )
    {
        b[i] = 2 * (h[i - 1] + h[i]);
    }

    c[0] = 0.0;
    c[n - 2] = 0.0;
    c[n - 1] = 0.0;
    for ( i = 1; i < n - 2; ++i )
    {
        c[i] = h[i];
    }

    d[0] = 0.0;
    d[n - 1] = 0.0;
    for ( i = 1; i < n - 1; ++i )
    {
        d[i] = 6 * ((f[i + 1] - f[i]) / h[i] - (f[i] - f[i - 1]) / h[i - 1]);
    }

    v[0] = 0;
    v[n - 1] = 0;
    Tridiagonal_Solve( &a[1], &b[1], &c[1], &d[1], &v[1], n - 2 );

    for ( i = 1; i < n; ++i )
    {
        coef[i - 1].d = (v[i] - v[i - 1]) / (6 * h[i - 1]);
        coef[i - 1].c = v[i] / 2;
        coef[i - 1].b = (f[i] - f[i - 1]) / h[i - 1] + h[i - 1] * (2 * v[i] + v[i - 1]) / 6;
        coef[i - 1].a = f[i];
    }

    sfree( a, "Natural_Cubic_Spline::a" );
    sfree( b, "Natural_Cubic_Spline::b" );
    sfree( c, "Natural_Cubic_Spline::c" );
    sfree( d, "Natural_Cubic_Spline::d" );
    sfree( v, "Natural_Cubic_Spline::v" );
}


static void Complete_Cubic_Spline( const real * const h, const real * const f,
        real v0, real vlast, cubic_spline_coef * const coef, unsigned int n )
{
    int i;
    real *a, *b, *c, *d, *v;

    a = smalloc( sizeof(real) * n, "Compute_Cubic_Spline::a" );
    b = smalloc( sizeof(real) * n, "Compute_Cubic_Spline::b" );
    c = smalloc( sizeof(real) * n, "Compute_Cubic_Spline::c" );
    d = smalloc( sizeof(real) * n, "Compute_Cubic_Spline::d" );
    v = smalloc( sizeof(real) * n, "Compute_Cubic_Spline::v" );

    /* build the linear system */
    a[0] = 0;
    for ( i = 1; i < n; ++i )
    {
        a[i] = h[i - 1];
    }

    b[0] = 2 * h[0];
    for ( i = 1; i < n; ++i )
    {
        b[i] = 2 * (h[i - 1] + h[i]);
    }

    c[n - 1] = 0;
    for ( i = 0; i < n - 1; ++i )
    {
        c[i] = h[i];
    }

    d[0] = 6 * (f[1] - f[0]) / h[0] - 6 * v0;
    d[n - 1] = 6 * vlast - 6 * (f[n - 1] - f[n - 2] / h[n - 2]);
    for ( i = 1; i < n - 1; ++i )
    {
        d[i] = 6 * ((f[i + 1] - f[i]) / h[i] - (f[i] - f[i - 1]) / h[i - 1]);
    }

    Tridiagonal_Solve( &(a[0]), &(b[0]), &(c[0]), &(d[0]), &(v[0]), n );

    for ( i = 1; i < n; ++i )
    {
        coef[i - 1].d = (v[i] - v[i - 1]) / (6 * h[i - 1]);
        coef[i - 1].c = v[i] / 2;
        coef[i - 1].b = (f[i] - f[i - 1]) / h[i - 1] + h[i - 1] * (2 * v[i] + v[i - 1]) / 6;
        coef[i - 1].a = f[i];
    }

    sfree( a, "Compute_Cubic_Spline::a" );
    sfree( b, "Compute_Cubic_Spline::b" );
    sfree( c, "Compute_Cubic_Spline::c" );
    sfree( d, "Compute_Cubic_Spline::d" );
    sfree( v, "Compute_Cubic_Spline::v" );
}


void LR_Lookup( LR_lookup_table * const t, real r, LR_data * const y )
{
    int i;
    real base, dif;

    i = (int)(r * t->inv_dx);
    if ( i == 0 )
    {
        ++i;
    }
    base = (real)(i + 1) * t->dx;
    dif = r - base;

    y->e_vdW = ((t->vdW[i].d * dif + t->vdW[i].c) * dif + t->vdW[i].b) * dif +
               t->vdW[i].a;
    y->CEvd = ((t->CEvd[i].d * dif + t->CEvd[i].c) * dif +
               t->CEvd[i].b) * dif + t->CEvd[i].a;

    y->e_ele = ((t->ele[i].d * dif + t->ele[i].c) * dif + t->ele[i].b) * dif +
               t->ele[i].a;
    y->CEclmb = ((t->CEclmb[i].d * dif + t->CEclmb[i].c) * dif + t->CEclmb[i].b) * dif +
                t->CEclmb[i].a;

    y->H = y->e_ele * EV_to_KCALpMOL / C_ELE;
}


void Init_Lookup_Tables( reax_system * const system, control_params * const control,
        storage * const workspace, mpi_datatypes * const mpi_data )
{
    int i, j, r, num_atom_types, ret;
    int existing_types[MAX_ATOM_TYPES], aggregated[MAX_ATOM_TYPES];
    real dr;
    real *h, *fh, *fvdw, *fele, *fCEvd, *fCEclmb;
    real v0_vdw, v0_ele, vlast_vdw, vlast_ele;
#if defined(HAVE_CUDA)
    real t_start, t_end;
#endif

    v0_vdw = 0;
    v0_ele = 0;
    vlast_vdw = 0;
    vlast_ele = 0;

    num_atom_types = system->reax_param.num_atom_types;
    dr = control->nonb_cut / control->tabulate;

    h = smalloc( sizeof(real) * (control->tabulate + 1), "Init_Lookup_Tables::h" );
    fh = smalloc( sizeof(real) * (control->tabulate + 1), "Init_Lookup_Tables::fh" );
    fvdw = smalloc( sizeof(real) * (control->tabulate + 1), "Init_Lookup_Tables::fvdw" );
    fCEvd = smalloc( sizeof(real) * (control->tabulate + 1), "Init_Lookup_Tables::fCEvd");
    fele = smalloc( sizeof(real) * (control->tabulate + 1), "Init_Lookup_Tables::fele" );
    fCEclmb = smalloc( sizeof(real) * (control->tabulate + 1),
            "Init_Lookup_Tables::fCEclmb" );

    /* allocate Long-Range LookUp Table space based on
     * number of atom types in the ffield file */
    workspace->LR = smalloc( sizeof(LR_lookup_table) * num_atom_types * num_atom_types,
            "Init_Lookup_Tables::LR" );

    /* most atom types in ffield file will not exist in the current
     * simulation. to avoid unnecessary lookup table space, determine
     * the atom types that exist in the current simulation */
    for ( i = 0; i < MAX_ATOM_TYPES; ++i )
    {
        existing_types[i] = 0;
    }
    for ( i = 0; i < system->n; ++i )
    {
        existing_types[ system->my_atoms[i].type ] = 1;
    }

    ret = MPI_Allreduce( existing_types, aggregated, MAX_ATOM_TYPES,
            MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    /* fill in the lookup table entries for existing atom types.
     * only lower half should be enough. */
    for ( i = 0; i < num_atom_types; ++i )
    {
        if ( aggregated[i] )
        {
            for ( j = i; j < num_atom_types; ++j )
            {
                if ( aggregated[j] )
                {

                    workspace->LR[ index_lr(i, j, num_atom_types) ].xmin = 0;
                    workspace->LR[ index_lr(i, j, num_atom_types) ].xmax = control->nonb_cut;
                    workspace->LR[ index_lr(i, j, num_atom_types) ].n = control->tabulate + 1;
                    workspace->LR[ index_lr(i, j, num_atom_types) ].dx = dr;
                    workspace->LR[ index_lr(i, j, num_atom_types) ].inv_dx = control->tabulate / control->nonb_cut;
                    workspace->LR[ index_lr(i, j, num_atom_types) ].y =
                            smalloc( workspace->LR[ sizeof(LR_data) * index_lr(i, j, num_atom_types) ].n,
                                    "Init_Lookup_Tables::LR[i,j].y" );
                    workspace->LR[ index_lr(i, j, num_atom_types) ].H =
                            smalloc( workspace->LR[ sizeof(cubic_spline_coef) * index_lr(i, j, num_atom_types) ].n,
                                    "Init_Lookup_Tables::LR[i,j].H" );
                    workspace->LR[ index_lr(i, j, num_atom_types) ].vdW =
                            smalloc( workspace->LR[ sizeof(cubic_spline_coef) * index_lr(i, j, num_atom_types) ].n,
                                    "Init_Lookup_Tables::LR[i,j].vdW" );
                    workspace->LR[ index_lr(i, j, num_atom_types) ].CEvd =
                            smalloc( workspace->LR[ sizeof(cubic_spline_coef) * index_lr(i, j, num_atom_types) ].n,
                                    "Init_Lookup_Tables::LR[i,j].CEvd" );
                    workspace->LR[ index_lr(i, j, num_atom_types) ].ele =
                            smalloc( workspace->LR[ sizeof(cubic_spline_coef) * index_lr(i, j, num_atom_types) ].n,
                                    "Init_Lookup_Tables::LR[i,j].ele" );
                    workspace->LR[ index_lr(i, j, num_atom_types) ].CEclmb =
                            smalloc( workspace->LR[ sizeof(cubic_spline_coef) * index_lr(i, j, num_atom_types) ].n,
                                    "Init_Lookup_Tables::LR[i,j].CEclmb" );

                    for ( r = 1; r <= control->tabulate; ++r )
                    {
                        LR_vdW_Coulomb( system, workspace, i, j, r * dr,
                                &workspace->LR[ index_lr(i, j, num_atom_types) ].y[r] );
                        h[r] = workspace->LR[ index_lr(i, j, num_atom_types) ].dx;
                        fh[r] = workspace->LR[ index_lr(i, j, num_atom_types) ].y[r].H;
                        fvdw[r] = workspace->LR[ index_lr(i, j, num_atom_types) ].y[r].e_vdW;
                        fCEvd[r] = workspace->LR[ index_lr(i, j, num_atom_types) ].y[r].CEvd;
                        fele[r] = workspace->LR[ index_lr(i, j, num_atom_types) ].y[r].e_ele;
                        fCEclmb[r] = workspace->LR[ index_lr(i, j, num_atom_types) ].y[r].CEclmb;

                        if ( r == 1 )
                        {
                            v0_vdw = workspace->LR[ index_lr(i, j, num_atom_types) ].y[r].CEvd;
                            v0_ele = workspace->LR[ index_lr(i, j, num_atom_types) ].y[r].CEclmb;
                        }
                        else if ( r == control->tabulate )
                        {
                            vlast_vdw = workspace->LR[ index_lr(i, j, num_atom_types) ].y[r].CEvd;
                            vlast_ele = workspace->LR[ index_lr(i, j, num_atom_types) ].y[r].CEclmb;
                        }
                    }

                    Natural_Cubic_Spline( &h[1], &fh[1],
                            &workspace->LR[ index_lr(i, j, num_atom_types) ].H[1], control->tabulate + 1 );

                    Complete_Cubic_Spline( &h[1], &fvdw[1], v0_vdw, vlast_vdw,
                            &workspace->LR[ index_lr(i, j, num_atom_types) ].vdW[1], control->tabulate + 1 );
                    Natural_Cubic_Spline( &h[1], &fCEvd[1],
                            &workspace->LR[ index_lr(i, j, num_atom_types) ].CEvd[1], control->tabulate + 1 );

                    Complete_Cubic_Spline( &h[1], &fele[1], v0_ele, vlast_ele,
                            &workspace->LR[ index_lr(i, j, num_atom_types) ].ele[1], control->tabulate + 1 );
                    Natural_Cubic_Spline( &h[1], &fCEclmb[1],
                            &workspace->LR[ index_lr(i, j, num_atom_types) ].CEclmb[1], control->tabulate + 1 );
                }
            }
        }
    }

    sfree( h, "Init_Lookup_Tables::h" );
    sfree( fh, "Init_Lookup_Tables::fh" );
    sfree( fvdw, "Init_Lookup_Tables::fvdw" );
    sfree( fCEvd, "Init_Lookup_Tables::fCEvd" );
    sfree( fele, "Init_Lookup_Tables::fele" );
    sfree( fCEclmb, "Init_Lookup_Tables::fCEclmb" );

#if defined(HAVE_CUDA)
    t_start = Get_Time( );
    copy_LR_table_to_device( system, control, workspace, aggregated );
    t_end = Get_Elapsed_Time( t_start );

    fprintf( stderr, "[INFO] time to copy LR Lookup table from device to host: %f \n", t_end );
#endif
}


void Finalize_LR_Lookup_Table( reax_system * const system, control_params * const control,
       storage * const workspace, mpi_datatypes * const mpi_data )
{
    int i, j, num_atom_types, ret;
    int existing_types[MAX_ATOM_TYPES], aggregated[MAX_ATOM_TYPES];

    num_atom_types = system->reax_param.num_atom_types;

    for ( i = 0; i < MAX_ATOM_TYPES; ++i )
    {
        existing_types[i] = 0;
    }
    for ( i = 0; i < system->N; ++i )
    {
        existing_types[ system->my_atoms[i].type ] = 1;
    }

    ret = MPI_Allreduce( existing_types, aggregated, MAX_ATOM_TYPES, MPI_INT,
            MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    for ( i = 0; i < num_atom_types; ++i )
    {
        if ( aggregated[i] )
        {
            for ( j = i; j < num_atom_types; ++j )
            {
                if ( aggregated[j] )
                {
                    sfree( workspace->LR[ index_lr(i, j, num_atom_types) ].y, "Finalize_LR_Lookup_Table::LR[ ].y" );
                    sfree( workspace->LR[ index_lr(i, j, num_atom_types) ].H, "Finalize_LR_Lookup_Table::LR[ ].H" );
                    sfree( workspace->LR[ index_lr(i, j, num_atom_types) ].vdW, "Finalize_LR_Lookup_Table::LR[ ].vdW" );
                    sfree( workspace->LR[ index_lr(i, j, num_atom_types) ].CEvd, "Finalize_LR_Lookup_Table::LR[ ].CEvd" );
                    sfree( workspace->LR[ index_lr(i, j, num_atom_types) ].ele, "Finalize_LR_Lookup_Table::LR[ ].ele" );
                    sfree( workspace->LR[ index_lr(i, j, num_atom_types) ].CEclmb, "Finalize_LR_Lookup_Table::LR[ ].CEclmb" );
                }
            }
        }
    }

    sfree( workspace->LR, "Finalize_LR_Lookup_Table::LR" );
}


/*
void copy_LR_table_to_device( reax_system * const system, control_params * const control, int aggregated )
{
  int i, j, r;
  int num_atom_types;
  LR_data *d_y;
  cubic_spline_coef *temp;

  num_atom_types = system->reaxprm.num_atom_types;

  fprintf( stderr, "Copying the LR Lookyp Table to the device ... \n" );

  cuda_malloc( (void **) &control->d_LR, sizeof (LR_lookup_table) * ( num_atom_types * num_atom_types ), FALSE, "LR_lookup:table" );

  for( i = 0; i < MAX_ATOM_TYPES; ++i )
  {
    existing_types[i] = 0;
  }

  for( i = 0; i < system->N; ++i )
  {
    existing_types[ system->atoms[i].type ] = 1;
  }

  copy_host_device ( workspace->LR, workspace->d_LR, sizeof (LR_lookup_table) * (num_atom_types * num_atom_types),
    cudaMemcpyHostToDevice, "LR_lookup:table");

  for( i = 0; i < num_atom_types; ++i )
    if( aggregated [i] )
      for( j = i; j < num_atom_types; ++j )

         if( aggregated [j] ) {

            cuda_malloc ((void **) &d_y, sizeof (LR_data) * (control->tabulate + 1), FALSE, "LR_lookup:d_y");
            copy_host_device ( workspace->LR [ index_lr (i, j, num_atom_types) ].y, d_y,
                    sizeof (LR_data) * (control->tabulate + 1), cudaMemcpyHostToDevice, "LR_lookup:y");
            copy_host_device ( &d_y, &workspace->d_LR [ index_lr (i, j, num_atom_types) ].y,
                    sizeof (LR_data *), cudaMemcpyHostToDevice, "LR_lookup:y");

            cuda_malloc ((void **) &temp, sizeof (cubic_spline_coef) * (control->tabulate + 1), FALSE, "LR_lookup:h");
            copy_host_device ( workspace->LR [ index_lr (i, j, num_atom_types) ].H, temp,
                    sizeof (cubic_spline_coef) * (control->tabulate + 1), cudaMemcpyHostToDevice, "LR_lookup:h");
            copy_host_device ( &temp, &workspace->d_LR [ index_lr (i, j, num_atom_types) ].H,
                    sizeof (cubic_spline_coef *), cudaMemcpyHostToDevice, "LR_lookup:h");

            cuda_malloc ((void **) &temp, sizeof (cubic_spline_coef) * (control->tabulate + 1), FALSE, "LR_lookup:vdW");
            copy_host_device ( workspace->LR [ index_lr (i, j, num_atom_types) ].vdW, temp,
                    sizeof (cubic_spline_coef) * (control->tabulate + 1), cudaMemcpyHostToDevice, "LR_lookup:vdW");
            copy_host_device ( &temp, &workspace->d_LR [ index_lr (i, j, num_atom_types) ].vdW,
                    sizeof (cubic_spline_coef *), cudaMemcpyHostToDevice, "LR_lookup:vdW");

            cuda_malloc ((void **) &temp, sizeof (cubic_spline_coef) * (control->tabulate + 1), FALSE, "LR_lookup:CEvd");
            copy_host_device ( workspace->LR [ index_lr (i, j, num_atom_types) ].CEvd, temp,
                    sizeof (cubic_spline_coef) * (control->tabulate + 1), cudaMemcpyHostToDevice, "LR_lookup:CEvd");
            copy_host_device ( &temp, &workspace->d_LR [ index_lr (i, j, num_atom_types) ].CEvd,
                    sizeof (cubic_spline_coef *), cudaMemcpyHostToDevice, "LR_lookup:CDvd");

            cuda_malloc ((void **) &temp, sizeof (cubic_spline_coef) * (control->tabulate + 1), FALSE, "LR_lookup:ele");
            copy_host_device ( workspace->LR [ index_lr (i, j, num_atom_types) ].ele, temp,
                    sizeof (cubic_spline_coef) * (control->tabulate + 1), cudaMemcpyHostToDevice, "LR_lookup:ele");
            copy_host_device ( &temp, &workspace->d_LR [ index_lr (i, j, num_atom_types) ].ele,
                    sizeof (cubic_spline_coef *), cudaMemcpyHostToDevice, "LR_lookup:ele");

            cuda_malloc ((void **) &temp, sizeof (cubic_spline_coef) * (control->tabulate + 1), FALSE, "LR_lookup:ceclmb");
            copy_host_device ( workspace->LR [ index_lr (i, j, num_atom_types) ].CEclmb, temp,
                    sizeof (cubic_spline_coef) * (control->tabulate + 1), cudaMemcpyHostToDevice, "LR_lookup:ceclmb");
            copy_host_device ( &temp, &workspace->d_LR [ index_lr (i, j, num_atom_types) ].CEclmb,
                    sizeof (cubic_spline_coef *), cudaMemcpyHostToDevice, "LR_lookup:ceclmb");
         }

   fprintf (stderr, "Copy of the LR Lookup Table to the device complete ... \n");
}
*/
