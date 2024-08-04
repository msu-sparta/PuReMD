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

#if (defined(HAVE_CONFIG_H) && !defined(__CONFIG_H_))
  #define __CONFIG_H_
  #include "../../common/include/config.h"
#endif

#if defined(PURE_REAX)
  #include "lookup.h"

  #include "comm_tools.h"
  #include "index_utils.h"
  #include "nonbonded.h"
  #include "tool_box.h"
#elif defined(LAMMPS_REAX)
  #include "reax_lookup.h"

  #include "reax_comm_tools.h"
  #include "reax_index_utils.h"
  #include "reax_nonbonded.h"
  #include "reax_tool_box.h"
#endif

#if defined(HAVE_CUDA)
  #include "cuda/gpu_lookup.h"
#elif defined(HAVE_HIP)
  #include "hip/gpu_lookup.h"
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

    /* solve via back substitution */
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
    a = smalloc( sizeof(real) * n, __FILE__, __LINE__ );
    b = smalloc( sizeof(real) * n, __FILE__, __LINE__ );
    c = smalloc( sizeof(real) * n, __FILE__, __LINE__ );
    d = smalloc( sizeof(real) * n, __FILE__, __LINE__ );
    v = smalloc( sizeof(real) * n, __FILE__, __LINE__ );

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
        b[i] = 2.0 * (h[i - 1] + h[i]);
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
        d[i] = 6.0 * ((f[i + 1] - f[i])
                / h[i] - (f[i] - f[i - 1]) / h[i - 1]);
    }

    v[0] = 0.0;
    v[n - 1] = 0.0;
    Tridiagonal_Solve( &a[1], &b[1], &c[1], &d[1], &v[1], n - 2 );

    for ( i = 1; i < n; ++i )
    {
        coef[i - 1].d = (v[i] - v[i - 1]) / (6.0 * h[i - 1]);
        coef[i - 1].c = v[i] / 2.0;
        coef[i - 1].b = (f[i] - f[i - 1]) / h[i - 1] + h[i - 1]
            * (2.0 * v[i] + v[i - 1]) / 6.0;
        coef[i - 1].a = f[i];
    }

    sfree( a, __FILE__, __LINE__ );
    sfree( b, __FILE__, __LINE__ );
    sfree( c, __FILE__, __LINE__ );
    sfree( d, __FILE__, __LINE__ );
    sfree( v, __FILE__, __LINE__ );
}


static void Complete_Cubic_Spline( const real * const h, const real * const f,
        real v0, real vlast, cubic_spline_coef * const coef, unsigned int n )
{
    int i;
    real *a, *b, *c, *d, *v;

    a = smalloc( sizeof(real) * n, __FILE__, __LINE__ );
    b = smalloc( sizeof(real) * n, __FILE__, __LINE__ );
    c = smalloc( sizeof(real) * n, __FILE__, __LINE__ );
    d = smalloc( sizeof(real) * n, __FILE__, __LINE__ );
    v = smalloc( sizeof(real) * n, __FILE__, __LINE__ );

    /* build the linear system */
    a[0] = 0.0;
    for ( i = 1; i < n; ++i )
    {
        a[i] = h[i - 1];
    }

    b[0] = 2.0 * h[0];
    for ( i = 1; i < n; ++i )
    {
        b[i] = 2.0 * (h[i - 1] + h[i]);
    }

    c[n - 1] = 0.0;
    for ( i = 0; i < n - 1; ++i )
    {
        c[i] = h[i];
    }

    d[0] = 6.0 * (f[1] - f[0]) / h[0] - 6.0 * v0;
    d[n - 1] = 6.0 * vlast - 6.0 * (f[n - 1] - f[n - 2] / h[n - 2]);
    for ( i = 1; i < n - 1; ++i )
    {
        d[i] = 6.0 * ((f[i + 1] - f[i])
                / h[i] - (f[i] - f[i - 1]) / h[i - 1]);
    }

    Tridiagonal_Solve( a, b, c, d, v, n );

    for ( i = 1; i < n; ++i )
    {
        coef[i - 1].d = (v[i] - v[i - 1]) / (6.0 * h[i - 1]);
        coef[i - 1].c = v[i] / 2.0;
        coef[i - 1].b = (f[i] - f[i - 1]) / h[i - 1]
            + h[i - 1] * (2.0 * v[i] + v[i - 1]) / 6.0;
        coef[i - 1].a = f[i];
    }

    sfree( a, __FILE__, __LINE__ );
    sfree( b, __FILE__, __LINE__ );
    sfree( c, __FILE__, __LINE__ );
    sfree( d, __FILE__, __LINE__ );
    sfree( v, __FILE__, __LINE__ );
}


/* Lookup and return a specified entry from a given LR lookup table
 *
 * t: LR lookup table
 * x: function value (atomic pairwise distance) used for the lookup
 * intr_type: type of interaction
 *
 * returns: the approximated atomic pairwise interaction value
 *  via polyonomial splines as specified in the lookup table */
real LR_Lookup_Entry( LR_lookup_table * const t, real x, int intr_type )
{
    int i;
    real base, dif, ret;

    i = (int) (x * t->inv_dx);
    if ( i == 0 )
    {
        ++i;
    }
    base = (real) (i + 1) * t->dx;
    dif = x - base;

    switch ( intr_type )
    {
        case LR_E_VDW:
            ret = ((t->vdW[i].d * dif + t->vdW[i].c) * dif + t->vdW[i].b) * dif
                + t->vdW[i].a;
            break;

        case LR_CE_VDW:
            ret = ((t->CEvd[i].d * dif + t->CEvd[i].c) * dif + t->CEvd[i].b) * dif
                + t->CEvd[i].a;
            break;

        case LR_E_CLMB:
            ret = ((t->ele[i].d * dif + t->ele[i].c) * dif + t->ele[i].b) * dif
                + t->ele[i].a;
            break;

        case LR_CE_CLMB:
            ret = ((t->CEclmb[i].d * dif + t->CEclmb[i].c) * dif + t->CEclmb[i].b) * dif
                + t->CEclmb[i].a;
            break;

        case LR_CM:
            ret = ((t->H[i].d * dif + t->H[i].c) * dif + t->H[i].b) * dif
                + t->H[i].a;
            break;

        default:
            ret = 0.0; // surpress compiler warning of uninitialized value
            fprintf( stderr, "[ERROR] LR_Lookup_Entry: unknown interaction type (%d). Terminating...\n", intr_type );
            MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
            break;
    }

    return ret;
}


/* Create lookup tables for approximating atomic pairwise interactions
 * based on distance between the atom pairs */
void Make_LR_Lookup_Table( reax_system * const system, control_params * const control,
        storage * const workspace, mpi_datatypes * const mpi_data )
{
    int i, j, r, num_atom_types, ret;
    int existing_types[MAX_ATOM_TYPES], aggregated[MAX_ATOM_TYPES];
    real dr;
    real *h, *fh, *fvdw, *fele, *fCEvd, *fCEclmb;
    real v0_vdw, v0_ele, vlast_vdw, vlast_ele;

    v0_vdw = 0.0;
    v0_ele = 0.0;
    vlast_vdw = 0.0;
    vlast_ele = 0.0;

    num_atom_types = system->reax_param.num_atom_types;
    dr = control->nonb_cut / control->tabulate;

    h = scalloc( (control->tabulate + 2), sizeof(real), __FILE__, __LINE__ );
    fh = scalloc( (control->tabulate + 2), sizeof(real), __FILE__, __LINE__ );
    fvdw = scalloc( (control->tabulate + 2), sizeof(real), __FILE__, __LINE__ );
    fCEvd = scalloc( (control->tabulate + 2), sizeof(real), __FILE__, __LINE__ );
    fele = scalloc( (control->tabulate + 2), sizeof(real), __FILE__, __LINE__ );
    fCEclmb = scalloc( (control->tabulate + 2), sizeof(real), __FILE__, __LINE__ );

    /* allocate Long-Range LookUp Table space based on
     * number of atom types in the ffield file */
    workspace->LR = smalloc( sizeof(LR_lookup_table) * num_atom_types * num_atom_types,
            __FILE__, __LINE__ );

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
        if ( aggregated[i] > 0 )
        {
            for ( j = i; j < num_atom_types; ++j )
            {
                if ( aggregated[j] > 0 )
                {
                    workspace->LR[ index_lr(i, j, num_atom_types) ].xmin = 0.0;
                    workspace->LR[ index_lr(i, j, num_atom_types) ].xmax = control->nonb_cut;
                    workspace->LR[ index_lr(i, j, num_atom_types) ].n = control->tabulate + 1;
                    workspace->LR[ index_lr(i, j, num_atom_types) ].dx = dr;
                    workspace->LR[ index_lr(i, j, num_atom_types) ].inv_dx = control->tabulate / control->nonb_cut;
                    workspace->LR[ index_lr(i, j, num_atom_types) ].y =
                            smalloc( sizeof(LR_data) * workspace->LR[ index_lr(i, j, num_atom_types) ].n,
                                    __FILE__, __LINE__ );
                    workspace->LR[ index_lr(i, j, num_atom_types) ].H =
                            smalloc( sizeof(cubic_spline_coef) * workspace->LR[ index_lr(i, j, num_atom_types) ].n,
                                    __FILE__, __LINE__ );
                    workspace->LR[ index_lr(i, j, num_atom_types) ].vdW =
                            smalloc( sizeof(cubic_spline_coef) * workspace->LR[ index_lr(i, j, num_atom_types) ].n,
                                    __FILE__, __LINE__ );
                    workspace->LR[ index_lr(i, j, num_atom_types) ].CEvd =
                            smalloc( sizeof(cubic_spline_coef) * workspace->LR[ index_lr(i, j, num_atom_types) ].n,
                                    __FILE__, __LINE__ );
                    workspace->LR[ index_lr(i, j, num_atom_types) ].ele =
                            smalloc( sizeof(cubic_spline_coef) * workspace->LR[ index_lr(i, j, num_atom_types) ].n,
                                    __FILE__, __LINE__ );
                    workspace->LR[ index_lr(i, j, num_atom_types) ].CEclmb =
                            smalloc( sizeof(cubic_spline_coef) * workspace->LR[ index_lr(i, j, num_atom_types) ].n,
                                    __FILE__, __LINE__ );

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
                            v0_vdw = workspace->LR[ index_lr(i, j, num_atom_types) ].y[r].e_vdW;
                            v0_ele = workspace->LR[ index_lr(i, j, num_atom_types) ].y[r].e_ele;
                        }
                        else if ( r == control->tabulate )
                        {
                            vlast_vdw = workspace->LR[ index_lr(i, j, num_atom_types) ].y[r].e_vdW;
                            vlast_ele = workspace->LR[ index_lr(i, j, num_atom_types) ].y[r].e_ele;
                        }
                    }

                    Natural_Cubic_Spline( &h[1], &fh[1],
                            &workspace->LR[ index_lr(i, j, num_atom_types) ].H[1],
                            workspace->LR[ index_lr(i, j, num_atom_types) ].n );

                    Complete_Cubic_Spline( &h[1], &fvdw[1], v0_vdw, vlast_vdw,
                            &workspace->LR[ index_lr(i, j, num_atom_types) ].vdW[1],
                            workspace->LR[ index_lr(i, j, num_atom_types) ].n );

                    Natural_Cubic_Spline( &h[1], &fCEvd[1],
                            &workspace->LR[ index_lr(i, j, num_atom_types) ].CEvd[1],
                            workspace->LR[ index_lr(i, j, num_atom_types) ].n );

                    Complete_Cubic_Spline( &h[1], &fele[1], v0_ele, vlast_ele,
                            &workspace->LR[ index_lr(i, j, num_atom_types) ].ele[1],
                            workspace->LR[ index_lr(i, j, num_atom_types) ].n );

                    Natural_Cubic_Spline( &h[1], &fCEclmb[1],
                            &workspace->LR[ index_lr(i, j, num_atom_types) ].CEclmb[1],
                            workspace->LR[ index_lr(i, j, num_atom_types) ].n );
                }
            }
        }
    }

    sfree( h, __FILE__, __LINE__ );
    sfree( fh, __FILE__, __LINE__ );
    sfree( fvdw, __FILE__, __LINE__ );
    sfree( fCEvd, __FILE__, __LINE__ );
    sfree( fele, __FILE__, __LINE__ );
    sfree( fCEclmb, __FILE__, __LINE__ );

#if defined(HAVE_CUDA) || defined(HAVE_HIP)
    GPU_Copy_LR_Lookup_Table_Host_to_Device( system, control, workspace, aggregated );
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
                    sfree( workspace->LR[ index_lr(i, j, num_atom_types) ].y,
                            __FILE__, __LINE__ );
                    sfree( workspace->LR[ index_lr(i, j, num_atom_types) ].H,
                            __FILE__, __LINE__ );
                    sfree( workspace->LR[ index_lr(i, j, num_atom_types) ].vdW,
                            __FILE__, __LINE__ );
                    sfree( workspace->LR[ index_lr(i, j, num_atom_types) ].CEvd,
                            __FILE__, __LINE__ );
                    sfree( workspace->LR[ index_lr(i, j, num_atom_types) ].ele,
                            __FILE__, __LINE__ );
                    sfree( workspace->LR[ index_lr(i, j, num_atom_types) ].CEclmb,
                            __FILE__, __LINE__ );
                }
            }
        }
    }

    sfree( workspace->LR, __FILE__, __LINE__ );
}
