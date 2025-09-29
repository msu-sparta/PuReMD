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

#include "lookup.h"

#include "nonbonded.h"
#include "tool_box.h"


/* Fills solution into x. Warning: will modify c and d! */
static void Tridiagonal_Solve( const real *a, const real *b,
        real *c, real *d, real *x, uint32_t n)
{
    uint32_t i;
    real id;

    /* Modify the coefficients. */
    c[0] /= b[0]; /* Division by zero risk. */
    d[0] /= b[0]; /* Division by zero would imply a singular matrix. */
    for (i = 1; i < n; i++) {
        id = (b[i] - c[i - 1] * a[i]); /* Division by zero risk. */
        c[i] /= id;         /* Last value calculated is redundant. */
        d[i] = (d[i] - d[i - 1] * a[i]) / id;
    }

    /* solve via back substitution */
    x[n - 1] = d[n - 1];
    for (i = n - 2; i < n; i--) {
        x[i] = d[i] - c[i] * x[i + 1];
    }
}


static void Natural_Cubic_Spline( const real *h, const real *f,
        cubic_spline_coef *coef, uint32_t n )
{
    uint32_t i;
    real *a, *b, *c, *d, *v;

    /* allocate space for linear system */
    a = smalloc( sizeof(real) * n, __FILE__, __LINE__ );
    b = smalloc( sizeof(real) * n, __FILE__, __LINE__ );
    c = smalloc( sizeof(real) * n, __FILE__, __LINE__ );
    d = smalloc( sizeof(real) * n, __FILE__, __LINE__ );
    v = smalloc( sizeof(real) * n, __FILE__, __LINE__ );

    /* build linear system */
    a[0] = 0.0;
    a[1] = 0.0;
    a[n - 1] = 0.0;
    for ( i = 2; i < n - 1; ++i ) {
        a[i] = h[i - 1];
    }

    b[0] = 0.0;
    b[n - 1] = 0.0;
    for ( i = 1; i < n - 1; ++i ) {
        b[i] = 2.0 * (h[i - 1] + h[i]);
    }

    c[0] = 0.0;
    c[n - 2] = 0.0;
    c[n - 1] = 0.0;
    for ( i = 1; i < n - 2; ++i ) {
        c[i] = h[i];
    }

    d[0] = 0.0;
    d[n - 1] = 0.0;
    for ( i = 1; i < n - 1; ++i ) {
        d[i] = 6.0 * ((f[i + 1] - f[i])
                / h[i] - (f[i] - f[i - 1]) / h[i - 1]);
    }

    v[0] = 0.0;
    v[n - 1] = 0.0;
    Tridiagonal_Solve( &a[1], &b[1], &c[1], &d[1], &v[1], n - 2 );

    for ( i = 1; i < n; ++i ) {
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


static void Complete_Cubic_Spline( const real *h, const real *f, real v0, real vlast,
        cubic_spline_coef *coef, uint32_t n )
{
    uint32_t i;
    real *a, *b, *c, *d, *v;

    /* allocate space for the linear system */
    a = smalloc( sizeof(real) * n, __FILE__, __LINE__ );
    b = smalloc( sizeof(real) * n, __FILE__, __LINE__ );
    c = smalloc( sizeof(real) * n, __FILE__, __LINE__ );
    d = smalloc( sizeof(real) * n, __FILE__, __LINE__ );
    v = smalloc( sizeof(real) * n, __FILE__, __LINE__ );

    /* build the linear system */
    a[0] = 0.0;
    for ( i = 1; i < n; ++i ) {
        a[i] = h[i - 1];
    }

    b[0] = 2.0 * h[0];
    for ( i = 1; i < n; ++i ) {
        b[i] = 2.0 * (h[i - 1] + h[i]);
    }

    c[n - 1] = 0.0;
    for ( i = 0; i < n - 1; ++i ) {
        c[i] = h[i];
    }

    d[0] = 6.0 * (f[1] - f[0]) / h[0] - 6.0 * v0;
    d[n - 1] = 6.0 * vlast - 6.0 * (f[n - 1] - f[n - 2] / h[n - 2]);
    for ( i = 1; i < n - 1; ++i ) {
        d[i] = 6.0 * ((f[i + 1] - f[i])
                / h[i] - (f[i] - f[i - 1]) / h[i - 1]);
    }

    Tridiagonal_Solve( a, b, c, d, v, n );

    for ( i = 1; i < n; ++i ) {
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


void Make_LR_Lookup_Table( reax_system *system, control_params *control,
       static_storage *workspace )
{
    uint32_t i, j, r, num_atom_types;
    bool existing_types[MAX_ATOM_TYPES];
    real dr;
    real *h, *fh, *fvdw, *fele, *fCEvd, *fCEclmb;
    real v0_vdw, v0_ele, vlast_vdw, vlast_ele;

    v0_vdw = 0.0;
    v0_ele = 0.0;
    vlast_vdw = 0.0;
    vlast_ele = 0.0;

    num_atom_types = system->reax_param.num_atom_types;
    dr = control->nonb_cut / control->tabulate;

    h = scalloc( control->tabulate + 2, sizeof(real), __FILE__, __LINE__ );
    fh = scalloc( control->tabulate + 2, sizeof(real), __FILE__, __LINE__ );
    fvdw = scalloc( control->tabulate + 2, sizeof(real), __FILE__, __LINE__ );
    fCEvd = scalloc( control->tabulate + 2, sizeof(real), __FILE__, __LINE__ );
    fele = scalloc( control->tabulate + 2, sizeof(real), __FILE__, __LINE__ );
    fCEclmb = scalloc( control->tabulate + 2, sizeof(real), __FILE__, __LINE__ );

    /* allocate Long-Range LookUp Table space based on
       number of atom types in the ffield file */
    workspace->LR = smalloc( sizeof(LR_lookup_table) * SQR(num_atom_types),
           __FILE__, __LINE__ );

    /* most atom types in ffield file will not exist in the current
       simulation. to avoid unnecessary lookup table space, determine
       the atom types that exist in the current simulation */
    for ( i = 0; i < MAX_ATOM_TYPES; ++i ) {
        existing_types[i] = FALSE;
    }
    for ( i = 0; i < system->N; ++i ) {
        existing_types[ system->atoms[i].type ] = TRUE;
    }

    /* fill in the lookup table entries for existing atom types.
       only lower half should be enough. */
    for ( i = 0; i < num_atom_types; ++i ) {
        if ( existing_types[i] == TRUE ) {
            for ( j = i; j < num_atom_types; ++j ) {
                if ( existing_types[j] == TRUE ) {
                    workspace->LR[IDX_LR(i, j, num_atom_types)].xmin = 0.0;
                    workspace->LR[IDX_LR(i, j, num_atom_types)].xmax = control->nonb_cut;
                    workspace->LR[IDX_LR(i, j, num_atom_types)].n = control->tabulate + 1;
                    workspace->LR[IDX_LR(i, j, num_atom_types)].dx = dr;
                    workspace->LR[IDX_LR(i, j, num_atom_types)].inv_dx = control->tabulate / control->nonb_cut;
                    workspace->LR[IDX_LR(i, j, num_atom_types)].y = 
                        smalloc( sizeof(LR_data) * workspace->LR[IDX_LR(i, j, num_atom_types)].n,
                              __FILE__, __LINE__ );
                    workspace->LR[IDX_LR(i, j, num_atom_types)].H = 
                        smalloc( sizeof(cubic_spline_coef) * workspace->LR[IDX_LR(i, j, num_atom_types)].n,
                              __FILE__, __LINE__ );
                    workspace->LR[IDX_LR(i, j, num_atom_types)].vdW = 
                        smalloc( sizeof(cubic_spline_coef) * workspace->LR[IDX_LR(i, j, num_atom_types)].n,
                              __FILE__, __LINE__ );
                    workspace->LR[IDX_LR(i, j, num_atom_types)].CEvd = 
                        smalloc( sizeof(cubic_spline_coef) * workspace->LR[IDX_LR(i, j, num_atom_types)].n,
                              __FILE__, __LINE__ );
                    workspace->LR[IDX_LR(i, j, num_atom_types)].ele = 
                        smalloc( sizeof(cubic_spline_coef) * workspace->LR[IDX_LR(i, j, num_atom_types)].n,
                              __FILE__, __LINE__ );
                    workspace->LR[IDX_LR(i, j, num_atom_types)].CEclmb = 
                        smalloc( sizeof(cubic_spline_coef) * workspace->LR[IDX_LR(i, j, num_atom_types)].n,
                              __FILE__, __LINE__ );

                    for ( r = 1; r <= control->tabulate; ++r ) {
                        LR_vdW_Coulomb( system, control, workspace, i, j, r * dr,
                                &workspace->LR[IDX_LR(i, j, num_atom_types)].y[r] );
                        h[r] = workspace->LR[IDX_LR(i, j, num_atom_types)].dx;
                        fh[r] = workspace->LR[IDX_LR(i, j, num_atom_types)].y[r].H;
                        fvdw[r] = workspace->LR[IDX_LR(i, j, num_atom_types)].y[r].e_vdW;
                        fCEvd[r] = workspace->LR[IDX_LR(i, j, num_atom_types)].y[r].CEvd;
                        fele[r] = workspace->LR[IDX_LR(i, j, num_atom_types)].y[r].e_ele;
                        fCEclmb[r] = workspace->LR[IDX_LR(i, j, num_atom_types)].y[r].CEclmb;

                        if ( r == 1 ) {
                            v0_vdw = workspace->LR[IDX_LR(i, j, num_atom_types)].y[r].e_vdW;
                            v0_ele = workspace->LR[IDX_LR(i, j, num_atom_types)].y[r].e_ele;
                        } else if ( r == control->tabulate ) {
                            vlast_vdw = workspace->LR[IDX_LR(i, j, num_atom_types)].y[r].e_vdW;
                            vlast_ele = workspace->LR[IDX_LR(i, j, num_atom_types)].y[r].e_ele;
                        }
                    }

                    Natural_Cubic_Spline( &h[1], &fh[1],
                            &workspace->LR[IDX_LR(i, j, num_atom_types)].H[1],
                            workspace->LR[IDX_LR(i, j, num_atom_types)].n );

                    Complete_Cubic_Spline( &h[1], &fvdw[1], v0_vdw, vlast_vdw,
                            &workspace->LR[IDX_LR(i, j, num_atom_types)].vdW[1],
                            workspace->LR[IDX_LR(i, j, num_atom_types)].n );

                    Natural_Cubic_Spline( &h[1], &fCEvd[1],
                            &workspace->LR[IDX_LR(i, j, num_atom_types)].CEvd[1],
                            workspace->LR[IDX_LR(i, j, num_atom_types)].n );

                    Complete_Cubic_Spline( &h[1], &fele[1], v0_ele, vlast_ele,
                            &workspace->LR[IDX_LR(i, j, num_atom_types)].ele[1],
                            workspace->LR[IDX_LR(i, j, num_atom_types)].n );

                    Natural_Cubic_Spline( &h[1], &fCEclmb[1],
                            &workspace->LR[IDX_LR(i, j, num_atom_types)].CEclmb[1],
                            workspace->LR[IDX_LR(i, j, num_atom_types)].n );
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
}


void Finalize_LR_Lookup_Table( reax_system *system, control_params *control,
       static_storage *workspace )
{
    uint32_t i, j, num_atom_types;
    bool existing_types[MAX_ATOM_TYPES];

    num_atom_types = system->reax_param.num_atom_types;

    for ( i = 0; i < MAX_ATOM_TYPES; ++i ) {
        existing_types[i] = FALSE;
    }
    for ( i = 0; i < system->N; ++i ) {
        existing_types[ system->atoms[i].type ] = TRUE;
    }

    for ( i = 0; i < num_atom_types; ++i ) {
        if ( existing_types[i] == TRUE ) {
            for ( j = i; j < num_atom_types; ++j ) {
                if ( existing_types[j] == TRUE ) {
                    sfree( workspace->LR[IDX_LR(i, j, num_atom_types)].y, __FILE__, __LINE__ );
                    sfree( workspace->LR[IDX_LR(i, j, num_atom_types)].H, __FILE__, __LINE__ );
                    sfree( workspace->LR[IDX_LR(i, j, num_atom_types)].vdW, __FILE__, __LINE__ );
                    sfree( workspace->LR[IDX_LR(i, j, num_atom_types)].CEvd, __FILE__, __LINE__ );
                    sfree( workspace->LR[IDX_LR(i, j, num_atom_types)].ele, __FILE__, __LINE__ );
                    sfree( workspace->LR[IDX_LR(i, j, num_atom_types)].CEclmb, __FILE__, __LINE__ );
                }
            }
        }
    }

    sfree( workspace->LR, __FILE__, __LINE__ );
}
