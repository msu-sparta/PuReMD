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

#include "tool_box.h"
#include "two_body_interactions.h"


void Make_Lookup_Table( real xmin, real xmax, int n,
        lookup_function f, lookup_table* t )
{
    int i;

    t->xmin = xmin;
    t->xmax = xmax;
    t->n = n;
    t->dx = (xmax - xmin) / (n - 1);
    t->inv_dx = 1.0 / t->dx;
    t->a = (n - 1) / (xmax - xmin);
    t->y = (real*) smalloc( n * sizeof(real),
            "Make_Lookup_Table::t->y" );

    for (i = 0; i < n; i++)
    {
        t->y[i] = f(i * t->dx + t->xmin);
    }

    // fprintf(stdout,"dx = %lf\n",t->dx);
    // for(i=0; i < n; i++)
    //   fprintf( stdout,"%d %lf %lf %lf\n",
    //            i, i/t->a+t->xmin, t->y[i], exp(i/t->a+t->xmin) );
}


/* Fills solution into x. Warning: will modify c and d! */
static void Tridiagonal_Solve( const real *a, const real *b,
        real *c, real *d, real *x, unsigned int n)
{
    int i;
    real id;

    /* Modify the coefficients. */
    c[0] /= b[0]; /* Division by zero risk. */
    d[0] /= b[0]; /* Division by zero would imply a singular matrix. */
    for (i = 1; i < n; i++)
    {
        id = (b[i] - c[i - 1] * a[i]); /* Division by zero risk. */
        c[i] /= id;         /* Last value calculated is redundant. */
        d[i] = (d[i] - d[i - 1] * a[i]) / id;
    }

    /* Now back substitute. */
    x[n - 1] = d[n - 1];
    for (i = n - 2; i >= 0; i--)
    {
        x[i] = d[i] - c[i] * x[i + 1];
    }
}


static void Natural_Cubic_Spline( const real *h, const real *f,
        cubic_spline_coef *coef, unsigned int n )
{
    int i;
    real *a, *b, *c, *d, *v;

    /* allocate space for the linear system */
    a = (real*) smalloc( n * sizeof(real),
           "Natural_Cubic_Spline::a" );
    b = (real*) smalloc( n * sizeof(real),
           "Natural_Cubic_Spline::b" );
    c = (real*) smalloc( n * sizeof(real),
           "Natural_Cubic_Spline::c" );
    d = (real*) smalloc( n * sizeof(real),
           "Natural_Cubic_Spline::d" );
    v = (real*) smalloc( n * sizeof(real),
           "Natural_Cubic_Spline::v" );

    /* build the linear system */
    a[0] = 0.0;
    a[1] = 0.0;
    a[n - 1] = 0.0;
    for ( i = 2; i < n - 1; ++i )
    {
        a[i] = h[i];
    }

    b[0] = 0.0;
    b[n - 1] = 0.0;
    for ( i = 1; i < n - 1; ++i )
    {
        b[i] = 2.0 * (h[i] + h[i + 1]);
    }

    c[0] = 0.0;
    c[n - 2] = 0.0;
    c[n - 1] = 0.0;
    for ( i = 1; i < n - 2; ++i )
    {
        c[i] = h[i + 1];
    }

    d[0] = 0.0;
    d[n - 1] = 0.0;
    for ( i = 1; i < n - 1; ++i )
    {
        d[i] = 6.0 * ((f[i + 2] - f[i + 1])
                / h[i + 1] - (f[i + 1] - f[i]) / h[i]);
    }

    /*fprintf( stderr, "i  a        b        c        d\n" );
      for( i = 0; i < n; ++i )
      fprintf( stderr, "%d  %f  %f  %f  %f\n", i, a[i], b[i], c[i], d[i] );*/
    v[0] = 0.0;
    v[n - 1] = 0.0;
    Tridiagonal_Solve( a + 1, b + 1, c + 1, d + 1, v + 1, n - 2 );

    for ( i = 1; i < n; ++i )
    {
        coef[i - 1].d = (v[i] - v[i - 1]) / (6.0 * h[i]);
        coef[i - 1].c = v[i] / 2.0;
        coef[i - 1].b = (f[i + 1] - f[i]) / h[i] + h[i]
            * (2.0 * v[i] + v[i - 1]) / 6.0;
        coef[i - 1].a = f[i + 1];
    }

    sfree( a, "Natural_Cubic_Spline::a" );
    sfree( b, "Natural_Cubic_Spline::b" );
    sfree( c, "Natural_Cubic_Spline::c" );
    sfree( d, "Natural_Cubic_Spline::d" );
    sfree( v, "Natural_Cubic_Spline::v" );
}


static void Complete_Cubic_Spline( const real *h, const real *f, real v0, real vlast,
        cubic_spline_coef *coef, unsigned int n )
{
    int i;
    real *a, *b, *c, *d, *v;

    /* allocate space for the linear system */
    a = (real*) smalloc( n * sizeof(real),
           "Complete_Cubic_Spline::a" );
    b = (real*) smalloc( n * sizeof(real),
           "Complete_Cubic_Spline::b" );
    c = (real*) smalloc( n * sizeof(real),
           "Complete_Cubic_Spline::c" );
    d = (real*) smalloc( n * sizeof(real),
           "Complete_Cubic_Spline::d" );
    v = (real*) smalloc( n * sizeof(real),
           "Complete_Cubic_Spline::v" );

    /* build the linear system */
    a[0] = 0.0;
    for ( i = 1; i < n; ++i )
    {
        a[i] = h[i];
    }

    b[0] = 2.0 * h[1];
    for ( i = 1; i < n; ++i )
    {
        b[i] = 2.0 * (h[i] + h[i + 1]);
    }

    c[n - 1] = 0.0;
    for ( i = 0; i < n - 1; ++i )
    {
        c[i] = h[i + 1];
    }

    d[0] = 6.0 * (f[2] - f[1]) / h[1] - 6.0 * v0;
    d[n - 1] = 6.0 * vlast - 6.0 * (f[n] - f[n - 1] / h[n - 1]);
    for ( i = 1; i < n - 1; ++i )
    {
        d[i] = 6.0 * ((f[i + 2] - f[i + 1])
                / h[i + 1] - (f[i + 1] - f[i]) / h[i]);
    }

    /*fprintf( stderr, "i  a        b        c        d\n" );
      for( i = 0; i < n; ++i )
      fprintf( stderr, "%d  %f  %f  %f  %f\n", i, a[i], b[i], c[i], d[i] );*/

    Tridiagonal_Solve( a, b, c, d, v, n );

    for ( i = 1; i < n; ++i )
    {
        coef[i - 1].d = (v[i] - v[i - 1]) / (6.0 * h[i]);
        coef[i - 1].c = v[i] / 2.0;
        coef[i - 1].b = (f[i + 1] - f[i]) / h[i] + h[i] * (2.0 * v[i] + v[i - 1]) / 6.0;
        coef[i - 1].a = f[i + 1];
    }

    sfree( a, "Complete_Cubic_Spline::a" );
    sfree( b, "Complete_Cubic_Spline::b" );
    sfree( c, "Complete_Cubic_Spline::c" );
    sfree( d, "Complete_Cubic_Spline::d" );
    sfree( v, "Complete_Cubic_Spline::v" );
}


static void LR_Lookup( LR_lookup_table *t, real r, LR_data *y )
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
    //fprintf( stderr, "r: %f, i: %d, base: %f, dif: %f\n", r, i, base, dif );

    y->e_vdW = ((t->vdW[i].d * dif + t->vdW[i].c) * dif + t->vdW[i].b) * dif +
               t->vdW[i].a;
    y->CEvd = ((t->CEvd[i].d * dif + t->CEvd[i].c) * dif +
               t->CEvd[i].b) * dif + t->CEvd[i].a;
    //y->CEvd = (3*t->vdW[i].d*dif + 2*t->vdW[i].c)*dif + t->vdW[i].b;

    y->e_ele = ((t->ele[i].d * dif + t->ele[i].c) * dif + t->ele[i].b) * dif +
               t->ele[i].a;
    y->CEclmb = ((t->CEclmb[i].d * dif + t->CEclmb[i].c) * dif + t->CEclmb[i].b) * dif +
                t->CEclmb[i].a;

    y->H = y->e_ele * EV_to_KCALpMOL / C_ele;
    //y->H = ((t->H[i].d*dif + t->H[i].c)*dif + t->H[i].b)*dif + t->H[i].a;
}


void Make_LR_Lookup_Table( reax_system *system, control_params *control,
       static_storage *workspace )
{
    int i, j, r;
    int num_atom_types;
    int existing_types[MAX_ATOM_TYPES];
    real dr;
    real *h, *fh, *fvdw, *fele, *fCEvd, *fCEclmb;
    real v0_vdw, v0_ele, vlast_vdw, vlast_ele;
    /* real rand_dist;
       real evdw_abserr, evdw_relerr, fvdw_abserr, fvdw_relerr;
       real eele_abserr, eele_relerr, fele_abserr, fele_relerr;
       real evdw_maxerr, eele_maxerr;
       LR_data y, y_spline; */

    /* initializations */
    vlast_ele = 0;
    vlast_vdw = 0;
    v0_ele = 0;
    v0_vdw = 0;

    num_atom_types = system->reaxprm.num_atom_types;
    dr = control->r_cut / control->tabulate;
    h = (real*) scalloc( (control->tabulate + 1), sizeof(real),
            "Make_LR_Lookup_Table::h" );
    fh = (real*) scalloc( (control->tabulate + 1), sizeof(real),
            "Make_LR_Lookup_Table::fh" );
    fvdw = (real*) scalloc( (control->tabulate + 1), sizeof(real),
            "Make_LR_Lookup_Table::fvdw" );
    fCEvd = (real*) scalloc( (control->tabulate + 1), sizeof(real),
            "Make_LR_Lookup_Table::fCEvd" );
    fele = (real*) scalloc( (control->tabulate + 1), sizeof(real),
            "Make_LR_Lookup_Table::fele" );
    fCEclmb = (real*) scalloc( (control->tabulate + 1), sizeof(real),
            "Make_LR_Lookup_Table::fCEclmb" );

    /* allocate Long-Range LookUp Table space based on
       number of atom types in the ffield file */
    workspace->LR = (LR_lookup_table**) smalloc( num_atom_types * sizeof(LR_lookup_table*),
           "Make_LR_Lookup_Table::LR" );
    for ( i = 0; i < num_atom_types; ++i )
    {
        workspace->LR[i] = (LR_lookup_table*) smalloc( num_atom_types * sizeof(LR_lookup_table),
                "Make_LR_Lookup_Table::LR[i]");
    }

    /* most atom types in ffield file will not exist in the current
       simulation. to avoid unnecessary lookup table space, determine
       the atom types that exist in the current simulation */
    for ( i = 0; i < MAX_ATOM_TYPES; ++i )
    {
        existing_types[i] = 0;
    }
    for ( i = 0; i < system->N; ++i )
    {
        existing_types[ system->atoms[i].type ] = 1;
    }

    /* fill in the lookup table entries for existing atom types.
       only lower half should be enough. */
    for ( i = 0; i < num_atom_types; ++i )
    {
        if ( existing_types[i] )
        {
            for ( j = i; j < num_atom_types; ++j )
            {
                if ( existing_types[j] )
                {
                    workspace->LR[i][j].xmin = 0;
                    workspace->LR[i][j].xmax = control->r_cut;
                    workspace->LR[i][j].n = control->tabulate + 1;
                    workspace->LR[i][j].dx = dr;
                    workspace->LR[i][j].inv_dx = control->tabulate / control->r_cut;
                    workspace->LR[i][j].y = (LR_data*)
                        smalloc( workspace->LR[i][j].n * sizeof(LR_data),
                              "Make_LR_Lookup_Table::LR[i][j].y" );
                    workspace->LR[i][j].H = (cubic_spline_coef*)
                        smalloc( workspace->LR[i][j].n * sizeof(cubic_spline_coef),
                              "Make_LR_Lookup_Table::LR[i][j].H" );
                    workspace->LR[i][j].vdW = (cubic_spline_coef*)
                        smalloc( workspace->LR[i][j].n * sizeof(cubic_spline_coef),
                              "Make_LR_Lookup_Table::LR[i][j].vdW" );
                    workspace->LR[i][j].CEvd = (cubic_spline_coef*)
                        smalloc( workspace->LR[i][j].n * sizeof(cubic_spline_coef),
                              "Make_LR_Lookup_Table::LR[i][j].CEvd" );
                    workspace->LR[i][j].ele = (cubic_spline_coef*)
                        smalloc( workspace->LR[i][j].n * sizeof(cubic_spline_coef),
                              "Make_LR_Lookup_Table::LR[i][j].ele" );
                    workspace->LR[i][j].CEclmb = (cubic_spline_coef*)
                        smalloc( workspace->LR[i][j].n * sizeof(cubic_spline_coef),
                              "Make_LR_Lookup_Table::LR[i][j].CEclmb" );

                    for ( r = 1; r <= control->tabulate; ++r )
                    {
                        LR_vdW_Coulomb( system, control, i, j, r * dr,
                                &(workspace->LR[i][j].y[r]) );
                        h[r] = workspace->LR[i][j].dx;
                        fh[r] = workspace->LR[i][j].y[r].H;
                        fvdw[r] = workspace->LR[i][j].y[r].e_vdW;
                        fCEvd[r] = workspace->LR[i][j].y[r].CEvd;
                        fele[r] = workspace->LR[i][j].y[r].e_ele;
                        fCEclmb[r] = workspace->LR[i][j].y[r].CEclmb;

                        if ( r == 1 )
                        {
                            v0_vdw = workspace->LR[i][j].y[r].CEvd;
                            v0_ele = workspace->LR[i][j].y[r].CEclmb;
                        }
                        else if ( r == control->tabulate )
                        {
                            vlast_vdw = workspace->LR[i][j].y[r].CEvd;
                            vlast_ele = workspace->LR[i][j].y[r].CEclmb;
                        }
                    }

                    Natural_Cubic_Spline( h, fh,
                            &(workspace->LR[i][j].H[1]), control->tabulate + 1 );

//                    fprintf( stderr, "%-6s  %-6s  %-6s\n", "r", "h", "fh" );
//                    for( r = 1; r <= control->tabulate; ++r )
//                        fprintf( stderr, "%f  %f  %f\n", r * dr, h[r], fh[r] );

                    Complete_Cubic_Spline( h, fvdw, v0_vdw, vlast_vdw,
                            &(workspace->LR[i][j].vdW[1]), control->tabulate + 1 );

//                    fprintf( stderr, "%-6s  %-6s  %-6s\n", "r", "h", "fvdw" );
//                    for( r = 1; r <= control->tabulate; ++r )
//                        fprintf( stderr, "%f  %f  %f\n", r * dr, h[r], fvdw[r] );
//                    fprintf( stderr, "v0_vdw: %f, vlast_vdw: %f\n", v0_vdw, vlast_vdw );

                    Natural_Cubic_Spline( h, fCEvd,
                            &(workspace->LR[i][j].CEvd[1]), control->tabulate + 1 );

//                    fprintf( stderr, "%-6s  %-6s  %-6s\n", "r", "h", "fele" );
//                    for( r = 1; r <= control->tabulate; ++r )
//                        fprintf( stderr, "%f  %f  %f\n", r * dr, h[r], fele[r] );
//                    fprintf( stderr, "v0_ele: %f, vlast_ele: %f\n", v0_ele, vlast_ele );

                    Complete_Cubic_Spline( h, fele, v0_ele, vlast_ele,
                            &(workspace->LR[i][j].ele[1]), control->tabulate + 1 );

//                    fprintf( stderr, "%-6s  %-6s  %-6s\n", "r", "h", "fele" );
//                    for( r = 1; r <= control->tabulate; ++r )
//                        fprintf( stderr, "%f  %f  %f\n", r * dr, h[r], fele[r] );
//                    fprintf( stderr, "v0_ele: %f, vlast_ele: %f\n", v0_ele, vlast_ele );

                    Natural_Cubic_Spline( h, fCEclmb,
                            &(workspace->LR[i][j].CEclmb[1]), control->tabulate + 1 );
                }
            }
        }
    }

    /***** //test LR-Lookup table
     evdw_maxerr = 0;
     eele_maxerr = 0;
     for( i = 0; i < num_atom_types; ++i )
     if( existing_types[i] )
     for( j = i; j < num_atom_types; ++j )
     if( existing_types[j] ) {
     for( r = 1; r <= 100; ++r ) {
     rand_dist = (real)rand()/RAND_MAX * control->r_cut;
     LR_vdW_Coulomb( system, control, i, j, rand_dist, &y );
     LR_Lookup( &(workspace->LR[i][j]), rand_dist, &y_spline );

     evdw_abserr = FABS(y.e_vdW - y_spline.e_vdW);
     evdw_relerr = FABS(evdw_abserr / y.e_vdW);
     fvdw_abserr = FABS(y.CEvd - y_spline.CEvd);
     fvdw_relerr = FABS(fvdw_abserr / y.CEvd);
     eele_abserr = FABS(y.e_ele - y_spline.e_ele);
     eele_relerr = FABS(eele_abserr / y.e_ele);
     fele_abserr = FABS(y.CEclmb - y_spline.CEclmb);
     fele_relerr = FABS(fele_abserr / y.CEclmb);

     if( evdw_relerr > 1e-10 || eele_relerr > 1e-10 ){
     fprintf( stderr, "rand_dist = %24.15e\n", rand_dist );
     fprintf( stderr, "%24.15e  %24.15e  %24.15e  %24.15e\n",
     y.H, y_spline.H,
     FABS(y.H-y_spline.H), FABS((y.H-y_spline.H)/y.H) );

     fprintf( stderr, "%24.15e  %24.15e  %24.15e  %24.15e\n",
     y.e_vdW, y_spline.e_vdW, evdw_abserr, evdw_relerr );
     fprintf( stderr, "%24.15e  %24.15e  %24.15e  %24.15e\n",
     y.CEvd, y_spline.CEvd, fvdw_abserr, fvdw_relerr );

     fprintf( stderr, "%24.15e  %24.15e  %24.15e  %24.15e\n",
     y.e_ele, y_spline.e_ele, eele_abserr, eele_relerr );
     fprintf( stderr, "%24.15e  %24.15e  %24.15e  %24.15e\n",
             y.CEclmb, y_spline.CEclmb, fele_abserr, fele_relerr );
             }

             if( evdw_relerr > evdw_maxerr )
             evdw_maxerr = evdw_relerr;
             if( eele_relerr > eele_maxerr )
             eele_maxerr = eele_relerr;
             }
             }
             fprintf( stderr, "evdw_maxerr: %24.15e\n", evdw_maxerr );
             fprintf( stderr, "eele_maxerr: %24.15e\n", eele_maxerr );
    *******/

    sfree( h, "Make_LR_Lookup_Table::h" );
    sfree( fh, "Make_LR_Lookup_Table::fh" );
    sfree( fvdw, "Make_LR_Lookup_Table::fvdw" );
    sfree( fCEvd, "Make_LR_Lookup_Table::fCEvd" );
    sfree( fele, "Make_LR_Lookup_Table::fele" );
    sfree( fCEclmb, "Make_LR_Lookup_Table::fCEclmb" );
}


void Finalize_LR_Lookup_Table( reax_system *system, control_params *control,
       static_storage *workspace )
{
    int i, j;
    int num_atom_types;
    int existing_types[MAX_ATOM_TYPES];

    num_atom_types = system->reaxprm.num_atom_types;

    for ( i = 0; i < MAX_ATOM_TYPES; ++i )
    {
        existing_types[i] = 0;
    }
    for ( i = 0; i < system->N; ++i )
    {
        existing_types[ system->atoms[i].type ] = 1;
    }

    for ( i = 0; i < num_atom_types; ++i )
    {
        if ( existing_types[i] )
        {
            for ( j = i; j < num_atom_types; ++j )
            {
                if ( existing_types[j] )
                {
                    sfree( workspace->LR[i][j].y, "Finalize_LR_Lookup_Table::LR[i][j].y" );
                    sfree( workspace->LR[i][j].H, "Finalize_LR_Lookup_Table::LR[i][j].H" );
                    sfree( workspace->LR[i][j].vdW, "Finalize_LR_Lookup_Table::LR[i][j].vdW" );
                    sfree( workspace->LR[i][j].CEvd, "Finalize_LR_Lookup_Table::LR[i][j].CEvd" );
                    sfree( workspace->LR[i][j].ele, "Finalize_LR_Lookup_Table::LR[i][j].ele" );
                    sfree( workspace->LR[i][j].CEclmb, "Finalize_LR_Lookup_Table::LR[i][j].CEclmb" );
                }
            }
        }

        sfree( workspace->LR[i], "Finalize_LR_Lookup_Table::LR[i]" );
    }

    sfree( workspace->LR, "Finalize_LR_Lookup_Table::LR" );
}


static int Lookup_Index_Of( real x, lookup_table* t )
{
    return (int)( t->a * ( x - t->xmin ) );
}


real Lookup( real x, lookup_table* t )
{
    real x1, x2;
    real b;
    int i;

    /*
    if ( x < t->xmin)
    {
       fprintf(stderr,"Domain check %lf > %lf\n",t->xmin,x);
       exit(0);
    }
    if ( x > t->xmax)
    {
       fprintf(stderr,"Domain check %lf < %lf\n",t->xmax,x);
       exit(0);
    }
    */

    i = Lookup_Index_Of( x, t );
    x1 = i * t->dx + t->xmin;
    x2 = (i + 1) * t->dx + t->xmin;

    b = ( x2 * t->y[i] - x1 * t->y[i + 1] ) * t->inv_dx;
    // fprintf( stdout,"SLookup_Entry: %d, %lf, %lf, %lf, %lf: %lf, %lf\n",
    //          i,x1,x2,x,b,t->one_over_dx*(t->y[i+1]-t->y[i])*x+b,exp(x));

    return t->inv_dx * ( t->y[i + 1] - t->y[i] ) * x + b;
}
