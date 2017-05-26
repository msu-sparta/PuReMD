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

#ifndef __TOOL_BOX_H_
#define __TOOL_BOX_H_

#include "reax_types.h"


/* from comm_tools.h */
int SumScan( int, int, int, MPI_Comm );

void SumScanB( int, int, int, int, MPI_Comm, int* );

/* from box.h */
void Transform_to_UnitBox( rvec, simulation_box*, char, rvec );

void Fit_to_Periodic_Box( simulation_box*, rvec* );

/* from geo_tools.h */
void Make_Point( real, real, real, rvec* );

int is_Valid_Serial( storage*, int );

int Check_Input_Range( int, int, int, char* );

void Trim_Spaces( char* );

/* from system_props.h */
#ifdef __cplusplus
extern "C" {
#endif

real Get_Time( );

real Get_Timing_Info( real );

void Update_Timing_Info( real*, real* );

#ifdef __cplusplus
}
#endif

/* from io_tools.h */
int Get_Atom_Type( reax_interaction*, char* );

char *Get_Element( reax_system*, int );

char *Get_Atom_Name( reax_system*, int );

int Allocate_Tokenizer_Space( char**, char**, char*** );

int Tokenize( char*, char*** );

/* from lammps */
void *smalloc( size_t, char* );

void* srealloc( void *ptr, size_t n, char *name );

void *scalloc( size_t, size_t, char* );

void sfree( void*, char* );


#if defined(LAMMPS_REAX) || defined(PURE_REAX)
/* determine the touch point, tp, of a box to
   its neighbor denoted by the relative coordinate rl */
static inline void Box_Touch_Point( simulation_box *box, ivec rl, rvec tp )
{
    int d;

    for ( d = 0; d < 3; ++d )
    {
        if ( rl[d] == -1 )
        {
            tp[d] = box->min[d];
        }
        else if ( rl[d] == 0 )
        {
            tp[d] = NEG_INF - 1.;
        }
        else
        {
            tp[d] = box->max[d];
        }
    }
}


/* determine whether point p is inside the box */
/* assumes orthogonal box */
static inline int is_Inside_Box( simulation_box *box, rvec p )
{
    if ( p[0] < box->min[0] || p[0] >= box->max[0] ||
            p[1] < box->min[1] || p[1] >= box->max[1] ||
            p[2] < box->min[2] || p[2] >= box->max[2] )
    {
        return FALSE;
    }

    return TRUE;
}


static inline int iown_midpoint( simulation_box *box, rvec p1, rvec p2 )
{
    rvec midp;

    midp[0] = (p1[0] + p2[0]) / 2;
    midp[1] = (p1[1] + p2[1]) / 2;
    midp[2] = (p1[2] + p2[2]) / 2;

    if ( midp[0] < box->min[0] || midp[0] >= box->max[0] ||
            midp[1] < box->min[1] || midp[1] >= box->max[1] ||
            midp[2] < box->min[2] || midp[2] >= box->max[2] )
    {
        return FALSE;
    }

    return TRUE;
}


static inline void GridCell_Closest_Point( grid_cell *gci, grid_cell *gcj,
        ivec ci, ivec cj, rvec cp )
{
    int  d;

    for ( d = 0; d < 3; d++ )
    {
        if ( cj[d] > ci[d] )
        {
            cp[d] = gcj->min[d];
        }
        else if ( cj[d] == ci[d] )
        {
            cp[d] = NEG_INF - 1.;
        }
        else
        {
            cp[d] = gcj->max[d];
        }
    }
}


static inline void GridCell_Touch_Point( grid_cell *gc, ivec rl, rvec fp )
{
    int d;

    for ( d = 0; d < 3; ++d )
    {
        if ( rl[d] == -1 )
        {
            fp[d] = gc->min[d];
        }
        else if ( rl[d] == 0 )
        {
            fp[d] = NEG_INF - 1.;
        }
        else
        {
            fp[d] = gc->max[d];
        }
    }
}



static inline real DistSqr_to_CP( rvec cp, rvec x )
{
    int  i;
    real d_sqr = 0;

    for ( i = 0; i < 3; ++i )
    {
        if ( cp[i] > NEG_INF )
        {
            d_sqr += SQR( cp[i] - x[i] );
        }
    }

    return d_sqr;
}


static inline int Relative_Coord_Encoding( ivec c )
{
    return 9 * (c[0] + 1) + 3 * (c[1] + 1) + (c[2] + 1);
}


static inline real DistSqr_to_Special_Point( rvec cp, rvec x )
{
    int  i;
    real d_sqr = 0;

    for ( i = 0; i < 3; ++i )
    {
        if ( cp[i] > NEG_INF )
        {
            d_sqr += SQR( cp[i] - x[i] );
        }
    }

    return d_sqr;
}
#endif

#endif
