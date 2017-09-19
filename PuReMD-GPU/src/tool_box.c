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

#include "tool_box.h"

#include <ctype.h>


/************** taken from box.c **************/
void Transform( rvec x1, simulation_box *box, char flag, rvec x2 )
{
    int i, j;
    real tmp;

    //  printf(">x1: (%lf, %lf, %lf)\n",x1[0],x1[1],x1[2]);

    if (flag > 0)
    {
        for (i = 0; i < 3; i++)
        {
            tmp = 0.0;
            for (j = 0; j < 3; j++)
                tmp += box->trans[i][j] * x1[j];
            x2[i] = tmp;
        }
    }
    else
    {
        for (i = 0; i < 3; i++)
        {
            tmp = 0.0;
            for (j = 0; j < 3; j++)
                tmp += box->trans_inv[i][j] * x1[j];
            x2[i] = tmp;
        }
    }
    //  printf(">x2: (%lf, %lf, %lf)\n", x2[0], x2[1], x2[2]);
}


void Transform_to_UnitBox( rvec x1, simulation_box *box, char flag, rvec x2 )
{
    Transform( x1, box, flag, x2 );

    x2[0] /= box->box_norms[0];
    x2[1] /= box->box_norms[1];
    x2[2] /= box->box_norms[2];
}


/* determine whether point p is inside the box */
void Fit_to_Periodic_Box( simulation_box *box, rvec *p )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        //TODO: verify box boundary coordinates -- assuming orthogonal box pinned at origin
        if ( (*p)[i] < 0. )
        {
            /* handle lower coords */
            while ( (*p)[i] < 0. )
                (*p)[i] += box->box_norms[i];
        }
        else if ( (*p)[i] >= box->box_norms[i] )
        {
            /* handle higher coords */
            while ( (*p)[i] >= box->box_norms[i] )
                (*p)[i] -= box->box_norms[i];
        }
//        if ( (*p)[i] < box->min[i] )
//        {
//            /* handle lower coords */
//            while ( (*p)[i] < box->min[i] )
//                (*p)[i] += box->box_norms[i];
//        }
//        else if ( (*p)[i] >= box->max[i] )
//        {
//            /* handle higher coords */
//            while ( (*p)[i] >= box->max[i] )
//                (*p)[i] -= box->box_norms[i];
//        }
    }
}


/* determine the touch point, tp, of a box to
   its neighbor denoted by the relative coordinate rl */
/*
inline void Box_Touch_Point( simulation_box *box, ivec rl, rvec tp )
{
    int d;

    for ( d = 0; d < 3; ++d )
        if ( rl[d] == -1 )
            tp[d] = box->min[d];
        else if ( rl[d] == 0 )
            tp[d] = NEG_INF - 1.;
        else
            tp[d] = box->max[d];
}
*/


/* determine whether point p is inside the box */
/* assumes orthogonal box */
/*
inline int is_Inside_Box( simulation_box *box, rvec p )
{
    if ( p[0] < box->min[0] || p[0] >= box->max[0] ||
            p[1] < box->min[1] || p[1] >= box->max[1] ||
            p[2] < box->min[2] || p[2] >= box->max[2] )
        return FALSE;

    return TRUE;
}
*/


/*
inline int iown_midpoint( simulation_box *box, rvec p1, rvec p2 )
{
    rvec midp;

    midp[0] = (p1[0] + p2[0]) / 2;
    midp[1] = (p1[1] + p2[1]) / 2;
    midp[2] = (p1[2] + p2[2]) / 2;

    if ( midp[0] < box->min[0] || midp[0] >= box->max[0] ||
            midp[1] < box->min[1] || midp[1] >= box->max[1] ||
            midp[2] < box->min[2] || midp[2] >= box->max[2] )
        return FALSE;

    return TRUE;
}
*/


/**************** from grid.c ****************/
/* finds the closest point of grid cell cj to ci.
   no need to consider periodic boundary conditions as in the serial case
   because the box of a process is not periodic in itself */
/*
inline void GridCell_Closest_Point( grid_cell *gci, grid_cell *gcj,
        ivec ci, ivec cj, rvec cp )
{
    int  d;

    for ( d = 0; d < 3; d++ )
        if ( cj[d] > ci[d] )
            cp[d] = gcj->min[d];
        else if ( cj[d] == ci[d] )
            cp[d] = NEG_INF - 1.;
        else
            cp[d] = gcj->max[d];
}


inline void GridCell_to_Box_Points( grid_cell *gc, ivec rl, rvec cp, rvec fp )
{
    int d;

    for ( d = 0; d < 3; ++d )
        if ( rl[d] == -1 )
        {
            cp[d] = gc->min[d];
            fp[d] = gc->max[d];
        }
        else if ( rl[d] == 0 )
        {
            cp[d] = fp[d] = NEG_INF - 1.;
        }
        else
        {
            cp[d] = gc->max[d];
            fp[d] = gc->min[d];
        }
}


inline real DistSqr_between_Special_Points( rvec sp1, rvec sp2 )
{
    int  i;
    real d_sqr = 0;

    for ( i = 0; i < 3; ++i )
    {
        if ( sp1[i] > NEG_INF && sp2[i] > NEG_INF )
        {
            d_sqr += SQR( sp1[i] - sp2[i] );
        }
    }

    return d_sqr;
}


inline real DistSqr_to_Special_Point( rvec cp, rvec x )
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


inline int Relative_Coord_Encoding( ivec c )
{
    return 9 * (c[0] + 1) + 3 * (c[1] + 1) + (c[2] + 1);
}
*/


/************** from geo_tools.c *****************/
void Make_Point( real x, real y, real z, rvec* p )
{
    (*p)[0] = x;
    (*p)[1] = y;
    (*p)[2] = z;
}


int is_Valid_Serial( static_storage *workspace, int serial )
{
    if( workspace->map_serials[ serial ] < 0 )
    {
        fprintf( stderr, "CONECT line includes invalid pdb serial number %d.\n", serial );
        fprintf( stderr, "Please correct the input file.Terminating...\n" );
        exit( INVALID_INPUT );
    }

    return TRUE;
}


int Check_Input_Range( int val, int lo, int hi, char *message )
{
    if ( val < lo || val > hi )
    {
        fprintf( stderr, "%s\nInput %d - Out of range %d-%d. Terminating...\n",
                 message, val, lo, hi );
        exit( INVALID_INPUT );
    }

    return SUCCESS;
}


void Trim_Spaces( char *element )
{
    int i, j;

    for ( i = 0; element[i] == ' '; ++i ); // skip initial space chars

    for ( j = i; j < (int)(strlen(element)) && element[j] != ' '; ++j )
    {
        element[j - i] = toupper( element[j] ); // make uppercase, offset to 0
    }
    element[j - i] = 0; // finalize the string
}


/************ from system_props.c *************/
real Get_Time( )
{
    gettimeofday(&tim, NULL );
    return ( tim.tv_sec + (tim.tv_usec / 1000000.0) );
}


real Get_Timing_Info( real t_start )
{
    gettimeofday(&tim, NULL );
    t_end = tim.tv_sec + (tim.tv_usec / 1000000.0);
    return (t_end - t_start);
}


void Update_Timing_Info( real *t_start, real *timing )
{
    gettimeofday(&tim, NULL );
    t_end = tim.tv_sec + (tim.tv_usec / 1000000.0);
    *timing += (t_end - *t_start);
    *t_start = t_end;
}


/*********** from io_tools.c **************/
int Get_Atom_Type( reax_interaction *reax_param, char *s )
{
    int i;

    for ( i = 0; i < reax_param->num_atom_types; ++i )
    {
        if ( !strcmp( reax_param->sbp[i].name, s ) )
        {
            return i;
        }
    }

    fprintf( stderr, "Unknown atom type %s. Terminating...\n", s );
    exit( UNKNOWN_ATOM_TYPE );

    return FAILURE;
}


char *Get_Element( reax_system *system, int i )
{
    return &( system->reaxprm.sbp[system->atoms[i].type].name[0] );
}


char *Get_Atom_Name( reax_system *system, int i )
{
    return &(system->atoms[i].name[0]);
}


int Allocate_Tokenizer_Space( char **line, char **backup, char ***tokens )
{
    int i;

    if ( (*line = (char*) malloc( sizeof(char) * MAX_LINE )) == NULL )
    {
        return FAILURE;
    }

    if ( (*backup = (char*) malloc( sizeof(char) * MAX_LINE )) == NULL )
    {
        return FAILURE;
    }

    if ( (*tokens = (char**) malloc( sizeof(char*) * MAX_TOKENS )) == NULL )
    {
        return FAILURE;
    }

    for ( i = 0; i < MAX_TOKENS; i++ )
    {
        if ( ((*tokens)[i] = (char*) malloc(sizeof(char) * MAX_TOKEN_LEN)) == NULL )
        {
            return FAILURE;
        }
    }

    return SUCCESS;
}


int Tokenize( char* s, char*** tok )
{
    char test[MAX_LINE];
    char *sep = "\t \n!=";
    char *word;
    int count = 0;

    strncpy( test, s, MAX_LINE );

    for ( word = strtok(test, sep); word; word = strtok(NULL, sep) )
    {
        strncpy( (*tok)[count], word, MAX_LINE );
        count++;
    }

    return count;
}


/***************** taken from lammps ************************/
/* safe malloc */
void *smalloc( long n, char *name )
{
    void *ptr;

    if ( n <= 0 )
    {
        fprintf( stderr, "WARNING: trying to allocate %ld bytes for array %s. ",
                 n, name );
        fprintf( stderr, "returning NULL.\n" );
        return NULL;
    }

    ptr = malloc( n );
    if ( ptr == NULL )
    {
        fprintf( stderr, "ERROR: failed to allocate %ld bytes for array %s",
                 n, name );
        exit( INSUFFICIENT_MEMORY );
    }

    return ptr;
}


/* safe calloc */
void *scalloc( int n, int size, char *name )
{
    void *ptr;

    if ( n <= 0 )
    {
        fprintf( stderr, "WARNING: trying to allocate %d elements for array %s. ",
                 n, name );
        fprintf( stderr, "returning NULL.\n" );
        return NULL;
    }

    if ( size <= 0 )
    {
        fprintf( stderr, "WARNING: elements size for array %s is %d. ",
                 name, size );
        fprintf( stderr, "returning NULL.\n" );
        return NULL;
    }

    ptr = calloc( n, size );
    if ( ptr == NULL )
    {
        fprintf( stderr, "ERROR: failed to allocate %d bytes for array %s",
                 n * size, name );
        exit( INSUFFICIENT_MEMORY );
    }

    return ptr;
}


/* safe free */
void sfree( void *ptr, char *name )
{
    if ( ptr == NULL )
    {
        fprintf( stderr, "WARNING: trying to free the already NULL pointer %s!\n",
                 name );
        return;
    }

    free( ptr );
    ptr = NULL;
}
