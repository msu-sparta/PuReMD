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
  #include "tool_box.h"
#elif defined(LAMMPS_REAX)
  #include "reax_tool_box.h"
#endif


/************** taken from comm_tools.c **************/
int SumScan( int n, int me, int root, MPI_Comm comm )
{
    int i, my_order, wsize;
    int *nbuf = NULL;

    if ( me == root )
    {
        MPI_Comm_size( comm, &wsize );
        nbuf = (int *) scalloc( wsize, sizeof(int), "SumScan:nbuf" );

        MPI_Gather( &n, 1, MPI_INT, nbuf, 1, MPI_INT, root, comm );

        for ( i = 0; i < wsize - 1; ++i )
        {
            nbuf[i + 1] += nbuf[i];
        }

        MPI_Scatter( nbuf, 1, MPI_INT, &my_order, 1, MPI_INT, root, comm );

        sfree( nbuf, "SumScan:nbuf" );
    }
    else
    {
        MPI_Gather( &n, 1, MPI_INT, nbuf, 1, MPI_INT, root, comm );
        MPI_Scatter( nbuf, 1, MPI_INT, &my_order, 1, MPI_INT, root, comm );
    }

    return my_order;
}


void SumScanB( int n, int me, int wsize, int root, MPI_Comm comm, int *nbuf )
{
    int i;

    MPI_Gather( &n, 1, MPI_INT, nbuf, 1, MPI_INT, root, comm );

    if ( me == root )
    {
        for ( i = 0; i < wsize - 1; ++i )
        {
            nbuf[i + 1] += nbuf[i];
        }
    }

    MPI_Bcast( nbuf, wsize, MPI_INT, root, comm );
}


/* determine whether point p is inside the box */
void Fit_to_Periodic_Box( simulation_box *box, rvec *p )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        if ( (*p)[i] < box->min[i] )
        {
            /* handle lower coords */
            while ( (*p)[i] < box->min[i] )
            {
                (*p)[i] += box->box_norms[i];
            }
        }
        else if ( (*p)[i] >= box->max[i] )
        {
            /* handle higher coords */
            while ( (*p)[i] >= box->max[i] )
            {
                (*p)[i] -= box->box_norms[i];
            }
        }
    }
}


/************** from geo_tools.c *****************/
void Make_Point( real x, real y, real z, rvec* p )
{
    (*p)[0] = x;
    (*p)[1] = y;
    (*p)[2] = z;
}


int is_Valid_Serial( storage *workspace, int serial )
{
//    if( workspace->map_serials[ serial ] < 0 )
//    {
//        fprintf( stderr, "CONECT line includes invalid pdb serial number %d.\n",
//                serial );
//        fprintf( stderr, "Please correct the input file.Terminating...\n" );
//        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
//    }

    return SUCCESS;
}


int Check_Input_Range( int val, int lo, int hi, char *message )
{
    if ( val < lo || val > hi )
    {
        fprintf( stderr, "%s\nInput %d - Out of range %d-%d. Terminating...\n",
                 message, val, lo, hi );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
    }

    return 1;
}


void Trim_Spaces( char *element )
{
    int i, j;

    // skip initial space chars
    for ( i = 0; element[i] == ' '; ++i );

    for ( j = i; j < (int)(strlen(element)) && element[j] != ' '; ++j )
    {
        // make uppercase, offset to 0
        element[j - i] = toupper( element[j] );
    }
    // finalize the string
    element[j - i] = 0;
}


/************ from system_props.c *************/
real Get_Time( )
{
    struct timeval tim;

    gettimeofday( &tim, NULL );

    return ( tim.tv_sec + (tim.tv_usec / 1000000.0) );
}


real Get_Timing_Info( real t_start )
{
    struct timeval tim;
    real t_end;

    gettimeofday(&tim, NULL );
    t_end = tim.tv_sec + (tim.tv_usec / 1000000.0);

    return (t_end - t_start);
}


void Update_Timing_Info( real *t_start, real *timing )
{
    struct timeval tim;
    real t_end;

    gettimeofday( &tim, NULL );

    t_end = tim.tv_sec + (tim.tv_usec / 1000000.0);
    *timing += (t_end - *t_start);
    *t_start = t_end;
}


/*********** from io_tools.c **************/
int Get_Atom_Type( reax_interaction *reax_param, char *s )
{
    int i, ret, flag;
    
    flag = FAILURE;
    ret = -1;

    for ( i = 0; i < reax_param->num_atom_types; ++i )
    {
        if ( strncmp( reax_param->sbp[i].name, s, 15 ) == 0 )
        {
            ret = i;
            flag = SUCCESS;
            break;
        }
    }

    if ( flag == FAILURE )
    {
        fprintf( stderr, "[ERROR] Unknown atom type (%s). Terminating...\n", s );
        MPI_Abort( MPI_COMM_WORLD, UNKNOWN_ATOM_TYPE );
    }

    return ret;
}


char *Get_Element( reax_system *system, int i )
{
    return &system->reax_param.sbp[system->my_atoms[i].type].name[0];
}


char *Get_Atom_Name( reax_system *system, int i )
{
    return &system->my_atoms[i].name[0];
}


void Allocate_Tokenizer_Space( char **line, char **backup, char ***tokens )
{
    int i;

    *line = (char*) smalloc( sizeof(char) * MAX_LINE, "Tokenizer:line" );

    *backup = (char*) smalloc( sizeof(char) * MAX_LINE, "Tokenizer:backup" );

    *tokens = (char**) smalloc( sizeof(char*) * MAX_TOKENS, "Tokenizer:tokens" );

    for ( i = 0; i < MAX_TOKENS; i++ )
    {
        (*tokens)[i] = (char*) smalloc(sizeof(char) * MAX_TOKEN_LEN, "Tokenizer:tokens[i]" );
    }
}


int Tokenize( const char* s, char*** tok )
{
    char test[MAX_LINE];
    char *sep = "\t \n!=";
    char *word;
    char *saveptr = NULL;
    int count = 0;

    if ( s == NULL )
    {
        fprintf( stderr, "[WARNING] passed null string to tokenizer. Returning...\n" );
        return count;
    }

    strncpy( test, s, MAX_LINE );

    for ( word = strtok_r( test, sep, &saveptr );
            word != NULL;
            word = strtok_r( NULL, sep, &saveptr ) )
    {
        strncpy( (*tok)[count], word, MAX_LINE );
        ++count;
    }

    return count;
}


/* Safe wrapper around libc malloc
 *
 * n: num. of bytes to allocated
 * name: message with details about pointer, used for warnings/errors
 *
 * returns: ptr to allocated memory
 * */
void * smalloc( size_t n, const char *name )
{
    void *ptr;

    if ( n == 0 )
    {
        fprintf( stderr, "[ERROR] failed to allocate %zu bytes for array %s.\n",
                n, name );
        exit( INSUFFICIENT_MEMORY );
    }

#if defined(DEBUG)
    fprintf( stderr, "[INFO] requesting %zu bytes for %s\n", n, name );
    fflush( stderr );
#endif

    ptr = malloc( n );

    if ( ptr == NULL )
    {
        fprintf( stderr, "[ERROR] failed to allocate %zu bytes for array %s.\n",
                n, name );
        exit( INSUFFICIENT_MEMORY );
    }

#if defined(DEBUG)
    fprintf( stderr, "[INFO] granted memory at address: %p\n", ptr );
    fflush( stderr );
#endif

    return ptr;
}


/* Safe wrapper around libc realloc
 *
 * n: num. of bytes to reallocated
 * name: message with details about pointer, used for warnings/errors
 *
 * returns: ptr to reallocated memory
 * */
void * srealloc( void *ptr, size_t n, const char *name )
{
    void *new_ptr;

    if ( n == 0 )
    {
        fprintf( stderr, "[ERROR] failed to reallocate %zu bytes for array %s.\n",
                n, name );
        exit( INSUFFICIENT_MEMORY );
    }

#if defined(DEBUG)
    if ( ptr == NULL )
    {
        fprintf( stderr, "[INFO] trying to allocate %zu NEW bytes for array %s.\n",
                n, name );
    }

    fprintf( stderr, "[INFO] requesting %zu bytes for %s\n", n, name );
    fflush( stderr );
#endif

    new_ptr = realloc( ptr, n );

    /* technically, ptr may still be allocated and valid,
     * but we needed more memory, so abort */
    if ( new_ptr == NULL )
    {
        fprintf( stderr, "[ERROR] failed to reallocate %zu bytes for array %s.\n",
                n, name );
        exit( INSUFFICIENT_MEMORY );
    }

#if defined(DEBUG)
    fprintf( stderr, "[INFO] granted memory at address: %p\n", new_ptr );
    fflush( stderr );
#endif

    return new_ptr;
}


/* Safe wrapper around libc calloc
 *
 * n: num. of elements to allocated (each of size bytes)
 * size: num. of bytes per element
 * name: message with details about pointer, used for warnings/errors
 *
 * returns: ptr to allocated memory, all bits initialized to zeros
 * */
void * scalloc( size_t n, size_t size, const char *name )
{
    void *ptr;

    if ( n == 0 )
    {
        fprintf( stderr, "[ERROR] failed to allocate %zu bytes for array %s.\n",
                n * size, name );
        exit( INSUFFICIENT_MEMORY );
    }

#if defined(DEBUG)
    fprintf( stderr, "[INFO] requesting %zu bytes for %s\n", n * size, name );
    fflush( stderr );
#endif

    ptr = calloc( n, size );

    if ( ptr == NULL )
    {
        fprintf( stderr, "[ERROR] failed to allocate %zu bytes for array %s.\n",
                n * size, name );
        exit( INSUFFICIENT_MEMORY );
    }

#if defined(DEBUG)
    fprintf( stderr, "[INFO] granted memory at address: %p\n", ptr );
    fflush( stderr );
#endif

    return ptr;
}


/* Safe wrapper around libc free
 *
 * ptr: pointer to dynamically allocated memory which will be deallocated
 * name: message with details about pointer, used for warnings/errors
 * */
void sfree( void *ptr, const char *name )
{
    if ( ptr == NULL )
    {
        fprintf( stderr, "[WARNING] trying to free the already NULL pointer %s!\n",
                name );
        return;
    }

#if defined(DEBUG)
    fprintf( stderr, "[INFO] trying to free pointer (%s), address: %p\n",
            name, (void *) ptr );
    fflush( stderr );
#endif

    free( ptr );
}


/* Safe wrapper around libc fopen
 *
 * fname: name of file to be opened
 * mode: mode in which to open file
 * msg: message to be printed in case of error
 * */
FILE * sfopen( const char * fname, const char * mode, const char * msg )
{
    FILE * ptr;

    if ( fname == NULL )
    {
        fprintf( stderr, "[ERROR] trying to open file: NULL file name (%s). Terminating...\n",
                msg );
        exit( INVALID_INPUT );
    }
    if ( mode == NULL )
    {
        fprintf( stderr, "[ERROR] trying to open file: NULL mode (%s). Terminating...\n",
                msg );
        exit( INVALID_INPUT );
    }

    ptr = fopen( fname, mode );

    if ( ptr == NULL )
    {
        fprintf( stderr, "[ERROR] failed to open file %s with mode %s (%s)\n",
              fname, mode, msg );
        exit( INVALID_INPUT );
    }

    return ptr;
}


/* Safe wrapper around libc fclose
 *
 * fname: name of file to be opened
 * mode: mode in which to open file
 * msg: message to be printed in case of error
 * */
void sfclose( FILE * fp, const char * msg )
{
    int ret;

    if ( fp == NULL )
    {
        fprintf( stderr, "[WARNING] trying to close NULL file pointer (%s). Returning...\n", msg );
        return;
    }

    ret = fclose( fp );

    if ( ret != 0 )
    {
        fprintf( stderr, "[ERROR] error detected when closing file (%s). Terminating...\n", msg );
        exit( INVALID_INPUT );
    }
}
