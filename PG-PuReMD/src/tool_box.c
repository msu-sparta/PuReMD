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
  #include "comm_tools.h"
#elif defined(LAMMPS_REAX)
  #include "reax_tool_box.h"
  #include "reax_comm_tools.h"
#endif


/************** taken from comm_tools.c **************/
int SumScan( int n, int me, int root, MPI_Comm comm )
{
    int i, my_order, wsize, *nbuf, ret;

    nbuf = NULL;

    if ( me == root )
    {
        MPI_Comm_size( comm, &wsize );
        nbuf = (int *) scalloc( wsize, sizeof(int), "SumScan:nbuf" );

        ret = MPI_Gather( &n, 1, MPI_INT, nbuf, 1, MPI_INT, root, comm );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        for ( i = 0; i < wsize - 1; ++i )
        {
            nbuf[i + 1] += nbuf[i];
        }

        ret = MPI_Scatter( nbuf, 1, MPI_INT, &my_order, 1, MPI_INT, root, comm );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        sfree( nbuf, "SumScan:nbuf" );
    }
    else
    {
        ret = MPI_Gather( &n, 1, MPI_INT, nbuf, 1, MPI_INT, root, comm );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        ret = MPI_Scatter( nbuf, 1, MPI_INT, &my_order, 1, MPI_INT, root, comm );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
    }

    return my_order;
}


void SumScanB( int n, int me, int wsize, int root, MPI_Comm comm, int *nbuf )
{
    int i, ret;

    ret = MPI_Gather( &n, 1, MPI_INT, nbuf, 1, MPI_INT, root, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

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


/* Get the current time
 *
 * returns: current time in seconds */
real Get_Time( )
{
    return MPI_Wtime( );
}


/* Get the elapsed time given a starting time
 *
 * t_start: starting time in seconds
 * returns: elapsed time in seconds */
real Get_Elapsed_Time( real t_start )
{
    return MPI_Wtime( ) - t_start;
}


/* Accumulate elapsed time into timing and update starting time
 * to be new time from this instant going forward
 *
 * t_start: previous starting time
 * timing: variable to accumulate elapsed time into */
void Update_Timing_Info( real *t_start, real *timing )
{
    double t_end;

    t_end = MPI_Wtime( );
    *timing += t_end - *t_start;
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

    *line = smalloc( sizeof(char) * MAX_LINE, "Allocate_Tokenizer_Space::line" );
    *backup =  smalloc( sizeof(char) * MAX_LINE, "Allocate_Tokenizer_Space::backup" );
    *tokens = smalloc( sizeof(char*) * MAX_TOKENS, "Allocate_Tokenizer_Space::tokens" );

    for ( i = 0; i < MAX_TOKENS; i++ )
    {
        (*tokens)[i] = smalloc( sizeof(char) * MAX_TOKEN_LEN, "Allocate_Tokenizer_Space::tokens[i]" );
    }
}


int Tokenize( char* s, char*** tok, size_t token_len )
{
    int count = 0;
    char test[MAX_LINE];
    char *sep = "\t \n!=";
    char *word, *saveptr;

    strncpy( test, s, sizeof(test) - 1 );
    test[sizeof(test) - 1] = '\0';

    for ( word = strtok_r(test, sep, &saveptr); word != NULL;
            word = strtok_r(NULL, sep, &saveptr) )
    {
        strncpy( (*tok)[count], word, token_len - 1 );
        (*tok)[count][token_len - 1] = '\0';
        count++;
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

#if defined(DEBUG_FOCUS)
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

#if defined(DEBUG_FOCUS)
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

#if defined(DEBUG_FOCUS)
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

#if defined(DEBUG_FOCUS)
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

#if defined(DEBUG_FOCUS)
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

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] granted memory at address: %p\n", ptr );
    fflush( stderr );
#endif

    return ptr;
}


/* Safe wrapper around check first and reallocate-if-needed routine:
 * checks if the amount of space currently allocated to ptr is sufficient,
 * and, if not, frees any space allocated to ptr before allocating the
 * requested amount of space
 *
 * ptr: pointer to memory allocation
 * cur_size: num. of bytes currently allocated
 * new_size: num. of bytes to be newly allocated, if needed
 * msg: message with details about pointer, used for warnings/errors
 * */
void check_smalloc( void **ptr, size_t *cur_size, size_t new_size, const char *msg )
{
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] requesting %zu bytes for %s (%zu currently allocated)\n",
            new_size, msg, *cur_size );
    fflush( stderr );
#endif

    assert( new_size > 0 );

    if ( new_size > *cur_size )
    {
        if ( *cur_size > 0 && *ptr == NULL )
        {
            sfree( *ptr, msg );
        }

        //TODO: look into using aligned alloc's
        /* intentionally over-allocate to reduce the number of allocation operations,
         * and record the new allocation size */
        *cur_size = new_size;
        *ptr = smalloc( *cur_size, msg );
    }
}


/* Safe wrapper around check first and reallocate-if-needed routine:
 * checks if the amount of space currently allocated to ptr is sufficient,
 * and, if not, frees any space allocated to ptr before allocating the
 * requested amount of space
 *
 * ptr: pointer to memory allocation
 * cur_size: num. of bytes currently allocated
 * new_size: num. of bytes to be newly allocated, if needed
 * msg: message with details about pointer, used for warnings/errors
 * */
void check_srealloc( void **ptr, size_t *cur_size, size_t new_size, const char *msg )
{
    void *new_ptr;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] requesting %zu bytes for %s (%zu currently allocated)\n",
            new_size, msg, *cur_size );
    fflush( stderr );
#endif

    assert( new_size > 0 );

    if ( new_size > *cur_size )
    {
        //TODO: look into using aligned alloc's
        /* intentionally over-allocate to reduce the number of allocation operations,
         * and record the new allocation size */
        *cur_size = new_size;
        new_ptr = srealloc( *ptr, *cur_size, msg );
        *ptr = new_ptr;
    }
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

#if defined(DEBUG_FOCUS)
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
