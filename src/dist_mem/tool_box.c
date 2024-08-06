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
  #include "../common/include/config.h"
#endif

#if defined(PURE_REAX)
  #include "tool_box.h"

  #include "comm_tools.h"
#elif defined(LAMMPS_REAX)
  #include "reax_tool_box.h"

  #include "reax_comm_tools.h"
#endif

#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <errno.h>
#include <time.h>

#if defined(HAVE_CUDA)
  #include "cuda/gpu_allocate.h"
#elif defined(HAVE_HIP)
  #include "hip/gpu_allocate.h"
#endif

/* base 10 for result of string-to-integer conversion */
#define INTBASE (10)


/************** taken from comm_tools.c **************/
int SumScan( int n, int me, int root, MPI_Comm comm )
{
    int i, my_order, wsize, *nbuf, ret;

    nbuf = NULL;

    if ( me == root )
    {
        MPI_Comm_size( comm, &wsize );
        nbuf = (int *) scalloc( wsize, sizeof(int), __FILE__, __LINE__ );

        ret = MPI_Gather( &n, 1, MPI_INT, nbuf, 1, MPI_INT, root, comm );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        for ( i = 0; i < wsize - 1; ++i )
        {
            nbuf[i + 1] += nbuf[i];
        }

        ret = MPI_Scatter( nbuf, 1, MPI_INT, &my_order, 1, MPI_INT, root, comm );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        sfree( nbuf, __FILE__, __LINE__ );
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

    ret = MPI_Bcast( nbuf, wsize, MPI_INT, root, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
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


void Trim_Spaces( char * const element, const size_t size )
{
    int i, j, n;

    element[size - 1] = '\0';
    n = strlen( element );

    /* buffer not NULL-terminated, abort */
    if ( n == size )
    {
        fprintf( stderr, "[ERROR] buffer not NULL-terminated (Trim_Spaces). Terminating...\n" );
        exit( RUNTIME_ERROR );
    }

    /* skip initial space chars */
    for ( i = 0; element[i] == ' '; ++i )
        ;

    /* make uppercase, offset to 0 */
    for ( j = i; j < n && element[j] != ' '; ++j )
    {
        element[j - i] = toupper( element[j] );
    }

    /* NULL terminate */
    element[j - i] = '\0';
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
 * t_start: previous starting time in seconds
 * timing: variable to accumulate elapsed time in seconds */
void Update_Timing_Info( real *t_start, real *timing )
{
    double t_end;

    t_end = MPI_Wtime( );
    *timing += t_end - *t_start;
    *t_start = t_end;
}


/*********** from io_tools.c **************/
int Get_Atom_Type( reax_interaction *reax_param, char *s, size_t n )
{
    int i, ret, flag;
    
    flag = FAILURE;

    for ( i = 0; i < reax_param->num_atom_types; ++i )
    {
        if ( strncmp( reax_param->sbp[i].name, s,
                    MIN( sizeof(reax_param->sbp[i].name), n ) ) == 0 )
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


void Allocate_Tokenizer_Space( char **line, size_t line_size,
        char **backup, size_t backup_size,
        char ***tokens, size_t num_tokens, size_t token_size )
{
    int i;

    *line = smalloc( sizeof(char) * line_size, __FILE__, __LINE__ );
    *backup = smalloc( sizeof(char) * backup_size, __FILE__, __LINE__ );
    *tokens = smalloc( sizeof(char*) * num_tokens, __FILE__, __LINE__ );

    for ( i = 0; i < num_tokens; i++ )
    {
        (*tokens)[i] = smalloc( sizeof(char) * token_size,
                __FILE__, __LINE__ );
    }
}


void Deallocate_Tokenizer_Space( char **line, char **backup,
        char ***tokens, size_t num_tokens )
{
    int i;

    for ( i = 0; i < num_tokens; i++ )
    {
        sfree( (*tokens)[i], __FILE__, __LINE__ );
    }

    sfree( *line, __FILE__, __LINE__ );
    sfree( *backup, __FILE__, __LINE__ );
    sfree( *tokens, __FILE__, __LINE__ );
}


int Tokenize( char* s, char*** tok, size_t token_len )
{
    int count = 0;
    char test[MAX_LINE];
    char *sep = "\t \n!=";
    char *word;

    strncpy( test, s, sizeof(test) - 1 );
    test[sizeof(test) - 1] = '\0';

    for ( word = strtok(test, sep); word != NULL;
            word = strtok(NULL, sep) )
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
 * filename: source filename of caller
 * line: source line of caller
 *
 * returns: ptr to allocated memory
 * */
void * smalloc( size_t n, const char * const filename, int line )
{
    void *ptr;

    if ( n == 0 )
    {
        fprintf( stderr, "[ERROR] failed to allocate %zu bytes for array\n",
                n );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        exit( INSUFFICIENT_MEMORY );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] requesting allocation of %zu bytes of memory at line %d in file %.*s\n",
            n, line, (int) strlen(filename), filename );
    fflush( stderr );
#endif

    ptr = malloc( n );

    if ( ptr == NULL )
    {
        fprintf( stderr, "[ERROR] failed to allocate %zu bytes for array\n",
                n );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        exit( INSUFFICIENT_MEMORY );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] address: %p [SMALLOC]\n", (void *) ptr );
    fflush( stderr );
#endif

    return ptr;
}


/* Safe wrapper with malloc-like functionality using pinned memory
 *
 * n: num. of bytes to allocated
 * filename: source filename of caller
 * line: source line of caller
 *
 * returns: ptr to allocated memory
 * */
void * smalloc_pinned( size_t n, const char * const filename, int line )
{
    void *ptr;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] requesting allocation of %zu bytes of pinned memory at line %d in file %.*s\n",
            n, line, (int) strlen(filename), filename );
    fflush( stderr );
#endif

#if defined(HAVE_CUDA)
    ptr = sCudaHostAllocWrapper( n, filename, line );
#elif defined(HAVE_HIP)
    ptr = sHipHostMallocWrapper( n, filename, line );
#else
    //TODO: use pinned memory for host-only functionality
    ptr = smalloc( n, filename, line );
#endif

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] address: %p [SMALLOC_PINNED]\n", (void *) ptr );
    fflush( stderr );
#endif

    return ptr;
}


/* Safe wrapper around libc realloc
 *
 * n: num. of bytes to reallocated
 * filename: source filename of caller
 * line: source line of caller
 *
 * returns: ptr to reallocated memory
 * */
void * srealloc( void *ptr, size_t n, const char * const filename, int line )
{
    void *new_ptr;

    if ( n == 0 )
    {
        fprintf( stderr, "[ERROR] failed to reallocate %zu bytes for array\n",
                n );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        exit( INSUFFICIENT_MEMORY );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] requesting reallocation of %zu bytes of memory at line %d in file %.*s\n",
            n, line, (int) strlen(filename), filename );
    fflush( stderr );
#endif

    new_ptr = realloc( ptr, n );

    /* technically, ptr may still be allocated and valid,
     * but we needed more memory, so abort */
    if ( new_ptr == NULL )
    {
        fprintf( stderr, "[ERROR] failed to reallocate %zu bytes for array\n",
                n );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        exit( INSUFFICIENT_MEMORY );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] address: %p [SREALLOC]\n", (void *) new_ptr );
    fflush( stderr );
#endif

    return new_ptr;
}


/* Safe wrapper with realloc-like functionality using pinned memory
 *
 * cur_size: num. of bytes currently allocated
 * new_size: num. of bytes to reallocated
 * filename: source filename of caller
 * line: source line of caller
 *
 * returns: ptr to reallocated memory
 * */
void * srealloc_pinned( void *ptr, size_t cur_size, size_t new_size,
        const char * const filename, int line )
{
    void *new_ptr;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] requesting reallocation of %zu bytes of pinned memory at line %d in file %.*s\n",
            new_size, line, (int) strlen(filename), filename );
    fflush( stderr );
#endif

#if defined(HAVE_CUDA)
    new_ptr = sCudaHostReallocWrapper( ptr, cur_size, new_size, filename, line );
#elif defined(HAVE_HIP)
    new_ptr = sHipHostReallocWrapper( ptr, cur_size, new_size, filename, line );
#else
    //TODO: use pinned memory for host-only functionality
    new_ptr = srealloc( ptr, new_size, filename, line );
#endif

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] address: %p [SREALLOC_PINNED]\n", (void *) new_ptr );
    fflush( stderr );
#endif

    return new_ptr;
}


/* Safe wrapper around libc calloc
 *
 * n: num. of elements to allocated (each of size bytes)
 * size: num. of bytes per element
 * filename: source filename of caller
 * line: source line of caller
 *
 * returns: ptr to allocated memory, all bits initialized to zeros
 * */
void * scalloc( size_t n, size_t size, const char * const filename, int line )
{
    void *ptr;

    if ( n == 0 )
    {
        fprintf( stderr, "[ERROR] failed to allocate %zu bytes for array\n",
                n * size );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        exit( INSUFFICIENT_MEMORY );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] requesting allocation of %zu bytes of zeroed memory at line %d in file %.*s\n",
            n * size, line, (int) strlen(filename), filename );
    fflush( stderr );
#endif

    ptr = calloc( n, size );

    if ( ptr == NULL )
    {
        fprintf( stderr, "[ERROR] failed to allocate %zu bytes for array\n",
                n * size );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        exit( INSUFFICIENT_MEMORY );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] address: %p [SCALLOC]\n", (void *) ptr );
    fflush( stderr );
#endif

    return ptr;
}


/* Safe wrapper with calloc-like functionality using pinned memory
 *
 * n: num. of elements to allocated (each of size bytes)
 * size: num. of bytes per element
 * filename: source filename of caller
 * line: source line of caller
 *
 * returns: ptr to allocated memory, all bits initialized to zeros
 * */
void * scalloc_pinned( size_t n, size_t size, const char * const filename, int line )
{
    void *ptr;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] requesting allocation of %zu bytes of zeroed pinned memory at line %d in file %.*s\n",
            n * size, line, (int) strlen(filename), filename );
    fflush( stderr );
#endif

#if defined(HAVE_CUDA)
    ptr = sCudaHostCallocWrapper( n, size, filename, line );
#elif defined(HAVE_HIP)
    ptr = sHipHostCallocWrapper( n, size, filename, line );
#else
    //TODO: use pinned memory for host-only functionality
    ptr = scalloc( n, size, filename, line );
#endif

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] address: %p [SCALLOC_PINNED]\n", (void *) ptr );
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
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 * */
void smalloc_check( void **ptr, size_t *cur_size, size_t new_size, 
        int over_alloc, real over_alloc_factor, const char * const filename, int line )
{
    assert( new_size > 0 || *cur_size > 0 );

    if ( new_size > *cur_size )
    {
#if defined(DEBUG_FOCUS)
        int rank;
    
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
        fprintf( stderr, "[INFO] smalloc_check: requesting %zu bytes (%zu currently allocated) at line %d in file %.*s on MPI processor %d\n",
                new_size, *cur_size, line, (int) strlen(filename), filename, rank );
        fflush( stderr );
#endif

        if ( *cur_size != 0 )
        {
            sfree( *ptr, filename, line );
        }

        if ( over_alloc == TRUE )
        {
            *cur_size = (int) CEIL( new_size * over_alloc_factor );
        }
        else
        {
            *cur_size = new_size;
        }

        //TODO: look into using aligned alloc's
        *ptr = smalloc( *cur_size, filename, line );
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
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 * */
void srealloc_check( void **ptr, size_t *cur_size, size_t new_size,
        int over_alloc, real over_alloc_factor, const char * const filename,
        int line )
{
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] check_srealloc requesting %zu bytes for %s (%zu currently allocated)\n",
            new_size, msg, *cur_size );
    fflush( stderr );
#endif

    if ( new_size > *cur_size )
    {
        if ( over_alloc == TRUE )
        {
            *cur_size = (size_t) CEIL( new_size * over_alloc_factor );
        }
        else
        {
            *cur_size = new_size;
        }

        *ptr = srealloc( *ptr, *cur_size, filename, line );
    }
}


/* Safe wrapper around libc free
 *
 * ptr: pointer to dynamically allocated memory which will be deallocated
 * filename: source filename of caller
 * line: source line of caller
 * */
void sfree( void *ptr, const char * const filename, int line )
{
    if ( ptr == NULL )
    {
        fprintf( stderr, "[WARNING] trying to free the already NULL pointer\n" );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        return;
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] trying to free pointer at line %d in file %.*s\n",
            line, (int) strlen(filename), filename );
    fflush( stderr );
    fprintf( stderr, "[INFO] address: %p [SFREE]\n", (void *) ptr );
    fflush( stderr );
#endif

    free( ptr );
}


/* Safe wrapper with free-like functionality for pinned memory
 *
 * ptr: pointer to dynamically allocated memory which will be deallocated
 * filename: source filename of caller
 * line: source line of caller
 * */
void sfree_pinned( void *ptr, const char * const filename, int line )
{
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] trying to free pinned pointer at line %d in file %.*s\n",
            line, (int) strlen(filename), filename );
    fflush( stderr );
    fprintf( stderr, "[INFO] address: %p [SFREE_PINNED]\n", (void *) ptr );
    fflush( stderr );
#endif

#if defined(HAVE_CUDA)
    sCudaFreeHostWrapper( ptr, filename, line );
#elif defined(HAVE_HIP)
    sHipHostFreeWrapper( ptr, filename, line );
#else
    //TODO: use pinned memory for host-only functionality
    sfree( ptr, filename, line );
#endif
}


/* Safe wrapper around libc fopen
 *
 * fname: name of file to be opened
 * mode: mode in which to open file
 * filename: source filename of caller
 * line: source line of caller
 * */
FILE * sfopen( const char * fname, const char * mode,
        const char * const filename, int line )
{
    FILE * ptr;

    if ( fname == NULL )
    {
        fprintf( stderr, "[ERROR] trying to open file\n" );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        fprintf( stderr, "  [INFO] NULL file name\n" );
        exit( INVALID_INPUT );
    }
    if ( mode == NULL )
    {
        fprintf( stderr, "[ERROR] trying to open file\n" );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        fprintf( stderr, "  [INFO] NULL mode\n" );
        exit( INVALID_INPUT );
    }

    ptr = fopen( fname, mode );

    if ( ptr == NULL )
    {
        fprintf( stderr, "[ERROR] failed to open file %s with mode %s\n",
              fname, mode );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        exit( INVALID_INPUT );
    }

    return ptr;
}


/* Safe wrapper around libc fclose
 *
 * fp: pointer to file to close
 * filename: source filename of caller
 * line: source line of caller
 * */
void sfclose( FILE * fp, const char * const filename, int line )
{
    int ret;

    if ( fp == NULL )
    {
        fprintf( stderr, "[WARNING] trying to close NULL file pointer. Returning...\n" );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        return;
    }

    ret = fclose( fp );

    if ( ret != 0 )
    {
        fprintf( stderr, "[ERROR] error detected when closing file\n" );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        exit( INVALID_INPUT );
    }
}


/* Safe wrapper around strtol
 *
 * str: string to be converted
 * filename: source filename of caller
 * line: source line of caller
 *
 * returns: result of conversion (integer)
 * */
int sstrtol( const char * const str,
        const char * const filename, int line )
{
    long ret;
    char *endptr;

    if ( str[0] == '\0' )
    {
        fprintf( stderr, "[ERROR] sstrtol: NULL string\n" );
        /* strlen safe here only if filename is NULL-terminated
         * before calling sconvert_string_to_int */
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        exit( INVALID_INPUT );
    }

    errno = 0;
    ret = strtol( str, &endptr, INTBASE );

    if ( (errno == ERANGE && (ret == LONG_MAX || ret == LONG_MIN) )
            || (errno != 0 && ret == 0) )
    {
        fprintf( stderr, "[ERROR] strtol: invalid string\n" );
        /* strlen safe here only if filename is NULL-terminated
         * before calling sconvert_string_to_int */
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        fprintf( stderr, "    [INFO] str: %.*s\n",
                (int) strlen(str), str );
        exit( INVALID_INPUT );
    }
    else if ( endptr == str )
    {
        fprintf( stderr, "[ERROR] strtol: no digits found\n" );
        /* strlen safe here only if filename is NULL-terminated
         * before calling sconvert_string_to_int */
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        fprintf( stderr, "    [INFO] str: %.*s\n",
                (int) strlen(str), str );
        exit( INVALID_INPUT );
    }
    else if ( *endptr != '\0' )
    {
        fprintf( stderr, "[ERROR] strtol: non-numeric trailing characters\n" );
        /* strlen safe here only if filename is NULL-terminated
         * before calling sconvert_string_to_int */
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        fprintf( stderr, "    [INFO] str: %.*s\n",
                (int) strlen(str), str );
        exit( INVALID_INPUT );
    }

    return (int) ret;
}


/* Safe wrapper around strtod
 *
 * str: string to be converted
 * filename: source filename of caller
 * line: source line of caller
 *
 * returns: result of conversion (double)
 * */
double sstrtod( const char * const str,
        const char * const filename, int line )
{
    double ret;
    char *endptr;

    if ( str[0] == '\0' )
    {
        fprintf( stderr, "[ERROR] sstrtod: NULL string\n" );
        /* strlen safe here only if filename is NULL-terminated
         * before calling sconvert_string_to_int */
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        exit( INVALID_INPUT );
    }

    errno = 0;
    ret = strtod( str, &endptr );

    if ( (errno == ERANGE && (ret == DBL_MAX || ret == DBL_MIN) )
            || (errno != 0 && ret == 0.0) )
    {
        fprintf( stderr, "[ERROR] strtod: invalid string\n" );
        /* strlen safe here only if filename is NULL-terminated
         * before calling sconvert_string_to_int */
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        fprintf( stderr, "    [INFO] str: %.*s\n",
                (int) strlen(str), str );
        exit( INVALID_INPUT );
    }
    else if ( endptr == str )
    {
        fprintf( stderr, "[ERROR] strtod: no digits found\n" );
        /* strlen safe here only if filename is NULL-terminated
         * before calling sconvert_string_to_int */
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        fprintf( stderr, "    [INFO] str: %.*s\n",
                (int) strlen(str), str );
        exit( INVALID_INPUT );
    }
    else if ( *endptr != '\0' )
    {
        fprintf( stderr, "[ERROR] strtod: non-numeric trailing characters\n" );
        /* strlen safe here only if filename is NULL-terminated
         * before calling sconvert_string_to_int */
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        fprintf( stderr, "    [INFO] str: %.*s\n",
                (int) strlen(str), str );
        exit( INVALID_INPUT );
    }

    return ret;
}
