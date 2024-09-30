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

#ifndef __TOOL_BOX_H_
#define __TOOL_BOX_H_

#include "reax_types.h"


/* from box.h */
void Transform( rvec, simulation_box *, int32_t, rvec );

void Transform_to_UnitBox( rvec, simulation_box *, int32_t, rvec );

void Fit_to_Periodic_Box( simulation_box *, rvec );

int32_t is_Inside_Box( simulation_box *, rvec );

/* from geo_tools.h */
void Make_Point( real, real, real, rvec * );

int32_t is_Valid_Serial( int32_t );

int32_t Check_Input_Range( int32_t, int32_t, int32_t, const char * const, int32_t );

void Trim_Spaces( char * const, const size_t );

/* from system_props.h */
real Get_Time( );

real Get_Timing_Info( real );

/* from io_tools.h */
int32_t Get_Atom_Type( reax_interaction *, char *, size_t );

char * Get_Element( reax_system *, int32_t );

char * Get_Atom_Name( reax_system *, int32_t );

void Allocate_Tokenizer_Space( char **, size_t, char **, size_t, char ***,
        size_t, size_t );

void Deallocate_Tokenizer_Space( char **, char **, char ***,
        size_t );

int32_t Tokenize( char *, char ***, size_t );

void * smalloc( size_t, const char * const, int32_t );

void * srealloc( void *, size_t, const char * const, int32_t );

void * scalloc( size_t, size_t, const char * const, int32_t );

void sfree( void *, const char * const, int32_t );

FILE * sfopen( const char *, const char *, const char * const, int32_t );

void sfclose( FILE *, const char * const, int32_t );

int32_t sstrtol( const char * const, const char * const, int32_t );

double sstrtod( const char * const, const char * const, int32_t );

#endif
