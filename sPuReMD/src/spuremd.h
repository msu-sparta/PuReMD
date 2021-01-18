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

#ifndef __SPUREMD_H_
#define __SPUREMD_H_

#include "reax_types.h"


#define SPUREMD_SUCCESS (0)
#define SPUREMD_FAILURE (-1)


#if defined(__cplusplus)
extern "C"  {
#endif

void * setup( const char * const, const char * const,
        const char * const );

int setup_callback( const void * const, const callback_function );

int simulate( const void * const );

int cleanup( const void * const );

int reset( const void * const, const char * const,
        const char * const, const char * const );

int get_atom_positions( const void * const, double * const,
        double * const, double * const );

int get_atom_velocities( const void * const, double * const,
        double * const, double * const );

int get_atom_forces( const void * const, double * const,
        double * const, double * const );

int get_atom_charges( const void * const, double * const );

int get_system_info( const void * const, double * const,
        double * const, double * const, double * const,
        double * const, double * const );

int set_output_enabled( const void * const, const int );

int set_control_parameter( const void * const, const char * const,
       const char ** const );

#if defined(QMMM)
void * setup_qmmm( int, const int * const,
        const double * const, const double * const,
        const double * const, int, const int * const,
        const double * const, const double * const,
        const double * const, const double * const,
        const double * const,
        const char * const, const char * const );

int reset_qmmm( const void * const, int,
        const int * const,
        const double * const, const double * const,
        const double * const, int, const int * const,
        const double * const, const double * const,
        const double * const, const double * const,
        const double * const,
        const char * const, const char * const );

int get_atom_positions_qmmm( const void * const, double * const,
        double * const, double * const, double * const,
        double * const, double * const );

int get_atom_velocities_qmmm( const void * const, double * const,
        double * const, double * const, double * const,
        double * const, double * const );

int get_atom_forces_qmmm( const void * const, double * const,
        double * const, double * const, double * const,
        double * const, double * const );

int get_atom_charges_qmmm( const void * const, double * const, double * const );
#endif

#if defined(QMMM_FORTRAN)
void setup_qmmm_( void *, const int * const, const int * const,
        const double * const, const double * const,
        const double * const, const int * const, const int * const,
        const double * const, const double * const,
        const double * const, const double * const,
        const double * const,
        const char * const, const char * const );

void reset_qmmm_( const void * const, const int * const, const int * const,
        const double * const, const double * const,
        const double * const, const int * const, const int * const,
        const double * const, const double * const,
        const double * const, const double * const,
        const double * const,
        const char * const, const char * const );

void simulate_( const void * const );

void cleanup_( const void * const );

void set_output_enabled_( const void * const, const int );

void set_control_parameter_( const void * const, const char * const,
       const char ** const );

void get_atom_positions_qmmm_( const void * const, double * const,
        double * const, double * const, double * const,
        double * const, double * const );

void get_atom_velocities_qmmm_( const void * const, double * const,
        double * const, double * const, double * const,
        double * const, double * const );

void get_atom_forces_qmmm_( const void * const, double * const,
        double * const, double * const, double * const,
        double * const, double * const );

void get_atom_charges_qmmm_( const void * const, double * const, double * const );

void get_system_info_( const void * const, double * const,
        double * const, double * const, double * const,
        double * const, double * const );
#endif

#if defined(__cplusplus)
}
#endif


#endif
