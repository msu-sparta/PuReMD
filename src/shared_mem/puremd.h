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

#ifndef __PUREMD_H_
#define __PUREMD_H_

#include "reax_types.h"


#define PUREMD_SUCCESS (0)
#define PUREMD_FAILURE (-1)


#if defined(__cplusplus)
extern "C"  {
#endif

void * setup( const char * const, const char * const,
        const char * const );

void * setup2( int32_t, const int32_t * const,
        const double * const, const double * const,
        const char * const, const char * const );

int32_t setup_callback( const void * const, const callback_function );

int32_t simulate( const void * const );

int32_t cleanup( const void * const );

int32_t reset( const void * const, const char * const,
        const char * const, const char * const );

int32_t reset2( const void * const, int32_t, const int32_t * const,
        const double * const, const double * const,
        const char * const, const char * const );

int32_t get_atom_positions( const void * const, double * const );

int32_t get_atom_velocities( const void * const, double * const );

int32_t get_atom_forces( const void * const, double * const );

int32_t get_atom_charges( const void * const, double * const );

int32_t get_system_info( const void * const, double * const,
        double * const, double * const, double * const,
        double * const, double * const );

int32_t get_total_energy( const void * const, double * const );

int32_t set_output_enabled( const void * const, const int32_t );

int32_t set_control_parameter( const void * const, const char * const,
       const char ** const );

int32_t set_contiguous_charge_constraints( const void * const, int32_t,
        const int32_t * const, const int32_t * const, const double * const );

int32_t set_custom_charge_constraints( const void * const,
        int32_t, const int32_t * const, const int32_t * const,
        const double * const, const double * const );

#if defined(QMMM)
void * setup_qmmm( int32_t, const char * const,
        const double * const, int32_t, const char * const,
        const double * const, const double * const,
        const char * const, const char * const );

int32_t reset_qmmm( const void * const, int32_t, const char * const,
        const double * const, int32_t, const char * const,
        const double * const, const double * const,
        const char * const, const char * const);

int32_t get_atom_positions_qmmm( const void * const, double * const,
        double * const );

int32_t get_atom_velocities_qmmm( const void * const, double * const,
        double * const );

int32_t get_atom_forces_qmmm( const void * const, double * const,
        double * const );

int32_t get_atom_charges_qmmm( const void * const, double * const, double * const );
#endif

#if defined(__cplusplus)
}
#endif


#endif
