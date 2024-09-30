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

#ifndef __TRAJ_H_
#define __TRAJ_H_

#include "reax_types.h"


/*
  Format for trajectory file

 {HEADER}
  size flag char (1)
  size of header to skip (int32_t)
  Title (char[80])
  size flag char (2)
  size of control param block (int32_t)
  Entire control param structure
  size of frame descriptor (int32_t)
  Frame descriptor Block
      [ Frame descriptor block
         No. of global quantities lines (say m) (int32_t)
     Format for each global quantity line [m].
     Comma separated names for each global quantity line [m].
      ]

  {FRAMES}
  size flag char (1)
  size of the entire frame to skip it (int32_t)
  size flag char (2)
  size of global quantities block to skip it (int32_t)
  Global quantities lines [m]
  Atom format line
  Bond format line
  Angle format line
  Torsion format line
  size flag char (2)
  size of atom block (int32_t)
  No. of atom lines (int32_t)
  Atom lines as per atom format
  size flag char (2)
  size to skip to the end of frame (int32_t)
  size flag char (3)
  size to skip bond block (int32_t)
  No. of bond entries (int32_t)
  Bond info lines as per bond format.
  size flag char (3)
  size to skip angle block (int32_t)
  No. of angle entries (int32_t)
  Angle info lines as per angle format.
  size flag char (3)
  size to skip torsion block (int32_t)
  No. of torsion entries (int32_t)
  Torsion info lines as per torsion format.
*/
int32_t Write_Custom_Header( reax_system*, control_params*,
        static_storage*, output_controls* );

int32_t Write_xyz_Header( reax_system*, control_params*,
        static_storage*, output_controls* );

/*
  Write_Traj_Header( gzfile file,
             int32_t No. of lines of global qunatities,
             char** format for global quantities,
             char** names for global quantities,
                 control_params* control);
 */
char Write_Traj_Header( FILE*, int32_t, char**, char**, control_params* );

int32_t Append_Custom_Frame( reax_system*, control_params*, simulation_data*,
        static_storage*, reax_list**, output_controls* );

int32_t Append_xyz_Frame( reax_system*, control_params*, simulation_data*,
        static_storage*, reax_list**, output_controls* );

#if defined(HAVE_ZLIB)
void Read_Traj_Compressed( output_controls*, char * );
#endif


#endif
