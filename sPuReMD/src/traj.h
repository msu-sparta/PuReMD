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

#ifndef __TRAJ_H__
#define __TRAJ_H__

#include "mytypes.h"
#include "zlib.h"


#define BLOCK_MARK "REAX_BLOCK_MARK "
#define BLOCK_MARK_LEN 16

#define HEADER_INIT "%-10d %-10d\n%-80s\n"
#define HEADER_INIT_LEN 81

#define CONTROL_BLOCK "num_atoms:\t\t%d\nrestart:\t\t%d\nrestart_from:\t\t%s\nrandom_vel:\t\t%d\nrestart_freq:\t\t%d\nensemble_type:\t\t%d\nnsteps:\t\t\t%d\ndt:\t\t\t%.5f\nreposition_atoms:\t%d\nrestrict_bonds:\t\t%d\ntabulate_long_range:\t%d\nnbrhood_cutoff:\t\t%.3f\nr_cut:\t\t\t%.3f\nbond_graph_cutoff:\t%.3f\nbond_order_cutoff:\t%.3f\nthb_cutoff:\t\t%.3f\nhbond_cutoff:\t\t%.3f\nq_err:\t\t\t%.10f\ntemp_init:\t\t%.3f\ntemp_final:\t\t%.3f\nt_mass:\t\t\t%.3f\nt_mode:\t\t\t%d\nt_rate:\t\t\t%.3f\nt_freq:\t\t\t%.3f\npressure:\t\t%.5f %.5f %.5f\np_mass:\t\t\t%.3f %.3f %.3f\ncompress:\t\t%.5f\npress_mode:\t\t%d\nremove_CoM_vel:\t\t%d\nwrite_freq:\t\t%d\ntraj_compress:\t\t%d\ntraj_format:\t\t%d\natom_line:\t\t%d\nbond_line:\t\t%d\nangle_line:\t\t%d\nenergy_update_freq:\t%d\nmolec_anal:\t\t%d\nfreq_molec_anal:\t%d\n"

#define NUM_FRAME_GLOBALS 27 // 26 floats, 1 integer
#define FRAME_GLOBALS_FORMAT "%10d  %8.3f  %15.3f  %15.3f  %15.3f  %15.3f\\n%15.3f  %15.3f  %15.3f  %15.3f  %15.3f  %8.2f  %8.2f  %8.2f\\n%15.3f  %15.3f  %15.3f  %15.3f  %15.3f  %15.3f  %15.3f  %15.3f  %15.3f  %15.3f  %15.3f  %15.3f  %15.3f\\n"
#define FRAME_GLOBALS "%10d  %8.3f  %15.3f  %15.3f  %15.3f  %15.3f\n%15.3f  %15.3f  %15.3f  %15.3f  %15.3f  %8.2f  %8.2f  %8.2f\n%15.3f  %15.3f  %15.3f  %15.3f  %15.3f  %15.3f  %15.3f  %15.3f  %15.3f  %15.3f  %15.3f  %15.3f  %15.3f\n"
#define FRAME_GLOBAL_NAMES "timestep, time, e_total, e_pot, e_kin, temperature, pressure, volume, x_norm, y_norm, z_norm, x_angle, y_angle, z_angle, e_be, e_ov, e_un, e_lp, e_ang, e_pen, e_coa, e_hb, e_tor, e_con, e_vdw, e_ele, e_pol"
#define FRAME_GLOBALS_LEN 11 + 20*9 + 70 // 1x10d int + 20x8.3f + CRYST line (6s + 3x9.3f + 3x7.2f + 11s + 4d + 1)

//AtomID AtomType (X Y Z) Charge
#define ATOM_BASIC "%9d %10.3f %10.3f %10.3f %10.3f\n"
#define ATOM_BASIC_LEN 54
//AtomID (X Y Z) (Vx Vy Vz) Charge
#define ATOM_wV "%9d %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n"
#define ATOM_wV_LEN 87
//AtomID (X Y Z) (Fx Fy Fz) Charge
#define ATOM_wF "%9d %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n"
#define ATOM_wF_LEN 87
//AtomID (X Y Z) (Vx Vy Vz) (Fx Fy Fz) Charge
#define ATOM_FULL "%9d %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n"
#define ATOM_FULL_LEN 120

// Atom1 Atom2 Dist Total_BO
#define BOND_BASIC "%9d %9d %10.3f %10.3f\n"
#define BOND_BASIC_LEN 42
// Atom1 Atom2 Dist Total_BO BOs BOpi BOpi2
#define BOND_FULL "%9d %9d %10.3f %10.3f %10.3f %10.3f %10.3f\n"
#define BOND_FULL_LEN 75

// Atom1 Atom2 Atom3 Theta
#define ANGLE_BASIC "%9d %9d %9d %10.3f\n"
#define ANGLE_BASIC_LEN 41

//AtomID - AtomType, AtomName, AtomMass mapping
#define ATOM_MAPPING "%9d %2d %4s %8.3f\n"
#define ATOM_MAPPING_LEN 33

#define SIZE_INFO_LINE2 "%-10d %-10d\n"
#define SIZE_INFO_LEN2 22

#define SIZE_INFO_LINE3 "%-10d %-10d %-10d\n"
#define SIZE_INFO_LEN3 33

enum ATOM_LINE_OPTS {OPT_NOATOM = 0, OPT_ATOM_BASIC = 4, OPT_ATOM_wF = 5,
                     OPT_ATOM_wV = 6, OPT_ATOM_FULL = 7
                    };
enum BOND_LINE_OPTS {OPT_NOBOND, OPT_BOND_BASIC, OPT_BOND_FULL};
enum ANGLE_LINE_OPTS {OPT_NOANGLE, OPT_ANGLE_BASIC};

typedef struct
{
    int no_of_sub_blocks;
    int size;
    char* buffer;
    struct block** sub_blocks;
} block;


int Write_Block( gzFile, block* );
int Read_Next_Block( gzFile, block*, int* );
int Skip_Next_Block( gzFile, int*);


/*
  Format for trajectory file

 {HEADER}
  size flag char (1)
  size of header to skip (int)
  Title (char[80])
  size flag char (2)
  size of control param block (int)
  Entire control param structure
  size of frame descriptor (int)
  Frame descriptor Block
      [ Frame descriptor block
         No. of global quantities lines (say m) (int)
     Format for each global quantity line [m].
     Comma separated names for each global quantity line [m].
      ]

  {FRAMES}
  size flag char (1)
  size of the entire frame to skip it (int)
  size flag char (2)
  size of global quantities block to skip it (int)
  Global quantities lines [m]
  Atom format line
  Bond format line
  Angle format line
  Torsion format line
  size flag char (2)
  size of atom block (int)
  No. of atom lines (int)
  Atom lines as per atom format
  size flag char (2)
  size to skip to the end of frame (int)
  size flag char (3)
  size to skip bond block (int)
  No. of bond entries (int)
  Bond info lines as per bond format.
  size flag char (3)
  size to skip angle block (int)
  No. of angle entries (int)
  Angle info lines as per angle format.
  size flag char (3)
  size to skip torsion block (int)
  No. of torsion entries (int)
  Torsion info lines as per torsion format.
*/


int Write_Custom_Header( reax_system*, control_params*,
                         static_storage*, output_controls* );
int Write_xyz_Header   ( reax_system*, control_params*,
                         static_storage*, output_controls* );

/*
  Write_Traj_Header( gzfile file,
             int No. of lines of global qunatities,
             char** format for global quantities,
             char** names for global quantities,
                 control_params* control);
 */
char Write_Traj_Header( FILE*, int, char**, char**, control_params* );


/*
  Push_Traj_Frame(gzfile file,
                  reax_system* system,
          control_params* control,
          simulation_data* data,
          static_storage* workspace,
          reax_list** lists,
          char** various flags);
*/
int Push_Traj_Frame( /*gzfile*/ FILE*, reax_system*, control_params*,
                                simulation_data*, static_storage*, reax_list**, char** );

/*
  Append_Traj_Frame( gzfile file,
                        reax_system* system,
                        control_params* control,
                simulation_data* data,
                static_storage* workspace,
                reax_list** lists,
                char** various flags);
*/
int Append_Custom_Frame( reax_system*, control_params*, simulation_data*,
                         static_storage*, reax_list**, output_controls* );
int Append_xyz_Frame   ( reax_system*, control_params*, simulation_data*,
                         static_storage*, reax_list**, output_controls* );


void Read_Traj( output_controls*, char * );

#endif
