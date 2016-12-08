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

#ifndef __GEO_TOOLS_H_
#define __GEO_TOOLS_H_

#include "mytypes.h"

// CUSTOM_BOXGEO: BOXGEO box_x box_y box_z  angle1 angle2 angle3
#define CUSTOM_BOXGEO_FORMAT " %s %lf %lf %lf %lf %lf %lf"
// CUSTOM ATOM: serial element name x y z
#define CUSTOM_ATOM_FORMAT " %d %s %s %lf %lf %lf"

char Read_Geo( char*, reax_system*, control_params*,
        simulation_data*, static_storage* );

/* PDB format :
http://www.rcsb.org/pdb/file_formats/pdb/pdbguide2.2/guide2.2_frame.html

#define PDB_ATOM_FORMAT   "%6s%5d%4s%c%4s%c%4d%c%8s%8s%8s%6s%6s%4s%2s%2s\n"

COLUMNS        DATA TYPE       FIELD         DEFINITION
--------------------------------------------------------------------------------
1 -  6        Record name     "ATOM  "
7 - 11        Integer         serial        Atom serial number.
13 - 16       Atom            name          Atom name.
17            Character       altLoc        Alternate location indicator.
18 - 20       Residue name    resName       Residue name.
22            Character       chainID       Chain identifier.
23 - 26       Integer         resSeq        Residue sequence number.
27            AChar           iCode         Code for insertion of residues.
31 - 38       Real(8.3)       x             Orthogonal coord for X in Angstroms
39 - 46       Real(8.3)       y             Orthogonal coord for Y in Angstroms
47 - 54       Real(8.3)       z             Orthogonal coord for Z in Angstroms
55 - 60       Real(6.2)       occupancy     Occupancy.
61 - 66       Real(6.2)       tempFactor    Temperature factor.
73 - 76       LString(4)      segID         Segment identifier, left-justified.
77 - 78       LString(2)      element       Element symbol, right-justified.
79 - 80       LString(2)      charge        Charge on the atom.
*/

/*
COLUMNS     DATA TYPE        FIELD         DEFINITION
--------------------------------------------------------------
 1 - 6      Record name      "HETATM"
 7 - 11     Integer          serial        Atom serial number.
13 - 16     Atom             name          Atom name.
17          Character        altLoc        Alternate location indicator.
18 - 20     Residue name     resName       Residue name.
22          Character        chainID       Chain identifier.
23 - 26     Integer          resSeq        Residue sequence number.
27          AChar            iCode         Code for insertion of residues.
31 - 38     Real(8.3)        x             Orthogonal coordinates for X.
39 - 46     Real(8.3)        y             Orthogonal coordinates for Y.
47 - 54     Real(8.3)        z             Orthogonal coordinates for Z.
55 - 60     Real(6.2)        occupancy     Occupancy.
61 - 66     Real(6.2)        tempFactor    Temperature factor.
77 - 78     LString(2)       element       Element symbol; right-justified.
79 - 80     LString(2)       charge        Charge on the atom.
*/

/*
COLUMNS       DATA TYPE       FIELD         DEFINITION
-------------------------------------------------------
1 -  6       Record name     "CONECT"
7 - 11       Integer         serial        Atom serial number
12 - 16      Integer         serial        Serial number of bonded atom
17 - 21      Integer         serial        Serial number of bonded atom
22 - 26      Integer         serial        Serial number of bonded atom
27 - 31      Integer         serial        Serial number of bonded atom
*/

/*
COLUMNS       DATA TYPE       FIELD         DEFINITION
----------------------------------------------------------
1 - 6        Record name     "CRYST1"
7 - 15       Real(9.3)       a             a (Angstroms)
16 - 24      Real(9.3)       b             b (Angstroms)
25 - 33      Real(9.3)       c             c (Angstroms)
34 - 40      Real(7.2)       alpha         alpha (degrees)
41 - 47      Real(7.2)       beta          beta (degrees)
48 - 54      Real(7.2)       gamma         gamma (degrees)
56 - 66      LString         sGroup        Space group
67 - 70      Integer         z             Z value
*/

//#define PDB_ATOM_FORMAT
//"ATOM  %4d%4s%c%3s%c%4d%c%8.3f%8.3f%8.3f%6.2f%6.2f%-4s%2s%2s\n"

#define PDB_ATOM_FORMAT   "%6s%5d%4s%c%4s%c%4d%c%8s%8s%8s%6s%6s%4s%2s%2s\n"
#define PDB_ATOM_FORMAT_LENGTH 71
#define PDB_HETATM_FORMAT "%6s%5d%4s%c%4s%c%4d%c%8s%8s%8s%6s%6s%2s%2s\n"
#define PDB_CONECT_FORMAT "%6s%5d%5d%5d%5d%5d\n"
#define PDB_CRYST1_FORMAT "%6s%9s%9s%9s%7s%7s%7s%11s%4s\n"

#define PDB_ATOM_FORMAT_O "%6s%5d %4s%c%3s %c%4d%c   %8.3f%8.3f%8.3f%6.2f%6.2f      %-4s%2s%2s\n"
#define PDB_ATOM_FORMAT_O_LENGTH 81
#define PDB_CRYST1_FORMAT_O "%6s%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f%11s%4d\n"

#define BGF_CRYSTX_FORMAT "%8s%11s%11s%11s%11s%11s%11s"

char Read_PDB( char*, reax_system*, control_params*,
        simulation_data*, static_storage* );

char Read_BGF( char*, reax_system*, control_params*,
        simulation_data*, static_storage* );

char Write_PDB( reax_system*, list*, simulation_data*,
        control_params*, static_storage*, output_controls* );

#endif
