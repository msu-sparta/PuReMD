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

#ifndef REAX_DEFS_H
#define REAX_DEFS_H

#if defined(__IBMC__)
#define inline __inline__
#endif /*IBMC*/

#define SUCCESS  1
#define FAILURE  0
#define TRUE  1
#define FALSE 0

#define SQR(x)        ((x)*(x))
#define CUBE(x)       ((x)*(x)*(x))
#define DEG2RAD(a)    ((a)*constPI/180.0)
#define RAD2DEG(a)    ((a)*180.0/constPI)
#define MAX(x,y)      (((x) > (y)) ? (x) : (y))
#define MIN(x,y)      (((x) < (y)) ? (x) : (y))
#define MAX3(x,y,z)   MAX( MAX(x,y), z)

#define constPI        3.14159265
#define C_ele          332.06371
//#define K_B         503.398008   // kcal/mol/K
#define K_B             0.831687   // amu A^2 / ps^2 / K
#define F_CONV          1e6 / 48.88821291 / 48.88821291   // --> amu A / ps^2
#define E_CONV          0.002391   // amu A^2 / ps^2 --> kcal/mol
#define EV_to_KCALpMOL 14.400000   // ElectronVolt --> KCAL per MOLe
#define KCALpMOL_to_EV 23.02       // 23.060549 //KCAL per MOLe --> ElectronVolt
#define ECxA_to_DEBYE   4.803204   // elem. charge * Ang -> debye
#define CAL_to_JOULES   4.184000   // CALories --> JOULES
#define JOULES_to_CAL   1/4.184000 // JOULES --> CALories
#define AMU_to_GRAM     1.6605e-24
#define ANG_to_CM       1e-8
#define AVOGNR          6.0221367e23
#define P_CONV          1e-24 * AVOGNR * JOULES_to_CAL

#define MAX_STR             1024
#define MAX_LINE            1024
#define MAX_TOKENS          1024
#define MAX_TOKEN_LEN       1024

#define MAX_ATOM_ID         100000
#define MAX_RESTRICT        15
#define MAX_MOLECULE_SIZE   20
#define MAX_ATOM_TYPES      25

#define NUM_INTRS      10
#define ALMOST_ZERO    1e-10
#define NEG_INF       -1e10
#define NO_BOND        1e-3  // 0.001
#define HB_THRESHOLD   1e-2  // 0.01

#define MIN_CAP        50
#define MIN_NBRS       100
#define MIN_HENTRIES   100
#define MAX_BONDS      30
#define MIN_BONDS      15
#define MIN_HBONDS     25
#define MIN_3BODIES    1000
#define MIN_GCELL_POPL 50
#define MIN_SEND       100
#define SAFE_ZONE      1.2
#define SAFER_ZONE     1.4
#define DANGER_ZONE    0.90
#define LOOSE_ZONE     0.75
#define MAX_3BODY_PARAM     5
#define MAX_4BODY_PARAM     5

#define MAX_dV              1.01
#define MIN_dV              0.99
#define MAX_dT              4.00
#define MIN_dT              0.00

#define MASTER_NODE 0
#define MAX_NBRS 6 //27
#define MAX_NT_NBRS 6
#define MYSELF   13  // encoding of relative coordinate (0,0,0)

#define MAX_ITR 10
#define RESTART 30



/******************* ENUMERATIONS *************************/
enum geo_formats { CUSTOM = 0, PDB = 1, ASCII_RESTART = 2, BINARY_RESTART = 3, GF_N = 4 };

enum restart_formats { WRITE_ASCII = 0, WRITE_BINARY = 1, RF_N = 2 };

enum ensembles { NVE = 0, bNVT = 1, nhNVT = 2, sNPT = 3, iNPT = 4, NPT = 5, ens_N = 6 };

enum lists { BONDS = 0, OLD_BONDS = 1, THREE_BODIES = 2,
             HBONDS = 3, FAR_NBRS = 4, DBOS = 5, DDELTAS = 6, LIST_N = 7
           };

enum interactions { TYP_VOID = 0, TYP_BOND = 1, TYP_THREE_BODY = 2,
                    TYP_HBOND = 3, TYP_FAR_NEIGHBOR = 4, TYP_DBO = 5, TYP_DDELTA = 6, TYP_N = 7
                  };

enum message_tags { INIT = 0, UPDATE = 1, BNDRY = 2, UPDATE_BNDRY = 3,
                    EXC_VEC1 = 4, EXC_VEC2 = 5, DIST_RVEC2 = 6, COLL_RVEC2 = 7,
                    DIST_RVECS = 8, COLL_RVECS = 9, INIT_DESCS = 10, ATOM_LINES = 11,
                    BOND_LINES = 12, ANGLE_LINES = 13, RESTART_ATOMS = 14, TAGS_N = 15
                  };

enum errors { FILE_NOT_FOUND = -10, UNKNOWN_ATOM_TYPE = -11,
              CANNOT_OPEN_FILE = -12, CANNOT_INITIALIZE = -13,
              INSUFFICIENT_MEMORY = -14, UNKNOWN_OPTION = -15,
              INVALID_INPUT = -16, INVALID_GEO = -17
            };

enum exchanges { NONE = 0, NEAR_EXCH = 1, FULL_EXCH = 2 };

enum gcell_types { NO_NBRS = 0, NEAR_ONLY = 1, HBOND_ONLY = 2, FAR_ONLY = 4,
                   NEAR_HBOND = 3, NEAR_FAR = 5, HBOND_FAR = 6, FULL_NBRS = 7,
                   NATIVE = 8
                 };

enum atoms { C_ATOM = 0, H_ATOM = 1, O_ATOM = 2, N_ATOM = 3,
             S_ATOM = 4, SI_ATOM = 5, GE_ATOM = 6, X_ATOM = 7
           };

enum traj_methods { REG_TRAJ = 0, MPI_TRAJ = 1, TF_N = 2 };

enum molecules { UNKNOWN = 0, WATER = 1 };


#endif
