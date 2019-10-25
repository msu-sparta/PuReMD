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

#ifndef __REAX_TYPES_H_
#define __REAX_TYPES_H_

#if (defined(HAVE_CONFIG_H) && !defined(__CONFIG_H_))
  #define __CONFIG_H_
  #include "../../common/include/config.h"
#endif

#include <ctype.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <zlib.h>

/************* SOME DEFS - crucial for reax_types.h *********/

#define PURE_REAX
//#define DUAL_SOLVER
//#define NEUTRAL_TERRITORY
//#define LAMMPS_REAX
//#define DEBUG
//#define DEBUG_FOCUS
//#define TEST_ENERGY
//#define TEST_FORCES
//#define CG_PERFORMANCE
#define LOG_PERFORMANCE
#define STANDARD_BOUNDARIES
//#define OLD_BOUNDARIES
//#define MIDPOINT_BOUNDARIES

#define SUCCESS (1)
#define FAILURE (0)
#define TRUE (1)
#define FALSE (0)

/* transcendental constant pi */
#if defined(M_PI)
  /* GNU C library (libc), defined in math.h */
  #define PI (M_PI)
#else
  #define PI (3.14159265) // Fortran ReaxFF code
#endif
/* Coulomb energy conversion */
#define C_ELE (332.0638) // Fortran ReaxFF code
//#define C_ELE (332.06371)
/* kcal/mol/K */
//#define K_B (503.398008)
/* amu A^2 / ps^2 / K */
//#define K_B (0.831687)
#define K_B (0.8314510) // Fortran ReaxFF code
/* --> amu A / ps^2 */
#define F_CONV (1e6 / 48.88821291 / 48.88821291)
/* amu A^2 / ps^2 --> kcal/mol */
#define E_CONV (0.002391)
/* conversion constant from electron volts to kilo-calories per mole */
#define EV_to_KCALpMOL (14.40)
/* conversion constant from kilo-calories per mole to electron volts */
//#define KCALpMOL_to_EV (23.060549) // value used in LAMMPS
//#define KCALpMOL_to_EV (23.0408) // value used in ReaxFF Fortran code
#define KCALpMOL_to_EV (23.02) // value used in ReaxFF Fortran code (ACKS2)
/* elem. charge * angstrom -> debye conv */
//#define ECxA_to_DEBYE (4.803204)
#define ECxA_to_DEBYE (4.80320679913) // ReaxFF Fortran code
/* CALories --> JOULES */
#define CAL_to_JOULES (4.1840)
/* JOULES --> CALories */
#define JOULES_to_CAL (1.0 / 4.1840)
/* */
#define AMU_to_GRAM (1.6605e-24)
/* */
#define ANG_to_CM (1.0e-8)
/* */
#define AVOGNR (6.0221367e23)
/* */
#define P_CONV (1.0e-24 * AVOGNR * JOULES_to_CAL)

#define MAX_STR (1024)
#define MAX_LINE (1024)
#define MAX_TOKENS (1024)
#define MAX_TOKEN_LEN (1024)

#define MAX_ATOM_ID (100000)
#define MAX_RESTRICT (15)
#define MAX_MOLECULE_SIZE (20)
#define MAX_ATOM_TYPES (25)

#define MAX_3BODY_PARAM (5)
#define MAX_4BODY_PARAM (5)
#define NUM_INTRS (10)

#define MAX_dV (1.01)
#define MIN_dV (0.99)
#define MAX_dT (4.00)
#define MIN_dT (0.00)

#define ZERO (0.000000000000000e+00)
#define ALMOST_ZERO (1e-10)
#define NEG_INF (-1e10)
#define NO_BOND (1e-3)
#define HB_THRESHOLD (1e-2)
#define MIN_CAP (50)
#define MIN_NBRS (100)
#define MIN_HENTRIES (100)
#define MAX_BONDS (30)
#define MIN_BONDS (15)
#define MIN_HBONDS (25)
#define MIN_3BODIES (1000)
#define MIN_GCELL_POPL (50)
#define MIN_SEND (100)
#define SAFE_ZONE (1.2)
#define SAFER_ZONE (1.4)
#define SAFE_ZONE_NT (2.0)
#define SAFER_ZONE_NT (2.5)
#define DANGER_ZONE (0.90)
#define LOOSE_ZONE (0.75)

/* NaN IEEE 754 representation for C99 in math.h
 * Note: function choice must match REAL typedef below */
#ifdef NAN
  #define IS_NAN_REAL(a) (isnan(a))
#else
  #warn "No support for NaN"
  #define IS_NAN_REAL(a) (0)
#endif
#define LOG (log)
#define EXP (exp)
#define SQRT (sqrt)
#define POW (pow)
#define ACOS (acos)
#define COS (cos)
#define SIN (sin)
#define TAN (tan)
#define CEIL (ceil)
#define FLOOR (floor)
#define FABS (fabs)
#define FMOD (fmod)
#define SQR(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))
#define DEG2RAD(a) ((a)*PI/180.0)
#define RAD2DEG(a) ((a)*180.0/PI)
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#define MAX3(x,y,z) (MAX( MAX(x,y), z))

#define MASTER_NODE (0)
#define MAX_NBRS (6) //27
#define MAX_NT_NBRS (6)
#define MYSELF (13)  // encoding of relative coordinate (0,0,0)


/* ensemble type */
enum ensemble
{
    NVE = 0,
    bNVT = 1,
    nhNVT = 2,
    sNPT = 3,
    iNPT = 4,
    NPT = 5,
    ens_N = 6,
};

/* interaction list type */
enum interaction_list_offets
{
    BONDS = 0,
    OLD_BONDS = 1,
    THREE_BODIES = 2,
    HBONDS = 3,
    FAR_NBRS = 4,
    DBOS = 5,
    DDELTAS = 6,
    LIST_N = 7,
};

/* interaction type */
enum interaction_type
{
    TYP_VOID = 0,
    TYP_BOND = 1,
    TYP_THREE_BODY = 2,
    TYP_HBOND = 3,
    TYP_FAR_NEIGHBOR = 4,
    TYP_DBO = 5,
    TYP_DDELTA = 6,
    TYP_N = 7,
};

/* error codes for simulation termination */
enum errors
{
    FILE_NOT_FOUND = -10,
    UNKNOWN_ATOM_TYPE = -11,
    CANNOT_OPEN_FILE = -12,
    CANNOT_INITIALIZE = -13,
    INSUFFICIENT_MEMORY = -14,
    UNKNOWN_OPTION = -15,
    INVALID_INPUT = -16,
    INVALID_GEO = -17,
    NUMERIC_BREAKDOWN = -18,
    MAX_RETRIES_REACHED = -19,
    RUNTIME_ERROR = -20,
};

/* restart file format */
enum restart_formats
{
    WRITE_ASCII = 0,
    WRITE_BINARY = 1,
    RF_N = 2,
};

/* geometry file format */
enum geo_formats
{
    CUSTOM = 0,
    PDB = 1,
    ASCII_RESTART = 2,
    BINARY_RESTART = 3,
    GF_N = 4,
};

/* method used for computing atomic charges */
enum charge_method
{
    QEQ_CM = 0,
    EE_CM = 1,
    ACKS2_CM = 2,
};

/* iterative linear solver used for computing atomic charges */
enum solver
{
    GMRES_S = 0,
    GMRES_H_S = 1,
    CG_S = 2,
    SDM_S = 3,
    BiCGStab_S = 4,
    PIPECG_S = 5,
    PIPECR_S = 6,
};

/* preconditioner used with iterative linear solver */
enum pre_comp
{
    NONE_PC = 0,
    JACOBI_PC = 1,
    ICHOLT_PC = 2,
    ILUT_PC = 3,
    ILUTP_PC = 4,
    FG_ILUT_PC = 5,
    SAI_PC = 6,
};

/* method used to apply preconditioner for 2-side incomplete factorizations (ICHOLT, ILU) */
enum pre_app
{
    TRI_SOLVE_PA = 0,
    TRI_SOLVE_LEVEL_SCHED_PA = 1,
    TRI_SOLVE_GC_PA = 2,
    JACOBI_ITER_PA = 3,
};

/* atom types as pertains to hydrogen bonding */
//enum hydrogen_bonding_atom_types
//{
//    NON_H_BONDING_ATOM = 0,
//    H_ATOM = 1,
//    H_BONDING_ATOM = 2,
//};

/* interaction list (reax_list) storage format */
enum reax_list_format
{
    /* store half of interactions, when i < j (atoms i and j) */
    HALF_LIST = 0,
    /* store all interactions */
    FULL_LIST = 1,
};

/* sparse matrix (sparse_matrix) storage format */
enum sparse_matrix_format
{
    /* store upper half of nonzeros in a symmetric matrix (a_{ij}, i >= j) */
    SYM_HALF_MATRIX = 0,
    /* store all nonzeros in a symmetric matrix */
    SYM_FULL_MATRIX = 1,
    /* store all nonzeros in a matrix */
    FULL_MATRIX = 2,
};

enum message_tags
{
    INIT = 0,
    UPDATE = 1,
    BNDRY = 2,
    UPDATE_BNDRY = 3,
    EXC_VEC1 = 4,
    EXC_VEC2 = 5,
    DIST_RVEC2 = 6,
    COLL_RVEC2 = 7,
    DIST_RVECS = 8,
    COLL_RVECS = 9,
    INIT_DESCS = 10,
    ATOM_LINES = 11,
    BOND_LINES = 12,
    ANGLE_LINES = 13,
    RESTART_ATOMS = 14,
    TAGS_N = 15,
};

enum exchanges
{
    NONE = 0,
    NEAR_EXCH = 1,
    FULL_EXCH = 2,
};

enum gcell_types
{
    NO_NBRS = 0,
    NEAR_ONLY = 1,
    HBOND_ONLY = 2,
    FAR_ONLY = 4,
    NEAR_HBOND = 3,
    NEAR_FAR = 5,
    HBOND_FAR = 6,
    FULL_NBRS = 7,
    NATIVE = 8,
    NT_NBRS = 9, // 9 through 14
};

enum atoms
{
    C_ATOM = 0,
    H_ATOM = 1,
    O_ATOM = 2,
    N_ATOM = 3,
    S_ATOM = 4,
    SI_ATOM = 5,
    GE_ATOM = 6,
    X_ATOM = 7,
};

enum traj_methods
{
    REG_TRAJ = 0,
    MPI_TRAJ = 1,
    TF_N = 2,
};

enum molecules
{
    UNKNOWN = 0,
    WATER = 1,
};


typedef int  ivec[3];
typedef double real;
typedef real rvec[3];
typedef real rtensor[3][3];
typedef real rvec2[2];
typedef real rvec4[4];

/* struct declarations, see definitions below for comments */
typedef struct restart_header restart_header;
typedef struct restart_atom restart_atom;
typedef struct mpi_atom mpi_atom;
typedef struct boundary_atom boundary_atom;
typedef struct mpi_out_data mpi_out_data;
typedef struct mpi_datatypes mpi_datatypes;
typedef struct global_parameters global_parameters;
typedef struct single_body_parameters single_body_parameters;
typedef struct two_body_parameters two_body_parameters;
typedef struct three_body_parameters three_body_parameters;
typedef struct three_body_header three_body_header;
typedef struct hbond_parameters hbond_parameters;
typedef struct four_body_parameters four_body_parameters;
typedef struct four_body_header four_body_header;
typedef struct reax_interaction reax_interaction;
typedef struct reax_atom reax_atom;
typedef struct simulation_box simulation_box;
typedef struct grid_cell grid_cell;
typedef struct grid grid;
typedef struct neighbor_proc neighbor_proc;
typedef struct boundary_estimate boundary_estimate;
typedef struct boundary_cutoff boundary_cutoff;
typedef struct reax_system reax_system;
typedef struct control_params control_params;
typedef struct thermostat thermostat;
typedef struct isotropic_barostat isotropic_barostat;
typedef struct flexible_barostat flexible_barostat;
typedef struct reax_timing reax_timing;
typedef struct energy_data energy_data;
typedef struct simulation_data simulation_data;
typedef struct three_body_interaction_data three_body_interaction_data;
typedef struct far_neighbor_data far_neighbor_data;
#if defined(NEUTRAL_TERRITORY)
typedef struct nt_neighbor_data nt_neighbor_data;
#endif
typedef struct hbond_data hbond_data;
typedef struct dDelta_data dDelta_data;
typedef struct dbond_data dbond_data;
typedef struct bond_order_data bond_order_data;
typedef struct bond_data bond_data;
typedef struct sparse_matrix_entry sparse_matrix_entry;
typedef struct sparse_matrix sparse_matrix;
typedef struct reallocate_data reallocate_data;
typedef struct storage storage;
typedef struct reax_list reax_list;
typedef struct output_controls output_controls;
typedef struct molecule molecule;
typedef struct LR_data LR_data;
typedef struct cubic_spline_coef cubic_spline_coef;
typedef struct LR_lookup_table LR_lookup_table;


/* function pointer for calculating a bonded interaction */
typedef void (*interaction_function)( reax_system*, control_params*,
        simulation_data*, storage*, reax_list**, output_controls* );
#if defined(TEST_FORCES)
/* function pointers for printed bonded interactions */
typedef void (*print_interaction)(reax_system*, control_params*, simulation_data*,
        storage*, reax_list**, output_controls* );
#endif
/* function pointer for evolving the atomic system (i.e., updating the positions)
 * given the pre-computed forces from the prescribed interactions */
typedef void (*evolve_function)( reax_system*, control_params*,
        simulation_data*, storage*, reax_list**,
        output_controls*, mpi_datatypes* );
/**/
typedef real (*lookup_function)(real);
/**/
typedef void (*message_sorter) (reax_system*, int, int, int, mpi_out_data*);
/**/
typedef void (*unpacker) ( reax_system*, int, void*, int, neighbor_proc*, int );
/**/
typedef void (*dist_packer) (void*, mpi_out_data*);
/**/
typedef void (*coll_unpacker) (void*, void*, mpi_out_data*);



struct restart_header
{
    int step, bigN;
    real T, xi, v_xi, v_xi_old, G_xi;
    rtensor box;
};


struct restart_atom
{
    int orig_id;
    int type;
    char name[8];
    rvec x;
    rvec v;
};


struct mpi_atom
{
    int  orig_id;
    int  imprt_id;
    int  type;
    int  num_bonds;
    int  num_hbonds;
    //int  pad;  // pad to 8-byte address boundary
    char name[8];
    rvec x;     // position
    rvec v;     // velocity
    rvec f_old; // old force
    rvec4 s, t;  // for calculating q
};


struct boundary_atom
{
    int orig_id;
    int imprt_id;
    int type;
    int num_bonds;
    int num_hbonds;
//    int pad;
    rvec x;     // position
};


struct mpi_out_data
{
    //int  ncells;
    //int *cnt_by_gcell;

    int  cnt;
    //int *block;
    int *index;
    //MPI_Datatype out_dtype;
    void *out_atoms;
};


struct mpi_datatypes
{
    MPI_Comm world;
    MPI_Comm comm_mesh3D;

    MPI_Datatype sys_info;
    MPI_Datatype mpi_atom_type;
    MPI_Datatype boundary_atom_type;
    MPI_Datatype mpi_rvec, mpi_rvec2;
    MPI_Datatype restart_atom_type;

    MPI_Datatype header_line;
    MPI_Datatype header_view;
    MPI_Datatype init_desc_line;
    MPI_Datatype init_desc_view;
    MPI_Datatype atom_line;
    MPI_Datatype atom_view;
    MPI_Datatype bond_line;
    MPI_Datatype bond_view;
    MPI_Datatype angle_line;
    MPI_Datatype angle_view;
    mpi_out_data out_buffers[MAX_NBRS];

    void *in1_buffer;
    void *in2_buffer;

#if defined(NEUTRAL_TERRITORY)
    mpi_out_data out_nt_buffers[MAX_NT_NBRS];
    void *in_nt_buffer[MAX_NT_NBRS];
#endif
};


/* Global params mapping */
/*
l[0]  = p_boc1
l[1]  = p_boc2
l[2]  = p_coa2
l[3]  = N/A
l[4]  = N/A
l[5]  = N/A
l[6]  = p_ovun6
l[7]  = N/A
l[8]  = p_ovun7
l[9]  = p_ovun8
l[10] = N/A
l[11] = swa
l[12] = swb
l[13] = N/A
l[14] = p_val6
l[15] = p_lp1
l[16] = p_val9
l[17] = p_val10
l[18] = N/A
l[19] = p_pen2
l[20] = p_pen3
l[21] = p_pen4
l[22] = N/A
l[23] = p_tor2
l[24] = p_tor3
l[25] = p_tor4
l[26] = N/A
l[27] = p_cot2
l[28] = p_vdW1
l[29] = v_par30
l[30] = p_coa4
l[31] = p_ovun4
l[32] = p_ovun3
l[33] = p_val8
l[34] = N/A
l[35] = N/A
l[36] = N/A
l[37] = version number
l[38] = p_coa3
*/
struct global_parameters
{
    int n_global;
    real* l;
    int vdw_type;
};


struct single_body_parameters
{
    /* Line one in field file */
    char name[15]; // Two character atom name

    real r_s;
    real valency;  // Valency of the atom
    real mass;     // Mass of atom
    real r_vdw;
    real epsilon;
    real gamma;
    real r_pi;
    real valency_e;
    real nlp_opt;

    /* Line two in field file */
    real alpha;
    real gamma_w;
    real valency_boc;
    real p_ovun5;
    real chi;
    real eta;
    int  p_hbond; // 1 for H, 2 for hbonding atoms (O,S,P,N), 0 for others

    /* Line three in field file */
    real r_pi_pi;
    real p_lp2;
    real b_o_131;
    real b_o_132;
    real b_o_133;

    /* Line four in the field file */
    real p_ovun2;
    real p_val3;
    real valency_val;
    real p_val5;
    real rcore2;
    real ecore2;
    real acore2;
};


/* Two Body Parameters */
struct two_body_parameters
{
    /* Bond Order parameters */
    real p_bo1, p_bo2, p_bo3, p_bo4, p_bo5, p_bo6;
    real r_s, r_p, r_pp;  // r_o distances in BO formula
    real p_boc3, p_boc4, p_boc5;

    /* Bond Energy parameters */
    real p_be1, p_be2;
    real De_s, De_p, De_pp;

    /* Over/Under coordination parameters */
    real p_ovun1;

    /* Van der Waal interaction parameters */
    real D;
    real alpha;
    real r_vdW;
    real gamma_w;
    real rcore, ecore, acore;

    /* electrostatic parameters */
    real gamma; // note: this parameter is gamma^-3 and not gamma.

    real v13cor, ovc;
};


/* 3-body parameters */
struct three_body_parameters
{
    /* valence angle */
    real theta_00;
    real p_val1, p_val2, p_val4, p_val7;

    /* penalty */
    real p_pen1;

    /* 3-body conjugation */
    real p_coa1;
};


struct three_body_header
{
    int cnt;
    three_body_parameters prm[MAX_3BODY_PARAM];
};


/* hydrogen-bond parameters */
struct hbond_parameters
{
    real r0_hb, p_hb1, p_hb2, p_hb3;
};


/* 4-body parameters */
struct four_body_parameters
{
    real V1, V2, V3;

    /* torsion angle */
    real p_tor1;

    /* 4-body conjugation */
    real p_cot1;
};


struct four_body_header
{
    int cnt;
    four_body_parameters prm[MAX_4BODY_PARAM];
};


struct reax_interaction
{
    int num_atom_types;
    global_parameters gp;
    single_body_parameters *sbp;
    two_body_parameters **tbp;
    three_body_header ***thbp;
    hbond_parameters ***hbp;
    four_body_header ****fbp;
};


struct reax_atom
{
    int  orig_id;
    int  imprt_id;
    int  type;
    char name[8];

    rvec x; // position
    rvec v; // velocity
    rvec f; // force
    rvec f_old;

    real q; // charge
    rvec4 s; // they take part in
    rvec4 t; // computing q

    int Hindex;
    int num_bonds;
    int num_hbonds;
    int renumber;
#if defined(NEUTRAL_TERRITORY)
    int nt_dir;
    int pos;
#endif
};


struct simulation_box
{
    real V;
    rvec min, max, box_norms;

    rtensor box, box_inv;
    rtensor trans, trans_inv;
    rtensor g;
};


struct grid_cell
{
    real cutoff;
    rvec min, max;
    ivec rel_box;

    int  mark;
    int  type;
    int  str;
    int  end;
    int  top;
    int* atoms;
    struct grid_cell** nbrs;
    ivec* nbrs_x;
    rvec* nbrs_cp;
};


struct grid
{
    int  total, max_atoms, max_nbrs;
    ivec ncells;
    rvec cell_len;
    rvec inv_len;

    ivec bond_span;
    ivec nonb_span;
    ivec vlist_span;

    ivec native_cells;
    ivec native_str;
    ivec native_end;

    real ghost_cut;
    ivec ghost_span;
    ivec ghost_nonb_span;
    ivec ghost_hbond_span;
    ivec ghost_bond_span;

    grid_cell*** cells;
    ivec *order;
};


struct neighbor_proc
{
    int  rank;
#if defined(NEUTRAL_TERRITORY)
    int  receive_rank;
#endif
    int  est_send, est_recv;
    int  atoms_str, atoms_cnt;
    ivec rltv, prdc;
    rvec bndry_min, bndry_max;

    int  send_type;
    int  recv_type;
    ivec str_send;
    ivec end_send;
    ivec str_recv;
    ivec end_recv;
};


struct bound_estimate
{
    int N;
    int exc_gcells;
    int exc_atoms;
};


struct boundary_cutoff
{
    real ghost_nonb;
    real ghost_hbond;
    real ghost_bond;
    real ghost_cutoff;
};


struct reax_system
{
    /* number of locally owned atoms by this processor */
    int n;
    /* number of locally owned and ghost atoms by this processor */
    int N;
    /* total number of atoms across all processors (sum of locally owned atoms) */
    int bigN;
    /* number of locally owned Hydrogen atoms by this processor */
    int numH;
    /**/
    int local_cap;
    /**/
    int total_cap;
    /**/
    int gcell_cap;
    /**/
    int Hcap;
    /**/
    int est_recv;
    /**/
    int est_trans;
    /**/
    int max_recved;
    /* size (in terms of number of processors) of MPI
     * global communication (MPI_COMM_WORLD) */
    int wsize;
    /* MPI rank of the process */
    int my_rank;
    /**/
    int num_nbrs;
    /**/
    ivec my_coords;
    /**/
    neighbor_proc my_nbrs[MAX_NBRS];
    /**/
    neighbor_proc my_nt_nbrs[MAX_NT_NBRS];
    /**/
    reax_atom *my_atoms;
    /* simulation space (a.k.a. box) parameters */
    simulation_box big_box;
    /**/
    simulation_box my_box;
    /**/
    simulation_box my_ext_box;
    /* grid structure used for binning atoms and tracking neighboring bins */
    grid my_grid;
    /**/
    boundary_cutoff bndry_cuts;
    /* atomic interaction parameters */
    reax_interaction reax_param;
#if defined(NEUTRAL_TERRITORY)
    /**/
    int num_nt_nbrs;
#endif
};


/* system control parameters */
struct control_params
{
    /* simulation name, as supplied via control file */
    char sim_name[MAX_STR];
    /* number of MPI processors, as supplied via control file */
    int nprocs;
    /* MPI processors per each simulation dimension (cartesian topology),
     * as supplied via control file */
    ivec procs_by_dim;
    /* ensemble type for simulation, values:
     * 0 : NVE
     * 1 : bNVT (Berendsen)
     * 2 : nhNVT (Nose-Hoover)
     * 3 : sNPT (Parrinello-Rehman-Nose-Hoover) semiisotropic
     * 4 : iNPT (Parrinello-Rehman-Nose-Hoover) isotropic
     * 5 : NPT  (Parrinello-Rehman-Nose-Hoover) Anisotropic */
    int ensemble;
    /* num. of simulation time steps */
    int nsteps;
    /* length of time step, in femtoseconds */
    real dt;
    /* format of geometry input file */
    int geo_format;
    /* format of restart file */
    int restart;

    /**/
    int restrict_bonds;
    /* flag to control if center of mass velocity is removed */
    int remove_CoM_vel;
    /* flag to control if atomic initial velocity is randomly assigned */
    int random_vel;
    /* flag to control how atom repositioning is performed, values:
     * 0: fit to periodic box
     * 1: put center of mass to box center
     * 2: put center of mass to box origin  */
    int reposition_atoms;

    /* flag to control the frequency (in terms of simulation time stesp)
     * at which atom reneighboring is performed */
    int reneighbor;
    /* far neighbor (Verlet list) interaction cutoff, in Angstroms */
    real vlist_cut;
    /* bond interaction cutoff, in Angstroms */
    real bond_cut;
    /* non-bonded interaction cutoff, in Angstroms */
    real nonb_cut;
    /* ???, as supplied by force field parameters, in Angstroms */
    real nonb_low;
    /* hydrogen bond interaction cutoff, in Angstroms */
    real hbond_cut;
    /* ghost region cutoff (user-supplied via control file), in Angstroms */
    real user_ghost_cut;

    /* bond graph cutoff, as supplied by control file, in Angstroms */
    real bg_cut;
    /* bond order cutoff, as supplied by force field parameters, in Angstroms */
    real bo_cut;
    /* three body interaction cutoff, as supplied by control file, in Angstroms */
    real thb_cut;

    /* flag to control if force computations are tablulated */
    int tabulate;

    /* method for computing atomic charges */
    unsigned int charge_method;
    /* frequency (in terms of simulation time steps) at which to
     * re-compute atomic charge distribution */
    int charge_freq;
    /* iterative linear solver type */
    unsigned int cm_solver_type;
    /* system net charge */
    real cm_q_net;
    /* max. iterations for linear solver */
    unsigned int cm_solver_max_iters;
    /* max. iterations before restarting in specific solvers, e.g., GMRES(k) */
    unsigned int cm_solver_restart;
    /* error tolerance of solution produced by charge distribution
     * sparse iterative linear solver */
    real cm_solver_q_err;
    /* ratio used in computing sparser charge matrix,
     * between 0.0 and 1.0 */
    real cm_domain_sparsity;
    /* TRUE if enabled, FALSE otherwise */
    unsigned int cm_domain_sparsify_enabled;
    /* order of spline extrapolation used for computing initial guess
     * to linear solver */
    unsigned int cm_init_guess_extrap1;
    /* order of spline extrapolation used for computing initial guess
     * to linear solver */
    unsigned int cm_init_guess_extrap2;
    /* preconditioner type for linear solver */
    unsigned int cm_solver_pre_comp_type;
    /* frequency (in terms of simulation time steps) at which to recompute
     * incomplete factorizations */
    unsigned int cm_solver_pre_comp_refactor;
    /* drop tolerance of incomplete factorization schemes (ILUT, ICHOLT, etc.)
     * used for preconditioning the iterative linear solver used in charge distribution */
    real cm_solver_pre_comp_droptol;
    /* num. of sweeps for computing preconditioner factors
     * in fine-grained iterative methods (FG-ICHOL, FG-ILU) */
    unsigned int cm_solver_pre_comp_sweeps;
    /* relative num. of non-zeros to charge matrix used to
     * compute the sparse approximate inverse preconditioner,
     * between 0.0 and 1.0 */
    real cm_solver_pre_comp_sai_thres;
    /* preconditioner application type */
    unsigned int cm_solver_pre_app_type;
    /* num. of iterations used to apply preconditioner via
     * Jacobi relaxation scheme (truncated Neumann series) */
    unsigned int cm_solver_pre_app_jacobi_iters;

    /* initial temperature of simulation, in Kelvin */
    real T_init;
    /* final temperature of simulation, in Kelvin */
    real T_final;
    /* current temperature of simulation, in Kelvin */
    real T;
    /**/
    real Tau_T;
    /**/
    int  T_mode;
    /**/
    real T_rate;
    /**/
    real T_freq;

    /**/
    int  virial;
    /**/
    rvec P;
    /**/
    rvec Tau_P;
    /**/
    rvec Tau_PT;
    /**/
    int press_mode;
    /**/
    real compressibility;

    /**/
    int molecular_analysis;
    /**/
    int num_ignored;
    /**/
    int  ignore[MAX_ATOM_TYPES];

    /**/
    int dipole_anal;
    /**/
    int freq_dipole_anal;
    /**/
    int diffusion_coef;
    /**/
    int freq_diffusion_coef;
    /**/
    int restrict_type;
    /* function pointers for bonded interactions */
    interaction_function intr_funcs[NUM_INTRS];
#if defined(TEST_FORCES)
    /* function pointers for printed bonded interactions */
    print_interaction print_intr_funcs[NUM_INTRS];
#endif
};


struct thermostat
{
    real T;
    real xi;
    real v_xi;
    real v_xi_old;
    real G_xi;
};


struct isotropic_barostat
{
    real P;
    real eps;
    real v_eps;
    real v_eps_old;
    real a_eps;

};


struct flexible_barostat
{
    rtensor P;
    real P_scalar;

    real eps;
    real v_eps;
    real v_eps_old;
    real a_eps;

    rtensor h0;
    rtensor v_g0;
    rtensor v_g0_old;
    rtensor a_g0;
};


struct reax_timing
{
    /* start time of event */
    real start;
    /* end time of event */
    real end;
    /* total elapsed time of event */
    real elapsed;
    /* total simulation time */
    real total;
    /* communication time */
    real comm;
    /* neighbor list generation time */
    real nbrs;
    /* force initialization time */
    real init_forces;
    /* bonded force calculation time */
    real bonded;
    /* non-bonded force calculation time */
    real nonb;
    /* distance between pairs calculation time */
    real init_dist;
    /* charge matrix calculation time */
    real init_cm;
    /* bonded interactions calculation time */
    real init_bond;
    /* atomic charge distribution calculation time */
    real cm;
    /**/
    real cm_sort;
    /**/
    real cm_solver_comm;
    /**/
    real cm_solver_allreduce;
    /**/
    real cm_solver_pre_comp;
    /**/
    real cm_solver_pre_app;
    /* num. of steps in iterative linear solver for charge distribution */
    int cm_solver_iters;
    /**/
    real cm_solver_spmv;
    /**/
    real cm_solver_vector_ops;
    /**/
    real cm_solver_orthog;
    /**/
    real cm_solver_tri_solve;
    /* time spent on last preconditioner computation */
    real cm_last_pre_comp;
    /* time lost for not refactoring */
    real cm_total_loss;
    /* solver time on last refactoring step */
    real cm_optimum;
    /* neighbor list generation time on last refactoring step */
    real last_nbrs;
};


struct energy_data
{
    real e_tot;
    real e_kin;                      // Total kinetic energy
    real e_pot;

    real e_bond;                     // Total bond energy
    real e_ov;                       // Total over coordination
    real e_un;                       // Total under coordination energy
    real e_lp;                       // Total under coordination energy
    real e_ang;                      // Total valance angle energy
    real e_pen;                      // Total penalty energy
    real e_coa;                      // Total three body conjgation energy
    real e_hb;                       // Total Hydrogen bond energy
    real e_tor;                      // Total torsional energy
    real e_con;                      // Total four body conjugation energy
    real e_vdW;                      // Total van der Waals energy
    real e_ele;                      // Total electrostatics energy
    real e_pol;                      // Polarization energy
};


struct simulation_data
{
    int step;
    int prev_steps;
    /* to decide when to compute preconditioner for dynamic refactoring */
    int refactor;
    /* last refactoring step for dynamic refactoring */
    int last_pc_step;
    real time;

    real M;              // Total Mass
    real inv_M;                      // 1 / Total Mass

    rvec xcm;                        // Center of mass
    rvec vcm;                        // Center of mass velocity
    rvec fcm;                        // Center of mass force
    rvec amcm;                       // Angular momentum of CoM
    rvec avcm;                       // Angular velocity of CoM
    real etran_cm;                   // Translational kinetic energy of CoM
    real erot_cm;                    // Rotational kinetic energy of CoM

    rtensor kinetic;                 // Kinetic energy tensor
    rtensor virial;                  // Hydrodynamic virial

    energy_data my_en;
    energy_data sys_en;

    real N_f;          //Number of degrees of freedom
    rvec t_scale;
    rtensor p_scale;
    thermostat therm;        // Used in Nose_Hoover method
    isotropic_barostat iso_bar;
    flexible_barostat flex_bar;
    real inv_W;

    real kin_press;
    rvec int_press;
    rvec my_ext_press;
    rvec ext_press;
    rvec tot_press;

    reax_timing timing;
};


struct three_body_interaction_data
{
    int thb;
    int pthb; // pointer to the third body on the central atom's nbrlist
    real theta;
    real cos_theta;
    rvec dcos_di;
    rvec dcos_dj;
    rvec dcos_dk;
};


struct far_neighbor_data
{
    /* neighbor atom IDs */
    int *nbr;
    /* set of three integers which deterimine if the neighbor
     * atom is a non-periodic neighbor (all zeros) or a periodic
     * neighbor and which perioidic image this neighbor comes from */
    ivec *rel_box;
    /* distance to the neighboring atom */
    real *d;
    /* difference between positions of this atom and its neighboring atom */
    rvec *dvec;
};


#if defined(NEUTRAL_TERRITORY)
struct nt_neighbor_data
{
    int nbr;
    ivec rel_box;
    real d;
    rvec dvec;
};
#endif


struct hbond_data
{
    /* neighbor atom ID */
    int nbr;
    /* ??? */
    int scl;
    /* position of neighbor in far neighbor list */
    int ptr;
};


struct dDelta_data
{
    int wrt;
    rvec dVal;
};


struct dbond_data
{
    int wrt;
    rvec dBO, dBOpi, dBOpi2;
};

struct bond_order_data
{
    real BO;
    real BO_s;
    real BO_pi;
    real BO_pi2;
    real Cdbo;
    real Cdbopi;
    real Cdbopi2;
    real C1dbo;
    real C2dbo;
    real C3dbo;
    real C1dbopi;
    real C2dbopi;
    real C3dbopi;
    real C4dbopi;
    real C1dbopi2;
    real C2dbopi2;
    real C3dbopi2;
    real C4dbopi2;
    rvec dBOp;
    rvec dln_BOp_s;
    rvec dln_BOp_pi;
    rvec dln_BOp_pi2;
};


struct bond_data
{
    int nbr;
    int sym_index;
    int dbond_index;
    ivec rel_box;
//    rvec ext_factor;
    real d;
    rvec dvec;
    bond_order_data bo_data;
};


struct sparse_matrix_entry
{
    int j;
    real val;
};


struct sparse_matrix
{
    /* matrix storage format */
    int format;
    int cap, n, m;
#if defined(NEUTRAL_TERRITORY)
    int NT;
#endif
    int *start, *end;
    sparse_matrix_entry *entries;
};


struct reallocate_data
{
    int num_far;
    int H, Htop;
    int hbonds, num_hbonds;
    int bonds, num_bonds;
    int num_3body;
    int gcell_atoms;
};


struct storage
{
    int allocated;

    /* communication storage */
    real *tmp_dbl[MAX_NBRS];
    rvec *tmp_rvec[MAX_NBRS];
    rvec2 *tmp_rvec2[MAX_NBRS];
    int  *within_bond_box;

    /* bond order related storage */
    real *total_bond_order;
    real *Deltap, *Deltap_boc;
    real *Delta, *Delta_lp, *Delta_lp_temp, *Delta_e, *Delta_boc;
    real *dDelta_lp, *dDelta_lp_temp;
    real *nlp, *nlp_temp, *Clp, *vlpex;
    rvec *dDeltap_self;
    int *bond_mark;

    /* charge matrix storage */
    sparse_matrix *H;
    sparse_matrix *L;
    sparse_matrix *U;
    sparse_matrix *H_full;
    sparse_matrix *H_spar_patt;
    sparse_matrix *H_spar_patt_full;
    sparse_matrix *H_app_inv;
    real *Hdia_inv;
    real *b_s;
    real *b_t;
    real *b_prc;
    real *b_prm;
    real *s;
    real *t;
    real *droptol;
    rvec2 *b;
    rvec2 *x;

    /* GMRES storage */
    real *hc;
    real *hs;
    real **h;
    real **v;

    /* GMRES, BiCGStab storage */
    real *g;
    real *y;

    /* GMRES, BiCGStab, PIPECG, PIPECR storage */
    real *z;

    /* CG, BiCGStab, PIPECG, PIPECR storage */
    real *r;
    real *d;
    real *q;
    real *p;

    /* BiCGStab storage */
    real *r_hat;
    real *q_hat;

    /* PIPECG, PIPECR storage */
    real *m;
    real *n;
    real *u;
    real *w;

    /* dual-CG storage */
    rvec2 *d2;
    rvec2 *p2;
    rvec2 *q2;
    rvec2 *r2;

    /* dual-PIPECG storage */
    rvec2 *m2;
    rvec2 *n2;
    rvec2 *u2;
    rvec2 *w2;
    rvec2 *z2;

    /* Taper */
    real Tap[8];

    /* storage for analysis */
    int *mark, *old_mark;
    rvec *x_old;

    /* storage space for bond restrictions */
    int *restricted;
    int **restricted_list;

    /* integrator */
    rvec *v_const;

    /* force calculations */
    real *CdDelta;  // coefficient of dDelta
    rvec *f;
#ifdef TEST_FORCES
    rvec *f_ele;
    rvec *f_vdw;
    rvec *f_bo;
    rvec *f_be;
    rvec *f_lp;
    rvec *f_ov;
    rvec *f_un;
    rvec *f_ang;
    rvec *f_coa;
    rvec *f_pen;
    rvec *f_hb;
    rvec *f_tor;
    rvec *f_con;
    rvec *f_tot;
    rvec *dDelta;   // calculated on the fly in bond_orders.c together with bo'

    int  *rcounts;
    int  *displs;
    int  *id_all;
    rvec *f_all;
#endif

    reallocate_data realloc;

    LR_lookup_table **LR;

    //int *num_bonds;
    /* hydrogen bonds */
    //int   num_H, Hcap;
    //int  *Hindex;
    //int *num_hbonds;
    //int *hash;
    //int *rev_hash;
};


union list_type
{
    //void *v;
    //three_body_interaction_data *three_body_list;
    //bond_data          *bond_list;
    //dbond_data         *dbo_list;
    //dDelta_data        *dDelta_list;
    //far_neighbor_data  *far_nbr_list;
    //hbond_data         *hbond_list;
};


struct reax_list
{
    int allocated;

    int n;
    int num_intrs;

    int *index;
    int *end_index;

    int type;
//    list_type select;

    /* list storage format (half or full) */
    int format;

    void *v;
    three_body_interaction_data *three_body_list;
    bond_data *bond_list;
    dbond_data *dbo_list;
    dDelta_data *dDelta_list;
    far_neighbor_data far_nbr_list;
#if defined(NEUTRAL_TERRITORY)
    nt_neighbor_data *nt_nbr_list;
#endif
    hbond_data *hbond_list;
};


struct output_controls
{
#if defined(PURE_REAX)
    MPI_File trj;
#endif
    FILE *strj;
    int   trj_offset;
    int   atom_line_len;
    int   bond_line_len;
    int   angle_line_len;
    int   write_atoms;
    int   write_bonds;
    int   write_angles;
    char *line;
    int   buffer_len;
    char *buffer;

    FILE *out;
    FILE *pot;
    FILE *log;
    FILE *mol, *ign;
    FILE *dpl;
    FILE *drft;
    FILE *pdb;
    FILE *prs;

    int   write_steps;
    int   traj_compress;
    int   traj_method;
    char  traj_title[MAX_STR];
    int   atom_info;
    int   bond_info;
    int   angle_info;

    int   restart_format;
    int   restart_freq;
    int   debug_level;
    int   energy_update_freq;

#ifdef TEST_ENERGY
    FILE *ebond;
    FILE *elp, *eov, *eun;
    FILE *eval, *epen, *ecoa;
    FILE *ehb;
    FILE *etor, *econ;
    FILE *evdw, *ecou;
#endif

#ifdef TEST_FORCES
    FILE *fbo, *fdbo;
    FILE *fbond;
    FILE *flp, *fov, *fun;
    FILE *fang, *fcoa, *fpen;
    FILE *fhb;
    FILE *ftor, *fcon;
    FILE *fvdw, *fele;
    FILE *ftot, *fcomp;
#endif

#if defined(TEST_ENERGY) || defined(TEST_FORCES)
    FILE *flist; // far neighbor list
    FILE *blist; // bond list
    FILE *nlist; // near neighbor list
#endif
};


struct molecule
{
    int atom_count;
    int atom_list[MAX_MOLECULE_SIZE];
    int mtypes[MAX_ATOM_TYPES];
};


struct LR_data
{
    real H;
    real e_vdW, CEvd;
    real e_ele, CEclmb;
};


struct cubic_spline_coef
{
    real a, b, c, d;
};


struct LR_lookup_table
{
    real xmin, xmax;
    int n;
    real dx, inv_dx;
    real a;
    real m;
    real c;

    LR_data *y;
    cubic_spline_coef *H;
    cubic_spline_coef *vdW, *CEvd;
    cubic_spline_coef *ele, *CEclmb;
};


#endif
