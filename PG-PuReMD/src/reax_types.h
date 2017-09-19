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

#if !(defined(__REAX_TYPES_H_) || defined(__CUDA_REAX_TYPES_H_))

#ifdef __CUDACC__
  #ifndef __CUDA_REAX_TYPES_H_
    #define __CUDA_REAX_TYPES_H_
    #define CUDA_HOST __host__
    #define CUDA_DEVICE __device__
    #define CUDA_GLOBAL __global__
    #define CUDA_HOST_DEVICE __host__ __device__
  #endif
#else
  #ifndef __REAX_TYPES_H_
    #define __REAX_TYPES_H_
    #define CUDA_HOST
    #define CUDA_DEVICE
    #define CUDA_GLOBAL
    #define CUDA_HOST_DEVICE
  #endif
#endif

#if (defined(HAVE_CONFIG_H) && !defined(__CONFIG_H_))
  #define __CONFIG_H_
  #include "config.h"
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
#ifdef HAVE_CUDA
  #include <cuda.h>
  #include <cuda_runtime.h>
#endif

#if defined(__IBMC__)
  #define inline __inline__
#endif /*IBMC*/

#define PURE_REAX
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

#define SUCCESS  1
#define FAILURE  0
#define TRUE  1
#define FALSE 0

#define EXP    exp
#define EXP2   exp2
#define LOG    log
#define LOG2   log2
#define SQRT   sqrt
#define POW    pow
#define COS    cos
#define ACOS   acos
#define SIN    sin
#define TAN    tan
#define ATAN2  atan2
#define CEIL   ceil
#define FLOOR  floor
#define FABS   fabs
#define FMOD   fmod

/* transcendental constant pi */
#if defined(M_PI)
  /* GNU C library (libc), defined in math.h */
  #define PI (M_PI)
#else
  #define PI            3.14159265
#endif

#define SQR(x)        ((x)*(x))
#define CUBE(x)       ((x)*(x)*(x))
#define DEG2RAD(a)    ((a)*PI/180.0)
#define RAD2DEG(a)    ((a)*180.0/PI)
#define MAX(x,y)      (((x) > (y)) ? (x) : (y))
#define MIN(x,y)      (((x) < (y)) ? (x) : (y))
#define MAX3(x,y,z)   MAX( MAX(x,y), z)

/* ??? */
#define C_ele          332.06371
/* ??? */
//#define K_B         503.398008   // kcal/mol/K
#define K_B             0.831687   // amu A^2 / ps^2 / K
/* ??? */
#define F_CONV          1e6 / 48.88821291 / 48.88821291   // --> amu A / ps^2
/**/
#define E_CONV          0.002391   // amu A^2 / ps^2 --> kcal/mol
/* conversion factor from electron volts to kilo calories per mole  */
#define EV_to_KCALpMOL 14.400000
/* conversion factor from kilo calories per mode to electron volts */
#define KCALpMOL_to_EV 23.060549   // 23.020000
/* conversion factor from (elemental charge * angstroms) to debye */
#define ECxA_to_DEBYE   4.803204   // elem. charge * Ang -> debye
/* conversion factor from calories to joules */
#define CAL_to_JOULES   4.184000
/* conversion factor from joules to calories */
#define JOULES_to_CAL   1/4.184000
/* conversion factor from (unified) atomic mass units to grams */
#define AMU_to_GRAM     1.6605e-24
/* conversion factor from angstroms to centimenters */
#define ANG_to_CM       1e-8
/* Avogadro's constant */
#define AVOGNR          6.0221367e23
/* ??? */
#define P_CONV          1e-24 * AVOGNR * JOULES_to_CAL

#define MAX_STR             1024
#define MAX_LINE            1024
#define MAX_TOKENS          1024
#define MAX_TOKEN_LEN       1024
#define MAX_ATOM_NAME_LEN   8

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
#define MIN_CM_ENTRIES 100
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
/* encoding of relative coordinate (0,0,0) */
#define MYSELF 13

#define MAX_ITR 10
#define RESTART 30

#define MAX_RETRIES 20

/* NaN IEEE 754 representation for C99 in math.h
 * Note: function choice must match REAL typedef below */
#if defined(NAN)
  #define IS_NAN_REAL(a) (isnan(a))
#else
  #warn "No support for NaN"
  #define NAN_REAL(a) (0)
#endif

/**************** RESOURCE CONSTANTS **********************/
/* 500 MB */
#define HOST_SCRATCH_SIZE               (1024 * 1024 * 500)
#ifdef HAVE_CUDA
/* 500 MB */
#define DEVICE_SCRATCH_SIZE             (1024 * 1024 * 500)
/* 500 MB */
#define RES_SCRATCH                     0x90

/* BLOCK SIZES for kernels */
#define HB_SYM_BLOCK_SIZE                   64
#define HB_KER_SYM_THREADS_PER_ATOM         16
#define HB_POST_PROC_BLOCK_SIZE             256
#define HB_POST_PROC_KER_THREADS_PER_ATOM   32

#if defined( __INIT_BLOCK_SIZE__)
  #define DEF_BLOCK_SIZE                      __INIT_BLOCK_SIZE__    /* all utility functions and all */
  #define CUDA_BLOCK_SIZE                     __INIT_BLOCK_SIZE__     /* init forces */
  #define ST_BLOCK_SIZE                       __INIT_BLOCK_SIZE__
#else
  #define DEF_BLOCK_SIZE                      256                     /* all utility functions and all */
  #define CUDA_BLOCK_SIZE                     256                     /* init forces */
  #define ST_BLOCK_SIZE                       256
#endif

#if defined( __NBRS_THREADS_PER_ATOM__ )
  #define NB_KER_THREADS_PER_ATOM             __NBRS_THREADS_PER_ATOM__
#else
  #define NB_KER_THREADS_PER_ATOM             16
#endif

#if defined( __NBRS_BLOCK_SIZE__)
  #define NBRS_BLOCK_SIZE                     __NBRS_BLOCK_SIZE__
#else
  #define NBRS_BLOCK_SIZE                     256
#endif

#if defined( __HB_THREADS_PER_ATOM__)
  #define HB_KER_THREADS_PER_ATOM             __HB_THREADS_PER_ATOM__
#else
  #define HB_KER_THREADS_PER_ATOM             32
#endif

#if defined(__HB_BLOCK_SIZE__)
  #define HB_BLOCK_SIZE                   __HB_BLOCK_SIZE__
#else
  #define HB_BLOCK_SIZE                       256
#endif

#if defined( __VDW_THREADS_PER_ATOM__ )
  #define VDW_KER_THREADS_PER_ATOM            __VDW_THREADS_PER_ATOM__
#else
  #define VDW_KER_THREADS_PER_ATOM            32
#endif

#if defined( __VDW_BLOCK_SIZE__)
  #define VDW_BLOCK_SIZE                      __VDW_BLOCK_SIZE__
#else
  #define VDW_BLOCK_SIZE                      256
#endif

#if defined( __MATVEC_THREADS_PER_ROW__ )
  #define MATVEC_KER_THREADS_PER_ROW      __MATVEC_THREADS_PER_ROW__
#else
  #define MATVEC_KER_THREADS_PER_ROW      32
#endif

#if defined( __MATVEC_BLOCK_SIZE__)
  #define MATVEC_BLOCK_SIZE                   __MATVEC_BLOCK_SIZE__
#else
  #define MATVEC_BLOCK_SIZE                   512
#endif

//Validation
#define GPU_TOLERANCE               1e-5

#endif


/******************* ENUMERATIONS *************************/
/* ensemble type */
enum ensembles
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
enum lists
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
enum interactions
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

/* MPI message tags */
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
    MAX_RETRIES_REACHED = -18,
    RUNTIME_ERROR = -19,
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

enum charge_method
{
    QEQ_CM = 0,
    EE_CM = 1,
    ACKS2_CM = 2,
};

enum solver
{
    GMRES_S = 0,
    GMRES_H_S = 1,
    CG_S = 2,
    SDM_S = 3,
};

enum pre_comp
{
    NONE_PC = 0,
    DIAG_PC = 1,
    ICHOLT_PC = 2,
    ILU_PAR_PC = 3,
    ILUT_PAR_PC = 4,
    ILU_SUPERLU_MT_PC = 5,
};

enum pre_app
{
    TRI_SOLVE_PA = 0,
    TRI_SOLVE_LEVEL_SCHED_PA = 1,
    TRI_SOLVE_GC_PA = 2,
    JACOBI_ITER_PA = 3,
};

/* ??? */
enum exchanges
{
    NONE = 0,
    NEAR_EXCH = 1,
    FULL_EXCH = 2,
};

/* ??? */
enum gcell_types
{
    NO_NBRS = 0,
    NEAR_ONLY = 1,
    HBOND_ONLY = 2,
    FAR_ONLY = 3,
    NEAR_HBOND = 4,
    NEAR_FAR = 5,
    HBOND_FAR = 6,
    FULL_NBRS = 7,
    NATIVE = 8,
};

/* atom types as pertains to hydrogen bonding */
enum hydrogen_bonding_atom_types
{
    NON_H_BONDING_ATOM = -1,
    H_ATOM = 1,
    H_BONDING_ATOM = 2,
};

/* trajectory file formats */
enum traj_methods
{
    REG_TRAJ = 0,
    MPI_TRAJ = 1,
    TF_N = 2,
};

/* ??? */
enum molecules
{
    UNKNOWN = 0,
    WATER = 1,
};


/********************** TYPE DEFINITIONS ********************/
/* 3D vector, integer values */
typedef int ivec[3];
/* double precision floating point */
typedef double real;
/* 3D vector, double precision floating point values */
typedef real rvec[3];
/* 3D tensor, double precision floating point values */
typedef real rtensor[3][3];
/* 2D vector, double precision floating point values */
typedef real rvec2[2];
/* 4D vector, double precision floating point values */
typedef real rvec4[4];


/* header used in restart file */
typedef struct
{
    /* current simulation time step */
    int step;
    /* total num. atoms in simulation */
    int bigN;
    /* thermostat temperature */
    real T;
    /* thrmostat ??? */
    real xi;
    /* thrmostat ??? */
    real v_xi;
    /* thrmostat ??? */
    real v_xi_old;
    /* thrmostat ??? */
    real G_xi;
    /* ??? */
    rtensor box;
} restart_header;


/* atom type used for restarting simulation */
typedef struct
{
    /* atom serial number as given in the geo file */
    int orig_id;
    /* non-negative integer used to indicate atom type,
     * as identified by short element string in force field file (single
     * body parameters section) */
    int type;
    /* atom name as given in the geo file */
    char name[MAX_ATOM_NAME_LEN];
    /* atomic position, 3D */
    rvec x;
    /* atomic velocity, 3D */
    rvec v;
} restart_atom;


/* atom type used for MPI communications */
typedef struct
{
    /* atom serial number as given in the geo file */
    int orig_id;
    /* local atom ID on neighbor processor ??? */
    int imprt_id;
    /* non-negative integer used to indicate atom type,
     * as identified by short element string in force field file (single
     * body parameters section) */
    int type;
    /* num. bonds associated with atom */
    int num_bonds;
    /* num. hydrogren bonds associated with atom */
    int num_hbonds;
    /* pad to 8-byte address boundary */
    //int  pad;
    /* atom name as given in the geo file */
    char name[MAX_ATOM_NAME_LEN];
    /* atomic position, 3D */
    rvec x;
    /* atomic velocity, 3D */
    rvec v;
    /* net force acting upon atom in previous time step, 3D */
    rvec f_old;
    /* atomic fictitious charge used during QEq to compute atomic charge,
     * multiple entries used to hold old values for extrapolation */
    rvec4 s;
    /* atomic fictitious charge used during QEq to compute atomic charge,
     * multiple entries used to hold old values for extrapolation */
    rvec4 t;
} mpi_atom;


/* atom type used for MPI communications at boundary regions */
typedef struct
{
    /* atom serial number as given in the geo file */
    int orig_id;
    /* local atom ID on neighbor processor ??? */
    int imprt_id;
    /* non-negative integer used to indicate atom type,
     * as identified by short element string in force field file (single
     * body parameters section) */
    int type;
    /* num. bonds associated with atom */
    int num_bonds;
    /* num. hydrogren bonds associated with atom */
    int num_hbonds;
    /* pad to 8-byte address boundary */
    //int pad;
    /* atomic position, 3D */
    rvec x;
} boundary_atom;


/**/
typedef struct
{
    /**/
    int cnt;
    /**/
    int *index;
    /**/
    void *out_atoms;
} mpi_out_data;


/**/
typedef struct
{
    /**/
    MPI_Comm world;
    /**/
    MPI_Comm comm_mesh3D;

    /**/
    MPI_Datatype sys_info;
    /**/
    MPI_Datatype mpi_atom_type;
    /**/
    MPI_Datatype boundary_atom_type;
    /**/
    MPI_Datatype mpi_rvec;
    /**/
    MPI_Datatype mpi_rvec2;
    /**/
    MPI_Datatype restart_atom_type;

    /**/
    MPI_Datatype header_line;
    /**/
    MPI_Datatype header_view;
    /**/
    MPI_Datatype init_desc_line;
    /**/
    MPI_Datatype init_desc_view;
    /**/
    MPI_Datatype atom_line;
    /**/
    MPI_Datatype atom_view;
    /**/
    MPI_Datatype bond_line;
    /**/
    MPI_Datatype bond_view;
    /**/
    MPI_Datatype angle_line;
    /**/
    MPI_Datatype angle_view;

    /**/
    mpi_out_data out_buffers[MAX_NBRS];
    /**/
    void *in1_buffer;
    /**/
    void *in2_buffer;
} mpi_datatypes;


/* Global parameters in force field parameters file, mapping:
 *
 * l[0]  = p_boc1
 * l[1]  = p_boc2
 * l[2]  = p_coa2
 * l[3]  = N/A
 * l[4]  = N/A
 * l[5]  = N/A
 * l[6]  = p_ovun6
 * l[7]  = N/A
 * l[8]  = p_ovun7
 * l[9]  = p_ovun8
 * l[10] = N/A
 * l[11] = swa
 * l[12] = swb
 * l[13] = N/A
 * l[14] = p_val6
 * l[15] = p_lp1
 * l[16] = p_val9
 * l[17] = p_val10
 * l[18] = N/A
 * l[19] = p_pen2
 * l[20] = p_pen3
 * l[21] = p_pen4
 * l[22] = N/A
 * l[23] = p_tor2
 * l[24] = p_tor3
 * l[25] = p_tor4
 * l[26] = N/A
 * l[27] = p_cot2
 * l[28] = p_vdW1
 * l[29] = v_par30
 * l[30] = p_coa4
 * l[31] = p_ovun4
 * l[32] = p_ovun3
 * l[33] = p_val8
 * l[34] = N/A
 * l[35] = N/A
 * l[36] = N/A
 * l[37] = version number
 * l[38] = p_coa3
 * */
typedef struct
{
    /* num. of global parameters, from the force field file */
    int n_global;
    /* global parameters, see above mapping */
    real* l;
    /* van der Waals interaction type, values:
     * 0: none (???)
     * 1: shielded Morse, no inner-wall
     * 2: inner wall, no shielding
     * 3: inner wall + shielding
     * */
    int vdw_type;
} global_parameters;


/* single body parameters in force field parameters file */
typedef struct
{
    /* Line one in field file */
    /* two character atom name */
    char name[15];
    /**/
    real r_s;
    /* valency of the atom */
    real valency;
    /* mass of atom */
    real mass;
    /**/
    real r_vdw;
    /**/
    real epsilon;
    /**/
    real gamma;
    /**/
    real r_pi;
    /**/
    real valency_e;
    /**/
    real nlp_opt;

    /* Line two in field file */
    /**/
    real alpha;
    /**/
    real gamma_w;
    /**/
    real valency_boc;
    /**/
    real p_ovun5;
    /**/
    real chi;
    /**/
    real eta;
    /* info related to hydrogen bonding
     * (values correspond to hydrogen_bonding_atom_types enum above):
     *  0: non-hydrogen bonding atom
     *  1: H atom
     *  2: hydrogen bonding atom (e.g., O, S, P, N) */
    int p_hbond;

    /* Line three in field file */
    /**/
    real r_pi_pi;
    /**/
    real p_lp2;
    /**/
    real b_o_131;
    /**/
    real b_o_132;
    /**/
    real b_o_133;

    /* Line four in the field file */
    /**/
    real p_ovun2;
    /**/
    real p_val3;
    /**/
    real valency_val;
    /**/
    real p_val5;
    /**/
    real rcore2;
    /**/
    real ecore2;
    /**/
    real acore2;
} single_body_parameters;


/* 2-body parameters for a single interaction type,
 * from the force field parameters file */
typedef struct
{
    /* Bond Order parameters */
    /**/
    real p_bo1;
    /**/
    real p_bo2;
    /**/
    real p_bo3;
    /**/
    real p_bo4;
    /**/
    real p_bo5;
    /**/
    real p_bo6;
    /**/
    real r_s;
    /**/
    real r_p;
    /**/
    real r_pp;  // r_o distances in BO formula
    /**/
    real p_boc3;
    /**/
    real p_boc4;
    /**/
    real p_boc5;

    /* Bond Energy parameters */
    /**/
    real p_be1;
    /**/
    real p_be2;
    /**/
    real De_s;
    /**/
    real De_p;
    /**/
    real De_pp;

    /* Over/Under coordination parameters */
    /**/
    real p_ovun1;

    /* Van der Waal interaction parameters */
    /**/
    real D;
    /**/
    real alpha;
    /**/
    real r_vdW;
    /**/
    real gamma_w;
    /**/
    real rcore;
    /**/
    real ecore;
    /**/
    real acore;

    /* electrostatic parameters,
     * note: this parameter is gamma^-3 and not gamma */
    real gamma;

    /**/
    real v13cor;
    /**/
    real ovc;
} two_body_parameters;


/* 3-body parameters for a single interaction type,
 * from the force field parameters file */
typedef struct
{
    /* valence angle */
    real theta_00;
    real p_val1, p_val2, p_val4, p_val7;

    /* penalty */
    real p_pen1;

    /* 3-body conjugation */
    real p_coa1;
} three_body_parameters;


/* three body interactions info. */
typedef struct
{
    /* num. of three body parameters */
    int cnt;
    /* collection of three body parameters, indexed by atomic types */
    three_body_parameters prm[MAX_3BODY_PARAM];
} three_body_header;


/* hydrogen bond parameters in force field parameters file */
typedef struct
{
    /**/
    real r0_hb;
    /**/
    real p_hb1;
    /**/
    real p_hb2;
    /**/
    real p_hb3;
} hbond_parameters;


/* 4-body parameters for a single interaction type,
 * from the force field parameters file */
typedef struct
{
    /**/
    real V1;
    /**/
    real V2;
    /**/
    real V3;

    /* torsion angle */
    real p_tor1;

    /* 4-body conjugation */
    real p_cot1;
} four_body_parameters;


/* four body interactions info. */
typedef struct
{
    /* num. of four body parameters */
    int cnt;
    /* collection of four body parameters, indexed by atomic types */
    four_body_parameters prm[MAX_4BODY_PARAM];
} four_body_header;


/* atomic interaction parameters */
typedef struct
{
    /* num. of atom types, from force field parameters file */
    int num_atom_types;

    /* global simulation parameters, from force field parameters file */
    global_parameters gp;
    /* simulation parameters for single body interactions */
    single_body_parameters *sbp;
    /* simulation parameters for two body interactions */
    two_body_parameters *tbp; 
    /* simulation parameters for three body interactions */
    three_body_header *thbp; 
    /* simulation parameters for hydrogen bonding interactions */
    hbond_parameters *hbp; 
    /* simulation parameters for four body interactions */
    four_body_header *fbp; 

#ifdef HAVE_CUDA
    /* global simulation parameters (GPU), from force field parameters file */
    global_parameters d_gp;
    /* simulation parameters for single body interactions (GPU) */
    single_body_parameters *d_sbp;
    /* simulation parameters for two body interactions (GPU) */
    two_body_parameters *d_tbp;
    /* simulation parameters for three body interactions (GPU) */
    three_body_header *d_thbp;
    /* simulation parameters for hydrogen bonding interactions (GPU) */
    hbond_parameters *d_hbp;
    /* simulation parameters for four body interactions (GPU) */
    four_body_header *d_fbp;
#endif
} reax_interaction;


/**/
typedef struct
{
    /* atom serial number as given in the geo file */
    int orig_id;
    /* local atom ID on neighbor processor ??? */
    int imprt_id;
    /* non-negative integer used to indicate atom type,
     * as identified by short element string in force field file (single
     * body parameters section) */
    int type;
    /* atom name as given in the geo file */
    char name[MAX_ATOM_NAME_LEN];

    /* atomic position, 3D */
    rvec x;
    /* atomic velocity, 3D */
    rvec v;
    /* net force acting upon atom, 3D */
    rvec f;
    /* net force acting upon atom in previous time step, 3D */
    rvec f_old;

    /* atomic charge, computed during coulombic interaction */
    real q;
    /* atomic fictitious charge used during QEq to compute atomic charge,
     * multiple entries used to hold old values for extrapolation */
    rvec4 s;
    /* atomic fictitious charge used during QEq to compute atomic charge,
     * multiple entries used to hold old values for extrapolation */
    rvec4 t;

    /* unique non-negative integer index of atom if it is a hydrogen atom,
     * -1 otherwise */
    int Hindex;
    /* num. bonds associated with atom */
    int num_bonds;
    /* num. hydrogren bonds associated with atom */
    int num_hbonds;
    /* ??? */
    int renumber;
} reax_atom;


/* Info. regarding 3D simulation space */
typedef struct
{
    /* total volume */
    real V;
    /* min. coordinate of box in Angstroms, 3D */
    rvec min;
    /* max. coordinate of box in Angstroms, 3D */
    rvec max;
    /* length of each dimension of the simulation box in Angstroms, 3D */
    rvec box_norms;

    /* ??? */
    rtensor box;
    /* ??? */
    rtensor box_inv;
    /* ??? */
    rtensor trans;
    /* ??? */
    rtensor trans_inv;
    /* ??? */
    rtensor g;
} simulation_box;


/**/
typedef struct
{
    /* min. cell coordinate (top-left) */
    rvec min;
    /* max. cell coordinate (bottom-right) */
    rvec max;
 
    /* ??? */
    int mark;
    /* native or ghost cells (contains atoms only of resp. type) */
    int type;
    /* count of num. of atoms currently within this grid cell */
    int top;
    /* IDs of atoms within this grid cell */
    int* atoms;
} grid_cell;


/* info. for 3D domain (i.e., spatial) partitioning of atoms
 * inside the simulation box */
typedef struct
{
    /* total number of grid cells (native AND ghost) */
    int total;
    /* max. num. of atoms with a grid cell can contain */
    int max_atoms;
    /**/
    int max_nbrs;
    /* num. of grid cells in each dimension, 3D */
    ivec ncells;
    /* lengths of each grid cell dimension, 3D */
    rvec cell_len;
    /* multiplicative inverses of lengths of each grid cell dimension, 3D */
    rvec inv_len;

    /* bond interaction cutoff in terms of num. of grid cells in each dimension, 3D */
    ivec bond_span;
    /* non-bonded interaction cutoff in terms of num. of grid cells in each dimension, 3D */
    ivec nonb_span;
    /* Verlet list (i.e., neighbor list) cutoff in terms of num. of grid cells in each dimension, 3D */
    ivec vlist_span;

    /* partitioning of ??? */
    ivec native_cells;
    /**/
    ivec native_str;
    /**/
    ivec native_end;

    /**/
    real ghost_cut;
    /**/
    ivec ghost_span;
    /**/
    ivec ghost_nonb_span;
    /**/
    ivec ghost_hbond_span;
    /**/
    ivec ghost_bond_span;

    /**/
    grid_cell* cells;
    /**/
    ivec *order;
 
    /**/
    int *str;
    /**/
    int *end;
    /**/
    real *cutoff;
    /* rel. positions of cells which fall within neighbor cut-off of a given cell */
    ivec *nbrs_x;
    /* corner points of cells which fall within neighbor cut-off of a given cell */
    rvec *nbrs_cp;
 
    /**/
    ivec *rel_box;
} grid;


/**/
typedef struct
{
    /**/
    int rank;
    /**/
    int est_send;
    /**/
    int est_recv;
    /**/
    int atoms_str;
    /**/
    int atoms_cnt;
    /**/
    ivec rltv;
    /**/
    ivec prdc;
    /**/
    rvec bndry_min;
    /**/
    rvec bndry_max;

    /**/
    int  send_type;
    /**/
    int  recv_type;
    /**/
    ivec str_send;
    /**/
    ivec end_send;
    /**/
    ivec str_recv;
    /**/
    ivec end_recv;
} neighbor_proc;


/**/
typedef struct
{
    /**/
    int N;
    /**/
    int exc_gcells;
    /**/
    int exc_atoms;
} bound_estimate;


/**/
typedef struct
{
    /**/
    real ghost_nonb;
    /**/
    real ghost_hbond;
    /**/
    real ghost_bond;
    /**/
    real ghost_cutoff;
} boundary_cutoff;


/**/
typedef struct
{
    /* atomic interaction parameters */
    reax_interaction reax_param;

    /* num. atoms (locally owned) within spatial domain of MPI process */
    int n;
    /* num. atoms (locally owned AND ghost region) within spatial domain of MPI process */
    int N;
    /* num. atoms within simulation */
    int bigN;
    /* dimension of sparse charge method matrix */
    int N_cm;
    /* num. hydrogen atoms */
    int numH;
    /* num. hydrogen atoms (GPU) */
    int *d_numH;
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
    /**/
    int my_rank;
    /**/
    int num_nbrs;
    /* coordinates of processor (according to rank) in MPI cartesian topology */
    ivec my_coords;
    /* list of neighbor processors */
    neighbor_proc my_nbrs[MAX_NBRS];

    /* global simulation box */
    simulation_box big_box;
    /* local simulation box of owned atoms per processor */
    simulation_box my_box;
    /* local simulation box of owned AND ghost atoms per processor */
    simulation_box my_ext_box;
    /* global simulation box (GPU) */
    simulation_box *d_big_box;
    /* local simulation box of owned atoms per processor (GPU) */
    simulation_box *d_my_box;
    /* local simulation box of owned AND ghost atoms per processor (GPU) */
    simulation_box *d_my_ext_box;

    /* grid specifying domain (i.e., spatial) decompisition
     * of atoms within simulation box */
    grid my_grid;
    /* grid specifying domain (i.e., spatial) decompisition
     * of atoms within simulation box (GPU) */
    grid d_my_grid;

    /* boundary cutoffs, in ??? */
    boundary_cutoff bndry_cuts;

    /* collection of atomic info. */
    reax_atom *my_atoms;
    /* collection of atomic info. (GPU) */
    reax_atom *d_my_atoms;

    /* current num. of far neighbors per atom */
    int *far_nbrs;
    /* current num. of far neighbors per atom (GPU) */
    int *d_far_nbrs;
    /* max num. of far neighbors per atom */
    int *max_far_nbrs;
    /* max num. of far neighbors per atom (GPU) */
    int *d_max_far_nbrs;
    /* total num. of (max) far neighbors across all atoms */
    int total_far_nbrs;
    /* total num. of (max) far neighbors across all atoms (GPU) */
    int *d_total_far_nbrs;
    /* TRUE if far neighbors list requires reallocation,
     * FALSE otherwise (GPU) */
    int *d_realloc_far_nbrs;

    /* num. bonds per atom */
    int *bonds;
    /* num. bonds per atom (GPU) */
    int *d_bonds;
    /* max. num. bonds per atom */
    int *max_bonds;
    /* max. num. bonds per atom (GPU) */
    int *d_max_bonds;
    /* total num. bonds (sum over max) */
    int total_bonds;
    /* total num. bonds (sum over max) (GPU) */
    int *d_total_bonds;
    /* TRUE if bonds list requires reallocation, FALSE otherwise (GPU) */
    int *d_realloc_bonds;

    /* num. hydrogen bonds per atom */
    int *hbonds;
    /* num. hydrogen bonds per atom (GPU) */
    int *d_hbonds;
    /* max. num. hydrogen bonds per atom */
    int max_hbonds;
    //int *max_hbonds;
    /* max. num. hydrogen bonds per atom (GPU) */
    int *d_max_hbonds;
    /* total num. hydrogen bonds (sum over max) */
    int total_hbonds;
    /* total num. hydrogen bonds (sum over max) (GPU) */
    int *d_total_hbonds;
    /* TRUE if hydrogen bonds list requires reallocation, FALSE otherwise (GPU) */
    int *d_realloc_hbonds;

    /* num. matrix entries per row (GPU) */
    int *d_cm_entries;
    /* max. num. matrix entries per row (GPU) */
    int *d_max_cm_entries;
    /* total num. matrix entries (sum over max) */
    int total_cm_entries;
    /* total num. matrix entries (sum over max) (GPU) */
    int *d_total_cm_entries;
    /* TRUE if charge matrix requires reallocation, FALSE otherwise (GPU) */
    int *d_realloc_cm_entries;

    /* total num. three body list indices */
    int total_thbodies_indices;
    /* total num. three body interactions */
    int total_thbodies;
    /* total num. three body interactions (GPU) */
    int *d_total_thbodies;
} reax_system;


/* system control parameters */
typedef struct
{
    /* simulation name, as supplied via control file */
    char sim_name[MAX_STR];
    /* number of MPI processors, as supplied via control file */
    int nprocs;
    /* number of GPUs per node, as supplied via control file */
    int gpus_per_node;
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

    /**/
    unsigned int charge_method;
    /* frequency (in terms of simulation time steps) at which to
     * re-compute atomic charge distribution */
    int charge_freq;
    /**/
    unsigned int cm_solver_type;
    /**/
    real cm_q_net;
    /**/
    unsigned int cm_solver_max_iters;
    /**/
    unsigned int cm_solver_restart;
    /* error tolerance of solution produced by charge distribution
     * sparse iterative linear solver */
    real cm_solver_q_err;
    /**/
    real cm_domain_sparsity;
    /**/
    unsigned int cm_domain_sparsify_enabled;
    /**/
    unsigned int cm_solver_pre_comp_type;
    /* frequency (in terms of simulation time steps) at which to recompute
     * incomplete factorizations */
    unsigned int cm_solver_pre_comp_refactor;
    /* drop tolerance of incomplete factorization schemes (ILUT, ICHOLT, etc.)
     * used for preconditioning the iterative linear solver used in charge distribution */
    real cm_solver_pre_comp_droptol;
    /**/
    unsigned int cm_solver_pre_comp_sweeps;
    /**/
    unsigned int cm_solver_pre_app_type;
    /**/
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
    int ignore[MAX_ATOM_TYPES];

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

    /* control parameters (GPU) */
    void *d_control_params;
} control_params;


typedef struct
{
    /**/
    real T;
    /**/
    real xi;
    /**/
    real v_xi;
    /**/
    real v_xi_old;
    /**/
    real G_xi;

} thermostat;


typedef struct
{
    /**/
    real P;
    /**/
    real eps;
    /**/
    real v_eps;
    /**/
    real v_eps_old;
    /**/
    real a_eps;

} isotropic_barostat;


typedef struct
{
    /**/
    rtensor P;
    /**/
    real P_scalar;

    /**/
    real eps;
    /**/
    real v_eps;
    /**/
    real v_eps_old;
    /**/
    real a_eps;

    /**/
    rtensor h0;
    /**/
    rtensor v_g0;
    /**/
    rtensor v_g0_old;
    /**/
    rtensor a_g0;

} flexible_barostat;


typedef struct
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
    /* neighbor (i.e., Verlet) list generation time */
    real nbrs;
    /* force initialization time */
    real init_forces;
    /* bonded force calculation time */
    real bonded;
    /* non-bonded force calculation time */
    real nonb;
    /* atomic charge distribution calculation time */
    real cm;
    /* num. of steps in iterative linear solver for charge distribution (QEq, first solve) */
    int s_matvecs;
    /* num. of steps in iterative linear solver for charge distribution (QEq, second solve) */
    int t_matvecs;
    /* num. of retries in main sim. loop */
    int num_retries;
} reax_timing;


typedef struct
{
    /* total energy */
    real e_tot;
    /* total kinetic energy */
    real e_kin;
    /* total potential energy */
    real e_pot;

    /* total bond energy */
    real e_bond;
    /* total over coordination */
    real e_ov;
    /* total under coordination energy */
    real e_un;
    /* total under coordination energy */
    real e_lp;
    /* total valance angle energy */
    real e_ang;
    /* total penalty energy */
    real e_pen;
    /* total three body conjugation energy */
    real e_coa;
    /* total Hydrogen bond energy */
    real e_hb;
    /* total torsional energy */
    real e_tor;
    /* total four body conjugation energy */
    real e_con;
    /* total van der Waals energy */
    real e_vdW;
    /* total electrostatics energy */
    real e_ele;
    /* polarization energy */
    real e_pol;
} energy_data;


/**/
typedef struct
{
    /**/
    int step;
    /**/
    int prev_steps;
    /**/
    real time;

    /**/
    real M;              // Total Mass
    /**/
    real inv_M;                      // 1 / Total Mass

    /**/
    rvec xcm;                        // Center of mass
    /**/
    rvec vcm;                        // Center of mass velocity
    /**/
    rvec fcm;                        // Center of mass force
    /**/
    rvec amcm;                       // Angular momentum of CoM
    /**/
    rvec avcm;                       // Angular velocity of CoM
    /**/
    real etran_cm;                   // Translational kinetic energy of CoM
    /**/
    real erot_cm;                    // Rotational kinetic energy of CoM

    /**/
    rtensor kinetic;                 // Kinetic energy tensor
    /**/
    rtensor virial;                  // Hydrodynamic virial

    /**/
    energy_data my_en;
    /**/
    energy_data sys_en;

    /**/
    real N_f;          //Number of degrees of freedom
    /**/
    rvec t_scale;
    /**/
    rtensor p_scale;
    /**/
    thermostat therm;        // Used in Nose_Hoover method
    /**/
    isotropic_barostat iso_bar;
    /**/
    flexible_barostat flex_bar;
    /**/
    real inv_W;

    /**/
    real kin_press;
    /**/
    rvec int_press;
    /**/
    rvec my_ext_press;
    /**/
    rvec ext_press;
    /**/
    rvec tot_press;

    /**/
    reax_timing timing;
    /**/
    reax_timing d_timing;

    /**/
    void *d_simulation_data;
} simulation_data;


/**/
typedef struct
{
    /**/
    int thb;
    /* pointer to the third body on the central atom's nbrlist */
    int pthb;
    /**/
    real theta;
    /**/
    real cos_theta;
    /**/
    rvec dcos_di;
    /**/
    rvec dcos_dj;
    /**/
    rvec dcos_dk;
} three_body_interaction_data;


/* info. about a far neighbor to an atom */
typedef struct
{
    /**/
    int nbr;
    /**/
    ivec rel_box;
    /**/
    real d;
    /**/
    rvec dvec;
} far_neighbor_data;


/**/
typedef struct
{
    /**/
    int nbr;
    /**/
    int scl;
    /**/
    far_neighbor_data *ptr;

    /*CUDA-specific*/
    /**/
    int sym_index;
    /**/
    rvec hb_f;
} hbond_data;


/**/
typedef struct
{
    /**/
    int wrt;
    /**/
    rvec dVal;
} dDelta_data;


/**/
typedef struct
{
    /**/
    int wrt;
    /**/
    rvec dBO;
    /**/
    rvec dBOpi;
    /**/
    rvec dBOpi2;
} dbond_data;


/**/
typedef struct
{
    /**/
    real BO;
    /**/
    real BO_s;
    /**/
    real BO_pi;
    /**/
    real BO_pi2;
    /**/
    real Cdbo;
    /**/
    real Cdbopi;
    /**/
    real Cdbopi2;
    /**/
    real C1dbo;
    /**/
    real C2dbo;
    /**/
    real C3dbo;
    /**/
    real C1dbopi;
    /**/
    real C2dbopi;
    /**/
    real C3dbopi;
    /**/
    real C4dbopi;
    /**/
    real C1dbopi2;
    /**/
    real C2dbopi2;
    /**/
    real C3dbopi2;
    /**/
    real C4dbopi2;
    /**/
    rvec dBOp;
    /**/
    rvec dln_BOp_s;
    /**/
    rvec dln_BOp_pi;
    /**/
    rvec dln_BOp_pi2;
} bond_order_data;


/**/
typedef struct
{
    /**/
    int nbr;
    /**/
    int sym_index;
    /**/
    int dbond_index;
    /**/
    ivec rel_box;
    //  rvec ext_factor;
    /**/
    real d;
    /**/
    rvec dvec;
    /**/
    bond_order_data bo_data;

    /*CUDA-specific*/
    /**/
    real ae_CdDelta;

    /**/
    real va_CdDelta;
    /**/
    rvec va_f;

    /**/
    real ta_CdDelta;
    /**/
    real ta_Cdbo;
    /**/
    rvec ta_f;

    /**/
    rvec hb_f;

    /**/
    rvec tf_f;
} bond_data;


/* Secondary structure for matrix in CRS format */
typedef struct
{
    /* column index for corresponding matrix entry */
    int j;
    /* matrix entry */
    real val;
} sparse_matrix_entry;


/* Matrix in compressed row storage (CRS) format,
 * with modifications for row end pointer and max entries per row (CUDA optimizations).
 * See, e.g.,
 *   http://netlib.org/linalg/html_templates/node91.html#SECTION00931100000000000000
 */
typedef struct
{
    /* number of rows */
    int n;
    /* number of nonzeros (NNZ) ALLOCATED */
    int m;
    /* row start pointer (last element contains ACTUAL NNZ) */
    int *start;
    /* row end pointer */
    int *end;
    /* secondary structure for matrix entry info */
    sparse_matrix_entry *entries;
} sparse_matrix;


/* used to determine if and how much space should be reallocated */
typedef struct
{
    /* TRUE if far neighbor list needs
     * to be reallocated, FALSE otherwise */
    int far_nbrs;
    /* TRUE if charge matrix needs
     * to be reallocated, FALSE otherwise */
    int cm;
    /**/
    int Htop;
    /**/
    int hbonds;
    /**/
    int num_hbonds;
    /* TRUE if bonds list needs
     * to be reallocated, FALSE otherwise */
    int bonds;
    /**/
    int num_bonds;
    /* TRUE if three body list needs
     * to be reallocated, FALSE otherwise */
    int thbody;
    /**/
    int num_3body;
    /**/
    int gcell_atoms;
} reallocate_data;


typedef struct
{
    /* 0 if struct members are NOT allocated, 1 otherwise */
    int allocated;

    /* communication storage */
    /**/
    real *tmp_dbl[MAX_NBRS];
    /**/
    rvec *tmp_rvec[MAX_NBRS];
    /**/
    rvec2 *tmp_rvec2[MAX_NBRS];
    /**/
    int  *within_bond_box;

    /* bond order related storage */
    /**/
    real *total_bond_order;
    /**/
    real *Deltap;
    /**/
    real *Deltap_boc;
    /**/
    real *Delta;
    /**/
    real *Delta_lp;
    /**/
    real *Delta_lp_temp;
    /**/
    real *Delta_e;
    /**/
    real *Delta_boc;
    /**/
    real *dDelta_lp;
    /**/
    real *dDelta_lp_temp;
    /**/
    real *nlp;
    /**/
    real *nlp_temp;
    /**/
    real *Clp;
    /**/
    real *vlpex;
    /**/
    rvec *dDeltap_self;
    /**/
    int *bond_mark;
    /**/
    int *done_after;

    /* charge matrix storage */
    /* charge matrix */
    sparse_matrix H;
    /* preconditioner */
    sparse_matrix L;
    /* preconditioner */
    sparse_matrix U;
    /* preconditioner */
    real *Hdia_inv;
    /**/
    real *b_s;
    /**/
    real *b_t;
    /**/
    real *b_prc;
    /**/
    real *b_prm;
    /**/
    real *s;
    /**/
    real *t;
    /**/
    real *droptol;
    /**/
    rvec2 *b;
    /**/
    rvec2 *x;

    /* GMRES storage */
    /**/
    real *y;
    /**/
    real *z;
    /**/
    real *g;
    /**/
    real *hc;
    /**/
    real *hs;
    /**/
    real *h;
    /**/
    real *v;

    /* CG storage */
    /**/
    real *r;
    /**/
    real *d;
    /**/
    real *q;
    /**/
    real *p;
    /**/
    rvec2 *r2;
    /**/
    rvec2 *d2;
    /**/
    rvec2 *q2;
    /**/
    rvec2 *p2;

    /* Taper */
    /* Tap7, Tap6, Tap5, Tap4, Tap3, Tap2, Tap1, Tap0 */
    real Tap[8];

    /* storage for analysis */
    /**/
    int *mark;
    /**/
    int *old_mark;
    /**/
    rvec *x_old;

    /* storage space for bond restrictions */
    /**/
    int *restricted;
    /**/
    int *restricted_list;

    /* integrator */
    /**/
    rvec *v_const;

    /* force calculations */
    /**/
    real *CdDelta;  // coefficient of dDelta
    /**/
    rvec *f;
#ifdef TEST_FORCES
    /**/
    rvec *f_ele;
    /**/
    rvec *f_vdw;
    /**/
    rvec *f_bo;
    /**/
    rvec *f_be;
    /**/
    rvec *f_lp;
    /**/
    rvec *f_ov;
    /**/
    rvec *f_un;
    /**/
    rvec *f_ang;
    /**/
    rvec *f_coa;
    /**/
    rvec *f_pen;
    /**/
    rvec *f_hb;
    /**/
    rvec *f_tor;
    /**/
    rvec *f_con;
    /**/
    rvec *f_tot;
    /**/
    rvec *dDelta;   // calculated on the fly in bond_orders.c together with bo'

    /**/
    int  *rcounts;
    /**/
    int  *displs;
    /**/
    int  *id_all;
    /**/
    rvec *f_all;
#endif

    /**/
    reallocate_data realloc;
} storage;


/* Union used for determining interaction list type */
typedef union
{
    /* void type */
    void *v;
    /* three body type */
    three_body_interaction_data *three_body_list;
    /* bond type */
    bond_data *bond_list;
    /* derivative bond order type */
    dbond_data *dbo_list;
    /* derivative delta type */
    dDelta_data *dDelta_list;
    /* far neighbor type */
    far_neighbor_data *far_nbr_list;
    /* hydrogen bond type */
    hbond_data *hbond_list;
} list_type;


/* Interaction list */
typedef struct
{
    /* 0 if struct members are NOT allocated, 1 otherwise */
    int allocated;

    /* total num. of entities, each of which correspond to one of more interactions */
    int n;
    /* total num. of interactions */
    int num_intrs;

    /* beginning position for interactions corresponding to a particular entity,
     * where the entity ID used for indexing is an integer between 0 and n - 1, inclusive */
    int *index;
    /* ending position for interactions corresponding to a particular entity,
     * where the entity ID used for indexing is an integer between 0 and n - 1, inclusive */
    int *end_index;

    /* interaction list type, as defined by interactions enum above */
    int type;
    /* interaction list, made purposely non-opaque via above union to avoid typecasts */
    list_type select;
} reax_list;


/**/
typedef struct
{
#if defined(PURE_REAX)
    /**/
    MPI_File trj;
#endif
    /**/
    FILE *strj;
    /**/
    int trj_offset;
    /**/
    int atom_line_len;
    /**/
    int bond_line_len;
    /**/
    int angle_line_len;
    /**/
    int write_atoms;
    /**/
    int write_bonds;
    /**/
    int write_angles;
    /**/
    char *line;
    /**/
    int buffer_len;
    /**/
    char *buffer;

    /**/
    FILE *out;
    /**/
    FILE *pot;
    /**/
    FILE *log;
    /**/
    FILE *mol;
    /**/
    FILE *ign;
    /**/
    FILE *dpl;
    /**/
    FILE *drft;
    /**/
    FILE *pdb;
    /**/
    FILE *prs;

    /**/
    int write_steps;
    /**/
    int traj_compress;
    /**/
    int traj_method;
    /**/
    char traj_title[81];
    /**/
    int atom_info;
    /**/
    int bond_info;
    /**/
    int angle_info;

    /**/
    int restart_format;
    /**/
    int restart_freq;
    /**/
    int debug_level;
    /**/
    int energy_update_freq;

#ifdef TEST_ENERGY
    /**/
    FILE *ebond;
    /**/
    FILE *elp;
    /**/
    FILE *eov;
    /**/
    FILE *eun;
    /**/
    FILE *eval;
    /**/
    FILE *epen;
    /**/
    FILE *ecoa;
    /**/
    FILE *ehb;
    /**/
    FILE *etor;
    /**/
    FILE *econ;
    /**/
    FILE *evdw;
    /**/
    FILE *ecou;
#endif

#ifdef TEST_FORCES
    /**/
    FILE *fbo;
    /**/
    FILE *fdbo;
    /**/
    FILE *fbond;
    /**/
    FILE *flp;
    /**/
    FILE *fov;
    /**/
    FILE *fun;
    /**/
    FILE *fang;
    /**/
    FILE *fcoa;
    /**/
    FILE *fpen;
    /**/
    FILE *fhb;
    /**/
    FILE *ftor;
    /**/
    FILE *fcon;
    /**/
    FILE *fvdw;
    /**/
    FILE *fele;
    /**/
    FILE *ftot;
    /**/
    FILE *fcomp;
#endif

#if defined(TEST_ENERGY) || defined(TEST_FORCES)
    /* far neighbor list */
    FILE *flist;
    /* bond list */
    FILE *blist;
    /* near neighbor list */
    FILE *nlist;
#endif
} output_controls;


/**/
typedef struct
{
    /**/
    int atom_count;
    /**/
    int atom_list[MAX_MOLECULE_SIZE];
    /**/
    int mtypes[MAX_ATOM_TYPES];
} molecule;


/**/
typedef struct
{
    /**/
    real H;
    /**/
    real e_vdW;
    /**/
    real CEvd;
    /**/
    real e_ele;
    /**/
    real CEclmb;
} LR_data;


/**/
typedef struct
{
    /**/
    real a;
    /**/
    real b;
    /**/
    real c;
    /**/
    real d;
} cubic_spline_coef;


/**/
typedef struct
{
    /**/
    real xmin;
    /**/
    real xmax;
    /**/
    int n;
    /**/
    real dx;
    /**/
    real inv_dx;
    /**/
    real a;
    /**/
    real m;
    /**/
    real c;

    /**/
    LR_data *y;
    /**/
    cubic_spline_coef *H;
    /**/
    cubic_spline_coef *vdW;
    /**/
    cubic_spline_coef *CEvd;
    /**/
    cubic_spline_coef *ele;
    /**/
    cubic_spline_coef *CEclmb;
} LR_lookup_table;


extern LR_lookup_table *LR;


/* function pointer defs */
typedef int (*evolve_function)(reax_system*, control_params*,
        simulation_data*, storage*, reax_list**, output_controls*, mpi_datatypes* );
#if defined(PURE_REAX)
extern evolve_function Evolve;
extern evolve_function Cuda_Evolve;
#endif

typedef void (*interaction_function)(reax_system*, control_params*,
        simulation_data*, storage*, reax_list**, output_controls*);

typedef void (*print_interaction)(reax_system*, control_params*,
        simulation_data*, storage*, reax_list**, output_controls*);

typedef real (*lookup_function)(real);

typedef void (*message_sorter)(reax_system*, int, int, int, mpi_out_data*);
typedef void (*unpacker)( reax_system*, int, void*, int, neighbor_proc*, int );

typedef void (*dist_packer)(void*, mpi_out_data*);
typedef void (*coll_unpacker)(void*, void*, mpi_out_data*);

/*CUDA-specific*/
extern reax_list **dev_lists;
extern storage *dev_workspace;
extern LR_lookup_table *d_LR;

extern void *scratch;
extern void *host_scratch;
extern int BLOCKS, BLOCKS_POW_2, BLOCK_SIZE;
extern int BLOCKS_N, BLOCKS_POW_2_N;
extern int MATVEC_BLOCKS;

#endif
