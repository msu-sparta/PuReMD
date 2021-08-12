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

#if defined(__CUDACC__)
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
#if defined(HAVE_CUDA)
  #include <cuda.h>
  #include <cuda_runtime.h>
#endif

/* IBMC */
#if defined(__IBMC__)
  #define inline __inline__
#endif

/* compile verbose debug logging */
//#define DEBUG_FOCUS
/* compile energy calculation test code */
//#define TEST_ENERGY
/* compile force calculation test code */
//#define TEST_FORCES
/* constants defined in reference Fortran ReaxFF code (useful for comparisons) */
#define USE_REF_FORTRAN_REAXFF_CONSTANTS
/* constants defined in reference Fortran eReaxFF code (useful for comparisons) */
//#define USE_REF_FORTRAN_EREAXFF_CONSTANTS
/* constants defined in LAMMPS ReaxFF code (useful for comparisons) */
//#define USE_LAMMPS_REAXFF_CONSTANTS
/* pack/unpack MPI buffers on device and use CUDA-aware MPI for messaging */
//TODO: rewrite this to use configure option to include
/* OpenMPI extensions for CUDA-aware support */
//#include <mpi-ext.h>
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  #define CUDA_DEVICE_PACK
#endif
/* aggregate atomic energies and forces using atomic operations */
#define CUDA_ACCUM_ATOMIC

/* disable assertions if NOT compiling with debug support --
 * the definition (or lack thereof) controls how the assert macro is defined */
#if !defined(DEBUG) && !defined(DEBUG_FOCUS)
  #define NDEBUG
#endif
#include <assert.h>

#define SUCCESS (1)
#define FAILURE (0)
#define TRUE (1)
#define FALSE (0)

#define EXP exp
#define EXP2 exp2
#define LOG log
#define LOG2 log2
#define SQRT sqrt
#define POW pow
#define COS cos
#define ACOS acos
#define SIN sin
#define TAN tan
#define ATAN2 atan2
#define CEIL ceil
#define FLOOR floor
#define FABS fabs
#define FMOD fmod

/* transcendental constant pi */
#if defined(M_PI)
  /* GNU C library (libc), defined in math.h */
  #define PI (M_PI)
#else
  #define PI (3.14159265)
#endif

#define SQR(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))
#define DEG2RAD(a) ((a)*PI/180.0)
#define RAD2DEG(a) ((a)*180.0/PI)
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#define MAX3(x,y,z) MAX( MAX((x),(y)), (z))

#if defined(USE_REF_FORTRAN_REAXFF_CONSTANTS)
  /* transcendental constant pi */
//  #define PI (3.14159265)
  /* unit conversion from ??? to kcal / mol */
  #define C_ELE (332.0638)
  /* Boltzmann constant, AMU * A^2 / (ps^2 * K) */
  #define K_B (0.831687)
//  #define K_B (0.8314510)
  /* unit conversion for atomic force to AMU * A / ps^2 */
  #define F_CONV (4.184e2)
  /* energy conversion constant from electron volts to kilo-calories per mole */
  #define KCALpMOL_to_EV (23.02)
  /* electric dipole moment conversion constant from elementary charge * angstrom to debye */
  #define ECxA_to_DEBYE (4.80320679913)
#elif defined(USE_REF_FORTRAN_EREAXFF_CONSTANTS)
  //TODO
  /* energy conversion constant from electron volts to kilo-calories per mole */
  #define KCALpMOL_to_EV (23.02)
  /* electric dipole moment conversion constant from elementary charge * angstrom to debye */
  #define ECxA_to_DEBYE (4.80320679913)
#elif defined(USE_LAMMPS_REAXFF_CONSTANTS)
  //TODO
  /* energy conversion constant from electron volts to kilo-calories per mole */
  #define KCALpMOL_to_EV (23.060549)
#endif

/* Coulomb energy conversion */
#if !defined(C_ELE)
  #define C_ELE (332.06371)
#endif
/* Boltzmann constant */
#if !defined(K_B)
  /* in ??? */
//  #define K_B (503.398008)
  /* Boltzmann constant, AMU * A^2 / (ps^2 * K) */
//  #define K_B (0.831687)
  #define K_B (0.8314510)
#endif
/* unit conversion for atomic force */
#if !defined(F_CONV)
  /* to AMU * A / ps^2 */
  #define F_CONV (1.0e6 / 48.88821291 / 48.88821291)
#endif
/* unit conversion for atomic energy */
#if !defined(E_CONV)
  /* AMU * Angstroms^2 / ps^2 --> kcal / mol */
  #define E_CONV (0.002391)
#endif
/* energy conversion constant from electron volts to kilo-calories per mole */
#if !defined(EV_to_KCALpMOL)
  #define EV_to_KCALpMOL (14.40)
#endif
/* energy conversion constant from electron volts to kilo-calories per mole */
#if !defined(KCALpMOL_to_EV)
  #define KCALpMOL_to_EV (23.0408)
#endif
/* electric dipole moment conversion constant from elementary charge * angstrom to debye */
#if !defined(ECxA_to_DEBYE)
  #define ECxA_to_DEBYE (4.803204)
#endif
/* energy conversion constant from (gram) calories to Joules (SI) */
#if !defined(CAL_to_JOULES)
  #define CAL_to_JOULES (4.1840)
#endif
/* energy conversion constant from Joules (SI) to (gram) calories */
#if !defined(JOULES_to_CAL)
  #define JOULES_to_CAL (1.0 / 4.1840)
#endif
/* mass conversion constant from unified atomic mass units (daltons) to grams */
#if !defined(AMU_to_GRAM)
  #define AMU_to_GRAM (1.6605e-24)
#endif
/* distance conversion constant from angstroms to centimeters */
#if !defined(ANG_to_CM)
  #define ANG_to_CM (1.0e-8)
#endif
/* Avogradro's constant */
#if !defined(AVOGNR)
  #define AVOGNR (6.0221367e23)
#endif
/* unit conversion for pressure:
 * (1 s / 10^12 ps) * (1 m / 10^10 Angstroms) * (6.0221367^23 atoms / mole) * (0.2390057 J / cal)
 * ps * Angstroms * moles * cals => s * m * atoms * J */
#if !defined(P_CONV)
  #define P_CONV (1.0e-24 * AVOGNR * JOULES_to_CAL)
#endif

/* max. num. of characters for string buffers */
#define MAX_STR (1024)
/* max. num. of characters for a line in files */
#define MAX_LINE (1024)
/* max. num. of tokens per line */
#define MAX_TOKENS (1024)
/* max. num. of characters per token */
#define MAX_TOKEN_LEN (1024)
/* max. num. of characters in atom name */
#define MAX_ATOM_NAME_LEN (8)

/* max. atom ID in geo file */
#define MAX_ATOM_ID (100000)
/* ??? */
#define MAX_RESTRICT (15)
/* max. num. atoms per molecule */
#define MAX_MOLECULE_SIZE (20)
/* max. num. atom types defined in the force field parameter file */
#define MAX_ATOM_TYPES (25)

/* max. num. of interaction functions */
#define NUM_INTRS (10)
/* constant for double precision zero */
#define ZERO (0.000000000000000e+00)
/* threshold for determining if values are sufficiently close to double precision zero */
#define ALMOST_ZERO (1.0e-10)
/* constant for indicating atom pairs are effectively a
 * negative infinite distance apart along a dimension */
#define NEG_INF (-1.0e10)
/* threshold for determining if bond order values are large enough
 * to constitute a hydrogen bond interaction */
#define HB_THRESHOLD (1.0e-2)

/* minimum capacity (entries) in various interaction lists */
#define MIN_CAP (64)
/* minimum number of interactions per entry in the far neighbor list */
#define MIN_NBRS (96)
/* minimum number of non-zero entries per row in the charge matrix */
#define MIN_CM_ENTRIES (96)
/* minimum number of interactions per entry in the bond list */
#define MIN_BONDS (32)
/* minimum number of interactions per entry in the hydrogen bond list */
#define MIN_HBONDS (32)
/* minimum capacity (entries) in 3-body interaction list */
#define MIN_3BODIES (1024)
/* minimum number of atoms per grid cell */
#define MIN_GCELL_POPL (64)
/* ??? */
#define MIN_SEND (100)
/* over-allocation factor for various data structures */
#define SAFE_ZONE (1.2)
/* larger over-allocation factor for various data structures */
#define SAFER_ZONE (1.4)
/* over-allocation factor for various neutral territory data structures */
#define SAFE_ZONE_NT (2.0)
/* larger over-allocation factor for various neutral territory data structures */
#define SAFER_ZONE_NT (2.5)
/* utilization threshold which when exceeded triggers reallocation
 * with increased capacity for various data structures */
#define DANGER_ZONE (0.90)
/* utilization threshold which when exceeded triggers reallocation
 * with increased capacity for various data structures */
#define LOOSE_ZONE (0.75)
/* ??? */
#define MAX_3BODY_PARAM (5)
/* ??? */
#define MAX_4BODY_PARAM (5)

/* max. pressure scaler for simulation box dimenion in NPT ensembles */
#define MAX_dV (1.01)
/* min. pressure scaler for simulation box dimenion in NPT ensembles */
#define MIN_dV (0.99)
/* max. temperature scaler for atomic positions and velocities in NPT ensembles */
#define MAX_dT (2.0)
/* min. temperature scaler for atomic positions and velocities in NPT ensembles */
#define MIN_dT (0.0)

/* ??? */
#define MASTER_NODE (0)
/* max neighbors in 3D Cartesian processor grid */
#define MAX_NBRS (6) // (27)
/* max neighbors in 3D Cartesian processor grid for neutral territory */
#define MAX_NT_NBRS (6)
/* encoding of relative coordinate (0,0,0) */
#define MYSELF (13)

/* max. num. of main simulation loop retries;
 * retries occur when memory allocation checks determine more memory is needed */
#define MAX_RETRIES (5)

/**************** RESOURCE CONSTANTS **********************/
#if defined(HAVE_CUDA)
  /* BLOCK SIZES for kernels */
  #define HB_SYM_BLOCK_SIZE (64)
  #define HB_KER_SYM_THREADS_PER_ATOM (16)
  #define HB_POST_PROC_BLOCK_SIZE (256)
  #define HB_POST_PROC_KER_THREADS_PER_ATOM (32)
  
  #if defined(__INIT_BLOCK_SIZE__)
    /* all utility functions and all */
    #define DEF_BLOCK_SIZE (__INIT_BLOCK_SIZE__)
  #else
    /* all utility functions and all */
    #define DEF_BLOCK_SIZE (256)
  #endif
  
  #if defined( __NBRS_THREADS_PER_ATOM__ )
    #define NB_KER_THREADS_PER_ATOM __NBRS_THREADS_PER_ATOM__
  #else
    #define NB_KER_THREADS_PER_ATOM (16)
  #endif
  
  #if defined( __NBRS_BLOCK_SIZE__)
    #define NBRS_BLOCK_SIZE __NBRS_BLOCK_SIZE__
  #else
    #define NBRS_BLOCK_SIZE (256)
  #endif
  
  #if defined( __HB_THREADS_PER_ATOM__)
    #define HB_KER_THREADS_PER_ATOM __HB_THREADS_PER_ATOM__
  #else
    #define HB_KER_THREADS_PER_ATOM (32)
  #endif
  
  #if defined(__HB_BLOCK_SIZE__)
    #define HB_BLOCK_SIZE __HB_BLOCK_SIZE__
  #else
    #define HB_BLOCK_SIZE (256)
  #endif
  
  #if defined( __VDW_THREADS_PER_ATOM__ )
    #define VDW_KER_THREADS_PER_ATOM __VDW_THREADS_PER_ATOM__
  #else
    #define VDW_KER_THREADS_PER_ATOM (32)
  #endif
  
  #if defined( __VDW_BLOCK_SIZE__)
    #define VDW_BLOCK_SIZE __VDW_BLOCK_SIZE__
  #else
    #define VDW_BLOCK_SIZE (256)
  #endif

  /* max. num. of active CUDA streams */
  #define MAX_CUDA_STREAMS (5)

#endif

/* max. num. of CUDA events used for synchronizing streams */
#define MAX_CUDA_STREAM_EVENTS (4)


/* ensemble type */
enum ensemble
{
    /* microcanonical ensemble */
    NVE = 0,
    /* Berendsen NVT ensemble */
    bNVT = 1,
    /* Nose-Hoover NVT ensemble */
    nhNVT = 2,
    /* semi-isotropic NPT ensemble */
    sNPT = 3,
    /* isotropic NPT ensemble */
    iNPT = 4,
    /* anisotropic NPT ensemble */
    NPT = 5,
    /* total number of ensemble types */
    ens_N = 6,
};

/* interaction list mapping from type to index */
enum lists
{
    /* far neighbor list */
    FAR_NBRS = 0,
    /* bond list */
    BONDS = 1,
    /* hydrogen list */
    HBONDS = 2,
    /* 3-body list */
    THREE_BODIES = 3,
#if defined(TEST_FORCES)
    /* derivative bond order list */
    DBOS = 4,
    /* derivative delta list */
    DDELTAS = 5,
    /* total number of list types */
    LIST_N = 6,
#else
    /* total number of list types */
    LIST_N = 4,
#endif
};

/* interaction type */
enum interactions
{
    TYP_VOID = 0,
    TYP_FAR_NEIGHBOR = 1,
    TYP_BOND = 2,
    TYP_HBOND = 3,
    TYP_THREE_BODY = 4,
#if defined(TEST_FORCES)
    TYP_DBO = 5,
    TYP_DDELTA = 6,
    TYP_N = 7,
#else
    TYP_N = 5,
#endif
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
    BGF = 2,
    ASCII_RESTART = 3,
    BINARY_RESTART = 4,
    GF_N = 5,
};

/* method for computing atomic charges */
enum charge_method
{
    QEQ_CM = 0,
    EE_CM = 1,
    ACKS2_CM = 2,
};

/* linear solver type used in charge method */
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

/* preconditioner computation type for charge method linear solver */
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

/* preconditioner application type for ICHOL/ILU preconditioners,
 * used for charge method linear solver */
enum pre_app
{
    TRI_SOLVE_PA = 0,
    TRI_SOLVE_LEVEL_SCHED_PA = 1,
    TRI_SOLVE_GC_PA = 2,
    JACOBI_ITER_PA = 3,
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
    NT_NBRS = 9, // 9 through 14
};

/* atom types as pertains to hydrogen bonding */
enum hydrogen_bonding_atom_types
{
    NON_H_BONDING_ATOM = 0,
    H_ATOM = 1,
    H_BONDING_ATOM = 2,
};

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

/* trajectory file formats */
enum traj_methods
{
    /* write trajectory file using standard I/O */
    REG_TRAJ = 0,
    /* write trajectory file using MPI I/O */
    MPI_TRAJ = 1,
    /* num. trajectory file formats */
    TF_N = 2,
};


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
typedef struct bound_estimate bound_estimate;
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
typedef struct hbond_data hbond_data;
typedef struct dDelta_data dDelta_data;
typedef struct dbond_data dbond_data;
typedef struct bond_order_data bond_order_data;
typedef struct bond_data bond_data;
typedef struct sparse_matrix sparse_matrix;
typedef struct reallocate_data reallocate_data;
typedef struct storage storage;
typedef struct reax_list reax_list;
typedef struct output_controls output_controls;
typedef struct molecule molecule;
typedef struct LR_data LR_data;
typedef struct cubic_spline_coef cubic_spline_coef;
typedef struct LR_lookup_table LR_lookup_table;
typedef struct puremd_handle puremd_handle;


/* function pointer definitions */
/**/
typedef int (*evolve_function)( reax_system * const, control_params * const,
        simulation_data * const, storage * const, reax_list ** const,
        output_controls * const, mpi_datatypes * const );
/**/
typedef void (*interaction_function)( reax_system * const, control_params * const,
        simulation_data * const, storage * const, reax_list ** const,
        output_controls * const );
/**/
typedef real (*lookup_function)( real );
/* function pointer type for counting data going into egress MPI buffers */
typedef void (*message_counter)( reax_system const * const, int, int, int,
        mpi_out_data * const, int * const );
/* function pointer type for sorting data into egress MPI buffers */
typedef void (*message_sorter)( reax_system * const, int, int, int,
        mpi_out_data * const, mpi_datatypes * const );
/**/
typedef void (*unpacker)( reax_system * const, int, void * const, int,
        neighbor_proc * const, int );
/**/
typedef void (*callback_function)( reax_atom * const, simulation_data * const,
        reax_list * const );


/* struct definitions */
/* header used in restart file */
struct restart_header
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
};


/* atom type used for restarting simulation */
struct restart_atom
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
};


/* atom type used for MPI communications */
struct mpi_atom
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
};


/* atom type used for MPI communications at boundary regions */
struct boundary_atom
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
    /* atomic position, 3D */
    rvec x;
};


/**/
struct mpi_out_data
{
    /* num. of elements currently contained in the egress buffer */
    int cnt;
    /* mapping between elements of egress buffer and another buffer */
    int *index;
    /* size of index, in bytes */
    size_t index_size;
    /* egress buffer */
    void *out_atoms;
    /* size of egress buffer, in bytes */
    size_t out_atoms_size;
};


/**/
struct mpi_datatypes
{
    /* communicator for neighboring processes in 3D Cartesian mesh topology */
    MPI_Comm comm_mesh3D;
    /* MPI datatype for mpi_atom */
    MPI_Datatype mpi_atom_type;
    /* MPI datatype for boundary_atom */
    MPI_Datatype boundary_atom_type;
    /* MPI datatype for rvec */
    MPI_Datatype mpi_rvec;
    /* MPI datatype for rvec2 */
    MPI_Datatype mpi_rvec2;
    /* MPI datatype for restart_atom */
    MPI_Datatype restart_atom_type;
    /**/
    MPI_Datatype header_line;
    /**/
    MPI_Datatype init_desc_line;
    /**/
    MPI_Datatype atom_line;
    /**/
    MPI_Datatype bond_line;
    /**/
    MPI_Datatype angle_line;
    /* ingress buffer for communications with neighbor 1 along
     * one dimension of the 3D Cartesian topology */
    void *in1_buffer;
    /* size of ingress buffer, in bytes */
    size_t in1_buffer_size;
    /* ingress buffer for communications with neighbor 2 along
     * one dimension of the 3D Cartesian topology */
    void *in2_buffer;
    /* size of ingress buffer, in bytes */
    size_t in2_buffer_size;
    /* egress buffers for communications with neighbors in
     * 3D Cartesian topology */
    mpi_out_data out_buffers[MAX_NBRS];
#if defined(NEUTRAL_TERRITORY)
    /**/
    void *in_nt_buffer[MAX_NT_NBRS];
    /* size of ingress buffer, in bytes */
    size_t in_nt_buffer_size;
    /**/
    mpi_out_data out_nt_buffers[MAX_NT_NBRS];
#endif
#if defined(HAVE_CUDA)
    /* ingress buffer for communications with neighbor 1 along
     * one dimension of the 3D Cartesian topology (GPU) */
    void *d_in1_buffer;
    /* size of ingress buffer, in bytes (GPU) */
    size_t d_in1_buffer_size;
    /* ingress buffer for communications with neighbor 2 along
     * one dimension of the 3D Cartesian topology (GPU) */
    void *d_in2_buffer;
    /* size of ingress buffer, in bytes (GPU) */
    size_t d_in2_buffer_size;
    /* egress buffers for communications with neighbors in
     * 3D Cartesian topology (GPU) */
    mpi_out_data d_out_buffers[MAX_NBRS];
#endif
};


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
 * l[34] = b_s_acks2 (ACKS2 bond softness)
 * l[35] = N/A
 * l[36] = N/A
 * l[37] = N/A
 * l[38] = p_coa3
 * */
struct global_parameters
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
};


/* single body parameters in force field parameters file */
struct single_body_parameters
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
    /* bond softness for ACKS2 */
    real b_s_acks2;

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
};


/* 2-body parameters for a single interaction type,
 * from the force field parameters file */
struct two_body_parameters
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
};


/* 3-body parameters for a single interaction type,
 * from the force field parameters file */
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


/* three body interactions info. */
struct three_body_header
{
    /* num. of three body parameters */
    int cnt;
    /* collection of three body parameters, indexed by atomic types */
    three_body_parameters prm[MAX_3BODY_PARAM];
};


/* hydrogen bond parameters in force field parameters file */
struct hbond_parameters
{
    /**/
    real r0_hb;
    /**/
    real p_hb1;
    /**/
    real p_hb2;
    /**/
    real p_hb3;
};


/* 4-body parameters for a single interaction type,
 * from the force field parameters file */
struct four_body_parameters
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
};


/* four body interactions info. */
struct four_body_header
{
    /* num. of four body parameters */
    int cnt;
    /* collection of four body parameters, indexed by atomic types */
    four_body_parameters prm[MAX_4BODY_PARAM];
};


/* atomic interaction parameters */
struct reax_interaction
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

#if defined(HAVE_CUDA)
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
};


/**/
struct reax_atom
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
#if defined(NEUTRAL_TERRITORY)
    /**/
    int nt_dir;
    /**/
    int pos;
#endif
};


/* Info. regarding 3D simulation space */
struct simulation_box
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
};


/**/
struct grid_cell
{
    /* min. cell coordinates */
    rvec min;
    /* max. cell coordinates */
    rvec max;
    /* ??? */
    int mark;
    /* native or ghost cells (contains atoms only of resp. type) */
    int type;
    /* count of num. of atoms currently within this grid cell */
    int top;
    /* IDs of atoms within this grid cell */
    int* atoms;
};


/* info. for 3D domain (i.e., spatial) partitioning of atoms
 * inside a processor's portion (local and ghost regoin) of the simulation box */
struct grid
{
    /* total number of grid cells (native AND ghost) */
    int total;
    /* max. num. of atoms a grid cell can contain */
    int max_atoms;
    /**/
    int max_nbrs;
    /* num. of grid cells in each dimension of grid */
    ivec ncells;
    /* lengths of each grid cell dimension of grid */
    rvec cell_len;
    /* multiplicative inverses of lengths of each grid cell dimension in grid */
    rvec inv_len;
    /* bond interaction cutoff in terms of
     * num. of grid cells in each dimension of grid */
    ivec bond_span;
    /* non-bonded interaction cutoff in terms of
     * num. of grid cells in each dimension of grid */
    ivec nonb_span;
    /* Verlet list (i.e., neighbor list) cutoff in terms of
     * num. of grid cells in each dimension in grid */
    ivec vlist_span;
    /* num. of grid cells containing native atoms */
    ivec native_cells;
    /* starting indices for grid cells containing native atoms */
    ivec native_str;
    /* ending indices for grid cells containing native atoms */
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
    /* grid cells within the grid */
    grid_cell *cells;
    /**/
    ivec *order;
    /* starting index for atom indices recorded for a grid cell */
    int *str;
    /* ending index for atom indices recorded for a grid cell */
    int *end;
    /**/
    real *cutoff;
    /* rel. positions of cells which fall within neighbor cut-off of a given cell */
    ivec *nbrs_x;
    /* corner points of cells which fall within neighbor cut-off of a given cell */
    rvec *nbrs_cp;
    /**/
    ivec *rel_box;
};


/**/
struct neighbor_proc
{
    /* MPI rank of neighbor MPI process */
    int rank;
    /* index to beginning of atom list for atom's
     * received from this process */
    int atoms_str;
    /* num. of atoms received from this process
     * during boundary atom exchanges */
    int atoms_cnt;
    /**/
    ivec prdc;
    /**/
    rvec bndry_min;
    /**/
    rvec bndry_max;
};


/**/
struct bound_estimate
{
    /**/
    int N;
    /**/
    int exc_gcells;
    /**/
    int exc_atoms;
};


/**/
struct boundary_cutoff
{
    /**/
    real ghost_nonb;
    /**/
    real ghost_hbond;
    /**/
    real ghost_bond;
    /**/
    real ghost_cutoff;
};


/**/
struct reax_system
{
    /* atomic interaction parameters */
    reax_interaction reax_param;
    /* num. atoms (locally owned) within spatial domain of MPI process */
    int n;
    /* num. atoms (locally owned AND ghost region) within spatial domain of MPI process */
    int N;
    /* num. atoms within simulation */
    int bigN;
    /* dimension of locally owned part of sparse charge matrix */
    int n_cm;
    /* num. hydrogen atoms */
    int numH;
    /* num. hydrogen atoms (GPU) */
    int *d_numH;
    /* upper bound of the num. of local atoms owned by this process */
    int local_cap;
    /* upper bound of the num. of local and ghost atoms owned by this process */
    int total_cap;
    /* MPI rank of this process */
    int my_rank;
    /* num. of neighboring processes to this MPI process */
    int num_nbrs;
    /* coordinates of processor (according to rank) in MPI cartesian topology */
    ivec my_coords;
    /* info on neighbor MPI processors */
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
    /* max num. of far neighbors per atom */
    int *max_far_nbrs;
    /* total num. of (max) far neighbors across all atoms */
    int total_far_nbrs;
    /* current num. of far neighbors per atom (GPU) */
    int *d_far_nbrs;
    /* max num. of far neighbors per atom (GPU) */
    int *d_max_far_nbrs;
    /* total num. of (max) far neighbors across all atoms (GPU) */
    int *d_total_far_nbrs;
    /* TRUE if far neighbors list requires reallocation,
     * FALSE otherwise (GPU) */
    int *d_realloc_far_nbrs;

    /* num. bonds per atom */
    int *bonds;
    /* max. num. bonds per atom */
    int *max_bonds;
    /* total num. bonds (sum over max) */
    int total_bonds;
    /* num. bonds per atom (GPU) */
    int *d_bonds;
    /* max. num. bonds per atom (GPU) */
    int *d_max_bonds;
    /* total num. bonds (sum over max) (GPU) */
    int *d_total_bonds;
    /* TRUE if bonds list requires reallocation, FALSE otherwise (GPU) */
    int *d_realloc_bonds;

    /* num. hydrogen bonds per atom */
    int *hbonds;
    /* max. num. hydrogen bonds per atom */
    int *max_hbonds;
    /* total num. hydrogen bonds (sum over max) */
    int total_hbonds;
    /* num. hydrogen bonds per atom (GPU) */
    int *d_hbonds;
    /* max. num. hydrogen bonds per atom (GPU) */
    int *d_max_hbonds;
    /* total num. hydrogen bonds (sum over max) (GPU) */
    int *d_total_hbonds;
    /* TRUE if hydrogen bonds list requires reallocation, FALSE otherwise (GPU) */
    int *d_realloc_hbonds;

    /* num. matrix entries per row */
    int *cm_entries;
    /* max. num. matrix entries per row */
    int *max_cm_entries;
    /* total num. matrix entries (sum over max) */
    int total_cm_entries;
    /* num. matrix entries per row (GPU) */
    int *d_cm_entries;
    /* max. num. matrix entries per row (GPU) */
    int *d_max_cm_entries;
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
};


/* system control parameters */
struct control_params
{
    /* simulation name, as supplied via control file */
    char sim_name[MAX_STR];
    /* number of MPI processors, as supplied via control file */
    int nprocs;
    /* number of GPUs per node, as supplied via control file */
    int gpus_per_node;
    /* number of CUDA streams per GPU, as supplied via control file */
    int gpu_streams;
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
    /* 1 if run from a restart file, 0 otherwise */
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
    /* upper taper radius (non-bonded interaction cutoff),
     * as supplied by force field parameters, in Angstroms */
    real nonb_cut;
    /* lower taper radius, as supplied by force field parameters, in Angstroms */
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
    /* TRUE if polarization energy calculation is enabled, FALSE otherwise */
    unsigned int polarization_energy_enabled;

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
    /* function pointer for ensemble used to evolve atomic system */
    evolve_function Evolve;
    /* function pointers for bonded interactions */
    interaction_function intr_funcs[NUM_INTRS];

#if defined(HAVE_CUDA)
    /* function pointer for ensemble used to evolve atomic system (GPU) */
    evolve_function Cuda_Evolve;
    /* control parameters (GPU) */
    void *d_control_params;
    /* num. CUDA blocks for kernels with 1 thread per atom (local) */
    int blocks;
    /* num. CUDA threads per block for kernels with 1 thread per atom (local) */
    int block_size;
    /* num. of CUDA blocks rounded up to the nearest power of 2
     * for kernels with 1 thread per atom (local) */
    int blocks_pow_2;
    /* num. CUDA blocks for kernels with 1 thread per atom (local AND ghost) */
    int blocks_n;
    /* num. CUDA threads per block for kernels with 1 thread per atom (local AND ghost) */
    int block_size_n;
    /* num. of CUDA blocks rounded up to the nearest power of 2
     * for kernels with 1 thread per atom (local AND ghost) */
    int blocks_pow_2_n;
    /* CUDA streams */
    cudaStream_t streams[MAX_CUDA_STREAMS];
    /* CUDA events for synchronizing streams */
    cudaEvent_t stream_events[MAX_CUDA_STREAM_EVENTS];
#endif
};


struct thermostat
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

};


struct isotropic_barostat
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

};


struct flexible_barostat
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

};


struct reax_timing
{
    /* simulation start time */
    real start;
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
    /* charge matrix entry sorting time */
    real cm_sort;
    /* charge solver communication time */
    real cm_solver_comm;
    /* charge solver communication time for all-reduce operations */
    real cm_solver_allreduce;
    /* charge solver preconditioner computation time */
    real cm_solver_pre_comp;
    /* charge solver preconditioner application time */
    real cm_solver_pre_app;
    /* num. of steps in iterative linear solver for charge distribution */
    int cm_solver_iters;
    /* charge solver sparse matrix-dense vector multiplication (SpMV) operation time */
    real cm_solver_spmv;
    /* charge solver dense vector operation time */
    real cm_solver_vector_ops;
    /* charge solver vector orthogonalization time */
    real cm_solver_orthog;
    /* charge solver triangular system (forward/backward substitution) time */
    real cm_solver_tri_solve;
    /* time spent on last preconditioner computation */
    real cm_last_pre_comp;
    /* time lost for not refactoring */
    real cm_total_loss;
    /* solver time on last refactoring step */
    real cm_optimum;
    /* neighbor list generation time on last refactoring step */
    real last_nbrs;
    /* num. of retries in main sim. loop */
    int num_retries;
};


struct energy_data
{
    /* total energy */
    real e_tot;
    /* kinetic energy */
    real e_kin;
    /* potential energy */
    real e_pot;
    /* bond energy */
    real e_bond;
    /* over coordination */
    real e_ov;
    /* under coordination energy */
    real e_un;
    /* lone pair energy */
    real e_lp;
    /* valance angle energy */
    real e_ang;
    /* penalty energy */
    real e_pen;
    /* three body conjugation energy */
    real e_coa;
    /* hydrogen bond energy */
    real e_hb;
    /* torsional energy */
    real e_tor;
    /* four body conjugation energy */
    real e_con;
    /* van der Waals energy */
    real e_vdW;
    /* electrostatics energy */
    real e_ele;
    /* polarization energy */
    real e_pol;
};


/**/
struct simulation_data
{
    /**/
    int step;
    /**/
    int prev_steps;
    /* to decide when to compute preconditioner for dynamic refactoring */
    int refactor;
    /* last refactoring step for dynamic refactoring */
    int last_pc_step;
    /**/
    real time;
    /* total mass */
    real M;
    /* multiplicative inverse of total mass */
    real inv_M;
    /* positional center of mass */
    rvec xcm;
    /* velocity center of mass */
    rvec vcm;
    /* force center of mass */
    rvec fcm;
    /* angular momentum of center of mass */
    rvec amcm;
    /* angular velocity of center of mass */
    rvec avcm;
    /* tranlational kinetic energy of center of mass */
    real etran_cm;
    /* rotational kinetic energy of center of mass */
    real erot_cm;
    /* kinetic energy tensor */
    rtensor kinetic;
    /* hydrodynamic virial */
    rtensor virial;
    /**/
    energy_data my_en;
    /**/
    energy_data sys_en;
    /* number of degrees of freedom */
    real N_f;
    /**/
    rvec t_scale;
    /**/
    rtensor p_scale;
    /* thermostat for Nose_Hoover method */
    thermostat therm;
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
    /* struct containing timing of various simulation functions */
    reax_timing timing;

    /* struct containing timing of various simulation functions (GPU) */
    reax_timing d_timing;
    /**/
    void *d_simulation_data;
};


/**/
struct three_body_interaction_data
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
};


/* info. about a far neighbor to an atom */
struct far_neighbor_data
{
    /* atom ID of neighbor */
    int *nbr;
    /* sets of three integers which deterimine if a neighbor
     * atom is a non-periodic neighbor (all zeros) or a periodic
     * neighbor and which perioidic image this neighbor comes from */
    ivec *rel_box;
    /* distances to the neighboring atom */
    real *d;
    /* differences between positions of atom and its neighboring atom */
    rvec *dvec;
};


/**/
struct hbond_data
{
    /* neighbor atom ID */
    int nbr;
    /**/
    int scl;
    /* position of neighbor in far neighbor list */
    int ptr;
#if defined(HAVE_CUDA) && !defined(CUDA_ACCUM_ATOMIC)
    /**/
    int sym_index;
    /**/
    rvec hb_f;
#endif
};


/**/
struct dDelta_data
{
    /**/
    int wrt;
    /**/
    rvec dVal;
};


/**/
struct dbond_data
{
    /**/
    int wrt;
    /**/
    rvec dBO;
    /**/
    rvec dBOpi;
    /**/
    rvec dBOpi2;
};


/**/
struct bond_order_data
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
};


/**/
struct bond_data
{
    /* local atom ID of neighboring bonded atom */
    int nbr;
    /* index in the bonds list of neighboring atom */
    int sym_index;
    /* index in the dbond list of neighboring atom */
    int dbond_index;
    /**/
    ivec rel_box;
//  rvec ext_factor;
    /* distance to neighboring atom */
    real d;
    /* component-wise difference of coordinates of this atom
     * and its neighboring bonded atom */
    rvec dvec;
    /* bond order data */
    bond_order_data bo_data;
#if defined(HAVE_CUDA) && !defined(CUDA_ACCUM_ATOMIC)
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
#endif
};


/* Matrix in compressed row storage (CRS) format,
 * with modifications for row end pointer and max entries per row (CUDA optimizations).
 * See, e.g.,
 *   http://netlib.org/linalg/html_templates/node91.html#SECTION00931100000000000000
 */
struct sparse_matrix
{
    /* 0 if struct members are NOT allocated, 1 otherwise */
    int allocated;
    /* matrix storage format */
    int format;
    /* number of rows active for this processor */
    int n;
    /* max. number of rows active for this processor */
    int n_max;
    /* number of nonzeros (NNZ) ALLOCATED */
    int m;
#if defined(NEUTRAL_TERRITORY)
    /* max. number of entries (neural territory) */
    int NT;
#endif
    /* row start pointer (last element contains ACTUAL NNZ) */
    int *start;
    /* row end pointer */
    int *end;
    /* column indices for corresponding matrix entries */
    int *j;
    /* matrix entries */
    real *val;
};


/* used to determine if and how much space should be reallocated */
struct reallocate_data
{
    /* TRUE if far neighbor list needs
     * to be reallocated, FALSE otherwise */
    int far_nbrs;
    /* TRUE if charge matrix needs
     * to be reallocated, FALSE otherwise */
    int cm;
    /* TRUE if hbond list needs
     * to be reallocated, FALSE otherwise */
    int hbonds;
    /* TRUE if bond list needs
     * to be reallocated, FALSE otherwise */
    int bonds;
    /* TRUE if three body list needs
     * to be reallocated, FALSE otherwise */
    int thbody;
    /**/
    int gcell_atoms;
};


struct storage
{
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

    /* charge method storage */
    /* charge matrix */
    sparse_matrix H;
    /* charge matrix (full) */
    sparse_matrix H_full;
    /* sparser charge matrix */
    sparse_matrix H_sp;
    /* permuted charge matrix (graph coloring) */
    sparse_matrix H_p;
    /* sparsity pattern of charge matrix, used in
     * computing a sparse approximate inverse preconditioner */
    sparse_matrix H_spar_patt;
    /* sparsity pattern of charge matrix (full), used in
     * computing a sparse approximate inverse preconditioner */
    sparse_matrix H_spar_patt_full;
    /* sparse approximate inverse preconditioner */
    sparse_matrix H_app_inv;
    /* incomplete Cholesky or LU preconditioner */
    sparse_matrix L;
    /* incomplete Cholesky or LU preconditioner */
    sparse_matrix U;
    /* Jacobi preconditioner */
    real *Hdia_inv;
    /* row drop tolerences for incomplete Cholesky preconditioner */
    real *droptol;
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
    rvec2 *b;
    /**/
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

    /* dual GMRES, dual BiCGStab storage */
    rvec2 *g2;
    rvec2 *y2;

    /* dual GMRES, dual BiCGStab, dual PIPECG, dual PIPECR storage */
    rvec2 *z2;

    /* dual CG, dual BiCGStab, dual PIPECG, dual PIPECR storage */
    rvec2 *r2;
    rvec2 *d2;
    rvec2 *q2;
    rvec2 *p2;

    /* dual BiCGStab storage */
    rvec2 *r_hat2;
    rvec2 *q_hat2;

    /* dual PIPECG, dual PIPECR storage */
    rvec2 *m2;
    rvec2 *n2;
    rvec2 *u2;
    rvec2 *w2;

    /* Taper */
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
#if defined(TEST_FORCES)
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
    /* lookup table for force tabulation */
    LR_lookup_table *LR;
#if defined(HAVE_CUDA)
    /* temporary host workspace */
    void *host_scratch;
    /* size of temporary host workspace, in bytes */
    size_t host_scratch_size;
    /* temporary workspace (GPU) */
    void *scratch[MAX_CUDA_STREAMS];
    /* size of temporary workspace (GPU), in bytes */
    size_t scratch_size[MAX_CUDA_STREAMS];
    /* lookup table for force tabulation (GPU) */
    LR_lookup_table *d_LR;
    /* storage (GPU) */
    storage *d_workspace;
#endif
};


/* Union used for determining interaction list type */
//typedef union
//{
//    /* void type */
//    void *v;
//    /* three body type */
//    three_body_interaction_data *three_body_list;
//    /* bond type */
//    bond_data *bond_list;
//    /* derivative bond order type */
//    dbond_data *dbo_list;
//    /* derivative delta type */
//    dDelta_data *dDelta_list;
//    /* far neighbor type */
//    far_neighbor_data *far_nbr_list;
//    /* hydrogen bond type */
//    hbond_data *hbond_list;
//} list_type;


/* Interaction list */
struct reax_list
{
    /* 0 if struct members are NOT allocated, 1 otherwise */
    int allocated;
    /* total num. of entities, each of which correspond to zero or more interactions */
    int n;
    /* max. num. of interactions for which space is allocated */
    int max_intrs;
    /* beginning position for interactions corresponding to a particular entity,
     * where the entity ID used for indexing is an integer between 0 and n - 1, inclusive */
    int *index;
    /* ending position for interactions corresponding to a particular entity,
     * where the entity ID used for indexing is an integer between 0 and n - 1, inclusive */
    int *end_index;
    /* interaction list type, as defined by interactions enum above */
    int type;
    /* interaction list, made purposely non-opaque via above union to avoid typecasts */
//    list_type select;
    /* list storage format (half or full) */
    int format;
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
    far_neighbor_data far_nbr_list;
    /* hydrogen bond type */
    hbond_data *hbond_list;
};


/**/
struct output_controls
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

#if defined(TEST_ENERGY)
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

#if defined(TEST_FORCES)
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
};


/**/
struct molecule
{
    /**/
    int atom_count;
    /**/
    int atom_list[MAX_MOLECULE_SIZE];
    /**/
    int mtypes[MAX_ATOM_TYPES];
};


/**/
struct LR_data
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
};


/* coefficients for cublic spline interpolation */
struct cubic_spline_coef
{
    /**/
    real a;
    /**/
    real b;
    /**/
    real c;
    /**/
    real d;
};


/* struct containing lookup table for pairwise atomic interactions
 * based on atomic distance */
struct LR_lookup_table
{
    /* min. distance between atom pairs contained in lookup table */
    real xmin;
    /* max. distance between atom pairs contained in lookup table */
    real xmax;
    /* num. of lookup table entries */
    int n;
    /* step size between consecutive entries in the lookup table, in angstroms */
    real dx;
    /* inverse step size for the lookup table, in angstroms */
    real inv_dx;
    /**/
    real a;
    /**/
    real m;
    /**/
    real c;

    /**/
    LR_data *y;
    /* spline coefficients for charge matrix entries */
    cubic_spline_coef *H;
    /* spline coefficients for van der Waals interactions */
    cubic_spline_coef *vdW;
    /* spline coefficients for van der Waals interactions */
    cubic_spline_coef *CEvd;
    /* spline coefficients for Coulomb interactions */
    cubic_spline_coef *ele;
    /* spline coefficients for Coulomb interactions */
    cubic_spline_coef *CEclmb;
};


/* Handle for working with an instance of the PuReMD library */
struct puremd_handle
{
    /* System info. struct pointer */
    reax_system *system;
    /* System struct pointer */
    control_params *control;
    /* Control parameters struct pointer */
    simulation_data *data;
    /* Internal workspace struct pointer */
    storage *workspace;
    /* Reax interaction list struct pointer */
    reax_list **lists;
    /* Output controls struct pointer */
    output_controls *out_control;
    /* MPI datatypes struct pointer */
    mpi_datatypes *mpi_data;
    /* TRUE if file I/O for simulation output enabled, FALSE otherwise */
    int output_enabled;
    /* Callback for getting simulation state at the end of each time step */
    callback_function callback;
};

#endif
