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

#ifndef __REAX_TYPES_H_
#define __REAX_TYPES_H_

#if (defined(HAVE_CONFIG_H) && !defined(__CONFIG_H_))
  #define __CONFIG_H_
  #include "config.h"
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

/* enables debugging code */
//#define DEBUG_FOCUS
/* enables test forces code */
//#define TEST_FORCES
/* enables test energy code */
//#define TEST_ENERGY
/* enables reordering atoms after neighbor list generation (optimization) */
#define REORDER_ATOMS
/* enables support for small simulation boxes (i.e. a simulation box with any
 * dimension less than twice the Verlet list cutoff distance, vlist_cut) */
#define SMALL_BOX_SUPPORT

#define SUCCESS (1)
#define FAILURE (0)
#define TRUE (1)
#define FALSE (0)

/* transcendental constant pi */
#if defined(M_PI)
  /* GNU C library (libc), defined in math.h */
  #define PI (M_PI)
#else
  #define PI (3.14159265)
#endif
#define C_ele (332.06371)
//#define K_B (503.398008)   // kcal/mol/K
#define K_B (0.831687)   // amu A^2 / ps^2 / K
#define F_CONV (1e6 / 48.88821291 / 48.88821291)   // --> amu A / ps^2
#define E_CONV (0.002391)   // amu A^2 / ps^2 --> kcal/mol
#define EV_to_KCALpMOL (14.400000)   // ElectronVolt --> KCAL per MOLe
#define KCALpMOL_to_EV (23.060549)   // 23.020000//KCAL per MOLe --> ElectronVolt
#define ECxA_to_DEBYE (4.803204)      // elem. charge * angstrom -> debye conv
#define CAL_to_JOULES (4.184000)      // CALories --> JOULES
#define JOULES_to_CAL (1.0 / 4.184000)    // JOULES --> CALories
#define AMU_to_GRAM (1.6605e-24)
#define ANG_to_CM (1.0e-8)
#define AVOGNR (6.0221367e23)
#define P_CONV (1.0e-24 * AVOGNR * JOULES_to_CAL)

#define MAX_STR (1024)
#define MAX_LINE (1024)
#define MAX_TOKENS (1024)
#define MAX_TOKEN_LEN (1024)

#define MAX_ATOM_ID (100000)
#define MAX_RESTRICT (15)
#define MAX_MOLECULE_SIZE (20)
#define MAX_ATOM_TYPES (25)

#define MAX_GRID (50)
#define MAX_3BODY_PARAM (5)
#define MAX_4BODY_PARAM (5)
#define NO_OF_INTERACTIONS (10)

#define MAX_dV (1.01)
#define MIN_dV (0.99)
#define MAX_dT (4.00)
#define MIN_dT (0.00)

#define ZERO (0.000000000000000e+00)
#define ALMOST_ZERO (1e-10)
#define NEG_INF (-1e10)
#define NO_BOND (1e-3)
#define HB_THRESHOLD (1e-2)
#define MAX_BONDS (40)
#define MIN_BONDS (15)
#define MIN_HBONDS (50)
#define SAFE_HBONDS (1.4)
#define MIN_GCELL_POPL (50)
#define SAFE_ZONE (1.2)
#define DANGER_ZONE (0.95)
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


/* ensemble type */
enum ensemble
{
    NVE = 0,
    bNVT = 1,
    nhNVT = 2,
    sNPT = 3,
    iNPT = 4,
    aNPT = 5,
    ens_N = 6,
};

/* interaction list type */
enum interaction_list_offets
{
    FAR_NBRS = 0,
    NEAR_NBRS = 1,
    THREE_BODIES = 2,
    BONDS = 3,
    OLD_BONDS = 4,
    HBONDS = 5,
    DBO = 6,
    DDELTA = 7,
    LIST_N = 8,
};

/* interaction type */
enum interaction_type
{
    TYP_VOID = 0,
    TYP_THREE_BODY = 1,
    TYP_BOND = 2,
    TYP_HBOND = 3,
    TYP_DBO = 4,
    TYP_DDELTA = 5,
    TYP_FAR_NEIGHBOR = 6,
    TYP_NEAR_NEIGHBOR = 7,
    TYP_N = 8,
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

enum molecular_analysis_type
{
    NO_ANALYSIS = 0,
    FRAGMENTS = 1,
    REACTIONS = 2,
    NUM_ANALYSIS = 3,
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
enum hydrogen_bonding_atom_types
{
    NON_H_BONDING_ATOM = 0,
    H_ATOM = 1,
    H_BONDING_ATOM = 2,
};


typedef double real;
typedef real rvec[3];
typedef int ivec[3];
typedef real rtensor[3][3];


/* struct declarations, see definitions below for comments */
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
typedef struct grid grid;
typedef struct reax_system reax_system;
typedef struct control_params control_params;
typedef struct thermostat thermostat;
typedef struct isotropic_barostat isotropic_barostat;
typedef struct flexible_barostat flexible_barostat;
typedef struct reax_timing reax_timing;
typedef struct simulation_data simulation_data;
typedef struct three_body_interaction_data three_body_interaction_data;
typedef struct near_neighbor_data near_neighbor_data;
typedef struct far_neighbor_data far_neighbor_data;
typedef struct hbond_data hbond_data;
typedef struct dDelta_data dDelta_data;
typedef struct dbond_data dbond_data;
typedef struct bond_order_data bond_order_data;
typedef struct bond_data bond_data;
typedef struct sparse_matrix sparse_matrix;
typedef struct reallocate_data reallocate_data;
typedef struct LR_data LR_data;
typedef struct cubic_spline_coef cubic_spline_coef;
typedef struct LR_lookup_table LR_lookup_table;
typedef struct static_storage static_storage;
typedef struct reax_list reax_list;
typedef struct output_controls output_controls;
typedef struct spuremd_handle spuremd_handle;


/* function pointer definitions */
/**/
typedef void (*interaction_function)( reax_system*, control_params*,
        simulation_data*, static_storage*, reax_list**, output_controls* );
/**/
typedef void (*evolve_function)( reax_system*, control_params*,
        simulation_data*, static_storage*,
        reax_list**, output_controls* );
/**/
typedef void (*callback_function)( reax_atom*, simulation_data*, reax_list** );


/* struct definitions */
/* Force field global parameters mapping
 * (contained in section 1 of file):
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
 * l[11] = N/A
 * l[12] = N/A
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
 * l[37] = version number
 * l[38] = p_coa3 */
struct global_parameters
{
    int n_global;
    real* l;
    int vdw_type;
};


struct single_body_parameters
{
    /* Line one in field file */
    /* Two character atom name */
    char name[15];

    real r_s;
    /* Valency of the atom */
    real valency;
    /* Mass of atom */
    real mass;
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
    /* Determines whether this type of atom participates in H_bonds:
     * 1 for H donor, 2 for acceptors (O,S,N), 0 for others */
    int p_hbond;

    /* Line three in field file */
    real r_pi_pi;
    real p_lp2;
    real b_o_131;
    real b_o_132;
    real b_o_133;
    /* bond softness for ACKS2 */
    real b_s_acks2;

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
    real p_bo1;
    real p_bo2;
    real p_bo3;
    real p_bo4;
    real p_bo5;
    real p_bo6;
    real r_s;
    real r_p;
    /* r_o distances in BO formula */
    real r_pp;
    real p_boc3;
    real p_boc4;
    real p_boc5;

    /* Bond Energy parameters */
    real p_be1;
    real p_be2;
    real De_s;
    real De_p;
    real De_pp;

    /* Over/Under coordination parameters */
    real p_ovun1;

    /* Van der Waal interaction parameters */
    real D;
    real alpha;
    real r_vdW;
    real gamma_w;
    real rcore;
    real ecore;
    real acore;

    /* electrostatic parameters */
    /* note: this parameter is gamma^-3 and not gamma */
    real gamma;

    real v13cor;
    real ovc;
};


/* 3-body parameters */
struct three_body_parameters
{
    /* valence angle */
    real theta_00;
    real p_val1;
    real p_val2;
    real p_val4;
    real p_val7;

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
    real r0_hb;
    real p_hb1;
    real p_hb2;
    real p_hb3;
};


/* 4-body parameters */
struct four_body_parameters
{
    real V1;
    real V2;
    real V3;

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
    /* Type of this atom */
    int type;
    /**/
    char name[8];
    /* position */
    rvec x;
    /* velocity */
    rvec v;
    /* force */
    rvec f;
    /* Charge on the atom */
    real q;
};


struct simulation_box
{
    real volume;

    rvec box_norms;
    rvec side_prop;
    rvec nbr_box_press[27];

    rtensor box;
    rtensor box_inv;
    rtensor old_box;
    rtensor trans;
    rtensor trans_inv;
    rtensor g;
};


struct grid
{
    int max_atoms;
    int max_nbrs;
    int total;
    real cell_size;
    ivec spread;

    ivec ncell;
    rvec len;
    rvec inv_len;

    int **** atoms;
    int *** top;
    int *** mark;
    int *** start;
    int *** end;
    ivec **** nbrs;
    rvec **** nbrs_cp;
};


struct reax_system
{
    /* number of atoms */
    int N;
    /* dimension of the N x N sparse charge method matrix H */
    int N_cm;
    /* atom info */
    reax_atom *atoms;
    /* atomic interaction parameters */
    reax_interaction reaxprm;
    /* simulation space (a.k.a. box) parameters */
    simulation_box box;
    /* grid structure used for binning atoms and tracking neighboring bins */
    grid g;
};


/* Simulation control parameters not related to the system */
struct control_params
{
    char sim_name[MAX_STR];
    char restart_from[MAX_STR];
    int restart;
    int random_vel;

    int reposition_atoms;

    /* ensemble values:
     * 0 : NVE
     * 1 : NVT  (Berendsen)
     * 2 : NVT  (Nose-Hoover)
     * 3 : sNPT (Parrinello-Rehman-Nose-Hoover) semi-isotropic
     * 4 : iNPT (Parrinello-Rehman-Nose-Hoover) isotropic
     * 5 : aNPT  (Parrinello-Rehman-Nose-Hoover) anisotropic */
    int ensemble;
    int nsteps;
    int periodic_boundaries;
    int restrict_bonds;
    int tabulate;
    real dt;

    int reneighbor;
    real vlist_cut;
    real nbr_cut;
    /* upper and lower taper */
    real r_cut;
    real r_sp_cut;
    real r_low;
    real bo_cut;
    real thb_cut;
    real hb_cut;

    real T_init;
    real T_final;
    real T;
    real Tau_T;
    int T_mode;
    real T_rate;
    real T_freq;

    real Tau_PT;
    rvec P;
    rvec Tau_P;
    int press_mode;
    real compressibility;

    int remove_CoM_vel;

    int geo_format;

    int dipole_anal;
    int freq_dipole_anal;

    int diffusion_coef;
    int freq_diffusion_coef;
    int restrict_type;

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
    /**/
    int molec_anal;
    /**/
    int freq_molec_anal;
    /**/
    real bg_cut;
    /**/
    int num_ignored;
    /**/
    int ignore[MAX_ATOM_TYPES];
    /**/
    int num_threads;
    /* function pointers for bonded interactions */
    interaction_function intr_funcs[NO_OF_INTERACTIONS];
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
    /* neighbor list generation time */
    real nbrs;
    /* force initialization time */
    real init_forces;
    /* bonded force calculation time */
    real bonded;
    /* non-bonded force calculation time */
    real nonb;
    /* atomic charge distribution calculation time */
    real cm;
    /**/
    real cm_sort_mat_rows;
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
    /* num. of retries in main sim. loop */
    int num_retries;
};


struct simulation_data
{
    int step;
    int prev_steps;
    real time;

    /* Total Mass */
    real M;
    /* 1.0 / Total Mass */
    real inv_M;

    rvec xcm;                        /* Center of mass */
    rvec vcm;                        /* Center of mass velocity */
    rvec fcm;                        /* Center of mass force */
    rvec amcm;                       /* Angular momentum of CoM */
    rvec avcm;                       /* Angular velocity of CoM */
    real etran_cm;                  /* Translational kinetic energy of CoM */
    real erot_cm;                    /* Rotational kinetic energy of CoM */

    rtensor kinetic;                 /* Kinetic energy tensor */
    rtensor virial;                  /* Hydrodynamic virial */

    real E_Tot;
    real E_Kin;                      /* Total kinetic energy */
    real E_Pot;

    real E_BE;                       /* Total bond energy */
    real E_Ov;                       /* Total over coordination */
    real E_Un;                       /* Total under coordination energy */
    real E_Lp;                       /* Total under coordination energy */
    real E_Ang;                      /* Total valance angle energy */
    real E_Pen;                      /* Total penalty energy */
    real E_Coa;                      /* Total three body conjgation energy */
    real E_HB;                       /* Total Hydrogen bond energy */
    real E_Tor;                      /* Total torsional energy */
    real E_Con;                      /* Total four body conjugation energy */
    real E_vdW;                      /* Total van der Waals energy */
    real E_Ele;                      /* Total electrostatics energy */
    real E_Pol;                      /* Polarization energy */

    real N_f;                        /* Number of degrees of freedom */
    rvec t_scale;
    rtensor p_scale;
    thermostat therm;                /* Used in Nose_Hoover method */
    isotropic_barostat iso_bar;
    flexible_barostat flex_bar;
    real inv_W;

    rvec int_press;
    rvec ext_press;
    real kin_press;
    rvec tot_press;

    reax_timing timing;
};


struct three_body_interaction_data
{
    int thb;
    /* pointer to the third body on the central atom's nbrlist */
    int pthb;
    real theta;
    real cos_theta;
    rvec dcos_di;
    rvec dcos_dj;
    rvec dcos_dk;
};


struct near_neighbor_data
{
    int nbr;
    ivec rel_box;
//    rvec ext_factor;
    real d;
    rvec dvec;
};


struct far_neighbor_data
{
    int nbr;
    ivec rel_box;
//    rvec ext_factor;
    real d;
    rvec dvec;
};


struct hbond_data
{
    int nbr;
    int scl;
    far_neighbor_data *ptr;
};


struct dDelta_data
{
    int wrt;
    rvec dVal;
};


struct dbond_data
{
    int wrt;
    rvec dBO;
    rvec dBOpi;
    rvec dBOpi2;
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


/* compressed row storage (crs) format
 * See, e.g.,
 *   http://netlib.org/linalg/html_templates/node91.html#SECTION00931100000000000000
 *
 *   m: number of nonzeros (NNZ) ALLOCATED
 *   n: number of rows
 *   start: row pointer (last element contains ACTUAL NNZ)
 *   j: column index for corresponding matrix entry
 *   val: matrix entry */
struct sparse_matrix
{
    unsigned int n;
    unsigned int m;
    unsigned int *start;
    unsigned int *j;
    real *val;
};


struct reallocate_data
{
    int num_far;
    int Htop;
    int hbonds;
    int num_hbonds;
    int bonds;
    int num_bonds;
    int num_3body;
    int gcell_atoms;
};


struct LR_data
{
    real H;
    real e_vdW;
    real CEvd;
    real e_ele;
    real CEclmb;
};


struct cubic_spline_coef
{
    real a;
    real b;
    real c;
    real d;
};


struct LR_lookup_table
{
    real xmin;
    real xmax;
    int n;
    real dx;
    real inv_dx;
    real a;
    real m;
    real c;

    LR_data *y;
    cubic_spline_coef *H;
    cubic_spline_coef *vdW;
    cubic_spline_coef *CEvd;
    cubic_spline_coef *ele;
    cubic_spline_coef *CEclmb;
};


struct static_storage
{
    /* bond order related storage */
    real *total_bond_order;
    real *Deltap;
    real *Deltap_boc;
    real *Delta;
    real *Delta_lp;
    real *Delta_lp_temp;
    real *Delta_e;
    real *Delta_boc;
    real *dDelta_lp;
    real *dDelta_lp_temp;
    real *nlp;
    real *nlp_temp;
    real *Clp;
    real *vlpex;
    rvec *dDeltap_self;

    /* charge method storage */
    /* charge matrix */
    sparse_matrix *H;
    /* charge matrix (full) */
    sparse_matrix *H_full;
    /* sparser charge matrix */
    sparse_matrix *H_sp;
    /* permuted charge matrix (graph coloring) */
    sparse_matrix *H_p;
    /* sparsity pattern of charge matrix, used in
     * computing a sparse approximate inverse preconditioner */
    sparse_matrix *H_spar_patt;
    /* sparsity pattern of charge matrix (full), used in
     * computing a sparse approximate inverse preconditioner */
    sparse_matrix *H_spar_patt_full;
    /* sparse approximate inverse preconditioner */
    sparse_matrix *H_app_inv;
    /* incomplete Cholesky or LU preconditioner */
    sparse_matrix *L;
    /* incomplete Cholesky or LU preconditioner */
    sparse_matrix *U;
    /* Jacobi preconditioner */
    real *Hdia_inv;
    /* row drop tolerences for incomplete Cholesky preconditioner */
    real *droptol;
    real *b;
    real *b_s;
    real *b_t;
    real *b_prc;
    real *b_prm;
    real **s;
    real **t;

    /* GMRES related storage */
    real *y;
    real *z;
    real *g;
    real *hc;
    real *hs;
    real **h;
    real **rn;
    real **v;

    /* CG, SDM, BiCGStab related storage */
    real *r;
    real *r_hat;
    real *d;
    real *q;
    real *p;

    /* SpMV related storage */
#ifdef _OPENMP
    real *b_local;
#endif

    /* Level scheduling related storage for applying, e.g. ICHOLT and ILU(T),
     * preconditioners */
    int levels_L;
    int levels_U;
    unsigned int *row_levels_L;
    unsigned int *level_rows_L;
    unsigned int *level_rows_cnt_L;
    unsigned int *row_levels_U;
    unsigned int *level_rows_U;
    unsigned int *level_rows_cnt_U;
    unsigned int *top;

    /* Graph coloring related storage for applying, e.g. ICHOLT and ILU(T),
     * preconditioners */
    unsigned int *color;
    unsigned int *to_color;
    unsigned int *conflict;
    unsigned int *conflict_cnt;
    unsigned int *recolor;
    unsigned int recolor_cnt;
    unsigned int *color_top;
    unsigned int *permuted_row_col;
    unsigned int *permuted_row_col_inv;
    real *y_p;
    real *x_p;

    /* Jacobi iteration related storage for applying, e.g. ICHOLT and ILU(T),
     * preconditioners */
    real *Dinv_L;
    real *Dinv_U;
    real *Dinv_b;
    real *rp;
    real *rp2;

    int num_H;
    int *hbond_index; // for hydrogen bonds

    rvec *v_const;
    rvec *f_old;
    rvec *a; // used in integrators

    real *CdDelta;  // coefficient of dDelta for force calculations

    /* Taper */
    real Tap[8];

    int *mark;
    int *old_mark;  // storage for analysis
    rvec *x_old;

    /* storage space for bond restrictions */
    int *map_serials;
    int *orig_id;
    int *restricted;
    int **restricted_list;

#ifdef _OPENMP
    /* local forces per thread */
    rvec *f_local;
#endif
    unsigned int temp_int_omp;
    real temp_real_omp;

    reallocate_data realloc;

    LR_lookup_table **LR;

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
    /* Calculated on the fly in bond_orders.c */
    rvec *dDelta;
#endif
};


/* interaction lists */
struct reax_list
{
    /* num. entries in list */
    int n;
    /* sum of max. interactions per atom */
    int total_intrs;
    /* starting position of atom's interactions */
    int *index;
    /* ending position of atom's interactions */
    int *end_index;
    /* max. num. of interactions per atom */
    int *max_intrs;
    /* interaction list (polymorphic via union dereference) */
    union
    {
        /* typeless interaction list */
        void *v;
        /* three body interaction list */
        three_body_interaction_data *three_body_list;
        /* bond interaction list */
        bond_data *bond_list;
        /* bond interaction list */
        dbond_data *dbo_list;
        /* test forces interaction list */
        dDelta_data *dDelta_list;
        /* far neighbor interaction list */
        far_neighbor_data *far_nbr_list;
        /* near neighbor interaction list */
        near_neighbor_data *near_nbr_list;
        /* hydrogen bond interaction list */
        hbond_data *hbond_list;
    } select;
};


struct output_controls
{
    FILE *trj;
    FILE *out;
    FILE *pot;
    FILE *log;
    FILE *mol;
    FILE *ign;
    FILE *dpl;
    FILE *drft;
    FILE *pdb;
    FILE *prs;

    int write_steps;
    int traj_compress;
    int traj_format;
    char traj_title[81];
    int atom_format;
    int bond_info;
    int angle_info;

    int restart_format;
    int restart_freq;
    int debug_level;
    int energy_update_freq;

    /* trajectory output function pointer definitions */
    int (* write_header)( reax_system*, control_params*, static_storage*, void* );
    int (* append_traj_frame)( reax_system*, control_params*,
            simulation_data*, static_storage*, reax_list **, void* );
    int (* write)( FILE *, const char *, ... );

#ifdef TEST_ENERGY
    FILE *ebond;
    FILE *elp;
    FILE *eov;
    FILE *eun;
    FILE *eval;
    FILE *epen;
    FILE *ecoa;
    FILE *ehb;
    FILE *etor;
    FILE *econ;
    FILE *evdw;
    FILE *ecou;
#endif

#ifdef TEST_FORCES
    FILE *fbo;
    FILE *fdbo;
    FILE *fbond;
    FILE *flp;
    FILE *fatom;
    FILE *f3body;
    FILE *fhb;
    FILE *f4body;
    FILE *fnonb;
    FILE *ftot;
    FILE *ftot2;
#endif
};


/* Handle for working with an instance of the sPuReMD library */
struct spuremd_handle
{
    /* System info. struct pointer */
    reax_system *system;
    /* System struct pointer */
    control_params *control;
    /* Control parameters struct pointer */
    simulation_data *data;
    /* Internal workspace struct pointer */
    static_storage *workspace;
    /* Reax interaction list struct pointer */
    reax_list **lists;
    /* Output controls struct pointer */
    output_controls *out_control;
    /* TRUE if file I/O for simulation output enabled, FALSE otherwise */
    int output_enabled;
    /* Callback for getting simulation state at the end of each time step */
    callback_function callback;
};


#endif