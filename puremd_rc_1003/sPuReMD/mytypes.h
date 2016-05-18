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

#ifndef __MYTYPES_H_
#define __MYTYPES_H_

#include "math.h"
#include "random.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "sys/time.h"
#include "time.h"
#include "zlib.h"

#ifdef _OPENMP
#include <omp.h>
#endif

//#define DEBUG_FOCUS
//#define TEST_FORCES
//#define TEST_ENERGY
//#define REORDER_ATOMS  // turns on nbrgen opt by re-ordering atoms
//#define LGJ

#define EXP    exp
#define SQRT   sqrt
#define POW    pow
#define ACOS   acos
#define COS    cos
#define SIN    sin
#define TAN    tan

#define SQR(x)        ((x)*(x))
#define CUBE(x)       ((x)*(x)*(x))
#define DEG2RAD(a)    ((a)*PI/180.0)
#define RAD2DEG(a)    ((a)*180.0/PI)
#define MAX( x, y )   (((x) > (y)) ? (x) : (y))
#define MIN( x, y )   (((x) < (y)) ? (x) : (y))

#define PI            3.14159265
#define C_ele          332.06371
//#define K_B         503.398008   // kcal/mol/K
#define K_B             0.831687   // amu A^2 / ps^2 / K
#define F_CONV          1e6 / 48.88821291 / 48.88821291   // --> amu A / ps^2
#define E_CONV          0.002391   // amu A^2 / ps^2 --> kcal/mol
#define EV_to_KCALpMOL 14.400000   // ElectronVolt --> KCAL per MOLe
#define KCALpMOL_to_EV 23.060549   // 23.020000//KCAL per MOLe --> ElectronVolt
#define ECxA_to_DEBYE   4.803204      // elem. charge * angstrom -> debye conv
#define CAL_to_JOULES   4.184000      // CALories --> JOULES
#define JOULES_to_CAL   1/4.184000    // JOULES --> CALories
#define AMU_to_GRAM     1.6605e-24
#define ANG_to_CM       1.0e-8
#define AVOGNR          6.0221367e23
#define P_CONV          1.0e-24 * AVOGNR * JOULES_to_CAL

#define MAX_STR             100      // MAX STRing length (used for naming)
#define MAX_ATOM_ID         100000
#define MAX_RESTRICT        15
#define MAX_MOLECULE_SIZE   20
#define MAX_ATOM_TYPES      25
#define MAX_GRID            50
#define MAX_3BODY_PARAM     5
#define MAX_4BODY_PARAM     5
#define NO_OF_INTERACTIONS  10

#define MAX_dV              1.01
#define MIN_dV              0.99
#define MAX_dT              4.00
#define MIN_dT              0.00

#define MAX_ITR             10
#define RESTART             50

#define FILE_NOT_FOUND_ERR    10
#define UNKNOWN_ATOM_TYPE_ERR 11
#define CANNOT_OPEN_OUTFILE   12
#define INIT_ERR              13
#define INSUFFICIENT_SPACE    14
#define UNKNOWN_OPTION        15
#define INVALID_INPUT         16
#define NUMERIC_BREAKDOWN     17

#define C_ATOM  0
#define H_ATOM  1
#define O_ATOM  2
#define N_ATOM  3
#define S_ATOM  4
#define SI_ATOM 5
#define GE_ATOM 6
#define X_ATOM  7

#define ZERO           0.000000000000000e+00
#define ALMOST_ZERO    1e-10
#define NEG_INF       -1e10
#define NO_BOND        1e-3
#define HB_THRESHOLD   1e-2
#define MAX_BONDS      40
#define MIN_BONDS      15
#define MIN_HBONDS     50
#define SAFE_HBONDS    1.4
#define MIN_GCELL_POPL 50
#define SAFE_ZONE   1.2
#define DANGER_ZONE 0.95
#define LOOSE_ZONE  0.75


typedef double real;
typedef real rvec[3];
typedef int  ivec[3];
typedef real rtensor[3][3];

enum {NVE = 0, NVT = 1, NPT = 2, sNPT = 3, iNPT = 4, ensNR = 5, bNVT = 6};
enum {FAR_NBRS = 0, NEAR_NBRS = 1, THREE_BODIES = 2, BONDS = 3, OLD_BONDS = 4,
      HBONDS = 5, DBO = 6, DDELTA = 7, LIST_N = 8
     };
enum {TYP_VOID = 0, TYP_THREE_BODY = 1, TYP_BOND = 2, TYP_HBOND = 3, TYP_DBO = 4,
      TYP_DDELTA = 5, TYP_FAR_NEIGHBOR = 6, TYP_NEAR_NEIGHBOR = 7, TYP_N = 8
     };
enum {UNKNOWN = 0, WATER = 1};
enum {NO_ANALYSIS = 0, FRAGMENTS = 1, REACTIONS = 2, NUM_ANALYSIS = 3};
enum {WRITE_ASCII = 0, WRITE_BINARY = 1, RF_N = 2};
enum {XYZ = 0, PDB = 1, BGF = 2, ASCII_RESTART = 3, BINARY_RESTART = 4, GF_N = 5};



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
l[11] = N/A
l[12] = N/A
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

typedef struct
{
    int n_global;
    real* l;
    int vdw_type;
} global_parameters;



typedef struct
{
    /* Line one in field file */
    char name[15];                     /* Two character atom name */

    real r_s;
    real valency;                     /* Valency of the atom */
    real mass;                        /* Mass of atom */
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
    int  p_hbond; /* Determines whether this type of atom participates in H_bonds.
           It is 1 for donor H, 2 for acceptors (O,S,N), 0 for others*/

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
} single_body_parameters;



/* Two Body Parameters */
typedef struct
{
    /* Bond Order parameters */
    real p_bo1, p_bo2, p_bo3, p_bo4, p_bo5, p_bo6;
    real r_s, r_p, r_pp;  /* r_o distances in BO formula */
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
} two_body_parameters;



/* 3-body parameters */
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


typedef struct
{
    int cnt;
    three_body_parameters prm[MAX_3BODY_PARAM];
} three_body_header;



/* hydrogen-bond parameters */
typedef struct
{
    real r0_hb, p_hb1, p_hb2, p_hb3;
} hbond_parameters;



/* 4-body parameters */
typedef struct
{
    real V1, V2, V3;

    /* torsion angle */
    real p_tor1;

    /* 4-body conjugation */
    real p_cot1;
} four_body_parameters;


typedef struct
{
    int cnt;
    four_body_parameters prm[MAX_4BODY_PARAM];
} four_body_header;


typedef struct
{
    int num_atom_types;
    global_parameters gp;
    single_body_parameters *sbp;
    two_body_parameters **tbp;
    three_body_header ***thbp;
    hbond_parameters ***hbp;
    four_body_header ****fbp;
} reax_interaction;


typedef struct
{
    rvec x, v, f;        /* Position, velocity, force on atom */
    real q;              /* Charge on the atom */
    int  type;           /* Type of this atom */
    char name[5];
} reax_atom;


typedef struct
{
    real volume;

    rvec box_norms;
    rvec side_prop;
    rvec nbr_box_press[27];
    // rvec lower_end;

    rtensor box, box_inv, old_box;
    rtensor trans, trans_inv;
    rtensor g;
} simulation_box;


typedef struct
{
    int  max_atoms;
    int  max_nbrs;
    int  total;
    real cell_size;
    ivec spread;

    ivec ncell;
    rvec len;
    rvec inv_len;

    int**** atoms;
    int***  top;
    int***  mark;
    int***  start;
    int***  end;
    ivec**** nbrs;
    rvec**** nbrs_cp;
} grid;


typedef struct
{
    int N;
    reax_atom *atoms;
    reax_interaction reaxprm;
    simulation_box box;
    grid g;
} reax_system;


/* Simulation control parameters not related to the system */
typedef struct
{
    char sim_name[MAX_STR];
    char restart_from[MAX_STR];
    int  restart;
    int  random_vel;

    int  reposition_atoms;

    /* ensemble values:
       0 : NVE
       1 : NVT  (Nose-Hoover)
       2 : NPT  (Parrinello-Rehman-Nose-Hoover) Anisotropic
       3 : sNPT (Parrinello-Rehman-Nose-Hoover) semiisotropic
       4 : iNPT (Parrinello-Rehman-Nose-Hoover) isotropic */
    int  ensemble;
    int  nsteps;
    int  periodic_boundaries;
    int  restrict_bonds;
    int  tabulate;
    ivec periodic_images;
    real dt;

    int reneighbor;
    real vlist_cut;
    real nbr_cut;
    real r_cut, r_sp_cut, r_low; // upper and lower taper
    real bo_cut;
    real thb_cut;
    real hb_cut;
    real Tap7, Tap6, Tap5, Tap4, Tap3, Tap2, Tap1, Tap0;
    real q_err;
    int  max_far_nbrs;

    real T_init, T_final, T;
    real Tau_T;
    int  T_mode;
    real T_rate, T_freq;

    real Tau_PT;
    rvec P, Tau_P;
    int  press_mode;
    real compressibility;

    int remove_CoM_vel;

    int geo_format;

    int dipole_anal;
    int freq_dipole_anal;

    int diffusion_coef;
    int freq_diffusion_coef;
    int restrict_type;

    int refactor;
    real droptol;

    int molec_anal;
    int freq_molec_anal;
    real bg_cut;
    int num_ignored;
    int  ignore[MAX_ATOM_TYPES];
} control_params;


typedef struct
{
    real T;
    real xi;
    real v_xi;
    real v_xi_old;
    real G_xi;
} thermostat;


typedef struct
{
    real P;
    real eps;
    real v_eps;
    real v_eps_old;
    real a_eps;

} isotropic_barostat;


typedef struct
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

} flexible_barostat;


typedef struct
{
    real start;
    real end;
    real elapsed;

    real total;
    real nbrs;
    real init_forces;
    real bonded;
    real nonb;
    real QEq;
    int matvecs;
    real pre_comp;
    real pre_app;
    real spmv;
} reax_timing;


typedef struct
{
    int step;
    int prev_steps;
    real time;

    real M;              /* Total Mass */
    real inv_M;                      /* 1 / Total Mass */

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

    real N_f;                        /*Number of degrees of freedom */
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
} simulation_data;


typedef struct
{
    int thb;
    int pthb; /* pointer to the third body on the central atom's nbrlist */
    real theta, cos_theta;
    rvec dcos_di, dcos_dj, dcos_dk;
} three_body_interaction_data;


typedef struct
{
    int nbr;
    ivec rel_box;
    //  rvec ext_factor;
    real d;
    rvec dvec;
} near_neighbor_data;


typedef struct
{
    int nbr;
    ivec rel_box;
    //  rvec ext_factor;
    real d;
    rvec dvec;
    // real H; //, Tap, inv_dr3gamij_1, inv_dr3gamij_3;
} far_neighbor_data;


typedef struct
{
    int nbr;
    int scl;
    far_neighbor_data *ptr;
} hbond_data;


typedef struct
{
    int wrt;
    rvec dVal;
} dDelta_data;


typedef struct
{
    int wrt;
    rvec dBO, dBOpi, dBOpi2;
} dbond_data;

typedef struct
{
    real BO, BO_s, BO_pi, BO_pi2;
    real Cdbo, Cdbopi, Cdbopi2;
    real C1dbo, C2dbo, C3dbo;
    real C1dbopi, C2dbopi, C3dbopi, C4dbopi;
    real C1dbopi2, C2dbopi2, C3dbopi2, C4dbopi2;
    rvec dBOp, dln_BOp_s, dln_BOp_pi, dln_BOp_pi2;
} bond_order_data;

typedef struct
{
    int nbr;
    int sym_index;
    int dbond_index;
    ivec rel_box;
    //  rvec ext_factor;
    real d;
    rvec dvec;
    bond_order_data bo_data;
} bond_data;


/* compressed row storage (crs) format
 *   j: column index for corresponding matrix entry
 *   val: matrix entry
 * */
// TODO: move pointers into below struct (question about addressing performance)
// Consider: __attribute__((packed)) or similar in structure definition
//   to avoid having the compiler add padding bytes (manual alignment along word boundary)
// Warning: padding is implementation (compiler) dependent, so not portable,
//   and also potentially dangerous
typedef struct
{
    int j;
    real val;
} sparse_matrix_entry;

/* compressed row storage (crs) format
 * See, e.g.,
 *   http://netlib.org/linalg/html_templates/node91.html#SECTION00931100000000000000
 *
 *   m: number of nonzeros (NNZ) ALLOCATED
 *   n: number of rows
 *   start: row pointer (last element contains ACTUAL NNZ)
 *   entries: (column index, val) pairs
 * */
typedef struct
{
    int n, m;
    int *start;
    sparse_matrix_entry *entries;
//  int *j;
//  real *val;
} sparse_matrix;


typedef struct
{
    int num_far;
    int Htop;
    int hbonds;
    int num_hbonds;
    int bonds;
    int num_bonds;
    int num_3body;
    int gcell_atoms;
} reallocate_data;

typedef struct
{
    /* bond order related storage */
    real *total_bond_order;
    real *Deltap, *Deltap_boc;
    real *Delta, *Delta_lp, *Delta_lp_temp, *Delta_e, *Delta_boc;
    real *dDelta_lp, *dDelta_lp_temp;
    real *nlp, *nlp_temp, *Clp, *vlpex;
    rvec *dDeltap_self;

    /* QEq storage */
    sparse_matrix *H, *H_sp, *L, *U;
    real *droptol;
    real *w;
    real *Hdia_inv;
    real *b, *b_s, *b_t, *b_prc, *b_prm;
    real **s, **t;
    real *s_t; //, *s_old, *t_old, *s_oldest, *t_oldest;

    /* GMRES related storage */
    real *y, *z, *g;
    real *hc, *hs;
    real **h, **rn, **v;
    /* CG related storage */
    real *r, *d, *q, *p;
    int   s_dims, t_dims;

    int num_H;
    int *hbond_index; // for hydrogen bonds

    rvec *v_const, *f_old, *a; // used in integrators

    real *CdDelta;  // coefficient of dDelta for force calculations

    int *mark, *old_mark;  // storage for analysis
    rvec *x_old;

    /* storage space for bond restrictions */
    int  *map_serials;
    int  *orig_id;
    int  *restricted;
    int **restricted_list;

    reallocate_data realloc;

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
    rvec *dDelta;       /* Calculated on the fly in bond_orders.c */
#endif
} static_storage;


typedef struct
{
    int n;
    int num_intrs;
    int *index;
    int *end_index;
    int type;
    union
    {
        void *v;
        three_body_interaction_data *three_body_list;
        bond_data *bond_list;
        dbond_data *dbo_list;
        dDelta_data *dDelta_list;
        far_neighbor_data *far_nbr_list;
        near_neighbor_data *near_nbr_list;
        hbond_data *hbond_list;
    } select;
} list;


typedef struct
{
    FILE *trj;
    FILE *out;
    FILE *pot;
    FILE *log;
    FILE *mol, *ign;
    FILE *dpl;
    FILE *drft;
    FILE *pdb;
    FILE *prs;

    int  write_steps;
    int  traj_compress;
    int  traj_format;
    char traj_title[81];
    int  atom_format;
    int  bond_info;
    int  angle_info;

    int  restart_format;
    int  restart_freq;
    int  debug_level;
    int  energy_update_freq;

    // trajectory output functions
    int (* write_header)( reax_system*, control_params*, static_storage*, void* );
    int (* append_traj_frame)(reax_system*, control_params*,
                              simulation_data*, static_storage*, list **, void* );
    int (* write)( FILE *, const char *, ... );

#ifdef TEST_ENERGY
    FILE *ebond;
    FILE *elp, *eov, *eun;
    FILE *eval, *epen, *ecoa;
    FILE *ehb;
    FILE *etor, *econ;
    FILE *evdw, *ecou;
#endif

    FILE *ftot;
#ifdef TEST_FORCES
    FILE *fbo, *fdbo;
    FILE *fbond;
    FILE *flp, *fatom;
    FILE *f3body;
    FILE *fhb;
    FILE *f4body;
    FILE *fnonb;
    FILE *ftot2;
#endif
} output_controls;


typedef struct
{
    int atom_count;
    int atom_list[MAX_MOLECULE_SIZE];
    int mtypes[MAX_ATOM_TYPES];
} molecule;


typedef struct
{
    real H;
    real e_vdW, CEvd;
    real e_ele, CEclmb;
} LR_data;



typedef struct
{
    real a, b, c, d;
} cubic_spline_coef;

typedef struct
{
    real xmin, xmax;
    int n;
    real dx, inv_dx;
    real a;

    real m;
    real c;

    real *y;
} lookup_table;


typedef struct
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
} LR_lookup_table;


typedef void (*interaction_function)(reax_system*, control_params*,
                                     simulation_data*, static_storage*,
                                     list**, output_controls*);
interaction_function Interaction_Functions[NO_OF_INTERACTIONS];

typedef void (*evolve_function)(reax_system*, control_params*,
                                simulation_data*, static_storage*,
                                list**, output_controls*);

typedef real (*lookup_function)(real);
lookup_table Exp, Sqrt, Cube_Root, Four_Third_Root, Cos, Sin, ACos;
LR_lookup_table **LR;


typedef void (*get_far_neighbors_function)(rvec, rvec, simulation_box*,
        control_params*, far_neighbor_data*,
        int*);
#endif
