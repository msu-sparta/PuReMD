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

/************* SOME DEFS - crucial for reax_types.h *********/

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

#define ZERO                    (0.000000000000000e+00)
#define REAX_MAX_STR            1024
#define REAX_MAX_NBRS           6
#define REAX_MAX_3BODY_PARAM    5
#define REAX_MAX_4BODY_PARAM    5
#define REAX_MAX_ATOM_TYPES     25
#define REAX_MAX_MOLECULE_SIZE  20

/* NaN IEEE 754 representation for C99 in math.h
 * Note: function choice must match REAL typedef below */
#ifdef NAN
#define IS_NAN_REAL(a) (isnan(a))
#else
#warn "No support for NaN"
#define NAN_REAL(a) (0)
#endif


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
};

/* preconditioner computation type for charge method linear solver */
enum pre_comp
{
    NONE_PC = 0,
    JACOBI_PC = 1,
    ICHOLT_PC = 2,
    ILUT__PC = 3,
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


/********************** TYPE DEFINITIONS ********************/
typedef int  ivec[3];
typedef double real;
typedef real rvec[3];
typedef real rtensor[3][3];
typedef real rvec2[2];
typedef real rvec4[4];


typedef struct
{
    int step, bigN;
    real T, xi, v_xi, v_xi_old, G_xi;
    rtensor box;
} restart_header;

typedef struct
{
    int orig_id, type;
    char name[8];
    rvec x, v;
} restart_atom;

typedef struct
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
} mpi_atom;


typedef struct
{
    int  orig_id;
    int  imprt_id;
    int  type;
    int  num_bonds;
    int  num_hbonds;
    //int  pad;
    rvec x;     // position
} boundary_atom;


typedef struct
{
    //int  ncells;
    //int *cnt_by_gcell;

    int  cnt;
    //int *block;
    int *index;
    //MPI_Datatype out_dtype;
    void *out_atoms;
} mpi_out_data;


typedef struct
{
    MPI_Comm     world;
    MPI_Comm     comm_mesh3D;

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

    //MPI_Request  send_req1[REAX_MAX_NBRS];
    //MPI_Request  send_req2[REAX_MAX_NBRS];
    //MPI_Status   send_stat1[REAX_MAX_NBRS];
    //MPI_Status   send_stat2[REAX_MAX_NBRS];
    //MPI_Status   recv_stat1[REAX_MAX_NBRS];
    //MPI_Status   recv_stat2[REAX_MAX_NBRS];

    mpi_out_data out_buffers[REAX_MAX_NBRS];
    void *in1_buffer;
    void *in2_buffer;
} mpi_datatypes;


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

typedef struct
{
    int n_global;
    real* l;
    int vdw_type;
} global_parameters;



typedef struct
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
} single_body_parameters;



/* Two Body Parameters */
typedef struct
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
    three_body_parameters prm[REAX_MAX_3BODY_PARAM];
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
    four_body_parameters prm[REAX_MAX_4BODY_PARAM];
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
} reax_atom;



typedef struct
{
    real V;
    rvec min, max, box_norms;

    rtensor box, box_inv;
    rtensor trans, trans_inv;
    rtensor g;
} simulation_box;



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

typedef struct grid_cell grid_cell;


typedef struct
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
} grid;


typedef struct
{
    int  rank;
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
} neighbor_proc;



typedef struct
{
    int N;
    int exc_gcells;
    int exc_atoms;
} bound_estimate;



typedef struct
{
    real ghost_nonb;
    real ghost_hbond;
    real ghost_bond;
    real ghost_cutoff;
} boundary_cutoff;



typedef struct
{
    reax_interaction reax_param;

    int              n, N, bigN, numH;
    int              local_cap, total_cap, gcell_cap, Hcap;
    int              est_recv, est_trans, max_recved;
    int              wsize, my_rank, num_nbrs;
    ivec             my_coords;
    neighbor_proc    my_nbrs[REAX_MAX_NBRS];
    int             *global_offset;
    simulation_box   big_box, my_box, my_ext_box;
    grid             my_grid;
    boundary_cutoff  bndry_cuts;

    reax_atom       *my_atoms;
} reax_system;



/* system control parameters */
typedef struct
{
    /* simulation name, as supplied via control file */
    char sim_name[REAX_MAX_STR];
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
    int  ignore[REAX_MAX_ATOM_TYPES];

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
    /* atomic charge distribution calculation time */
    real cm;
    /**/
    real cm_sort_mat_rows;
    /**/
    real cm_solver_comm;
    /**/
    real cm_solver_allreduce;
    /**/
    real cm_solver_pre_comp;
    /**/
    real cm_solver_pre_app; // update CG()
    /* num. of steps in iterative linear solver for charge distribution */
    int cm_solver_iters;
    /**/
    real cm_solver_spmv; // update CG()
    /**/
    real cm_solver_vector_ops; // update CG()
    /**/
    real cm_solver_orthog;
    /**/
    real cm_solver_tri_solve;
} reax_timing;


typedef struct
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
} energy_data;

typedef struct
{
    int  step;
    int  prev_steps;
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

    real               N_f;          //Number of degrees of freedom
    rvec               t_scale;
    rtensor            p_scale;
    thermostat         therm;        // Used in Nose_Hoover method
    isotropic_barostat iso_bar;
    flexible_barostat  flex_bar;
    real               inv_W;

    real kin_press;
    rvec int_press;
    rvec my_ext_press;
    rvec ext_press;
    rvec tot_press;

    reax_timing timing;
} simulation_data;


typedef struct
{
    int thb;
    int pthb; // pointer to the third body on the central atom's nbrlist
    real theta;
    real cos_theta;
    rvec dcos_di;
    rvec dcos_dj;
    rvec dcos_dk;
} three_body_interaction_data;


typedef struct
{
    int nbr;
    ivec rel_box;
    real d;
    rvec dvec;
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
} bond_order_data;

typedef struct
{
    int nbr;
    int sym_index;
    int dbond_index;
    ivec rel_box;
//    rvec ext_factor;
    real d;
    rvec dvec;
    bond_order_data bo_data;
} bond_data;


typedef struct
{
    int j;
    real val;
} sparse_matrix_entry;

typedef struct
{
    /* matrix storage format */
    int format;
    int cap, n, m;
    int *start, *end;
    sparse_matrix_entry *entries;
} sparse_matrix;


typedef struct
{
    int num_far;
    int H, Htop;
    int hbonds, num_hbonds;
    int bonds, num_bonds;
    int num_3body;
    int gcell_atoms;
} reallocate_data;


typedef struct
{
    int allocated;

    /* communication storage */
    real *tmp_dbl[REAX_MAX_NBRS];
    rvec *tmp_rvec[REAX_MAX_NBRS];
    rvec2 *tmp_rvec2[REAX_MAX_NBRS];
    int  *within_bond_box;

    /* bond order related storage */
    real *total_bond_order;
    real *Deltap, *Deltap_boc;
    real *Delta, *Delta_lp, *Delta_lp_temp, *Delta_e, *Delta_boc;
    real *dDelta_lp, *dDelta_lp_temp;
    real *nlp, *nlp_temp, *Clp, *vlpex;
    rvec *dDeltap_self;
    int *bond_mark, *done_after;

    /* QEq storage */
    sparse_matrix *H;
    sparse_matrix *L;
    sparse_matrix *U;
    sparse_matrix *H_full;
    sparse_matrix *H_spar_patt;
    sparse_matrix *H_spar_patt_full;
    sparse_matrix *H_app_inv;
    real *Hdia_inv, *b_s, *b_t, *b_prc, *b_prm, *s, *t;
    real *droptol;
    rvec2 *b, *x;

    /* GMRES storage */
    real *y, *z, *g;
    real *hc, *hs;
    real **h, **v;
    /* CG storage */
    real *r, *d, *q, *p;
    rvec2 *r2, *d2, *q2, *p2;
    /* Taper */
    real Tap[8]; //Tap7, Tap6, Tap5, Tap4, Tap3, Tap2, Tap1, Tap0;

    /* storage for analysis */
    int  *mark, *old_mark;
    rvec *x_old;

    /* storage space for bond restrictions */
    int  *restricted;
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
    //int *num_bonds;
    /* hydrogen bonds */
    //int   num_H, Hcap;
    //int  *Hindex;
    //int *num_hbonds;
    //int *hash;
    //int *rev_hash;
} storage;


typedef union
{
    //void *v;
    //three_body_interaction_data *three_body_list;
    //bond_data          *bond_list;
    //dbond_data         *dbo_list;
    //dDelta_data        *dDelta_list;
    //far_neighbor_data  *far_nbr_list;
    //hbond_data         *hbond_list;
} list_type;


typedef struct
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
    far_neighbor_data *far_nbr_list;
    hbond_data *hbond_list;
} reax_list;


typedef struct
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
    char  traj_title[81];
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
} output_controls;


typedef struct
{
    int atom_count;
    int atom_list[REAX_MAX_MOLECULE_SIZE];
    int mtypes[REAX_MAX_ATOM_TYPES];
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

    LR_data *y;
    cubic_spline_coef *H;
    cubic_spline_coef *vdW, *CEvd;
    cubic_spline_coef *ele, *CEclmb;
} LR_lookup_table;
extern LR_lookup_table **LR;

/* function pointer defs */
typedef void (*evolve_function)(reax_system*, control_params*,
                                simulation_data*, storage*, reax_list**,
                                output_controls*, mpi_datatypes* );
#if defined(PURE_REAX)
evolve_function  Evolve;
#endif

typedef void (*interaction_function) (reax_system*, control_params*,
                                      simulation_data*, storage*,
                                      reax_list**, output_controls*);

typedef void (*print_interaction)(reax_system*, control_params*,
                                  simulation_data*, storage*,
                                  reax_list**, output_controls*);

typedef real (*lookup_function)(real);

typedef void (*message_sorter) (reax_system*, int, int, int, mpi_out_data*);
typedef void (*unpacker) ( reax_system*, int, void*, int, neighbor_proc*, int );

typedef void (*dist_packer) (void*, mpi_out_data*);
typedef void (*coll_unpacker) (void*, void*, mpi_out_data*);
#endif
