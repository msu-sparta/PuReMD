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
#define MAX(x,y)      (((x) > (y)) ? (x) : (y))
#define MIN(x,y)      (((x) < (y)) ? (x) : (y))
#define MAX3(x,y,z)   MAX( MAX(x,y), z)

#define PI            3.14159265
#define C_ele          332.06371
//#define K_B         503.398008   // kcal/mol/K
#define K_B             0.831687   // amu A^2 / ps^2 / K
#define F_CONV          1e6 / 48.88821291 / 48.88821291   // --> amu A / ps^2
#define E_CONV          0.002391   // amu A^2 / ps^2 --> kcal/mol
#define EV_to_KCALpMOL 14.400000   // ElectronVolt --> KCAL per MOLe
#define KCALpMOL_to_EV 23.060549   // 23.020000 //KCAL per MOLe --> ElectronVolt
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
#define MYSELF   13  // encoding of relative coordinate (0,0,0)

#define MAX_ITR 10
#define RESTART 30

/**************** RESOURCE CONSTANTS **********************/
#ifdef HAVE_CUDA
//#define			CUDA_BLOCK_SIZE				256
#define			SCRATCH_SIZE					(1024 * 1024 * 20)
#define			HOST_SCRATCH_SIZE				(1024 * 1024 * 20)
#define			RES_SCRATCH						0x90

/* BLOCK SIZES for kernels */
#define				HB_SYM_BLOCK_SIZE					64
#define				HB_KER_SYM_THREADS_PER_ATOM			16
#define				HB_POST_PROC_BLOCK_SIZE				256
#define				HB_POST_PROC_KER_THREADS_PER_ATOM	32

#if defined( __INIT_BLOCK_SIZE__)
#define				DEF_BLOCK_SIZE						__INIT_BLOCK_SIZE__    /* all utility functions and all */
#define				CUDA_BLOCK_SIZE						__INIT_BLOCK_SIZE__    	/* init forces */
#define				ST_BLOCK_SIZE						__INIT_BLOCK_SIZE__    
#else
#define				DEF_BLOCK_SIZE						256						/* all utility functions and all */
#define				CUDA_BLOCK_SIZE						256						/* init forces */
#define				ST_BLOCK_SIZE						256	
#endif

#if defined( __NBRS_THREADS_PER_ATOM__ )
#define				NB_KER_THREADS_PER_ATOM				__NBRS_THREADS_PER_ATOM__
#else
#define				NB_KER_THREADS_PER_ATOM				16
#endif

#if defined( __NBRS_BLOCK_SIZE__)
#define				NBRS_BLOCK_SIZE						__NBRS_BLOCK_SIZE__
#else
#define				NBRS_BLOCK_SIZE						256
#endif

#if defined( __HB_THREADS_PER_ATOM__)
#define				HB_KER_THREADS_PER_ATOM				__HB_THREADS_PER_ATOM__
#else
#define				HB_KER_THREADS_PER_ATOM				32
#endif

#if defined(__HB_BLOCK_SIZE__)
#define				HB_BLOCK_SIZE					__HB_BLOCK_SIZE__
#else
#define				HB_BLOCK_SIZE						256
#endif

#if defined( __VDW_THREADS_PER_ATOM__ )
#define				VDW_KER_THREADS_PER_ATOM			__VDW_THREADS_PER_ATOM__
#else
#define				VDW_KER_THREADS_PER_ATOM			32
#endif

#if defined( __VDW_BLOCK_SIZE__)
#define				VDW_BLOCK_SIZE						__VDW_BLOCK_SIZE__
#else
#define				VDW_BLOCK_SIZE						256
#endif

#if defined( __MATVEC_THREADS_PER_ROW__ )
#define 			MATVEC_KER_THREADS_PER_ROW		__MATVEC_THREADS_PER_ROW__
#else
#define 			MATVEC_KER_THREADS_PER_ROW		32
#endif

#if defined( __MATVEC_BLOCK_SIZE__)
#define				MATVEC_BLOCK_SIZE					__MATVEC_BLOCK_SIZE__
#else
#define				MATVEC_BLOCK_SIZE					512
#endif

//Validation
#define				GPU_TOLERANCE				1e-5

#endif



/******************* ENUMERATIONS *************************/
enum geo_formats { CUSTOM, PDB, ASCII_RESTART, BINARY_RESTART, GF_N };

enum restart_formats { WRITE_ASCII, WRITE_BINARY, RF_N };

enum ensembles { NVE, bNVT, nhNVT, sNPT, iNPT, NPT, ens_N };

enum lists { BONDS, OLD_BONDS, THREE_BODIES, 
	     HBONDS, FAR_NBRS, DBOS, DDELTAS, LIST_N };

enum interactions { TYP_VOID, TYP_BOND, TYP_THREE_BODY, 
		    TYP_HBOND, TYP_FAR_NEIGHBOR, TYP_DBO, TYP_DDELTA, TYP_N };

enum message_tags { INIT, UPDATE, BNDRY, UPDATE_BNDRY,
		    EXC_VEC1, EXC_VEC2, DIST_RVEC2, COLL_RVEC2, 
		    DIST_RVECS, COLL_RVECS, INIT_DESCS, ATOM_LINES, 
		    BOND_LINES, ANGLE_LINES, RESTART_ATOMS, TAGS_N };

enum errors { FILE_NOT_FOUND = -10, UNKNOWN_ATOM_TYPE = -11, 
	      CANNOT_OPEN_FILE = -12, CANNOT_INITIALIZE = -13, 
	      INSUFFICIENT_MEMORY = -14, UNKNOWN_OPTION = -15,
	      INVALID_INPUT = -16, INVALID_GEO = -17 };

enum exchanges { NONE, NEAR_EXCH, FULL_EXCH };

enum gcell_types { NO_NBRS=0, NEAR_ONLY=1, HBOND_ONLY=2, FAR_ONLY=4, 
		   NEAR_HBOND=3, NEAR_FAR=5, HBOND_FAR=6, FULL_NBRS=7, 
		   NATIVE=8 };

enum atoms { C_ATOM = 0, H_ATOM = 1, O_ATOM = 2, N_ATOM = 3, 
	     S_ATOM = 4, SI_ATOM = 5, GE_ATOM = 6, X_ATOM = 7 };

enum traj_methods { REG_TRAJ, MPI_TRAJ, TF_N };

enum molecules { UNKNOWN, WATER };

enum list_on { TYP_HOST, TYP_DEVICE };



/********************** TYPE DEFINITIONS ********************/
typedef int  ivec[3];
typedef double real;
typedef real rvec[3];
typedef real rtensor[3][3];
typedef real rvec2[2];
typedef real rvec4[4];


typedef struct {
  int step, bigN;
  real T, xi, v_xi, v_xi_old, G_xi;
  rtensor box;
} restart_header;

typedef struct {
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

  //MPI_Request  send_req1[MAX_NBRS];
  //MPI_Request  send_req2[MAX_NBRS];
  //MPI_Status   send_stat1[MAX_NBRS];
  //MPI_Status   send_stat2[MAX_NBRS];
  //MPI_Status   recv_stat1[MAX_NBRS];
  //MPI_Status   recv_stat2[MAX_NBRS];

  mpi_out_data out_buffers[MAX_NBRS];
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
typedef struct {
  /* Bond Order parameters */
  real p_bo1,p_bo2,p_bo3,p_bo4,p_bo5,p_bo6;
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
typedef struct {
  /* valence angle */
  real theta_00;
  real p_val1, p_val2, p_val4, p_val7;

  /* penalty */
  real p_pen1;

  /* 3-body conjugation */
  real p_coa1;
} three_body_parameters;


typedef struct{
  int cnt;
  three_body_parameters prm[MAX_3BODY_PARAM];
} three_body_header;



/* hydrogen-bond parameters */
typedef struct{
  real r0_hb, p_hb1, p_hb2, p_hb3;
} hbond_parameters;



/* 4-body parameters */
typedef struct {
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
  global_parameters d_gp;

  single_body_parameters *sbp;
  single_body_parameters *d_sbp;

  two_body_parameters *tbp; //changed
  two_body_parameters *d_tbp; //changed

  three_body_header *thbp; //changed
  three_body_header *d_thbp; //changed

  hbond_parameters *hbp; //changed
  hbond_parameters *d_hbp; //changed

  four_body_header *fbp; //changed
  four_body_header *d_fbp; //changed
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
  //real cutoff;
  rvec min, max;
  //ivec rel_box;

  int  mark;
  int  type;
  //int  str;
  //int  end;
  int  top;
  int* atoms;
  //struct grid_cell** nbrs; //changed
  //ivec* nbrs_x;
  //rvec* nbrs_cp;
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

  grid_cell* cells; //changed
  ivec *order;

  //GRID
  int *str;
  int *end;
  real *cutoff;
  ivec *nbrs_x;
  rvec *nbrs_cp;

  ivec *rel_box;
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
  neighbor_proc    my_nbrs[MAX_NBRS];
  int             *global_offset;

  simulation_box   big_box, my_box, my_ext_box;
  simulation_box   *d_big_box, *d_my_box, *d_my_ext_box;

  grid             my_grid;
  grid             d_my_grid;

  boundary_cutoff  bndry_cuts;

  reax_atom       *my_atoms;
  reax_atom       *d_my_atoms;

/*CUDA-specific*/
  int					max_sparse_entries;
  int 				init_thblist;
  int					num_thbodies;

  int					max_bonds;
  int					max_hbonds;
} reax_system;



/* system control parameters */
typedef struct
{
  char sim_name[MAX_STR];
  int  nprocs;
  int  gpus_per_node;
  ivec procs_by_dim;
  /* ensemble values:
     0 : NVE
     1 : bNVT (Berendsen) 
     2 : nhNVT (Nose-Hoover)
     3 : sNPT (Parrinello-Rehman-Nose-Hoover) semiisotropic
     4 : iNPT (Parrinello-Rehman-Nose-Hoover) isotropic 
     5 : NPT  (Parrinello-Rehman-Nose-Hoover) Anisotropic*/
  int  ensemble;
  int  nsteps;
  real dt;
  int  geo_format;
  int  restart;

  int  restrict_bonds;
  int  remove_CoM_vel;
  int  random_vel;
  int  reposition_atoms;
  
  int  reneighbor;
  real vlist_cut;
  real bond_cut;
  real nonb_cut, nonb_low;
  real hbond_cut;
  real user_ghost_cut;

  real bg_cut;
  real bo_cut;
  real thb_cut;

  int tabulate;

  int qeq_freq;
  real q_err;
  int refactor;
  real droptol;  

  real T_init, T_final, T;
  real Tau_T;
  int  T_mode;
  real T_rate, T_freq;
  
  int  virial;  
  rvec P, Tau_P, Tau_PT;
  int  press_mode;
  real compressibility;

  int  molecular_analysis;
  int  num_ignored;
  int  ignore[MAX_ATOM_TYPES];

  int  dipole_anal;
  int  freq_dipole_anal;
  int  diffusion_coef;
  int  freq_diffusion_coef;
  int  restrict_type;

  void *d_control_params;
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
  real comm;
  real nbrs;
  real init_forces;
  real bonded;
  real nonb;
  real qEq;
  int  s_matvecs;
  int  t_matvecs;
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

  real M;			   // Total Mass 
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
  reax_timing d_timing;

  void *d_simulation_data;
} simulation_data;


typedef struct{
  int thb;
  int pthb; // pointer to the third body on the central atom's nbrlist 
  real theta, cos_theta;
  rvec dcos_di, dcos_dj, dcos_dk;
} three_body_interaction_data;


typedef struct {
  int nbr;
  ivec rel_box;
  real d;
  rvec dvec;
  
} far_neighbor_data;


typedef struct {
  int nbr;
  int scl;
  far_neighbor_data *ptr;

/*CUDA-specific*/
  int sym_index;
  rvec hb_f;
} hbond_data;


typedef struct{
  int wrt;
  rvec dVal;
} dDelta_data;


typedef struct{
  int wrt;
  rvec dBO, dBOpi, dBOpi2;
} dbond_data;

typedef struct{
  real BO, BO_s, BO_pi, BO_pi2;
  real Cdbo, Cdbopi, Cdbopi2;
  real C1dbo, C2dbo, C3dbo;
  real C1dbopi, C2dbopi, C3dbopi, C4dbopi;
  real C1dbopi2, C2dbopi2, C3dbopi2, C4dbopi2;
  rvec dBOp, dln_BOp_s, dln_BOp_pi, dln_BOp_pi2;
} bond_order_data;

typedef struct {
  int nbr;
  int sym_index;
  int dbond_index;
  ivec rel_box;
  //  rvec ext_factor;
  real d;
  rvec dvec;
  bond_order_data bo_data;

/*CUDA-specific*/
  real ae_CdDelta;

  real va_CdDelta;
  rvec va_f;

  real ta_CdDelta;
  real ta_Cdbo;
  rvec ta_f;

  rvec hb_f;

  rvec tf_f;
} bond_data;


typedef struct {
  int j;
  real val;
} sparse_matrix_entry;

typedef struct {
  int cap, n, m;
  int *start, *end;
  sparse_matrix_entry *entries;
} sparse_matrix;


typedef struct { 
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
  int *bond_mark, *done_after;

  /* QEq storage */
  sparse_matrix H, L, U; //CHANGED
  real *Hdia_inv, *b_s, *b_t, *b_prc, *b_prm, *s, *t;
  real *droptol;
  rvec2 *b, *x;
  
  /* GMRES storage */
  real *y, *z, *g;
  real *hc, *hs;
  real *h, *v; //changed
  /* CG storage */
  real *r, *d, *q, *p;
  rvec2 *r2, *d2, *q2, *p2;
  /* Taper */
  real Tap[8]; //Tap7, Tap6, Tap5, Tap4, Tap3, Tap2, Tap1, Tap0;
  real d_Tap;

  /* storage for analysis */
  int  *mark, *old_mark;
  rvec *x_old;
  
  /* storage space for bond restrictions */
  int  *restricted;
  int *restricted_list;   //changed

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
  void *v;
  three_body_interaction_data *three_body_list;
  bond_data          *bond_list;
  dbond_data         *dbo_list;
  dDelta_data        *dDelta_list;
  far_neighbor_data  *far_nbr_list;
  hbond_data         *hbond_list;
} list_type;


typedef struct
{
  int allocated;

  int n;
  int num_intrs;

  int *index;
  int *end_index;

  int type;
  list_type select;
} reax_list;


typedef struct
{
  MPI_File trj;
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

  LR_data *y;
  cubic_spline_coef *H;
  cubic_spline_coef *vdW, *CEvd;
  cubic_spline_coef *ele, *CEclmb;
} LR_lookup_table;

extern LR_lookup_table *LR; //changed

/* function pointer defs */
typedef void (*evolve_function)(reax_system*, control_params*, 
				simulation_data*, storage*, reax_list**, 
				output_controls*, mpi_datatypes* );
#if defined(PURE_REAX)
extern evolve_function  Evolve;
extern evolve_function Cuda_Evolve;
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

/*CUDA-specific*/
extern reax_list **dev_lists;
extern storage *dev_workspace;
extern storage *dev_storage;
extern LR_lookup_table *d_LR;

extern void *scratch;
extern void *host_scratch;
extern int BLOCKS, BLOCKS_POW_2, BLOCK_SIZE;
extern int BLOCKS_N, BLOCKS_POW_2_N;
extern int MATVEC_BLOCKS;

#endif
