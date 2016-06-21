/*----------------------------------------------------------------------
  PuReMD-GPU - Reax Force Field Simulator

  Copyright (2014) Purdue University
  Sudhir Kylasa, skylasa@purdue.edu
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
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

#include "init_md.h"
#include "allocate.h"
#include "box.h"
#include "forces.h"
#include "grid.h"
#include "GMRES.h"
#include "integrate.h"
#include "neighbors.h"
#include "list.h"
#include "lookup.h"
#include "print_utils.h"
#include "reset_utils.h"
#include "system_props.h"
#include "traj.h"
#include "vector.h"


#include "cuda_init.h"
#include "cuda_copy.h"
#include "cuda_utils.h"
#include "helpers.h"
#include "reduction.h"

#include     "index_utils.h"

#include "validation.h"

void Generate_Initial_Velocities(reax_system *system, real T )
{
    int i;
    real scale, norm;


    if( T <= 0.1 ) {
        for (i=0; i < system->N; i++)
            rvec_MakeZero( system->atoms[i].v );
#if defined(DEBUG)
        fprintf( stderr, "no random velocities...\n" );
#endif
    }
    else {
        for( i = 0; i < system->N; i++ ) {
            rvec_Random( system->atoms[i].v );

            norm = rvec_Norm_Sqr( system->atoms[i].v );
            scale = SQRT( system->reaxprm.sbp[ system->atoms[i].type ].mass * 
                    norm / (3.0 * K_B * T) );

            rvec_Scale( system->atoms[i].v, 1.0/scale, system->atoms[i].v );

            /*
               fprintf( stderr, "v = %f %f %f\n", 
               system->atoms[i].v[0],system->atoms[i].v[1],system->atoms[i].v[2]);
               fprintf( stderr, "scale = %f\n", scale );
               fprintf( stderr, "v = %f %f %f\n",
               system->atoms[i].v[0],system->atoms[i].v[1],system->atoms[i].v[2]);
             */
        }
    }
}


void Init_System( reax_system *system, control_params *control, 
        simulation_data *data )
{
    int i;
    rvec dx;

    if( !control->restart )
        Reset_Atoms( system );

    Compute_Total_Mass( system, data );

    Compute_Center_of_Mass( system, data, stderr );

    /* reposition atoms */
    // just fit the atoms to the periodic box
    if( control->reposition_atoms == 0 ) {
        rvec_MakeZero( dx );
    }
    // put the center of mass to the center of the box
    else if( control->reposition_atoms == 1 ) {
        rvec_Scale( dx, 0.5, system->box.box_norms );
        rvec_ScaledAdd( dx, -1., data->xcm );
    }
    // put the center of mass to the origin
    else if( control->reposition_atoms == 2 ) {
        rvec_Scale( dx, -1., data->xcm );
    }
    else {
        fprintf( stderr, "UNKNOWN OPTION: reposition_atoms. Terminating...\n" );
        exit( UNKNOWN_OPTION );
    }

    for( i = 0; i < system->N; ++i ) {
        Inc_on_T3( system->atoms[i].x, dx, &(system->box) );
        /*fprintf( stderr, "%6d%2d%8.3f%8.3f%8.3f\n", 
          i, system->atoms[i].type, 
          system->atoms[i].x[0], system->atoms[i].x[1], system->atoms[i].x[2] );*/
    }

    /* Initialize velocities so that desired init T can be attained */
    if( !control->restart || (control->restart && control->random_vel) )  {
        Generate_Initial_Velocities( system, control->T_init );
    }

    Setup_Grid( system );
}


void Cuda_Init_System( reax_system *system, control_params *control, 
        simulation_data *data )
{
    int i;
    rvec dx;

    if( !control->restart )
        Cuda_Reset_Atoms( system );

    Cuda_Compute_Total_Mass( system, data );

    Cuda_Compute_Center_of_Mass( system, data, stderr );

    /* reposition atoms */
    // just fit the atoms to the periodic box
    if( control->reposition_atoms == 0 ) {
        rvec_MakeZero( dx );
    }
    // put the center of mass to the center of the box
    else if( control->reposition_atoms == 1 ) {
        rvec_Scale( dx, 0.5, system->box.box_norms );
        rvec_ScaledAdd( dx, -1., data->xcm );
    }
    // put the center of mass to the origin
    else if( control->reposition_atoms == 2 ) {
        rvec_Scale( dx, -1., data->xcm );
    }
    else {
        fprintf( stderr, "UNKNOWN OPTION: reposition_atoms. Terminating...\n" );
        exit( UNKNOWN_OPTION );
    }

    compute_Inc_on_T3 <<<BLOCKS_POW_2, BLOCK_SIZE>>>
        (system->d_atoms, system->N, system->d_box, dx[0], dx[1], dx[2]);
    cudaThreadSynchronize ();
    cudaCheckError ();

    //copy back the atoms from device to the host
    copy_host_device (system->atoms, system->d_atoms, REAX_ATOM_SIZE * system->N , 
            cudaMemcpyDeviceToHost, RES_SYSTEM_ATOMS );

    /* Initialize velocities so that desired init T can be attained */
    if( !control->restart || (control->restart && control->random_vel) )  {
        Generate_Initial_Velocities( system, control->T_init );
    }

    Setup_Grid( system );
}



void Init_Simulation_Data( reax_system *system, control_params *control, 
        simulation_data *data, output_controls *out_control, 
        evolve_function *Evolve )
{

    Reset_Simulation_Data( data );

    if( !control->restart )  
        data->step = data->prev_steps = 0;

    switch( control->ensemble ) {
        case NVE:
            data->N_f = 3 * system->N;
            *Evolve = Velocity_Verlet_NVE;
            break;


        case NVT:
            data->N_f = 3 * system->N + 1;
            //control->Tau_T = 100 * data->N_f * K_B * control->T_final;
            if( !control->restart || (control->restart && control->random_vel) ) {
                data->therm.G_xi = control->Tau_T * (2.0 * data->E_Kin - 
                        data->N_f * K_B * control->T );
                data->therm.v_xi = data->therm.G_xi * control->dt;
                data->therm.v_xi_old = 0;
                data->therm.xi = 0;
#if defined(DEBUG_FOCUS)
                fprintf( stderr, "init_md: G_xi=%f Tau_T=%f E_kin=%f N_f=%f v_xi=%f\n",
                        data->therm.G_xi, control->Tau_T, data->E_Kin, 
                        data->N_f, data->therm.v_xi );
#endif
            }

            *Evolve = Velocity_Verlet_Nose_Hoover_NVT_Klein;
            break;


        case NPT: // Anisotropic NPT
            fprintf( stderr, "THIS OPTION IS NOT YET IMPLEMENTED! TERMINATING...\n" );
            exit( UNKNOWN_OPTION );
            data->N_f = 3 * system->N + 9;
            if( !control->restart ) {
                data->therm.G_xi = control->Tau_T * (2.0 * data->E_Kin - 
                        data->N_f * K_B * control->T );
                data->therm.v_xi = data->therm.G_xi * control->dt;
                data->iso_bar.eps = 0.33333 * log(system->box.volume);
                //data->inv_W = 1. / (data->N_f*K_B*control->T*SQR(control->Tau_P));
                //Compute_Pressure( system, data, workspace );
            }
            *Evolve = Velocity_Verlet_Berendsen_Isotropic_NPT;
            break;


        case sNPT: // Semi-Isotropic NPT
            data->N_f = 3 * system->N + 4;
            *Evolve = Velocity_Verlet_Berendsen_SemiIsotropic_NPT;
            break;


        case iNPT: // Isotropic NPT
            data->N_f = 3 * system->N + 2;
            *Evolve = Velocity_Verlet_Berendsen_Isotropic_NPT;
            break;

        case bNVT: //berendensen NVT
            data->N_f = 3 * system->N + 1; 
            *Evolve = Velocity_Verlet_Berendsen_NVT;
            break;

        default:
            break;
    }

    Compute_Kinetic_Energy( system, data );

    /* init timing info for the host*/
    data->timing.start = Get_Time( );
    data->timing.total = data->timing.start;
    data->timing.nbrs = 0;
    data->timing.init_forces = 0;
    data->timing.bonded = 0;
    data->timing.nonb = 0;
    data->timing.QEq = 0;
    data->timing.matvecs = 0;
}


void Cuda_Init_Simulation_Data( reax_system *system, control_params *control, 
        simulation_data *data, output_controls *out_control, 
        evolve_function *Evolve )
{

    Reset_Simulation_Data( data );

    if( !control->restart )  
        data->step = data->prev_steps = 0;

    switch( control->ensemble ) {
        case NVE:
            data->N_f = 3 * system->N;
            *Evolve = Cuda_Velocity_Verlet_NVE;
            break;


        case NVT:
            data->N_f = 3 * system->N + 1;
            //control->Tau_T = 100 * data->N_f * K_B * control->T_final;
            if( !control->restart || (control->restart && control->random_vel) ) {
                data->therm.G_xi = control->Tau_T * (2.0 * data->E_Kin - 
                        data->N_f * K_B * control->T );
                data->therm.v_xi = data->therm.G_xi * control->dt;
                data->therm.v_xi_old = 0;
                data->therm.xi = 0;
#if defined(DEBUG_FOCUS)
                fprintf( stderr, "init_md: G_xi=%f Tau_T=%f E_kin=%f N_f=%f v_xi=%f\n",
                        data->therm.G_xi, control->Tau_T, data->E_Kin, 
                        data->N_f, data->therm.v_xi );
#endif
            }

            *Evolve = Cuda_Velocity_Verlet_Nose_Hoover_NVT_Klein;
            break;


        case NPT: // Anisotropic NPT
            fprintf( stderr, "THIS OPTION IS NOT YET IMPLEMENTED! TERMINATING...\n" );
            exit( UNKNOWN_OPTION );
            data->N_f = 3 * system->N + 9;
            if( !control->restart ) {
                data->therm.G_xi = control->Tau_T * (2.0 * data->E_Kin - 
                        data->N_f * K_B * control->T );
                data->therm.v_xi = data->therm.G_xi * control->dt;
                data->iso_bar.eps = 0.33333 * log(system->box.volume);
                //data->inv_W = 1. / (data->N_f*K_B*control->T*SQR(control->Tau_P));
                //Compute_Pressure( system, data, workspace );
            }
            *Evolve = Velocity_Verlet_Berendsen_Isotropic_NPT;
            break;


        case sNPT: // Semi-Isotropic NPT
            fprintf( stderr, "THIS OPTION IS NOT YET IMPLEMENTED! TERMINATING...\n" );
            exit( UNKNOWN_OPTION );
            data->N_f = 3 * system->N + 4;
            *Evolve = Velocity_Verlet_Berendsen_SemiIsotropic_NPT;
            break;


        case iNPT: // Isotropic NPT
            fprintf( stderr, "THIS OPTION IS NOT YET IMPLEMENTED! TERMINATING...\n" );
            exit( UNKNOWN_OPTION );
            data->N_f = 3 * system->N + 2;
            *Evolve = Velocity_Verlet_Berendsen_Isotropic_NPT;
            break;

        case bNVT: //berendensen NVT
            data->N_f = 3 * system->N + 1; 
            *Evolve = Cuda_Velocity_Verlet_Berendsen_NVT;
            break;

        default:
            break;
    }

    Cuda_Compute_Kinetic_Energy (system, data);

#ifdef __BUILD_DEBUG__
    real t_E_Kin = 0;
    t_E_Kin = data->E_Kin;
#endif

    copy_host_device (&data->E_Kin, &((simulation_data *)data->d_simulation_data)->E_Kin, 
            REAL_SIZE, cudaMemcpyDeviceToHost, RES_SIMULATION_DATA );
    data->therm.T = (2. * data->E_Kin) / (data->N_f * K_B);
    if ( fabs(data->therm.T) < ALMOST_ZERO ) // avoid T being an absolute zero! 
        data->therm.T = ALMOST_ZERO;

#ifdef __BUILD_DEBUG__
    if (check_zero (t_E_Kin, data->E_Kin)){
        fprintf (stderr, "SimulationData:E_Kin does not match between host and device (%f %f) \n", t_E_Kin, data->E_Kin );
        exit (1);
    }
    //validate_data ( system, data );
#endif

    /* init timing info for the host*/
    data->timing.start = Get_Time( );
    data->timing.total = data->timing.start;
    data->timing.nbrs = 0;
    data->timing.init_forces = 0;
    data->timing.bonded = 0;
    data->timing.nonb = 0;
    data->timing.QEq = 0;
    data->timing.matvecs = 0;

    /* init timing info for the device */
    d_timing.start = Get_Time( );
    d_timing.total = data->timing.start;
    d_timing.nbrs = 0;
    d_timing.init_forces = 0;
    d_timing.bonded = 0;
    d_timing.nonb = 0;
    d_timing.QEq = 0;
    d_timing.matvecs = 0;
}


void Init_Workspace( reax_system *system, control_params *control, 
        static_storage *workspace )
{  
    int i;

    /* Allocate space for hydrogen bond list */
    workspace->hbond_index = (int *) malloc( system->N * sizeof( int ) );

    /* bond order related storage  */
    workspace->total_bond_order = (real *) malloc( system->N * sizeof( real ) );
    workspace->Deltap           = (real *) malloc( system->N * sizeof( real ) );
    workspace->Deltap_boc       = (real *) malloc( system->N * sizeof( real ) );
    workspace->dDeltap_self     = (rvec *) malloc( system->N * sizeof( rvec ) );

    workspace->Delta          = (real *) malloc( system->N * sizeof( real ) );
    workspace->Delta_lp          = (real *) malloc( system->N * sizeof( real ) );
    workspace->Delta_lp_temp    = (real *) malloc( system->N * sizeof( real ) );
    workspace->dDelta_lp          = (real *) malloc( system->N * sizeof( real ) );
    workspace->dDelta_lp_temp   = (real *) malloc( system->N * sizeof( real ) );
    workspace->Delta_e          = (real *) malloc( system->N * sizeof( real ) );
    workspace->Delta_boc        = (real *) malloc( system->N * sizeof( real ) );
    workspace->nlp          = (real *) malloc( system->N * sizeof( real ) );
    workspace->nlp_temp          = (real *) malloc( system->N * sizeof( real ) );
    workspace->Clp          = (real *) malloc( system->N * sizeof( real ) );
    workspace->CdDelta          = (real *) malloc( system->N * sizeof( real ) );
    workspace->vlpex          = (real *) malloc( system->N * sizeof( real ) );

    /* QEq storage */
    //workspace->H        = NULL;
    //workspace->L        = NULL;
    //workspace->U        = NULL;
    //
    workspace->H.start        = NULL;
    workspace->L.start        = NULL;
    workspace->U.start        = NULL;

    workspace->H.entries         = NULL;
    workspace->L.entries         = NULL;
    workspace->U.entries        = NULL;

    workspace->droptol  = (real *) calloc( system->N, sizeof( real ) );
    workspace->w        = (real *) calloc( system->N, sizeof( real ) );
    workspace->Hdia_inv = (real *) calloc( system->N, sizeof( real ) );
    workspace->b        = (real *) calloc( system->N * 2, sizeof( real ) );
    workspace->b_s      = (real *) calloc( system->N, sizeof( real ) );
    workspace->b_t      = (real *) calloc( system->N, sizeof( real ) );
    workspace->b_prc    = (real *) calloc( system->N * 2, sizeof( real ) );
    workspace->b_prm    = (real *) calloc( system->N * 2, sizeof( real ) );
    workspace->s_t      = (real *) calloc( system->N * 2, sizeof( real ) );
    workspace->s        = (real *) calloc( 5 * system->N, sizeof( real ) );
    workspace->t        = (real *) calloc( 5 * system->N, sizeof( real ) );
    // workspace->s_old    = (real *) calloc( system->N, sizeof( real ) );
    // workspace->t_old    = (real *) calloc( system->N, sizeof( real ) );
    // workspace->s_oldest = (real *) calloc( system->N, sizeof( real ) );
    // workspace->t_oldest = (real *) calloc( system->N, sizeof( real ) );

    for( i = 0; i < system->N; ++i ) {
        workspace->Hdia_inv[i] = 1./system->reaxprm.sbp[system->atoms[i].type].eta;
        workspace->b_s[i] = -system->reaxprm.sbp[ system->atoms[i].type ].chi;
        workspace->b_t[i] = -1.0;

        workspace->b[i] = -system->reaxprm.sbp[ system->atoms[i].type ].chi;
        workspace->b[i+system->N] = -1.0;
    }

    /* GMRES storage */
    workspace->y  = (real *)  calloc( RESTART+1, sizeof( real ) );
    workspace->z  = (real *)  calloc( RESTART+1, sizeof( real ) );
    workspace->g  = (real *)  calloc( RESTART+1, sizeof( real ) );
    workspace->hs = (real *)  calloc( RESTART+1, sizeof( real ) );
    workspace->hc = (real *)  calloc( RESTART+1, sizeof( real ) );

    workspace->rn = (real *) calloc( (RESTART+1)*system->N*2, sizeof( real) );
    workspace->v  = (real *) calloc( (RESTART+1)*system->N, sizeof( real) );
    workspace->h  = (real *) calloc( (RESTART+1)*(RESTART+1), sizeof( real) );

    /* CG storage */
    workspace->r = (real *) calloc( system->N, sizeof( real ) );
    workspace->d = (real *) calloc( system->N, sizeof( real ) );
    workspace->q = (real *) calloc( system->N, sizeof( real ) );
    workspace->p = (real *) calloc( system->N, sizeof( real ) );


    /* integrator storage */
    workspace->a = (rvec *) malloc( system->N * sizeof( rvec ) );
    workspace->f_old = (rvec *) malloc( system->N * sizeof( rvec ) );
    workspace->v_const = (rvec *) malloc( system->N * sizeof( rvec ) );


    /* storage for analysis */
    if( control->molec_anal || control->diffusion_coef )
    {
        workspace->mark = (int *) calloc( system->N, sizeof(int) );
        workspace->old_mark = (int *) calloc( system->N, sizeof(int) );
    }
    else 
        workspace->mark = workspace->old_mark = NULL;

    if( control->diffusion_coef )
        workspace->x_old = (rvec *) calloc( system->N, sizeof( rvec ) );
    else workspace->x_old = NULL;


#ifdef TEST_FORCES
    workspace->dDelta = (rvec *) malloc( system->N * sizeof( rvec ) );
    workspace->f_ele = (rvec *) malloc( system->N * sizeof( rvec ) );
    workspace->f_vdw = (rvec *) malloc( system->N * sizeof( rvec ) );
    workspace->f_bo = (rvec *) malloc( system->N * sizeof( rvec ) );
    workspace->f_be = (rvec *) malloc( system->N * sizeof( rvec ) );
    workspace->f_lp = (rvec *) malloc( system->N * sizeof( rvec ) );
    workspace->f_ov = (rvec *) malloc( system->N * sizeof( rvec ) );
    workspace->f_un = (rvec *) malloc( system->N * sizeof( rvec ) );
    workspace->f_ang = (rvec *) malloc( system->N * sizeof( rvec ) );
    workspace->f_coa = (rvec *) malloc( system->N * sizeof( rvec ) );
    workspace->f_pen = (rvec *) malloc( system->N * sizeof( rvec ) );
    workspace->f_hb = (rvec *) malloc( system->N * sizeof( rvec ) );
    workspace->f_tor = (rvec *) malloc( system->N * sizeof( rvec ) );
    workspace->f_con = (rvec *) malloc( system->N * sizeof( rvec ) );
#endif

    workspace->realloc.num_far = -1;
    workspace->realloc.Htop = -1;
    workspace->realloc.hbonds = -1;
    workspace->realloc.bonds = -1;
    workspace->realloc.num_3body = -1;
    workspace->realloc.gcell_atoms = -1;

    Reset_Workspace( system, workspace );
}

void compare_far_neighbors (int *test, int *start, int *end, far_neighbor_data *data, list *slist, int N)
{
    int index = 0;
    int count = 0;
    int jicount = 0;
    int end_index, gpu_index, gpu_end, k;
    far_neighbor_data gpu, cpu;

    /*
       for (int i = 0; i < N ; i++ )
       {
       if (test[i] != start[i]) {
       fprintf (stderr, "start index does not match \n");
       exit (0);
       }

       if (test[i+1] != (end[i]) ){
       fprintf (stderr, "end index does not match for atom %d (cpu: %d gpu: %d) \n", i, test[i+1], end[i]);
       exit (0);
       }
       }
     */


    for (int i = 0; i < N; i++){
        index = Start_Index (i, slist);
        //fprintf (stderr, "GPU : Neighbors of atom --> %d (start: %d , end: %d )\n", i, start[i], end[i]);


        for (int j = start[i]; j < end[i]; j++){
            gpu = data[j];

            if (i < data[j].nbr) continue;
            /*
               if (i < data[j].nbr) {
            //fprintf (stderr, " atom %d and neighbor %d @ index %d\n", i, data[j].nbr, j);
            int src = data[j].nbr;
            int dest = i;
            int x;


            for (x = start[src]; x < end[src]; x++) {
            if (data[x].nbr != dest) continue;

            gpu = data[x];
            cpu = data[j];

            if (  (gpu.d != cpu.d) ||
            (cpu.dvec[0] != gpu.dvec[0]) || (cpu.dvec[1] != gpu.dvec[1]) || (cpu.dvec[2] != gpu.dvec[2]) ||
            (cpu.rel_box[0] != gpu.rel_box[0]) || (cpu.rel_box[1] != gpu.rel_box[1]) || (cpu.rel_box[2] != gpu.rel_box[2])) {
            fprintf (stderr, " atom %d neighbor %d  (%f, %d, %d, %d - %f %f %f) \n", i, data[j].nbr, 
            data[j].d, 
            data[j].rel_box[0],
            data[j].rel_box[1],
            data[j].rel_box[2],
            data[j].dvec[0], 
            data[j].dvec[1], 
            data[j].dvec[2] 
            );
            fprintf (stderr, " atom %d neighbor %d  (%f, %d, %d, %d - %f %f %f) \n", data[j].nbr, data[x].nbr,
            data[x].d,
            data[x].rel_box[0],
            data[x].rel_box[1],
            data[x].rel_box[2],
            data[x].dvec[0],
            data[x].dvec[1],
            data[x].dvec[2]
            );
            jicount++;
            }
            break;
            }

            if (x >= end[src]) {
            fprintf (stderr, "could not find the neighbor duplicate data for ij (%d %d)\n", i, src );
            exit (0);
            }

            continue;
            }
             */

            cpu = slist->select.far_nbr_list[index];
            //if ( (gpu.nbr != cpu.nbr) || (gpu.d != cpu.d) ){
            //if ( (gpu->d != cpu->d) ){
            if (  (gpu.nbr != cpu.nbr) || (gpu.d != cpu.d) ||
                    (cpu.dvec[0] != gpu.dvec[0]) || (cpu.dvec[1] != gpu.dvec[1]) || (cpu.dvec[2] != gpu.dvec[2]) ||
                    (cpu.rel_box[0] != gpu.rel_box[0]) || (cpu.rel_box[1] != gpu.rel_box[1]) || (cpu.rel_box[2] != gpu.rel_box[2])) {
                //if ( (gpu.dvec[0] != i) || (gpu.dvec[1] != i) ||(gpu.dvec[2] != i) ||
                //        (gpu.rel_box[0] != i) || (gpu.rel_box[1] != i) ||(gpu.rel_box[2] != i) ) {
                //if (memcmp (&gpu, &cpu, FAR_NEIGHBOR_SIZE - RVEC_SIZE - INT_SIZE )){

                fprintf (stderr, "GPU:atom --> %d (s: %d , e: %d, i: %d ) (%d %d %d) \n", i, start[i], end[i], j, gpu.rel_box[0], gpu.rel_box[1], gpu.rel_box[2] );
                fprintf (stderr, "CPU:atom --> %d (s: %d , e: %d, i: %d )\n", i, Start_Index(i, slist), End_Index (i, slist), index);

                /*
                   fprintf (stdout, "Far neighbors does not match atom: %d \n", i );
                   fprintf (stdout, "neighbor %d ,  %d \n",  cpu.nbr, gpu.nbr);
                   fprintf (stdout, "d %f ,  %f \n", slist->select.far_nbr_list[index].d, data[j].d);
                   fprintf (stdout, "dvec (%f %f %f) (%f %f %f) \n", 
                   cpu.dvec[0], cpu.dvec[1], cpu.dvec[2],
                   gpu.dvec[0], gpu.dvec[1], gpu.dvec[2] );

                   fprintf (stdout, "ivec (%d %d %d) (%d %d %d) \n", 
                   cpu.rel_box[0], cpu.rel_box[1], cpu.rel_box[2],
                   gpu.rel_box[0], gpu.rel_box[1], gpu.rel_box[2] );

                 */
                count ++;
            }

            //fprintf (stderr, "GPU (neighbor %d , d %d )\n", gpu->nbr, gpu->d);
            index ++;
            }

            if (index != End_Index (i, slist))
            {
                fprintf (stderr, "End index does not match for atom --> %d end index (%d) Cpu (%d, %d ) gpu (%d, %d)\n", i, index, Start_Index (i, slist), End_Index(i, slist),
                        start[i], end[i]);
                exit (10);
            }
            }

            fprintf (stderr, "Far neighbors MATCH between CPU and GPU -->%d  reverse %d \n", count, jicount);

            /*
               for (int i = 0; i < N; i++) 
               {
               index = Start_Index (i, slist);
               end_index = End_Index (i, slist);

               gpu_index = start[i];
               gpu_end = end[i];
               for (int j = index; j < end_index; j++) 
               {
               far_neighbor_data *cpu = &slist->select.far_nbr_list[j];
               far_neighbor_data *gpu;

               for (k = gpu_index; k < gpu_end; k++) {
               gpu = &data[k];
               if (gpu->nbr == cpu->nbr) break;
               }

               if (k == gpu_end) { fprintf (stderr, " could not find neighbor for atom %d \n", i); exit (1); }

               if ( (gpu->nbr != cpu->nbr) || (gpu->d != cpu->d) ||
               ((cpu->dvec[0] || gpu->dvec[0]) || (cpu->dvec[1] || gpu->dvec[1]) || (cpu->dvec[2] || gpu->dvec[2])) ||
               ((cpu->rel_box[0] || gpu->rel_box[0]) || (cpu->rel_box[1] || gpu->rel_box[1]) || (cpu->rel_box[2] || gpu->rel_box[2])) ) {

               fprintf (stderr, "Far neighbors does not match atom: %d \n", i );
               fprintf (stderr, "neighbor %d ,  %d \n",  cpu->nbr, gpu->nbr);
               fprintf (stderr, "d %d ,  %d \n", cpu->d, gpu->d);
               fprintf (stderr, "dvec (%f %f %f) (%f %f %f) \n", 
               cpu->dvec[0], cpu->dvec[1], cpu->dvec[2],
               gpu->dvec[0], gpu->dvec[1], gpu->dvec[2] );

               fprintf (stderr, "ivec (%d %d %d) (%d %d %d) \n", 
               cpu->rel_box[0], cpu->rel_box[1], cpu->rel_box[2],
               gpu->rel_box[0], gpu->rel_box[1], gpu->rel_box[2] );
               fprintf (stderr, "GPU start %d GPU End %d \n", gpu_index, gpu_end );

               exit (1);
               }
               }
               }

             */
        }

        int Estimate_Device_Matrix (reax_system *system, control_params *control, 
                simulation_data *data, static_storage *workspace, 
                list **lists, output_controls *out_control )
        {
            int *indices, *Htop;
            list *far_nbrs = dev_lists + FAR_NBRS;
            int max_sparse_entries = 0;
            real t1, t2;

            indices = (int *) scratch;
            cuda_memset ( indices, 0, INT_SIZE * system->N, RES_SCRATCH );

            t1 = Get_Time ();

            Estimate_Sparse_Matrix_Entries <<<BLOCKS, BLOCK_SIZE>>>
                ( system->d_atoms, (control_params *)control->d_control, 
                  (simulation_data *)data->d_simulation_data, (simulation_box *)system->d_box, 
                  *far_nbrs, system->N, indices );
            cudaThreadSynchronize ();
            cudaCheckError ();

            t2 = Get_Timing_Info ( t1 );

            //fprintf (stderr, " Time to estimate sparse matrix entries --- > %f \n", t2 );

            Htop = (int *) malloc (INT_SIZE * (system->N + 1));
            memset (Htop, 0, INT_SIZE * (system->N + 1));
            copy_host_device (Htop, indices, system->N * INT_SIZE, cudaMemcpyDeviceToHost, __LINE__);

            for (int i = 0; i < system->N; i++) 
            {
                if (max_sparse_entries < Htop[i]) {
                    max_sparse_entries = Htop[i];
                }    
            }

#ifdef __DEBUG_CUDA__
            fprintf (stderr, " Max sparse entries for this run are ---> %d \n", max_sparse_entries );
#endif

            return max_sparse_entries * SAFE_ZONE;
            //return max_sparse_entries;
        }

        void Allocate_Device_Matrix (reax_system *system, control_params *control, 
                simulation_data *data, static_storage *workspace, 
                list **lists, output_controls *out_control )
        {

            //Allocate space for the sparse Matrix entries here. 
            system->max_sparse_matrix_entries = 
                Estimate_Device_Matrix (system, control, data, workspace, lists, out_control );
            dev_workspace->H.n = system->N ;
            dev_workspace->H.m = system->N * system->max_sparse_matrix_entries;
            Cuda_Init_Sparse_Matrix (&dev_workspace->H, system->max_sparse_matrix_entries * system->N, system->N );

#ifdef __CUDA_MEM__
            fprintf( stderr, "Device memory allocated: sparse matrix= %ld (MB)\n", 
                    system->max_sparse_matrix_entries * system->N * sizeof(sparse_matrix_entry) / (1024*1024) );
#endif
        }

        void Cuda_Init_Lists( reax_system *system, control_params *control, 
                simulation_data *data, static_storage *workspace, 
                list **lists, output_controls *out_control )
        {
            int i, num_nbrs, num_hbonds, num_bonds, num_3body, Htop;
            int *hb_top, *bond_top;

            real t_start, t_elapsed;

            grid *g = &( system->g );
            int *d_indices = (int *) scratch;
            int total = g->ncell[0] * g->ncell[1] * g->ncell[2];

            cuda_memset ( d_indices, 0, INT_SIZE * system->N, RES_SCRATCH );

#ifdef __BUILD_DEBUG__
            for (int i = 0; i < g->max_nbrs; i ++) {
                if ((g->nbrs[i][0] >= g->ncell[0]) ||
                        (g->nbrs[i][1] >= g->ncell[1]) ||
                        (g->nbrs[i][2] >= g->ncell[2]) ) {
                    fprintf (stderr, " Grid Incorrectly built.... \n");
                    exit (1);
                }

            }
#endif

            dim3 blockspergrid (system->g.ncell[0], system->g.ncell[1], system->g.ncell[2]);
            dim3 threadsperblock (system->g.max_atoms);

#ifdef __BUILD_DEBUG__
            fprintf (stderr, "Blocks per grid (%d %d %d)\n", system->g.ncell[0], system->g.ncell[1], system->g.ncell[2]);
            fprintf (stderr, "Estimate Num  Neighbors with threads per block as %d \n", system->d_g.max_atoms);
            fprintf (stderr, "Max nbrs %d \n", system->d_g.max_nbrs);
#endif 


            //First Bin atoms and they sync the host and the device for the grid.
            //This will copy the atoms from host to device.
            Cuda_Bin_Atoms (system, workspace);
            Sync_Host_Device (&system->g, &system->d_g, cudaMemcpyHostToDevice );

            Estimate_NumNeighbors <<<blockspergrid, threadsperblock >>>
                (system->d_atoms, system->d_g, system->d_box, 
                 (control_params *)control->d_control, d_indices);
            cudaThreadSynchronize ();
            cudaCheckError ();

            int *nbrs_indices = (int *) malloc( INT_SIZE * (system->N+1) );
            memset (nbrs_indices , 0, INT_SIZE * (system->N + 1));

            nbrs_indices [0] = 0;
            copy_host_device (&nbrs_indices [1], d_indices, INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__); 

            for (int i = 1; i <= system->N; i++)
                nbrs_indices [i] += nbrs_indices [i-1];

            num_nbrs = nbrs_indices [system->N] ;
            system->num_nbrs = num_nbrs;

#ifdef __DEBUG_CUDA__
            fprintf (stderr, "Total neighbors %d \n", nbrs_indices[system->N]);
            fprintf (stderr, "Corrected Total neighbors %d \n", num_nbrs);
#endif


            list *far_nbrs = (dev_lists + FAR_NBRS);
            if( !Make_List(system->N, num_nbrs, TYP_FAR_NEIGHBOR, far_nbrs, TYP_DEVICE) ) {
                fprintf(stderr, "Problem in initializing far nbrs list. Terminating!\n");
                exit( INIT_ERR );
            }

#ifdef __CUDA_MEM__
            fprintf( stderr, "Device memory allocated: far_nbrs = %ld (MB)\n", 
                    num_nbrs * sizeof(far_neighbor_data) / (1024*1024) );
#endif

            copy_host_device (nbrs_indices, far_nbrs->index, INT_SIZE * system->N, cudaMemcpyHostToDevice, __LINE__ );
            copy_host_device (nbrs_indices, far_nbrs->end_index, INT_SIZE * system->N, cudaMemcpyHostToDevice, __LINE__ );
            Cuda_Generate_Neighbor_Lists (system, workspace, control, false);

#ifdef __BUILD_DEBUG__

            int *end = (int *)malloc (sizeof (int) * system->N);
            int *start = (int *) malloc (sizeof (int) * system->N );

            copy_host_device (start, far_nbrs->index, INT_SIZE * system->N, cudaMemcpyDeviceToHost, 0);
            copy_host_device (end, far_nbrs->end_index, INT_SIZE * system->N, cudaMemcpyDeviceToHost, 0);

            far_neighbor_data *far_data = (far_neighbor_data *) 
                malloc (FAR_NEIGHBOR_SIZE * num_nbrs);
            copy_host_device (far_data, far_nbrs->select.far_nbr_list, 
                    FAR_NEIGHBOR_SIZE * num_nbrs, cudaMemcpyDeviceToHost, 0);

            compare_far_neighbors (nbrs_indices, start, end, far_data, *lists + FAR_NBRS, system->N);

            free (start);
            free (end);
#endif

            int *output, size;
            size = INT_SIZE * 2 * system->N + 2;
            output = (int *) malloc (size);
            Cuda_Estimate_Storage_Sizes (system, control, output);

            Htop = output[0];
            num_3body  = output[1];
            hb_top = &output[ 2 ]; 
            bond_top = &output[ 2 + system->N ];

#ifdef __DEBUG_CUDA__
            int max_hbonds = 0;
            int min_hbonds = 1000;
            int max_bonds = 0;
            int min_bonds = 1000;
            for (int i = 0; i < system->N; i++) {
                if ( max_hbonds < hb_top[i])
                    max_hbonds = hb_top[i];
                if (min_hbonds > hb_top[i])
                    min_hbonds = hb_top[i];

                if (max_bonds < bond_top [i])
                    max_bonds = bond_top[i];
                if (min_bonds > bond_top[i])
                    min_bonds = bond_top[i];
            }

            fprintf (stderr, "Max Hbonds %d min Hbonds %d \n", max_hbonds, min_hbonds );
            fprintf (stderr, "Max bonds %d min bonds %d \n", max_bonds, min_bonds );
            fprintf (stderr, "Device HTop --> %d and num_3body --> %d \n", Htop, num_3body );
#endif

            Allocate_Device_Matrix (system, control, data, workspace, lists, out_control );

            dev_workspace->num_H = 0;

            if( control->hb_cut > 0 ) {

                int *hbond_index = (int *) malloc ( INT_SIZE * system->N );
                // init H indexes 
                num_hbonds = 0;
                for( i = 0; i < system->N; ++i )
                    if( system->reaxprm.sbp[ system->atoms[i].type ].p_hbond == 1 || 
                            system->reaxprm.sbp[ system->atoms[i].type ].p_hbond == 2  ) // H atom
                        //hbond_index[i] = workspace->num_H++;
                        hbond_index[i] = num_hbonds ++;
                    else 
                        hbond_index[i] = -1;

                copy_host_device (hbond_index, dev_workspace->hbond_index, 
                        system->N * INT_SIZE, cudaMemcpyHostToDevice, RES_STORAGE_HBOND_INDEX );
                dev_workspace->num_H = num_hbonds;

#ifdef __DEBUG_CUDA__
                fprintf (stderr, "Device num_H --> %d \n", dev_workspace->num_H );
#endif

                Cuda_Allocate_HBond_List( system->N, dev_workspace->num_H, dev_workspace->hbond_index, 
                        hb_top, (dev_lists+HBONDS) );
                num_hbonds = hb_top[system->N-1];
                system->num_hbonds = num_hbonds;

#ifdef __CUDA_MEM__
                fprintf (stderr, "Device memory allocated: Hydrogen Bonds list: %ld (MB) \n", 
                        sizeof (hbond_data) * num_hbonds / (1024*1024));
#endif

#ifdef __DEBUG_CUDA__
                fprintf (stderr, "Device Total number of HBonds --> %d \n", num_hbonds );
#endif

                free (hbond_index);
            }

            // bonds list 
            Cuda_Allocate_Bond_List( system->N, bond_top, dev_lists+BONDS );
            num_bonds = bond_top[system->N-1];
            system->num_bonds = num_bonds;

#ifdef __CUDA_MEM__
            fprintf (stderr, "Device memory allocated: Bonds list: %ld (MB) \n", 
                    sizeof (bond_data) * num_bonds / (1024*1024));
#endif

#ifdef __DEBUG_CUDA__
            fprintf (stderr, "Device Total Bonds --> %d \n", num_bonds );
#endif

            //    system->max_thb_intrs = num_3body;
            // 3bodies list 
            //if(!Make_List(num_bonds, num_bonds * MAX_THREE_BODIES, TYP_THREE_BODY, dev_lists + THREE_BODIES, TYP_DEVICE)) {
            //  fprintf( stderr, "Problem in initializing angles list. Terminating!\n" );
            //  exit( INIT_ERR );
            //}

            //fprintf( stderr, "***memory allocated: three_body = %ldMB\n", 
            //   num_bonds * MAX_THREE_BODIES *sizeof(three_body_interaction_data) / (1024*1024) );
            //fprintf (stderr, "size of (three_body_interaction_data) : %d \n", sizeof (three_body_interaction_data));

            //Free local resources
            free (output);
            free (nbrs_indices);
        }


        void Init_Lists( reax_system *system, control_params *control, 
                simulation_data *data, static_storage *workspace, 
                list **lists, output_controls *out_control )
        {
            int i, num_nbrs, num_hbonds, num_bonds, num_3body, Htop;
            int *hb_top, *bond_top;

            real t_start, t_elapsed;

            num_nbrs = Estimate_NumNeighbors( system, control, workspace, lists );

#ifdef __DEBUG_CUDA__
            fprintf (stderr, "Serial NumNeighbors ---> %d \n", num_nbrs);
#endif

            if( !Make_List(system->N, num_nbrs, TYP_FAR_NEIGHBOR, (*lists)+FAR_NBRS) ) {
                fprintf(stderr, "Problem in initializing far nbrs list. Terminating!\n");
                exit( INIT_ERR );
            }
#if defined(DEBUG_FOCUS)
            fprintf( stderr, "memory allocated: far_nbrs = %ldMB\n", 
                    num_nbrs * sizeof(far_neighbor_data) / (1024*1024) );
#endif

            t_start = Get_Time ();
            Generate_Neighbor_Lists(system,control,data,workspace,lists,out_control);
            t_elapsed = Get_Timing_Info ( t_start );

#ifdef __DEBUG_CUDA__
            fprintf (stderr, " Timing Generate Neighbors %lf \n", t_elapsed );
#endif

            Htop = 0;
            hb_top = (int*) calloc( system->N, sizeof(int) );
            bond_top = (int*) calloc( system->N, sizeof(int) );
            num_3body = 0;
            Estimate_Storage_Sizes( system, control, lists, 
                    &Htop, hb_top, bond_top, &num_3body );

            Allocate_Matrix( &(workspace->H), system->N, Htop );
#if defined(DEBUG_FOCUS)
            fprintf( stderr, "estimated storage - Htop: %d\n", Htop );
            fprintf( stderr, "memory allocated: H = %ldMB\n", 
                    Htop * sizeof(sparse_matrix_entry) / (1024*1024) );
#endif

            workspace->num_H = 0;
            if( control->hb_cut > 0 ) {
                /* init H indexes */
                for( i = 0; i < system->N; ++i )
                    if( system->reaxprm.sbp[ system->atoms[i].type ].p_hbond == 1 ) // H atom
                        workspace->hbond_index[i] = workspace->num_H++;
                    else workspace->hbond_index[i] = -1;

                Allocate_HBond_List( system->N, workspace->num_H, workspace->hbond_index, 
                        hb_top, (*lists)+HBONDS );
                num_hbonds = hb_top[system->N-1];

#ifdef __DEBUG_CUDA__
                fprintf( stderr, "Serial num_hbonds: %d\n", num_hbonds );
#endif

#if defined(DEBUG_FOCUS)
                fprintf( stderr, "estimated storage - num_hbonds: %d\n", num_hbonds );
                fprintf( stderr, "memory allocated: hbonds = %ldMB\n", 
                        num_hbonds * sizeof(hbond_data) / (1024*1024) );
#endif
            }

            /* bonds list */
            Allocate_Bond_List( system->N, bond_top, (*lists)+BONDS );
            num_bonds = bond_top[system->N-1];
#if defined(DEBUG_FOCUS)
            fprintf( stderr, "estimated storage - num_bonds: %d\n", num_bonds );
            fprintf( stderr, "memory allocated: bonds = %ldMB\n", 
                    num_bonds * sizeof(bond_data) / (1024*1024) );
#endif

#ifdef __DEBUG_CUDA__
            fprintf (stderr, " host num_3body : %d \n", num_3body);
            fprintf (stderr, " host num_bonds : %d \n", num_bonds);
#endif

            /* 3bodies list */
            if(!Make_List(num_bonds, num_3body, TYP_THREE_BODY, (*lists)+THREE_BODIES)) {
                fprintf( stderr, "Problem in initializing angles list. Terminating!\n" );
                exit( INIT_ERR );
            }
#if defined(DEBUG_FOCUS)
            fprintf( stderr, "estimated storage - num_3body: %d\n", num_3body );
            fprintf( stderr, "memory allocated: 3-body = %ldMB\n", 
                    num_3body * sizeof(three_body_interaction_data) / (1024*1024) );
#endif
#ifdef TEST_FORCES
            if(!Make_List( system->N, num_bonds * 8, TYP_DDELTA, (*lists) + DDELTA )) {
                fprintf( stderr, "Problem in initializing dDelta list. Terminating!\n" );
                exit( INIT_ERR );
            }

            if( !Make_List( num_bonds, num_bonds*MAX_BONDS*3, TYP_DBO, (*lists)+DBO ) ) {
                fprintf( stderr, "Problem in initializing dBO list. Terminating!\n" );
                exit( INIT_ERR );
            }
#endif

            free( hb_top );
            free( bond_top );
        }


        void Init_Out_Controls(reax_system *system, control_params *control, 
                static_storage *workspace, output_controls *out_control)
        {
            char temp[1000];

            /* Init trajectory file */
            if( out_control->write_steps > 0 ) { 
                strcpy( temp, control->sim_name );
                strcat( temp, ".trj" );
                out_control->trj = fopen( temp, "w" );
                out_control->write_header( system, control, workspace, out_control );
            }

            if( out_control->energy_update_freq > 0 ) {
                /* Init out file */
                strcpy( temp, control->sim_name );
                strcat( temp, ".out" );
                out_control->out = fopen( temp, "w" );
                fprintf( out_control->out, "%-6s%16s%16s%16s%11s%11s%13s%13s%13s\n",
                        "step", "total energy", "poten. energy", "kin. energy", 
                        "temp.", "target", "volume", "press.", "target" );
                fflush( out_control->out );

                /* Init potentials file */
                strcpy( temp, control->sim_name );
                strcat( temp, ".pot" );
                out_control->pot = fopen( temp, "w" );
                fprintf( out_control->pot, 
                        "%-6s%13s%13s%13s%13s%13s%13s%13s%13s%13s%13s%13s\n",
                        "step", "ebond", "eatom", "elp", "eang", "ecoa", "ehb", 
                        "etor", "econj", "evdw","ecoul", "epol" );
                fflush( out_control->pot );

                /* Init log file */
                strcpy( temp, control->sim_name );
                strcat( temp, ".log" );
                out_control->log = fopen( temp, "w" );
                fprintf( out_control->log, "%-6s%10s%10s%10s%10s%10s%10s%10s\n", 
                        "step", "total", "neighbors", "init", "bonded", 
                        "nonbonded", "QEq", "matvec" );
            }

            /* Init pressure file */
            if( control->ensemble == NPT || 
                    control->ensemble == iNPT || 
                    control->ensemble == sNPT ) {
                strcpy( temp, control->sim_name );
                strcat( temp, ".prs" );
                out_control->prs = fopen( temp, "w" );
                fprintf( out_control->prs, "%-6s%13s%13s%13s%13s%13s%13s%13s%13s\n",
                        "step", "norm_x", "norm_y", "norm_z", 
                        "press_x", "press_y", "press_z", "target_p", "volume" );
                fflush( out_control->prs );
            }

            /* Init molecular analysis file */
            if( control->molec_anal ) {
                sprintf( temp, "%s.mol", control->sim_name );
                out_control->mol = fopen( temp, "w" );
                if( control->num_ignored ) {
                    sprintf( temp, "%s.ign", control->sim_name );
                    out_control->ign = fopen( temp, "w" );
                } 
            }

            /* Init electric dipole moment analysis file */
            if( control->dipole_anal ) {
                strcpy( temp, control->sim_name );
                strcat( temp, ".dpl" );
                out_control->dpl = fopen( temp, "w" );
                fprintf( out_control->dpl, 
                        "Step      Molecule Count  Avg. Dipole Moment Norm\n" );
                fflush( out_control->dpl );
            }

            /* Init diffusion coef analysis file */
            if( control->diffusion_coef ) {
                strcpy( temp, control->sim_name );
                strcat( temp, ".drft" );
                out_control->drft = fopen( temp, "w" );
                fprintf( out_control->drft, "Step     Type Count   Avg Squared Disp\n" );
                fflush( out_control->drft );
            }


#ifdef TEST_ENERGY
            /* open bond energy file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".ebond" );
            out_control->ebond = fopen( temp, "w" );

            /* open lone-pair energy file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".elp" );
            out_control->elp = fopen( temp, "w" );

            /* open overcoordination energy file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".eov" );
            out_control->eov = fopen( temp, "w" );

            /* open undercoordination energy file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".eun" );
            out_control->eun = fopen( temp, "w" );

            /* open angle energy file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".eval" );
            out_control->eval = fopen( temp, "w" );

            /* open penalty energy file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".epen" );
            out_control->epen = fopen( temp, "w" );

            /* open coalition energy file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".ecoa" );
            out_control->ecoa = fopen( temp, "w" );

            /* open hydrogen bond energy file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".ehb" );
            out_control->ehb = fopen( temp, "w" );

            /* open torsion energy file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".etor" );
            out_control->etor = fopen( temp, "w" );

            /* open conjugation energy file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".econ" );
            out_control->econ = fopen( temp, "w" );

            /* open vdWaals energy file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".evdw" );
            out_control->evdw = fopen( temp, "w" );

            /* open coulomb energy file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".ecou" );
            out_control->ecou = fopen( temp, "w" );
#endif


#ifdef TEST_FORCES
            /* open bond orders file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".fbo" );
            out_control->fbo = fopen( temp, "w" );

            /* open bond orders derivatives file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".fdbo" );
            out_control->fdbo = fopen( temp, "w" );

            /* open bond forces file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".fbond" );
            out_control->fbond = fopen( temp, "w" );

            /* open lone-pair forces file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".flp" );
            out_control->flp = fopen( temp, "w" );

            /* open overcoordination forces file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".fatom" );
            out_control->fatom = fopen( temp, "w" );

            /* open angle forces file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".f3body" );
            out_control->f3body = fopen( temp, "w" );

            /* open hydrogen bond forces file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".fhb" );
            out_control->fhb = fopen( temp, "w" );

            /* open torsion forces file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".f4body" );
            out_control->f4body = fopen( temp, "w" );

            /* open nonbonded forces file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".fnonb" );
            out_control->fnonb = fopen( temp, "w" );

            /* open total force file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".ftot" );
            out_control->ftot = fopen( temp, "w" );

            /* open coulomb forces file */
            strcpy( temp, control->sim_name );
            strcat( temp, ".ftot2" );
            out_control->ftot2 = fopen( temp, "w" );
#endif


            /* Error handling */
            /* if ( out_control->out == NULL || out_control->pot == NULL || 
               out_control->log == NULL || out_control->mol == NULL || 
               out_control->dpl == NULL || out_control->drft == NULL ||       
               out_control->pdb == NULL )
               {
               fprintf( stderr, "FILE OPEN ERROR. TERMINATING..." );
               exit( CANNOT_OPEN_OUTFILE );
               }*/
        }


        void Initialize(reax_system *system, control_params *control, 
                simulation_data *data, static_storage *workspace, list **lists, 
                output_controls *out_control, evolve_function *Evolve)
        {
            Randomize();

            Init_System( system, control, data );

            Init_Simulation_Data( system, control, data, out_control, Evolve );

            Init_Workspace( system, control, workspace );

            Init_Lists( system, control, data, workspace, lists, out_control );

            Init_Out_Controls( system, control, workspace, out_control );

            /* These are done in forces.c, only forces.c can see all those functions */
            Init_Bonded_Force_Functions( control );
#ifdef TEST_FORCES
            Init_Force_Test_Functions( );
#endif

            if( control->tabulate )
                Make_LR_Lookup_Table( system, control );

#if defined(DEBUG_FOCUS)
            fprintf( stderr, "data structures have been initialized...\n" ); 
#endif
        }

        void Cuda_Initialize(reax_system *system, control_params *control, 
                simulation_data *data, static_storage *workspace, list **lists, 
                output_controls *out_control, evolve_function *Evolve)
        {
            Randomize ();

            Cuda_Init_Scratch ();

            //System
            Cuda_Init_System (system);
            Sync_Host_Device ( system, cudaMemcpyHostToDevice );
            Cuda_Init_System (system, control, data );

            //Simulation Data
            copy_host_device (system->atoms, system->d_atoms, REAX_ATOM_SIZE * system->N , 
                    cudaMemcpyHostToDevice, RES_SYSTEM_ATOMS );
            Cuda_Init_Simulation_Data (data);
            //Sync_Host_Device (data, (simulation_data *)data->d_simulation_data, cudaMemcpyHostToDevice);
            Cuda_Init_Simulation_Data( system, control, data, out_control, Evolve );
            Sync_Host_Device (data, (simulation_data *)data->d_simulation_data, cudaMemcpyHostToDevice);

            //static storage
            Cuda_Init_Workspace_System ( system, dev_workspace );
            Cuda_Init_Workspace ( system, control, dev_workspace );
            Cuda_Init_Workspace_Device (workspace);

            //control
            Cuda_Init_Control (control);

            //Grid
            Cuda_Init_Grid (&system->g, &system->d_g );

            //lists
            Cuda_Init_Lists (system, control, data, workspace, lists, out_control );

            Init_Out_Controls( system, control, workspace, out_control );

            if( control->tabulate ) {
                real start, end;
                start = Get_Time ();
                Make_LR_Lookup_Table( system, control );
                copy_LR_table_to_device (system, control );
                end = Get_Timing_Info ( start );

#ifdef __DEBUG_CUDA__
                fprintf (stderr, "Done copying the LR table to the device ---> %f \n", end );
#endif
            }
        }
