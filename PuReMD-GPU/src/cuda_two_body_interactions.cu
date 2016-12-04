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

#include "cuda_two_body_interactions.h"

#include "bond_orders.h"
#include "list.h"
#include "lookup.h"
#include "vector.h"
#include "index_utils.h"

#include "cuda_helpers.h"


GLOBAL void Cuda_Bond_Energy ( reax_atom *atoms, global_parameters g_params, 
        single_body_parameters *sbp, two_body_parameters *tbp, 
        simulation_data *data,
        static_storage p_workspace, list p_bonds, 
        int N, int num_atom_types, real *E_BE)
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    real ebond, pow_BOs_be2, exp_be12, CEbo;
    real gp3, gp4, gp7, gp10, gp37;
    real exphu, exphua1, exphub1, exphuov, hulpov, estriph;
    real decobdbo, decobdboua, decobdboub;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    bond_order_data *bo_ij;
    list *bonds;
    static_storage *workspace;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= N ) return;

    bonds = &p_bonds;
    workspace = &p_workspace;

    gp3 = g_params.l[3];
    gp4 = g_params.l[4];
    gp7 = g_params.l[7];
    gp10 = g_params.l[10];
    gp37 = (int) g_params.l[37];

    //for( i=0; i < system->N; ++i )
    start_i = Start_Index(i, bonds);
    end_i = End_Index(i, bonds);
    //fprintf( stderr, "i=%d start=%d end=%d\n", i, start_i, end_i );
    for( pj = start_i; pj < end_i; ++pj )
    {
        //TODO
        //if( i < bonds->select.bond_list[pj].nbr ) 
        if( i < bonds->select.bond_list[pj].nbr ) 
        {
            //TODO
            /* set the pointers */
            j = bonds->select.bond_list[pj].nbr;
            type_i = atoms[i].type;
            type_j = atoms[j].type;
            sbp_i = &( sbp[type_i] );
            sbp_j = &( sbp[type_j] );
            twbp = &( tbp[ index_tbp(type_i,type_j,num_atom_types) ] );
            bo_ij = &( bonds->select.bond_list[pj].bo_data );

            /* calculate the constants */
            pow_BOs_be2 = POW( bo_ij->BO_s, twbp->p_be2 );
            exp_be12 = EXP( twbp->p_be1 * ( 1.0 - pow_BOs_be2 ) );
            CEbo = -twbp->De_s * exp_be12 * 
                ( 1.0 - twbp->p_be1 * twbp->p_be2 * pow_BOs_be2 );

            /* calculate the Bond Energy */
            ebond = 
                -twbp->De_s * bo_ij->BO_s * exp_be12 
                -twbp->De_p * bo_ij->BO_pi 
                -twbp->De_pp * bo_ij->BO_pi2;

            //PERFORMANCE IMAPCT
            //MYATOMICADD(&data->E_BE, ebond);
            //TODO
            //E_BE [ i ] += ebond/2.0;
            E_BE [ i ] += ebond;
            //data->E_BE += ebond;

            /* calculate derivatives of Bond Orders */
            bo_ij->Cdbo += CEbo;
            bo_ij->Cdbopi -= (CEbo + twbp->De_p);
            bo_ij->Cdbopi2 -= (CEbo + twbp->De_pp);

#ifdef TEST_ENERGY
            //TODO
            //fprintf( out_control->ebond, "%6d%6d%24.15e%24.15e\n", 
            //     workspace->orig_id[i], workspace->orig_id[j], 
            // i+1, j+1, 
            //     bo_ij->BO, ebond/*, data->E_BE*/ );
            /*
               fprintf( out_control->ebond, "%6d%6d%12.6f%12.6f%12.6f\n", 
               workspace->orig_id[i], workspace->orig_id[j], 
               CEbo, -twbp->De_p, -twbp->De_pp );*/
#endif
#ifdef TEST_FORCES
            //TODO
            /*
               Add_dBO( system, lists, i, pj, CEbo, workspace->f_be );
               Add_dBOpinpi2( system, lists, i, pj, 
               -(CEbo + twbp->De_p), -(CEbo + twbp->De_pp), 
               workspace->f_be, workspace->f_be );
             */
            //TODO
#endif

            /* Stabilisation terminal triple bond */
            if( bo_ij->BO >= 1.00 ) {
                if( gp37 == 2 ||
                        (sbp_i->mass == 12.0000 && sbp_j->mass == 15.9990) || 
                        (sbp_j->mass == 12.0000 && sbp_i->mass == 15.9990) ) {
                    // ba = SQR(bo_ij->BO - 2.50);
                    exphu = EXP( -gp7 * SQR(bo_ij->BO - 2.50) );
                    //oboa=abo(j1)-boa;
                    //obob=abo(j2)-boa;
                    exphua1 = EXP(-gp3*(workspace->total_bond_order[i]-bo_ij->BO));
                    exphub1 = EXP(-gp3*(workspace->total_bond_order[j]-bo_ij->BO));
                    //ovoab=abo(j1)-aval(it1)+abo(j2)-aval(it2);
                    exphuov = EXP(gp4*(workspace->Delta[i] + workspace->Delta[j]));
                    hulpov = 1.0 / (1.0 + 25.0 * exphuov);

                    estriph = gp10 * exphu * hulpov * (exphua1 + exphub1);
                    //estrain(j1) = estrain(j1) + 0.50*estriph;
                    //estrain(j2) = estrain(j2) + 0.50*estriph;

                    //PERFORMANCE IMPACT
                    //MYATOMICADD(&data->E_BE, estriph);
                    E_BE [ i] += estriph;
                    //data->E_BE += estriph;

                    decobdbo = gp10 * exphu * hulpov * (exphua1 + exphub1) * 
                        ( gp3 - 2.0 * gp7 * (bo_ij->BO-2.50) );
                    decobdboua = -gp10 * exphu * hulpov * 
                        (gp3*exphua1 + 25.0*gp4*exphuov*hulpov*(exphua1+exphub1));
                    decobdboub = -gp10 * exphu * hulpov * 
                        (gp3*exphub1 + 25.0*gp4*exphuov*hulpov*(exphua1+exphub1));

                    bo_ij->Cdbo += decobdbo;

                    //PERFORMANCE IMAPCT
                    workspace->CdDelta[i] += decobdboua;
                    //MYATOMICADD(&workspace->CdDelta[j], decobdboub);
                    //CdDelta [ i * N + i ] += decobdboua;
                    //CdDelta [ i * N + j ] += decobdboua;
                    //workspace->CdDelta [i] += decobdboua;
                    //workspace->CdDelta [j] += decobdboub;

#ifdef TEST_ENERGY
                    /*
                       fprintf( out_control->ebond, 
                       "%6d%6d%24.15e%24.15e%24.15e%24.15e\n",
                       workspace->orig_id[i], workspace->orig_id[j],
                    //i+1, j+1, 
                    estriph, decobdbo, decobdboua, decobdboub );
                     */
#endif
#ifdef TEST_FORCES
                    /*
                       Add_dBO( system, lists, i, pj, decobdbo, workspace->f_be );
                       Add_dDelta( system, lists, i, decobdboua, workspace->f_be );
                       Add_dDelta( system, lists, j, decobdboub, workspace->f_be );
                     */
#endif
                }
            }
        }
    } //TODO commented out the if statement for processing i < j. 
    // we process all teh bonds and add only half the energy
}


DEVICE void LR_vdW_Coulomb(global_parameters g_params, two_body_parameters *tbp,
        control_params *control, int i, int j, real r_ij, LR_data *lr, int num_atom_types )
{
    real p_vdW1 = g_params.l[28];
    real p_vdW1i = 1.0 / p_vdW1;
    real powr_vdW1, powgi_vdW1;
    real tmp, fn13, exp1, exp2;
    real Tap, dTap, dfn13;
    real dr3gamij_1, dr3gamij_3;
    real e_core, de_core;
    two_body_parameters *twbp;

    twbp = &(tbp[ index_tbp (i, j, num_atom_types) ]);
    e_core = 0;
    de_core = 0;

    /* calculate taper and its derivative */
    Tap = control->Tap7 * r_ij + control->Tap6;
    Tap = Tap * r_ij + control->Tap5;
    Tap = Tap * r_ij + control->Tap4;
    Tap = Tap * r_ij + control->Tap3;
    Tap = Tap * r_ij + control->Tap2;
    Tap = Tap * r_ij + control->Tap1;
    Tap = Tap * r_ij + control->Tap0;

    dTap = 7 * control->Tap7 * r_ij + 6 * control->Tap6;
    dTap = dTap * r_ij + 5 * control->Tap5;
    dTap = dTap * r_ij + 4 * control->Tap4;
    dTap = dTap * r_ij + 3 * control->Tap3;
    dTap = dTap * r_ij + 2 * control->Tap2;
    dTap += control->Tap1 / r_ij;


    /* vdWaals calculations */
    powr_vdW1 = POW(r_ij, p_vdW1);
    powgi_vdW1 = POW( 1.0 / twbp->gamma_w, p_vdW1);

    fn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i );
    exp1 = EXP( twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );
    exp2 = EXP( 0.5 * twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );

    lr->e_vdW = Tap * twbp->D * (exp1 - 2.0 * exp2);
    /* fprintf(stderr,"vdW: Tap:%f, r: %f, f13:%f, D:%f, Energy:%f,\
       Gamma_w:%f, p_vdw: %f, alpha: %f, r_vdw: %f, %lf %lf\n",
       Tap, r_ij, fn13, twbp->D, Tap * twbp->D * (exp1 - 2.0 * exp2),
       powgi_vdW1, p_vdW1, twbp->alpha, twbp->r_vdW, exp1, exp2); */

    dfn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i - 1.0) * POW(r_ij, p_vdW1 - 2.0);

    lr->CEvd = dTap * twbp->D * (exp1 - 2 * exp2) -
               Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * (exp1 - exp2) * dfn13;

    /*vdWaals Calculations*/
    if (g_params.vdw_type == 1 || g_params.vdw_type == 3)
    {
        // shielding
        powr_vdW1 = POW(r_ij, p_vdW1);
        powgi_vdW1 = POW( 1.0 / twbp->gamma_w, p_vdW1);

        fn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i );
        exp1 = EXP( twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );
        exp2 = EXP( 0.5 * twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );

        lr->e_vdW = Tap * twbp->D * (exp1 - 2.0 * exp2);

        dfn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i - 1.0) *
                POW(r_ij, p_vdW1 - 2.0);

        lr->CEvd = dTap * twbp->D * (exp1 - 2.0 * exp2) -
                   Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * (exp1 - exp2) * dfn13;
    }
    else  // no shielding
    {
        exp1 = EXP( twbp->alpha * (1.0 - r_ij / twbp->r_vdW) );
        exp2 = EXP( 0.5 * twbp->alpha * (1.0 - r_ij / twbp->r_vdW) );

        lr->e_vdW = Tap * twbp->D * (exp1 - 2.0 * exp2);

        lr->CEvd = dTap * twbp->D * (exp1 - 2.0 * exp2) -
                   Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * (exp1 - exp2);
    }

    if (g_params.vdw_type == 2 || g_params.vdw_type == 3)
    {
        // innner wall
        e_core = twbp->ecore * EXP(twbp->acore * (1.0 - (r_ij / twbp->rcore)));
        lr->e_vdW += Tap * e_core;

        de_core = -(twbp->acore / twbp->rcore) * e_core;
        lr->CEvd += dTap * e_core + Tap * de_core;
    }

    /* Coulomb calculations */
    dr3gamij_1 = ( r_ij * r_ij * r_ij + twbp->gamma );
    dr3gamij_3 = POW( dr3gamij_1 , 0.33333333333333 );

    tmp = Tap / dr3gamij_3;
    lr->H = EV_to_KCALpMOL * tmp;
    lr->e_ele = C_ele * tmp;
    /* fprintf( stderr,"i:%d(%d), j:%d(%d), gamma:%f,\
       Tap:%f, dr3gamij_3:%f, qi: %f, qj: %f\n",
       i, system->atoms[i].type, j, system->atoms[j].type,
       twbp->gamma, Tap, dr3gamij_3,
       system->atoms[i].q, system->atoms[j].q ); */

    lr->CEclmb = C_ele * ( dTap -  Tap * r_ij / dr3gamij_1 ) / dr3gamij_3;
    /* fprintf( stdout, "%d %d\t%g\t%g  %g\t%g  %g\t%g  %g\n",
       i+1, j+1, r_ij, e_vdW, CEvd * r_ij,
       system->atoms[i].q, system->atoms[j].q, e_ele, CEclmb * r_ij ); */

    /* fprintf( stderr,"LR_Lookup:%3d%3d%5.3f-%8.5f,%8.5f%8.5f,%8.5f%8.5f\n",
       i, j, r_ij, lr->H, lr->e_vdW, lr->CEvd, lr->e_ele, lr->CEclmb ); */
}



/*

   GLOBAL void Cuda_vdW_Coulomb_Energy( reax_atom *atoms,     
   two_body_parameters *tbp,
   global_parameters g_p,
   control_params *control, 
   simulation_data *data,  
   list p_far_nbrs, 
   real *E_vdW, real *E_Ele, rvec *aux_ext_press, 
   int num_atom_types, int N )
   {
   int  i, j, pj;
   int  start_i, end_i;
   real self_coef;
   real p_vdW1, p_vdW1i;
   real powr_vdW1, powgi_vdW1;
   real tmp, r_ij, fn13, exp1, exp2;
   real Tap, dTap, dfn13, CEvd, CEclmb;
   real dr3gamij_1, dr3gamij_3;
   real e_ele, e_vdW, e_core, de_core;
   rvec temp, ext_press;
// rtensor temp_rtensor, total_rtensor;
two_body_parameters *twbp;
far_neighbor_data *nbr_pj;
list *far_nbrs = &p_far_nbrs;

i = blockIdx.x * blockDim.x + threadIdx.x;
if ( i >= N ) return;

p_vdW1 = g_p.l[28];
p_vdW1i = 1.0 / p_vdW1;
e_ele = 0;
e_vdW = 0;
e_core = 0;
de_core = 0;

//for( i = 0; i < system->N; ++i ) {
start_i = Start_Index(i, far_nbrs);
end_i   = End_Index(i, far_nbrs);
// fprintf( stderr, "i: %d, start: %d, end: %d\n",
//     i, start_i, end_i );

for( pj = start_i; pj < end_i; ++pj )
if( far_nbrs->select.far_nbr_list[pj].d <= control->r_cut ) {
nbr_pj = &( far_nbrs->select.far_nbr_list[pj] );
j = nbr_pj->nbr;
r_ij = nbr_pj->d;
twbp = &(tbp[ index_tbp(atoms[i].type, atoms[j].type, num_atom_types) ]);
self_coef = (i == j) ? 0.5 : 1.0; // for supporting small boxes!

//CHANGE ORIGINAL
//if (i <= j) continue;
//CHANGE ORIGINAL

// Calculate Taper and its derivative 
// Tap = nbr_pj->Tap;   -- precomputed during compte_H
Tap = control->Tap7 * r_ij + control->Tap6;
Tap = Tap * r_ij + control->Tap5;
Tap = Tap * r_ij + control->Tap4;
Tap = Tap * r_ij + control->Tap3;
Tap = Tap * r_ij + control->Tap2;
Tap = Tap * r_ij + control->Tap1;
Tap = Tap * r_ij + control->Tap0;

dTap = 7*control->Tap7 * r_ij + 6*control->Tap6;
dTap = dTap * r_ij + 5*control->Tap5;
dTap = dTap * r_ij + 4*control->Tap4;
dTap = dTap * r_ij + 3*control->Tap3;
dTap = dTap * r_ij + 2*control->Tap2;
dTap += control->Tap1/r_ij;

//vdWaals Calculations
if(g_p.vdw_type==1 || g_p.vdw_type==3) {
    // shielding
    powr_vdW1 = POW(r_ij, p_vdW1);
    powgi_vdW1 = POW( 1.0 / twbp->gamma_w, p_vdW1);

    fn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i );
    exp1 = EXP( twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );
    exp2 = EXP( 0.5 * twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );

    e_vdW = self_coef * Tap * twbp->D * (exp1 - 2.0 * exp2);        
    E_vdW [i] += e_vdW / 2.0;

    dfn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i - 1.0) * 
        POW(r_ij, p_vdW1 - 2.0);

    CEvd = self_coef * ( dTap * twbp->D * (exp1 - 2 * exp2) - 
            Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * 
            (exp1 - exp2) * dfn13 );
}
else{ // no shielding
    exp1 = EXP( twbp->alpha * (1.0 - r_ij / twbp->r_vdW) );
    exp2 = EXP( 0.5 * twbp->alpha * (1.0 - r_ij / twbp->r_vdW) );

    e_vdW = self_coef * Tap * twbp->D * (exp1 - 2.0 * exp2);        
    E_vdW [i] += e_vdW / 2.0;

    CEvd = self_coef * ( dTap * twbp->D * (exp1 - 2.0 * exp2) - 
            Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * 
            (exp1 - exp2) );
}

if(g_p.vdw_type==2 || g_p.vdw_type==3) {
    // innner wall
    e_core = twbp->ecore * EXP(twbp->acore * (1.0-(r_ij/twbp->rcore)));
    e_vdW = self_coef * Tap * e_core;

    //TODO check this
    E_vdW [i] += e_vdW / 2.0;
    //TODO check this

    de_core = -(twbp->acore/twbp->rcore) * e_core;
    CEvd += self_coef * ( dTap * e_core + Tap * de_core );
}

//Coulomb Calculations
dr3gamij_1 = ( r_ij * r_ij * r_ij + twbp->gamma );
dr3gamij_3 = POW( dr3gamij_1 , 0.33333333333333 );

tmp = Tap / dr3gamij_3;
//tmp = Tap * nbr_pj->inv_dr3gamij_3; -- precomputed during compte_H
e_ele = 
self_coef * C_ele * atoms[i].q * atoms[j].q * tmp;
E_Ele [i] += e_ele / 2.0;

CEclmb = self_coef * C_ele * atoms[i].q * atoms[j].q *
( dTap -  Tap * r_ij / dr3gamij_1 ) / dr3gamij_3;
//CEclmb = self_coef*C_ele*system->atoms[i].q*system->atoms[j].q* 
// ( dTap- Tap*r_ij*nbr_pj->inv_dr3gamij_1 )*nbr_pj->inv_dr3gamij_3;

if( control->ensemble == NVE || control->ensemble == NVT || control->ensemble == bNVT) {
    if (i >= j)
        rvec_ScaledAdd( atoms[i].f, -(CEvd+CEclmb), nbr_pj->dvec );
    else
        rvec_ScaledAdd( atoms[i].f, +(CEvd+CEclmb), nbr_pj->dvec );
}
else { // NPT, iNPT or sNPT
    // for pressure coupling, terms not related to bond order 
    //  derivatives are added directly into pressure vector/tensor 
    rvec_Scale( temp, CEvd + CEclmb, nbr_pj->dvec );

    if ( i >= j)
        rvec_ScaledAdd( atoms[i].f, -1., temp );
    else
        rvec_Add( atoms[i].f, temp );

    rvec_iMultiply( ext_press, nbr_pj->rel_box, temp );

    //rvec_Add( data->ext_press, ext_press );
    rvec_Copy (aux_ext_press[i], ext_press);

    //TODO CHECK THIS calculation here, it should be divided by two somehow.
}
}
//}
}

*/


GLOBAL void Cuda_vdW_Coulomb_Energy( reax_atom *atoms,     
        two_body_parameters *tbp,
        global_parameters g_p,
        control_params *control, 
        simulation_data *data,  
        list p_far_nbrs, 
        real *E_vdW, real *E_Ele, rvec *aux_ext_press, 
        int num_atom_types, int N )
{
    extern __shared__ real _vdw[];
    extern __shared__ real _ele[];
    extern __shared__ rvec _force [];

    real *sh_vdw;
    real *sh_ele;
    rvec *sh_force;

    int  i, j, pj;
    int  start_i, end_i;
    real self_coef;
    real p_vdW1, p_vdW1i;
    real powr_vdW1, powgi_vdW1;
    real tmp, r_ij, fn13, exp1, exp2;
    real Tap, dTap, dfn13, CEvd, CEclmb;
    real dr3gamij_1, dr3gamij_3;
    real e_ele, e_vdW, e_core, de_core;
    rvec temp, ext_press;
    // rtensor temp_rtensor, total_rtensor;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    list *far_nbrs = &p_far_nbrs;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warpid = thread_id / VDW_THREADS_PER_ATOM;
    int laneid = thread_id & (VDW_THREADS_PER_ATOM -1);

    i = warpid;

    sh_vdw = _vdw;
    sh_ele = _vdw + blockDim.x;
    sh_force = (rvec *)( _vdw + 2*blockDim.x);

    sh_vdw[threadIdx.x] = 0.0; 
    sh_ele[threadIdx.x] = 0.0; 
    rvec_MakeZero ( sh_force [threadIdx.x] );

    if (i < N)
    {

        p_vdW1 = g_p.l[28];
        p_vdW1i = 1.0 / p_vdW1;
        e_ele = 0;
        e_vdW = 0;
        e_core = 0;
        de_core = 0;

        //for( i = 0; i < system->N; ++i ) {
        start_i = Start_Index(i, far_nbrs);
        end_i   = End_Index(i, far_nbrs);
        // fprintf( stderr, "i: %d, start: %d, end: %d\n",
        //     i, start_i, end_i );

        pj = start_i + laneid;
        //for( pj = start_i; pj < end_i; ++pj )
        while (pj < end_i)
        {
            if( far_nbrs->select.far_nbr_list[pj].d <= control->r_cut ) {
                nbr_pj = &( far_nbrs->select.far_nbr_list[pj] );
                j = nbr_pj->nbr;
                r_ij = nbr_pj->d;
                twbp = &(tbp[ index_tbp(atoms[i].type, atoms[j].type, num_atom_types) ]);
                self_coef = (i == j) ? 0.5 : 1.0; // for supporting small boxes!

                //CHANGE ORIGINAL
                //if (i <= j) continue;
                //CHANGE ORIGINAL

                // Calculate Taper and its derivative 
                // Tap = nbr_pj->Tap;   -- precomputed during compte_H
                Tap = control->Tap7 * r_ij + control->Tap6;
                Tap = Tap * r_ij + control->Tap5;
                Tap = Tap * r_ij + control->Tap4;
                Tap = Tap * r_ij + control->Tap3;
                Tap = Tap * r_ij + control->Tap2;
                Tap = Tap * r_ij + control->Tap1;
                Tap = Tap * r_ij + control->Tap0;

                dTap = 7*control->Tap7 * r_ij + 6*control->Tap6;
                dTap = dTap * r_ij + 5*control->Tap5;
                dTap = dTap * r_ij + 4*control->Tap4;
                dTap = dTap * r_ij + 3*control->Tap3;
                dTap = dTap * r_ij + 2*control->Tap2;
                dTap += control->Tap1/r_ij;

                //vdWaals Calculations
                if(g_p.vdw_type==1 || g_p.vdw_type==3) {
                    // shielding
                    powr_vdW1 = POW(r_ij, p_vdW1);
                    powgi_vdW1 = POW( 1.0 / twbp->gamma_w, p_vdW1);

                    fn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i );
                    exp1 = EXP( twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );
                    exp2 = EXP( 0.5 * twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );

                    e_vdW = self_coef * Tap * twbp->D * (exp1 - 2.0 * exp2);        


                    //E_vdW [i] += e_vdW / 2.0;
                    sh_vdw [threadIdx.x] += e_vdW/2.0;

                    dfn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i - 1.0) * 
                        POW(r_ij, p_vdW1 - 2.0);

                    CEvd = self_coef * ( dTap * twbp->D * (exp1 - 2 * exp2) - 
                            Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * 
                            (exp1 - exp2) * dfn13 );
                }
                else{ // no shielding
                    exp1 = EXP( twbp->alpha * (1.0 - r_ij / twbp->r_vdW) );
                    exp2 = EXP( 0.5 * twbp->alpha * (1.0 - r_ij / twbp->r_vdW) );

                    e_vdW = self_coef * Tap * twbp->D * (exp1 - 2.0 * exp2);        


                    //E_vdW [i] += e_vdW / 2.0;
                    sh_vdw [threadIdx.x] += e_vdW/2.0;

                    CEvd = self_coef * ( dTap * twbp->D * (exp1 - 2.0 * exp2) - 
                            Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * 
                            (exp1 - exp2) );
                }

                if(g_p.vdw_type==2 || g_p.vdw_type==3) {
                    // innner wall
                    e_core = twbp->ecore * EXP(twbp->acore * (1.0-(r_ij/twbp->rcore)));
                    e_vdW = self_coef * Tap * e_core;

                    //TODO check this
                    //E_vdW [i] += e_vdW / 2.0;
                    sh_vdw [threadIdx.x] += e_vdW / 2.0;
                    //TODO check this

                    de_core = -(twbp->acore/twbp->rcore) * e_core;
                    CEvd += self_coef * ( dTap * e_core + Tap * de_core );
                }

                //Coulomb Calculations
                dr3gamij_1 = ( r_ij * r_ij * r_ij + twbp->gamma );
                dr3gamij_3 = POW( dr3gamij_1 , 0.33333333333333 );

                tmp = Tap / dr3gamij_3;
                //tmp = Tap * nbr_pj->inv_dr3gamij_3; -- precomputed during compte_H
                e_ele = 
                    self_coef * C_ele * atoms[i].q * atoms[j].q * tmp;

                //E_Ele [i] += e_ele / 2.0;
                sh_ele [threadIdx.x] += e_ele / 2.0;

                CEclmb = self_coef * C_ele * atoms[i].q * atoms[j].q *
                    ( dTap -  Tap * r_ij / dr3gamij_1 ) / dr3gamij_3;
                //CEclmb = self_coef*C_ele*system->atoms[i].q*system->atoms[j].q* 
                // ( dTap- Tap*r_ij*nbr_pj->inv_dr3gamij_1 )*nbr_pj->inv_dr3gamij_3;

                if( control->ensemble == NVE || control->ensemble == NVT || control->ensemble == bNVT) {
                    if (i >= j){
                        //rvec_ScaledAdd( atoms[i].f, -(CEvd+CEclmb), nbr_pj->dvec );
                        rvec_ScaledAdd( sh_force[threadIdx.x], -(CEvd+CEclmb), nbr_pj->dvec );
                    }
                    else
                    {
                        //rvec_ScaledAdd( atoms[i].f, +(CEvd+CEclmb), nbr_pj->dvec );
                        rvec_ScaledAdd( sh_force[threadIdx.x], +(CEvd+CEclmb), nbr_pj->dvec );
                    }
                }
                else { // NPT, iNPT or sNPT
                    // for pressure coupling, terms not related to bond order 
                    //  derivatives are added directly into pressure vector/tensor 
                    rvec_Scale( temp, CEvd + CEclmb, nbr_pj->dvec );

                    if ( i >= j)
                    {
                        //rvec_ScaledAdd( atoms[i].f, -1., temp );
                        rvec_ScaledAdd( sh_force[threadIdx.x], -1., temp );
                    }
                    else
                    {
                        //rvec_Add( atoms[i].f, temp );
                        rvec_Add( sh_force[threadIdx.x], temp );
                    }

                    rvec_iMultiply( ext_press, nbr_pj->rel_box, temp );

                    //rvec_Add( data->ext_press, ext_press );
                    rvec_Copy (aux_ext_press[i], ext_press);

                    //TODO CHECK THIS calculation here, it should be divided by two somehow.
                }
            } // if condition for far neighbors


            pj += VDW_THREADS_PER_ATOM;

        } // end of while loop for pj < end_i condition
    } // if (i < N ) condition
    //}

    __syncthreads ();

    if (laneid < 16) {
        sh_vdw[threadIdx.x] += sh_vdw[threadIdx.x + 16];
        sh_ele[threadIdx.x] += sh_ele[threadIdx.x + 16];
        rvec_Add (sh_force [threadIdx.x], sh_force [threadIdx.x + 16] );
    }
    __syncthreads ();
    if (laneid < 8) {
        sh_vdw[threadIdx.x] += sh_vdw[threadIdx.x + 8];
        sh_ele[threadIdx.x] += sh_ele[threadIdx.x + 8];
        rvec_Add (sh_force [threadIdx.x], sh_force [threadIdx.x + 8] );
    }
    __syncthreads ();
    if (laneid < 4) {
        sh_vdw[threadIdx.x] += sh_vdw[threadIdx.x + 4];
        sh_ele[threadIdx.x] += sh_ele[threadIdx.x + 4];
        rvec_Add (sh_force [threadIdx.x], sh_force [threadIdx.x + 4] );
    }
    __syncthreads ();
    if (laneid < 2) {
        sh_vdw[threadIdx.x] += sh_vdw[threadIdx.x + 2];
        sh_ele[threadIdx.x] += sh_ele[threadIdx.x + 2];
        rvec_Add (sh_force [threadIdx.x], sh_force [threadIdx.x + 2] );
    }
    __syncthreads ();
    if (laneid < 1) {
        sh_vdw[threadIdx.x] += sh_vdw[threadIdx.x + 1];
        sh_ele[threadIdx.x] += sh_ele[threadIdx.x + 1];
        rvec_Add (sh_force [threadIdx.x], sh_force [threadIdx.x + 1] );
    }
    __syncthreads ();
    if (laneid == 0) {
        E_vdW [i] += sh_vdw[threadIdx.x];
        E_Ele [i] += sh_ele[threadIdx.x];
        rvec_Add (atoms[i].f, sh_force [ threadIdx.x ]);
    }
}


GLOBAL void Cuda_Tabulated_vdW_Coulomb_Energy(reax_atom *atoms, 
        control_params *control,
        simulation_data *data, 
        list p_far_nbrs, 
        real *E_vdW, real *E_Ele, rvec *aux_ext_press, 
        LR_lookup_table *d_LR,
        int num_atom_types,
        int energy_update_freq,
        int N  )
{

    extern __shared__ real _vdw[];
    extern __shared__ real _ele[];
    extern __shared__ rvec _force [];

    real *sh_vdw;
    real *sh_ele;
    rvec *sh_force;

    int i, j, pj, r, steps, update_freq, update_energies;
    int type_i, type_j, tmin, tmax;
    int start_i, end_i;
    real r_ij, self_coef, base, dif;
    real e_vdW, e_ele;
    real CEvd, CEclmb;
    rvec temp, ext_press;
    far_neighbor_data *nbr_pj;
    LR_lookup_table *t;
    list *far_nbrs = &p_far_nbrs;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warpid = thread_id / VDW_THREADS_PER_ATOM;
    int laneid = thread_id & (VDW_THREADS_PER_ATOM -1);

    i = warpid;

    sh_vdw = _vdw;
    sh_ele = _vdw + blockDim.x;
    sh_force = (rvec *)( _vdw + 2*blockDim.x);

    sh_vdw[threadIdx.x] = 0.0; 
    sh_ele[threadIdx.x] = 0.0; 
    rvec_MakeZero ( sh_force [threadIdx.x] );

    if ( i < N ) 
    {

        reax_atom local_atom ;
        local_atom.q =  atoms[i].q;
        //local_atom.q =  d_far_data.q[i];
        local_atom.type = atoms[i].type;
        //local_atom.type = d_far_data.type[i];

        /*
           sh_vdw = _vdw;
           sh_ele = _vdw + warpid;
           sh_force = (rvec *)( _vdw + 2*warpid);

           sh_vdw[threadIdx.x] = 0.0; 
           sh_ele[threadIdx.x] = 0.0; 
           rvec_MakeZero ( sh_force [threadIdx.x] );
         */


        steps = data->step - data->prev_steps;
        update_freq = energy_update_freq;
        update_energies = update_freq > 0 && steps % update_freq == 0;

        //for( i = 0; i < system->N; ++i ) {
        type_i  = local_atom.type;
        start_i = Start_Index(i,far_nbrs);
        end_i   = End_Index(i,far_nbrs);

        pj = start_i + laneid;

        //for( pj = start_i; pj < end_i; ++pj ) 
        while (pj < end_i)
        {
            if( far_nbrs->select.far_nbr_list[pj].d <= control->r_cut ) 
                //if( d_far_data.d[pj] <= control->r_cut ) 
            {
                nbr_pj = &( far_nbrs->select.far_nbr_list[pj] );
                j      = nbr_pj->nbr;
                //j      = d_far_data.nbrs[pj];
                type_j = atoms[j].type;
                //type_j = d_far_data.type[j];
                r_ij   = nbr_pj->d;
                //r_ij   = d_far_data.d[pj];
                self_coef = (i == j) ? 0.5 : 1.0;
                tmin  = MIN( type_i, type_j );
                tmax  = MAX( type_i, type_j );
                t = &( d_LR[ index_lr (tmin,tmax,num_atom_types) ] ); 

                //TODO
                //CHANGE ORIGINAL
                //if (i <= j) { pj += blockDim.x; continue; }
                //CHANGE ORIGINAL

                /* Cubic Spline Interpolation */
                r = (int)(r_ij * t->inv_dx);
                if( r == 0 )  ++r;
                base = (real)(r+1) * t->dx;
                dif = r_ij - base;

                if(( update_energies )) 
                {
                    e_vdW = ((t->vdW[r].d*dif + t->vdW[r].c)*dif + t->vdW[r].b)*dif + 
                        t->vdW[r].a;
                    e_vdW *= self_coef;

                    e_ele = ((t->ele[r].d*dif + t->ele[r].c)*dif + t->ele[r].b)*dif + t->ele[r].a;
                    e_ele *= self_coef * local_atom.q * atoms[j].q;


                    //data->E_vdW += e_vdW;
                    //TODO
                    //E_vdW [i] += e_vdW / 2.0;
                    //E_vdW [i] = __dadd_rd (E_vdW [i], e_vdW/2.0);
                    sh_vdw [threadIdx.x] += e_vdW/2.0;
                    //E_vdW [i] += e_vdW;

                    //TODO
                    //data->E_Ele += e_ele;
                    //E_Ele [i] += e_ele / 2.0;
                    //E_Ele [i] = __dadd_rd ( E_Ele [i], e_ele / 2.0);
                    sh_ele [threadIdx.x] += e_ele/2.0;
                    //E_Ele [i] += e_ele;
                }    

                CEvd = ((t->CEvd[r].d*dif + t->CEvd[r].c)*dif + t->CEvd[r].b)*dif + 
                    t->CEvd[r].a;
                CEvd *= self_coef;

                CEclmb = ((t->CEclmb[r].d*dif+t->CEclmb[r].c)*dif+t->CEclmb[r].b)*dif + 
                    t->CEclmb[r].a;
                CEclmb *= self_coef * local_atom.q * atoms[j].q;
                //CEclmb *= self_coef * local_atom.q * d_far_data.q[j];

                if( control->ensemble == NVE || control->ensemble == NVT || control->ensemble == bNVT) {
                    if ( i >= j)
                        //rvec_ScaledAdd( atoms[i].f, -(CEvd + CEclmb), nbr_pj->dvec );
                        rvec_ScaledAdd( sh_force [threadIdx.x], -(CEvd + CEclmb), nbr_pj->dvec );
                    //rvec_ScaledAdd( sh_force [threadIdx.x], -(CEvd + CEclmb), d_far_data.dvec[pj] );
                    else 
                        //rvec_ScaledAdd( atoms[i].f, +(CEvd + CEclmb), nbr_pj->dvec );
                        rvec_ScaledAdd( sh_force [threadIdx.x], +(CEvd + CEclmb), nbr_pj->dvec );
                    //rvec_ScaledAdd( sh_force [threadIdx.x], +(CEvd + CEclmb), d_far_data.dvec[pj] );
                }
                else { // NPT, iNPT or sNPT
                    // for pressure coupling, terms not related to bond order 
                    //  derivatives are added directly into pressure vector/tensor /
                    rvec_Scale( temp, CEvd + CEclmb, nbr_pj->dvec );
                    if (i >= j)
                        rvec_ScaledAdd( atoms[i].f, -1., temp );
                    else
                        rvec_Add( atoms[i].f, temp );
                    rvec_iMultiply( ext_press, nbr_pj->rel_box, temp );

                    //rvec_Add( data->ext_press, ext_press );
                    rvec_Copy (aux_ext_press [i], ext_press );

                    //TODO CHECK THIS
                }



            }

            pj += VDW_THREADS_PER_ATOM;
        }

    }// if i < n condition

    __syncthreads ();

    if (laneid < 16) {
        sh_vdw[threadIdx.x] += sh_vdw[threadIdx.x + 16];
        sh_ele[threadIdx.x] += sh_ele[threadIdx.x + 16];
        rvec_Add (sh_force [threadIdx.x], sh_force [threadIdx.x + 16] );
    }
    __syncthreads ();
    if (laneid < 8) {
        sh_vdw[threadIdx.x] += sh_vdw[threadIdx.x + 8];
        sh_ele[threadIdx.x] += sh_ele[threadIdx.x + 8];
        rvec_Add (sh_force [threadIdx.x], sh_force [threadIdx.x + 8] );
    }
    __syncthreads ();
    if (laneid < 4) {
        sh_vdw[threadIdx.x] += sh_vdw[threadIdx.x + 4];
        sh_ele[threadIdx.x] += sh_ele[threadIdx.x + 4];
        rvec_Add (sh_force [threadIdx.x], sh_force [threadIdx.x + 4] );
    }
    __syncthreads ();
    if (laneid < 2) {
        sh_vdw[threadIdx.x] += sh_vdw[threadIdx.x + 2];
        sh_ele[threadIdx.x] += sh_ele[threadIdx.x + 2];
        rvec_Add (sh_force [threadIdx.x], sh_force [threadIdx.x + 2] );
    }
    __syncthreads ();
    if (laneid < 1) {
        sh_vdw[threadIdx.x] += sh_vdw[threadIdx.x + 1];
        sh_ele[threadIdx.x] += sh_ele[threadIdx.x + 1];
        rvec_Add (sh_force [threadIdx.x], sh_force [threadIdx.x + 1] );
    }
    __syncthreads ();
    if (laneid == 0) {
        E_vdW [i] += sh_vdw[threadIdx.x];
        E_Ele [i] += sh_ele[threadIdx.x];
        rvec_Add (atoms[i].f, sh_force [ threadIdx.x ]);
    }
}


GLOBAL void Cuda_Tabulated_vdW_Coulomb_Energy_1(reax_atom *atoms, 
        control_params *control,
        simulation_data *data, 
        list p_far_nbrs, 
        real *E_vdW, real *E_Ele, rvec *aux_ext_press, 
        LR_lookup_table *d_LR,
        int num_atom_types,
        int energy_update_freq,
        int N )
{

    extern __shared__ real _vdw[];
    extern __shared__ real _ele[];

    real *sh_vdw;
    real *sh_ele;

    int i, j, pj, r, steps, update_freq, update_energies;
    int type_i, type_j, tmin, tmax;
    int start_i, end_i;
    real r_ij, self_coef, base, dif;
    real e_vdW, e_ele;
    real CEvd, CEclmb;
    rvec temp, ext_press;
    far_neighbor_data *nbr_pj;
    LR_lookup_table *t;
    list *far_nbrs = &p_far_nbrs;

    i = blockIdx.x;

    reax_atom local_atom;
    local_atom.q =  atoms[i].q;
    local_atom.type = atoms[i].type;

    sh_vdw = _vdw;
    sh_ele = _vdw + blockDim.x;

    sh_vdw[threadIdx.x] = 0.0; 
    sh_ele[threadIdx.x] = 0.0; 


    steps = data->step - data->prev_steps;
    update_freq = energy_update_freq;
    update_energies = update_freq > 0 && steps % update_freq == 0;

    type_i  = local_atom.type;
    start_i = Start_Index(i,far_nbrs);
    end_i   = End_Index(i,far_nbrs);

    pj = start_i + threadIdx.x;

    while (pj < end_i)
    {
        if( far_nbrs->select.far_nbr_list[pj].d <= control->r_cut ) 
        {
            nbr_pj = &( far_nbrs->select.far_nbr_list[pj] );
            j      = nbr_pj->nbr;
            type_j = atoms[j].type;
            r_ij   = nbr_pj->d;
            self_coef = (i == j) ? 0.5 : 1.0;
            tmin  = MIN( type_i, type_j );
            tmax  = MAX( type_i, type_j );
            t = &( d_LR[ index_lr (tmin,tmax,num_atom_types) ] ); 

            /* Cubic Spline Interpolation */
            r = (int)(r_ij * t->inv_dx);
            if( r == 0 )  ++r;
            base = (real)(r+1) * t->dx;
            dif = r_ij - base;

            if(( update_energies )) 
            {
                e_vdW = ((t->vdW[r].d*dif + t->vdW[r].c)*dif + t->vdW[r].b)*dif + 
                    t->vdW[r].a;
                e_vdW *= self_coef;

                e_ele = ((t->ele[r].d*dif + t->ele[r].c)*dif + t->ele[r].b)*dif + 
                    t->ele[r].a;
                e_ele *= self_coef * local_atom.q * atoms[j].q;

                sh_vdw [threadIdx.x] += e_vdW/2.0;
                sh_ele [threadIdx.x] += e_ele/2.0;
            }    
        }

        pj += blockDim.x;
    }

    // now do a reduce inside the warp for E_vdW, E_Ele and force.
    if (threadIdx.x < 16) {
        sh_vdw[threadIdx.x] += sh_vdw[threadIdx.x + 16];
        sh_ele[threadIdx.x] += sh_ele[threadIdx.x + 16];
    }
    if (threadIdx.x < 8) {
        sh_vdw[threadIdx.x] += sh_vdw[threadIdx.x + 8];
        sh_ele[threadIdx.x] += sh_ele[threadIdx.x + 8];
    }
    if (threadIdx.x < 4) {
        sh_vdw[threadIdx.x] += sh_vdw[threadIdx.x + 4];
        sh_ele[threadIdx.x] += sh_ele[threadIdx.x + 4];
    }
    if (threadIdx.x < 2) {
        sh_vdw[threadIdx.x] += sh_vdw[threadIdx.x + 2];
        sh_ele[threadIdx.x] += sh_ele[threadIdx.x + 2];
    }
    if (threadIdx.x < 1) {
        sh_vdw[threadIdx.x] += sh_vdw[threadIdx.x + 1];
        sh_ele[threadIdx.x] += sh_ele[threadIdx.x + 1];
    }
    if (threadIdx.x == 0) {
        E_vdW [i] += sh_vdw[0];
        E_Ele [i] += sh_ele[0];
    }
}


GLOBAL void Cuda_Tabulated_vdW_Coulomb_Energy_2(reax_atom *atoms, 
        control_params *control,
        simulation_data *data, 
        list p_far_nbrs, 
        real *E_vdW, real *E_Ele, rvec *aux_ext_press, 
        LR_lookup_table *d_LR,
        int num_atom_types,
        int energy_update_freq,
        int N )
{

    extern __shared__ rvec _force [];

    rvec *sh_force;

    int i, j, pj, r, steps, update_freq, update_energies;
    int type_i, type_j, tmin, tmax;
    int start_i, end_i;
    real r_ij, self_coef, base, dif;
    real e_vdW, e_ele;
    real CEvd, CEclmb;
    rvec temp, ext_press;
    far_neighbor_data *nbr_pj;
    LR_lookup_table *t;
    list *far_nbrs = &p_far_nbrs;

    i = blockIdx.x;

    reax_atom local_atom;
    local_atom.q =  atoms[i].q;
    local_atom.type = atoms[i].type;

    sh_force = _force;
    rvec_MakeZero ( sh_force [threadIdx.x] );


    steps = data->step - data->prev_steps;
    update_freq = energy_update_freq;
    update_energies = update_freq > 0 && steps % update_freq == 0;

    //for( i = 0; i < system->N; ++i ) {
    type_i  = local_atom.type;
    start_i = Start_Index(i,far_nbrs);
    end_i   = End_Index(i,far_nbrs);

    pj = start_i + threadIdx.x;

    while (pj < end_i)
    {
        if( far_nbrs->select.far_nbr_list[pj].d <= control->r_cut ) 
        {
            nbr_pj = &( far_nbrs->select.far_nbr_list[pj] );
            j      = nbr_pj->nbr;
            type_j = atoms[j].type;
            r_ij   = nbr_pj->d;
            self_coef = (i == j) ? 0.5 : 1.0;
            tmin  = MIN( type_i, type_j );
            tmax  = MAX( type_i, type_j );
            t = &( d_LR[ index_lr (tmin,tmax,num_atom_types) ] ); 

            /* Cubic Spline Interpolation */
            r = (int)(r_ij * t->inv_dx);
            if( r == 0 )  ++r;
            base = (real)(r+1) * t->dx;
            dif = r_ij - base;

            CEvd = ((t->CEvd[r].d*dif + t->CEvd[r].c)*dif + t->CEvd[r].b)*dif + 
                t->CEvd[r].a;
            CEvd *= self_coef;

            CEclmb = ((t->CEclmb[r].d*dif+t->CEclmb[r].c)*dif+t->CEclmb[r].b)*dif + 
                t->CEclmb[r].a;
            CEclmb *= self_coef * local_atom.q * atoms[j].q;

            if( control->ensemble == NVE || control->ensemble == NVT || control->ensemble == bNVT ) {
                if ( i >= j)
                    rvec_ScaledAdd( sh_force [threadIdx.x], -(CEvd + CEclmb), nbr_pj->dvec );
                else 
                    rvec_ScaledAdd( sh_force [threadIdx.x], +(CEvd + CEclmb), nbr_pj->dvec );
            }
            else { // NPT, iNPT or sNPT
                // for pressure coupling, terms not related to bond order 
                //  derivatives are added directly into pressure vector/tensor /
                rvec_Scale( temp, CEvd + CEclmb, nbr_pj->dvec );
                if (i >= j)
                    rvec_ScaledAdd( atoms[i].f, -1., temp );
                else
                    rvec_Add( atoms[i].f, temp );
                rvec_iMultiply( ext_press, nbr_pj->rel_box, temp );

                rvec_Copy (aux_ext_press [i], ext_press );
            }
        }

        pj += blockDim.x;
    }

    // now do a reduce inside the warp for E_vdW, E_Ele and force.
    if (threadIdx.x < 16) {
        rvec_Add (sh_force [threadIdx.x], sh_force [threadIdx.x + 16] );
    }
    if (threadIdx.x < 8) {
        rvec_Add (sh_force [threadIdx.x], sh_force [threadIdx.x + 8] );
    }
    if (threadIdx.x < 4) {
        rvec_Add (sh_force [threadIdx.x], sh_force [threadIdx.x + 4] );
    }
    if (threadIdx.x < 2) {
        rvec_Add (sh_force [threadIdx.x], sh_force [threadIdx.x + 2] );
    }
    if (threadIdx.x < 1) {
        rvec_Add (sh_force [threadIdx.x], sh_force [threadIdx.x + 1] );
    }
    if (threadIdx.x == 0) {
        rvec_Add (atoms[i].f, sh_force [ 0 ]);
    }
}
