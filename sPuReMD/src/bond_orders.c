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

#include "bond_orders.h"

#include "list.h"
#include "lookup.h"
#include "print_utils.h"
#include "vector.h"


static inline real Cf45( real p1, real p2 )
{
    return  -EXP(-p2 / 2.0) /
        ( SQR( EXP(-p1 / 2.0) + EXP(p1 / 2.0) ) * (EXP(-p2 / 2.0) + EXP(p2 / 2.0)) );
}


#ifdef TEST_FORCES
void Get_dBO( reax_system *system, reax_list **lists,
        int i, int pj, real C, rvec *v )
{
    reax_list *bonds;
    reax_list *dBOs;
    int start_pj, end_pj, k;

    bonds = &(*lists)[BONDS];
    dBOs = &(*lists)[DBO];
    pj = bonds->select.bond_list[pj].dbond_index;
    start_pj = Start_Index(pj, dBOs);
    end_pj = End_Index(pj, dBOs);

    for ( k = start_pj; k < end_pj; ++k )
    {
        rvec_Scale( v[dBOs->select.dbo_list[k].wrt],
                C, dBOs->select.dbo_list[k].dBO );
    }
}


void Get_dBOpinpi2( reax_system *system, reax_list **lists,
        int i, int pj, real Cpi, real Cpi2, rvec *vpi, rvec *vpi2 )
{
    reax_list *bonds;
    reax_list *dBOs;
    dbond_data *dbo_k;
    int start_pj, end_pj, k;

    bonds = &(*lists)[BONDS];
    dBOs = &(*lists)[DBO];
    pj = bonds->select.bond_list[pj].dbond_index;
    start_pj = Start_Index(pj, dBOs);
    end_pj = End_Index(pj, dBOs);

    for ( k = start_pj; k < end_pj; ++k )
    {
        dbo_k = &(dBOs->select.dbo_list[k]);
        rvec_Scale( vpi[dbo_k->wrt], Cpi, dbo_k->dBOpi );
        rvec_Scale( vpi2[dbo_k->wrt], Cpi2, dbo_k->dBOpi2 );
    }
}


void Add_dBO( reax_system *system, reax_list **lists,
        int i, int pj, real C, rvec *v )
{
    reax_list *bonds;
    reax_list *dBOs;
    int start_pj, end_pj, k;

    bonds = &(*lists)[BONDS];
    dBOs = &(*lists)[DBO];
    pj = bonds->select.bond_list[pj].dbond_index;
    start_pj = Start_Index(pj, dBOs);
    end_pj = End_Index(pj, dBOs);

    //fprintf( stderr, "i=%d j=%d start=%d end=%d\n", i, pj, start_pj, end_pj );

    for ( k = start_pj; k < end_pj; ++k )
    {
        rvec_ScaledAdd( v[dBOs->select.dbo_list[k].wrt],
                C, dBOs->select.dbo_list[k].dBO );
    }
}


void Add_dBOpinpi2( reax_system *system, reax_list **lists,
        int i, int pj, real Cpi, real Cpi2, rvec *vpi, rvec *vpi2 )
{
    reax_list *bonds;
    reax_list *dBOs;
    dbond_data *dbo_k;
    int start_pj, end_pj, k;

    bonds = &(*lists)[BONDS];
    dBOs = &(*lists)[DBO];
    pj = bonds->select.bond_list[pj].dbond_index;
    start_pj = Start_Index(pj, dBOs);
    end_pj = End_Index(pj, dBOs);

    for ( k = start_pj; k < end_pj; ++k )
    {
        dbo_k = &(dBOs->select.dbo_list[k]);
        rvec_ScaledAdd( vpi[dbo_k->wrt], Cpi, dbo_k->dBOpi );
        rvec_ScaledAdd( vpi2[dbo_k->wrt], Cpi2, dbo_k->dBOpi2 );
    }
}


void Add_dBO_to_Forces( reax_system *system, reax_list **lists,
                        int i, int pj, real C )
{
    reax_list *bonds;
    reax_list *dBOs;
    int start_pj, end_pj, k;

    bonds = &(*lists)[BONDS];
    dBOs = &(*lists)[DBO];
    pj = bonds->select.bond_list[pj].dbond_index;
    start_pj = Start_Index(pj, dBOs);
    end_pj = End_Index(pj, dBOs);

    for ( k = start_pj; k < end_pj; ++k )
    {
        rvec_ScaledAdd( system->atoms[dBOs->select.dbo_list[k].wrt].f,
                C, dBOs->select.dbo_list[k].dBO );
    }
}


void Add_dBOpinpi2_to_Forces( reax_system *system, reax_list **lists,
        int i, int pj, real Cpi, real Cpi2 )
{
    reax_list *bonds;
    reax_list *dBOs;
    dbond_data *dbo_k;
    int start_pj, end_pj, k;

    bonds = &(*lists)[BONDS];
    dBOs = &(*lists)[DBO];
    pj = bonds->select.bond_list[pj].dbond_index;
    start_pj = Start_Index(pj, dBOs);
    end_pj = End_Index(pj, dBOs);

    for ( k = start_pj; k < end_pj; ++k )
    {
        dbo_k = &(dBOs->select.dbo_list[k]);
        rvec_ScaledAdd( system->atoms[dbo_k->wrt].f, Cpi, dbo_k->dBOpi );
        rvec_ScaledAdd( system->atoms[dbo_k->wrt].f, Cpi2, dbo_k->dBOpi2 );
    }
}


void Add_dDelta( reax_system *system, reax_list **lists, int i, real C, rvec *v )
{
    reax_list *dDeltas;
    int start, end, k;

    dDeltas = &((*lists)[DDELTA]);
    start = Start_Index(i, dDeltas);
    end = End_Index(i, dDeltas);

    for ( k = start; k < end; ++k )
    {
        rvec_ScaledAdd( v[dDeltas->select.dDelta_list[k].wrt],
                C, dDeltas->select.dDelta_list[k].dVal );
    }
}


void Add_dDelta_to_Forces( reax_system *system, reax_list **lists, int i, real C )
{
    reax_list *dDeltas;
    int start, end, k;

    dDeltas = &((*lists)[DDELTA]);
    start = Start_Index(i, dDeltas);
    end = End_Index(i, dDeltas);

    for ( k = start; k < end; ++k )
    {
        rvec_ScaledAdd( system->atoms[dDeltas->select.dDelta_list[k].wrt].f,
                C, dDeltas->select.dDelta_list[k].dVal );
    }
}


void Calculate_dBO( int i, int pj, static_storage *workspace, reax_list **lists,
        int *top )
{
    int j, k, l, start_i, end_i, end_j;
    rvec dDeltap_self, dBOp;
    reax_list *bonds, *dBOs;
    bond_data *nbr_l, *nbr_k;
    bond_order_data *bo_ij;
    dbond_data *top_dbo;

    /* Initializations */
    bonds = &(*lists)[BONDS];
    dBOs = &(*lists)[DBO];
    j = bonds->select.bond_list[pj].nbr;
    bo_ij = &(bonds->select.bond_list[pj].bo_data);
    start_i = Start_Index( i, bonds );
    end_i = End_Index( i, bonds );
    l = Start_Index( j, bonds );
    end_j = End_Index( j, bonds );
    top_dbo = &(dBOs->select.dbo_list[ (*top) ]);

    for ( k = start_i; k < end_i; ++k )
    {
        nbr_k = &(bonds->select.bond_list[k]);

        for ( ; l < end_j && bonds->select.bond_list[l].nbr < nbr_k->nbr; ++l )
        {
            /* These are the neighbors of j which aren't in the neighbor_list of i
            Note that they might also include i! */
            nbr_l = &(bonds->select.bond_list[l]);
            top_dbo->wrt = nbr_l->nbr;
            rvec_Copy( dBOp, nbr_l->bo_data.dBOp );

            rvec_Scale( top_dbo->dBO, -bo_ij->C3dbo, dBOp );  // dBO, 3rd
            rvec_Scale( top_dbo->dBOpi, -bo_ij->C4dbopi, dBOp );  // dBOpi, 4th
            rvec_Scale( top_dbo->dBOpi2, -bo_ij->C4dbopi2, dBOp );// dBOpipi, 4th

            if ( nbr_l->nbr == i )
            {
                rvec_Copy( dDeltap_self, workspace->dDeltap_self[i] );

                /* dBO */
                rvec_ScaledAdd( top_dbo->dBO, bo_ij->C1dbo, bo_ij->dBOp ); //1st
                rvec_ScaledAdd( top_dbo->dBO, bo_ij->C2dbo, dDeltap_self ); //2nd

                /* dBOpi */
                rvec_ScaledAdd( top_dbo->dBOpi, bo_ij->C1dbopi, bo_ij->dln_BOp_pi ); //1
                rvec_ScaledAdd( top_dbo->dBOpi, bo_ij->C2dbopi, bo_ij->dBOp ); //2nd
                rvec_ScaledAdd( top_dbo->dBOpi, bo_ij->C3dbopi, dDeltap_self ); //3rd

                /* dBOpp, 1st */
                rvec_ScaledAdd( top_dbo->dBOpi2, bo_ij->C1dbopi2, bo_ij->dln_BOp_pi2 );
                rvec_ScaledAdd( top_dbo->dBOpi2, bo_ij->C2dbopi2, bo_ij->dBOp ); //2nd
                rvec_ScaledAdd( top_dbo->dBOpi2, bo_ij->C3dbopi2, dDeltap_self ); //3rd

            }

            ++(*top);
            ++top_dbo;
        }

        /* Now we are processing neighbor k of i. */
        top_dbo->wrt = nbr_k->nbr;
        rvec_Copy( dBOp, nbr_k->bo_data.dBOp );

        rvec_Scale( top_dbo->dBO, -bo_ij->C2dbo, dBOp );      //dBO-2
        rvec_Scale( top_dbo->dBOpi, -bo_ij->C3dbopi, dBOp );  //dBOpi-3
        rvec_Scale( top_dbo->dBOpi2, -bo_ij->C3dbopi2, dBOp );//dBOpp-3

        if ( l < end_j && bonds->select.bond_list[l].nbr == nbr_k->nbr )
        {
            /* This is a common neighbor of i and j. */
            nbr_l = &(bonds->select.bond_list[l]);
            rvec_Copy( dBOp, nbr_l->bo_data.dBOp );

            rvec_ScaledAdd( top_dbo->dBO, -bo_ij->C3dbo, dBOp );      //dBO,3rd
            rvec_ScaledAdd( top_dbo->dBOpi, -bo_ij->C4dbopi, dBOp );  //dBOpi,4th
            rvec_ScaledAdd( top_dbo->dBOpi2, -bo_ij->C4dbopi2, dBOp );//dBOpp.4th
            ++l;
        }
        else if ( k == pj )
        {
            /* This negihbor is j. */
            rvec_Copy( dDeltap_self, workspace->dDeltap_self[j] );

            rvec_ScaledAdd( top_dbo->dBO, -bo_ij->C1dbo, bo_ij->dBOp );// 1st, dBO
            rvec_ScaledAdd( top_dbo->dBO, bo_ij->C3dbo, dDeltap_self );// 3rd, dBO

            /* dBOpi, 1st */
            rvec_ScaledAdd( top_dbo->dBOpi, -bo_ij->C1dbopi, bo_ij->dln_BOp_pi );
            rvec_ScaledAdd( top_dbo->dBOpi, -bo_ij->C2dbopi, bo_ij->dBOp );    //2nd
            rvec_ScaledAdd( top_dbo->dBOpi, bo_ij->C4dbopi, dDeltap_self );  //4th

            /* dBOpi2, 1st */
            rvec_ScaledAdd( top_dbo->dBOpi2, -bo_ij->C1dbopi2, bo_ij->dln_BOp_pi2 );
            rvec_ScaledAdd( top_dbo->dBOpi2, -bo_ij->C2dbopi2, bo_ij->dBOp ); //2nd
            rvec_ScaledAdd( top_dbo->dBOpi2, bo_ij->C4dbopi2, dDeltap_self ); //4th
        }

        ++(*top), ++top_dbo;
    }

    for ( ; l < end_j; ++l )
    {
        /* These are the remaining neighbors of j which are not in the
           neighbor_list of i. Note that they might also include i!*/
        nbr_l = &(bonds->select.bond_list[l]);
        top_dbo->wrt = nbr_l->nbr;
        rvec_Copy( dBOp, nbr_l->bo_data.dBOp );

        rvec_Scale( top_dbo->dBO, -bo_ij->C3dbo, dBOp );      //3rd, dBO
        rvec_Scale( top_dbo->dBOpi, -bo_ij->C4dbopi, dBOp );  //4th, dBOpi
        rvec_Scale( top_dbo->dBOpi2, -bo_ij->C4dbopi2, dBOp );//4th, dBOpp

        if ( nbr_l->nbr == i )
        {
            /* do the adjustments on i */
            rvec_Copy( dDeltap_self, workspace->dDeltap_self[i] );

            /* dBO, 1st */
            rvec_ScaledAdd( top_dbo->dBO, bo_ij->C1dbo, bo_ij->dBOp );
            rvec_ScaledAdd( top_dbo->dBO, bo_ij->C2dbo, dDeltap_self ); //2nd, dBO

            /* dBOpi, 1st */
            rvec_ScaledAdd( top_dbo->dBOpi, bo_ij->C1dbopi, bo_ij->dln_BOp_pi );
            rvec_ScaledAdd( top_dbo->dBOpi, bo_ij->C2dbopi, bo_ij->dBOp );  //2nd
            rvec_ScaledAdd( top_dbo->dBOpi, bo_ij->C3dbopi, dDeltap_self ); //3rd

            /* dBOpipi, 1st */
            rvec_ScaledAdd(top_dbo->dBOpi2, bo_ij->C1dbopi2, bo_ij->dln_BOp_pi2);
            rvec_ScaledAdd( top_dbo->dBOpi2, bo_ij->C2dbopi2, bo_ij->dBOp ); //2nd
            rvec_ScaledAdd( top_dbo->dBOpi2, bo_ij->C3dbopi2, dDeltap_self );//3rd
        }

        ++(*top), ++top_dbo;
    }
}
#endif


void Add_dBond_to_Forces_NPT( int i, int pj, reax_system *system,
        simulation_data *data, static_storage *workspace, reax_list **lists )
{
    reax_list *bonds;
    bond_data *nbr_j, *nbr_k;
    bond_order_data *bo_ij, *bo_ji;
    dbond_coefficients coef;
    rvec temp, ext_press;
    ivec rel_box;
    int pk, k, j;
    rvec *f_i, *f_j, *f_k;
#ifdef _OPENMP
    int tid = omp_get_thread_num( );
#endif

    /* Initializations */
    bonds = &(*lists)[BONDS];
    nbr_j = &(bonds->select.bond_list[pj]);
    j = nbr_j->nbr;
    bo_ij = &(nbr_j->bo_data);
    bo_ji = &(bonds->select.bond_list[ nbr_j->sym_index ].bo_data);
#ifdef _OPENMP
    f_i = &(workspace->f_local[tid * system->N + i]);
    f_j = &(workspace->f_local[tid * system->N + j]);
#else
    f_i = &(system->atoms[i].f);
    f_j = &(system->atoms[j].f);
#endif

    coef.C1dbo = bo_ij->C1dbo * (bo_ij->Cdbo + bo_ji->Cdbo);
    coef.C2dbo = bo_ij->C2dbo * (bo_ij->Cdbo + bo_ji->Cdbo);
    coef.C3dbo = bo_ij->C3dbo * (bo_ij->Cdbo + bo_ji->Cdbo);

    coef.C1dbopi = bo_ij->C1dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
    coef.C2dbopi = bo_ij->C2dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
    coef.C3dbopi = bo_ij->C3dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
    coef.C4dbopi = bo_ij->C4dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);

    coef.C1dbopi2 = bo_ij->C1dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
    coef.C2dbopi2 = bo_ij->C2dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
    coef.C3dbopi2 = bo_ij->C3dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
    coef.C4dbopi2 = bo_ij->C4dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);

    coef.C1dDelta = bo_ij->C1dbo * (workspace->CdDelta[i] + workspace->CdDelta[j]);
    coef.C2dDelta = bo_ij->C2dbo * (workspace->CdDelta[i] + workspace->CdDelta[j]);
    coef.C3dDelta = bo_ij->C3dbo * (workspace->CdDelta[i] + workspace->CdDelta[j]);

    /************************************
    * forces related to atom i          *
    * first neighbors of atom i         *
    ************************************/
    for ( pk = Start_Index(i, bonds); pk < End_Index(i, bonds); ++pk )
    {
        nbr_k = &(bonds->select.bond_list[pk]);
        k = nbr_k->nbr;
#ifdef _OPENMP
        f_k = &(workspace->f_local[tid * system->N + k]);
#else
        f_k = &(system->atoms[k].f);
#endif

        rvec_Scale( temp, -coef.C2dbo, nbr_k->bo_data.dBOp );       /*2nd,dBO*/
        rvec_ScaledAdd( temp, -coef.C2dDelta, nbr_k->bo_data.dBOp );/*dDelta*/
        rvec_ScaledAdd( temp, -coef.C3dbopi, nbr_k->bo_data.dBOp ); /*3rd,dBOpi*/
        rvec_ScaledAdd( temp, -coef.C3dbopi2, nbr_k->bo_data.dBOp );/*3rd,dBOpi2*/

        /* force */
        rvec_Add( *f_k, temp );
        /* pressure */
        rvec_iMultiply( ext_press, nbr_k->rel_box, temp );
#ifdef _OPENMP
        #pragma omp critical (Add_dBond_to_Forces_NPT_ext_press)
#endif
        {
            rvec_Add( data->ext_press, ext_press );
        }

        /* if( !ivec_isZero( nbr_k->rel_box ) )
           fprintf( stderr, "%3d %3d %3d: dvec[%10.6f %10.6f %10.6f]
           ext[%3d %3d %3d] f[%10.6f %10.6f %10.6f]\n",
           i+1,
           system->atoms[i].x[0],system->atoms[i].x[1],system->atoms[i].x[2],
           j+1, k+1,
           system->atoms[k].x[0], system->atoms[k].x[1], system->atoms[k].x[2],
           nbr_k->dvec[0], nbr_k->dvec[1], nbr_k->dvec[2],
           nbr_k->rel_box[0], nbr_k->rel_box[1], nbr_k->rel_box[2],
           temp[0], temp[1], temp[2] ); */
    }

    /* then atom i itself  */
    rvec_Scale( temp, coef.C1dbo, bo_ij->dBOp );                      /*1st, dBO*/
    rvec_ScaledAdd( temp, coef.C2dbo, workspace->dDeltap_self[i] );   /*2nd, dBO*/

    rvec_ScaledAdd( temp, coef.C1dDelta, bo_ij->dBOp );               /*1st, dBO*/
    rvec_ScaledAdd( temp, coef.C2dDelta, workspace->dDeltap_self[i] );/*2nd, dBO*/

    rvec_ScaledAdd( temp, coef.C1dbopi, bo_ij->dln_BOp_pi );         /*1st,dBOpi*/
    rvec_ScaledAdd( temp, coef.C2dbopi, bo_ij->dBOp );               /*2nd,dBOpi*/
    rvec_ScaledAdd( temp, coef.C3dbopi, workspace->dDeltap_self[i] );/*3rd,dBOpi*/

    rvec_ScaledAdd( temp, coef.C1dbopi2, bo_ij->dln_BOp_pi2 ) ;      /*1st,dBO_pi2*/
    rvec_ScaledAdd( temp, coef.C2dbopi2, bo_ij->dBOp );              /*2nd,dBO_pi2*/
    rvec_ScaledAdd( temp, coef.C3dbopi2, workspace->dDeltap_self[i] );/*3rd,dBO_pi2*/

    /* force */
    rvec_Add( *f_i, temp );
    /* ext pressure due to i dropped, counting force on j only will be enough */

    /****************************************************************************
     * forces and pressure related to atom j                                    *
     * first neighbors of atom j                                                *
     ***************************************************************************/
    for ( pk = Start_Index(j, bonds); pk < End_Index(j, bonds); ++pk )
    {
        nbr_k = &(bonds->select.bond_list[pk]);
        k = nbr_k->nbr;
#ifdef _OPENMP
        f_k = &(workspace->f_local[tid * system->N + k]);
#else
        f_k = &(system->atoms[k].f);
#endif

        rvec_Scale( temp, -coef.C3dbo, nbr_k->bo_data.dBOp );       /*3rd,dBO*/
        rvec_ScaledAdd( temp, -coef.C3dDelta, nbr_k->bo_data.dBOp );/*dDelta*/
        rvec_ScaledAdd( temp, -coef.C4dbopi, nbr_k->bo_data.dBOp ); /*4th,dBOpi*/
        rvec_ScaledAdd( temp, -coef.C4dbopi2, nbr_k->bo_data.dBOp );/*4th,dBOpi2*/

        /* force */
        rvec_Add( *f_k, temp );
        /* pressure */
        if ( k != i )
        {
            ivec_Sum( rel_box, nbr_k->rel_box, nbr_j->rel_box );//k's rel_box  wrt i
            rvec_iMultiply( ext_press, rel_box, temp );
#ifdef _OPENMP
            #pragma omp critical (Add_dBond_to_Forces_NPT_ext_press)
#endif
            {
                rvec_Add( data->ext_press, ext_press );
            }

            /* if( !ivec_isZero( rel_box ) )
            fprintf( stderr, "%3d %3d %3d: dvec[%10.6f %10.6f %10.6f]
             ext[%3d %3d %3d] f[%10.6f %10.6f %10.6f]\n",
             i+1, j+1,
             system->atoms[j].x[0],system->atoms[j].x[1],system->atoms[j].x[2],
             k+1,
             system->atoms[k].x[0], system->atoms[k].x[1], system->atoms[k].x[2],
             nbr_k->dvec[0], nbr_k->dvec[1], nbr_k->dvec[2],
             rel_box[0], rel_box[1], rel_box[2],
             temp[0], temp[1], temp[2] ); */
        }
    }

    /* then atom j itself */
    rvec_Scale( temp, -coef.C1dbo, bo_ij->dBOp );                     /*1st, dBO*/
    rvec_ScaledAdd( temp, coef.C3dbo, workspace->dDeltap_self[j] );   /*2nd, dBO*/

    rvec_ScaledAdd( temp, -coef.C1dDelta, bo_ij->dBOp );              /*1st, dBO*/
    rvec_ScaledAdd( temp, coef.C3dDelta, workspace->dDeltap_self[j] );/*2nd, dBO*/

    rvec_ScaledAdd( temp, -coef.C1dbopi, bo_ij->dln_BOp_pi );        /*1st,dBOpi*/
    rvec_ScaledAdd( temp, -coef.C2dbopi, bo_ij->dBOp );              /*2nd,dBOpi*/
    rvec_ScaledAdd( temp, coef.C4dbopi, workspace->dDeltap_self[j] );/*3rd,dBOpi*/

    rvec_ScaledAdd( temp, -coef.C1dbopi2, bo_ij->dln_BOp_pi2 );       /*1st,dBOpi2*/
    rvec_ScaledAdd( temp, -coef.C2dbopi2, bo_ij->dBOp );              /*2nd,dBOpi2*/
    rvec_ScaledAdd( temp, coef.C4dbopi2, workspace->dDeltap_self[j] );/*3rd,dBOpi2*/

    /* force */
    rvec_Add( *f_j, temp );
    /* pressure */
    rvec_iMultiply( ext_press, nbr_j->rel_box, temp );
#ifdef _OPENMP
    #pragma omp critical (Add_dBond_to_Forces_NPT_ext_press)
#endif
    {
        rvec_Add( data->ext_press, ext_press );
    }

    /* if( !ivec_isZero( nbr_j->rel_box ) )
       fprintf( stderr, "%3d %3d %3d: dvec[%10.6f %10.6f %10.6f]
       ext[%3d %3d %3d] f[%10.6f %10.6f %10.6f]\n",
       i+1, system->atoms[i].x[0], system->atoms[i].x[1], system->atoms[i].x[2],
       j+1, system->atoms[j].x[0], system->atoms[j].x[1], system->atoms[j].x[2],
       j+1, nbr_j->dvec[0], nbr_j->dvec[1], nbr_j->dvec[2],
       nbr_j->rel_box[0], nbr_j->rel_box[1], nbr_j->rel_box[2],
       temp[0], temp[1], temp[2] ); */
}


void Add_dBond_to_Forces( int i, int pj, reax_system *system,
        simulation_data *data, static_storage *workspace, reax_list **lists )
{
    reax_list *bonds;
    bond_data *nbr_j, *nbr_k;
    bond_order_data *bo_ij, *bo_ji;
    dbond_coefficients coef;
    int pk, k, j;
    rvec *f_i, *f_j, *f_k;
#ifdef _OPENMP
    int tid = omp_get_thread_num( );
#endif

    /* Initializations */
    bonds = &(*lists)[BONDS];
    nbr_j = &(bonds->select.bond_list[pj]);
    j = nbr_j->nbr;
    bo_ij = &(nbr_j->bo_data);
    bo_ji = &(bonds->select.bond_list[ nbr_j->sym_index ].bo_data);
#ifdef _OPENMP
    f_i = &(workspace->f_local[tid * system->N + i]);
    f_j = &(workspace->f_local[tid * system->N + j]);
#else
    f_i = &(system->atoms[i].f);
    f_j = &(system->atoms[j].f);
#endif

    coef.C1dbo = bo_ij->C1dbo * (bo_ij->Cdbo + bo_ji->Cdbo);
    coef.C2dbo = bo_ij->C2dbo * (bo_ij->Cdbo + bo_ji->Cdbo);
    coef.C3dbo = bo_ij->C3dbo * (bo_ij->Cdbo + bo_ji->Cdbo);

    coef.C1dbopi = bo_ij->C1dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
    coef.C2dbopi = bo_ij->C2dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
    coef.C3dbopi = bo_ij->C3dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
    coef.C4dbopi = bo_ij->C4dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);

    coef.C1dbopi2 = bo_ij->C1dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
    coef.C2dbopi2 = bo_ij->C2dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
    coef.C3dbopi2 = bo_ij->C3dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
    coef.C4dbopi2 = bo_ij->C4dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);

    coef.C1dDelta = bo_ij->C1dbo * (workspace->CdDelta[i] + workspace->CdDelta[j]);
    coef.C2dDelta = bo_ij->C2dbo * (workspace->CdDelta[i] + workspace->CdDelta[j]);
    coef.C3dDelta = bo_ij->C3dbo * (workspace->CdDelta[i] + workspace->CdDelta[j]);

    for ( pk = Start_Index(i, bonds); pk < End_Index(i, bonds); ++pk )
    {
        nbr_k = &(bonds->select.bond_list[pk]);
        k = nbr_k->nbr;
#ifdef _OPENMP
        f_k = &(workspace->f_local[tid * system->N + k]);
#else
        f_k = &(system->atoms[k].f);
#endif

        rvec_ScaledAdd( *f_k, -coef.C2dbo, nbr_k->bo_data.dBOp );
        /*2nd, dBO*/
        rvec_ScaledAdd( *f_k, -coef.C2dDelta, nbr_k->bo_data.dBOp );
        /*dDelta*/
        rvec_ScaledAdd( *f_k, -coef.C3dbopi, nbr_k->bo_data.dBOp );
        /*3rd, dBOpi*/
        rvec_ScaledAdd( *f_k, -coef.C3dbopi2, nbr_k->bo_data.dBOp );
        /*3rd, dBOpi2*/
    }

    rvec_ScaledAdd( *f_i, coef.C1dbo, bo_ij->dBOp );
    /*1st, dBO*/
    rvec_ScaledAdd( *f_i, coef.C2dbo, workspace->dDeltap_self[i] );
    /*2nd, dBO*/

    rvec_ScaledAdd( *f_i, coef.C1dDelta, bo_ij->dBOp );
    /*1st, dBO*/
    rvec_ScaledAdd( *f_i, coef.C2dDelta, workspace->dDeltap_self[i] );
    /*2nd, dBO*/

    rvec_ScaledAdd( *f_i, coef.C1dbopi, bo_ij->dln_BOp_pi );
    /*1st, dBOpi*/
    rvec_ScaledAdd( *f_i, coef.C2dbopi, bo_ij->dBOp );
    /*2nd, dBOpi*/
    rvec_ScaledAdd( *f_i, coef.C3dbopi, workspace->dDeltap_self[i] );
    /*3rd, dBOpi*/

    rvec_ScaledAdd( *f_i, coef.C1dbopi2, bo_ij->dln_BOp_pi2 );
    /*1st, dBO_pi2*/
    rvec_ScaledAdd( *f_i, coef.C2dbopi2, bo_ij->dBOp );
    /*2nd, dBO_pi2*/
    rvec_ScaledAdd( *f_i, coef.C3dbopi2, workspace->dDeltap_self[i] );
    /*3rd, dBO_pi2*/

    for ( pk = Start_Index(j, bonds); pk < End_Index(j, bonds); ++pk )
    {
        nbr_k = &(bonds->select.bond_list[pk]);
        k = nbr_k->nbr;
#ifdef _OPENMP
        f_k = &(workspace->f_local[tid * system->N + k]);
#else
        f_k = &(system->atoms[k].f);
#endif

        rvec_ScaledAdd( *f_k, -coef.C3dbo, nbr_k->bo_data.dBOp );
        /*3rd, dBO*/
        rvec_ScaledAdd( *f_k, -coef.C3dDelta, nbr_k->bo_data.dBOp );
        /*dDelta*/
        rvec_ScaledAdd( *f_k, -coef.C4dbopi, nbr_k->bo_data.dBOp );
        /*4th, dBOpi*/
        rvec_ScaledAdd( *f_k, -coef.C4dbopi2, nbr_k->bo_data.dBOp );
        /*4th, dBOpi2*/
    }

    rvec_ScaledAdd( *f_j, -coef.C1dbo, bo_ij->dBOp );
    /*1st, dBO*/
    rvec_ScaledAdd( *f_j, coef.C3dbo, workspace->dDeltap_self[j] );
    /*2nd, dBO*/

    rvec_ScaledAdd( *f_j, -coef.C1dDelta, bo_ij->dBOp );
    /*1st, dBO*/
    rvec_ScaledAdd( *f_j, coef.C3dDelta, workspace->dDeltap_self[j] );
    /*2nd, dBO*/

    rvec_ScaledAdd( *f_j, -coef.C1dbopi, bo_ij->dln_BOp_pi );
    /*1st, dBOpi*/
    rvec_ScaledAdd( *f_j, -coef.C2dbopi, bo_ij->dBOp );
    /*2nd, dBOpi*/
    rvec_ScaledAdd( *f_j, coef.C4dbopi, workspace->dDeltap_self[j] );
    /*3rd, dBOpi*/

    rvec_ScaledAdd( *f_j, -coef.C1dbopi2, bo_ij->dln_BOp_pi2 );
    /*1st, dBOpi2*/
    rvec_ScaledAdd( *f_j, -coef.C2dbopi2, bo_ij->dBOp );
    /*2nd, dBOpi2*/
    rvec_ScaledAdd( *f_j, coef.C4dbopi2, workspace->dDeltap_self[j] );
    /*3rd, dBOpi2*/
}


/* Locate j on i's list.
   This function assumes that j is there for sure!
   And this is the case given our method of neighbor generation*/
static int Locate_Symmetric_Bond( reax_list *bonds, int i, int j )
{
    int start = Start_Index(i, bonds);
    int end = End_Index(i, bonds);
    int mid = (start + end) / 2;
    int mid_nbr;

    while ( (mid_nbr = bonds->select.bond_list[mid].nbr) != j )
    {
        /*fprintf( stderr, "\tstart: %d   end: %d   mid: %d\n",
        start, end, mid );*/
        if ( mid_nbr < j )
        {
            start = mid + 1;
        }
        else
        {
            end = mid - 1;
        }

        mid = (start + end) / 2;
    }

    return mid;
}


static inline void Copy_Neighbor_Data( bond_data *dest, near_neighbor_data *src )
{
    dest->nbr = src->nbr;
    dest->d = src->d;
    rvec_Copy( dest->dvec, src->dvec );
    ivec_Copy( dest->rel_box, src->rel_box );
    /* rvec_Copy( dest->ext_factor, src->ext_factor );*/
}


static inline void Copy_Bond_Order_Data( bond_order_data *dest, bond_order_data *src )
{
    dest->BO = src->BO;
    dest->BO_s = src->BO_s;
    dest->BO_pi = src->BO_pi;
    dest->BO_pi2 = src->BO_pi2;

    rvec_Scale( dest->dBOp, -1.0, src->dBOp );
    rvec_Scale( dest->dln_BOp_s, -1.0, src->dln_BOp_s );
    rvec_Scale( dest->dln_BOp_pi, -1.0, src->dln_BOp_pi );
    rvec_Scale( dest->dln_BOp_pi2, -1.0, src->dln_BOp_pi2 );
}


static int compare_bonds( const void *p1, const void *p2 )
{
    return ((bond_data *)p1)->nbr - ((bond_data *)p2)->nbr;
}


/* A very important and crucial assumption here is that each segment
   belonging to a different atom in nbrhoods->nbr_list is sorted in its own.
   This can either be done in the general coordinator function or here */
void Calculate_Bond_Orders( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace,
        reax_list **lists, output_controls *out_control )
{
    real p_lp1;
    real p_boc1, p_boc2;
    reax_list *bonds;

    p_lp1 = system->reaxprm.gp.l[15];
    p_boc1 = system->reaxprm.gp.l[0];
    p_boc2 = system->reaxprm.gp.l[1];
    bonds = &(*lists)[BONDS];

#ifdef _OPENMP
    #pragma omp parallel default(shared)
#endif
    {
        int i, j, pj, type_i, type_j;
        int start_i, end_i;
        int sym_index;
        real val_i, Deltap_i, Deltap_boc_i;
        real val_j, Deltap_j, Deltap_boc_j;
        real temp, f1, f2, f3, f4, f5, f4f5, exp_f4, exp_f5;
        real exp_p1i, exp_p2i, exp_p1j, exp_p2j;
        real u1_ij, u1_ji, Cf1A_ij, Cf1B_ij, Cf1_ij, Cf1_ji;
        real Cf45_ij, Cf45_ji;
        real A0_ij, A1_ij, A2_ij, A2_ji, A3_ij, A3_ji;
        real explp1;
        two_body_parameters *twbp;
        bond_order_data *bo_ij, *bo_ji;
        single_body_parameters *sbp_i, *sbp_j;
#if defined(TEST_FORCES)
        int k, pk, start_j, end_j;
        int top_dbo, top_dDelta;
        dbond_data *pdbo;
        dDelta_data *ptop_dDelta;
        reax_list *dDeltas;
        reax_list *dBOs;

        top_dbo = 0;
        top_dDelta = 0;
        dDeltas = &(*lists)[DDELTA];
        dBOs = &(*lists)[DBO];
#endif

        /* Calculate Deltaprime, Deltaprime_boc values */
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for ( i = 0; i < system->N; ++i )
        {
            type_i = system->atoms[i].type;
            sbp_i = &(system->reaxprm.sbp[type_i]);
            workspace->Deltap[i] = workspace->total_bond_order[i] - sbp_i->valency;
            workspace->Deltap_boc[i] =
                workspace->total_bond_order[i] - sbp_i->valency_val;
            workspace->total_bond_order[i] = 0.0;
        }

        /* wait until initialization complete */
#ifdef _OPENMP
        #pragma omp barrier
#endif

        /* Corrected Bond Order calculations */
#ifdef _OPENMP
        #pragma omp for schedule(guided)
#endif
        for ( i = 0; i < system->N; ++i )
        {
            type_i = system->atoms[i].type;
            sbp_i = &(system->reaxprm.sbp[type_i]);
            val_i = sbp_i->valency;
            Deltap_i = workspace->Deltap[i];
            Deltap_boc_i = workspace->Deltap_boc[i];
            start_i = Start_Index(i, bonds);
            end_i = End_Index(i, bonds);

            for ( pj = start_i; pj < end_i; ++pj )
            {
                j = bonds->select.bond_list[pj].nbr;
                type_j = system->atoms[j].type;
                bo_ij = &( bonds->select.bond_list[pj].bo_data );

                if ( i < j )
                {
                    twbp = &( system->reaxprm.tbp[type_i][type_j] );

#ifdef TEST_FORCES
                    Set_Start_Index( pj, top_dbo, dBOs );
                    /* fprintf( stderr, "%6d%6d%23.15e%23.15e%23.15e\n",
                       workspace->reverse_map[i], workspace->reverse_map[j],
                       twbp->ovc, twbp->v13cor, bo_ij->BO ); */
#endif

                    if ( twbp->ovc < 0.001 && twbp->v13cor < 0.001 )
                    {
                        /* There is no correction to bond orders nor to derivatives of
                           bond order prime! So we leave bond orders unchanged and
                           set derivative of bond order coefficients s.t.
                           dBO = dBOp & dBOxx = dBOxxp in Add_dBO_to_Forces */
                        bo_ij->C1dbo = 1.000000;
                        bo_ij->C2dbo = 0.000000;
                        bo_ij->C3dbo = 0.000000;

                        bo_ij->C1dbopi = bo_ij->BO_pi;
                        bo_ij->C2dbopi = 0.000000;
                        bo_ij->C3dbopi = 0.000000;
                        bo_ij->C4dbopi = 0.000000;

                        bo_ij->C1dbopi2 = bo_ij->BO_pi2;
                        bo_ij->C2dbopi2 = 0.000000;
                        bo_ij->C3dbopi2 = 0.000000;
                        bo_ij->C4dbopi2 = 0.000000;

#ifdef TEST_FORCES
                        pdbo = &(dBOs->select.dbo_list[ top_dbo ]);

                        /* compute dBO_ij/dr_i */
                        pdbo->wrt = i;
                        rvec_Copy( pdbo->dBO, bo_ij->dBOp );
                        rvec_Scale( pdbo->dBOpi, bo_ij->BO_pi, bo_ij->dln_BOp_pi );
                        rvec_Scale( pdbo->dBOpi2, bo_ij->BO_pi2, bo_ij->dln_BOp_pi2 );

                        /* compute dBO_ij/dr_j */
                        pdbo++;
                        pdbo->wrt = j;
                        rvec_Scale( pdbo->dBO, -1.0, bo_ij->dBOp );
                        rvec_Scale( pdbo->dBOpi, -bo_ij->BO_pi, bo_ij->dln_BOp_pi );
                        rvec_Scale( pdbo->dBOpi2, -bo_ij->BO_pi2, bo_ij->dln_BOp_pi2 );

                        top_dbo += 2;
#endif
                    }
                    else
                    {
                        val_j = system->reaxprm.sbp[type_j].valency;
                        Deltap_j = workspace->Deltap[j];
                        Deltap_boc_j = workspace->Deltap_boc[j];

                        /* on page 1 */
                        if ( twbp->ovc >= 0.001 )
                        {
                            /* Correction for overcoordination */
                            exp_p1i = EXP( -p_boc1 * Deltap_i );
                            exp_p2i = EXP( -p_boc2 * Deltap_i );
                            exp_p1j = EXP( -p_boc1 * Deltap_j );
                            exp_p2j = EXP( -p_boc2 * Deltap_j );

                            f2 = exp_p1i + exp_p1j;
                            f3 = -1.0 / p_boc2 * log( 0.5 * ( exp_p2i  + exp_p2j ) );
                            f1 = 0.5 * ( ( val_i + f2 ) / ( val_i + f2 + f3 ) +
                                    ( val_j + f2 ) / ( val_j + f2 + f3 ) );

                            /* Now come the derivates */
                            /* Bond Order pages 5-7, derivative of f1 */
                            temp = f2 + f3;
                            u1_ij = val_i + temp;
                            u1_ji = val_j + temp;
                            Cf1A_ij = 0.5 * f3 * (1.0 / SQR( u1_ij ) + 1.0 / SQR( u1_ji ));
                            Cf1B_ij = -0.5 * (( u1_ij - f3 ) / SQR( u1_ij ) +
                                    ( u1_ji - f3 ) / SQR( u1_ji ));

                            //Cf1_ij = -Cf1A_ij * p_boc1 * exp_p1i +
                            //          Cf1B_ij * exp_p2i / ( exp_p2i + exp_p2j );
                            Cf1_ij = 0.50 * ( -p_boc1 * exp_p1i / u1_ij -
                                    ((val_i + f2) / SQR(u1_ij)) * ( -p_boc1 * exp_p1i +
                                    exp_p2i / ( exp_p2i + exp_p2j ) ) + -p_boc1 * exp_p1i / u1_ji -
                                    ((val_j + f2) / SQR(u1_ji)) * ( -p_boc1 * exp_p1i +
                                    exp_p2i / ( exp_p2i + exp_p2j ) ));

                            Cf1_ji = -Cf1A_ij * p_boc1 * exp_p1j +
                                Cf1B_ij * exp_p2j / ( exp_p2i + exp_p2j );
                        }
                        else
                        {
                            /* No overcoordination correction! */
                            f1 = 1.0;
                            Cf1_ij = Cf1_ji = 0.0;
                        }

                        if ( twbp->v13cor >= 0.001 )
                        {
                            /* Correction for 1-3 bond orders */
                            exp_f4 = EXP(-(twbp->p_boc4 * SQR( bo_ij->BO ) -
                                           Deltap_boc_i) * twbp->p_boc3 + twbp->p_boc5);
                            exp_f5 = EXP(-(twbp->p_boc4 * SQR( bo_ij->BO ) -
                                           Deltap_boc_j) * twbp->p_boc3 + twbp->p_boc5);

                            f4 = 1. / (1. + exp_f4);
                            f5 = 1. / (1. + exp_f5);
                            f4f5 = f4 * f5;

                            /* Bond Order pages 8-9, derivative of f4 and f5 */
                            /*temp = twbp->p_boc5 -
                              twbp->p_boc3 * twbp->p_boc4 * SQR( bo_ij->BO );
                              u_ij = temp + twbp->p_boc3 * Deltap_boc_i;
                              u_ji = temp + twbp->p_boc3 * Deltap_boc_j;
                              Cf45_ij = Cf45( u_ij, u_ji ) / f4f5;
                              Cf45_ji = Cf45( u_ji, u_ij ) / f4f5;*/
                            Cf45_ij = -f4 * exp_f4;
                            Cf45_ji = -f5 * exp_f5;
                        }
                        else
                        {
                            f4 = f5 = f4f5 = 1.0;
                            Cf45_ij = Cf45_ji = 0.0;
                        }

                        /* Bond Order page 10, derivative of total bond order */
                        A0_ij = f1 * f4f5;
                        A1_ij = -2 * twbp->p_boc3 * twbp->p_boc4 * bo_ij->BO *
                            (Cf45_ij + Cf45_ji);
                        A2_ij = Cf1_ij / f1 + twbp->p_boc3 * Cf45_ij;
                        A2_ji = Cf1_ji / f1 + twbp->p_boc3 * Cf45_ji;
                        A3_ij = A2_ij + Cf1_ij / f1;
                        A3_ji = A2_ji + Cf1_ji / f1;

                        /* find corrected bond order values and their deriv coefs */
                        bo_ij->BO = bo_ij->BO * A0_ij;
                        bo_ij->BO_pi = bo_ij->BO_pi * A0_ij * f1;
                        bo_ij->BO_pi2 = bo_ij->BO_pi2 * A0_ij * f1;
                        bo_ij->BO_s = bo_ij->BO - ( bo_ij->BO_pi + bo_ij->BO_pi2 );

                        bo_ij->C1dbo = A0_ij + bo_ij->BO * A1_ij;
                        bo_ij->C2dbo = bo_ij->BO * A2_ij;
                        bo_ij->C3dbo = bo_ij->BO * A2_ji;

                        bo_ij->C1dbopi = f1 * f1 * f4 * f5;
                        bo_ij->C2dbopi = bo_ij->BO_pi * A1_ij;
                        bo_ij->C3dbopi = bo_ij->BO_pi * A3_ij;
                        bo_ij->C4dbopi = bo_ij->BO_pi * A3_ji;

                        bo_ij->C1dbopi2 = f1 * f1 * f4 * f5;
                        bo_ij->C2dbopi2 = bo_ij->BO_pi2 * A1_ij;
                        bo_ij->C3dbopi2 = bo_ij->BO_pi2 * A3_ij;
                        bo_ij->C4dbopi2 = bo_ij->BO_pi2 * A3_ji;

#ifdef TEST_FORCES
                        /*fprintf( stderr, "%6d%6d%13.6f%13.6f%13.6f%13.6f\n",
                          i+1, j+1, bo_ij->BO, bo_ij->C1dbo, Cf45_ij, Cf45_ji );*/

                        /* fprintf( stderr, "%6d%6d%13.6f%13.6f%13.6f%13.6f\n",
                        //"%6d%6d%10.6f%10.6f%10.6f%10.6f\n%10.6f%10.6f%10.6f\n%10.6f%10.6f%10.6f%10.6f\n%10.6f%10.6f%10.6f%10.6f\n\n",
                        workspace->orig_id[i], workspace->orig_id[j]
                        A0_ij, A1_ij, A2_ij, A2_ji, A3_ij, A3_ji
                        bo_ij->BO, bo_ij->BO_pi, bo_ij->BO_pi2, bo_ij->BO_s,
                        bo_ij->C1dbo, bo_ij->C2dbo, bo_ij->C3dbo,
                        bo_ij->C1dbopi,bo_ij->C2dbopi,bo_ij->C3dbopi,bo_ij->C4dbopi,
                        bo_ij->C1dbopi2,bo_ij->C2dbopi2,bo_ij->C3dbopi2,bo_ij->C4dbopi2
                        ); */

                        Calculate_dBO( i, pj, workspace, lists, &top_dbo );
#endif
                    }

                    /* neglect bonds that are < 1e-10 */
                    if ( bo_ij->BO < 1e-10 )
                    {
                        bo_ij->BO = 0.0;
                    }
                    if ( bo_ij->BO_s < 1e-10 )
                    {
                        bo_ij->BO_s = 0.0;
                    }
                    if ( bo_ij->BO_pi < 1e-10 )
                    {
                        bo_ij->BO_pi = 0.0;
                    }
                    if ( bo_ij->BO_pi2 < 1e-10 )
                    {
                        bo_ij->BO_pi2 = 0.0;
                    }

                    workspace->total_bond_order[i] += bo_ij->BO; // now keeps total_BO

#ifdef TEST_FORCES
                    Set_End_Index( pj, top_dbo, dBOs );
                    Add_dBO( system, lists, i, pj, 1.0, workspace->dDelta );
#endif
                }
            }

#ifdef TEST_FORCES
            Set_Start_Index( i, top_dDelta, dDeltas );
            ptop_dDelta = &( dDeltas->select.dDelta_list[top_dDelta] );

            for ( pj = start_i; pj < end_i; ++pj )
            {
                j = bonds->select.bond_list[pj].nbr;

                if ( !rvec_isZero( workspace->dDelta[j] ) )
                {
                    ptop_dDelta->wrt = j;
                    rvec_Copy( ptop_dDelta->dVal, workspace->dDelta[j] );
                    rvec_MakeZero( workspace->dDelta[j] );
                    ++top_dDelta, ++ptop_dDelta;
                }

                start_j = Start_Index(j, bonds);
                end_j = End_Index(j, bonds);
                for ( pk = start_j; pk < end_j; ++pk )
                {
                    k = bonds->select.bond_list[pk].nbr;
                    if ( !rvec_isZero( workspace->dDelta[k] ) )
                    {
                        ptop_dDelta->wrt = k;
                        rvec_Copy( ptop_dDelta->dVal, workspace->dDelta[k] );
                        rvec_MakeZero( workspace->dDelta[k] );
                        ++top_dDelta, ++ptop_dDelta;
                    }
                }
            }

            Set_End_Index( i, top_dDelta, dDeltas );

            /*for( pj=Start_Index(i,dDeltas); pj<End_Index(i,dDeltas); ++pj )
              fprintf( stdout, "dDel: %d %d [%g %g %g]\n",
              i+1, dDeltas->select.dDelta_list[pj].wrt+1,
              dDeltas->select.dDelta_list[pj].dVal[0],
              dDeltas->select.dDelta_list[pj].dVal[1],
              dDeltas->select.dDelta_list[pj].dVal[2] );*/
#endif
        }

        /* wait for bo_ij to be updated */
#ifdef _OPENMP
        #pragma omp barrier
#endif

#ifdef _OPENMP
        #pragma omp for schedule(guided)
#endif
        for ( i = 0; i < system->N; ++i )
        {
            type_i = system->atoms[i].type;
            if ( type_i < 0 )
            {
                continue;
            }
            start_i = Start_Index(i, bonds);
            end_i = End_Index(i, bonds);

            for ( pj = start_i; pj < end_i; ++pj )
            {
                j = bonds->select.bond_list[pj].nbr;
                type_j = system->atoms[j].type;
                if ( type_j < 0 )
                {
                    continue;
                }

                if ( i < j )
                {
                    /* computed in previous for-loop */
                }
                else
                {
                    /* We only need to update bond orders from bo_ji
                       everything else is set in uncorrected_bo calculations */
                    sym_index = bonds->select.bond_list[pj].sym_index;

                    bo_ij = &(bonds->select.bond_list[ pj ].bo_data);
                    bo_ji = &(bonds->select.bond_list[ sym_index ].bo_data);
                    bo_ij->BO = bo_ji->BO;
                    bo_ij->BO_s = bo_ji->BO_s;
                    bo_ij->BO_pi = bo_ji->BO_pi;
                    bo_ij->BO_pi2 = bo_ji->BO_pi2;

                    /* now keeps total_BO */
                    workspace->total_bond_order[i] += bo_ij->BO;

#ifdef TEST_FORCES
                    Add_dBO( system, lists, j, sym_index, 1.0, workspace->dDelta );
#endif
                }
            }
        }

        /* need to wait for total_bond_order to be accumulated */
#ifdef _OPENMP
        #pragma omp barrier
#endif

        /* Calculate some helper variables that are  used at many places
           throughout force calculations */
#ifdef _OPENMP
        #pragma omp for schedule(guided)
#endif
        for ( j = 0; j < system->N; ++j )
        {
            type_j = system->atoms[j].type;
            sbp_j = &(system->reaxprm.sbp[ type_j ]);

            workspace->Delta[j] = workspace->total_bond_order[j] - sbp_j->valency;
            workspace->Delta_e[j] = workspace->total_bond_order[j] - sbp_j->valency_e;
            workspace->Delta_boc[j] = workspace->total_bond_order[j] -
                sbp_j->valency_boc;

            workspace->vlpex[j] = workspace->Delta_e[j] -
                2.0 * (int)(workspace->Delta_e[j] / 2.0);
            explp1 = EXP(-p_lp1 * SQR(2.0 + workspace->vlpex[j]));
            workspace->nlp[j] = explp1 - (int)(workspace->Delta_e[j] / 2.0);
            workspace->Delta_lp[j] = sbp_j->nlp_opt - workspace->nlp[j];
            workspace->Clp[j] = 2.0 * p_lp1 * explp1 * (2.0 + workspace->vlpex[j]);
            /* Adri uses different dDelta_lp values than the ones in notes... */
            workspace->dDelta_lp[j] = workspace->Clp[j];
            //workspace->dDelta_lp[j] = workspace->Clp[j] + (0.5-workspace->Clp[j]) *
            //((FABS(workspace->Delta_e[j]/2.0 -
            //       (int)(workspace->Delta_e[j]/2.0)) < 0.1) ? 1 : 0 );

            if ( sbp_j->mass > 21.0 )
            {
                workspace->nlp_temp[j] = 0.5 * (sbp_j->valency_e - sbp_j->valency);
                workspace->Delta_lp_temp[j] = sbp_j->nlp_opt - workspace->nlp_temp[j];
                workspace->dDelta_lp_temp[j] = 0.0;
            }
            else
            {
                workspace->nlp_temp[j] = workspace->nlp[j];
                workspace->Delta_lp_temp[j] = sbp_j->nlp_opt - workspace->nlp_temp[j];
                workspace->dDelta_lp_temp[j] = workspace->Clp[j];
            }
        }
    }

#if defined(DEBUG)
    Print_Bond_Orders( system, control, data, workspace, lists, out_control );
#endif
}
