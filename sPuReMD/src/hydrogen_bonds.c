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

#include "hydrogen_bonds.h"

#include "bond_orders.h"
#include "list.h"
#include "lookup.h"
#include "valence_angles.h"
#include "vector.h"

#include "tool_box.h"


void Hydrogen_Bonds( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace,
        reax_list **lists, output_controls *out_control )
{
#ifdef TEST_FORCES
    int num_hb_intrs;
#endif
    real e_hb_total;

    e_hb_total = 0.0;
#ifdef TEST_FORCES
    num_hb_intrs = 0;
#endif

#ifdef _OPENMP
    #pragma omp parallel default(shared) reduction(+: e_hb_total)
#endif
    {
        int i, j, k, pi, pk, itr, top;
        int type_i, type_j, type_k;
        int start_j, end_j, hb_start_j, hb_end_j;
        int hblist[MAX_BONDS];
        real r_ij, r_jk, theta, cos_theta, sin_xhz4, cos_xhz1, sin_theta2;
        real e_hb, exp_hb2, exp_hb3, CEhb1, CEhb2, CEhb3;
        rvec dcos_theta_di, dcos_theta_dj, dcos_theta_dk;
        rvec dvec_jk, force, ext_press;
        ivec rel_jk;
        //rtensor temp_rtensor, total_rtensor;
        hbond_parameters *hbp;
        bond_order_data *bo_ij;
        bond_data *pbond_ij;
        far_neighbor_data *nbr_jk;
        reax_list *bonds, *hbonds;
        bond_data *bond_list;
        hbond_data *hbond_list;
        rvec *f_i, *f_j, *f_k;
#ifdef _OPENMP
        int tid = omp_get_thread_num( );
#endif

        bonds = lists[BONDS];
        bond_list = bonds->bond_list;
        hbonds = lists[HBONDS];
        hbond_list = hbonds->hbond_list;

        /* loops below discover the Hydrogen bonds between i-j-k triplets.
         * here j is H atom and there has to be some bond between i and j.
         * Hydrogen bond is between j and k.
         * so in this function i->X, j->H, k->Z when we map
         * variables onto the ones in the handout. */
#ifdef _OPENMP
        #pragma omp for schedule(guided)
#endif
        for ( j = 0; j < system->N; ++j )
        {
            /* j must be H */
            if ( system->reax_param.sbp[system->atoms[j].type].p_hbond == 1 )
            {
                /* set j's variables */
                type_j = system->atoms[j].type;
                start_j = Start_Index( j, bonds );
                end_j = End_Index( j, bonds );
                hb_start_j = Start_Index( workspace->hbond_index[j], hbonds );
                hb_end_j = End_Index( workspace->hbond_index[j], hbonds );
#ifdef _OPENMP
                f_j = &workspace->f_local[tid * system->N + j];
#else
                f_j = &system->atoms[j].f;
#endif

                top = 0;
                for ( pi = start_j; pi < end_j; ++pi )
                {
                    pbond_ij = &bond_list[pi];
                    i = pbond_ij->nbr;
                    bo_ij = &pbond_ij->bo_data;
                    type_i = system->atoms[i].type;

                    if ( system->reax_param.sbp[type_i].p_hbond == 2
                            && bo_ij->BO >= HB_THRESHOLD )
                    {
                        hblist[top++] = pi;
                    }
                }

                for ( pk = hb_start_j; pk < hb_end_j; ++pk )
                {
                    /* set k's varibles */
                    k = hbond_list[pk].nbr;
                    type_k = system->atoms[k].type;
                    nbr_jk = hbond_list[pk].ptr;
                    r_jk = nbr_jk->d;
                    rvec_Scale( dvec_jk, hbond_list[pk].scl, nbr_jk->dvec );
#ifdef _OPENMP
                    f_k = &workspace->f_local[tid * system->N + k];
#else
                    f_k = &system->atoms[k].f;
#endif

                    for ( itr = 0; itr < top; ++itr )
                    {
                        pi = hblist[itr];
                        pbond_ij = &bond_list[pi];
                        i = pbond_ij->nbr;

                        if ( i != k )
                        {
                            bo_ij = &pbond_ij->bo_data;
                            type_i = system->atoms[i].type;
                            r_ij = pbond_ij->d;
                            hbp = &system->reax_param.hbp[ type_i ][ type_j ][ type_k ];
#ifdef _OPENMP
                            f_i = &workspace->f_local[tid * system->N + i];
#else
                            f_i = &system->atoms[i].f;
#endif

#ifdef TEST_FORCES
#ifdef _OPENMP
                            #pragma omp atomic
#endif
                            ++num_hb_intrs;
#endif

                            Calculate_Theta( pbond_ij->dvec, pbond_ij->d, dvec_jk, r_jk,
                                    &theta, &cos_theta );
                            /* the derivative of cos(theta) */
                            Calculate_dCos_Theta( pbond_ij->dvec, pbond_ij->d, dvec_jk, r_jk,
                                    &dcos_theta_di, &dcos_theta_dj, &dcos_theta_dk );

                            /* hydrogen bond energy */
                            sin_theta2 = SIN( theta / 2.0 );
                            sin_xhz4 = SQR( sin_theta2 );
                            sin_xhz4 *= sin_xhz4;
                            cos_xhz1 = ( 1.0 - cos_theta );
                            exp_hb2 = EXP( -hbp->p_hb2 * bo_ij->BO );
                            exp_hb3 = EXP( -hbp->p_hb3 * ( hbp->r0_hb / r_jk
                                        + r_jk / hbp->r0_hb - 2.0 ) );

                            e_hb = hbp->p_hb1 * (1.0 - exp_hb2) * exp_hb3 * sin_xhz4;
                            e_hb_total += e_hb;

                            CEhb1 = hbp->p_hb1 * hbp->p_hb2 * exp_hb2 * exp_hb3 * sin_xhz4;
                            CEhb2 = -hbp->p_hb1 / 2.0 * (1.0 - exp_hb2) * exp_hb3 * cos_xhz1;
                            CEhb3 = -hbp->p_hb3 * e_hb * (-hbp->r0_hb / SQR( r_jk )
                                    + 1.0 / hbp->r0_hb);

                            /* hydrogen bond forces */
                            /* dbo term,
                             * note: safe to update across threads as this points
                             * to the bond_order_data struct inside atom j's list,
                             * and threads are partitioned across all j's */
#ifdef _OPENMP
                            #pragma omp atomic
#endif
                            bo_ij->Cdbo += CEhb1;

                            if ( control->ensemble == NVE || control->ensemble == nhNVT
                                    || control->ensemble == bNVT )
                            {
                                /* dcos terms */
                                rvec_ScaledAdd( *f_i, +CEhb2, dcos_theta_di );
                                rvec_ScaledAdd( *f_j, +CEhb2, dcos_theta_dj );
                                rvec_ScaledAdd( *f_k, +CEhb2, dcos_theta_dk );

                                /* dr terms */
                                rvec_ScaledAdd( *f_j, -CEhb3 / r_jk, dvec_jk );
                                rvec_ScaledAdd( *f_k, +CEhb3 / r_jk, dvec_jk );
                            }
                            else
                            {
                                /* for pressure coupling, terms that are not related
                                   to bond order derivatives are added directly into
                                   pressure vector/tensor */

                                /* dcos terms */
                                rvec_Scale( force, +CEhb2, dcos_theta_di );
                                rvec_Add( *f_i, force );
                                rvec_iMultiply( ext_press, pbond_ij->rel_box, force );
#ifdef _OPENMP
                                #pragma omp critical (Hydrogen_Bonds_ext_press)
#endif
                                {
                                    rvec_ScaledAdd( data->ext_press, 1.0, ext_press );
                                }

                                rvec_ScaledAdd( *f_j, +CEhb2, dcos_theta_dj );

                                ivec_Scale( rel_jk, hbond_list[pk].scl, nbr_jk->rel_box );
                                rvec_Scale( force, +CEhb2, dcos_theta_dk );
                                rvec_Add( *f_k, force );
                                rvec_iMultiply( ext_press, rel_jk, force );
#ifdef _OPENMP
                                #pragma omp critical (Hydrogen_Bonds_ext_press)
#endif
                                {
                                    rvec_ScaledAdd( data->ext_press, 1.0, ext_press );
                                }

                                /* dr terms */
                                rvec_ScaledAdd( *f_j, -CEhb3 / r_jk, dvec_jk );

                                rvec_Scale( force, CEhb3 / r_jk, dvec_jk );
                                rvec_Add( *f_k, force );
                                rvec_iMultiply( ext_press, rel_jk, force );
#ifdef _OPENMP
                                #pragma omp critical (Hydrogen_Bonds_ext_press)
#endif
                                {
                                    rvec_ScaledAdd( data->ext_press, 1.0, ext_press );
                                }

                                /* This part is intended for a fully-flexible box */
                                /* rvec_OuterProduct( temp_rtensor,
                                   dcos_theta_di, system->atoms[i].x );
                                   rtensor_Scale( total_rtensor, -CEhb2, temp_rtensor );

                                   rvec_ScaledSum( temp_rvec, -CEhb2, dcos_theta_dj,
                                   -CEhb3/r_jk, pbond_jk->dvec );
                                   rvec_OuterProduct( temp_rtensor,
                                   temp_rvec, system->atoms[j].x );
                                   rtensor_Add( total_rtensor, temp_rtensor );

                                   rvec_ScaledSum( temp_rvec, -CEhb2, dcos_theta_dk,
                                   +CEhb3/r_jk, pbond_jk->dvec );
                                   rvec_OuterProduct( temp_rtensor,
                                   temp_rvec, system->atoms[k].x );
                                   rtensor_Add( total_rtensor, temp_rtensor );

                                   if( pbond_ij->imaginary || pbond_jk->imaginary )
                                   rtensor_ScaledAdd( data->flex_bar.P, -1.0, total_rtensor );
                                   else
                                   rtensor_Add( data->flex_bar.P, total_rtensor ); */
                            }

#ifdef TEST_ENERGY
                            /*fprintf( out_control->ehb,
                              "%23.15e%23.15e%23.15e\n%23.15e%23.15e%23.15e\n%23.15e%23.15e%23.15e\n",
                              dcos_theta_di[0], dcos_theta_di[1], dcos_theta_di[2],
                              dcos_theta_dj[0], dcos_theta_dj[1], dcos_theta_dj[2],
                              dcos_theta_dk[0], dcos_theta_dk[1], dcos_theta_dk[2]);
                              fprintf( out_control->ehb, "%23.15e%23.15e%23.15e\n",
                              CEhb1, CEhb2, CEhb3 ); */
                            fprintf( stderr, //out_control->ehb,
                                     "%6d%6d%6d%23.15e%23.15e%23.15e%23.15e%23.15e\n",
                                     workspace->orig_id[i],
                                     workspace->orig_id[j],
                                     workspace->orig_id[k],
                                     r_jk, theta, bo_ij->BO, e_hb, data->E_HB );
#endif

#ifdef TEST_FORCES
                            /* dbo term */
                            Add_dBO( system, lists, j, pi, +CEhb1, workspace->f_hb );
                            /* dcos terms */
                            rvec_ScaledAdd( workspace->f_hb[i], +CEhb2, dcos_theta_di );
                            rvec_ScaledAdd( workspace->f_hb[j], +CEhb2, dcos_theta_dj );
                            rvec_ScaledAdd( workspace->f_hb[k], +CEhb2, dcos_theta_dk );
                            /* dr terms */
                            rvec_ScaledAdd( workspace->f_hb[j], -CEhb3 / r_jk, dvec_jk );
                            rvec_ScaledAdd( workspace->f_hb[k], +CEhb3 / r_jk, dvec_jk );
#endif
                        }
                    }
                }
            }
        }
    }

    data->E_HB += e_hb_total;

#ifdef TEST_FORCES
    fprintf( stderr, "Number of hydrogen bonds: %d\n", num_hb_intrs );
    fprintf( stderr, "Hydrogen Bond Energy: %g\n", data->E_HB );
#endif
}