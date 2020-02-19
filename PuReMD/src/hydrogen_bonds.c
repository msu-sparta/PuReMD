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

#include "reax_types.h"
#if defined(PURE_REAX)
#include "hydrogen_bonds.h"
#include "bond_orders.h"
#include "list.h"
#include "valence_angles.h"
#include "vector.h"
#elif defined(LAMMPS_REAX)
#include "reax_hydrogen_bonds.h"
#include "reax_bond_orders.h"
#include "reax_list.h"
#include "reax_valence_angles.h"
#include "reax_vector.h"
#endif


void Hydrogen_Bonds( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control )
{
    int i, j, k, pi, pk, itr, top;
    int type_i, type_j, type_k;
    int start_j, end_j, hb_start_j, hb_end_j;
    int hblist[MAX_BONDS];
#if defined(DEBUG_FOCUS)
    int num_hb_intrs;
#endif
    ivec rel_jk;
    real r_ij, r_jk, theta, cos_theta, sin_xhz4, cos_xhz1, sin_theta2;
    real e_hb, exp_hb2, exp_hb3, CEhb1, CEhb2, CEhb3;
    rvec dcos_theta_di, dcos_theta_dj, dcos_theta_dk;
    rvec dvec_jk, force, ext_press;
    hbond_parameters *hbp;
    bond_order_data *bo_ij;
    bond_data *pbond_ij;
    int nbr_jk;
    reax_list *far_nbr_list, *bond_list, *hbond_list;

    far_nbr_list = lists[FAR_NBRS];
    bond_list = lists[BONDS];
    hbond_list = lists[HBONDS];
#if defined(DEBUG_FOCUS)
    num_hb_intrs = 0;
#endif

    /* loops below discover the Hydrogen bonds between i-j-k triplets.
     * here j is H atom and there has to be some bond between i and j.
     * Hydrogen bond is between j and k.
     * so in this function i->X, j->H, k->Z when we map
     * variables onto the ones in the handout. */
    for ( j = 0; j < system->n; ++j )
    {
        /* j must be a hydrogen atom */
        if ( system->reax_param.sbp[system->my_atoms[j].type].p_hbond == H_ATOM )
        {
            type_j = system->my_atoms[j].type;
            start_j = Start_Index( j, bond_list );
            end_j = End_Index( j, bond_list );
            hb_start_j = Start_Index( system->my_atoms[j].Hindex, hbond_list );
            hb_end_j = End_Index( system->my_atoms[j].Hindex, hbond_list );

            top = 0;
            for ( pi = start_j; pi < end_j; ++pi )
            {
                pbond_ij = &bond_list->bond_list[pi];
                i = pbond_ij->nbr;
                bo_ij = &pbond_ij->bo_data;
                type_i = system->my_atoms[i].type;

                if ( system->reax_param.sbp[type_i].p_hbond == H_BONDING_ATOM
                        && bo_ij->BO >= HB_THRESHOLD )
                {
                    hblist[top++] = pi;
                }
            }

            for ( pk = hb_start_j; pk < hb_end_j; ++pk )
            {
                k = hbond_list->hbond_list[pk].nbr;
                type_k = system->my_atoms[k].type;
                nbr_jk = hbond_list->hbond_list[pk].ptr;
                r_jk = far_nbr_list->far_nbr_list.d[nbr_jk];

                rvec_Scale( dvec_jk, hbond_list->hbond_list[pk].scl,
                        far_nbr_list->far_nbr_list.dvec[nbr_jk] );

                for ( itr = 0; itr < top; ++itr )
                {
                    pi = hblist[itr];
                    pbond_ij = &bond_list->bond_list[pi];
                    i = pbond_ij->nbr;

                    if ( system->my_atoms[i].orig_id != system->my_atoms[k].orig_id )
                    {
                        bo_ij = &pbond_ij->bo_data;
                        type_i = system->my_atoms[i].type;
                        r_ij = pbond_ij->d;
                        hbp = &system->reax_param.hbp[ type_i ][ type_j ][ type_k ];

#if defined(DEBUG_FOCUS)
                        ++num_hb_intrs;
#endif

                        Calculate_Theta( pbond_ij->dvec, r_ij, dvec_jk, r_jk,
                                &theta, &cos_theta );
                        /* the derivative of cos(theta) */
                        Calculate_dCos_Theta( pbond_ij->dvec, r_ij, dvec_jk, r_jk,
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
                        data->my_en.e_hb += e_hb;

                        CEhb1 = hbp->p_hb1 * hbp->p_hb2 * exp_hb2 * exp_hb3 * sin_xhz4;
                        CEhb2 = -hbp->p_hb1 / 2.0 * (1.0 - exp_hb2) * exp_hb3 * cos_xhz1;
                        CEhb3 = -hbp->p_hb3 * e_hb * (-hbp->r0_hb / SQR( r_jk )
                                + 1.0 / hbp->r0_hb);

                        /* hydrogen bond forces */
                        /* dbo term */
                        bo_ij->Cdbo += CEhb1;

                        if ( control->virial == 0 )
                        {
                            /* dcos terms */
                            rvec_ScaledAdd( workspace->f[i], +CEhb2, dcos_theta_di );
                            rvec_ScaledAdd( workspace->f[j], +CEhb2, dcos_theta_dj );
                            rvec_ScaledAdd( workspace->f[k], +CEhb2, dcos_theta_dk );
                            /* dr terms */
                            rvec_ScaledAdd( workspace->f[j], -CEhb3 / r_jk, dvec_jk );
                            rvec_ScaledAdd( workspace->f[k], +CEhb3 / r_jk, dvec_jk );
                        }
                        else
                        {
                            /* for pressure coupling, terms that are not related
                             * to bond order derivatives are added directly into
                             * pressure vector/tensor */

                            /* dcos terms */
                            rvec_Scale( force, +CEhb2, dcos_theta_di );
                            rvec_Add( workspace->f[i], force );
                            rvec_iMultiply( ext_press, pbond_ij->rel_box, force );
                            rvec_ScaledAdd( data->my_ext_press, 1.0, ext_press );

                            rvec_ScaledAdd( workspace->f[j], +CEhb2, dcos_theta_dj );

                            ivec_Scale( rel_jk, hbond_list->hbond_list[pk].scl,
                                    far_nbr_list->far_nbr_list.rel_box[nbr_jk] );
                            rvec_Scale( force, +CEhb2, dcos_theta_dk );
                            rvec_Add( workspace->f[k], force );
                            rvec_iMultiply( ext_press, rel_jk, force );
                            rvec_ScaledAdd( data->my_ext_press, 1.0, ext_press );

                            /* dr terms */
                            rvec_ScaledAdd( workspace->f[j], -CEhb3 / r_jk, dvec_jk );

                            rvec_Scale( force, CEhb3 / r_jk, dvec_jk );
                            rvec_Add( workspace->f[k], force );
                            rvec_iMultiply( ext_press, rel_jk, force );
                            rvec_ScaledAdd( data->my_ext_press, 1.0, ext_press );
                        }

#if defined(TEST_ENERGY)
                        /* fprintf( out_control->ehb,
                           "%24.15e%24.15e%24.15e\n%24.15e%24.15e%24.15e\n%24.15e%24.15e%24.15e\n",
                           dcos_theta_di[0], dcos_theta_di[1], dcos_theta_di[2],
                           dcos_theta_dj[0], dcos_theta_dj[1], dcos_theta_dj[2],
                           dcos_theta_dk[0], dcos_theta_dk[1], dcos_theta_dk[2]);
                           fprintf( out_control->ehb, "%24.15e%24.15e%24.15e\n",
                           CEhb1, CEhb2, CEhb3 ); */
                        fprintf( out_control->ehb,
                                 //"%6d%6d%6d%24.15e%24.15e%24.15e%24.15e%24.15e\n",
                                 "%6d%6d%6d%12.4f%12.4f%12.4f%12.4f%12.4f\n",
                                 system->my_atoms[i].orig_id, system->my_atoms[j].orig_id,
                                 system->my_atoms[k].orig_id,
                                 r_jk, theta, bo_ij->BO, e_hb, data->my_en.e_hb );
#endif

#if defined(TEST_FORCES)
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

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "Number of hydrogen bonds: %d\n", num_hb_intrs );
    fprintf( stderr, "Hydrogen Bond Energy: %g\n", data->my_en.e_hb );
#endif
}
