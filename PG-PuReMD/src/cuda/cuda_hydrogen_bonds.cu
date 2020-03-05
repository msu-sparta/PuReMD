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

#include "cuda_hydrogen_bonds.h"

#include "cuda_valence_angles.h"
#include "cuda_helpers.h"
#include "cuda_list.h"
#include "cuda_shuffle.h"

#include "../index_utils.h"
#include "../vector.h"


CUDA_GLOBAL void Cuda_Hydrogen_Bonds( reax_atom *my_atoms, single_body_parameters *sbp, 
        hbond_parameters *d_hbp, global_parameters gp,
        control_params *control, storage p_workspace,
        reax_list p_far_nbr_list, reax_list p_bond_list, reax_list p_hbond_list, int n, 
        int num_atom_types, real *data_e_hb, rvec *data_ext_press, int rank, int step )
{
    int i, j, k, pi, pk;
    int type_i, type_j, type_k;
    int start_j, end_j, hb_start_j, hb_end_j;
    int hblist[MAX_BONDS];
    int itr, top;
    ivec rel_jk;
    real r_ij, r_jk, theta, cos_theta, sin_xhz4, cos_xhz1, sin_theta2;
    real e_hb, exp_hb2, exp_hb3, CEhb1, CEhb2, CEhb3;
    rvec dcos_theta_di, dcos_theta_dj, dcos_theta_dk;
    rvec dvec_jk, force, ext_press;
    hbond_parameters *hbp;
    bond_order_data *bo_ij;
    int nbr_jk;
    reax_list *far_nbr_list, *bond_list, *hbond_list;
    bond_data *pbond_ij;
    hbond_data *phbond_jk;
    storage *workspace;

    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= n )
    {
        return;
    }

    far_nbr_list = &p_far_nbr_list;
    bond_list = &p_bond_list;
    hbond_list = &p_hbond_list;
    workspace = &p_workspace;

    /* discover the Hydrogen bonds between i-j-k triplets.
     * here j is H atom and there has to be some bond between i and j.
     * Hydrogen bond is between j and k.
     * so in this function i->X, j->H, k->Z when we map 
     * variables onto the ones in the handout. */

    /* j must be a hydrogen atom */
    if ( sbp[ my_atoms[j].type ].p_hbond == H_ATOM )
    {
        type_j = my_atoms[j].type;
        start_j = Start_Index( j, bond_list );
        end_j = End_Index( j, bond_list );
        hb_start_j = Start_Index( my_atoms[j].Hindex, hbond_list );
        hb_end_j = End_Index( my_atoms[j].Hindex, hbond_list );
        top = 0;

        /* search bonded atoms i to atom j (hydrogen atom) for potential hydrogen bonding */
        for ( pi = start_j; pi < end_j; ++pi )
        {
            pbond_ij = &bond_list->bond_list[pi];
            i = pbond_ij->nbr;
            bo_ij = &pbond_ij->bo_data;
            type_i = my_atoms[i].type;

            if ( sbp[type_i].p_hbond == H_BONDING_ATOM
                    && bo_ij->BO >= HB_THRESHOLD )
            {
                hblist[top++] = pi;
            }
        }

        /* for each hbond of atom j */
        for ( pk = hb_start_j; pk < hb_end_j; ++pk )
        {
            phbond_jk = &hbond_list->hbond_list[pk];
            k = phbond_jk->nbr;
            type_k = my_atoms[k].type;
            nbr_jk = phbond_jk->ptr;
            r_jk = far_nbr_list->far_nbr_list.d[nbr_jk];

            rvec_Scale( dvec_jk, hbond_list->hbond_list[pk].scl,
                    far_nbr_list->far_nbr_list.dvec[nbr_jk] );

            rvec_MakeZero( phbond_jk->hb_f );

            /* find matching hbond to atoms j and k */
            for ( itr = 0; itr < top; ++itr )
            {
                pi = hblist[itr];
                pbond_ij = &bond_list->bond_list[pi];
                i = pbond_ij->nbr;

                if ( my_atoms[i].orig_id != my_atoms[k].orig_id )
                {
                    bo_ij = &pbond_ij->bo_data;
                    type_i = my_atoms[i].type;
                    r_ij = pbond_ij->d;
                    hbp = &d_hbp[ index_hbp(type_i, type_j, type_k, num_atom_types) ];

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
                    exp_hb2 = EXP( -1.0 * hbp->p_hb2 * bo_ij->BO );
                    exp_hb3 = EXP( -1.0 * hbp->p_hb3 * ( hbp->r0_hb / r_jk
                                + r_jk / hbp->r0_hb - 2.0 ) );

                    e_hb = hbp->p_hb1 * (1.0 - exp_hb2) * exp_hb3 * sin_xhz4;
                    data_e_hb[j] += e_hb;

                    CEhb1 = hbp->p_hb1 * hbp->p_hb2 * exp_hb2 * exp_hb3 * sin_xhz4;
                    CEhb2 = -0.5 * hbp->p_hb1 * (1.0 - exp_hb2) * exp_hb3 * cos_xhz1;
                    CEhb3 = hbp->p_hb3 * e_hb * (hbp->r0_hb / SQR( r_jk )
                            + -1.0 / hbp->r0_hb);

                    /* hydrogen bond forces */
                    /* dbo term */
                    bo_ij->Cdbo += CEhb1;

                    if ( control->virial == 0 )
                    {
                        /* dcos terms */
                        //rvec_ScaledAdd( workspace->f[i], CEhb2, dcos_theta_di ); 
                        //atomic_rvecScaledAdd( workspace->f[i], CEhb2, dcos_theta_di );
                        rvec_ScaledAdd( pbond_ij->hb_f, CEhb2, dcos_theta_di ); 

                        rvec_ScaledAdd( workspace->f[j], CEhb2, dcos_theta_dj );

                        //rvec_ScaledAdd( workspace->f[k], CEhb2, dcos_theta_dk );
                        //atomic_rvecScaledAdd( workspace->f[k], CEhb2, dcos_theta_dk );
                        rvec_ScaledAdd( phbond_jk->hb_f, CEhb2, dcos_theta_dk );

                        /* dr terms */
                        rvec_ScaledAdd( workspace->f[j], -1.0 * CEhb3 / r_jk, dvec_jk ); 

                        //rvec_ScaledAdd( workspace->f[k], CEhb3 / r_jk, dvec_jk );
                        //atomic_rvecScaledAdd( workspace->f[k], CEhb3 / r_jk, dvec_jk );
                        rvec_ScaledAdd( phbond_jk->hb_f, CEhb3 / r_jk, dvec_jk );
                    }
                    else
                    {
                        /* for pressure coupling, terms that are not related to bond order
                         * derivatives are added directly into pressure vector/tensor */
                        /* dcos terms */
                        rvec_Scale( force, CEhb2, dcos_theta_di );
                        //rvec_Add( workspace->f[i], force );
                        rvec_Add( pbond_ij->hb_f, force );
                        rvec_iMultiply( ext_press, pbond_ij->rel_box, force );
                        rvec_ScaledAdd( data_ext_press[j], 1.0, ext_press );

                        rvec_ScaledAdd( workspace->f[j], CEhb2, dcos_theta_dj );

                        ivec_Scale( rel_jk, hbond_list->hbond_list[pk].scl,
                                far_nbr_list->far_nbr_list.rel_box[nbr_jk] );
                        rvec_Scale( force, CEhb2, dcos_theta_dk );
                        //rvec_Add( workspace->f[k], force );
                        rvec_Add( phbond_jk->hb_f, force );
                        rvec_iMultiply( ext_press, rel_jk, force );
                        rvec_ScaledAdd( data_ext_press[j], 1.0, ext_press );

                        /* dr terms */
                        rvec_ScaledAdd( workspace->f[j], -1.0 * CEhb3 / r_jk, dvec_jk ); 

                        rvec_Scale( force, CEhb3 / r_jk, dvec_jk );
                        //rvec_Add( workspace->f[k], force );
                        rvec_Add( phbond_jk->hb_f, force );
                        rvec_iMultiply( ext_press, rel_jk, force );
                        rvec_ScaledAdd( data_ext_press[j], 1.0, ext_press );
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
                    Add_dBO( system, lists, j, pi, CEhb1, workspace->f_hb );

                    /* dcos terms */
                    rvec_ScaledAdd( workspace->f_hb[i], CEhb2, dcos_theta_di );
                    rvec_ScaledAdd( workspace->f_hb[j], CEhb2, dcos_theta_dj );
                    rvec_ScaledAdd( workspace->f_hb[k], CEhb2, dcos_theta_dk );

                    /* dr terms */
                    rvec_ScaledAdd( workspace->f_hb[j], -1.0 * CEhb3 / r_jk, dvec_jk ); 
                    rvec_ScaledAdd( workspace->f_hb[k], CEhb3 / r_jk, dvec_jk );
#endif
                }
            }
        }
    }
}


CUDA_GLOBAL void Cuda_Hydrogen_Bonds_MT( reax_atom *my_atoms, single_body_parameters *sbp, 
        hbond_parameters *d_hbp, global_parameters gp, control_params *control, storage p_workspace,
        reax_list p_far_nbr_list, reax_list p_bond_list, reax_list p_hbond_list, int n, 
        int num_atom_types, real *data_e_hb, rvec *data_ext_press )
{
#if defined( __SM_35__)
    real sh_hb;
    real sh_cdbo;
    rvec sh_atomf;
    rvec sh_hf;
#else
    extern __shared__ real _s[];
    real *sh_hb = _s;
    real *sh_cdbo = &_s[blockDim.x];
    rvec *sh_atomf = (rvec *)(&sh_cdbo[blockDim.x]);
    rvec *sh_hf = (rvec *)(&sh_atomf[blockDim.x]);
#endif
    int __THREADS_PER_ATOM__, thread_id, group_id, lane_id; 
    int i, j, k, pi, pk;
    int type_i, type_j, type_k;
    int start_j, end_j, hb_start_j, hb_end_j;
    //TODO: re-write and remove
    int hblist[MAX_BONDS];
    int itr, top;
    int loopcount, count;
    ivec rel_jk;
    real r_ij, r_jk, theta, cos_theta, sin_xhz4, cos_xhz1, sin_theta2;
    real e_hb, exp_hb2, exp_hb3, CEhb1, CEhb2, CEhb3;
    rvec dcos_theta_di, dcos_theta_dj, dcos_theta_dk;
    rvec dvec_jk, force, ext_press;
    hbond_parameters *hbp;
    bond_order_data *bo_ij;
    int nbr_jk;
    reax_list *far_nbr_list, *bond_list, *hbond_list;
    bond_data *pbond_ij;
    hbond_data *phbond_jk;
    storage *workspace;

    __THREADS_PER_ATOM__ = HB_KER_THREADS_PER_ATOM;
    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    group_id = thread_id / __THREADS_PER_ATOM__;
    lane_id = thread_id & (__THREADS_PER_ATOM__ - 1); 

    if ( group_id >= n )
    {
        return;
    }

    workspace = &p_workspace;
    far_nbr_list = &p_far_nbr_list;
    bond_list = &p_bond_list;
    hbond_list = &p_hbond_list;
    j = group_id;

    /* loops below discover the Hydrogen bonds between i-j-k triplets.
       here j is H atom and there has to be some bond between i and j.
       Hydrogen bond is between j and k.
       so in this function i->X, j->H, k->Z when we map 
       variables onto the ones in the handout.*/
    //for( j = 0; j < system->n; ++j )

#if defined( __SM_35__)
    sh_hb = 0;
    rvec_MakeZero( sh_atomf );
#else
    sh_hb[threadIdx.x] = 0;
    rvec_MakeZero( sh_atomf[threadIdx.x] );
#endif

    /* j has to be of type H */
    if ( sbp[ my_atoms[j].type ].p_hbond == H_ATOM )
    {
        type_j = my_atoms[j].type;
        start_j = Start_Index( j, bond_list );
        end_j = End_Index( j, bond_list );
        hb_start_j = Start_Index( my_atoms[j].Hindex, hbond_list );
        hb_end_j = End_Index( my_atoms[j].Hindex, hbond_list );
        top = 0;

        /* search bonded atoms i to atom j (hydrogen atom) for potential hydrogen bonding */
        for ( pi = start_j; pi < end_j; ++pi ) 
        {
            pbond_ij = &bond_list->bond_list[pi];
            i = pbond_ij->nbr;
            bo_ij = &pbond_ij->bo_data;
            type_i = my_atoms[i].type;

            if ( sbp[type_i].p_hbond == H_BONDING_ATOM
                    && bo_ij->BO >= HB_THRESHOLD )
            {
                hblist[top++] = pi;
            }
        }

        /* find matching hbond to atoms j and k */
        for ( itr = 0; itr < top; ++itr )
        {
            pi = hblist[itr];
            pbond_ij = &bond_list->bond_list[pi];
            i = pbond_ij->nbr;

#if defined( __SM_35__)
            rvec_MakeZero( sh_hf );
            sh_cdbo = 0;
#else
            rvec_MakeZero( sh_hf[threadIdx.x] );
            sh_cdbo[threadIdx.x] = 0;
#endif

            //for( pk = hb_start_j; pk < hb_end_j; ++pk ) {
            loopcount = (hb_end_j - hb_start_j) / HB_KER_THREADS_PER_ATOM + 
                (((hb_end_j - hb_start_j) % HB_KER_THREADS_PER_ATOM == 0) ? 0 : 1);

            count = 0;
            pk = hb_start_j + lane_id;
            while ( count < loopcount )
            {
                /* only allow threads with an actual hbond */
                if ( pk < hb_end_j )
                {
                    phbond_jk = &hbond_list->hbond_list[pk];

                    /* set k's varibles */
                    k = hbond_list->hbond_list[pk].nbr;
                    type_k = my_atoms[k].type;
                    nbr_jk = hbond_list->hbond_list[pk].ptr;
                    r_jk = far_nbr_list->far_nbr_list.d[nbr_jk];

                    rvec_Scale( dvec_jk, phbond_jk->scl,
                            far_nbr_list->far_nbr_list.dvec[nbr_jk] );
                }
                else
                {
                    k = -1;
                }

                if ( my_atoms[i].orig_id != my_atoms[k].orig_id && k != -1 )
                {
                    bo_ij = &pbond_ij->bo_data;
                    type_i = my_atoms[i].type;
                    r_ij = pbond_ij->d;
                    hbp = &d_hbp[ index_hbp(type_i,type_j,type_k,num_atom_types) ];

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
                    exp_hb2 = EXP( -1.0 * hbp->p_hb2 * bo_ij->BO );
                    exp_hb3 = EXP( -1.0 * hbp->p_hb3 * ( hbp->r0_hb / r_jk
                                + r_jk / hbp->r0_hb - 2.0 ) );

                    e_hb = hbp->p_hb1 * (1.0 - exp_hb2) * exp_hb3 * sin_xhz4;
#if defined( __SM_35__)
                    sh_hb += e_hb;
#else
                    sh_hb[threadIdx.x] += e_hb;
#endif

                    CEhb1 = hbp->p_hb1 * hbp->p_hb2 * exp_hb2 * exp_hb3 * sin_xhz4;
                    CEhb2 = -0.5 * hbp->p_hb1 * (1.0 - exp_hb2) * exp_hb3 * cos_xhz1;
                    CEhb3 = hbp->p_hb3 * e_hb * (hbp->r0_hb / SQR( r_jk )
                            + -1.0 / hbp->r0_hb);

                    /* hydrogen bond forces */
                    /* dbo term */
#if defined( __SM_35__)
                    sh_cdbo += CEhb1;
#else
                    sh_cdbo[threadIdx.x] += CEhb1;
#endif

                    if ( control->virial == 0 )
                    {
                        /* dcos terms */
#if defined( __SM_35__)
                        rvec_ScaledAdd( sh_hf, CEhb2, dcos_theta_di ); 
#else
                        rvec_ScaledAdd( sh_hf[threadIdx.x], CEhb2, dcos_theta_di ); 
#endif

#if defined( __SM_35__)
                        rvec_ScaledAdd( sh_atomf, CEhb2, dcos_theta_dj );
#else
                        rvec_ScaledAdd( sh_atomf[threadIdx.x], CEhb2, dcos_theta_dj );
#endif

                        rvec_ScaledAdd( phbond_jk->hb_f, CEhb2, dcos_theta_dk );

                        /* dr terms */
#if defined( __SM_35__)
                        rvec_ScaledAdd( sh_atomf, -1.0 * CEhb3 / r_jk, dvec_jk ); 
#else
                        rvec_ScaledAdd( sh_atomf[threadIdx.x], -1.0 * CEhb3 / r_jk, dvec_jk ); 
#endif

                        rvec_ScaledAdd( phbond_jk->hb_f, CEhb3 / r_jk, dvec_jk );
                    }
                    else
                    {
                        /* for pressure coupling, terms that are not related to bond order
                         * derivatives are added directly into pressure vector/tensor */
                        /* dcos terms */
                        rvec_Scale( force, CEhb2, dcos_theta_di );
                        rvec_Add( pbond_ij->hb_f, force );
                        rvec_iMultiply( ext_press, pbond_ij->rel_box, force );
                        rvec_ScaledAdd( data_ext_press [j], 1.0, ext_press );

                        rvec_ScaledAdd( workspace->f[j], CEhb2, dcos_theta_dj );

                        ivec_Scale( rel_jk, hbond_list->hbond_list[pk].scl,
                                far_nbr_list->far_nbr_list.rel_box[nbr_jk] );
                        rvec_Scale( force, CEhb2, dcos_theta_dk );
                        rvec_Add( phbond_jk->hb_f, force );
                        rvec_iMultiply( ext_press, rel_jk, force );
                        rvec_ScaledAdd( data_ext_press[j], 1.0, ext_press );

                        /* dr terms */
                        rvec_ScaledAdd( workspace->f[j], -1.0 * CEhb3 / r_jk, dvec_jk ); 

                        rvec_Scale( force, CEhb3 / r_jk, dvec_jk );
                        rvec_Add( phbond_jk->hb_f, force );
                        rvec_iMultiply( ext_press, rel_jk, force );
                        rvec_ScaledAdd( data_ext_press[j], 1.0, ext_press );
                    }

                } //orid id end

                pk += __THREADS_PER_ATOM__;
                count++;

            } //for itr loop end

            /* reduction */
#if defined( __SM_35__)
            for ( int s = __THREADS_PER_ATOM__ >> 1; s >= 1; s/=2 )
            {
                sh_cdbo += shfl( sh_cdbo, s);
                sh_hf[0] += shfl( sh_hf[0], s);
                sh_hf[1] += shfl( sh_hf[1], s);
                sh_hf[2] += shfl( sh_hf[2], s);
            }
            //end of the shuffle
            if ( lane_id == 0 )
            {
                bo_ij->Cdbo += sh_cdbo ;
                rvec_Add( pbond_ij->hb_f, sh_hf );
            }
#else
            if ( lane_id < 16 )
            {
                sh_cdbo[threadIdx.x] += sh_cdbo[threadIdx.x + 16];
                rvec_Add( sh_hf [threadIdx.x], sh_hf[threadIdx.x + 16] );
            }
            if ( lane_id < 8 )
            {
                sh_cdbo[threadIdx.x] += sh_cdbo[threadIdx.x + 8];
                rvec_Add( sh_hf [threadIdx.x], sh_hf[threadIdx.x + 8] );
            }
            if ( lane_id < 4 )
            {
                sh_cdbo[threadIdx.x] += sh_cdbo[threadIdx.x + 4];
                rvec_Add( sh_hf [threadIdx.x], sh_hf[threadIdx.x + 4] );
            }
            if ( lane_id < 2 )
            {
                sh_cdbo[threadIdx.x] += sh_cdbo[threadIdx.x + 2];
                rvec_Add( sh_hf [threadIdx.x], sh_hf[threadIdx.x + 2] );
            }
            if ( lane_id < 1 )
            {
                sh_cdbo[threadIdx.x] += sh_cdbo[threadIdx.x + 1];
                rvec_Add( sh_hf [threadIdx.x], sh_hf[threadIdx.x + 1] );

                bo_ij->Cdbo += sh_cdbo[threadIdx.x];
                rvec_Add( pbond_ij->hb_f, sh_hf[threadIdx.x] );
            }
#endif
        } // for loop hbonds end
    } //if Hbond check end

#if defined( __SM_35__)
    for ( int s = __THREADS_PER_ATOM__ >> 1; s >= 1; s/=2 )
    {
        sh_hb += shfl( sh_hb, s);
        sh_atomf[0] += shfl( sh_atomf[0], s);
        sh_atomf[1] += shfl( sh_atomf[1], s);
        sh_atomf[2] += shfl( sh_atomf[2], s);
    }
    if ( lane_id == 0 )
    {
        data_e_hb[j] += sh_hb;
        rvec_Add( workspace->f[j], sh_atomf );
    }
#else
    if ( lane_id < 16 )
    {
        sh_hb[threadIdx.x] += sh_hb[threadIdx.x + 16];
        rvec_Add ( sh_atomf [threadIdx.x], sh_atomf[threadIdx.x + 16] );
    }
    if ( lane_id < 8 )
    {
        sh_hb[threadIdx.x] += sh_hb[threadIdx.x + 8];
        rvec_Add ( sh_atomf [threadIdx.x], sh_atomf[threadIdx.x + 8] );
    }
    if ( lane_id < 4 )
    {
        sh_hb[threadIdx.x] += sh_hb[threadIdx.x + 4];
        rvec_Add ( sh_atomf [threadIdx.x], sh_atomf[threadIdx.x + 4] );
    }
    if ( lane_id < 2 )
    {
        sh_hb[threadIdx.x] += sh_hb[threadIdx.x + 2];
        rvec_Add ( sh_atomf [threadIdx.x], sh_atomf[threadIdx.x + 2] );
    }
    if ( lane_id < 1 )
    {
        sh_hb[threadIdx.x] += sh_hb[threadIdx.x + 1];
        rvec_Add ( sh_atomf [threadIdx.x], sh_atomf[threadIdx.x + 1] );

        data_e_hb[j] += sh_hb[threadIdx.x];
        rvec_Add( workspace->f[j], sh_atomf[threadIdx.x] );
    }
#endif
}


CUDA_GLOBAL void Cuda_Hydrogen_Bonds_PostProcess( reax_atom *atoms,
        storage p_workspace, reax_list p_bond_list, int N )
{
    int i, pj;
    storage *workspace;
    bond_data *pbond;
    bond_data *sym_index_bond;
    reax_list *bond_list;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    workspace = &p_workspace;
    bond_list = &p_bond_list;

    for ( pj = Start_Index(i, bond_list); pj < End_Index(i, bond_list); ++pj )
    {
        pbond = &bond_list->bond_list[pj];
        sym_index_bond = &bond_list->bond_list[pbond->sym_index];

        //rvec_Add( atoms[i].f, sym_index_bond->hb_f );
        rvec_Add( workspace->f[i], sym_index_bond->hb_f );
    }
}


CUDA_GLOBAL void Cuda_Hydrogen_Bonds_HNbrs( reax_atom *atoms,
        storage p_workspace, reax_list p_hbond_list )
{
#if defined(__SM_35__)
    rvec __f;
#else
    extern __shared__ rvec __f[];
#endif
    int i, pj;
    int start, end;
    storage *workspace;
    hbond_data *nbr_pj, *sym_index_nbr;
    reax_list *hbond_list;

    i = blockIdx.x;
    workspace = &p_workspace;
    hbond_list = &p_hbond_list;

    start = Start_Index( atoms[i].Hindex, hbond_list );
    end = End_Index( atoms[i].Hindex, hbond_list );
    pj = start + threadIdx.x;
#if defined(__SM_35__)
    rvec_MakeZero( __f );
#else
    rvec_MakeZero( __f[threadIdx.x] );
#endif

    while ( pj < end )
    {
        nbr_pj = &hbond_list->hbond_list[pj];

        sym_index_nbr = &hbond_list->hbond_list[ nbr_pj->sym_index ];

#if defined(__SM_35__)
        rvec_Add( __f, sym_index_nbr->hb_f );
#else
        rvec_Add( __f[threadIdx.x], sym_index_nbr->hb_f );
#endif

        pj += blockDim.x;
    }

    __syncthreads( );

#if defined(__SM_35__)
    for ( int s = 16; s >= 1; s /= 2 )
    {
        __f[0] += shfl( __f[0], s );
        __f[1] += shfl( __f[1], s );
        __f[2] += shfl( __f[2], s );
    }

    if ( threadIdx.x == 0 )
    {
        rvec_Add( workspace->f[i], __f );
    }
#else
    if ( threadIdx.x < 16 )
    {
        rvec_Add( __f[threadIdx.x], __f[threadIdx.x + 16] );
    }
    __syncthreads( );

    if ( threadIdx.x < 8 )
    {
        rvec_Add( __f[threadIdx.x], __f[threadIdx.x + 8] );
    }
    __syncthreads( );

    if ( threadIdx.x < 4 )
    {
        rvec_Add( __f[threadIdx.x], __f[threadIdx.x + 4] );
    }
    __syncthreads( );

    if ( threadIdx.x < 2 )
    {
        rvec_Add( __f[threadIdx.x], __f[threadIdx.x + 2] );
    }
    __syncthreads( );

    if ( threadIdx.x < 1 )
    {
        rvec_Add( __f[threadIdx.x], __f[threadIdx.x + 1] );
    }
    __syncthreads( );

    if ( threadIdx.x == 0 )
    {
        //rvec_Add( atoms[i].f, __f[0] );
        rvec_Add( workspace->f[i], __f[0] );
    }
#endif
}


CUDA_GLOBAL void Cuda_Hydrogen_Bonds_HNbrs_BL( reax_atom *atoms,
        storage p_workspace, reax_list p_hbond_list, int N )
{
#if defined(__SM_35__)
    rvec __f;
    int s;
#else
    extern __shared__ rvec __f[];
#endif
    int i, pj;
    int start, end;
    storage *workspace;
    hbond_data *nbr_pj, *sym_index_nbr;
    reax_list *hbond_list;
    int __THREADS_PER_ATOM__;
    int thread_id;
    int group_id;
    int lane_id; 

    __THREADS_PER_ATOM__ = HB_POST_PROC_KER_THREADS_PER_ATOM;
    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    group_id = thread_id / __THREADS_PER_ATOM__;
    lane_id = thread_id & (__THREADS_PER_ATOM__ - 1);

    if ( group_id >= N )
    {
        return;
    }

    workspace = &p_workspace;
    hbond_list = &p_hbond_list;
    i = group_id;
    start = Start_Index( atoms[i].Hindex, hbond_list );
    end = End_Index( atoms[i].Hindex, hbond_list );
    pj = start + lane_id;
#if defined(__SM_35__)
    rvec_MakeZero( __f );
#else
    rvec_MakeZero( __f[threadIdx.x] );
#endif

    while ( pj < end )
    {
        nbr_pj = &hbond_list->hbond_list[pj];

        sym_index_nbr = &hbond_list->hbond_list[ nbr_pj->sym_index ];
#if defined(__SM_35__)
        rvec_Add( __f, sym_index_nbr->hb_f );
#else
        rvec_Add( __f[threadIdx.x], sym_index_nbr->hb_f );
#endif

        pj += __THREADS_PER_ATOM__;
    }

    __syncthreads( );

#if defined(__SM_35__)
    for ( s = __THREADS_PER_ATOM__ >> 1; s >= 1; s /= 2 )
    {
        __f[0] += shfl( __f[0], s );
        __f[1] += shfl( __f[1], s );
        __f[2] += shfl( __f[2], s );
    }

    if ( lane_id == 0 )
    {
        rvec_Add( workspace->f[i], __f );
    }
#else
    if ( lane_id < 16 )
    {
        rvec_Add( __f[threadIdx.x], __f[threadIdx.x + 16] );
    }
    __syncthreads( );

    if ( lane_id < 8 )
    {
        rvec_Add( __f[threadIdx.x], __f[threadIdx.x + 8] );
    }
    __syncthreads( );

    if ( lane_id < 4 )
    {
        rvec_Add( __f[threadIdx.x], __f[threadIdx.x + 4] );
    }
    __syncthreads( );

    if ( lane_id < 2 )
    {
        rvec_Add( __f[threadIdx.x], __f[threadIdx.x + 2] );
    }
    __syncthreads( );

    if ( lane_id < 1 )
    {
        rvec_Add( __f[threadIdx.x], __f[threadIdx.x + 1] );
    }
    __syncthreads( );

    if ( lane_id == 0 )
    {
        rvec_Add( workspace->f[i], __f[threadIdx.x] );
    }
#endif
}
