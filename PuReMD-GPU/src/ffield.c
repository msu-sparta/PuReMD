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

#include <ctype.h>

#include "ffield.h"
#include "tool_box.h"


char Read_Force_Field( FILE* fp, reax_interaction* reax )
{
    char *s;
    char **tmp;
    char *tor_flag;
    int c, i, j, k, l, m, n, o, p, cnt;
    real val;
    int __N;
    int index1, index2;

    s = (char*) malloc(sizeof(char) * MAX_LINE);
    tmp = (char**) malloc(sizeof(char*)*MAX_TOKENS);
    for (i = 0; i < MAX_TOKENS; i++)
    {
        tmp[i] = (char*) malloc(sizeof(char) * MAX_TOKEN_LEN);
    }


    /* reading first header comment */
    fgets( s, MAX_LINE, fp );

    /* line 2 is number of global parameters */
    fgets( s, MAX_LINE, fp );
    c = Tokenize( s, &tmp );

    /* reading the number of global parameters */
    n = atoi(tmp[0]);
    if (n < 1)
    {
        fprintf( stderr, "WARNING: number of globals in ffield file is 0!\n" );
        exit( INVALID_INPUT );
    }

    reax->gp.n_global = n;
    reax->gp.l = (real*) malloc(sizeof(real) * n);

    /* see mytypes.h for mapping between l[i] and the lambdas used in ff */
    for (i = 0; i < n; i++)
    {
        fgets(s, MAX_LINE, fp);
        c = Tokenize(s, &tmp);

        val = (real) atof(tmp[0]);

        reax->gp.l[i] = val;
    }

    /* next line is number of atom types and some comments */
    fgets( s, MAX_LINE, fp );
    c = Tokenize( s, &tmp );
    reax->num_atom_types = atoi(tmp[0]);
    __N = reax->num_atom_types;

    /* 3 lines of comments */
    fgets(s, MAX_LINE, fp);
    fgets(s, MAX_LINE, fp);
    fgets(s, MAX_LINE, fp);

    /* Allocating structures in reax_interaction */
    reax->sbp = (single_body_parameters*)
                calloc( reax->num_atom_types, sizeof(single_body_parameters) );

    reax->tbp = (two_body_parameters*)
                calloc( pow (reax->num_atom_types, 2), sizeof(two_body_parameters) );

    reax->thbp = (three_body_header*)
                 calloc( pow (reax->num_atom_types, 3), sizeof(three_body_header) );
    reax->hbp = (hbond_parameters*)
                calloc( pow (reax->num_atom_types, 3), sizeof(hbond_parameters) );

    reax->fbp = (four_body_header*)
                calloc( pow (reax->num_atom_types, 4), sizeof(four_body_header) );

    tor_flag  = (char*)
                calloc( pow (reax->num_atom_types, 4), sizeof(char) );

    // vdWaals type: 1: Shielded Morse, no inner-wall
    //               2: inner wall, no shielding
    //               3: inner wall+shielding
    reax->gp.vdw_type = 0;

    /* reading single atom parameters */
    /* there are 4 lines of each single atom parameters in ff files. these
       parameters later determine some of the pair and triplet parameters using
       combination rules. */
    for ( i = 0; i < reax->num_atom_types; i++ )
    {
        /* line one */
        fgets( s, MAX_LINE, fp );
        c = Tokenize( s, &tmp );

        for ( j = 0; j < strlen( tmp[0] ); ++j )
        {
            reax->sbp[i].name[j] = toupper( tmp[0][j] );
        }

        val = atof(tmp[1]);
        reax->sbp[i].r_s        = val;
        val = atof(tmp[2]);
        reax->sbp[i].valency    = val;
        val = atof(tmp[3]);
        reax->sbp[i].mass       = val;
        val = atof(tmp[4]);
        reax->sbp[i].r_vdw      = val;
        val = atof(tmp[5]);
        reax->sbp[i].epsilon    = val;
        val = atof(tmp[6]);
        reax->sbp[i].gamma      = val;
        val = atof(tmp[7]);
        reax->sbp[i].r_pi       = val;
        val = atof(tmp[8]);
        reax->sbp[i].valency_e  = val;
        reax->sbp[i].nlp_opt = 0.5 * (reax->sbp[i].valency_e - reax->sbp[i].valency);

        /* line two */
        fgets( s, MAX_LINE, fp );
        c = Tokenize( s, &tmp );

        val = atof(tmp[0]);
        reax->sbp[i].alpha      = val;
        val = atof(tmp[1]);
        reax->sbp[i].gamma_w    = val;
        val = atof(tmp[2]);
        reax->sbp[i].valency_boc = val;
        val = atof(tmp[3]);
        reax->sbp[i].p_ovun5    = val;
        val = atof(tmp[4]);
        val = atof(tmp[5]);
        reax->sbp[i].chi        = val;
        val = atof(tmp[6]);
        reax->sbp[i].eta        = 2.0 * val;
        /* this is the parameter that is used to determine
           which type of atoms participate in h-bonds.
           1 is for H - 2 for O, N, S - 0 for all others.*/
        val = atof(tmp[7]);
        reax->sbp[i].p_hbond = (int)(val + 0.1);
        //0.1 is to avoid from truncating down!

        /* line 3 */
        fgets( s, MAX_LINE, fp );
        c = Tokenize( s, &tmp );

        val = atof(tmp[0]);
        reax->sbp[i].r_pi_pi    = val;
        val = atof(tmp[1]);
        reax->sbp[i].p_lp2      = val;
        val = atof(tmp[2]);
        val = atof(tmp[3]);
        reax->sbp[i].b_o_131    = val;
        val = atof(tmp[4]);
        reax->sbp[i].b_o_132    = val;
        val = atof(tmp[5]);
        reax->sbp[i].b_o_133    = val;
        val = atof(tmp[6]);
        val = atof(tmp[7]);

        /* line 4  */
        fgets( s, MAX_LINE, fp );
        c = Tokenize( s, &tmp );

        val = atof(tmp[0]);
        reax->sbp[i].p_ovun2    = val;
        val = atof(tmp[1]);
        reax->sbp[i].p_val3     = val;
        val = atof(tmp[2]);
        val = atof(tmp[3]);
        reax->sbp[i].valency_val = val;
        val = atof(tmp[4]);
        reax->sbp[i].p_val5     = val;
        val = atof(tmp[5]);
        reax->sbp[i].rcore2     = val;
        val = atof(tmp[6]);
        reax->sbp[i].ecore2     = val;
        val = atof(tmp[7]);
        reax->sbp[i].acore2     = val;

        if ( reax->sbp[i].rcore2 > 0.01 && reax->sbp[i].acore2 > 0.01 ) // Inner-wall
        {
            if ( reax->sbp[i].gamma_w > 0.5 ) // Shielding vdWaals
            {
                if ( reax->gp.vdw_type != 0 && reax->gp.vdw_type != 3 )
                {
                    fprintf( stderr, "Warning: inconsistent vdWaals-parameters\n" \
                             "Force field parameters for element %s\n"        \
                             "indicate inner wall+shielding, but earlier\n"   \
                             "atoms indicate different vdWaals-method.\n"     \
                             "This may cause division-by-zero errors.\n"      \
                             "Keeping vdWaals-setting for earlier atoms.\n",
                             reax->sbp[i].name );
                }
                else
                {
                    reax->gp.vdw_type = 3;

#if defined(DEBUG)
                    fprintf( stderr, "vdWaals type for element %s: Shielding+inner-wall",
                             reax->sbp[i].name );
#endif
                }
            }
            else    // No shielding vdWaals parameters present
            {
                if ( reax->gp.vdw_type != 0 && reax->gp.vdw_type != 2 )
                {
                    fprintf( stderr, "Warning: inconsistent vdWaals-parameters\n" \
                             "Force field parameters for element %s\n"        \
                             "indicate inner wall without shielding, but earlier\n" \
                             "atoms indicate different vdWaals-method.\n"     \
                             "This may cause division-by-zero errors.\n"      \
                             "Keeping vdWaals-setting for earlier atoms.\n",
                             reax->sbp[i].name );
                }
                else
                {
                    reax->gp.vdw_type = 2;

#if defined(DEBUG)
                    fprintf( stderr, "vdWaals type for element%s: No Shielding,inner-wall",
                             reax->sbp[i].name );
#endif
                }
            }
        }
        else  // No Inner wall parameters present
        {
            if ( reax->sbp[i].gamma_w > 0.5 ) // Shielding vdWaals
            {
                if ( reax->gp.vdw_type != 0 && reax->gp.vdw_type != 1 )
                    fprintf( stderr, "Warning: inconsistent vdWaals-parameters\n" \
                             "Force field parameters for element %s\n"        \
                             "indicate  shielding without inner wall, but earlier\n" \
                             "atoms indicate different vdWaals-method.\n"     \
                             "This may cause division-by-zero errors.\n"      \
                             "Keeping vdWaals-setting for earlier atoms.\n",
                             reax->sbp[i].name );
                else
                {
                    reax->gp.vdw_type = 1;
#if defined(DEBUG)
                    fprintf( stderr, "vdWaals type for element%s: Shielding,no inner-wall",
                             reax->sbp[i].name );
#endif
                }
            }
            else
            {
                fprintf( stderr, "Error: inconsistent vdWaals-parameters\n"\
                         "No shielding or inner-wall set for element %s\n",
                         reax->sbp[i].name );
                exit( INVALID_INPUT );
            }
        }
    }

    /* next line is number of two body combination and some comments */
    fgets(s, MAX_LINE, fp);
    c = Tokenize(s, &tmp);
    l = atoi(tmp[0]);

    /* a line of comments */
    fgets(s, MAX_LINE, fp);

    for (i = 0; i < l; i++)
    {
        /* line 1 */
        fgets(s, MAX_LINE, fp);
        c = Tokenize(s, &tmp);

        j = atoi(tmp[0]) - 1;
        k = atoi(tmp[1]) - 1;

        index1 = j * __N + k;
        index2 = k * __N + j;

        if (j < reax->num_atom_types && k < reax->num_atom_types)
        {

            val = atof(tmp[2]);
            reax->tbp[index1].De_s      = val;
            reax->tbp[index2].De_s      = val;
            val = atof(tmp[3]);
            reax->tbp[index1].De_p      = val;
            reax->tbp[index2].De_p      = val;
            val = atof(tmp[4]);
            reax->tbp[index1].De_pp     = val;
            reax->tbp[index2].De_pp     = val;
            val = atof(tmp[5]);
            reax->tbp[index1].p_be1     = val;
            reax->tbp[index2].p_be1     = val;
            val = atof(tmp[6]);
            reax->tbp[index1].p_bo5     = val;
            reax->tbp[index2].p_bo5     = val;
            val = atof(tmp[7]);
            reax->tbp[index1].v13cor    = val;
            reax->tbp[index2].v13cor    = val;

            val = atof(tmp[8]);
            reax->tbp[index1].p_bo6     = val;
            reax->tbp[index2].p_bo6     = val;
            val = atof(tmp[9]);
            reax->tbp[index1].p_ovun1 = val;
            reax->tbp[index2].p_ovun1 = val;

            /* line 2 */
            fgets(s, MAX_LINE, fp);
            c = Tokenize(s, &tmp);

            val = atof(tmp[0]);
            reax->tbp[index1].p_be2     = val;
            reax->tbp[index2].p_be2     = val;
            val = atof(tmp[1]);
            reax->tbp[index1].p_bo3     = val;
            reax->tbp[index2].p_bo3     = val;
            val = atof(tmp[2]);
            reax->tbp[index1].p_bo4     = val;
            reax->tbp[index2].p_bo4     = val;
            val = atof(tmp[3]);

            val = atof(tmp[4]);
            reax->tbp[index1].p_bo1     = val;
            reax->tbp[index2].p_bo1     = val;
            val = atof(tmp[5]);
            reax->tbp[index1].p_bo2     = val;
            reax->tbp[index2].p_bo2     = val;
            val = atof(tmp[6]);
            reax->tbp[index1].ovc       = val;
            reax->tbp[index2].ovc       = val;

            val = atof(tmp[7]);
        }
    }

    /* calculating combination rules and filling up remaining fields. */

    for (i = 0; i < reax->num_atom_types; i++)
    {
        for (j = i; j < reax->num_atom_types; j++)
        {
            index1 = i * __N + j;
            index2 = j * __N + i;

            reax->tbp[index1].r_s = 0.5 *
                                    (reax->sbp[i].r_s + reax->sbp[j].r_s);
            reax->tbp[index2].r_s = 0.5 *
                                    (reax->sbp[j].r_s + reax->sbp[i].r_s);

            reax->tbp[index1].r_p = 0.5 *
                                    (reax->sbp[i].r_pi + reax->sbp[j].r_pi);
            reax->tbp[index2].r_p = 0.5 *
                                    (reax->sbp[j].r_pi + reax->sbp[i].r_pi);

            reax->tbp[index1].r_pp = 0.5 *
                                     (reax->sbp[i].r_pi_pi + reax->sbp[j].r_pi_pi);
            reax->tbp[index2].r_pp = 0.5 *
                                     (reax->sbp[j].r_pi_pi + reax->sbp[i].r_pi_pi);

            reax->tbp[index1].p_boc3 =
                sqrt(reax->sbp[i].b_o_132 *
                     reax->sbp[j].b_o_132);
            reax->tbp[index2].p_boc3 =
                sqrt(reax->sbp[j].b_o_132 *
                     reax->sbp[i].b_o_132);

            reax->tbp[index1].p_boc4 =
                sqrt(reax->sbp[i].b_o_131 *
                     reax->sbp[j].b_o_131);
            reax->tbp[index2].p_boc4 =
                sqrt(reax->sbp[j].b_o_131 *
                     reax->sbp[i].b_o_131);

            reax->tbp[index1].p_boc5 =
                sqrt(reax->sbp[i].b_o_133 *
                     reax->sbp[j].b_o_133);
            reax->tbp[index2].p_boc5 =
                sqrt(reax->sbp[j].b_o_133 *
                     reax->sbp[i].b_o_133);

            reax->tbp[index1].D =
                sqrt(reax->sbp[i].epsilon *
                     reax->sbp[j].epsilon);
            reax->tbp[index2].D =
                sqrt(reax->sbp[j].epsilon *
                     reax->sbp[i].epsilon);

            reax->tbp[index1].alpha =
                sqrt(reax->sbp[i].alpha *
                     reax->sbp[j].alpha);
            reax->tbp[index2].alpha =
                sqrt(reax->sbp[j].alpha *
                     reax->sbp[i].alpha);

            reax->tbp[index1].r_vdW =
                2.0 * sqrt(reax->sbp[i].r_vdw * reax->sbp[j].r_vdw);
            reax->tbp[index2].r_vdW =
                2.0 * sqrt(reax->sbp[j].r_vdw * reax->sbp[i].r_vdw);

            reax->tbp[index1].gamma_w =
                sqrt(reax->sbp[i].gamma_w *
                     reax->sbp[j].gamma_w);
            reax->tbp[index2].gamma_w =
                sqrt(reax->sbp[j].gamma_w *
                     reax->sbp[i].gamma_w);

            reax->tbp[index1].gamma =
                POW(reax->sbp[i].gamma *
                    reax->sbp[j].gamma, -1.5);
            reax->tbp[index2].gamma =
                POW(reax->sbp[j].gamma *
                    reax->sbp[i].gamma, -1.5);
        }
    }

    /* next line is number of 2-body offdiagonal combinations and some comments */
    /* these are two body offdiagonal terms that are different from the
       combination rules defined above */
    fgets(s, MAX_LINE, fp);
    c = Tokenize(s, &tmp);
    l = atoi(tmp[0]);

    for (i = 0; i < l; i++)
    {
        fgets(s, MAX_LINE, fp);
        c = Tokenize(s, &tmp);

        j = atoi(tmp[0]) - 1;
        k = atoi(tmp[1]) - 1;
        index1 = j * __N + k;
        index2 = k * __N + j;

        if (j < reax->num_atom_types && k < reax->num_atom_types)
        {
            val = atof(tmp[2]);
            if (val > 0.0)
            {
                reax->tbp[index1].D = val;
                reax->tbp[index2].D = val;
            }

            val = atof(tmp[3]);
            if (val > 0.0)
            {
                reax->tbp[index1].r_vdW = 2 * val;
                reax->tbp[index2].r_vdW = 2 * val;
            }

            val = atof(tmp[4]);
            if (val > 0.0)
            {
                reax->tbp[index1].alpha = val;
                reax->tbp[index2].alpha = val;
            }

            val = atof(tmp[5]);
            if (val > 0.0)
            {
                reax->tbp[index1].r_s = val;
                reax->tbp[index2].r_s = val;
            }

            val = atof(tmp[6]);
            if (val > 0.0)
            {
                reax->tbp[index1].r_p = val;
                reax->tbp[index2].r_p = val;
            }

            val = atof(tmp[7]);
            if (val > 0.0)
            {
                reax->tbp[index1].r_pp = val;
                reax->tbp[index2].r_pp = val;
            }
        }
    }

    /* 3-body parameters -
       supports multi-well potentials (upto MAX_3BODY_PARAM in mytypes.h) */
    /* clear entries first */
    for ( i = 0; i < reax->num_atom_types; ++i )
        for ( j = 0; j < reax->num_atom_types; ++j )
            for ( k = 0; k < reax->num_atom_types; ++k )
                reax->thbp[i * __N * __N + j * __N + k].cnt = 0;

    /* next line is number of 3-body params and some comments */
    fgets( s, MAX_LINE, fp );
    c = Tokenize( s, &tmp );
    l = atoi( tmp[0] );

    for ( i = 0; i < l; i++ )
    {
        fgets(s, MAX_LINE, fp);
        c = Tokenize(s, &tmp);

        j = atoi(tmp[0]) - 1;
        k = atoi(tmp[1]) - 1;
        m = atoi(tmp[2]) - 1;

        index1 = j * __N * __N + k * __N + m;
        index2 = m * __N * __N + k * __N + j;

        if (j < reax->num_atom_types &&
                k < reax->num_atom_types &&
                m < reax->num_atom_types)
        {
            cnt = reax->thbp[index1].cnt;
            reax->thbp[index1].cnt++;
            reax->thbp[index2].cnt++;

            val = atof(tmp[3]);
            reax->thbp[index1].prm[cnt].theta_00 = val;
            reax->thbp[index2].prm[cnt].theta_00 = val;

            val = atof(tmp[4]);
            reax->thbp[index1].prm[cnt].p_val1 = val;
            reax->thbp[index2].prm[cnt].p_val1 = val;

            val = atof(tmp[5]);
            reax->thbp[index1].prm[cnt].p_val2 = val;
            reax->thbp[index2].prm[cnt].p_val2 = val;

            val = atof(tmp[6]);
            reax->thbp[index1].prm[cnt].p_coa1 = val;
            reax->thbp[index2].prm[cnt].p_coa1 = val;

            val = atof(tmp[7]);
            reax->thbp[index1].prm[cnt].p_val7 = val;
            reax->thbp[index2].prm[cnt].p_val7 = val;

            val = atof(tmp[8]);
            reax->thbp[index1].prm[cnt].p_pen1 = val;
            reax->thbp[index2].prm[cnt].p_pen1 = val;

            val = atof(tmp[9]);
            reax->thbp[index1].prm[cnt].p_val4 = val;
            reax->thbp[index2].prm[cnt].p_val4 = val;
        }
    }

    /* 4-body parameters are entered in compact form. i.e. 0-X-Y-0
       correspond to any type of pair of atoms in 1 and 4
       position. However, explicit X-Y-Z-W takes precedence over the
       default description.
       supports multi-well potentials (upto MAX_4BODY_PARAM in mytypes.h)
       IMPORTANT: for now, directions on how to read multi-entries from ffield
       is not clear */

    /* clear all entries first */
    for ( i = 0; i < reax->num_atom_types; ++i )
    {
        for ( j = 0; j < reax->num_atom_types; ++j )
        {
            for ( k = 0; k < reax->num_atom_types; ++k )
            {
                for ( m = 0; m < reax->num_atom_types; ++m )
                {
                    reax->fbp[i * __N * __N * __N + j * __N * __N + k * __N + m].cnt = 0;
                    tor_flag[i * __N * __N * __N + j * __N * __N + k * __N + m] = 0;
                }
            }
        }
    }

    /* next line is number of 4-body params and some comments */
    fgets( s, MAX_LINE, fp );
    c = Tokenize( s, &tmp );
    l = atoi( tmp[0] );

    for ( i = 0; i < l; i++ )
    {
        fgets( s, MAX_LINE, fp );
        c = Tokenize( s, &tmp );

        j = atoi(tmp[0]) - 1;
        k = atoi(tmp[1]) - 1;
        m = atoi(tmp[2]) - 1;
        n = atoi(tmp[3]) - 1;
        index1 = j * __N * __N * __N + k * __N * __N + m * __N + n;
        index2 = n * __N * __N * __N + m * __N * __N + k * __N + j;

        if (j >= 0 && n >= 0) // this means the entry is not in compact form
        {
            if (j < reax->num_atom_types &&
                    k < reax->num_atom_types &&
                    m < reax->num_atom_types &&
                    n < reax->num_atom_types)
            {
                /* these flags ensure that this entry take precedence
                over the compact form entries */
                tor_flag[index1] = 1;
                tor_flag[index2] = 1;

                reax->fbp[index1].cnt = 1;
                reax->fbp[index2].cnt = 1;
                /* cnt = reax->fbp[j][k][m][n].cnt;
                reax->fbp[j][k][m][n].cnt++;
                 reax->fbp[n][m][k][j].cnt++; */

                val = atof(tmp[4]);
                reax->fbp[index1].prm[0].V1 = val;
                reax->fbp[index2].prm[0].V1 = val;

                val = atof(tmp[5]);
                reax->fbp[index1].prm[0].V2 = val;
                reax->fbp[index2].prm[0].V2 = val;

                val = atof(tmp[6]);
                reax->fbp[index1].prm[0].V3 = val;
                reax->fbp[index2].prm[0].V3 = val;

                val = atof(tmp[7]);
                reax->fbp[index1].prm[0].p_tor1 = val;
                reax->fbp[index2].prm[0].p_tor1 = val;

                val = atof(tmp[8]);
                reax->fbp[index1].prm[0].p_cot1 = val;
                reax->fbp[index2].prm[0].p_cot1 = val;
            }
        }
        else /* This means the entry is of the form 0-X-Y-0 */
        {
            if ( k < reax->num_atom_types && m < reax->num_atom_types )
            {
                for ( p = 0; p < reax->num_atom_types; p++ )
                {
                    for ( o = 0; o < reax->num_atom_types; o++ )
                    {
                        index1 = p * __N * __N * __N + k * __N * __N + m * __N + o;
                        index2 = o * __N * __N * __N + m * __N * __N + k * __N + p;
                        reax->fbp[index1].cnt = 1;
                        reax->fbp[index2].cnt = 1;
                        /* cnt = reax->fbp[p][k][m][o].cnt;
                           reax->fbp[p][k][m][o].cnt++;
                           reax->fbp[o][m][k][p].cnt++; */

                        if (tor_flag[index1] == 0)
                        {
                            reax->fbp[index1].prm[0].V1 = atof(tmp[4]);
                            reax->fbp[index1].prm[0].V2 = atof(tmp[5]);
                            reax->fbp[index1].prm[0].V3 = atof(tmp[6]);
                            reax->fbp[index1].prm[0].p_tor1 = atof(tmp[7]);
                            reax->fbp[index1].prm[0].p_cot1 = atof(tmp[8]);
                        }

                        if (tor_flag[index2] == 0)
                        {
                            reax->fbp[index2].prm[0].V1 = atof(tmp[4]);
                            reax->fbp[index2].prm[0].V2 = atof(tmp[5]);
                            reax->fbp[index2].prm[0].V3 = atof(tmp[6]);
                            reax->fbp[index2].prm[0].p_tor1 = atof(tmp[7]);
                            reax->fbp[index2].prm[0].p_cot1 = atof(tmp[8]);
                        }
                    }
                }
            }
        }
    }


    /* next line is number of hydrogen bond params and some comments */
    fgets( s, MAX_LINE, fp );
    c = Tokenize( s, &tmp );
    l = atoi( tmp[0] );

    for ( i = 0; i < l; i++ )
    {
        fgets( s, MAX_LINE, fp );
        c = Tokenize( s, &tmp );

        j = atoi(tmp[0]) - 1;
        k = atoi(tmp[1]) - 1;
        m = atoi(tmp[2]) - 1;
        index1 = j * __N * __N + k * __N + m;


        if (j < reax->num_atom_types && m < reax->num_atom_types)
        {
            val = atof(tmp[3]);
            reax->hbp[index1].r0_hb = val;

            val = atof(tmp[4]);
            reax->hbp[index1].p_hb1 = val;

            val = atof(tmp[5]);
            reax->hbp[index1].p_hb2 = val;

            val = atof(tmp[6]);
            reax->hbp[index1].p_hb3 = val;

        }
    }

    /* deallocate helper storage */
    for ( i = 0; i < MAX_TOKENS; i++ )
    {
        free( tmp[i] );
    }
    free( tmp );
    free( s );

    /* deallocate tor_flag */
    free( tor_flag );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "force field read\n" );
#endif

    return SUCCESS;
}
