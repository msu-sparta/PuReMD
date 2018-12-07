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

#include "ffield.h"

#include <ctype.h>

#include "tool_box.h"


void Read_Force_Field( FILE* fp, reax_interaction* reax )
{
    char *s;
    char **tmp;
    char ****tor_flag;
    int i, j, k, l, m, n, o, p, cnt;
    real val;

    s = (char*) smalloc( sizeof(char) * MAX_LINE,
            "Read_Force_Field::s" );
    tmp = (char**) smalloc( sizeof(char*) * MAX_TOKENS,
            "Read_Force_Field::tmp" );
    for (i = 0; i < MAX_TOKENS; i++)
    {
        tmp[i] = (char*) smalloc( sizeof(char) * MAX_TOKEN_LEN,
                "Read_Force_Field::tmp[i]" );
    }

    /* reading first header comment */
    fgets( s, MAX_LINE, fp );

    /* line 2 is number of global parameters */
    fgets( s, MAX_LINE, fp );
    Tokenize( s, &tmp );

    /* reading the number of global parameters */
    n = atoi(tmp[0]);
    if (n < 1)
    {
        fprintf( stderr, "[WARNING] number of globals in ffield file is 0!\n" );
        return;
    }

    reax->gp.n_global = n;
    reax->gp.l = (real*) smalloc( sizeof(real) * n,
           "Read_Force_Field::reax->gp-l" );

    /* see reax_types.h for mapping between l[i] and the lambdas used in ff */
    for (i = 0; i < n; i++)
    {
        fgets(s, MAX_LINE, fp);
        Tokenize(s, &tmp);

        val = (real) atof(tmp[0]);

        reax->gp.l[i] = val;
    }

    /* next line is number of atom types and some comments */
    fgets( s, MAX_LINE, fp );
    Tokenize( s, &tmp );
    reax->num_atom_types = atoi(tmp[0]);

    /* 3 lines of comments */
    fgets(s, MAX_LINE, fp);
    fgets(s, MAX_LINE, fp);
    fgets(s, MAX_LINE, fp);

    /* Allocating structures in reax_interaction */
    reax->sbp = (single_body_parameters*) scalloc( reax->num_atom_types, sizeof(single_body_parameters),
            "Read_Force_Field::reax->sbp" );
    reax->tbp = (two_body_parameters**) scalloc( reax->num_atom_types, sizeof(two_body_parameters*),
            "Read_Force_Field::reax->tbp" );
    reax->thbp = (three_body_header***) scalloc( reax->num_atom_types, sizeof(three_body_header**),
            "Read_Force_Field::reax->thbp" );
    reax->hbp = (hbond_parameters***) scalloc( reax->num_atom_types, sizeof(hbond_parameters**),
            "Read_Force_Field::reax->hbp" );
    reax->fbp = (four_body_header****) scalloc( reax->num_atom_types, sizeof(four_body_header***),
            "Read_Force_Field::reax->fbp" );
    tor_flag  = (char****) scalloc( reax->num_atom_types, sizeof(char***),
            "Read_Force_Field::tor_flag" );

    for ( i = 0; i < reax->num_atom_types; i++ )
    {
        reax->tbp[i] = (two_body_parameters*) scalloc( reax->num_atom_types, sizeof(two_body_parameters),
                "Read_Force_Field::reax->tbp[i]" );
        reax->thbp[i] = (three_body_header**) scalloc( reax->num_atom_types, sizeof(three_body_header*),
                "Read_Force_Field::reax->thbp[i]" );
        reax->hbp[i] = (hbond_parameters**) scalloc( reax->num_atom_types, sizeof(hbond_parameters*),
                "Read_Force_Field::reax->hbp[i]" );
        reax->fbp[i] = (four_body_header***) scalloc( reax->num_atom_types, sizeof(four_body_header**),
                "Read_Force_Field::reax->fbp[i]" );
        tor_flag[i] = (char***) scalloc( reax->num_atom_types, sizeof(char**),
                "Read_Force_Field::tor_flag[i]" );

        for ( j = 0; j < reax->num_atom_types; j++ )
        {
            reax->thbp[i][j] = (three_body_header*) scalloc( reax->num_atom_types, sizeof(three_body_header),
                    "Read_Force_Field::reax->thbp[i][j]" );
            reax->hbp[i][j] = (hbond_parameters*) scalloc( reax->num_atom_types, sizeof(hbond_parameters),
                    "Read_Force_Field::reax->hbp[i][j]" );
            reax->fbp[i][j] = (four_body_header**) scalloc( reax->num_atom_types, sizeof(four_body_header*),
                    "Read_Force_Field::reax->fbp[i][j]" );
            tor_flag[i][j]  = (char**) scalloc( reax->num_atom_types, sizeof(char*),
                    "Read_Force_Field::tor_flag[i][j]" );

            for (k = 0; k < reax->num_atom_types; k++)
            {
                reax->fbp[i][j][k] = (four_body_header*) scalloc( reax->num_atom_types, sizeof(four_body_header),
                        "Read_Force_Field::reax->fbp[i][j][k]" );
                tor_flag[i][j][k]  = (char*) scalloc( reax->num_atom_types, sizeof(char),
                        "Read_Force_Field::tor_flag[i][j][k]" );
            }
        }
    }

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
        Tokenize( s, &tmp );

        for ( j = 0; j < strnlen( tmp[0], MAX_TOKEN_LEN ); ++j )
        {
            reax->sbp[i].name[j] = toupper( tmp[0][j] );
        }

        val = atof(tmp[1]);
        reax->sbp[i].r_s = val;
        val = atof(tmp[2]);
        reax->sbp[i].valency = val;
        val = atof(tmp[3]);
        reax->sbp[i].mass = val;
        val = atof(tmp[4]);
        reax->sbp[i].r_vdw = val;
        val = atof(tmp[5]);
        reax->sbp[i].epsilon = val;
        val = atof(tmp[6]);
        reax->sbp[i].gamma = val;
        val = atof(tmp[7]);
        reax->sbp[i].r_pi = val;
        val = atof(tmp[8]);
        reax->sbp[i].valency_e  = val;
        reax->sbp[i].nlp_opt = 0.5 * (reax->sbp[i].valency_e - reax->sbp[i].valency);

        /* line two */
        fgets( s, MAX_LINE, fp );
        Tokenize( s, &tmp );

        val = atof(tmp[0]);
        reax->sbp[i].alpha = val;
        val = atof(tmp[1]);
        reax->sbp[i].gamma_w = val;
        val = atof(tmp[2]);
        reax->sbp[i].valency_boc = val;
        val = atof(tmp[3]);
        reax->sbp[i].p_ovun5 = val;
        val = atof(tmp[4]);
        val = atof(tmp[5]);
        reax->sbp[i].chi = val;
        val = atof(tmp[6]);
        reax->sbp[i].eta = 2.0 * val;
        /* this is the parameter that is used to determine
           which type of atoms participate in h-bonds.
           1 is for H - 2 for O, N, S - 0 for all others.*/
        val = atof(tmp[7]);
        reax->sbp[i].p_hbond = (int)(val + 0.1);
        //0.1 is to avoid from truncating down!

        /* line 3 */
        fgets( s, MAX_LINE, fp );
        Tokenize( s, &tmp );

        val = atof(tmp[0]);
        reax->sbp[i].r_pi_pi = val;
        val = atof(tmp[1]);
        reax->sbp[i].p_lp2 = val;
        val = atof(tmp[2]);
        val = atof(tmp[3]);
        reax->sbp[i].b_o_131 = val;
        val = atof(tmp[4]);
        reax->sbp[i].b_o_132 = val;
        val = atof(tmp[5]);
        reax->sbp[i].b_o_133 = val;
        val = atof(tmp[6]);
        reax->sbp[i].b_s_acks2 = val;
        val = atof(tmp[7]);

        /* line 4  */
        fgets( s, MAX_LINE, fp );
        Tokenize( s, &tmp );

        val = atof(tmp[0]);
        reax->sbp[i].p_ovun2 = val;
        val = atof(tmp[1]);
        reax->sbp[i].p_val3 = val;
        val = atof(tmp[2]);
        val = atof(tmp[3]);
        reax->sbp[i].valency_val = val;
        val = atof(tmp[4]);
        reax->sbp[i].p_val5 = val;
        val = atof(tmp[5]);
        reax->sbp[i].rcore2 = val;
        val = atof(tmp[6]);
        reax->sbp[i].ecore2 = val;
        val = atof(tmp[7]);
        reax->sbp[i].acore2 = val;

        /* inner-wall */
        if ( reax->sbp[i].rcore2 > 0.01 && reax->sbp[i].acore2 > 0.01 ) // Inner-wall
        {
            /* shielding vdWaals */
            if ( reax->sbp[i].gamma_w > 0.5 )
            {
                if ( reax->gp.vdw_type != 0 && reax->gp.vdw_type != 3 )
                    fprintf( stderr, "[WARNING] Inconsistent vdWaals-parameters!\n" \
                             "  Force field parameters for element %s\n"        \
                             "  indicate inner wall+shielding, but earlier\n"   \
                             "  atoms indicate different vdWaals-method.\n"     \
                             "  This may cause division-by-zero errors.\n"      \
                             "  Keeping vdWaals-setting for earlier atoms.\n",
                             reax->sbp[i].name );
                else
                {
                    reax->gp.vdw_type = 3;

#if defined(DEBUG)
                    fprintf( stderr, "vdWaals type for element %s: Shielding+inner-wall",
                             reax->sbp[i].name );
#endif
                }
            }
            /* no shielding vdWaals parameters present */
            else
            {
                if ( reax->gp.vdw_type != 0 && reax->gp.vdw_type != 2 )
                {
                    fprintf( stderr, "[WARNING] Inconsistent vdWaals-parameters!\n" \
                             "  Force field parameters for element %s\n"        \
                             "  indicate inner wall without shielding, but earlier\n" \
                             "  atoms indicate different vdWaals-method.\n"     \
                             "  This may cause division-by-zero errors.\n"      \
                             "  Keeping vdWaals-setting for earlier atoms.\n",
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
        /* no Inner wall parameters present */
        else
        {
            /* shielding vdWaals */
            if ( reax->sbp[i].gamma_w > 0.5 )
            {
                if ( reax->gp.vdw_type != 0 && reax->gp.vdw_type != 1 )
                    fprintf( stderr, "[WARNING] Inconsistent vdWaals-parameters!\n" \
                             "  Force field parameters for element %s\n"        \
                             "  indicate shielding without inner wall, but earlier\n" \
                             "  atoms indicate different vdWaals-method.\n"     \
                             "  This may cause division-by-zero errors.\n"      \
                             "  Keeping vdWaals-setting for earlier atoms.\n",
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
                fprintf( stderr, "[ERROR] Inconsistent vdWaals-parameters\n" \
                         "  No shielding or inner-wall set for element %s\n",
                         reax->sbp[i].name );
                exit( INVALID_INPUT );
            }
        }
    }


    /* next line is number of two body combination and some comments */
    fgets(s, MAX_LINE, fp);
    Tokenize(s, &tmp);
    l = atoi(tmp[0]);

    /* a line of comments */
    fgets(s, MAX_LINE, fp);

    for (i = 0; i < l; i++)
    {
        /* line 1 */
        fgets(s, MAX_LINE, fp);
        Tokenize(s, &tmp);

        j = atoi(tmp[0]) - 1;
        k = atoi(tmp[1]) - 1;

        if (j < reax->num_atom_types && k < reax->num_atom_types)
        {

            val = atof(tmp[2]);
            reax->tbp[j][k].De_s = val;
            reax->tbp[k][j].De_s = val;
            val = atof(tmp[3]);
            reax->tbp[j][k].De_p = val;
            reax->tbp[k][j].De_p = val;
            val = atof(tmp[4]);
            reax->tbp[j][k].De_pp = val;
            reax->tbp[k][j].De_pp = val;
            val = atof(tmp[5]);
            reax->tbp[j][k].p_be1 = val;
            reax->tbp[k][j].p_be1 = val;
            val = atof(tmp[6]);
            reax->tbp[j][k].p_bo5 = val;
            reax->tbp[k][j].p_bo5 = val;
            val = atof(tmp[7]);
            reax->tbp[j][k].v13cor = val;
            reax->tbp[k][j].v13cor = val;

            val = atof(tmp[8]);
            reax->tbp[j][k].p_bo6 = val;
            reax->tbp[k][j].p_bo6 = val;
            val = atof(tmp[9]);
            reax->tbp[j][k].p_ovun1 = val;
            reax->tbp[k][j].p_ovun1 = val;

            /* line 2 */
            fgets(s, MAX_LINE, fp);
            Tokenize(s, &tmp);

            val = atof(tmp[0]);
            reax->tbp[j][k].p_be2 = val;
            reax->tbp[k][j].p_be2 = val;
            val = atof(tmp[1]);
            reax->tbp[j][k].p_bo3 = val;
            reax->tbp[k][j].p_bo3 = val;
            val = atof(tmp[2]);
            reax->tbp[j][k].p_bo4 = val;
            reax->tbp[k][j].p_bo4 = val;
            val = atof(tmp[3]);

            val = atof(tmp[4]);
            reax->tbp[j][k].p_bo1 = val;
            reax->tbp[k][j].p_bo1 = val;
            val = atof(tmp[5]);
            reax->tbp[j][k].p_bo2 = val;
            reax->tbp[k][j].p_bo2 = val;
            val = atof(tmp[6]);
            reax->tbp[j][k].ovc = val;
            reax->tbp[k][j].ovc = val;

            val = atof(tmp[7]);
        }
    }

    /* calculating combination rules and filling up remaining fields. */
    for (i = 0; i < reax->num_atom_types; i++)
    {
        for (j = i; j < reax->num_atom_types; j++)
        {
            reax->tbp[i][j].r_s = 0.5 * (reax->sbp[i].r_s + reax->sbp[j].r_s);
            reax->tbp[j][i].r_s = 0.5 * (reax->sbp[j].r_s + reax->sbp[i].r_s);

            reax->tbp[i][j].r_p = 0.5 * (reax->sbp[i].r_pi + reax->sbp[j].r_pi);
            reax->tbp[j][i].r_p = 0.5 * (reax->sbp[j].r_pi + reax->sbp[i].r_pi);

            reax->tbp[i][j].r_pp = 0.5 * (reax->sbp[i].r_pi_pi + reax->sbp[j].r_pi_pi);
            reax->tbp[j][i].r_pp = 0.5 * (reax->sbp[j].r_pi_pi + reax->sbp[i].r_pi_pi);

            reax->tbp[i][j].p_boc3 = SQRT(reax->sbp[i].b_o_132 * reax->sbp[j].b_o_132);

            reax->tbp[j][i].p_boc3 = SQRT(reax->sbp[j].b_o_132 * reax->sbp[i].b_o_132);

            reax->tbp[i][j].p_boc4 = SQRT(reax->sbp[i].b_o_131 * reax->sbp[j].b_o_131);
            reax->tbp[j][i].p_boc4 = SQRT(reax->sbp[j].b_o_131 * reax->sbp[i].b_o_131);

            reax->tbp[i][j].p_boc5 = SQRT(reax->sbp[i].b_o_133 * reax->sbp[j].b_o_133);
            reax->tbp[j][i].p_boc5 = SQRT(reax->sbp[j].b_o_133 * reax->sbp[i].b_o_133);

            reax->tbp[i][j].D = SQRT(reax->sbp[i].epsilon * reax->sbp[j].epsilon);
            reax->tbp[j][i].D = SQRT(reax->sbp[j].epsilon * reax->sbp[i].epsilon);

            reax->tbp[i][j].alpha = SQRT(reax->sbp[i].alpha * reax->sbp[j].alpha);
            reax->tbp[j][i].alpha = SQRT(reax->sbp[j].alpha * reax->sbp[i].alpha);

            reax->tbp[i][j].r_vdW = 2.0 * SQRT(reax->sbp[i].r_vdw * reax->sbp[j].r_vdw);
            reax->tbp[j][i].r_vdW = 2.0 * SQRT(reax->sbp[j].r_vdw * reax->sbp[i].r_vdw);

            reax->tbp[i][j].gamma_w = SQRT(reax->sbp[i].gamma_w * reax->sbp[j].gamma_w);
            reax->tbp[j][i].gamma_w = SQRT(reax->sbp[j].gamma_w * reax->sbp[i].gamma_w);

            reax->tbp[i][j].gamma = POW(reax->sbp[i].gamma * reax->sbp[j].gamma, -1.5);
            reax->tbp[j][i].gamma = POW(reax->sbp[j].gamma * reax->sbp[i].gamma, -1.5);

            reax->tbp[i][j].acore = SQRT( reax->sbp[i].acore2 * reax->sbp[j].acore2 );
            reax->tbp[j][i].acore = SQRT( reax->sbp[j].acore2 * reax->sbp[i].acore2 );

            reax->tbp[i][j].ecore = SQRT( reax->sbp[i].ecore2 * reax->sbp[j].ecore2 );
            reax->tbp[j][i].ecore = SQRT( reax->sbp[j].ecore2 * reax->sbp[i].ecore2 );

            reax->tbp[i][j].rcore = SQRT( reax->sbp[i].rcore2 * reax->sbp[j].rcore2 );
            reax->tbp[j][i].rcore = SQRT( reax->sbp[j].rcore2 * reax->sbp[i].rcore2 );
        }
    }

    /* next line is number of 2-body offdiagonal combinations and some comments */
    /* these are two body off-diagonal terms that are different from the
     * combination rules defined above */
    fgets(s, MAX_LINE, fp);
    Tokenize(s, &tmp);
    l = atoi(tmp[0]);

    for (i = 0; i < l; i++)
    {
        fgets(s, MAX_LINE, fp);
        Tokenize(s, &tmp);

        j = atoi(tmp[0]) - 1;
        k = atoi(tmp[1]) - 1;

        if (j < reax->num_atom_types && k < reax->num_atom_types)
        {
            val = atof(tmp[2]);
            if (val > 0.0)
            {
                reax->tbp[j][k].D = val;
                reax->tbp[k][j].D = val;
            }

            val = atof(tmp[3]);
            if (val > 0.0)
            {
                reax->tbp[j][k].r_vdW = 2 * val;
                reax->tbp[k][j].r_vdW = 2 * val;
            }

            val = atof(tmp[4]);
            if (val > 0.0)
            {
                reax->tbp[j][k].alpha = val;
                reax->tbp[k][j].alpha = val;
            }

            val = atof(tmp[5]);
            if (val > 0.0)
            {
                reax->tbp[j][k].r_s = val;
                reax->tbp[k][j].r_s = val;
            }

            val = atof(tmp[6]);
            if (val > 0.0)
            {
                reax->tbp[j][k].r_p = val;
                reax->tbp[k][j].r_p = val;
            }

            val = atof(tmp[7]);
            if (val > 0.0)
            {
                reax->tbp[j][k].r_pp = val;
                reax->tbp[k][j].r_pp = val;
            }
        }
    }

    /* 3-body parameters -
     * supports multi-well potentials (upto MAX_3BODY_PARAM in reax_types.h) */
    /* clear entries first */
    for ( i = 0; i < reax->num_atom_types; ++i )
    {
        for ( j = 0; j < reax->num_atom_types; ++j )
        {
            for ( k = 0; k < reax->num_atom_types; ++k )
            {
                reax->thbp[i][j][k].cnt = 0;
            }
        }
    }

    /* next line is number of 3-body params and some comments */
    fgets( s, MAX_LINE, fp );
    Tokenize( s, &tmp );
    l = atoi( tmp[0] );

    for ( i = 0; i < l; i++ )
    {
        fgets(s, MAX_LINE, fp);
        Tokenize(s, &tmp);

        j = atoi(tmp[0]) - 1;
        k = atoi(tmp[1]) - 1;
        m = atoi(tmp[2]) - 1;

        if (j < reax->num_atom_types &&
                k < reax->num_atom_types &&
                m < reax->num_atom_types)
        {
            cnt = reax->thbp[j][k][m].cnt;
            reax->thbp[j][k][m].cnt++;
            reax->thbp[m][k][j].cnt++;

            val = atof(tmp[3]);
            reax->thbp[j][k][m].prm[cnt].theta_00 = val;
            reax->thbp[m][k][j].prm[cnt].theta_00 = val;

            val = atof(tmp[4]);
            reax->thbp[j][k][m].prm[cnt].p_val1 = val;
            reax->thbp[m][k][j].prm[cnt].p_val1 = val;

            val = atof(tmp[5]);
            reax->thbp[j][k][m].prm[cnt].p_val2 = val;
            reax->thbp[m][k][j].prm[cnt].p_val2 = val;

            val = atof(tmp[6]);
            reax->thbp[j][k][m].prm[cnt].p_coa1 = val;
            reax->thbp[m][k][j].prm[cnt].p_coa1 = val;

            val = atof(tmp[7]);
            reax->thbp[j][k][m].prm[cnt].p_val7 = val;
            reax->thbp[m][k][j].prm[cnt].p_val7 = val;

            val = atof(tmp[8]);
            reax->thbp[j][k][m].prm[cnt].p_pen1 = val;
            reax->thbp[m][k][j].prm[cnt].p_pen1 = val;

            val = atof(tmp[9]);
            reax->thbp[j][k][m].prm[cnt].p_val4 = val;
            reax->thbp[m][k][j].prm[cnt].p_val4 = val;
        }
    }

    /* 4-body parameters are entered in compact form. i.e. 0-X-Y-0
     * correspond to any type of pair of atoms in 1 and 4
     * position. However, explicit X-Y-Z-W takes precedence over the
     * default description.
     * supports multi-well potentials (upto MAX_4BODY_PARAM in reax_types.h)
     * IMPORTANT: for now, directions on how to read multi-entries from ffield
     * is not clear */

    /* clear all entries first */
    for ( i = 0; i < reax->num_atom_types; ++i )
    {
        for ( j = 0; j < reax->num_atom_types; ++j )
        {
            for ( k = 0; k < reax->num_atom_types; ++k )
            {
                for ( m = 0; m < reax->num_atom_types; ++m )
                {
                    reax->fbp[i][j][k][m].cnt = 0;
                    tor_flag[i][j][k][m] = 0;
                }
            }
        }
    }

    /* next line is number of 4-body params and some comments */
    fgets( s, MAX_LINE, fp );
    Tokenize( s, &tmp );
    l = atoi( tmp[0] );

    for ( i = 0; i < l; i++ )
    {
        fgets( s, MAX_LINE, fp );
        Tokenize( s, &tmp );

        j = atoi(tmp[0]) - 1;
        k = atoi(tmp[1]) - 1;
        m = atoi(tmp[2]) - 1;
        n = atoi(tmp[3]) - 1;

        /* this means the entry is not in compact form */
        if (j >= 0 && n >= 0)
        {
            if (j < reax->num_atom_types &&
                    k < reax->num_atom_types &&
                    m < reax->num_atom_types &&
                    n < reax->num_atom_types)
            {
                /* these flags ensure that this entry take precedence
                 * over the compact form entries */
                tor_flag[j][k][m][n] = 1;
                tor_flag[n][m][k][j] = 1;

                reax->fbp[j][k][m][n].cnt = 1;
                reax->fbp[n][m][k][j].cnt = 1;

//                cnt = reax->fbp[j][k][m][n].cnt;
//                reax->fbp[j][k][m][n].cnt++;
//                reax->fbp[n][m][k][j].cnt++;

                val = atof(tmp[4]);
                reax->fbp[j][k][m][n].prm[0].V1 = val;
                reax->fbp[n][m][k][j].prm[0].V1 = val;

                val = atof(tmp[5]);
                reax->fbp[j][k][m][n].prm[0].V2 = val;
                reax->fbp[n][m][k][j].prm[0].V2 = val;

                val = atof(tmp[6]);
                reax->fbp[j][k][m][n].prm[0].V3 = val;
                reax->fbp[n][m][k][j].prm[0].V3 = val;

                val = atof(tmp[7]);
                reax->fbp[j][k][m][n].prm[0].p_tor1 = val;
                reax->fbp[n][m][k][j].prm[0].p_tor1 = val;

                val = atof(tmp[8]);
                reax->fbp[j][k][m][n].prm[0].p_cot1 = val;
                reax->fbp[n][m][k][j].prm[0].p_cot1 = val;
            }
        }
        /* This means the entry is of the form 0-X-Y-0 */
        else
        {
            if ( k < reax->num_atom_types && m < reax->num_atom_types )
            {
                for ( p = 0; p < reax->num_atom_types; p++ )
                {
                    for ( o = 0; o < reax->num_atom_types; o++ )
                    {
                        reax->fbp[p][k][m][o].cnt = 1;
                        reax->fbp[o][m][k][p].cnt = 1;

//                        cnt = reax->fbp[p][k][m][o].cnt;
//                        reax->fbp[p][k][m][o].cnt++;
//                        reax->fbp[o][m][k][p].cnt++;

                        if (tor_flag[p][k][m][o] == 0)
                        {
                            reax->fbp[p][k][m][o].prm[0].V1 = atof(tmp[4]);
                            reax->fbp[p][k][m][o].prm[0].V2 = atof(tmp[5]);
                            reax->fbp[p][k][m][o].prm[0].V3 = atof(tmp[6]);
                            reax->fbp[p][k][m][o].prm[0].p_tor1 = atof(tmp[7]);
                            reax->fbp[p][k][m][o].prm[0].p_cot1 = atof(tmp[8]);
                        }

                        if (tor_flag[o][m][k][p] == 0)
                        {
                            reax->fbp[o][m][k][p].prm[0].V1 = atof(tmp[4]);
                            reax->fbp[o][m][k][p].prm[0].V2 = atof(tmp[5]);
                            reax->fbp[o][m][k][p].prm[0].V3 = atof(tmp[6]);
                            reax->fbp[o][m][k][p].prm[0].p_tor1 = atof(tmp[7]);
                            reax->fbp[o][m][k][p].prm[0].p_cot1 = atof(tmp[8]);
                        }
                    }
                }
            }
        }
    }

    /* next line is number of hydrogen bond params and some comments */
    fgets( s, MAX_LINE, fp );
    Tokenize( s, &tmp );
    l = atoi( tmp[0] );

    for ( i = 0; i < l; i++ )
    {
        fgets( s, MAX_LINE, fp );
        Tokenize( s, &tmp );

        j = atoi(tmp[0]) - 1;
        k = atoi(tmp[1]) - 1;
        m = atoi(tmp[2]) - 1;

        if ( j < reax->num_atom_types && m < reax->num_atom_types )
        {
            val = atof(tmp[3]);
            reax->hbp[j][k][m].r0_hb = val;

            val = atof(tmp[4]);
            reax->hbp[j][k][m].p_hb1 = val;

            val = atof(tmp[5]);
            reax->hbp[j][k][m].p_hb2 = val;

            val = atof(tmp[6]);
            reax->hbp[j][k][m].p_hb3 = val;

        }
    }

    /* deallocate helper storage */
    for ( i = 0; i < MAX_TOKENS; i++ )
    {
        sfree( tmp[i], "Read_Force_Field::tmp[i]" );
    }
    sfree( tmp, "Read_Force_Field::tmp" );
    sfree( s, "Read_Force_Field::s" );

    /* deallocate tor_flag */
    for ( i = 0; i < reax->num_atom_types; i++ )
    {
        for ( j = 0; j < reax->num_atom_types; j++ )
        {
            for ( k = 0; k < reax->num_atom_types; k++ )
            {
                sfree( tor_flag[i][j][k], "Read_Force_Field::tor_flag[i][j][k]" );
            }

            sfree( tor_flag[i][j], "Read_Force_Field::tor_flag[i][j]" );
        }

        sfree( tor_flag[i], "Read_Force_Field::tor_flag[i]" );
    }

    sfree( tor_flag, "Read_Force_Field::tor_flag" );
}
