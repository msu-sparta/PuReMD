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

#include "param.h"
#include "traj.h"
#include "ctype.h"


int Get_Atom_Type( reax_interaction *reaxprm, char *s )
{
    int i;

    for ( i = 0; i < reaxprm->num_atom_types; ++i )
        if ( !strcmp( reaxprm->sbp[i].name, s ) )
            return i;

    fprintf( stderr, "Unknown atom type %s. Terminating...\n", s );
    exit( UNKNOWN_ATOM_TYPE_ERR );
}


int Tokenize(char* s, char*** tok)
{
    char test[MAX_LINE];
    char *sep = "\t \n!=";
    char *word;
    int count = 0;

    strncpy( test, s, MAX_LINE );

    // fprintf( stderr, "|%s|\n", test );

    for ( word = strtok(test, sep); word; word = strtok(NULL, sep) )
    {
        strncpy( (*tok)[count], word, MAX_LINE );
        count++;
    }

    return count;
}


/* Initialize Taper params */
void Init_Taper( control_params *control )
{
    real d1, d7;
    real swa, swa2, swa3;
    real swb, swb2, swb3;

    swa = control->r_low;
    swb = control->r_cut;

    if ( fabs( swa ) > 0.01 )
        fprintf( stderr, "Warning: non-zero value for lower Taper-radius cutoff\n" );

    if ( swb < 0 )
    {
        fprintf( stderr, "Negative value for upper Taper-radius cutoff\n" );
        exit( INVALID_INPUT );
    }
    else if ( swb < 5 )
        fprintf( stderr, "Warning: low value for upper Taper-radius cutoff:%f\n",
                 swb );

    d1 = swb - swa;
    d7 = POW( d1, 7.0 );
    swa2 = SQR( swa );
    swa3 = CUBE( swa );
    swb2 = SQR( swb );
    swb3 = CUBE( swb );

    control->Tap7 =  20.0 / d7;
    control->Tap6 = -70.0 * (swa + swb) / d7;
    control->Tap5 =  84.0 * (swa2 + 3.0 * swa * swb + swb2) / d7;
    control->Tap4 = -35.0 * (swa3 + 9.0 * swa2 * swb + 9.0 * swa * swb2 + swb3 ) / d7;
    control->Tap3 = 140.0 * (swa3 * swb + 3.0 * swa2 * swb2 + swa * swb3 ) / d7;
    control->Tap2 = -210.0 * (swa3 * swb2 + swa2 * swb3) / d7;
    control->Tap1 = 140.0 * swa3 * swb3 / d7;
    control->Tap0 = (-35.0 * swa3 * swb2 * swb2 + 21.0 * swa2 * swb3 * swb2 +
                     7.0 * swa * swb3 * swb3 + swb3 * swb3 * swb ) / d7;
}



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
        tmp[i] = (char*) malloc(sizeof(char) * MAX_TOKEN_LEN);


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
        return 1;
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
            reax->sbp[i].name[j] = toupper( tmp[0][j] );

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
                    fprintf( stderr, "Warning: inconsistent vdWaals-parameters\n" \
                             "Force field parameters for element %s\n"        \
                             "indicate inner wall+shielding, but earlier\n"   \
                             "atoms indicate different vdWaals-method.\n"     \
                             "This may cause division-by-zero errors.\n"      \
                             "Keeping vdWaals-setting for earlier atoms.\n",
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
            else    // No shielding vdWaals parameters present
            {
                if ( reax->gp.vdw_type != 0 && reax->gp.vdw_type != 2 )
                    fprintf( stderr, "Warning: inconsistent vdWaals-parameters\n" \
                             "Force field parameters for element %s\n"        \
                             "indicate inner wall without shielding, but earlier\n" \
                             "atoms indicate different vdWaals-method.\n"     \
                             "This may cause division-by-zero errors.\n"      \
                             "Keeping vdWaals-setting for earlier atoms.\n",
                             reax->sbp[i].name );
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
        for ( j = 0; j < reax->num_atom_types; ++j )
            for ( k = 0; k < reax->num_atom_types; ++k )
                for ( m = 0; m < reax->num_atom_types; ++m )
                {
                    reax->fbp[i * __N * __N * __N + j * __N * __N + k * __N + m].cnt = 0;
                    tor_flag[i * __N * __N * __N + j * __N * __N + k * __N + m] = 0;
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
                for ( p = 0; p < reax->num_atom_types; p++ )
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
        free( tmp[i] );
    free( tmp );
    free( s );


    /* deallocate tor_flag */
    free( tor_flag );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "force field read\n" );
#endif

    return 0;
}


char Read_Control_File( FILE* fp, reax_system *system, control_params* control,
                        output_controls *out_control )
{
    char *s, **tmp;
    int c, i;
    real val;
    int ival;

    /* assign default values */
    strcpy( control->sim_name, "default.sim" );

    control->restart = 0;
    out_control->restart_format = 1;
    out_control->restart_freq = 0;
    strcpy( control->restart_from, "default.res" );
    out_control->restart_freq = 0;
    control->random_vel = 0;

    control->reposition_atoms = 0;

    control->ensemble = 0;
    control->nsteps = 0;
    control->dt = 0.25;

    control->geo_format = 1;
    control->restrict_bonds = 0;

    control->periodic_boundaries = 1;
    control->periodic_images[0] = 0;
    control->periodic_images[1] = 0;
    control->periodic_images[2] = 0;

    control->reneighbor = 1;
    control->vlist_cut = 0;
    control->nbr_cut = 4.;
    control->r_cut = 10;
    control->max_far_nbrs = 1000;
    control->bo_cut = 0.01;
    control->thb_cut = 0.001;
    control->hb_cut = 7.50;

    control->q_err = 0.000001;
    control->tabulate = 0;
    //TODO
    control->refactor = 100;
    //TODO -- change this to 5.

    control->droptol = 0.01;

    control->T_init = 0.;
    control->T_final = 300.;
    control->Tau_T = 1.0;
    control->T_mode = 0.;
    control->T_rate = 1.;
    control->T_freq = 1.;

    control->P[0] = 0.000101325;
    control->P[1] = 0.000101325;
    control->P[2] = 0.000101325;
    control->Tau_P[0]  = 500.0;
    control->Tau_P[1]  = 500.0;
    control->Tau_P[2]  = 500.0;
    control->Tau_PT = 500.0;
    control->compressibility = 1.0;
    control->press_mode = 0;

    control->remove_CoM_vel = 25;

    out_control->debug_level = 0;
    out_control->energy_update_freq = 10;

    out_control->write_steps = 100;
    out_control->traj_compress = 0;
    out_control->write = fprintf;
    out_control->traj_format = 0;
    out_control->write_header =
        (int (*)( reax_system*, control_params*,
                  static_storage*, void* )) Write_Custom_Header;
    out_control->append_traj_frame =
        (int (*)( reax_system*, control_params*, simulation_data*,
                  static_storage*, list **, void* )) Append_Custom_Frame;

    strcpy( out_control->traj_title, "default_title" );
    out_control->atom_format = 0;
    out_control->bond_info = 0;
    out_control->angle_info = 0;

    control->molec_anal = 0;
    control->freq_molec_anal = 0;
    control->bg_cut = 0.3;
    control->num_ignored = 0;
    memset( control->ignore, 0, sizeof(int)*MAX_ATOM_TYPES );

    control->dipole_anal = 0;
    control->freq_dipole_anal = 0;

    control->diffusion_coef = 0;
    control->freq_diffusion_coef = 0;
    control->restrict_type = 0;

    /* memory allocations */
    s = (char*) malloc(sizeof(char) * MAX_LINE);
    tmp = (char**) malloc(sizeof(char*)*MAX_TOKENS);
    for (i = 0; i < MAX_TOKENS; i++)
        tmp[i] = (char*) malloc(sizeof(char) * MAX_LINE);

    /* read control parameters file */
    while (!feof(fp))
    {
        fgets(s, MAX_LINE, fp);
        c = Tokenize(s, &tmp);

        if ( strcmp(tmp[0], "simulation_name") == 0 )
        {
            strcpy( control->sim_name, tmp[1] );
        }
        //else if( strcmp(tmp[0], "restart") == 0 ) {
        //  ival = atoi(tmp[1]);
        //  control->restart = ival;
        //}
        else if ( strcmp(tmp[0], "restart_format") == 0 )
        {
            ival = atoi(tmp[1]);
            out_control->restart_format = ival;
        }
        else if ( strcmp(tmp[0], "restart_freq") == 0 )
        {
            ival = atoi(tmp[1]);
            out_control->restart_freq = ival;
        }
        else if ( strcmp(tmp[0], "random_vel") == 0 )
        {
            ival = atoi(tmp[1]);
            control->random_vel = ival;
        }
        else if ( strcmp(tmp[0], "reposition_atoms") == 0 )
        {
            ival = atoi(tmp[1]);
            control->reposition_atoms = ival;
        }
        else if ( strcmp(tmp[0], "ensemble_type") == 0 )
        {
            ival = atoi(tmp[1]);
            control->ensemble = ival;
        }
        else if ( strcmp(tmp[0], "nsteps") == 0 )
        {
            ival = atoi(tmp[1]);
            control->nsteps = ival;
        }
        else if ( strcmp(tmp[0], "dt") == 0 )
        {
            val = atof(tmp[1]);
            control->dt = val * 1.e-3;  // convert dt from fs to ps!
        }
        else if ( strcmp(tmp[0], "periodic_boundaries") == 0 )
        {
            ival = atoi( tmp[1] );
            control->periodic_boundaries = ival;
        }
        else if ( strcmp(tmp[0], "periodic_images") == 0 )
        {
            ival = atoi(tmp[1]);
            control->periodic_images[0] = ival;
            ival = atoi(tmp[2]);
            control->periodic_images[1] = ival;
            ival = atoi(tmp[3]);
            control->periodic_images[2] = ival;
        }
        else if ( strcmp(tmp[0], "geo_format") == 0 )
        {
            ival = atoi( tmp[1] );
            control->geo_format = ival;
        }
        else if ( strcmp(tmp[0], "restrict_bonds") == 0 )
        {
            ival = atoi( tmp[1] );
            control->restrict_bonds = ival;
        }
        else if ( strcmp(tmp[0], "tabulate_long_range") == 0 )
        {
            ival = atoi( tmp[1] );
            control->tabulate = ival;
        }
        else if ( strcmp(tmp[0], "reneighbor") == 0 )
        {
            ival = atoi( tmp[1] );
            control->reneighbor = ival;
        }
        else if ( strcmp(tmp[0], "vlist_buffer") == 0 )
        {
            val = atof(tmp[1]);
            control->vlist_cut = val;
        }
        else if ( strcmp(tmp[0], "nbrhood_cutoff") == 0 )
        {
            val = atof(tmp[1]);
            control->nbr_cut = val;
        }
        else if ( strcmp(tmp[0], "thb_cutoff") == 0 )
        {
            val = atof(tmp[1]);
            control->thb_cut = val;
        }
        else if ( strcmp(tmp[0], "hbond_cutoff") == 0 )
        {
            val = atof( tmp[1] );
            control->hb_cut = val;
        }
        else if ( strcmp(tmp[0], "q_err") == 0 )
        {
            val = atof( tmp[1] );
            control->q_err = val;
        }
        else if ( strcmp(tmp[0], "ilu_refactor") == 0 )
        {
            ival = atoi( tmp[1] );
            control->refactor = ival;
        }
        else if ( strcmp(tmp[0], "ilu_droptol") == 0 )
        {
            val = atof( tmp[1] );
            control->droptol = val;
        }
        else if ( strcmp(tmp[0], "temp_init") == 0 )
        {
            val = atof(tmp[1]);
            control->T_init = val;

            if ( control->T_init < 0.001 )
                control->T_init = 0.001;
        }
        else if ( strcmp(tmp[0], "temp_final") == 0 )
        {
            val = atof(tmp[1]);
            control->T_final = val;

            if ( control->T_final < 0.1 )
                control->T_final = 0.1;
        }
        else if ( strcmp(tmp[0], "t_mass") == 0 )
        {
            val = atof(tmp[1]);
            control->Tau_T = val * 1.e-3;    // convert t_mass from fs to ps
        }
        else if ( strcmp(tmp[0], "t_mode") == 0 )
        {
            ival = atoi(tmp[1]);
            control->T_mode = ival;
        }
        else if ( strcmp(tmp[0], "t_rate") == 0 )
        {
            val = atof(tmp[1]);
            control->T_rate = val;
        }
        else if ( strcmp(tmp[0], "t_freq") == 0 )
        {
            val = atof(tmp[1]);
            control->T_freq = val;
        }
        else if ( strcmp(tmp[0], "pressure") == 0 )
        {
            if ( control->ensemble == iNPT )
            {
                val = atof(tmp[1]);
                control->P[0] = control->P[1] = control->P[2] = val;
            }
            else if ( control->ensemble == sNPT )
            {
                val = atof(tmp[1]);
                control->P[0] = val;

                val = atof(tmp[2]);
                control->P[1] = val;

                val = atof(tmp[3]);
                control->P[2] = val;
            }
        }
        else if ( strcmp(tmp[0], "p_mass") == 0 )
        {
            if ( control->ensemble == iNPT )
            {
                val = atof(tmp[1]);
                control->Tau_P[0] = val * 1.e-3;   // convert p_mass from fs to ps
            }
            else if ( control->ensemble == sNPT )
            {
                val = atof(tmp[1]);
                control->Tau_P[0] = val * 1.e-3;   // convert p_mass from fs to ps

                val = atof(tmp[2]);
                control->Tau_P[1] = val * 1.e-3;   // convert p_mass from fs to ps

                val = atof(tmp[3]);
                control->Tau_P[2] = val * 1.e-3;   // convert p_mass from fs to ps
            }
        }
        else if ( strcmp(tmp[0], "pt_mass") == 0 )
        {
            val = atof(tmp[1]);
            control->Tau_PT = val * 1.e-3;  // convert pt_mass from fs to ps
        }
        else if ( strcmp(tmp[0], "compress") == 0 )
        {
            val = atof(tmp[1]);
            control->compressibility = val;
        }
        else if ( strcmp(tmp[0], "press_mode") == 0 )
        {
            val = atoi(tmp[1]);
            control->press_mode = val;
        }
        else if ( strcmp(tmp[0], "remove_CoM_vel") == 0 )
        {
            val = atoi(tmp[1]);
            control->remove_CoM_vel = val;
        }
        else if ( strcmp(tmp[0], "debug_level") == 0 )
        {
            ival = atoi(tmp[1]);
            out_control->debug_level = ival;
        }
        else if ( strcmp(tmp[0], "energy_update_freq") == 0 )
        {
            ival = atoi(tmp[1]);
            out_control->energy_update_freq = ival;
        }
        else if ( strcmp(tmp[0], "write_freq") == 0 )
        {
            ival = atoi(tmp[1]);
            out_control->write_steps = ival;
        }
        else if ( strcmp(tmp[0], "traj_compress") == 0 )
        {
            ival = atoi(tmp[1]);
            out_control->traj_compress = ival;

            if ( out_control->traj_compress )
                out_control->write = (int (*)(FILE *, const char *, ...)) gzprintf;
            else out_control->write = fprintf;
        }
        else if ( strcmp(tmp[0], "traj_format") == 0 )
        {
            ival = atoi(tmp[1]);
            out_control->traj_format = ival;

            if ( out_control->traj_format == 0 )
            {
                out_control->write_header =
                    (int (*)( reax_system*, control_params*,
                              static_storage*, void* )) Write_Custom_Header;
                out_control->append_traj_frame =
                    (int (*)(reax_system*, control_params*, simulation_data*,
                             static_storage*, list **, void*)) Append_Custom_Frame;
            }
            else if ( out_control->traj_format == 1 )
            {
                out_control->write_header =
                    (int (*)( reax_system*, control_params*,
                              static_storage*, void* )) Write_xyz_Header;
                out_control->append_traj_frame =
                    (int (*)( reax_system*,  control_params*, simulation_data*,
                              static_storage*, list **, void* )) Append_xyz_Frame;
            }
        }
        else if ( strcmp(tmp[0], "traj_title") == 0 )
        {
            strcpy( out_control->traj_title, tmp[1] );
        }
        else if ( strcmp(tmp[0], "atom_info") == 0 )
        {
            ival = atoi(tmp[1]);
            out_control->atom_format += ival * 4;
        }
        else if ( strcmp(tmp[0], "atom_velocities") == 0 )
        {
            ival = atoi(tmp[1]);
            out_control->atom_format += ival * 2;
        }
        else if ( strcmp(tmp[0], "atom_forces") == 0 )
        {
            ival = atoi(tmp[1]);
            out_control->atom_format += ival * 1;
        }
        else if ( strcmp(tmp[0], "bond_info") == 0 )
        {
            ival = atoi(tmp[1]);
            out_control->bond_info = ival;
        }
        else if ( strcmp(tmp[0], "angle_info") == 0 )
        {
            ival = atoi(tmp[1]);
            out_control->angle_info = ival;
        }
        else if ( strcmp(tmp[0], "test_forces") == 0 )
        {
            ival = atoi(tmp[1]);
        }
        else if ( strcmp(tmp[0], "molec_anal") == 0 )
        {
            ival = atoi(tmp[1]);
            control->molec_anal = ival;
        }
        else if ( strcmp(tmp[0], "freq_molec_anal") == 0 )
        {
            ival = atoi(tmp[1]);
            control->freq_molec_anal = ival;
        }
        else if ( strcmp(tmp[0], "bond_graph_cutoff") == 0 )
        {
            val = atof(tmp[1]);
            control->bg_cut = val;
        }
        else if ( strcmp(tmp[0], "ignore") == 0 )
        {
            control->num_ignored = atoi(tmp[1]);
            for ( i = 0; i < control->num_ignored; ++i )
                control->ignore[atoi(tmp[i + 2])] = 1;
        }
        else if ( strcmp(tmp[0], "dipole_anal") == 0 )
        {
            ival = atoi(tmp[1]);
            control->dipole_anal = ival;
        }
        else if ( strcmp(tmp[0], "freq_dipole_anal") == 0 )
        {
            ival = atoi(tmp[1]);
            control->freq_dipole_anal = ival;
        }
        else if ( strcmp(tmp[0], "diffusion_coef") == 0 )
        {
            ival = atoi(tmp[1]);
            control->diffusion_coef = ival;
        }
        else if ( strcmp(tmp[0], "freq_diffusion_coef") == 0 )
        {
            ival = atoi(tmp[1]);
            control->freq_diffusion_coef = ival;
        }
        else if ( strcmp(tmp[0], "restrict_type") == 0 )
        {
            ival = atoi(tmp[1]);
            control->restrict_type = ival;
        }
        else
        {
            fprintf( stderr, "WARNING: unknown parameter %s\n", tmp[0] );
            exit( 15 );
        }
    }


    /* determine target T */
    if ( control->T_mode == 0 )
        control->T = control->T_final;
    else control->T = control->T_init;


    /* near neighbor and far neighbor cutoffs */
    control->bo_cut = 0.01 * system->reaxprm.gp.l[29];
    control->r_low  = system->reaxprm.gp.l[11];
    control->r_cut  = system->reaxprm.gp.l[12];
    control->vlist_cut += control->r_cut;

    system->g.cell_size = control->vlist_cut / 2.;
    for ( i = 0; i < 3; ++i )
        system->g.spread[i] = 2;


    /* Initialize Taper function */
    Init_Taper( control );


    /* free memory allocations at the top */
    for ( i = 0; i < MAX_TOKENS; i++ )
        free( tmp[i] );
    free( tmp );
    free( s );

#if defined(DEBUG_FOCUS)
    fprintf( stderr,
             "en=%d steps=%d dt=%.5f opt=%d T=%.5f P=%.5f %.5f %.5f\n",
             control->ensemble, control->nsteps, control->dt, control->tabulate,
             control->T, control->P[0], control->P[1], control->P[2] );

    fprintf(stderr, "control file read\n" );
#endif
    return 0;
}