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

#if (defined(HAVE_CONFIG_H) && !defined(__CONFIG_H_))
  #define __CONFIG_H_
  #include "../../common/include/config.h"
#endif

#if defined(PURE_REAX)
  #include "ffield.h"

  #include "tool_box.h"
#elif defined(LAMMPS_REAX)
  #include "reax_ffield.h"

  #include "reax_tool_box.h"
#endif


void Read_Force_Field_File( const char * const ffield_file, reax_interaction * const reax,
        reax_system * const system, control_params * const control )
{
    FILE *fp;
    char *s;
    char **tmp;
    char *tor_flag;
    int c, i, j, k, l, m, n, o, p, cnt;
    real val;
    int __N;
    int index1, index2;

    /* open force field file */
    fp = sfopen( ffield_file, "r", "Read_Force_Field::fp" );

    s = smalloc( sizeof(char) * MAX_LINE, "Read_Force_Field::s" );
    tmp = smalloc( sizeof(char *) * MAX_TOKENS, "Read_Force_Field::tmp");
    for (i = 0; i < MAX_TOKENS; i++)
    {
        tmp[i] = smalloc( sizeof(char) * MAX_TOKEN_LEN, "Read_Force_Field::tmp[i]" );
    }

    /* reading first header comment */
    fgets( s, MAX_LINE, fp );

    /* line 2 is number of global parameters */
    fgets( s, MAX_LINE, fp );
    c = Tokenize( s, &tmp, MAX_TOKEN_LEN );

    /* reading the number of global parameters */
    if ( c > 0 )
    {
        n = atoi(tmp[0]);
    }
    else
    {
        n = 0;
    }

    if ( n < 1 )
    {
        fprintf( stderr, "[WARNING] p%d: number of globals in ffield file is 0!\n",
              system->my_rank );
        return;
    }

    reax->gp.n_global = n;
    reax->gp.l = smalloc( sizeof(real) * n, "Read_Force_Field::reax->gp.l" );

    /* see reax_types.h for mapping between l[i] and the lambdas used in ff */
    for (i = 0; i < n; i++)
    {
        fgets( s, MAX_LINE, fp );
        c = Tokenize( s, &tmp, MAX_TOKEN_LEN );

        if ( c > 0 )
        {
            val = (real) atof(tmp[0]);
            reax->gp.l[i] = val;
        }
    }

    control->bo_cut = 0.01 * reax->gp.l[29];
    control->nonb_low  = reax->gp.l[11];
    control->nonb_cut  = reax->gp.l[12];

    /* next line is number of atom types and some comments */
    fgets( s, MAX_LINE, fp );
    c = Tokenize( s, &tmp, MAX_TOKEN_LEN );
    if ( c > 0 )
    {
        reax->num_atom_types = atoi(tmp[0]);
    }

    /* 3 lines of comments */
    fgets( s, MAX_LINE, fp );
    fgets( s, MAX_LINE, fp );
    fgets( s, MAX_LINE, fp );

    /* Allocating structures in reax_interaction */
    __N = reax->num_atom_types;

    reax->sbp = scalloc( reax->num_atom_types, sizeof(single_body_parameters),
                "Read_Force_Field::reax->sbp" );

    reax->tbp = scalloc( POW(reax->num_atom_types, 2.0), sizeof(two_body_parameters),
              "Read_Force_Field::reax->tbp" );

    reax->thbp = scalloc( POW(reax->num_atom_types, 3.0), sizeof(three_body_header),
              "Read_Force_Field::reax->thbp" );

    reax->hbp = scalloc( POW(reax->num_atom_types, 3.0), sizeof(hbond_parameters),
              "Read_Force_Field::reax->hbp" );

    reax->fbp = scalloc( POW(reax->num_atom_types, 4.0), sizeof(four_body_header),
              "Read_Force_Field::reax->fbp" );

    tor_flag = scalloc( POW(reax->num_atom_types, 4.0), sizeof(char),
           "Read_Force_Field::tor_flag" );

    /* vdWaals type:
     * 1: Shielded Morse, no inner-wall
     * 2: inner wall, no shielding
     * 3: inner wall+shielding */
    reax->gp.vdw_type = 0;

    /* reading single atom parameters */
    /* there are 4 lines of each single atom parameters in ff files. these
     * parameters later determine some of the pair and triplet parameters using
     * combination rules. */
    for ( i = 0; i < reax->num_atom_types; i++ )
    {
        /* line one */
        fgets( s, MAX_LINE, fp );
        c = Tokenize( s, &tmp, MAX_TOKEN_LEN );

        /* strlen safe here as tmp[0] is NULL-terminated in Tokenize */
        for ( j = 0; j < (int) strlen(tmp[0]); ++j )
        {
            if ( c > 0 )
            {
                reax->sbp[i].name[j] = toupper( tmp[0][j] );
                --c;
            }
        }

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: Atom Name in the force field : %s \n",
                system->my_rank, reax->sbp[i].name );
#endif

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
        reax->sbp[i].valency_e = val;
        reax->sbp[i].nlp_opt = 0.5 * (reax->sbp[i].valency_e - reax->sbp[i].valency);

        /* line two */
        fgets( s, MAX_LINE, fp );
        c = Tokenize( s, &tmp, MAX_TOKEN_LEN );

        if ( c >= 8 )
        {
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
             * which type of atoms participate in h-bonds.
             * 1 is for H - 2 for O, N, S - 0 for all others.*/
            val = atof(tmp[7]);
            /* 0.1 is to avoid from truncating down! */
            reax->sbp[i].p_hbond = (int)(val + 0.1);
        }

        /* line 3 */
        fgets( s, MAX_LINE, fp );
        c = Tokenize( s, &tmp, MAX_TOKEN_LEN );

        if ( c >= 8 )
        {
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
        }

        /* line 4  */
        fgets( s, MAX_LINE, fp );
        c = Tokenize( s, &tmp, MAX_TOKEN_LEN );

        if ( c >= 8 )
        {
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
        }

        /* Inner-wall */
        if ( reax->sbp[i].rcore2 > 0.01 && reax->sbp[i].acore2 > 0.01 )
        {
            /* Shielding vdWaals */
            if ( reax->sbp[i].gamma_w > 0.5 )
            {
                if ( reax->gp.vdw_type != 0 && reax->gp.vdw_type != 3 )
                {
                    fprintf( stderr, "[WARNING] p%d: inconsistent vdWaals-parameters\n"
                            "Force field parameters for element %s\n"
                            "indicate inner wall+shielding, but earlier\n"
                            "atoms indicate different vdWaals-method.\n"
                            "This may cause division-by-zero errors.\n"
                            "Keeping vdWaals-setting for earlier atoms.\n",
                            system->my_rank, reax->sbp[i].name );
                }
                else
                {
                    reax->gp.vdw_type = 3;
#if defined(DEBUG_FOCUS)
                    fprintf( stderr, "p%d: vdWaals type for element %s: Shielding+inner-wall",
                            system->my_rank, reax->sbp[i].name );
#endif
                }
            }
            /* No shielding vdWaals parameters present */
            else
            {
                if ( reax->gp.vdw_type != 0 && reax->gp.vdw_type != 2 )
                {
                    fprintf( stderr, "[WARNING] p%d: inconsistent vdWaals-parameters\n",
                            system->my_rank );
                    fprintf( stderr, "    [INFO] Force field parameters for element %s\n", reax->sbp[i].name );
                    fprintf( stderr, "    [INFO] indicate inner wall without shielding, but earlier\n" );
                    fprintf( stderr, "    [INFO] atoms indicate different vdWaals-method.\n" );
                    fprintf( stderr, "    [INFO] This may cause division-by-zero errors.\n" );
                    fprintf( stderr, "    [INFO] Keeping vdWaals-setting for earlier atoms.\n" );
                }
                else
                {
                    reax->gp.vdw_type = 2;
#if defined(DEBUG_FOCUS)
                    fprintf( stderr, "p%d: vdWaals type for element%s: No Shielding,inner-wall",
                            system->my_rank, reax->sbp[i].name );
#endif
                }
            }
        }
        /* No Inner wall parameters present */
        else
        {
            /* Shielding vdWaals */
            if ( reax->sbp[i].gamma_w > 0.5 )
            {
                if ( reax->gp.vdw_type != 0 && reax->gp.vdw_type != 1 )
                    fprintf( stderr, "[WARNING] p%d: inconsistent vdWaals-parameters\n" \
                            "    [INFO] Force field parameters for element %s\n"        \
                            "    [INFO] indicate  shielding without inner wall, but earlier\n" \
                            "    [INFO] atoms indicate different vdWaals-method.\n"     \
                            "    [INFO] This may cause division-by-zero errors.\n"      \
                            "    [INFO] Keeping vdWaals-setting for earlier atoms.\n",
                            system->my_rank, reax->sbp[i].name );
                else
                {
                    reax->gp.vdw_type = 1;
#if defined(DEBUG_FOCUS)
                    fprintf( stderr, "p%d, vdWaals type for element%s: Shielding,no inner-wall",
                            system->my_rank, reax->sbp[i].name );
#endif
                }
            }
            else
            {
                fprintf( stderr, "[ERROR] p%d: inconsistent vdWaals-parameters\n" \
                         "    [INFO] No shielding or inner-wall set for element %s\n",
                         system->my_rank, reax->sbp[i].name );
                MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
            }
        }
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: vdWaals type: %d\n", system->my_rank, reax->gp.vdw_type );
#endif

    /* Equate vval3 to valf for first-row elements (25/10/2004) */
    for ( i = 0; i < reax->num_atom_types; i++ )
    {
        if ( reax->sbp[i].mass < 21 &&
                reax->sbp[i].valency_val != reax->sbp[i].valency_boc )
        {
            fprintf( stderr, "[WARNING] p%d: changed valency_val to valency_boc for atom type %s\n",
                    system->my_rank, reax->sbp[i].name );
            reax->sbp[i].valency_val = reax->sbp[i].valency_boc;
        }
    }

    /* next line is number of two body combination and some comments */
    fgets( s, MAX_LINE, fp );
    c = Tokenize( s, &tmp, MAX_TOKEN_LEN );
    l = atoi( tmp[0] );

    /* a line of comments */
    fgets( s, MAX_LINE, fp );

    for ( i = 0; i < l; i++ )
    {
        /* line 1 */
        fgets( s, MAX_LINE, fp );
        c = Tokenize(s, &tmp, MAX_TOKEN_LEN);

        j = atoi(tmp[0]) - 1;
        k = atoi(tmp[1]) - 1;
        index1 = j * __N + k;
        index2 = k * __N + j;

        if ( j < reax->num_atom_types && k < reax->num_atom_types )
        {
            val = atof(tmp[2]);
            reax->tbp[ index1 ].De_s = val;
            reax->tbp[ index2 ].De_s = val;
            val = atof(tmp[3]);
            reax->tbp[ index1 ].De_p = val;
            reax->tbp[ index2 ].De_p = val;
            val = atof(tmp[4]);
            reax->tbp[ index1 ].De_pp = val;
            reax->tbp[ index2 ].De_pp = val;
            val = atof(tmp[5]);
            reax->tbp[ index1 ].p_be1 = val;
            reax->tbp[ index2 ].p_be1 = val;
            val = atof(tmp[6]);
            reax->tbp[ index1 ].p_bo5 = val;
            reax->tbp[ index2 ].p_bo5 = val;
            val = atof(tmp[7]);
            reax->tbp[ index1 ].v13cor = val;
            reax->tbp[ index2 ].v13cor = val;

            val = atof(tmp[8]);
            reax->tbp[ index1 ].p_bo6 = val;
            reax->tbp[ index2 ].p_bo6 = val;
            val = atof(tmp[9]);
            reax->tbp[ index1 ].p_ovun1 = val;
            reax->tbp[ index2 ].p_ovun1 = val;

            /* line 2 */
            fgets( s, MAX_LINE, fp );
            c = Tokenize( s, &tmp, MAX_TOKEN_LEN );

            val = atof(tmp[0]);
            reax->tbp[ index1 ].p_be2 = val;
            reax->tbp[ index2 ].p_be2 = val;
            val = atof(tmp[1]);
            reax->tbp[ index1 ].p_bo3 = val;
            reax->tbp[ index2 ].p_bo3 = val;
            val = atof(tmp[2]);
            reax->tbp[ index1 ].p_bo4 = val;
            reax->tbp[ index2 ].p_bo4 = val;
            val = atof(tmp[3]);

            val = atof(tmp[4]);
            reax->tbp[ index1 ].p_bo1 = val;
            reax->tbp[ index2 ].p_bo1 = val;
            val = atof(tmp[5]);
            reax->tbp[ index1 ].p_bo2 = val;
            reax->tbp[ index2 ].p_bo2 = val;
            val = atof(tmp[6]);
            reax->tbp[ index1 ].ovc = val;
            reax->tbp[ index2 ].ovc = val;

            val = atof(tmp[7]);
        }
    }

    /* calculating combination rules and filling up remaining fields. */
    for ( i = 0; i < reax->num_atom_types; i++ )
    {
        for ( j = i; j < reax->num_atom_types; j++ )
        {
            index1 = i * __N + j;
            index2 = j * __N + i;

            reax->tbp[index1].r_s = 0.5 * (reax->sbp[i].r_s + reax->sbp[j].r_s);
            reax->tbp[index2].r_s = 0.5 * (reax->sbp[j].r_s + reax->sbp[i].r_s);

            reax->tbp[index1].r_p = 0.5 * (reax->sbp[i].r_pi + reax->sbp[j].r_pi);
            reax->tbp[index2].r_p = 0.5 * (reax->sbp[j].r_pi + reax->sbp[i].r_pi);

            reax->tbp[index1].r_pp = 0.5 * (reax->sbp[i].r_pi_pi + reax->sbp[j].r_pi_pi);
            reax->tbp[index2].r_pp = 0.5 * (reax->sbp[j].r_pi_pi + reax->sbp[i].r_pi_pi);

            reax->tbp[index1].p_boc3 = SQRT( reax->sbp[i].b_o_132 * reax->sbp[j].b_o_132 );
            reax->tbp[index2].p_boc3 = SQRT( reax->sbp[j].b_o_132 * reax->sbp[i].b_o_132 );

            reax->tbp[index1].p_boc4 = SQRT( reax->sbp[i].b_o_131 * reax->sbp[j].b_o_131 );
            reax->tbp[index2].p_boc4 = SQRT( reax->sbp[j].b_o_131 * reax->sbp[i].b_o_131 );

            reax->tbp[index1].p_boc5 = SQRT( reax->sbp[i].b_o_133 * reax->sbp[j].b_o_133 );
            reax->tbp[index2].p_boc5 = SQRT( reax->sbp[j].b_o_133 * reax->sbp[i].b_o_133 );

            reax->tbp[index1].D = SQRT( reax->sbp[i].epsilon * reax->sbp[j].epsilon );
            reax->tbp[index2].D = SQRT( reax->sbp[j].epsilon * reax->sbp[i].epsilon );

            reax->tbp[index1].alpha = SQRT( reax->sbp[i].alpha * reax->sbp[j].alpha);
            reax->tbp[index2].alpha = SQRT( reax->sbp[j].alpha * reax->sbp[i].alpha);

            reax->tbp[index1].r_vdW = 2.0 * SQRT( reax->sbp[i].r_vdw * reax->sbp[j].r_vdw );
            reax->tbp[index2].r_vdW = 2.0 * SQRT( reax->sbp[j].r_vdw * reax->sbp[i].r_vdw );

            reax->tbp[index1].gamma_w = SQRT( reax->sbp[i].gamma_w * reax->sbp[j].gamma_w );
            reax->tbp[index2].gamma_w = SQRT( reax->sbp[j].gamma_w * reax->sbp[i].gamma_w );

            reax->tbp[index1].gamma = SQRT( reax->sbp[i].gamma * reax->sbp[j].gamma );
            reax->tbp[index2].gamma = SQRT( reax->sbp[j].gamma * reax->sbp[i].gamma );

            /* additions for additional vdWaals interaction types - inner core */
            reax->tbp[index1].rcore = SQRT( reax->sbp[i].rcore2 * reax->sbp[j].rcore2 );
            reax->tbp[index2].rcore = SQRT( reax->sbp[j].rcore2 * reax->sbp[i].rcore2 );

            reax->tbp[index1].ecore = SQRT( reax->sbp[i].ecore2 * reax->sbp[j].ecore2 );
            reax->tbp[index2].ecore = SQRT( reax->sbp[j].ecore2 * reax->sbp[i].ecore2 );

            reax->tbp[index1].acore = SQRT( reax->sbp[i].acore2 * reax->sbp[j].acore2 );
            reax->tbp[index2].acore = SQRT( reax->sbp[j].acore2 * reax->sbp[i].acore2 );
        }
    }

    /* next line is number of two body offdiagonal combinations and comments */
    /* these are two body offdiagonal terms that are different from the
       combination rules defined above */
    fgets( s, MAX_LINE, fp );
    c = Tokenize( s, &tmp, MAX_TOKEN_LEN );
    l = atoi(tmp[0]);

    for ( i = 0; i < l; i++ )
    {
        fgets( s, MAX_LINE, fp );
        c = Tokenize( s, &tmp, MAX_TOKEN_LEN );

        j = atoi(tmp[0]) - 1;
        k = atoi(tmp[1]) - 1;

        index1 = j * __N + k;
        index2 = k * __N + j;

        if ( j < reax->num_atom_types && k < reax->num_atom_types )
        {
            val = atof(tmp[2]);
            if ( val > 0.0 )
            {
                reax->tbp[index1].D = val;
                reax->tbp[index2].D = val;
            }

            val = atof(tmp[3]);
            if ( val > 0.0 )
            {
                reax->tbp[index1].r_vdW = 2 * val;
                reax->tbp[index2].r_vdW = 2 * val;
            }

            val = atof(tmp[4]);
            if ( val > 0.0 )
            {
                reax->tbp[index1].alpha = val;
                reax->tbp[index2].alpha = val;
            }

            val = atof(tmp[5]);
            if ( val > 0.0 )
            {
                reax->tbp[index1].r_s = val;
                reax->tbp[index2].r_s = val;
            }

            val = atof(tmp[6]);
            if ( val > 0.0 )
            {
                reax->tbp[index1].r_p = val;
                reax->tbp[index2].r_p = val;
            }

            val = atof(tmp[7]);
            if ( val > 0.0 )
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
    {
        for ( j = 0; j < reax->num_atom_types; ++j )
        {
            for ( k = 0; k < reax->num_atom_types; ++k )
            {
                reax->thbp[i * __N * __N + j * __N + k].cnt = 0;
            }
        }
    }

    /* next line is number of 3-body params and some comments */
    fgets( s, MAX_LINE, fp );
    c = Tokenize( s, &tmp, MAX_TOKEN_LEN );
    l = atoi( tmp[0] );

    for ( i = 0; i < l; i++ )
    {
        fgets( s, MAX_LINE, fp );
        c = Tokenize( s, &tmp, MAX_TOKEN_LEN );

        j = atoi(tmp[0]) - 1;
        k = atoi(tmp[1]) - 1;
        m = atoi(tmp[2]) - 1;
        index1 = j * __N * __N + k * __N + m;
        index2 = m * __N * __N + k * __N + j;

        if ( j < reax->num_atom_types && k < reax->num_atom_types
                && m < reax->num_atom_types )
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
     * correspond to any type of pair of atoms in 1 and 4
     * position. However, explicit X-Y-Z-W takes precedence over the
     * default description.
     * supports multi-well potentials (upto MAX_4BODY_PARAM in mytypes.h)
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
                    reax->fbp[i * __N * __N * __N + j * __N * __N + k * __N + m].cnt = 0;
                    tor_flag[i * __N * __N * __N + j * __N * __N + k * __N + m] = 0;

                }
            }
        }
    }

    /* next line is number of 4-body params and some comments */
    fgets( s, MAX_LINE, fp );
    c = Tokenize( s, &tmp, MAX_TOKEN_LEN );
    l = atoi( tmp[0] );

    for ( i = 0; i < l; i++ )
    {
        fgets( s, MAX_LINE, fp );
        c = Tokenize( s, &tmp, MAX_TOKEN_LEN );

        j = atoi(tmp[0]) - 1;
        k = atoi(tmp[1]) - 1;
        m = atoi(tmp[2]) - 1;
        n = atoi(tmp[3]) - 1;
        index1 = j * __N * __N * __N + k * __N * __N + m * __N + n;
        index2 = n * __N * __N * __N + m * __N * __N + k * __N + j;

        /* this means the entry is not in compact form */
        if ( j >= 0 && n >= 0 )
        {
            if ( j < reax->num_atom_types && k < reax->num_atom_types &&
                    m < reax->num_atom_types && n < reax->num_atom_types )
            {
                /* these flags ensure that this entry take precedence
                   over the compact form entries */
                tor_flag[index1] = 1;
                tor_flag[index2] = 1;

                reax->fbp[index1].cnt = 1;
                reax->fbp[index2].cnt = 1;

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
        /* This means the entry is of the form 0-X-Y-0 */
        else
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
    c = Tokenize( s, &tmp, MAX_TOKEN_LEN );
    l = atoi( tmp[0] );

    for ( i = 0; i < l; i++ )
    {
        fgets( s, MAX_LINE, fp );
        c = Tokenize( s, &tmp, MAX_TOKEN_LEN );

        j = atoi(tmp[0]) - 1;
        k = atoi(tmp[1]) - 1;
        m = atoi(tmp[2]) - 1;
        index1 = j * __N * __N + k * __N + m;

        if ( j < reax->num_atom_types && m < reax->num_atom_types )
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
        sfree( tmp[i], "Read_Force_Field::tmp[i]" );
    }
    sfree( tmp, "Read_Force_Field::tmp" );
    sfree( s, "Read_Force_Field::s" );
    sfree( tor_flag, "Read_Force_Field::tor_flag" );

    sfclose( fp, "Read_Force_Field::fp" );
}
