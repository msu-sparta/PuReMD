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


void Read_Force_Field( const char * const ffield_file,
        reax_system * const system, reax_interaction * const reax )
{
    char *s;
    char **tmp;
    bool *tor_flag;
    uint32_t i, j, k, l, m, n, o, p, cnt;
    int32_t jj, kk, mm, nn;
    real val, *gamma;
    FILE *fp;

    fp = sfopen( ffield_file, "r", __FILE__, __LINE__ );

    assert( fp != NULL );

    if ( fp != NULL ) {
        s = smalloc( sizeof(char) * MAX_LINE, __FILE__, __LINE__ );
        tmp = smalloc( sizeof(char *) * MAX_TOKENS, __FILE__, __LINE__ );
        for ( i = 0; i < MAX_TOKENS; i++ ) {
            tmp[i] = smalloc( sizeof(char) * MAX_TOKEN_LEN, __FILE__, __LINE__ );
        }

        /* reading first header comment */
        if ( fgets( s, MAX_LINE, fp ) == NULL ) {
            fprintf( stderr, "[ERROR] reading force field failed\n" \
                     "  [INFO] reading first header comment\n" );
            exit( INVALID_INPUT );
        }

        /* line 2 is number of global parameters */
        if ( fgets( s, MAX_LINE, fp ) == NULL ) {
            fprintf( stderr, "[ERROR] reading force field failed\n" \
                     "  [INFO] reading number of global parameters\n" );
            exit( INVALID_INPUT );
        }
        Tokenize( s, &tmp, MAX_TOKEN_LEN );

        /* reading the number of global parameters */
        n = sstrtoul( tmp[0], __FILE__, __LINE__ );
        if ( n < 1 ) {
            fprintf( stderr, "[WARNING] number of globals in ffield file is 0!\n" );
            return;
        }

        if ( system->ffield_params_allocated == FALSE ) {
            reax->gp.l = smalloc( sizeof(real) * n, __FILE__, __LINE__ );
            reax->gp.max_n_global = n;
        } else if ( reax->gp.max_n_global < n ) {
            reax->gp.l = srealloc( reax->gp.l, sizeof(real) * n, __FILE__, __LINE__ );
            reax->gp.max_n_global = n;
        }
        reax->gp.n_global = n;

        /* see reax_types.h for mapping between l[i] and the lambdas used in ff */
        for ( i = 0; i < n; i++ ) {
            if ( fgets( s, MAX_LINE, fp ) == NULL ) {
                fprintf( stderr, "[ERROR] reading force field failed\n" \
                         "  [INFO] reading global parameters (entry %d)\n", i );
                exit( INVALID_INPUT );
            }
            Tokenize( s, &tmp, MAX_TOKEN_LEN );

            val = (real) sstrtod( tmp[0], __FILE__, __LINE__ );

            reax->gp.l[i] = val;
        }

        /* next line is number of atom types and some comments */
        if ( fgets( s, MAX_LINE, fp ) == NULL ) {
            fprintf( stderr, "[ERROR] reading force field failed\n" \
                     "  [INFO] reading number of single body parameters\n" );
            exit( INVALID_INPUT );
        }
        Tokenize( s, &tmp, MAX_TOKEN_LEN );
        n = sstrtoul( tmp[0], __FILE__, __LINE__ );

        /* 3 lines of comments */
        if ( fgets( s, MAX_LINE, fp ) == NULL
                || fgets( s, MAX_LINE, fp ) == NULL
                || fgets( s, MAX_LINE, fp ) == NULL ) {
            fprintf( stderr, "[ERROR] reading force field failed\n" \
                     "  [INFO] reading single body comments\n" );
            exit( INVALID_INPUT );
        }

        if ( system->ffield_params_allocated == FALSE ) {
            system->ffield_params_allocated = TRUE;

            reax->sbp = scalloc( n, sizeof(single_body_parameters), __FILE__, __LINE__ );
            reax->tbp = scalloc( SQR(n), sizeof(two_body_parameters), __FILE__, __LINE__ );
            reax->thbp = scalloc( CUBE(n), sizeof(three_body_header), __FILE__, __LINE__ );
            reax->hbp = scalloc( CUBE(n), sizeof(hbond_parameters), __FILE__, __LINE__ );
            reax->fbp = scalloc( FOURTH(n), sizeof(four_body_header), __FILE__, __LINE__ );

            reax->max_num_atom_types = n;
        } else if ( reax->max_num_atom_types < n ) {
            sfree( reax->sbp, __FILE__, __LINE__ );
            sfree( reax->tbp, __FILE__, __LINE__ );
            sfree( reax->thbp, __FILE__, __LINE__ );
            sfree( reax->hbp, __FILE__, __LINE__ );
            sfree( reax->fbp, __FILE__, __LINE__ );

            reax->sbp = scalloc( n, sizeof(single_body_parameters), __FILE__, __LINE__ );
            reax->tbp = scalloc( SQR(n), sizeof(two_body_parameters), __FILE__, __LINE__ );
            reax->thbp = scalloc( CUBE(n), sizeof(three_body_header), __FILE__, __LINE__ );
            reax->hbp = scalloc( CUBE(n), sizeof(hbond_parameters), __FILE__, __LINE__ );
            reax->fbp = scalloc( FOURTH(n), sizeof(four_body_header), __FILE__, __LINE__ );

            reax->max_num_atom_types = n;
        }

        tor_flag = smalloc( FOURTH(n) * sizeof(bool), __FILE__, __LINE__ );

        reax->num_atom_types = n;
        gamma = smalloc( n * sizeof(real), __FILE__, __LINE__ );

        // vdWaals type: 1: Shielded Morse, no inner-wall
        //               2: inner wall, no shielding
        //               3: inner wall+shielding
        reax->gp.vdw_type = 0;

        /* reading single atom parameters:
         * there are 4 lines of each single atom parameters in ff files. these
         * parameters later determine some of the pair and triplet parameters using
         * combination rules. */
        for ( i = 0; i < reax->num_atom_types; i++ ) {
            /* line one */
            if ( fgets( s, MAX_LINE, fp ) == NULL ) {
                fprintf( stderr, "[ERROR] reading force field failed\n" \
                         "  [INFO] reading single body parameters (entry %d, line 1)\n", i );
                exit( INVALID_INPUT );
            }
            Tokenize( s, &tmp, MAX_TOKEN_LEN );

            strncpy( reax->sbp[i].name, tmp[0], sizeof(reax->sbp[i].name) - 1 );
            reax->sbp[i].name[sizeof(reax->sbp[i].name) - 1] = '\0';
            for ( j = 0; j < strlen( reax->sbp[i].name ); ++j ) {
                reax->sbp[i].name[j] = toupper( reax->sbp[i].name[j] );
            }

            for ( j = 0; j < i; j++ ) {
                if ( strncmp( reax->sbp[i].name, reax->sbp[j].name, sizeof(reax->sbp[i].name) - 1 ) == 0 ) {
                    fprintf( stderr, "[ERROR] reading force field failed\n" \
                             "  [INFO] duplicate single body parameters (entries %d and %d)\n", j + 1, i + 1 );
                    exit( INVALID_INPUT );
                }
            }

            val = sstrtod( tmp[1], __FILE__, __LINE__ );
            reax->sbp[i].r_s = val;
            val = sstrtod( tmp[2], __FILE__, __LINE__ );
            reax->sbp[i].valency = val;
            val = sstrtod( tmp[3], __FILE__, __LINE__ );
            reax->sbp[i].mass = val;
            val = sstrtod( tmp[4], __FILE__, __LINE__ );
            reax->sbp[i].r_vdw = val;
            val = sstrtod( tmp[5], __FILE__, __LINE__ );
            reax->sbp[i].epsilon = val;
            val = sstrtod( tmp[6], __FILE__, __LINE__ );
            gamma[i] = val;
            val = sstrtod( tmp[7], __FILE__, __LINE__ );
            reax->sbp[i].r_pi = val;
            val = sstrtod( tmp[8], __FILE__, __LINE__ );
            reax->sbp[i].valency_e  = val;
            reax->sbp[i].nlp_opt = 0.5 * (reax->sbp[i].valency_e - reax->sbp[i].valency);

            /* line two */
            if ( fgets( s, MAX_LINE, fp ) == NULL ) {
                fprintf( stderr, "[ERROR] reading force field failed\n" \
                         "  [INFO] reading single body parameters (entry %d, line 2)\n", i );
                exit( INVALID_INPUT );
            }
            Tokenize( s, &tmp, MAX_TOKEN_LEN );

            val = sstrtod( tmp[0], __FILE__, __LINE__ );
            reax->sbp[i].alpha = val;
            val = sstrtod( tmp[1], __FILE__, __LINE__ );
            reax->sbp[i].gamma_w = val;
            val = sstrtod( tmp[2], __FILE__, __LINE__ );
            reax->sbp[i].valency_boc = val;
            val = sstrtod( tmp[3], __FILE__, __LINE__ );
            reax->sbp[i].p_ovun5 = val;
            val = sstrtod( tmp[4], __FILE__, __LINE__ );
            val = sstrtod( tmp[5], __FILE__, __LINE__ );
            reax->sbp[i].chi = val;
            val = sstrtod( tmp[6], __FILE__, __LINE__ );
            reax->sbp[i].eta = 2.0 * val;
            /* this is the parameter that is used to determine
             * which type of atoms participate in h-bonds.
             * 1 is for H - 2 for O, N, S - 0 for all others.*/
            val = sstrtod( tmp[7], __FILE__, __LINE__ );
            reax->sbp[i].p_hbond = (int32_t) (val + 0.1);
            //0.1 is to avoid from truncating down!

            /* line 3 */
            if ( fgets( s, MAX_LINE, fp ) == NULL ) {
                fprintf( stderr, "[ERROR] reading force field failed\n" \
                         "  [INFO] reading single body parameters (entry %d, line 3)\n", i );
                exit( INVALID_INPUT );
            }
            Tokenize( s, &tmp, MAX_TOKEN_LEN );

            val = sstrtod( tmp[0], __FILE__, __LINE__ );
            reax->sbp[i].r_pi_pi = val;
            val = sstrtod( tmp[1], __FILE__, __LINE__ );
            reax->sbp[i].p_lp2 = val;
            val = sstrtod( tmp[2], __FILE__, __LINE__ );
            val = sstrtod( tmp[3], __FILE__, __LINE__ );
            reax->sbp[i].b_o_131 = val;
            val = sstrtod( tmp[4], __FILE__, __LINE__ );
            reax->sbp[i].b_o_132 = val;
            val = sstrtod( tmp[5], __FILE__, __LINE__ );
            reax->sbp[i].b_o_133 = val;
            val = sstrtod( tmp[6], __FILE__, __LINE__ );
            reax->sbp[i].b_s_acks2 = val;
            val = sstrtod( tmp[7], __FILE__, __LINE__ );

            /* line 4  */
            if ( fgets( s, MAX_LINE, fp ) == NULL ) {
                fprintf( stderr, "[ERROR] reading force field failed\n" \
                         "  [INFO] reading single body parameters (entry %d, line 4)\n", i );
                exit( INVALID_INPUT );
            }
            Tokenize( s, &tmp, MAX_TOKEN_LEN );

            val = sstrtod( tmp[0], __FILE__, __LINE__ );
            reax->sbp[i].p_ovun2 = val;
            val = sstrtod( tmp[1], __FILE__, __LINE__ );
            reax->sbp[i].p_val3 = val;
            val = sstrtod( tmp[2], __FILE__, __LINE__ );
            val = sstrtod( tmp[3], __FILE__, __LINE__ );
            reax->sbp[i].valency_val = val;
            val = sstrtod( tmp[4], __FILE__, __LINE__ );
            reax->sbp[i].p_val5 = val;
            val = sstrtod( tmp[5], __FILE__, __LINE__ );
            reax->sbp[i].rcore2 = val;
            val = sstrtod( tmp[6], __FILE__, __LINE__ );
            reax->sbp[i].ecore2 = val;
            val = sstrtod( tmp[7], __FILE__, __LINE__ );
            reax->sbp[i].acore2 = val;

            /* inner-wall parameters present */
            if ( reax->sbp[i].rcore2 > 0.01 && reax->sbp[i].acore2 > 0.01 ) {
                /* shielding vdWaals parameters present */
                if ( reax->sbp[i].gamma_w > 0.5 ) {
                    if ( reax->gp.vdw_type != 0 && reax->gp.vdw_type != 3 )
                        fprintf( stderr, "[WARNING] Inconsistent vdWaals-parameters!\n" \
                                 "[INFO] Force field parameters for element %s\n"        \
                                 "[INFO] indicate inner wall+shielding, but earlier\n"   \
                                 "[INFO] atoms indicate different vdWaals-method.\n"     \
                                 "[INFO] This may cause division-by-zero errors.\n"      \
                                 "[INFO] Keeping vdWaals-setting for earlier atoms.\n",
                                 reax->sbp[i].name );
                    else {
                        reax->gp.vdw_type = 3;

#if defined(DEBUG_FOCUS)
                        fprintf( stderr, "[INFO] vdWaals type for element %s: Shielding+inner-wall\n",
                                 reax->sbp[i].name );
#endif
                    }
                }
                /* no shielding vdWaals parameters present */
                else {
                    if ( reax->gp.vdw_type != 0 && reax->gp.vdw_type != 2 ) {
                        fprintf( stderr, "[WARNING] Inconsistent vdWaals-parameters!\n" \
                                 "[INFO] Force field parameters for element %s\n"        \
                                 "[INFO] indicate inner wall without shielding, but earlier\n" \
                                 "[INFO] atoms indicate different vdWaals-method.\n"     \
                                 "[INFO] This may cause division-by-zero errors.\n"      \
                                 "[INFO] Keeping vdWaals-setting for earlier atoms.\n",
                                 reax->sbp[i].name );
                    } else {
                        reax->gp.vdw_type = 2;

#if defined(DEBUG_FOCUS)
                        fprintf( stderr, "[INFO] vdWaals type for element%s: No Shielding,inner-wall\n",
                                 reax->sbp[i].name );
#endif
                    }
                }
            }
            /* no inner wall parameters present */
            else {
                /* shielding vdWaals parameters present */
                if ( reax->sbp[i].gamma_w > 0.5 ) {
                    if ( reax->gp.vdw_type != 0 && reax->gp.vdw_type != 1 )
                        fprintf( stderr, "[WARNING] Inconsistent vdWaals-parameters!\n" \
                                 "[INFO] Force field parameters for element %s\n"        \
                                 "[INFO] indicate shielding without inner wall, but earlier\n" \
                                 "[INFO] atoms indicate different vdWaals-method.\n"     \
                                 "[INFO] This may cause division-by-zero errors.\n"      \
                                 "[INFO] Keeping vdWaals-setting for earlier atoms.\n",
                                 reax->sbp[i].name );
                    else {
                        reax->gp.vdw_type = 1;

#if defined(DEBUG_FOCUS)
                        fprintf( stderr, "[INFO] vdWaals type for element%s: Shielding,no inner-wall\n",
                                 reax->sbp[i].name );
#endif
                    }
                } else {
                    fprintf( stderr, "[ERROR] Inconsistent vdWaals-parameters\n" \
                             "[INFO] No shielding or inner-wall set for element %s\n",
                             reax->sbp[i].name );
                    exit( INVALID_INPUT );
                }
            }
        }

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "[INFO] vdWaals type: %d\n", reax->gp.vdw_type );
#endif

        /* Equate vval3 to valf for first-row elements (25/10/2004) */
        for ( i = 0; i < reax->num_atom_types; i++ ) {
            if ( reax->sbp[i].mass < 21.0
                    && reax->sbp[i].valency_val != reax->sbp[i].valency_boc ) {
                fprintf( stderr, "[WARNING] changed valency_val to valency_boc for %s\n",
                         reax->sbp[i].name );
                reax->sbp[i].valency_val = reax->sbp[i].valency_boc;
            }
        }

        /* next line is number of two body combination and some comments */
        if ( fgets( s, MAX_LINE, fp ) == NULL ) {
            fprintf( stderr, "[ERROR] reading force field failed\n" \
                     "  [INFO] reading number of two body parameters\n" );
            exit( INVALID_INPUT );
        }
        Tokenize( s, &tmp, MAX_TOKEN_LEN );
        l = sstrtoul( tmp[0], __FILE__, __LINE__ );

        /* a line of comments */
        if ( fgets( s, MAX_LINE, fp ) == NULL ) {
            fprintf( stderr, "[ERROR] reading force field failed\n" \
                     "  [INFO] reading two body comments\n" );
            exit( INVALID_INPUT );
        }

        for ( i = 0; i < l; i++ ) {
            /* line 1 */
            if ( fgets( s, MAX_LINE, fp ) == NULL ) {
                fprintf( stderr, "[ERROR] reading force field failed\n" \
                         "  [INFO] reading two body parameters (entry %d, line 1)\n", i );
                exit( INVALID_INPUT );
            }
            Tokenize( s, &tmp, MAX_TOKEN_LEN );

            j = sstrtoul( tmp[0], __FILE__, __LINE__ ) - 1;
            k = sstrtoul( tmp[1], __FILE__, __LINE__ ) - 1;

            if ( j < reax->num_atom_types && k < reax->num_atom_types ) {
                val = sstrtod( tmp[2], __FILE__, __LINE__ );
                reax->tbp[IDX_TBP(j, k, n)].De_s = val;
                reax->tbp[IDX_TBP(k, j, n)].De_s = val;
                val = sstrtod( tmp[3], __FILE__, __LINE__ );
                reax->tbp[IDX_TBP(j, k, n)].De_p = val;
                reax->tbp[IDX_TBP(k, j, n)].De_p = val;
                val = sstrtod( tmp[4], __FILE__, __LINE__ );
                reax->tbp[IDX_TBP(j, k, n)].De_pp = val;
                reax->tbp[IDX_TBP(k, j, n)].De_pp = val;
                val = sstrtod( tmp[5], __FILE__, __LINE__ );
                reax->tbp[IDX_TBP(j, k, n)].p_be1 = val;
                reax->tbp[IDX_TBP(k, j, n)].p_be1 = val;
                val = sstrtod( tmp[6], __FILE__, __LINE__ );
                reax->tbp[IDX_TBP(j, k, n)].p_bo5 = val;
                reax->tbp[IDX_TBP(k, j, n)].p_bo5 = val;
                val = sstrtod( tmp[7], __FILE__, __LINE__ );
                reax->tbp[IDX_TBP(j, k, n)].v13cor = val;
                reax->tbp[IDX_TBP(k, j, n)].v13cor = val;

                val = sstrtod( tmp[8], __FILE__, __LINE__ );
                reax->tbp[IDX_TBP(j, k, n)].p_bo6 = val;
                reax->tbp[IDX_TBP(k, j, n)].p_bo6 = val;
                val = sstrtod( tmp[9], __FILE__, __LINE__ );
                reax->tbp[IDX_TBP(j, k, n)].p_ovun1 = val;
                reax->tbp[IDX_TBP(k, j, n)].p_ovun1 = val;

                /* line 2 */
                if ( fgets( s, MAX_LINE, fp ) == NULL ) {
                    fprintf( stderr, "[ERROR] reading force field failed\n" \
                             "  [INFO] reading two body parameters (entry %d, line 2)\n", i );
                    exit( INVALID_INPUT );
                }
                Tokenize( s, &tmp, MAX_TOKEN_LEN );

                val = sstrtod( tmp[0], __FILE__, __LINE__ );
                reax->tbp[IDX_TBP(j, k, n)].p_be2 = val;
                reax->tbp[IDX_TBP(k, j, n)].p_be2 = val;
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                reax->tbp[IDX_TBP(j, k, n)].p_bo3 = val;
                reax->tbp[IDX_TBP(k, j, n)].p_bo3 = val;
                val = sstrtod( tmp[2], __FILE__, __LINE__ );
                reax->tbp[IDX_TBP(j, k, n)].p_bo4 = val;
                reax->tbp[IDX_TBP(k, j, n)].p_bo4 = val;
                val = sstrtod( tmp[3], __FILE__, __LINE__ );

                val = sstrtod( tmp[4], __FILE__, __LINE__ );
                reax->tbp[IDX_TBP(j, k, n)].p_bo1 = val;
                reax->tbp[IDX_TBP(k, j, n)].p_bo1 = val;
                val = sstrtod( tmp[5], __FILE__, __LINE__ );
                reax->tbp[IDX_TBP(j, k, n)].p_bo2 = val;
                reax->tbp[IDX_TBP(k, j, n)].p_bo2 = val;
                val = sstrtod( tmp[6], __FILE__, __LINE__ );
                reax->tbp[IDX_TBP(j, k, n)].ovc = val;
                reax->tbp[IDX_TBP(k, j, n)].ovc = val;

                val = sstrtod( tmp[7], __FILE__, __LINE__ );
            }
        }

        /* calculating combination rules and filling up remaining fields. */
        for ( i = 0; i < reax->num_atom_types; i++ ) {
            for ( j = i; j < reax->num_atom_types; j++ ) {
                val = gamma[i] * gamma[j];

                reax->tbp[IDX_TBP(i, j, n)].r_s = 0.5 * (reax->sbp[i].r_s + reax->sbp[j].r_s);
                reax->tbp[IDX_TBP(j, i, n)].r_s = 0.5 * (reax->sbp[j].r_s + reax->sbp[i].r_s);

                reax->tbp[IDX_TBP(i, j, n)].r_p = 0.5 * (reax->sbp[i].r_pi + reax->sbp[j].r_pi);
                reax->tbp[IDX_TBP(j, i, n)].r_p = 0.5 * (reax->sbp[j].r_pi + reax->sbp[i].r_pi);

                reax->tbp[IDX_TBP(i, j, n)].r_pp = 0.5 * (reax->sbp[i].r_pi_pi + reax->sbp[j].r_pi_pi);
                reax->tbp[IDX_TBP(j, i, n)].r_pp = 0.5 * (reax->sbp[j].r_pi_pi + reax->sbp[i].r_pi_pi);

                reax->tbp[IDX_TBP(i, j, n)].p_boc3 = SQRT( reax->sbp[i].b_o_132 * reax->sbp[j].b_o_132);
                reax->tbp[IDX_TBP(j, i, n)].p_boc3 = SQRT( reax->sbp[j].b_o_132 * reax->sbp[i].b_o_132 );

                reax->tbp[IDX_TBP(i, j, n)].p_boc4 = SQRT( reax->sbp[i].b_o_131 * reax->sbp[j].b_o_131 );
                reax->tbp[IDX_TBP(j, i, n)].p_boc4 = SQRT( reax->sbp[j].b_o_131 * reax->sbp[i].b_o_131 );

                reax->tbp[IDX_TBP(i, j, n)].p_boc5 = SQRT( reax->sbp[i].b_o_133 * reax->sbp[j].b_o_133 );
                reax->tbp[IDX_TBP(j, i, n)].p_boc5 = SQRT( reax->sbp[j].b_o_133 * reax->sbp[i].b_o_133 );

                reax->tbp[IDX_TBP(i, j, n)].D = SQRT( reax->sbp[i].epsilon * reax->sbp[j].epsilon );
                reax->tbp[IDX_TBP(j, i, n)].D = SQRT( reax->sbp[j].epsilon * reax->sbp[i].epsilon );

                reax->tbp[IDX_TBP(i, j, n)].alpha = SQRT( reax->sbp[i].alpha * reax->sbp[j].alpha );
                reax->tbp[IDX_TBP(j, i, n)].alpha = SQRT( reax->sbp[j].alpha * reax->sbp[i].alpha );

                reax->tbp[IDX_TBP(i, j, n)].r_vdW = 2.0 * SQRT( reax->sbp[i].r_vdw * reax->sbp[j].r_vdw );
                reax->tbp[IDX_TBP(j, i, n)].r_vdW = 2.0 * SQRT( reax->sbp[j].r_vdw * reax->sbp[i].r_vdw );

                reax->tbp[IDX_TBP(i, j, n)].gamma_w = SQRT( reax->sbp[i].gamma_w * reax->sbp[j].gamma_w );
                reax->tbp[IDX_TBP(j, i, n)].gamma_w = SQRT( reax->sbp[j].gamma_w * reax->sbp[i].gamma_w );

                /* avoid division-by-zero when gamma_{i,j} is 0.0,
                 * which implies that electrostatic shielding is disabled */
                if ( FABS( val ) >= 1.0e-10 ) {
                    reax->tbp[IDX_TBP(i, j, n)].gamma = 1.0 / CUBE(SQRT(val));
                    reax->tbp[IDX_TBP(j, i, n)].gamma = 1.0 / CUBE(SQRT(val));
                } else {
                    reax->tbp[IDX_TBP(i, j, n)].gamma = 0.0;
                    reax->tbp[IDX_TBP(j, i, n)].gamma = 0.0;
                }

                reax->tbp[IDX_TBP(i, j, n)].acore = SQRT( reax->sbp[i].acore2 * reax->sbp[j].acore2 );
                reax->tbp[IDX_TBP(j, i, n)].acore = SQRT( reax->sbp[j].acore2 * reax->sbp[i].acore2 );

                reax->tbp[IDX_TBP(i, j, n)].ecore = SQRT( reax->sbp[i].ecore2 * reax->sbp[j].ecore2 );
                reax->tbp[IDX_TBP(j, i, n)].ecore = SQRT( reax->sbp[j].ecore2 * reax->sbp[i].ecore2 );

                reax->tbp[IDX_TBP(i, j, n)].rcore = SQRT( reax->sbp[i].rcore2 * reax->sbp[j].rcore2 );
                reax->tbp[IDX_TBP(j, i, n)].rcore = SQRT( reax->sbp[j].rcore2 * reax->sbp[i].rcore2 );
            }
        }

        /* next line is number of 2-body offdiagonal combinations and some comments */
        /* these are two body off-diagonal terms that are different from the
         * combination rules defined above */
        if ( fgets( s, MAX_LINE, fp ) == NULL ) {
            fprintf( stderr, "[ERROR] reading force field failed\n" \
                     "  [INFO] reading number of two body off-diagonal parameters\n" );
            exit( INVALID_INPUT );
        }
        Tokenize( s, &tmp, MAX_TOKEN_LEN );
        l = sstrtoul( tmp[0], __FILE__, __LINE__ );

        for ( i = 0; i < l; i++ ) {
            if ( fgets( s, MAX_LINE, fp ) == NULL ) {
                fprintf( stderr, "[ERROR] reading force field failed\n" \
                         "  [INFO] reading two body off-diagonal parameters (entry %d)\n", i );
                exit( INVALID_INPUT );
            }
            Tokenize( s, &tmp, MAX_TOKEN_LEN );

            j = sstrtoul( tmp[0], __FILE__, __LINE__ ) - 1;
            k = sstrtoul( tmp[1], __FILE__, __LINE__ ) - 1;

            if ( j < reax->num_atom_types && k < reax->num_atom_types ) {
                val = sstrtod( tmp[2], __FILE__, __LINE__ );
                if ( val > 0.0 ) {
                    reax->tbp[IDX_TBP(j, k, n)].D = val;
                    reax->tbp[IDX_TBP(k, j, n)].D = val;
                }

                val = sstrtod( tmp[3], __FILE__, __LINE__ );
                if ( val > 0.0 ) {
                    reax->tbp[IDX_TBP(j, k, n)].r_vdW = 2.0 * val;
                    reax->tbp[IDX_TBP(k, j, n)].r_vdW = 2.0 * val;
                }

                val = sstrtod( tmp[4], __FILE__, __LINE__ );
                if ( val > 0.0 ) {
                    reax->tbp[IDX_TBP(j, k, n)].alpha = val;
                    reax->tbp[IDX_TBP(k, j, n)].alpha = val;
                }

                val = sstrtod( tmp[5], __FILE__, __LINE__ );
                if ( val > 0.0 ) {
                    reax->tbp[IDX_TBP(j, k, n)].r_s = val;
                    reax->tbp[IDX_TBP(k, j, n)].r_s = val;
                }

                val = sstrtod( tmp[6], __FILE__, __LINE__ );
                if ( val > 0.0 ) {
                    reax->tbp[IDX_TBP(j, k, n)].r_p = val;
                    reax->tbp[IDX_TBP(k, j, n)].r_p = val;
                }

                val = sstrtod( tmp[7], __FILE__, __LINE__ );
                if ( val > 0.0 ) {
                    reax->tbp[IDX_TBP(j, k, n)].r_pp = val;
                    reax->tbp[IDX_TBP(k, j, n)].r_pp = val;
                }
            }
        }

        /* 3-body parameters -
         * supports multi-well potentials (upto MAX_3BODY_PARAM in reax_types.h) */
        /* clear entries first */
        for ( i = 0; i < reax->num_atom_types; ++i ) {
            for ( j = 0; j < reax->num_atom_types; ++j ) {
                for ( k = 0; k < reax->num_atom_types; ++k ) {
                    reax->thbp[IDX_THBP(i, j, k, n)].cnt = 0;
                }
            }
        }

        /* next line is number of 3-body params and some comments */
        if ( fgets( s, MAX_LINE, fp ) == NULL ) {
            fprintf( stderr, "[ERROR] reading force field failed\n" \
                     "  [INFO] reading number of three body parameters\n" );
            exit( INVALID_INPUT );
        }
        Tokenize( s, &tmp, MAX_TOKEN_LEN );
        l = sstrtoul( tmp[0], __FILE__, __LINE__ );

        for ( i = 0; i < l; i++ ) {
            if ( fgets( s, MAX_LINE, fp ) == NULL ) {
                fprintf( stderr, "[ERROR] reading force field failed\n" \
                         "  [INFO] reading three body parameters (entry %d)\n", i );
                exit( INVALID_INPUT );
            }
            Tokenize( s, &tmp, MAX_TOKEN_LEN );

            j = sstrtoul( tmp[0], __FILE__, __LINE__ ) - 1;
            k = sstrtoul( tmp[1], __FILE__, __LINE__ ) - 1;
            m = sstrtoul( tmp[2], __FILE__, __LINE__ ) - 1;

            if ( j < reax->num_atom_types
                    && k < reax->num_atom_types
                    && m < reax->num_atom_types ) {
                cnt = reax->thbp[IDX_THBP(j, k, m, n)].cnt;
                reax->thbp[IDX_THBP(j, k, m, n)].cnt++;
                reax->thbp[IDX_THBP(m, k, j, n)].cnt++;

                val = sstrtod( tmp[3], __FILE__, __LINE__ );
                reax->thbp[IDX_THBP(j, k, m, n)].prm[cnt].theta_00 = val;
                reax->thbp[IDX_THBP(m, k, j, n)].prm[cnt].theta_00 = val;

                val = sstrtod( tmp[4], __FILE__, __LINE__ );
                reax->thbp[IDX_THBP(j, k, m, n)].prm[cnt].p_val1 = val;
                reax->thbp[IDX_THBP(m, k, j, n)].prm[cnt].p_val1 = val;

                val = sstrtod( tmp[5], __FILE__, __LINE__ );
                reax->thbp[IDX_THBP(j, k, m, n)].prm[cnt].p_val2 = val;
                reax->thbp[IDX_THBP(m, k, j, n)].prm[cnt].p_val2 = val;

                val = sstrtod( tmp[6], __FILE__, __LINE__ );
                reax->thbp[IDX_THBP(j, k, m, n)].prm[cnt].p_coa1 = val;
                reax->thbp[IDX_THBP(m, k, j, n)].prm[cnt].p_coa1 = val;

                val = sstrtod( tmp[7], __FILE__, __LINE__ );
                reax->thbp[IDX_THBP(j, k, m, n)].prm[cnt].p_val7 = val;
                reax->thbp[IDX_THBP(m, k, j, n)].prm[cnt].p_val7 = val;

                val = sstrtod( tmp[8], __FILE__, __LINE__ );
                reax->thbp[IDX_THBP(j, k, m, n)].prm[cnt].p_pen1 = val;
                reax->thbp[IDX_THBP(m, k, j, n)].prm[cnt].p_pen1 = val;

                val = sstrtod( tmp[9], __FILE__, __LINE__ );
                reax->thbp[IDX_THBP(j, k, m, n)].prm[cnt].p_val4 = val;
                reax->thbp[IDX_THBP(m, k, j, n)].prm[cnt].p_val4 = val;
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
        for ( i = 0; i < reax->num_atom_types; ++i ) {
            for ( j = 0; j < reax->num_atom_types; ++j ) {
                for ( k = 0; k < reax->num_atom_types; ++k ) {
                    for ( m = 0; m < reax->num_atom_types; ++m ) {
                        reax->fbp[IDX_FBP(i, j, k, m, n)].cnt = 0;
                        tor_flag[IDX_FBP(i, j, k, m, n)] = FALSE;
                    }
                }
            }
        }

        /* next line is number of 4-body params and some comments */
        if ( fgets( s, MAX_LINE, fp ) == NULL ) {
            fprintf( stderr, "[ERROR] reading force field failed\n" \
                     "  [INFO] reading number of four body parameters\n" );
            exit( INVALID_INPUT );
        }
        Tokenize( s, &tmp, MAX_TOKEN_LEN );
        l = sstrtoul( tmp[0], __FILE__, __LINE__ );

        for ( i = 0; i < l; i++ ) {
            if ( fgets( s, MAX_LINE, fp ) == NULL ) {
                fprintf( stderr, "[ERROR] reading force field failed\n" \
                         "  [INFO] reading four body parameters (entry %d)\n", i );
                exit( INVALID_INPUT );
            }
            Tokenize( s, &tmp, MAX_TOKEN_LEN );

            jj = sstrtol( tmp[0], __FILE__, __LINE__ ) - 1;
            kk = sstrtol( tmp[1], __FILE__, __LINE__ ) - 1;
            mm = sstrtol( tmp[2], __FILE__, __LINE__ ) - 1;
            nn = sstrtol( tmp[3], __FILE__, __LINE__ ) - 1;

            /* this means the entry is not in compact form */
            if ( jj >= 0 && nn >= 0 ) {
                if ( (uint32_t) jj < reax->num_atom_types
                        && (uint32_t) kk < reax->num_atom_types
                        && (uint32_t) mm < reax->num_atom_types
                        && (uint32_t) nn < reax->num_atom_types ) {
                    /* these flags ensure that this entry take precedence
                     * over the compact form entries */
                    tor_flag[IDX_FBP(jj, kk, mm, nn, n)] = TRUE;
                    tor_flag[IDX_FBP(nn, mm, kk, jj, n)] = TRUE;

                    reax->fbp[IDX_FBP(jj, kk, mm, nn, n)].cnt = 1;
                    reax->fbp[IDX_FBP(nn, mm, kk, jj, n)].cnt = 1;

//                    cnt = reax->fbp[IDX_FBP(jj, kk, mm, nn, n)].cnt;
//                    reax->fbp[IDX_FBP(jj, kk, mm, nn, n)].cnt++;
//                    reax->fbp[IDX_FBP(nn, mm, kk, jj, n)].cnt++;

                    val = sstrtod( tmp[4], __FILE__, __LINE__ );
                    reax->fbp[IDX_FBP(jj, kk, mm, nn, n)].prm[0].V1 = val;
                    reax->fbp[IDX_FBP(nn, mm, kk, jj, n)].prm[0].V1 = val;

                    val = sstrtod( tmp[5], __FILE__, __LINE__ );
                    reax->fbp[IDX_FBP(jj, kk, mm, nn, n)].prm[0].V2 = val;
                    reax->fbp[IDX_FBP(nn, mm, kk, jj, n)].prm[0].V2 = val;

                    val = sstrtod( tmp[6], __FILE__, __LINE__ );
                    reax->fbp[IDX_FBP(jj, kk, mm, nn, n)].prm[0].V3 = val;
                    reax->fbp[IDX_FBP(nn, mm, kk, jj, n)].prm[0].V3 = val;

                    val = sstrtod( tmp[7], __FILE__, __LINE__ );
                    reax->fbp[IDX_FBP(jj, kk, mm, nn, n)].prm[0].p_tor1 = val;
                    reax->fbp[IDX_FBP(nn, mm, kk, jj, n)].prm[0].p_tor1 = val;

                    val = sstrtod( tmp[8], __FILE__, __LINE__ );
                    reax->fbp[IDX_FBP(jj, kk, mm, nn, n)].prm[0].p_cot1 = val;
                    reax->fbp[IDX_FBP(nn, mm, kk, jj, n)].prm[0].p_cot1 = val;
                }
            }
            /* This means the entry is of the form 0-X-Y-0 */
            else {
                if ( (uint32_t) kk < reax->num_atom_types && (uint32_t) mm < reax->num_atom_types ) {
                    for ( p = 0; p < reax->num_atom_types; p++ ) {
                        for ( o = 0; o < reax->num_atom_types; o++ ) {
                            reax->fbp[IDX_FBP(p, kk, mm, o, n)].cnt = 1;
                            reax->fbp[IDX_FBP(o, mm, kk, p, n)].cnt = 1;

//                            cnt = reax->fbp[IDX_FBP(p, kk, mm, o, n)].cnt;
//                            reax->fbp[IDX_FBP(p, kk, mm, o, n)].cnt++;
//                            reax->fbp[IDX_FBP(o, mm, kk, p, n)].cnt++;

                            if ( tor_flag[IDX_FBP(p, kk, mm, o, n)] == FALSE ) {
                                reax->fbp[IDX_FBP(p, kk, mm, o, n)].prm[0].V1 = sstrtod( tmp[4], __FILE__, __LINE__ );
                                reax->fbp[IDX_FBP(p, kk, mm, o, n)].prm[0].V2 = sstrtod( tmp[5], __FILE__, __LINE__ );
                                reax->fbp[IDX_FBP(p, kk, mm, o, n)].prm[0].V3 = sstrtod( tmp[6], __FILE__, __LINE__ );
                                reax->fbp[IDX_FBP(p, kk, mm, o, n)].prm[0].p_tor1 = sstrtod( tmp[7], __FILE__, __LINE__ );
                                reax->fbp[IDX_FBP(p, kk, mm, o, n)].prm[0].p_cot1 = sstrtod( tmp[8], __FILE__, __LINE__ );
                            }

                            if ( tor_flag[IDX_FBP(o, mm, kk, p, n)] == FALSE ) {
                                reax->fbp[IDX_FBP(o, mm, kk, p, n)].prm[0].V1 = sstrtod( tmp[4], __FILE__, __LINE__ );
                                reax->fbp[IDX_FBP(o, mm, kk, p, n)].prm[0].V2 = sstrtod( tmp[5], __FILE__, __LINE__ );
                                reax->fbp[IDX_FBP(o, mm, kk, p, n)].prm[0].V3 = sstrtod( tmp[6], __FILE__, __LINE__ );
                                reax->fbp[IDX_FBP(o, mm, kk, p, n)].prm[0].p_tor1 = sstrtod( tmp[7], __FILE__, __LINE__ );
                                reax->fbp[IDX_FBP(o, mm, kk, p, n)].prm[0].p_cot1 = sstrtod( tmp[8], __FILE__, __LINE__ );
                            }
                        }
                    }
                }
            }
        }

        /* next line is number of hydrogen bond params and some comments */
        if ( fgets( s, MAX_LINE, fp ) == NULL ) {
            fprintf( stderr, "[ERROR] reading force field failed\n" \
                     "  [INFO] reading number of hydrogen bond parameters\n" );
            exit( INVALID_INPUT );
        }
        Tokenize( s, &tmp, MAX_TOKEN_LEN );
        l = sstrtoul( tmp[0], __FILE__, __LINE__ );

        for ( i = 0; i < reax->max_num_atom_types; ++i ) {
            for ( j = 0; j < reax->max_num_atom_types; ++j ) {
                for ( k = 0; k < reax->max_num_atom_types; ++k ) {
                    reax->hbp[IDX_HBP(i, j, k, n)].is_valid = FALSE;
                }
            }
        }

        for ( i = 0; i < l; i++ ) {
            if ( fgets( s, MAX_LINE, fp ) == NULL ) {
                fprintf( stderr, "[ERROR] reading force field failed\n" \
                         "  [INFO] reading hydrogen bond parameters (entry %d)\n", i );
                exit( INVALID_INPUT );
            }
            Tokenize( s, &tmp, MAX_TOKEN_LEN );

            j = sstrtoul( tmp[0], __FILE__, __LINE__ ) - 1;
            k = sstrtoul( tmp[1], __FILE__, __LINE__ ) - 1;
            m = sstrtoul( tmp[2], __FILE__, __LINE__ ) - 1;

            if ( j < reax->num_atom_types && m < reax->num_atom_types ) {
                val = sstrtod( tmp[3], __FILE__, __LINE__ );
                reax->hbp[IDX_HBP(j, k, m, n)].r0_hb = val;

                val = sstrtod( tmp[4], __FILE__, __LINE__ );
                reax->hbp[IDX_HBP(j, k, m, n)].p_hb1 = val;

                val = sstrtod( tmp[5], __FILE__, __LINE__ );
                reax->hbp[IDX_HBP(j, k, m, n)].p_hb2 = val;

                val = sstrtod( tmp[6], __FILE__, __LINE__ );
                reax->hbp[IDX_HBP(j, k, m, n)].p_hb3 = val;

                reax->hbp[IDX_HBP(j, k, m, n)].is_valid = TRUE;
            }
        }

        /* deallocate helper storage */
        for ( i = 0; i < MAX_TOKENS; i++ ) {
            sfree( tmp[i], __FILE__, __LINE__ );
        }
        sfree( tmp, __FILE__, __LINE__ );
        sfree( s, __FILE__, __LINE__ );

        sfree( tor_flag, __FILE__, __LINE__ );
        sfree( gamma, __FILE__, __LINE__ );
    }

    sfclose( fp, __FILE__, __LINE__ );
}
