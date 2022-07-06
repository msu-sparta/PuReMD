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
  #include "io_tools.h"

  #include "basic_comm.h"
  #include "comm_tools.h"
  #include "index_utils.h"
  #include "list.h"
  #include "reset_tools.h"
  #include "system_props.h"
  #include "tool_box.h"
  #include "traj.h"
  #include "vector.h"
#elif defined(LAMMPS_REAX)
  #include "reax_io_tools.h"

  #include "reax_basic_comm.h"
  #include "reax_comm_tools.h"
  #include "reax_index_utils.h"
  #include "reax_list.h"
  #include "reax_reset_tools.h"
  #include "reax_system_props.h"
  #include "reax_tool_box.h"
  #include "reax_traj.h"
  #include "reax_vector.h"
#endif


/************************ initialize output controls ************************/
void Init_Output_Files( reax_system *system, control_params *control,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int ret;
    char temp[MAX_STR];

    /* trajectory file */
    if ( out_control->write_steps > 0 )
    {
        Init_Traj( system, control, out_control, mpi_data );
    }

    if ( system->my_rank == MASTER_NODE )
    {
        if ( out_control->energy_update_freq > 0 )
        {
            /* out file */
            sprintf( temp, "%s.out", control->sim_name );
            out_control->out = sfopen( temp, "w", __FILE__, __LINE__ );

#if !defined(DEBUG) && !defined(DEBUG_FOCUS)
            fprintf( out_control->out, "%-6s%14s%14s%14s%11s%13s%13s\n",
                     "step", "total energy", "potential", "kinetic",
                     "T(K)", "V(A^3)", "P(Gpa)" );
#else
            fprintf( out_control->out, "%-6s%24s%24s%24s%13s%16s%13s\n",
                     "step", "total energy", "potential", "kinetic",
                     "T(K)", "V(A^3)", "P(GPa)" );
            fflush( out_control->out );
#endif

            /* potential file */
            sprintf( temp, "%s.pot", control->sim_name );
            out_control->pot = sfopen( temp, "w", __FILE__, __LINE__ );

#if !defined(DEBUG) && !defined(DEBUG_FOCUS)
            fprintf( out_control->pot,
                     "%-6s%14s%14s%14s%14s%14s%14s%14s%14s%14s%14s%14s\n",
                     "step", "ebond", "eatom", "elp",
                     "eang", "ecoa", "ehb", "etor", "econj",
                     "evdw", "ecoul", "epol" );
#else
            fprintf( out_control->pot,
                     "%-6s%24s%24s%24s%24s%24s%24s%24s%24s%24s%24s%24s\n",
                     "step", "ebond", "eatom", "elp",
                     "eang", "ecoa", "ehb", "etor", "econj",
                     "evdw", "ecoul", "epol" );
            fflush( out_control->pot );
#endif

#if defined(LOG_PERFORMANCE)
            /* log file */
            sprintf( temp, "%s.log", control->sim_name );
            out_control->log = sfopen( temp, "w", __FILE__, __LINE__ );

//            fprintf( out_control->log, "%6s%8s%8s%8s%8s%8s%8s%8s%8s%8s\n",
//                     "step", "total", "comm", "nbrs", "init", "bonded", "nonb",
//                     "charges", "siters", "retries" );

            fprintf( out_control->log, "%6s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
                     "step", "total", "comm", "nbrs", "init",
                     "init_dist", "init_cm", "init_bond", "init_hbond",
                     "bonded", "BO", "bonds", "lpovun", "val", "tors", "hbonds", "nonb",
                     "cm", "cm_sort", "cm_iters", "cm_p_comp", "cm_p_app",
                     "cm_comm", "cm_allr", "cm_spmv", "cm_vec_ops", "cm_orthog", "cm_t_solve",
                     "retries" );

#if defined(DEBUG)
            fflush( out_control->log );
#endif
#endif
        }

        /* pressure file */
        if ( control->ensemble == NPT  ||
                control->ensemble == iNPT ||
                control->ensemble == sNPT )
        {
            sprintf( temp, "%s.prs", control->sim_name );
            out_control->prs = sfopen( temp, "w", __FILE__, __LINE__ );

            fprintf( out_control->prs, "%8s%13s%13s%13s%13s%13s%13s%13s\n",
                    "step", "Pint/norm[x]", "Pint/norm[y]", "Pint/norm[z]",
                    "Pext/Ptot[x]", "Pext/Ptot[y]", "Pext/Ptot[z]", "Pkin/V" );
#if defined(DEBUG)
            fflush( out_control->prs );
#endif
        }

        /* electric dipole moment analysis file */
        if ( control->dipole_anal )
        {
            sprintf( temp, "%s.dpl", control->sim_name );

            out_control->dpl = sfopen( temp, "w", __FILE__, __LINE__ );

            fprintf( out_control->dpl, "%6s%20s%30s",
                     "step", "molecule count", "avg dipole moment norm" );
#if defined(DEBUG)
            fflush( out_control->dpl );
#endif
        }

        /* diffusion coef analysis file */
        if ( control->diffusion_coef )
        {
            sprintf( temp, "%s.drft", control->sim_name );
            out_control->drft = sfopen( temp, "w", __FILE__, __LINE__ );

            fprintf( out_control->drft, "%7s%20s%20s\n",
                     "step", "type count", "avg disp^2" );
#if defined(DEBUG)
            fflush( out_control->drft );
#endif
        }
    }

    /* molecular analysis file:
     * proc0 opens this file and shares it with everyone.
     * then all processors write into it in a round-robin
     * fashion controlled by their rank */
    if ( control->molecular_analysis )
    {
        if ( system->my_rank == MASTER_NODE )
        {
            sprintf( temp, "%s.mol", control->sim_name );
            out_control->mol = sfopen( temp, "w", __FILE__, __LINE__ );
        }

        ret = MPI_Bcast( &out_control->mol, 1, MPI_LONG, 0, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
    }

#if defined(TEST_ENERGY)
    /* bond energy file */
    sprintf( temp, "%s.ebond.%d", control->sim_name, system->my_rank );
    out_control->ebond = sfopen( temp, "w", __FILE__, __LINE__ );

    /* lone-pair energy file */
    sprintf( temp, "%s.elp.%d", control->sim_name, system->my_rank );
    out_control->elp = sfopen( temp, "w", __FILE__, __LINE__ );

    /* overcoordination energy file */
    sprintf( temp, "%s.eov.%d", control->sim_name, system->my_rank );
    out_control->eov = sfopen( temp, "w", __FILE__, __LINE__ );

    /* undercoordination energy file */
    sprintf( temp, "%s.eun.%d", control->sim_name, system->my_rank );
    out_control->eun = sfopen( temp, "w", __FILE__, __LINE__ );

    /* angle energy file */
    sprintf( temp, "%s.eval.%d", control->sim_name, system->my_rank );
    out_control->eval = sfopen( temp, "w", __FILE__, __LINE__ );

    /* coalition energy file */
    sprintf( temp, "%s.ecoa.%d", control->sim_name, system->my_rank );
    out_control->ecoa = sfopen( temp, "w", __FILE__, __LINE__ );

    /* penalty energy file */
    sprintf( temp, "%s.epen.%d", control->sim_name, system->my_rank );
    out_control->epen = sfopen( temp, "w", __FILE__, __LINE__ );

    /* torsion energy file */
    sprintf( temp, "%s.etor.%d", control->sim_name, system->my_rank );
    out_control->etor = sfopen( temp, "w", __FILE__, __LINE__ );

    /* conjugation energy file */
    sprintf( temp, "%s.econ.%d", control->sim_name, system->my_rank );
    out_control->econ = sfopen( temp, "w", __FILE__, __LINE__ );

    /* hydrogen bond energy file */
    sprintf( temp, "%s.ehb.%d", control->sim_name, system->my_rank );
    out_control->ehb = sfopen( temp, "w", __FILE__, __LINE__ );

    /* vdWaals energy file */
    sprintf( temp, "%s.evdw.%d", control->sim_name, system->my_rank );
    out_control->evdw = sfopen( temp, "w", __FILE__, __LINE__ );

    /* coulomb energy file */
    sprintf( temp, "%s.ecou.%d", control->sim_name, system->my_rank );
    out_control->ecou = sfopen( temp, "w", __FILE__, __LINE__ );
#endif

#if defined(TEST_FORCES)
    /* bond orders file */
    sprintf( temp, "%s.fbo.%d", control->sim_name, system->my_rank );
    out_control->fbo = sfopen( temp, "w", __FILE__, __LINE__ );

    /* bond orders derivatives file */
    sprintf( temp, "%s.fdbo.%d", control->sim_name, system->my_rank );
    out_control->fdbo = sfopen( temp, "w", __FILE__, __LINE__ );

    /* produce a single force file - to be written by p0 */
    if ( system->my_rank == MASTER_NODE )
    {
        /* bond forces file */
        sprintf( temp, "%s.fbond", control->sim_name );
        out_control->fbond = sfopen( temp, "w", __FILE__, __LINE__ );

        /* lone-pair forces file */
        sprintf( temp, "%s.flp", control->sim_name );
        out_control->flp = sfopen( temp, "w", __FILE__, __LINE__ );

        /* overcoordination forces file */
        sprintf( temp, "%s.fov", control->sim_name );
        out_control->fov = sfopen( temp, "w", __FILE__, __LINE__ );

        /* undercoordination forces file */
        sprintf( temp, "%s.fun", control->sim_name );
        out_control->fun = sfopen( temp, "w", __FILE__, __LINE__ );

        /* angle forces file */
        sprintf( temp, "%s.fang", control->sim_name );
        out_control->fang = sfopen( temp, "w", __FILE__, __LINE__ );

        /* coalition forces file */
        sprintf( temp, "%s.fcoa", control->sim_name );
        out_control->fcoa = sfopen( temp, "w", __FILE__, __LINE__ );

        /* penalty forces file */
        sprintf( temp, "%s.fpen", control->sim_name );
        out_control->fpen = sfopen( temp, "w", __FILE__, __LINE__ );

        /* torsion forces file */
        sprintf( temp, "%s.ftor", control->sim_name );
        out_control->ftor = sfopen( temp, "w", __FILE__, __LINE__ );

        /* conjugation forces file */
        sprintf( temp, "%s.fcon", control->sim_name );
        out_control->fcon = sfopen( temp, "w", __FILE__, __LINE__ );

        /* hydrogen bond forces file */
        sprintf( temp, "%s.fhb", control->sim_name );
        out_control->fhb = sfopen( temp, "w", __FILE__, __LINE__ );

        /* vdw forces file */
        sprintf( temp, "%s.fvdw", control->sim_name );
        out_control->fvdw = sfopen( temp, "w", __FILE__, __LINE__ );

        /* nonbonded forces file */
        sprintf( temp, "%s.fele", control->sim_name );
        out_control->fele = sfopen( temp, "w", __FILE__, __LINE__ );

        /* total force file */
        sprintf( temp, "%s.ftot", control->sim_name );
        out_control->ftot = sfopen( temp, "w", __FILE__, __LINE__ );

        /* force comprison file */
        sprintf( temp, "%s.fcomp", control->sim_name );
        out_control->fcomp = sfopen( temp, "w", __FILE__, __LINE__ );
    }
#endif

#if defined(PURE_REAX)
#if defined(TEST_FORCES) || defined(TEST_ENERGY)
    /* far neighbor list file */
    sprintf( temp, "%s.far_nbrs_list.%d", control->sim_name, system->my_rank );
    out_control->flist = sfopen( temp, "w", __FILE__, __LINE__ );

    /* bond list file */
    sprintf( temp, "%s.bond_list.%d", control->sim_name, system->my_rank );
    out_control->blist = sfopen( temp, "w", __FILE__, __LINE__ );

    /* near neighbor list file */
    sprintf( temp, "%s.near_nbrs_list.%d", control->sim_name, system->my_rank );
    out_control->nlist = sfopen( temp, "w", __FILE__, __LINE__ );
#endif
#endif
}


void Finalize_Output_Files( reax_system *system, control_params *control,
        output_controls *out_control )
{
    if ( out_control->write_steps > 0 )
    {
        End_Traj( system->my_rank, out_control );
    }

    if ( system->my_rank == MASTER_NODE )
    {
        if ( out_control->energy_update_freq > 0 )
        {
            if ( out_control->out )
            {
                sfclose( out_control->out, __FILE__, __LINE__ );
            }
            if ( out_control->pot )
            {
                sfclose( out_control->pot, __FILE__, __LINE__ );
            }

#if defined(LOG_PERFORMANCE)
            if ( out_control->log )
            {
                sfclose( out_control->log, __FILE__, __LINE__ );
            }
#endif
        }

        if ( control->ensemble == NPT || control->ensemble == iNPT ||
                control->ensemble == sNPT )
        {
            if ( out_control->prs )
            {
                sfclose( out_control->prs, __FILE__, __LINE__ );
            }
        }

        if ( control->dipole_anal )
        {
            sfclose( out_control->dpl, __FILE__, __LINE__ );
        }

        if ( control->diffusion_coef )
        {
            sfclose( out_control->drft, __FILE__, __LINE__ );
        }

        if ( control->molecular_analysis )
        {
            sfclose( out_control->mol, __FILE__, __LINE__ );
        }
    }

#if defined(TEST_ENERGY)
    sfclose( out_control->ebond, __FILE__, __LINE__ );
    sfclose( out_control->elp, __FILE__, __LINE__ );
    sfclose( out_control->eov, __FILE__, __LINE__ );
    sfclose( out_control->eun, __FILE__, __LINE__ );
    sfclose( out_control->eval, __FILE__, __LINE__ );
    sfclose( out_control->epen, __FILE__, __LINE__ );
    sfclose( out_control->ecoa, __FILE__, __LINE__ );
    sfclose( out_control->ehb, __FILE__, __LINE__ );
    sfclose( out_control->etor, __FILE__, __LINE__ );
    sfclose( out_control->econ, __FILE__, __LINE__ );
    sfclose( out_control->evdw, __FILE__, __LINE__ );
    sfclose( out_control->ecou, __FILE__, __LINE__ );
#endif

#if defined(TEST_FORCES)
    sfclose( out_control->fbo, __FILE__, __LINE__ );
    sfclose( out_control->fdbo, __FILE__, __LINE__ );

    if ( system->my_rank == MASTER_NODE )
    {
        sfclose( out_control->fbond, __FILE__, __LINE__ );
        sfclose( out_control->flp, __FILE__, __LINE__ );
        sfclose( out_control->fov, __FILE__, __LINE__ );
        sfclose( out_control->fun, __FILE__, __LINE__ );
        sfclose( out_control->fang, __FILE__, __LINE__ );
        sfclose( out_control->fcoa, __FILE__, __LINE__ );
        sfclose( out_control->fpen, __FILE__, __LINE__ );
        sfclose( out_control->ftor, __FILE__, __LINE__ );
        sfclose( out_control->fcon, __FILE__, __LINE__ );
        sfclose( out_control->fhb, __FILE__, __LINE__ );
        sfclose( out_control->fvdw, __FILE__, __LINE__ );
        sfclose( out_control->fele, __FILE__, __LINE__ );
        sfclose( out_control->ftot, __FILE__, __LINE__ );
        sfclose( out_control->fcomp, __FILE__, __LINE__ );
    }
#endif

#if defined(PURE_REAX)
#if defined(TEST_FORCES) || defined(TEST_ENERGY)
    sfclose( out_control->flist, __FILE__, __LINE__ );
    sfclose( out_control->blist, __FILE__, __LINE__ );
    sfclose( out_control->nlist, __FILE__, __LINE__ );
#endif
#endif
}


void Print_Box( simulation_box* box, char *name, FILE *out )
{
    // int i, j;

    fprintf( out, "%s:\n", name );
    fprintf( out, "\tmin[%8.3f %8.3f %8.3f]\n",
             box->min[0], box->min[1], box->min[2] );
    fprintf( out, "\tmax[%8.3f %8.3f %8.3f]\n",
             box->max[0], box->max[1], box->max[2] );
    fprintf( out, "\tdims[%8.3f%8.3f%8.3f]\n",
             box->box_norms[0], box->box_norms[1], box->box_norms[2] );

    // fprintf( out, "box: {" );
    // for( i = 0; i < 3; ++i )
    //   {
    //     fprintf( out, "{" );
    //     for( j = 0; j < 3; ++j )
    //       fprintf( out, "%8.3f ", box->box[i][j] );
    //     fprintf( out, "}" );
    //   }
    // fprintf( out, "}\n" );

    // fprintf( out, "box_trans: {" );
    // for( i = 0; i < 3; ++i )
    //   {
    //     fprintf( out, "{" );
    //     for( j = 0; j < 3; ++j )
    //    fprintf( out, "%8.3f ", box->trans[i][j] );
    //     fprintf( out, "}" );
    //   }
    // fprintf( out, "}\n" );

    // fprintf( out, "box_trinv: {" );
    // for( i = 0; i < 3; ++i )
    //   {
    //     fprintf( out, "{" );
    //     for( j = 0; j < 3; ++j )
    //    fprintf( out, "%8.3f ", box->trans_inv[i][j] );
    //     fprintf( out, "}" );
    //   }
    // fprintf( out, "}\n" );
}


void Print_Grid( grid* g, FILE *out )
{
    int x, y, z, gc_type;
    ivec gc_str;
    char gcell_type_text[10][12] =
    {
        "NO_NBRS", "NEAR_ONLY", "HBOND_ONLY", "FAR_ONLY",
        "NEAR_HBOND", "NEAR_FAR", "HBOND_FAR", "FULL_NBRS", "NATIVE"
    };

    fprintf( out, "\tnumber of grid cells: %d %d %d\n",
             g->ncells[0], g->ncells[1], g->ncells[2] );
    fprintf( out, "\tgcell lengths: %8.3f %8.3f %8.3f\n",
             g->cell_len[0], g->cell_len[1], g->cell_len[2] );
    fprintf( out, "\tinverses of gcell lengths: %8.3f %8.3f %8.3f\n",
             g->inv_len[0], g->inv_len[1], g->inv_len[2] );
    fprintf( out, "\t---------------------------------\n" );
    fprintf( out, "\tnumber of native gcells: %d %d %d\n",
             g->native_cells[0], g->native_cells[1], g->native_cells[2] );
    fprintf( out, "\tnative gcell span: %d-%d  %d-%d  %d-%d\n",
             g->native_str[0], g->native_end[0],
             g->native_str[1], g->native_end[1],
             g->native_str[2], g->native_end[2] );
    fprintf( out, "\t---------------------------------\n" );
    fprintf( out, "\tvlist gcell stretch: %d %d %d\n",
             g->vlist_span[0], g->vlist_span[1], g->vlist_span[2] );
    fprintf( out, "\tnonbonded nbrs gcell stretch: %d %d %d\n",
             g->nonb_span[0], g->nonb_span[1], g->nonb_span[2] );
    fprintf( out, "\tbonded nbrs gcell stretch: %d %d %d\n",
             g->bond_span[0], g->bond_span[1], g->bond_span[2] );
    fprintf( out, "\t---------------------------------\n" );
    fprintf( out, "\tghost gcell span: %d %d %d\n",
             g->ghost_span[0], g->ghost_span[1], g->ghost_span[2] );
    fprintf( out, "\tnonbonded ghost gcell span: %d %d %d\n",
             g->ghost_nonb_span[0], g->ghost_nonb_span[1], g->ghost_nonb_span[2]);
    fprintf(out, "\thbonded ghost gcell span: %d %d %d\n",
            g->ghost_hbond_span[0], g->ghost_hbond_span[1], g->ghost_hbond_span[2]);
    fprintf( out, "\tbonded ghost gcell span: %d %d %d\n",
             g->ghost_bond_span[0], g->ghost_bond_span[1], g->ghost_bond_span[2]);
    fprintf( out, "\t---------------------------------\n" );

    fprintf( stderr, "GCELL MARKS:\n" );
    gc_type = g->cells[ index_grid_3d(0, 0, 0, g) ].type;
    ivec_MakeZero( gc_str );

    x = y = z = 0;
    for ( x = 0; x < g->ncells[0]; ++x )
        for ( y = 0; y < g->ncells[1]; ++y )
            for ( z = 0; z < g->ncells[2]; ++z )
                if ( g->cells[ index_grid_3d(x, y, z, g) ].type != gc_type )
                {
                    fprintf( stderr,
                             "\tgcells from(%2d %2d %2d) to (%2d %2d %2d): %d - %s\n",
                             gc_str[0], gc_str[1], gc_str[2], x, y, z,
                             gc_type, gcell_type_text[gc_type] );
                    gc_type = g->cells[ index_grid_3d(x, y, z, g) ].type;
                    gc_str[0] = x;
                    gc_str[1] = y;
                    gc_str[2] = z;
                }
    fprintf( stderr, "\tgcells from(%2d %2d %2d) to (%2d %2d %2d): %d - %s\n",
             gc_str[0], gc_str[1], gc_str[2], x, y, z,
             gc_type, gcell_type_text[gc_type] );
    fprintf( out, "-------------------------------------\n" );
}


void Print_Native_GCells( reax_system *system )
{
    int i, j, k, l;
    char fname[100];
    FILE *f;
    grid *g;
    grid_cell *gc;
    char gcell_type_text[10][12] =
    {
        "NO_NBRS", "NEAR_ONLY", "HBOND_ONLY", "FAR_ONLY",
        "NEAR_HBOND", "NEAR_FAR", "HBOND_FAR", "FULL_NBRS", "NATIVE"
    };

    sprintf( fname, "native_gcells.%d", system->my_rank );
    f = sfopen( fname, "w", __FILE__, __LINE__ );
    g = &system->my_grid;

    for ( i = g->native_str[0]; i < g->native_end[0]; i++ )
    {
        for ( j = g->native_str[1]; j < g->native_end[1]; j++ )
        {
            for ( k = g->native_str[2]; k < g->native_end[2]; k++ )
            {
                gc = &g->cells[ index_grid_3d(i, j, k, g) ];

                fprintf( f, "p%d gcell(%2d %2d %2d) of type %d(%s)\n",
                         system->my_rank, i, j, k,
                         gc->type, gcell_type_text[gc->type] );

                //fprintf( f, "\tatom list start: %d, end: %d\n\t", gc->str, gc->end );
                fprintf( f, "\tatom list start: %d, end: %d\n\t",
                        g->str[index_grid_3d(i, j, k, g)],
                        g->end[index_grid_3d(i, j, k, g)] );

                //for( l = gc->str; l < gc->end; ++l )
                for ( l = g->str[index_grid_3d(i, j, k, g)];
                        l < g->end[index_grid_3d(i, j, k, g)]; ++l )
                {
                    fprintf( f, "%5d", system->my_atoms[l].orig_id );
                }
                fprintf( f, "\n" );
            }
        }
    }

    sfclose( f, __FILE__, __LINE__ );
}


void Print_All_GCells( reax_system *system )
{
    int i, j, k, l;
    char fname[100];
    FILE *f;
    grid *g;
    grid_cell *gc;
    char gcell_type_text[10][12] =
    {
        "NO_NBRS", "NEAR_ONLY", "HBOND_ONLY", "FAR_ONLY",
        "NEAR_HBOND", "NEAR_FAR", "HBOND_FAR", "FULL_NBRS", "NATIVE"
    };

    sprintf( fname, "all_gcells.%d", system->my_rank );
    f = sfopen( fname, "w", __FILE__, __LINE__ );
    g = &system->my_grid;

    for ( i = 0; i < g->ncells[0]; i++ )
    {
        for ( j = 0; j < g->ncells[1]; j++ )
        {
            for ( k = 0; k < g->ncells[2]; k++ )
            {
                gc = &g->cells[ index_grid_3d(i, j, k, g) ];

                fprintf( f, "p%d gcell(%2d %2d %2d) of type %d(%s)\n",
                         system->my_rank, i, j, k,
                         gc->type, gcell_type_text[gc->type] );

                //fprintf( f, "\tatom list start: %d, end: %d\n\t", gc->str, gc->end );
                fprintf( f, "\tatom list start: %d, end: %d\n\t",
                        g->str[index_grid_3d(i, j, k, g)],
                        g->end[index_grid_3d(i, j, k, g)] );

                //for( l = gc->str; l < gc->end; ++l )
                for ( l = g->str[index_grid_3d(i, j, k, g)];
                        l < g->end[index_grid_3d(i, j, k, g)]; ++l )
                {
                    fprintf( f, "%5d", system->my_atoms[l].orig_id );
                }
                fprintf( f, "\n" );
            }
        }
    }

    sfclose( f, __FILE__, __LINE__ );
}


void Print_My_Atoms( reax_system *system, control_params *control, int step )
{
    int i;
    char fname[100];
    FILE *fh;

    sprintf( fname, "%s.my_atoms.%d.%d", control->sim_name, step, system->my_rank );
    fh = sfopen( fname, "w", __FILE__, __LINE__ );

    // fprintf( stderr, "p%d had %d atoms\n",
    //   system->my_rank, system->n );

    for ( i = 0; i < system->n; ++i )
    {
        fprintf( fh, "p%-2d %-5d %2d %24.15e%24.15e%24.15e\n",
                 system->my_rank,
                 system->my_atoms[i].orig_id, system->my_atoms[i].type,
                 system->my_atoms[i].x[0],
                 system->my_atoms[i].x[1],
                 system->my_atoms[i].x[2] );
    }

    sfclose( fh, __FILE__, __LINE__ );
}


void Print_My_Ext_Atoms( reax_system *system )
{
    int i;
    char fname[100];
    FILE *fh;

    sprintf( fname, "my_ext_atoms.%d", system->my_rank );
    fh = sfopen( fname, "w", __FILE__, __LINE__ );

    // fprintf( stderr, "p%d had %d atoms\n",
    //   system->my_rank, system->n );

    for ( i = 0; i < system->N; ++i )
    {
        fprintf( fh, "p%-2d %-5d imprt%-5d %2d %24.15e%24.15e%24.15e\n",
                 system->my_rank, system->my_atoms[i].orig_id,
                 system->my_atoms[i].imprt_id, system->my_atoms[i].type,
                 system->my_atoms[i].x[0],
                 system->my_atoms[i].x[1],
                 system->my_atoms[i].x[2] );
    }

    sfclose( fh, __FILE__, __LINE__ );
}


void Print_Far_Neighbors( reax_system *system, reax_list **lists,
        control_params *control )
{
    int i, j, id_i, id_j, nbr;
    char fname[100];
    FILE *fout;
    reax_list *far_nbrs;

    sprintf( fname, "%s.far_nbrs.%d", control->sim_name, system->my_rank );
    fout = sfopen( fname, "w", __FILE__, __LINE__ );
    far_nbrs = lists[FAR_NBRS];

    if ( far_nbrs->format == HALF_LIST )
    {
        for ( i = 0; i < system->N; ++i )
        {
            id_i = system->my_atoms[i].orig_id;

            for ( j = Start_Index(i, far_nbrs); j < End_Index(i, far_nbrs); ++j )
            {
                nbr = far_nbrs->far_nbr_list.nbr[j];
                id_j = system->my_atoms[nbr].orig_id;

                fprintf( fout, "%6d %6d %9d %9d (%3d,%3d,%3d) %12.7e %12.7e %12.7e %12.7e\n",
                         id_i, id_j, i, j,
                         far_nbrs->far_nbr_list.rel_box[j][0],
                         far_nbrs->far_nbr_list.rel_box[j][1],
                         far_nbrs->far_nbr_list.rel_box[j][2],
                         far_nbrs->far_nbr_list.d[j],
                         far_nbrs->far_nbr_list.dvec[j][0],
                         far_nbrs->far_nbr_list.dvec[j][1],
                         far_nbrs->far_nbr_list.dvec[j][2] );

                fprintf( fout, "%6d %6d %9d %9d (%3d,%3d,%3d) %12.7e %12.7e %12.7e %12.7e\n",
                         id_j, id_i, j, i,
                         -far_nbrs->far_nbr_list.rel_box[j][0],
                         -far_nbrs->far_nbr_list.rel_box[j][1],
                         -far_nbrs->far_nbr_list.rel_box[j][2],
                         far_nbrs->far_nbr_list.d[j],
                         -far_nbrs->far_nbr_list.dvec[j][0],
                         -far_nbrs->far_nbr_list.dvec[j][1],
                         -far_nbrs->far_nbr_list.dvec[j][2] );
            }
        }
    }
    else
    {
        for ( i = 0; i < system->N; ++i )
        {
            id_i = system->my_atoms[i].orig_id;

            for ( j = Start_Index(i, far_nbrs); j < End_Index(i, far_nbrs); ++j )
            {
                nbr = far_nbrs->far_nbr_list.nbr[j];
                id_j = system->my_atoms[nbr].orig_id;

                fprintf( fout, "%6d %6d %9d %9d (%3d,%3d,%3d) %12.7e %12.7e %12.7e %12.7e\n",
                         id_i, id_j, i, j,
                         far_nbrs->far_nbr_list.rel_box[j][0],
                         far_nbrs->far_nbr_list.rel_box[j][1],
                         far_nbrs->far_nbr_list.rel_box[j][2],
                         far_nbrs->far_nbr_list.d[j],
                         far_nbrs->far_nbr_list.dvec[j][0],
                         far_nbrs->far_nbr_list.dvec[j][1],
                         far_nbrs->far_nbr_list.dvec[j][2] );
            }
        }
    }


    sfclose( fout, __FILE__, __LINE__ );
}


void Print_Sparse_Matrix( reax_system *system, sparse_matrix *A )
{
    int i, pj;

    for ( i = 0; i < A->n; ++i )
    {
        for ( pj = A->start[i]; pj < A->end[i]; ++pj )
        {
            fprintf( stderr, "%d %d %.15e\n",
                    system->my_atoms[i].orig_id,
                    system->my_atoms[A->j[pj]].orig_id,
                    A->val[pj] );
        }
    }
}


void Print_Sparse_Matrix2( reax_system const * const system,
        sparse_matrix const * const A, char const * const fname )
{
    int i, pj;
    FILE *f;
    
    f = sfopen( fname, "w", __FILE__, __LINE__ );

    for ( i = 0; i < A->n; ++i )
    {
        for ( pj = A->start[i]; pj < A->end[i]; ++pj )
        {
            fprintf( f, "%d %d %.15e\n",
                    system->my_atoms[i].orig_id,
                    system->my_atoms[A->j[pj]].orig_id,
                    A->val[pj] );
        }
    }

    sfclose( f, __FILE__, __LINE__ );
}


void Print_Symmetric_Sparse( reax_system *system, sparse_matrix *A, char *fname )
{
    int i, j;
    reax_atom *ai, *aj;
    FILE *f;

    f = sfopen( fname, "w", __FILE__, __LINE__ );

    for ( i = 0; i < A->n; ++i )
    {
        ai = &system->my_atoms[i];

        for ( j = A->start[i]; j < A->end[i]; ++j )
        {
            aj = &system->my_atoms[A->j[j]];

            fprintf( f, "%d %d %.15e\n",
                    ai->renumber, aj->renumber, A->val[j] );

            if ( A->j[j] < system->n && ai->renumber != aj->renumber )
            {
                fprintf( f, "%d %d %.15e\n",
                         aj->renumber, ai->renumber, A->val[j] );
            }
        }
    }

    sfclose( f, __FILE__, __LINE__ );
}


void Print_Linear_System( reax_system *system, control_params *control,
        storage *workspace, int step )
{
    int i;
//    int j;
    char fname[100];
    reax_atom *ai;
//    reax_atom *aj;
//    sparse_matrix *H;
    FILE *out;

    /* print rhs and init guesses for QEq */
    sprintf( fname, "%s.p%dstate%d", control->sim_name, system->my_rank, step );
    out = sfopen( fname, "w", __FILE__, __LINE__ );

    for ( i = 0; i < system->n; i++ )
    {
        ai = &system->my_atoms[i];
        fprintf( out, "%6d%2d%24.15e%24.15e%24.15e%24.15e%24.15e%24.15e%24.15e\n",
                 ai->renumber, ai->type, ai->x[0], ai->x[1], ai->x[2],
                 workspace->s[i], workspace->b_s[i],
                 workspace->t[i], workspace->b_t[i] );
    }
    sfclose( out, __FILE__, __LINE__ );

    /* print charge matrix */
    sprintf( fname, "%s.p%dH%d", control->sim_name, system->my_rank, step );
    Print_Symmetric_Sparse( system, &workspace->H, fname ); //MATRIX CHANGES

    /* print the incomplete H matrix */
//    sprintf( fname, "%s.p%dHinc%d", control->sim_name, system->my_rank, step );
//    out = sfopen( fname, "w", __FILE__, __LINE__ );
//    H = workspace->H;
//    for( i = 0; i < H->n; ++i )
//    {
//        ai = &(system->my_atoms[i]);
//        for( j = H->start[i]; j < H->end[i]; ++j )
//        {
//            if( H->j[j] < system->n )
//            {
//                aj = &(system->my_atoms[H->j[j]]);
//                fprintf( out, "%d %d %.15e\n",
//                ai->orig_id, aj->orig_id, H->val[j] );
//                if( ai->orig_id != aj->orig_id )
//                    fprintf( out, "%d %d %.15e\n",
//                aj->orig_id, ai->orig_id, H->val[j] );
//            }
//        }
//    }
//    sfclose( out, __FILE__, __LINE__ );

    // print the L from incomplete cholesky decomposition
//    sprintf( fname, "%s.p%dL%d", control->sim_name, system->my_rank, step );
//    Print_Sparse_Matrix2( system, workspace->L, fname );
}


void Print_LinSys_Soln( reax_system *system, real *x, real *b_prm, real *b )
{
    int i;
    char fname[100];
    FILE *fout;

    sprintf( fname, "qeq.%d.out", system->my_rank );
    fout = sfopen( fname, "w", __FILE__, __LINE__ );

    for ( i = 0; i < system->n; ++i )
    {
        fprintf( fout, "%6d%10.4f%10.4f%10.4f\n",
                 system->my_atoms[i].orig_id, x[i], b_prm[i], b[i] );
    }

    sfclose( fout, __FILE__, __LINE__ );
}


void Print_Charges( reax_system *system )
{
    int i;
    char fname[100];
    FILE *fout;

    sprintf( fname, "q.%d.out", system->my_rank );
    fout = sfopen( fname, "w", __FILE__, __LINE__ );

    for ( i = 0; i < system->n; ++i )
    {
        fprintf( fout, "%6d %10.7f %10.7f %10.7f\n",
                 system->my_atoms[i].orig_id,
                 system->my_atoms[i].s[0],
                 system->my_atoms[i].t[0],
                 system->my_atoms[i].q );
    }

    sfclose( fout, __FILE__, __LINE__ );
}


void Print_HBonds( reax_system *system, reax_list **lists,
        control_params *control, int step )
{
    int i, pj; 
    char fname[MAX_STR]; 
    hbond_data *phbond;
    FILE *fout;
    reax_list *far_nbr_list, *hbond_list;

    far_nbr_list = lists[FAR_NBRS];
    hbond_list = lists[HBONDS];

    sprintf( fname, "%s.hbonds.%d.%d", control->sim_name, step, system->my_rank );
    fout = sfopen( fname, "w", __FILE__, __LINE__ );

    for ( i = 0; i < hbond_list->n; ++i )
    {
        for ( pj = Start_Index(i, hbond_list); pj < End_Index(i, hbond_list); ++pj )
        {
            phbond = &hbond_list->hbond_list[pj];

            fprintf( fout, "%8d%8d %24.15e %24.15e %24.15e\n", i, phbond->nbr,
                    far_nbr_list->far_nbr_list.dvec[phbond->ptr][0],
                    far_nbr_list->far_nbr_list.dvec[phbond->ptr][1],
                    far_nbr_list->far_nbr_list.dvec[phbond->ptr][2] );
//            fprintf( fout, "%8d%8d %8d %8d\n", i, phbond->nbr,
//                  phbond->scl, phbond->sym_index );
        }
    }

    sfclose( fout, __FILE__, __LINE__ );
}

 
void Print_HBond_Indices( reax_system *system, reax_list **lists,
        control_params *control, int step )
{
    int i; 
    char fname[MAX_STR]; 
    FILE *fout;
    reax_list *hbonds = lists[HBONDS];

    sprintf( fname, "%s.hbonds_indices.%d.%d", control->sim_name, step, system->my_rank );
    fout = sfopen( fname, "w", __FILE__, __LINE__ );

    for ( i = 0; i < system->N; ++i )
    {
        fprintf( fout, "%8d: start: %8d, end: %8d\n",
                i, Start_Index(i, hbonds), End_Index(i, hbonds) );
    }

    sfclose( fout, __FILE__, __LINE__ );
}


void Print_Bonds( reax_system *system, reax_list **lists,
        control_params *control, int step )
{
    int i, pj; 
    char fname[MAX_STR]; 
    bond_data *pbond;
    bond_order_data *bo_ij;
    FILE *fout;
    reax_list *bonds = lists[BONDS];

    sprintf( fname, "%s.bonds.%d.%d", control->sim_name, step, system->my_rank );
    fout = sfopen( fname, "w", __FILE__, __LINE__ );

    for ( i = 0; i < system->N; ++i )
    {
        for ( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj )
        {
            pbond = &bonds->bond_list[pj];
            bo_ij = &pbond->bo_data;
//            fprintf( fout, "%6d%6d%23.15e%23.15e%23.15e%23.15e%23.15e\n",
//                    system->my_atoms[i].orig_id, system->my_atoms[j].orig_id,
//                    pbond->d, bo_ij->BO, bo_ij->BO_s, bo_ij->BO_pi, bo_ij->BO_pi2 );
            fprintf( fout, "%8d%8d %24.15f %24.15f\n",
                    i, pbond->nbr, //system->my_atoms[i].orig_id, system->my_atoms[j].orig_id,
                    pbond->d, bo_ij->BO );
        }
    }

    sfclose( fout, __FILE__, __LINE__ );
}


void Print_Three_Bodies( reax_system *system, reax_list **lists,
        control_params *control, int step )
{
    int i, pj; 
    char fname[MAX_STR]; 
    three_body_interaction_data *thb_data;
    FILE *fout;
    reax_list *thb_list = lists[THREE_BODIES];

    sprintf( fname, "%s.thbs.%d.%d", control->sim_name, step, system->my_rank );
    fout = sfopen( fname, "w", __FILE__, __LINE__ );

    for ( i = 0; i < thb_list->n; ++i )
    {
        for ( pj = Start_Index(i, thb_list); pj < End_Index(i, thb_list); ++pj )
        {
            thb_data = &thb_list->three_body_list[pj];

            fprintf( fout, "%12d %12d %24.15f %24.15f %24.15f %24.15f %24.15f %24.15f %24.15f %24.15f %24.15f %24.15f %24.15f\n",
                    thb_data->thb, thb_data->pthb, thb_data->theta, thb_data->cos_theta,
                    thb_data->dcos_di[0], thb_data->dcos_di[1], thb_data->dcos_di[2],
                    thb_data->dcos_dj[0], thb_data->dcos_dj[1], thb_data->dcos_dj[2],
                    thb_data->dcos_dk[0], thb_data->dcos_dk[1], thb_data->dcos_dk[2] );
        }
    }

    sfclose( fout, __FILE__, __LINE__ );
}


int fn_qsort_intcmp( const void *a, const void *b )
{
    return ( *(int *)a - * (int *)b );
}


void Print_Bond_List2( reax_system *system, reax_list *bonds, char *fname )
{
    int i, j, id_i, id_j, nbr, pj;
    FILE *f;
    int temp[500];
    int num;

    num = 0;
    f = sfopen( fname, "w", __FILE__, __LINE__ );

    for ( i = 0; i < system->n; ++i )
    {
        num = 0;
        id_i = system->my_atoms[i].orig_id;

        fprintf( f, "%6d:", id_i );

        for ( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj )
        {
            nbr = bonds->bond_list[pj].nbr;
            id_j = system->my_atoms[nbr].orig_id;

            if ( id_i < id_j )
            {
                temp[num++] = id_j;
            }
        }

        qsort( &temp, num, sizeof(int), fn_qsort_intcmp );
        for ( j = 0; j < num; j++ )
        {
            fprintf(f, "%6d", temp[j] );
        }
        fprintf(f, "\n");
    }
}


void Print_Total_Force( reax_system *system, simulation_data *data,
        storage *workspace )
{
    int i;

    fprintf( stderr, "step: %d\n", data->step );
    fprintf( stderr, "%6s\t%-38s\n", "atom", "atom.f[0,1,2]");

    for ( i = 0; i < system->N; ++i )
        fprintf( stderr, "%6d %f %f %f\n",
                 //"%6d%24.15e%24.15e%24.15e\n",
                 system->my_atoms[i].orig_id,
                 workspace->f[i][0], workspace->f[i][1], workspace->f[i][2] );
}


/* Print reax interaction list in adjacency list format */
void Print_Far_Neighbors_List_Adj_Format( reax_system *system,
        control_params *control, reax_list *list, int step )
{
    int i, pj, id_i, id_j, nbr, cnt;
    int num_intrs, *intrs;
    char fname[MAX_STR]; 
    FILE *fout;

    sprintf( fname, "%s.far.%d.%d", control->sim_name, step, system->my_rank );
    fout = sfopen( fname, "w", __FILE__, __LINE__ );

    num_intrs = 0;
    intrs = NULL;

    if ( fout == NULL )
    {
        fprintf( stderr, "[WARNING] null file pointer, returning without printing...\n" );
        return;
    }

    for ( i = 0; i < system->n; ++i )
    {
        cnt = 0;
        id_i = system->my_atoms[i].orig_id;
        fprintf( fout, "%d: ", id_i );

        if ( Num_Entries( i, list ) > num_intrs )
        {
            num_intrs = Num_Entries( i, list );
            intrs = srealloc( intrs, num_intrs * sizeof(int), __FILE__, __LINE__ );
        }

        for ( pj = Start_Index(i, list); pj < End_Index(i, list); ++pj )
        {
            nbr = list->far_nbr_list.nbr[pj];
            id_j = system->my_atoms[nbr].orig_id;
            intrs[cnt++] = id_j;
        }

        if ( cnt > 0 )
        {
            qsort( (void *) intrs, (size_t) cnt, sizeof(int), fn_qsort_intcmp );
        }

        for ( pj = 0; pj < cnt; ++pj )
        {
            fprintf( fout, "%d, ", intrs[pj] );
        }
        fprintf( fout, "\n" );
    }

    if ( intrs != NULL )
    {
        sfree( intrs, __FILE__, __LINE__ );
    }

    sfclose( fout, __FILE__, __LINE__ );
}


void Output_Results( reax_system *system, control_params *control,
        simulation_data *data, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
#if defined(LOG_PERFORMANCE)
    int ret;
    real my_timings[26], total_timings[26], t_elapsed, denom;
#endif

    if ( (out_control->energy_update_freq > 0
                && (data->step - data->prev_steps) % out_control->energy_update_freq == 0)
            || (out_control->write_steps > 0
                && (data->step - data->prev_steps) % out_control->write_steps == 0) )
    {
        /* output energies */
        if ( out_control->energy_update_freq > 0
                && (data->step - data->prev_steps) % out_control->energy_update_freq == 0 )
        {
#if defined(LOG_PERFORMANCE)
            my_timings[0] = data->timing.comm;
            my_timings[1] = data->timing.nbrs;
            my_timings[2] = data->timing.init_forces;
            my_timings[3] = data->timing.init_dist;
            my_timings[4] = data->timing.init_cm;
            my_timings[5] = data->timing.init_bond;
            my_timings[6] = data->timing.init_hbond;
            my_timings[7] = data->timing.bonded;
            my_timings[8] = data->timing.bond_order;
            my_timings[9] = data->timing.bonds;
            my_timings[10] = data->timing.lpovun;
            my_timings[11] = data->timing.valence;
            my_timings[12] = data->timing.torsion;
            my_timings[13] = data->timing.hbonds;
            my_timings[14] = data->timing.nonb;
            my_timings[15] = data->timing.cm;
            my_timings[16] = data->timing.cm_sort;
            my_timings[17] = (double) data->timing.cm_solver_iters;
            my_timings[18] = data->timing.cm_solver_pre_comp;
            my_timings[19] = data->timing.cm_solver_pre_app;
            my_timings[20] = data->timing.cm_solver_comm;
            my_timings[21] = data->timing.cm_solver_allreduce;
            my_timings[22] = data->timing.cm_solver_spmv;
            my_timings[23] = data->timing.cm_solver_vector_ops;
            my_timings[24] = data->timing.cm_solver_orthog;
            my_timings[25] = data->timing.cm_solver_tri_solve;

            ret = MPI_Reduce( my_timings, total_timings, 26, MPI_DOUBLE,
                    MPI_SUM, MASTER_NODE, MPI_COMM_WORLD );
            Check_MPI_Error( ret, __FILE__, __LINE__ );

            if ( system->my_rank == MASTER_NODE )
            {
                data->timing.comm = total_timings[0] / control->nprocs;
                data->timing.nbrs = total_timings[1] / control->nprocs;
                data->timing.init_forces = total_timings[2] / control->nprocs;
                data->timing.init_dist = total_timings[3] / control->nprocs;
                data->timing.init_cm = total_timings[4] / control->nprocs;
                data->timing.init_bond = total_timings[5] / control->nprocs;
                data->timing.init_hbond = total_timings[6] / control->nprocs;
                data->timing.bonded = total_timings[7] / control->nprocs;
                data->timing.bond_order = total_timings[8] / control->nprocs;
                data->timing.bonds = total_timings[9] / control->nprocs;
                data->timing.lpovun = total_timings[10] / control->nprocs;
                data->timing.valence = total_timings[11] / control->nprocs;
                data->timing.torsion = total_timings[12] / control->nprocs;
                data->timing.hbonds = total_timings[13] / control->nprocs;
                data->timing.nonb = total_timings[14] / control->nprocs;
                data->timing.cm = total_timings[15] / control->nprocs;
                data->timing.cm_sort = total_timings[16] / control->nprocs;
                data->timing.cm_solver_iters = total_timings[17] / control->nprocs;
                data->timing.cm_solver_pre_comp = total_timings[18] / control->nprocs;
                data->timing.cm_solver_pre_app = total_timings[19] / control->nprocs;
                data->timing.cm_solver_comm = total_timings[20] / control->nprocs;
                data->timing.cm_solver_allreduce = total_timings[21] / control->nprocs;
                data->timing.cm_solver_spmv = total_timings[22] / control->nprocs;
                data->timing.cm_solver_vector_ops = total_timings[23] / control->nprocs;
                data->timing.cm_solver_orthog = total_timings[24] / control->nprocs;
                data->timing.cm_solver_tri_solve = total_timings[25] / control->nprocs;
            }
#endif

            if ( system->my_rank == MASTER_NODE )
            {
#if !defined(DEBUG) && !defined(DEBUG_FOCUS)
                fprintf( out_control->out,
                         "%-6d%14.2f%14.2f%14.2f%11.2f%13.2f%13.5f\n",
                         data->step, data->sys_en.e_tot, data->sys_en.e_pot,
                         E_CONV * data->sys_en.e_kin, data->therm.T,
                         system->big_box.V, data->iso_bar.P );

                fprintf( out_control->pot,
                         "%-6d%14.2f%14.2f%14.2f%14.2f%14.2f%14.2f%14.2f%14.2f%14.2f%14.2f%14.2f\n",
                         data->step,
                         data->sys_en.e_bond,
                         data->sys_en.e_ov + data->sys_en.e_un,  data->sys_en.e_lp,
                         data->sys_en.e_ang + data->sys_en.e_pen, data->sys_en.e_coa,
                         data->sys_en.e_hb,
                         data->sys_en.e_tor, data->sys_en.e_con,
                         data->sys_en.e_vdW, data->sys_en.e_ele, data->sys_en.e_pol);
#else
                fprintf( out_control->out,
                        "%-6d%24.15f%24.15f%24.15f%13.5f%16.5f%13.5f\n",
                        data->step, data->sys_en.e_tot, data->sys_en.e_pot,
                        E_CONV * data->sys_en.e_kin, data->therm.T,
                        system->big_box.V, data->iso_bar.P );

                fprintf( out_control->pot,
                        "%-6d%24.15f%24.15f%24.15f%24.15f%24.15f%24.15f%24.15f%24.15f%24.15f%24.15f%24.15f\n",
                        data->step,
                        data->sys_en.e_bond,
                        data->sys_en.e_ov + data->sys_en.e_un,  data->sys_en.e_lp,
                        data->sys_en.e_ang + data->sys_en.e_pen, data->sys_en.e_coa,
                        data->sys_en.e_hb,
                        data->sys_en.e_tor, data->sys_en.e_con,
                        data->sys_en.e_vdW, data->sys_en.e_ele, data->sys_en.e_pol);

                fflush( out_control->out );
                fflush( out_control->pot );
#endif //DEBUG

#if defined(LOG_PERFORMANCE)
                t_elapsed = Get_Elapsed_Time( data->timing.total );

                /* average times according to logging simulation step frequency */
                if ( data->step - data->prev_steps > 0 )
                {
                    denom = 1.0 / out_control->energy_update_freq;
                }
                else
                {
                    denom = 1.0;
                }

//                fprintf( out_control->log,
//                        "%6d%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8d%8d\n",
//                        data->step, t_elapsed * denom, data->timing.comm * denom,
//                        data->timing.nbrs * denom, data->timing.init_forces * denom,
//                        data->timing.bonded * denom, data->timing.nonb * denom,
//                        data->timing.cm * denom,
//                        (int) (data->timing.cm_solver_iters * denom),
//                        data->timing.num_retries );

                fprintf( out_control->log,
                        "%6d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.2f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.2f\n",
                        data->step,
                        t_elapsed * denom,
                        data->timing.comm * denom,
                        data->timing.nbrs * denom,
                        data->timing.init_forces * denom,
                        data->timing.init_dist * denom,
                        data->timing.init_cm * denom,
                        data->timing.init_bond * denom,
                        data->timing.init_hbond * denom,
                        data->timing.bonded * denom,
                        data->timing.bond_order * denom,
                        data->timing.bonds * denom,
                        data->timing.lpovun * denom,
                        data->timing.valence * denom,
                        data->timing.torsion * denom,
                        data->timing.hbonds * denom,
                        (data->timing.nonb + data->timing.cm) * denom,
                        data->timing.cm * denom,
                        data->timing.cm_sort * denom,
                        (double) (data->timing.cm_solver_iters * denom),
                        data->timing.cm_solver_pre_comp * denom,
                        data->timing.cm_solver_pre_app * denom,
                        data->timing.cm_solver_comm * denom,
                        data->timing.cm_solver_allreduce * denom,
                        data->timing.cm_solver_spmv * denom,
                        data->timing.cm_solver_vector_ops * denom,
                        data->timing.cm_solver_orthog * denom,
                        data->timing.cm_solver_tri_solve * denom,
                        data->timing.num_retries * denom );

#if defined(DEBUG)
                fflush( out_control->log );
#endif
#endif //LOG_PERFORMANCE

                if ( control->virial == 1 )
                {
                    fprintf( out_control->prs,
                             "%8d%13.6f%13.6f%13.6f%13.6f%13.6f%13.6f%13.6f\n",
                             data->step,
                             data->int_press[0], data->int_press[1], data->int_press[2],
                             data->ext_press[0], data->ext_press[1], data->ext_press[2],
                             data->kin_press );

                    fprintf( out_control->prs,
                             "%8s%13.6f%13.6f%13.6f%13.6f%13.6f%13.6f%13.6f\n",
                             "", system->big_box.box_norms[0], system->big_box.box_norms[1],
                             system->big_box.box_norms[2],
                             data->tot_press[0], data->tot_press[1], data->tot_press[2],
                             system->big_box.V );

#if defined(DEBUG)
                    fflush( out_control->prs );
#endif
                }
            }

#if defined(LOG_PERFORMANCE)
            Reset_Timing( &data->timing );
#endif
        }

        /* write current frame */
        if ( out_control->write_steps > 0
                && (data->step - data->prev_steps) % out_control->write_steps == 0 )
        {
            Append_Frame( system, control, data, lists, out_control, mpi_data );
        }
    }
}


#if defined(TEST_ENERGY)
void Debug_Marker_Bonded( output_controls *out_control, int step )
{
    fprintf( out_control->ebond, "step: %d\n%6s%6s%12s%12s%12s\n",
             step, "atom1", "atom2", "bo", "ebond", "total" );
    fprintf( out_control->elp, "step: %d\n%6s%12s%12s%12s\n",
             step, "atom", "nlp", "elp", "total" );
    fprintf( out_control->eov, "step: %d\n%6s%12s%12s\n",
             step, "atom", "eov", "total" );
    fprintf( out_control->eun, "step: %d\n%6s%12s%12s\n",
             step, "atom", "eun", "total" );
    fprintf( out_control->eval, "step: %d\n%6s%6s%6s%12s%12s%12s%12s%12s%12s\n",
             step, "atom1", "atom2", "atom3", "angle", "theta0",
             "bo(12)", "bo(23)", "eval", "total" );
    fprintf( out_control->epen, "step: %d\n%6s%6s%6s%12s%12s%12s%12s%12s\n",
             step, "atom1", "atom2", "atom3", "angle", "bo(12)", "bo(23)",
             "epen", "total" );
    fprintf( out_control->ecoa, "step: %d\n%6s%6s%6s%12s%12s%12s%12s%12s\n",
             step, "atom1", "atom2", "atom3", "angle", "bo(12)", "bo(23)",
             "ecoa", "total" );
    fprintf( out_control->ehb,  "step: %d\n%6s%6s%6s%12s%12s%12s%12s%12s\n",
             step, "atom1", "atom2", "atom3", "r(23)", "angle", "bo(12)",
             "ehb", "total" );
    fprintf( out_control->etor, "step: %d\n%6s%6s%6s%6s%12s%12s%12s%12s\n",
             step, "atom1", "atom2", "atom3", "atom4", "phi", "bo(23)",
             "etor", "total" );
    fprintf( out_control->econ, "step:%d\n%6s%6s%6s%6s%12s%12s%12s%12s%12s%12s\n",
             step, "atom1", "atom2", "atom3", "atom4",
             "phi", "bo(12)", "bo(23)", "bo(34)", "econ", "total" );
}


void Debug_Marker_Nonbonded( output_controls *out_control, int step )
{
    fprintf( out_control->evdw, "step: %d\n%6s%6s%12s%12s%12s\n",
             step, "atom1", "atom2", "r12", "evdw", "total" );
    fprintf( out_control->ecou, "step: %d\n%6s%6s%12s%12s%12s%12s%12s\n",
             step, "atom1", "atom2", "r12", "q1", "q2", "ecou", "total" );
}

#endif


#if defined(TEST_FORCES)
void Dummy_Printer( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control )
{
}


void Print_Bond_Orders( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control )
{
    int i, pj, pk;
    bond_order_data *bo_ij;
    dbond_data *dbo_k;
    reax_list *bonds;
    reax_list *dBOs;

    bonds = lists[BONDS];
    dBOs = lists[DBOS];

    /* bond orders */
    fprintf( out_control->fbo, "step: %d\n", data->step );
    fprintf( out_control->fbo, "%6s%6s%12s%12s%12s%12s%12s\n",
             "atom1", "atom2", "r_ij", "total_bo", "bo_s", "bo_p", "bo_pp" );

    for ( i = 0; i < system->N; ++i )
    {
        for ( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj )
        {
            bo_ij = &bonds->bond_list[pj].bo_data;
            fprintf( out_control->fbo,
                     "%6d%6d%24.15e%24.15e%24.15e%24.15e%24.15e\n",
                     system->my_atoms[i].orig_id,
                     system->my_atoms[bonds->bond_list[pj].nbr].orig_id,
                     bonds->bond_list[pj].d,
                     bo_ij->BO, bo_ij->BO_s, bo_ij->BO_pi, bo_ij->BO_pi2 );
        }
    }

    /* derivatives of bond orders */
    fprintf( out_control->fdbo, "step: %d\n", data->step );
    fprintf( out_control->fdbo, "%6s%6s%6s%24s%24s%24s\n",
             "atom1", "atom2", "atom2", "dBO", "dBOpi", "dBOpi2" );
    for ( i = 0; i < system->N; ++i )
    {
        for ( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj )
        {
            /* fprintf( out_control->fdbo, "%6d %6d\tstart: %6d\tend: %6d\n",
            system->my_atoms[i].orig_id,
             system->my_atoms[bonds->bond_list[pj].nbr].orig_id,
             Start_Index( pj, dBOs ), End_Index( pj, dBOs ) ); */
            for ( pk = Start_Index(pj, dBOs); pk < End_Index(pj, dBOs); ++pk )
            {
                dbo_k = &dBOs->dbo_list[pk];
                fprintf( out_control->fdbo, "%6d%6d%6d%24.15e%24.15e%24.15e\n",
                         system->my_atoms[i].orig_id,
                         system->my_atoms[bonds->bond_list[pj].nbr].orig_id,
                         system->my_atoms[dbo_k->wrt].orig_id,
                         dbo_k->dBO[0], dbo_k->dBO[1], dbo_k->dBO[2] );

                fprintf( out_control->fdbo, "%6d%6d%6d%24.15e%24.15e%24.15e\n",
                         system->my_atoms[i].orig_id,
                         system->my_atoms[bonds->bond_list[pj].nbr].orig_id,
                         system->my_atoms[dbo_k->wrt].orig_id,
                         dbo_k->dBOpi[0], dbo_k->dBOpi[1], dbo_k->dBOpi[2] );

                fprintf( out_control->fdbo, "%6d%6d%6d%24.15e%24.15e%24.15e\n",
                         system->my_atoms[i].orig_id,
                         system->my_atoms[bonds->bond_list[pj].nbr].orig_id,
                         system->my_atoms[dbo_k->wrt].orig_id,
                         dbo_k->dBOpi2[0], dbo_k->dBOpi2[1], dbo_k->dBOpi2[2] );
            }
        }
    }
}


void Print_Forces( FILE *f, storage *workspace, int N, int step )
{
    int i;

    fprintf( f, "step: %d\n", step );
    for ( i = 0; i < N; ++i )
        //fprintf( f, "%6d %23.15e %23.15e %23.15e\n",
        //fprintf( f, "%6d%12.6f%12.6f%12.6f\n",
        fprintf( f, "%6d %19.9e %19.9e %19.9e\n",
                 workspace->id_all[i], workspace->f_all[i][0],
                 workspace->f_all[i][1], workspace->f_all[i][2] );
}


void Print_Force_Files( reax_system *system, control_params *control,
                        simulation_data *data, storage *workspace,
                        reax_list **lists, output_controls *out_control,
                        mpi_datatypes *mpi_data )
{
    int i, d;

    Coll_ids_at_Master( system, workspace, mpi_data );

    Print_Bond_Orders( system, control, data, workspace, lists, out_control );

    Coll_rvecs_at_Master( system, workspace, mpi_data, workspace->f_be );
    if ( system->my_rank == MASTER_NODE )
        Print_Forces( out_control->fbond, workspace, system->bigN, data->step );

    Coll_rvecs_at_Master( system, workspace, mpi_data, workspace->f_lp );
    if ( system->my_rank == MASTER_NODE )
        Print_Forces( out_control->flp, workspace, system->bigN, data->step );

    Coll_rvecs_at_Master( system, workspace, mpi_data, workspace->f_ov );
    if ( system->my_rank == MASTER_NODE )
        Print_Forces( out_control->fov, workspace, system->bigN, data->step );

    Coll_rvecs_at_Master( system, workspace, mpi_data, workspace->f_un );
    if ( system->my_rank == MASTER_NODE )
        Print_Forces( out_control->fun, workspace, system->bigN, data->step );

    Coll_rvecs_at_Master( system, workspace, mpi_data, workspace->f_ang );
    if ( system->my_rank == MASTER_NODE )
        Print_Forces( out_control->fang, workspace, system->bigN, data->step );

    Coll_rvecs_at_Master( system, workspace, mpi_data, workspace->f_coa );
    if ( system->my_rank == MASTER_NODE )
        Print_Forces( out_control->fcoa, workspace, system->bigN, data->step );

    Coll_rvecs_at_Master( system, workspace, mpi_data, workspace->f_pen );
    if ( system->my_rank == MASTER_NODE )
        Print_Forces( out_control->fpen, workspace, system->bigN, data->step );

    Coll_rvecs_at_Master( system, workspace, mpi_data, workspace->f_tor );
    if ( system->my_rank == MASTER_NODE )
        Print_Forces( out_control->ftor, workspace, system->bigN, data->step );

    Coll_rvecs_at_Master( system, workspace, mpi_data, workspace->f_con );
    if ( system->my_rank == MASTER_NODE )
        Print_Forces( out_control->fcon, workspace, system->bigN, data->step );

    Coll_rvecs_at_Master( system, workspace, mpi_data, workspace->f_hb );
    if ( system->my_rank == MASTER_NODE )
        Print_Forces( out_control->fhb, workspace, system->bigN, data->step );

    Coll_rvecs_at_Master( system, workspace, mpi_data, workspace->f_vdw );
    if ( system->my_rank == MASTER_NODE )
        Print_Forces( out_control->fvdw, workspace, system->bigN, data->step );

    Coll_rvecs_at_Master( system, workspace, mpi_data, workspace->f_ele );
    if ( system->my_rank == MASTER_NODE )
        Print_Forces( out_control->fele, workspace, system->bigN, data->step );

    Coll_rvecs_at_Master( system, workspace, mpi_data, workspace->f );
    if ( system->my_rank == MASTER_NODE )
        Print_Forces( out_control->ftot, workspace, system->bigN, data->step );

    for ( i = 0; i < system->n; ++i )
    {
        for ( d = 0; d < 3; ++d )
            workspace->f_tot[i][d] = workspace->f_be[i][d] +
                                     workspace->f_lp[i][d] + workspace->f_ov[i][d] + workspace->f_un[i][d] +
                                     workspace->f_ang[i][d] + workspace->f_pen[i][d] + workspace->f_coa[i][d] +
                                     workspace->f_tor[i][d] + workspace->f_con[i][d] +
                                     workspace->f_vdw[i][d] + workspace->f_ele[i][d] +
                                     workspace->f_hb[i][d];
    }

    Coll_rvecs_at_Master( system, workspace, mpi_data, workspace->f_tot );
    if ( system->my_rank == MASTER_NODE )
        Print_Forces( out_control->fcomp, workspace, system->bigN, data->step );
}
#endif


#if defined(TEST_FORCES) || defined(TEST_ENERGY)
/* Print far neighbors list in adjacency list format */
void Print_Far_Neighbors_List( reax_system *system, reax_list **lists,
        control_params *control, simulation_data *data,
        output_controls *out_control )
{
    int i, j, id_i, id_j, nbr, natoms;
    int num;
    int temp[500];
    reax_list *far_nbrs;

    num = 0;
    far_nbrs = lists[FAR_NBRS];

    fprintf( out_control->flist, "step: %d\n", data->step );
    fprintf( out_control->flist, "%6s\t%-38s\n", "atom", "Far_nbrs_list");

    natoms = system->n;
    for ( i = 0; i < natoms; ++i )
    {
        id_i = system->my_atoms[i].orig_id;
        fprintf( out_control->flist, "%6d:", id_i );
        num = 0;

        for ( j = Start_Index(i, far_nbrs); j < End_Index(i, far_nbrs); ++j )
        {
            nbr = far_nbrs->far_nbr_list.nbr[j];
            id_j = system->my_atoms[nbr].orig_id;
            temp[num++] = id_j;
        }

        qsort( &temp, num, sizeof(int), fn_qsort_intcmp );

        for ( j = 0; j < num; j++ )
        {
            fprintf( out_control->flist, "%6d", temp[j] );
        }
        fprintf( out_control->flist, "\n" );
    }
}


void Print_Bond_List( reax_system *system, control_params *control,
        simulation_data *data, reax_list **lists,
        output_controls *out_control)
{
    int i, j, id_i, id_j, nbr, pj;
    reax_list *bonds = lists[BONDS];

    int temp[500];
    int num = 0;

    fprintf( out_control->blist, "step: %d\n", data->step );
    fprintf( out_control->blist, "%6s\t%-38s\n", "atom", "Bond_list" );

    /* bond list */
    for ( i = 0; i < system->n; ++i )
    {
        num = 0;
        id_i = system->my_atoms[i].orig_id;
        fprintf( out_control->blist, "%6d:", id_i );
        for ( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj )
        {
            nbr = bonds->bond_list[pj].nbr;
            id_j = system->my_atoms[nbr].orig_id;
            if ( id_i < id_j )
            {
                temp[num++] = id_j;
            }
        }

        qsort( &temp, num, sizeof(int), fn_qsort_intcmp );
        for ( j = 0; j < num; j++ )
        {
            fprintf( out_control->blist, "%6d", temp[j] );
        }
        fprintf( out_control->blist, "\n" );
    }
}


#endif


#if defined(OLD_VERSION)
void Print_Init_Atoms( reax_system *system, storage *workspace )
{
    int i;

    fprintf( stderr, "p%d had %d atoms\n",
             system->my_rank, workspace->init_cnt );

    for ( i = 0; i < workspace->init_cnt; ++i )
        fprintf( stderr, "p%d, atom%d: %d  %s  %8.3f %8.3f %8.3f\n",
                 system->my_rank, i,
                 workspace->init_atoms[i].type, workspace->init_atoms[i].name,
                 workspace->init_atoms[i].x[0],
                 workspace->init_atoms[i].x[1],
                 workspace->init_atoms[i].x[2] );
}
#endif //OLD_VERSION


/*void Print_Bond_Forces( reax_system *system, control_params *control,
            simulation_data *data, storage *workspace,
            reax_list **lists, output_controls *out_control )
{
  int i;

  fprintf( out_control->fbond, "step: %d\n", data->step );
  fprintf( out_control->fbond, "%6s%24s%24s%24s\n",
       "atom", "f_be[0]", "f_be[1]", "f_be[2]" );

  for( i = 0; i < system->bigN; ++i )
    fprintf(out_control->fbond, "%6d%24.15e%24.15e%24.15e\n",
        system->my_atoms[i].orig_id,
        workspace->f_all[i][0], workspace->f_all[i][1],
        workspace->f_all[i][2]);
}

void Print_LonePair_Forces( reax_system *system, control_params *control,
                simulation_data *data, storage *workspace,
                reax_list **lists, output_controls *out_control )
{
  int i;

  fprintf( out_control->flp, "step: %d\n", data->step );
  fprintf( out_control->flp, "%6s%24s\n", "atom", "f_lonepair" );

  for( i = 0; i < system->bigN; ++i )
    fprintf(out_control->flp, "%6d%24.15e%24.15e%24.15e\n",
        system->my_atoms[i].orig_id,
        workspace->f_all[i][0], workspace->f_all[i][1],
        workspace->f_all[i][2]);
}


void Print_OverCoor_Forces( reax_system *system, control_params *control,
                simulation_data *data, storage *workspace,
                reax_list **lists, output_controls *out_control )
{
  int i;

  fprintf( out_control->fov, "step: %d\n", data->step );
  fprintf( out_control->fov, "%6s%-38s%-38s%-38s\n",
       "atom","f_over[0]", "f_over[1]", "f_over[2]" );

  for( i = 0; i < system->bigN; ++i )
    fprintf( out_control->fov,
         "%6d %24.15e%24.15e%24.15e 0 0 0\n",
         system->my_atoms[i].orig_id,
         workspace->f_all[i][0], workspace->f_all[i][1],
         workspace->f_all[i][2] );
}


void Print_UnderCoor_Forces( reax_system *system, control_params *control,
                 simulation_data *data, storage *workspace,
                 reax_list **lists, output_controls *out_control )
{
  int i;

  fprintf( out_control->fun, "step: %d\n", data->step );
  fprintf( out_control->fun, "%6s%-38s%-38s%-38s\n",
       "atom","f_under[0]", "f_under[1]", "f_under[2]" );

  for( i = 0; i < system->bigN; ++i )
    fprintf( out_control->fun,
         "%6d %24.15e%24.15e%24.15e 0 0 0\n",
         system->my_atoms[i].orig_id,
         workspace->f_all[i][0], workspace->f_all[i][1],
         workspace->f_all[i][2] );
}


void Print_ValAngle_Forces( reax_system *system, control_params *control,
                simulation_data *data, storage *workspace,
                reax_list **lists, output_controls *out_control )
{
  int j;

  fprintf( out_control->f3body, "step: %d\n", data->step );
  fprintf( out_control->f3body, "%6s%-37s%-37s%-37s%-38s\n",
       "atom", "3-body total", "f_ang", "f_pen", "f_coa" );

  for( j = 0; j < system->N; ++j ){
    if( rvec_isZero(workspace->f_pen[j]) && rvec_isZero(workspace->f_coa[j]) )
      fprintf( out_control->f3body,
           "%6d %24.15e%24.15e%24.15e  0 0 0  0 0 0\n",
           system->my_atoms[j].orig_id,
           workspace->f_ang[j][0], workspace->f_ang[j][1],
           workspace->f_ang[j][2] );
    else if( rvec_isZero(workspace->f_coa[j]) )
      fprintf( out_control->f3body,
           "%6d %24.15e%24.15e%24.15e %24.15e%24.15e%24.15e "   \
           "%24.15e%24.15e%24.15e\n",
           system->my_atoms[j].orig_id,
           workspace->f_ang[j][0] + workspace->f_pen[j][0],
           workspace->f_ang[j][1] + workspace->f_pen[j][1],
           workspace->f_ang[j][2] + workspace->f_pen[j][2],
           workspace->f_ang[j][0], workspace->f_ang[j][1],
           workspace->f_ang[j][2],
           workspace->f_pen[j][0], workspace->f_pen[j][1],
           workspace->f_pen[j][2] );
    else{
      fprintf( out_control->f3body, "%6d %24.15e%24.15e%24.15e ",
         system->my_atoms[j].orig_id,
           workspace->f_ang[j][0] + workspace->f_pen[j][0] +
           workspace->f_coa[j][0],
           workspace->f_ang[j][1] + workspace->f_pen[j][1] +
           workspace->f_coa[j][1],
           workspace->f_ang[j][2] + workspace->f_pen[j][2] +
           workspace->f_coa[j][2] );

      fprintf( out_control->f3body,
           "%24.15e%24.15e%24.15e %24.15e%24.15e%24.15e "\
           "%24.15e%24.15e%24.15e\n",
           workspace->f_ang[j][0], workspace->f_ang[j][1],
           workspace->f_ang[j][2],
           workspace->f_pen[j][0], workspace->f_pen[j][1],
           workspace->f_pen[j][2],
           workspace->f_coa[j][0], workspace->f_coa[j][1],
           workspace->f_coa[j][2] );
    }
  }
}


void Print_Hydrogen_Bond_Forces( reax_system *system, control_params *control,
                 simulation_data *data, storage *workspace,
                 reax_list **lists, output_controls *out_control)
{
  int j;

  fprintf( out_control->fhb, "step: %d\n", data->step );
  fprintf( out_control->fhb, "%6s\t%-38s\n", "atom", "f_hb[0,1,2]" );

  for( j = 0; j < system->N; ++j )
    fprintf(out_control->fhb, "%6d%24.15e%24.15e%24.15e\n",
         system->my_atoms[j].orig_id,
         workspace->f_hb[j][0],
         workspace->f_hb[j][1],
         workspace->f_hb[j][2] );
}


void Print_Four_Body_Forces( reax_system *system, control_params *control,
                 simulation_data *data, storage *workspace,
                 reax_list **lists, output_controls *out_control )
{
  int j;

  fprintf( out_control->f4body, "step: %d\n", data->step );
  fprintf( out_control->f4body, "%6s\t%-38s%-38s%-38s\n",
       "atom", "4-body total", "f_tor", "f_con" );

  for( j = 0; j < system->N; ++j ){
    if( !rvec_isZero( workspace->f_con[j] ) )
      fprintf( out_control->f4body,
           "%6d %24.15e%24.15e%24.15e %24.15e%24.15e%24.15e "\
           "%24.15e%24.15e%24.15e\n",
         system->my_atoms[j].orig_id,
           workspace->f_tor[j][0] + workspace->f_con[j][0],
           workspace->f_tor[j][1] + workspace->f_con[j][1],
           workspace->f_tor[j][2] + workspace->f_con[j][2],
           workspace->f_tor[j][0], workspace->f_tor[j][1],
           workspace->f_tor[j][2],
           workspace->f_con[j][0], workspace->f_con[j][1],
           workspace->f_con[j][2] );
    else
      fprintf( out_control->f4body,
           "%6d %24.15e%24.15e%24.15e  0 0 0\n",
           system->my_atoms[j].orig_id, workspace->f_tor[j][0],
           workspace->f_tor[j][1], workspace->f_tor[j][2] );
  }

}


void Print_vdW_Coulomb_Forces( reax_system *system, control_params *control,
                   simulation_data *data, storage *workspace,
                   reax_list **lists, output_controls *out_control )
{
  int  i;

  return;

  fprintf( out_control->fnonb, "step: %d\n", data->step );
  fprintf( out_control->fnonb, "%6s\t%-38s%-38s%-38s\n",
       "atom", "nonbonded_total[0,1,2]", "f_vdw[0,1,2]", "f_ele[0,1,2]" );

  for( i = 0; i < system->N; ++i )
    fprintf( out_control->fnonb,
         "%6d%24.15e%24.15e%24.15e%24.15e%24.15e%24.15e%24.15e%24.15e%24.15e\n",
         system->my_atoms[i].orig_id,
         workspace->f_vdw[i][0] + workspace->f_ele[i][0],
         workspace->f_vdw[i][1] + workspace->f_ele[i][1],
         workspace->f_vdw[i][2] + workspace->f_ele[i][2],
         workspace->f_vdw[i][0],
         workspace->f_vdw[i][1],
         workspace->f_vdw[i][2],
         workspace->f_ele[i][0],
         workspace->f_ele[i][1],
         workspace->f_ele[i][2] );
}


void Print_Total_Force( reax_system *system, control_params *control,
            simulation_data *data, storage *workspace,
            reax_list **lists, output_controls *out_control )
{
  int    i;

  return;

  fprintf( out_control->ftot, "step: %d\n", data->step );
  fprintf( out_control->ftot, "%6s\t%-38s\n", "atom", "atom.f[0,1,2]");

  for( i = 0; i < system->n; ++i )
    fprintf( out_control->ftot, "%6d%24.15e%24.15e%24.15e\n",
         system->my_atoms[i].orig_id,
         system->my_atoms[i].f[0],
         system->my_atoms[i].f[1],
         system->my_atoms[i].f[2] );
}


void Compare_Total_Forces( reax_system *system, control_params *control,
               simulation_data *data, storage *workspace,
               reax_list **lists, output_controls *out_control )
{
  int i;

  return;

  fprintf( out_control->ftot2, "step: %d\n", data->step );
  fprintf( out_control->ftot2, "%6s\t%-38s%-38s\n",
       "atom", "f_total[0,1,2]", "test_force_total[0,1,2]" );

  for( i = 0; i < system->N; ++i )
    fprintf( out_control->ftot2, "%6d%24.15e%24.15e%24.15e%24.15e%24.15e%24.15e\n",
         system->my_atoms[i].orig_id,
         system->my_atoms[i].f[0],
         system->my_atoms[i].f[1],
         system->my_atoms[i].f[2],
         workspace->f_be[i][0] + workspace->f_lp[i][0] +
         workspace->f_ov[i][0] + workspace->f_un[i][0] +
         workspace->f_ang[i][0]+ workspace->f_pen[i][0]+
         workspace->f_coa[i][0]+ + workspace->f_hb[i][0] +
         workspace->f_tor[i][0] + workspace->f_con[i][0] +
         workspace->f_vdw[i][0] + workspace->f_ele[i][0],
         workspace->f_be[i][1] + workspace->f_lp[i][1] +
         workspace->f_ov[i][1] + workspace->f_un[i][1] +
             workspace->f_ang[i][1]+ workspace->f_pen[i][1]+
         workspace->f_coa[i][1]+ + workspace->f_hb[i][1] +
         workspace->f_tor[i][1] + workspace->f_con[i][1] +
         workspace->f_vdw[i][1] + workspace->f_ele[i][1],
         workspace->f_be[i][2] + workspace->f_lp[i][2] +
         workspace->f_ov[i][2] + workspace->f_un[i][2] +
             workspace->f_ang[i][2]+ workspace->f_pen[i][2] +
         workspace->f_coa[i][2]+ + workspace->f_hb[i][2] +
         workspace->f_tor[i][2] + workspace->f_con[i][2] +
         workspace->f_vdw[i][2] + workspace->f_ele[i][2] );
}*/

/*void Init_Force_Test_Functions( )
{
  Print_Interactions[0] = Print_Bond_Orders;
  Print_Interactions[1] = Print_Bond_Forces;
  Print_Interactions[2] = Print_LonePair_Forces;
  Print_Interactions[3] = Print_OverUnderCoor_Forces;
  Print_Interactions[4] = Print_Three_Body_Forces;
  Print_Interactions[5] = Print_Four_Body_Forces;
  Print_Interactions[6] = Print_Hydrogen_Bond_Forces;
  Print_Interactions[7] = Print_vdW_Coulomb_Forces;
  Print_Interactions[8] = Print_Total_Force;
  Print_Interactions[9] = Compare_Total_Forces;
  }*/
