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
  #include "allocate.h"
  #include "list.h"
  #include "reset_tools.h"
  #include "tool_box.h"
  #include "vector.h"
#elif defined(LAMMPS_REAX)
  #include "reax_allocate.h"
  #include "reax_list.h"
  #include "reax_reset_tools.h"
  #include "reax_tool_box.h"
  #include "reax_vector.h"
#endif

#include "index_utils.h"


void Init_Matrix_Row_Indices( sparse_matrix *H, int * max_row_entries )
{
    int i;

    /* exclusive prefix sum on max_row_entries replaces start,
     * set end indices to the same as start indices for safety */
    H->start[0] = 0;
    H->end[0] = 0;
    for ( i = 1; i < H->n; ++i )
    {
        H->start[i] = H->start[i - 1] + max_row_entries[i - 1];
        H->end[i] = H->start[i];
    }
}


/* allocate space for my_atoms
 * important: we cannot know the exact number of atoms that will fall into a
 * process's box throughout the whole simulation. therefore
 * we need to make upper bound estimates for various data structures */
void PreAllocate_Space( reax_system *system, control_params *control,
        storage *workspace )
{
    /* determine capacity based on box vol & est atom volume */
    system->local_cap = MAX( (int)(system->n * SAFE_ZONE), MIN_CAP );
    system->total_cap = MAX( (int)(system->N * SAFE_ZONE), MIN_CAP );

#if defined(DEBUG)||defined(__CUDA_DEBUG_LOG__)
    fprintf( stderr, "p%d: local_cap=%d total_cap=%d\n",
            system->my_rank, system->local_cap, system->total_cap );
#endif

    system->my_atoms = scalloc( system->total_cap, sizeof(reax_atom),
            "PreAllocate_Space::system->my_atoms" );

    /* space for keeping restriction info, if any */
    if ( control->restrict_bonds )
    {
        workspace->restricted = scalloc( system->local_cap, sizeof(int),
                "PreAllocate_Space::workspace->restricted_atoms" );
        workspace->restricted_list = scalloc( system->local_cap * MAX_RESTRICT,
                sizeof(int), "PreAllocate_Space::workspace->restricted_list" );
    }
}


/*************       system        *************/
void ReAllocate_System( reax_system *system, int local_cap, int total_cap )
{
    system->my_atoms = srealloc( system->my_atoms, sizeof(reax_atom) * total_cap,
            "ReAllocate_System::system->my_atoms" );

    /* list management */
    system->far_nbrs = srealloc( system->far_nbrs, sizeof(int) * total_cap,
            "ReAllocate_System::system->far_nbrs" );
    system->max_far_nbrs = srealloc( system->max_far_nbrs, sizeof(int) * total_cap,
            "ReAllocate_System::system->max_far_nbrs" );

    system->bonds = srealloc( system->bonds, sizeof(int) * total_cap,
            "ReAllocate_System::system->bonds" );
    system->max_bonds = srealloc( system->max_bonds, sizeof(int) * total_cap,
            "ReAllocate_System::system->max_bonds" );

    system->hbonds = srealloc( system->hbonds, sizeof(int) * total_cap,
            "ReAllocate_System::system->hbonds" );
    system->max_hbonds = srealloc( system->max_hbonds, sizeof(int) * total_cap,
            "ReAllocate_System::system->max_hbonds" );

    system->cm_entries = srealloc( system->cm_entries, sizeof(int) * local_cap,
            "ReAllocate_System::system->cm_entries" );
    system->max_cm_entries = srealloc( system->max_cm_entries, sizeof(int) * local_cap,
            "ReAllocate_System::system->max_cm_entries" );
}


/*************       workspace        *************/
void DeAllocate_Workspace( control_params *control, storage *workspace )
{
    int i;

    if ( workspace->allocated == FALSE )
    {
        return;
    }

    workspace->allocated = FALSE;

    /* communication storage */
    for ( i = 0; i < MAX_NBRS; ++i )
    {
        sfree( workspace->tmp_dbl[i], "DeAllocate_Workspace::tmp_dbl[i]" );
        sfree( workspace->tmp_rvec[i], "DeAllocate_Workspace::tmp_rvec[i]" );
        sfree( workspace->tmp_rvec2[i], "DeAllocate_Workspace::tmp_rvec2[i]" );
    }

    /* bond order storage */
    sfree( workspace->within_bond_box, "DeAllocate_Workspace::skin" );
    sfree( workspace->total_bond_order, "DeAllocate_Workspace::total_bo" );
    sfree( workspace->Deltap, "DeAllocate_Workspace::Deltap" );
    sfree( workspace->Deltap_boc, "DeAllocate_Workspace::Deltap_boc" );
    sfree( workspace->dDeltap_self, "DeAllocate_Workspace::dDeltap_self" );
    sfree( workspace->Delta, "DeAllocate_Workspace::Delta" );
    sfree( workspace->Delta_lp, "DeAllocate_Workspace::Delta_lp" );
    sfree( workspace->Delta_lp_temp, "DeAllocate_Workspace::Delta_lp_temp" );
    sfree( workspace->dDelta_lp, "DeAllocate_Workspace::dDelta_lp" );
    sfree( workspace->dDelta_lp_temp, "DeAllocate_Workspace::dDelta_lp_temp" );
    sfree( workspace->Delta_e, "DeAllocate_Workspace::Delta_e" );
    sfree( workspace->Delta_boc, "DeAllocate_Workspace::Delta_boc" );
    sfree( workspace->nlp, "DeAllocate_Workspace::nlp" );
    sfree( workspace->nlp_temp, "DeAllocate_Workspace::nlp_temp" );
    sfree( workspace->Clp, "DeAllocate_Workspace::Clp" );
    sfree( workspace->vlpex, "DeAllocate_Workspace::vlpex" );
    sfree( workspace->bond_mark, "DeAllocate_Workspace::bond_mark" );

    /* QEq storage */
    sfree( workspace->Hdia_inv, "DeAllocate_Workspace::Hdia_inv" );
    sfree( workspace->b_s, "DeAllocate_Workspace::b_s" );
    sfree( workspace->b_t, "DeAllocate_Workspace::b_t" );
    sfree( workspace->b_prc, "DeAllocate_Workspace::b_prc" );
    sfree( workspace->b_prm, "DeAllocate_Workspace::b_prm" );
    sfree( workspace->s, "DeAllocate_Workspace::s" );
    sfree( workspace->t, "DeAllocate_Workspace::t" );
    sfree( workspace->droptol, "DeAllocate_Workspace::droptol" );
    sfree( workspace->b, "DeAllocate_Workspace::b" );
    sfree( workspace->x, "DeAllocate_Workspace::x" );

    /* GMRES storage */
    /*
    for( i = 0; i < RESTART+1; ++i ) {
      sfree( workspace->h[i], "DeAllocate_Workspace::h[i]" );
      sfree( workspace->v[i], "DeAllocate_Workspace::v[i]" );
    }
    */
    sfree( workspace->h, "DeAllocate_Workspace::h" );
    sfree( workspace->v, "DeAllocate_Workspace::v" );
    sfree( workspace->y, "DeAllocate_Workspace::y" );
    sfree( workspace->z, "DeAllocate_Workspace::z" );
    sfree( workspace->g, "DeAllocate_Workspace::g" );
    sfree( workspace->hs, "DeAllocate_Workspace::hs" );
    sfree( workspace->hc, "DeAllocate_Workspace::hc" );
    /* CG storage */
    sfree( workspace->r, "DeAllocate_Workspace::r" );
    sfree( workspace->d, "DeAllocate_Workspace::d" );
    sfree( workspace->q, "DeAllocate_Workspace::q" );
    sfree( workspace->p, "DeAllocate_Workspace::p" );
    sfree( workspace->r2, "DeAllocate_Workspace::r2" );
    sfree( workspace->d2, "DeAllocate_Workspace::d2" );
    sfree( workspace->q2, "DeAllocate_Workspace::q2" );
    sfree( workspace->p2, "DeAllocate_Workspace::p2" );

    /* integrator */
    // sfree( workspace->f_old, "DeAllocate_Workspace::f_old" );
    sfree( workspace->v_const, "DeAllocate_Workspace::v_const" );

    /*workspace->realloc.far_nbrs = -1;
      workspace->realloc.Htop = -1;
      workspace->realloc.hbonds = -1;
      workspace->realloc.bonds = -1;
      workspace->realloc.num_3body = -1;
      workspace->realloc.gcell_atoms = -1;*/

    /* storage for analysis */
    if ( control->molecular_analysis || control->diffusion_coef )
    {
        sfree( workspace->mark, "DeAllocate_Workspace::mark" );
        sfree( workspace->old_mark, "DeAllocate_Workspace::old_mark" );
    }

    if ( control->diffusion_coef )
    {
        sfree( workspace->x_old, "DeAllocate_Workspace::x_old" );
    }

    /* force-related storage */
    sfree( workspace->f, "DeAllocate_Workspace::f" );
    sfree( workspace->CdDelta, "DeAllocate_Workspace::CdDelta" );

#ifdef TEST_FORCES
    sfree(workspace->dDelta, "DeAllocate_Workspace::dDelta" );
    sfree( workspace->f_ele, "DeAllocate_Workspace::f_ele" );
    sfree( workspace->f_vdw, "DeAllocate_Workspace::f_vdw" );
    sfree( workspace->f_bo, "DeAllocate_Workspace::f_bo" );
    sfree( workspace->f_be, "DeAllocate_Workspace::f_be" );
    sfree( workspace->f_lp, "DeAllocate_Workspace::f_lp" );
    sfree( workspace->f_ov, "DeAllocate_Workspace::f_ov" );
    sfree( workspace->f_un, "DeAllocate_Workspace::f_un" );
    sfree( workspace->f_ang, "DeAllocate_Workspace::f_ang" );
    sfree( workspace->f_coa, "DeAllocate_Workspace::f_coa" );
    sfree( workspace->f_pen, "DeAllocate_Workspace::f_pen" );
    sfree( workspace->f_hb, "DeAllocate_Workspace::f_hb" );
    sfree( workspace->f_tor, "DeAllocate_Workspace::f_tor" );
    sfree( workspace->f_con, "DeAllocate_Workspace::f_con" );
    sfree( workspace->f_tot, "DeAllocate_Workspace::f_tot" );

    sfree( workspace->rcounts, "DeAllocate_Workspace::rcounts" );
    sfree( workspace->displs, "DeAllocate_Workspace::displs" );
    sfree( workspace->id_all, "DeAllocate_Workspace::id_all" );
    sfree( workspace->f_all, "DeAllocate_Workspace::f_all" );
#endif
}


void Allocate_Workspace( reax_system *system, control_params *control,
        storage *workspace, int local_cap, int total_cap )
{
    int i, total_real, total_rvec, local_rvec;

    workspace->allocated = TRUE;
    total_real = sizeof(real) * total_cap;
    total_rvec = sizeof(rvec) * total_cap;
    local_rvec = sizeof(rvec) * local_cap;

    /* communication storage */
    for ( i = 0; i < MAX_NBRS; ++i )
    {
        workspace->tmp_dbl[i] = scalloc( total_cap, sizeof(real),
                "Allocate_Workspace::tmp_dbl[i]" );
        workspace->tmp_rvec[i] = scalloc( total_cap, sizeof(rvec),
                "Allocate_Workspace::tmp_rvec[i]" );
        workspace->tmp_rvec2[i] = scalloc( total_cap, sizeof(rvec2),
                "Allocate_Workspace::tmp_rvec2[i]" );
    }

    /* bond order related storage  */
    workspace->within_bond_box = scalloc( total_cap, sizeof(int),
            "Allocate_Workspace::skin" );
    workspace->total_bond_order = smalloc( total_real, "Allocate_Workspace::total_bo" );
    workspace->Deltap = smalloc( total_real, "Allocate_Workspace::Deltap" );
    workspace->Deltap_boc = smalloc( total_real, "Allocate_Workspace::Deltap_boc" );
    workspace->dDeltap_self = smalloc( total_rvec, "Allocate_Workspace::dDeltap_self" );
    workspace->Delta = smalloc( total_real, "Allocate_Workspace::Delta" );
    workspace->Delta_lp = smalloc( total_real, "Allocate_Workspace::Delta_lp" );
    workspace->Delta_lp_temp = smalloc( total_real, "Allocate_Workspace::Delta_lp_temp" );
    workspace->dDelta_lp = smalloc( total_real, "Allocate_Workspace::dDelta_lp" );
    workspace->dDelta_lp_temp = smalloc( total_real, "Allocate_Workspace::dDelta_lp_temp" );
    workspace->Delta_e = smalloc( total_real, "Allocate_Workspace::Delta_e" );
    workspace->Delta_boc = smalloc( total_real, "Allocate_Workspace::Delta_boc" );
    workspace->nlp = smalloc( total_real, "Allocate_Workspace::nlp" );
    workspace->nlp_temp = smalloc( total_real, "Allocate_Workspace::nlp_temp" );
    workspace->Clp = smalloc( total_real, "Allocate_Workspace::Clp" );
    workspace->CdDelta = scalloc( total_cap, sizeof(real), "Allocate_Workspace::CdDelta" );
    workspace->vlpex = smalloc( total_real, "Allocate_Workspace::vlpex" );
    workspace->bond_mark = scalloc( total_cap, sizeof(int),
            "Allocate_Workspace::bond_mark" );

    /* charge method storage */
    switch ( control->charge_method )
    {
        case QEQ_CM:
            system->n_cm = system->N;
            break;
        case EE_CM:
            system->n_cm = system->N + 1;
            break;
        case ACKS2_CM:
            system->n_cm = 2 * system->N + 2;
            break;
        default:
            fprintf( stderr, "[ERROR] Unknown charge method type. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }

    workspace->Hdia_inv = scalloc( total_cap, sizeof(real),
            "Allocate_Workspace::Hdia_inv" );
    workspace->b_s = scalloc( total_cap, sizeof(real),
            "Allocate_Workspace::b_s" );
    workspace->b_t = scalloc( total_cap, sizeof(real),
            "Allocate_Workspace::b_t" );
    workspace->b_prc = scalloc( total_cap, sizeof(real),
            "Allocate_Workspace::b_prc" );
    workspace->b_prm = scalloc( total_cap, sizeof(real),
            "Allocate_Workspace::b_prm" );
    workspace->s = scalloc( total_cap, sizeof(real), "Allocate_Workspace::s" );
    workspace->t = scalloc( total_cap, sizeof(real), "Allocate_Workspace::t" );
    if ( control->cm_solver_pre_comp_type == ICHOLT_PC ||
            control->cm_solver_pre_comp_type == ILUT_PAR_PC )
    {
        workspace->droptol = scalloc( total_cap, sizeof(real),
                "Allocate_Workspace::droptol" );
    }
    workspace->b = scalloc( total_cap, sizeof(rvec2), "Allocate_Workspace::b" );
    workspace->x = scalloc( total_cap, sizeof(rvec2), "Allocate_Workspace::x" );

    switch ( control->cm_solver_type )
    {
        /* GMRES storage */
        case GMRES_S:
        case GMRES_H_S:
            workspace->y = scalloc( RESTART + 1, sizeof(real),
                    "Allocate_Workspace::y" );
            workspace->z = scalloc( RESTART + 1, sizeof(real),
                    "Allocate_Workspace::z" );
            workspace->g = scalloc( RESTART + 1, sizeof(real),
                    "Allocate_Workspace::g" );
            workspace->h = scalloc ( (RESTART + 1) * (RESTART + 1), sizeof(real),
                    "Allocate_Workspace::h");
            workspace->hs = scalloc( RESTART + 1, sizeof(real),
                    "Allocate_Workspace::hs" );
            workspace->hc = scalloc( RESTART + 1, sizeof(real),
                    "Allocate_Workspace::hc" );
            workspace->v = scalloc ( (RESTART + 1) * (RESTART + 1), sizeof(real),
                    "Allocate_Workspace::v");
            break;

        /* CG storage */
        case CG_S:
            workspace->r = scalloc( total_cap, sizeof(real),
                    "Allocate_Workspace::r" );
            workspace->d = scalloc( total_cap, sizeof(real),
                    "Allocate_Workspace::d" );
            workspace->q = scalloc( total_cap, sizeof(real),
                    "Allocate_Workspace::q" );
            workspace->p = scalloc( total_cap, sizeof(real),
                    "Allocate_Workspace::p" );
            workspace->r2 = scalloc( total_cap, sizeof(rvec2),
                    "Allocate_Workspace::r2" );
            workspace->d2 = scalloc( total_cap, sizeof(rvec2),
                    "Allocate_Workspace::d2" );
            workspace->q2 = scalloc( total_cap, sizeof(rvec2),
                    "Allocate_Workspace::q2" );
            workspace->p2 = scalloc( total_cap, sizeof(rvec2),
                    "Allocate_Workspace::p2" );
            break;

        case SDM_S:
            workspace->r = scalloc( total_cap, sizeof(real), "Allocate_Workspace::r" );
            workspace->d = scalloc( total_cap, sizeof(real), "Allocate_Workspace::d" );
            workspace->q = scalloc( total_cap, sizeof(real), "Allocate_Workspace::q" );
            workspace->p = scalloc( total_cap, sizeof(real), "Allocate_Workspace::p" );
            workspace->r2 = scalloc( total_cap, sizeof(rvec2), "Allocate_Workspace::r2" );
            workspace->d2 = scalloc( total_cap, sizeof(rvec2), "Allocate_Workspace::d2" );
            workspace->q2 = scalloc( total_cap, sizeof(rvec2), "Allocate_Workspace::q2" );
            workspace->p2 = scalloc( total_cap, sizeof(rvec2), "Allocate_Workspace::p2" );
            break;

        default:
            fprintf( stderr, "[ERROR] Unknown charge method linear solver type. Terminating...\n" );
            exit( UNKNOWN_OPTION );
            break;
    }

    /* integrator storage */
    workspace->v_const = smalloc( local_rvec, "Allocate_Workspace::v_const" );

    /* storage for analysis */
    if ( control->molecular_analysis || control->diffusion_coef )
    {
        workspace->mark = scalloc( local_cap, sizeof(int),
                "Allocate_Workspace::mark" );
        workspace->old_mark = scalloc( local_cap, sizeof(int),
                "Allocate_Workspace::old_mark" );
    }
    else
    {
        workspace->mark = NULL;
        workspace->old_mark = NULL;
    }

    if ( control->diffusion_coef )
    {
        workspace->x_old = scalloc( local_cap, sizeof(rvec),
                "Allocate_Workspace::x_old" );
    }
    else
    {
        workspace->x_old = NULL;
    }

    /* force related storage */
    workspace->f = scalloc( total_cap, sizeof(rvec),
            "Allocate_Workspace::f" );

#ifdef TEST_FORCES
    workspace->dDelta = smalloc( total_rvec, "Allocate_Workspace::dDelta" );
    workspace->f_ele = smalloc( total_rvec, "Allocate_Workspace::f_ele" );
    workspace->f_vdw = smalloc( total_rvec, "Allocate_Workspace::f_vdw" );
    workspace->f_bo = smalloc( total_rvec, "Allocate_Workspace::f_bo" );
    workspace->f_be = smalloc( total_rvec, "Allocate_Workspace::f_be" );
    workspace->f_lp = smalloc( total_rvec, "Allocate_Workspace::f_lp" );
    workspace->f_ov = smalloc( total_rvec, "Allocate_Workspace::f_ov" );
    workspace->f_un = smalloc( total_rvec, "Allocate_Workspace::f_un" );
    workspace->f_ang = smalloc( total_rvec, "Allocate_Workspace::f_ang" );
    workspace->f_coa = smalloc( total_rvec, "Allocate_Workspace::f_coa" );
    workspace->f_pen = smalloc( total_rvec, "Allocate_Workspace::f_pen" );
    workspace->f_hb = smalloc( total_rvec, "Allocate_Workspace::f_hb" );
    workspace->f_tor = smalloc( total_rvec, "Allocate_Workspace::f_tor" );
    workspace->f_con = smalloc( total_rvec, "Allocate_Workspace::f_con" );
    workspace->f_tot = smalloc( total_rvec, "Allocate_Workspace::f_tot" );

    if ( system->my_rank == MASTER_NODE )
    {
        workspace->rcounts = smalloc( sizeof(int) * system->nprocs,
                "Allocate_Workspace::rcounts" );
        workspace->displs = smalloc( sizeof(int) * system->nprocs,
                "Allocate_Workspace::displs" );
        workspace->id_all = smalloc( sizeof(int) * system->bigN,
                "Allocate_Workspace::id_all" );
        workspace->f_all = smalloc( sizeof(rvec) * system->bigN,
                "Allocate_Workspace::f_all" );
    }
    else
    {
        workspace->rcounts = NULL;
        workspace->displs = NULL;
        workspace->id_all = NULL;
        workspace->f_all = NULL;
    }
#endif
}


void Reallocate_Neighbor_List( reax_list *far_nbr_list, int n, int max_intrs )
{
    Delete_List( far_nbr_list );
    Make_List( n, max_intrs, TYP_FAR_NEIGHBOR, far_nbr_list );
}


void Allocate_Matrix( sparse_matrix *H, int n, int n_max, int m )
{
    H->n = n;
    H->n_max = n_max;
    H->m = m;

    H->start = smalloc( sizeof(int) * n_max, "Allocate_Matrix::start" );
    H->end = smalloc( sizeof(int) * n_max, "Allocate_Matrix::end" );
    H->entries = smalloc( sizeof(sparse_matrix_entry) * m, "Allocate_Matrix::entries" );
}


void Deallocate_Matrix( sparse_matrix *H )
{
    H->n = 0;
    H->n_max = 0;
    H->m = 0;

    sfree( H->start, "Deallocate_Matrix::start" );
    sfree( H->end, "Deallocate_Matrix::end" );
    sfree( H->entries, "Deallocate_Matrix::entries" );
}


static void Reallocate_Matrix( sparse_matrix *H, int n, int n_max, int m )
{
    Deallocate_Matrix( H );
    Allocate_Matrix( H, n, n_max, m );
}


void Reallocate_HBonds_List( reax_system *system, reax_list *hbond_list )
{
    Delete_List( hbond_list );
//    Make_List( system->Hcap, system->total_hbonds, TYP_HBOND, hbond_list );
    Make_List( system->total_cap, system->total_hbonds, TYP_HBOND, hbond_list );
}


void Reallocate_Bonds_List( reax_system *system, reax_list *bond_list )
{
    Delete_List( bond_list );
    Make_List( system->total_cap, system->total_bonds, TYP_BOND, bond_list );
}


/*************       grid        *************/
int Estimate_GCell_Population( reax_system* system, MPI_Comm comm )
{
    int d, i, j, k, l, max_atoms, my_max, all_max;
    ivec c;
    grid *g;
    grid_cell *gc;
    simulation_box *my_ext_box;
    reax_atom *atoms;

    my_ext_box = &system->my_ext_box;
    g = &system->my_grid;
    atoms = system->my_atoms;
    Reset_Grid( g );

    for ( l = 0; l < system->n; l++ )
    {
        for ( d = 0; d < 3; ++d )
        {
            c[d] = (int)((atoms[l].x[d] - my_ext_box->min[d]) * g->inv_len[d]);

            if ( c[d] >= g->native_end[d] )
            {
                c[d] = g->native_end[d] - 1;
            }
            else if ( c[d] < g->native_str[d] )
            {
                c[d] = g->native_str[d];
            }
        }

#if defined(DEBUG)
        fprintf( stderr, "p%d bin_my_atoms: l:%d - atom%d @ %.5f %.5f %.5f" \
                "--> cell: %d %d %d\n",
                system->my_rank, l, atoms[l].orig_id,
                atoms[l].x[0], atoms[l].x[1], atoms[l].x[2],
                c[0], c[1], c[2] );
#endif

        gc = &g->cells[ index_grid_3d_v(c, g) ];
        gc->top++;
    }

    max_atoms = 0;
    for ( i = 0; i < g->ncells[0]; i++ )
    {
        for ( j = 0; j < g->ncells[1]; j++ )
        {
            for ( k = 0; k < g->ncells[2]; k++ )
            {
                gc = &g->cells[ index_grid_3d(i, j, k, g) ];
                if ( max_atoms < gc->top )
                {
                    max_atoms = gc->top;
                }

#if defined(DEBUG)
                fprintf( stderr, "p%d gc[%d,%d,%d]->top=%d\n",
                        system->my_rank, i, j, k, gc->top );
#endif
            }
        }
    }

    my_max = MAX( max_atoms * SAFE_ZONE, MIN_GCELL_POPL );
    MPI_Allreduce( &my_max, &all_max, 1, MPI_INT, MPI_MAX, comm );

#if defined(DEBUG)
    fprintf( stderr, "p%d max_atoms=%d, my_max=%d, all_max=%d\n",
            system->my_rank, max_atoms, my_max, all_max );
#endif

    return all_max;
}


void Allocate_Grid( reax_system *system, MPI_Comm comm )
{
    int i, j, k;
    grid *g;
    grid_cell *gc;
    int total;

    g = &system->my_grid;
    total = g->ncells[0] * g->ncells[1] * g->ncells[2];

    /* allocate gcell reordering space */
    g->order = scalloc( g->total + 1, sizeof(ivec), "Allocate_Grid::g->order" );

    /* allocate the gcells for the new grid */
    g->max_nbrs = (2 * g->vlist_span[0] + 1) * (2 * g->vlist_span[1] + 1) *
            (2 * g->vlist_span[2] + 1) + 3;

    g->cells = scalloc( total, sizeof(grid_cell), "Allocate_Grid::g->cells" );

    for (i = 0; i < total; i++)
    {
        gc = &g->cells[i];
        gc->top = 0;
        gc->mark = 0;
    }

    g->str = scalloc( total, sizeof(int),"Allocate_Grid::grid->str" );
    g->end = scalloc( total, sizeof(int), "Allocate_Grid::grid->end" );
    g->cutoff = scalloc( total, sizeof(real), "Allocate_Grid::grid->cutoff" );
    g->nbrs_x = scalloc( total * g->max_nbrs, sizeof(ivec), "Allocate_Grid::grid->nbrs_x" );
    g->nbrs_cp = scalloc( total * g->max_nbrs, sizeof(rvec), "Allocate_Grid::grid->nbrs_cp" );

    for ( i = 0; i < total * g->max_nbrs; i++ )
    {
        g->nbrs_x[i][0] = -1;
        g->nbrs_x[i][1] = -1;
        g->nbrs_x[i][2] = -1;
    }
    g->rel_box = scalloc( total, sizeof(ivec), "Allocate_Grid::grid->rel_box" );

    /* allocate atom id storage in gcells */
    g->max_atoms = Estimate_GCell_Population( system, comm );

    /* space for storing atom id's is required only for native cells */
    for ( i = g->native_str[0]; i < g->native_end[0]; ++i )
    {
        for ( j = g->native_str[1]; j < g->native_end[1]; ++j )
        {
            for ( k = g->native_str[2]; k < g->native_end[2]; ++k )
            {
                g->cells[ index_grid_3d(i, j, k, g) ].atoms =
                    scalloc( g->max_atoms, sizeof(int), "Allocate_Grid::g->cells[ ].atoms" );
            }
        }
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d-allocated %dx%dx%d grid: nbrs=%d atoms=%d space=%dMB\n",
             system->my_rank, g->ncells[0], g->ncells[1], g->ncells[2],
             g->max_nbrs, g->max_atoms,
             (int)
             ((g->total * sizeof(grid_cell) + g->total * g->max_nbrs * sizeof(int*) +
               g->total * g->max_nbrs * sizeof(rvec) +
               (g->native_end[0] - g->native_str[0]) *
               (g->native_end[1] - g->native_str[1]) *
               (g->native_end[2] - g->native_str[2]) * g->max_atoms * sizeof(int)) /
              (1024 * 1024)) );
#endif
}


void Deallocate_Grid( grid *g )
{
    int i, j, k;
    grid_cell *gc;

    sfree( g->order, "Deallocate_Grid::g->order" );
    sfree( g->str, "Deallocate_Grid::g->str" );
    sfree( g->end, "Deallocate_Grid::g->end" );
    sfree( g->cutoff, "Deallocate_Grid::g->cutoff" );
    sfree( g->nbrs_x, "Deallocate_Grid::g->nbrs_x" );
    sfree( g->nbrs_cp, "Deallocate_Grid::g->nbrs_cp" );

    /* deallocate the grid cells */
    for ( i = 0; i < g->ncells[0]; i++ )
    {
        for ( j = 0; j < g->ncells[1]; j++ )
        {
            for ( k = 0; k < g->ncells[2]; k++ )
            {
                gc = &g->cells[ index_grid_3d(i, j, k, g)] ;
                if ( gc->atoms != NULL )
                {
                    sfree( gc->atoms, "Deallocate_Grid::g->cells[ ].atoms" );
                }
            }
        }
    }
    sfree( g->cells, "Deallocate_Grid::g->cells" );
}


/************ mpi buffers ********************/
/* make the allocations based on the largest need by the three comms:
 * 1- transfer an atom who has moved into other proc's domain (mpi_atom)
 * 2- exchange boundary atoms (boundary_atom)
 * 3- update position info for boundary atoms (mpi_rvec)
 *
 * the largest space by far is required for the 2nd comm operation above.
 *
 * Note: buffers are (void *), type cast to the correct pointer type to access
 * the allocated buffers */
void Allocate_MPI_Buffers( mpi_datatypes *mpi_data, int est_recv,
        neighbor_proc *my_nbrs )
{
    int i;
    mpi_out_data *mpi_buf;

    /* buffers for incoming messages,
     * see SendRecv for MPI datatypes sent */
    mpi_data->in1_buffer = scalloc( est_recv,
            MAX3( sizeof(mpi_atom), sizeof(boundary_atom), sizeof(rvec) ),
            "Allocate_MPI_Buffers::in1_buffer" );
    mpi_data->in2_buffer = scalloc( est_recv,
            MAX3( sizeof(mpi_atom), sizeof(boundary_atom), sizeof(rvec) ),
            "Allocate_MPI_Buffers::in2_buffer" );

    /* buffers for outgoing messages,
     * see SendRecv for MPI datatypes sent */
    for ( i = 0; i < MAX_NBRS; ++i )
    {
        mpi_buf = &mpi_data->out_buffers[i];

        /* allocate storage for the neighbor processor i */
        mpi_buf->index = scalloc( my_nbrs[i].est_send, sizeof(int),
                "Allocate_MPI_Buffers::mpi_buf->index" );
        mpi_buf->out_atoms = scalloc( my_nbrs[i].est_send,
                MAX3( sizeof(mpi_atom), sizeof(boundary_atom), sizeof(rvec) ),
                "Allocate_MPI_Buffers::mpi_buf->out_atoms" );
    }
}


void Deallocate_MPI_Buffers( mpi_datatypes *mpi_data )
{
    int i;
    mpi_out_data  *mpi_buf;

    sfree( mpi_data->in1_buffer, "Deallocate_MPI_Buffers::in1_buffer" );
    sfree( mpi_data->in2_buffer, "Deallocate_MPI_Buffers::in2_buffer" );

    for ( i = 0; i < MAX_NBRS; ++i )
    {
        mpi_buf = &mpi_data->out_buffers[i];
        sfree( mpi_buf->index, "Deallocate_MPI_Buffers::mpi_buf->index" );
        sfree( mpi_buf->out_atoms, "Deallocate_MPI_Buffers::mpi_buf->out_atoms" );
    }
}


void ReAllocate( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        mpi_datatypes *mpi_data )
{
    int i, j, k;
    int nflag, Nflag, Hflag, mpi_flag, total_send;
    int renbr;
    reallocate_data *realloc;
    sparse_matrix *H;
    grid *g;
    neighbor_proc *nbr_pr;
    mpi_out_data *nbr_data;

    realloc = &workspace->realloc;
    g = &system->my_grid;
    H = &workspace->H;

    /* IMPORTANT: LOOSE ZONES CHECKS ARE DISABLED FOR NOW BY &&'ing with FALSE!!! */
    nflag = FALSE;
    if ( system->n >= DANGER_ZONE * system->local_cap ||
            (FALSE && system->n <= LOOSE_ZONE * system->local_cap) )
    {
        nflag = TRUE;
        system->local_cap = (int)(system->n * SAFE_ZONE);
    }

    Nflag = FALSE;
    if ( system->N >= DANGER_ZONE * system->total_cap ||
            (FALSE && system->N <= LOOSE_ZONE * system->total_cap) )
    {
        Nflag = TRUE;
        system->total_cap = (int)(system->N * SAFE_ZONE);
    }

    if ( Nflag == TRUE )
    {
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: reallocating system and workspace -"\
                "n=%d  N=%d  local_cap=%d  total_cap=%d\n",
                system->my_rank, system->n, system->N,
                system->local_cap, system->total_cap );
#endif

        ReAllocate_System( system, system->local_cap, system->total_cap );

        DeAllocate_Workspace( control, workspace );
        Allocate_Workspace( system, control, workspace, system->local_cap,
                system->total_cap );
    }

    renbr = (data->step - data->prev_steps) % control->reneighbor == 0 ? TRUE : FALSE;
    /* far neighbors */
    if ( renbr == TRUE )
    {
        if ( Nflag == TRUE || realloc->far_nbrs == TRUE )
        {
#if defined(DEBUG_FOCUS)
            fprintf( stderr, "p%d: reallocating far_nbrs: far_nbrs=%d, space=%dMB\n",
                     system->my_rank, (int)(system->total_far_nbrs * SAFE_ZONE),
                     (int)(system->total_far_nbrs * SAFE_ZONE * sizeof(far_neighbor_data) /
                           (1024 * 1024)) );
#endif

            Reallocate_Neighbor_List( lists[FAR_NBRS], system->total_cap, system->total_far_nbrs );
            Init_List_Indices( lists[FAR_NBRS], system->max_far_nbrs );
            realloc->far_nbrs = FALSE;
        }
    }

    /* charge matrix */
    if ( nflag == TRUE || realloc->cm == TRUE )
    {
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: reallocating H matrix: Htop=%d, space=%dMB\n",
                system->my_rank, (int)(realloc->Htop * SAFE_ZONE),
                (int)(realloc->Htop * SAFE_ZONE * sizeof(sparse_matrix_entry) /
                (1024 * 1024)) );
#endif

        Reallocate_Matrix( H, system->n, system->local_cap, system->total_cm_entries );
        Init_Matrix_Row_Indices( H, system->max_cm_entries );
        realloc->cm = FALSE;
    }

    /* hydrogen bonds list */
    if ( control->hbond_cut > 0.0 )
    {
        Hflag = FALSE;
//        if ( system->numH >= DANGER_ZONE * system->Hcap
//                || (0 && system->numH <= LOOSE_ZONE * system->Hcap) )
//        {
//            Hflag = TRUE;
//            //system->Hcap = (int)(system->numH * SAFE_ZONE);
//            // Tried fix
//            system->Hcap = (int)(system->n * SAFE_ZONE);
//        }

        if ( Hflag == TRUE || realloc->hbonds == TRUE )
        {
#if defined(DEBUG_FOCUS)
            fprintf(stderr, "p%d: reallocating hbonds: total_hbonds=%d space=%dMB\n",
                    system->my_rank, ret, (int)(ret * sizeof(hbond_data) / (1024 * 1024)));
#endif

            Reallocate_HBonds_List( system, lists[HBONDS] );
            Init_List_Indices( lists[HBONDS], system->max_hbonds );
            realloc->hbonds = FALSE;
        }
    }

    /* bonds list */
    if ( Nflag == TRUE || realloc->bonds == TRUE )
    {
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: reallocating bonds: total_bonds=%d, space=%dMB\n",
                 system->my_rank, system->total_bonds,
                 (int)(system->total_bonds * sizeof(bond_data) / (1024 * 1024)) );
#endif

        Reallocate_Bonds_List( system, lists[BONDS] );
        Init_List_Indices( lists[BONDS], system->max_bonds );
        realloc->bonds = FALSE;
        realloc->thbody = TRUE;
    }

    /* 3-body list */
    if ( realloc->thbody == TRUE )
    {
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: reallocating 3body list: num_3body=%d, space=%dMB\n",
                system->my_rank, realloc->num_3body,
                (int)(realloc->num_3body * sizeof(three_body_interaction_data) /
                (1024 * 1024)) );
#endif

        Delete_List( lists[THREE_BODIES] );
        Make_List( system->total_bonds, system->total_thbodies,
                TYP_THREE_BODY, lists[THREE_BODIES] );
        realloc->thbody = FALSE;
    }

    /* grid */
    if ( renbr == TRUE && realloc->gcell_atoms > -1 )
    {
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "reallocating gcell: g->max_atoms: %d\n", g->max_atoms );
#endif

        for ( i = g->native_str[0]; i < g->native_end[0]; i++ )
        {
            for ( j = g->native_str[1]; j < g->native_end[1]; j++ )
            {
                for ( k = g->native_str[2]; k < g->native_end[2]; k++ )
                {
                    sfree( g->cells[ index_grid_3d(i, j, k, g) ].atoms, "ReAllocate::g->cells[ ].atoms" );
                    g->cells[ index_grid_3d(i, j, k, g) ].atoms = scalloc( realloc->gcell_atoms,
                            sizeof(int), "ReAllocate::g->cells[ ].atoms" );
                }
            }
        }

        realloc->gcell_atoms = -1;
    }

    /* mpi buffers */
    // we have to be at a renbring step -
    // to ensure correct values at mpi_buffers for update_boundary_positions
    if ( renbr == FALSE )
    {
        mpi_flag = FALSE;
    }
    /* check whether in_buffer capacity is enough */
    else if ( system->max_recved >= system->est_recv * DANGER_ZONE )
    {
        mpi_flag = TRUE;
    }
    else
    {
        /* otherwise check individual outgoing buffers */
        mpi_flag = FALSE;
        for ( i = 0; i < MAX_NBRS; ++i )
        {
            nbr_pr = &( system->my_nbrs[i] );
            nbr_data = &( mpi_data->out_buffers[i] );
            if ( nbr_data->cnt >= nbr_pr->est_send * DANGER_ZONE )
            {
                mpi_flag = TRUE;
                break;
            }
        }
    }

    if ( mpi_flag == TRUE )
    {
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: reallocating mpi_buf: old_recv=%d\n",
                system->my_rank, system->est_recv );
        for ( i = 0; i < MAX_NBRS; ++i )
        {
            fprintf( stderr, "p%d: nbr%d old_send=%d\n",
                    system->my_rank, i, system->my_nbrs[i].est_send );
        }
#endif

        /* update mpi buffer estimates based on last comm */
        system->est_recv = MAX( system->max_recved * SAFER_ZONE, MIN_SEND );
        system->est_trans =
            (system->est_recv * sizeof(boundary_atom)) / sizeof(mpi_atom);
        total_send = 0;
        for ( i = 0; i < MAX_NBRS; ++i )
        {
            nbr_pr = &( system->my_nbrs[i] );
            nbr_data = &( mpi_data->out_buffers[i] );
            nbr_pr->est_send = MAX( nbr_data->cnt * SAFER_ZONE, MIN_SEND );
            total_send += nbr_pr->est_send;
        }

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: reallocating mpi_buf: recv=%d send=%d total=%dMB\n",
               system->my_rank, system->est_recv, total_send,
               (int)((system->est_recv + total_send) * sizeof(boundary_atom) /
               (1024 * 1024)));
        for ( i = 0; i < MAX_NBRS; ++i )
        {
            fprintf( stderr, "p%d: nbr%d new_send=%d\n",
                    system->my_rank, i, system->my_nbrs[i].est_send );
        }
#endif

        /* reallocate mpi buffers */
        Deallocate_MPI_Buffers( mpi_data );
        Allocate_MPI_Buffers( mpi_data, system->est_recv, system->my_nbrs );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d: reallocate done\n",
            system->my_rank, data->step );
    MPI_Barrier( MPI_COMM_WORLD );
#endif
}
