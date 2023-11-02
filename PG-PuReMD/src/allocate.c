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
  #include "allocate.h"

  #include "comm_tools.h"
  #include "index_utils.h"
  #include "list.h"
  #include "reset_tools.h"
  #include "tool_box.h"
  #include "vector.h"
#elif defined(LAMMPS_REAX)
  #include "reax_allocate.h"

  #include "reax_comm_tools.h"
  #include "reax_index_utils.h"
  #include "reax_list.h"
  #include "reax_reset_tools.h"
  #include "reax_tool_box.h"
  #include "reax_vector.h"
#endif


void Init_Matrix_Row_Indices( sparse_matrix * const H,
        int * const max_row_entries )
{
    int i;

    /* exclusive prefix sum on max_row_entries replaces start,
     * set end indices to the same as start indices for safety */
    H->start[0] = 0;
    H->end[0] = 0;
    for ( i = 1; i < H->n_max; ++i )
    {
        H->start[i] = H->start[i - 1] + max_row_entries[i - 1];
        H->end[i] = H->start[i];
    }
}


/* allocate space for my_atoms
 * important: we cannot know the exact number of atoms that will fall into a
 * process's box throughout the whole simulation. therefore
 * we need to make upper bound estimates for various data structures */
void PreAllocate_Space( reax_system * const system, control_params * const control,
        storage * const workspace )
{
    /* determine capacity based on box vol & est atom volume */
    system->local_cap = MAX( (int) CEIL( system->n * SAFE_ZONE ), MIN_CAP );
    system->total_cap = MAX( (int) CEIL( system->N * SAFE_ZONE ), MIN_CAP );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: local_cap=%d total_cap=%d\n",
            system->my_rank, system->local_cap, system->total_cap );
#endif

    system->my_atoms = scalloc_pinned( system->total_cap, sizeof(reax_atom),
            __FILE__, __LINE__ );

    /* space for keeping restriction info, if any */
    if ( control->restrict_bonds )
    {
        workspace->restricted = scalloc( system->local_cap, sizeof(int),
                __FILE__, __LINE__ );
        workspace->restricted_list = scalloc( system->local_cap * MAX_RESTRICT,
                sizeof(int), __FILE__, __LINE__ );
    }
}


void Reallocate_System_Part1( reax_system * const system, int local_cap )
{
    system->cm_entries = srealloc( system->cm_entries, sizeof(int) * local_cap,
            __FILE__, __LINE__ );
    system->max_cm_entries = srealloc( system->max_cm_entries, sizeof(int) * local_cap,
            __FILE__, __LINE__ );
}



void Reallocate_System_Part2( reax_system * const system, int total_cap_old,
        int total_cap )
{
    system->my_atoms = srealloc_pinned( system->my_atoms,
            sizeof(reax_atom) * total_cap_old,
            sizeof(reax_atom) * total_cap, __FILE__, __LINE__ );
//    system->my_atoms = srealloc( system->my_atoms, sizeof(reax_atom) * total_cap,
//            __FILE__, __LINE__ );

    /* list management */
    system->far_nbrs = srealloc( system->far_nbrs, sizeof(int) * total_cap,
            __FILE__, __LINE__ );
    system->max_far_nbrs = srealloc( system->max_far_nbrs, sizeof(int) * total_cap,
            __FILE__, __LINE__ );

    system->bonds = srealloc( system->bonds, sizeof(int) * total_cap,
            __FILE__, __LINE__ );
    system->max_bonds = srealloc( system->max_bonds, sizeof(int) * total_cap,
            __FILE__, __LINE__ );

    system->hbonds = srealloc( system->hbonds, sizeof(int) * total_cap,
            __FILE__, __LINE__ );
    system->max_hbonds = srealloc( system->max_hbonds, sizeof(int) * total_cap,
            __FILE__, __LINE__ );
}


static void Deallocate_Workspace_Part1( control_params * const control,
        storage * const workspace )
{
    /* Nose-Hoover integrator */
    if ( control->ensemble == nhNVT )
    {
        sfree( workspace->v_const, __FILE__, __LINE__ );
    }

    /* storage for analysis */
    if ( control->molecular_analysis || control->diffusion_coef )
    {
        sfree( workspace->mark, __FILE__, __LINE__ );
        sfree( workspace->old_mark, __FILE__, __LINE__ );
    }

    if ( control->diffusion_coef )
    {
        sfree( workspace->x_old, __FILE__, __LINE__ );
    }
}


static void Deallocate_Workspace_Part2( control_params * const control,
        storage * const workspace )
{
    /* bond order storage */
    sfree( workspace->total_bond_order, __FILE__, __LINE__ );
    sfree( workspace->Deltap, __FILE__, __LINE__ );
    sfree( workspace->Deltap_boc, __FILE__, __LINE__ );
    sfree( workspace->dDeltap_self, __FILE__, __LINE__ );
    sfree( workspace->Delta, __FILE__, __LINE__ );
    sfree( workspace->Delta_lp, __FILE__, __LINE__ );
    sfree( workspace->Delta_lp_temp, __FILE__, __LINE__ );
    sfree( workspace->dDelta_lp, __FILE__, __LINE__ );
    sfree( workspace->dDelta_lp_temp, __FILE__, __LINE__ );
    sfree( workspace->Delta_e, __FILE__, __LINE__ );
    sfree( workspace->Delta_boc, __FILE__, __LINE__ );
    sfree( workspace->nlp, __FILE__, __LINE__ );
    sfree( workspace->nlp_temp, __FILE__, __LINE__ );
    sfree( workspace->Clp, __FILE__, __LINE__ );
    sfree( workspace->vlpex, __FILE__, __LINE__ );
    sfree( workspace->bond_mark, __FILE__, __LINE__ );

    /* charge matrix storage */
    if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        sfree( workspace->Hdia_inv, __FILE__, __LINE__ );
    }
    if ( control->cm_solver_pre_comp_type == ICHOLT_PC
            || control->cm_solver_pre_comp_type == ILUT_PC
            || control->cm_solver_pre_comp_type == ILUTP_PC
            || control->cm_solver_pre_comp_type == FG_ILUT_PC )
    {
        sfree( workspace->droptol, __FILE__, __LINE__ );
    }
    sfree( workspace->b_s, __FILE__, __LINE__ );
    sfree( workspace->b_t, __FILE__, __LINE__ );
    sfree( workspace->b_prc, __FILE__, __LINE__ );
    sfree( workspace->b_prm, __FILE__, __LINE__ );
    sfree( workspace->s, __FILE__, __LINE__ );
    sfree( workspace->t, __FILE__, __LINE__ );

    switch ( control->cm_solver_type )
    {
        case GMRES_S:
        case GMRES_H_S:
            sfree( workspace->y, __FILE__, __LINE__ );
            sfree( workspace->z, __FILE__, __LINE__ );
            sfree( workspace->g, __FILE__, __LINE__ );
            sfree( workspace->h, __FILE__, __LINE__ );
            sfree( workspace->hs, __FILE__, __LINE__ );
            sfree( workspace->hc, __FILE__, __LINE__ );
            sfree( workspace->v, __FILE__, __LINE__ );
            break;

        case CG_S:
            sfree( workspace->r, __FILE__, __LINE__ );
            sfree( workspace->d, __FILE__, __LINE__ );
            sfree( workspace->q, __FILE__, __LINE__ );
            sfree( workspace->p, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            sfree( workspace->r2, __FILE__, __LINE__ );
            sfree( workspace->d2, __FILE__, __LINE__ );
            sfree( workspace->q2, __FILE__, __LINE__ );
            sfree( workspace->p2, __FILE__, __LINE__ );
#endif
            break;

        case SDM_S:
            sfree( workspace->r, __FILE__, __LINE__ );
            sfree( workspace->d, __FILE__, __LINE__ );
            sfree( workspace->q, __FILE__, __LINE__ );
            sfree( workspace->p, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            sfree( workspace->r2, __FILE__, __LINE__ );
            sfree( workspace->d2, __FILE__, __LINE__ );
            sfree( workspace->q2, __FILE__, __LINE__ );
            sfree( workspace->p2, __FILE__, __LINE__ );
#endif
            break;

        case BiCGStab_S:
            sfree( workspace->y, __FILE__, __LINE__ );
            sfree( workspace->g, __FILE__, __LINE__ );
            sfree( workspace->z, __FILE__, __LINE__ );
            sfree( workspace->r, __FILE__, __LINE__ );
            sfree( workspace->d, __FILE__, __LINE__ );
            sfree( workspace->q, __FILE__, __LINE__ );
            sfree( workspace->p, __FILE__, __LINE__ );
            sfree( workspace->r_hat, __FILE__, __LINE__ );
            sfree( workspace->q_hat, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            sfree( workspace->y2, __FILE__, __LINE__ );
            sfree( workspace->g2, __FILE__, __LINE__ );
            sfree( workspace->z2, __FILE__, __LINE__ );
            sfree( workspace->r2, __FILE__, __LINE__ );
            sfree( workspace->d2, __FILE__, __LINE__ );
            sfree( workspace->q2, __FILE__, __LINE__ );
            sfree( workspace->p2, __FILE__, __LINE__ );
            sfree( workspace->r_hat2, __FILE__, __LINE__ );
            sfree( workspace->q_hat2, __FILE__, __LINE__ );
#endif
            break;

        case PIPECG_S:
            sfree( workspace->z, __FILE__, __LINE__ );
            sfree( workspace->r, __FILE__, __LINE__ );
            sfree( workspace->d, __FILE__, __LINE__ );
            sfree( workspace->q, __FILE__, __LINE__ );
            sfree( workspace->p, __FILE__, __LINE__ );
            sfree( workspace->m, __FILE__, __LINE__ );
            sfree( workspace->n, __FILE__, __LINE__ );
            sfree( workspace->u, __FILE__, __LINE__ );
            sfree( workspace->w, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            sfree( workspace->z2, __FILE__, __LINE__ );
            sfree( workspace->r2, __FILE__, __LINE__ );
            sfree( workspace->d2, __FILE__, __LINE__ );
            sfree( workspace->q2, __FILE__, __LINE__ );
            sfree( workspace->p2, __FILE__, __LINE__ );
            sfree( workspace->m2, __FILE__, __LINE__ );
            sfree( workspace->n2, __FILE__, __LINE__ );
            sfree( workspace->u2, __FILE__, __LINE__ );
            sfree( workspace->w2, __FILE__, __LINE__ );
#endif
            break;

        case PIPECR_S:
            sfree( workspace->z, __FILE__, __LINE__ );
            sfree( workspace->r, __FILE__, __LINE__ );
            sfree( workspace->d, __FILE__, __LINE__ );
            sfree( workspace->q, __FILE__, __LINE__ );
            sfree( workspace->p, __FILE__, __LINE__ );
            sfree( workspace->m, __FILE__, __LINE__ );
            sfree( workspace->n, __FILE__, __LINE__ );
            sfree( workspace->u, __FILE__, __LINE__ );
            sfree( workspace->w, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            sfree( workspace->z2, __FILE__, __LINE__ );
            sfree( workspace->r2, __FILE__, __LINE__ );
            sfree( workspace->d2, __FILE__, __LINE__ );
            sfree( workspace->q2, __FILE__, __LINE__ );
            sfree( workspace->p2, __FILE__, __LINE__ );
            sfree( workspace->m2, __FILE__, __LINE__ );
            sfree( workspace->n2, __FILE__, __LINE__ );
            sfree( workspace->u2, __FILE__, __LINE__ );
            sfree( workspace->w2, __FILE__, __LINE__ );
#endif
            break;

        default:
            fprintf( stderr, "[ERROR] Unknown charge method linear solver type. Terminating...\n" );
            exit( UNKNOWN_OPTION );
            break;
    }

    /* force-related storage */
    sfree( workspace->f, __FILE__, __LINE__ );
    sfree( workspace->CdDelta, __FILE__, __LINE__ );

#if defined(TEST_FORCES)
    sfree( workspace->dDelta, __FILE__, __LINE__ );
    sfree( workspace->f_ele, __FILE__, __LINE__ );
    sfree( workspace->f_vdw, __FILE__, __LINE__ );
    sfree( workspace->f_bo, __FILE__, __LINE__ );
    sfree( workspace->f_be, __FILE__, __LINE__ );
    sfree( workspace->f_lp, __FILE__, __LINE__ );
    sfree( workspace->f_ov, __FILE__, __LINE__ );
    sfree( workspace->f_un, __FILE__, __LINE__ );
    sfree( workspace->f_ang, __FILE__, __LINE__ );
    sfree( workspace->f_coa, __FILE__, __LINE__ );
    sfree( workspace->f_pen, __FILE__, __LINE__ );
    sfree( workspace->f_hb, __FILE__, __LINE__ );
    sfree( workspace->f_tor, __FILE__, __LINE__ );
    sfree( workspace->f_con, __FILE__, __LINE__ );
    sfree( workspace->f_tot, __FILE__, __LINE__ );

    sfree( workspace->rcounts, __FILE__, __LINE__ );
    sfree( workspace->displs, __FILE__, __LINE__ );
    sfree( workspace->id_all, __FILE__, __LINE__ );
    sfree( workspace->f_all, __FILE__, __LINE__ );
#endif
}


void Allocate_Workspace_Part1( reax_system * const system, control_params * const control,
        storage * const workspace, int local_cap )
{
    const size_t local_rvec = sizeof(rvec) * local_cap;

    /* integrator storage */
    if ( control->ensemble == nhNVT )
    {
        workspace->v_const = smalloc( local_rvec, __FILE__, __LINE__ );
    }

    /* storage for analysis */
    if ( control->molecular_analysis || control->diffusion_coef )
    {
        workspace->mark = scalloc( local_cap, sizeof(int),
                __FILE__, __LINE__ );
        workspace->old_mark = scalloc( local_cap, sizeof(int),
                __FILE__, __LINE__ );
    }
    else
    {
        workspace->mark = NULL;
        workspace->old_mark = NULL;
    }

    if ( control->diffusion_coef )
    {
        workspace->x_old = scalloc( local_cap, sizeof(rvec),
                __FILE__, __LINE__ );
    }
    else
    {
        workspace->x_old = NULL;
    }
}


void Allocate_Workspace_Part2( reax_system * const system, control_params * const control,
        storage * const workspace, int total_cap )
{
    const size_t total_real = sizeof(real) * total_cap;
    const size_t total_rvec = sizeof(rvec) * total_cap;

    /* bond order related storage  */
    workspace->total_bond_order = smalloc( total_real, __FILE__, __LINE__ );
    workspace->Deltap = smalloc( total_real, __FILE__, __LINE__ );
    workspace->Deltap_boc = smalloc( total_real, __FILE__, __LINE__ );
    workspace->dDeltap_self = smalloc( total_rvec, __FILE__, __LINE__ );
    workspace->Delta = smalloc( total_real, __FILE__, __LINE__ );
    workspace->Delta_lp = smalloc( total_real, __FILE__, __LINE__ );
    workspace->Delta_lp_temp = smalloc( total_real, __FILE__, __LINE__ );
    workspace->dDelta_lp = smalloc( total_real, __FILE__, __LINE__ );
    workspace->dDelta_lp_temp = smalloc( total_real, __FILE__, __LINE__ );
    workspace->Delta_e = smalloc( total_real, __FILE__, __LINE__ );
    workspace->Delta_boc = smalloc( total_real, __FILE__, __LINE__ );
    workspace->nlp = smalloc( total_real, __FILE__, __LINE__ );
    workspace->nlp_temp = smalloc( total_real, __FILE__, __LINE__ );
    workspace->Clp = smalloc( total_real, __FILE__, __LINE__ );
    workspace->CdDelta = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
    workspace->vlpex = smalloc( total_real, __FILE__, __LINE__ );
    workspace->bond_mark = scalloc( total_cap, sizeof(int),
            __FILE__, __LINE__ );

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

    workspace->b_s = scalloc( total_cap, sizeof(real),
            __FILE__, __LINE__ );
    workspace->b_t = scalloc( total_cap, sizeof(real),
            __FILE__, __LINE__ );
    workspace->b_prc = scalloc( total_cap, sizeof(real),
            __FILE__, __LINE__ );
    workspace->b_prm = scalloc( total_cap, sizeof(real),
            __FILE__, __LINE__ );
    workspace->s = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
    workspace->t = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
    workspace->b = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
    workspace->x = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );

    if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        workspace->Hdia_inv = scalloc( total_cap, sizeof(real),
                __FILE__, __LINE__ );
    }
    if ( control->cm_solver_pre_comp_type == ICHOLT_PC
            || control->cm_solver_pre_comp_type == ILUT_PC
            || control->cm_solver_pre_comp_type == ILUTP_PC
            || control->cm_solver_pre_comp_type == FG_ILUT_PC )
    {
        workspace->droptol = scalloc( total_cap, sizeof(real),
                __FILE__, __LINE__ );
    }

    switch ( control->cm_solver_type )
    {
        case GMRES_S:
        case GMRES_H_S:
            workspace->y = scalloc( control->cm_solver_restart + 1, sizeof(real), __FILE__, __LINE__ );
            workspace->z = scalloc( control->cm_solver_restart + 1, sizeof(real), __FILE__, __LINE__ );
            workspace->g = scalloc( control->cm_solver_restart + 1, sizeof(real), __FILE__, __LINE__ );
            workspace->h = scalloc( SQR(control->cm_solver_restart + 1), sizeof(real), __FILE__, __LINE__ );
            workspace->hs = scalloc( control->cm_solver_restart + 1, sizeof(real), __FILE__, __LINE__ );
            workspace->hc = scalloc( control->cm_solver_restart + 1, sizeof(real), __FILE__, __LINE__ );
            workspace->v = scalloc( SQR(control->cm_solver_restart + 1), sizeof(real), __FILE__, __LINE__ );
            break;

        case CG_S:
            workspace->r = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->d = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->q = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->p = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            workspace->r2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->d2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->q2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->p2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
#endif
            break;

        case SDM_S:
            workspace->r = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->d = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->q = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->p = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            workspace->r2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->d2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->q2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->p2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
#endif
            break;

        case BiCGStab_S:
            workspace->y = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->g = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->z = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->r = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->d = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->q = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->p = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->r_hat = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->q_hat = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            workspace->y2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->g2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->z2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->r2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->d2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->q2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->p2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->r_hat2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->q_hat2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
#endif
            break;

        case PIPECG_S:
            workspace->z = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->r = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->d = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->q = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->p = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->m = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->n = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->u = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->w = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            workspace->z2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->r2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->d2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->q2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->p2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->m2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->n2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->u2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->w2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
#endif
            break;

        case PIPECR_S:
            workspace->z = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->r = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->d = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->q = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->p = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->m = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->n = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->u = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
            workspace->w = scalloc( total_cap, sizeof(real), __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            workspace->z2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->r2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->d2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->q2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->p2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->m2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->n2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->u2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
            workspace->w2 = scalloc( total_cap, sizeof(rvec2), __FILE__, __LINE__ );
#endif
            break;

        default:
            fprintf( stderr, "[ERROR] Unknown charge method linear solver type. Terminating...\n" );
            exit( UNKNOWN_OPTION );
            break;
    }

    /* force related storage */
    workspace->f = scalloc( total_cap, sizeof(rvec),
            __FILE__, __LINE__ );

#if defined(TEST_FORCES)
    workspace->dDelta = smalloc( total_rvec, __FILE__, __LINE__ );
    workspace->f_ele = smalloc( total_rvec, __FILE__, __LINE__ );
    workspace->f_vdw = smalloc( total_rvec, __FILE__, __LINE__ );
    workspace->f_bo = smalloc( total_rvec, __FILE__, __LINE__ );
    workspace->f_be = smalloc( total_rvec, __FILE__, __LINE__ );
    workspace->f_lp = smalloc( total_rvec, __FILE__, __LINE__ );
    workspace->f_ov = smalloc( total_rvec, __FILE__, __LINE__ );
    workspace->f_un = smalloc( total_rvec, __FILE__, __LINE__ );
    workspace->f_ang = smalloc( total_rvec, __FILE__, __LINE__ );
    workspace->f_coa = smalloc( total_rvec, __FILE__, __LINE__ );
    workspace->f_pen = smalloc( total_rvec, __FILE__, __LINE__ );
    workspace->f_hb = smalloc( total_rvec, __FILE__, __LINE__ );
    workspace->f_tor = smalloc( total_rvec, __FILE__, __LINE__ );
    workspace->f_con = smalloc( total_rvec, __FILE__, __LINE__ );
    workspace->f_tot = smalloc( total_rvec, __FILE__, __LINE__ );

    if ( system->my_rank == MASTER_NODE )
    {
        workspace->rcounts = smalloc( sizeof(int) * system->nprocs,
                __FILE__, __LINE__ );
        workspace->displs = smalloc( sizeof(int) * system->nprocs,
                __FILE__, __LINE__ );
        workspace->id_all = smalloc( sizeof(int) * system->bigN,
                __FILE__, __LINE__ );
        workspace->f_all = smalloc( sizeof(rvec) * system->bigN,
                __FILE__, __LINE__ );
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


void Reallocate_Neighbor_List( reax_list * const far_nbr_list,
        int n, int max_intrs )
{
    int format;

    format = far_nbr_list->format;

    Delete_List( far_nbr_list );
    Make_List( n, max_intrs, TYP_FAR_NEIGHBOR, format, far_nbr_list );
}


/* Allocate sparse matrix struc
 *
 * H: pointer to struct
 * n: currently utilized number of rows
 * n_max: max number of rows allocated
 * m: max number of entries allocated
 * format: sparse matrix format
 */
void Allocate_Matrix( sparse_matrix * const H, int n, int n_max, int m,
        int format )
{
    H->allocated = TRUE;
    H->n = n;
    H->n_max = n_max;
    H->m = m;
    H->format = format;

    H->start = smalloc( sizeof(int) * n_max, __FILE__, __LINE__ );
    H->end = smalloc( sizeof(int) * n_max, __FILE__, __LINE__ );
    H->j = smalloc( sizeof(int) * m, __FILE__, __LINE__ );
    H->val = smalloc( sizeof(real) * m, __FILE__, __LINE__ );
}


void Deallocate_Matrix( sparse_matrix * const H )
{
    H->allocated = FALSE;
    H->n = 0;
    H->n_max = 0;
    H->m = 0;

    sfree( H->start, __FILE__, __LINE__ );
    sfree( H->end, __FILE__, __LINE__ );
    sfree( H->j, __FILE__, __LINE__ );
    sfree( H->val, __FILE__, __LINE__ );
}


static void Reallocate_Matrix( sparse_matrix * const H, int n, int n_max, int m )
{
    int format;

    format = H->format;

    Deallocate_Matrix( H );
    Allocate_Matrix( H, n, n_max, m, format );
}


static void Reallocate_List( reax_list * const list, int n, int max_intrs,
        int type, int format )
{
    Delete_List( list );
    Make_List( n, max_intrs, type, format, list );
}


void Reallocate_Bonds_List( reax_system * const system, reax_list * const bond_list )
{
    int format;

    format = bond_list->format;

    Delete_List( bond_list );
    Make_List( system->total_cap, system->total_bonds, TYP_BOND, format, bond_list );
}


int Estimate_GCell_Population( reax_system * const system, MPI_Comm comm )
{
    int d, i, j, k, l, max_atoms, my_max, all_max, ret;
    ivec c;
    grid * const g = &system->my_grid;
    grid_cell *gc;
    simulation_box * const my_ext_box = &system->my_ext_box;
    reax_atom * const atoms = system->my_atoms;

    Reset_Grid( g );

    for ( l = 0; l < system->n; l++ )
    {
        for ( d = 0; d < 3; ++d )
        {
            c[d] = (int) ((atoms[l].x[d] - my_ext_box->min[d]) * g->inv_len[d]);

            if ( c[d] >= g->native_end[d] )
            {
                c[d] = g->native_end[d] - 1;
            }
            else if ( c[d] < g->native_str[d] )
            {
                c[d] = g->native_str[d];
            }
        }

#if defined(DEBUG_FOCUS)
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

#if defined(DEBUG_FOCUS)
                fprintf( stderr, "p%d gc[%d,%d,%d]->top=%d\n",
                        system->my_rank, i, j, k, gc->top );
#endif
            }
        }
    }

    my_max = MAX( (int) CEIL( max_atoms * SAFE_ZONE ), MIN_GCELL_POPL );
    ret = MPI_Allreduce( &my_max, &all_max, 1, MPI_INT, MPI_MAX, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d max_atoms=%d, my_max=%d, all_max=%d\n",
            system->my_rank, max_atoms, my_max, all_max );
#endif

    return all_max;
}


void Allocate_Grid( reax_system * const system, MPI_Comm comm )
{
    int i, j, k;
    grid * const g = &system->my_grid;
    grid_cell *gc;
    int total;

    total = g->ncells[0] * g->ncells[1] * g->ncells[2];

    /* allocate gcell reordering space */
    g->order = scalloc( g->total + 1, sizeof(ivec), __FILE__, __LINE__ );

    /* allocate the gcells for the new grid */
    g->max_nbrs = (2 * g->vlist_span[0] + 1) * (2 * g->vlist_span[1] + 1)
        * (2 * g->vlist_span[2] + 1) + 3;

    g->cells = scalloc( total, sizeof(grid_cell), __FILE__, __LINE__ );

    for ( i = 0; i < total; i++ )
    {
        gc = &g->cells[i];
        gc->top = 0;
        gc->mark = 0;
    }

    g->str = scalloc( total, sizeof(int), __FILE__, __LINE__ );
    g->end = scalloc( total, sizeof(int), __FILE__, __LINE__ );
    g->cutoff = scalloc( total, sizeof(real), __FILE__, __LINE__ );
    g->nbrs_x = scalloc( total * g->max_nbrs, sizeof(ivec), __FILE__, __LINE__ );
    g->nbrs_cp = scalloc( total * g->max_nbrs, sizeof(rvec), __FILE__, __LINE__ );

    for ( i = 0; i < total * g->max_nbrs; i++ )
    {
        g->nbrs_x[i][0] = -1;
        g->nbrs_x[i][1] = -1;
        g->nbrs_x[i][2] = -1;
    }
    g->rel_box = scalloc( total, sizeof(ivec), __FILE__, __LINE__ );

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
                    scalloc( g->max_atoms, sizeof(int),
                            __FILE__, __LINE__ );
            }
        }
    }
}


void Deallocate_Grid( grid * const g )
{
    int i, j, k;
    grid_cell *gc;

    sfree( g->order, __FILE__, __LINE__ );
    sfree( g->str, __FILE__, __LINE__ );
    sfree( g->end, __FILE__, __LINE__ );
    sfree( g->cutoff, __FILE__, __LINE__ );
    sfree( g->nbrs_x, __FILE__, __LINE__ );
    sfree( g->nbrs_cp, __FILE__, __LINE__ );
    sfree( g->rel_box, __FILE__, __LINE__ );

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
                    sfree( gc->atoms, __FILE__, __LINE__ );
                }
            }
        }
    }
    sfree( g->cells, __FILE__, __LINE__ );
}


void Deallocate_MPI_Buffers( mpi_datatypes * const mpi_data )
{
    int i;
    mpi_out_data *mpi_buf;

    sfree( mpi_data->in1_buffer, __FILE__, __LINE__ );
    mpi_data->in1_buffer_size = 0;
    sfree( mpi_data->in2_buffer, __FILE__, __LINE__ );
    mpi_data->in2_buffer_size = 0;

    for ( i = 0; i < MAX_NBRS; ++i )
    {
        mpi_buf = &mpi_data->out_buffers[i];

        mpi_buf->cnt = 0;
        sfree( mpi_buf->index, __FILE__, __LINE__ );
        mpi_buf->index_size = 0;
        sfree( mpi_buf->out_atoms, __FILE__, __LINE__ );
        mpi_buf->out_atoms_size = 0;
    }

#if defined(NEUTRAL_TERRITORY)
    for ( i = 0; i < MAX_NT_NBRS; ++i )
    {
        sfree( mpi_data->in_nt_buffer[i], __FILE__, __LINE__ );

        mpi_buf = &mpi_data->out_nt_buffers[i];
        mpi_buf->cnt = 0;
        sfree( mpi_buf->index, __FILE__, __LINE__ );
        mpi_buf->index_size = 0;
        sfree( mpi_buf->out_atoms, __FILE__, __LINE__ );
        mpi_buf->out_atoms_size = 0;
    }
#endif
}


void Reallocate_Part1( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        mpi_datatypes * const mpi_data )
{
    int i, j, k, renbr;
    int * const realloc = workspace->realloc;
    grid * const g = &system->my_grid;

    renbr = (data->step - data->prev_steps) % control->reneighbor == 0 ? TRUE : FALSE;

    /* grid */
    if ( renbr == TRUE && realloc[RE_GCELL_ATOMS] > -1 )
    {
        for ( i = g->native_str[0]; i < g->native_end[0]; i++ )
        {
            for ( j = g->native_str[1]; j < g->native_end[1]; j++ )
            {
                for ( k = g->native_str[2]; k < g->native_end[2]; k++ )
                {
                    sfree( g->cells[ index_grid_3d(i, j, k, g) ].atoms,
                            __FILE__, __LINE__ );
                    g->cells[ index_grid_3d(i, j, k, g) ].atoms = scalloc( realloc[RE_GCELL_ATOMS],
                            sizeof(int), __FILE__, __LINE__ );
                }
            }
        }

        realloc[RE_GCELL_ATOMS] = -1;
    }
}


void Reallocate_Part2( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        mpi_datatypes * const mpi_data )
{
    int nflag, Nflag, renbr, total_cap_old;
    int * const realloc = workspace->realloc;
    sparse_matrix * const H = &workspace->H;

    renbr = (data->step - data->prev_steps) % control->reneighbor == 0 ? TRUE : FALSE;

    /* IMPORTANT: LOOSE ZONES CHECKS ARE DISABLED FOR NOW BY &&'ing with FALSE!!! */
    nflag = FALSE;
    if ( system->n >= (int) CEIL( DANGER_ZONE * system->local_cap )
            || (FALSE && system->n <= (int) CEIL( LOOSE_ZONE * system->local_cap )) )
    {
#if !defined(NEUTRAL_TERRITORY)
        nflag = TRUE;
#endif
        system->local_cap = (int) CEIL( system->n * SAFE_ZONE );
    }

#if defined(NEUTRAL_TERRITORY)
    if ( workspace->H->NT >= (int) CEIL( DANGER_ZONE * workspace->H->n_max ) )
    {
        nflag = TRUE;
        workspace->H->cap = (int) CEIL( workspace->H->NT * SAFE_ZONE_NT );
    }
#endif

    Nflag = FALSE;
    if ( system->N >= (int) CEIL( DANGER_ZONE * system->total_cap )
            || (FALSE && system->N <= (int) CEIL( LOOSE_ZONE * system->total_cap )) )
    {
        Nflag = TRUE;
        total_cap_old = system->total_cap;
        system->total_cap = (int) CEIL( system->N * SAFE_ZONE );
    }

    if ( nflag == TRUE )
    {
        Reallocate_System_Part1( system, system->local_cap );

        Deallocate_Workspace_Part1( control, workspace );
        Allocate_Workspace_Part1( system, control, workspace, system->local_cap );
    }

    if ( Nflag == TRUE )
    {
        Reallocate_System_Part2( system, total_cap_old, system->total_cap );

        Deallocate_Workspace_Part2( control, workspace );
        Allocate_Workspace_Part2( system, control, workspace, system->total_cap );
    }

    /* far neighbors */
    if ( renbr == TRUE && (Nflag == TRUE || realloc[RE_FAR_NBRS] == TRUE) )
    {
        Reallocate_Neighbor_List( lists[FAR_NBRS], system->total_cap, system->total_far_nbrs );
        Init_List_Indices( lists[FAR_NBRS], system->max_far_nbrs );
        realloc[RE_FAR_NBRS] = FALSE;
    }

    /* charge matrix */
    if ( nflag == TRUE || realloc[RE_CM] == TRUE )
    {
#if defined(NEUTRAL_TERRITORY)
        Reallocate_Matrix( H, H->n, system->local_cap,
                (int) CEIL( system->total_cm_entries * SAFE_ZONE_NT ) );
#else
        Reallocate_Matrix( H, system->n, system->local_cap, system->total_cm_entries );
#endif

        realloc[RE_CM] = FALSE;
    }

    /* bonds list */
    if ( Nflag == TRUE || realloc[RE_BONDS] == TRUE )
    {
        Reallocate_List( lists[BONDS], system->total_cap, system->total_bonds,
               TYP_BOND, lists[BONDS]->format );

        realloc[RE_BONDS] = FALSE;
        realloc[RE_THBODY] = TRUE;
    }

    /* hydrogen bonds list */
    if ( control->hbond_cut > 0.0
            && (Nflag == TRUE || realloc[RE_HBONDS] == TRUE) )
    {
        Reallocate_List( lists[HBONDS], system->total_cap, system->total_hbonds,
               TYP_HBOND, lists[HBONDS]->format );

        realloc[RE_HBONDS] = FALSE;
    }

    /* 3-body list */
    if ( realloc[RE_THBODY] == TRUE )
    {
        Reallocate_List( lists[THREE_BODIES], system->total_bonds, system->total_thbodies,
               TYP_THREE_BODY, lists[THREE_BODIES]->format );

        realloc[RE_THBODY] = FALSE;
    }
}
