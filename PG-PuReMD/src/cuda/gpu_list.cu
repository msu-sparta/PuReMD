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

#include "gpu_utils.h"


/* Allocate space for interaction list
 *
 * n: num. of elements to be allocated for list
 * max_intrs: max. num. of interactions for which to allocate space
 * type: list interaction type
 * l: pointer to list to be allocated
 * */
extern "C" void GPU_Make_List( int n, int max_intrs, int type,
        reax_list * const l )
{
    if ( l->allocated == TRUE )
    {
        fprintf( stderr, "[WARNING] attempted to allocate list which was already allocated."
                " Returning without allocation...\n" );
        return;
    }

    l->allocated = TRUE;
    l->n = n;
    l->max_intrs = max_intrs;
    l->type = type;
//    l->format = format;

    sCudaMalloc( (void **) &l->index, sizeof(int) * n, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &l->end_index, sizeof(int) * n, __FILE__, __LINE__ );

    switch ( l->type )
    {
        case TYP_FAR_NEIGHBOR:
            sCudaMalloc( (void **) &l->far_nbr_list.nbr, 
                    sizeof(int) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->far_nbr_list.rel_box, 
                    sizeof(ivec) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->far_nbr_list.d, 
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->far_nbr_list.dvec, 
                    sizeof(rvec) * l->max_intrs, __FILE__, __LINE__ );
            break;

        case TYP_BOND:
            sCudaMalloc( (void **) &l->bond_list_gpu.nbr,
                    sizeof(int) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.sym_index,
                    sizeof(int) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.dbond_index,
                    sizeof(int) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.rel_box,
                    sizeof(ivec) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.d,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.dvec,
                    sizeof(rvec) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.BO,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.BO_s,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.BO_pi,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.BO_pi2,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.Cdbo,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.Cdbopi,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.Cdbopi2,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.C1dbo,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.C2dbo,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.C3dbo,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.C1dbopi,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.C2dbopi,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.C3dbopi,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.C4dbopi,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.C1dbopi2,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.C2dbopi2,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.C3dbopi2,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.C4dbopi2,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.dBOp,
                    sizeof(rvec) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.dln_BOp_s,
                    sizeof(rvec) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.dln_BOp_pi,
                    sizeof(rvec) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.dln_BOp_pi2,
                    sizeof(rvec) * l->max_intrs, __FILE__, __LINE__ );
#if !defined(GPU_KERNEL_ATOMIC)
            sCudaMalloc( (void **) &l->bond_list_gpu.CdDelta_multi,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.CdDelta_val,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.CdDelta_tor,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.f_val,
                    sizeof(rvec) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.f_tor,
                    sizeof(rvec) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.f_hb,
                    sizeof(rvec) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.f_bo,
                    sizeof(rvec) * l->max_intrs, __FILE__, __LINE__ );
#endif
#if !defined(GPU_STREAM_SINGLE_ACCUM)
            sCudaMalloc( (void **) &l->bond_list_gpu.Cdbo_bonds,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.Cdbo_multi,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.Cdbo_hbonds,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.Cdbo_tor,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.Cdbopi_bonds,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.Cdbopi_multi,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.Cdbopi_tor,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.Cdbopi2_bonds,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->bond_list_gpu.Cdbopi2_multi,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
#endif
            break;

        case TYP_HBOND:
            sCudaMalloc( (void **) &l->hbond_list.nbr, 
                    sizeof(int) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->hbond_list.scl, 
                    sizeof(int) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->hbond_list.ptr, 
                    sizeof(int) * l->max_intrs, __FILE__, __LINE__ );
#if (defined(HAVE_CUDA) || defined(HAVE_HIP)) && !defined(GPU_KERNEL_ATOMIC)
//            sCudaMalloc( (void **) &l->hbond_list.sym_index, 
//                    sizeof(int) * l->max_intrs, __FILE__, __LINE__ );
//            sCudaMalloc( (void **) &l->hbond_list.f_hb, 
//                    sizeof(rvec) * l->max_intrs, __FILE__, __LINE__ );
#endif
            break;            

        case TYP_THREE_BODY:
            sCudaMalloc( (void **) &l->three_body_list_gpu.thb,
                    sizeof(int) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->three_body_list_gpu.pthb,
                    sizeof(int) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->three_body_list_gpu.theta,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->three_body_list_gpu.cos_theta,
                    sizeof(real) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->three_body_list_gpu.dcos_di,
                    sizeof(rvec) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->three_body_list_gpu.dcos_dj,
                    sizeof(rvec) * l->max_intrs, __FILE__, __LINE__ );
            sCudaMalloc( (void **) &l->three_body_list_gpu.dcos_dk,
                    sizeof(rvec) * l->max_intrs, __FILE__, __LINE__ );
            break;

        default:
            fprintf( stderr, "[ERROR] unknown devive list type (%d)\n", l->type );
            MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
            break;
    }
}


extern "C" void GPU_Delete_List( reax_list * const l )
{
    if ( l->allocated == FALSE )
    {
        fprintf( stderr, "[WARNING] attempted to free list which was not allocated."
                " Returning without deallocation...\n" );
        return;
    }

    l->allocated = FALSE;
    l->n = 0;
    l->max_intrs = 0;

    sCudaFree( l->index, __FILE__, __LINE__ );
    sCudaFree( l->end_index, __FILE__, __LINE__ );

    switch ( l->type )
    {
        case TYP_FAR_NEIGHBOR:
            sCudaFree( l->far_nbr_list.nbr, __FILE__, __LINE__ );
            sCudaFree( l->far_nbr_list.rel_box, __FILE__, __LINE__ );
            sCudaFree( l->far_nbr_list.d, __FILE__, __LINE__ );
            sCudaFree( l->far_nbr_list.dvec, __FILE__, __LINE__ );
            break;

        case TYP_BOND:
            sCudaFree( l->bond_list_gpu.nbr, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.sym_index, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.dbond_index, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.rel_box, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.d, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.dvec, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.BO, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.BO_s, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.BO_pi, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.BO_pi2, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.Cdbo, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.Cdbopi, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.Cdbopi2, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.C1dbo, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.C2dbo, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.C3dbo, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.C1dbopi, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.C2dbopi, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.C3dbopi, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.C4dbopi, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.C1dbopi2, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.C2dbopi2, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.C3dbopi2, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.C4dbopi2, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.dBOp, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.dln_BOp_s, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.dln_BOp_pi, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.dln_BOp_pi2, __FILE__, __LINE__ );
#if !defined(GPU_KERNEL_ATOMIC)
            sCudaFree( l->bond_list_gpu.CdDelta_multi, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.CdDelta_val, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.CdDelta_tor, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.f_val, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.f_tor, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.f_hb, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.f_bo, __FILE__, __LINE__ );
#endif
#if !defined(GPU_STREAM_SINGLE_ACCUM)
            sCudaFree( l->bond_list_gpu.Cdbo_bonds, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.Cdbo_multi, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.Cdbo_hbonds, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.Cdbo_tor, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.Cdbopi_bonds, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.Cdbopi_multi, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.Cdbopi_tor, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.Cdbopi2_bonds, __FILE__, __LINE__ );
            sCudaFree( l->bond_list_gpu.Cdbopi2_multi, __FILE__, __LINE__ );
#endif
            break;

        case TYP_HBOND:
            sCudaFree( l->hbond_list.nbr, __FILE__, __LINE__ );
            sCudaFree( l->hbond_list.scl, __FILE__, __LINE__ );
            sCudaFree( l->hbond_list.ptr, __FILE__, __LINE__ );
#if (defined(HAVE_CUDA) || defined(HAVE_HIP)) && !defined(GPU_KERNEL_ATOMIC)
//            sCudaFree( l->hbond_list.sym_index, __FILE__, __LINE__ );
//            sCudaFree( l->hbond_list.f_hb, __FILE__, __LINE__ );
#endif
            break;

        case TYP_THREE_BODY:
            sCudaFree( l->three_body_list_gpu.thb, __FILE__, __LINE__ );
            sCudaFree( l->three_body_list_gpu.pthb, __FILE__, __LINE__ );
            sCudaFree( l->three_body_list_gpu.theta, __FILE__, __LINE__ );
            sCudaFree( l->three_body_list_gpu.cos_theta, __FILE__, __LINE__ );
            sCudaFree( l->three_body_list_gpu.dcos_di, __FILE__, __LINE__ );
            sCudaFree( l->three_body_list_gpu.dcos_dj, __FILE__, __LINE__ );
            sCudaFree( l->three_body_list_gpu.dcos_dk, __FILE__, __LINE__ );
            break;

        default:
            fprintf( stderr, "[ERROR] unknown devive list type (%d)\n", l->type );
            MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
            break;
    }
}
