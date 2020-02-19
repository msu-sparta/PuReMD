
#include "cuda_forces.h"

#include "cuda_bonds.h"
#include "cuda_bond_orders.h"
#include "cuda_charges.h"
#include "cuda_helpers.h"
#include "cuda_hydrogen_bonds.h"
#include "cuda_lin_alg.h"
#include "cuda_list.h"
#include "cuda_multi_body.h"
#include "cuda_neighbors.h"
#include "cuda_nonbonded.h"
#include "cuda_reduction.h"
#include "cuda_torsion_angles.h"
#include "cuda_utils.h"
#include "cuda_valence_angles.h"

#include "../basic_comm.h"
#include "../forces.h"
#include "../index_utils.h"
#include "../tool_box.h"
#include "../vector.h"


typedef enum
{
    DIAGONAL = 0,
    OFF_DIAGONAL = 1,
} MATRIX_ENTRY_POSITION;


CUDA_DEVICE real Init_Charge_Matrix_Entry( single_body_parameters *sbp_i, real *ctap,
        control_params *control, int i, int j, real r_ij, real gamma, MATRIX_ENTRY_POSITION pos )
{
    real taper, dr3gamij_1, dr3gamij_3, ret;

    ret = 0.0;

    switch ( control->charge_method )
    {
    case QEQ_CM:
    case EE_CM:
    case ACKS2_CM:
        switch ( pos )
        {
            case OFF_DIAGONAL:
                taper = ctap[7] * r_ij + ctap[6];
                taper = taper * r_ij + ctap[5];
                taper = taper * r_ij + ctap[4];
                taper = taper * r_ij + ctap[3];
                taper = taper * r_ij + ctap[2];
                taper = taper * r_ij + ctap[1];
                taper = taper * r_ij + ctap[0];    

                /* shielding */
                dr3gamij_1 = r_ij * r_ij * r_ij + gamma;
                dr3gamij_3 = POW( dr3gamij_1 , 1.0 / 3.0 );

                //TODO: investigate why conditional is excluded (OpenMP code below)
//                ret = ((i == j) ? 0.5 : 1.0) * Tap * EV_to_KCALpMOL / dr3gamij_3;
                ret = taper * EV_to_KCALpMOL / dr3gamij_3;
                break;

            case DIAGONAL:
                ret = sbp_i->eta;
                break;

            default:
//                fprintf( stderr, "[ERROR] Invalid matrix position. Terminating...\n" );
//                exit( INVALID_INPUT );
                break;
        }
        break;


    default:
//        fprintf( stderr, "[ERROR] Invalid charge method. Terminating...\n" );
//        exit( INVALID_INPUT );
        break;
    }

    return ret;
}


CUDA_DEVICE real Init_Charge_Matrix_Entry_Tab( LR_lookup_table *t_LR, real r_ij,
        int ti, int tj, int num_atom_types )
{
    int r, tmin, tmax;
    real val, dif, base;
    LR_lookup_table *t; 

    tmin = MIN( ti, tj );
    tmax = MAX( ti, tj );
    t = &t_LR[ index_lr(tmin,tmax, num_atom_types) ];

    /* cubic spline interpolation */
    r = (int)(r_ij * t->inv_dx);
    if ( r == 0 )
    {
        ++r;
    }
    base = (real)(r + 1) * t->dx;
    dif = r_ij - base;
    val = ((t->ele[r].d * dif + t->ele[r].c) * dif + t->ele[r].b) * dif
        + t->ele[r].a;
    val *= EV_to_KCALpMOL / C_ELE;

    return val;
}


CUDA_GLOBAL void k_disable_hydrogen_bonding( control_params *control )
{
    control->hbond_cut = 0.0;
}


CUDA_GLOBAL void k_init_end_index( int * intr_cnt, int *indices, int *end_indices, int N )
{
    int i;

    i = blockIdx.x * blockDim.x  + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    end_indices[i] = indices[i] + intr_cnt[i];
}


CUDA_GLOBAL void k_setup_hindex( reax_atom *my_atoms, int N )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    my_atoms[i].Hindex = i;
}


CUDA_GLOBAL void k_init_hbond_indices( reax_atom * atoms, single_body_parameters *sbp,
        int *hbonds, int *max_hbonds, int *indices, int *end_indices, int N )
{
    int i, hindex, my_hbonds;

    i = blockIdx.x * blockDim.x  + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    hindex = atoms[i].Hindex;

    if ( sbp[ atoms[i].type ].p_hbond == H_ATOM
            || sbp[ atoms[i].type ].p_hbond == H_BONDING_ATOM )
    {
        my_hbonds = hbonds[i];
        indices[hindex] = max_hbonds[i];
        end_indices[hindex] = indices[hindex] + hbonds[i];
    }
    else
    {
        my_hbonds = 0;
        indices[hindex] = 0;
        end_indices[hindex] = 0;
    }
    atoms[i].num_hbonds = my_hbonds;
}


CUDA_GLOBAL void k_print_hbond_info( reax_atom *my_atoms, single_body_parameters *sbp, 
        control_params *control, reax_list hbond_list, int N )
{
    int i;
    int type_i;
    int ihb, ihb_top;
    single_body_parameters *sbp_i;
    reax_atom *atom_i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    atom_i = &my_atoms[i];
    type_i = atom_i->type;
    sbp_i = &sbp[type_i];

    if ( control->hbond_cut > 0.0 )
    {
        ihb = sbp_i->p_hbond;

        if ( ihb == H_ATOM  || ihb == H_BONDING_ATOM )
        {
            ihb_top = Start_Index( atom_i->Hindex, &hbond_list );
        }
        else
        {
            ihb_top = -1;
        }
    }

    printf( "atom %6d: ihb = %2d, ihb_top = %2d\n", i, ihb, ihb_top );
}


CUDA_GLOBAL void k_init_forces( reax_atom *my_atoms, single_body_parameters *sbp, 
        two_body_parameters *tbp, storage workspace, control_params *control, 
        reax_list far_nbr_list, reax_list bond_list, reax_list hbond_list, 
        LR_lookup_table *t_LR, int n, int N, int num_atom_types, int renbr,
        int *max_cm_entries, int *realloc_cm_entries,
        int *max_bonds, int *realloc_bonds,
        int *max_hbonds, int *realloc_hbonds )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int Htop, btop_i, ihb, jhb, ihb_top;
    int num_bonds, num_hbonds, num_cm_entries;
    int local, flag, flag2, flag3;
    real r_ij, cutoff;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    reax_atom *atom_i, *atom_j;
    sparse_matrix *H;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    H = &workspace.H;
    Htop = H->start[i];

    atom_i = &my_atoms[i];
    type_i = atom_i->type;
    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    btop_i = Start_Index( i, &bond_list );
    sbp_i = &sbp[type_i];

    if ( i < n )
    {
        local = TRUE;
        cutoff = control->nonb_cut;

        //update bond mark here
        workspace.bond_mark[i] = 0;
    }
    else
    {
        local = FALSE;
        cutoff = control->bond_cut;

        //update bond mark here
        workspace.bond_mark[i] = 1000;
    }

    ihb = NON_H_BONDING_ATOM;
    ihb_top = -1;

    if ( local == TRUE )
    {
        H->entries[Htop].j = i;
        H->entries[Htop].val = Init_Charge_Matrix_Entry( sbp_i, workspace.Tap, control,
                i, H->entries[Htop].j, 0.0, 0.0, DIAGONAL );
        ++Htop;
    }

    if ( control->hbond_cut > 0.0 )
    {
        ihb = sbp_i->p_hbond;

        if ( ihb == H_ATOM || ihb == H_BONDING_ATOM )
        {
            ihb_top = Start_Index( atom_i->Hindex, &hbond_list );
        }
        else
        {
            ihb_top = -1;
        }
    }

    /* update i-j distance - check if j is within cutoff */
    for ( pj = start_i; pj < end_i; ++pj )
    {
        j = far_nbr_list.far_nbr_list.nbr[pj];
        atom_j = &my_atoms[j];

        if ( renbr )
        {
            if ( far_nbr_list.far_nbr_list.d[pj] <= cutoff )
            {
                flag = TRUE;
            }
            else
            {
                flag = FALSE;
            }

            if ( far_nbr_list.far_nbr_list.d[pj] <= control->nonb_cut )
            {
                flag2 = TRUE;
            }
            else
            {
                flag2 = FALSE;
            }

        }
        else
        {
            if ( i < j )
            {
                far_nbr_list.far_nbr_list.dvec[pj][0] = atom_j->x[0] - atom_i->x[0];
                far_nbr_list.far_nbr_list.dvec[pj][1] = atom_j->x[1] - atom_i->x[1];
                far_nbr_list.far_nbr_list.dvec[pj][2] = atom_j->x[2] - atom_i->x[2];
            }
            else
            {
                far_nbr_list.far_nbr_list.dvec[pj][0] = atom_i->x[0] - atom_j->x[0];
                far_nbr_list.far_nbr_list.dvec[pj][1] = atom_i->x[1] - atom_j->x[1];
                far_nbr_list.far_nbr_list.dvec[pj][2] = atom_i->x[2] - atom_j->x[2];
            }
            far_nbr_list.far_nbr_list.d[pj] = rvec_Norm_Sqr( far_nbr_list.far_nbr_list.dvec[pj] );

            if ( far_nbr_list.far_nbr_list.d[pj] <= SQR( control->nonb_cut ) )
            {
                flag2 = TRUE;
            }
            else
            {
                flag2 = FALSE;
            }

            if ( far_nbr_list.far_nbr_list.d[pj] <= SQR( control->nonb_cut ) )
            {
                far_nbr_list.far_nbr_list.d[pj] = SQRT( far_nbr_list.far_nbr_list.d[pj] );
                flag = TRUE;
            }
            else
            {
                flag = FALSE;
            }
        }
        if ( flag2 == TRUE )
        {
            type_j = atom_j->type;
            sbp_j = &sbp[type_j];
            ihb = sbp_i->p_hbond;
            jhb = sbp_j->p_hbond;

            /* atom i: H bonding, ghost
             * atom j: H atom, native */
            if ( control->hbond_cut > 0.0
                    && far_nbr_list.far_nbr_list.d[pj] <= control->hbond_cut
                    && ihb == H_BONDING_ATOM && jhb == H_ATOM && i >= n && j < n ) 
            {
                hbond_list.hbond_list[ihb_top].nbr = j;
                hbond_list.hbond_list[ihb_top].scl = -1;
                hbond_list.hbond_list[ihb_top].ptr = pj;

                /* CUDA-specific */
                hbond_list.hbond_list[ihb_top].sym_index = -1;
                rvec_MakeZero( hbond_list.hbond_list[ihb_top].hb_f );

                ++ihb_top;
            }

            //if ((i < n) || (j < n))
            //if (local == TRUE || ((i >= n) &&(j < n)))

            flag3 = FALSE;
            if ( i < j && i < n && (j < n || atom_i->orig_id < atom_j->orig_id) )
            {
                flag3 = TRUE;
            }
            else if ( i > j && i >= n && j < n && atom_j->orig_id < atom_i->orig_id )
            {
                flag3 = TRUE;
            }
            else if ( i > j && i < n && (j < n || atom_j->orig_id < atom_i->orig_id ) )
            {
                flag3 = TRUE;
            }

            if ( flag3 == TRUE )
            {
                twbp = &tbp[ index_tbp(type_i,type_j,num_atom_types) ];
                r_ij = far_nbr_list.far_nbr_list.d[pj];

                //if (renbr) {
                H->entries[Htop].j = j;
                if ( control->tabulate == 0 )
                {
                    H->entries[Htop].val = Init_Charge_Matrix_Entry( sbp_i, workspace.Tap,
                            control, i, H->entries[Htop].j, r_ij, twbp->gamma, OFF_DIAGONAL );
                }
                else
                {
                    H->entries[Htop].val = Init_Charge_Matrix_Entry_Tab( t_LR, r_ij, type_i, type_j,num_atom_types );
                }
                //}
                ++Htop;
            }
        }

        if ( flag == TRUE )
        {
            type_j = atom_j->type;
            r_ij = far_nbr_list.far_nbr_list.d[pj];
            sbp_j = &sbp[type_j];
            twbp = &tbp[ index_tbp(type_i, type_j, num_atom_types) ];

            if ( local == TRUE )
            {
                /* H matrix entry */
//                if( j < n || atom_i->orig_id < atom_j->orig_id ) {//tryQEq||1
//                    H->entries[Htop].j = j;
//                    if( control->tabulate == 0 )
//                            H->entries[Htop].val = Init_Charge_Matrix_Entry( sbp_i, workspace.Tap,
//                                    control, i, H->entries[Htop].j, r_ij, twbp->gamma, OFF_DIAGONAL );
//                    else
//                        H->entries[Htop].val = Init_Charge_Matrix_Entry_Tab(t_LR, r_ij, type_i, type_j,num_atom_types);
//                    ++Htop;
//                } 
//                else if( j < n || atom_i->orig_id > atom_j->orig_id ) {//tryQEq||1
//                    H->entries[Htop].j = j;
//                    if( control->tabulate == 0 )
//                            H->entries[Htop].val = Init_Charge_Matrix_Entry( sbp_i, workspace.Tap,
//                                    control, i, H->entries[Htop].j, r_ij, twbp->gamma, OFF_DIAGONAL );
//                    else
//                        H->entries[Htop].val = Init_Charge_Matrix_Entry_Tab(t_LR, r_ij, type_i, type_j,num_atom_types);
//                    ++Htop;
//                } 

                /* hydrogen bond lists */
                if ( control->hbond_cut > 0.0
                        && (ihb == H_ATOM || ihb == H_BONDING_ATOM)
                        && far_nbr_list.far_nbr_list.d[pj] <= control->hbond_cut )
                {
                    jhb = sbp_j->p_hbond;

                    /* atom i: H atom, native
                     * atom j: H bonding atom */
                    if ( ihb == H_ATOM && jhb == H_BONDING_ATOM )
                    {
                        hbond_list.hbond_list[ihb_top].nbr = j;

                        if ( i < j )
                        {
                            hbond_list.hbond_list[ihb_top].scl = 1;
                        }
                        else
                        {
                            hbond_list.hbond_list[ihb_top].scl = -1;
                        }
                        hbond_list.hbond_list[ihb_top].ptr = pj;

                        /* CUDA-specific */
                        hbond_list.hbond_list[ihb_top].sym_index = -1;
                        rvec_MakeZero( hbond_list.hbond_list[ihb_top].hb_f );

                        ++ihb_top;
                    }
                    /* atom i: H bonding atom, native
                     * atom j: H atom, native */
                    else if ( ihb == H_BONDING_ATOM && jhb == H_ATOM && j < n )
                    {
                        //jhb_top = End_Index( atom_j->Hindex, &hbond_list );
                        hbond_list.hbond_list[ihb_top].nbr = j;
                        hbond_list.hbond_list[ihb_top].scl = -1;
                        hbond_list.hbond_list[ihb_top].ptr = pj;

                        //CUDA SPECIFIC
                        hbond_list.hbond_list[ihb_top].sym_index = -1;
                        rvec_MakeZero( hbond_list.hbond_list[ihb_top].hb_f );

                        ++ihb_top;
                    }
                }
            }

            /* uncorrected bond orders */
            if ( far_nbr_list.far_nbr_list.d[pj] <= control->bond_cut
                    && Cuda_BOp( bond_list, control->bo_cut,
                        i, btop_i, far_nbr_list.far_nbr_list.nbr[pj],
                        &far_nbr_list.far_nbr_list.rel_box[pj], far_nbr_list.far_nbr_list.d[pj],
                        &far_nbr_list.far_nbr_list.dvec[pj], far_nbr_list.format,
                        sbp_i, sbp_j, twbp, workspace.dDeltap_self,
                        workspace.total_bond_order ) == TRUE )
            {
                ++btop_i;

                /* TODO: Need to do later... since i and j are parallel */
//                if( workspace->bond_mark[j] > workspace->bond_mark[i] + 1 )
//                {
//                    workspace->bond_mark[j] = workspace->bond_mark[i] + 1;
//                }
//                else if( workspace->bond_mark[i] > workspace->bond_mark[j] + 1 )
//                {
//                    workspace->bond_mark[i] = workspace->bond_mark[j] + 1;
//                }
            }
        }
    }

    Set_End_Index( i, btop_i, &bond_list );
    H->end[i] = Htop;
//    if( local == TRUE )
//    {
        if ( control->hbond_cut > 0.0 && ihb_top > 0 && (ihb == H_ATOM || ihb == H_BONDING_ATOM) )
        {
            Set_End_Index( atom_i->Hindex, ihb_top, &hbond_list );
        }
//    }

    num_bonds = btop_i - Start_Index( i, &bond_list );
    num_hbonds = ihb_top - Start_Index( atom_i->Hindex, &hbond_list );
    num_cm_entries = Htop - H->start[i];

    /* copy (h)bond info to atom structure
     * (needed for atom ownership transfer via MPI) */
    my_atoms[i].num_bonds = num_bonds;
    my_atoms[i].num_hbonds = num_hbonds;

    /* reallocation checks */
    if ( num_bonds > max_bonds[i] )
    {
        *realloc_bonds = TRUE;
    }

    if ( num_hbonds > max_hbonds[i] )
    {
        *realloc_hbonds = TRUE;
    }

    if ( num_cm_entries > max_cm_entries[i] )
    {
        *realloc_cm_entries = TRUE;
    }
}


CUDA_GLOBAL void k_estimate_storages( reax_atom *my_atoms, 
        single_body_parameters *sbp, two_body_parameters *tbp,
        control_params *control, reax_list p_far_nbr_list, 
        int num_atom_types, int n, int N, int total_cap,
        int *cm_entries, int *max_cm_entries,
        int *bonds, int *max_bonds,
        int *hbonds, int *max_hbonds )
{
    int i, j, pj; 
    int start_i, end_i;
    int type_i, type_j;
    int ihb, jhb;
    int local;
    int num_bonds, num_hbonds, num_cm_entries;
    real cutoff;
    real r_ij; 
    real C12, C34, C56;
    real BO, BO_s, BO_pi, BO_pi2;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    reax_atom *atom_i, *atom_j;
    reax_list *far_nbr_list;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= total_cap )
    {
        return;
    }

    far_nbr_list = &p_far_nbr_list;
    num_bonds = 0;
    num_hbonds = 0;
    num_cm_entries = 0;

    if ( i < N )
    {
        atom_i = &my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, far_nbr_list );
        end_i = End_Index( i, far_nbr_list );
        sbp_i = &sbp[type_i];

        if ( i < n )
        { 
            local = TRUE;
            cutoff = control->nonb_cut;
            ++num_cm_entries;
//            ihb = sbp_i->p_hbond;
        }   
        else
        {
            local = FALSE;
            cutoff = control->bond_cut;
//            ihb = NON_H_BONDING_ATOM; 
        } 

        ihb = NON_H_BONDING_ATOM; 

        for ( pj = start_i; pj < end_i; ++pj )
        { 
            j = far_nbr_list->far_nbr_list.nbr[pj];
            atom_j = &my_atoms[j];

            if ( far_nbr_list->far_nbr_list.d[pj] <= control->nonb_cut )
            {
                type_j = my_atoms[j].type;
                sbp_j = &sbp[type_j];
                ihb = sbp_i->p_hbond;
                jhb = sbp_j->p_hbond;

                if ( local == TRUE )
                {
                    if ( i < j && (j < n || atom_i->orig_id < atom_j->orig_id) )
                    {
                        ++num_cm_entries;
                    }
                    else if ( i > j && (j < n || atom_j->orig_id > atom_i->orig_id) )
                    {
                        ++num_cm_entries;
                    }
                }
                else
                {
                    if ( i > j && j < n && atom_j->orig_id < atom_i->orig_id )
                    {
                        ++num_cm_entries;
                    }
                }

                /* atom i: H bonding, ghost
                 * atom j: H atom, native */
                if ( control->hbond_cut > 0.0
                        && far_nbr_list->far_nbr_list.d[pj] <= control->hbond_cut 
                        && ihb == H_BONDING_ATOM && jhb == H_ATOM && i >= n && j < n )
                {
                    ++num_hbonds;
                }

//                if ( i >= n )
//                {
//                    ihb = NON_H_BONDING_ATOM;
//                }
            }

            if ( far_nbr_list->far_nbr_list.d[pj] <= cutoff )
            {
                type_j = my_atoms[j].type;
                r_ij = far_nbr_list->far_nbr_list.d[pj];
                sbp_j = &sbp[type_j];
                twbp = &tbp[ index_tbp(type_i ,type_j, num_atom_types) ];

                if ( local == TRUE )
                {
                    /* atom i: H atom OR H bonding atom, native */
                    if ( control->hbond_cut > 0.0
                            && (ihb == H_ATOM || ihb == H_BONDING_ATOM)
                            && far_nbr_list->far_nbr_list.d[pj] <= control->hbond_cut )
                    {
                        jhb = sbp_j->p_hbond;

                        /* atom i: H atom, native
                         * atom j: H bonding atom */
                        if( ihb == H_ATOM && jhb == H_BONDING_ATOM )
                        {
                            ++num_hbonds;
                        }
                        /* atom i: H bonding atom, native
                         * atom j: H atom, native */
                        else if( ihb == H_BONDING_ATOM && jhb == H_ATOM && j < n )
                        {
                            ++num_hbonds;
                        }
                    }
                }

                /* uncorrected bond orders */
                if ( far_nbr_list->far_nbr_list.d[pj] <= control->bond_cut )
                {
                    if ( sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0 )
                    {
                        C12 = twbp->p_bo1 * POW( r_ij / twbp->r_s, twbp->p_bo2 );
                        BO_s = (1.0 + control->bo_cut) * EXP( C12 );
                    }
                    else
                    {
                        C12 = 0.0;
                        BO_s = 0.0;
                    }

                    if ( sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0 )
                    {
                        C34 = twbp->p_bo3 * POW( r_ij / twbp->r_p, twbp->p_bo4 );
                        BO_pi = EXP( C34 );
                    }
                    else
                    {
                        C34 = 0.0;
                        BO_pi = 0.0;
                    }

                    if ( sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0 )
                    {
                        C56 = twbp->p_bo5 * POW( r_ij / twbp->r_pp, twbp->p_bo6 );
                        BO_pi2= EXP( C56 );
                    }
                    else
                    {
                        C56 = 0.0;
                        BO_pi2 = 0.0;
                    }

                    /* initially BO values are the uncorrected ones, page 1 */
                    BO = BO_s + BO_pi + BO_pi2;

                    if ( BO >= control->bo_cut )
                    {
                        ++num_bonds;
                    }
                }
            }
        }
    }

    bonds[i] = num_bonds;
    max_bonds[i] = MAX( (int)(num_bonds * 2), MIN_BONDS );

    hbonds[i] = num_hbonds;
    max_hbonds[i] = MAX( (int)(num_hbonds * SAFE_ZONE), MIN_HBONDS );

    cm_entries[i] = num_cm_entries;
    max_cm_entries[i] = MAX( (int)(num_cm_entries * SAFE_ZONE), MIN_CM_ENTRIES );
}


CUDA_GLOBAL void k_init_bond_mark( int offset, int n, int *bond_mark )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    bond_mark[offset + threadIdx.x] = 1000;
}


CUDA_GLOBAL void k_update_sym_dbond_indices( reax_list bond_list, int N )
{
    int i, pj, pk, nbr_ij, nbr_jk;
    bond_data *ibond, *jbond;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    /* i-j bonds */
    for ( pj = Start_Index(i, &bond_list); pj < End_Index(i, &bond_list); ++pj )
    {
        ibond = &bond_list.bond_list[pj];
        nbr_ij = ibond->nbr;

        /* j-k bonds */
        for ( pk = Start_Index(nbr_ij, &bond_list); pk < End_Index(nbr_ij, &bond_list); ++pk )
        {
            jbond = &bond_list.bond_list[pk];
            nbr_jk = jbond->nbr;

            if ( i == nbr_jk && i > nbr_ij )
            {
                ibond->dbond_index = pj;
                jbond->dbond_index = pj;

                ibond->sym_index = pk;
                jbond->sym_index = pj;
            }
        }
    }
}


CUDA_GLOBAL void k_update_sym_hbond_indices( reax_atom *my_atoms, reax_list hbond_list, int N )
{
    int i, j, k;
    int nbr, nbrstart, nbrend;
    int start, end;
    hbond_data *ihbond, *jhbond;
    int __THREADS_PER_ATOM__;
    int thread_id;
    int warp_id;
    int lane_id;

    __THREADS_PER_ATOM__ = HB_KER_SYM_THREADS_PER_ATOM;
    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    warp_id = thread_id / __THREADS_PER_ATOM__;

    if ( warp_id > N )
    {
        return;
    }

    lane_id = thread_id & (__THREADS_PER_ATOM__ - 1);
    i = warp_id;
    start = Start_Index( my_atoms[i].Hindex, &hbond_list );
    end = End_Index( my_atoms[i].Hindex, &hbond_list );
    j = start + lane_id;

    while ( j < end )
    {
        ihbond = &hbond_list.hbond_list[j];
        nbr = ihbond->nbr;

        nbrstart = Start_Index( my_atoms[nbr].Hindex, &hbond_list );
        nbrend = End_Index( my_atoms[nbr].Hindex, &hbond_list );

        for ( k = nbrstart; k < nbrend; k++ )
        {
            jhbond = &hbond_list.hbond_list[k];

            if ( jhbond->nbr == i )
            {
                ihbond->sym_index = k;
                jhbond->sym_index = j;
                break;
            }
        }

        j += __THREADS_PER_ATOM__;
    }
}


#if defined(DEBUG_FOCUS)
CUDA_GLOBAL void k_print_forces( reax_atom *my_atoms, rvec *f, int n )
{
    int i; 

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    printf( "%8d: %24.15f, %24.15f, %24.15f\n",
            my_atoms[i].orig_id,
            f[i][0],
            f[i][1],
            f[i][2] );
}


CUDA_GLOBAL void k_print_hbonds( reax_atom *my_atoms, reax_list hbond_list, int n, int rank, int step )
{
    int i, k, pj, start, end; 
    hbond_data *hbond_jk;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    start = Start_Index( my_atoms[i].Hindex, &hbond_list );
    end = End_Index( my_atoms[i].Hindex, &hbond_list );

    for ( pj = start; pj < end; ++pj )
    {
        k = hbond_list.hbond_list[pj].nbr;
        hbond_jk = &hbond_list.hbond_list[pj];

        printf( "p%03d, step %05d: %8d: %8d, %24.15f, %24.15f, %24.15f\n",
                rank, step, my_atoms[i].Hindex, k,
                hbond_jk->hb_f[0],
                hbond_jk->hb_f[1],
                hbond_jk->hb_f[2] );
    }
}
#endif


CUDA_GLOBAL void k_bond_mark( reax_list p_bond_list, storage p_workspace, int N )
{
    int i, j, k;
    reax_list *bond_list;
    storage *workspace;

//    i = blockIdx.x * blockDim.x + threadIdx.x;
//    if ( i >= N )
//    {
//        return;
//    }

    bond_list = &p_bond_list;
    workspace = &p_workspace;

    for ( i = 0; i < N; i++ )
    {
        for ( k = Start_Index( i, bond_list ); k < End_Index( i, bond_list ); k++ )
        {
            j = bond_list->bond_list[k].nbr;

            if ( i < j )
            {
                if ( workspace->bond_mark[j] > (workspace->bond_mark[i] + 1) )
                {
                    workspace->bond_mark[j] = workspace->bond_mark[i] + 1;    
                }
                else if ( workspace->bond_mark[i] > (workspace->bond_mark[j] + 1) )
                {
                    workspace->bond_mark[i] = workspace->bond_mark[j] + 1;
                }
            }
        }
    }
}


static int Cuda_Estimate_Storage_Three_Body( reax_system *system, control_params *control, 
        storage *workspace, int step, reax_list **lists, int *thbody )
{
    int ret;

    ret = SUCCESS;

    cuda_memset( thbody, 0, system->total_bonds * sizeof(int),
            "Cuda_Estimate_Storage_Three_Body::thbody" );

    Estimate_Cuda_Valence_Angles <<< control->blocks_n, control->block_size >>>
        ( system->d_my_atoms, (control_params *)control->d_control_params, 
          *(lists[BONDS]), system->n, system->N, thbody );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    Cuda_Reduction_Sum( thbody, system->d_total_thbodies, system->total_bonds );

    copy_host_device( &system->total_thbodies, system->d_total_thbodies, sizeof(int),
            cudaMemcpyDeviceToHost, "Cuda_Estimate_Storage_Three_Body::d_total_thbodies" );

    if ( step == 0 )
    {
        system->total_thbodies = MAX( (int)(system->total_thbodies * SAFE_ZONE), MIN_3BODIES );
        system->total_thbodies_indices = system->total_bonds;

        /* create Three-body list */
        Cuda_Make_List( system->total_thbodies_indices, system->total_thbodies,
                TYP_THREE_BODY, lists[THREE_BODIES] );
    }

    if ( system->total_thbodies > lists[THREE_BODIES]->max_intrs ||
            system->total_bonds > lists[THREE_BODIES]->n )
    {
        if ( system->total_thbodies > lists[THREE_BODIES]->max_intrs )
        {
            system->total_thbodies = MAX( (int)(lists[THREE_BODIES]->max_intrs * SAFE_ZONE),
                    system->total_thbodies );
        }
        if ( system->total_bonds > lists[THREE_BODIES]->n )
        {
            system->total_thbodies_indices = MAX( (int)(lists[THREE_BODIES]->n * SAFE_ZONE),
                    system->total_bonds );
        }

        workspace->d_workspace->realloc.thbody = TRUE;
        ret = FAILURE;
    }

    return ret;
}


#if defined(DEBUG_FOCUS)
static void Print_Forces( reax_system *system )
{
    int blocks;
    
    blocks = (system->n) / DEF_BLOCK_SIZE + 
        (((system->n % DEF_BLOCK_SIZE) == 0) ? 0 : 1);

    k_print_forces <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, workspace->d_workspace->f, system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


static void Print_HBonds( reax_system *system, int step )
{
    int blocks;
    
    blocks = (system->n) / DEF_BLOCK_SIZE + 
        (((system->n % DEF_BLOCK_SIZE) == 0) ? 0 : 1);

    k_print_hbonds <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, *(lists[HBONDS]), system->n, system->my_rank, step );
    cudaDeviceSynchronize( );
    cudaCheckError( );
}
#endif


/* Initialize indices for far neighbors list post reallocation
 *
 * system: atomic system info. */
void Cuda_Init_Neighbor_Indices( reax_system *system, reax_list **lists )
{
    int blocks;
    reax_list *far_nbr_list = lists[FAR_NBRS];

    /* init indices */
    Cuda_Scan_Excl_Sum( system->d_max_far_nbrs, far_nbr_list->index, system->total_cap );

    /* init end_indices */
    blocks = system->total_cap / DEF_BLOCK_SIZE + 
        ((system->total_cap % DEF_BLOCK_SIZE == 0) ? 0 : 1);
    k_init_end_index <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_far_nbrs, far_nbr_list->index, far_nbr_list->end_index, system->total_cap );
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


/* Initialize indices for far hydrogen bonds list post reallocation
 *
 * system: atomic system info. */
void Cuda_Init_HBond_Indices( reax_system *system, storage *workspace,
        reax_list **lists )
{
    int blocks, *temp;
    reax_list *hbond_list;

    hbond_list = lists[HBONDS];
    temp = (int *) workspace->scratch;

    /* init Hindices */
    blocks = system->N / DEF_BLOCK_SIZE + 
        ((system->N % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_setup_hindex <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->N );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    /* init indices and end_indices */
    Cuda_Scan_Excl_Sum( system->d_max_hbonds, temp, system->total_cap );

    k_init_hbond_indices <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, system->d_hbonds, temp, 
          hbond_list->index, hbond_list->end_index, system->N );
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


/* Initialize indices for far bonds list post reallocation
 *
 * system: atomic system info. */
void Cuda_Init_Bond_Indices( reax_system *system, reax_list **lists )
{
    int blocks;
    reax_list *bond_list;

    bond_list = lists[BONDS];

    /* init indices */
    Cuda_Scan_Excl_Sum( system->d_max_bonds, bond_list->index, system->total_cap );

    /* init end_indices */
    blocks = system->N / DEF_BLOCK_SIZE + 
        ((system->N % DEF_BLOCK_SIZE == 0) ? 0 : 1);
    k_init_end_index <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_bonds, bond_list->index, bond_list->end_index, system->N );
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


/* Initialize indices for charge matrix post reallocation
 *
 * system: atomic system info.
 * H: charge matrix */
void Cuda_Init_Sparse_Matrix_Indices( reax_system *system, sparse_matrix *H )
{
    int blocks;

    /* init indices */
    Cuda_Scan_Excl_Sum( system->d_max_cm_entries, H->start, system->total_cap );

    /* init end_indices */
    blocks = system->total_cap / DEF_BLOCK_SIZE
        + ((system->total_cap % DEF_BLOCK_SIZE == 0) ? 0 : 1);
    k_init_end_index <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_cm_entries, H->start, H->end, system->total_cap );
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


/* Initialize indices for three body list post reallocation
 *
 * indices: list indices
 * entries: num. of entries in list */
void Cuda_Init_Three_Body_Indices( int *indices, int entries, reax_list **lists )
{
    reax_list *thbody;

    thbody = lists[THREE_BODIES];

    Cuda_Scan_Excl_Sum( indices, thbody->index, entries );
}


void Cuda_Estimate_Storages( reax_system *system, control_params *control, 
        reax_list **lists, int realloc_bonds, int realloc_hbonds, int realloc_cm,
        int step )
{
    int blocks;

    blocks = system->total_cap / ST_BLOCK_SIZE + 
        (((system->total_cap % ST_BLOCK_SIZE == 0)) ? 0 : 1);

    k_estimate_storages <<< blocks, ST_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, system->reax_param.d_tbp, 
          (control_params *)control->d_control_params,
          *(lists[FAR_NBRS]), system->reax_param.num_atom_types,
          system->n, system->N, system->total_cap,
          system->d_cm_entries, system->d_max_cm_entries,
          system->d_bonds, system->d_max_bonds,
          system->d_hbonds, system->d_max_hbonds );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    if ( realloc_bonds == TRUE )
    {
        Cuda_Reduction_Sum( system->d_max_bonds, system->d_total_bonds,
                system->total_cap );
        copy_host_device( &system->total_bonds, system->d_total_bonds, sizeof(int), 
                cudaMemcpyDeviceToHost, "Cuda_Estimate_Storages::d_total_bonds" );
    }

    if ( system->numH > 0 && control->hbond_cut > 0.0 )
    {
        if ( realloc_hbonds == TRUE )
        {
            Cuda_Reduction_Sum( system->d_max_hbonds, system->d_total_hbonds,
                    system->total_cap );
            copy_host_device( &system->total_hbonds, system->d_total_hbonds, sizeof(int), 
                    cudaMemcpyDeviceToHost, "Cuda_Estimate_Storages::d_total_hbonds" );
        }
    }
    else
    {
        if ( step == 0 )
        {
#if defined(DEBUG_FOCUS)
            if ( system->numH == 0 )
            {
                fprintf( stderr, "[INFO] DISABLING HYDROGEN BOND COMPUTATION: NO HYDROGEN ATOMS FOUND\n" );
            }
#endif

#if defined(DEBUG_FOCUS)
            if ( control->hbond_cut <= 0.0 )
            {
                fprintf( stderr, "[INFO] DISABLING HYDROGEN BOND COMPUTATION: BOND CUTOFF LENGTH IS ZERO\n" );
            }
#endif

            control->hbond_cut = 0.0;
            k_disable_hydrogen_bonding <<< 1, 1 >>> ( (control_params *)control->d_control_params );
        }
    }

    if ( realloc_cm == TRUE )
    {
        Cuda_Reduction_Sum( system->d_max_cm_entries, system->d_total_cm_entries, system->total_cap );
        copy_host_device( &system->total_cm_entries, system->d_total_cm_entries, sizeof(int),
                cudaMemcpyDeviceToHost, "Cuda_Estimate_Storages::d_total_cm_entries" );
    }
}


int Cuda_Init_Forces( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control ) 
{
    int ret, ret_bonds, ret_hbonds, ret_cm;
    int blocks, hblocks;

    /* init the workspace (bond_mark) */
//    cuda_memset( workspace->d_workspace->bond_mark, 0, sizeof(int) * system->n, "bond_mark" );
//
//    blocks = (system->N - system->n) / DEF_BLOCK_SIZE + 
//       (((system->N - system->n) % DEF_BLOCK_SIZE == 0) ? 0 : 1);
//    k_init_bond_mark <<< blocks, DEF_BLOCK_SIZE >>>
//       ( system->n, (system->N - system->n), workspace->d_workspace->bond_mark );
//    cudaDeviceSynchronize( );
//    cudaCheckError( );

    blocks = (system->N) / DEF_BLOCK_SIZE + 
        (((system->N % DEF_BLOCK_SIZE) == 0) ? 0 : 1);

//    k_print_hbond_info <<< blocks, DEF_BLOCK_SIZE >>>
//        ( system->d_my_atoms, system->reax_param.d_sbp,
//          (control_params *)control->d_control_params,
//          *(lists[HBONDS]), system->N );
//    cudaDeviceSynchronize( );
//    cudaCheckError( );

    /* reset reallocation flags on device */
    cuda_memset( system->d_realloc_bonds, FALSE, sizeof(int), 
            "Cuda_Init_Forces::d_realloc_bonds" );
    cuda_memset( system->d_realloc_hbonds, FALSE, sizeof(int), 
            "Cuda_Init_Forces::d_realloc_hbonds" );
    cuda_memset( system->d_realloc_cm_entries, FALSE, sizeof(int), 
            "Cuda_Init_Forces::d_realloc_cm_entries" );

    k_init_forces <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->reax_param.d_sbp,
          system->reax_param.d_tbp, *(workspace->d_workspace), (control_params *)control->d_control_params,
          *(lists[FAR_NBRS]), *(lists[BONDS]), *(lists[HBONDS]),
          workspace->d_LR, system->n, system->N, system->reax_param.num_atom_types,
          (((data->step-data->prev_steps) % control->reneighbor) == 0),
          system->d_max_cm_entries, system->d_realloc_cm_entries,
          system->d_max_bonds, system->d_realloc_bonds,
          system->d_max_hbonds, system->d_realloc_hbonds );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    /* check reallocation flags on device */
    copy_host_device( &ret_bonds, system->d_realloc_bonds, sizeof(int), 
            cudaMemcpyDeviceToHost, "Cuda_Init_Forces::d_realloc_bonds" );
    copy_host_device( &ret_hbonds, system->d_realloc_hbonds, sizeof(int), 
            cudaMemcpyDeviceToHost, "Cuda_Init_Forces::d_realloc_hbonds" );
    copy_host_device( &ret_cm, system->d_realloc_cm_entries, sizeof(int), 
            cudaMemcpyDeviceToHost, "Cuda_Init_Forces::d_realloc_cm_entries" );

    ret = (ret_bonds == FALSE && ret_hbonds == FALSE && ret_cm == FALSE) ? SUCCESS : FAILURE;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] p%d, step %d: ret = %d, ret_bonds = %d, ret_hbonds = %d, ret_cm = %d\n",
            system->my_rank, data->step, ret, ret_bonds, ret_hbonds, ret_cm );
#endif

    if ( ret == SUCCESS )
    {
        k_update_sym_dbond_indices <<< blocks, control->block_size >>> 
            ( *(lists[BONDS]), system->N );
        cudaDeviceSynchronize( );
        cudaCheckError( );

        if ( control->hbond_cut > 0.0 && system->numH > 0 )
        {
            /* make hbond_list symmetric */
            hblocks = (system->N * HB_KER_SYM_THREADS_PER_ATOM / HB_SYM_BLOCK_SIZE) + 
                ((((system->N * HB_KER_SYM_THREADS_PER_ATOM) % HB_SYM_BLOCK_SIZE) == 0) ? 0 : 1);

            k_update_sym_hbond_indices <<< hblocks, HB_BLOCK_SIZE >>>
                ( system->d_my_atoms, *(lists[HBONDS]), system->N );
            cudaDeviceSynchronize( );
            cudaCheckError( );
        }

        /* update bond_mark */
//        k_bond_mark <<< blocks, DEF_BLOCK_SIZE >>>
//        k_bond_mark <<< 1, 1 >>>
//            ( *(lists[BONDS]), *(workspace->d_workspace), system->N );
//        cudaDeviceSynchronize( );
//        cudaCheckError( );
    }
    else
    {
        Cuda_Estimate_Storages( system, control, lists,
               ret_bonds, ret_hbonds, ret_cm, data->step );

        workspace->d_workspace->realloc.bonds = ret_bonds;
        workspace->d_workspace->realloc.hbonds = ret_hbonds;
        workspace->d_workspace->realloc.cm = ret_cm;
    }

    return ret;
}


int Cuda_Init_Forces_No_Charges( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control ) 
{
    //TODO: implement later when figure out bond_mark usage
    return FAILURE;
}


int Cuda_Compute_Bonded_Forces( reax_system *system, control_params *control, 
        simulation_data *data, storage *workspace, 
        reax_list **lists, output_controls *out_control )
{
    int update_energy, ret;
//    int hbs, hnbrs_blocks;
    int *thbody;
    static int compute_bonded_part1 = FALSE;
    real *spad;
    rvec *rvec_spad;
#if defined(DEBUG_FOCUS)
    real t_start, t_elapsed;
#endif

    spad = (real *) workspace->scratch;
    update_energy = (out_control->energy_update_freq > 0
            && data->step % out_control->energy_update_freq == 0) ? TRUE : FALSE;
    ret = SUCCESS;

    if ( compute_bonded_part1 == FALSE )
    {
        /* 1. Bond Order Interactions */
#if defined(DEBUG_FOCUS)
        t_start = Get_Time( );

        fprintf( stderr, " Begin Bonded Forces ... %d x %d\n",
                control->blocks_n, control->block_size );
#endif

        Cuda_BO_Part1 <<< control->blocks_n, control->block_size >>>
            ( system->d_my_atoms, system->reax_param.d_sbp, 
              *(workspace->d_workspace), system->N );
        cudaDeviceSynchronize( );
        cudaCheckError( );

        Cuda_BO_Part2 <<< control->blocks_n, control->block_size >>>
            ( system->d_my_atoms, system->reax_param.d_gp, system->reax_param.d_sbp, 
              system->reax_param.d_tbp, *(workspace->d_workspace), 
              *(lists[BONDS]),
              system->reax_param.num_atom_types, system->N );
        cudaDeviceSynchronize( );
        cudaCheckError( );

        Cuda_BO_Part3 <<< control->blocks_n, control->block_size >>>
            ( *(workspace->d_workspace), *(lists[BONDS]), system->N );
        cudaDeviceSynchronize( );
        cudaCheckError( );

        Cuda_BO_Part4 <<< control->blocks_n, control->block_size >>>
            ( system->d_my_atoms, system->reax_param.d_gp, system->reax_param.d_sbp, 
             *(workspace->d_workspace), system->N );
        cudaDeviceSynchronize( );
        cudaCheckError( );

#if defined(DEBUG_FOCUS)
        t_elapsed = Get_Timing_Info( t_start );

        fprintf( stderr, "Bond Orders... return value --> %d --- Timing %lf \n",
                cudaGetLastError( ), t_elapsed );
        fprintf( stderr, "Cuda_Calculate_Bond_Orders Done... \n" );
#endif

        /* 2. Bond Energy Interactions */
#if defined(DEBUG_FOCUS)
        t_start = Get_Time( );
#endif

        cuda_memset( spad, 0, sizeof(real) * 2 * system->N,
                "Compute_Bonded_Forces::spad" );

        Cuda_Bonds <<< control->blocks, control->block_size, sizeof(real) * control->block_size >>>
            ( system->d_my_atoms, system->reax_param.d_gp, system->reax_param.d_sbp, system->reax_param.d_tbp,
              *(workspace->d_workspace), *(lists[BONDS]), 
              system->n, system->reax_param.num_atom_types, spad );
        cudaDeviceSynchronize( );
        cudaCheckError( );

        /* reduction for E_BE */
        if ( update_energy == TRUE )
        {
            Cuda_Reduction_Sum( spad, &((simulation_data *)data->d_simulation_data)->my_en.e_bond,
                    system->n );
        }

#if defined(DEBUG_FOCUS)
        t_elapsed = Get_Timing_Info( t_start );

        fprintf( stderr, "Cuda_Bond_Energy ... return value --> %d --- Timing %lf \n",
                cudaGetLastError( ), t_elapsed );
        fprintf( stderr, "Cuda_Bond_Energy Done... \n" );
#endif

        /* 3. Atom Energy Interactions */
#if defined(DEBUG_FOCUS)
        t_start = Get_Time( );
#endif

        cuda_memset( spad, 0, sizeof(real) * 3 * system->n,
                "Compute_Bonded_Forces::spad" );

        Cuda_Atom_Energy_Part1 <<< control->blocks, control->block_size >>>
            ( system->d_my_atoms, system->reax_param.d_gp,
              system->reax_param.d_sbp, system->reax_param.d_tbp, *(workspace->d_workspace),
              *(lists[BONDS]), system->n, system->reax_param.num_atom_types,
              spad, &spad[system->n], &spad[2 * system->n] );
        cudaDeviceSynchronize( );
        cudaCheckError( );

        Cuda_Atom_Energy_Part2 <<< control->blocks, control->block_size >>>
            ( *(lists[BONDS]), *(workspace->d_workspace), system->n );
        cudaDeviceSynchronize( );
        cudaCheckError( );

        /* reduction for E_Lp */
        if ( update_energy == TRUE )
        {
            Cuda_Reduction_Sum( spad, &((simulation_data *)data->d_simulation_data)->my_en.e_lp,
                    system->n );
        }

        /* reduction for E_Ov */
        if ( update_energy == TRUE )
        {
            Cuda_Reduction_Sum( &spad[system->n],
                    &((simulation_data *)data->d_simulation_data)->my_en.e_ov,
                    system->n );
        }

        /* reduction for E_Un */
        if ( update_energy == TRUE )
        {
            Cuda_Reduction_Sum( &spad[2 * system->n],
                    &((simulation_data *)data->d_simulation_data)->my_en.e_un,
                    system->n );
        }

#if defined(DEBUG_FOCUS)
        t_elapsed = Get_Timing_Info( t_start );

        fprintf( stderr, "test_LonePair_postprocess ... return value --> %d --- Timing %lf \n",
                cudaGetLastError( ), t_elapsed );
        fprintf( stderr, "test_LonePair_postprocess Done... \n");
#endif

        compute_bonded_part1 = TRUE;
    }

    /* 4. Valence Angles Interactions */
#if defined(DEBUG_FOCUS)
    t_start = Get_Time( );
#endif

    thbody = (int *) workspace->scratch;
    ret = Cuda_Estimate_Storage_Three_Body( system, control, workspace,
            data->step, lists, thbody );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "system->total_thbodies = %d, lists:THREE_BODIES->max_intrs = %d,\n",
            system->total_thbodies, lists[THREE_BODIES]->max_intrs );
    fprintf( stderr, "lists:THREE_BODIES->n = %d, lists:BONDS->max_intrs = %d,\n",
            lists[THREE_BODIES]->n, lists[BONDS]->max_intrs );
    fprintf( stderr, "system->total_thbodies = %d\n", system->total_thbodies );
#endif

    if ( ret == SUCCESS )
    {
        Cuda_Init_Three_Body_Indices( thbody, system->total_thbodies_indices, lists );

        cuda_memset( spad, 0, 6 * sizeof(real) * system->N + sizeof(rvec) * system->N * 2,
                "Cuda_Compute_Bonded_Forces::spad" );

        Cuda_Valence_Angles_Part1 <<< control->blocks_n, control->block_size >>>
            ( system->d_my_atoms, system->reax_param.d_gp, 
              system->reax_param.d_sbp, system->reax_param.d_thbp, 
              (control_params *)control->d_control_params,
              *(workspace->d_workspace), *(lists[BONDS]), *(lists[THREE_BODIES]),
              system->n, system->N, system->reax_param.num_atom_types, 
              spad, &spad[2 * system->N], &spad[4 * system->N], (rvec *)(&spad[6 * system->N]) );
        cudaDeviceSynchronize( );
        cudaCheckError( );

        /* reduction for E_Ang */
        if ( update_energy == TRUE )
        {
            Cuda_Reduction_Sum( spad, &((simulation_data *)data->d_simulation_data)->my_en.e_ang,
                    system->N );
        }

        /* reduction for E_Pen */
        if ( update_energy == TRUE )
        {
            Cuda_Reduction_Sum( &spad[2 * system->N],
                    &((simulation_data *)data->d_simulation_data)->my_en.e_pen,
                    system->N );
        }

        /* reduction for E_Coa */
        if ( update_energy == TRUE )
        {
            Cuda_Reduction_Sum( &spad[4 * system->N],
                    &((simulation_data *)data->d_simulation_data)->my_en.e_coa,
                    system->N );
        }

        /* reduction for ext_pres */
        rvec_spad = (rvec *) (&spad[6 * system->N]);
        k_reduction_rvec <<< control->blocks_n, control->block_size, sizeof(rvec) * control->block_size >>>
            ( rvec_spad, rvec_spad + system->N,  system->N );
        cudaDeviceSynchronize( );
        cudaCheckError( );

        k_reduction_rvec <<< 1, control->blocks_pow_2_n, sizeof(rvec) * control->blocks_pow_2_n >>>
            ( rvec_spad + system->N, &((simulation_data *)data->d_simulation_data)->my_ext_press, control->blocks_n );
        cudaDeviceSynchronize ();
        cudaCheckError( );
//        Cuda_Reduction_Sum( rvec_spad,
//                &((simulation_data *)data->d_simulation_data)->my_ext_press,
//                system->N );

        Cuda_Valence_Angles_Part2 <<< control->blocks_n, control->block_size >>>
            ( system->d_my_atoms, (control_params *)control->d_control_params,
              *(workspace->d_workspace), *(lists[BONDS]), system->N );
        cudaDeviceSynchronize( );
        cudaCheckError( );

#if defined(DEBUG_FOCUS)
        t_elapsed = Get_Timing_Info( t_start );

        fprintf( stderr, "Three_Body_Interactions ...  Timing %lf \n",
                t_elapsed );
        fprintf( stderr, "Three_Body_Interactions Done... \n" );
#endif

        /* 5. Torsion Angles Interactions */
#if defined(DEBUG_FOCUS)
        t_start = Get_Time( );
#endif

        cuda_memset( spad, 0, 4 * sizeof(real) * system->n + sizeof(rvec) * system->n * 2,
                "Cuda_Compute_Bonded_Forces::spad" );

        Cuda_Torsion_Angles_Part1 <<< control->blocks, control->block_size >>>
            ( system->d_my_atoms, system->reax_param.d_gp, system->reax_param.d_fbp,
              (control_params *) control->d_control_params, *(lists[BONDS]),
              *(lists[THREE_BODIES]), *(workspace->d_workspace), system->n,
              system->reax_param.num_atom_types, 
              spad, &spad[2 * system->n], (rvec *) (&spad[4 * system->n]) );
        cudaDeviceSynchronize( );
        cudaCheckError( );

        /* reduction for E_Tor */
        if ( update_energy == TRUE )
        {
            Cuda_Reduction_Sum( spad, &((simulation_data *)data->d_simulation_data)->my_en.e_tor,
                    system->n );
        }

        /* reduction for E_Con */
        if ( update_energy == TRUE )
        {
            Cuda_Reduction_Sum( &spad[2 * system->n],
                    &((simulation_data *)data->d_simulation_data)->my_en.e_con,
                    system->n );
        }

        /* reduction for ext_pres */
        rvec_spad = (rvec *) (&spad[4 * system->n]);
        k_reduction_rvec <<< control->blocks, control->block_size, sizeof(rvec) * control->block_size >>>
            ( rvec_spad, rvec_spad + system->n,  system->n );
        cudaDeviceSynchronize( );
        cudaCheckError( );

        k_reduction_rvec <<< 1, control->blocks_pow_2, sizeof(rvec) * control->blocks_pow_2 >>>
                ( rvec_spad + system->n,
                  &((simulation_data *)data->d_simulation_data)->my_ext_press, control->blocks );
        cudaDeviceSynchronize( );
        cudaCheckError( );
//        Cuda_Reduction_Sum( rvec_spad,
//                &((simulation_data *)data->d_simulation_data)->my_ext_press,
//                system->n );

        Cuda_Torsion_Angles_Part2 <<< control->blocks_n, control->block_size >>>
                ( system->d_my_atoms, *(workspace->d_workspace), *(lists[BONDS]),
                  system->N );
        cudaDeviceSynchronize( );
        cudaCheckError( );

#if defined(DEBUG_FOCUS)
        t_elapsed = Get_Timing_Info( t_start );

        fprintf( stderr, "Four_Body_post process return value --> %d --- Four body Timing %lf \n",
                cudaGetLastError( ), t_elapsed );
        fprintf( stderr, " Four_Body_ Done... \n");
#endif

        /* 6. Hydrogen Bonds Interactions */
        if ( control->hbond_cut > 0.0 && system->numH > 0 )
        {
#if defined(DEBUG_FOCUS)
            t_start = Get_Time( );
#endif

            cuda_memset( spad, 0,
                    2 * sizeof(real) * system->n + sizeof(rvec) * system->n * 2,
                    "Cuda_Compute_Bonded_Forces::spad" );

//            hbs = (system->n * HB_KER_THREADS_PER_ATOM / HB_BLOCK_SIZE) + 
//                (((system->n * HB_KER_THREADS_PER_ATOM) % HB_BLOCK_SIZE) == 0 ? 0 : 1);

            Cuda_Hydrogen_Bonds <<< control->blocks, control->block_size >>>
//            Cuda_Hydrogen_Bonds_MT <<< hbs, HB_BLOCK_SIZE, 
//                    HB_BLOCK_SIZE * (2 * sizeof(real) + 2 * sizeof(rvec)) >>>
                    ( system->d_my_atoms, system->reax_param.d_sbp,
                      system->reax_param.d_hbp, system->reax_param.d_gp,
                      (control_params *) control->d_control_params,
                      *(workspace->d_workspace),
                      *(lists[FAR_NBRS]), *(lists[BONDS]), *(lists[HBONDS]),
                      system->n, system->reax_param.num_atom_types,
                      spad, (rvec *) (&spad[2 * system->n]), system->my_rank, data->step );
            cudaDeviceSynchronize( );
            cudaCheckError( );

//            if ( data->step == 10 )
//            {
//                Print_HBonds( system, data->step );
//            }

            /* reduction for E_HB */
            if ( update_energy == TRUE )
            {
                Cuda_Reduction_Sum( spad,
                        &((simulation_data *)data->d_simulation_data)->my_en.e_hb,
                        system->n );
            }

            /* reduction for ext_pres */
            rvec_spad = (rvec *) (&spad[2 * system->n]);
            k_reduction_rvec <<< control->blocks, control->block_size, sizeof(rvec) * control->block_size >>>
                (rvec_spad, rvec_spad + system->n,  system->n);
            cudaDeviceSynchronize( );
            cudaCheckError( );

            k_reduction_rvec <<< 1, control->blocks_pow_2, sizeof(rvec) * control->blocks_pow_2 >>>
                ( rvec_spad + system->n,
                  &((simulation_data *)data->d_simulation_data)->my_ext_press,
                  control->blocks );
            cudaDeviceSynchronize( );
            cudaCheckError( );
//            Cuda_Reduction_Sum( rvec_spad,
//                    &((simulation_data *)data->d_simulation_data)->my_ext_press,
//                    system->n );

            /* post process step1 */
            Cuda_Hydrogen_Bonds_PostProcess <<< control->blocks_n, control->block_size, control->block_size * sizeof(rvec) >>>
                ( system->d_my_atoms, *(workspace->d_workspace),
                  *(lists[BONDS]), system->N );
            cudaDeviceSynchronize( );
            cudaCheckError( );

            /* post process step2 */
//            hnbrs_blocks = (system->N * HB_POST_PROC_KER_THREADS_PER_ATOM / HB_POST_PROC_BLOCK_SIZE) +
//                (((system->N * HB_POST_PROC_KER_THREADS_PER_ATOM) % HB_POST_PROC_BLOCK_SIZE) == 0 ? 0 : 1);

            Cuda_Hydrogen_Bonds_HNbrs <<< system->N, 32, 32 * sizeof(rvec) >>>
                ( system->d_my_atoms, *(workspace->d_workspace), *(lists[HBONDS]) );
//            Cuda_Hydrogen_Bonds_HNbrs_BL <<< hnbrs_blocks, HB_POST_PROC_BLOCK_SIZE, 
//                    HB_POST_PROC_BLOCK_SIZE * sizeof(rvec) >>>
//                ( system->d_my_atoms, *(workspace->d_workspace), *(lists[HBONDS]), system->N );
            cudaDeviceSynchronize( );
            cudaCheckError( );

#if defined(DEBUG_FOCUS)
            t_elapsed = Get_Timing_Info( t_start );

            fprintf( stderr,
                    "Hydrogen bonds return value --> %d --- HydrogenBonds Timing %lf \n",
                    cudaGetLastError( ), t_elapsed );
            fprintf( stderr, "Hydrogen_Bond Done... \n" );
#endif
        }

        compute_bonded_part1 = FALSE;
    }

    return ret;
}


void Cuda_Compute_NonBonded_Forces( reax_system *system, control_params *control, 
        simulation_data *data, storage *workspace, 
        reax_list **lists, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    Cuda_NonBonded_Energy( system, control, workspace, data,
            lists, out_control, (control->tabulate == 0) ? false: true );
}


void Cuda_Compute_Total_Force( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, mpi_datatypes *mpi_data )
{
    rvec *f;

    f = (rvec *) workspace->host_scratch;
    memset( f, 0, sizeof(rvec) * system->N );

    Cuda_Total_Forces( system, control, data, workspace, lists );

#if defined(PURE_REAX)
    /* now all forces are computed to their partially-final values
     * based on the neighbors information each processor has had.
     * final values of force on each atom needs to be computed by adding up
     * all partially-final pieces */
    copy_host_device( f, workspace->d_workspace->f, sizeof(rvec) * system->N ,
            cudaMemcpyDeviceToHost, "Cuda_Compute_Total_Force::workspace->d_workspace->f" );

    Coll( system, mpi_data, f, RVEC_PTR_TYPE, mpi_data->mpi_rvec );

    copy_host_device( f, workspace->d_workspace->f, sizeof(rvec) * system->N,
            cudaMemcpyHostToDevice, "Cuda_Compute_Total_Force::workspace->d_workspace->f" );

    Cuda_Total_Forces_PURE( system, workspace );
#endif

}


int Cuda_Compute_Forces( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int charge_flag, retVal;
    static int init_forces_done = FALSE;

#if defined(LOG_PERFORMANCE)
    real t_start = 0;

    if ( system->my_rank == MASTER_NODE )
    {
        t_start = Get_Time( );
    }
#endif

    retVal = SUCCESS;

    if ( control->charge_freq && (data->step - data->prev_steps) % control->charge_freq == 0 )
    {
        charge_flag = TRUE;
    }
    else
    {
        charge_flag = FALSE;
    }

    if ( init_forces_done == FALSE )
    {
        if ( charge_flag == TRUE )
        {
            retVal = Cuda_Init_Forces( system, control, data, workspace, lists, out_control );
        }
        else
        {
            retVal = Cuda_Init_Forces_No_Charges( system, control, data, workspace, lists, out_control );
        }

        if ( retVal == SUCCESS )
        {
            init_forces_done = TRUE;
        }
    }

    if ( retVal == SUCCESS )
    {
#if defined(LOG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &data->timing.init_forces );
        }
#endif

        retVal = Cuda_Compute_Bonded_Forces( system, control, data,
                workspace, lists, out_control );

#if defined(LOG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &data->timing.bonded );
        }
#endif
    }

    if ( retVal == SUCCESS )
    {
#if defined(PURE_REAX)
        if ( charge_flag == TRUE )
        {
            Cuda_Compute_Charges( system, control, data, workspace, out_control, mpi_data );
        }

#if defined(LOG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &data->timing.cm );
        }
#endif

#endif //PURE_REAX

        Cuda_Compute_NonBonded_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );

#if defined(LOG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &data->timing.nonb );
        }
#endif

        Cuda_Compute_Total_Force( system, control, data, workspace, lists, mpi_data );

#if defined(LOG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &data->timing.bonded );
        }
#endif

        init_forces_done = FALSE;
    }

    return retVal;
}
