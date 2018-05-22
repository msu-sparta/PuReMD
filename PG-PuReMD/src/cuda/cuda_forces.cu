
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
#if defined(DEBUG)
  #include "cuda_validation.h"
#endif

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

    if ( sbp[ atoms[i].type ].p_hbond == H_ATOM || 
            sbp[ atoms[i].type ].p_hbond == H_BONDING_ATOM )
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


/* Initialize indices for far neighbors list post reallocation
 *
 * system: atomic system info. */
void Cuda_Init_Neighbor_Indices( reax_system *system )
{
    int blocks;
    reax_list *far_nbrs = dev_lists[FAR_NBRS];

    /* init indices */
    Cuda_Scan_Excl_Sum( system->d_max_far_nbrs, far_nbrs->index, system->total_cap );

    /* init end_indices */
    blocks = system->N / DEF_BLOCK_SIZE + 
        ((system->N % DEF_BLOCK_SIZE == 0) ? 0 : 1);
    k_init_end_index <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_far_nbrs, far_nbrs->index, far_nbrs->end_index, system->N );
    cudaThreadSynchronize( );
    cudaCheckError( );
}


/* Initialize indices for far hydrogen bonds list post reallocation
 *
 * system: atomic system info. */
void Cuda_Init_HBond_Indices( reax_system *system )
{
    int blocks;
    int *temp;
    reax_list *hbonds = dev_lists[HBONDS];

    temp = (int *) scratch;

    /* init Hindices */
    blocks = system->N / DEF_BLOCK_SIZE + 
        ((system->N % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_setup_hindex <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->N );
    cudaThreadSynchronize( );
    cudaCheckError( );

    /* init indices and end_indices */
    Cuda_Scan_Excl_Sum( system->d_max_hbonds, temp, system->total_cap );

    k_init_hbond_indices <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, system->d_hbonds, temp, 
          hbonds->index, hbonds->end_index, system->N );
    cudaThreadSynchronize( );
    cudaCheckError( );
}


/* Initialize indices for far bonds list post reallocation
 *
 * system: atomic system info. */
void Cuda_Init_Bond_Indices( reax_system *system )
{
    int blocks;
    reax_list *bonds = dev_lists[BONDS];

    /* init indices */
    Cuda_Scan_Excl_Sum( system->d_max_bonds, bonds->index, system->total_cap );

    /* init end_indices */
    blocks = system->N / DEF_BLOCK_SIZE + 
        ((system->N % DEF_BLOCK_SIZE == 0) ? 0 : 1);
    k_init_end_index <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_bonds, bonds->index, bonds->end_index, system->N );
    cudaThreadSynchronize( );
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
    Cuda_Scan_Excl_Sum( system->d_max_cm_entries, H->start, system->N );

    /* init end_indices */
    blocks = system->N / DEF_BLOCK_SIZE + 
        ((system->N % DEF_BLOCK_SIZE == 0) ? 0 : 1);
    k_init_end_index <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_cm_entries, H->start, H->end, system->N );
    cudaThreadSynchronize( );
    cudaCheckError( );
}


/* Initialize indices for three body list post reallocation
 *
 * indices: list indices
 * entries: num. of entries in list */
void Cuda_Init_Three_Body_Indices( int *indices, int entries )
{
    reax_list *thbody = dev_lists[THREE_BODIES];

    Cuda_Scan_Excl_Sum( indices, thbody->index, entries );
}


CUDA_GLOBAL void k_estimate_storages( reax_atom *my_atoms, 
        single_body_parameters *sbp, two_body_parameters *tbp,
        control_params *control, reax_list far_nbrs, 
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
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= total_cap )
    {
        return;
    }

    num_bonds = 0;
    num_hbonds = 0;
    num_cm_entries = 0;

    if ( i < N )
    {
        atom_i = &(my_atoms[i]);
        type_i = atom_i->type;
        start_i = Dev_Start_Index( i, &far_nbrs );
        end_i = Dev_End_Index( i, &far_nbrs );
        sbp_i = &(sbp[type_i]);

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
            nbr_pj = &( far_nbrs.far_nbr_list[pj] );
            j = nbr_pj->nbr;
            atom_j = &(my_atoms[j]);

            if ( nbr_pj->d <= control->nonb_cut )
            {
                type_j = my_atoms[j].type;
                sbp_j = &(sbp[type_j]);
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
                if ( control->hbond_cut > 0.0 && nbr_pj->d <= control->hbond_cut 
                        && ihb == H_BONDING_ATOM && jhb == H_ATOM && i >= n && j < n )
                {
                    ++num_hbonds;
                }

//                if ( i >= n )
//                {
//                    ihb = NON_H_BONDING_ATOM;
//                }
            }

            if ( nbr_pj->d <= cutoff )
            {
                type_j = my_atoms[j].type;
                r_ij = nbr_pj->d;
                sbp_j = &(sbp[type_j]);
                twbp = &(tbp[ index_tbp(type_i ,type_j, num_atom_types) ]);

                if ( local == TRUE )
                {
                    /* atom i: H atom OR H bonding atom, native */
                    if ( control->hbond_cut > 0.0 && (ihb == H_ATOM || ihb == H_BONDING_ATOM) &&
                            nbr_pj->d <= control->hbond_cut )
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
                if ( nbr_pj->d <= control->bond_cut )
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
          *(dev_lists[FAR_NBRS]), system->reax_param.num_atom_types,
          system->n, system->N, system->total_cap,
          system->d_cm_entries, system->d_max_cm_entries,
          system->d_bonds, system->d_max_bonds,
          system->d_hbonds, system->d_max_hbonds );
    cudaThreadSynchronize( );
    cudaCheckError( );

    if ( realloc_bonds == TRUE )
    {
        Cuda_Reduction_Sum( system->d_max_bonds, system->d_total_bonds,
                system->total_cap );
        copy_host_device( &(system->total_bonds), system->d_total_bonds, sizeof(int), 
                cudaMemcpyDeviceToHost, "Cuda_Estimate_Storages::d_total_bonds" );
    }

    if ( system->numH > 0 && control->hbond_cut > 0.0 )
    {
        if ( realloc_hbonds == TRUE )
        {
            Cuda_Reduction_Sum( system->d_max_hbonds, system->d_total_hbonds,
                    system->total_cap );
            copy_host_device( &(system->total_hbonds), system->d_total_hbonds, sizeof(int), 
                    cudaMemcpyDeviceToHost, "Cuda_Estimate_Storages::d_total_hbonds" );
        }
    }
    else
    {
        if ( step == 0 )
        {
#if defined(DEBUG)
            if ( system->numH == 0 )
            {
                fprintf( stderr, "[INFO] DISABLING HYDROGEN BOND COMPUTATION: NO HYDROGEN ATOMS FOUND\n" );
            }
#endif

#if defined(DEBUG)
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
        copy_host_device( &(system->total_cm_entries), system->d_total_cm_entries, sizeof(int),
                cudaMemcpyDeviceToHost, "Cuda_Estimate_Storages::d_total_cm_entries" );
    }

#if defined(DEBUG)
    fprintf( stderr, "p:%d -->\n", system->my_rank );
    fprintf( stderr, " TOTAL DEVICE BOND COUNT: %d \n", system->total_bonds );
    fprintf( stderr, " TOTAL DEVICE HBOND COUNT: %d \n", system->total_hbonds );
    fprintf( stderr, " TOTAL DEVICE SPARSE COUNT: %d \n", system->total_cm_entries );
#endif
}


int Cuda_Estimate_Storage_Three_Body( reax_system *system, control_params *control, 
        int step, reax_list **lists, int *thbody )
{
    int ret;

    ret = SUCCESS;

    cuda_memset( thbody, 0, system->total_bonds * sizeof(int), "scratch::thbody" );

    Estimate_Cuda_Valence_Angles <<< BLOCKS_N, BLOCK_SIZE >>>
        ( system->d_my_atoms, (control_params *)control->d_control_params, 
          *(dev_lists[BONDS]), system->n, system->N, thbody );
    cudaThreadSynchronize( );
    cudaCheckError( );

    Cuda_Reduction_Sum( thbody, system->d_total_thbodies, system->total_bonds );

    copy_host_device( &(system->total_thbodies), system->d_total_thbodies, sizeof(int),
            cudaMemcpyDeviceToHost, "Cuda_Estimate_Storage_Three_Body::d_total_thbodies" );

    if ( step == 0 )
    {
        system->total_thbodies = MAX( (int)(system->total_thbodies * SAFE_ZONE), MIN_3BODIES );
        system->total_thbodies_indices = system->total_bonds;

        /* create Three-body list */
        Dev_Make_List( system->total_thbodies_indices, system->total_thbodies,
                TYP_THREE_BODY, dev_lists[THREE_BODIES] );
    }

    if ( system->total_thbodies > dev_lists[THREE_BODIES]->max_intrs ||
            system->total_bonds > dev_lists[THREE_BODIES]->n )
    {
        if ( system->total_thbodies > dev_lists[THREE_BODIES]->max_intrs )
        {
            system->total_thbodies = MAX( (int)(dev_lists[THREE_BODIES]->max_intrs * SAFE_ZONE),
                    system->total_thbodies );
        }
        if ( system->total_bonds > dev_lists[THREE_BODIES]->n )
        {
            system->total_thbodies_indices = MAX( (int)(dev_lists[THREE_BODIES]->n * SAFE_ZONE),
                    system->total_bonds );
        }

        dev_workspace->realloc.thbody = TRUE;
        ret = FAILURE;
    }

    return ret;
}


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
    t = &( t_LR[ index_lr(tmin,tmax, num_atom_types) ] );    

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
    val *= EV_to_KCALpMOL / C_ele;

    return val;
}


//static void Init_Charge_Matrix_Remaining_Entries( reax_system *system,
//        control_params *control, reax_list *far_nbrs,
//        sparse_matrix * H, sparse_matrix * H_sp,
//        int * Htop, int * H_sp_top )
//{
//    int i, j, pj;
//    real d, xcut, bond_softness, * X_diag;
//
//    switch ( control->charge_method )
//    {
//        case QEQ_CM:
//            break;
//
//        case EE_CM:
//            H->start[system->N_cm - 1] = *Htop;
//            H_sp->start[system->N_cm - 1] = *H_sp_top;
//
//            for ( i = 0; i < system->N_cm - 1; ++i )
//            {
//                H->j[*Htop] = i;
//                H->val[*Htop] = 1.0;
//                *Htop = *Htop + 1;
//
//                H_sp->j[*H_sp_top] = i;
//                H_sp->val[*H_sp_top] = 1.0;
//                *H_sp_top = *H_sp_top + 1;
//            }
//
//            H->j[*Htop] = system->N_cm - 1;
//            H->val[*Htop] = 0.0;
//            *Htop = *Htop + 1;
//
//            H_sp->j[*H_sp_top] = system->N_cm - 1;
//            H_sp->val[*H_sp_top] = 0.0;
//            *H_sp_top = *H_sp_top + 1;
//            break;
//
//        case ACKS2_CM:
//            X_diag = (real*) scalloc( system->N, sizeof(real),
//                    "Init_Charge_Matrix_Remaining_Entries::X_diag" );
//
//            H->start[system->N] = *Htop;
//            H_sp->start[system->N] = *H_sp_top;
//
//            for ( i = 0; i < system->N; ++i )
//            {
//                H->j[*Htop] = i;
//                H->val[*Htop] = 1.0;
//                *Htop = *Htop + 1;
//
//                H_sp->j[*H_sp_top] = i;
//                H_sp->val[*H_sp_top] = 1.0;
//                *H_sp_top = *H_sp_top + 1;
//            }
//
//            H->j[*Htop] = system->N;
//            H->val[*Htop] = 0.0;
//            *Htop = *Htop + 1;
//
//            H_sp->j[*H_sp_top] = system->N;
//            H_sp->val[*H_sp_top] = 0.0;
//            *H_sp_top = *H_sp_top + 1;
//
//            for ( i = 0; i < system->N; ++i )
//            {
//                H->start[system->N + i + 1] = *Htop;
//                H_sp->start[system->N + i + 1] = *H_sp_top;
//
//                H->j[*Htop] = i;
//                H->val[*Htop] = 1.0;
//                *Htop = *Htop + 1;
//
//                H_sp->j[*H_sp_top] = i;
//                H_sp->val[*H_sp_top] = 1.0;
//                *H_sp_top = *H_sp_top + 1;
//
//                for ( pj = Start_Index(i, far_nbrs); pj < End_Index(i, far_nbrs); ++pj )
//                {
//                    if ( far_nbrs->far_nbr_list[pj].d <= control->r_cut )
//                    {
//                        j = far_nbrs->far_nbr_list[pj].nbr;
//
//                        xcut = ( system->reaxprm.sbp[ system->atoms[i].type ].b_s_acks2
//                                + system->reaxprm.sbp[ system->atoms[j].type ].b_s_acks2 )
//                            / 2.0;
//
//                        if ( far_nbrs->far_nbr_list[pj].d < xcut &&
//                                far_nbrs->far_nbr_list[pj].d > 0.001 )
//                        {
//                            d = far_nbrs->far_nbr_list[pj].d / xcut;
//                            bond_softness = system->reaxprm.gp.l[34] * POW( d, 3.0 ) * POW( 1.0 - d, 6.0 );
//
//                            H->j[*Htop] = system->N + j + 1;
//                            H->val[*Htop] = MAX( 0.0, bond_softness );
//                            *Htop = *Htop + 1;
//
//                            H_sp->j[*H_sp_top] = system->N + j + 1;
//                            H_sp->val[*H_sp_top] = MAX( 0.0, bond_softness );
//                            *H_sp_top = *H_sp_top + 1;
//
//                            X_diag[i] -= bond_softness;
//                            X_diag[j] -= bond_softness;
//                        }
//                    }
//                }
//
//                H->j[*Htop] = system->N + i + 1;
//                H->val[*Htop] = 0.0;
//                *Htop = *Htop + 1;
//
//                H_sp->j[*H_sp_top] = system->N + i + 1;
//                H_sp->val[*H_sp_top] = 0.0;
//                *H_sp_top = *H_sp_top + 1;
//            }
//
//            H->start[system->N_cm - 1] = *Htop;
//            H_sp->start[system->N_cm - 1] = *H_sp_top;
//
//            for ( i = system->N + 1; i < system->N_cm - 1; ++i )
//            {
//                for ( pj = H->start[i]; pj < H->start[i + 1]; ++pj )
//                {
//                    if ( H->j[pj] == i )
//                    {
//                        H->val[pj] = X_diag[i - system->N - 1];
//                        break;
//                    }
//                }
//
//                for ( pj = H_sp->start[i]; pj < H_sp->start[i + 1]; ++pj )
//                {
//                    if ( H_sp->j[pj] == i )
//                    {
//                        H_sp->val[pj] = X_diag[i - system->N - 1];
//                        break;
//                    }
//                }
//            }
//
//            for ( i = system->N + 1; i < system->N_cm - 1; ++i )
//            {
//                H->j[*Htop] = i;
//                H->val[*Htop] = 1.0;
//                *Htop = *Htop + 1;
//
//                H_sp->j[*H_sp_top] = i;
//                H_sp->val[*H_sp_top] = 1.0;
//                *H_sp_top = *H_sp_top + 1;
//            }
//
//            H->j[*Htop] = system->N_cm - 1;
//            H->val[*Htop] = 0.0;
//            *Htop = *Htop + 1;
//
//            H_sp->j[*H_sp_top] = system->N_cm - 1;
//            H_sp->val[*H_sp_top] = 0.0;
//            *H_sp_top = *H_sp_top + 1;
//
//            sfree( X_diag, "Init_Charge_Matrix_Remaining_Entries::X_diag" );
//            break;
//
//        default:
//            break;
//    }
//}


CUDA_GLOBAL void k_print_hbond_info( reax_atom *my_atoms, single_body_parameters *sbp, 
        control_params *control, reax_list hbonds, int N )
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

    atom_i = &(my_atoms[i]);
    type_i = atom_i->type;
    sbp_i = &(sbp[type_i]);

    if ( control->hbond_cut > 0.0 )
    {
        ihb = sbp_i->p_hbond;
        if ( ihb == H_ATOM  || ihb == H_BONDING_ATOM )
        {
            ihb_top = Dev_Start_Index( atom_i->Hindex, &hbonds );
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
        reax_list far_nbrs_list, reax_list bonds_list, reax_list hbonds_list, 
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
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;
    sparse_matrix *H;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    H = &(workspace.H);
    Htop = H->start[i];

    atom_i = &(my_atoms[i]);
    type_i = atom_i->type;
    start_i = Dev_Start_Index( i, &far_nbrs_list );
    end_i = Dev_End_Index( i, &far_nbrs_list );
    btop_i = Dev_Start_Index( i, &bonds_list );
    sbp_i = &(sbp[type_i]);

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
                i, H->entries[Htop].j, r_ij, twbp->gamma, DIAGONAL );
        ++Htop;
    }

    if ( control->hbond_cut > 0.0 )
    {
        ihb = sbp_i->p_hbond;

        if ( ihb == H_ATOM || ihb == H_BONDING_ATOM )
        {
            ihb_top = Dev_Start_Index( atom_i->Hindex, &hbonds_list );
        }
        else
        {
            ihb_top = -1;
        }
    }

    /* update i-j distance - check if j is within cutoff */
    for ( pj = start_i; pj < end_i; ++pj )
    {
        nbr_pj = &( far_nbrs_list.far_nbr_list[pj] );
        j = nbr_pj->nbr;
        atom_j = &(my_atoms[j]);

        if ( renbr )
        {
            if ( nbr_pj->d <= cutoff )
            {
                flag = TRUE;
            }
            else
            {
                flag = FALSE;
            }

            if ( nbr_pj->d <= control->nonb_cut )
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
                nbr_pj->dvec[0] = atom_j->x[0] - atom_i->x[0];
                nbr_pj->dvec[1] = atom_j->x[1] - atom_i->x[1];
                nbr_pj->dvec[2] = atom_j->x[2] - atom_i->x[2];
                nbr_pj->d = rvec_Norm_Sqr( nbr_pj->dvec );
            }
            else
            {
                nbr_pj->dvec[0] = atom_i->x[0] - atom_j->x[0];
                nbr_pj->dvec[1] = atom_i->x[1] - atom_j->x[1];
                nbr_pj->dvec[2] = atom_i->x[2] - atom_j->x[2];
                nbr_pj->d = rvec_Norm_Sqr( nbr_pj->dvec );
            }

            if ( nbr_pj->d <= SQR( control->nonb_cut ) )
            {
                flag2 = TRUE;
            }
            else
            {
                flag2 = FALSE;
            }

            if ( nbr_pj->d <= SQR( control->nonb_cut ) )
            {
                nbr_pj->d = SQRT( nbr_pj->d );
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
            sbp_j = &(sbp[type_j]);
            ihb = sbp_i->p_hbond;
            jhb = sbp_j->p_hbond;

            /* atom i: H bonding, ghost
             * atom j: H atom, native */
            if ( control->hbond_cut > 0.0 && nbr_pj->d <= control->hbond_cut
                    && ihb == H_BONDING_ATOM && jhb == H_ATOM && i >= n && j < n ) 
            {
                hbonds_list.hbond_list[ihb_top].nbr = j;
                hbonds_list.hbond_list[ihb_top].scl = -1;
                hbonds_list.hbond_list[ihb_top].ptr = nbr_pj;

                //CUDA SPECIFIC
                hbonds_list.hbond_list[ihb_top].sym_index = -1;
                rvec_MakeZero( hbonds_list.hbond_list[ihb_top].hb_f );

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
                twbp = &(tbp[ index_tbp(type_i,type_j,num_atom_types) ]);
                r_ij = nbr_pj->d;

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
            r_ij = nbr_pj->d;
            sbp_j = &(sbp[type_j]);
            twbp = &(tbp[ index_tbp(type_i, type_j, num_atom_types) ]);

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
                //bool condition = !((i >= n) && (j >= n));

                /* hydrogen bond lists */
                if ( control->hbond_cut > 0.0 && (ihb == H_ATOM || ihb == H_BONDING_ATOM) &&
                        nbr_pj->d <= control->hbond_cut )
                {
                    jhb = sbp_j->p_hbond;

                    /* atom i: H atom, native
                     * atom j: H bonding atom */
                    if ( ihb == H_ATOM && jhb == H_BONDING_ATOM )
                    {
                        hbonds_list.hbond_list[ihb_top].nbr = j;

                        if ( i < j )
                        {
                            hbonds_list.hbond_list[ihb_top].scl = 1;
                        }
                        else
                        {
                            hbonds_list.hbond_list[ihb_top].scl = -1;
                        }
                        hbonds_list.hbond_list[ihb_top].ptr = nbr_pj;

                        //CUDA SPECIFIC
                        hbonds_list.hbond_list[ihb_top].sym_index = -1;
                        rvec_MakeZero( hbonds_list.hbond_list[ihb_top].hb_f );

                        ++ihb_top;
                    }
                    /* atom i: H bonding atom, native
                     * atom j: H atom, native */
                    else if ( ihb == H_BONDING_ATOM && jhb == H_ATOM && j < n )
                    {
                        //jhb_top = End_Index( atom_j->Hindex, hbonds_list );
                        hbonds_list.hbond_list[ihb_top].nbr = j;
                        hbonds_list.hbond_list[ihb_top].scl = -1;
                        hbonds_list.hbond_list[ihb_top].ptr = nbr_pj;

                        //CUDA SPECIFIC
                        hbonds_list.hbond_list[ihb_top].sym_index = -1;
                        rvec_MakeZero( hbonds_list.hbond_list[ihb_top].hb_f );

                        ++ihb_top;
                    }
                }
            }

            /* uncorrected bond orders */
            if ( nbr_pj->d <= control->bond_cut &&
                    Dev_BOp( bonds_list, control->bo_cut, i, btop_i, nbr_pj,
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

    Dev_Set_End_Index( i, btop_i, &bonds_list );
    H->end[i] = Htop;
//    if( local == TRUE )
//    {
        if ( control->hbond_cut > 0.0 && ihb_top > 0 && (ihb == H_ATOM || ihb == H_BONDING_ATOM) )
        {
            Dev_Set_End_Index( atom_i->Hindex, ihb_top, &hbonds_list );
        }
//    }

    num_bonds = btop_i - Dev_Start_Index( i, &bonds_list );
    num_hbonds = ihb_top - Dev_Start_Index( atom_i->Hindex, &hbonds_list );
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


CUDA_GLOBAL void k_init_bond_mark( int offset, int n, int *bond_mark )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }

    bond_mark[offset + threadIdx.x] = 1000;
}


CUDA_GLOBAL void New_fix_sym_dbond_indices( reax_list pbonds, int N )
{
    int i, j, k, nbr;
    bond_data *ibond, *jbond;
    int atom_j;
    reax_list *bonds;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    bonds = &pbonds;

    for ( j = Dev_Start_Index(i, bonds); j < Dev_End_Index(i, bonds); j++ )
    {
        ibond = &( bonds->bond_list[j] );
        nbr = ibond->nbr;

        for ( k = Dev_Start_Index(nbr, bonds); k < Dev_End_Index(nbr, bonds); k++ )
        {
            jbond = &( bonds->bond_list[k] );
            atom_j = jbond->nbr;

            if ( atom_j == i )
            {
                if ( i > nbr )
                {
                    ibond->dbond_index = j;
                    jbond->dbond_index = j;

                    ibond->sym_index = k;
                    jbond->sym_index = j;
                }
            }
        }
    }
}


CUDA_GLOBAL void New_fix_sym_hbond_indices( reax_atom *my_atoms, reax_list hbonds, int N )
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
    start = Dev_Start_Index( my_atoms[i].Hindex, &hbonds );
    end = Dev_End_Index( my_atoms[i].Hindex, &hbonds );
    j = start + lane_id;

    while ( j < end )
    {
        ihbond = &( hbonds.hbond_list[j] );
        nbr = ihbond->nbr;

        nbrstart = Dev_Start_Index( my_atoms[nbr].Hindex, &hbonds );
        nbrend = Dev_End_Index( my_atoms[nbr].Hindex, &hbonds );

        for ( k = nbrstart; k < nbrend; k++ )
        {
            jhbond = &( hbonds.hbond_list[k] );

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


CUDA_GLOBAL void k_update_bonds( reax_atom *my_atoms, reax_list bonds, int n )
{
    int i;
    
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    my_atoms[i].num_bonds = Dev_Num_Entries( i, &bonds );
}


CUDA_GLOBAL void k_update_hbonds( reax_atom *my_atoms, reax_list hbonds, int n )
{
    int Hindex;
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    Hindex = my_atoms[i].Hindex;
    my_atoms[i].num_hbonds = Dev_Num_Entries( Hindex, &hbonds );
}


#if defined(DEBUG)
CUDA_GLOBAL void k_print_forces( reax_atom *my_atoms, rvec *f, int n )
{
    int i; 

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
    {
        return;
    }

    printf( "%8d: %24.15f, %24.15f, %24.15f\n",
            my_atoms[i].orig_id,
            f[i][0],
            f[i][1],
            f[i][2] );
}


static void Print_Forces( reax_system *system )
{
    int blocks;
    
    blocks = (system->n) / DEF_BLOCK_SIZE + 
        (((system->n % DEF_BLOCK_SIZE) == 0) ? 0 : 1);

    k_print_forces <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, dev_workspace->f, system->n );
    cudaThreadSynchronize( );
    cudaCheckError( );
}


CUDA_GLOBAL void k_print_hbonds( reax_atom *my_atoms, reax_list p_hbonds, int n, int rank, int step )
{
    int i, k, pj, start, end; 
    reax_list *hbonds;
    hbond_data *hbond_list, *hbond_jk;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    hbonds = &( p_hbonds );
    start = Dev_Start_Index( my_atoms[i].Hindex, hbonds );
    end = Dev_End_Index( my_atoms[i].Hindex, hbonds );
    hbond_list = hbonds->hbond_list;

    for ( pj = start; pj < end; ++pj )
    {
        k = hbond_list[pj].nbr;
        hbond_jk = &( hbond_list[pj] );

        printf( "p%03d, step %05d: %8d: %8d, %24.15f, %24.15f, %24.15f\n",
                rank, step, my_atoms[i].Hindex, k,
                hbond_jk->hb_f[0],
                hbond_jk->hb_f[1],
                hbond_jk->hb_f[2] );
    }
}


static void Print_HBonds( reax_system *system, int step )
{
    int blocks;
    
    blocks = (system->n) / DEF_BLOCK_SIZE + 
        (((system->n % DEF_BLOCK_SIZE) == 0) ? 0 : 1);

    k_print_hbonds <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, *(dev_lists[HBONDS], system->n, system->my_rank, step );
    cudaThreadSynchronize( );
    cudaCheckError( );
}
#endif


CUDA_GLOBAL void k_init_bond_orders( reax_atom *my_atoms, reax_list far_nbrs, 
        reax_list bonds, real *total_bond_order, int N )
{
    int i, pj; 
//    int j; 
    int start_i, end_i;
//    far_neighbor_data *nbr_pj;
//    reax_atom *atom_i, *atom_j;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

//    atom_i = &(my_atoms[i]);
    start_i = Dev_Start_Index(i, &far_nbrs);
    end_i = Dev_End_Index(i, &far_nbrs);

    for( pj = start_i; pj < end_i; ++pj )
    { 
//        nbr_pj = &( far_nbrs.far_nbr_list[pj] );
//        j = nbr_pj->nbr;
//        atom_j = &(my_atoms[j]);
//
//        total_bond_order[i]++;
//        atom_i->Hindex++;
    }
}


CUDA_GLOBAL void k_bond_mark( reax_list p_bonds, storage p_workspace, int N )
{
    int i, j, k;
    reax_list *bonds = &( p_bonds );
    storage *workspace = &( p_workspace );

//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    if ( i >= N )
//    {
//        return;
//    }

    for ( i = 0; i < N; i++ )
    {
        for (k = Dev_Start_Index(i, bonds); k < Dev_End_Index(i, bonds); k++)
        {
            bond_data *bdata = &( bonds->bond_list[k] );
            j = bdata->nbr;

            if (i < j )
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


int Cuda_Init_Forces( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control ) 
{
    int ret, ret_bonds, ret_hbonds, ret_cm;
    int blocks, hblocks;

    /* init the workspace (bond_mark) */
//    cuda_memset( dev_workspace->bond_mark, 0, sizeof(int) * system->n, "bond_mark" );
//
//    blocks = (system->N - system->n) / DEF_BLOCK_SIZE + 
//       (((system->N - system->n) % DEF_BLOCK_SIZE == 0) ? 0 : 1);
//    k_init_bond_mark <<< blocks, DEF_BLOCK_SIZE >>>
//       ( system->n, (system->N - system->n), dev_workspace->bond_mark );
//    cudaThreadSynchronize( );
//    cudaCheckError( );

    blocks = (system->N) / DEF_BLOCK_SIZE + 
        (((system->N % DEF_BLOCK_SIZE) == 0) ? 0 : 1);

//    k_init_bond_orders <<< blocks, DEF_BLOCK_SIZE >>>
//        ( system->d_my_atoms, *(dev_lists[FAR_NBRS]), *(dev_lists[BONDS]),
//          dev_workspace->total_bond_order, system->N );
//    cudaThreadSynchronize( );
//    cudaCheckError( );
//
//    k_print_hbond_info <<< blocks, DEF_BLOCK_SIZE >>>
//        ( system->d_my_atoms, system->reax_param.d_sbp,
//          (control_params *)control->d_control_params,
//          *(dev_lists[HBONDS]), system->N );
//    cudaThreadSynchronize( );
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
          system->reax_param.d_tbp, *dev_workspace,
          (control_params *)control->d_control_params,
          *(dev_lists[FAR_NBRS]), *(dev_lists[BONDS]),
          *(dev_lists[HBONDS]), control->d_LR, system->n,
          system->N, system->reax_param.num_atom_types,
          (((data->step-data->prev_steps) % control->reneighbor) == 0),
          system->d_max_cm_entries, system->d_realloc_cm_entries,
          system->d_max_bonds, system->d_realloc_bonds,
          system->d_max_hbonds, system->d_realloc_hbonds );
    cudaThreadSynchronize( );
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
        /* fix sym_index and dbond_index */
        New_fix_sym_dbond_indices <<< blocks, BLOCK_SIZE >>> 
            ( *(dev_lists[BONDS]), system->N );
        cudaThreadSynchronize( );
        cudaCheckError( );

        if ( control->hbond_cut > 0 && system->numH > 0 )
        {
            /* make hbond_list symmetric */
            hblocks = (system->N * HB_KER_SYM_THREADS_PER_ATOM / HB_SYM_BLOCK_SIZE) + 
                ((((system->N * HB_KER_SYM_THREADS_PER_ATOM) % HB_SYM_BLOCK_SIZE) == 0) ? 0 : 1);

            New_fix_sym_hbond_indices <<< hblocks, HB_BLOCK_SIZE >>>
                ( system->d_my_atoms, *(dev_lists[HBONDS]), system->N );
            cudaThreadSynchronize( );
            cudaCheckError( );
        }

        /* update bond_mark */
//        k_bond_mark <<< blocks, DEF_BLOCK_SIZE >>>
//        k_bond_mark <<< 1, 1 >>>
//            ( *(dev_lists[BONDS]), *dev_workspace, system->N );
//        cudaThreadSynchronize( );
//        cudaCheckError( );
    }
    else
    {
        Cuda_Estimate_Storages( system, control, dev_lists,
               ret_bonds, ret_hbonds, ret_cm, data->step );

        dev_workspace->realloc.bonds = ret_bonds;
        dev_workspace->realloc.hbonds = ret_hbonds;
        dev_workspace->realloc.cm = ret_cm;
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
    real *spad = (real *) scratch;
    rvec *rvec_spad;
#if defined(DEBUG)
    real t_start, t_elapsed;
#endif

    update_energy = (out_control->energy_update_freq > 0
            && data->step % out_control->energy_update_freq == 0) ? TRUE : FALSE;
    ret = SUCCESS;

    if ( compute_bonded_part1 == FALSE )
    {
        /* 1. Bond Order Interactions */
#if defined(DEBUG)
        t_start = Get_Time( );

        fprintf( stderr, " Begin Bonded Forces ... %d x %d\n",
                BLOCKS_N, BLOCK_SIZE );
#endif

        Cuda_Calculate_BO_init <<< BLOCKS_N, BLOCK_SIZE >>>
            ( system->d_my_atoms, system->reax_param.d_sbp, 
              *dev_workspace, system->N );
        cudaThreadSynchronize( );
        cudaCheckError( );

        Cuda_Calculate_BO <<< BLOCKS_N, BLOCK_SIZE >>>
            ( system->d_my_atoms, system->reax_param.d_gp, system->reax_param.d_sbp, 
              system->reax_param.d_tbp, *dev_workspace, 
              *(dev_lists[BONDS]),
              system->reax_param.num_atom_types, system->N );
        cudaThreadSynchronize( );
        cudaCheckError( );

        Cuda_Update_Uncorrected_BO <<< BLOCKS_N, BLOCK_SIZE >>>
            ( *dev_workspace, *(dev_lists[BONDS]), system->N );
        cudaThreadSynchronize( );
        cudaCheckError( );

        Cuda_Update_Workspace_After_BO <<< BLOCKS_N, BLOCK_SIZE >>>
            ( system->d_my_atoms, system->reax_param.d_gp, system->reax_param.d_sbp, 
             *dev_workspace, system->N );
        cudaThreadSynchronize( );
        cudaCheckError( );

#if defined(DEBUG)
        t_elapsed = Get_Timing_Info( t_start );

        fprintf( stderr, "Bond Orders... return value --> %d --- Timing %lf \n",
                cudaGetLastError( ), t_elapsed );
        fprintf( stderr, "Cuda_Calculate_Bond_Orders Done... \n" );
#endif

        /* 2. Bond Energy Interactions */
#if defined(DEBUG)
        t_start = Get_Time( );
#endif

        cuda_memset( spad, 0, system->N * (2 * sizeof(real)) , "scratch" );

        Cuda_Bonds <<< BLOCKS, BLOCK_SIZE, sizeof(real)* BLOCK_SIZE >>>
            ( system->d_my_atoms, system->reax_param.d_gp, system->reax_param.d_sbp, system->reax_param.d_tbp,
              *dev_workspace, *(dev_lists[BONDS]), 
              system->n, system->reax_param.num_atom_types, spad );
        cudaThreadSynchronize( );
        cudaCheckError( );

        /* reduction for E_BE */
        if ( update_energy == TRUE )
        {
            Cuda_Reduction_Sum( spad, &((simulation_data *)data->d_simulation_data)->my_en.e_bond,
                    system->n );
        }

#if defined(DEBUG)
        t_elapsed = Get_Timing_Info( t_start );

        fprintf( stderr, "Cuda_Bond_Energy ... return value --> %d --- Timing %lf \n",
                cudaGetLastError( ), t_elapsed );
        fprintf( stderr, "Cuda_Bond_Energy Done... \n" );
#endif

        /* 3. Atom Energy Interactions */
#if defined(DEBUG)
        t_start = Get_Time( );
#endif

        cuda_memset( spad, 0, ( 6 * sizeof(real) * system->n ), "scratch" );

        Cuda_Atom_Energy <<< BLOCKS, BLOCK_SIZE >>>
            ( system->d_my_atoms, system->reax_param.d_gp,
              system->reax_param.d_sbp, system->reax_param.d_tbp, *dev_workspace,
              *(dev_lists[BONDS]), system->n, system->reax_param.num_atom_types,
              spad, spad + 2 * system->n, spad + 4 * system->n);
        cudaThreadSynchronize( );
        cudaCheckError( );

//        Cuda_Atom_Energy_PostProcess <<< BLOCKS, BLOCK_SIZE >>>
//            ( *(dev_lists[BONDS]), *dev_workspace, system->n );
        Cuda_Atom_Energy_PostProcess <<< BLOCKS_N, BLOCK_SIZE >>>
            ( *(dev_lists[BONDS]), *dev_workspace, system->N );
        cudaThreadSynchronize( );
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
            Cuda_Reduction_Sum( spad + 2 * system->n,
                    &((simulation_data *)data->d_simulation_data)->my_en.e_ov,
                    system->n );
        }

        /* reduction for E_Un */
        if ( update_energy == TRUE )
        {
            Cuda_Reduction_Sum( spad + 4 * system->n,
                    &((simulation_data *)data->d_simulation_data)->my_en.e_ov,
                    system->n );
        }

#if defined(DEBUG)
        t_elapsed = Get_Timing_Info( t_start );

        fprintf( stderr, "test_LonePair_postprocess ... return value --> %d --- Timing %lf \n",
                cudaGetLastError( ), t_elapsed );
        fprintf( stderr, "test_LonePair_postprocess Done... \n");
#endif

        compute_bonded_part1 = TRUE;
    }

    /* 4. Valence Angles Interactions */
#if defined(DEBUG)
    t_start = Get_Time( );
#endif

    thbody = (int *) scratch;
    ret = Cuda_Estimate_Storage_Three_Body( system, control, data->step,
            dev_lists, thbody );

#if defined(DEBUG)
    fprintf( stderr, "system->total_thbodies = %d, lists:THREE_BODIES->max_intrs = %d,\n",
            system->total_thbodies, lists[THREE_BODIES]->max_intrs );
    fprintf( stderr, "lists:THREE_BODIES->n = %d, lists:BONDS->max_intrs = %d,\n",
            lists[THREE_BODIES]->n, lists[BONDS]->max_intrs );
    fprintf( stderr, "system->total_thbodies = %d\n", system->total_thbodies );
#endif

    if ( ret == SUCCESS )
    {
        Cuda_Init_Three_Body_Indices( thbody, system->total_thbodies_indices );

        cuda_memset( spad, 0, 6 * sizeof(real) * system->N + sizeof(rvec) * system->N * 2, "scratch" );

        Cuda_Valence_Angles <<< BLOCKS_N, BLOCK_SIZE >>>
            ( system->d_my_atoms, system->reax_param.d_gp, 
              system->reax_param.d_sbp, system->reax_param.d_thbp, 
              (control_params *)control->d_control_params,
              *dev_workspace, *(dev_lists[BONDS]), *(dev_lists[THREE_BODIES]),
              system->n, system->N, system->reax_param.num_atom_types, 
              spad, spad + 2 * system->N, spad + 4 * system->N, (rvec *)(spad + 6 * system->N) );
        cudaThreadSynchronize( );
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
            Cuda_Reduction_Sum( spad + 2 * system->N,
                    &((simulation_data *)data->d_simulation_data)->my_en.e_pen,
                    system->N );
        }

        /* reduction for E_Coa */
        if ( update_energy == TRUE )
        {
            Cuda_Reduction_Sum( spad + 4 * system->N,
                    &((simulation_data *)data->d_simulation_data)->my_en.e_coa,
                    system->N );
        }

        /* reduction for ext_pres */
        rvec_spad = (rvec *) (spad + 6 * system->N);
        k_reduction_rvec <<< BLOCKS_N, BLOCK_SIZE, sizeof(rvec) * BLOCK_SIZE >>>
            ( rvec_spad, rvec_spad + system->N,  system->N );
        cudaThreadSynchronize( );
        cudaCheckError( );

        k_reduction_rvec <<< 1, BLOCKS_POW_2_N, sizeof(rvec) * BLOCKS_POW_2_N >>>
            ( rvec_spad + system->N, &((simulation_data *)data->d_simulation_data)->my_ext_press, BLOCKS_N );
        cudaThreadSynchronize ();
        cudaCheckError( );
//        Cuda_Reduction_Sum( rvec_spad,
//                &((simulation_data *)data->d_simulation_data)->my_ext_press,
//                system->N );

        Cuda_Valence_Angles_PostProcess <<< BLOCKS_N, BLOCK_SIZE >>>
            ( system->d_my_atoms, (control_params *)control->d_control_params,
              *dev_workspace, *(dev_lists[BONDS]), system->N );
        cudaThreadSynchronize( );
        cudaCheckError( );

#if defined(DEBUG)
        t_elapsed = Get_Timing_Info( t_start );

        fprintf( stderr, "Three_Body_Interactions ...  Timing %lf \n",
                t_elapsed );
        fprintf( stderr, "Three_Body_Interactions Done... \n" );
#endif

        /* 5. Torsion Angles Interactions */
#if defined(DEBUG)
        t_start = Get_Time( );
#endif

        cuda_memset( spad, 0, 4 * sizeof(real) * system->n + sizeof(rvec) * system->n * 2,
                "scratch" );

        Cuda_Torsion_Angles <<< BLOCKS, BLOCK_SIZE >>>
            ( system->d_my_atoms, system->reax_param.d_gp, system->reax_param.d_fbp,
              (control_params *) control->d_control_params, *(dev_lists[BONDS]),
              *(dev_lists[THREE_BODIES]), *dev_workspace, system->n,
              system->reax_param.num_atom_types, 
              spad, spad + 2 * system->n, (rvec *) (spad + 4 * system->n) );
        cudaThreadSynchronize( );
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
            Cuda_Reduction_Sum( spad + 2 * system->n,
                    &((simulation_data *)data->d_simulation_data)->my_en.e_con,
                    system->n );
        }

        /* reduction for ext_pres */
        rvec_spad = (rvec *) (spad + 4 * system->n);
        k_reduction_rvec <<< BLOCKS, BLOCK_SIZE, sizeof(rvec) * BLOCK_SIZE >>>
            ( rvec_spad, rvec_spad + system->n,  system->n );
        cudaThreadSynchronize( );
        cudaCheckError( );

        k_reduction_rvec <<< 1, BLOCKS_POW_2, sizeof(rvec) * BLOCKS_POW_2 >>>
                ( rvec_spad + system->n,
                  &((simulation_data *)data->d_simulation_data)->my_ext_press, BLOCKS );
        cudaThreadSynchronize( );
        cudaCheckError( );
//        Cuda_Reduction_Sum( rvec_spad,
//                &((simulation_data *)data->d_simulation_data)->my_ext_press,
//                system->n );

        Cuda_Torsion_Angles_PostProcess <<< BLOCKS_N, BLOCK_SIZE >>>
                ( system->d_my_atoms, *dev_workspace, *(dev_lists[BONDS]),
                  system->N );
        cudaThreadSynchronize( );
        cudaCheckError( );

#if defined(DEBUG)
        t_elapsed = Get_Timing_Info( t_start );

        fprintf( stderr, "Four_Body_post process return value --> %d --- Four body Timing %lf \n",
                cudaGetLastError( ), t_elapsed );
        fprintf( stderr, " Four_Body_ Done... \n");
#endif

        /* 6. Hydrogen Bonds Interactions */
        if ( control->hbond_cut > 0.0 && system->numH > 0 )
        {
#if defined(DEBUG)
            t_start = Get_Time( );
#endif

            cuda_memset( spad, 0,
                    2 * sizeof(real) * system->n + sizeof(rvec) * system->n * 2, "scratch" );

//            hbs = (system->n * HB_KER_THREADS_PER_ATOM / HB_BLOCK_SIZE) + 
//                (((system->n * HB_KER_THREADS_PER_ATOM) % HB_BLOCK_SIZE) == 0 ? 0 : 1);

            Cuda_Hydrogen_Bonds <<< BLOCKS, BLOCK_SIZE >>>
//            Cuda_Hydrogen_Bonds_MT <<< hbs, HB_BLOCK_SIZE, 
//                    HB_BLOCK_SIZE * (2 * sizeof(real) + 2 * sizeof(rvec)) >>>
                    ( system->d_my_atoms, system->reax_param.d_sbp,
                      system->reax_param.d_hbp, system->reax_param.d_gp,
                      (control_params *) control->d_control_params,
                      *dev_workspace, *(dev_lists[BONDS]), *(dev_lists[HBONDS]),
                      system->n, system->reax_param.num_atom_types,
                      spad, (rvec *) (spad + 2 * system->n), system->my_rank, data->step );
            cudaThreadSynchronize( );
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
            rvec_spad = (rvec *) (spad + 2 * system->n);
            k_reduction_rvec <<< BLOCKS, BLOCK_SIZE, sizeof(rvec) * BLOCK_SIZE >>>
                (rvec_spad, rvec_spad + system->n,  system->n);
            cudaThreadSynchronize( );
            cudaCheckError( );

            k_reduction_rvec <<< 1, BLOCKS_POW_2, sizeof(rvec) * BLOCKS_POW_2 >>>
                (rvec_spad + system->n, &((simulation_data *)data->d_simulation_data)->my_ext_press, BLOCKS);
            cudaThreadSynchronize( );
            cudaCheckError( );
//            Cuda_Reduction_Sum( rvec_spad,
//                    &((simulation_data *)data->d_simulation_data)->my_ext_press,
//                    system->n );

            /* post process step1 */
            Cuda_Hydrogen_Bonds_PostProcess <<< BLOCKS_N, BLOCK_SIZE, BLOCK_SIZE * sizeof(rvec) >>>
                ( system->d_my_atoms, *dev_workspace,
                  *(dev_lists[BONDS]), system->N );
            cudaThreadSynchronize( );
            cudaCheckError( );

            /* post process step2 */
//            hnbrs_blocks = (system->N * HB_POST_PROC_KER_THREADS_PER_ATOM / HB_POST_PROC_BLOCK_SIZE) +
//                (((system->N * HB_POST_PROC_KER_THREADS_PER_ATOM) % HB_POST_PROC_BLOCK_SIZE) == 0 ? 0 : 1);

            Cuda_Hydrogen_Bonds_HNbrs <<< system->N, 32, 32 * sizeof(rvec) >>>
                ( system->d_my_atoms, *dev_workspace, *(dev_lists[HBONDS]) );
//            Cuda_Hydrogen_Bonds_HNbrs_BL <<< hnbrs_blocks, HB_POST_PROC_BLOCK_SIZE, 
//                    HB_POST_PROC_BLOCK_SIZE * sizeof(rvec) >>>
//                ( system->d_my_atoms, *dev_workspace, *(dev_lists[HBONDS]), system->N );
            cudaThreadSynchronize( );
            cudaCheckError( );

#if defined(DEBUG)
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
    /* van der Waals and Coulomb interactions */
    Cuda_NonBonded_Energy( system, control, workspace, data,
            lists, out_control, (control->tabulate == 0) ? false: true );
}


void Cuda_Compute_Total_Force( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, mpi_datatypes *mpi_data )
{
    rvec *f;

    f = (rvec *) host_scratch;
    memset( f, 0, sizeof(rvec) * system->N );

    Cuda_Total_Forces( system, control, data, workspace );

#if defined(PURE_REAX)
    /* now all forces are computed to their partially-final values
     * based on the neighbors information each processor has had.
     * final values of force on each atom needs to be computed by adding up
     * all partially-final pieces */

    //MVAPICH2
    copy_host_device( f, dev_workspace->f, sizeof(rvec) * system->N ,
            cudaMemcpyDeviceToHost, "total_force:f:get" );

    Coll( system, mpi_data, f, RVEC_PTR_TYPE, mpi_data->mpi_rvec );

    copy_host_device( f, dev_workspace->f, sizeof(rvec) * system->N,
            cudaMemcpyHostToDevice, "total_force:f:put" );

    Cuda_Total_Forces_PURE( system, dev_workspace );
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

    //MPI_Barrier( MPI_COMM_WORLD );
    if ( system->my_rank == MASTER_NODE )
    {
        t_start = Get_Time( );
    }
#endif

    retVal = SUCCESS;

    /********* init forces ************/
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
        //validate_sparse_matrix( system, workspace );

#if defined(LOG_PERFORMANCE)
        //MPI_Barrier( MPI_COMM_WORLD );
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &(data->timing.init_forces) );
        }
#endif

        /********* bonded interactions ************/
        retVal = Cuda_Compute_Bonded_Forces( system, control, data, workspace, lists, out_control );

#if defined(LOG_PERFORMANCE)
        //MPI_Barrier( MPI_COMM_WORLD );
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &(data->timing.bonded) );
        }
#endif

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d @ step%d: completed bonded\n",
                 system->my_rank, data->step );
        MPI_Barrier( MPI_COMM_WORLD );
#endif
    }

    if ( retVal == SUCCESS )
    {
    /**************** charges ************************/
#if defined(PURE_REAX)
        if ( charge_flag == TRUE )
        {
            Cuda_Compute_Charges( system, control, data, workspace, out_control, mpi_data );
        }

#if defined(LOG_PERFORMANCE)
        //MPI_Barrier( MPI_COMM_WORLD );
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &(data->timing.cm) );
        }
#endif

#if defined(DEBUG_FOCUS)
        fprintf(stderr, "p%d @ step%d: qeq completed\n", system->my_rank, data->step);
        MPI_Barrier( MPI_COMM_WORLD );
#endif

#endif //PURE_REAX

        /********* nonbonded interactions ************/
        Cuda_Compute_NonBonded_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );

#if defined(LOG_PERFORMANCE)
        //MPI_Barrier( MPI_COMM_WORLD );
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &(data->timing.nonb) );
        }
#endif

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d @ step%d: nonbonded forces completed\n",
                system->my_rank, data->step );
        MPI_Barrier( MPI_COMM_WORLD );
#endif

        /*********** total force ***************/
        Cuda_Compute_Total_Force( system, control, data, workspace, lists, mpi_data );

#if defined(LOG_PERFORMANCE)
        //MPI_Barrier( MPI_COMM_WORLD );
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &(data->timing.bonded) );
        }
#endif

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d @ step%d: total forces computed\n",
                system->my_rank, data->step );
//        Print_Total_Force( system, data, workspace );
        MPI_Barrier( MPI_COMM_WORLD );

#endif

        init_forces_done = FALSE;
    }

    return retVal;
}
