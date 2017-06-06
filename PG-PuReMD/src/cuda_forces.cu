
#include "cuda_forces.h"

#include "reax_types.h"
#include "reax_types.h"
#include "dev_list.h"
#include "cuda_utils.h"
#include "cuda_helpers.h"
#include "index_utils.h"
#include "vector.h"

#include "cuda_neighbors.h"

#include "forces.h"
#include "cuda_bond_orders.h"
#include "reduction.h"
#include "cuda_bonds.h"
#include "cuda_multi_body.h"
#include "cuda_valence_angles.h"
#include "cuda_torsion_angles.h"
#include "cuda_hydrogen_bonds.h"
#include "tool_box.h"
#include "cuda_nonbonded.h"


extern "C" void Make_List( int, int, int, reax_list* );
extern "C" void Delete_List( reax_list* );


CUDA_GLOBAL void k_disable_hydrogen_bonding( control_params *control )
{
    control->hbond_cut = 0.0;
}


CUDA_GLOBAL void k_estimate_storages( reax_atom *my_atoms, 
        single_body_parameters *sbp, two_body_parameters *tbp,
        control_params *control, reax_list far_nbrs, 
        int num_atom_types, int n, int N, int Hcap, int total_cap,
        int *Htop, int *bond_top, int *hb_top )
{
    int i, j, pj; 
    int start_i, end_i;
    int type_i, type_j;
    int ihb, jhb;
    int local;
    real cutoff;
    real r_ij, r2; 
    real C12, C34, C56;
    real BO, BO_s, BO_pi, BO_pi2;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
    {
        return;
    }

    atom_i = &(my_atoms[i]);
    type_i  = atom_i->type;
    start_i = Dev_Start_Index(i, &far_nbrs);
    end_i = Dev_End_Index(i, &far_nbrs);
    sbp_i = &(sbp[type_i]);

    if ( i < n )
    { 
        local = 1;
        cutoff = control->nonb_cut;
        atomicAdd( Htop, 1 );
        ihb = sbp_i->p_hbond;
    }   
    else
    {
        local = 0;
        cutoff = control->bond_cut;
        ihb = -1; 
    } 

    for ( pj = start_i; pj < end_i; ++pj )
    { 
        nbr_pj = &( far_nbrs.select.far_nbr_list[pj] );
        j = nbr_pj->nbr;
        atom_j = &(my_atoms[j]);

        if ( nbr_pj->d <= control->nonb_cut )
        {
            type_j = my_atoms[j].type;
            sbp_j = &(sbp[type_j]);
            ihb = sbp_i->p_hbond;
            jhb = sbp_j->p_hbond;
            if ( (control->hbond_cut > 0.1) 
                    && (nbr_pj->d <= control->hbond_cut) 
                    && (ihb == 2) && (jhb == 1) && (j < n) && (i > n) )
            {
                atomicAdd( &hb_top[i], 1 );
            }

            if ( i >= n )
            {
                ihb = -1;
            }
        }

        if ( nbr_pj->d <= cutoff )
        {
            type_j = my_atoms[j].type;
            r_ij = nbr_pj->d;
            sbp_j = &(sbp[type_j]);
            twbp = &(tbp[index_tbp (type_i,type_j,num_atom_types)]);

            if ( local )
            {
                if ( j < n || atom_i->orig_id < atom_j->orig_id ) //tryQEq ||1
                {
                    atomicAdd( Htop, 1 );
                }
                else if ( j < n || atom_i->orig_id > atom_j->orig_id ) //tryQEq ||1
                {
                    atomicAdd( Htop, 1 );
                }

                if ( control->hbond_cut > 0.1 && (ihb==1 || ihb==2) &&
                        nbr_pj->d <= control->hbond_cut )
                {
                    jhb = sbp_j->p_hbond;
                    if( ihb == 1 && jhb == 2 )
                    {
                        atomicAdd( &hb_top[i], 1 );
                    }
                    else if( ihb == 2 && jhb == 1 && j < n )
                    {
                        atomicAdd( &hb_top[i], 1 );
                    }
                }
            }

            // uncorrected bond orders 
            if ( nbr_pj->d <= control->bond_cut )
            {
                r2 = SQR( r_ij );

                if ( sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0 )
                {
                    C12 = twbp->p_bo1 * POW( r_ij / twbp->r_s, twbp->p_bo2 );
                    BO_s = (1.0 + control->bo_cut) * EXP( C12 );
                }
                else
                {
                    BO_s = C12 = 0.0;
                }

                if ( sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0 )
                {
                    C34 = twbp->p_bo3 * POW( r_ij / twbp->r_p, twbp->p_bo4 );
                    BO_pi = EXP( C34 );
                }
                else
                {
                    BO_pi = C34 = 0.0;
                }

                if ( sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0 )
                {
                    C56 = twbp->p_bo5 * POW( r_ij / twbp->r_pp, twbp->p_bo6 );
                    BO_pi2= EXP( C56 );
                }
                else
                {
                    BO_pi2 = C56 = 0.0;
                }

                // Initially BO values are the uncorrected ones, page 1 
                BO = BO_s + BO_pi + BO_pi2;

                if ( BO >= control->bo_cut )
                {
                    atomicAdd( &bond_top[i], 1 );
                    //atomicAdd( &bond_top[j], 1 );
                }
            }
        }
    }
}


CUDA_GLOBAL void k_init_system_atoms( reax_atom *my_atoms, int N, 
        int *hb_top, int *bond_top )
{
    int i;
    
    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
    {
        return;
    }

    my_atoms[i].num_bonds = bond_top[i];
    my_atoms[i].num_hbonds = hb_top[i];
}


void Cuda_Estimate_Storages( reax_system *system, control_params *control, 
        reax_list **lists, int *Htop, int *hb_top, int *bond_top )
{
    int i;
    int blocks = 0;
    int *d_Htop, *d_hb_top, *d_bond_top;
    int * tmp = (int*) scratch;
    int bond_count = 0;
    int hbond_count = 0;
    int max_hbonds = 0, min_hbonds = 999999;

    *Htop = 0;
    memset( hb_top, 0, sizeof(int) * system->total_cap );
    memset( bond_top, 0, sizeof(int) * system->total_cap );
	
    cuda_memset( tmp, 0, sizeof(int) *
            (1 + system->total_cap + system->total_cap), "Cuda_Estimate_Storages" );
 
    d_Htop = tmp; 
    d_hb_top = d_Htop + 1;
    d_bond_top = d_hb_top + system->total_cap;
   
    blocks = (int) CEIL((real)system->N / ST_BLOCK_SIZE);

    k_estimate_storages <<< blocks, ST_BLOCK_SIZE>>>
        (system->d_my_atoms, system->reax_param.d_sbp, system->reax_param.d_tbp, 
         (control_params *)control->d_control_params,
         *(*dev_lists + FAR_NBRS), system->reax_param.num_atom_types,
         system->n, system->N, system->Hcap, system->total_cap, 
         d_Htop, d_bond_top, d_hb_top );
    cudaThreadSynchronize( );
    cudaCheckError( );

    copy_host_device( Htop, d_Htop, sizeof(int),
            cudaMemcpyDeviceToHost, "Htop");
    copy_host_device( hb_top, d_hb_top, sizeof(int) * system->total_cap,
            cudaMemcpyDeviceToHost, "hb_top");
    copy_host_device( bond_top, d_bond_top, sizeof(int) * system->total_cap,
            cudaMemcpyDeviceToHost, "bond_top");

    for ( i = 0; i < system->total_cap; i++ )
    {
        system->max_bonds[i] = MAX( (int)CEIL( bond_top[i] * SAFE_ZONE ), MIN_BONDS );
        bond_count += system->max_bonds[i];
    }

    //TODO: change
    for ( i = 0; i < system->N; i++ )
    {
        if ( hb_top[i] >= max_hbonds )
        {
            max_hbonds = hb_top[i];
        }
        if ( hb_top[i] <= min_hbonds )
        {
            min_hbonds = hb_top[i];
        }

        hbond_count += hb_top[i];
    }
    system->max_hbonds = max_hbonds * SAFER_ZONE;

#if defined(DEBUG)
    fprintf( stderr, "p:%d -->\n", system->my_rank );
    fprintf( stderr, " TOTAL DEVICE BOND COUNT: %d \n", bond_count );
    fprintf( stderr, " TOTAL DEVICE HBOND COUNT: %d \n", hbond_count );
    fprintf( stderr, " TOTAL DEVICE SPARSE COUNT: %d \n", *Htop );
#endif

    /* if number of hydrogen atoms is 0, disable hydrogen bond functionality */
    if ( hbond_count == 0 )
    {
        control->hbond_cut = 0.0;
        k_disable_hydrogen_bonding <<<1,1>>> ( (control_params *)control->d_control_params );
    }

    k_init_system_atoms <<<blocks, ST_BLOCK_SIZE>>>
        (system->d_my_atoms, system->N, d_hb_top, d_bond_top );

    cudaThreadSynchronize( );
    cudaCheckError( );
}


void Cuda_Estimate_Storages_Three_Body( reax_system *system, control_params *control, 
        reax_list **lists, int *num_3body, int *thbody )
{
    int i;
    real *spad = (real *) scratch;

    cuda_memset( spad, 0, (*dev_lists + BONDS)->num_intrs * sizeof(int), "scratch" );

    Estimate_Cuda_Valence_Angles <<<BLOCKS_N, BLOCK_SIZE>>>
        ( system->d_my_atoms, (control_params *)control->d_control_params, 
          *(*dev_lists + BONDS), system->n, system->N, (int *)spad );
    cudaThreadSynchronize( );
    cudaCheckError( );

    copy_host_device( thbody, spad, (*dev_lists + BONDS)->num_intrs * sizeof(int),
            cudaMemcpyDeviceToHost, "thb:offsets" );

    *num_3body = 0;
    for ( i = 0; i < (*dev_lists + BONDS)->num_intrs; i++ )
    {
        *num_3body += thbody[i];
        thbody[i] += thbody[i - 1];
    }

    system->num_thbodies = thbody[(*dev_lists + BONDS)->num_intrs - 1];
}


CUDA_DEVICE real Compute_H( real r, real gamma, real *ctap )
{
    real taper, dr3gamij_1, dr3gamij_3;

    taper = ctap[7] * r + ctap[6];
    taper = taper * r + ctap[5];
    taper = taper * r + ctap[4];
    taper = taper * r + ctap[3];
    taper = taper * r + ctap[2];
    taper = taper * r + ctap[1];
    taper = taper * r + ctap[0];    

    dr3gamij_1 = ( r*r*r + gamma );
    dr3gamij_3 = POW( dr3gamij_1 , 1.0 / 3.0 );

    return taper * EV_to_KCALpMOL / dr3gamij_3;
}


CUDA_DEVICE real Compute_tabH( LR_lookup_table *t_LR, real r_ij, int ti, int tj, int num_atom_types )
{
    int r, tmin, tmax;
    real val, dif, base;
    LR_lookup_table *t; 

    tmin  = MIN( ti, tj );
    tmax  = MAX( ti, tj );
    t = &( t_LR[index_lr (tmin,tmax, num_atom_types)] );    

    /* cubic spline interpolation */
    r = (int)(r_ij * t->inv_dx);
    if( r == 0 )
    {
        ++r;
    }
    base = (real)(r+1) * t->dx;
    dif = r_ij - base;
    val = ((t->ele[r].d*dif + t->ele[r].c)*dif + t->ele[r].b)*dif
            + t->ele[r].a;
    val *= EV_to_KCALpMOL / C_ele;

    return val;
}


CUDA_GLOBAL void k_estimate_sparse_matrix (reax_atom *my_atoms, control_params *control, 
        reax_list p_far_nbrs, int n, int N, int renbr, int *indices)
{
    int i, j, pj;
    int start_i, end_i;
    int flag;
    real cutoff;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;
    reax_list *far_nbrs = &( p_far_nbrs );

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
    {
        return;
    }

    atom_i = &(my_atoms[i]);
    start_i = Dev_Start_Index(i, far_nbrs);
    end_i   = Dev_End_Index(i, far_nbrs);

    cutoff = control->nonb_cut;

    //++Htop;
    if ( i < n )
    {
        indices[i]++;
    }

    /* update i-j distance - check if j is within cutoff */
    for( pj = start_i; pj < end_i; ++pj )
    {
        nbr_pj = &( far_nbrs->select.far_nbr_list[pj] );
        j = nbr_pj->nbr;
        atom_j = &(my_atoms[j]);
        if( renbr )
        {
            if(nbr_pj->d <= cutoff)
            {
                flag = 1;
            }
            else
            {
                flag = 0;
            }
        }
        else
        {
            if (i < j)
            {
                nbr_pj->dvec[0] = atom_j->x[0] - atom_i->x[0];
                nbr_pj->dvec[1] = atom_j->x[1] - atom_i->x[1];
                nbr_pj->dvec[2] = atom_j->x[2] - atom_i->x[2];
            }
            else
            {
                nbr_pj->dvec[0] = atom_i->x[0] - atom_j->x[0];
                nbr_pj->dvec[1] = atom_i->x[1] - atom_j->x[1];
                nbr_pj->dvec[2] = atom_i->x[2] - atom_j->x[2];
            }
            nbr_pj->d = rvec_Norm_Sqr( nbr_pj->dvec );
            //TODO
            //if( nbr_pj->d <= (cutoff) ) {
            if( nbr_pj->d <= SQR(cutoff) )
            {
                nbr_pj->d = sqrt(nbr_pj->d);
                flag = 1;
            }
            else
            {
                flag = 0;
            }
        }

        if( flag )
        {
            /* H matrix entry */
            //if( j < n || atom_i->orig_id < atom_j->orig_id )
            //++Htop;
            //    indices [i] ++;
            //else if (j < n || atom_i->orig_id > atom_j->orig_id )
            //    indices [i] ++;

            //if ((i < n) || (j < n))
            //    indices [i] ++;
            //if ((i < n) && (i < j) && ((j < n) || atom_i->orig_id < atom_j->orig_id))
            //    indices [i] ++;
            //if ( i >= n && j < n && atom_i->orig_id > atom_j->orig_id)
            //    indices [i] ++;
            //else if ((i >=n) && (i > j) && ((j < n) || (atom_i->orig_id > atom_j->orig_id)))
            //    indices [i] ++;
            //THIS IS THE HOST CONDITION
            //if (i < n && i < j && ( j < n || atom_i->orig_id < atom_j->orig_id ))
            //if (i < n && i < j && atom_i->orig_id < atom_j->orig_id && j >=n)
            //    indices [i] ++;
            //THIS IS THE DEVICE CONDITION
            //if ( i > j && i >= n && j < n && atom_j->orig_id < atom_i->orig_id)
            //    indices [i] ++;

            //this is the working condition
            if (i < j && i < n && ( j < n || atom_i->orig_id < atom_j->orig_id))
            {
                indices[i]++;
            }
            else if (i > j && i >= n && j < n && atom_j->orig_id < atom_i->orig_id)
            {
                indices[i]++;
            }
            else if (i > j && i < n && ( j < n || atom_j->orig_id < atom_i->orig_id ))
            {
                indices[i]++;
            }
        }
    }
}


int Cuda_Estimate_Sparse_Matrix( reax_system *system, control_params *control, 
        simulation_data *data, reax_list **lists )
{
    int blocks, max_sp_entries;
    int *indices = (int *) scratch;
    int *h_indices = (int *) host_scratch;
    int total_sparse = 0;

    cuda_memset( indices, 0, sizeof(int) * system->N, "sp_matrix:indices" );

    blocks = system->N / DEF_BLOCK_SIZE + 
        ((system->N % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    //TODO
    k_estimate_sparse_matrix  <<< blocks, DEF_BLOCK_SIZE >>>
        (system->d_my_atoms, (control_params *)control->d_control_params, 
         *(*dev_lists + FAR_NBRS), system->n, system->N, 
         (((data->step-data->prev_steps) % control->reneighbor) == 0), indices);
    cudaThreadSynchronize( );
    cudaCheckError( );

    copy_host_device( h_indices, indices, sizeof(int) * system->N, 
            cudaMemcpyDeviceToHost, "sp_matrix:indices" );
    max_sp_entries = 0;    
    for (int i = 0; i < system->N; i++)
    {
        total_sparse += h_indices [i];
        if (max_sp_entries < h_indices[i])
        {
            max_sp_entries = h_indices[i];
        }
    }

    //fprintf (stderr, " TOTAL DEVICE SPARSE ENTRIES: %d \n", total_sparse );
    //fprintf (stderr, "p%d: Max sparse entries -> %d \n", system->my_rank, max_sp_entries );
    system->max_sparse_entries = max_sp_entries * SAFE_ZONE;

    return SUCCESS;
}


CUDA_GLOBAL void k_init_forces( reax_atom *my_atoms, single_body_parameters *sbp, 
        two_body_parameters *tbp, storage workspace, control_params *control, 
        reax_list far_nbrs, reax_list bonds, reax_list hbonds, 
        LR_lookup_table *t_LR, int n, int N, int num_atom_types, 
        int max_sparse_entries, int renbr, int max_hbonds )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int Htop;
    int btop_i, ihb, jhb, ihb_top;
    //int btop_j, jhb, jhb_top;
    int local, flag, flag2, flag3;
    real r_ij, cutoff;
    //reax_list *far_nbrs, *bonds, *hbonds;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;
    sparse_matrix *H = &(workspace.H);

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
    {
        return;
    }

    Htop = i * max_sparse_entries;
    btop_i = 0;

    //Commented for CUDA KERNEL
    //for( i = 0; i < system->N; ++i ) {
    atom_i = &(my_atoms[i]);
    type_i = atom_i->type;
    start_i = Dev_Start_Index( i, &far_nbrs );
    end_i = Dev_End_Index( i, &far_nbrs );
    btop_i = Dev_Start_Index( i, &bonds );

    sbp_i = &(sbp[type_i]);

    if( i < n )
    {
        local = 1;
        cutoff = control->nonb_cut;

        //update bond mark here
        workspace.bond_mark[i] = 0;
    }
    else
    {
        local = 0;
        cutoff = control->bond_cut;

        //update bond mark here
        workspace.bond_mark[i] = 1000;
    }

    ihb = -1;
    ihb_top = -1;
    //CHANGE ORIGINAL
    H->start[i] = Htop;

    if( local )
    {
        H->entries[Htop].j = i;
        H->entries[Htop].val = sbp_i->eta;
        ++Htop;
    }
    //CHANGE ORIGINAL

    if( control->hbond_cut > 0.0 )
    {
        ihb = sbp_i->p_hbond;
        //CHANGE ORIGINAL
        if( ihb == 1  || ihb == 2)
        {
            //CHANGE ORIGINAL
            //ihb_top = Dev_Start_Index( atom_i->Hindex, &hbonds );
            ihb_top = i * max_hbonds;
            Dev_Set_Start_Index( atom_i->Hindex, ihb_top, &hbonds );
        }
        else
        {
            ihb_top = -1;
        }
    }

    /* update i-j distance - check if j is within cutoff */
    for( pj = start_i; pj < end_i; ++pj )
    {
        nbr_pj = &( far_nbrs.select.far_nbr_list[pj] );
        j = nbr_pj->nbr;
        atom_j = &(my_atoms[j]);
        if( renbr )
        {
            if(nbr_pj->d <= cutoff)
            {
                flag = TRUE;
            }
            else
            {
                flag = FALSE;
            }

            if(nbr_pj->d <= control->nonb_cut)
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
            if (i < j)
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

            if( nbr_pj->d <= SQR( control->nonb_cut ) )
            {
                flag2 = TRUE;
            }
            else
            {
                flag2 = FALSE;
            }

            if( nbr_pj->d <= SQR( control->nonb_cut ) )
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
            ihb = sbp_i->p_hbond;
            type_j = atom_j->type;
            sbp_j = &(sbp[type_j]);
            jhb = sbp_j->p_hbond;
            if( control->hbond_cut > 0.0 && nbr_pj->d <= control->hbond_cut
                    && ihb == 2 && jhb == 1 && i >= n && j < n ) 
            {
                hbonds.select.hbond_list[ihb_top].nbr = j;
                hbonds.select.hbond_list[ihb_top].scl = -1;
                hbonds.select.hbond_list[ihb_top].ptr = nbr_pj;

                //CUDA SPECIFIC
                hbonds.select.hbond_list[ihb_top].sym_index = -1;
                rvec_MakeZero( hbonds.select.hbond_list[ihb_top].hb_f );

                ++ihb_top;
            }

            //if ((i < n) || (j < n))
            //if (local || ((i >= n) &&(j < n)))

            flag3 = FALSE;
            if (i < j && i < n && ( j < n || atom_i->orig_id < atom_j->orig_id))
            {
                flag3 = TRUE;
            }
            else if (i > j && i >= n && j < n && atom_j->orig_id < atom_i->orig_id)
            {
                flag3 = TRUE;
            }
            else if (i > j && i < n && ( j < n || atom_j->orig_id < atom_i->orig_id ))
            {
                flag3 = TRUE;
            }

            if ( flag3 == TRUE )
            {
                twbp = &(tbp[ index_tbp (type_i,type_j,num_atom_types)]);
                r_ij = nbr_pj->d;

                //if (renbr) {
                H->entries[Htop].j = j;
                if( control->tabulate == 0 )
                {
                    H->entries[Htop].val = Compute_H(r_ij,twbp->gamma,workspace.Tap);
                }
                else
                {
                    H->entries[Htop].val = Compute_tabH(t_LR, r_ij, type_i, type_j,num_atom_types);
                }
                //}
                ++Htop;
            }
        }

        if( flag == TRUE )
        {
            type_j = atom_j->type;
            r_ij = nbr_pj->d;
            sbp_j = &(sbp[type_j]);
            twbp = &(tbp[ index_tbp (type_i,type_j,num_atom_types)]);

            if ( local )
            {
                /* H matrix entry */
                /*
                   if( j < n || atom_i->orig_id < atom_j->orig_id ) {//tryQEq||1
                   H->entries[Htop].j = j;
                   if( control->tabulate == 0 )
                   H->entries[Htop].val = Compute_H(r_ij,twbp->gamma,workspace.Tap);
                   else H->entries[Htop].val = Compute_tabH(t_LR, r_ij, type_i, type_j,num_atom_types);
                   ++Htop;
                   } 
                   else if( j < n || atom_i->orig_id > atom_j->orig_id ) {//tryQEq||1
                   H->entries[Htop].j = j;
                   if( control->tabulate == 0 )
                   H->entries[Htop].val = Compute_H(r_ij,twbp->gamma,workspace.Tap);
                   else H->entries[Htop].val = Compute_tabH(t_LR, r_ij, type_i, type_j,num_atom_types);
                   ++Htop;
                   } 
                 */

                //bool condition = !((i >= n) && (j >= n));
                /* hydrogen bond lists */
                if( control->hbond_cut > 0 && (ihb==1 || ihb==2) &&
                        nbr_pj->d <= control->hbond_cut // && i < j
                  )
                {
                    jhb = sbp_j->p_hbond;
                    if( ihb == 1 && jhb == 2 )
                    {
                        hbonds.select.hbond_list[ihb_top].nbr = j;
                        if (i < j) 
                        {
                            hbonds.select.hbond_list[ihb_top].scl = 1;
                        }
                        else
                        {
                            hbonds.select.hbond_list[ihb_top].scl = -1;
                        }
                        hbonds.select.hbond_list[ihb_top].ptr = nbr_pj;

                        //CUDA SPECIFIC
                        hbonds.select.hbond_list[ihb_top].sym_index = -1;
                        rvec_MakeZero( hbonds.select.hbond_list[ihb_top].hb_f );

                        ++ihb_top;
                    }
                    else if( ihb == 2 && jhb == 1 && j < n )
                    {
                        //jhb_top = End_Index( atom_j->Hindex, hbonds );
                        hbonds.select.hbond_list[ihb_top].nbr = j;
                        hbonds.select.hbond_list[ihb_top].scl = -1;
                        hbonds.select.hbond_list[ihb_top].ptr = nbr_pj;

                        //CUDA SPECIFIC
                        hbonds.select.hbond_list[ihb_top].sym_index = -1;
                        rvec_MakeZero (hbonds.select.hbond_list[ihb_top].hb_f);

                        ++ihb_top;

                        //Set_End_Index( atom_j->Hindex, jhb_top+1, hbonds );
                        //++num_hbonds;
                    }
                }
            }

            /* uncorrected bond orders */
            if( nbr_pj->d <= control->bond_cut &&
                    Dev_BOp( bonds, control->bo_cut, i , btop_i, nbr_pj,
                        sbp_i, sbp_j, twbp, workspace.dDeltap_self,
                        workspace.total_bond_order ) == TRUE )
            {
                //num_bonds += 2;
                ++btop_i;

                /* Need to do later... since i and j are parallel
                   if( workspace->bond_mark[j] > workspace->bond_mark[i] + 1 )
                   workspace->bond_mark[j] = workspace->bond_mark[i] + 1;
                   else if( workspace->bond_mark[i] > workspace->bond_mark[j] + 1 ) {
                   workspace->bond_mark[i] = workspace->bond_mark[j] + 1;
                   }
                 */
            }
        }
    }

    Dev_Set_End_Index( i, btop_i, &bonds );
    //    if( local ) {
    H->end[i] = Htop;
    //   }
    //CHANGE ORIGINAL
    if ( ( ihb == 1 || ihb == 2 ) && ihb_top > 0 && control->hbond_cut > 0.0 )
    {
        Dev_Set_End_Index( atom_i->Hindex, ihb_top, &hbonds );
    }
    //} Commented for cuda kernel
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

    reax_list *bonds = &pbonds;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
    {
        return;
    }

    for ( j = Dev_Start_Index(i, bonds); j < Dev_End_Index(i, bonds); j++ )
    {
        ibond = &( bonds->select.bond_list [j] );
        nbr = ibond->nbr;

        for ( k = Dev_Start_Index(nbr, bonds); k < Dev_End_Index(nbr, bonds); k++ )
        {
            jbond = &( bonds->select.bond_list[k] );
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
    int __THREADS_PER_ATOM__ = HB_KER_SYM_THREADS_PER_ATOM;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id / __THREADS_PER_ATOM__;
    int lane_id = thread_id & (__THREADS_PER_ATOM__ - 1);
    int my_bucket = threadIdx.x / __THREADS_PER_ATOM__;

    if ( warp_id > N )
    {
        return;
    }

    i = warp_id;
    j = start + lane_id;
    start = Dev_Start_Index( my_atoms[i].Hindex, &hbonds );
    end = Dev_End_Index( my_atoms[i].Hindex, &hbonds );

    while ( j < end )
    {
        ihbond = &( hbonds.select.hbond_list [j] );
        nbr = ihbond->nbr;

        nbrstart = Dev_Start_Index (my_atoms[nbr].Hindex, &hbonds);
        nbrend = Dev_End_Index (my_atoms[nbr].Hindex, &hbonds);

        for ( k = nbrstart; k < nbrend; k++ )
        {
            jhbond = &( hbonds.select.hbond_list [k] );

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


////////////////////////
// HBOND ISSUE
CUDA_GLOBAL void k_update_bonds( reax_atom *my_atoms, 
        reax_list bonds, int n )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
    {
        return;
    }

    my_atoms[i].num_bonds = 
        MAX( Dev_Num_Entries(i, &bonds) * 2, MIN_BONDS );
}


CUDA_GLOBAL void k_update_hbonds( reax_atom *my_atoms, 
        reax_list hbonds, int n )
{
    int Hindex;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
    {
        return;
    }

    Hindex = my_atoms[i].Hindex;
    my_atoms[i].num_hbonds = 
        MAX( Dev_Num_Entries(Hindex, &hbonds) * SAFER_ZONE, MIN_HBONDS );
}
////////////////////////
////////////////////////
////////////////////////


int Cuda_Validate_Lists( reax_system *system, storage *workspace,
        reax_list **lists, control_params *control, 
        int step, int numH )
{
    int blocks;
    int i, comp, Hindex, ret;
    int total_3body, *thbody;
    int *index, *end_index;
    reax_list *bonds, *hbonds;
    reallocate_data *realloc;
    int max_sp_entries, num_hbonds, num_bonds;
    int total_sp_entries;
    int max_hbonds;
    real *spad = (real *) scratch;

    realloc = &( dev_workspace->realloc );
    blocks = system->n / DEF_BLOCK_SIZE + 
        ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);
    ret = SUCCESS;

    k_update_bonds <<< blocks, DEF_BLOCK_SIZE >>>
        (system->d_my_atoms, *(*lists + BONDS), 
         system->n);
    cudaThreadSynchronize( );
    cudaCheckError( );

    ////////////////////////
    // HBOND ISSUE
    //FIX - 4 - Added this check for hydrogen bond issue
    if ( control->hbond_cut > 0.0 && system->numH > 0 )
    {
        k_update_hbonds <<< blocks, DEF_BLOCK_SIZE >>>
            (system->d_my_atoms, *(*lists + HBONDS), 
             system->n);
        cudaThreadSynchronize( );
        cudaCheckError( );
    }

    /* validate charge matrix */
    memset( host_scratch, 0, 2 * system->N * sizeof(int) );
    index = (int *) host_scratch;
    end_index = index + system->N;
    copy_host_device( index, dev_workspace->H.start, system->N * sizeof(int), 
            cudaMemcpyDeviceToHost, "sparse_matrix:start" );
    copy_host_device( end_index, dev_workspace->H.end, system->N * sizeof(int), 
            cudaMemcpyDeviceToHost, "sparse_matrix:end" );
    max_sp_entries = total_sp_entries = 0;
    for (i = 0; i < system->N; i++ )
    {
        //if (i < system->N-1)
        //    comp = index [i+1];
        //else
        //    comp = dev_workspace->H.m;

        total_sp_entries += end_index [i] - index[i];
        if ( end_index[i] - index[i] > system->max_sparse_entries )
        {
            fprintf( stderr, "step%d-sparsemat-chk failed: i=%d start(i)=%d end(i)=%d \n",
                    step, i, index[i], end_index[i] );
            ret = FAILURE;
        }
        else if ( end_index[i] >= dev_workspace->H.m )
        {
            //SUDHIR_FIX_SPARSE_MATRIX
            //TODO move this carver
            fprintf( stderr, "p:%d - step%d-sparsemat-chk failed (exceed limits): i=%d start(i)=%d end(i)=%d \n", 
                    system->my_rank, step, i, index[i], end_index[i] );
            //TODO move this carver
            ret = FAILURE;
        }
        else
        {
            if ( max_sp_entries <= end_index[i] - index[i] )
            {
                max_sp_entries = end_index[i] - index[i];
            }
        }
    }
    //if (max_sp_entries <= end_index[i] - index[i])
    //    max_sp_entries = end_index[i] - index[i];

    //update the current step max_sp_entries;
    realloc->Htop = max_sp_entries;

#if defined(DEBUG)
    fprintf( stderr, "p:%d - Cuda_Reallocate: Total H matrix entries: %d, cap: %d, used: %d \n", 
            system->my_rank, dev_workspace->H.n, dev_workspace->H.m, total_sp_entries );
#endif

    if (total_sp_entries >= dev_workspace->H.m)
    {
        fprintf( stderr, "p:%d - **ran out of space for sparse matrix: step: %d, allocated: %d, used: %d \n", 
                system->my_rank, step, dev_workspace->H.m, total_sp_entries );

        ret = FAILURE;
    }

    /* validate bond list */
    if ( system->N > 0 )
    {
        num_bonds = 0;

        bonds = *lists + BONDS;
        memset( host_scratch, 0, 2 * system->N * sizeof(int) );
        index = (int *) host_scratch;
        end_index = index + system->N;

        copy_host_device( index, bonds->index, system->N * sizeof(int), 
                cudaMemcpyDeviceToHost, "dev_bonds:index" );
        copy_host_device( end_index, bonds->end_index, system->N * sizeof(int), 
                cudaMemcpyDeviceToHost, "dev_bonds:end_index" );

        for ( i = 0; i < system->N; i++ )
        {
            if ( (end_index[i] - index[i]) >= system->max_bonds[i] )
            {
                fprintf( stderr, "p:%d - **ran out of space for bonds: atom: %d, allocated: %d, used: %d \n",
                        system->my_rank, i, system->max_bonds[i], end_index[i] - index[i] );
                system->max_bonds[i] =
                        MAX( (int)CEIL( (end_index[i] - index[i]) * SAFE_ZONE ), MIN_BONDS );
                ret = FAILURE;
                realloc->bonds = TRUE;
            }
        }
        for ( i = system->N; i < system->total_cap; i++ )
        {
            system->max_bonds[i] = MIN_NBRS;
        }
    }

    /* validate Hbonds list */
    num_hbonds = 0;
    // FIX - 4 - added additional check here
    if ( numH > 0 && control->hbond_cut > 0.0 )
    {
        hbonds = *lists + HBONDS;
        memset( host_scratch, 0, 2 * hbonds->n * sizeof(int) );
        index = (int *) host_scratch;
        end_index = index + hbonds->n;

        copy_host_device( index, hbonds->index, hbonds->n * sizeof(int), 
                cudaMemcpyDeviceToHost, "hbonds:index" );
        copy_host_device( end_index, hbonds->end_index, hbonds->n * sizeof(int), 
                cudaMemcpyDeviceToHost, "hbonds:end_index" );

        /*
           for (i = 0; i < N-1; i++) {
           Hindex = my_atoms [i].Hindex;
           if (Hindex > -1) 
           comp = index [Hindex + 1];
           else
           comp = hbonds->num_intrs;

           if (end_index [Hindex] > comp) {
           fprintf(stderr,"step%d-atom:%d hbondchk failed: H=%d start(H)=%d end(H)=%d str(H+1)=%d\n",
           step, i, Hindex, index[Hindex], end_index[Hindex], comp );
           return FAILURE;
           }

           num_hbonds += MAX( (end_index [Hindex] - index [Hindex]) * 2, MIN_HBONDS * 2);
           }
           if (end_index [my_atoms[i].Hindex] > hbonds->num_intrs) {
           fprintf(stderr,"step%d-atom:%d hbondchk failed: H=%d start(H)=%d end(H)=%d num_intrs=%d\n",
           step, i, Hindex, index[Hindex], end_index[Hindex], hbonds->num_intrs);
           return FAILURE;
           }

           num_hbonds += MIN( (end_index [my_atoms[i].Hindex] - index [my_atoms[i].Hindex]) * 2, 
           2 * MIN_HBONDS);
           num_hbonds = MAX( num_hbonds, MIN_CAP*MIN_HBONDS );
           realloc->num_hbonds = num_hbonds;
         */

        max_hbonds = 0;
        for (i = 0; i < system->N; i++)
        {
            if (end_index[i] - index[i] >= system->max_hbonds)
            {
                //TODO: update
//                fprintf( stderr, "step%d-hbondchk failed: i=%d start(i)=%d end(i)=%d max_hbonds=%d\n",
//                        step, i, index[i], end_index[i], system->max_hbonds );
//                return FAILURE;
            }
            if (end_index[i] - index[i] >= max_hbonds)
            {
                max_hbonds = end_index[i] - index[i];
            }
        }
        realloc->num_hbonds = max_hbonds;
    }

    /* three body interactions */
    cuda_memset( spad, 0, (*lists + BONDS)->num_intrs * sizeof (int), "scratch" );
    Estimate_Cuda_Valence_Angles <<<BLOCKS_N, BLOCK_SIZE>>>
        ( system->d_my_atoms, (control_params *)control->d_control_params, 
          *(*lists + BONDS), system->n, system->N, (int *)spad);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [ESTIMATE_CUDA_VALENCE_ANGLES]\n" );

    thbody = (int *) host_scratch;
    memset( thbody, 0, sizeof(int) * (*lists + BONDS)->num_intrs );
    copy_host_device( thbody, spad, (*lists + BONDS)->num_intrs * sizeof(int),
            cudaMemcpyDeviceToHost, "thb:offsets" );

    total_3body = 0;
    for (i = 0; i < (*lists + BONDS)->num_intrs; i++)
    {
        total_3body += thbody[i];
        thbody[i] += thbody[i - 1];
    }

    system->num_thbodies = thbody[(*lists + BONDS)->num_intrs - 1];

    if ( system->num_thbodies > (*lists + THREE_BODIES)->num_intrs ||
                (*lists + THREE_BODIES)->n < (*lists + BONDS)->num_intrs )
    {
        realloc->num_3body = total_3body;
        system->num_thbodies = total_3body;
        ret = FAILURE;
    }
    fprintf( stderr, "system->num_thbodies = %d, lists:THREE_BODIES->num_intrs = %d,\n",
            system->num_thbodies, (*lists + THREE_BODIES)->num_intrs );
    fprintf( stderr, "lists:THREE_BODIES->n = %d, lists:BONDS->num_intrs = %d,\n",
            (*lists + THREE_BODIES)->n, (*lists + BONDS)->num_intrs );
    fprintf( stderr, "total_3body = %d\n", total_3body );

    if ( ret == SUCCESS )
    {
        /* copy the indexes into the thb list */
        copy_host_device( thbody, (*lists + THREE_BODIES)->index + 1,
                sizeof(int) * ((*lists + BONDS)->num_intrs - 1),
                cudaMemcpyHostToDevice, "dev_thb:index" );
        copy_host_device( thbody, (*lists + THREE_BODIES)->end_index + 1,
                sizeof(int) * ((*lists + BONDS)->num_intrs - 1),
                cudaMemcpyHostToDevice, "dev_thb:end_index" );
    }


    return ret;
}


CUDA_GLOBAL void k_init_bond_orders( reax_atom *my_atoms, reax_list far_nbrs, 
        reax_list bonds, real *total_bond_order, int N )
{
    int i, j, pj; 
    int start_i, end_i;
    int type_i, type_j;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N)
    {
        return;
    }

    atom_i = &(my_atoms[i]);
    start_i = Dev_Start_Index(i, &far_nbrs);
    end_i = Dev_End_Index(i, &far_nbrs);

    for( pj = start_i; pj < end_i; ++pj )
    { 
        // nbr_pj = &( far_nbrs.select.far_nbr_list[pj] );
        // j = nbr_pj->nbr;
        // atom_j = &(my_atoms[j]);

        //total_bond_order [i] ++;
        //atom_i->Hindex ++;
    }
}


CUDA_GLOBAL void k_bond_mark( reax_list p_bonds, storage p_workspace, int N )
{
    reax_list *bonds = &( p_bonds );
    storage *workspace = &( p_workspace );
    int j;

    //int i = blockIdx.x * blockDim.x + threadIdx.x;
    //if (i >= N) return;

    for (int i = 0; i < N; i++) 
    {
        for (int k = Dev_Start_Index (i, bonds); k < Dev_End_Index (i, bonds); k++)
        {
            bond_data *bdata = &( bonds->select.bond_list [k] );
            j = bdata->nbr;

            if (i < j )
            {
                if ( workspace->bond_mark [j] > (workspace->bond_mark [i] + 1) )
                {
                    workspace->bond_mark [j] = workspace->bond_mark [i] + 1;    
                }
                else if ( workspace->bond_mark [i] > (workspace->bond_mark [j] + 1) )
                {
                    workspace->bond_mark [i] = workspace->bond_mark [j] + 1;
                }
            }
        }
    }
}


int Cuda_Init_Forces( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control ) 
{
    int i, ret;
    int init_blocks, hblocks;

    //init the workspace (bond_mark)
    /*
       int blocks;
       cuda_memset (dev_workspace->bond_mark, 0, sizeof (int) * system->n, "bond_mark");

       blocks = (system->N - system->n) / DEF_BLOCK_SIZE + 
       (((system->N - system->n) % DEF_BLOCK_SIZE == 0) ? 0 : 1);
       k_init_bond_mark <<< blocks, DEF_BLOCK_SIZE >>>
       (system->n, (system->N - system->n), dev_workspace->bond_mark);
       cudaThreadSynchronize ();
       cudaCheckError ();
     */
    //validate total_bond_orders

    //main kernel
    init_blocks = (system->N) / DEF_BLOCK_SIZE + 
        (((system->N % DEF_BLOCK_SIZE) == 0) ? 0 : 1);
    //fprintf (stderr, " Total atoms: %d, blocks: %d \n", system->N, init_blocks );

    //    k_init_bond_orders <<<init_blocks, DEF_BLOCK_SIZE >>>
    //            ( system->d_my_atoms, *(*dev_lists + FAR_NBRS), *(*dev_lists + BONDS), 
    //                dev_workspace->total_bond_order, system->N);
    //    cudaThreadSynchronize ();
    //    cudaCheckError ();
    //    fprintf (stderr, " DONE WITH VALIDATION \n");

    k_init_forces <<<init_blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->reax_param.d_sbp,
          system->reax_param.d_tbp, *dev_workspace,
          (control_params *)control->d_control_params,
          *(*dev_lists + FAR_NBRS), *(*dev_lists + BONDS),
          *(*dev_lists + HBONDS), d_LR, system->n,
          system->N, system->reax_param.num_atom_types,
          system->max_sparse_entries, (((data->step-data->prev_steps) %
                  control->reneighbor) == 0), system->max_hbonds );
    cudaThreadSynchronize( );
    cudaCheckError( );

    //fix - sym_index and dbond_index
    New_fix_sym_dbond_indices <<<init_blocks, BLOCK_SIZE>>> 
        (*(*dev_lists + BONDS), system->N);
    cudaThreadSynchronize( );
    cudaCheckError( );

    ///////////////////////
    ///////////////////////
    // FIX - 4 - HBOND ISSUE
    if ( control->hbond_cut > 0 && system->numH > 0 )
    {
        //make hbond_list symmetric
        hblocks = (system->N * HB_KER_SYM_THREADS_PER_ATOM) / HB_SYM_BLOCK_SIZE + 
            ((((system->N * HB_KER_SYM_THREADS_PER_ATOM) % HB_SYM_BLOCK_SIZE) == 0) ? 0 : 1);
        //New_fix_sym_hbond_indices <<<hblocks, HB_BLOCK_SIZE >>> 
        New_fix_sym_hbond_indices <<<hblocks, HB_BLOCK_SIZE >>> 
            (system->d_my_atoms, *(*dev_lists + HBONDS), system->N);
        cudaThreadSynchronize( );
        cudaCheckError( );
    }

    //update bond_mark
    //k_bond_mark <<< init_blocks, DEF_BLOCK_SIZE>>>
    /*
       k_bond_mark <<< 1, 1>>>
       ( *(*dev_lists + BONDS), *dev_workspace, system->N);
       cudaThreadSynchronize ();
       cudaCheckError ();
     */

    //TODO
    //1. update the sparse matrix count for reallocation
    //2. update the bonds count for reallocation
    //3. update the hydrogen bonds count for reallocation

    /* validate lists */
    ret = Cuda_Validate_Lists( system, workspace, dev_lists, control,
            data->step, system->numH );

    return ret;
}


int Cuda_Init_Forces_No_Charges( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control ) 
{
    //TODO Implement later
    // when you figure out the bond_mark usage.

    return FAILURE;
}


void Cuda_Compute_Bonded_Forces( reax_system *system, control_params *control, 
        simulation_data *data, storage *workspace, 
        reax_list **lists, output_controls *out_control )
{
    int i;
    real t_start, t_elapsed;
    real *spad = (real *) scratch;
    rvec *rvec_spad;

    //1. Bond Order Interactions. - bond_orders.c
    t_start = Get_Time( );
    //fprintf (stderr, " Begin Bonded Forces ... %d x %d\n", BLOCKS_N, BLOCK_SIZE);
    Cuda_Calculate_BO_init  <<< BLOCKS_N, BLOCK_SIZE >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, 
          *dev_workspace, system->N );
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [CUDA_CALCULATE_BO_INIT]\n" );

    Cuda_Calculate_BO <<< BLOCKS_N, BLOCK_SIZE >>>
        ( system->d_my_atoms, system->reax_param.d_gp, system->reax_param.d_sbp, 
          system->reax_param.d_tbp, *dev_workspace, 
          *(*dev_lists + BONDS),
          system->reax_param.num_atom_types, system->N );
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [CUDA_CALCULATE_BO]\n" );


    Cuda_Update_Uncorrected_BO <<<BLOCKS_N, BLOCK_SIZE>>>
        (*dev_workspace, *(*dev_lists + BONDS), system->N);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [CUDA_UPDATE_UNCORRECTED_BO]\n" );

    Cuda_Update_Workspace_After_BO <<<BLOCKS_N, BLOCK_SIZE>>>
        (system->d_my_atoms, system->reax_param.d_gp, system->reax_param.d_sbp, 
         *dev_workspace, system->N);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [CUDA_UPDATE_WORKSPACE_AFTER_BO]\n" );

    t_elapsed = Get_Timing_Info( t_start );
    //fprintf (stderr, "Bond Orders... return value --> %d --- Timing %lf \n", cudaGetLastError (), t_elapsed );
    //fprintf (stderr, "Cuda_Calculate_Bond_Orders Done... \n");

    //2. Bond Energy Interactions. - bonds.c
    t_start = Get_Time( );
    cuda_memset( spad, 0, system->N * (2 * sizeof(real)) , "scratch" );

    Cuda_Bonds <<< BLOCKS, BLOCK_SIZE, sizeof(real)* BLOCK_SIZE >>>
        ( system->d_my_atoms, system->reax_param.d_gp, system->reax_param.d_sbp, system->reax_param.d_tbp,
          *dev_workspace, *(*dev_lists + BONDS), 
          system->n, system->reax_param.num_atom_types, spad );
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [CUDA_BONDS]\n" );

    //Reduction for E_BE
    k_reduction <<<BLOCKS, BLOCK_SIZE, sizeof(real) * BLOCK_SIZE >>>  
        (spad, spad + system->n,  system->n);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    k_reduction <<<1, BLOCKS_POW_2, sizeof(real) * BLOCKS_POW_2>>> 
        (spad + system->n, &((simulation_data *)data->d_simulation_data)->my_en.e_bond, BLOCKS_POW_2);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    t_elapsed = Get_Timing_Info( t_start );
    //fprintf (stderr, "Cuda_Bond_Energy ... return value --> %d --- Timing %lf \n", cudaGetLastError (), t_elapsed );
    //fprintf (stderr, "Cuda_Bond_Energy Done... \n");

    //3. Atom Energy Interactions. 
    t_start = Get_Time( );
    cuda_memset( spad, 0, ( 6 * sizeof(real) * system->n ), "scratch");

    Cuda_Atom_Energy <<<BLOCKS, BLOCK_SIZE>>>( system->d_my_atoms, system->reax_param.d_gp, 
            system->reax_param.d_sbp, system->reax_param.d_tbp, 
            *dev_workspace, 
            *(*dev_lists + BONDS), system->n, system->reax_param.num_atom_types, 
            spad, spad + 2 * system->n, spad + 4*system->n);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [CUDA_ATOM_ENERGY]\n" );

    //CHANGE ORIGINAL
    //Cuda_Atom_Energy_PostProcess <<<BLOCKS, BLOCK_SIZE >>>
    //                    ( *(*dev_lists + BONDS), *dev_workspace, system->n );
    Cuda_Atom_Energy_PostProcess <<<BLOCKS_N, BLOCK_SIZE >>>
        ( *(*dev_lists + BONDS), *dev_workspace, system->N );
    //CHANGE ORIGINAL
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [CUDA_ATOM_ENERGY_POST_PROCESS]\n" );

    //Reduction for E_Lp
    k_reduction <<<BLOCKS, BLOCK_SIZE, sizeof(real) * BLOCK_SIZE >>>  
        (spad, spad + system->n, system->n);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    k_reduction <<<1, BLOCKS_POW_2, sizeof(real) * BLOCKS_POW_2 >>>  
        (spad + system->n, &((simulation_data *)data->d_simulation_data)->my_en.e_lp, BLOCKS);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    //Reduction for E_Ov
    k_reduction <<<BLOCKS, BLOCK_SIZE, sizeof (real) * BLOCK_SIZE >>>  
        (spad + 2*system->n, spad + 3*system->n,  system->n);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    k_reduction <<<1, BLOCKS_POW_2, sizeof (real) * BLOCKS_POW_2 >>>  
        (spad + 3*system->n, &((simulation_data *)data->d_simulation_data)->my_en.e_ov, BLOCKS);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    //Reduction for E_Un
    k_reduction <<<BLOCKS, BLOCK_SIZE, sizeof (real) * BLOCK_SIZE >>>  
        (spad + 4*system->n, spad + 5*system->n,  system->n);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    k_reduction <<<1, BLOCKS_POW_2, sizeof (real) * BLOCKS_POW_2 >>>  
        (spad + 5*system->n, &((simulation_data *)data->d_simulation_data)->my_en.e_un, BLOCKS);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    t_elapsed = Get_Timing_Info( t_start );
    //fprintf (stderr, "test_LonePair_postprocess ... return value --> %d --- Timing %lf \n", cudaGetLastError (), t_elapsed );
    //fprintf (stderr, "test_LonePair_postprocess Done... \n");

    //4. Valence Angles Interactions. 
    t_start = Get_Time( );

    cuda_memset( spad, 0, 6 * sizeof(real) * system->N + sizeof(rvec) * system->N * 2, "scratch" );
    Cuda_Valence_Angles <<< BLOCKS_N, BLOCK_SIZE >>>
        ( system->d_my_atoms, system->reax_param.d_gp, 
          system->reax_param.d_sbp, system->reax_param.d_thbp, 
          (control_params *)control->d_control_params,
          *dev_workspace, *(*dev_lists + BONDS), *(*dev_lists + THREE_BODIES),
          system->n, system->N, system->reax_param.num_atom_types, 
          spad, spad + 2*system->N, spad + 4*system->N, (rvec *)(spad + 6*system->N));
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [CUDA_VALENCE_ANGLES]\n" );

    //Reduction for E_Ang
    k_reduction <<<BLOCKS_N, BLOCK_SIZE, sizeof (real) * BLOCK_SIZE >>>  
        (spad, spad + system->N,  system->N);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    k_reduction <<<1, BLOCKS_POW_2_N, sizeof (real) * BLOCKS_POW_2_N >>>
        (spad + system->N, &((simulation_data *)data->d_simulation_data)->my_en.e_ang, BLOCKS_N);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    //Reduction for E_Pen
    k_reduction <<<BLOCKS_N, BLOCK_SIZE, sizeof (real) * BLOCK_SIZE >>>
        (spad + 2*system->N, spad + 3*system->N,  system->N);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    k_reduction <<<1, BLOCKS_POW_2_N, sizeof (real) * BLOCKS_POW_2_N >>>
        (spad + 3*system->N, &((simulation_data *)data->d_simulation_data)->my_en.e_pen, BLOCKS_N);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    //Reduction for E_Coa
    k_reduction <<<BLOCKS_N, BLOCK_SIZE, sizeof (real) * BLOCK_SIZE >>>
        (spad + 4*system->N, spad + 5*system->N,  system->N);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    k_reduction <<<1, BLOCKS_POW_2_N, sizeof (real) * BLOCKS_POW_2_N >>>
        (spad + 5*system->N, &((simulation_data *)data->d_simulation_data)->my_en.e_coa, BLOCKS_N);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    //Reduction for ext_pres
    rvec_spad = (rvec *) (spad + 6*system->N);
    k_reduction_rvec <<<BLOCKS_N, BLOCK_SIZE, sizeof (rvec) * BLOCK_SIZE >>>
        (rvec_spad, rvec_spad + system->N,  system->N);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    k_reduction_rvec <<<1, BLOCKS_POW_2_N, sizeof (rvec) * BLOCKS_POW_2_N >>>
        (rvec_spad + system->N, &((simulation_data *)data->d_simulation_data)->my_ext_press, BLOCKS_N);
    cudaThreadSynchronize ();
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    Cuda_Valence_Angles_PostProcess <<< BLOCKS_N, BLOCK_SIZE >>>
        (  system->d_my_atoms,
           (control_params *)control->d_control_params,
           *dev_workspace,
           *(*dev_lists + BONDS),
           system->N );
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [CUDA_VALENCE_ANGLES_POST_PROCESS]\n" );

    t_elapsed = Get_Timing_Info( t_start );
    //fprintf (stderr, "Three_Body_Interactions ...  Timing %lf \n", t_elapsed );
    //fprintf (stderr, "Three_Body_Interactions Done... \n");

    //5. Torsion Angles Interactions. 
    t_start = Get_Time( );
    cuda_memset (spad, 0, ( 4 * sizeof (real) * system->n + sizeof (rvec) * system->n * 2), "scratch");
    Cuda_Torsion_Angles <<< BLOCKS, BLOCK_SIZE >>>
        ( system->d_my_atoms,
          system->reax_param.d_gp,
          system->reax_param.d_fbp,
          (control_params *)control->d_control_params,
          *(*dev_lists + BONDS), *(*dev_lists + THREE_BODIES),
          *dev_workspace,
          system->n, system->reax_param.num_atom_types, 
          spad, spad + 2*system->n, (rvec *) (spad + 4*system->n));
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [CUDA_TORSION_ANGLES]\n" );

    //Reduction for E_Tor
    k_reduction <<<BLOCKS, BLOCK_SIZE, sizeof (real) * BLOCK_SIZE >>>
        (spad, spad + system->n,  system->n);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    k_reduction <<<1, BLOCKS_POW_2, sizeof (real) * BLOCKS_POW_2 >>>
        (spad + system->n, &((simulation_data *)data->d_simulation_data)->my_en.e_tor, BLOCKS);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    //Reduction for E_Con
    k_reduction <<<BLOCKS, BLOCK_SIZE, sizeof (real) * BLOCK_SIZE >>>
        (spad + 2*system->n, spad + 3*system->n,  system->n);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    k_reduction <<<1, BLOCKS_POW_2, sizeof (real) * BLOCKS_POW_2 >>>
        (spad + 3*system->n, &((simulation_data *)data->d_simulation_data)->my_en.e_con, BLOCKS);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    //Reduction for ext_pres
    rvec_spad = (rvec *) (spad + 4*system->n);
    k_reduction_rvec <<<BLOCKS, BLOCK_SIZE, sizeof (rvec) * BLOCK_SIZE >>>
        (rvec_spad, rvec_spad + system->n,  system->n);
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    k_reduction_rvec <<<1, BLOCKS_POW_2, sizeof (rvec) * BLOCKS_POW_2 >>>
            ( rvec_spad + system->n,
            &((simulation_data *)data->d_simulation_data)->my_ext_press, BLOCKS );
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [REDUCTION]\n" );

    //Post process here
    Cuda_Torsion_Angles_PostProcess   <<< BLOCKS_N, BLOCK_SIZE >>>
            ( system->d_my_atoms, *dev_workspace, *(*dev_lists + BONDS),
            system->N );
    cudaThreadSynchronize( );
    cudaCheckError( );
    fprintf( stderr, "      [CUDA_TORSION_POST_PROCESS]\n" );

    t_elapsed = Get_Timing_Info( t_start );
    //fprintf (stderr, "Four_Body_post process return value --> %d --- Four body Timing %lf \n", cudaGetLastError (), t_elapsed );
    //fprintf (stderr, " Four_Body_ Done... \n");

    //6. Hydrogen Bonds Interactions.
    // FIX - 4 - Added additional check here
    if ((control->hbond_cut > 0) && (system->numH > 0))
    {
        t_start = Get_Time( );
        cuda_memset( spad, 0,
                2 * sizeof(real) * system->n + sizeof(rvec) * system->n * 2, "scratch" );

        int hbs = ((system->n * HB_KER_THREADS_PER_ATOM)/ HB_BLOCK_SIZE) + 
            (((system->n * HB_KER_THREADS_PER_ATOM) % HB_BLOCK_SIZE) == 0 ? 0 : 1);
        Cuda_Hydrogen_Bonds_MT <<<hbs, HB_BLOCK_SIZE, 
                HB_BLOCK_SIZE * (2 * sizeof(real) + 2 * sizeof(rvec)) >>>
        //Cuda_Hydrogen_Bonds <<< BLOCKS, BLOCK_SIZE>>>
                        ( system->d_my_atoms, system->reax_param.d_sbp,
                        system->reax_param.d_hbp, system->reax_param.d_gp,
                        (control_params *)control->d_control_params,
                        *dev_workspace, *(*dev_lists + BONDS), *(*dev_lists + HBONDS),
                        system->n, system->reax_param.num_atom_types,
                        spad, (rvec *) (spad + 2*system->n));
        cudaThreadSynchronize( );
        cudaCheckError( );
        fprintf( stderr, "      [CUDA_HYDROGEN_BONDS_MT]\n" );

        //Reduction for E_HB
        k_reduction <<<BLOCKS, BLOCK_SIZE, sizeof(real) * BLOCK_SIZE >>>
            (spad, spad + system->n,  system->n);
        cudaThreadSynchronize( );
        cudaCheckError( );
        fprintf( stderr, "      [REDUCTION]\n" );

        k_reduction <<<1, BLOCKS_POW_2, sizeof(real) * BLOCKS_POW_2 >>>
            (spad + system->n, &((simulation_data *)data->d_simulation_data)->my_en.e_hb, BLOCKS);
        cudaThreadSynchronize( );
        cudaCheckError( );
        fprintf( stderr, "      [REDUCTION]\n" );

        //Reduction for ext_pres
        rvec_spad = (rvec *) (spad + 2*system->n);
        k_reduction_rvec <<<BLOCKS, BLOCK_SIZE, sizeof (rvec) * BLOCK_SIZE >>>
            (rvec_spad, rvec_spad + system->n,  system->n);
        cudaThreadSynchronize( );
        cudaCheckError( );
        fprintf( stderr, "      [REDUCTION]\n" );

        k_reduction_rvec <<<1, BLOCKS_POW_2, sizeof (rvec) * BLOCKS_POW_2 >>>
            (rvec_spad + system->n, &((simulation_data *)data->d_simulation_data)->my_ext_press, BLOCKS);
        cudaThreadSynchronize( );
        cudaCheckError( );
        fprintf( stderr, "      [REDUCTION]\n" );

        //post process step1:
        Cuda_Hydrogen_Bonds_PostProcess <<< BLOCKS_N, BLOCK_SIZE, BLOCK_SIZE * sizeof (rvec) >>>
            (  system->d_my_atoms, *dev_workspace,
               *(*dev_lists + BONDS), system->N );
        cudaThreadSynchronize( );
        cudaCheckError( );
        fprintf( stderr, "      [CUDA_HYDROGEN_BONDS_POST_PROCESS]\n" );

        //post process step2:
        /*
           Cuda_Hydrogen_Bonds_HNbrs <<< system->N, 32, 32 * sizeof (rvec)>>>
           (  system->d_my_atoms,
         *dev_workspace,
         *(*dev_lists + HBONDS));
         */
        int hnbrs_bl = ((system->N * HB_POST_PROC_KER_THREADS_PER_ATOM)/ HB_POST_PROC_BLOCK_SIZE) + 
            (((system->N * HB_POST_PROC_KER_THREADS_PER_ATOM) % HB_POST_PROC_BLOCK_SIZE) == 0 ? 0 : 1);
        Cuda_Hydrogen_Bonds_HNbrs_BL <<< hnbrs_bl, HB_POST_PROC_BLOCK_SIZE, 
                HB_POST_PROC_BLOCK_SIZE * sizeof (rvec)>>>
                        ( system->d_my_atoms, *dev_workspace,
                        *(*dev_lists + HBONDS), system->N);
        cudaThreadSynchronize( );
        cudaCheckError( );
        fprintf( stderr, "      [CUDA_HYDROGEN_BONDS_HNBRS_BL]\n" );

        t_elapsed = Get_Timing_Info( t_start );
        //fprintf (stderr, "Hydrogen bonds return value --> %d --- HydrogenBonds Timing %lf \n", cudaGetLastError (), t_elapsed );
        //fprintf (stderr, "Hydrogen_Bond Done... \n");    
    }
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
