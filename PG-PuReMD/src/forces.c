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

#include "index_utils.h"
#ifdef HAVE_CUDA
  #include "cuda_forces.h"
  #include "cuda_linear_solvers.h"
  #include "cuda_neighbors.h"
  #include "cuda_utils.h"
  #include "validation.h"
#endif

#if defined(PURE_REAX)
  #include "forces.h"
  #include "bond_orders.h"
  #include "bonds.h"
  #include "basic_comm.h"
  #include "hydrogen_bonds.h"
  #include "io_tools.h"
  #include "list.h"
  #include "lookup.h"
  #include "multi_body.h"
  #include "nonbonded.h"
  #include "qEq.h"
  #include "tool_box.h"
  #include "torsion_angles.h"
  #include "valence_angles.h"
  #include "vector.h"
#elif defined(LAMMPS_REAX)
  #include "reax_forces.h"
  #include "reax_bond_orders.h"
  #include "reax_bonds.h"
  #include "reax_basic_comm.h"
  #include "reax_hydrogen_bonds.h"
  #include "reax_io_tools.h"
  #include "reax_list.h"
  #include "reax_lookup.h"
  #include "reax_multi_body.h"
  #include "reax_nonbonded.h"
  #include "reax_tool_box.h"
  #include "reax_torsion_angles.h"
  #include "reax_valence_angles.h"
  #include "reax_vector.h"
#endif


#ifdef HAVE_CUDA
void Cuda_Total_Forces (reax_system *, control_params *, simulation_data *, storage *);
void Cuda_Total_Forces_PURE (reax_system *, storage *);
#endif


interaction_function Interaction_Functions[NUM_INTRS];


/* placeholder for unused interactions in interaction list
 * Interaction_Functions, which is initialized in Init_Force_Functions */
void Dummy_Interaction( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control )
{
}


void Init_Force_Functions( control_params *control )
{
    Interaction_Functions[0] = BO;
    Interaction_Functions[1] = Bonds; //Dummy_Interaction;
    Interaction_Functions[2] = Atom_Energy; //Dummy_Interaction;
    Interaction_Functions[3] = Valence_Angles; //Dummy_Interaction;
    Interaction_Functions[4] = Torsion_Angles; //Dummy_Interaction;
    if ( control->hbond_cut > 0.0 )
    {
        Interaction_Functions[5] = Hydrogen_Bonds;
    }
    else
    {
        Interaction_Functions[5] = Dummy_Interaction;
    }
    Interaction_Functions[6] = Dummy_Interaction; //empty
    Interaction_Functions[7] = Dummy_Interaction; //empty
    Interaction_Functions[8] = Dummy_Interaction; //empty
    Interaction_Functions[9] = Dummy_Interaction; //empty
}


void Compute_Bonded_Forces( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control )
{
    int i;

    /* Mark beginning of a new timestep in bonded energy files */
#if defined(TEST_ENERGY)
    Debug_Marker_Bonded( out_control, data->step );
#endif

    /* Implement all force calls as function pointers */
    for( i = 0; i < NUM_INTRS; i++ )
    {
#if defined(DEBUG)
        fprintf( stderr, "p%d: starting f%d\n", system->my_rank, i );
        MPI_Barrier( MPI_COMM_WORLD );
#endif

        (Interaction_Functions[i])( system, control, data, workspace, lists, out_control );

#if defined(DEBUG)
        fprintf( stderr, "p%d: f%d done\n", system->my_rank, i );
        MPI_Barrier( MPI_COMM_WORLD );
#endif
    }
}


void Compute_NonBonded_Forces( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    /* Mark beginning of a new timestep in nonbonded energy files */
#if defined(TEST_ENERGY)
    Debug_Marker_Nonbonded( out_control, data->step );
#endif

    /* van der Waals and Coulomb interactions */
    if ( control->tabulate == 0 )
    {
        vdW_Coulomb_Energy( system, control, data, workspace, lists, out_control );
    }
    else
    {
        Tabulated_vdW_Coulomb_Energy( system, control, data, workspace, lists, out_control );
    }

#if defined(DEBUG)
    fprintf( stderr, "p%d: nonbonded forces done\n", system->my_rank );
    MPI_Barrier( MPI_COMM_WORLD );
#endif
}


/* this version of Compute_Total_Force computes forces from
   coefficients accumulated by all interaction functions.
   Saves enormous time & space! */
void Compute_Total_Force( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        mpi_datatypes *mpi_data )
{
    int i, pj;
    reax_list *bonds = (*lists) + BONDS;

    for ( i = 0; i < system->N; ++i )
    {
        for ( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj )
        {
            if ( i < bonds->select.bond_list[pj].nbr )
            {
                if ( control->virial == 0 )
                {
                    Add_dBond_to_Forces( i, pj, workspace, lists );
                }
                else
                {
                    Add_dBond_to_Forces_NPT( i, pj, data, workspace, lists );
                }
            }
        }
    }

    //Print_Total_Force( system, data, workspace );
#if defined(PURE_REAX)
    /* now all forces are computed to their partially-final values
       based on the neighbors information each processor has had.
       final values of force on each atom needs to be computed by adding up
       all partially-final pieces */
    Coll( system, mpi_data, workspace->f, mpi_data->mpi_rvec,
            sizeof(rvec) / sizeof(void), rvec_unpacker );
    for ( i = 0; i < system->n; ++i )
    {
        rvec_Copy( system->my_atoms[i].f, workspace->f[i] );
    }

#if defined(TEST_FORCES)
    Coll( system, mpi_data, workspace->f_ele, mpi_data->mpi_rvec, rvec_unpacker);
    Coll( system, mpi_data, workspace->f_vdw, mpi_data->mpi_rvec, rvec_unpacker);
    Coll( system, mpi_data, workspace->f_be, mpi_data->mpi_rvec, rvec_unpacker );
    Coll( system, mpi_data, workspace->f_lp, mpi_data->mpi_rvec, rvec_unpacker );
    Coll( system, mpi_data, workspace->f_ov, mpi_data->mpi_rvec, rvec_unpacker );
    Coll( system, mpi_data, workspace->f_un, mpi_data->mpi_rvec, rvec_unpacker );
    Coll( system, mpi_data, workspace->f_ang, mpi_data->mpi_rvec, rvec_unpacker);
    Coll( system, mpi_data, workspace->f_coa, mpi_data->mpi_rvec, rvec_unpacker);
    Coll( system, mpi_data, workspace->f_pen, mpi_data->mpi_rvec, rvec_unpacker);
    Coll( system, mpi_data, workspace->f_hb, mpi_data->mpi_rvec, rvec_unpacker );
    Coll( system, mpi_data, workspace->f_tor, mpi_data->mpi_rvec, rvec_unpacker);
    Coll( system, mpi_data, workspace->f_con, mpi_data->mpi_rvec, rvec_unpacker);
#endif

#endif
}


#ifdef HAVE_CUDA
void Cuda_Compute_Total_Force( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, mpi_datatypes *mpi_data )
{
    rvec *f = (rvec *) host_scratch;
    memset( f, 0, sizeof(rvec) * system->N );

    Cuda_Total_Forces( system, control, data, workspace );

#if defined(PURE_REAX)
    /* now all forces are computed to their partially-final values
       based on the neighbors information each processor has had.
       final values of force on each atom needs to be computed by adding up
       all partially-final pieces */

    //MVAPICH2
    copy_host_device( f, dev_workspace->f, sizeof(rvec) * system->N ,
            cudaMemcpyDeviceToHost, "total_force:f:get" );

    Coll( system, mpi_data, f, mpi_data->mpi_rvec,
          sizeof(rvec) / sizeof(void), rvec_unpacker );

    copy_host_device( f, dev_workspace->f, sizeof(rvec) * system->N,
            cudaMemcpyHostToDevice, "total_force:f:put" );

    Cuda_Total_Forces_PURE( system, dev_workspace );
#endif

}
#endif


// Essentially no-cuda copies of cuda kernels, to be used only in the mpi-not-gpu version
////////////////////////
// HBOND ISSUE
void mpi_not_gpu_update_bonds( reax_atom *my_atoms, reax_list bonds, int n )
{
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //  if (i >= n) return;
    int i;
    for (i = 0; i < n; i++)
    {
        my_atoms [i].num_bonds =
            MAX(Num_Entries(i, &bonds) * 2, MIN_BONDS);
    }
}


void mpi_not_gpu_update_hbonds( reax_atom *my_atoms, reax_list hbonds, int n )
{
    int Hindex;
    int i;
    //int i = blockIdx.x * blockDim.x + threadIdx.x;
    //if (i >= n) return;
    for (i = 0; i < n; i++)
    {
        Hindex = my_atoms[i].Hindex;
        my_atoms [i].num_hbonds =
            MAX(Num_Entries(Hindex, &hbonds) * SAFER_ZONE, MIN_HBONDS);
    }
}


// Essentially a copy of cuda_validate_lists, but with all cuda-dependent kernels turned into serial versions
int MPI_Not_GPU_Validate_Lists( reax_system *system, storage *workspace,
        reax_list **lists, control_params *control, int step, int n, int N, int numH )
{
    int blocks;
    int i, comp, Hindex;
    int *index, *end_index;
    reax_list *bonds, *hbonds;
    reax_atom *my_atoms;
    reallocate_data *realloc;
    realloc = &( workspace->realloc);

    int max_sp_entries, num_hbonds, num_bonds;
    int total_sp_entries;

    //blocks = system->n / DEF_BLOCK_SIZE +
    //    ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    //ker_update_bonds <<< blocks, DEF_BLOCK_SIZE >>>
    //    (system->d_my_atoms, *(*lists + BONDS),
    //   system->n);
    //cudaThreadSynchronize ();
    //cudaCheckError ();
    mpi_not_gpu_update_bonds(system->my_atoms, *(*lists + BONDS), system->n);

    ////////////////////////
    // HBOND ISSUE
    //FIX - 4 - Added this check for hydrogen bond issue
    if ((control->hbond_cut > 0) && (system->numH > 0))
    {
        //ker_update_hbonds <<< blocks, DEF_BLOCK_SIZE >>>
        //    (system->d_my_atoms, *(*lists + HBONDS),
        //     system->n);
        //cudaThreadSynchronize ();
        //cudaCheckError ();
        mpi_not_gpu_update_hbonds(system->my_atoms, *(*lists + HBONDS), system->n);

    }

    //validate sparse matrix entries.
    //memset (host_scratch, 0, 2 * system->N * sizeof (int));
    //index = (int *) host_scratch;
    //end_index = index + system->N;

    index = workspace->H.start;
    end_index = workspace->H.end;

    // immediately set these to host version since there is no device version.
    //memcpy(index, workspace->H.start, system->N * sizeof (int));
    //memcpy(end_index, workspace->H.end, system->N * sizeof (int));

    // don't need these, everything is already at host
    //copy_host_device (index, dev_workspace->H.start, system->N * sizeof (int),
    //        cudaMemcpyDeviceToHost, "sparse_matrix:start" );
    //copy_host_device (end_index, dev_workspace->H.end, system->N * sizeof (int),
    //        cudaMemcpyDeviceToHost, "sparse_matrix:end" );
    max_sp_entries = total_sp_entries = 0;

    for (i = 0; i < n; i++ )
    {
        //if (i < N-1)
        //    comp = index [i+1];
        //else
        //    comp = dev_workspace->H.m;

        total_sp_entries += end_index [i] - index[i];
        if (end_index [i] - index[i] > system->max_sparse_entries)
        {
            fprintf( stderr, "step%d-sparsemat-chk failed: i=%d start(i)=%d end(i)=%d \n",
                     step, i, index[i], end_index[i] );
            return FAILURE;
        }
        else if (end_index[i] >= workspace->H.m)
        {
            //SUDHIR_FIX_SPARSE_MATRIX
            //TODO move this carver
            fprintf (stderr, "p:%d - step%d-sparsemat-chk failed (exceed limits): i=%d start(i)=%d end(i)=%d \n",
                     system->my_rank, step, i, index[i], end_index[i]);
            //TODO move this carver
            return FAILURE;
        }
        else
        {
            if (max_sp_entries <= end_index[i] - index [i])
                max_sp_entries = end_index[i] - index [i];
        }
    }
    //if (max_sp_entries <= end_index[i] - index [i])
    //    max_sp_entries = end_index[i] - index [i];

    //update the current step max_sp_entries;
    realloc->Htop = max_sp_entries;

#if defined(DEBUG)
    fprintf (stderr, "p:%d - MPI-Not-GPU Reallocate: Total H matrix entries: %d, cap: %d, used: %d \n",
             system->my_rank, workspace->H.n, workspace->H.m, total_sp_entries);
#endif

    if (total_sp_entries >= workspace->H.m)
    {
        fprintf (stderr, "p:%d - **ran out of space for sparse matrix: step: %d, allocated: %d, used: %d \n",
                 system->my_rank, step, workspace->H.m, total_sp_entries);

        return FAILURE;
    }

    //validate Bond list
    if (N > 0)
    {
        num_bonds = 0;

        bonds = *lists + BONDS;
        //  memset (host_scratch, 0, 2 * bonds->n * sizeof (int));

        //  index = (int *) host_scratch;
        // end_index = index + bonds->n;
        index = bonds->index;
        end_index = bonds->end_index;


        //  memcpy(index, bonds->index, bonds->n * sizeof (int));
        // memcpy(end_index, bonds->end_index, bonds->n * sizeof (int));
        /*
                copy_host_device (index, bonds->index, bonds->n * sizeof (int),
                        cudaMemcpyDeviceToHost, "bonds:index");
                copy_host_device (end_index, bonds->end_index, bonds->n * sizeof (int),
                        cudaMemcpyDeviceToHost, "bonds:end_index");
        */
        /*
           for (i = 0; i < N; i++) {
           if (i < N-1)
           comp = index [i+1];
           else
           comp = bonds->num_intrs;

           if (end_index [i] > comp) {
           fprintf( stderr, "step%d-bondchk failed: i=%d start(i)=%d end(i)=%d str(i+1)=%d\n",
           step, i, index[i], end_index[i], comp );
           return FAILURE;
           }

           num_bonds += MAX( (end_index[i] - index[i]) * 4, MIN_BONDS);
           }

           if (end_index[N-1] >= bonds->num_intrs) {
           fprintf( stderr, "step%d-bondchk failed(end): i=N-1 start(i)=%d end(i)=%d num_intrs=%d\n",
           step, index[N-1], end_index[N-1], bonds->num_intrs);
           return FAILURE;
           }
           num_bonds = MAX( num_bonds, MIN_CAP*MIN_BONDS );
        //check the condition for reallocation
        realloc->num_bonds = num_bonds;
         */

        int max_bonds = 0;
        for (i = 0; i < N; i++)
        {
            if (end_index[i] - index[i] >= system->max_bonds[i])
            {
#if defined(DEBUG)
                fprintf( stderr, "MPI-Not-GPU step%d-bondchk failed: i=%d start(i)=%d end(i)=%d max_bonds=%d\n",
                        step, i, index[i], end_index[i], system->max_bonds[i]);
#endif

                return FAILURE;
            }
            if (end_index[i] - index[i] >= max_bonds)
                max_bonds = end_index[i] - index[i];
        }
        realloc->num_bonds = max_bonds;

    }

    //validate Hbonds list
    num_hbonds = 0;
    // FIX - 4 - added additional check here
    if ((numH > 0) && (control->hbond_cut > 0))
    {
        hbonds = *lists + HBONDS;
        memset (host_scratch, 0, 2 * hbonds->n * sizeof (int) + sizeof (reax_atom) * system->N);
        index = (int *) host_scratch;
        end_index = index + hbonds->n;
        my_atoms = (reax_atom *)(end_index + hbonds->n);
        /*
                copy_host_device (index, hbonds->index, hbonds->n * sizeof (int),
                        cudaMemcpyDeviceToHost, "hbonds:index");
                copy_host_device (end_index, hbonds->end_index, hbonds->n * sizeof (int),
                        cudaMemcpyDeviceToHost, "hbonds:end_index");
                copy_host_device (my_atoms, system->d_my_atoms, system->N * sizeof (reax_atom),
                        cudaMemcpyDeviceToHost, "system:d_my_atoms");
        */
        //fprintf (stderr, " Total local atoms: %d \n", n);

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

        int max_hbonds = 0;
        for (i = 0; i < N; i++)
        {
            if (end_index[i] - index[i] >= system->max_hbonds)
            {
                fprintf( stderr, "step%d-hbondchk failed: i=%d start(i)=%d end(i)=%d max_hbonds=%d\n",
                         step, i, index[i], end_index[i], system->max_hbonds);
                return FAILURE;
            }
            if (end_index[i] - index[i] >= max_hbonds)
                max_hbonds = end_index[i] - index[i];
        }
        realloc->num_hbonds = max_hbonds;
    }

    return SUCCESS;
}


/*
void Validate_Lists( reax_system *system, storage *workspace, reax_list **lists,
                     int step, int n, int N, int numH )
{
    int i, comp, Hindex;
    reax_list *bonds, *hbonds;
    reallocate_data *realloc;
    realloc = &(workspace->realloc);

    // bond list
    if ( N > 0 )
    {
        bonds = *lists + BONDS;

        for ( i = 0; i < N; ++i )
        {
            if ( i < n )
                system->my_atoms[i].num_bonds = MAX(Num_Entries(i, bonds) * 2, MIN_BONDS);

            //if( End_Index(i, bonds) >= Start_Index(i+1, bonds)-2 )
            //workspace->realloc.bonds = 1;

            if ( i < N - 1 )
                comp = Start_Index(i + 1, bonds);
            else comp = bonds->num_intrs;

            if ( End_Index(i, bonds) > comp )
            {
                fprintf( stderr, "step%d-bondchk failed: i=%d end(i)=%d str(i+1)=%d\n",
                         step, i, End_Index(i, bonds), comp );
                MPI_Abort( MPI_COMM_WORLD, INSUFFICIENT_MEMORY );
            }
        }
    }


    // hbonds list
    if ( numH > 0 )
    {
        hbonds = *lists + HBONDS;

        for ( i = 0; i < n; ++i )
        {
            Hindex = system->my_atoms[i].Hindex;
            if ( Hindex > -1 )
            {
                system->my_atoms[i].num_hbonds =
                    MAX( Num_Entries(Hindex, hbonds) * SAFER_ZONE, MIN_HBONDS );
*/
//if( Num_Entries(i, hbonds) >=
//(Start_Index(i+1,hbonds)-Start_Index(i,hbonds))*0.90/*DANGER_ZONE*/){
//  workspace->realloc.hbonds = 1;
/*
                //TODO
                if ( Hindex < system->n - 1 )
                    comp = Start_Index(Hindex + 1, hbonds);
                else comp = hbonds->num_intrs;

                if ( End_Index(Hindex, hbonds) > comp )
                {
                    fprintf(stderr, "step%d-hbondchk failed: H=%d end(H)=%d str(H+1)=%d\n",
                            step, Hindex, End_Index(Hindex, hbonds), comp );
                    MPI_Abort( MPI_COMM_WORLD, INSUFFICIENT_MEMORY );
                }
            }
        }
    }
}*/


void Validate_Lists( reax_system *system, storage *workspace, reax_list **lists,
        int step, int n, int N, int numH, MPI_Comm comm )
{
    int i, comp, Hindex;
    reax_list *bonds, *hbonds;
    reallocate_data *realloc;
    realloc = &(workspace->realloc);

    /* bond list */
    if ( N > 0 )
    {
        bonds = *lists + BONDS;

        for ( i = 0; i < N; ++i )
        {
            // if( i < n ) - we need to update ghost estimates for delayed nbrings
            system->my_atoms[i].num_bonds = MAX(Num_Entries(i, bonds) * 2, MIN_BONDS);

            //if( End_Index(i, bonds) >= Start_Index(i+1, bonds)-2 )
            //workspace->realloc.bonds = 1;

            if ( i < N - 1 )
                comp = Start_Index(i + 1, bonds);
            else comp = bonds->num_intrs;

            if ( End_Index(i, bonds) > comp )
            {
                fprintf( stderr, "step%d-bondchk failed: i=%d end(i)=%d str(i+1)=%d\n",
                         step, i, End_Index(i, bonds), comp );
                MPI_Abort( comm, INSUFFICIENT_MEMORY );
            }
        }
    }

    /* hbonds list */
    if ( numH > 0 )
    {
        hbonds = *lists + HBONDS;

        for ( i = 0; i < n; ++i )
        {
            Hindex = system->my_atoms[i].Hindex;
            if ( Hindex > -1 )
            {
                system->my_atoms[i].num_hbonds =
                    (int)(MAX( Num_Entries(Hindex, hbonds) * SAFER_ZONE, MIN_HBONDS ));

                //if( Num_Entries(i, hbonds) >=
                //(Start_Index(i+1,hbonds)-Start_Index(i,hbonds))*0.90/*DANGER_ZONE*/){
                //  workspace->realloc.hbonds = 1;

                if ( Hindex < numH - 1 )
                    comp = Start_Index(Hindex + 1, hbonds);
                else comp = hbonds->num_intrs;

                if ( End_Index(Hindex, hbonds) > comp )
                {
                    fprintf(stderr, "step%d-hbondchk failed: H=%d end(H)=%d str(H+1)=%d\n",
                            step, Hindex, End_Index(Hindex, hbonds), comp );
                    MPI_Abort( comm, INSUFFICIENT_MEMORY );
                }
            }
            /*
                        if ( Hindex > -1 )
                        {
                            system->my_atoms[i].num_hbonds =
                                MAX( Num_Entries(Hindex, hbonds) * SAFER_ZONE, MIN_HBONDS );
            */
            //if( Num_Entries(i, hbonds) >=
            //(Start_Index(i+1,hbonds)-Start_Index(i,hbonds))*0.90/*DANGER_ZONE*/){
            //  workspace->realloc.hbonds = 1;
            /*
                            //TODO
                            if ( Hindex < system->n - 1 )
                                comp = Start_Index(Hindex + 1, hbonds);
                            else comp = hbonds->num_intrs;

                            if ( End_Index(Hindex, hbonds) > comp )
                            {
                                fprintf(stderr, "step%d-hbondchk failed: H=%d end(H)=%d str(H+1)=%d\n",
                                        step, Hindex, End_Index(Hindex, hbonds), comp );
                                MPI_Abort( MPI_COMM_WORLD, INSUFFICIENT_MEMORY );
                            }
                        }

            */
        }
    }
}


#if defined(OLD_VALIDATE)
void Validate_Lists( storage *workspace, reax_list **lists,
                     int step, int n, int N, int numH )
{
    int i, flag;
    reax_list *bonds, *hbonds;
    reallocate_data *realloc;
    realloc = &(workspace->realloc);

    /* bond list */
    if ( N > 0 )
    {
        flag = -1;
        bonds = *lists + BONDS;
        for ( i = 0; i < N - 1; ++i )
            if ( End_Index(i, bonds) >= Start_Index(i + 1, bonds) - 2 )
            {
                workspace->realloc.bonds = 1;
                if ( End_Index(i, bonds) > Start_Index(i + 1, bonds) )
                    flag = i;
            }

        if ( flag > -1 )
        {
            fprintf( stderr, "step%d-bondchk failed: i=%d end(i)=%d str(i+1)=%d\n",
                     step, flag, End_Index(flag, bonds), Start_Index(flag + 1, bonds) );
            MPI_Abort( MPI_COMM_WORLD, INSUFFICIENT_MEMORY );
        }

        if ( End_Index(i, bonds) >= bonds->num_intrs - 2 )
        {
            workspace->realloc.bonds = 1;
            if ( End_Index(i, bonds) > bonds->num_intrs )
            {
                fprintf( stderr, "step%d-bondchk failed: i=%d end(i)=%d bond_end=%d\n",
                         step, flag, End_Index(i, bonds), bonds->num_intrs );
                MPI_Abort( MPI_COMM_WORLD, INSUFFICIENT_MEMORY );
            }
        }
    }


    /* hbonds list */
    if ( numH > 0 )
    {
        flag = -1;
        hbonds = *lists + HBONDS;
        for ( i = 0; i < numH - 1; ++i )
        {
            if ( Num_Entries(i, hbonds) >=
                    (Start_Index(i + 1, hbonds) - Start_Index(i, hbonds)) * 0.90/*DANGER_ZONE*/ )
            {
                workspace->realloc.hbonds = 1;
                if ( End_Index(i, hbonds) > Start_Index(i + 1, hbonds) )
                {
                    flag = i;
                }
            }
        }

        if ( flag > -1 )
        {
            fprintf( stderr, "step%d-hbondchk failed: i=%d end(i)=%d str(i+1)=%d\n",
                     step, flag, End_Index(flag, hbonds), Start_Index(flag + 1, hbonds) );
            MPI_Abort( MPI_COMM_WORLD, INSUFFICIENT_MEMORY );
        }

        if ( Num_Entries(i, hbonds) >=
                (hbonds->num_intrs - Start_Index(i, hbonds)) * 0.90/*DANGER_ZONE*/ )
        {
            workspace->realloc.hbonds = 1;
            if ( End_Index(i, hbonds) > hbonds->num_intrs )
            {
                fprintf( stderr, "step%d-hbondchk failed: i=%d end(i)=%d hbondend=%d\n",
                         step, flag, End_Index(i, hbonds), hbonds->num_intrs );
                MPI_Abort( MPI_COMM_WORLD, INSUFFICIENT_MEMORY );
            }
        }
    }
}
#endif


static inline real Compute_H( real r, real gamma, real *ctap )
{
    real taper, dr3gamij_1, dr3gamij_3;

    taper = ctap[7] * r + ctap[6];
    taper = taper * r + ctap[5];
    taper = taper * r + ctap[4];
    taper = taper * r + ctap[3];
    taper = taper * r + ctap[2];
    taper = taper * r + ctap[1];
    taper = taper * r + ctap[0];

    dr3gamij_1 = ( r * r * r + gamma );
    dr3gamij_3 = POW( dr3gamij_1 , 1.0 / 3.0 );

    return taper * EV_to_KCALpMOL / dr3gamij_3;
}


static inline real Compute_tabH( real r_ij, int ti, int tj, int num_atom_types )
{
    int r, tmin, tmax;
    real val, dif, base;
    LR_lookup_table *t;

    tmin  = MIN( ti, tj );
    tmax  = MAX( ti, tj );
    //SUDHIR
    //t = &( LR[tmin][tmax] );
    t = &( LR[index_lr (tmin, tmax, num_atom_types)] );

    /* cubic spline interpolation */
    r = (int)(r_ij * t->inv_dx);
    if ( r == 0 )
    {
        ++r;
    }
    base = (real)(r + 1) * t->dx;
    dif = r_ij - base;
    val = ((t->ele[r].d * dif + t->ele[r].c) * dif + t->ele[r].b) * dif +
          t->ele[r].a;
    val *= EV_to_KCALpMOL / C_ele;

    return val;
}


int Init_Forces( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int Htop, btop_i, btop_j, num_bonds, num_hbonds;
    int ihb, jhb, ihb_top, jhb_top;
    int local, flag, renbr;
    real r_ij, cutoff;
    sparse_matrix *H;
    reax_list *far_nbrs, *bonds, *hbonds;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;

    far_nbrs = *lists + FAR_NBRS;
    bonds = *lists + BONDS;
    hbonds = *lists + HBONDS;

    for ( i = 0; i < system->n; ++i )
    {
        workspace->bond_mark[i] = 0;
    }
    for ( i = system->n; i < system->N; ++i )
    {
        /* put ghost atoms to an infinite distance */
        workspace->bond_mark[i] = 1000;
        //workspace->done_after[i] = Start_Index( i, far_nbrs );
    }

    H = &(workspace->H); //MATRIX CHANGES
    H->n = system->n;
    Htop = 0;
    num_bonds = 0;
    num_hbonds = 0;
    btop_i = btop_j = 0;
    renbr = (data->step - data->prev_steps) % control->reneighbor == 0;

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &(system->my_atoms[i]);
        type_i  = atom_i->type;
        start_i = Start_Index(i, far_nbrs);
        end_i   = End_Index(i, far_nbrs);
        btop_i = End_Index( i, bonds );
        sbp_i = &(system->reax_param.sbp[type_i]);

        if ( i < system->n )
        {
            local = 1;
            cutoff = control->nonb_cut;
        }
        else
        {
            local = 0;
            cutoff = control->bond_cut;
        }

        ihb = -1;
        ihb_top = -1;
        if ( local )
        {
            H->start[i] = Htop;
            H->entries[Htop].j = i;
            H->entries[Htop].val = sbp_i->eta;
            ++Htop;

            if ( control->hbond_cut > 0.0 )
            {
                ihb = sbp_i->p_hbond;
                if ( ihb == 1 )
                {
                    ihb_top = End_Index( atom_i->Hindex, hbonds );
                }
                else
                {
                    ihb_top = -1;
                }
            }
        }

        /* update i-j distance - check if j is within cutoff */
        for ( pj = start_i; pj < end_i; ++pj )
        {
            nbr_pj = &( far_nbrs->select.far_nbr_list[pj] );
            j = nbr_pj->nbr;
            atom_j = &(system->my_atoms[j]);

            //fprintf( stderr, "%d%d i=%d x_i: %f %f %f,j=%d x_j: %f %f %f, d=%f\n",
            //     MIN(atom_i->orig_id, atom_j->orig_id),
            //     MAX(atom_i->orig_id, atom_j->orig_id),
            //     i, atom_i->x[0], atom_i->x[1], atom_i->x[2],
            //     j, atom_j->x[0], atom_j->x[1], atom_j->x[2], nbr_pj->d );

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
            }
            else
            {
                nbr_pj->dvec[0] = atom_j->x[0] - atom_i->x[0];
                nbr_pj->dvec[1] = atom_j->x[1] - atom_i->x[1];
                nbr_pj->dvec[2] = atom_j->x[2] - atom_i->x[2];
                nbr_pj->d = rvec_Norm_Sqr( nbr_pj->dvec );
                if ( nbr_pj->d <= SQR(cutoff) )
                {
                    nbr_pj->d = SQRT( nbr_pj->d );
                    flag = TRUE;
                }
                else
                {
                    flag = FALSE;
                }
            }

            if ( flag == TRUE )
            {
                type_j = atom_j->type;
                r_ij = nbr_pj->d;
                sbp_j = &(system->reax_param.sbp[type_j]);
                twbp = &(system->reax_param.tbp[ index_tbp(type_i, type_j, system->reax_param.num_atom_types)]);

                if ( local )
                {
                    /* H matrix entry */
                    if ( j < system->n || atom_i->orig_id < atom_j->orig_id ) //tryQEq||1
                    {
                        H->entries[Htop].j = j;

                        //fprintf( stdout, "%d%d %d %d\n",
                        //     MIN(atom_i->orig_id, atom_j->orig_id),
                        //     MAX(atom_i->orig_id, atom_j->orig_id),
                        //     MIN(atom_i->orig_id, atom_j->orig_id),
                        //     MAX(atom_i->orig_id, atom_j->orig_id) );

                        if ( control->tabulate == 0 )
                        {
                            H->entries[Htop].val = Compute_H(r_ij, twbp->gamma, workspace->Tap);
                        }
                        else
                        {
                            H->entries[Htop].val = Compute_tabH(r_ij, type_i, type_j, system->reax_param.num_atom_types);
                        }
                        ++Htop;
                    }

                    /* hydrogen bond lists */
                    if ( control->hbond_cut > 0 && (ihb == 1 || ihb == 2) &&
                            nbr_pj->d <= control->hbond_cut )
                    {
                        // fprintf( stderr, "%d %d\n", atom1, atom2 );

                        jhb = sbp_j->p_hbond;
                        if ( ihb == 1 && jhb == 2 )
                        {
                            hbonds->select.hbond_list[ihb_top].nbr = j;
                            hbonds->select.hbond_list[ihb_top].scl = 1;
                            hbonds->select.hbond_list[ihb_top].ptr = nbr_pj;
                            ++ihb_top;
                            ++num_hbonds;
                        }
                        else if ( j < system->n && ihb == 2 && jhb == 1 )
                        {
                            jhb_top = End_Index( atom_j->Hindex, hbonds );
                            hbonds->select.hbond_list[jhb_top].nbr = i;
                            hbonds->select.hbond_list[jhb_top].scl = -1;
                            hbonds->select.hbond_list[jhb_top].ptr = nbr_pj;
                            Set_End_Index( atom_j->Hindex, jhb_top + 1, hbonds );
                            ++num_hbonds;
                        }
                    }
                }

                /* uncorrected bond orders */
                if ( //(workspace->bond_mark[i] < 3 || workspace->bond_mark[j] < 3) &&
                    nbr_pj->d <= control->bond_cut &&
                    BOp( workspace, bonds, control->bo_cut,
                        i, btop_i, nbr_pj, sbp_i, sbp_j, twbp ) == TRUE )
                {
                    num_bonds += 2;
                    ++btop_i;

                    if ( workspace->bond_mark[j] > workspace->bond_mark[i] + 1 )
                    {
                        workspace->bond_mark[j] = workspace->bond_mark[i] + 1;
                    }
                    else if ( workspace->bond_mark[i] > workspace->bond_mark[j] + 1 )
                    {
                        workspace->bond_mark[i] = workspace->bond_mark[j] + 1;
                        //if( workspace->bond_mark[i] == 1000 )
                        //  workspace->done_after[i] = pj;
                    }

                    //fprintf( stdout, "%d%d - %d(%d) %d(%d)\n",
                    //   i , j, i, workspace->bond_mark[i], j, workspace->bond_mark[j] );
                }
            }
        }

//        H->end[i] = Htop;

        Set_End_Index( i, btop_i, bonds );
        if ( local )
        {
            //printf("Htop: %d \n", Htop);

            H->end[i] = Htop;
            if ( ihb == 1 )
            {
                Set_End_Index( atom_i->Hindex, ihb_top, hbonds );
            }
        }
    }

    //fprintf( stderr, "after the first init loop\n" );
    /*for( i = system->n; i < system->N; ++i )
      if( workspace->bond_mark[i] > 3 ) {
        start_i = Start_Index(i, bonds);
        end_i = End_Index(i, bonds);
        num_bonds -= (end_i - start_i);
        Set_End_Index(i, start_i, bonds );
        }*/

    /*for( i = system->n; i < system->N; ++i ) {
      start_i = Start_Index(i, far_nbrs);
      end_i = workspace->done_after[i];

      if( workspace->bond_mark[i] >= 2 && start_i < end_i ) {
        atom_i = &(system->my_atoms[i]);
        type_i = atom_i->type;
        btop_i = End_Index( i, bonds );
        sbp_i = &(system->reax_param.sbp[type_i]);

        for( pj = start_i; pj < end_i; ++pj ) {
    nbr_pj = &( far_nbrs->select.far_nbr_list[pj] );
    j = nbr_pj->nbr;

    if( workspace->bond_mark[j] >= 2 && nbr_pj->d <= control->bond_cut ) {
      atom_j = &(system->my_atoms[j]);
      type_j = atom_j->type;
      sbp_j = &(system->reax_param.sbp[type_j]);
      twbp = &(system->reax_param.tbp[type_i][type_j]);

      if( BOp( workspace, bonds, control->bo_cut,
         i , btop_i, nbr_pj, sbp_i, sbp_j, twbp ) ) {
        num_bonds += 2;
        ++btop_i;

        if( workspace->bond_mark[j] > workspace->bond_mark[i] + 1 )
          workspace->bond_mark[j] = workspace->bond_mark[i] + 1;
        else if( workspace->bond_mark[i] > workspace->bond_mark[j] + 1 )
          workspace->bond_mark[i] = workspace->bond_mark[j] + 1;

        //fprintf( stdout, "%d%d - %d(%d) %d(%d) new\n",
        // i , j, i, workspace->bond_mark[i], j, workspace->bond_mark[j] );
      }
    }
        }
        Set_End_Index( i, btop_i, bonds );
      }
      }*/

    workspace->realloc.Htop = Htop;
    workspace->realloc.num_bonds = num_bonds;
    workspace->realloc.num_hbonds = num_hbonds;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d: Htop = %d num_bonds = %d num_hbonds = %d\n",
             system->my_rank, data->step, Htop, num_bonds, num_hbonds );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

//    return Validate_Lists( system, workspace, lists, control,
//            data->step, system->n, system->N, system->numH );

    return MPI_Not_GPU_Validate_Lists( system, workspace, lists, control,
            data->step, system->n, system->N, system->numH );
}


int Init_Forces_No_Charges( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int btop_i, btop_j, num_bonds, num_hbonds;
    int ihb, jhb, ihb_top, jhb_top;
    int local, flag, renbr;
    real r_ij, cutoff;
    reax_list *far_nbrs, *bonds, *hbonds;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;

    far_nbrs = *lists + FAR_NBRS;
    bonds = *lists + BONDS;
    hbonds = *lists + HBONDS;

    for ( i = 0; i < system->n; ++i )
    {
        workspace->bond_mark[i] = 0;
    }
    for ( i = system->n; i < system->N; ++i )
    {
        workspace->bond_mark[i] = 1000; // put ghost atoms to an infinite distance
        //workspace->done_after[i] = Start_Index( i, far_nbrs );
    }

    num_bonds = 0;
    num_hbonds = 0;
    btop_i = btop_j = 0;
    renbr = (data->step - data->prev_steps) % control->reneighbor == 0;

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &(system->my_atoms[i]);
        type_i  = atom_i->type;
        start_i = Start_Index(i, far_nbrs);
        end_i   = End_Index(i, far_nbrs);
        btop_i = End_Index( i, bonds );
        sbp_i = &(system->reax_param.sbp[type_i]);

        if ( i < system->n )
        {
            local = 1;
            cutoff = MAX( control->hbond_cut, control->bond_cut );
        }
        else
        {
            local = 0;
            cutoff = control->bond_cut;
        }

        ihb = -1;
        ihb_top = -1;
        if ( local && control->hbond_cut > 0 )
        {
            ihb = sbp_i->p_hbond;
            if ( ihb == 1 )
            {
                ihb_top = End_Index( atom_i->Hindex, hbonds );
            }
            else
            {
                ihb_top = -1;
            }
        }

        /* update i-j distance - check if j is within cutoff */
        for ( pj = start_i; pj < end_i; ++pj )
        {
            nbr_pj = &( far_nbrs->select.far_nbr_list[pj] );
            j = nbr_pj->nbr;
            atom_j = &(system->my_atoms[j]);

            if ( renbr )
            {
                if ( nbr_pj->d <= cutoff )
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
                nbr_pj->dvec[0] = atom_j->x[0] - atom_i->x[0];
                nbr_pj->dvec[1] = atom_j->x[1] - atom_i->x[1];
                nbr_pj->dvec[2] = atom_j->x[2] - atom_i->x[2];
                nbr_pj->d = rvec_Norm_Sqr( nbr_pj->dvec );
                if ( nbr_pj->d <= SQR(cutoff) )
                {
                    nbr_pj->d = sqrt(nbr_pj->d);
                    flag = 1;
                }
                else
                {
                    flag = 0;
                }
            }

            if ( flag )
            {
                type_j = atom_j->type;
                r_ij = nbr_pj->d;
                sbp_j = &(system->reax_param.sbp[type_j]);
                //SUDHIR
                //twbp = &(system->reax_param.tbp[type_i][type_j]);
                twbp = &(system->reax_param.tbp[index_tbp(type_i, type_j, system->reax_param.num_atom_types)]);

                if ( local )
                {
                    /* hydrogen bond lists */
                    if ( control->hbond_cut > 0 && (ihb == 1 || ihb == 2) &&
                            nbr_pj->d <= control->hbond_cut )
                    {
                        // fprintf( stderr, "%d %d\n", atom1, atom2 );
                        jhb = sbp_j->p_hbond;
                        if ( ihb == 1 && jhb == 2 )
                        {
                            hbonds->select.hbond_list[ihb_top].nbr = j;
                            hbonds->select.hbond_list[ihb_top].scl = 1;
                            hbonds->select.hbond_list[ihb_top].ptr = nbr_pj;
                            ++ihb_top;
                            ++num_hbonds;
                        }
                        else if ( j < system->n && ihb == 2 && jhb == 1 )
                        {
                            jhb_top = End_Index( atom_j->Hindex, hbonds );
                            hbonds->select.hbond_list[jhb_top].nbr = i;
                            hbonds->select.hbond_list[jhb_top].scl = -1;
                            hbonds->select.hbond_list[jhb_top].ptr = nbr_pj;
                            Set_End_Index( atom_j->Hindex, jhb_top + 1, hbonds );
                            ++num_hbonds;
                        }
                    }
                }


                /* uncorrected bond orders */
                if ( //(workspace->bond_mark[i] < 3 || workspace->bond_mark[j] < 3) &&
                    nbr_pj->d <= control->bond_cut &&
                    BOp( workspace, bonds, control->bo_cut,
                         i , btop_i, nbr_pj, sbp_i, sbp_j, twbp ) )
                {
                    num_bonds += 2;
                    ++btop_i;

                    if ( workspace->bond_mark[j] > workspace->bond_mark[i] + 1 )
                        workspace->bond_mark[j] = workspace->bond_mark[i] + 1;
                    else if ( workspace->bond_mark[i] > workspace->bond_mark[j] + 1 )
                    {
                        workspace->bond_mark[i] = workspace->bond_mark[j] + 1;
                        //if( workspace->bond_mark[i] == 1000 )
                        //  workspace->done_after[i] = pj;
                    }
                    //fprintf( stdout, "%d%d - %d(%d) %d(%d)\n",
                    //   i , j, i, workspace->bond_mark[i], j, workspace->bond_mark[j] );
                }
            }
        }

        Set_End_Index( i, btop_i, bonds );
        if ( local && ihb == 1 )
        {
            Set_End_Index( atom_i->Hindex, ihb_top, hbonds );
        }
    }

    for ( i = system->n; i < system->N; ++i )
    {
        if ( workspace->bond_mark[i] > 3 )
        {
            start_i = Start_Index(i, bonds);
            end_i = End_Index(i, bonds);
            num_bonds -= (end_i - start_i);
            Set_End_Index(i, start_i, bonds );
        }
    }

    workspace->realloc.num_bonds = num_bonds;
    workspace->realloc.num_hbonds = num_hbonds;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d: num_bonds = %d num_hbonds = %d\n",
             system->my_rank, data->step, num_bonds, num_hbonds );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

#if defined( DEBUG )
    Print_Bonds( system, bonds, "debugbonds.out" );
    Print_Bond_List2( system, bonds, "pbonds.out" );
#endif

    return MPI_Not_GPU_Validate_Lists( system, workspace, lists, control,
            data->step, system->n, system->N, system->numH );
}


void Host_Estimate_Sparse_Matrix( reax_atom *my_atoms, control_params *control,
        reax_list p_far_nbrs, int n, int N, int renbr, int *indices )
{
    int i, j, pj;
    int start_i, end_i;
    int flag;
    real cutoff;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;
    reax_list *far_nbrs = &( p_far_nbrs );

    //i = blockIdx.x * blockDim.x + threadIdx.x;
    //if (i >= N) return;
    for (i = 0; i < N; i++)
    {
        atom_i = &(my_atoms[i]);
        start_i = Start_Index(i, far_nbrs);
        end_i   = End_Index(i, far_nbrs);

        cutoff = control->nonb_cut;

        //++Htop;
        if ( i < n)
            indices [i] ++;

        /* update i-j distance - check if j is within cutoff */
        for ( pj = start_i; pj < end_i; ++pj )
        {
            nbr_pj = &( far_nbrs->select.far_nbr_list[pj] );
            j = nbr_pj->nbr;
            atom_j = &(my_atoms[j]);
            if ( renbr )
            {
                if (nbr_pj->d <= cutoff)
                    flag = 1;
                else flag = 0;
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
                if ( nbr_pj->d <= SQR(cutoff) )
                {
                    nbr_pj->d = sqrt(nbr_pj->d);
                    flag = 1;
                }
                else
                {
                    flag = 0;
                }
            }

            if ( flag )
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
                    indices [i]++;
                }
                else if (i > j && i >= n && j < n && atom_j->orig_id < atom_i->orig_id)
                {
                    indices [i] ++;
                }
                else if (i > j && i < n && ( j < n || atom_j->orig_id < atom_i->orig_id ))
                {
                    indices [i] ++;
                }
            }
        }
    }
}


#ifdef HAVE_CUDA
void Estimate_Storages( reax_system *system, control_params *control,
        reax_list **lists, int *Htop, int *hb_top, int *bond_top, int *num_3body )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int ihb, jhb;
    int local;
    int hbond_count, bond_count;
    real cutoff;
    real r_ij, r2;
    real C12, C34, C56;
    real BO, BO_s, BO_pi, BO_pi2;
    reax_list *far_nbrs;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;

    far_nbrs = *lists + FAR_NBRS;
    *Htop = 0;
    //printf("Hcap: %d \n", system->Hcap);
    memset( hb_top, 0, sizeof(int) * system->Hcap );
    memset( bond_top, 0, sizeof(int) * system->total_cap );
    *num_3body = 0;

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &(system->my_atoms[i]);
        type_i  = atom_i->type;
        start_i = Start_Index(i, far_nbrs);
        end_i   = End_Index(i, far_nbrs);
        sbp_i = &(system->reax_param.sbp[type_i]);

        if ( i < system->n )
        {
            local = 1;
            cutoff = control->nonb_cut;
            ++(*Htop);
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
            nbr_pj = &( far_nbrs->select.far_nbr_list[pj] );
            j = nbr_pj->nbr;
            atom_j = &(system->my_atoms[j]);

            if (nbr_pj->d <= cutoff)
            {
                type_j = system->my_atoms[j].type;
                r_ij = nbr_pj->d;
                sbp_j = &(system->reax_param.sbp[type_j]);
                //SUDHIR
                //twbp = &(system->reax_param.tbp[type_i][type_j]);
                twbp = &(system->reax_param.tbp[index_tbp (type_i, type_j, system->reax_param.num_atom_types)]);

                if ( local )
                {
                    if ( j < system->n || atom_i->orig_id < atom_j->orig_id ) //tryQEq ||1
                        ++(*Htop);


                    if ( control->hbond_cut > 0.1 && (ihb == 1 || ihb == 2) &&
                            nbr_pj->d <= control->hbond_cut )
                    {
                        jhb = sbp_j->p_hbond;
                        if ( ihb == 1 && jhb == 2 )
                            ++hb_top[i];
                        else if ( j < system->n && ihb == 2 && jhb == 1 )
                        {
                            ++hb_top[j];

                        }
                    }
                }

                // uncorrected bond orders
                if ( nbr_pj->d <= control->bond_cut )
                {
                    r2 = SQR(r_ij);

                    if ( sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0)
                    {
                        C12 = twbp->p_bo1 * POW( r_ij / twbp->r_s, twbp->p_bo2 );
                        BO_s = (1.0 + control->bo_cut) * EXP( C12 );
                    }
                    else BO_s = C12 = 0.0;

                    if ( sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0)
                    {
                        C34 = twbp->p_bo3 * POW( r_ij / twbp->r_p, twbp->p_bo4 );
                        BO_pi = EXP( C34 );
                    }
                    else BO_pi = C34 = 0.0;

                    if ( sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0)
                    {
                        C56 = twbp->p_bo5 * POW( r_ij / twbp->r_pp, twbp->p_bo6 );
                        BO_pi2 = EXP( C56 );
                    }
                    else BO_pi2 = C56 = 0.0;

                    // Initially BO values are the uncorrected ones, page 1
                    BO = BO_s + BO_pi + BO_pi2;

                    if ( BO >= control->bo_cut )
                    {
                        ++bond_top[i];
                        ++bond_top[j];
                    }
                }
            }
        }
    }

    fprintf( stderr, "HOST SPARSE MATRIX ENTRIES: %d \n",  *Htop );
    *Htop = MAX( *Htop * SAFE_ZONE, MIN_CAP * MIN_HENTRIES );

    hbond_count = 0;
    for ( i = 0; i < system->n; ++i )
    {
        hbond_count += hb_top[i];
        hb_top[i] = MAX( hb_top[i] * SAFER_ZONE, MIN_HBONDS );
    }
    fprintf( stderr, "HOST HBOND COUNT: %d \n", hbond_count );

    bond_count = 0;
    for ( i = 0; i < system->N; ++i )
    {
        bond_count += bond_top[i];
        *num_3body += SQR(bond_top[i]);
        //if( i < system->n )
        bond_top[i] = MAX( bond_top[i] * 2, MIN_BONDS );
        //else bond_top[i] = MAX_BONDS;
    }
    fprintf (stderr, "HOST BOND COUNT: %d \n", bond_count);

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ estimate storages: Htop = %d, num_3body = %d\n",
             system->my_rank, *Htop, *num_3body );
    MPI_Barrier( MPI_COMM_WORLD );
#endif
}


#else
void Estimate_Storages( reax_system *system, control_params *control,
        reax_list **lists, int *Htop, int *hb_top,
        int *bond_top, int *num_3body)
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
    reax_list *far_nbrs;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;

    far_nbrs = *lists + FAR_NBRS;
    *Htop = 0;
    memset( hb_top, 0, sizeof(int) * system->local_cap );
    memset( bond_top, 0, sizeof(int) * system->total_cap );
    *num_3body = 0;

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &(system->my_atoms[i]);
        type_i  = atom_i->type;
        start_i = Start_Index(i, far_nbrs);
        end_i   = End_Index(i, far_nbrs);
        sbp_i = &(system->reax_param.sbp[type_i]);

        if ( i < system->n )
        {
            local = 1;
            cutoff = control->nonb_cut;
            ++(*Htop);
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
            //nbr_pj = &( far_nbrs->far_nbr_list[pj] );
            nbr_pj = &( far_nbrs->select.far_nbr_list[pj]);
            j = nbr_pj->nbr;
            atom_j = &(system->my_atoms[j]);

            if (nbr_pj->d <= cutoff)
            {
                type_j = system->my_atoms[j].type;
                r_ij = nbr_pj->d;
                sbp_j = &(system->reax_param.sbp[type_j]);
                //twbp = &(system->reax_param.tbp[type_i][type_j]);
                twbp = &(system->reax_param.tbp[index_tbp (type_i, type_j, system->reax_param.num_atom_types)]);

                if ( local )
                {
                    if ( j < system->n || atom_i->orig_id < atom_j->orig_id ) //tryQEq ||1
                        ++(*Htop);

                    /* hydrogen bond lists */
                    if ( control->hbond_cut > 0.1 && (ihb == 1 || ihb == 2) &&
                            nbr_pj->d <= control->hbond_cut )
                    {
                        jhb = sbp_j->p_hbond;
                        if ( ihb == 1 && jhb == 2 )
                            ++hb_top[i];
                        else if ( j < system->n && ihb == 2 && jhb == 1 )
                            ++hb_top[j];
                    }
                }

                /* uncorrected bond orders */
                if ( nbr_pj->d <= control->bond_cut )
                {
                    r2 = SQR(r_ij);

                    if ( sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0)
                    {
                        C12 = twbp->p_bo1 * pow( r_ij / twbp->r_s, twbp->p_bo2 );
                        BO_s = (1.0 + control->bo_cut) * exp( C12 );
                    }
                    else BO_s = C12 = 0.0;

                    if ( sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0)
                    {
                        C34 = twbp->p_bo3 * pow( r_ij / twbp->r_p, twbp->p_bo4 );
                        BO_pi = exp( C34 );
                    }
                    else BO_pi = C34 = 0.0;

                    if ( sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0)
                    {
                        C56 = twbp->p_bo5 * pow( r_ij / twbp->r_pp, twbp->p_bo6 );
                        BO_pi2 = exp( C56 );
                    }
                    else BO_pi2 = C56 = 0.0;

                    /* Initially BO values are the uncorrected ones, page 1 */
                    BO = BO_s + BO_pi + BO_pi2;

                    if ( BO >= control->bo_cut )
                    {
                        ++bond_top[i];
                        ++bond_top[j];
                    }
                }
            }
        }
    }

    *Htop = (int)(MAX( *Htop * SAFE_ZONE, MIN_CAP * MIN_HENTRIES ));

    // Set max sparse entries, needed for first iteration of validate_list
    system->max_sparse_entries = *Htop * SAFE_ZONE;

    for ( i = 0; i < system->n; ++i )
    {
        hb_top[i] = (int)(MAX( hb_top[i] * SAFER_ZONE, MIN_HBONDS ));
    }

    for ( i = 0; i < system->N; ++i )
    {
        *num_3body += SQR(bond_top[i]);
        //if( i < system->n )
        bond_top[i] = MAX( bond_top[i] * 2, MIN_BONDS );
        //else bond_top[i] = MAX_BONDS;
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ estimate storages: Htop = %d, num_3body = %d\n",
            system->my_rank, *Htop, *num_3body );
    MPI_Barrier( MPI_COMM_WORLD );
#endif
}
#endif


int Compute_Forces( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    int charge_flag, ret;
#if defined(LOG_PERFORMANCE)
    real t_start = 0;

    //MPI_Barrier( MPI_COMM_WORLD );
    if ( system->my_rank == MASTER_NODE )
    {
        t_start = Get_Time( );
    }
#endif

    /********* init forces ************/
    if ( control->charge_freq && (data->step - data->prev_steps) % control->charge_freq == 0 )
    {
        charge_flag = TRUE;
    }
    else
    {
        charge_flag = FALSE;
    }

    if ( charge_flag == TRUE )
    {
        ret = Init_Forces( system, control, data, workspace, lists, out_control );
    }
    else
    {
        ret = Init_Forces_No_Charges( system, control, data, workspace, lists, out_control );
    }

#if defined(LOG_PERFORMANCE)
    //MPI_Barrier( MPI_COMM_WORLD );
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &(data->timing.init_forces) );
    }
#endif

    if ( ret == SUCCESS )
    {
        /********* bonded interactions ************/
        Compute_Bonded_Forces( system, control, data, workspace, lists, out_control );

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

    /**************** charges ************************/
#if defined(PURE_REAX)
        if ( charge_flag == TRUE )
        {
            QEq( system, control, data, workspace, out_control, mpi_data );
        }

#if defined(LOG_PERFORMANCE)
        //MPI_Barrier( MPI_COMM_WORLD );
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &(data->timing.qEq) );
        }
#endif
#if defined(DEBUG_FOCUS)
        fprintf(stderr, "p%d @ step%d: qeq completed\n", system->my_rank, data->step);
        MPI_Barrier( MPI_COMM_WORLD );
#endif
#endif //PURE_REAX
    
        /********* nonbonded interactions ************/
        Compute_NonBonded_Forces( system, control, data, workspace,
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
        Compute_Total_Force( system, control, data, workspace, lists, mpi_data );
    
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
        //Print_Total_Force( system, data, workspace );
        MPI_Barrier( MPI_COMM_WORLD );
#endif

#if defined(TEST_FORCES)
        Print_Force_Files( system, control, data, workspace, lists, out_control, mpi_data );
#endif
    }

    return ret;
}


#ifdef HAVE_CUDA
int Cuda_Compute_Forces( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int charge_flag, retVal;

#if defined(LOG_PERFORMANCE)
    real t_start = 0;

    //MPI_Barrier( MPI_COMM_WORLD );
    if ( system->my_rank == MASTER_NODE )
    {
        t_start = Get_Time( );
    }
#endif

    /********* init forces ************/
    if ( control->charge_freq && (data->step - data->prev_steps) % control->charge_freq == 0 )
    {
        charge_flag = TRUE;
    }
    else
    {
        charge_flag = FALSE;
    }

    if ( charge_flag == TRUE )
    {
        retVal = Cuda_Init_Forces( system, control, data, workspace, lists, out_control );
        fprintf( stderr, "    [CUDA_INIT_FORCES: %d] STEP %d\n", retVal, data->step );
    }
    else
    {
        retVal = Cuda_Init_Forces_No_Charges( system, control, data, workspace, lists, out_control );
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
        Cuda_Compute_Bonded_Forces( system, control, data, workspace, lists, out_control );
        fprintf( stderr, "    [CUDA_COMPUTE_BONDED_FORCES: %d] STEP %d\n", retVal, data->step );

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

    /**************** charges ************************/
#if defined(PURE_REAX)
        if ( charge_flag == TRUE )
        {
            Cuda_QEq( system, control, data, workspace, out_control, mpi_data );
            fprintf( stderr, "    [CUDA_QEQ: %d] STEP %d\n", retVal, data->step );
        }

#if defined(LOG_PERFORMANCE)
        //MPI_Barrier( MPI_COMM_WORLD );
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &(data->timing.qEq) );
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
        fprintf( stderr, "    [CUDA_COMPUTE_NONBONDED_FORCES: %d] STEP %d\n", retVal, data->step );

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
        fprintf( stderr, "    [CUDA_COMPUTE_TOTAL_FORCE: %d] STEP %d\n", retVal, data->step );

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
        //Print_Total_Force( system, data, workspace );
        MPI_Barrier( MPI_COMM_WORLD );
#endif
    }

    return retVal;
}
#endif


int validate_device( reax_system *system, simulation_data *data,
        storage *workspace, reax_list **lists )
{
    int retval = FAILURE;

#if defined(__CUDA_DEBUG__)
    //retval |= validate_neighbors (system, lists);
    //retval |= validate_sym_dbond_indices (system, workspace, lists);
    //retval |= validate_hbonds (system, workspace, lists);
    //retval |= validate_workspace (system, workspace);
    //retval |= validate_bonds (system, workspace, lists);
    //retval |= validate_three_bodies (system, workspace, lists );
    retval |= validate_sparse_matrix (system, workspace);
    //retval |= validate_data (system, data);
    //retval |= validate_atoms (system, lists);
    //analyze_hbonds (system, workspace, lists);

    if (!retval)
    {
        fprintf( stderr, "Result *DOES NOT* match between device and host\n" );
    }
#endif

    return retval;

}
