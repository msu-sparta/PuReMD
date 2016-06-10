
#include "cuda_reset_tools.h"
#include "cuda_utils.h"
#include "dev_list.h"

CUDA_GLOBAL void ker_reset_hbond_list (reax_atom *my_atoms, 
													reax_list hbonds, 
													int N)
{
	int Hindex = 0;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	Hindex = my_atoms[i].Hindex;
	if (Hindex > 1) {
		Dev_Set_End_Index ( Hindex, Dev_Start_Index (Hindex, &hbonds), &hbonds);
	}
}

CUDA_GLOBAL void ker_reset_bond_list (reax_atom *my_atoms, 
													reax_list bonds, 
													int N)
{
	int Hindex = 0;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	Dev_Set_End_Index ( i, Dev_Start_Index (i, &bonds), &bonds);
}

extern "C"
{

void Cuda_Reset_Workspace (reax_system *system, storage *workspace)
{
	cuda_memset ( dev_workspace->total_bond_order, 0, system->total_cap * sizeof (real), "total_bond_order");
	cuda_memset ( dev_workspace->dDeltap_self, 0, system->total_cap * sizeof (rvec), "dDeltap_self");
	cuda_memset ( dev_workspace->CdDelta, 0, system->total_cap * sizeof (real), "CdDelta");
	cuda_memset ( dev_workspace->f, 0, system->total_cap * sizeof (rvec), "f");
}

CUDA_GLOBAL void ker_reset_hindex (reax_atom *my_atoms, int N)
{
	int Hindex = 0;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	my_atoms[i].Hindex = i;
}

void Cuda_Reset_Atoms( reax_system* system, control_params *control )
{
  int i;
  reax_atom *atom;
  int blocks;

	/*
  if( control->hbond_cut > 0 ) 
  //TODO
    for( i = 0; i < system->N; ++i ) { 
      atom = &(system->my_atoms[i]);
      //if( system->reax_param.sbp[ atom->type ].p_hbond == 1 ) 
   atom->Hindex = system->numH++;
      //else atom->Hindex = -1; 
    }   
	 //TODO
	 */
////////////////////////////////
////////////////////////////////
////////////////////////////////
////////////////////////////////
// FIX - 3 - Commented out this line for Hydrogen Bond fix
// FIX - HBOND ISSUE
// FIX - HBOND ISSUE
// FIX - HBOND ISSUE
// COMMENTED OUT THIS LINE BELOW
  //system->numH = system->N;
// FIX - HBOND ISSUE
// FIX - HBOND ISSUE
// FIX - HBOND ISSUE
////////////////////////////////
////////////////////////////////
////////////////////////////////
////////////////////////////////
////////////////////////////////


		blocks = system->N / DEF_BLOCK_SIZE + 
					((system->N % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);
		ker_reset_hindex <<<blocks, DEF_BLOCK_SIZE>>>
			(system->d_my_atoms, system->N);
		cudaThreadSynchronize ();
		cudaCheckError ();

}

int Cuda_Reset_Neighbor_Lists( reax_system *system, control_params *control,
            storage *workspace, reax_list **lists )
{
	int i, total_bonds, Hindex, total_hbonds;
	reax_list *bonds, *hbonds;
	int blocks;

	if (system->N > 0) {
		bonds = *dev_lists + BONDS;
		total_bonds = 0;

		//cuda_memset (bonds->index, 0, sizeof (int) * system->total_cap, "bonds:index");
		//cuda_memset (bonds->end_index, 0, sizeof (int) * system->total_cap, "bonds:end_index");
		blocks = system->N / DEF_BLOCK_SIZE + 
					((system->N % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);
		ker_reset_bond_list <<<blocks, DEF_BLOCK_SIZE>>>
			(system->d_my_atoms, *(*dev_lists + BONDS), system->N);
		cudaThreadSynchronize ();
		cudaCheckError ();

		total_bonds = 0;// TODO compute the total bonds here.

    	/* is reallocation needed? */
	 	if( total_bonds >= bonds->num_intrs * DANGER_ZONE ) { 
			workspace->realloc.bonds = 1;
			if( total_bonds >= bonds->num_intrs ) { 
				fprintf(stderr, "p%d: not enough space for bonds! total=%d allocated=%d\n", 
									system->my_rank, total_bonds, bonds->num_intrs );
				return FAILURE;
			}   
		}   
	}

	//HBonds processing
	//FIX - 4 - Added additional check
  	if( (control->hbond_cut > 0) && (system->numH > 0)) { 
   	hbonds = (*dev_lists) + HBONDS;
		total_hbonds = 0;
			     
		/* reset start-end indexes */
		//TODO
		blocks = system->N / DEF_BLOCK_SIZE + 
					((system->N % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);
		ker_reset_hbond_list <<<blocks, DEF_BLOCK_SIZE>>>
			(system->d_my_atoms, *(*dev_lists + HBONDS), system->N);
		cudaThreadSynchronize ();
		cudaCheckError ();

		//TODO compute the total hbonds here
		total_hbonds = 0;
																	      
		/* is reallocation needed? */
		if( total_hbonds >= hbonds->num_intrs * 0.90/*DANGER_ZONE*/ ) { 
			workspace->realloc.hbonds = 1;
			if( total_hbonds >= hbonds->num_intrs ) {
				fprintf(stderr, "p%d: not enough space for hbonds! total=%d allocated=%d\n",
									system->my_rank, total_hbonds, hbonds->num_intrs );
				return FAILURE;
			}
		}
	}

	return SUCCESS;
}

}
