
#ifndef __DEV_SYSTEM_PROPS_H__
#define __DEV_SYSTEM_PROPS_H__

#include "reax_types.h"

#ifdef __cplusplus
extern "C"  {  
#endif

	void dev_compute_total_mass (reax_system *, real *);
	void dev_compute_kinetic_energy (reax_system *, simulation_data *, real *);
	void dev_compute_momentum (reax_system *, rvec, rvec, rvec );
	void dev_compute_inertial_tensor (reax_system *, real *, rvec my_xcm);

	void dev_sync_simulation_data (simulation_data *);
	//void dev_compute_kinetic_energy (reax_system *, simulation_data *, real *);

#ifdef __cplusplus
}
#endif

#endif
