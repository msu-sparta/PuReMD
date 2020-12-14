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
  #include "system_props.h"

  #include "comm_tools.h"
  #include "tool_box.h"
  #include "vector.h"
#elif defined(LAMMPS_REAX)
  #include "reax_system_props.h"

  #include "reax_comm_tools.h"
  #include "reax_tool_box.h"
  #include "reax_vector.h"
#endif

#if defined(HAVE_CUDA)
  #include "cuda/cuda_copy.h"
#endif


void Temperature_Control( control_params *control, simulation_data *data )
{
    real tmp;

    /* step-wise temperature control */
    if ( control->T_mode == 1 )
    {
        if ((data->step - data->prev_steps) % ((int)(control->T_freq / control->dt)) == 0)
        {
            if ( FABS( control->T - control->T_final ) >= FABS( control->T_rate ) )
            {
                control->T += control->T_rate;
            }
            else
            {
                control->T = control->T_final;
            }
        }
    }
    /* constant slope control */
    else if ( control->T_mode == 2 )
    {
        tmp = control->T_rate * control->dt / control->T_freq;

        if ( FABS( control->T - control->T_final ) >= FABS( tmp ) )
        {
            control->T += tmp;
        }
    }
}


void Compute_Kinetic_Energy( reax_system* system, simulation_data* data,
        MPI_Comm comm )
{
    int i, ret;
    rvec p;
    real m;

    data->my_en.e_kin = 0.0;

    for ( i = 0; i < system->n; i++ )
    {
        m = system->reax_param.sbp[system->my_atoms[i].type].mass;

        rvec_Scale( p, m, system->my_atoms[i].v );
        data->my_en.e_kin += rvec_Dot( p, system->my_atoms[i].v );
    }
    data->my_en.e_kin *= 0.5;

    ret = MPI_Allreduce( &data->my_en.e_kin, &data->sys_en.e_kin, 1, MPI_DOUBLE,
            MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    data->therm.T = (2.0 * data->sys_en.e_kin) / (data->N_f * K_B);

    /* avoid T being an absolute zero, might cause F.P.E! */
    if ( FABS(data->therm.T) < ALMOST_ZERO )
    {
        data->therm.T = ALMOST_ZERO;
    }
}


void Compute_Total_Energy( reax_system *system, simulation_data *data,
        MPI_Comm comm )
{
    int ret;
    real my_en[14], sys_en[14];

    //TODO: remove this is an UGLY fix
    my_en[13] = data->my_en.e_kin;

#if defined(HAVE_CUDA)
    Cuda_Copy_Simulation_Data_Device_to_Host( data,
            (simulation_data *)data->d_simulation_data );
#endif

    my_en[0] = data->my_en.e_bond;
    my_en[1] = data->my_en.e_ov;
    my_en[2] = data->my_en.e_un;
    my_en[3] = data->my_en.e_lp;
    my_en[4] = data->my_en.e_ang;
    my_en[5] = data->my_en.e_pen;
    my_en[6] = data->my_en.e_coa;
    my_en[7] = data->my_en.e_hb;
    my_en[8] = data->my_en.e_tor;
    my_en[9] = data->my_en.e_con;
    my_en[10] = data->my_en.e_vdW;
    my_en[11] = data->my_en.e_ele;
    my_en[12] = data->my_en.e_pol;
//    my_en[13] = data->my_en.e_kin;

    ret = MPI_Reduce( my_en, sys_en, 14, MPI_DOUBLE, MPI_SUM, MASTER_NODE, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    data->my_en.e_pot = data->my_en.e_bond
        + data->my_en.e_ov + data->my_en.e_un  + data->my_en.e_lp
        + data->my_en.e_ang + data->my_en.e_pen + data->my_en.e_coa
        + data->my_en.e_hb + data->my_en.e_tor + data->my_en.e_con
        + data->my_en.e_vdW + data->my_en.e_ele + data->my_en.e_pol;

    data->my_en.e_tot = data->my_en.e_pot + E_CONV * data->my_en.e_kin;

    if ( system->my_rank == MASTER_NODE )
    {
        data->sys_en.e_bond = sys_en[0];
        data->sys_en.e_ov = sys_en[1];
        data->sys_en.e_un = sys_en[2];
        data->sys_en.e_lp = sys_en[3];
        data->sys_en.e_ang = sys_en[4];
        data->sys_en.e_pen = sys_en[5];
        data->sys_en.e_coa = sys_en[6];
        data->sys_en.e_hb = sys_en[7];
        data->sys_en.e_tor = sys_en[8];
        data->sys_en.e_con = sys_en[9];
        data->sys_en.e_vdW = sys_en[10];
        data->sys_en.e_ele = sys_en[11];
        data->sys_en.e_pol = sys_en[12];
        data->sys_en.e_kin = sys_en[13];

        data->sys_en.e_pot = data->sys_en.e_bond
            + data->sys_en.e_ov + data->sys_en.e_un  + data->sys_en.e_lp
            + data->sys_en.e_ang + data->sys_en.e_pen + data->sys_en.e_coa
            + data->sys_en.e_hb + data->sys_en.e_tor + data->sys_en.e_con
            + data->sys_en.e_vdW + data->sys_en.e_ele + data->sys_en.e_pol;

        data->sys_en.e_tot = data->sys_en.e_pot + E_CONV * data->sys_en.e_kin;
    }
}


void Compute_Total_Mass( reax_system *system, simulation_data *data,
        MPI_Comm comm  )
{
    int i, ret;
    real tmp;

    tmp = 0;
    for ( i = 0; i < system->n; i++ )
    {
        tmp += system->reax_param.sbp[ system->my_atoms[i].type ].mass;
    }

    ret = MPI_Allreduce( &tmp, &data->M, 1, MPI_DOUBLE, MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    data->inv_M = 1.0 / data->M;
}


void Compute_Center_of_Mass( reax_system *system, simulation_data *data,
        mpi_datatypes *mpi_data, MPI_Comm comm )
{
    int i, ret;
    real m, det; //xx, xy, xz, yy, yz, zz;
    real tmp_mat[6], tot_mat[6];
    rvec my_xcm, my_vcm, my_amcm, my_avcm;
    rvec tvec, diff;
    rtensor mat, inv;

    rvec_MakeZero( my_xcm );  // position of CoM
    rvec_MakeZero( my_vcm );  // velocity of CoM
    rvec_MakeZero( my_amcm ); // angular momentum of CoM
    rvec_MakeZero( my_avcm ); // angular velocity of CoM

    /* Compute the position, vel. and ang. momentum about the centre of mass */
    for ( i = 0; i < system->n; ++i )
    {
        m = system->reax_param.sbp[ system->my_atoms[i].type ].mass;

        rvec_ScaledAdd( my_xcm, m, system->my_atoms[i].x );
        rvec_ScaledAdd( my_vcm, m, system->my_atoms[i].v );

        rvec_Cross( tvec, system->my_atoms[i].x, system->my_atoms[i].v );
        rvec_ScaledAdd( my_amcm, m, tvec );
    }

    ret = MPI_Allreduce( my_xcm, data->xcm, 3, MPI_DOUBLE, MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Allreduce( my_vcm, data->vcm, 3, MPI_DOUBLE, MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Allreduce( my_amcm, data->amcm, 3, MPI_DOUBLE, MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    rvec_Scale( data->xcm, data->inv_M, data->xcm );
    rvec_Scale( data->vcm, data->inv_M, data->vcm );
    rvec_Cross( tvec, data->xcm, data->vcm );
    rvec_ScaledAdd( data->amcm, -data->M, tvec );
    data->etran_cm = 0.5 * data->M * rvec_Norm_Sqr( data->vcm );

    /* Calculate and then invert the inertial tensor */
    for ( i = 0; i < 6; ++i )
    {
        tmp_mat[i] = 0.0;
    }
    //my_xx = my_xy = my_xz = my_yy = my_yz = my_zz = 0;

    for ( i = 0; i < system->n; ++i )
    {
        m = system->reax_param.sbp[ system->my_atoms[i].type ].mass;
        rvec_ScaledSum( diff, 1.0, system->my_atoms[i].x, -1.0, data->xcm );

        tmp_mat[0]/*my_xx*/ += diff[0] * diff[0] * m;
        tmp_mat[1]/*my_xy*/ += diff[0] * diff[1] * m;
        tmp_mat[2]/*my_xz*/ += diff[0] * diff[2] * m;
        tmp_mat[3]/*my_yy*/ += diff[1] * diff[1] * m;
        tmp_mat[4]/*my_yz*/ += diff[1] * diff[2] * m;
        tmp_mat[5]/*my_zz*/ += diff[2] * diff[2] * m;
    }

    ret = MPI_Reduce( tmp_mat, tot_mat, 6, MPI_DOUBLE, MPI_SUM, MASTER_NODE, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    if ( system->my_rank == MASTER_NODE )
    {
        mat[0][0] = tot_mat[3] + tot_mat[5];  // yy + zz;
        mat[0][1] = mat[1][0] = -tot_mat[1];  // -xy;
        mat[0][2] = mat[2][0] = -tot_mat[2];  // -xz;
        mat[1][1] = tot_mat[0] + tot_mat[5];  // xx + zz;
        mat[2][1] = mat[1][2] = -tot_mat[4];  // -yz;
        mat[2][2] = tot_mat[0] + tot_mat[3];  // xx + yy;

        /* invert the inertial tensor */
        det = ( mat[0][0] * mat[1][1] * mat[2][2] +
                mat[0][1] * mat[1][2] * mat[2][0] +
                mat[0][2] * mat[1][0] * mat[2][1] ) -
              ( mat[0][0] * mat[1][2] * mat[2][1] +
                mat[0][1] * mat[1][0] * mat[2][2] +
                mat[0][2] * mat[1][1] * mat[2][0] );

        inv[0][0] = mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1];
        inv[0][1] = mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2];
        inv[0][2] = mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1];
        inv[1][0] = mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2];
        inv[1][1] = mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0];
        inv[1][2] = mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2];
        inv[2][0] = mat[1][0] * mat[2][1] - mat[2][0] * mat[1][1];
        inv[2][1] = mat[2][0] * mat[0][1] - mat[0][0] * mat[2][1];
        inv[2][2] = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];

        if ( det > ALMOST_ZERO )
        {
            rtensor_Scale( inv, 1.0 / det, inv );
        }
        else
        {
            rtensor_MakeZero( inv );
        }

        /* Compute the angular velocity about the centre of mass */
        rtensor_MatVec( data->avcm, inv, data->amcm );
    }

    ret = MPI_Bcast( data->avcm, 3, MPI_DOUBLE, MASTER_NODE, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    /* Compute the rotational energy */
    data->erot_cm = 0.5 * E_CONV * rvec_Dot( data->avcm, data->amcm );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "xcm:  %24.15e %24.15e %24.15e\n",
             data->xcm[0], data->xcm[1], data->xcm[2] );
    fprintf( stderr, "vcm:  %24.15e %24.15e %24.15e\n",
             data->vcm[0], data->vcm[1], data->vcm[2] );
    fprintf( stderr, "amcm: %24.15e %24.15e %24.15e\n",
             data->amcm[0], data->amcm[1], data->amcm[2] );
    fprintf( stderr, "mat:  %f %f %f\n     %f %f %f\n     %f %f %f\n",
       mat[0][0], mat[0][1], mat[0][2],
       mat[1][0], mat[1][1], mat[1][2],
       mat[2][0], mat[2][1], mat[2][2] );
    fprintf( stderr, "inv:  %g %g %g\n     %g %g %g\n     %g %g %g\n",
       inv[0][0], inv[0][1], inv[0][2],
       inv[1][0], inv[1][1], inv[1][2],
       inv[2][0], inv[2][1], inv[2][2] ); 
    fprintf( stderr, "avcm: %24.15e %24.15e %24.15e\n",
             data->avcm[0], data->avcm[1], data->avcm[2] );
#endif
}


void Check_Energy( simulation_data* data )
{
    if ( !isfinite(data->my_en.e_pol) )
    {
        fprintf( stderr, "[ERROR] NaN or infinite detected for polarization energy. Terminating...\n" );
        exit( NUMERIC_BREAKDOWN );
    }

    if ( !isfinite(data->my_en.e_pot) )
    {
        fprintf( stderr, "[ERROR] NaN or infinite detected for potential energy. Terminating...\n" );
        exit( NUMERIC_BREAKDOWN );
    }

    if ( !isfinite(data->my_en.e_tot) )
    {
        fprintf( stderr, "[ERROR] NaN or infinite detected for total energy. Terminating...\n" );
        exit( NUMERIC_BREAKDOWN );
    }
}


/* IMPORTANT: This function assumes that current kinetic energy
 * the system is already computed
 *
 * IMPORTANT: In Klein's paper, it is stated that a dU/dV term needs
 *  to be added when there are long-range interactions or long-range
 *  corrections to short-range interactions present.
 *  We may want to add that for more accuracy.
 */
void Compute_Pressure( reax_system* system, control_params *control,
        simulation_data* data, mpi_datatypes *mpi_data )
{
    int i, ret;
    reax_atom *p_atom;
    rvec tmp, tx, int_press;
    simulation_box *big_box;

    big_box = &system->big_box;

    /* Calculate internal pressure */
    rvec_MakeZero( int_press );

    // 0: both int and ext, 1: ext only, 2: int only
    if ( control->press_mode == 0 || control->press_mode == 2 )
    {
        for ( i = 0; i < system->n; ++i )
        {
            p_atom = &system->my_atoms[i];

            /* transform x into unitbox coordinates */
            Transform_to_UnitBox( p_atom->x, big_box, 1, tx );

            /* this atom's contribution to internal pressure */
            rvec_Multiply( tmp, p_atom->f, tx );
            rvec_Add( int_press, tmp );

#if defined(DEBUG_FOCUS)
            fprintf( stderr, "%8d%8.2f%8.2f%8.2f",
                     i + 1, p_atom->x[0], p_atom->x[1], p_atom->x[2] );
            fprintf( stderr, "%8.2f%8.2f%8.2f",
                     p_atom->f[0], p_atom->f[1], p_atom->f[2] );
            fprintf( stderr, "%8.2f%8.2f%8.2f\n",
                     int_press[0], int_press[1], int_press[2] );
#endif
        }
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d:p_int(%10.5f %10.5f %10.5f)p_ext(%10.5f %10.5f %10.5f)\n",
            system->my_rank, int_press[0], int_press[1], int_press[2],
            data->my_ext_press[0], data->my_ext_press[1], data->my_ext_press[2] );
#endif

    /* sum up internal and external pressure */
    ret = MPI_Allreduce( int_press, data->int_press, 3, MPI_DOUBLE,
            MPI_SUM, mpi_data->comm_mesh3D );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Allreduce( data->my_ext_press, data->ext_press, 3, MPI_DOUBLE,
            MPI_SUM, mpi_data->comm_mesh3D );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: %10.5f %10.5f %10.5f\n",
             system->my_rank,
             data->int_press[0], data->int_press[1], data->int_press[2] );
    fprintf( stderr, "p%d: %10.5f %10.5f %10.5f\n",
             system->my_rank,
             data->ext_press[0], data->ext_press[1], data->ext_press[2] );
#endif

    /* kinetic contribution */
    data->kin_press = 2.0 * (E_CONV * data->sys_en.e_kin)
        / (3.0 * big_box->V * P_CONV);

    /* Calculate total pressure in each direction */
    data->tot_press[0] = data->kin_press -
        (( data->int_press[0] + data->ext_press[0] ) /
         ( big_box->box_norms[1] * big_box->box_norms[2] * P_CONV ));

    data->tot_press[1] = data->kin_press -
        (( data->int_press[1] + data->ext_press[1] ) /
         ( big_box->box_norms[0] * big_box->box_norms[2] * P_CONV ));

    data->tot_press[2] = data->kin_press -
        (( data->int_press[2] + data->ext_press[2] ) /
         ( big_box->box_norms[0] * big_box->box_norms[1] * P_CONV ));

    /* Average pressure for the whole box */
    data->iso_bar.P =
        ( data->tot_press[0] + data->tot_press[1] + data->tot_press[2] ) / 3.;
}


/*
void Compute_Pressure_Isotropic_Klein( reax_system* system,
                       simulation_data* data )
{
  int i;
  reax_atom *p_atom;
  rvec dx;

  // IMPORTANT: This function assumes that current kinetic energy and
  // the center of mass of the system is already computed before.
  data->iso_bar.P = 2.0 * data->my_en.e_kin;

  for( i = 0; i < system->N; ++i ) {
    p_atom = &( system->my_atoms[i] );
    rvec_ScaledSum(dx,1.0,p_atom->x,-1.0,data->xcm);
    data->iso_bar.P += ( -F_CONV * rvec_Dot(p_atom->f, dx) );
  }

  data->iso_bar.P /= (3.0 * system->my_box.V);

  // IMPORTANT: In Klein's paper, it is stated that a dU/dV term needs
  // to be added when there are long-range interactions or long-range
  // corrections to short-range interactions present.
  // We may want to add that for more accuracy.
}


void Compute_Pressure( reax_system* system, simulation_data* data )
{
  int i;
  reax_atom *p_atom;
  rtensor temp;

  rtensor_MakeZero( data->flex_bar.P );

  for( i = 0; i < system->N; ++i ) {
    p_atom = &( system->my_atoms[i] );

    // Distance_on_T3_Gen( data->rcm, p_atom->x, &(system->my_box), &dx );

    rvec_OuterProduct( temp, p_atom->v, p_atom->v );
    rtensor_ScaledAdd( data->flex_bar.P,
               system->reax_param.sbp[ p_atom->type ].mass, temp );

    // rvec_OuterProduct(temp,workspace->virial_forces[i],p_atom->x); //dx);
    rtensor_ScaledAdd( data->flex_bar.P, -F_CONV, temp );
  }

  rtensor_Scale( data->flex_bar.P, 1.0 / system->my_box.V, data->flex_bar.P );

  data->iso_bar.P = rtensor_Trace( data->flex_bar.P ) / 3.0;
}
*/
