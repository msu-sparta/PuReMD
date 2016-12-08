/*----------------------------------------------------------------------
  SerialReax - Reax Force Field Simulator

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

#include "system_props.h"
#include "tool_box.h"
#include "vector.h"


void Temperature_Control( control_params *control, simulation_data *data,
                          output_controls *out_control )
{
    real tmp;

    if ( control->T_mode == 1 )  // step-wise temperature control
    {
        if ( (data->step - data->prev_steps) %
                ((int)(control->T_freq / control->dt)) == 0 )
        {
            if ( fabs( control->T - control->T_final ) >= fabs( control->T_rate ) )
                control->T += control->T_rate;
            else control->T = control->T_final;
        }
    }
    else if ( control->T_mode == 2 )  // constant slope control
    {
        tmp = control->T_rate * control->dt / control->T_freq;

        if ( fabs( control->T - control->T_final ) >= fabs( tmp ) )
            control->T += tmp;
    }
}


void Compute_Total_Mass( reax_system *system, simulation_data *data )
{
    int i;

    data->M = 0;

    for ( i = 0; i < system->N; i++ )
        data->M += system->reaxprm.sbp[ system->atoms[i].type ].mass;

    //fprintf ( stderr, "Compute_total_Mass -->%f<-- \n", data->M );
    data->inv_M = 1. / data->M;
}


void Compute_Center_of_Mass( reax_system *system, simulation_data *data,
                             FILE *fout )
{
    int i;
    real m, xx, xy, xz, yy, yz, zz, det;
    rvec tvec, diff;
    rtensor mat, inv;

    rvec_MakeZero( data->xcm );  // position of CoM
    rvec_MakeZero( data->vcm );  // velocity of CoM
    rvec_MakeZero( data->amcm ); // angular momentum of CoM
    rvec_MakeZero( data->avcm ); // angular velocity of CoM


    /* Compute the position, velocity and angular momentum about the CoM */
    for ( i = 0; i < system->N; ++i )
    {
        m = system->reaxprm.sbp[ system->atoms[i].type ].mass;

        rvec_ScaledAdd( data->xcm, m, system->atoms[i].x );
        rvec_ScaledAdd( data->vcm, m, system->atoms[i].v );

        rvec_Cross( tvec, system->atoms[i].x, system->atoms[i].v );
        rvec_ScaledAdd( data->amcm, m, tvec );

        /*fprintf( fout,"%3d  %g %g %g\n",
          i+1,
          system->atoms[i].v[0], system->atoms[i].v[1], system->atoms[i].v[2]  );
          fprintf( fout, "vcm:  %g %g %g\n",
          data->vcm[0], data->vcm[1], data->vcm[2] );
        */
        /* fprintf( stderr, "amcm: %12.6f %12.6f %12.6f\n",
           data->amcm[0], data->amcm[1], data->amcm[2] ); */
    }

    rvec_Scale( data->xcm, data->inv_M, data->xcm );
    rvec_Scale( data->vcm, data->inv_M, data->vcm );

    rvec_Cross( tvec, data->xcm, data->vcm );
    rvec_ScaledAdd( data->amcm, -data->M, tvec );

    data->etran_cm = 0.5 * data->M * rvec_Norm_Sqr( data->vcm );

    /* Calculate and then invert the inertial tensor */
    xx = xy = xz = yy = yz = zz = 0;

    for ( i = 0; i < system->N; ++i )
    {
        m = system->reaxprm.sbp[ system->atoms[i].type ].mass;

        rvec_ScaledSum( diff, 1., system->atoms[i].x, -1., data->xcm );
        xx += diff[0] * diff[0] * m;
        xy += diff[0] * diff[1] * m;
        xz += diff[0] * diff[2] * m;
        yy += diff[1] * diff[1] * m;
        yz += diff[1] * diff[2] * m;
        zz += diff[2] * diff[2] * m;
    }

    mat[0][0] = yy + zz;
    mat[0][1] = mat[1][0] = -xy;
    mat[0][2] = mat[2][0] = -xz;
    mat[1][1] = xx + zz;
    mat[2][1] = mat[1][2] = -yz;
    mat[2][2] = xx + yy;

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

    if ( fabs(det) > ALMOST_ZERO )
        rtensor_Scale( inv, 1. / det, inv );
    else
        rtensor_MakeZero( inv );

    /* Compute the angular velocity about the centre of mass */
    rtensor_MatVec( data->avcm, inv, data->amcm );
    data->erot_cm = 0.5 * E_CONV * rvec_Dot( data->avcm, data->amcm );

#if defined(DEBUG)
    fprintf( stderr, "xcm:  %24.15e %24.15e %24.15e\n",
             data->xcm[0], data->xcm[1], data->xcm[2] );
    fprintf( stderr, "vcm:  %24.15e %24.15e %24.15e\n",
             data->vcm[0], data->vcm[1], data->vcm[2] );
    fprintf( stderr, "amcm: %24.15e %24.15e %24.15e\n",
             data->amcm[0], data->amcm[1], data->amcm[2] );
    /* fprintf( fout, "mat:  %f %f %f\n     %f %f %f\n     %f %f %f\n",
       mat[0][0], mat[0][1], mat[0][2],
       mat[1][0], mat[1][1], mat[1][2],
       mat[2][0], mat[2][1], mat[2][2] );
       fprintf( fout, "inv:  %g %g %g\n     %g %g %g\n     %g %g %g\n",
       inv[0][0], inv[0][1], inv[0][2],
       inv[1][0], inv[1][1], inv[1][2],
       inv[2][0], inv[2][1], inv[2][2] );
       fflush( fout ); */
    fprintf( stderr, "avcm:  %24.15e %24.15e %24.15e\n",
             data->avcm[0], data->avcm[1], data->avcm[2] );
#endif
}


void Compute_Kinetic_Energy( reax_system* system, simulation_data* data )
{
    int i;
    rvec p;
    real m;

    data->E_Kin = 0.0;

    for (i = 0; i < system->N; i++)
    {
        m = system->reaxprm.sbp[system->atoms[i].type].mass;

        rvec_Scale( p, m, system->atoms[i].v );
        data->E_Kin += 0.5 * rvec_Dot( p, system->atoms[i].v );

        /* fprintf(stderr,"%d, %lf, %lf, %lf %lf\n",
           i,system->atoms[i].v[0], system->atoms[i].v[1], system->atoms[i].v[2],
           system->reaxprm.sbp[system->atoms[i].type].mass); */
    }

    data->therm.T = (2. * data->E_Kin) / (data->N_f * K_B);

    if ( fabs(data->therm.T) < ALMOST_ZERO ) /* avoid T being an absolute zero! */
        data->therm.T = ALMOST_ZERO;
}


/* IMPORTANT: This function assumes that current kinetic energy and
 *  the center of mass of the system is already computed before.
 *
 * IMPORTANT: In Klein's paper, it is stated that a dU/dV term needs
 *  to be added when there are long-range interactions or long-range
 *  corrections to short-range interactions present.
 *  We may want to add that for more accuracy.
 */
void Compute_Pressure_Isotropic( reax_system* system, control_params *control,
                                 simulation_data* data,
                                 output_controls *out_control )
{
    int i;
    reax_atom *p_atom;
    rvec tx;
    rvec tmp;
    simulation_box *box = &(system->box);

    /* Calculate internal pressure */
    rvec_MakeZero( data->int_press );

    // 0: both int and ext, 1: ext only, 2: int only
    if ( control->press_mode == 0 || control->press_mode == 2 )
    {
        for ( i = 0; i < system->N; ++i )
        {
            p_atom = &( system->atoms[i] );

            /* transform x into unitbox coordinates */
            Transform_to_UnitBox( p_atom->x, box, 1, tx );

            /* this atom's contribution to internal pressure */
            rvec_Multiply( tmp, p_atom->f, tx );
            rvec_Add( data->int_press, tmp );

            if ( out_control->debug_level > 0 )
            {
                fprintf( out_control->prs, "%-8d%8.2f%8.2f%8.2f",
                         i + 1, p_atom->x[0], p_atom->x[1], p_atom->x[2] );
                fprintf( out_control->prs, "%8.2f%8.2f%8.2f",
                         p_atom->f[0], p_atom->f[1], p_atom->f[2] );
                fprintf( out_control->prs, "%8.2f%8.2f%8.2f\n",
                         data->int_press[0], data->int_press[1], data->int_press[2]);
            }
        }
    }

    /* kinetic contribution */
    data->kin_press = 2. * (E_CONV * data->E_Kin) / ( 3. * box->volume * P_CONV );

    /* Calculate total pressure in each direction */
    data->tot_press[0] = data->kin_press -
                         ((data->int_press[0] + data->ext_press[0]) /
                          (box->box_norms[1] * box->box_norms[2] * P_CONV));

    data->tot_press[1] = data->kin_press -
                         ((data->int_press[1] + data->ext_press[1]) /
                          (box->box_norms[0] * box->box_norms[2] * P_CONV));

    data->tot_press[2] = data->kin_press -
                         ((data->int_press[2] + data->ext_press[2]) /
                          (box->box_norms[0] * box->box_norms[1] * P_CONV));

    /* Average pressure for the whole box */
    data->iso_bar.P = (data->tot_press[0] + data->tot_press[1] + data->tot_press[2]) / 3;
}


void Compute_Pressure_Isotropic_Klein( reax_system* system,
                                       simulation_data* data )
{
    int i;
    reax_atom *p_atom;
    rvec dx;

    // IMPORTANT: This function assumes that current kinetic energy and
    // the center of mass of the system is already computed before.
    data->iso_bar.P = 2.0 * data->E_Kin;

    for ( i = 0; i < system->N; ++i )
    {
        p_atom = &( system->atoms[i] );
        rvec_ScaledSum(dx, 1.0, p_atom->x, -1.0, data->xcm);
        data->iso_bar.P += ( -F_CONV * rvec_Dot(p_atom->f, dx) );
    }

    data->iso_bar.P /= (3.0 * system->box.volume);

    // IMPORTANT: In Klein's paper, it is stated that a dU/dV term needs
    // to be added when there are long-range interactions or long-range
    // corrections to short-range interactions present.
    // We may want to add that for more accuracy.
}


void Compute_Pressure( reax_system* system, simulation_data* data,
                       static_storage *workspace )
{
    int i;
    reax_atom *p_atom;
    rtensor temp;

    rtensor_MakeZero( data->flex_bar.P );

    for ( i = 0; i < system->N; ++i )
    {
        p_atom = &( system->atoms[i] );
        // Distance_on_T3_Gen( data->rcm, p_atom->x, &(system->box), &dx );
        rvec_OuterProduct( temp, p_atom->v, p_atom->v );
        rtensor_ScaledAdd( data->flex_bar.P,
                           system->reaxprm.sbp[ p_atom->type ].mass, temp );
        // rvec_OuterProduct(temp, workspace->virial_forces[i], p_atom->x );
        rtensor_ScaledAdd( data->flex_bar.P, -F_CONV, temp );
    }

    rtensor_Scale( data->flex_bar.P, 1.0 / system->box.volume, data->flex_bar.P );
    data->iso_bar.P = rtensor_Trace( data->flex_bar.P ) / 3.0;
}
