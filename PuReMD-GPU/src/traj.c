/*----------------------------------------------------------------------
  PuReMD-GPU - Reax Force Field Simulator

  Copyright (2014) Purdue University
  Sudhir Kylasa, skylasa@purdue.edu
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
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

#include "traj.h"

#include "list.h"

#ifdef __PRINT_CPU_RESULTS__
  #include "cuda_copy.h"
#endif


/************************************************/
/*      CUSTOM FORMAT ROUTINES                  */
/************************************************/
int Write_Custom_Header(reax_system *system, control_params *control, 
        static_storage *workspace, output_controls *out_control)
{
    int i, header_len, control_block_len, frame_format_len;
    // char buffer[2048];
    char control_block[2048];
    char frame_format[2048];
    char atom_format[100], bond_format[100], angle_format[100];

    sprintf( control_block, CONTROL_BLOCK,
            system->N,
            control->restart,
            control->restart_from,
            control->random_vel,
            out_control->restart_freq,
            control->ensemble,
            control->nsteps,
            control->dt,
            control->reposition_atoms,
            control->restrict_bonds,
            control->tabulate,
            control->nbr_cut,
            control->r_cut,
            control->bg_cut,
            control->bo_cut,
            control->thb_cut,
            control->hb_cut,
            control->q_err,
            control->T_init,
            control->T_final,
            control->Tau_T,
            control->T_mode,
            control->T_rate,
            control->T_freq,
            control->P[0], control->P[1], control->P[2], 
            control->Tau_P[0], control->Tau_P[1], control->Tau_P[2],
            control->compressibility,
            control->press_mode,
            control->remove_CoM_vel,
            out_control->write_steps,
            out_control->traj_compress,
            out_control->traj_format,
            out_control->atom_format,
            out_control->bond_info,
            out_control->angle_info,
            out_control->energy_update_freq,
            control->molec_anal,
            control->freq_molec_anal );

            control_block_len = strlen( control_block );


            sprintf( frame_format, "Frame Format: %d\n%s\n%s\n", 
                    NUM_FRAME_GLOBALS, FRAME_GLOBALS_FORMAT, FRAME_GLOBAL_NAMES );

            atom_format[0] = OPT_NOATOM;
            switch( out_control->atom_format )
            {
                case OPT_ATOM_BASIC: sprintf( atom_format, "Atom_Basic: %s", ATOM_BASIC );
                             break;
                case OPT_ATOM_wF: sprintf( atom_format, "Atom_wF: %s", ATOM_wF );
                          break;
                case OPT_ATOM_wV: sprintf( atom_format, "Atom_wV: %s", ATOM_wV );
                          break;
                case OPT_ATOM_FULL: sprintf( atom_format, "Atom_Full: %s", ATOM_FULL );
                            break;
            }
            strcat( frame_format, atom_format );

            bond_format[0] = OPT_NOBOND;
            if( out_control->bond_info == OPT_BOND_BASIC )
                sprintf( bond_format, "Bond_Line: %s", BOND_BASIC );
            else if( out_control->bond_info == OPT_BOND_FULL )
                sprintf( bond_format, "Bond_Line_Full: %s", BOND_FULL );
            strcat( frame_format, bond_format );

            angle_format[0] = OPT_NOANGLE;
            if( out_control->angle_info == OPT_ANGLE_BASIC )
                sprintf( angle_format, "Angle_Line: %s", ANGLE_BASIC );
            strcat( frame_format, angle_format );

            frame_format_len = strlen( frame_format );


            header_len = HEADER_INIT_LEN + (control_block_len + SIZE_INFO_LEN2)+ 
                (frame_format_len + SIZE_INFO_LEN2) + 
                (ATOM_MAPPING_LEN * system->N + SIZE_INFO_LEN2);

            out_control->write( out_control->trj, HEADER_INIT, 
                    header_len, HEADER_INIT_LEN, out_control->traj_title );

            out_control->write( out_control->trj, SIZE_INFO_LINE2,
                    control_block_len + (frame_format_len + SIZE_INFO_LEN2) + 
                    (ATOM_MAPPING_LEN * system->N + SIZE_INFO_LEN2), 
                    control_block_len );
            out_control->write( out_control->trj, "%s", control_block );

            out_control->write( out_control->trj, SIZE_INFO_LINE2, 
                    frame_format_len + 
                    (ATOM_MAPPING_LEN * system->N + SIZE_INFO_LEN2), 
                    frame_format_len );
            out_control->write( out_control->trj, "%s", frame_format );

            out_control->write( out_control->trj, SIZE_INFO_LINE2, 
                    ATOM_MAPPING_LEN * system->N, 
                    ATOM_MAPPING_LEN * system->N );

            for( i = 0; i < system->N; ++i )
                out_control->write( out_control->trj, ATOM_MAPPING,  
                        workspace->orig_id[i], 
                        system->atoms[i].type, 
                        system->atoms[i].name, 
                        system->reaxprm.sbp[ system->atoms[i].type ].mass ); 

            fflush( out_control->trj );

            return 0;
}


int Append_Custom_Frame( reax_system *system, control_params *control, 
        simulation_data *data, static_storage *workspace, 
        list **lists, output_controls *out_control )
{
    int i, j, pi, pk, pk_j;
    int write_atoms, write_bonds, write_angles;
    int frame_len, atom_line_len, bond_line_len, angle_line_len, rest_of_frame_len;
    int frame_globals_len, num_bonds, num_thb_intrs;
    real P;
    char buffer[2048];
    list *bonds = (*lists) + BONDS;
    list *thb_intrs =  (*lists) + THREE_BODIES;
    bond_data *bo_ij;


    /* IMPORTANT: This whole part will go to init_trj after finalized! */
    switch( out_control->atom_format )
    {
        case OPT_ATOM_BASIC: 
            atom_line_len = ATOM_BASIC_LEN;
            write_atoms = 1;
            break;
        case OPT_ATOM_wF: 
            atom_line_len = ATOM_wF_LEN; 
            write_atoms = 1;
            break;
        case OPT_ATOM_wV: 
            atom_line_len = ATOM_wV_LEN; 
            write_atoms = 1;
            break;
        case OPT_ATOM_FULL: 
            atom_line_len = ATOM_FULL_LEN; 
            write_atoms = 1;
            break;
        default: 
            atom_line_len = 0;
            write_atoms = 0;
    }


    /* bond preparations */
    bond_line_len = write_bonds = 0;
    if( out_control->bond_info == OPT_BOND_BASIC )
    {
        bond_line_len = BOND_BASIC_LEN;
        write_bonds = 1;
    }
    else if( out_control->bond_info == OPT_BOND_FULL )
    {
        bond_line_len = BOND_FULL_LEN;
        write_bonds = 1;
    }

#ifdef __DEBUG_CUDA__
    fprintf (stderr, "Append Custom Frame -- write_bonds --> %d \n", write_bonds);
#endif

    num_bonds = 0;
    if( write_bonds )
    {

#ifdef __PRINT_CPU_RESULTS__
        //fprintf (stderr, "Synching bonds from device for printing ....\n");
        Sync_Host_Device (bonds, (dev_lists + BONDS), TYP_BOND );
#endif

        for( i = 0; i < system->N; ++i )
            for( j = Start_Index( i, bonds ); j < End_Index( i, bonds ); ++j )
                if( i < bonds->select.bond_list[j].nbr && 
                        bonds->select.bond_list[j].bo_data.BO >= control->bg_cut )
                    ++num_bonds;
    }


    /* angle preparations */
    if( out_control->angle_info == OPT_ANGLE_BASIC )
    {
        angle_line_len = ANGLE_BASIC_LEN;
        write_angles = 1;
    }
    else 
    {
        angle_line_len = 0;
        write_angles = 0;
    }

#ifdef __DEBUG_CUDA__
    fprintf (stderr, "Append Custom Frame -- write-angles --> %d \n", write_angles );
#endif

    num_thb_intrs = 0;
    if( write_angles ) {

#ifdef __PRINT_CPU_RESULTS__
        //fprintf (stderr, "Synching three bodies from deivce for printing ... \n");
        Sync_Host_Device (thb_intrs, dev_lists + THREE_BODIES, TYP_THREE_BODY );
        if ( !write_bonds) {
            //fprintf (stderr, "Synching bonds for three bodies from device for printing ... \n");
            Sync_Host_Device (bonds, (dev_lists + BONDS), TYP_BOND );
        }
#endif 

        for( j = 0; j < system->N; ++j )
            for( pi = Start_Index(j, bonds); pi < End_Index(j, bonds); ++pi )
                if( bonds->select.bond_list[pi].bo_data.BO >= control->bg_cut ) 
                    // physical j&i bond
                    for( pk = Start_Index( pi, thb_intrs ); 
                            pk < End_Index( pi, thb_intrs ); ++pk )
                        if( bonds->select.bond_list[pi].nbr < 
                                thb_intrs->select.three_body_list[pk].thb ) {
                            // get k's pointer on j's bond list
                            pk_j = thb_intrs->select.three_body_list[pk].pthb;

                            if( bonds->select.bond_list[pk_j].bo_data.BO >= control->bg_cut ) 
                                // physical j&k bond
                                ++num_thb_intrs;
                        }
    }



    /* get correct pressure */
    if( control->ensemble == NPT || control->ensemble == sNPT )
        P = data->flex_bar.P_scalar;
    else  if( control->ensemble == iNPT )
        P = data->iso_bar.P;
    else P = 0;


    /* calculate total frame length*/
    sprintf( buffer, FRAME_GLOBALS,
            data->step, data->time, 
            data->E_Tot, data->E_Pot, E_CONV * data->E_Kin, data->therm.T,
            P, system->box.volume,
            system->box.box_norms[0], 
            system->box.box_norms[1], 
            system->box.box_norms[2],
            90.0, 90.0, 90.0, // IMPORTANT: need to rewrite for flexible boxes!
            data->E_BE,
            data->E_Ov,  data->E_Un,  data->E_Lp,
            data->E_Ang, data->E_Pen, data->E_Coa, data->E_HB,
            data->E_Tor, data->E_Con, 
            data->E_vdW, data->E_Ele, data->E_Pol );
    frame_globals_len = strlen( buffer );

    frame_len = frame_globals_len + 
        write_atoms  * SIZE_INFO_LEN3 + system->N * atom_line_len +
        write_bonds  * SIZE_INFO_LEN3 + num_bonds * bond_line_len +
        write_angles * SIZE_INFO_LEN3 + num_thb_intrs * angle_line_len;


    /* write size info & frame globals */
    out_control->write( out_control->trj, SIZE_INFO_LINE2, 
            frame_len, frame_globals_len );
    out_control->write( out_control->trj, "%s", buffer );


    /* write size info & atom lines */  
    if( write_atoms ) 
    {
        rest_of_frame_len = system->N * atom_line_len +
            write_bonds  * SIZE_INFO_LEN3 + num_bonds * bond_line_len +
            write_angles * SIZE_INFO_LEN3 + num_thb_intrs * angle_line_len;

        out_control->write( out_control->trj, SIZE_INFO_LINE3, 
                rest_of_frame_len, system->N * atom_line_len, 
                system->N );
    }

    switch( out_control->atom_format )
    {
        case 4: 
            for( i = 0; i < system->N; ++i )
                out_control->write( out_control->trj, ATOM_BASIC, 
                        workspace->orig_id[i], 
                        system->atoms[i].x[0], 
                        system->atoms[i].x[1], 
                        system->atoms[i].x[2],
                        system->atoms[i].q );
            break;
        case 5:
            for( i = 0; i < system->N; ++i )
                out_control->write( out_control->trj, ATOM_wF, 
                        workspace->orig_id[i],
                        system->atoms[i].x[0], 
                        system->atoms[i].x[1], 
                        system->atoms[i].x[2],
                        system->atoms[i].f[0], 
                        system->atoms[i].f[1], 
                        system->atoms[i].f[2],
                        system->atoms[i].q );
            break;
        case 6: 
            for( i = 0; i < system->N; ++i )
                out_control->write( out_control->trj, ATOM_wV, 
                        workspace->orig_id[i], 
                        system->atoms[i].x[0], 
                        system->atoms[i].x[1], 
                        system->atoms[i].x[2],
                        system->atoms[i].v[0], 
                        system->atoms[i].v[1], 
                        system->atoms[i].v[2],
                        system->atoms[i].q );
            break;
        case 7: 
            for( i = 0; i < system->N; ++i )
                out_control->write( out_control->trj, ATOM_FULL, 
                        workspace->orig_id[i], 
                        system->atoms[i].x[0], 
                        system->atoms[i].x[1], 
                        system->atoms[i].x[2],
                        system->atoms[i].v[0], 
                        system->atoms[i].v[1], 
                        system->atoms[i].v[2],
                        system->atoms[i].f[0], 
                        system->atoms[i].f[1], 
                        system->atoms[i].f[2],
                        system->atoms[i].q );
            break;
    }
    fflush( out_control->trj );


    /* write size info & bond lines */
    if( write_bonds )
    {
        rest_of_frame_len = num_bonds * bond_line_len +
            write_angles * SIZE_INFO_LEN3 + num_thb_intrs * angle_line_len;

        out_control->write( out_control->trj, SIZE_INFO_LINE3, 
                rest_of_frame_len, num_bonds * bond_line_len, 
                num_bonds );
    }

    if( out_control->bond_info == 1 ) {
        for( i = 0; i < system->N; ++i )
            for( j = Start_Index( i, bonds ); j < End_Index( i, bonds ); ++j )
                if( i < bonds->select.bond_list[j].nbr && 
                        bonds->select.bond_list[j].bo_data.BO >= control->bg_cut ) {
                    bo_ij = &( bonds->select.bond_list[j] );
                    out_control->write( out_control->trj, BOND_BASIC, 
                            workspace->orig_id[i], 
                            workspace->orig_id[bo_ij->nbr], 
                            bo_ij->d, bo_ij->bo_data.BO );
                }
    }
    else if( out_control->bond_info == 2 ) {
        for( i = 0; i < system->N; ++i )
            for( j = Start_Index( i, bonds ); j < End_Index( i, bonds ); ++j )
                if( i < bonds->select.bond_list[j].nbr && 
                        bonds->select.bond_list[j].bo_data.BO >= control->bg_cut ) {
                    bo_ij = &( bonds->select.bond_list[j] );
                    out_control->write( out_control->trj, BOND_FULL, 
                            workspace->orig_id[i], 
                            workspace->orig_id[bo_ij->nbr], 
                            bo_ij->d, bo_ij->bo_data.BO, bo_ij->bo_data.BO_s, 
                            bo_ij->bo_data.BO_pi, bo_ij->bo_data.BO_pi2 );
                }
    }

    fflush( out_control->trj );


    /* write size info & angle lines */
    if( out_control->angle_info ) {
        out_control->write( out_control->trj, SIZE_INFO_LINE3,
                num_thb_intrs * angle_line_len, 
                num_thb_intrs * angle_line_len, num_thb_intrs );

        for( j = 0; j < system->N; ++j )
            for( pi = Start_Index(j, bonds); pi < End_Index(j, bonds); ++pi )
                if( bonds->select.bond_list[pi].bo_data.BO >= control->bg_cut ) 
                    // physical j&i bond
                    for( pk = Start_Index( pi, thb_intrs ); 
                            pk < End_Index( pi, thb_intrs ); ++pk )
                        if( bonds->select.bond_list[pi].nbr < 
                                thb_intrs->select.three_body_list[pk].thb ) {
                            pk_j = thb_intrs->select.three_body_list[pk].pthb; 
                            // get k's pointer on j's bond list

                            if( bonds->select.bond_list[pk_j].bo_data.BO >= control->bg_cut ) 
                                // physical j&k bond
                                out_control->write( out_control->trj, ANGLE_BASIC,
                                        workspace->orig_id[bonds->select.bond_list[pi].nbr], 
                                        workspace->orig_id[j], 
                                        workspace->orig_id[thb_intrs->select.three_body_list[pk].thb], 
                                        RAD2DEG(thb_intrs->select.three_body_list[pk].theta) );
                        }
    }

    fflush( out_control->trj );

    return 0;
}

/*
   void Read_Traj( output_controls *out_control, char *traj_name )
   {
   int skip_all, skip_part, n;
   char size_buffer[50];
// char read_buffer[2048];

out_control->trj = (FILE *)gzopen( traj_name, "r" );

fprintf( stderr, "file opened!\n" );

while( !gzeof( out_control->trj ) )
{
if( gzgets( out_control->trj, size_buffer, 50 ) == Z_NULL )
break;

fprintf( stderr, "read line\n" );

if( strlen( size_buffer ) >= SIZE_INFO_LEN3 )
sscanf( size_buffer, "%d %d %d", &skip_all, &skip_part, &n );
else
sscanf( size_buffer, "%d %d", &skip_all, &skip_part );

fprintf( stderr, "%d %d\n", skip_all, skip_part );

gzseek( out_control->trj, skip_part, SEEK_CUR );
}

gzclose( out_control->trj );
}
 */



/********************************************************/
/************      XYZ FORMAT ROUTINES    ***************/
/********************************************************/

int Write_xyz_Header( reax_system *system, control_params *control, 
        static_storage* workspace, output_controls *out_control )
{
    fflush( out_control->trj );

    return 1;
}


int Append_xyz_Frame( reax_system *system, control_params *control, 
        simulation_data *data, static_storage *workspace, 
        list **lists, output_controls *out_control )
{
    int i;

    out_control->write( out_control->trj, "%d\n", system->N );

    out_control->write( out_control->trj, "%d\t%8.3f\t%8.3f\t%8.3f\t%8.3f\n",
            data->step,
            data->E_Tot, data->E_Pot, 
            E_CONV*data->E_Kin, data->therm.T );

    for( i = 0; i < system->N; ++i )
        out_control->write( out_control->trj, "%3s %10.5f %10.5f %10.5f\n",
                system->reaxprm.sbp[ system->atoms[i].type ].name,
                system->atoms[i].x[0], 
                system->atoms[i].x[1], 
                system->atoms[i].x[2] );

    fflush( out_control->trj );

    return 1;
}
