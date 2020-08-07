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

#include "analyze.h"

#include "box.h"
#include "list.h"
#include "vector.h"

#define MAX_FRAGMENT_TYPES (100)

void Print_Molecule( reax_system *system, molecule *m, int mode,
        char * const s, size_t size )
{
    int j, atom;

    if ( mode == 1 )
    {
        /* print molecule summary */
        for ( j = 0; j < MAX_ATOM_TYPES; ++j )
        {
            if ( m->mtypes[j] )
            {
                snprintf( s, size - 1, "%s%d",
                        system->reax_param.sbp[j].name, m->mtypes[j] );
                s[size - 1] = '\0';
            }
        }
    }
    else if ( mode == 2 )
    {
        /* print molecule details */
        for ( j = 0; j < m->atom_count; ++j )
        {
            atom = m->atom_list[j];

            snprintf( s, size - 1, "%s(%d)",
                    system->reax_param.sbp[system->my_atoms[atom].type].name,
                    atom );
            s[size - 1] = '\0';
        }
    }
}


void Visit_Bonds( int atom, int *mark, int *type, reax_system *system,
                  control_params *control, reax_list *bonds, int ignore )
{
    int i, t, start, end, nbr;
    real bo;

    mark[atom] = 1;
    t = system->my_atoms[atom].type;
    if ( ignore && control->ignore[t] )
    {
        return;
    }
    type[t]++;

    start = Start_Index( atom, bonds );
    end = End_Index( atom, bonds );

    for ( i = start; i < end; ++i )
    {
        nbr = bonds->bond_list[i].nbr;
        bo = bonds->bond_list[i].bo_data.BO;

        if ( bo >= control->bg_cut && !mark[nbr] )
        {
            Visit_Bonds( nbr, mark, type, system, control, bonds, ignore );
        }
    }
}


void Analyze_Fragments( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, FILE *fout, int ignore )
{
    int atom, i, flag, *mark;
    int num_fragments, num_fragment_types, fragment_count[MAX_FRAGMENT_TYPES];
    char fragment[MAX_ATOM_TYPES], fragments[MAX_FRAGMENT_TYPES][MAX_ATOM_TYPES];
    molecule m;
    reax_list *new_bonds;

    mark = workspace->mark;
    new_bonds = lists[BONDS];

    /* fragment analysis */
    fprintf( fout, "step%d fragments\n", data->step );
    num_fragments = 0;
    num_fragment_types = 0;
    memset( mark, 0, system->N * sizeof(int) );

    for ( atom = 0; atom < system->N; ++atom )
    {
        if ( !mark[atom] )
        {
            /* discover a new fragment */
            memset( m.mtypes, 0, MAX_ATOM_TYPES * sizeof(int) );
            Visit_Bonds( atom, mark, m.mtypes, system, control, new_bonds, ignore );
            ++num_fragments;
            Print_Molecule( system, &m, 1, fragment, sizeof(fragment) );

            /* check if a similar fragment already exists */
            flag = 0;
            for ( i = 0; i < num_fragment_types; ++i )
            {
                if ( strncmp( fragments[i], fragment, MAX_ATOM_TYPES ) == 0 )
                {
                    fragment_count[i]++;
                    flag = 1;
                    break;
                }
            }

            if ( flag == 0 )
            {
                /* it is a new one, add to the fragments list */
                strncpy( fragments[num_fragment_types], fragment, sizeof(fragments[num_fragment_types]) - 1 );
                fragments[num_fragment_types][sizeof(fragments[num_fragment_types]) - 1] = '\0';
                fragment_count[num_fragment_types] = 1;
                ++num_fragment_types;
            }
        }
    }

    /* output the results of fragment analysis */
    for ( i = 0; i < num_fragment_types; ++i )
    {
        /* strlen safe here as fragments[i] is NULL-terminated above */
        if ( strlen(fragments[i]) )
        {
            fprintf( fout, "%d of %s\n", fragment_count[i], fragments[i] );
        }
    }
    fprintf( fout, "\n" );
#if defined(DEBUG)
    fflush( fout );
#endif
}


void Analysis( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    int steps;

    steps = data->step - data->prev_steps;
    // fprintf( stderr, "prev_steps: %d\n", data->prev_steps );

    /****** Molecular Analysis ******/
    if ( control->molecular_analysis &&
            steps % control->molecular_analysis == 0 )
    {

        /* discover molecules */
        Analyze_Fragments( system, control, data, workspace, lists,
                           out_control->mol, 0 );
        /* discover fragments without the ignored atoms */
        if ( control->num_ignored )
            Analyze_Fragments( system, control, data, workspace, lists,
                               out_control->ign, 1 );
    }
}
