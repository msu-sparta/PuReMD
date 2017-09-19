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

#ifndef __PRINT_UTILS_H_
#define __PRINT_UTILS_H_

#include "mytypes.h"


typedef void (*print_interaction)(reax_system*, control_params*, simulation_data*,
        static_storage*, list**, output_controls*);

extern print_interaction Print_Interactions[NO_OF_INTERACTIONS];


void Print_Near_Neighbors( reax_system*, control_params*, static_storage*, list** );

void Print_Far_Neighbors( reax_system*, control_params*, static_storage*, list** );

void Print_Total_Force( reax_system*, control_params*, simulation_data*,
        static_storage*, list**, output_controls* );

void Output_Results( reax_system*, control_params*, simulation_data*,
        static_storage*, list**, output_controls* );

void Print_Bond_Orders( reax_system*, control_params*, simulation_data*,
        static_storage*, list**, output_controls* );

void Print_Linear_System( reax_system*, control_params*, static_storage*, int );

void Print_Charges( reax_system*, control_params*, static_storage*, int );

void Print_Soln( static_storage*, real*, real*, real*, int );

void Print_Sparse_Matrix( sparse_matrix* );

void Print_Sparse_Matrix2( sparse_matrix*, char* );

void Print_Bonds( reax_system*, list*, char* );
void Print_Bond_List2( reax_system*, list*, char* );

#ifdef TEST_FORCES
void Dummy_Printer( reax_system*, control_params*, simulation_data*,
        static_storage*, list**, output_controls* );
void Print_Bond_Forces( reax_system*, control_params*, simulation_data*,
        static_storage*, list**, output_controls* );
void Print_LonePair_Forces( reax_system*, control_params*, simulation_data*,
        static_storage*, list**, output_controls* );
void Print_OverUnderCoor_Forces(reax_system*, control_params*,
        simulation_data*, static_storage*, list**, output_controls*);
void Print_Three_Body_Forces( reax_system*, control_params*, simulation_data*,
        static_storage*, list**, output_controls* );
void Print_Hydrogen_Bond_Forces(reax_system*, control_params*,
        simulation_data*, static_storage*, list**, output_controls*);
void Print_Four_Body_Forces( reax_system*, control_params*, simulation_data*,
        static_storage*, list**, output_controls* );
void Print_vdW_Coulomb_Forces( reax_system*, control_params*, simulation_data*,
        static_storage*, list**, output_controls* );
void Compare_Total_Forces( reax_system*, control_params*, simulation_data*,
        static_storage*, list**, output_controls* );
void Init_Force_Test_Functions( );
#endif

#endif
