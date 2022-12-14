! module for interfacing with PuReMD code (ReaxFF+EEM in QM/MM mode)
module qm2_extern_reaxff_puremd_module
        use, intrinsic :: iso_c_binding
        implicit none

        interface
                type(c_ptr) function setup_qmmm &
                        (num_qm_atoms, qm_symbols, qm_pos, &
                        num_mm_atoms, mm_symbols, mm_pos_q, sim_box_info, &
                        ffield_filename, control_filename) &
                        bind(C, name='setup_qmmm') 
                        use, intrinsic :: iso_c_binding
                        implicit none
                        integer (c_int), value :: num_qm_atoms
                        type(c_ptr), value :: qm_symbols
                        type(c_ptr), value :: qm_pos
                        integer (c_int), value :: num_mm_atoms
                        type(c_ptr), value :: mm_symbols
                        type(c_ptr), value :: mm_pos_q
                        type(c_ptr), value :: sim_box_info
                        type(c_ptr), value :: ffield_filename
                        type(c_ptr), value :: control_filename
                end function setup_qmmm

                integer (c_int) function reset_qmmm &
                        (handle, num_qm_atoms, qm_symbols, qm_pos, &
                        num_mm_atoms, mm_symbols, mm_pos_q, sim_box_info, &
                        ffield_filename, control_filename) &
                        bind(C, name='reset_qmmm') 
                        use, intrinsic :: iso_c_binding
                        implicit none
                        type(c_ptr), value :: handle
                        integer (c_int), value :: num_qm_atoms
                        type(c_ptr), value :: qm_symbols
                        type(c_ptr), value :: qm_pos
                        integer (c_int), value :: num_mm_atoms
                        type(c_ptr), value :: mm_symbols
                        type(c_ptr), value :: mm_pos_q
                        type(c_ptr), value :: sim_box_info
                        type(c_ptr), value :: ffield_filename
                        type(c_ptr), value :: control_filename
                end function reset_qmmm

                integer (c_int) function simulate &
                        (handle) &
                        bind(C, name='simulate') 
                        use, intrinsic :: iso_c_binding
                        implicit none
                        type(c_ptr), value :: handle
                end function simulate

                integer (c_int) function cleanup &
                        (handle) &
                        bind(C, name='cleanup') 
                        use, intrinsic :: iso_c_binding
                        implicit none
                        type(c_ptr), value :: handle
                end function cleanup

                integer (c_int) function set_control_parameter &
                        (handle, keyword, values) &
                        bind(C, name='set_control_parameter') 
                        use, intrinsic :: iso_c_binding
                        implicit none
                        type(c_ptr), value :: handle
                        type(c_ptr), value :: keyword
                        type(c_ptr) :: values
                end function set_control_parameter

                integer (c_int) function set_output_enabled &
                        (handle, is_enabled) &
                        bind(C, name='set_output_enabled') 
                        use, intrinsic :: iso_c_binding
                        implicit none
                        type(c_ptr), value :: handle
                        integer (c_int), value :: is_enabled
                end function set_output_enabled

                integer (c_int) function get_atom_forces_qmmm &
                        (handle, qm_f, mm_f) &
                        bind(C, name='get_atom_forces_qmmm') 
                        use, intrinsic :: iso_c_binding
                        implicit none
                        type(c_ptr), value :: handle
                        type(c_ptr), value :: qm_f
                        type(c_ptr), value :: mm_f
                end function get_atom_forces_qmmm

                integer (c_int) function get_atom_charges_qmmm &
                        (handle, qm_q, mm_q) &
                        bind(C, name='get_atom_charges_qmmm') 
                        use, intrinsic :: iso_c_binding
                        implicit none
                        type(c_ptr), value :: handle
                        type(c_ptr), value :: qm_q
                        type(c_ptr), value :: mm_q
                end function get_atom_charges_qmmm

                integer (c_int) function get_system_info &
                        (handle, e_potential, e_kinetic, e_total, temperature, &
                        volume, pressure) &
                        bind(C, name='get_system_info') 
                        use, intrinsic :: iso_c_binding
                        implicit none
                        type(c_ptr), value :: handle
                        type(c_ptr), value :: e_potential
                        type(c_ptr), value :: e_kinetic
                        type(c_ptr), value :: e_total
                        type(c_ptr), value :: temperature
                        type(c_ptr), value :: volume
                        type(c_ptr), value :: pressure
                end function get_system_info
        end interface

        public :: get_reaxff_puremd_forces
        public :: reaxff_puremd_finalize

        character(len=*), parameter, public :: module_name = "qm2_extern_reaxff_puremd_module"
        type(c_ptr), save, private :: handle = c_null_ptr

contains
        subroutine get_reaxff_puremd_forces( num_qm_atoms, qm_pos, qm_symbols, &
                        qm_q, num_mm_atoms, mm_pos_q, mm_symbols, e_total, &
                        qm_f, mm_f, qmcharge )
                use, intrinsic :: iso_c_binding
                implicit none

                integer (c_int), intent(in) :: num_qm_atoms                     ! number of QM atoms
                real (c_double), target, intent(in) :: qm_pos(3,num_qm_atoms)   ! QM atom coordinates
                character(len=2), dimension(num_qm_atoms), target, intent(in) :: qm_symbols   ! QM atom types
                real (c_double), target, intent(inout) :: qm_q(num_qm_atoms)    ! QM atom charges (nuclear charge in au)
                integer (c_int), intent(in) :: num_mm_atoms                     ! number of MM atoms
                real (c_double), target, intent(in) :: mm_pos_q(4,num_mm_atoms) ! MM atom coordinates and charges (nuclear charge in au)
                character(len=2), dimension(num_qm_atoms), target, intent(in) :: mm_symbols   ! MM atom types
                real (c_double), target, intent(out) :: e_total                 ! SCF energy (kcal/mol)
                real (c_double), target, intent(out) :: qm_f(3,num_qm_atoms)    ! SCF QM force (AMU * Angstroms / ps^2)
                real (c_double), target, intent(out) :: mm_f(3,num_mm_atoms)    ! SCF MM force (AMU * Angstroms / ps^2)
                integer (c_int), intent(in) :: qmcharge                         ! total charge of the QM region
                logical, save :: first_call = .true.
                integer :: ix
                integer (c_int) :: ret
                character(kind=c_char, len=1024), target :: ffield_filename, keyword, values
                ! triplets for lengths and angles of QM region simulation box (Angstroms and degrees)
                real (c_double), target :: sim_box_info(6)

                ffield_filename = "../AVE/ffield" // char(0)

                ! NOTE: PuReMD must run with periodic boundary conditions (PBCs) ON,
                !       so to compensate the simulation box will have void space added around it
                !       (20 angstroms, as the long-range cut-off is 10 angstroms) in order
                !       negate the effect of PBCs for QMMM
                do ix = 1, 3
                        sim_box_info(ix) = MAX(MAXVAL(mm_pos_q(ix,:)), MAXVAL(qm_pos(ix,:))) &
                                - MIN(MINVAL(mm_pos_q(ix,:)), MINVAL(qm_pos(ix,:))) + 20.0
                end do
                ! orthogonal simulation box
                sim_box_info(4:6) = 90.0

                if ( first_call ) then
                        first_call = .false.

                        handle = setup_qmmm( num_qm_atoms, c_loc(qm_symbols), &
                                c_loc(qm_pos), num_mm_atoms, c_loc(mm_symbols), &
                                c_loc(mm_pos_q), c_loc(sim_box_info), c_loc(ffield_filename), c_null_ptr )

                        ! NVE ensemble
                        keyword = "ensemble_type" // char(0)
                        values = "0" // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! MD steps
                        keyword = "nsteps" // char(0)
                        values = "0" // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! time step length (in fs)
                        keyword = "dt" // char(0)
                        values = "0.25" // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! enable periodic boundary conditions
                        keyword = "periodic_boundaries" // char(0)
                        values = "1" // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! do not remap atom coordinates within simulation box boundaries
                        keyword = "reposition_atoms" // char(0)
                        values = "0" // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! recompute Verlet neighbor lists at every (1) MD step
                        keyword = "reneighbor" // char(0)
                        values = "1" // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! disable force and energy tabulation for Coulomb interactions
                        keyword = "tabulate_long_range" // char(0)
                        values = "0" // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! calculate energies at every (1) MD step
                        keyword = "energy_update_freq" // char(0)
                        values = "1" // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! add a 2.5 Angstrom buffer to Verlet neighbor list cut-off
                        keyword = "vlist_buffer" // char(0)
                        values = "2.5" // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! 5.0 Angstrom bond interaction cut-off
                        keyword = "nbrhood_cutoff" // char(0)
                        values = "5.0" // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! 0.001 threshold for valence angle interactions
                        keyword = "thb_cutoff" // char(0)
                        values = "0.005" // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! 7.5 Angstrom hydrogen bond interaction cut-off
                        keyword = "hbond_cutoff" // char(0)
                        values = "7.5" // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! 0.3 Angstrom bond graph calculation cut-off
                        keyword = "bond_graph_cutoff" // char(0)
                        values = "0.3" // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! EEM model (full system) for charge calculations
                        keyword = "charge_method" // char(0)
                        values = "1" // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! net charge for system (in Coulombs)
                        keyword = "cm_q_net" // char(0)
                        write (values, *) qmcharge
                        values = trim(adjustl(values)) // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! conjugate gradient algorithm in charge solver
                        keyword = "cm_solver_type" // char(0)
                        values = "2" // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! max. iterations in charge solver
                        keyword = "cm_solver_max_iters" // char(0)
                        values = "200" // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! tolerance in charge solver
                        keyword = "cm_solver_q_err" // char(0)
                        values = "1.0e-14" // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! Jacobi preconditioner in charge solver
                        keyword = "cm_solver_pre_comp_type" // char(0)
                        values = "1" // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ! disable file I/O
                        ret = set_output_enabled( handle, 0 )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_output_enabled"

                        ret = simulate( handle )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::simulate"

                        ret = get_atom_forces_qmmm( handle, c_loc(qm_f), c_loc(mm_f) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::get_atom_forces_qmmm"

                        ! disregard MM atom charges, as static (input-only)
                        ret = get_atom_charges_qmmm( handle, c_loc(qm_q), c_null_ptr )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::get_atom_charges_qmmm"

                        ! disregard all values except total energy
                        ret = get_system_info( handle, c_null_ptr, c_null_ptr, &
                                c_loc(e_total), c_null_ptr, c_null_ptr, c_null_ptr )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::get_system_info"
                else
                        ret = reset_qmmm( handle, num_qm_atoms, c_loc(qm_symbols), &
                                c_loc(qm_pos), num_mm_atoms, c_loc(mm_symbols), &
                                c_loc(mm_pos_q), c_loc(sim_box_info), c_null_ptr, c_null_ptr )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::reset_qmmm"

                        ! net charge for system (in Coulombs)
                        keyword = "cm_q_net" // char(0)
                        write (values, *) qmcharge
                        values = trim(adjustl(values)) // char(0)
                        ret = set_control_parameter( handle, c_loc(keyword), c_loc(values) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::set_control_parameter"

                        ret = simulate( handle )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::simulate"

                        ret = get_atom_forces_qmmm( handle, c_loc(qm_f), c_loc(mm_f) )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::get_atom_forces_qmmm"

                        ! disregard MM atom charges, as static (input-only)
                        ret = get_atom_charges_qmmm( handle, c_loc(qm_q), c_null_ptr )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::get_atom_charges_qmmm"

                        ! disregard all values except total energy
                        ret = get_system_info( handle, c_null_ptr, c_null_ptr, &
                                c_loc(e_total), c_null_ptr, c_null_ptr, c_null_ptr )
                        if ( ret /= 0_c_int ) stop "ERROR: get_reaxff_puremd_forces::get_system_info"
                end if
        end subroutine get_reaxff_puremd_forces


        subroutine reaxff_puremd_finalize()
                use, intrinsic :: iso_c_binding
                implicit none

                integer (c_int) :: ret

                if ( c_associated(handle) ) then
                        ret = cleanup( handle )
                        if ( ret /= 0_c_int ) stop "ERROR: reaxff_puremd_finalize::cleanup"
                endif
        end subroutine reaxff_puremd_finalize
end module qm2_extern_reaxff_puremd_module
