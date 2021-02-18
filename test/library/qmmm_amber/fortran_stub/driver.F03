module driver_utils
        use, intrinsic :: iso_c_binding

        implicit none

contains
        subroutine parse_bgf_file( filename, num_atoms, sim_box_info, &
                        num_qm_atoms, qm_types, qm_pos, qm_q, num_mm_atoms, &
                        mm_types, mm_pos_q )
                implicit none

                character(len=1024), intent(in) :: filename
                integer, intent(out) :: num_atoms
                integer (c_int), intent(in) :: num_qm_atoms, num_mm_atoms
                real (c_double), intent(out) :: sim_box_info(6)
                real (c_double), intent(out) :: qm_pos(3*num_qm_atoms), qm_q(num_qm_atoms), mm_pos_q(4*num_mm_atoms)
                character(len=5), intent(out) :: qm_types(num_qm_atoms), mm_types(num_mm_atoms)

                character(len=200) :: line
                integer :: bgf_version, num_descrp, num_remark, num_format, line_num
                integer :: nqm, nmm

                open (unit=3, file=filename, action='read', status='old')

                read (3,'(a40)', end=46, err=40) line

                read (line,'(6x,i4)') bgf_version
                num_descrp = 0
                num_remark = 0
                num_format = 0
                line_num = 0
                nqm = 0
                nmm = 0

                30 read (3,'(a200)', end=46, err=40) line
                line_num = line_num + 1

                if (line(1:6) == 'DESCRP') then
!                        read (line, '(7x,a40)', end=46, err=46) descrps(num_descrp)
                        num_descrp = num_descrp + 1
                end if

                if (line(1:6) == 'REMARK') then
!                        read (line, '(7x,a40)', end=46, err=46) remarks(num_remark)
                        num_remark = num_remark + 1
                end if

                if (line(1:6) == 'FORMAT') then
!                        read(line, '(7x,a40)', end=46, err=46) formats(num_format)
                        num_format = num_format + 1
                end if

                if (line(1:6) == 'CRYSTX') then
                        read (line, '(7x,6f7.3)', end=46, err=46) &
                                sim_box_info(1), sim_box_info(2), sim_box_info(3), &
                                sim_box_info(4), sim_box_info(5), sim_box_info(6)
                end if

                if (line(1:6) == 'HETATM') then
                        if (bgf_version < 400) then
                                if (nqm < num_qm_atoms) then
                                        read (line, '(30x,3f10.5,1x,a5,6x,f10.5)', end=40, err=40) &
                                                qm_pos(3*nqm+1), qm_pos(3*nqm+2), qm_pos(3*nqm+3), &
                                                qm_types(nqm+1), qm_q(nqm+1)

                                        nqm = nqm + 1
                                else
                                        ! take the MM charges from BGF file;
                                        ! Amber would set instead in real QM/MM simulation
                                        read (line, '(30x,3f10.5,1x,a5,6x,f10.5)', end=40, err=40) &
                                                mm_pos_q(4*nmm+1), mm_pos_q(4*nmm+2), mm_pos_q(4*nmm+3), &
                                                mm_types(nmm+1), mm_pos_q(4*nmm+4)

                                        nmm = nmm + 1
                                endif
                        else
                                stop 'Unsupported Biograf-version'
                        end if
                end if

                goto 30

                40 write (*,*) 'Error on line ', line_num + 1, ' of Biograf-input'
                stop
                46 continue

                close (3)

                num_atoms = nqm + nmm
        end subroutine parse_bgf_file
end module driver_utils


! driver for PuReMD library interface to Amber in Fortran
program driver
        use, intrinsic :: iso_c_binding
        use driver_utils
        use reaxff_puremd

        implicit none

#ifdef f2003
        use, intrinsic :: iso_fortran_env, only: stdin=>input_unit, &
                stdout=>output_unit, &
                stderr=>error_unit
#else
#define stdin  5
#define stdout 6
#define stderr 0
#endif

        integer, parameter :: max_sim_steps = 0

        character(len=1024), dimension(:), allocatable :: args
        character(len=1024) :: filename
        integer :: num_args, ix, num_atoms
        integer (c_int) :: num_qm_atoms, num_mm_atoms
        real (c_double) :: sim_box_info(6), e_total
        real (c_double), dimension(:), allocatable :: qm_pos, qm_q, qm_f, mm_pos_q, mm_f
        character(len=5), dimension(:), allocatable :: qm_types, mm_types
        integer (c_int), dimension(:), allocatable :: qm_types_map, mm_types_map

        num_args = command_argument_count( )
        if (num_args /= 3) then
                write (stderr, *) 'ERROR: incorrect number of command-line arguments'
                write (stderr, *) 'usage: ./driver bgf_filename num_qm_atoms num_mm_atoms'
                stop
        endif

        allocate( args(num_args) )

        do ix = 1, num_args
        call get_command_argument( ix, args(ix) )
        end do

        filename = args(1)
        read (args(2), *) num_qm_atoms
        read (args(3), *) num_mm_atoms

        allocate( qm_types(num_qm_atoms) )
        allocate( qm_types_map(num_qm_atoms) )
        allocate( qm_pos(3*num_qm_atoms) )
        allocate( qm_q(num_qm_atoms) )
        allocate( qm_f(3*num_qm_atoms) )
        allocate( mm_types(num_mm_atoms) )
        allocate( mm_types_map(num_mm_atoms) )
        allocate( mm_pos_q(4*num_mm_atoms) )
        allocate( mm_f(3*num_mm_atoms) )

        call parse_bgf_file( filename, num_atoms, sim_box_info, num_qm_atoms, qm_types, &
                qm_pos, qm_q, num_mm_atoms, mm_types, mm_pos_q )

        if (num_atoms /= (num_qm_atoms + num_mm_atoms)) then
                write (stderr, *) 'ERROR: num_atoms in BGF file != num_qm_atoms + num_mm_atoms'
                write (stderr, *) 'INFO: num_atoms = ', num_atoms
                write (stderr, *) 'INFO: num_qm_atoms = ', num_qm_atoms
                write (stderr, *) 'INFO: num_mm_atoms = ', num_mm_atoms
                stop
        endif

        do ix = 1, num_qm_atoms
        if (trim(adjustl(qm_types(ix))) == 'C') then
                qm_types_map(ix) = 0
        else if (trim(adjustl(qm_types(ix))) == 'H') then
                qm_types_map(ix) = 1
        else if (trim(adjustl(qm_types(ix))) == 'O') then
                qm_types_map(ix) = 2
        else
                stop 'ERROR: unrecogized QM atom type'
        endif
        end do

        do ix = 1, num_mm_atoms
        if (trim(adjustl(mm_types(ix))) == 'C') then
                mm_types_map(ix) = 0
        else if (trim(adjustl(mm_types(ix))) == 'H') then
                mm_types_map(ix) = 1
        else if (trim(adjustl(mm_types(ix))) == 'O') then
                mm_types_map(ix) = 2
        else
                stop 'ERROR: unrecogized MM atom type'
        endif
        end do

        call get_reaxff_puremd_forces( max_sim_steps, 0, sim_box_info, &
                num_qm_atoms, qm_pos, qm_types_map, qm_q, num_mm_atoms, &
                mm_pos_q, mm_types_map, e_total, qm_f, mm_f )

        write (stdout, fmt=75) 'Total energy:', e_total
        write (stdout, *)
        write (stdout, fmt=79) 'i', 'F_x', 'F_y', 'F_z', 'Q'
        do ix = 1, num_qm_atoms
                write (stdout, fmt=80) ix, qm_f(3 * (ix - 1) + 1), &
                        qm_f(3 * (ix - 1) + 2), qm_f(3 * (ix - 1) + 3), qm_q(ix)
        end do
        do ix = 1, num_mm_atoms
                write (stdout, fmt=80) ix + num_qm_atoms, mm_f(3 * (ix - 1) + 1), &
                        mm_f(3 * (ix - 1) + 2), mm_f(3 * (ix - 1) + 3), mm_pos_q(4 * (ix - 1) + 3)
        end do

        deallocate(args)
        deallocate(qm_types)
        deallocate(qm_types_map)
        deallocate(qm_pos)
        deallocate(qm_q)
        deallocate(qm_f)
        deallocate(mm_types)
        deallocate(mm_types_map)
        deallocate(mm_pos_q)
        deallocate(mm_f)

        75 format (A13, 1X, F24.15)
        79 format (A6, '|', A24, '|', A24, '|', A24, '|', A24)
        80 format (I6, 1X, F24.15, 1X, F24.15, 1X, F24.15, 1X, F24.15)
end program driver
