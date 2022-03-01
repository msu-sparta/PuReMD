#!/bin/python3

from ctypes import c_int, c_double, c_char, c_char_p, c_void_p, \
        Structure, Union, POINTER, CFUNCTYPE, cdll
import sqlite3 as sq3
from os import path


class BondOrderData(Structure):
    _fields_ = [
            ("BO", c_double),
            ("BO_s", c_double),
            ("BO_pi", c_double),
            ("BO_pi2", c_double),
            ("Cdbo", c_double),
            ("Cdbopi", c_double),
            ("Cdbopi2", c_double),
            ("C1dbo", c_double),
            ("C2dbo", c_double),
            ("C3dbo", c_double),
            ("C1dbopi", c_double),
            ("C2dbopi", c_double),
            ("C3dbopi", c_double),
            ("C4dbopi", c_double),
            ("C1dbopi2", c_double),
            ("C2dbopi2", c_double),
            ("C3dbopi2", c_double),
            ("C4dbopi2", c_double),
            ("dBOp", c_double * 3),
            ("dln_BOp_s", c_double * 3),
            ("dln_BOp_pi", c_double * 3),
            ("dln_BOp_pi2", c_double * 3),
            ]


class ThreeBodyData(Structure):
    _fields_ = [
            ("thb", c_int),
            ("pthb", c_int),
            ("theta", c_double),
            ("cos_theta", c_double),
            ("dcos_di", c_double * 3),
            ("dcos_dj", c_double * 3),
            ("dcos_dk", c_double * 3),
            ]


class BondData(Structure):
    _fields_ = [
            ("nbr", c_int),
            ("sym_index", c_int),
            ("dbond_index", c_int),
            ("rel_box", c_int * 3),
            ("d", c_double),
            ("dvec", c_double * 3),
            ("bo_data", BondOrderData),
            ]


class DBondData(Structure):
    _fields_ = [
            ("wrt", c_int),
            ("dBO", c_double * 3),
            ("dBOpi", c_double * 3),
            ("dBOpi2", c_double * 3),
            ]


class DDeltaData(Structure):
    _fields_ = [
            ("wrt", c_int),
            ("dVal", c_double * 3),
            ]


class FarNbrData(Structure):
    _fields_ = [
            ("nbr", c_int),
            ("rel_box", c_int * 3),
            ("d", c_double),
            ("dvec", c_double * 3),
            ]


class NearNbrData(Structure):
    _fields_ = [
            ("nbr", c_int),
            ("rel_box", c_int * 3),
            ("d", c_double),
            ("dvec", c_double * 3),
            ]


class HBondData(Structure):
    _fields_ = [
            ("nbr", c_int),
            ("scl", c_int),
            ("ptr", POINTER(FarNbrData)),
            ]


class Thermostat(Structure):
    _fields_ = [
            ("T", c_double),
            ("xi", c_double),
            ("v_xi", c_double),
            ("v_xi_old", c_double),
            ("G_xi", c_double),
            ]


class IsotropicBarostat(Structure):
    _fields_ = [
            ("P", c_double),
            ("eps", c_double),
            ("v_eps", c_double),
            ("v_eps_old", c_double),
            ("a_eps", c_double),
            ]


class FlexibleBarostat(Structure):
    _fields_ = [
            ("P", c_double * 9),
            ("P_scalar", c_double),
            ("eps", c_double),
            ("v_eps", c_double),
            ("v_eps_old", c_double),
            ("a_eps", c_double),
            ("h0", c_double * 9),
            ("v_g0", c_double * 9),
            ("v_g0_old", c_double * 9),
            ("a_g0", c_double * 9),
            ]


class ReaxTiming(Structure):
    _fields_ = [
            ("start", c_double),
            ("end", c_double),
            ("elapsed", c_double),
            ("total", c_double),
            ("nbrs", c_double),
            ("init_forces", c_double),
            ("bonded", c_double),
            ("nonb", c_double),
            ("cm", c_double),
            ("cm_sort_mat_rows", c_double),
            ("cm_solver_pre_comp", c_double),
            ("cm_solver_pre_app", c_double),
            ("cm_solver_iters", c_int),
            ("cm_solver_spmv", c_double),
            ("cm_solver_vector_ops", c_double),
            ("cm_solver_orthog", c_double),
            ("cm_solver_tri_solve", c_double),
            ("cm_last_pre_comp", c_double),
            ("cm_total_loss", c_double),
            ("cm_optimum", c_double),
            ("num_retries", c_int),
            ]


class SimulationData(Structure):
    _fields_ = [
            ("sim_id", c_int),
            ("step", c_int),
            ("prev_step", c_int),
            ("time", c_double),
            ("M", c_double),
            ("inv_M", c_double),
            ("xcm", c_double * 3),
            ("vcm", c_double * 3),
            ("fcm", c_double * 3),
            ("amcm", c_double * 3),
            ("avcm", c_double * 3),
            ("etran_cm", c_double),
            ("erot_cm", c_double),
            ("kinetic", c_double * 9),
            ("virial", c_double * 9),
            ("E_Tot", c_double),
            ("E_Kin", c_double),
            ("E_Pot", c_double),
            ("E_BE", c_double),
            ("E_Ov", c_double),
            ("E_Un", c_double),
            ("E_Lp", c_double),
            ("E_Ang", c_double),
            ("E_Pen", c_double),
            ("E_Coa", c_double),
            ("E_HB", c_double),
            ("E_Tor", c_double),
            ("E_Con", c_double),
            ("E_vdW", c_double),
            ("E_Ele", c_double),
            ("E_Pol", c_double),
            ("N_f", c_double),
            ("t_scale", c_double * 3),
            ("p_scale", c_double * 9),
            ("therm", Thermostat),
            ("iso_bar", IsotropicBarostat),
            ("flex_bar", FlexibleBarostat),
            ("inv_W", c_double),
            # requires OpenMP enabled
            ("press_local", c_double * 9),
            ("press", c_double * 9),
            ("kin_press", c_double * 9),
            ("tot_press", c_double * 3),
            ("timing", ReaxTiming),
            ]


class ReaxAtom(Structure):
    _fields_ = [
            ("type", c_int),
            ("is_dummy", c_int),
            ("rel_map", c_int * 3),
            ("name", c_char * 9),
            ("x", c_double * 3),
            ("v", c_double * 3),
            ("f", c_double * 3),
            ("q", c_double),
            # requires QM/MM support enabled
            ("qmmm_mask", c_int),
            # requires QM/MM support enabled
            ("q_init", c_double),
            ]


def create_db(name='spuremd.db'):
    conn = sq3.connect(name)

    conn.executescript("""
        CREATE TABLE simulation(
            id integer,
            date text,
            name text,
            ensemble_type integer,
            steps integer,
            time_step integer,
            restart_format integer,
            random_velocity integer,
            reposition_atoms integer,
            peroidic_boundary integer,
            geo_format integer,
            restrict_bonds integer,
            tabulate_long_range integer,
            reneighbor integer,
            vlist_cutoff real,
            neighbor_cutoff real,
            three_body_cutoff real,
            hydrogen_bond_cutoff real,
            bond_graph_cutoff real,
            charge_method integer,
            cm_q_net real,
            cm_solver_type integer,
            cm_solver_max_iters integer,
            cm_solver_restart integer,
            cm_solver_q_err real,
            cm_domain_sparsity real,
            cm_solver_pre_comp_type integer,
            cm_solver_pre_comp_refactor integer,
            cm_solver_pre_comp_droptol real,
            cm_solver_pre_comp_sweeps integer,
            cm_solver_pre_comp_sai_thres real,
            cm_solver_pre_app_type integer,
            cm_solver_pre_app_jacobi_iters integer,
            temp_init real,
            temp_final real,
            temp_mass real,
            temp_mode integer,
            temp_rate real,
            temp_freq integer,
            pressure real,
            pressure_mass real,
            compress integer,
            pressure_mode integer,
            remove_center_of_mass integer,
            debug_level integer,
            write_freq integer,
            traj_compress integer,
            traj_format integer,
            traj_title text,
            atom_info integer,
            atom_velocities integer,
            atom_forces integer,
            bond_info integer,
            angle_info integer,
            test_forces integer,
            molecule_analysis integer,
            freq_molecule_analysis integer,
            ignore integer,
            dipole_analysis integer,
            freq_dipole_analysis integer,
            diffusion_coefficient integer,
            freq_diffusion_coefficient integer,
            restrict_type integer,
            PRIMARY KEY (id)
        );

        CREATE TABLE system_properties(
            id integer,
            step integer,
            total_energy real,
            potential_energy real,
            kinetic_energy real,
            temperature real,
            volume real,
            pressure real,
            PRIMARY KEY (id, step)
        );

        CREATE TABLE potential(
            id integer,
            step integer,
            bond_energy real,
            atom_energy real,
            lone_pair_energy real,
            angle_energy real,
            coa_energy real,
            hydrogen_bond_energy real,
            torsion_energy real,
            conjugation_energy real,
            van_der_waals_energy real,
            coulombic_energy real,
            polarization_energy real,
            PRIMARY KEY (id, step)
        );

        CREATE TABLE trajectory(
            id integer,
            step integer,
            atom_id integer,
            position_x real,
            position_y real,
            position_z real,
            charge real,
            PRIMARY KEY (id, step, atom_id)
        );

        CREATE TABLE performance(
            id integer,
            step integer,
            time_total real,
            time_nbrs real,
            time_init real,
            time_bonded real,
            time_nonbonded real,
            time_cm real,
            time_cm_sort real,
            cm_solver_iters integer,
            time_cm_pre_comp real,
            time_cm_pre_app real,
            time_cm_solver_spmv real,
            time_cm_solver_vec_ops real,
            time_cm_solver_orthog real,
            time_cm_solver_tri_solve real,
            PRIMARY KEY (id, step)
        );
    """)

    conn.close()


if __name__ == '__main__':
    lib = cdll.LoadLibrary("libspuremd.so.1")

    setup_qmmm = lib.setup_qmmm
    setup_qmmm.argtypes = [c_int, c_char_p, POINTER(c_double),
            c_int, c_char_p, POINTER(c_double),
            POINTER(c_double), c_char_p, c_char_p]
    setup_qmmm.restype = c_void_p

    simulate = lib.simulate
    simulate.argtypes = [c_void_p]
    simulate.restype = c_int

    cleanup = lib.cleanup
    cleanup.argtypes = [c_void_p]
    cleanup.restype = c_int

    reset_qmmm = lib.reset_qmmm
    reset_qmmm.argtypes = [c_void_p, c_int, c_char_p, POINTER(c_double),
            c_int, c_char_p, POINTER(c_double),
            POINTER(c_double), c_char_p, c_char_p]
    reset_qmmm.restype = c_int

    CALLBACKFUNC = CFUNCTYPE(None, c_int, POINTER(ReaxAtom),
            POINTER(SimulationData))

    setup_callback = lib.setup_callback
    setup_callback.argtypes = [c_void_p, CALLBACKFUNC]
    setup_callback.restype = c_int

    set_control_parameter = lib.set_control_parameter
    set_control_parameter.argtypes = [c_void_p, c_char_p, POINTER(c_char_p)]
    set_control_parameter.restype = c_int

    get_atom_positions_qmmm = lib.get_atom_positions_qmmm
    get_atom_positions_qmmm.argtypes = [c_void_p, POINTER(c_double), POINTER(c_double)]
    get_atom_positions_qmmm.restype = c_int

    get_atom_forces_qmmm = lib.get_atom_forces_qmmm
    get_atom_forces_qmmm.argtypes = [c_void_p, POINTER(c_double), POINTER(c_double)]
    get_atom_forces_qmmm.restype = c_int

    get_atom_charges_qmmm = lib.get_atom_charges_qmmm
    get_atom_charges_qmmm.argtypes = [c_void_p, POINTER(c_double), POINTER(c_double)]
    get_atom_charges_qmmm.restype = c_int

    def get_simulation_step_results(num_atoms, atoms, data):
        print("{0:24.15f} {1:24.15f} {2:24.15f}".format(
            data[0].E_Tot, data[0].E_Kin, data[0].E_Pot))

    # bulk water
    sim_box = (c_double * 6)(40.299, 40.299, 40.299, 90.0, 90.0, 90.0)
    num_qm_atoms = 10
    num_mm_atoms = 10
    num_atoms = num_qm_atoms + num_mm_atoms
    qm_types = ''.join('{0:<2}'.format(s) for s in ['H', 'O', 'O', 'H', 'O', 'O', 'H', 'O', 'O', 'H']).encode('utf-8')
    # (x-position, y-position, z-position) atom tuples (units of Angstroms)
    qm_p = (c_double * (3 * num_qm_atoms))(5.690, 12.751, 11.651,
            4.760, 12.681, 11.281,
            5.800, 13.641, 12.091,
            15.551, 15.111, 7.030,
            14.981, 14.951, 7.840,
            14.961, 15.211, 6.230,
            17.431, 6.180, 8.560,
            17.761, 7.120, 8.560,
            17.941, 5.640, 9.220,
            11.351, 7.030, 7.170)
    mm_types = ''.join('{0:<2}'.format(s) for s in ['O', 'O', 'H', 'O', 'O', 'H', 'O', 'O', 'H', 'O']).encode('utf-8')
    # (x-position, y-position, z-position, charge) atom tuples (units of Angstroms / Coulombs)
    mm_p_q = (c_double * (4 * num_mm_atoms))(11.921, 7.810, 6.920, -2.0,
            10.751, 7.290, 7.930, 1.0,
            17.551, 6.070, 2.310, 1.0,
            17.431, 5.940, 1.320, -2.0,
            17.251, 5.260, 2.800, 1.0,
            7.680, 11.441, 10.231, 1.0,
            6.900, 11.611, 10.831, -2.0,
            8.020, 12.311, 9.871, 1.0,
            8.500, 7.980, 18.231, 1.0,
            8.460, 8.740, 18.881, -2.0)

    handle = setup_qmmm(c_int(num_qm_atoms), qm_types, qm_p,
            c_int(num_mm_atoms), mm_types, mm_p_q, sim_box,
            b"data/benchmarks/water/ffield.water",
            b"environ/control_water")

    ret = setup_callback(handle, CALLBACKFUNC(get_simulation_step_results))

    if ret != 0:
        print("[ERROR] setup_callback returned {0}".format(ret))

    keyword = b"nsteps"
    values = (c_char_p)(b"10")
    ret = set_control_parameter(handle, keyword, values)

    if ret != 0:
        print("[ERROR] set_control_parameter returned {0}".format(ret))

    keyword = b"charge_method"
    values = (c_char_p)(b"1")
    ret = set_control_parameter(handle, keyword, values)

    if ret != 0:
        print("[ERROR] set_control_parameter returned {0}".format(ret))

    print("{0:24}|{1:24}|{2:24}".format("Total Energy", "Kinetic Energy", "Potential Energy"))

    ret = simulate(handle)

    if ret != 0:
        print("[ERROR] simulate returned {0}".format(ret))

    qm_p = (c_double * (3 * num_qm_atoms))()
    mm_p = (c_double * (3 * num_mm_atoms))()
    ret = get_atom_positions_qmmm(handle, qm_p, mm_p)

    if ret != 0:
        print("[ERROR] get_atom_positions_qmmm returned {0}".format(ret))

    qm_f = (c_double * (3 * num_qm_atoms))()
    mm_f = (c_double * (3 * num_mm_atoms))()
    ret = get_atom_forces_qmmm(handle, qm_f, mm_f)

    if ret != 0:
        print("[ERROR] get_atom_forces_qmmm returned {0}".format(ret))

    qm_q = (c_double * num_qm_atoms)()
    mm_q = (c_double * num_mm_atoms)()
    ret = get_atom_charges_qmmm(handle, qm_q, mm_q)

    if ret != 0:
        print("[ERROR] get_atom_charges_qmmm returned {0}".format(ret))

    # silica
    sim_box = (c_double * 6)(36.477, 50.174, 52.110, 90.0, 90.0, 90.0)
    num_qm_atoms = 15
    num_mm_atoms = 15
    num_atoms = num_qm_atoms + num_mm_atoms
    qm_types = ''.join('{0:<2}'.format(s) for s in ['O', 'O', 'O', 'O', 'Si', 'O', 'Si', 'O', 'O', 'O', 'Si', 'O', 'Si', 'O', 'O']).encode('utf-8')
    # (x-position, y-position, z-position) atom tuples (units of Angstroms)
    qm_p = (c_double * (3 * num_qm_atoms))(56.987, 39.868, 41.795,
            32.795, 24.104, 25.968,
            26.543, 26.261, 36.254,
            27.616, 27.534, 24.459,
            26.560, 39.146, 32.281,
            54.035, 39.112, 23.745,
            54.425, 37.117, 32.278,
            29.979, 43.558, 21.696,
            38.008, 47.170, 27.275,
            48.769, 42.454, 20.461,
            57.113, 35.565, 31.366,
            26.458, 35.477, 23.522,
            52.299, 39.113, 26.519,
            55.789, 41.444, 30.466,
            45.752, 44.237, 10.521)
    mm_types = ''.join('{0:<2}'.format(s) for s in ['Si', 'O', 'Si', 'O', 'O', 'O', 'Si', 'O', 'Si', 'O', 'O', 'Si', 'O', 'O', 'Si']).encode('utf-8')
    # (x-position, y-position, z-position, charge) atom tuples (units of Angstroms / Coulombs)
    mm_p_q = (c_double * (4 * num_mm_atoms))(49.617, 42.379, 21.790, -1.0,
            26.736, 37.815, 33.119, -2.0,
            58.166, 5.488, 0.945, -1.0,
            36.655, 48.615, 25.664, -2.0,
            55.773, 36.249, 31.954, -2.0,
            53.905, 1.450, 22.720, -2.0,
            52.254, 35.399, 33.665, -1.0,
            32.196, 6.595, 23.147, -2.0,
            53.817, 8.586, 31.684, -1.0,
            56.164, 1.585, 23.550, -2.0,
            28.678, 7.208, 30.801, -2.0,
            58.824, 36.067, 40.036, -1.0,
            24.807, 6.365, 24.849, -2.0,
            51.299, 25.234, 29.146, -2.0,
            28.950, 11.190, 24.542, -1.0)

    ret = reset_qmmm(handle, c_int(num_qm_atoms), qm_types, qm_p,
            c_int(num_mm_atoms), mm_types, mm_p_q, sim_box,
            b"data/benchmarks/silica/ffield-bio",
            b"environ/control_silica")

    if ret != 0:
        print("[ERROR] reset_qmmm returned {0}".format(ret))

    keyword = b"nsteps"
    values = (c_char_p)(b"10")
    ret = set_control_parameter(handle, keyword, values)

    if ret != 0:
        print("[ERROR] set_control_parameter returned {0}".format(ret))

    keyword = b"charge_method"
    values = (c_char_p)(b"1")
    ret = set_control_parameter(handle, keyword, values)

    if ret != 0:
        print("[ERROR] set_control_parameter returned {0}".format(ret))

    print("\n{0:24}|{1:24}|{2:24}".format("Total Energy", "Kinetic Energy", "Potential Energy"))

    ret = simulate(handle)

    if ret != 0:
        print("[ERROR] simulate returned {0}".format(ret))

    qm_p = (c_double * (3 * num_qm_atoms))()
    mm_p = (c_double * (3 * num_mm_atoms))()
    ret = get_atom_positions_qmmm(handle, qm_p, mm_p)

    if ret != 0:
        print("[ERROR] get_atom_positions_qmmm returned {0}".format(ret))

    qm_f = (c_double * (3 * num_qm_atoms))()
    mm_f = (c_double * (3 * num_mm_atoms))()
    ret = get_atom_forces_qmmm(handle, qm_f, mm_f)

    if ret != 0:
        print("[ERROR] get_atom_forces_qmmm returned {0}".format(ret))

    qm_q = (c_double * num_qm_atoms)()
    mm_q = (c_double * num_mm_atoms)()
    ret = get_atom_charges_qmmm(handle, qm_q, mm_q)

    if ret != 0:
        print("[ERROR] get_atom_charges_qmmm returned {0}".format(ret))

    cleanup(handle)
