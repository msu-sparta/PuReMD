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
            ("int_press", c_double * 3),
            ("ext_press", c_double * 3),
            ("kin_press", c_double),
            ("tot_press", c_double * 3),
            ("timing", ReaxTiming),
            ]


class ReaxAtom(Structure):
    _fields_ = [
            ("type", c_int),
            ("rel_map", c_int * 3),
            ("name", c_char * 9),
            ("x", c_double * 3),
            ("v", c_double * 3),
            ("f", c_double * 3),
            ("q", c_double),
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

    setup_qmmm = lib.setup_qmmm_
    setup_qmmm.argtypes = [c_int, POINTER(c_int),
            POINTER(c_double), POINTER(c_double), POINTER(c_double),
            c_int, POINTER(c_int), POINTER(c_double), POINTER(c_double),
            POINTER(c_double), POINTER(c_double), POINTER(c_double),
            c_char_p, c_char_p]
    setup_qmmm.restype = c_void_p

    simulate = lib.simulate
    simulate.argtypes = [c_void_p]
    simulate.restype = c_int

    cleanup = lib.cleanup
    cleanup.argtypes = [c_void_p]
    cleanup.restype = c_int

    reset_qmmm = lib.reset_qmmm_
    reset_qmmm.argtypes = [c_void_p, c_int, POINTER(c_int),
            POINTER(c_double), POINTER(c_double), POINTER(c_double),
            c_int, POINTER(c_int), POINTER(c_double), POINTER(c_double),
            POINTER(c_double), POINTER(c_double), POINTER(c_double),
            c_char_p, c_char_p]
    reset_qmmm.restype = c_int

    CALLBACKFUNC = CFUNCTYPE(None, c_int, POINTER(ReaxAtom),
            POINTER(SimulationData))

    setup_callback = lib.setup_callback
    setup_callback.argtypes = [c_void_p, CALLBACKFUNC]
    setup_callback.restype = c_int

    set_control_parameter = lib.set_control_parameter
    set_control_parameter.argtypes = [c_void_p, c_char_p, POINTER(c_char_p)]
    set_control_parameter.restype = c_int

    get_atom_positions_qmmm = lib.get_atom_positions_qmmm_
    get_atom_positions_qmmm.argtypes = [c_void_p,
            POINTER(c_double), POINTER(c_double), POINTER(c_double),
            POINTER(c_double), POINTER(c_double), POINTER(c_double)]
    get_atom_positions_qmmm.restype = c_int

    get_atom_forces_qmmm = lib.get_atom_forces_qmmm_
    get_atom_forces_qmmm.argtypes = [c_void_p,
            POINTER(c_double), POINTER(c_double), POINTER(c_double),
            POINTER(c_double), POINTER(c_double), POINTER(c_double)]
    get_atom_forces_qmmm.restype = c_int

    get_atom_charges_qmmm = lib.get_atom_charges_qmmm_
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
    qm_types = (c_int * num_qm_atoms)(2, 1, 1, 2, 1, 1, 2, 1, 1, 2)
    qm_p_x = (c_double * num_qm_atoms)(5.690, 4.760, 5.800, 15.551, 14.981, 14.961, 17.431, 17.761, 17.941, 11.351)
    qm_p_y = (c_double * num_qm_atoms)(12.751, 12.681, 13.641, 15.111, 14.951, 15.211, 6.180, 7.120, 5.640, 7.030)
    qm_p_z = (c_double * num_qm_atoms)(11.651, 11.281, 12.091, 7.030, 7.840, 6.230, 8.560, 8.560, 9.220, 7.170)
    mm_types = (c_int * num_qm_atoms)(1, 1, 2, 1, 1, 2, 1, 1, 2, 1)
    mm_p_x = (c_double * num_mm_atoms)(11.921, 10.751, 17.551, 17.431, 17.251, 7.680, 6.900, 8.020, 8.500, 8.460)
    mm_p_y = (c_double * num_mm_atoms)(7.810, 7.290, 6.070, 5.940, 5.260, 11.441, 11.611, 12.311, 7.980, 8.740)
    mm_p_z = (c_double * num_mm_atoms)(6.920, 7.930, 2.310, 1.320, 2.800, 10.231, 10.831, 9.871, 18.231, 18.881)
    mm_q = (c_double * num_mm_atoms)(-2.0, 1.0, 1.0, -2.0, 1.0, 1.0, -2.0, 1.0, 1.0, -2.0)

    handle = setup_qmmm(c_int(num_qm_atoms), qm_types, qm_p_x, qm_p_y, qm_p_z,
            c_int(num_mm_atoms), mm_types, mm_p_x, mm_p_y, mm_p_z, mm_q, sim_box,
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

    print("{0:24}|{1:24}|{2:24}".format("Total Energy", "Kinetic Energy", "Potential Energy"))

    ret = simulate(handle)

    if ret != 0:
        print("[ERROR] simulate returned {0}".format(ret))

    qm_p_x = (c_double * num_qm_atoms)()
    qm_p_y = (c_double * num_qm_atoms)()
    qm_p_z = (c_double * num_qm_atoms)()
    mm_p_x = (c_double * num_mm_atoms)()
    mm_p_y = (c_double * num_mm_atoms)()
    mm_p_z = (c_double * num_mm_atoms)()
    ret = get_atom_positions_qmmm(handle, qm_p_x, qm_p_y, qm_p_z,
            mm_p_x, mm_p_y, mm_p_z)

    if ret != 0:
        print("[ERROR] get_atom_positions_qmmm returned {0}".format(ret))

    qm_f_x = (c_double * num_qm_atoms)()
    qm_f_y = (c_double * num_qm_atoms)()
    qm_f_z = (c_double * num_qm_atoms)()
    mm_f_x = (c_double * num_mm_atoms)()
    mm_f_y = (c_double * num_mm_atoms)()
    mm_f_z = (c_double * num_mm_atoms)()
    ret = get_atom_forces_qmmm(handle, qm_f_x, qm_f_y, qm_f_z, mm_f_x, mm_f_y, mm_f_z)

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
    qm_types = (c_int * num_qm_atoms)(2, 2, 2, 2, 10, 2, 10, 2, 2, 10, 2, 10, 2, 2)
    qm_p_x = (c_double * num_qm_atoms)(56.987, 32.795, 26.543, 27.616, 26.560, 54.035, 54.425, 29.979, 38.008, 48.769, 57.113, 26.458, 52.299, 55.789, 45.752)
    qm_p_y = (c_double * num_qm_atoms)(39.868, 24.104, 26.261, 27.534, 39.146, 39.112, 37.117, 43.558, 47.170, 42.454, 35.565, 35.477, 39.113, 41.444, 44.237)
    qm_p_z = (c_double * num_qm_atoms)(41.795, 25.968, 36.254, 24.459, 32.281, 23.745, 32.278, 21.696, 27.275, 20.461, 31.366, 23.522, 26.519, 30.466, 10.521)
    mm_types = (c_int * num_qm_atoms)(10, 2, 10, 2, 2, 2, 10, 2, 10, 2, 2, 10, 2, 2, 10)
    mm_p_x = (c_double * num_mm_atoms)(49.617, 26.736, 58.166, 36.655, 55.773, 53.905, 52.254, 32.196, 53.817, 56.164, 28.678, 58.824, 24.807, 51.299, 28.950)
    mm_p_y = (c_double * num_mm_atoms)(42.379, 37.815, 5.488, 48.615, 36.249, 1.450, 35.399, 6.595, 8.586, 1.585, 7.208, 36.067, 6.365, 25.234, 11.190)
    mm_p_z = (c_double * num_mm_atoms)(21.790, 33.119, 0.945, 25.664, 31.954, 22.720, 33.665, 23.147, 31.684, 23.550, 30.801, 40.036, 24.849, 29.146, 24.542)
    mm_q = (c_double * num_mm_atoms)(-1.0, -2.0, -1.0, -2.0, -2.0, -2.0, -1.0, -2.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0, -1.0)

    ret = reset_qmmm(handle, c_int(num_qm_atoms), qm_types, qm_p_x, qm_p_y, qm_p_z,
            c_int(num_mm_atoms), mm_types, mm_p_x, mm_p_y, mm_p_z, mm_q, sim_box,
            b"data/benchmarks/silica/ffield-bio",
            b"environ/control_silica")

    print("\n{0:24}|{1:24}|{2:24}".format("Total Energy", "Kinetic Energy", "Potential Energy"))

    ret = simulate(handle)

    if ret != 0:
        print("[ERROR] simulate returned {0}".format(ret))

    qm_p_x = (c_double * num_qm_atoms)()
    qm_p_y = (c_double * num_qm_atoms)()
    qm_p_z = (c_double * num_qm_atoms)()
    mm_p_x = (c_double * num_mm_atoms)()
    mm_p_y = (c_double * num_mm_atoms)()
    mm_p_z = (c_double * num_mm_atoms)()
    ret = get_atom_positions_qmmm(handle, qm_p_x, qm_p_y, qm_p_z,
            mm_p_x, mm_p_y, mm_p_z)

    if ret != 0:
        print("[ERROR] get_atom_positions_qmmm returned {0}".format(ret))

    qm_f_x = (c_double * num_qm_atoms)()
    qm_f_y = (c_double * num_qm_atoms)()
    qm_f_z = (c_double * num_qm_atoms)()
    mm_f_x = (c_double * num_mm_atoms)()
    mm_f_y = (c_double * num_mm_atoms)()
    mm_f_z = (c_double * num_mm_atoms)()
    ret = get_atom_forces_qmmm(handle, qm_f_x, qm_f_y, qm_f_z, mm_f_x, mm_f_y, mm_f_z)

    if ret != 0:
        print("[ERROR] get_atom_forces_qmmm returned {0}".format(ret))

    qm_q = (c_double * num_qm_atoms)()
    mm_q = (c_double * num_mm_atoms)()
    ret = get_atom_charges_qmmm(handle, qm_q, mm_q)

    if ret != 0:
        print("[ERROR] get_atom_charges_qmmm returned {0}".format(ret))

    cleanup(handle)
