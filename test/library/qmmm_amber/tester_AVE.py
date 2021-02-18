#!/bin/python3

from ctypes import c_int, c_double, c_char, c_char_p, c_void_p, \
        Structure, Union, POINTER, CFUNCTYPE, cdll
import numpy as np
import sqlite3 as sq3
import pandas as pd
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

    setup_qmmm = lib.setup_qmmm
    setup_qmmm.argtypes = [c_int, POINTER(c_int), POINTER(c_double),
            c_int, POINTER(c_int), POINTER(c_double), POINTER(c_double),
            c_char_p, c_char_p]
    setup_qmmm.restype = c_void_p

    simulate = lib.simulate
    simulate.argtypes = [c_void_p]
    simulate.restype = c_int

    cleanup = lib.cleanup
    cleanup.argtypes = [c_void_p]
    cleanup.restype = c_int

    reset_qmmm = lib.reset_qmmm
    reset_qmmm.argtypes = [c_void_p, c_int, POINTER(c_int), POINTER(c_double),
            c_int, POINTER(c_int), POINTER(c_double), POINTER(c_double),
            POINTER(c_double), c_char_p, c_char_p]
    reset_qmmm.restype = c_int

    CALLBACKFUNC = CFUNCTYPE(None, c_int, POINTER(ReaxAtom),
            POINTER(SimulationData))

    setup_callback = lib.setup_callback
    setup_callback.argtypes = [c_void_p, CALLBACKFUNC]
    setup_callback.restype = c_int

    set_output_enabled = lib.set_output_enabled
    set_output_enabled.argtypes = [c_void_p, c_int]
    set_output_enabled.restype = c_int

    set_control_parameter = lib.set_control_parameter
    set_control_parameter.argtypes = [c_void_p, c_char_p, POINTER(c_char_p)]
    set_control_parameter.restype = c_int

    get_atom_positions_qmmm = lib.get_atom_positions_qmmm
    get_atom_positions_qmmm.argtypes = [c_void_p, POINTER(c_double),
            POINTER(c_double)]
    get_atom_positions_qmmm.restype = c_int

    get_atom_forces_qmmm = lib.get_atom_forces_qmmm
    get_atom_forces_qmmm.argtypes = [c_void_p, POINTER(c_double),
            POINTER(c_double)]
    get_atom_forces_qmmm.restype = c_int

    get_atom_charges_qmmm = lib.get_atom_charges_qmmm
    get_atom_charges_qmmm.argtypes = [c_void_p, POINTER(c_double), POINTER(c_double)]
    get_atom_charges_qmmm.restype = c_int

    def get_simulation_step_results(num_atoms, atoms, data):
        print("{0:24.15f} {1:24.15f} {2:24.15f}".format(
            data[0].E_Tot, data[0].E_Kin, data[0].E_Pot))

    # data from Amber
    sim_box_info = (c_double * 6)(80.0, 80.0, 80.0, 90.0, 90.0, 90.0)
    num_qm_atoms = 14
    num_mm_atoms = 759
    num_atoms = num_qm_atoms + num_mm_atoms

    df = pd.read_csv('AVE/fort.3', sep='\s+', skiprows=[0,1,2,3,777,778],
            names=['Keyword', 'Num', 'Tag', 'P_x', 'P_y', 'P_z', 'Elem',
                'Tag2', 'Tag3', 'Q'],
            dtype={'Keyword': str, 'Num': np.int,
                'Tag': str, 'P_x': np.float64, 'P_y': np.float64,
                'P_z': np.float64, 'Elem': str,
                'Tag2': str, 'Tag3': str, 'Q': np.float64})

    types_s = df['Elem'].to_list()
    elems = {'C': 0, 'H': 1, 'O': 2}
    types = [elems[t] for t in types_s]
    p = df[['P_x', 'P_y', 'P_z']].values.flatten().tolist()
    q = df['Q'].to_list()

    qm_types = (c_int * num_qm_atoms)(*types[0:num_qm_atoms])
    qm_p = (c_double * (3 * num_qm_atoms))(*p[0:(3 * num_qm_atoms)])
    mm_types = (c_int * num_mm_atoms)(*types[num_qm_atoms:])
    p_q = []
    for i in range(num_mm_atoms):
        p_q.append(p[3 * (num_qm_atoms + i)])
        p_q.append(p[3 * (num_qm_atoms + i) + 1])
        p_q.append(p[3 * (num_qm_atoms + i) + 2])
        p_q.append(q[num_qm_atoms + i])
    mm_p_q = (c_double * (4 * num_mm_atoms))(*p_q)

    handle = setup_qmmm(c_int(num_qm_atoms), qm_types, qm_p,
            c_int(num_mm_atoms), mm_types, mm_p_q, sim_box_info,
            b"AVE/ffield", None)

    d = {
            b"simulation_name": (c_char_p)(b"AVE"),
            b"ensemble_type": (c_char_p)(b"0"),
            b"nsteps": (c_char_p)(b"0"),
            b"dt": (c_char_p)(b"0.25"),
            b"periodic_boundaries": (c_char_p)(b"1"),
            b"reposition_atoms": (c_char_p)(b"0"),
            b"reneighbor": (c_char_p)(b"1"),
            b"tabulate_long_range": (c_char_p)(b"0"),
            b"energy_update_freq": (c_char_p)(b"1"),
            b"vlist_buffer": (c_char_p)(b"2.5"),
            b"nbrhood_cutoff": (c_char_p)(b"5.0"),
            b"thb_cutoff": (c_char_p)(b"0.005"),
            b"hbond_cutoff": (c_char_p)(b"7.5"),
            b"bond_graph_cutoff": (c_char_p)(b"0.3"),
            b"charge_method": (c_char_p)(b"1"),
            b"cm_q_net": (c_char_p)(b"0.0"),
            b"cm_solver_type": (c_char_p)(b"2"),
            b"cm_solver_max_iters": (c_char_p)(b"200"),
            b"cm_solver_q_err": (c_char_p)(b"1.0e-14"),
            b"cm_solver_pre_comp_type": (c_char_p)(b"1"),
            }
    for keyword, values in d.items():
        ret = set_control_parameter(handle, keyword, values)
        if ret != 0:
            print("[ERROR] set_control_parameter returned {0}".format(ret))

    ret = setup_callback(handle, CALLBACKFUNC(get_simulation_step_results))
    if ret != 0:
        print("[ERROR] setup_callback returned {0}".format(ret))

    ret = set_output_enabled(handle, c_int(1))
    if ret != 0:
        print("[ERROR] set_output_enabled returned {0}".format(ret))

    print("{0:24}|{1:24}|{2:24}".format("Total Energy", "Kinetic Energy", "Potential Energy"))

    ret = simulate(handle)
    if ret != 0:
        print("[ERROR] simulate returned {0}".format(ret))

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

    print("\n{0:6}|{1:24}|{2:24}|{3:24}|{4:24}".format("i", "F_x", "F_y", "F_z", "Q"))
    for i in range(num_qm_atoms):
        print("{0:6d} {1:24.12f} {2:24.12f} {3:24.12f} {4:24.12f}".format(i + 1,
            qm_f[3 * i], qm_f[3 * i + 1], qm_f[3 * i + 2], qm_q[i]))
    for i in range(num_mm_atoms):
        print("{0:6d} {1:24.12f} {2:24.12f} {3:24.12f} {4:24.12f}".format(i + num_qm_atoms + 1,
            mm_f[3 * i], mm_f[3 * i + 1], mm_f[3 * i + 2], mm_q[i]))

    cleanup(handle)
