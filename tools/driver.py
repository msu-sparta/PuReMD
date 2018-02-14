#!/bin/python3

from ctypes import *


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


class ReaxListSelector(Union):
    _fields_ = [
            ("v", c_void_p),
            ("three_body_interaction_data", POINTER(ThreeBodyData)),
            ("bond_data", POINTER(BondData)),
            ("dbond_data", POINTER(DBondData)),
            ("dDelta_data", POINTER(DDeltaData)),
            ("far_neighbor_data", POINTER(FarNbrData)),
            ("near_neighbor_data", POINTER(NearNbrData)),
            ("hbond_data", POINTER(HBondData)),
            ]


class ReaxList(Structure):
    _fields_ = [
            ("n", c_int),
            ("total_intrs", c_int),
            ("index", POINTER(c_int)),
            ("end_index", POINTER(c_int)),
            ("max_intrs", POINTER(c_int)),
            ("select", ReaxListSelector),
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
            ]


class SimulationData(Structure):
    _fields_ = [
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
            ("tot_press", c_double * 3),
            ("timing", ReaxTiming),
            ]


class ReaxAtom(Structure):
    _fields_ = [
            ("name", c_char * 8),
            ("x", c_double * 3),
            ("v", c_double * 3),
            ("f", c_double * 3),
            ("q", c_double),
            ]


if __name__ == '__main__':
    lib = cdll.LoadLibrary("libspuremd.so.1")

    setup = lib.setup
    setup.argtypes = [c_char_p, c_char_p, c_char_p]
    setup.restype = c_void_p

    simulate = lib.simulate
    simulate.argtypes = [c_void_p]
    simulate.restype = c_int

    cleanup = lib.cleanup
    cleanup.argtypes = [c_void_p]
    cleanup.restype = c_int

    get_atoms = lib.get_atoms
    get_atoms.argtypes = [c_void_p]
    get_atoms.restype = POINTER(ReaxAtom)

    CALLBACKFUNC = CFUNCTYPE(None, POINTER(ReaxAtom),
            POINTER(SimulationData), POINTER(ReaxList))

    def get_simulation_step_results(atoms, data, lists):
        print("{0:24.15f} {1:24.15f} {2:24.15f}".format(
            data[0].E_Tot, data[0].E_Kin, data[0].E_Pot))

    setup_callback = lib.setup_callback
    setup_callback.restype = c_int

    handle = setup(b"data/benchmarks/water/water_6540.pdb",
            b"data/benchmarks/water/ffield.water",
            b"environ/param.gpu.water")

    ret = setup_callback(handle, CALLBACKFUNC(get_simulation_step_results))

    print("{0:24}|{1:24}|{2:24}".format("Total Energy", "Kinetic Energy", "Potential Energy"))
    ret = simulate(handle)

    atoms = get_atoms(handle)

    print()
    print("{0:9}|{1:24}|{2:24}|{3:24}|{4:24}".format("Atom Num", "x-Position", "y-Position", "z-Position", "Charge"))
    for i in range(10):
        print("{0:9d} {1:24.15f} {2:24.15f} {3:24.15f} {4:24.15f}".format(
            i + 1, atoms[i].x[0], atoms[i].x[1], atoms[i].x[2], atoms[i].q))

    cleanup(handle)
