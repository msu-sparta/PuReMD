#!/bin/python3

from ctypes import *


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

    handle = setup(b"data/benchmarks/water/water_6540.pdb",
            b"data/benchmarks/water/ffield.water",
            b"environ/param.gpu.water")

    simulate(handle)

    cleanup(handle)
