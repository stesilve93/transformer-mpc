import numpy as np
from scipy.linalg import expm
from Asteroid_scenario.dynamics.polyhedron_model import ROP2BP_dynamics


def jacobian_2(
    x_k_ctrl,
    Omega,
    vertices,
    faces,
    edges,
    edge_faces,
    face_normals,
    face_centroids,
    rho,
    G,
):
    """
    Computes the Jacobian of the dynamics ROP2BP in x_k_ctrl.
    """
    x_k_ctrl = np.asarray(x_k_ctrl, dtype=float).reshape(
        6,
    )

    # Permutation Matrix: ctrl -> dyn
    idx = np.array([0, 3, 1, 4, 2, 5], dtype=int)
    P = np.eye(6)[idx, :]

    x_k_dyn = P @ x_k_ctrl

    def f(x):
        return ROP2BP_dynamics(
            0.0,
            x,
            Omega,
            vertices,
            faces,
            edges,
            edge_faces,
            face_normals,
            face_centroids,
            rho,
            G,
        )

    n = 6
    A_dyn = np.zeros((n, n), dtype=float)
    fx = f(x_k_dyn)

    eps = np.finfo(float).eps
    for i in range(n):
        hi = np.sqrt(eps) * max(1.0, abs(x_k_dyn[i]))
        dx = np.zeros(n, dtype=float)
        dx[i] = hi
        f1 = f(x_k_dyn + dx)
        A_dyn[:, i] = (f1 - fx) / hi

    # Change order: ctrl -> dyn
    A_ctrl = P.T @ A_dyn @ P
    return A_ctrl


# ----------------------------------------------------------
# Linear matrices A and B (discretized) along the trajectory
# ----------------------------------------------------------
def linear_matrices(
    xx_ref,
    tt_vect,
    Omega,
    vertices,
    faces,
    edges,
    edge_faces,
    face_normals,
    face_centroids,
    rho,
    G,
):
    """
    - xx_ref: (N, 6) reference trajectory (order ctrl)
    - tt_vect: (N,) times
    Discretization: A_k = expm(A_ct * dt)
    Impulsive control at the beginning of the step: B_k = A_k @ S
    """
    xx_ref = np.asarray(xx_ref, dtype=float)
    tt_vect = np.asarray(tt_vect, dtype=float).reshape(
        -1,
    )
    N = xx_ref.shape[0]
    assert xx_ref.shape[1] == 6, "xx_ref must have 6 columns"
    assert tt_vect.shape[0] == N, "tt_vect and xx_ref must have the same length N"

    A_vect = np.zeros((6, 6, N - 1), dtype=float)
    B_vect = np.zeros((6, 3, N - 1), dtype=float)
    S = np.vstack([np.zeros((3, 3)), np.eye(3)])  # (6x3)

    for i in range(N - 1):
        A_k_ct = jacobian_2(
            xx_ref[i, :],
            Omega,
            vertices,
            faces,
            edges,
            edge_faces,
            face_normals,
            face_centroids,
            rho,
            G,
        )
        dt = float(tt_vect[i + 1] - tt_vect[i])

        # Discretization of A
        A_k = expm(A_k_ct * dt)

        A_vect[:, :, i] = A_k
        B_vect[:, :, i] = A_k @ S

    return A_vect, B_vect


# ----------------------------------------------------------
# Conversion in adimensional
# ----------------------------------------------------------
def convert_mat_to_adimensional(xx_ref, A_vect, B_vect, LU, VU, CU):
    """
    Conversion A_vect e B_vect in adimensional
    """
    xx_ref = np.asarray(xx_ref, dtype=float)
    N = xx_ref.shape[0]

    A_vect_AD = np.zeros((6, 6, N - 1), dtype=float)
    B_vect_AD = np.zeros((6, 3, N - 1), dtype=float)

    S = np.block(
        [
            [LU * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), VU * np.eye(3)],
        ]
    )
    S_inv = np.block(
        [
            [(1.0 / LU) * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), (1.0 / VU) * np.eye(3)],
        ]
    )

    for i in range(N - 1):
        A_k_AD = S_inv @ A_vect[:, :, i] @ S
        A_vect_AD[:, :, i] = A_k_AD

        # S_inv --> [6, 6]
        # B_vect --> [6,3,288]
        # CU --> float
        B_k_AD = (S_inv @ B_vect[:, :, i]) * CU
        B_vect_AD[:, :, i] = B_k_AD

    return A_vect_AD, B_vect_AD


# ----------------------------------------------------------
# Linear Propagation
# ----------------------------------------------------------
def linear_propagation(x0, Uvec, time_instant_index, N_prop, A_vect_AD, B_vect_AD):
    """
    Propagation of the state using the linearized adimensional model.

    x0        : (6,)
    Uvec      : (3, N_prop)  adimensional commands
    time_idx  : 1-based
    N_prop    : number of commands
    A_vect_AD : (6,6, N-1)
    B_vect_AD : (6,3, N-1)

    Return:
      traj_lin : (N_prop+1, 6)  [include x0, then x1..x_{N_prop}]
    """
    x = np.asarray(x0, dtype=float).reshape(
        6,
    )
    Uvec = np.asarray(Uvec, dtype=float)
    assert Uvec.shape == (3, N_prop), (
        f"Uvec deve essere (3, N_prop), trovato {Uvec.shape}"
    )

    traj_lin = np.zeros((N_prop + 1, 6), dtype=float)
    traj_lin[0, :] = x

    start_idx = int(time_instant_index) - 1  # 1-based -> 0-based
    for j in range(N_prop):
        idx = start_idx + j
        if idx < 0 or idx >= A_vect_AD.shape[2]:
            raise IndexError(
                f"Indice A/B fuori range: idx={idx}, max={A_vect_AD.shape[2] - 1}"
            )
        A_k = A_vect_AD[:, :, idx]
        B_k = B_vect_AD[:, :, idx]
        u_j = Uvec[:, j]
        x = A_k @ x + B_k @ u_j
        traj_lin[j + 1, :] = x

    return traj_lin
