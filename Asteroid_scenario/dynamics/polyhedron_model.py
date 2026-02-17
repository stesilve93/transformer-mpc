import numpy as np
from scipy.integrate import solve_ivp
from numba import njit, prange


def model_opening(filename):
    vertices, faces = [], []
    with open(filename, "r") as file:
        for line in file:
            if line.startswith("v "):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face = [int(p.split("//")[0]) if "//" in p else int(p) for p in parts]
                faces.append(face)
    return np.array(vertices), np.array(faces) - 1


def extract_unique_edges(faces):
    all_edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(all_edges, axis=1), axis=0)


def build_face_edge_map(faces, edges):
    edge_faces = -np.ones((edges.shape[0], 2), dtype=int)
    for i in range(edges.shape[0]):
        count = 0
        for j in range(faces.shape[0]):
            face = faces[j]
            if np.isin(edges[i], face).all():
                edge_faces[i, count] = j
                count += 1
                if count == 2:
                    break
    return edge_faces


def preprocess_geometry(vertices, faces):
    face_normals, face_centroids = [], []
    for face in faces:
        P1, P2, P3 = vertices[face]
        v1, v2 = P2 - P1, P3 - P1
        n_raw = np.cross(v2, v1)
        norm_n = np.linalg.norm(n_raw)
        n_f = (n_raw / norm_n) if norm_n else np.zeros(3)
        if np.dot(n_raw, P1) < 0:
            n_f = -n_f
        face_normals.append(n_f)
        face_centroids.append((P1 + P2 + P3) / 3)
    return np.array(face_normals), np.array(face_centroids)


@njit(parallel=True)
def compute_polyhedral_acc(
    vertices,
    faces,
    edges,
    edge_faces,
    face_normals,
    face_centroids,
    rho,
    G,
    field_point,
):
    sum_1 = np.zeros(3)
    for i in prange(faces.shape[0]):
        face = faces[i]
        P1, P2, P3 = vertices[face]
        n_f = face_normals[i]
        F_f = np.outer(n_f, n_f)
        r1, r2, r3 = P1 - field_point, P2 - field_point, P3 - field_point
        num = np.dot(r1, np.cross(r2, r3))
        den = (
            np.linalg.norm(r1) * np.linalg.norm(r2) * np.linalg.norm(r3)
            + np.dot(r1, r3) * np.linalg.norm(r1)
            + np.dot(r3, r1) * np.linalg.norm(r2)
            + np.dot(r1, r2) * np.linalg.norm(r3)
        )
        omega_f = 2.0 * np.arctan2(num, den)
        sum_1 += F_f @ (P1 - field_point) * omega_f

    sum_2 = np.zeros(3)
    for i in prange(edges.shape[0]):
        e = edges[i]
        f1, f2 = edge_faces[i]
        if f1 < 0 or f2 < 0:
            continue
        v0, v1_pt = vertices[e[0]], vertices[e[1]]
        r_e = v0 - field_point
        l1, l2 = np.linalg.norm(r_e), np.linalg.norm(v1_pt - field_point)
        e_vec = v1_pt - v0
        e_len = np.linalg.norm(e_vec)
        if l1 + l2 - e_len <= 0:
            continue
        L_e = np.log((l1 + l2 + e_len) / (l1 + l2 - e_len))
        e_dir = e_vec / e_len
        n1, n2 = face_normals[f1], face_normals[f2]
        c1, c2 = face_centroids[f1], face_centroids[f2]
        edge_mid = (v0 + v1_pt) / 2.0
        n_e1 = np.cross(n1, e_dir)
        n_e1 /= np.linalg.norm(n_e1) if np.linalg.norm(n_e1) else 1
        if np.dot(n_e1, edge_mid - c1) < 0:
            n_e1 = -n_e1
        n_e2 = np.cross(n2, e_dir)
        n_e2 /= np.linalg.norm(n_e2) if np.linalg.norm(n_e2) else 1
        if np.dot(n_e2, edge_mid - c2) < 0:
            n_e2 = -n_e2
        E_e = np.outer(n1, n_e1) + np.outer(n2, n_e2)
        sum_2 += E_e @ r_e * L_e

    return G * rho * (sum_1 - sum_2)


def ROP2BP_dynamics(
    t,
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
):
    field_point = x[[0, 2, 4]]
    acc = compute_polyhedral_acc(
        vertices,
        faces,
        edges,
        edge_faces,
        face_normals,
        face_centroids,
        rho,
        G,
        field_point,
    )
    dxdt = np.zeros(6)
    dxdt[0] = x[1]
    dxdt[1] = 2 * Omega * x[3] + Omega**2 * x[0] + acc[0]
    dxdt[2] = x[3]
    dxdt[3] = -2 * Omega * x[1] + Omega**2 * x[2] + acc[1]
    dxdt[4] = x[5]
    dxdt[5] = acc[2]
    return dxdt


def dynamics_propagator_body_fast(
    x0,
    time_vector,
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
    x0_new = [x0[0], x0[3], x0[1], x0[4], x0[2], x0[5]]
    sol = solve_ivp(
        lambda t, x: ROP2BP_dynamics(
            t,
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
        ),
        [time_vector[0], time_vector[-1]],
        x0_new,
        t_eval=time_vector,
        method="RK45",
        rtol=1e-9,
        atol=1e-6,
    )
    xx = sol.y.T
    return sol.t, np.column_stack(
        (xx[:, 0], xx[:, 2], xx[:, 4], xx[:, 1], xx[:, 3], xx[:, 5])
    )

# General dynamics (no spin along z axis) -----------------------------------------
def ROP2BP_dynamics_general(
    t,
    x,
    omega,        # np.array shape (3,), constant angular velocity vector ω
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
    Dynamics in the rotating frame with constant angular velocity vector omega:

        r¨ + 2 ω × r˙ + ω × (ω × r) = a_T

    Here a_T = a_G (polyhedral gravity).
    """

    # Position and velocity in body-fixed frame
    r = x[[0, 2, 4]]  # [x, y, z]
    v = x[[1, 3, 5]]  # [vx, vy, vz]

    # Gravitational acceleration from polyhedral model (a_G)
    acc = compute_polyhedral_acc(
        vertices,
        faces,
        edges,
        edge_faces,
        face_normals,
        face_centroids,
        rho,
        G,
        r,
    )

    # Rotating-frame terms:
    # Coriolis:      -2 ω × v
    # Centrifugal:   -ω × (ω × r)
    a_coriolis = -2.0 * np.cross(omega, v)
    a_centrifugal = -np.cross(omega, np.cross(omega, r))

    # Total acceleration in rotating frame
    a_total = acc + a_coriolis + a_centrifugal

    # Build state derivative
    dxdt = np.zeros(6)
    dxdt[0] = v[0]
    dxdt[2] = v[1]
    dxdt[4] = v[2]

    dxdt[1] = a_total[0]
    dxdt[3] = a_total[1]
    dxdt[5] = a_total[2]

    return dxdt



def dynamics_propagator_body_fast_general(
    x0,
    time_vector,
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
    x0_new = [x0[0], x0[3], x0[1], x0[4], x0[2], x0[5]]
    sol = solve_ivp(
        lambda t, x: ROP2BP_dynamics_general(
            t,
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
        ),
        [time_vector[0], time_vector[-1]],
        x0_new,
        t_eval=time_vector,
        method="RK45",
        rtol=1e-9,
        atol=1e-6,
    )
    xx = sol.y.T
    return sol.t, np.column_stack(
        (xx[:, 0], xx[:, 2], xx[:, 4], xx[:, 1], xx[:, 3], xx[:, 5])
    )
