import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



# Geometry

BF, GF, EF = 165, 522, 215
DA, AE, CD, HC, GH = 237, 100, 506, 230, 337
AB = 170

EG = EF + GF
DH = CD + HC

xA, yA = 0.0, 0.0
xB, yB = AB, 0.0

omega2 = 2 * np.pi
g = 9.81

theta2_vals = np.linspace(np.deg2rad(40), np.deg2rad(130), 500)


# Kinematic core

def wrap_to_near(angle, ref):
    return ref + np.arctan2(np.sin(angle - ref), np.cos(angle - ref))

def loop1(vars, theta2):
    th3, th4 = vars
    eq1 = xB + BF*np.cos(theta2) - EF*np.cos(th3) - AE*np.cos(th4) - xA
    eq2 = yB + BF*np.sin(theta2) - EF*np.sin(th3) + AE*np.sin(th4) - yA
    return [eq1, eq2]

def loop2(vars, xD, yD, xG, yG):
    th5, th6 = vars
    eq1 = xD + DH*np.cos(th5) - xG - GH*np.cos(th6)
    eq2 = yD + DH*np.sin(th5) - yG - GH*np.sin(th6)
    return [eq1, eq2]

def simulate(theta_range):
    nodes = {name: [] for name in ['A', 'B', 'F', 'E', 'D', 'G', 'H', 'C']}
    angles = {name: [] for name in ['theta2', 'theta3', 'theta4', 'theta5', 'theta6']}

    guess1 = [np.deg2rad(80), np.deg2rad(120)]
    guess2 = [np.deg2rad(40), np.deg2rad(15)]

    for th2 in theta_range:
        th3, th4 = fsolve(loop1, guess1, args=(th2,))
        th3 = wrap_to_near(th3, guess1[0])
        th4 = wrap_to_near(th4, guess1[1])
        guess1 = [th3, th4]

        xF = xB + BF*np.cos(th2)
        yF = yB + BF*np.sin(th2)

        xE = xA + AE*np.cos(th4)
        yE = yA - AE*np.sin(th4)

        xD = xA - DA*np.cos(th4)
        yD = yA + DA*np.sin(th4)

        xG = xE + EG*np.cos(th3)
        yG = yE + EG*np.sin(th3)

        th5, th6 = fsolve(loop2, guess2, args=(xD, yD, xG, yG))
        th5 = wrap_to_near(th5, guess2[0])
        th6 = wrap_to_near(th6, guess2[1])
        guess2 = [th5, th6]

        xH = xD + DH*np.cos(th5)
        yH = yD + DH*np.sin(th5)

        xC = xD + (CD/DH)*(xH - xD)
        yC = yD + (CD/DH)*(yH - yD)

        nodes['A'].append((xA, yA))
        nodes['B'].append((xB, yB))
        nodes['F'].append((xF, yF))
        nodes['E'].append((xE, yE))
        nodes['D'].append((xD, yD))
        nodes['G'].append((xG, yG))
        nodes['H'].append((xH, yH))
        nodes['C'].append((xC, yC))

        angles['theta2'].append(th2)
        angles['theta3'].append(th3)
        angles['theta4'].append(th4)
        angles['theta5'].append(th5)
        angles['theta6'].append(th6)

    for k in nodes:
        nodes[k] = np.array(nodes[k], dtype=float)

    for k in angles:
        angles[k] = np.unwrap(np.array(angles[k], dtype=float))

    return nodes, angles

nodes, angles = simulate(theta2_vals)

t = (theta2_vals - theta2_vals[0]) / omega2

vel = {}
acc = {}

for k in nodes:
    vel[k] = np.gradient(nodes[k], t, axis=0)
    acc[k] = np.gradient(vel[k], t, axis=0)

omega = {}
alpha = {}

for k in angles:
    omega[k] = np.gradient(angles[k], t)
    alpha[k] = np.gradient(omega[k], t)

omega['theta4'] = -omega['theta4']
alpha['theta4'] = -alpha['theta4']

ratio_C = CD / DH
nodes['C'] = nodes['D'] + ratio_C * (nodes['H'] - nodes['D'])
vel['C'] = vel['D'] + ratio_C * (vel['H'] - vel['D'])
acc['C'] = acc['D'] + ratio_C * (acc['H'] - acc['D'])

t_plot = t.copy()

vel_plot = {}
acc_plot = {}
omega_plot = {}
alpha_plot = {}

for k in vel:
    vel_plot[k] = vel[k][::-1]
    acc_plot[k] = acc[k][::-1]

for k in omega:
    omega_plot[k] = omega[k][::-1]
    alpha_plot[k] = alpha[k][::-1]


# Instant-query kinematics

def interp_angle_array(th, arr):
    return np.interp(th, theta2_vals, arr)

def kinematics_from_dataset(theta_deg):
    th = np.deg2rad(theta_deg)

    vel_i = {}
    acc_i = {}
    nodes_i = {}
    omega_i = {}
    alpha_i = {}

    for k in nodes:
        x = np.interp(th, theta2_vals, nodes[k][:, 0])
        y = np.interp(th, theta2_vals, nodes[k][:, 1])
        nodes_i[k] = np.array([x, y], dtype=float)

        vx = np.interp(th, theta2_vals, vel[k][:, 0])
        vy = np.interp(th, theta2_vals, vel[k][:, 1])
        vel_i[k] = np.array([vx, vy], dtype=float)

        axx = np.interp(th, theta2_vals, acc[k][:, 0])
        ayy = np.interp(th, theta2_vals, acc[k][:, 1])
        acc_i[k] = np.array([axx, ayy], dtype=float)

    nodes_i['C'] = nodes_i['D'] + ratio_C * (nodes_i['H'] - nodes_i['D'])
    vel_i['C'] = vel_i['D'] + ratio_C * (vel_i['H'] - vel_i['D'])
    acc_i['C'] = acc_i['D'] + ratio_C * (acc_i['H'] - acc_i['D'])

    for k in omega:
        omega_i[k] = np.interp(th, theta2_vals, omega[k])
        alpha_i[k] = np.interp(th, theta2_vals, alpha[k])

    return nodes_i, vel_i, acc_i, omega_i, alpha_i

def initial_guess_from_dataset(theta2):
    th = np.clip(theta2, theta2_vals[0], theta2_vals[-1])
    g1 = [
        interp_angle_array(th, angles['theta3']),
        interp_angle_array(th, angles['theta4'])
    ]
    g2 = [
        interp_angle_array(th, angles['theta5']),
        interp_angle_array(th, angles['theta6'])
    ]
    return g1, g2

def solve_angle(theta2):
    guess1, guess2 = initial_guess_from_dataset(theta2)

    th3, th4 = fsolve(loop1, guess1, args=(theta2,))
    th3 = wrap_to_near(th3, guess1[0])
    th4 = wrap_to_near(th4, guess1[1])

    xF = xB + BF*np.cos(theta2)
    yF = yB + BF*np.sin(theta2)

    xE = xA + AE*np.cos(th4)
    yE = yA - AE*np.sin(th4)

    xD = xA - DA*np.cos(th4)
    yD = yA + DA*np.sin(th4)

    xG = xE + EG*np.cos(th3)
    yG = yE + EG*np.sin(th3)

    th5, th6 = fsolve(loop2, guess2, args=(xD, yD, xG, yG))
    th5 = wrap_to_near(th5, guess2[0])
    th6 = wrap_to_near(th6, guess2[1])

    xH = xD + DH*np.cos(th5)
    yH = yD + DH*np.sin(th5)

    xC = xD + (CD/DH)*(xH - xD)
    yC = yD + (CD/DH)*(yH - yD)

    nodes_out = {
        'A': np.array([xA, yA], dtype=float),
        'B': np.array([xB, yB], dtype=float),
        'F': np.array([xF, yF], dtype=float),
        'E': np.array([xE, yE], dtype=float),
        'D': np.array([xD, yD], dtype=float),
        'G': np.array([xG, yG], dtype=float),
        'H': np.array([xH, yH], dtype=float),
        'C': np.array([xC, yC], dtype=float)
    }

    angles_out = {
        'theta2': theta2,
        'theta3': th3,
        'theta4': th4,
        'theta5': th5,
        'theta6': th6
    }

    return nodes_out, angles_out

def angle_difference(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))

def kinematics(theta_deg):
    th = np.deg2rad(theta_deg)
    dth = 1e-4
    dt = dth / omega2

    nodes_m, ang_m = solve_angle(th - dth)
    nodes_0, ang_0 = solve_angle(th)
    nodes_p, ang_p = solve_angle(th + dth)

    vel_out = {}
    acc_out = {}
    omega_i = {}
    alpha_i = {}

    for k in nodes_0:
        pm = nodes_m[k]
        p0 = nodes_0[k]
        pp = nodes_p[k]
        vel_out[k] = (pp - pm) / (2*dt)
        acc_out[k] = (pp - 2*p0 + pm) / (dt**2)

    vel_out['C'] = vel_out['D'] + ratio_C * (vel_out['H'] - vel_out['D'])
    acc_out['C'] = acc_out['D'] + ratio_C * (acc_out['H'] - acc_out['D'])

    for k in ang_0:
        dm = angle_difference(ang_0[k], ang_m[k])
        dp = angle_difference(ang_p[k], ang_0[k])
        omega_i[k] = (dp + dm) / (2*dt)
        alpha_i[k] = (dp - dm) / (dt**2)

    omega_i['theta4'] = -omega_i['theta4']
    alpha_i['theta4'] = -alpha_i['theta4']

    return nodes_0, vel_out, acc_out, omega_i, alpha_i

def interpolate_point(nodes_in, vel_in, acc_in, P, Q, s):
    pos = (1 - s) * nodes_in[P] + s * nodes_in[Q]
    v = (1 - s) * vel_in[P] + s * vel_in[Q]
    a = (1 - s) * acc_in[P] + s * acc_in[Q]
    return pos, v, a


# Plotting / Animation

def plot_all_trajectories():
    plt.figure(figsize=(9, 8))
    for k in nodes:
        plt.plot(nodes[k][:, 0], nodes[k][:, 1], label=f'{k}')
        plt.scatter(nodes[k][0, 0], nodes[k][0, 1], s=25)
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.title("Trajectory of All Nodes")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_important_trajectories():
    plt.figure(figsize=(9, 8))
    important_points = ['F', 'G', 'H', 'C']
    for k in important_points:
        plt.plot(nodes[k][:, 0], nodes[k][:, 1], linewidth=2, label=f'Trajectory of {k}')
        plt.scatter(nodes[k][0, 0], nodes[k][0, 1], s=40)
        plt.scatter(nodes[k][-1, 0], nodes[k][-1, 1], s=40)
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.title("Trajectory of Important Points")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_speed_all_nodes():
    plt.figure(figsize=(10, 6))
    for k in nodes:
        speed = np.linalg.norm(vel_plot[k], axis=1)
        plt.plot(t_plot, speed, label=f'|v_{k}|')
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [mm/s]")
    plt.title("Velocity Magnitude of All Nodes")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_acc_all_nodes():
    plt.figure(figsize=(10, 6))
    for k in nodes:
        acc_mag = np.linalg.norm(acc_plot[k], axis=1)
        plt.plot(t_plot, acc_mag, label=f'|a_{k}|')
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [mm/s²]")
    plt.title("Acceleration Magnitude of All Nodes")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_omega_links():
    plt.figure(figsize=(10, 6))
    for k in omega:
        plt.plot(t_plot, omega_plot[k], label=f'ω_{k}')
    plt.xlabel("Time [s]")
    plt.ylabel("Angular velocity [rad/s]")
    plt.title("Angular Velocity of Links")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_alpha_links():
    plt.figure(figsize=(10, 6))
    for k in alpha:
        plt.plot(t_plot, alpha_plot[k], label=f'α_{k}')
    plt.xlabel("Time [s]")
    plt.ylabel("Angular acceleration [rad/s²]")
    plt.title("Angular Acceleration of Links")
    plt.legend()
    plt.grid(True)
    plt.show()

def animate_mechanism():
    fig, ax = plt.subplots(figsize=(8, 8))
    all_x = np.concatenate([nodes[k][:, 0] for k in nodes])
    all_y = np.concatenate([nodes[k][:, 1] for k in nodes])
    margin = 50
    ax.set_xlim(all_x.min()-margin, all_x.max()+margin)
    ax.set_ylim(all_y.min()-margin, all_y.max()+margin)
    ax.set_aspect('equal'); ax.grid(True)
    ax.set_title("Hart Inversor Mechanism Animation")

    (line_AB,) = ax.plot([], [], 'o-', lw=2)
    (line_BF,) = ax.plot([], [], 'o-', lw=2)
    (line_EG,) = ax.plot([], [], 'o-', lw=2)
    (line_AE,) = ax.plot([], [], 'o-', lw=2)
    (line_AD,) = ax.plot([], [], 'o-', lw=2)
    (line_DH,) = ax.plot([], [], 'o-', lw=2)
    (line_GH,) = ax.plot([], [], 'o-', lw=2)
    (points,) = ax.plot([], [], 'ko', ms=5)
    (trace_C,) = ax.plot([], [], 'r.', ms=2)

    labels = {name: ax.text(0, 0, name) for name in nodes}

    def set_line(line, P, Q):
        line.set_data([P[0], Q[0]], [P[1], Q[1]])

    def update(frame):
        pos = {k: nodes[k][frame] for k in nodes}
        set_line(line_AB, pos['A'], pos['B'])
        set_line(line_BF, pos['B'], pos['F'])
        set_line(line_EG, pos['E'], pos['G'])
        set_line(line_AE, pos['A'], pos['E'])
        set_line(line_AD, pos['A'], pos['D'])
        set_line(line_DH, pos['D'], pos['H'])
        set_line(line_GH, pos['G'], pos['H'])

        xs = [pos[k][0] for k in nodes]; ys = [pos[k][1] for k in nodes]
        points.set_data(xs, ys)
        trace_C.set_data(nodes['C'][:frame, 0], nodes['C'][:frame, 1])

        for name, P in pos.items():
            labels[name].set_position((P[0]+8, P[1]+8))

        return (line_AB, line_BF, line_EG, line_AE, line_AD, line_DH, line_GH,
                points, trace_C, *labels.values())

    ani = FuncAnimation(fig, update, frames=len(theta2_vals), interval=30, blit=True)
    plt.show()
    return ani


# Dynamics

rho = 1.0  # kg/m

def link_mass_and_inertia(L_mm):
    L_m = L_mm / 1000.0
    m = rho * L_m
    I = (1.0 / 12.0) * m * L_m**2
    return m, I

L2_mm, L3_mm, L4_mm, L5_mm, L6_mm = BF, EG, DA + AE, DH, GH

m2, I2 = link_mass_and_inertia(L2_mm)
m3, I3 = link_mass_and_inertia(L3_mm)
m4, I4 = link_mass_and_inertia(L4_mm)
m5, I5 = link_mass_and_inertia(L5_mm)
m6, I6 = link_mass_and_inertia(L6_mm)

# link 2 moment of inertia about point B
I2_B = (1.0 / 3.0) * m2 * (L2_mm / 1000.0)**2

_COLS = {
    'A': (0, 1),
    'B': (2, 3),
    'F': (4, 5),
    'E': (6, 7),
    'D': (8, 9),
    'G': (10, 11),
    'H': (12, 13)
}

COL_M2 = 14
N_UNKNOWN = 15

def _add_force_eq(A_mat, b_vec, row_fx, row_fy, joints_list, m_link, acom_link):
    for _, cx, _, sign in joints_list:
        A_mat[row_fx, cx] += sign
    b_vec[row_fx] = m_link * acom_link[0]

    for _, _, cy, sign in joints_list:
        A_mat[row_fy, cy] += sign
    b_vec[row_fy] = m_link * acom_link[1] + m_link * g

def _add_moment_eq(A_mat, b_vec, points, row_m, joints_list, ref_pt, actual_com,
                   m_link, I_link, alpha_link, include_M2=False):
    for j_name, cx, cy, sign in joints_list:
        P = points[j_name]
        r = P - ref_pt
        A_mat[row_m, cx] += sign * (-r[1])
        A_mat[row_m, cy] += sign * (r[0])

    r_com = actual_com - ref_pt
    b_vec[row_m] = I_link * alpha_link + m_link * g * r_com[0]

    if include_M2:
        A_mat[row_m, COL_M2] += 1.0

def solve_dynamics_at(points, acom, alpha_link, coms):
    A_mat = np.zeros((N_UNKNOWN, N_UNKNOWN))
    b_vec = np.zeros(N_UNKNOWN)

    def cols(name):
        return _COLS[name]

    joints2 = [('B', *cols('B'), +1), ('F', *cols('F'), +1)]
    _add_force_eq(A_mat, b_vec, 0, 1, joints2, m2, acom['2'])
    _add_moment_eq(A_mat, b_vec, points, 2, joints2, points['B'], coms['2'],
                   m2, I2_B, alpha_link['theta2'], include_M2=True)

    joints3 = [('F', *cols('F'), -1), ('E', *cols('E'), +1), ('G', *cols('G'), +1)]
    _add_force_eq(A_mat, b_vec, 3, 4, joints3, m3, acom['3'])
    _add_moment_eq(A_mat, b_vec, points, 5, joints3, coms['3'], coms['3'],
                   m3, I3, alpha_link['theta3'])

    joints4 = [('A', *cols('A'), +1), ('E', *cols('E'), -1), ('D', *cols('D'), +1)]
    _add_force_eq(A_mat, b_vec, 6, 7, joints4, m4, acom['4'])
    _add_moment_eq(A_mat, b_vec, points, 8, joints4, coms['4'], coms['4'],
                   m4, I4, alpha_link['theta4'])

    joints5 = [('D', *cols('D'), -1), ('H', *cols('H'), +1)]
    _add_force_eq(A_mat, b_vec, 9, 10, joints5, m5, acom['5'])
    _add_moment_eq(A_mat, b_vec, points, 11, joints5, coms['5'], coms['5'],
                   m5, I5, alpha_link['theta5'])

    joints6 = [('G', *cols('G'), -1), ('H', *cols('H'), -1)]
    _add_force_eq(A_mat, b_vec, 12, 13, joints6, m6, acom['6'])
    _add_moment_eq(A_mat, b_vec, points, 14, joints6, coms['6'], coms['6'],
                   m6, I6, alpha_link['theta6'])

    sol = np.linalg.solve(A_mat, b_vec)
    joint_forces = {name: sol[c[0]:c[1]+1] for name, c in _COLS.items()}
    torque2_Nmm = sol[COL_M2] * 1000.0
    return joint_forces, torque2_Nmm

def _coms_and_acoms(points_mm, acc_mm):
    mm = 1e-3
    P = {k: v * mm for k, v in points_mm.items()}
    Acc = {k: v * mm for k, v in acc_mm.items()}

    coms = {
        '2': 0.5 * (P['B'] + P['F']),
        '3': 0.5 * (P['E'] + P['G']),
        '4': 0.5 * (P['E'] + P['D']),
        '5': 0.5 * (P['D'] + P['H']),
        '6': 0.5 * (P['G'] + P['H']),
    }

    acom = {
        '2': 0.5 * (Acc['B'] + Acc['F']),
        '3': 0.5 * (Acc['E'] + Acc['G']),
        '4': 0.5 * (Acc['E'] + Acc['D']),
        '5': 0.5 * (Acc['D'] + Acc['H']),
        '6': 0.5 * (Acc['G'] + Acc['H']),
    }

    return P, coms, acom

def compute_dynamics_time_series():
    n = len(theta2_vals)
    joint_forces_ts = {k: np.zeros((n, 2)) for k in _COLS}
    torque2_ts = np.zeros(n)

    for i in range(n):
        points_mm = {k: nodes[k][i] for k in nodes}
        acc_mm = {k: acc[k][i] for k in acc}
        P, coms, acom = _coms_and_acoms(points_mm, acc_mm)
        alpha_link = {k: alpha[k][i] for k in ['theta2', 'theta3', 'theta4', 'theta5', 'theta6']}

        jf, torque = solve_dynamics_at(P, acom, alpha_link, coms)

        for name in _COLS:
            joint_forces_ts[name][i] = jf[name]
        torque2_ts[i] = torque

    return joint_forces_ts, torque2_ts

def instantaneous_dynamics(theta_deg):
    nodes_i, vel_i, acc_i, omega_i, alpha_i = kinematics(theta_deg)
    P, coms, acom = _coms_and_acoms(nodes_i, acc_i)
    jf, torque = solve_dynamics_at(P, acom, alpha_i, coms)
    return nodes_i, vel_i, acc_i, omega_i, alpha_i, jf, torque

joint_forces_ts, torque2_ts = compute_dynamics_time_series()


# Dynamic plots

def _smart_ylim(*arrays, margin=0.15, lo_pct=2, hi_pct=98):
   
    all_vals = np.concatenate([np.asarray(a).ravel() for a in arrays])
    lo = np.percentile(all_vals, lo_pct)
    hi = np.percentile(all_vals, hi_pct)
    span = hi - lo
    if span == 0:
        span = max(abs(hi), 1.0)
    return lo - margin*span, hi + margin*span

def plot_torque_link2():
    theta2_deg = np.rad2deg(theta2_vals)
    ylo, yhi = _smart_ylim(torque2_ts)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(theta2_deg, torque2_ts, lw=2)
    axes[0].set_ylim(ylo, yhi)
    axes[0].set_xlabel("theta2 [deg]"); axes[0].set_ylabel("Driving torque M2 [N.mm]")
    axes[0].set_title("Zoomed in (near-singularity spike clipped off-scale)")
    axes[0].grid(True)

    axes[1].plot(theta2_deg, torque2_ts, lw=2)
    axes[1].set_yscale('symlog')   # linear near 0, log further out -- handles the sign change
    axes[1].set_xlabel("theta2 [deg]"); axes[1].set_ylabel("Driving torque M2 [N.mm] (symlog)")
    axes[1].set_title("Full range (symlog scale)")
    axes[1].grid(True)

    fig.suptitle("Driving Torque on Link 2 vs theta2")
    plt.tight_layout()
    plt.show()

def plot_joint_force(name):
    if name not in joint_forces_ts:
        print("Invalid joint name.")
        return

    theta2_deg = np.rad2deg(theta2_vals)
    fx = joint_forces_ts[name][:, 0]
    fy = joint_forces_ts[name][:, 1]
    mag = np.sqrt(fx**2 + fy**2)

    ylo, yhi = _smart_ylim(fx, fy, mag)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(theta2_deg, fx, label=f'{name}x')
    axes[0].plot(theta2_deg, fy, label=f'{name}y')
    axes[0].plot(theta2_deg, mag, label=f'|{name}|', lw=2)
    axes[0].set_ylim(ylo, yhi)
    axes[0].set_xlabel("theta2 [deg]"); axes[0].set_ylabel("Joint force [N]")
    axes[0].set_title("Zoomed in (near-singularity spike clipped off-scale)")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(theta2_deg, fx, label=f'{name}x')
    axes[1].plot(theta2_deg, fy, label=f'{name}y')
    axes[1].plot(theta2_deg, mag, label=f'|{name}|', lw=2)
    axes[1].set_yscale('symlog')
    axes[1].set_xlabel("theta2 [deg]"); axes[1].set_ylabel("Joint force [N] (symlog)")
    axes[1].set_title("Full range (symlog scale)")
    axes[1].legend(); axes[1].grid(True)

    fig.suptitle(f"Joint Force at {name} vs theta2")
    plt.tight_layout()
    plt.show()



# Main Menu

if __name__ == "__main__":
    print("\nHart Inversor Mechanism -- Kinematic & Newton-Euler Dynamic Solver")

    while True:
        print("\n1. Node positions at a given theta2")
        print("2. Point kinematics on a link")
        print("3. Instantaneous kinematics at a given theta2")
        print("4. Plot trajectories of all nodes")
        print("5. Plot trajectories of F,G,H,C")
        print("6. Plot speed of all nodes")
        print("7. Plot acceleration of all nodes")
        print("8. Plot angular velocities of links")
        print("9. Plot angular accelerations of links")
        print("10. Animate mechanism")
        print("11. Plot driving torque on link 2 vs theta2")
        print("12. Plot joint force vs theta2")
        print("13. Instantaneous dynamic solve at a given theta2")
        print("0. Exit")

        choice = input("Enter choice: ").strip()

        if choice == "0":
            break

        elif choice == "1":
            th = float(input("Enter theta2 in degrees (40-130): "))
            nodes_i, _ = solve_angle(np.deg2rad(th))

            print("\nNode positions (mm):")
            for k in nodes_i:
                print(f"{k}: x={nodes_i[k][0]:.3f}, y={nodes_i[k][1]:.3f}")

        elif choice == "2":
            th = float(input("Enter theta2 in degrees (40-130): "))
            print(list(nodes.keys()))
            P = input("First node: ").strip()
            Q = input("Second node: ").strip()
            s = float(input("Fraction from first to second node (0-1): "))

            nodes_i, vel_i, acc_i, omega_i, alpha_i = kinematics_from_dataset(th)
            pos, v, a = interpolate_point(nodes_i, vel_i, acc_i, P, Q, s)

            print(f"\nPosition: x={pos[0]:.3f}, y={pos[1]:.3f}")
            print(f"Velocity: vx={v[0]:.3f}, vy={v[1]:.3f}, |v|={np.linalg.norm(v):.3f}")
            print(f"Acceleration: ax={a[0]:.3f}, ay={a[1]:.3f}, |a|={np.linalg.norm(a):.3f}")

            print("\nAngular velocities:")
            for k in omega_i:
                print(f"{k}: {omega_i[k]:.4f}")

            print("\nAngular accelerations:")
            for k in alpha_i:
                print(f"{k}: {alpha_i[k]:.4f}")

        elif choice == "3":
            th = float(input("Enter theta2 in degrees (40-130): "))
            nodes_i, vel_i, acc_i, omega_i, alpha_i = kinematics(th)

            print("\nNode positions (mm):")
            for k in nodes_i:
                print(f"{k}: x={nodes_i[k][0]:.3f}, y={nodes_i[k][1]:.3f}")

            print("\nNode velocities (mm/s):")
            for k in vel_i:
                print(f"{k}: vx={vel_i[k][0]:.3f}, vy={vel_i[k][1]:.3f}, |v|={np.linalg.norm(vel_i[k]):.3f}")

            print("\nNode accelerations (mm/s^2):")
            for k in acc_i:
                print(f"{k}: ax={acc_i[k][0]:.3f}, ay={acc_i[k][1]:.3f}, |a|={np.linalg.norm(acc_i[k]):.3f}")

            print("\nAngular velocities:")
            for k in omega_i:
                print(f"{k}: {omega_i[k]:.4f}")

            print("\nAngular accelerations:")
            for k in alpha_i:
                print(f"{k}: {alpha_i[k]:.4f}")

        elif choice == "4":
            plot_all_trajectories()

        elif choice == "5":
            plot_important_trajectories()

        elif choice == "6":
            plot_speed_all_nodes()

        elif choice == "7":
            plot_acc_all_nodes()

        elif choice == "8":
            plot_omega_links()

        elif choice == "9":
            plot_alpha_links()

        elif choice == "10":
            animate_mechanism()

        elif choice == "11":
            plot_torque_link2()

        elif choice == "12":
            print("Available joints:", list(joint_forces_ts.keys()))
            jname = input("Enter joint name: ").strip()
            plot_joint_force(jname)

        elif choice == "13":
            th = float(input("Enter theta2 in degrees (40-130): "))
            nodes_i, vel_i, acc_i, omega_i, alpha_i, jf, torque = instantaneous_dynamics(th)

            print("\nInstantaneous Dynamic Solve")
            print(f"Driving torque M2 = {torque:.4f} N.mm")

            print("\nJoint forces [N]:")
            for name, val in jf.items():
                print(f"{name}: Fx={val[0]:.4f}, Fy={val[1]:.4f}, |F|={np.linalg.norm(val):.4f}")

      

        else:
            print("Invalid choice.")
