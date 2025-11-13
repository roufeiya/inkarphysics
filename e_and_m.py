# app/routes/e_and_m.py

from flask import (
    Blueprint, render_template, request, Response
)
# Бұл файлға қажетті барлық кітапханаларды осында импорттаймыз
import numpy as np
import matplotlib.pyplot as plt
import io
import matplotlib
matplotlib.use('Agg')
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import cmath
import math

# 'e_and_m' (Электр және Магнетизм) атты жаңа "бөлім"
bp = Blueprint('e_and_m', __name__)


# ################################################
# ### E&M БӨЛІМІНІҢ НЕГІЗГІ БЕТІ ###
# ################################################
@bp.route('/', methods=['GET'])
def index():
    # Бұл /e_and_m/ бетін ашады
    return render_template('e_and_m_index.html')

# ################################################
# ### МОДУЛЬ 2: ЭЛЕКТР ӨРІСІ ###
# ################################################

@bp.route('/theory/efield', methods=['GET'])
def theory_efield():
    return render_template('theory_efield.html')

@bp.route('/simulator', methods=['GET', 'POST'])
def simulator_page():
    charges_data = None
    if request.method == 'POST':
         charges_data = { "x1": request.form.get('x1'), "y1": request.form.get('y1'), "x2": request.form.get('x2'), "y2": request.form.get('y2') }
    else:
        charges_data = { "x1": '-2', "y1": '0', "x2": '2', "y2": '0' }
    return render_template('simulator.html', charges=charges_data)

@bp.route('/plot.png')
def plot_png():
    try:
        x1 = float(request.args.get('x1', '-2')); y1 = float(request.args.get('y1', '0'))
        x2 = float(request.args.get('x2', '2')); y2 = float(request.args.get('y2', '0'))
        x = np.linspace(-10, 10, 100); y = np.linspace(-10, 10, 100); X, Y = np.meshgrid(x, y)
        q1_pos = np.array([x1, y1]); q2_pos = np.array([x2, y2])
        r1 = np.sqrt((X - q1_pos[0])**2 + (Y - q1_pos[1])**2); r2 = np.sqrt((X - q2_pos[0])**2 + (Y - q2_pos[1])**2)
        r1[r1 == 0] = 1e-6; r2[r2 == 0] = 1e-6
        Ex = (X - q1_pos[0]) / r1**3 + (-1)*(X - q2_pos[0]) / r2**3
        Ey = (Y - q1_pos[1]) / r1**3 + (-1)*(Y - q2_pos[1]) / r2**3
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.streamplot(X, Y, Ex, Ey, density=1.2, color='b', linewidth=0.8)
        ax.plot(q1_pos[0], q1_pos[1], 'ro', markersize=10, label='q+ (1-заряд)')
        ax.plot(q2_pos[0], q2_pos[1], 'bo', markersize=10, label='q- (2-заряд)')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
        ax.set_aspect('equal'); ax.legend(); ax.grid(True)
        output = io.BytesIO(); plt.savefig(output, format='png', bbox_inches='tight'); plt.close(fig)
        return Response(output.getvalue(), mimetype='image/png')
    except Exception as e:
        print(f"Error plotting: {e}")
        return "Графикті құру кезінде қате", 500

# ################################################
# ### МОДУЛЬ 4: RLC КОНТУРЫ ###
# ################################################

@bp.route('/theory/rlc', methods=['GET'])
def theory_rlc():
    return render_template('theory_rlc.html')

@bp.route('/rlc', methods=['GET'])
def rlc_page():
    rlc_params = { "R": request.args.get('R', '5'), "L": request.args.get('L', '20'), "C": request.args.get('C', '5') }
    return render_template('rlc.html', params=rlc_params)

@bp.route('/plot_rlc.png')
def plot_rlc_png():
    try:
        R = float(request.args.get('R', '5')); L = float(request.args.get('L', '20')); C = float(request.args.get('C', '5')) * 1e-3
        def rlc_model(t, y):
            Q, I = y; dQ_dt = I; dI_dt = -(R/L) * I - (1/(L*C)) * Q
            return [dQ_dt, dI_dt]
        Q0 = 10.0; I0 = 0.0; y0 = [Q0, I0]
        t_span = [0, 50]; t_eval = np.linspace(t_span[0], t_span[1], 1000)
        sol = solve_ivp(rlc_model, t_span, y0, t_eval=t_eval)
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)
        ax1.plot(sol.t, sol.y[0], 'b-', label=f'Заряд Q(t)'); ax1.set_ylabel('Заряд (Q), Кулон')
        ax1.set_title(f'RLC Тербелістері (R={R}, L={L}, C={C}e-3)'); ax1.legend(); ax1.grid(True)
        ax2.plot(sol.t, sol.y[1], 'r-', label=f'Ток I(t) = dQ/dt'); ax2.set_xlabel('Уақыт (t), секунд')
        ax2.set_ylabel('Ток (I), Ампер'); ax2.legend(); ax2.grid(True)
        output = io.BytesIO(); plt.savefig(output, format='png', bbox_inches='tight'); plt.close(fig)
        return Response(output.getvalue(), mimetype='image/png')
    except Exception as e:
        print(f"RLC салу қатесі: {e}")
        return "RLC графигін салу кезінде қате", 500

# ################################################
# ### МОДУЛЬ 5: ЛОРЕНЦ КҮШІ ###
# ################################################

@bp.route('/theory/lorentz', methods=['GET'])
def theory_lorentz():
    return render_template('theory_lorentz.html')

@bp.route('/lorentz', methods=['GET', 'POST'])
def lorentz_page():
    lorentz_params = None
    if request.method == 'POST':
        lorentz_params = { "particle": request.form.get('particle'), "Bz": request.form.get('Bz'),
                           "vx": request.form.get('vx'), "vy": request.form.get('vy'), "vz": request.form.get('vz') }
    else:
        lorentz_params = { "particle": 'electron', "Bz": '0.01', "vx": '1.0e6', "vy": '0.0', "vz": '1.0e5' }
    return render_template('lorentz.html', params=lorentz_params)

@bp.route('/plot_lorentz.png')
def plot_lorentz_png():
    try:
        particle_type = request.args.get('particle'); B_z = float(request.args.get('Bz'))
        v_x_0 = float(request.args.get('vx')); v_y_0 = float(request.args.get('vy')); v_z_0 = float(request.args.get('vz'))
        if particle_type == 'proton':
            q = 1.602e-19; m = 1.672e-27; color = 'r'; label_text = "Протон траекториясы"
        else:
            q = -1.602e-19; m = 9.109e-31; color = 'b'; label_text = "Электрон траекториясы"
        def lorentz_model(t, y):
            q_over_m = q / m; d_x = y[3]; d_y = y[4]; d_z = y[5]
            d_vx = q_over_m * (y[4] * B_z); d_vy = -q_over_m * (y[3] * B_z); d_vz = 0.0
            return [d_x, d_y, d_z, d_vx, d_vy, d_vz]
        y0 = [0.0, 0.0, 0.0, v_x_0, v_y_0, v_z_0]; t_span = [0, 5e-7]
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        sol = solve_ivp(lorentz_model, t_span, y0, t_eval=t_eval)
        fig = plt.figure(figsize=(8, 8)); ax = fig.add_subplot(111, projection='3d')
        ax.plot(sol.y[0], sol.y[1], sol.y[2], color=color, label=label_text)
        ax.plot([sol.y[0][0]], [sol.y[1][0]], [sol.y[2][0]], 'go', markersize=8, label="Бастапқы нүкте (t=0)")
        ax.set_xlabel('X осі (m)'); ax.set_ylabel('Y осі (m)'); ax.set_zlabel('Z осі (m)')
        ax.set_title(f'Зарядтың қозғалысы (B_z = {B_z} T)'); ax.legend(); ax.grid(True)
        all_coords = np.concatenate([sol.y[0], sol.y[1], sol.y[2]])
        max_range = np.max(all_coords) - np.min(all_coords)
        mid_x = (np.max(sol.y[0]) + np.min(sol.y[0])) / 2; mid_y = (np.max(sol.y[1]) + np.min(sol.y[1])) / 2
        mid_z = (np.max(sol.y[2]) + np.min(sol.y[2])) / 2
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2); ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
        output = io.BytesIO(); plt.savefig(output, format='png', bbox_inches='tight'); plt.close(fig)
        return Response(output.getvalue(), mimetype='image/png')
    except Exception as e:
        print(f"Лоренц салу қатесі: {e}")
        return "Лоренц графигін салу кезінде қате", 500

# ################################################
# ### МОДУЛЬ 6: МАГНИТ ӨРІСІ (БИО-САВАР) ###
# ################################################

@bp.route('/theory/biot_savart', methods=['GET'])
def theory_biot_savart():
    return render_template('theory_biot_savart.html')

@bp.route('/biot_savart', methods=['GET', 'POST'])
def biot_savart_page():
    bs_params = None
    if request.method == 'POST':
        bs_params = { "x1": request.form.get('x1'), "y1": request.form.get('y1'), "i1": request.form.get('i1'),
                      "x2": request.form.get('x2'), "y2": request.form.get('y2'), "i2": request.form.get('i2'),
                      "x3": request.form.get('x3'), "y3": request.form.get('y3'), "i3": request.form.get('i3')}
    else:
        bs_params = { "x1": '-3', "y1": '0', "i1": '10', "x2": '3', "y2": '0', "i2": '-10', "x3": '0', "y3": '5', "i3": '0'}
    return render_template('biot_savart.html', params=bs_params)

@bp.route('/plot_biot_savart.png')
def plot_biot_savart_png():
    try:
        params = [
            {"pos": (float(request.args.get('x1')), float(request.args.get('y1'))), "I": float(request.args.get('i1'))},
            {"pos": (float(request.args.get('x2')), float(request.args.get('y2'))), "I": float(request.args.get('i2'))},
            {"pos": (float(request.args.get('x3')), float(request.args.get('y3'))), "I": float(request.args.get('i3'))}
        ]
        x = np.linspace(-10, 10, 100); y = np.linspace(-10, 10, 100); X, Y = np.meshgrid(x, y)
        Bx_total = np.zeros_like(X); By_total = np.zeros_like(Y)
        fig, ax = plt.subplots(figsize=(7, 7)); K = 1.0
        for wire in params:
            I = wire["I"];
            if I == 0: continue
            x_i, y_i = wire["pos"]; r_sq = (X - x_i)**2 + (Y - y_i)**2; r_sq[r_sq < 0.01] = 0.01
            Bx_total += -K * I * (Y - y_i) / r_sq; By_total += +K * I * (X - x_i) / r_sq
            if I > 0:
                ax.plot(x_i, y_i, 'bo', markersize=10, markerfacecolor='w', markeredgecolor='b', markeredgewidth=3, label=f'I = {I} A (X)')
            else:
                ax.plot(x_i, y_i, 'ro', markersize=10, markerfacecolor='r', label=f'I = {I} A (•)')
        ax.streamplot(X, Y, Bx_total, By_total, density=1.5, color='k', linewidth=0.8)
        ax.set_xlabel('X осі'); ax.set_ylabel('Y осі'); ax.set_title('Тогы бар өткізгіштердің магнит өрісі')
        ax.set_xlim(-10, 10); ax.set_ylim(-10, 10); ax.set_aspect('equal'); ax.legend(); ax.grid(True)
        output = io.BytesIO(); plt.savefig(output, format='png', bbox_inches='tight'); plt.close(fig)
        return Response(output.getvalue(), mimetype='image/png')
    except Exception as e:
        print(f"Био-Савар салу қатесі: {e}")
        return "Био-Савар графигін салу кезінде қате", 500

# ################################################
# ### МОДУЛЬ 7: ИМПЕДАНС (AC ТІЗБЕК) ###
# ################################################

@bp.route('/theory/impedance', methods=['GET'])
def theory_impedance():
    return render_template('theory_impedance.html')

@bp.route('/impedance', methods=['GET', 'POST'])
def impedance_calculator():
    results = None; error = None
    if request.method == 'POST':
        try:
            R = float(request.form['R']); L = float(request.form['L']) * 1e-3; C = float(request.form['C']) * 1e-6; f = float(request.form['f'])
            if f <= 0: error = "Жиілік (f) нөлден үлкен болуы керек."
            elif C <= 0: error = "Сыйымдылық (C) нөлден үлкен болуы керек."
            else:
                w = 2 * cmath.pi * f; Z_R = complex(R, 0); Z_L = complex(0, w * L); Z_C = complex(0, -1 / (w * C))
                Z_total = Z_R + Z_L + Z_C; Z_magnitude = abs(Z_total); Z_phase_rad = cmath.phase(Z_total)
                Z_phase_deg = math.degrees(Z_phase_rad)
                results = { "Z_total_str": f"{Z_total.real:.2f} + ({Z_total.imag:.2f}j) Ом",
                            "Z_magnitude": round(Z_magnitude, 2), "Z_phase_deg": round(Z_phase_deg, 2) }
        except ValueError: error = "Барлық өрістерге жарамды сандарды енгізіңіз."
        except Exception as e: error = f"Белгісіз қате: {e}"
    return render_template('impedance.html', results=results, error=error)


# ################################################
# ### МОДУЛЬ 8: РЕЗОНАНС (АЖЖ) ###
# ################################################

@bp.route('/theory/resonance', methods=['GET'])
def theory_resonance():
    return render_template('theory_resonance.html')

@bp.route('/resonance', methods=['GET'])
def resonance_page():
    res_params = { "R": request.args.get('R', '50'), "L": request.args.get('L', '100'), "C": request.args.get('C', '0.1') }
    return render_template('resonance.html', params=res_params)

@bp.route('/plot_resonance.png')
def plot_resonance_png():
    try:
        R = float(request.args.get('R', '50')); L = float(request.args.get('L', '100')) * 1e-3; C = float(request.args.get('C', '0.1')) * 1e-6; V = 1.0
        if L <= 0 or C <= 0: return "L және C нөлден үлкен болуы керек", 500
        f_res = 1 / (2 * cmath.pi * (L * C)**0.5)
        frequencies = np.linspace(f_res * 0.1, f_res * 2.0, 500)
        currents = []; impedances = []; phases_deg = []
        for f in frequencies:
            w = 2 * cmath.pi * f; Z_total = complex(R, (w * L - 1 / (w * C))); Z_magnitude = abs(Z_total)
            I_amplitude = V / Z_magnitude; Z_phase_rad = cmath.phase(Z_total); Z_phase_deg = math.degrees(Z_phase_rad)
            currents.append(I_amplitude); impedances.append(Z_magnitude); phases_deg.append(Z_phase_deg)
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), sharex=True)
        ax1.plot(frequencies, currents, 'r-', label='Ток амплитудасы I(f)')
        ax1.set_ylabel('Ток (I), Ампер (V=1В)'); ax1.set_title(f'RLC Резонанс Тақтасы (R={R} Ом, L={L*1000:.1f} мГн, C={C*1e6:.2f} мкФ)')
        ax1.axvline(x=f_res, color='k', linestyle='--', label=f'f₀ = {f_res:.0f} Гц'); ax1.legend(); ax1.grid(True)
        ax2.plot(frequencies, impedances, 'b-', label='Толық кедергі |Z(f)|')
        ax2.set_ylabel('Импеданс |Z|, Ом'); ax2.axvline(x=f_res, color='k', linestyle='--'); ax2.legend(); ax2.grid(True)
        ax3.plot(frequencies, phases_deg, 'g-', label='Фаза ығысуы φ(f)')
        ax3.set_xlabel('Жиілік (f), Гц'); ax3.set_ylabel('Фаза (φ), градус')
        ax3.axvline(x=f_res, color='k', linestyle='--'); ax3.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
        ax3.legend(); ax3.grid(True)
        output = io.BytesIO(); plt.savefig(output, format='png', bbox_inches='tight'); plt.close(fig)
        return Response(output.getvalue(), mimetype='image/png')
    except Exception as e:
        print(f"Резонанс салу қатесі: {e}")
        return "Резонанс графигін салу кезінде қате", 500

# ################################################
# ### МОДУЛЬ 9: ТОЛҚЫНДЫҚ ИНТЕРФЕРЕНЦИЯ ###
# ################################################

@bp.route('/theory/interference', methods=['GET'])
def theory_interference():
    return render_template('theory_interference.html')

@bp.route('/interference', methods=['GET', 'POST'])
def interference_page():
    int_params = None
    if request.method == 'POST':
        int_params = {
            "lambda": request.form.get('wavelength', '1.5'),
            "d": request.form.get('distance', '5')
        }
    else:
        int_params = { "lambda": '1.5', "d": '5' }
    return render_template('interference.html', params=int_params)

@bp.route('/plot_interference.png')
def plot_interference_png():
    try:
        lambda_ = float(request.args.get('lambda', '1.5'))
        d = float(request.args.get('d', '5'))
        x = np.linspace(-20, 20, 400); y = np.linspace(0, 30, 400); X, Y = np.meshgrid(x, y)
        k = 2 * np.pi / lambda_; source1_pos = np.array([-d/2, 0]); source2_pos = np.array([d/2, 0])
        r1 = np.sqrt((X - q1_pos[0])**2 + (Y - q1_pos[1])**2)
        r2 = np.sqrt((X - q2_pos[0])**2 + (Y - q2_pos[1])**2)
        r1[r1 == 0] = 1e-6; r2[r2 == 0] = 1e-6
        wave1 = (1/r1) * np.cos(k * r1); wave2 = (1/r2) * np.cos(k * r2)
        total_wave = wave1 + wave2; intensity = total_wave**2
        vmax_limit = np.percentile(intensity, 99)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(intensity, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='hot', aspect='equal', vmax=vmax_limit)
        ax.plot([-d/2, d/2], [0, 0], 'co', markersize=10, label='Толқын көздері')
        ax.set_xlabel('X осі'); ax.set_ylabel('Y осі'); ax.set_title(f'Интерференция суреті (λ={lambda_}, d={d})')
        ax.legend(); ax.grid(False)
        output = io.BytesIO(); plt.savefig(output, format='png', bbox_inches='tight'); plt.close(fig)
        return Response(output.getvalue(), mimetype='image/png')
    except Exception as e:
        print(f"Интерференция салу қатесі: {e}")
        return "Интерференция графигін салу кезінде қате", 500