# app/routes/mechanics.py
from flask import (
    Blueprint, render_template, request, Response
)
import numpy as np
import matplotlib.pyplot as plt
import io
import math
import matplotlib

matplotlib.use('Agg')
from scipy.integrate import solve_ivp

bp = Blueprint('mechanics', __name__)


# Бұл /mechanics/ деп ашылады
@bp.route('/', methods=['GET'])
def index():
    return render_template('mechanics.html')


# ################################################
# ### МОДУЛЬ 1: БАЛЛИСТИКА ###
# ################################################

@bp.route('/theory/ballistics', methods=['GET'])
def theory_ballistics():
    return render_template('theory_ballistics.html')


@bp.route('/ballistics', methods=['GET', 'POST'])
def ballistics():
    ballistics_params = None
    if request.method == 'POST':
        ballistics_params = {
            "v0": request.form.get('v0', '100'),
            "theta": request.form.get('theta', '45')
        }
    else:
        ballistics_params = {"v0": '100', "theta": '45'}
    return render_template('ballistics.html', params=ballistics_params)


@bp.route('/plot_ballistics.png')
def plot_ballistics_png():
    try:
        v0 = float(request.args.get('v0', '100'))
        theta_deg = float(request.args.get('theta', '45'))
        g = 9.81
        theta_rad = math.radians(theta_deg)
        if g == 0 or v0 == 0:
            T_total = 1.0
        else:
            T_total = (2 * v0 * math.sin(theta_rad)) / g
            if T_total <= 0: T_total = 1.0

        t = np.linspace(0, T_total, 300)
        x = v0 * math.cos(theta_rad) * t
        y = v0 * math.sin(theta_rad) * t - 0.5 * g * t ** 2

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y, 'b-', label='Траектория')
        max_height = np.max(y);
        max_range = np.max(x)
        ax.set_title(f'Баллистика (v₀={v0} м/с, θ={theta_deg}°)')
        ax.set_xlabel('Қашықтық (x), м');
        ax.set_ylabel('Биіктік (y), м')
        if max_range > 0 or max_height > 0:
            ax.text(max_range / 2, max_height * 0.7,
                    f' H_max = {max_height:.1f} м\n L_max = {max_range:.1f} м',
                    ha='center')
        ax.legend();
        ax.grid(True);
        ax.set_aspect('equal', 'box');
        ax.set_ylim(bottom=0)
        output = io.BytesIO();
        plt.savefig(output, format='png', bbox_inches='tight');
        plt.close(fig)
        return Response(output.getvalue(), mimetype='image/png')

    except Exception as e:
        print(f"Баллистика салу қатесі: {e}")
        return "Баллистика графигін салу кезінде қате", 500


# ################################################
# ### МОДУЛЬ 2: СЕРІППЕЛІ МАЯТНИК ###
# ################################################

@bp.route('/theory/spring', methods=['GET'])
def theory_spring():
    return render_template('theory_spring.html')


@bp.route('/spring', methods=['GET'])
def spring():
    spring_params = {
        "m": request.args.get('m', '1.0'),
        "k": request.args.get('k', '20.0'),
        "b": request.args.get('b', '0.5')
    }
    return render_template('spring.html', params=spring_params)


@bp.route('/plot_spring.png')
def plot_spring_png():
    try:
        m = float(request.args.get('m', '1.0'))
        k = float(request.args.get('k', '20.0'))
        b = float(request.args.get('b', '0.5'))

        def spring_model(t, y):
            x, v = y;
            dx_dt = v
            dv_dt = -(b / m) * v - (k / m) * x
            return [dx_dt, dv_dt]

        x0 = 1.0;
        v0 = 0.0;
        y0 = [x0, v0]
        t_span = [0, 25];
        t_eval = np.linspace(t_span[0], t_span[1], 500)

        sol = solve_ivp(spring_model, t_span, y0, t_eval=t_eval)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sol.t, sol.y[0], 'r-', label='Маятниктің орны x(t)')
        ax.set_title(f'Серіппелі маятник (m={m}, k={k}, b={b})')
        ax.set_xlabel('Уақыт (t), секунд');
        ax.set_ylabel('Ауытқу (x), метр')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax.legend();
        ax.grid(True)

        output = io.BytesIO();
        plt.savefig(output, format='png', bbox_inches='tight');
        plt.close(fig)
        return Response(output.getvalue(), mimetype='image/png')
    except Exception as e:
        print(f"Маятник салу қатесі: {e}")
        return "Маятник графигін салу кезінде қате", 500


# ################################################
# ### МОДУЛЬ 3: ГРАВИТАЦИЯ (ОРБИТА) ###
# ################################################

# ### ЖАҢА: ТЕОРИЯ БЕТІ ###
@bp.route('/theory/gravity', methods=['GET'])
def theory_gravity():
    return render_template('theory_gravity.html')


@bp.route('/gravity', methods=['GET'])
def gravity():
    gravity_params = {
        "x0": request.args.get('x0', '10'),
        "vy0": request.args.get('vy0', '1.0')
    }
    return render_template('gravity.html', params=gravity_params)


@bp.route('/plot_gravity.png')
def plot_gravity_png():
    try:
        x0 = float(request.args.get('x0', '10'))
        vy0 = float(request.args.get('vy0', '1.0'))
        GM = 10.0

        def gravity_model(t, y):
            x, y, vx, vy = y
            r = np.sqrt(x ** 2 + y ** 2)
            if r == 0: r = 1e-6
            r_cubed = r ** 3
            dx_dt = vx
            dy_dt = vy
            dvx_dt = -GM * x / r_cubed
            dvy_dt = -GM * y / r_cubed
            return [dx_dt, dy_dt, dvx_dt, dvy_dt]

        y0 = [x0, 0.0, 0.0, vy0]
        t_span = [0, 50];
        t_eval = np.linspace(t_span[0], t_span[1], 1500)
        sol = solve_ivp(gravity_model, t_span, y0, t_eval=t_eval)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(0, 0, 'yo', markersize=15, label='Күн (M)')
        ax.plot(sol.y[0], sol.y[1], 'b-', label=f'Планета (v_y={vy0})')
        ax.plot(x0, 0, 'gX', markersize=10, label='Бастапқы нүкте')
        ax.set_title('Екі дене есебі (Гравитация)')
        ax.set_xlabel('X осі');
        ax.set_ylabel('Y осі')
        ax.legend();
        ax.grid(True);
        ax.set_aspect('equal', 'box')

        output = io.BytesIO();
        plt.savefig(output, format='png', bbox_inches='tight');
        plt.close(fig)
        return Response(output.getvalue(), mimetype='image/png')
    except Exception as e:
        print(f"Гравитация салу қатесі: {e}")
        return "Гравитация графигін салу кезінде қате", 500