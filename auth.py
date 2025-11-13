# app/routes/auth.py
from flask import Blueprint, render_template, request, flash, redirect, url_for
from app import db
from app.models import User
from flask_login import login_user, logout_user, login_required, current_user

# 'auth' атты жаңа "бөлім"
bp = Blueprint('auth', __name__)


@bp.route('/register', methods=['GET', 'POST'])
def register():
    # Егер пайдаланушы кіріп тұрса, оны басты бетке жібереміз
    if current_user.is_authenticated:
        return redirect(url_for('main.calculators'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        password2 = request.form.get('password2')

        # Тексерулер
        if not username or not password or not password2:
            flash('Барлық өрістерді толтырыңыз!', 'danger')
            return redirect(url_for('auth.register'))

        if password != password2:
            flash('Құпия сөздер сәйкес келмейді!', 'danger')
            return redirect(url_for('auth.register'))

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Бұл пайдаланушы аты бос емес!', 'danger')
            return redirect(url_for('auth.register'))

        # Жаңа пайдаланушыны жасау
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash('Сіз сәтті тіркелдіңіз! Енді кіре аласыз.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('register.html')


@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.calculators'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        # Пайдаланушыны және құпия сөзді тексеру
        if user is None or not user.check_password(password):
            flash('Пайдаланушы аты немесе құпия сөз қате!', 'danger')
            return redirect(url_for('auth.login'))

        # Пайдаланушыны жүйеге кіргізу (сессияны есте сақтау)
        login_user(user, remember=True)
        flash('Сәтті кірдіңіз!', 'success')

        # "Келесі" бетке бағыттау (егер болса)
        next_page = request.args.get('next')
        if not next_page:
            next_page = url_for('main.calculators')
        return redirect(next_page)

    return render_template('login.html')


@bp.route('/logout')
@login_required  # Бұл бетті тек кірген пайдаланушылар көре алады
def logout():
    logout_user()
    flash('Сіз жүйеден шықтыңыз.', 'info')
    return redirect(url_for('main.calculators'))