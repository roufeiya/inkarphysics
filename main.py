# physics_site/app/routes/main.py

from flask import (
    Blueprint, render_template, request, session, redirect, url_for
)
import cmath
import math

# 'main' атты жаңа "бөлім" (Blueprint) құрамыз
bp = Blueprint('main', __name__)

# ### 30 СҰРАҚТАН ТҰРАТЫН БАЗА ###
QUIZ_QUESTIONS = [
    # --- Электростатика (10 сұрақ) ---
    {
        "question": "Электр зарядының өлшем бірлігі қандай?",
        "options": ["Ампер", "Вольт", "Ом", "Кулон"],
        "answer": "Кулон"
    },
    {
        "question": "Зарядталған денелердің өзара әрекеттесуін сипаттайтын заң?",
        "options": ["Ом заңы", "Кулон заңы", "Ампер заңы", "Фарадей заңы"],
        "answer": "Кулон заңы"
    },
    {
        "question": "Электр өрісін сипаттайтын күштік шама?",
        "options": ["Кернеу", "Кернеулік", "Кедергі", "Ток күші"],
        "answer": "Кернеулік"
    },
    {
        "question": "Электр өрісінің потенциалының өлшем бірлігі?",
        "options": ["Вольт", "Ватт", "Джоуль", "Ньютон"],
        "answer": "Вольт"
    },
    {
        "question": "Өткізгіштің зарядтарды жинақтау қабілетін сипаттайтын шама?",
        "options": ["Электр сыйымдылығы", "Индуктивтілік", "Кедергі", "Өткізгіштік"],
        "answer": "Электр сыйымдылығы"
    },
    {
        "question": "Электр сыйымдылығының өлшем бірлігі?",
        "options": ["Генри", "Тесла", "Фарад", "Вебер"],
        "answer": "Фарад"
    },
    {
        "question": "Конденсатор не үшін қолданылады?",
        "options": ["Токты күшейту үшін", "Магнит өрісін жасау үшін", "Электр зарядын жинақтау үшін", "Кедергіні азайту үшін"],
        "answer": "Электр зарядын жинақтау үшін"
    },
    {
        "question": "Позитивті (оң) заряд иесі қандай бөлшек?",
        "options": ["Электрон", "Нейтрон", "Протон", "Фотон"],
        "answer": "Протон"
    },
    {
        "question": "Эбонит таяқшаны жүнге үйкегенде, ол қалай зарядталады?",
        "options": ["Оң", "Теріс", "Зарядталмайды", "Алдымен оң, сосын теріс"],
        "answer": "Теріс"
    },
    {
        "question": "Электр өрісін жақсы өткізбейтін заттар қалай аталады?",
        "options": ["Өткізгіштер", "Жартылай өткізгіштер", "Диэлектриктер (Оқшаулағыштар)", "Асқын өткізгіштер"],
        "answer": "Диэлектриктер (Оқшаулағыштар)"
    },

    # --- Тұрақты ток (11 сұрақ) ---
    {
        "question": "Ток күшінің өлшем бірлігі?",
        "options": ["Ампер", "Вольт", "Ом", "Кулон"],
        "answer": "Ампер"
    },
    {
        "question": "Тізбек бөлігі үшін Ом заңының формуласын көрсетіңіз?",
        "options": ["I = U / R", "I = U * R", "I = R / U", "P = U * I"],
        "answer": "I = U / R"
    },
    {
        "question": "Электр кедергісінің өлшем бірлігі?",
        "options": ["Ампер", "Вольт", "Ом", "Ватт"],
        "answer": "Ом"
    },
    {
        "question": "Ток күшін өлшейтін құрал?",
        "options": ["Вольтметр", "Омметр", "Амперметр", "Ваттметр"],
        "answer": "Амперметр"
    },
    {
        "question": "Кернеуді өлшейтін құрал?",
        "options": ["Вольтметр", "Амперметр", "Динамометр", "Манометр"],
        "answer": "Вольтметр"
    },
    {
        "question": "Амперметр тізбекке қалай қосылады?",
        "options": ["Параллель", "Тізбектей", "Кез келген әдіспен", "Тізбектен тыс"],
        "answer": "Тізбектей"
    },
    {
        "question": "Вольтметр тізбекке қалай қосылады?",
        "options": ["Параллель", "Тізбектей", "Кез келген әдіспен", "Тізбектен тыс"],
        "answer": "Параллель"
    },
    {
        "question": "Токтың жұмысының өлшем бірлігі?",
        "options": ["Ватт", "Джоуль", "Ампер", "Ньютон"],
        "answer": "Джоуль"
    },
    {
        "question": "Токтың қуатының өлшем бірлігі?",
        "options": ["Ватт", "Джоуль", "Вольт", "Ом"],
        "answer": "Ватт"
    },
    {
        "question": "Өткізгіштерді тізбектей жалғағанда, қай шама тұрақты болады?",
        "options": ["Ток күші", "Кернеу", "Кедергі", "Ешқайсысы"],
        "answer": "Ток күші"
    },
    {
        "question": "Өткізгіштерді параллель жалғағанда, қай шама тұрақты болады?",
        "options": ["Ток күші", "Кернеу", "Кедергі", "Ешқайсысы"],
        "answer": "Кернеу"
    },

    # --- Магнетизм (9 сұрақ) ---
    {
        "question": "Магнит өрісін сипаттайтын негізгі шама?",
        "options": ["Магнит индукциясы (B)", "Магнит ағыны (Ф)", "Кернеулік (E)", "Ток күші (I)"],
        "answer": "Магнит индукциясы (B)"
    },
    {
        "question": "Магнит индукциясының өлшем бірлігі?",
        "options": ["Вебер", "Генри", "Тесла", "Фарад"],
        "answer": "Тесла"
    },
    {
        "question": "Магнит өрісін не тудырады?",
        "options": ["Тыныштықтағы зарядтар", "Қозғалыстағы зарядтар (электр тогы)", "Гравитация", "Нейтрондар"],
        "answer": "Қозғалыстағы зарядтар (электр тогы)"
    },
    {
        "question": "Тогы бар өткізгішке магнит өрісі тарапынан әсер ететін күш?",
        "options": ["Кулон күші", "Ампер күші", "Лоренц күші", "Архимед күші"],
        "answer": "Ампер күші"
    },
    {
        "question": "Магнит өрісінде қозғалатын зарядқа әсер ететін күш?",
        "options": ["Кулон күші", "Ампер күші", "Лоренц күші", "Үйкеліс күші"],
        "answer": "Лоренц күші"
    },
    {
        "question": "Тұйық контур арқылы өтетін магнит ағынының өзгеруі кезінде ЭҚК пайда болу құбылысы?",
        "options": ["Электростатика", "Электромагниттік индукция", "Термоэлектронды эмиссия", "Асқын өткізгіштік"],
        "answer": "Электромагниттік индукция"
    },
    {
        "question": "Магнит ағынының өлшем бірлігі?",
        "options": ["Тесла", "Вебер", "Генри", "Вольт"],
        "answer": "Вебер"
    },
    {
        "question": "Индуктивтіліктің өлшем бірлігі?",
        "options": ["Ом", "Фарад", "Генри", "Тесла"],
        "answer": "Генри"
    },
    {
        "question": "Айнымалы токты трансформациялауға (өзгертуге) арналған құрылғы?",
        "options": ["Генератор", "Трансформатор", "Резистор", "Конденсатор"],
        "answer": "Трансформатор"
    }
]


# --- МОДУЛЬ 1: КАЛЬКУЛЯТОРЛАР (БАСТЫ БЕТ) ---
@bp.route('/', methods=['GET', 'POST'])
def calculators():
    current = None
    ohm_error = None
    total_resistance = None
    resistor_error = None
    if request.method == 'POST':
        if 'calculate_ohm' in request.form:
            try:
                voltage = float(request.form['voltage'])
                resistance = float(request.form['resistance'])
                if resistance == 0:
                    ohm_error = "Кедергі нөлге тең бола алмайды."
                else:
                    current = round(voltage / resistance, 3)
            except ValueError:
                ohm_error = "Қате (Ом): Тек сандарды енгізіңіз."
        elif 'calculate_resistors' in request.form:
            try:
                r1 = float(request.form['r1'])
                r2 = float(request.form['r2'])
                connection_type = request.form['connection_type']
                if connection_type == 'series':
                    total_resistance = round(r1 + r2, 3)
                elif connection_type == 'parallel':
                    if (r1 + r2) == 0:
                        resistor_error = "Кедергілер қосындысы нөл бола алмайды."
                    else:
                        total_resistance = round((r1 * r2) / (r1 + r2), 3)
            except ValueError:
                resistor_error = "Қате (Кедергі): Тек сандарды енгізіңіз."
    return render_template('index.html',
                           result_current=current,
                           error_message=ohm_error,
                           result_resistance=total_resistance,
                           error_resistance=resistor_error)


# --- МОДУЛЬ 3: ТЕСТ ЛОГИКАСЫ ---
@bp.route('/quiz', methods=['GET', 'POST'])
def quiz():
    if 'question_index' not in session:
        session['question_index'] = 0
        session['score'] = 0
    current_index = session['question_index']
    if request.method == 'POST':
        user_answer = request.form.get('option')
        prev_index = current_index - 1
        if 0 <= prev_index < len(QUIZ_QUESTIONS):
            correct_answer = QUIZ_QUESTIONS[prev_index]['answer']
            if user_answer == correct_answer:
                session['score'] += 1
    if current_index < len(QUIZ_QUESTIONS):
        question_data = QUIZ_QUESTIONS[current_index]
        session['question_index'] += 1
        return render_template('quiz.html',
                               question=question_data,
                               question_number=current_index + 1)
    else:
        score = session['score']
        total = len(QUIZ_QUESTIONS)
        session.pop('question_index', None)
        session.pop('score', None)
        return render_template('quiz_results.html',
                               score=score,
                               total=total)

@bp.route('/quiz_reset')
def quiz_reset():
    session.pop('question_index', None)
    session.pop('score', None)
    # МАҢЫЗДЫ: 'main.quiz' деп өзгертілді (бөлімнің аты + функция аты)
    return redirect(url_for('main.quiz'))