import cv2
from ultralytics import YOLO
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from collections import Counter
import os
import json
import logging

# Настройка логирования
logging.basicConfig(filename='app.log', encoding='UTF-8', level=logging.INFO, format='%(asctime)s - %(message)s')

# Инициализация модели YOLO
model = YOLO('yolov8m-seg.pt')

# Инициализация Dash приложения
app = dash.Dash(__name__)

# Список классов объектов
class_names = model.names
image_dir = 'images'
users_file = 'users.json'

# Инициализация состояния авторизации
is_authenticated = False
users = {}

# Загрузка пользователей из файла
def load_users():
    global users
    if os.path.exists(users_file):
        with open(users_file, 'r') as f:
            users = json.load(f)

# Сохранение пользователей в файл
def save_users():
    with open(users_file, 'w') as f:
        json.dump(users, f)

# Обработка изображений один раз
def process_images():
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    object_types = []

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is not None:
            result = model(frame, iou=0.4, conf=0.6)

            for box in result[0].boxes:
                object_types.append(class_names[int(box.cls.item())])

    return object_types

# Загрузка пользователей при старте приложения
load_users()

# Макет приложения
app.layout = html.Div([
    html.Div(id='auth-container', children=[
        dcc.Input(id='username-input', type='text', placeholder='Имя пользователя'),
        dcc.Input(id='password-input', type='password', placeholder='Пароль'),
        html.Button('Войти', id='login-button', n_clicks=0),
        html.Button('Зарегистрироваться', id='register-button', n_clicks=0),
        html.Div(id='access-output')
    ]),
    html.Div(id='dashboard-container', style={'display': 'none'}, children=[
        html.H1("Анализ объектов на изображениях"),
        html.Button('Обработать изображения', id='process-button', n_clicks=0),
        html.Div(id='output-container'),
        dcc.Graph(id='object-distribution-area'),
        dcc.Graph(id='object-distribution-heatmap'),
    ])
])

# Обновление графиков в Dash
@app.callback(
    Output('object-distribution-area', 'figure'),
    Output('object-distribution-heatmap', 'figure'),
    Output('output-container', 'children'),
    Output('access-output', 'children'),
    Output('dashboard-container', 'style'),
    Output('auth-container', 'style'),
    Input('process-button', 'n_clicks'),
    Input('login-button', 'n_clicks'),
    Input('register-button', 'n_clicks'),
    Input('username-input', 'value'),
    Input('password-input', 'value')
)
def update_graph(process_clicks, login_clicks, register_clicks, username, password):
    global is_authenticated, users

    # Начальные значения для всех выходных параметров
    area_fig = {}
    heatmap_fig = {}
    output_message = ''
    access_message = ''
    dashboard_style = {'display': 'none'}
    auth_style = {'display': 'block'}

    if int(process_clicks) != 0:
        if not is_authenticated:
            output_message = 'Необходимо войти в систему для обработки изображений.'
            return area_fig, heatmap_fig, output_message, '', dashboard_style, auth_style

        object_types = process_images()
        object_count = Counter(object_types)
        labels = list(object_count.keys())
        values = list(object_count.values())

        # График с областями
        area_fig = {
            'data': [{
                'x': labels,
                'y': values,
                'type': 'scatter',
                'mode': 'lines+fill',
                'name': 'Распределение типов объектов',
                'fill': 'tozeroy'
            }],
            'layout': {
                'title': 'График с областями',
                'xaxis': {'title': 'Типы объектов'},
                'yaxis': {'title': 'Количество'}
            }
        }

        # Тепловая карта
        heatmap_fig = {
            'data': [{
                'z': [values],
                'x': labels,
                'y': ['Объекты'],
                'type': 'heatmap',
                'colorscale': 'Viridis'
            }],
            'layout': {
                'title': 'Тепловая карта распределения объектов',
                'xaxis': {'title': 'Типы объектов'},
                'yaxis': {'title': 'Объекты'},
                'height': 300
            }
        }

        output_message = f'Обработано объектов: {sum(values)}'
        dashboard_style = {'display': 'block'}
        auth_style = {'display': 'none'}


    if login_clicks > 0:
        if username in users and users[username] == password:
            is_authenticated = True
            access_message = "Доступ предоставлен."
            # Обнуляем клики
            return area_fig, heatmap_fig, '', access_message, {'display': 'block'}, {'display': 'none'}
        else:
            access_message = "Неверные учетные данные. Попробуйте снова."
            return area_fig, heatmap_fig, '', access_message, dashboard_style, auth_style

    if register_clicks > 0:
        if username in users:
            access_message = "Пользователь уже существует."
        else:
            users[username] = password
            save_users()  # Сохранить пользователей после регистрации
            access_message = "Регистрация успешна. Теперь вы можете войти."
        # Обнуляем клики
        return area_fig, heatmap_fig, '', access_message, dashboard_style, auth_style

    return area_fig, heatmap_fig, output_message, access_message, dashboard_style, auth_style

if __name__ == '__main__':
    app.run_server(debug=True)
