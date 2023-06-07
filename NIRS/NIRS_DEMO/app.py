from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Загрузка данных
data = pd.read_csv('/Users/temas/Downloads/Air Quality.csv')
data = data.fillna(method='ffill')

features = ['NOx(GT)', 'NO2(GT)']
X = data[features]
y = data['CO(GT)']

X = X.astype(str)  # Преобразование признаков в строковый тип данных
y = y.astype(str)  # Преобразование целевой переменной в строковый тип данных


# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация модели
model = LinearRegression()

# Обучение модели на обучающей выборке
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Получение новых значений гиперпараметров из запроса
    nox_value = float(request.form['nox'])
    no2_value = float(request.form['no2'])

    # Создание нового входного примера на основе новых значений гиперпараметров
    new_data = np.array([[nox_value, no2_value]])

    # Предсказание с использованием обновленной модели
    prediction = model.predict(new_data)

    return render_template('index.html', prediction=str(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)