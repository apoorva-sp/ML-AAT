from datetime import datetime

import googlemaps
import pandas as pd
from flask import Flask, render_template, request
from random import uniform
from main import pre_processing, linear_regression, decision_tree, random_forest, evaluate, encode, random_traffic, \
    random_weather
from path import path, api_key
from pre_processing import eval_models_with_pca, eval_models

gmaps = googlemaps.Client(key=api_key)
app = Flask(__name__)

models = []
performances = []
performances_measure = []
names = ["Linear Regression", "Decision Tree CART", "Random Forest"]

df = pd.read_csv(path)
x_train, x_test, y_train, y_test = pre_processing(df)

linear_regression_model = linear_regression(x_train, x_test, y_train)
decision_tree_model = decision_tree(x_train, x_test, y_train)
random_forest_model = random_forest(x_train, x_test, y_train)
evaluated_with_pca = eval_models_with_pca()
evaluated_without_pca = eval_models()
models.extend([linear_regression_model[0], decision_tree_model[0], random_forest_model[0]])

for p in range(3):
    performances_measure.append([names[p], evaluated_without_pca[p]])

for i in range(3):
    performances_measure.append([names[i] + " with PCA", evaluated_with_pca[i]])


@app.route('/')
def root():
    return render_template('home.html')


@app.route('/performance')
def performance():
    return render_template('performance.html', performance_data=performances_measure)


@app.route('/predicted', methods=["POST"])
def home():
    now = datetime.now()
    directions_result = gmaps.directions(request.form['source'],
                                         request.form['destination'],
                                         mode="driving",
                                         departure_time=now)

    distance = directions_result[0]["legs"][0]["distance"]["value"] / 1000
    duration = directions_result[0]["legs"][0]["duration"]["value"] / 60
    data = {
        'Trip_Distance_km': [distance],
        'Time_of_Day': [request.form['time']],
        'Day_of_Week': [request.form['day_type']],
        'Passenger_Count': [int(request.form['passengers'])],
        'Traffic_Conditions': [random_traffic()],
        'Weather': [random_weather()],
        'Base_Fare': [uniform(2.01, 5.0)],
        'Per_Km_Rate': [uniform(0.5, 2.0)],
        'Per_Minute_Rate': [uniform(0.1, 0.5)],
        'Trip_Duration_Minutes': [duration]
    }
    user_data_frame = pd.DataFrame(data)
    training_columns = x_train.columns
    new_data = encode(user_data_frame).reindex(columns=training_columns, fill_value=0)
    predicted_fare = 0
    for m in range(3):
        predicted_fare += models[m].predict(new_data)[0]
    predicted_fare = round(predicted_fare / 3, 2)
    if int(data['Trip_Distance_km'][0]) == 0 or int(data['Trip_Duration_Minutes'][0]) == 0:
        predicted_fare = 0
    return render_template('predicted.html', value=round(predicted_fare, 2))


if __name__ == '__main__':
    app.run(debug=True)
