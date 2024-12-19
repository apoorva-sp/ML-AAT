import pandas as pd
from flask import Flask, render_template, request
from random import uniform
from main import pre_processing, linear_regression, decision_tree, random_forest, evaluate, encode, random_traffic, \
    random_weather
from path import path

app = Flask(__name__)

models = []
performances = []
performances_measure = []

df = pd.read_csv(path)
x_train, x_test, y_train, y_test = pre_processing(df)

linear_regression_model = linear_regression(x_train, x_test, y_train)
decision_tree_model = decision_tree(x_train, x_test, y_train)
random_forest_model = random_forest(x_train, x_test, y_train)

models.extend([linear_regression_model[0], decision_tree_model[0], random_forest_model[0]])
performances.extend([linear_regression_model[1], decision_tree_model[1], random_forest_model[1]])

for p in performances:
    performances_measure.append(evaluate(y_test, p))

@app.route('/')
def root():
    return render_template('home.html')


@app.route('/performance')
def performance():
    return render_template('performance.html', performance_data=performances_measure)


@app.route('/predicted', methods=["POST"])
def home():
    data = {
        'Trip_Distance_km': [float(request.form['distance'] or 0)],
        'Time_of_Day': [request.form['time']],
        'Day_of_Week': [request.form['day_type']],
        'Passenger_Count': [int(request.form['passengers'])],
        'Traffic_Conditions': [random_traffic()],
        'Weather': [random_weather()],
        'Base_Fare': [uniform(2.01,5.0)],
        'Per_Km_Rate': [uniform(0.5,2.0)],
        'Per_Minute_Rate': [uniform(0.1,0.5)],
        'Trip_Duration_Minutes': [float(request.form['duration'] or 0)]
    }
    user_data_frame = pd.DataFrame(data)
    training_columns = x_train.columns
    new_data = encode(user_data_frame).reindex(columns=training_columns, fill_value=0)
    model_selected = int(request.form['model'])
    predicted_fare = models[model_selected].predict(new_data)
    if int(data['Trip_Distance_km'][0]) == 0 or int(data['Trip_Duration_Minutes'][0]) == 0:
        predicted_fare = [0]
    return render_template('predicted.html', value=round(predicted_fare[0],2))


if __name__ == '__main__':
    app.run(debug=True)
