from flask import Flask, render_template, request
from preprocessing import train_model, get_attributes, measures, predict
from random import uniform

x_train, x_test, y_train, y_test = get_attributes()
train_model(x_train, y_train)
# print(x_test)
f = open('model.dat', "rb")
mse, mae, rmse, r2 = measures(x_test, y_test, f)
f.close()

app = Flask(__name__)


@app.route('/')
def root():
    return render_template('index.html')


@app.route('/predicted', methods=["POST"])
def home():
    data = request.form
    Cycle_Index = float(data['cycleIndex'])
    Discharge_Time = float(data['dischargeTime'])
    Decrement = uniform(-397645.908,406703.768)
    Maxi = float(data['maxi'])
    Mini = float(data['mini'])
    T4 = uniform(-113.584,245101.117)
    T = float(data['timeConst'])
    Charging_time = float(data['chargingTime'])
    data = [Cycle_Index, Discharge_Time, Decrement, Maxi, Mini, T4, T, Charging_time]
    F = open('model.dat', "rb")
    prediction = predict(F, [data])[0]
    F.close()
    return render_template('predicted.html', value=round(prediction, 2))


if __name__ == '__main__':
    app.run(debug=True)
