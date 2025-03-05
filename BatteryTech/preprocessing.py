import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from pandas import read_csv
from pickle import load, dump


def get_attributes():
    df = read_csv("D:\ML-AAT\BatteryTech\Battery_RUL.csv", encoding='utf-8')
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    return x_train, x_test, y_train, y_test

def train_model(x_train, y_train):
    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    f = open("model.dat", "wb")
    dump(linear_model, f)
    f.close()


def measures(x_test,y_test,f):
    linear_model = load(f)
    y_pred = linear_model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, rmse, r2

def predict(f,data):
    linear_model = load(f)
    return linear_model.predict(data)

def main():
    f = open("model.dat","rb")
    x_train,x_test,y_train,y_test = get_attributes()
    # train_model(x_train,y_train)
    # only uncomment when needed to train
    mse, mae, rmse, r2 = measures(x_test, y_test,f)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("MAE: ", mae)
    print("R2: ", r2)
    f.close()
