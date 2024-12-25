import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from random import choice


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def encode(df):
    cat_cols = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather', 'Passenger_Count']
    df.dropna(inplace=True)
    df = one_hot_encoder(df, cat_cols, drop_first=True)
    return df


def pre_processing(df):
    df = encode(df)
    y = df["Trip_Price"]
    X = df.drop(["Trip_Price"], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return x_train, x_test, y_train, y_test


def linear_regression(x_train, x_test, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return model, y_pred


def decision_tree(x_train, x_test, y_train):
    model = DecisionTreeRegressor(max_depth=5)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return model, y_pred


def random_forest(x_train, x_test, y_train):
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return model, y_pred


def evaluate(y_test, y_pred):
    mse = round(mean_squared_error(y_test, y_pred), 2)
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    r2 = round(r2_score(y_test, y_pred), 2)
    return mse, mae, r2


def random_traffic():
    return choice(["Low", "High", "Medium"])

def random_weather():
    return choice(["Clear", "Rain", "Snow"])

# df = pd.read_csv(path)
# x_train, x_test, y_train, y_test = pre_processing(df)
#
# lrmodel, y_pred = linear_regression(x_train, x_test, y_train)
# print(y_pred[:5])
# print(y_test[:5])
# mse, mae, r2 = evaluate(y_test, y_pred)
# print(f"Linear Regression - MSE: {mse}, MAE: {mae}, R2: {r2}")
#
# rand_model, y_pred = random_forest(x_train, x_test, y_train)
# print(y_pred[:5])
# print(y_test[:5])
# mse, mae, r2 = evaluate(y_test, y_pred)
# print(f"Linear Regression - MSE: {mse}, MAE: {mae}, R2: {r2}")
#
# decision_tree_model, y_pred = decision_tree(x_train, x_test, y_train)
# print(y_pred[:5])
# print(y_test[:5])
# mse, mae, r2 = evaluate(y_test, y_pred)
# print(f"Linear Regression - MSE: {mse}, MAE: {mae}, R2: {r2}")
#
# # prompt: give a sample data to predict the output
#
#
# # Create a sample DataFrame (replace with your actual data)
# data = {
#     'Trip_Distance_km': [120],
#     'Time_of_Day': ['Morning'],
#     'Day_of_Week': ['Weekday'],
#     'Passenger_Count': [1],
#     'Traffic_Conditions': ['Light'],
#     'Weather': ['Sunny'],
#     'Base_Fare': [5.0],
#     'Per_Km_Rate': [1.5],
#     'Per_Minute_Rate': [0.5],
#     'Trip_Duration_Minutes': [10]
# }
# df1 = pd.DataFrame(data)
#
# training_columns = x_train.columns
#
# new_data = encode(df1).reindex(columns=training_columns, fill_value=0)
#
# predicted_fare = rand_model.predict(new_data)
# print(predicted_fare)
# pred = lrmodel.predict(new_data)
# print(pred)
# p = decision_tree_model.predict(new_data)
# print(p)
