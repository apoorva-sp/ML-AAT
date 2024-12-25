from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from main import *
from path import *
from sklearn.decomposition import PCA


def pre_processing_dataset():
    """
        num_col = ['Trip_Distance_km', 'Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes',
                    'Trip_Price']
        cat_cols = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather', 'Passenger_Count']
        columns = {0: 'Trip_Distance_km', 1: 'Time_of_Day', 2: 'Day_of_Week', 3: 'Passenger_Count',
                    4: 'Traffic_Conditions',5: 'Weather', 6: 'Base_Fare', 7: 'Per_Km_Rate', 8: 'Per_Minute_Rate',
                     9: 'Trip_Duration_Minutes', 10: 'Trip_Price'}
   """
    cat_col_indexes = [1, 2, 3, 4, 5]
    num_col_indexes = [0, 6, 7, 8, 9]

    df = pd.read_csv(path)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cat_col_indexes)], remainder='passthrough')

    x[:, num_col_indexes] = imp.fit_transform(x[:, num_col_indexes])
    y = y.reshape(-1, 1)
    y = imp.fit_transform(y).ravel()
    x = np.array(ct.fit_transform(x))
    sc = StandardScaler()
    x = sc.fit_transform(x)
    return train_test_split(x, y, test_size=0.2, random_state=0)


def eval_models():
    x_train, x_test, y_train, y_test = pre_processing_dataset()
    lr = linear_regression(x_train, x_test, y_train)
    dt = decision_tree(x_train, x_test, y_train)
    rf = random_forest(x_train, x_test, y_train)
    return evaluate(y_test, lr[1]), evaluate(y_test, dt[1]), evaluate(y_test, rf[1])


def pre_processing_dataset_with_pca():
    """
        num_col = ['Trip_Distance_km', 'Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes',
                    'Trip_Price']
        cat_cols = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather', 'Passenger_Count']
        columns = {0: 'Trip_Distance_km', 1: 'Time_of_Day', 2: 'Day_of_Week', 3: 'Passenger_Count',
                    4: 'Traffic_Conditions',5: 'Weather', 6: 'Base_Fare', 7: 'Per_Km_Rate', 8: 'Per_Minute_Rate',
                     9: 'Trip_Duration_Minutes', 10: 'Trip_Price'}
   """
    cat_col_indexes = [1, 2, 3, 4, 5]
    num_col_indexes = [0, 6, 7, 8, 9]

    df = pd.read_csv(path)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cat_col_indexes)], remainder='passthrough')

    x[:, num_col_indexes] = imp.fit_transform(x[:, num_col_indexes])
    y = y.reshape(-1, 1)
    y = imp.fit_transform(y).ravel()
    x = np.array(ct.fit_transform(x))
    sc = StandardScaler()
    x = sc.fit_transform(x)
    pca = PCA()
    x_reduced = pca.fit_transform(x)
    return train_test_split(x_reduced, y, test_size=0.2, random_state=0)


def eval_models_with_pca():
    x_train, x_test, y_train, y_test = pre_processing_dataset_with_pca()
    lr = linear_regression(x_train, x_test, y_train)
    dt = decision_tree(x_train, x_test, y_train)
    rf = random_forest(x_train, x_test, y_train)
    return evaluate(y_test, lr[1]), evaluate(y_test, dt[1]), evaluate(y_test, rf[1])

# eval_models()
# eval_models_with_pca()
