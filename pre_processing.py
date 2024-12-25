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
    imp_categorical = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cat_col_indexes)], remainder='passthrough')

    x[:, num_col_indexes] = imp.fit_transform(x[:, num_col_indexes])
    x[:, cat_col_indexes] = imp_categorical.fit_transform(x[:, cat_col_indexes])
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

    df_pca = pd.read_csv(path)
    x = df_pca.iloc[:, :-1].values
    y = df_pca.iloc[:, -1].values

    imp_numerical = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp_categorical = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cat_col_indexes)], remainder='passthrough')

    x[:, num_col_indexes] = imp_numerical.fit_transform(x[:, num_col_indexes])
    # x[:, cat_col_indexes] = imp_categorical.fit_transform(x[:, cat_col_indexes])
    y = y.reshape(-1, 1)
    y = imp_numerical.fit_transform(y).ravel()
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


def check_null_values_after_preprocessing(x, y):
    null_in_x = np.isnan(x).any()
    null_in_y = np.isnan(y).any()
    return {"null_in_features": null_in_x, "null_in_target": null_in_y}


def check_nan_values():
    x_train, x_test, y_train, y_test = pre_processing_dataset()
    null_check_no_pca_train = check_null_values_after_preprocessing(x_train, y_train)
    null_check_no_pca_test = check_null_values_after_preprocessing(x_test, y_test)

    print("Null check after preprocessing (no PCA):")
    print("Training Data:", null_check_no_pca_train)
    print("Testing Data:", null_check_no_pca_test)

    x_train_pca, x_test_pca, y_train_pca, y_test_pca = pre_processing_dataset_with_pca()
    null_check_with_pca_train = check_null_values_after_preprocessing(x_train_pca, y_train_pca)
    null_check_with_pca_test = check_null_values_after_preprocessing(x_test_pca, y_test_pca)

    print("\nNull check after preprocessing (with PCA):")
    print("Training Data:", null_check_with_pca_train)
    print("Testing Data:", null_check_with_pca_test)


# check_nan_values()
