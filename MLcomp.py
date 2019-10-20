import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer


def replace_nan_categorical(x, cat_index):
    for j in cat_index:
        temporary = x[:, j]
        nan_index = pd.isnull(x[:, j])
        temporary[nan_index] = "unknown"
        temporary[temporary == "0"] = "unknown"
        temporary[temporary == "Unknown"] = "unknown"
        x[:, j] = temporary
    return x


def replace_nan_numerical(x, num_index):
    for j in num_index:
        simp = SimpleImputer(missing_values=np.nan, strategy='mean')
        simp.fit(x[:, j].reshape(-1, 1))
        x[:, j] = simp.transform(x[:, j].reshape(-1, 1))[:, 0]
    return x


train = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")
test = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")

train = train.drop(labels='Instance', axis=1)
test = test.drop(labels='Instance', axis=1)

mat_x = train.iloc[:, :-1].values
mat_y = train.iloc[:, -1].values
test_mat_x = test.iloc[:, :-1].values

categorical_index = [1, 3, 5, 6, 8]
numerical_index = [0, 2, 4, 7, 9]

mat_x = replace_nan_numerical(mat_x, numerical_index)
test_mat_x = replace_nan_numerical(test_mat_x, numerical_index)
mat_x = replace_nan_categorical(mat_x, categorical_index)
test_mat_x = replace_nan_categorical(test_mat_x, categorical_index)


concat_mat_x = LabelEncoder()
for i in categorical_index:
    category_list = np.concatenate((mat_x[:, i], test_mat_x[:, i]))
    unique_category_list = np.unique(category_list)
    concat_mat_x.fit(unique_category_list)
    mat_x[:, i] = concat_mat_x.transform(mat_x[:, i])
    test_mat_x[:, i] = concat_mat_x.transform(test_mat_x[:, i])

hot_encode = OneHotEncoder(categorical_features=categorical_index, sparse=False)
mat_x = hot_encode.fit_transform(mat_x)
test_mat_x = hot_encode.transform(test_mat_x)

scale = MinMaxScaler()
mat_x[:, -5:] = scale.fit_transform(mat_x[:, -5:])
test_mat_x[:, -5:] = scale.fit_transform(test_mat_x[:, -5:])
lr = LinearRegression()
lr.fit(mat_x, mat_y)
predict = lr.predict(mat_x)
np.savetxt("result.csv", predict, fmt="%f", newline='\n')

