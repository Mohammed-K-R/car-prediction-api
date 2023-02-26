from datetime import datetime

import dill
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def drop_columns(df):
    columns_to_drop = ['id',
                       'url',
                       'region',
                       'region_url',
                       'price',
                       'manufacturer',
                       'image_url',
                       'description',
                       'posting_date',
                       'lat',
                       'long']

    return df.drop(columns_to_drop, axis=1)


def restore_year(df):
    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)

        return boundaries

    border = calculate_outliers(df['year'])

    df.loc[df['year'] < border[0], 'year'] = round(border[0])
    df.loc[df['year'] > border[1], 'year'] = round(border[1])

    return df


def new_short_columns(df):
    def short_model(x):
        if not pd.isna(x):
            return x.lower().split(' ')[0]
        else:
            return x

    df.loc[:, 'short_model'] = df['model'].apply(short_model)
    df.loc[:, 'age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))

    return df


def find_best_model(X, y, models, preprocessor):
    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy', n_jobs=-1)
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    return best_pipe, best_score


def main():
    df = pd.read_csv("data/homework.csv", sep=",")

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)

    numerical_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('drop_columns_before', FunctionTransformer(drop_columns)),
        ('restore_year', FunctionTransformer(restore_year)),
        ('new_short_columns', FunctionTransformer(new_short_columns)),
        ('column_transformer', column_transformer)
    ])

    X = df.drop(['price_category'], axis=1)
    y = df['price_category']

    models = [
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    ]

    types = {'int64': 'int',
             'float64': 'float',
             'object': 'str'}
    column_types = dict()
    for column, value in df.dtypes.iteritems():
        column_types[column] = types.get(str(value), 'str')

    best_pipe, best_score = find_best_model(X, y, models, preprocessor)
    best_pipe.fit(X, y)
    model_file = 'car_pipe.pkl'
    with open(model_file, 'wb') as file:
        dill.dump({'model': best_pipe,
                   'metadata': {
                       'name': 'Car price prediction model',
                       'author': 'Mohammed Karimov',
                       'version': 1,
                       'date': datetime.now(),
                       'type': type(best_pipe.named_steps["classifier"]).__name__,
                       'accuracy': best_score
                   },
                   'column_types': column_types
                   }, file)


if __name__ == '__main__':
    main()
