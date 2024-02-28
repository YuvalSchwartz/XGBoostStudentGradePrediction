import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
import optuna
from xgboost import plot_importance
import matplotlib.pyplot as plt


def load_data():
    mat_data_path = os.path.abspath('data\\student-mat.csv')
    mat_df = pd.read_csv(mat_data_path)
    mat_df['course'] = 'mathematics'
    por_data_path = os.path.abspath('data\\student-por.csv')
    por_df = pd.read_csv(por_data_path)
    por_df['course'] = 'portuguese'
    students_df = pd.concat([mat_df, por_df])
    cat_columns = students_df.select_dtypes(['object']).columns
    students_df[cat_columns] = students_df[cat_columns].astype('category')
    return students_df


def find_optimal_params(X, y, seed):
    def objective(trial):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
            'random_state': trial.suggest_int('random_state', 1, 1000)
        }
        model = xgb.XGBRegressor(**params, enable_categorical=True)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred = np.round(y_pred)
        y_pred = np.clip(y_pred, 0, 20)
        mae = mean_absolute_error(y_test, y_pred)
        return mae

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1000)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
    return trial.params


def custom_round(x, threshold):
    if isinstance(x, np.ndarray):
        return np.floor(x) + (x % 1 >= threshold)
    else:
        return np.floor(x) + (x % 1 >= threshold)


def run_regressor(X, y, optimal_params, seed):
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    mae_list = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = xgb.XGBRegressor(**optimal_params, enable_categorical=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = custom_round(y_pred, 0.55)
        y_pred = np.clip(y_pred, 0, 20)

        mae = mean_absolute_error(y_test, y_pred)
        mae_list.append(mae)

    avg_mae = np.mean(mae_list)
    print(f'Average Mean Absolute Error across {n_splits} folds: {avg_mae}')

    # importance = model.get_booster().get_score(importance_type='weight')
    # print(importance)

    # fig, ax = plt.subplots(figsize=(10, 8))
    # plot_importance(model, ax=ax, importance_type='weight')
    # plt.show()


def main():
    seed = 42
    students_df = load_data()
    # students_df.drop(['address', 'famsup', 'nursery', 'internet'], axis=1, inplace=True)
    X = students_df.drop(['G3'], axis=1)
    y = students_df['G3']
    optimal_params = {
        'max_depth': 3,
        'learning_rate': 0.12029386320296027,
        'n_estimators': 50,
        'min_child_weight': 15,
        'gamma': 0.41122087230906346,
        'subsample': 0.8653434717474726,
        'colsample_bytree': 0.9708333738490725,
        'reg_alpha': 0.2221104520628081,
        'reg_lambda': 0.04597590979587908,
        'random_state': 660
    }
    run_regressor(X, y, optimal_params, seed)


if __name__ == '__main__':
    main()
