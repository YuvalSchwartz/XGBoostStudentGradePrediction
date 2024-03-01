import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna
from xgboost import plot_importance
from evaluation import Evaluation


def load_data():
    mat_data_path = os.path.abspath('data\\student-mat.csv')
    mat_df = pd.read_csv(mat_data_path)
    mat_df['course'] = 'mathematics'
    por_data_path = os.path.abspath('data\\student-por.csv')
    por_df = pd.read_csv(por_data_path)
    por_df['course'] = 'portuguese'
    students_df = pd.concat([mat_df, por_df])
    return students_df


def feature_selection(X):
    # X.drop(['higher', 'Pstatus'], axis=1, inplace=True)
    return X


def feature_extraction(X):
    # X['G1_G2_interaction'] = np.abs(X['G1'] - X['G2'])
    # X['G1_to_G2_improvement_rate'] = (X['G2'] - X['G1']) / 2
    return X


def convert_object_to_category(X):
    cat_columns = X.select_dtypes(['object']).columns
    X[cat_columns] = X[cat_columns].astype('category')
    return X


def find_optimal_params(X, y):
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 750),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
            'random_state': trial.suggest_int('random_state', 1, 1000)
        }
        model_results = run_xgboost(X, y, params)
        mae = mean_absolute_error(model_results['y_test'], model_results['y_pred'])
        return mae
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=3000)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
    return trial.params


def run_xgboost(X, y, params=None):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    if params is None:
        model = xgb.XGBRegressor(enable_categorical=True)
    else:
        model = xgb.XGBRegressor(enable_categorical=True, **params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = np.round(y_pred)
    y_pred = np.clip(y_pred, 0, 20)
    importance = model.get_booster().get_score(importance_type='weight')
    features_not_used = set(X.columns) - set(importance.keys())
    if len(features_not_used) > 0:
        print(f"Features not used: {features_not_used}")
    print(importance)
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_importance(model, ax=ax, importance_type='weight')
    plt.show()
    results = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    return results


def main():
    np.random.seed(42)
    students_df = load_data()
    X = students_df.drop(['G3'], axis=1)
    X = feature_selection(X)
    X = feature_extraction(X)
    X = convert_object_to_category(X)
    y = students_df['G3']
    # optimal_params = find_optimal_params(X, y)
    # optimal_params = {
    #     'max_depth': 3,
    #     'learning_rate': 0.010322048435422228,
    #     'n_estimators': 384,
    #     'min_child_weight': 19,
    #     'gamma': 0.9272196990826382,
    #     'subsample': 0.918293434562889,
    #     'colsample_bytree': 0.9986801767250817,
    #     'reg_alpha': 0.4746275842672103,
    #     'reg_lambda': 0.7525318986291032,
    #     'random_state': 908
    # }
    # run_regressor(X, y, optimal_params)
    model_results = run_xgboost(X, y)
    model_results.to_csv('results.csv')
    evaluation = Evaluation()
    evaluation.bootstrap_hypothesis_test()
    evaluation.plot_maes_histogram()

if __name__ == '__main__':
    main()
