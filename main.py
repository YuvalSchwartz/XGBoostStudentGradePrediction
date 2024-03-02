import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna
from evaluation import Evaluation
from sklearn.linear_model import LinearRegression


seed = 42


def load_data():
    mat_data_path = os.path.abspath('data\\student-mat.csv')
    mat_df = pd.read_csv(mat_data_path)
    mat_df['course'] = 'mathematics'
    por_data_path = os.path.abspath('data\\student-por.csv')
    por_df = pd.read_csv(por_data_path)
    por_df['course'] = 'portuguese'
    students_df = pd.concat([mat_df, por_df])
    return students_df


def plot_g3_distribution(segment_to_bins=False):
    students_df = load_data()
    if segment_to_bins:
        data = pd.cut(students_df['G3'], bins=[-1, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20], labels=False)
        number_of_bins = 16 + 1
        color = 'g'
    else:
        data = students_df['G3']
        number_of_bins = 21 + 1
        color = 'y'

    # Plot histogram
    plt.figure()
    bins = np.arange(number_of_bins) - 0.5
    plt.hist(data, bins, density=True, alpha=0.6, color=color, edgecolor='black')
    plt.xticks(range(number_of_bins - 1))
    plt.xlim([-1, number_of_bins - 1])

    # Plot formatting
    plt.title(f"Distribution of Final Grades (G3){' Bins' if segment_to_bins else ''}")
    plt.xlabel(f"Final Grade (G3){' Bin' if segment_to_bins else ''}")
    plt.ylabel('Density')

    # Save the histogram image
    histogram_image_path = os.path.abspath(f"data\\g3{'_bins' if segment_to_bins else ''}_distribution_histogram.png")
    plt.savefig(histogram_image_path, dpi=300)
    plt.close()


def feature_selection(x, features_to_remove=[]):
    if 'school' in features_to_remove:
        x.drop(['school'], axis=1, inplace=True)
    if 'sex' in features_to_remove:
        x.drop(['sex'], axis=1, inplace=True)
    if 'age' in features_to_remove:
        x.drop(['age'], axis=1, inplace=True)
    if 'address' in features_to_remove:
        x.drop(['address'], axis=1, inplace=True)
    if 'famsize' in features_to_remove:
        x.drop(['famsize'], axis=1, inplace=True)
    if 'Pstatus' in features_to_remove:
        x.drop(['Pstatus'], axis=1, inplace=True)
    if 'Medu' in features_to_remove:
        x.drop(['Medu'], axis=1, inplace=True)
    if 'Fedu' in features_to_remove:
        x.drop(['Fedu'], axis=1, inplace=True)
    if 'Mjob' in features_to_remove:
        x.drop(['Mjob'], axis=1, inplace=True)
    if 'Fjob' in features_to_remove:
        x.drop(['Fjob'], axis=1, inplace=True)
    if 'reason' in features_to_remove:
        x.drop(['reason'], axis=1, inplace=True)
    if 'guardian' in features_to_remove:
        x.drop(['guardian'], axis=1, inplace=True)
    if 'traveltime' in features_to_remove:
        x.drop(['traveltime'], axis=1, inplace=True)
    if 'studytime' in features_to_remove:
        x.drop(['studytime'], axis=1, inplace=True)
    if 'failures' in features_to_remove:
        x.drop(['failures'], axis=1, inplace=True)
    if 'schoolsup' in features_to_remove:
        x.drop(['schoolsup'], axis=1, inplace=True)
    if 'famsup' in features_to_remove:
        x.drop(['famsup'], axis=1, inplace=True)
    if 'paid' in features_to_remove:
        x.drop(['paid'], axis=1, inplace=True)
    if 'activities' in features_to_remove:
        x.drop(['activities'], axis=1, inplace=True)
    if 'nursery' in features_to_remove:
        x.drop(['nursery'], axis=1, inplace=True)
    if 'higher' in features_to_remove:
        x.drop(['higher'], axis=1, inplace=True)
    if 'internet' in features_to_remove:
        x.drop(['internet'], axis=1, inplace=True)
    if 'romantic' in features_to_remove:
        x.drop(['romantic'], axis=1, inplace=True)
    if 'famrel' in features_to_remove:
        x.drop(['famrel'], axis=1, inplace=True)
    if 'freetime' in features_to_remove:
        x.drop(['freetime'], axis=1, inplace=True)
    if 'goout' in features_to_remove:
        x.drop(['goout'], axis=1, inplace=True)
    if 'Dalc' in features_to_remove:
        x.drop(['Dalc'], axis=1, inplace=True)
    if 'Walc' in features_to_remove:
        x.drop(['Walc'], axis=1, inplace=True)
    if 'health' in features_to_remove:
        x.drop(['health'], axis=1, inplace=True)
    if 'absences' in features_to_remove:
        x.drop(['absences'], axis=1, inplace=True)
    if 'G1' in features_to_remove:
        x.drop(['G1'], axis=1, inplace=True)
    if 'G2' in features_to_remove:
        x.drop(['G2'], axis=1, inplace=True)
    if 'course' in features_to_remove:
        x.drop(['course'], axis=1, inplace=True)
    if 'G1_G2_interaction' in features_to_remove:
        x.drop(['G1_G2_interaction'], axis=1, inplace=True)
    if 'G1_to_G2_improvement_rate' in features_to_remove:
        x.drop(['G1_to_G2_improvement_rate'], axis=1, inplace=True)
    if 'alcohol_consumption_ratio' in features_to_remove:
        x.drop(['alcohol_consumption_ratio'], axis=1, inplace=True)
    if 'parental_education_difference' in features_to_remove:
        x.drop(['parental_education_difference'], axis=1, inplace=True)
    if 'parental_occupation_difference' in features_to_remove:
        x.drop(['parental_occupation_difference'], axis=1, inplace=True)
    if 'family_relationship_score' in features_to_remove:
        x.drop(['family_relationship_score'], axis=1, inplace=True)
    if 'study_free_time_ratio' in features_to_remove:
        x.drop(['study_free_time_ratio'], axis=1, inplace=True)
    if 'G3_is_zero' in features_to_remove:
        x.drop(['G3_is_zero'], axis=1, inplace=True)


def find_worst_feature_to_remove():
    initial_results_df = run_xgboost([])
    initial_score, _, _ = Evaluation(initial_results_df).bootstrap_hypothesis_test()
    print(f"Initial Score: {initial_score}")

    # Define all possible features including engineered ones
    all_features = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
                    'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities',
                    'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',
                    'health', 'absences', 'G1', 'G2', 'course']

    current_features = all_features.copy()
    best_score = float('inf')
    features_to_remove = []

    while True:
        scores = {}
        for feature in current_features:
            temp_features_to_remove = features_to_remove + [feature]
            results_df = run_xgboost(temp_features_to_remove)
            score, _, _ = Evaluation(results_df).bootstrap_hypothesis_test()
            scores[feature] = score
            print(f"Removing {temp_features_to_remove} resulted in a score of {score}")

        best_feature_to_remove = min(scores, key=scores.get)
        if scores[best_feature_to_remove] < best_score:
            print(f"Removing {best_feature_to_remove} improved score to {scores[best_feature_to_remove]}")
            best_score = scores[best_feature_to_remove]
            features_to_remove.append(best_feature_to_remove)
            current_features.remove(best_feature_to_remove)
        else:
            print("No improvement from removing any more features.")
            break

    print("Features to Remove:", features_to_remove)


def convert_object_to_category(x):
    cat_columns = x.select_dtypes(['object']).columns
    x[cat_columns] = x[cat_columns].astype('category')


def find_optimal_params(features_to_remove=[]):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 750),
            'max_depth': trial.suggest_int('max_depth', 1, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.25),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 1.0),
            'alpha': trial.suggest_float('reg_alpha', 0.01, 10.0),
            'lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
            'gamma': trial.suggest_float('gamma', 0.01, 10.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100),
            'base_score': trial.suggest_float('base_score', 0.1, 0.5),
            'random_state': seed
        }
        results_df = run_xgboost(features_to_remove, params)
        bootstrap_mae, _, _ = Evaluation(results_df).bootstrap_hypothesis_test()
        return bootstrap_mae
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


def run_xgboost(features_to_remove=[], params=None):
    np.random.seed(seed)
    students_df = load_data()

    students_df['G3_binned'] = pd.cut(students_df['G3'], bins=[-1, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20], labels=False)

    x = students_df.drop(['G3'], axis=1)
    y = students_df['G3']
    feature_selection(x, features_to_remove)
    convert_object_to_category(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=students_df['G3_binned'])
    x_train.drop(['G3_binned'], axis=1, inplace=True)
    x_test.drop(['G3_binned'], axis=1, inplace=True)

    if params is None:
        model = xgb.XGBRegressor(enable_categorical=True)
    else:
        model = xgb.XGBRegressor(enable_categorical=True, **params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # y_pred = np.round(y_pred)
    threshold = 0.54
    y_pred = np.where(y_pred - np.floor(y_pred) < threshold, np.floor(y_pred), np.ceil(y_pred))
    y_pred = np.clip(y_pred, 0, 20)
    return pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})


def high_significance_impact_features_ablation():
    high_significance_impact_features = ['Medu', 'Fedu', 'schoolsup', 'studytime', 'higher', 'internet', 'Dalc']
    pass


def moderate_significance_impact_features_ablation():
    moderate_significance_impact_features = ['school', 'reason', 'sex', 'address', 'Pstatus', 'Mjob', 'Fjob', 'paid', 'activities', 'nursery']
    pass


def low_significance_impact_features_ablation():
    low_significance_impact_features = ['age', 'famsize', 'guardian', 'traveltime', 'failures', 'romantic', 'famrel', 'freetime', 'health']
    pass


def main():
    # features_to_remove = find_worst_feature_to_remove()
    features_to_remove = ['Medu', 'internet', 'age']
    # optimal_params = find_optimal_params(features_to_remove)
    optimal_params = {
        'max_depth': 7,
        'learning_rate': 0.18738960451491402,
        'n_estimators': 338,
        'min_child_weight': 1,
        'gamma': 4.855706080517466,
        'subsample': 0.9999457493872772,
        'colsample_bytree': 0.937642220368115,
        'reg_alpha': 4.079739682294067,
        'reg_lambda': 0.01160143803332787,
        'scale_pos_weight': 1.0186083476091863,
        'random_state': seed
    }
    results_df = run_xgboost(features_to_remove, optimal_params)
    results_df.to_csv('results.csv')
    evaluation = Evaluation(results_df)
    evaluation.bootstrap_hypothesis_test()
    evaluation.plot_maes_histogram()
    evaluation.plot_actual_vs_predicted()


if __name__ == '__main__':
    main()
