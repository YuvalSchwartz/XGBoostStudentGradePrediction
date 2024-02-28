import os
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import evaluations

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


def main():
    seed = 56
    students_df = load_data()
    x = students_df.drop(['G3'], axis=1)
    y = students_df['G3']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    model = xgb.XGBRegressor(objective='reg:squarederror', enable_categorical=True, random_state=seed)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    results = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    results.to_csv('results.csv')
    mae = (abs(y_test - y_pred)).mean()
    print(f'Mean Absolute Error: {mae}')
    
    evaluator = evaluations.Evaluation()
    mae = evaluator.calculateMAE(results)
    print(f'Mean Absolute Error: {mae}')
    mae = evaluator.calaculateMAE_WithRounding(results)
    print(f'Mean Absolute Error with Rounding: {mae}')
    std = evaluator.calculateStandardDeviation(results)
    
    isSignificant = evaluator.oneSampleTTest(maeBaseline=mae, maeModified=0.803, standartDeviation=std, sampleSize=1104, alpha=0.05)
    print(f'Is the difference significant? {isSignificant}')

if __name__ == '__main__':
    main()
