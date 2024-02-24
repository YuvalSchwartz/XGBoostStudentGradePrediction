import os
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb


def load_data():
    mat_data_path = os.path.abspath('data\\student-mat.csv')
    mat_df = pd.read_csv(mat_data_path)
    mat_df['course'] = 'mathematics'
    por_data_path = os.path.abspath('data\\student-por.csv')
    por_df = pd.read_csv(por_data_path)
    por_df['course'] = 'portuguese'
    students_df = pd.concat([mat_df, por_df])
    return students_df


def main():
    students_df = load_data()
    x = students_df.drop(['G3'], axis=1)
    y = students_df['G3']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier()
    model.fit(x_train, y_train)


if __name__ == '__main__':
    main()
