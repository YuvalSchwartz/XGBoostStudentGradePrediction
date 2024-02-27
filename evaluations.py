import pandas as pd

class Evaluation:
    def __init__(self):
        pass

    def calculate_mae(self, df):
        if not all(column in df.columns for column in ['index', 'actual value', 'predicted value']):
            raise ValueError("DataFrame must contain columns: 'index', 'y_test', 'y_pred'")

        df['absolute error'] = abs(df['actual value'] - df['predicted value'])
        mae_score = df['absolute error'].mean()
        return mae_score