import pandas as pd
import numpy as np
import scipy.stats as stats

class Evaluation:
    def __init__(self):
        pass

    def calculateMAE(self, df) -> float:
        return (abs(df['y_test'] - df['y_pred'])).mean()
        

    def calaculateMAE_WithRounding(self, df) -> float:
        return (abs(df['y_test'] - df['y_pred'].round())).mean()
    
    def calculateStandardDeviation(self, df) -> float:
        return (abs(df['y_test'] - df['y_pred'])).std()
    
    def confidenceItervalTest(self, maeBaseline, maeModified=0.801, standartDeviation=1, alpha=0.05) -> bool:
        z_score = np.quantile(np.random.normal(0, 1, 10000), 1 - alpha/2)
        print(f'Z-score: {z_score}')
        
        ci_lower = maeModified - z_score * standartDeviation
        ci_upper = maeModified + z_score * standartDeviation
        
        if ci_lower <= maeBaseline <= ci_upper:
            print(f"The SOTA score ({maeBaseline}) falls within the confidence interval of your model ({ci_lower:.4f} , {ci_upper:.4f}) at {alpha*100:.0f}% confidence level.")
            print(f"The difference is not significant.")
            return False
        else:
            print(f"The SOTA score ({maeBaseline}) falls outside the confidence interval of your model ({ci_lower:.4f} , {ci_upper:.4f}) at {alpha*100:.0f}% confidence level.")
            print(f"The difference is significant.")
            return True
