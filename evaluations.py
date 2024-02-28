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
    
    def oneSampleTTest(self, maeBaseline, maeModified, standartDeviation, sampleSize, alpha=0.05) -> bool:
        t_statistic, p_value = stats.ttest_1samp([maeModified], maeBaseline, 0)
        
        print (f't_statistic: {t_statistic}')
        print (f'p_value: {p_value}')
        
        if p_value < alpha:
            return True # Significant
        else:
            return False 