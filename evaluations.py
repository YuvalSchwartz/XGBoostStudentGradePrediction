import pandas as pd
import numpy as np
import scipy.stats as stats
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

class Evaluation:
    def __init__(self):
        alpha = 0.05

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
            print(f"The baseline score ({maeBaseline}) falls within the confidence interval of the modified model ({ci_lower:.4f} , {ci_upper:.4f}) at {alpha*100:.0f}% confidence level.")
            print(f"The difference is not significant.")
            return False
        else:
            print(f"The baseline score ({maeBaseline}) falls outside the confidence interval of the modified model ({ci_lower:.4f} , {ci_upper:.4f}) at {alpha*100:.0f}% confidence level.")
            print(f"The difference is significant.")
            return True

    def bootstrap_hypothesis_test(alpha_value=0.05, dataset_file="results.csv", num_bootstrap=1000, reference_mae=0.801):
        
        dataset = pd.read_csv(dataset_file)
        original_mae = reference_mae

        bootstrap_mae_scores = []
        for _ in range(num_bootstrap):
            bootstrap_sample = dataset.sample(frac=1, replace=True)
            mae = mean_absolute_error(bootstrap_sample['y_test'], bootstrap_sample['y_pred'])
            bootstrap_mae_scores.append(mae)
        
        lower_percentile = (1 - 0.05) / 2 * 100
        upper_percentile = 100 - lower_percentile
        lower_bound = np.percentile(bootstrap_mae_scores, lower_percentile)
        upper_bound = np.percentile(bootstrap_mae_scores, upper_percentile)
        
        # Hypothesis testing
        if original_mae < lower_bound:
            print("The original model performs significantly better than the new model.")
        elif original_mae > upper_bound:
            print("The new model performs significantly better than the original model.")
        else:
            print("There is no significant difference between the performance of the original and new models.")
        
        # Print confidence interval
        print("Bootstrap confidence interval ({}%): ({:.4f}, {:.4f})".format((1-0.05)*100, lower_bound, upper_bound))
        