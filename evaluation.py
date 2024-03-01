import pandas as pd
import numpy as np
import scipy.stats as stats
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_error

class Evaluation:
    def __init__(self, results_file_path="results.csv"):
        self.results_df = pd.read_csv(results_file_path)
        self.alpha = 0.05
        self.number_of_bootstrap_samples = 1000

    def calculateMAE(self) -> float:
        return (abs(self.results_df['y_test'] - self.results_df['y_pred'])).mean()

    def calaculateMAE_WithRounding(self) -> float:
        return (abs(self.results_df['y_test'] - self.results_df['y_pred'].round())).mean()

    def calculateStandardDeviation(self) -> float:
        return (abs(self.results_df['y_test'] - self.results_df['y_pred'])).std()

    def confidenceItervalTest(self, maeBaseline, maeModified=0.801, standartDeviation=1) -> bool:
        z_score = np.quantile(np.random.normal(0, 1, 10000), 1 - self.alpha / 2)
        print(f'Z-score: {z_score}')
        
        ci_lower = maeModified - z_score * standartDeviation
        ci_upper = maeModified + z_score * standartDeviation
        
        if ci_lower <= maeBaseline <= ci_upper:
            print(f"The baseline score ({maeBaseline}) falls within the confidence interval of the modified model ({ci_lower:.3f} , {ci_upper:.3f}) at {self.alpha * 100:.0f}% confidence level.")
            print(f"The difference is not significant.")
            return False
        else:
            print(f"The baseline score ({maeBaseline}) falls outside the confidence interval of the modified model ({ci_lower:.3f} , {ci_upper:.3f}) at {self.alpha * 100:.0f}% confidence level.")
            print(f"The difference is significant.")
            return True

    def bootstrap_hypothesis_test(self, reference_mae=0.801):
        bootstrap_mae_scores = []
        for _ in range(self.number_of_bootstrap_samples):
            bootstrap_sample = self.results_df.sample(frac=1, replace=True)
            mae = mean_absolute_error(bootstrap_sample['y_test'], bootstrap_sample['y_pred'])
            bootstrap_mae_scores.append(mae)

        lower_percentile = self.alpha / 2 * 100
        upper_percentile = 100 - lower_percentile
        lower_bound = np.percentile(bootstrap_mae_scores, lower_percentile)
        upper_bound = np.percentile(bootstrap_mae_scores, upper_percentile)

        # Print MAE and confidence interval
        print(f"MAE = {np.mean(bootstrap_mae_scores):.3f}, {(1 - self.alpha) * 100:g}% CI: {lower_bound:.3f}-{upper_bound:.3f} (based on {self.number_of_bootstrap_samples} bootstraps).")
        
        # Hypothesis testing
        if reference_mae < lower_bound:
            print("The original model performs significantly better than the new model.")
        elif reference_mae > upper_bound:
            print("The new model performs significantly better than the original model.")
        else:
            print("There is no significant difference between the performance of the original and new models.")

    def plot_maes_histogram(self):
        bootstrap_mae_scores = []
        for _ in range(self.number_of_bootstrap_samples):
            bootstrap_sample = self.results_df.sample(frac=1, replace=True)
            mae = mean_absolute_error(bootstrap_sample['y_test'], bootstrap_sample['y_pred'])
            bootstrap_mae_scores.append(mae)

        lower_percentile = self.alpha / 2 * 100
        upper_percentile = 100 - lower_percentile
        lower_bound = np.percentile(bootstrap_mae_scores, lower_percentile)
        upper_bound = np.percentile(bootstrap_mae_scores, upper_percentile)

        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(bootstrap_mae_scores, bins=12, color='lightsteelblue', ec="k")
        ax_hist.set_title(f'Distribution of MAEs Using {self.number_of_bootstrap_samples} Bootstraps')
        ax_hist.set_xlabel('MAEs')
        ax_hist.set_ylabel('Frequency')
        fig_hist.subplots_adjust(bottom=0.18, left=0.18)
        ax_hist.axvline(x=lower_bound, color='dimgrey', linestyle='--', linewidth=1.25)
        ax_hist.axvline(x=upper_bound, color='dimgrey', linestyle='--', linewidth=1.25)
        legend_patch = Patch(color='lightsteelblue', label=f"XGBoost (MAE = {np.mean(bootstrap_mae_scores):.3f}, {(1 - self.alpha) * 100:g}% CI: {lower_bound:.3f}-{upper_bound:.3f})")
        ax_hist.legend(handles=[legend_patch], loc='upper center', bbox_to_anchor=(0.5, -0.15))

        # Save the histogram image
        histogram_image_name = f"maes_histogram.png"
        plt.savefig(histogram_image_name, dpi=300)
        plt.close(fig_hist)
