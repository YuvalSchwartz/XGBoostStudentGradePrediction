import pandas as pd
import numpy as np
import scipy.stats as stats
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_error


class Evaluation:
    def __init__(self, results_df):
        self.results_df = results_df
        self.alpha = 0.05
        self.number_of_bootstrap_samples = 10000
        self.mae_bootstraps, self.mean_mae, self.lower_bound, self.upper_bound = self.calculate_bootstrap()

    def get_bootstrap_mae_scores(self):
        return self.mae_bootstraps

    def get_mean_mae(self):
        return self.mean_mae

    def get_lower_bound(self):
        return self.lower_bound

    def get_upper_bound(self):
        return self.upper_bound

    def get_mae_and_ci_string(self):
        return f"MAE = {self.mean_mae:.3f}, {(1 - self.alpha) * 100:g}% CI: {self.lower_bound:.3f}-{self.upper_bound:.3f} (based on {self.number_of_bootstrap_samples} bootstraps)."


    # def calculateMAE(self) -> float:
    #     return (abs(self.results_df['y_test'] - self.results_df['y_pred'])).mean()

    # def calaculateMAE_WithRounding(self) -> float:
    #     return (abs(self.results_df['y_test'] - self.results_df['y_pred'].round())).mean()

    # def calculateStandardDeviation(self) -> float:
    #     return (abs(self.results_df['y_test'] - self.results_df['y_pred'])).std()

    # def confidenceItervalTest(self, maeBaseline, maeModified=0.801, standartDeviation=1) -> bool:
    #     z_score = np.quantile(np.random.normal(0, 1, 10000), 1 - self.alpha / 2)
    #     print(f'Z-score: {z_score}')
    #
    #     ci_lower = maeModified - z_score * standartDeviation
    #     ci_upper = maeModified + z_score * standartDeviation
    #
    #     if ci_lower <= maeBaseline <= ci_upper:
    #         print(f"The baseline score ({maeBaseline}) falls within the confidence interval of the modified model ({ci_lower:.3f} , {ci_upper:.3f}) at {self.alpha * 100:.0f}% confidence level.")
    #         print(f"The difference is not significant.")
    #         return False
    #     else:
    #         print(f"The baseline score ({maeBaseline}) falls outside the confidence interval of the modified model ({ci_lower:.3f} , {ci_upper:.3f}) at {self.alpha * 100:.0f}% confidence level.")
    #         print(f"The difference is significant.")
    #         return True

    def calculate_bootstrap(self):
        mae_bootstraps = []
        for _ in range(self.number_of_bootstrap_samples):
            bootstrap_sample = self.results_df.sample(frac=1, replace=True)
            mae = mean_absolute_error(bootstrap_sample['y_test'], bootstrap_sample['y_pred'])
            mae_bootstraps.append(mae)

        mean_mae = np.mean(mae_bootstraps)
        lower_percentile = self.alpha / 2 * 100
        upper_percentile = 100 - lower_percentile
        lower_bound = np.percentile(mae_bootstraps, lower_percentile)
        upper_bound = np.percentile(mae_bootstraps, upper_percentile)
        return mae_bootstraps, mean_mae, lower_bound, upper_bound

    def bootstrap_hypothesis_test(self, reference_mae=0.801):
        if reference_mae < self.lower_bound:
            print("The original model performs significantly better than the given model.")
        elif reference_mae > self.upper_bound:
            print("The given model performs significantly better than the original model.")
        else:
            print("There is no significant difference between the performance of the original and given models.")

    def plot_maes_histogram(self):
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(self.mae_bootstraps, bins=12, color='lightsteelblue', ec="k")
        ax_hist.set_title(f'Distribution of MAEs Using {self.number_of_bootstrap_samples} Bootstraps')
        ax_hist.set_xlabel('MAEs')
        ax_hist.set_ylabel('Frequency')
        fig_hist.subplots_adjust(bottom=0.18, left=0.18)
        ax_hist.axvline(x=self.lower_bound, color='dimgrey', linestyle='--', linewidth=1.25)
        ax_hist.axvline(x=self.upper_bound, color='dimgrey', linestyle='--', linewidth=1.25)
        legend_patch = Patch(color='lightsteelblue', label=f"XGBoost (MAE = {self.mean_mae:.3f}, {(1 - self.alpha) * 100:g}% CI: {self.lower_bound:.3f}-{self.upper_bound:.3f})")
        ax_hist.legend(handles=[legend_patch], loc='upper center', bbox_to_anchor=(0.5, -0.15))

        # Save the histogram image
        histogram_image_name = f"maes_histogram.png"
        plt.savefig(histogram_image_name, dpi=300)
        plt.close(fig_hist)

    def plot_actual_vs_predicted(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(self.results_df['y_test'], self.results_df['y_pred'], color='b', alpha=0.4, zorder=5, label=f"XGBoostRegressor (MAE = {self.mean_mae:.3f}, {(1 - self.alpha) * 100:g}% CI: {self.lower_bound:.3f}-{self.upper_bound:.3f})")
        ax.set_aspect('equal')
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="dimgrey")
        plt.xticks(range(0, 21, 1))
        plt.yticks(range(0, 21, 1))
        ax.set_title("Actual vs. Predicted 'G3' Values")
        ax.set_xlabel("Actual 'G3' Values")
        ax.set_ylabel("Predicted 'G3' Values")
        fig.subplots_adjust(bottom=0.18, left=0.18)
        plt.grid(alpha=0.4)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fontsize=17)
        scatter_plot_image_name = f"actual_vs_predicted.png"
        plt.savefig(scatter_plot_image_name, dpi=300)
        plt.close(fig)
