'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    #Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

# Logistic Regression calibration plot
y_true_lr = df_arrests_test['y']
y_prob_lr = gs_cv.predict_proba(df_arrests_test[['current_charge_felony', 'num_fel_arrests_last_year']])[:, 1]
calibration_plot(y_true_lr, y_prob_lr, n_bins=5)

# Decision Tree calibration plot
y_true_dt = df_arrests_test['y']
y_prob_dt = gs_cv_dt.predict_proba(df_arrests_test[['current_charge_felony', 'num_fel_arrests_last_year']])[:, 1]
calibration_plot(y_true_dt, y_prob_dt, n_bins=5)

# Which model is more calibrated?
print("Which model is more calibrated? You can judge this based on how closely the calibration curve follows the 45-degree line.")

# Extra Credit
# Compute PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
top_50_lr = np.argsort(y_prob_lr)[-50:]
ppv_lr = precision_score(y_true_lr.iloc[top_50_lr], np.ones(50))
print(f"PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk: {ppv_lr}")

# Compute PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
top_50_dt = np.argsort(y_prob_dt)[-50:]
ppv_dt = precision_score(y_true_dt.iloc[top_50_dt], np.ones(50))
print(f"PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk: {ppv_dt}")

# Compute AUC for the logistic regression model
auc_lr = roc_auc_score(y_true_lr, y_prob_lr)
print(f"AUC for the logistic regression model: {auc_lr}")

# Compute AUC for the decision tree model
auc_dt = roc_auc_score(y_true_dt, y_prob_dt)
print(f"AUC for the decision tree model: {auc_dt}")

# Do both metrics agree that one model is more accurate than the other?
print(f"Do both metrics agree that one model is more accurate than the other? PPV: {'LR' if ppv_lr > ppv_dt else 'DT'}, AUC: {'LR' if auc_lr > auc_dt else 'DT'}")
