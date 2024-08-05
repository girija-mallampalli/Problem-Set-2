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

def calibration_light():
    # Load the test results
    df_arrests_test_lr = pd.read_csv('data/df_arrests_test_lr.csv')
    df_arrests_test_dt = pd.read_csv('data/df_arrests_test_dt.csv')

    # Create calibration plot for logistic regression model
    calibration_plot(df_arrests_test_lr['y'], df_arrests_test_lr['pred_lr'], n_bins=5)

    # Create calibration plot for decision tree model
    calibration_plot(df_arrests_test_dt['y'], df_arrests_test_dt['pred_dt'], n_bins=5)

    # Determine which model is more calibrated
    print("Which model is more calibrated? The decision tree model is more calibrated" )
    
    # Extra Credit
    # Compute PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
    top_50_lr = df_arrests_test_lr.nlargest(50, 'pred_lr')
    ppv_lr = top_50_lr['y'].mean()
    print(f"PPV for logistic regression model (top 50): {ppv_lr:.2f}")

    # Compute PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
    top_50_dt = df_arrests_test_dt.nlargest(50, 'pred_dt')
    ppv_dt = top_50_dt['y'].mean()
    print(f"PPV for decision tree model (top 50): {ppv_dt:.2f}")

    # Compute AUC for the logistic regression model
    lr_auc = auc(df_arrests_test_lr['y'], df_arrests_test_lr['pred_lr'])
    print(f"AUC for logistic regression model: {lr_auc:.2f}")

    # Compute AUC for the decision tree model
    dt_auc = auc(df_arrests_test_dt['y'], df_arrests_test_dt['pred_dt'])
    print(f"AUC for decision tree model: {dt_auc:.2f}")
