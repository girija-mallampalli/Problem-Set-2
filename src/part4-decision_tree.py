'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC

# Read in the dataframes from PART 3
df_arrests_train = pd.read_csv('data/df_arrests_train.csv')
df_arrests_test = pd.read_csv('data/df_arrests_test.csv')

# Create a parameter grid for tree depth
parameter_grid_dt = {'max_depth': [5, 10, 15]}

# Initialize the Decision Tree model
dt_model = DecisionTreeClassifier()

# Initialize GridSearchCV with the decision tree model and parameter grid
gs_cv_dt = GridSearchCV(dt_model, parameter_grid_dt, cv=5)

# Fit the model on the training data
gs_cv_dt.fit(df_arrests_train[['current_charge_felony', 'num_fel_arrests_last_year']], df_arrests_train['y'])

# Get the optimal value for max_depth
opt_max_depth = gs_cv_dt.best_params_['max_depth']
print(f"What was the optimal value for max_depth? {opt_max_depth}")
print("Did it have the most or least regularization? Or in the middle?")
if opt_max_depth == 5:
  print("Most regularization")
elif opt_max_depth == 15:
  print("Least regularization")
else:
  print("In the middle")

# Predict for the test set
df_arrests_test['pred_dt'] = gs_cv_dt.predict(df_arrests_test[['current_charge_felony', 'num_fel_arrests_last_year']])

# Save the training and test datasets for further use
df_arrests_train.to_csv('data/df_arrests_train.csv', index=False)
df_arrests_test.to_csv('data/df_arrests_test.csv', index=False)

# Return the training and test dataframes for use in main.py
return df_arrests_train, df_arrests_test
