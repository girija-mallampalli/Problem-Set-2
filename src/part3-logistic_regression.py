'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: pred_universe, num_fel_arrests_last_year
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr


# Your code here
# Read in df_arrests
df_arrests = pd.read_csv('data/df_arrests.csv')

# Use train_test_split to create training and testing datasets
df_arrests_train, df_arrests_test = train_test_split(df_arrests, test_size=0.3, shuffle=True, stratify=df_arrests['y'])

# Create a list of feature names
features_list = ['current_charge_felony', 'num_fel_arrests_last_year']

# Create a parameter grid for the C hyperparameter
parameter_grid = {'C': [0.01, 0.1, 1, 10, 100]}

# Initialize the Logistic Regression model
lr_model = LogisticRegression()

# Initialize GridSearchCV with the logistic regression model and parameter grid
gs_cv = GridSearchCV(lr_model, parameter_grid, cv=5)

# Fit the model on the training data
gs_cv.fit(df_arrests_train[features_list], df_arrests_train['y'])

# Get the optimal value for C
optimal_C = gs_cv.best_params_['C']

# Determine the level of regularization
regularization_level = 'most regularization' if optimal_C == min(parameter_grid['C']) else 'least regularization' if optimal_C == max(parameter_grid['C']) else 'in the middle'

# Print the optimal value for C and the regularization level
print(f"What was the optimal value for C? {optimal_C}")
print(f"Did it have the most or least regularization? Or in the middle? {regularization_level}")

# Predict for the test set
df_arrests_test['pred_lr'] = gs_cv.predict(df_arrests_test[features_list])

# Save the training and test datasets for further use
df_arrests_train.to_csv('data/df_arrests_train.csv', index=False)
df_arrests_test.to_csv('data/df_arrests_test.csv', index=False)

# Return the training and test dataframes for use in main.py
df_arrests_train, df_arrests_test
