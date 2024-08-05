'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    etl.etl()

    # PART 2: Call functions/instanciate objects from preprocessing
    df_arrests = preprocessing.preprocess()

    # PART 3: Call functions/instanciate objects from logistic_regression
    lr_model, df_arrests_test_lr = logistic_regression.logistic_regression()

    # PART 4: Call functions/instanciate objects from decision_tree
    dt_model, df_arrests_test_dt = decision_tree.decision_tree()
    
    # PART 5: Call functions/instanciate objects from calibration_plot
    calibration_plot.calibration_light()

if __name__ == "__main__":
    main()
