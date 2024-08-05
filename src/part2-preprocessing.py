'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd

# Your code here

# Load the CSV files into dataframes
pred_universe = pd.read_csv('data/pred_universe_raw.csv')
arrest_events = pd.read_csv('data/arrest_events_raw.csv')

# Perform a full outer join on 'person_id'
df_arrests = pd.merge(pred_universe, arrest_events, on='person_id', how='outer')

# Create 'y' column for rearrests within 365 days
df_arrests['y'] = 0


# Loop through each row in df_arrests to calculate 'y'
for idx, row in df_arrests.iterrows():
    arrest_date = row['arrest_date_event']
    person_id = row['person_id']
    if not pd.isnull(arrest_date):
        start_date = arrest_date + pd.DateOffset(days=1)
        end_date = arrest_date + pd.DateOffset(years=1)
        felony = arrest_events[(arrest_events['person_id'] == person_id) & 
                                 (arrest_events['arrest_date_event'] >= start_date) & 
                                 (arrest_events['arrest_date_event'] <= end_date) & 
                                 (arrest_events['charge_type'] == 'felony')]
        if not felony.empty:
            df_arrests.at[idx, 'y'] = 1
          
#Calculate the share of arrestees rearrested for a felony crime in the next year
share_rearrested = df_arrests['y'].mean()
print(f"What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year? {share_rearrested}")

# Create 'current_charge_felony' column to indicate if the current charge is a felony
df_arrests['current_charge_felony'] = df_arrests['charge_type'].apply(lambda x: 1 if x == 'felony' else 0)

# Calculate the share of current charges that are felonies
share_felony_charges = df_arrests['current_charge_felony'].mean()
print(f"What share of current charges are felonies? {share_felony_charges}")

# Create 'num_fel_arrests_last_year' column to count felony arrests in the past year
df_arrests['num_fel_arrests_last_year'] = 0

# Loop through each row in df_arrests to calculate 'num_fel_arrests_last_year'
for idx, row in df_arrests.iterrows():
    arrest_date = row['arrest_date_event']
    person_id = row['person_id']
    if not pd.isnull(arrest_date):
        start_date = pd.to_datetime(arrest_date) - pd.Timedelta(days=365)
        end_date = pd.to_datetime(arrest_date) - pd.Timedelta(days=1)
        felonies = arrest_events[(arrest_events['person_id'] == person_id) & 
                                 (arrest_events['arrest_date_event'] >= start_date) & 
                                 (arrest_events['arrest_date_event'] <= end_date) & 
                                 (arrest_events['charge_type'] == 'felony')]
        df_arrests.at[idx, 'num_fel_arrests_last_year'] = len(felonies)

# Calculate the average number of felony arrests in the last year
avg_felony_arrests_last_year = df_arrests['num_fel_arrests_last_year'].mean()
print(f"What is the average number of felony arrests in the last year? {avg_felony_arrests_last_year}")

# Print the mean of 'num_fel_arrests_last_year'
print(f"Mean of 'num_fel_arrests_last_year': {df_arrests['num_fel_arrests_last_year'].mean()}")

# Print the first few rows of pred_universe
print(pred_universe.head())

# Save df_arrests for use in main.py
df_arrests.to_csv('data/df_arrests.csv', index=False)

# Return df_arrests for use in main.py
df_arrests
