# Classic libraries
import pandas as pd
import numpy as np
import pickle

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import imblearn

# Sklearn models
from sklearn.linear_model import LogisticRegression

# ----------------------------------------------- Final preprocessing functions --------------------------------------------

def final_cleaning(data):
    
    # Dropping keys
    keys = data['SK_ID_CURR']
    if 'Unnamed: 0' in data.columns:
        data.drop(['Unnamed: 0', 'SK_ID_CURR'], axis=1, inplace=True)
    else:
        data.drop(['SK_ID_CURR'], axis=1, inplace=True)

    # Splitting inputs and labels
    y = data['TARGET']
    x = data.drop(['TARGET'], axis=1)

    # We replace inf values by NaN
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return x, y, keys

def final_transform(x, categorical_cols, numerical_cols, existing_preprocessor=None):

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('stdscaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    if not existing_preprocessor: # If no trained preprocessor is passed we create a new one
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        # Preprocess datas
        x = preprocessor.fit_transform(x)
    else: # Else we use the existing trained preprocessor to transform datas
        preprocessor = existing_preprocessor
        x = preprocessor.transform(x)
    
    return x, preprocessor

def final_balance(x, y):
    # Balance datas
    over = imblearn.over_sampling.SMOTE(sampling_strategy=0.1)
    under = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=0.5)
    x, y = over.fit_resample(x, y)
    x, y = under.fit_resample(x, y)
    
    return x, y

def get_column_names(x, categorical_cols):
    
    # Apply one-hot encoder to each column with categorical data
    mode_impute = SimpleImputer(strategy='most_frequent')
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    transformed = mode_impute.fit_transform(x[categorical_cols])
    transformed = OH_encoder.fit_transform(transformed)
    transformed_df = pd.DataFrame(transformed, columns=OH_encoder.get_feature_names_out(input_features=categorical_cols))

    # One-hot encoding removed index; put it back
    transformed_df.index = x.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_data = x.drop(categorical_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    full_cols = pd.concat([num_data, transformed_df], axis=1)

    # Transforming our matrices in a df with the columns names
    x = pd.DataFrame(x, columns=full_cols.columns)
    
    return full_cols

def final_preprocessing(data, is_balance=True, existing_preprocessor=None, is_existing_cols=False, full_cols=None):
    """Preprocess datas for training our model"""
    
    # Drop keys, split x, y and replace infinites values
    x, y, keys = final_cleaning(data)
    
    # Defining numerical and categorical columns
    categorical_cols = [col for col in x.columns if x[col].dtype == 'object']
    numerical_cols = list(x.drop(categorical_cols, axis=1).columns)
    
    if not is_existing_cols:

        # Get new columns names after OH encoding
        full_cols = get_column_names(x, categorical_cols)
    
    # Preprocessing with imputation, standardization and encoding
    x, preprocessor = final_transform(x, categorical_cols, numerical_cols, existing_preprocessor=existing_preprocessor)
    
    # Over and undersampling to balance classes
    if is_balance:
        x, y = final_balance(x, y)
    
    # Put back x in a df with column names and client idx
    x = pd.DataFrame(x, columns=full_cols, index=keys)

    return x, y, preprocessor


# ----------------------------------------------- Joint and preprocess dataframes --------------------------------------------

def select_one_mode_value(x):
    """ 
    After an aggregation of mode values in a dataframe for a given variable, 
    this function selects only one mode value if several are returned during aggregation.
    Use this function with apply on a pd.Series with modes values for each row (ex : df[var].apply(lambda x: select_one_mode_value(x)))
    
    Parameters
    ----------
    - x : one-row result from a pandas mode aggregation (list, str)
    """

    if isinstance(x, str): # If we have only one value, the type is a str. We keep this value
        return x

    elif isinstance(x, np.ndarray):
        if x.size == 0: # If the value is a NaN we have an empty array
            return np.nan
        else: # If we have several value, it's stored in a nparray, we take only the first value of this array
            return x[0]

def bureau_balance_preprocess(bureau_balance, bureau):
    """Preprocess bureau_balance df and merge with bureau"""
    
    # Check if we have at least one corresponding key between bureau_balance and bureau
    is_corresp = False

    for id_bureau in bureau_balance['SK_ID_BUREAU']:
        if id_bureau in set(bureau['SK_ID_BUREAU']):
            is_corresp = True
            break

    # If empty bureau_balance or no correspondance with bureau we return bureau with NaN for new columns
    if bureau_balance.empty or not is_corresp:
        add_cols = ['STATUS_0_PERC', 'STATUS_1_PERC', 'STATUS_2_PERC', 'STATUS_3_PERC', 'STATUS_4_PERC', 'STATUS_5_PERC', 'STATUS_X_PERC']
        for col in add_cols:
            bureau[col] = np.nan
        return bureau
    
    else:
    
        # Creation of a temporary df
        df_temp = pd.DataFrame()

        # Duration of the credit in months (only months for which the credit isn't closed) 
        df_temp['CREDIT_DURATION'] = bureau_balance[bureau_balance['STATUS'] != 'C'].groupby(by='SK_ID_BUREAU')['STATUS'].apply(len)

        # Percentage of time of the credit per status
        status_list = list(bureau_balance['STATUS'].unique())
        status_list.remove('C')

        for s in status_list:
            df_temp['STATUS_' + s + '_DURATION'] = bureau_balance[bureau_balance['STATUS'] == s].groupby(by='SK_ID_BUREAU')['STATUS'].apply(len)
            df_temp['STATUS_' + s + '_PERC'] = df_temp['STATUS_' + s + '_DURATION'] / df_temp['CREDIT_DURATION']
            del df_temp['STATUS_' + s + '_DURATION']

        # Replace NaN values which corresponds to 0 days for this status
        df_temp.fillna(0, inplace=True)

        bureau = bureau.join(df_temp, how='left', on='SK_ID_BUREAU')
        bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
        del df_temp
        del bureau_balance

        return bureau

def bureau_preprocess(bureau, application_data, imputer=None):
    """Preprocess bureau df and merge with application_data"""

    # Check if we have at least one corresponding key between bureau and application_data
    is_corresp = False

    for id_curr in bureau['SK_ID_CURR']:
        if id_curr in set(application_data['SK_ID_CURR']):
            is_corresp = True
            break
    
    if bureau.empty or not is_corresp:
        return application_data, imputer    
    else:
    
        # -----------------We first process datas to impute missing values of annuity-----------------

        # We focus only on active credits
        active_credits = bureau[bureau['CREDIT_ACTIVE'] == 'Active']

        # Get indexes
        closed_indexes = active_credits[(active_credits['DAYS_CREDIT_ENDDATE'] < 0) & (active_credits['AMT_CREDIT_SUM'] == 0)].index

        # Correct status on bureau df
        bureau.iloc[closed_indexes]['CREDIT_ACTIVE'] == 'Closed'

        # Delete rows on active_credits df
        active_credits.drop(closed_indexes, inplace=True)

        # And set values with DAYS_CREDIT_ENDDATE < 0 and AMT_CREDIT_SUM > 0 to NaN
        active_credits.loc[active_credits['DAYS_CREDIT_ENDDATE'] < 0, 'DAYS_CREDIT_ENDDATE'] = np.nan

        # For values at 0 for AMT_ANNUITY and DAYS_CREDIT_ENDDATE, we will assume it corresponds to NaN values
        active_credits['AMT_ANNUITY'].replace(0, np.nan, inplace=True)
        active_credits['DAYS_CREDIT_ENDDATE'].replace(0, np.nan, inplace=True)

        if imputer:
            my_imputer = imputer
        else:
            # Imputation by the mean
            my_imputer = SimpleImputer(strategy='mean')

            # Fitting on all active credits
            my_imputer.fit(active_credits['DAYS_CREDIT_ENDDATE'].values.reshape(-1,1))

        # Transforming only on active credits without AMT_ANNUITY
        mask = active_credits['AMT_ANNUITY'].isna()
        imputed_DAYS_CREDIT_ENDDATE = my_imputer.transform(active_credits[mask]['DAYS_CREDIT_ENDDATE'].values.reshape(-1,1))

        # Replacing NaN with imputed values
        active_credits.loc[mask, 'DAYS_CREDIT_ENDDATE'] = imputed_DAYS_CREDIT_ENDDATE

        nan_annuity = active_credits[mask] # Mask for active credits without AMT_ANNUITY

        # Define our formula
        def f(x):
            i = 0.025 # Assumption
            return x['E'] * (i / (1-((1+i)**(-x['n']))))

        # Apply on masked dataframe
        nan_annuity['n'] = nan_annuity['DAYS_CREDIT_ENDDATE'].apply(lambda p: p / 365) # We assume payments are done yearly
        nan_annuity['E'] = nan_annuity['AMT_CREDIT_SUM'] # We compute the annuity as if it's a new loan for the amount of remaining capital to be paid
        nan_annuity['A'] = nan_annuity.apply(f, axis=1)

        # Replace values on active_credits df
        active_credits.loc[mask, 'AMT_ANNUITY'] = nan_annuity['A']
        del nan_annuity

        # -----------------Feature engineering-----------------

        # Creation of a temporary df
        df_temp = pd.DataFrame()

        # Conditional aggregation on active loans
        df_temp['BUREAU_TOTAL_ACTIVE_AMT_CREDIT_SUM'] = active_credits.groupby(by='SK_ID_CURR')['AMT_CREDIT_SUM'].sum()
        df_temp['BUREAU_TOTAL_ACTIVE_AMT_ANNUITY'] = active_credits.groupby(by='SK_ID_CURR')['AMT_ANNUITY'].sum()
        df_temp['BUREAU_TOTAL_AMT_CREDIT_SUM_OVERDUE'] = active_credits.groupby(by='SK_ID_CURR')['AMT_CREDIT_SUM_OVERDUE'].sum()
        df_temp['BUREAU_TOTAL_DAYS_CREDIT_ENDDATE'] = active_credits.groupby(by='SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].sum()

        del active_credits

        # Aggregation on numeric variables (description of variable in comments)
        aggregations = {
            'DAYS_CREDIT': ['min', 'max', 'mean'], # How many days before current application did client apply for Credit Bureau credit
            'CREDIT_DAY_OVERDUE': ['max', 'mean'], # Number of days past due on CB credit at the time of application for related loan 
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'], # Remaining duration of CB credit (in days) at the time of application in Home Credit
            'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'], # Days since CB credit ended at the time of application in Home Credit
            'AMT_CREDIT_MAX_OVERDUE': ['max'], # Maximal amount overdue on the Credit Bureau credit so far (at application date of loan)
            'CNT_CREDIT_PROLONG': ['sum'], # How many times was the Credit Bureau credit prolonged
            'AMT_CREDIT_SUM': ['max', 'mean', 'sum'], # Current credit amount for the Credit Bureau credit
            'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'], # Current debt on Credit Bureau credit
            'AMT_CREDIT_SUM_LIMIT': ['max', 'mean', 'sum'], # Current credit limit of credit card reported in Credit Bureau
            'AMT_CREDIT_SUM_OVERDUE': ['mean'], # Current amount overdue on Credit Bureau credit
            'DAYS_CREDIT_UPDATE': ['mean'], # How many days before loan application did last information about the Credit Bureau credit come
            'AMT_ANNUITY': ['max', 'mean', 'sum'], # Annuity of the Credit Bureau credit
            'STATUS_0_PERC': ['mean'], # Feature enginnering from previous step
            'STATUS_X_PERC': ['mean'], # Feature enginnering from previous step
            'STATUS_1_PERC': ['mean'], # Feature enginnering from previous step
            'STATUS_2_PERC': ['mean'], # Feature enginnering from previous step
            'STATUS_3_PERC': ['mean'], # Feature enginnering from previous step
            'STATUS_4_PERC': ['mean'], # Feature enginnering from previous step
            'STATUS_5_PERC': ['mean'], # Feature enginnering from previous step
        }

        bureau_agg = bureau.groupby('SK_ID_CURR').agg(aggregations)
        bureau_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

        bureau_agg = bureau_agg.join(df_temp, how='left', on='SK_ID_CURR')
        del df_temp

        # Aggregation on categorical variables, we will apply mode (description of variable in comments)
        cat_var = [
            'CREDIT_ACTIVE', # Status of the Credit Bureau (CB) reported credits
            'CREDIT_CURRENCY', # Recoded currency of the Credit Bureau credit
            'CREDIT_TYPE' # Type of Credit Bureau credit (Car, cash,...)
        ]

        for var in cat_var:
            bureau_agg['BUREAU_' + var + '_MODE'] = bureau.groupby('SK_ID_CURR')[var].agg(pd.Series.mode)
            bureau_agg['BUREAU_' + var + '_MODE'] = bureau_agg['BUREAU_' + var + '_MODE'].apply(lambda x: select_one_mode_value(x))

        # -----------------Jointure-----------------

        application_data = application_data.join(bureau_agg, how='left', on='SK_ID_CURR')
        del bureau_agg
        del bureau

        return application_data, my_imputer

def previous_application_preprocess(previous_application, application_data):
    """Preprocess previous_application df and merge with application_data"""
    
   # Check if we have at least one corresponding key between previous_application and application_data
    is_corresp = False

    for id_curr in previous_application['SK_ID_CURR']:
        if id_curr in set(application_data['SK_ID_CURR']):
            is_corresp = True
            break   
    
    if previous_application.empty or not is_corresp:
        return application_data
    else:
    
        # -----------------Cleaning-----------------

        # Some days values are 365.243, we will replace by NaN
        previous_application['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
        previous_application['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
        previous_application['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
        previous_application['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
        previous_application['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)

        # -----------------Feature engineering-----------------

        # Value ask / value received percentage
        previous_application['APP_CREDIT_PERC'] = previous_application['AMT_APPLICATION'] / previous_application['AMT_CREDIT']

        # LTV (% of goods financed with the credit)
        previous_application['LTV_RATIO'] = previous_application['AMT_CREDIT'] / previous_application['AMT_GOODS_PRICE']

        # Previous applications numeric features
        num_aggregations = {
            'AMT_ANNUITY': ['min', 'max', 'mean'], # Annuity of previous application
            'AMT_APPLICATION': ['min', 'max', 'mean'], # For how much credit did client ask on the previous application
            'AMT_CREDIT': ['min', 'max', 'mean'], # Final credit amount on the previous application. This differs from AMT_APPLICATION in a way that the AMT_APPLICATION is the amount for which the client initially applied for, but during our approval process he could have received different amount - AMT_CREDIT
            'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'], # Down payment on the previous application
            'AMT_GOODS_PRICE': ['min', 'max', 'mean'], # Goods price of good that client asked for (if applicable) on the previous application
            'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'], # Approximately at what day hour did the client apply for the previous application
            #'NFLAG_LAST_APPL_IN_DAY' : ['min', 'max', 'mean'], # Flag if the application was the last application per day of the client. Sometimes clients apply for more applications a day. Rarely it could also be error in our system that one application is in the database twice
            #'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'], # Down payment rate normalized on previous credit
            #'RATE_INTEREST_PRIMARY' : ['min', 'max', 'mean'], # Interest rate normalized on previous credit
            #'RATE_INTEREST_PRIVILEGED' : ['min', 'max', 'mean'], # Interest rate normalized on previous credit
            'DAYS_DECISION': ['min', 'max', 'mean'], # Relative to current application when was the decision about previous application made
            #'SELLERPLACE_AREA', ['min', 'max', 'mean'], # Selling area of seller place of the previous application
            'CNT_PAYMENT': ['mean', 'sum'], # Term of previous credit at application of the previous application
            #'DAYS_FIRST_DRAWING' : ['min', 'max', 'mean'], # Relative to application date of current application when was the first disbursement of the previous application
            #'DAYS_FIRST_DUE' : ['min', 'max', 'mean'], # Relative to application date of current application when was the first due supposed to be of the previous application
            #'DAYS_LAST_DUE_1ST_VERSION' : ['min', 'max', 'mean'], # Relative to application date of current application when was the first due of the previous application
            #'DAYS_LAST_DUE' : ['min', 'max', 'mean'], # Relative to application date of current application when was the last due date of the previous application
            #'DAYS_TERMINATION' : ['min', 'max', 'mean'], # Relative to application date of current application when was the expected termination of the previous application
            #'NFLAG_INSURED_ON_APPROVAL' : ['min', 'max', 'mean'], # Did the client requested insurance during the previous application
            'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'], # Feature enginnering from previous step
            'LTV_RATIO' : ['min', 'max', 'mean', 'var'], # Feature enginnering from previous step
        }

        prev_agg = previous_application.groupby('SK_ID_CURR').agg(num_aggregations)
        prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

        # Previous Applications: Approved Applications - only numerical features
        approved = previous_application[previous_application['NAME_CONTRACT_STATUS'] == 'Approved']
        approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
        approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
        prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

        # Previous Applications: Refused Applications - only numerical features
        refused = previous_application[previous_application['NAME_CONTRACT_STATUS'] == 'Refused']
        refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
        refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
        prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
        del refused, refused_agg, approved, approved_agg

        # Aggregation on categorical variables, we will apply mode (description of variable in comments)
        cat_var = [
            'NAME_CONTRACT_TYPE', # Contract product type (Cash loan, consumer loan [POS] ,...) of the previous application
            'WEEKDAY_APPR_PROCESS_START', # On which day of the week did the client apply for previous application
            'FLAG_LAST_APPL_PER_CONTRACT', # Flag if it was last application for the previous contract. Sometimes by mistake of client or our clerk there could be more applications for one single contract
            'NFLAG_LAST_APPL_IN_DAY', # Flag if the application was the last application per day of the client. Sometimes clients apply for more applications a day. Rarely it could also be error in our system that one application is in the database twice
            'NAME_CASH_LOAN_PURPOSE', # Purpose of the cash loan
            'NAME_CONTRACT_STATUS', # Contract status (approved, cancelled, ...) of previous application
            'NAME_PAYMENT_TYPE', # Payment method that client chose to pay for the previous application
            'CODE_REJECT_REASON', # Why was the previous application rejected
            'NAME_TYPE_SUITE', # Who accompanied client when applying for the previous application
            'NAME_CLIENT_TYPE', # Was the client old or new client when applying for the previous application
            'NAME_GOODS_CATEGORY', # What kind of goods did the client apply for in the previous application
            'NAME_PORTFOLIO', # Was the previous application for CASH, POS, CAR, …
            'NAME_PRODUCT_TYPE', # Was the previous application x-sell o walk-in
            'CHANNEL_TYPE', # Through which channel we acquired the client on the previous application
            'NAME_SELLER_INDUSTRY', # The industry of the seller
            'NAME_YIELD_GROUP', # Grouped interest rate into small medium and high of the previous application
            'PRODUCT_COMBINATION', # Detailed product combination of the previous application
        ]

        for var in cat_var:
            prev_agg['PREV_' + var + '_MODE'] = previous_application.groupby('SK_ID_CURR')[var].agg(pd.Series.mode)
            prev_agg['PREV_' + var + '_MODE'] = prev_agg['PREV_' + var + '_MODE'].apply(lambda x: select_one_mode_value(x))

        # -----------------Jointure-----------------

        application_data = application_data.join(prev_agg, how='left', on='SK_ID_CURR')
        del prev_agg
        del previous_application

        return application_data

def credit_card_balance_preprocess(credit_card_balance, application_data):
    """Preprocess credit_card_balance df and merge with application_data"""
    
   # Check if we have at least one corresponding key between credit_card_balance and application_data
    is_corresp = False

    for id_curr in credit_card_balance['SK_ID_CURR']:
        if id_curr in set(application_data['SK_ID_CURR']):
            is_corresp = True
            break   
    
    if credit_card_balance.empty or not is_corresp:
        return application_data
    else:
    
        # Aggregations on numeric variables (description of variable in comments)
        aggregations = {
            'MONTHS_BALANCE': ['max', 'mean', 'sum', 'var'], # Month of balance relative to application date (-1 means the freshest balance date)
            'AMT_BALANCE': ['max', 'mean', 'sum', 'var'], # Balance during the month of previous credit
            'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean', 'sum', 'var'], # Credit card limit during the month of the previous credit
            'AMT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum', 'var'], # Amount drawing at ATM during the month of the previous credit
            'AMT_DRAWINGS_CURRENT': ['max', 'mean', 'sum', 'var'], # Amount drawing during the month of the previous credit
            'AMT_DRAWINGS_OTHER_CURRENT': ['max', 'mean', 'sum', 'var'], # Amount of other drawings during the month of the previous credit
            'AMT_DRAWINGS_POS_CURRENT': ['max', 'mean', 'sum', 'var'], # Amount drawing or buying goods during the month of the previous credit
            'AMT_INST_MIN_REGULARITY': ['max', 'mean', 'sum', 'var'], # Minimal installment for this month of the previous credit
            'AMT_PAYMENT_CURRENT': ['max', 'mean', 'sum', 'var'], # How much did the client pay during the month on the previous credit
            'AMT_PAYMENT_TOTAL_CURRENT': ['max', 'mean', 'sum', 'var'], # How much did the client pay during the month in total on the previous credit
            'AMT_RECEIVABLE_PRINCIPAL': ['max', 'mean', 'sum', 'var'], # Amount receivable for principal on the previous credit
            'AMT_RECIVABLE': ['max', 'mean', 'sum', 'var'], # Amount receivable on the previous credit
            'AMT_TOTAL_RECEIVABLE': ['max', 'mean', 'sum', 'var'], # Total amount receivable on the previous credit
            'CNT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum', 'var'], # Number of drawings at ATM during this month on the previous credit
            'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum', 'var'], # Number of drawings during this month on the previous credit
            'CNT_DRAWINGS_OTHER_CURRENT': ['max', 'mean', 'sum', 'var'], # Number of other drawings during this month on the previous credit
            'CNT_DRAWINGS_POS_CURRENT': ['max', 'mean', 'sum', 'var'], # Number of drawings for goods during this month on the previous credit
            'CNT_INSTALMENT_MATURE_CUM': ['max', 'mean', 'sum', 'var'], # Number of paid installments on the previous credit
            'SK_DPD': ['max', 'mean', 'sum', 'var'], # DPD (Days past due) during the month on the previous credit
            'SK_DPD_DEF': ['max', 'mean', 'sum', 'var'], # DPD (Days past due) during the month with tolerance (debts with low loan amounts are ignored) of the previous credit
        }

        cc_agg = credit_card_balance.groupby('SK_ID_CURR').agg(aggregations)
        cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

        # Count installments accounts
        cc_agg['CC_COUNT'] = credit_card_balance.groupby('SK_ID_CURR').size()

        # Aggregation on categorical variables, we will apply mode
        cat_var = [
            'NAME_CONTRACT_STATUS' # Contract status (active signed,...) on the previous credit
        ]

        for var in cat_var:
            cc_agg['CC_' + var + '_MODE'] = credit_card_balance.groupby('SK_ID_CURR')[var].agg(pd.Series.mode)
            cc_agg['CC_' + var + '_MODE'] = cc_agg['CC_' + var + '_MODE'].apply(lambda x: select_one_mode_value(x))

        # Merging dataframes
        application_data = application_data.join(cc_agg, how='left', on='SK_ID_CURR')
        del cc_agg
        del credit_card_balance

        return application_data

def POS_CASH_balance_preprocess(POS_CASH_balance, application_data):
    """Preprocess POS_CASH_balance df and merge with application_data"""

   # Check if we have at least one corresponding key between POS_CASH_balance and application_data
    is_corresp = False

    for id_curr in POS_CASH_balance['SK_ID_CURR']:
        if id_curr in set(application_data['SK_ID_CURR']):
            is_corresp = True
            break   
    
    if POS_CASH_balance.empty or not is_corresp:
        return application_data
    else:
        
        # Aggregations on numeric variables (description of variable in comments)
        aggregations = {
            'MONTHS_BALANCE': ['max', 'mean', 'size'], # Month of balance relative to application date (-1 means the information to the freshest monthly snapshot, 0 means the information at application - often it will be the same as -1 as many banks are not updating the information to Credit Bureau regularly )
            #'CNT_INSTALMENT': ['max', 'mean', 'sum', 'var'], # Term of previous credit (can change over time)
            #'CNT_INSTALMENT_FUTURE': ['max', 'mean', 'sum', 'var'], # Installments left to pay on the previous credit
            'SK_DPD': ['max', 'mean'], # DPD (days past due) during the month of previous credit
            'SK_DPD_DEF': ['max', 'mean'], # DPD during the month with tolerance (debts with low loan amounts are ignored) of the previous credit
        }

        pos_agg = POS_CASH_balance.groupby('SK_ID_CURR').agg(aggregations)
        pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

        # Count installments accounts
        pos_agg['POS_COUNT'] = POS_CASH_balance.groupby('SK_ID_CURR').size()

        # Aggregation on categorical variables, we will apply mode
        cat_var = [
            'NAME_CONTRACT_STATUS' # Contract status during the month
        ]

        for var in cat_var:
            pos_agg['POS_' + var + '_MODE'] = POS_CASH_balance.groupby('SK_ID_CURR')[var].agg(pd.Series.mode)
            pos_agg['POS_' + var + '_MODE'] = pos_agg['POS_' + var + '_MODE'].apply(lambda x: select_one_mode_value(x))

        # Merging dataframes
        application_data = application_data.join(pos_agg, how='left', on='SK_ID_CURR')
        del pos_agg
        del POS_CASH_balance

        return application_data

def installments_payments_preprocess(installments_payments, application_data):
    """Preprocess installments_payments df and merge with application_data"""
    
   # Check if we have at least one corresponding key between installments_payments and application_data
    is_corresp = False

    for id_curr in installments_payments['SK_ID_CURR']:
        if id_curr in set(application_data['SK_ID_CURR']):
            is_corresp = True
            break   
    
    if installments_payments.empty or not is_corresp:
        return application_data
    else:
    
        # Feature engineering

        # Percentage and difference paid in each installment (amount paid and installment value)
        installments_payments['PAYMENT_PERC'] = installments_payments['AMT_PAYMENT'] / installments_payments['AMT_INSTALMENT']
        installments_payments['PAYMENT_DIFF'] = installments_payments['AMT_INSTALMENT'] - installments_payments['AMT_PAYMENT']

        # Days past due and days before due (no negative values)
        installments_payments['DPD'] = installments_payments['DAYS_ENTRY_PAYMENT'] - installments_payments['DAYS_INSTALMENT']
        installments_payments['DBD'] = installments_payments['DAYS_INSTALMENT'] - installments_payments['DAYS_ENTRY_PAYMENT']
        installments_payments['DPD'] = installments_payments['DPD'].apply(lambda x: x if x > 0 else 0)
        installments_payments['DBD'] = installments_payments['DBD'].apply(lambda x: x if x > 0 else 0)

        # Features: Perform aggregations (description of variable in comments)
        aggregations = {
            'NUM_INSTALMENT_VERSION': ['nunique'], # Version of installment calendar (0 is for credit card) of previous credit. Change of installment version from month to month signifies that some parameter of payment calendar has changed
            #'NUM_INSTALMENT_NUMBER': ['nunique'], # On which installment we observe payment
            #'DAYS_INSTALMENT': ['max', 'mean', 'sum'], # When the installment of previous credit was supposed to be paid (relative to application date of current loan)
            'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'], # When was the installments of previous credit paid actually (relative to application date of current loan)
            'AMT_INSTALMENT': ['max', 'mean', 'sum'], # What was the prescribed installment amount of previous credit on this installment
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'], # What the client actually paid on previous credit on this installment
            'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'], # Feature enginnering from previous step
            'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'], # Feature enginnering from previous step
            'DPD': ['max', 'mean', 'sum'], # Feature enginnering from previous step
            'DBD': ['max', 'mean', 'sum'], # Feature enginnering from previous step
        }

        ins_agg = installments_payments.groupby('SK_ID_CURR').agg(aggregations)
        ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

        # Count installments accounts
        ins_agg['INSTAL_COUNT'] = installments_payments.groupby('SK_ID_CURR').size()

        # Merging dataframes
        application_data = application_data.join(ins_agg, how='left', on='SK_ID_CURR')
        del ins_agg
        del installments_payments

        return application_data

def application_data_preprocess(application_data):
    """Feature engineering on final application_data df"""
    
    # Cleaning

    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    application_data = application_data[application_data['CODE_GENDER'] != 'XNA']

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    application_data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    
    # Ratios on current loan
    application_data['CURRENT_LOAN_EFFORT_RATE'] = application_data['AMT_ANNUITY'] / application_data['AMT_INCOME_TOTAL']
    application_data['CURRENT_LOAN_LTV'] = application_data['AMT_CREDIT'] / application_data['AMT_GOODS_PRICE']
    application_data['CURRENT_LOAN_INCOME_CREDIT_PERC'] = application_data['AMT_INCOME_TOTAL'] / application_data['AMT_CREDIT']
    application_data['CURRENT_LOAN_PAYMENT_RATE'] = application_data['AMT_ANNUITY'] / application_data['AMT_CREDIT']

    # Aggregation of amounts on current loan + other actives bureau loans
    application_data['TOTAL_AMT_ANNUITY'] = application_data['AMT_ANNUITY'] + application_data['BUREAU_TOTAL_ACTIVE_AMT_ANNUITY']
    application_data['TOTAL_AMT_CREDIT'] = application_data['AMT_CREDIT'] + application_data['BUREAU_TOTAL_ACTIVE_AMT_CREDIT_SUM']

    # Ratios on current loan + other actives bureau loans
    application_data['TOTAL_EFFORT_RATE'] = application_data['TOTAL_AMT_ANNUITY'] / application_data['AMT_INCOME_TOTAL']
    application_data['TOTAL_INCOME_CREDIT_PERC'] = application_data['AMT_INCOME_TOTAL'] / application_data['TOTAL_AMT_CREDIT']
    application_data['TOTAL_PAYMENT_RATE'] = application_data['TOTAL_AMT_ANNUITY'] / application_data['TOTAL_AMT_CREDIT']

    # Other relevant ratios
    application_data['DAYS_EMPLOYED_PERC'] = application_data['DAYS_EMPLOYED'] / application_data['DAYS_BIRTH']
    application_data['INCOME_PER_PERSON'] = application_data['AMT_INCOME_TOTAL'] / application_data['CNT_FAM_MEMBERS']
    
    return application_data

def preprocess_joint(application_data, bureau_balance, bureau, previous_application, credit_card_balance, POS_CASH_balance, installments_payments, imputer=None):
    """From initial dataframes, operates preprocessing, cleaning, feature engineering and jointures"""
    
    bureau = bureau_balance_preprocess(bureau_balance, bureau)
    application_data, imputer = bureau_preprocess(bureau, application_data, imputer=imputer)
    application_data = previous_application_preprocess(previous_application, application_data)
    application_data = credit_card_balance_preprocess(credit_card_balance, application_data)
    application_data = POS_CASH_balance_preprocess(POS_CASH_balance, application_data)
    application_data = installments_payments_preprocess(installments_payments, application_data)
    
    # Adding missing columns if applicable
    true_cols = pd.read_csv('./Datas/cols_after_preprocess_joint.csv')
    true_cols.drop('Unnamed: 0', axis=1, inplace=True)
    diff_cols = set(true_cols.columns) - set(application_data.columns)
    for col in diff_cols:
        application_data[col] = np.nan
    
    application_data = application_data_preprocess(application_data)
    
    return application_data, imputer